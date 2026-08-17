# -----------------------------------------------------------------------------
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""
Script to download checkpoint from S3 and convert to HuggingFace format for vLLM serving.

This script is designed to be uploaded to S3 and executed in the Lepton endpoint container.
It handles the checkpoint download, conversion, and tokenizer setup.
"""

import argparse
import io
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from typing import Generator, Union
from urllib.parse import urlparse

import boto3
import torch
from botocore.config import Config
from botocore.exceptions import ClientError
from loguru import logger as log
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.filesystem import FileSystemBase
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict



class S3Stream(io.BytesIO):
    """
    Workaround for PyTorch manually closing the stream before we can upload it to S3. We override the close() as noop
    and instead call our own _true_close() method to close the stream after we are done using it.
    The commit at fault is https://github.com/pytorch/pytorch/commit/9c909bf3bb122db2cce95e2eb7459bbe50dfa15a
    """

    def close(self):
        self.flush()
        # No close

    def _true_close(self):
        super().close()


class S3FileSystem(FileSystemBase):
    """Implementation of FileSystemBase for AWS S3 storage."""

    def __init__(
        self,
        credential_path: str,
        max_attempts: int = 20,
        initial_backoff: float = 1.0,
        max_backoff: float = 30.0,
        backoff_factor: float = 2.0,
    ) -> None:
        """
        Initialize S3FileSystem with retry configuration.

        Args:
            credential_path: Path to AWS credentials JSON file
            max_attempts: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            max_backoff: Maximum backoff time in seconds
            backoff_factor: Multiplicative factor for backoff time
        """
        with open(credential_path, "r") as f:
            conf = json.load(f)

        # Configure boto3 with retry settings
        config = Config(
            retries=dict(max_attempts=max_attempts, mode="adaptive"),  # Adaptive mode automatically handles throttling
            connect_timeout=60,
            read_timeout=60,
            # Turn this on with gcp checkpoint, but it will require upgrad docker
            # request_checksum_calculation="when_required",  # Data integrity check for uploads and downloads
            # response_checksum_validation="when_required",  # Data integrity check for uploads and downloads
        )

        self.s3_client = boto3.client("s3", config=config, **conf)
        self.max_attempts = max_attempts
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.backoff_factor = backoff_factor

    def _retry_with_backoff(self, operation_func, *args, **kwargs):
        """
        Execute an operation with exponential backoff retry logic.

        Args:
            operation_func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result of the operation function

        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        backoff = self.initial_backoff

        for attempt in range(self.max_attempts):
            try:
                return operation_func(*args, **kwargs)
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "")
                log.info(f"S3 Filesystem: Received ClientError: {error_code}", rank0_only=False)

                # Handle specific error cases
                if error_code in ["SlowDown", "ThrottlingException", "RequestLimitExceeded", "InternalError"]:
                    last_exception = e
                    if attempt < self.max_attempts - 1:  # Don't sleep on last attempt
                        current_backoff = min(backoff, self.max_backoff)
                        log.info(f"S3 Filesystem: Retrying in {current_backoff} seconds", rank0_only=False)
                        time.sleep(current_backoff)
                        backoff *= self.backoff_factor
                        continue
                # For other client errors, raise immediately
                raise
            except Exception as e:
                log.info(f"S3 Filesystem: Received Exception: {str(e)}", rank0_only=False)
                last_exception = e
                if attempt < self.max_attempts - 1:
                    current_backoff = min(backoff, self.max_backoff)
                    log.info(f"S3 Filesystem: Retrying in {current_backoff} seconds", rank0_only=False)
                    time.sleep(current_backoff)
                    backoff *= self.backoff_factor
                    continue

        # pyrefly: ignore [bad-raise]
        raise last_exception

    @contextmanager
    def create_stream(self, path: Union[str, os.PathLike], mode: str) -> Generator[io.IOBase, None, None]:
        """Create a stream for reading from or writing to S3 with retry logic."""
        path_str = str(path)
        bucket, key = self._parse_s3_uri(path_str)
        log.info(f"S3 Filesystem: Creating stream for {key} in bucket {bucket}", rank0_only=False)

        if mode == "rb":
            stream = io.BytesIO()
            try:

                def download_operation():
                    self.s3_client.download_fileobj(bucket, key, stream)
                    stream.seek(0)

                log.info(f"S3 Filesystem: Downloading {key} from bucket {bucket}", rank0_only=False)
                self._retry_with_backoff(download_operation)
                log.info("S3 Filesystem: Download complete", rank0_only=False)
                yield stream
            finally:
                stream.close()
        elif mode == "wb":
            stream = S3Stream()
            try:
                yield stream

                def upload_operation():
                    stream.seek(0)
                    self.s3_client.upload_fileobj(stream, bucket, key)

                log.info(f"S3 Filesystem: Uploading {key} to bucket {bucket}", rank0_only=False)
                self._retry_with_backoff(upload_operation)
                log.info("S3 Filesystem: Upload complete", rank0_only=False)
            finally:
                stream._true_close()
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def concat_path(self, path: Union[str, os.PathLike], suffix: str) -> Union[str, os.PathLike]:
        """Concatenate S3 path with suffix."""
        path_str = str(path)
        if path_str.endswith("/"):
            return f"{path_str}{suffix}"
        return f"{path_str}/{suffix}"

    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        """Initialize and validate S3 path."""
        path_str = str(path)
        if not path_str.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {path_str}. Must start with 's3://'")
        return path_str

    def rename(self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]) -> None:
        """Rename (move) an object in S3 with retry logic."""
        src_bucket, src_key = self._parse_s3_uri(str(path))
        dst_bucket, dst_key = self._parse_s3_uri(str(new_path))

        def copy_operation():
            copy_source = {"Bucket": src_bucket, "Key": src_key}
            self.s3_client.copy(copy_source, dst_bucket, dst_key)

        self._retry_with_backoff(copy_operation)

        def delete_operation():
            self.s3_client.delete_object(Bucket=src_bucket, Key=src_key)

        self._retry_with_backoff(delete_operation)

    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        """
        Create a "directory" in S3.

        Note: S3 doesn't have real directories, but we can create an empty object
        with a trailing slash to simulate a directory.
        """
        # Creating same buckets from different ranks can cause rate limit issues in GCP.
        # In object store, we don't need to create a directory.
        pass

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        """Validate if the checkpoint_id is a valid S3 URI."""
        checkpoint_id_str = str(checkpoint_id)
        try:
            if not checkpoint_id_str.startswith("s3://"):
                return False
            parsed = urlparse(checkpoint_id_str)
            return bool(parsed.netloc and parsed.path)  # Must have bucket and key
        except Exception:
            return False

    def exists(self, path: Union[str, os.PathLike]) -> bool:
        """Check if an object exists in S3 with retry logic."""
        bucket, key = self._parse_s3_uri(str(path))
        try:

            def head_operation():
                self.s3_client.head_object(Bucket=bucket, Key=key)

            self._retry_with_backoff(head_operation)
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code", "") == "404":
                return False
            raise

    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        """Remove a file from S3 with retry logic."""
        bucket, key = self._parse_s3_uri(str(path))

        def delete_operation():
            self.s3_client.delete_object(Bucket=bucket, Key=key)

        self._retry_with_backoff(delete_operation)

    def _parse_s3_uri(self, uri: str) -> tuple[str, str]:
        """
        Parse an S3 URI into bucket and key.

        Args:
            uri: S3 URI in the format s3://bucket-name/key

        Returns:
            Tuple of (bucket_name, key)

        Raises:
            ValueError: If the URI is invalid
        """
        uri = uri if isinstance(uri, str) else str(uri)
        if not uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI: {uri}. Must start with 's3://'")

        parsed = urlparse(uri)
        bucket = parsed.netloc

        # Remove leading slash from key
        key = parsed.path.lstrip("/")

        if not bucket:
            raise ValueError(f"Invalid S3 URI: {uri}. No bucket specified")

        return bucket, key


class S3StorageWriter(FileSystemWriter):
    def __init__(
        self,
        credential_path: str,
        path: str,
        **kwargs,
    ) -> None:
        """
        Initialize an S3 writer for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            path (str): The S3 URI to write checkpoints to.
            kwargs (dict): Keyword arguments to pass to the parent :class:`FileSystemWriter`.
        """
        super().__init__(
            path=path,
            sync_files=False,  # FIXME: setting this to True makes the run to fail (L#333: `os.fsync(stream.fileno())`)
            **kwargs,
        )
        self.fs = S3FileSystem(credential_path)  # type: ignore
        self.path = self.fs.init_path(path)

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)


class S3StorageReader(FileSystemReader):
    def __init__(self, credential_path: str, path: Union[str, os.PathLike]) -> None:
        """
        Initialize an S3 reader for distributed checkpointing.

        Args:
            region (str): The AWS region for S3.
            path (Union[str, os.PathLike]): The S3 path to read checkpoints from.
        """
        super().__init__(path)
        self.fs = S3FileSystem(credential_path)  # type: ignore
        self.path = self.fs.init_path(path)
        self.sync_files = False

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return S3FileSystem.validate_checkpoint_id(checkpoint_id)


def download_s3_directory(s3_path, local_path, s3_credential: str):
    """Download all files from an S3 directory to local path."""
    import subprocess

    # Use AWS CLI for faster parallel downloads
    credential_dict = json.load(open(s3_credential))
    cmd_prefix = f"AWS_ACCESS_KEY_ID={credential_dict['aws_access_key_id']} AWS_SECRET_ACCESS_KEY={credential_dict['aws_secret_access_key']} AWS_ENDPOINT_URL={credential_dict['endpoint_url']} AWS_REGION={credential_dict['region_name']}"
    cmd = f"{cmd_prefix} aws s3 sync {s3_path} {local_path} --no-progress"
    log.info(f"Downloading checkpoint files: {cmd}")
    tic = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        log.info(f"Error downloading checkpoint: {result.stderr}")
        raise RuntimeError(f"Failed to download checkpoint from {s3_path}")

    log.info(f"Successfully downloaded checkpoint to {local_path} in {time.time() - tic:.2f} seconds")
    return local_path


def load_dcp_state_dict(
    ckpt_dir, checkpoint_cred: str, download_files=True, remove_temp_dir=1, temp_prefix="/workspace/checkpoint"
):
    """Load a Distributed Checkpoint (DCP) from S3 into a PyTorch state_dict."""
    log.info(f"Loading DCP checkpoint from {ckpt_dir}... with checkpoint_cred: {checkpoint_cred}")
    state_dict = {}

    if ckpt_dir.startswith("s3://"):
        if download_files:
            # Create temp directory for checkpoint
            temp_dir = os.path.join(temp_prefix, ckpt_dir.split("s3://")[-1])
            log.info(f"Created temp directory: {temp_dir}")
            log.info(f"Downloading checkpoint from {ckpt_dir} to {temp_dir} with checkpoint_cred: {checkpoint_cred}")
            local_ckpt_dir = download_s3_directory(ckpt_dir, temp_dir, checkpoint_cred)

            # Load from local filesystem
            fs_storage_reader = torch.distributed.checkpoint.FileSystemReader(local_ckpt_dir)
            log.info(f"Loaded checkpoint from local filesystem: {local_ckpt_dir}")
            _load_state_dict(
                state_dict=state_dict,
                storage_reader=fs_storage_reader,
                planner=_EmptyStateDictLoadPlanner(),
                no_dist=True,
            )
            log.info(f"Loaded checkpoint from local filesystem: {local_ckpt_dir}")

            # Cleanup temp directory
            if remove_temp_dir:
                import shutil

                shutil.rmtree(temp_dir)

        else:  # read online
            storage_reader = S3StorageReader(
                credential_path=checkpoint_cred,
                path=ckpt_dir,
            )
            _load_state_dict(
                state_dict=state_dict, storage_reader=storage_reader, planner=_EmptyStateDictLoadPlanner(), no_dist=True
            )

    else:
        # Load from local filesystem
        fs_storage_reader = torch.distributed.checkpoint.FileSystemReader(ckpt_dir)
        _load_state_dict(
            state_dict=state_dict, storage_reader=fs_storage_reader, planner=_EmptyStateDictLoadPlanner(), no_dist=True
        )

    print("DCP checkpoint loaded successfully.")
    return state_dict


def download_tokenizer(model_path, output_dir, credentials="credentials/s3_training.secret"):
    """Download tokenizer files from a local directory or S3 path."""
    print(f"Downloading tokenizer files for {model_path} to {output_dir}")
    if os.path.isdir(model_path):
        shutil.copytree(model_path, output_dir, dirs_exist_ok=True)
        return
    if model_path.startswith("s3://"):
        parsed = urlparse(model_path)
        s3_bucket = parsed.netloc
        s3_prefix = parsed.path.lstrip("/")
        from projects.cosmos3.vlm.utils.pretrained_models_downloader import parallel_download_s3_prefix_to_dir
        os.makedirs(output_dir, exist_ok=True)
        parallel_download_s3_prefix_to_dir(s3_bucket, s3_prefix, output_dir, credentials)
        return
    raise ValueError(f"--model must be a local directory or an s3:// path, got: {model_path}")


def is_huggingface_format(ckpt_dir: str) -> bool:
    """Return True if the checkpoint directory looks like HuggingFace format (config.json + model weights)."""
    if not os.path.isdir(ckpt_dir):
        log.info(f"Model directory {ckpt_dir} does not exist")
        return False
    files = os.listdir(ckpt_dir)
    has_config = "config.json" in files
    log.info(f"has_config: {has_config}")
    has_single = "model.safetensors" in files
    log.info(f"has_single: {has_single}")
    shards = [
        f
        for f in files
        if f.endswith(".safetensors")
        and (
            (f.startswith("model-") and "-of-" in f)  # model-00001-of-00005.safetensors
            or os.path.splitext(f)[0].isdigit()  # 00000.safetensors
        )
    ]
    has_shards = len(shards) > 0
    log.info(f"has_shards: {has_shards}")
    return bool(has_config and (has_single or has_shards))


def check_model_shards_complete(model_dir):
    """Check if all model shards are completely downloaded."""
    if not os.path.exists(model_dir):
        return False

    files = os.listdir(model_dir)
    shard_files = [f for f in files if f.startswith("model-") and f.endswith(".safetensors")]

    if not shard_files:
        return False

    # Extract total number of shards
    for f in shard_files:
        parts = f.split("-of-")
        if len(parts) == 2:
            total_shards = int(parts[1].split(".")[0])
            break
    else:
        return False

    # Check if all shards present
    expected_shards = set(f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors" for i in range(total_shards))
    actual_shards = set(shard_files)

    return expected_shards.issubset(actual_shards)


def split_state_dict_for_safetensors(state_dict, max_shard_size=5 * 1024**3):
    """Split state dict into shards for safetensors format."""
    shards = []
    current_shard = {}
    current_size = 0

    for key, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()

        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    return shards


def save_to_hf_format(state_dict, save_path, max_shard_size=5 * 1024**3):
    """Save state dict in HuggingFace safetensors format."""
    log.info(f"Saving model to HuggingFace format at {save_path}...")
    os.makedirs(save_path, exist_ok=True)

    # Split into shards if needed
    shards = split_state_dict_for_safetensors(state_dict, max_shard_size=max_shard_size)

    if len(shards) == 1:
        # Single file
        safetensors_path = os.path.join(save_path, "model.safetensors")
        save_file(shards[0], safetensors_path)
        log.info(f"Saved single model file: {safetensors_path}")
    else:
        # Multiple shards
        total_shards = len(shards)
        index = {"metadata": {"total_size": 0}, "weight_map": {}}

        for i, shard in enumerate(shards):
            shard_filename = f"model-{i + 1:05d}-of-{total_shards:05d}.safetensors"
            shard_path = os.path.join(save_path, shard_filename)
            save_file(shard, shard_path)
            log.info(f"Saved shard {i + 1}/{total_shards}: {shard_filename}")

            # Update index
            for key in shard.keys():
                index["weight_map"][key] = shard_filename

        # Save index file
        index_path = os.path.join(save_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            index_target = json.load(open(index_path, "r"))
            weight_map_key_target = set(index_target["weight_map"].keys())
            weight_map_key_source = set(index["weight_map"].keys())
            diff = weight_map_key_target - weight_map_key_source
            if "vision_model.radio_model.input_conditioner.norm_mean" in diff:  # handle special case for nemotron VL
                log.warning(f"diff: {diff}")
            elif diff:
                log.info(f"diff: {diff}")
                raise ValueError(f"diff: {diff}")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        log.info(f"Saved index file: {index_path}")

    log.info("Model saving complete.")


def map_cosmos_weight_to_hf_qwen3_vl_moe(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    log.info(f"remapping cosmos weight to hf qwen3 vl moe")
    state_dict_renamed = {}
    for k, v in state_dict.items():
        if "lm_head" in k:
            state_dict_renamed["lm_head.weight"] = v
        elif k.startswith("visual"):
            state_dict_renamed["model." + k] = v
        elif k.startswith("layers"):
            if "mlp.gate_proj.weight" in k:
                up_proj = state_dict[k.replace("gate", "up")]
                gate_proj = v
                v_cat = (
                    torch.cat([gate_proj, up_proj], dim=1).permute(0, 2, 1).contiguous()
                )  # [128,1536,2048] to [128, 2048, 1536]
                layer_id = k.split("layers.")[1].split(".")[0]
                state_dict_renamed[f"model.language_model.layers.{layer_id}.mlp.experts.gate_up_proj"] = v_cat
            elif "mlp.up_proj" in k:
                continue
            elif "gate_and_up_projs" in k:
                layer_id = k.split("layers.")[1].split(".")[0]
                state_dict_renamed[f"model.language_model.layers.{layer_id}.mlp.experts.gate_up_proj"] = v.permute(
                    0, 2, 1
                ).contiguous()
            elif "mlp.down_proj.weight" in k:
                layer_id = k.split("layers.")[1].split(".")[0]
                state_dict_renamed[f"model.language_model.layers.{layer_id}.mlp.experts.down_proj"] = v.permute(
                    0, 2, 1
                ).contiguous()
            elif "down_projs" in k:
                layer_id = k.split("layers.")[1].split(".")[0]
                state_dict_renamed[f"model.language_model.layers.{layer_id}.mlp.experts.down_proj"] = v.permute(
                    0, 2, 1
                ).contiguous()
            else:
                state_dict_renamed["model.language_model." + k] = v
        elif k.startswith("embed_tokens"):
            state_dict_renamed["model.language_model." + k] = v
        elif k.startswith("norm.weight"):
            state_dict_renamed["model.language_model." + k] = v
        else:
            raise ValueError(f"Unsupported weight: {k}")

    state_dict = state_dict_renamed

    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Download and convert checkpoint for vLLM serving")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--s3_credential", type=str, default="credentials/s3_training_eu.secret")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--skip_if_exists", action="store_true")
    parser.add_argument("--remove_model_prefix", action="store_true")
    parser.add_argument("--tokenizer_credential", type=str, default=None)
    parser.add_argument("--remove_temp_dir", default=1)
    parser.add_argument("--temp_prefix", default="/tmp/checkpoint")
    args = parser.parse_args()

    if args.tokenizer_credential is None:
        args.tokenizer_credential = args.s3_credential

    # Check if already converted
    if args.skip_if_exists and check_model_shards_complete(args.output_dir):
        print(f"Model already exists at {args.output_dir}, skipping conversion")
        return 0

    # Download tokenizer
    download_tokenizer(args.model_path, args.output_dir, credentials=args.tokenizer_credential)

    # Resolve local checkpoint path (download from S3 if needed)
    temp_dir = None
    if args.checkpoint_path.startswith("s3://"):
        temp_dir = os.path.join(args.temp_prefix, args.checkpoint_path.split("s3://")[-1])
        local_ckpt_dir = download_s3_directory(args.checkpoint_path, temp_dir, args.s3_credential)
    else:
        local_ckpt_dir = args.checkpoint_path

    # If already HuggingFace format, copy to output_dir and skip DCP conversion
    ckpt_is_hf = is_huggingface_format(local_ckpt_dir)
    if ckpt_is_hf:
        log.info("Checkpoint is already in HuggingFace format, skipping DCP conversion")
        shutil.copytree(local_ckpt_dir, args.output_dir, dirs_exist_ok=True)
        if temp_dir and args.remove_temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
        log.info(f"Successfully prepared model at {args.output_dir} (HuggingFace format copied)")

    if "nemotron" in args.model_path.lower():
        # patch tokenizer_config.json
        from projects.cosmos3.vlm.processors.nemotronvl_processor import nemotron_chat_template

        with open(os.path.join(args.output_dir, "tokenizer_config.json"), "r") as f:
            tokenizer_config = json.load(f)
            tokenizer_config["chat_template"] = nemotron_chat_template
        with open(os.path.join(args.output_dir, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        log.info(f"Changed chat template to nemotron chat template, new tokenizer_config: {tokenizer_config}")

    if ckpt_is_hf:
        return 0

    # DCP path: load and convert
    state_dict = load_dcp_state_dict(
        local_ckpt_dir, args.s3_credential, download_files=False, remove_temp_dir=0, temp_prefix=args.temp_prefix
    )
    log.info(f"Loaded state_dict, keys: {state_dict.keys()}")

    state_dict = {k: v for k, v in state_dict.items() if "_fp32_params" not in k}
    log.info(f"Filtered _fp32_params keys, remaining keys: {list(state_dict.keys())[:10]}")

    if args.remove_model_prefix:
        state_dict_renamed = {}
        for k, v in state_dict.items():
            state_dict_renamed[k.replace("model.", "", 1)] = v
        state_dict = state_dict_renamed
        log.info(f"Removed model prefix from state_dict, new state_dict keys: {state_dict.keys()}")

    if args.model_path in [
        "Qwen/Qwen3-VL-30B-A3B-Thinking",
        "Qwen/Qwen3-VL-235B-A22B-Thinking",
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "Qwen/Qwen3-VL-235B-A22B-Instruct",
    ]:
        state_dict = map_cosmos_weight_to_hf_qwen3_vl_moe(state_dict)
        log.info(f"Mapped cosmos weight to hf qwen3 vl moe, new state_dict keys: {state_dict.keys()}")

    # Save in HF format
    if "nemotron" in args.model_path.lower():
        max_shard_size = 4 * 1024**3
    else:
        max_shard_size = 5 * 1024**3
    save_to_hf_format(state_dict, args.output_dir, max_shard_size=max_shard_size)

    if temp_dir and args.remove_temp_dir:
        shutil.rmtree(temp_dir, ignore_errors=True)

    log.info(f"Successfully prepared model at {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())