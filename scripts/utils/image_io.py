# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact Cosmos Lab at cosmoslab@exchange.nvidia.com.
# -----------------------------------------------------------------------------


import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import fsspec
from PIL import Image


def video_bytes_to_data_url(video_bytes: bytes, video_fmt: str = "mp4") -> str:
    mime_types = {
        "mp4": "video/mp4",
        "webm": "video/webm",
        "avi": "video/avi",
    }
    mime_type = mime_types.get(video_fmt.lower(), f"video/{video_fmt}")
    return f"data:{mime_type};base64,{base64.b64encode(video_bytes).decode('utf-8')}"


def hash_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


S3_CREDENTIAL_FILE = Path(__file__).resolve().parents[2] / "credentials" / "gcs.secret"


def encode_image(data: bytes, fmt: str = "webp", quality: int = 85, method: int = 5) -> bytes:
    fmt = fmt.lower()
    if fmt == "png" and data[:8] == b"\x89PNG\r\n\x1a\n":
        return data
    image = Image.open(io.BytesIO(data))
    buf = io.BytesIO()
    if fmt == "webp":
        image.save(buf, format="WEBP", quality=quality, method=method)
    elif fmt == "jpeg":
        image.convert("RGB").save(buf, format="JPEG", quality=quality)
    elif fmt == "png":
        image.save(buf, format="PNG")
    else:
        raise ValueError(f"unsupported image format {fmt!r}; use webp, jpeg or png")
    return buf.getvalue()


def put(data: bytes, path: str | Path) -> str:
    url = str(path)
    scheme = urlparse(url).scheme
    if scheme in ("", "file"):
        target = Path(url[len("file://"):] if scheme == "file" else url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return url
    assert scheme == "s3", f"unsupported destination {url!r}; use a local path or s3://bucket/key"
    import boto3
    from botocore.config import Config

    cred = json.loads(S3_CREDENTIAL_FILE.read_text())
    client = boto3.client(
        "s3",
        endpoint_url=cred["endpoint_url"],
        region_name=cred["region_name"],
        aws_access_key_id=cred["aws_access_key_id"],
        aws_secret_access_key=cred["aws_secret_access_key"],
        config=Config(s3={"addressing_style": "path"}, request_checksum_calculation="when_required"),
    )
    parsed = urlparse(url)
    client.put_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"), Body=data)
    return url


def sniff_image_fmt(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return "png"


PBSS_TO_GCS = {
    "logged_images": ("nv-00-10206-images", "logged_images"),
    "annotated_data": ("nv-00-10206-images", "annotated_data"),
    "webdataset": ("nv-00-10206-webdataset-images", ""),
    "debug": ("nv-00-10206-images", "debug"),
}
GCS_NATIVE_BUCKETS = {"nv-00-10206-images", "nv-00-10206-webdataset-images"}
S3R_CLIENT = None


def parse_s3r(uri: str) -> tuple[str, str, int, int]:
    rest = uri[len("s3r:"):].split("//", 1)[1].rsplit("@", 1)[0]
    bucket_and_path, range_part = rest.rsplit(":", 1)
    bucket, object_path = bucket_and_path.split("/", 1)
    start, length = map(int, range_part.split("-"))
    if bucket in PBSS_TO_GCS:
        bucket, prefix = PBSS_TO_GCS[bucket]
        object_path = f"{prefix}/{object_path}" if prefix else object_path
    elif bucket not in GCS_NATIVE_BUCKETS:
        raise ValueError(f"unknown s3r bucket {bucket!r} in {uri}")
    return bucket, object_path, start, length


def read_s3r_bytes(uri: str) -> bytes:
    global S3R_CLIENT
    if S3R_CLIENT is None:
        import boto3
        from botocore.config import Config

        cred = json.loads(S3_CREDENTIAL_FILE.read_text())
        S3R_CLIENT = boto3.client(
            "s3",
            endpoint_url=cred["endpoint_url"],
            region_name=cred["region_name"],
            aws_access_key_id=cred["aws_access_key_id"],
            aws_secret_access_key=cred["aws_secret_access_key"],
            config=Config(s3={"addressing_style": "path"}),
        )
    bucket, key, start, length = parse_s3r(uri)
    return S3R_CLIENT.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{start + length - 1}")["Body"].read()


def read_url_bytes(url: str) -> bytes:
    scheme = urlparse(url).scheme
    if scheme == "s3r":
        return read_s3r_bytes(url)
    if scheme == "data":
        return base64.b64decode(url.split(",", 1)[1])
    if scheme in ("", "file", "http", "https", "s3", "gs"):
        with fsspec.open(url, "rb") as f:
            return f.read()
    raise ValueError(f"unsupported url scheme {scheme!r} in {url}")


def image_conversion(value: Any, src_fmt: str = "", dst_fmt: str = "bytes") -> Any:
    if not src_fmt:
        if isinstance(value, (bytes, bytearray)):
            src_fmt = "bytes"
        elif isinstance(value, Image.Image):
            src_fmt = "pil"
        else:
            src_fmt = "data_url" if urlparse(str(value)).scheme == "data" else "url"
    if src_fmt == "data_url" and dst_fmt == "data_url":
        return value

    if src_fmt == "bytes":
        data = bytes(value)
    elif src_fmt == "pil":
        buf = io.BytesIO()
        value.save(buf, format="PNG")
        data = buf.getvalue()
    elif src_fmt in ("data_url", "url"):
        data = read_url_bytes(str(value))
    else:
        raise ValueError(f"unknown image src_fmt {src_fmt!r}")

    if dst_fmt == "bytes":
        return data
    if dst_fmt == "pil":
        image = Image.open(io.BytesIO(data))
        image.load()
        return image
    if dst_fmt == "data_url":
        mime_type = f"image/{sniff_image_fmt(data)}"
        return f"data:{mime_type};base64,{base64.b64encode(data).decode('utf-8')}"
    raise ValueError(f"unknown image dst_fmt {dst_fmt!r}")
