# MSC Batch S3 Data Operations

Utilities for setting up and using **MultiStorageClient (MSC)** to read/write files from S3-compatible storage. Supports multiple storage backends (AWS S3, S8K, GCS via S3 API) and uses async + semaphore-based concurrency for batch downloads and uploads.

---

## Path Utilities

Converts an `s3://` URI into an MSC profile-prefixed path (e.g. `input/bucket/key`). MSC uses profile names to route requests to the correct configured backend.

```python
from multistorageclient.contrib.async_fs import MultiStorageAsyncFileSystem
fs = MultiStorageAsyncFileSystem()

def reformat_path_s3_to_msc(path: str, profile: str = "input") -> str:
    return path.replace("s3://", f"{profile}/")
```

---

## setup_msc

Initializes the MSC configuration from per-profile credential JSON files and broadcasts the config path to all distributed ranks via `MSC_CONFIG`. Only rank 0 builds and writes the config; all other ranks receive the path via `distributed.broadcast_object`.

**Key behaviors:**
- Reads credentials from JSON files keyed by profile name (`endpoint_url`, `region_name`, `aws_access_key_id`, `aws_secret_access_key`)
- Auto-detects storage backend type from endpoint hostname: `s8k`, `gcs_s3`, or plain `s3`
- Writes the merged config to a temp `.secret` file; registers `atexit` cleanup on rank 0
- Sets `MSC_MAX_WORKERS` and `MSC_CONFIG` env vars consumed by the MSC library

```python
def setup_msc(cred_dict: dict[str, str | None], max_workers: int = 32):
    rank = distributed.get_rank()
    os.environ["MSC_MAX_WORKERS"] = str(max_workers)

    config_dict: dict[str, Any] = {
        "retry": {
            "attempts": 8,
            "delay": 0.05,
            "backoff_multiplier": 2,
        }
    }

    def _append_config_with_s3_credential_path(msc_config_dict, s3_credential_path, profile):
        with open(s3_credential_path, "r") as f:
            authinfo = json.load(f)

        msc_config_dict["profiles"] = msc_config_dict.get("profiles", {})
        msc_config_dict["profiles"][profile] = msc_config_dict["profiles"].get(profile, {})

        storage_provider_type: str = "s3"
        parsed_endpoint_url = urlparse(authinfo["endpoint_url"])

        if parsed_endpoint_url.hostname.endswith(".s8k.io"):
            storage_provider_type = "s8k"
        elif parsed_endpoint_url.hostname.startswith("storage.") and parsed_endpoint_url.hostname.endswith(
            ".googleapis.com"
        ):
            storage_provider_type = "gcs_s3"

        msc_config_dict["profiles"][profile]["storage_provider"] = {
            "type": storage_provider_type,
            "options": {
                "base_path": "",
                "endpoint_url": authinfo["endpoint_url"],
                "region_name": authinfo["region_name"],
            },
        }
        msc_config_dict["profiles"][profile]["credentials_provider"] = {
            "type": "S3Credentials",
            "options": {
                "access_key": authinfo["aws_access_key_id"],
                "secret_key": authinfo["aws_secret_access_key"],
            },
        }

    if rank == 0:
        for profile, cred_path in cred_dict.items():
            if cred_path is not None:
                _append_config_with_s3_credential_path(config_dict, cred_path, profile)
        shared_tmp_dir = os.path.expanduser("~/tmp")
        os.makedirs(shared_tmp_dir, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".secret", delete=False, dir=shared_tmp_dir)
        json.dump(config_dict, tmp, indent=2)
        tmp.flush()
        default_path = tmp.name
        logger.info(f"MSC config written to temporary file: {default_path}")
        atexit.register(lambda p=default_path: (os.remove(p), logger.info(f"Removed temporary MSC config: {p}")))
    else:
        default_path = None

    default_path = distributed.broadcast_object(default_path)
    assert isinstance(default_path, str)
    os.environ["MSC_CONFIG"] = default_path
```

---

## msc_download_many

Batch-downloads a list of remote paths concurrently using the async MSC filesystem. Returns `None` for any path that fails (with a warning logged), so callers must handle sparse results.

**Key behaviors:**
- Concurrency capped by `MSC_MAX_WORKERS` env var (default 32)
- Failures are caught per-file and return `None` rather than raising
- Progress tracked with tqdm

```python
async def msc_download_many(
    fs,
    remote_paths: list[str],
    pbar_desc: Optional[str] = None,
) -> list[Optional[bytes]]:
    """Batch download files from S3 using MultiStorageAsyncFileSystem."""
    max_concurrency = int(os.getenv("MSC_MAX_WORKERS", "32"))
    sem = asyncio.Semaphore(max_concurrency)
    pbar = tqdm(total=len(remote_paths), desc=pbar_desc, disable=pbar_desc is None)

    async def _one(p: str) -> Optional[bytes]:
        async with sem:
            try:
                return await fs._cat_file(p)
            except Exception as e:
                logger.warning(f"Failed to download {p}: {e}")
                return None
            finally:
                pbar.update(1)

    results = await asyncio.gather(*[asyncio.create_task(_one(p)) for p in remote_paths])
    pbar.close()
    return results
```

---

## msc_upload_many

Batch-uploads byte payloads to remote paths concurrently. Returns a `bool` per path indicating success. Same concurrency and error-handling pattern as `msc_download_many`.

```python
async def msc_upload_many(
    fs,
    remote_paths: list[str],
    payloads: list[bytes],
    pbar_desc: Optional[str] = None,
) -> list[bool]:
    """Batch upload files to S3 using MultiStorageAsyncFileSystem."""
    max_concurrency = int(os.getenv("MSC_MAX_WORKERS", "32"))
    sem = asyncio.Semaphore(max_concurrency)
    pbar = tqdm(total=len(remote_paths), desc=pbar_desc, disable=pbar_desc is None)

    async def _one(p: str, data: bytes) -> bool:
        async with sem:
            try:
                await fs._pipe_file(p, data)
                return True
            except Exception as e:
                logger.warning(f"Failed to upload {p}: {e}")
                return False
            finally:
                pbar.update(1)

    results = await asyncio.gather(*[asyncio.create_task(_one(p, d)) for p, d in zip(remote_paths, payloads)])
    pbar.close()
    return results
```
