# Create caption-pickle tars from meta tars

Use this when an existing meta-only WebDS layout already carries per-sample LanceDB rows as `<uuid>.json`, and you need to produce a sibling caption-only WebDS layout where each sample is a `<uuid>.pickle` matching the cosmos-captioner shard schema (the format consumed by the captioner-aware data loaders).

For each existing `{webds_path}/<multiple/bucket/paths>/{meta_key}/<shard>.tar`, the script writes a sibling `{webds_path}/<multiple/bucket/paths>/{out_key}/<shard>.tar` whose entries are `<uuid>.pickle` files with this exact shape:

```python
{
  "key": "<uuid>",
  "caption": {
    "caption_cosmos_captioner_image": "<JSON string from table_meta>",
  },
}
```

The caption value is read verbatim from `meta["table_meta"]["captioning_cosmos_captioner_image_v1_full_caption_cosmos_captioner_image"]`. Samples missing that field are silently dropped from the output tar; tars that end up with zero usable samples are not uploaded.

Lives at `pipelines/image/text_rendering/create_caption_pickle.py` in `imaginaire4_sila`.

## Step 1 — collect information

- **Check `./data_common_root.md` for some common root path settings** 
- `<meta_key>` = the source meta-tar subdir; usually `metas`, but ask user
- `<out_key>` = the caption-tar subdir to write into. Default `captions_cosmos_captioner_v1p1` (matches the cosmos-captioner v1.1 convention seen in production GCS).
- User sometime copy a path with `gcs:`, do the auto converion `gcs:`→`s3://`.
- `--metadata_to_caption_type` = `sgd_structured_caption_v1` For new synthetic datasets (i.e. Round 1 (202605)), otherwise `team_structured_caption_v1`
- Ask user whether use `--overwrite`.

### Input Information

| Arg                          | Value                            | Notes                                                                                                                                                                      |
|------------------------------|----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--webds_path`               | `<webds_path>`                   | Target WebDS root. Look `./data_common_root.md`                                                                                                                            |
| `--webds_credential`         | `credentials/gcs.secret`         | Credential file for the WebDS bucket. Default is `credentials/gcs.secret`.                                                                                                 |
| `--meta_key`                 | **ASK USER**                     | Source key under which meta tars live, e.g. `metas_20260502`. Used both as the discovery filter (`/{meta_key}/`) and the substring to swap when computing the output path. |
| `--out_key`                  | `captions_cosmos_captioner_v1p1` | Destination key for the caption-pickle tars. Override only if the downstream loader expects a different folder name.                                                       |
| `--num_concurrency`          | `8`                              | MSC worker pool size for single-threaded, only affects per-call download/upload concurrency.                                                                               |
| `--overwrite`                | _flag_                           | Off by default — tars already present under `/{out_key}/` are skipped or deleted up front                                                                                  |
| `--metadata_to_caption_type` | **ASK USER**                     | Where to get and how to format the pickle, `team_structured_caption_v1` or `sgd_structured_caption_v1`                                                                     |

## Step 2 — compose the formatted command and show user for confirmation

### The `slaunch`-way launch template

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x1 caption_pickle_<dataset_name> \
    pipelines/image/text_rendering/create_caption_pickle.py \
    --webds_path <webds_path> \
    --webds_credential credentials/gcs.secret \
    --meta_key <meta_key> \
    --out_key <out_key> \
    --metadata_to_caption_type <metadata_to_caption_type> \
    --num_concurrency 8
```


## Step 3 — launch

- **No silent run by yourself, confirmation is always required!**
- Ask the user which cluster he needs to launch the command.
- Sanity check if the run is duplicated (usually with same run name), if duplicated, stop and inform the user.
- When use confirm the run and no duplication, you need to use your skill `/ssh_run` to help launch the run
