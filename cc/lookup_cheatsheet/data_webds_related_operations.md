# WebDataset Operations

Scripts live in `pipelines/image/text_rendering/` in `imaginaire4_sila`.

## `webds_get_cluster_distribution.py` — count samples per category across wdinfo.json files

Purpose: for one or more webdataset base URIs, walk `<category>/<resolution>/<aspect>/wdinfo.json` and sum `total_key_count` per category, folding known category-name variants into a canonical set (`CATEGORY_GROUPS`). Prints a per-dataset breakdown table and, when multiple `--base_urls` are given, a cross-dataset summary table, and safe it as a report JSON on designated location.

Example run command is below:

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x1 webds_get_cluster_distribution_20260818 \
    pipelines/image/text_rendering/webds_get_cluster_distribution.py \
    --input_wdinfo_path \
        "s3://nv-00-10206-webdataset-images/webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/nvcommercial_700m" \
        "s3://nv-00-10206-webdataset-images/webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/MMC4" \
        "s3://nv-00-10206-webdataset-images/webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/coyo_700m" \
        "s3://nv-00-10206-webdataset-images/webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/red" \
        "s3://nv-00-10206-webdataset-images/webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1_high_quality/wdinfo" \
        "s3://nv-00-10206-webdataset-images/webdataset_cosmos_lab_image_v1/v1_pexels_residual_trustedK1_v2/wdinfo/pexels_residual_trustedK1_v2" \
    --input_dataset_name \
        nvcommercial_700m \
        MMC4 \
        coyo_700m \
        red \
        v1_high_quality \
        pexels_residual_trustedK1_v2 \
    --input_cred credentials/gcs.secret \
    --workers 32 \
    --output_json s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/webds_cluster_dist/stats_20260818.json \
    --output_cred credentials/gcs.secret
```

Notes:
- `--input_wdinfo_path` takes regular `s3://` paths (converted internally via `reformat_path_s3_to_msc`), not `gcs:`.
- `--input_cred` / `--output_cred` both default to `credentials/gcs.secret`.
- `--input_dataset_name` is optional and must 1:1 match `--input_wdinfo_path` if given; otherwise the dataset name in tables/JSON falls back to the last path segment of each `--input_wdinfo_path` entry.
- `--output_json` is optional — when set, writes `{"per_dataset": {...per-category counts + TOTAL...}, "grand_total": ...}` to that `s3://` path (via `msc_upload_many`), auto fill `<YYYYMMDD>` with the current date.