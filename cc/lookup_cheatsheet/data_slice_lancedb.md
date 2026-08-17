# Slice a LanceDB table by `source_dataset`

Use this when you have a big joint LanceDB table (e.g. `image_meta_table_full.lance`) and want to extract one or more per-dataset sub-tables, written out as new Lance datasets.

The script does **distributed single-pass slicing**: every rank scans a disjoint shard of input fragments, filters by `source_dataset IN (...)`, and accumulates rows in a per-rank `LanceDBBuffer` keyed by source_dataset. Every read round is followed by a collective `sync` that gathers all ranks' rows onto rank 0; rank 0 then drains as many full `MAX_ROW_PER_FILE`-sized segments as it can per output Lance. After the loop, rank 0 flushes any remainder rows.

Lives at `pipelines/image/text_rendering/slice_lancedb.py` in `imaginaire4_sila`.



## Step 1 — Collect Information.

- Read `~/home/xingqianx~/Project/trichord/cc/lookup_cheatsheet/data_common_root.md`
- Ask user what dataset(s) he/she will be working on.
- Try figure out the true webds / lanceDB locations from `~/Project/trichord/cc/lookup_cheatsheet/data_common_root.md`

## Step 2 — Compose Command (cmd) and show user.

At this stage, you need to show a full cmd to user. The code support multiple sliding from one run. Template is shown below:

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 1x8 slice_<dataset_name> \
    pipelines/image/text_rendering/slice_lancedb.py \
    --input_lancedb_path gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance \
    --dataset_name <dsetA> <dsetB> <dsetC> \
    --output_lancedb_path \
        <output_lancedb_path_A> \
        <output_lancedb_path_B> \
        <output_lancedb_path_C> \
    --lancedb_credential credentials/gcs.secret
```

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 8x1 slice_all_datasets \
    pipelines/image/text_rendering/slice_lancedb.py \
    --input_lancedb_path gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance \
    --dataset_name \
        nvcommercial_700m \
        coyo_700m \
        generations_qwen_image_2512_filtered_photoreal \
        wordnet_captions_20260224 \
        self_improving_synthetic_2026-02-09 \
        self_improving_synthetic_2026-02-14 \
        MMC4 \
        red \
    --output_lancedb_path \
        gs://nv-00-10206-vfm/lancedb/image/regular/nvcommercial_700m_slice_from_maintable_20260731.lance \
        gs://nv-00-10206-vfm/lancedb/image/regular/coyo_700m_slice_from_maintable_20260731.lance \
        gs://nv-00-10206-vfm/lancedb/image/synthetic/generations_qwen_image_2512_filtered_photoreal_slice_from_maintable_20260731.lance \
        gs://nv-00-10206-vfm/lancedb/image/synthetic/wordnet_captions_20260224_slice_from_maintable_20260731.lance \
        gs://nv-00-10206-vfm/lancedb/image/synthetic/self_improving_synthetic_2026-02-09_slice_from_maintable_20260731.lance \
        gs://nv-00-10206-vfm/lancedb/image/synthetic/self_improving_synthetic_2026-02-14_slice_from_maintable_20260731.lance \
        gs://nv-00-10206-vfm/lancedb/image/regular/mmc4_slice_from_maintable_20260731.lance \
        gs://nv-00-10206-vfm/lancedb/image/regular/red_slice_from_maintable_20260731.lance \
    --lancedb_credential credentials/gcs.secret
```


```
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_sila \
slaunch cpu 16x1 slice_all_sync \
    pipelines/image/text_rendering/slice_lancedb.py \
    --input_lancedb_path gs://nv-00-10206-lancedb/prod/image/image_meta_table_full.lance \
    --dataset_name \
        generations_qwen_image_2512_filtered_photoreal \
        wordnet_captions_20260224 \
        self_improving_synthetic_2026-02-09 \
        self_improving_synthetic_2026-02-14 \
        v1_agent_distilled_v19_99827 \
        v1_agent_distilled_v6a_57230 \
        v1_agent_distilled_v7m_31806 \
    --output_lancedb_path \
        gs://nv-00-10206-vfm/lancedb/image/synthetic/generations_qwen_image_2512_filtered_photoreal_slice_from_maintable_20260731.lance \
        gs://nv-00-10206-vfm/lancedb/image/synthetic/wordnet_captions_20260224_slice_from_maintable_20260731.lance \
        gs://nv-00-10206-vfm/lancedb/image/synthetic/self_improving_synthetic_2026-02-09_slice_from_maintable_20260731.lance \
        gs://nv-00-10206-vfm/lancedb/image/synthetic/self_improving_synthetic_2026-02-14_slice_from_maintable_20260731.lance \
        gs://nv-00-10206-vfm/lancedb/image/synthetic/v1_agent_distilled_v19_99827_slice_from_maintable_20260731.lance \
        gs://nv-00-10206-vfm/lancedb/image/synthetic/v1_agent_distilled_v6a_57230_slice_from_maintable_20260731.lance \
        gs://nv-00-10206-vfm/lancedb/image/synthetic/v1_agent_distilled_v7m_31806_slice_from_maintable_20260731.lance \
    --lancedb_credential credentials/gcs.secret \
    --fragment_group_size 4
```

## Step 3 — Launch the cmd

- Ask user which cluster it want to run. Use your skill `ssh_run` to launch the cmd when user approved.
