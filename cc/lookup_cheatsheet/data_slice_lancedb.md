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

## Step 3 — Launch the cmd

- Ask user which cluster and node it want to run. Use your skill `ssh_run` to launch the cmd when user approved.
