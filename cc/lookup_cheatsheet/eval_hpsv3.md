# HPSv3 evaluation

Scoring-only pipeline: computes HPSv3 (Human Preference Score v3) for a local UniGenBench-style generated-image folder. Input must be staged locally first (no S3 input support).

## Score Base Template

Runs on CPU (or any GPU with ≥48 GB VRAM for faster scoring). The model is loaded in `bfloat16` by default to fit 48 GB cards.

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
slaunch cpu 1x1 hpsv3_score_<some_run_name> \
    projects/cosmos3/vfm/evaluation/text_to_image/compute_hpsv3_metric.py \
    --input_folder /path/to/local/unigenbench/run \
    --batch_size 4 \
    --device cuda
```

- `--input_folder`: local folder produced by `inference_unigenbench_distributed.py` after staging. Must contain `config.json` (with a `prompts_csv` field) and `<index>_<generation_index>.png` images.
- `--batch_size 4`: safe default for 48 GB GPU in bf16. Reduce to 1 if OOM.
- `--device cuda` or `cpu`.

## Key flags

| Flag | Default | Purpose |
|---|---|---|
| `--input_folder` | required | Local staged image folder |
| `--benchmark_csv` | from `config.json` | Override the prompts CSV (local or `s3://`) |
| `--extension` | `png` | Image file extension |
| `--batch_size` | `4` | HPSv3 inference batch size |
| `--device` | `cuda` | Device for HPSv3 model |
| `--limit` | None | Only score the first N images (debug) |
| `--dry_run` | False | Validate discovery without loading HPSv3 |
| `--force_rescore` | False | Discard existing CSV cache and rescore all |
| `--hpsv3_dtype` | `bfloat16` | Model weight dtype (`bfloat16` / `float16` / `float32`) |
| `--hpsv3_config_path` | None | Optional HPSv3 config override |
| `--hpsv3_checkpoint_path` | None | Optional HPSv3 checkpoint override |

## Resume behavior

The script writes an incremental per-image cache CSV (`hpsv3_scoring_results.csv`) into `--input_folder`. On re-run it skips already-scored images (matched by `task_id`). Use `--force_rescore` to wipe and redo everything.

## Output files

Both written into `--input_folder`:

| File | Contents |
|---|---|
| `hpsv3_scoring_results.csv` | Per-image incremental cache (task_id, score, sigma, success, error, elapsed) |
| `hpsv3_result.json` | Final aggregate metric summary |

## Reporting performance / status

When the user asks "is X done?" or "what's the HPSv3 result?", **read the result JSON** — do not parse the slurm log.

Result JSON path:
```
<input_folder>/hpsv3_result.json
```

Fields to report:

1. *Overall mean score* — `stats.all.mean_hpsv3_score`
2. *Orig subset* — `stats.orig.mean_hpsv3_score` (images whose index starts with `orig`)
3. *Phi subset* — `stats.phi.mean_hpsv3_score` (images whose index starts with `phi`)
4. *Success count* — `success_count` (e.g. `"1170/1170"`)
5. *num_prompts / num_images* — `stats.all.num_prompts`, `stats.all.num_images`

`stats` keys: `all` (always present), `orig` / `phi` (only when those images exist).
Each stats block also has `std_hpsv3_score`, `min_hpsv3_score`, `max_hpsv3_score`, `mean_hpsv3_sigma`. Per-image detail lives under `breakdown[<image_name>]` — don't paste unless asked.

## Install note

HPSv3 is an optional dependency that pins `transformers==4.45.2`. Install it in a separate scoring environment:

```bash
pip install hpsv3
```
