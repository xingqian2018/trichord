# Upsample UniGenBench prompts

Reads a UniGenBench CSV (id, prompt, extra_info), calls `PromptUpsampler` (Gemini-backed) on each prompt, and writes a new CSV plus a sidecar JSON with full upsampling metadata.

Lives at `projects/cosmos3/vfm/evaluation/text_to_image/prompt_upsampling_ugb.py` in `imaginaire4_alt`.

## What it writes

- `<output_csv>` — same header row, `upsampled_prompt` JSON replaces the original prompt column.
- `<output_csv without .csv>.json` — full benchmark JSON with `benchmark[]` (id / prompt / upsampled_prompt / extra_info) and a `settings` block.

## Step 1 — collect information

| Arg                        | Default                                                                    | Notes                                                                                                                |
|----------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `--input_csv`              | `s3://nv-00-10206-vfm/debug/xingqianx/evaluation/unigenbench/v2_1170L.csv` | Source UGB CSV. Usually the default; ask user if a different benchmark version.                                      |
| `--input_credential_path`  | `credentials/gcs.secret`                                                   | Credential for reading the input CSV bucket.                                                                         |
| `--output_csv`             | *(required)*                                                               | S3 path for the output CSV. Ask user — e.g. `s3://nv-00-10206-vfm/debug/xingqianx/evaluation/unigenbench/<name>.csv` |
| `--output_credential_path` | `credentials/gcs.secret`                                                   | Credential for writing the output bucket.                                                                            |
| `--output_ensure_ascii`    | on (flag)                                                                  | Add this flag only if output must be ASCII-safe.                                                                     |
| `--model`                  | `gemini-3.1-pro`                                                           | Upsampler LLM. Ask user if a different model is requested.                                                           |
| `--upsampler_version`      | `v5`                                                                       | Upsampler prompt template version.                                                                                   |
| `--aspect_ratio`           | `1,1`                                                                      | Target aspect ratio passed to the upsampler.                                                                         |
| `--resolution`             | `768`                                                                      | Target resolution passed to the upsampler.                                                                           |
| `--num_concurrency`        | `128`                                                                      | Number of concurrent Gemini calls.                                                                                   |
| `--batch_size`             | `1170`                                                                     | Rows processed per batch loop iteration.                                                                             |
| `--max_try`                | `3`                                                                        | Retries per failed sample before dropping it.                                                                        |

## Step 2 — compose the formatted command and show user for confirmation

```bash
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
slaunch cpu 1x1 prompt_upsampling_ugb_<name> \
    projects/cosmos3/vfm/evaluation/text_to_image/prompt_upsampling_ugb.py \
    --input_csv s3://nv-00-10206-vfm/debug/xingqianx/evaluation/unigenbench/v2_1170L.csv \
    --input_credential_path credentials/gcs.secret \
    --output_csv <output_csv> \
    --output_credential_path credentials/gcs.secret \
    --model gemini-3.1-pro \
    --upsampler_version v5 \
    --aspect_ratio 1,1 \
    --resolution 768 \
    --num_concurrency 128 \
    --batch_size 1170 \
    --max_try 3
```

Omit args that equal their defaults to keep the command clean. Add `--output_ensure_ascii` only if the user requests it.

## Step 3 — launch

- **No silent run by yourself, confirmation is always required!**
- Ask the user which cluster to launch the command on.
- Sanity check if the run is duplicated (same run name); if duplicated, stop and inform the user.
- When user confirms and no duplication, use your skill `/ssh_run` to launch the run.

## Notes

- `1x1` CPU is sufficient — concurrency is handled internally via `--num_concurrency`.
- Samples that fail all `--max_try` retries are silently dropped from the output (not an error exit). Check `len(samples_result)` in the log to confirm coverage.
- The sidecar JSON is always written alongside the CSV at `<output_csv>.json` (extension swapped). It is safe to rerun — outputs are overwritten.
