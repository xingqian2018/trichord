# S3 path lookup & I/O (s3_omni.py)

## Purpose

Perform S3 object-store I/O using the existing helper at `~/Project/bashrc/s3_omni.py`. **Always prefer this helper over `aws s3`, `boto3`, or `s5cmd`** — it handles the project's multi-profile endpoints (AWS, PBSS, GCS) and the custom `profile:bucket/prefix` path syntax.

## Path syntax

All S3/GCS paths use the form:

```
<profile>:<bucket>/<prefix-or-key>
```

Supported profiles (from `s3_omni.py`):

| Profile | Endpoint | Notes |
|---|---|---|
| `s3-training` | s3.amazonaws.com | AWS — checkpoints |
| `team-dir` / `team-dir-share` | pbss.s8k.io | PBSS us-east |
| `team-cosmos` / `team-cosmos-benchmark` / `team-dir-pdx` | pdx.s8k.io | PBSS PDX |
| `gcs` | storage.googleapis.com | GCS |

Examples:
- `gcs:nv-00-10206-vfm/debug/xingqianx/foo.png`
- `team-cosmos:cosmos_generation/animal/webdataset/full_v2_train_1219/v0/`
- `s3-training:checkpoints-us-east-1/run123/iter_001000/`

## Frequently used paths

Curated from `s3hint`. When the user asks about one of these by short name (e.g. "cvtg", "unigenbench", "coyo"), resolve to the full path below before running an op.

**Current working roots (the user's "latest sharded data" lives here):**
- Real data → `gcs:nv-00-10206-vfm/webdataset_image_text_related/<dataset_name>/`
- Synthetic data → `gcs:nv-00-10206-vfm/webdataset_synthetic/<dataset_name>/`

When the user asks "where did my latest data shard to?" / "where's my newest webdataset?", `ls` these two roots and pick the most recent `<dataset_name>` (often dated, e.g. `wordnet_captions_20260224`).

| Purpose | Path | Comment |
|---|---|---|
| Eval bench | `gcs:nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/image_captioning/CosCapBench/v1/general/xingqianx_x0` | CosCapBench image-captioning eval outputs |
| Eval bench | `gcs:nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/cvtg/` | CVTG text-to-image eval |
| Eval bench | `gcs:nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/unigenbench/v2_1170L_G3F/` | Unigenbench v2 (1170L_G3F) eval |
| Experiment | `gcs:nv-00-10206-checkpoint-experiments/cosmos3_vfm/t2i_mot_expDAITR/` | T2I MOT expDAITR run series (e.g. `004`, `005`) |
| Debug | `gcs:nv-00-10206-vfm/debug/xingqianx/evaluation/` | Personal debug eval outputs |
| Debug | `gcs:nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/` | CosCapBench debug images |
| Debug | `gcs:nv-00-10206-vfm/debug/xingqianx/synthetic_scene_text/` | Synthetic scene-text debug |
| Debug | `gcs:nv-00-10206-vfm/debug/xingqianx/synthetic_chinese_scene_text_v0/` | Chinese synthetic scene-text debug |
| Debug | `gcs:nv-00-10206-vfm/debug/xingqianx/vfm_aid/` | VFM AID debug |
| Logged | `gcs:nv-00-10206-images/logged_images/synthetic_scene_text_v0/` | Logged synthetic scene-text images |
| Logged | `gcs:nv-00-10206-images/logged_metas/synthetic_scene_text_v0/` | Logged synthetic scene-text metadata |
| Logged | `gcs:nv-00-10206-images/debug/logged_images/synthetic_scene_text_v0` | Debug variant of logged scene-text images |
| LanceDB | `gcs:nv-00-10206-lancedb/prod/image/synthetic_scene_text/synthetic_scene_text_v0.lance/` | Prod LanceDB for synthetic scene-text v0 |
| Webdataset | `gcs:nv-00-10206-webdataset-images/webdataset_image_text_related/` | Image-text-related webdataset |
| Webdataset | `gcs:nv-00-10206-webdataset-images/webdataset_image_v5/coyo_700m/` | COYO-700M webdataset (v5) |
| Webdataset | `gcs:nv-00-10206-webdataset-images/webdataset_image_v5_filter_highquality_ablation_40M/` | High-quality 40M ablation webdataset |
| Webdataset | `gcs:nv-00-10206-webdataset-images/webdataset_synthetic/synthetic_scene_text/` | Synthetic scene-text webdataset |
| Webdataset | `gcs:nv-00-10206-webdataset-images/webdataset_synthetic/synthetic_chinese_scene_text_v0/` | Chinese synthetic scene-text webdataset |
| Text webdataset real data root path | `gcs:nv-00-10206-vfm/webdataset_image_text_related/` | This is our latest text webdataset location |
| Text webdataset sgd data root path | `gcs:nv-00-10206-vfm/webdataset_synthetic/` | This is our latest text webdataset location |
| PBSS bench | `team-cosmos-benchmark:datasets/cvtg/cvtg_2kl/` | CVTG 2k-L benchmark dataset |
| PBSS bench | `team-cosmos-benchmark:datasets/unigenbench/` | Unigenbench dataset |
| PBSS bench | `team-cosmos-benchmark:datasets/vfm_aid/` | VFM AID benchmark dataset |
| PBSS train | `team-cosmos:cosmos_generation/animal/webdataset/full_v2_train_1219/v0/` | Animal training webdataset |
| PBSS train | `team-cosmos:cosmos_generation/faces/ohv/data/20260117_720p_16fps_qwen3_235b_fp8/` | OHV faces training data |
| PBSS train | `team-cosmos:cosmos_generation/faces/pexels/data/20260106_720p_16fps_qwen3_235b_fp8/` | Pexels faces training data |
| PBSS train | `team-cosmos:cosmos_generation/food_cutting_v2/search_with_curation_20251125_235012/` | Food-cutting v2 training data |
| PBSS train | `team-cosmos:cosmos_generation/sports/sports_curated/sharded_qwen3/v0/` | Curated sports training data |

If the user asks for a short name not in this table, run `python ~/Project/bashrc/s3_omni.py hint` to refresh — the live list may have grown.

## Webdataset path layout

When an S3 path points at a webdataset-formatted image dataset (any of the "Webdataset" rows above, or anything under the working roots), it follows this convention:

```
<root>/<dataset_name>/<bucket-path>/<keys>/<subpath>/<tarid>.tar
```

Segment meanings:

| Segment | What it is | Example |
|---|---|---|
| `<root>` | Collection root holding many datasets side-by-side | `gcs:nv-00-10206-vfm/webdataset_image_text_related/`, `gcs:nv-00-10206-vfm/webdataset_synthetic/` |
| `<dataset_name>` | One dataset (often dated) | `wordnet_captions_20260224`, `synthetic_scene_text_v0` |
| `<bucket-path>` | Bucketed split — **one or more** path components. Axes: category / resolution / aspect-ratio / (duration, for video). For image webdatasets the duration axis is absent. | `landscape/720p/16x9/`, `general/480p/1x1/` |
| `<keys>` | The data keys themselves. For image webdatasets the keys are images (and their sidecar captions) packed inside tars. | `keys/` |
| `<subpath>/<tarid>.tar` | Remaining path to the addressable tar shard, **one or more** path components, typically `partXXXXX/XXXXX.tar` | `part00003/00042.tar` |

Quick orientation:
- **Find a dataset's buckets**: `ls <root>/<dataset_name>/` and walk down (each level peels off one bucket axis).
- **Peek at one shard without pulling the whole bucket**: `dl <full_tar_path> /tmp/peek.tar` then untar locally.
- **Count tars in a bucket**: `cnt <root>/<dataset_name>/<bucket-path>/` (caps at 11k — see `cnt` op).
- A path that ends mid-bucket (e.g. `<root>/<dataset_name>/landscape/`) is a **prefix**, not a shard — `ls` it, don't `dl` it as a single object.

When the user gives a partial webdataset path, identify which segment it stops at (root / dataset / bucket / keys / shard) before deciding what op to run.

## Inspecting a webdataset's keys + tar counts

For any "what's in this webdataset / how many tars per key" question, **prefer the dedicated helper** over hand-rolled `ls`/`cnt` walks:

```bash
~/Project/trichord/helper/webds_tarcnt_by_key.py <dataset_name_or_full_path>
```

- Accepts a bare `<dataset_name>` (auto-resolves under both working roots) or a full `gcs:.../<dataset>` path.
- Walks the tree, finds every `wdinfo.json`, treats each containing dir as a "bucket leaf", and reports tar counts per key (`images/`, `metas/`, `metas_YYYYMMDD/`, …) — both totals and a per-leaf breakdown.
- Filters out the parallel `wdinfo/` mirror tree so you don't see noise rows.

Use this whenever the goal is *information gathering* about a webdataset's structure or completeness. Fall back to raw `ls`/`cnt` only when the helper can't express what you need (e.g. byte-size totals, non-tar files, custom filters).

## Helper invocation

The script lives at `~/Project/bashrc/s3_omni.py`. Shell aliases exist (`s3ls`, `s3dl`, `s3autodl`, `s3cnt`, `s3hint`, `s3a`) but the portable form is:

```bash
python ~/Project/bashrc/s3_omni.py <op> <args...> [--maxjob 64]
```

### Operations

| Op | Usage | What it does |
|---|---|---|
| `ls` | `ls [profile:bucket/prefix]` | List buckets / prefixes / files. No arg → list profiles. |
| `cnt` | `cnt profile:bucket/prefix` | Count entries under prefix (caps at ~11k, prints `11000+`). |
| `dl` | `dl <s3_src> <local_dest>` | Download file or folder. Same-size files skipped. |
| `autodl` | `autodl <s3_src>` | Download to `~/.cache/imaginaire4/<auto-category>/` based on src pattern. |
| `ul` | `ul <local_src> <s3_dest>` | Upload file or folder. Same-size files skipped. |
| `cp` | `cp <s3_src> <s3_dest>` | Copy S3→S3. Same profile: server-side `copy()`. Cross profile (e.g. `team-cosmos:...` → `gcs:...`): streams via `get_object` → `put_object`, with multipart for files >8MB. Same-size keys are skipped; non-empty destinations trigger an overwrite prompt. |
| `rm` | `rm <s3_src>` | Delete file or all keys under prefix. **Prompts for confirmation.** |
| `hint` | `hint` | Print the curated hint list (`s3_omni_hint.txt`). Good starting point for "where do X live?" |
| `hintadd` / `hintrm` | `hintadd <path>` / `hintrm <path>` | Manage the hint list. |
| `auto <path>` | `auto profile:...` | Best-effort: treat arg as a path and run `ls`. Useful when unsure. |

Cross-profile `cp` works (streams through this machine — not server-side), so you no longer need to `dl` → `ul` manually for the `team-cosmos:...` ↔ `gcs:...` case.

There is no CLI `cpf` op anymore. For force-overwrite, call from Python: `s3cp([src, dest], max_workers=64, force=True)`.

## Execution rules

1. **Always run the helper — never roll your own boto3 call.** The profile → endpoint mapping is non-obvious, and re-implementing it will silently hit the wrong backend.
2. **Parse the user's request into exactly one `<op>` + args**, then run `python ~/Project/bashrc/s3_omni.py <op> ...`. Use absolute path so it works from any CWD.
3. **Paths must include the profile prefix.** If the user gives a bare `bucket/key` without a profile, ask which profile. Do not guess.
4. **Destructive ops (`rm`, `cpf`, overwriting `ul`/`dl`):** show the exact command and the target before running. For `rm`, the helper itself prompts — let the prompt through (do not auto-answer `y`).
5. **Interactive prompts inside the helper:** `s3dl`, `s3ul`, `s3rm` may ask merge/nest/overwrite questions. Run the command in the foreground so the user can answer. Do not pipe `echo y |` unless the user explicitly asked for non-interactive behavior.
6. **Large transfers:** default `--maxjob 64`. Raise to 128–256 only if the user wants more parallelism; lower for shared nodes.
7. **When the user is exploring ("what's in X?"):** run `ls`, then if they want more, run `cnt` or `hint`. Don't chain long pipelines — the helper's output is already paginated/capped.
8. **Never invoke the `vim` op.** It requires an interactive terminal which this skill cannot drive. When the user says "open that file" / "show me what's in <s3-file>" / "read <s3-file>": `dl` the file to a temp path (e.g. `/tmp/s3io_<basename>`), then use the `Read` tool on that local path. For binary/images, `dl` + appropriate local tool. Do not attempt a write-back unless the user explicitly asks you to `ul` the edited file.

## Parsing examples

| User says | Run |
|---|---|
| "list gcs:nv-00-10206-vfm/debug/xingqianx/" | `python ~/Project/bashrc/s3_omni.py ls gcs:nv-00-10206-vfm/debug/xingqianx/` |
| "how many files in <path>" | `python ~/Project/bashrc/s3_omni.py cnt <path>` |
| "download <s3> to ./out/" | `python ~/Project/bashrc/s3_omni.py dl <s3> ./out/` |
| "autodl <s3>" | `python ~/Project/bashrc/s3_omni.py autodl <s3>` |
| "upload ./model.pt to <s3-folder>/" | `python ~/Project/bashrc/s3_omni.py ul ./model.pt <s3-folder>/` |
| "copy s3 folder A to folder B (same or different profile)" | `python ~/Project/bashrc/s3_omni.py cp <A> <B>` |
| "copy <s3-A> to <s3-B> across profiles" | `python ~/Project/bashrc/s3_omni.py cp <A> <B>`  (streams through local machine — slower than intra-profile) |
| "delete <s3-prefix>" | `python ~/Project/bashrc/s3_omni.py rm <s3-prefix>` (confirm at prompt) |
| "show me the hint list" / "common paths" | `python ~/Project/bashrc/s3_omni.py hint` |
| "read <s3-file>" / "open <s3-file>" / "show me what's in <s3-file>" | `python ~/Project/bashrc/s3_omni.py dl <s3-file> /tmp/s3io_<basename>` then `Read` that local path. **Do not use the `vim` op.** |

## Python use from code

If the user wants to call these operations from Python (not shell), import directly:

```python
import sys
sys.path.insert(0, "/home/xingqian/Project/bashrc")
from s3_omni import s3ls, s3dl, s3ul, s3cp, s3rm, s3ls_core, parse_s3input

# Examples:
s3dl(["gcs:my-bucket/path/file.pt", "./local.pt"], max_workers=64)
s3ul(["./out_dir", "gcs:my-bucket/dst/"], max_workers=64)
s3cp(["team-cosmos:bkt/src/", "gcs:bkt/dst/"], max_workers=64, force=True)  # cross-profile, no overwrite prompt
entries, truncated = s3ls_core(["gcs:my-bucket/prefix/"])
```

Core building blocks (for custom pipelines):
- `parse_s3input(raw)` → `(profile, bucket, prefix)`
- `make_async_s3client(profile, endpoint_url, region)` → async client factory
- `list_keys_with_size(client, bucket, prefix)` → `[(key, rel, size), ...]`
- `list_local_with_size(path)` → local counterpart
- `s3key_check(client, bucket, prefix)` → `'file' | 'folder' | None`

Prefer the top-level ops (`s3dl`, `s3ul`, …) unless you need per-object control.

## When NOT to use this skill

- S3-compatible storage outside the profiles above (MinIO, custom endpoints): use `boto3` directly.
- Streaming reads (e.g. reading a parquet shard into memory without writing to disk): use `s3fs` / `fsspec` directly — `s3_omni.py` is download/upload oriented.
- Generating presigned URLs or managing IAM: use `aws` CLI.

## Output conventions

- The helper prints `[OK]`, `[SKIP]`, `[FAIL]` per file, then a final `Done. Success: X, Failures: Y, Skipped: Z, Total: ...`. Relay that summary — don't reprint every line.
- On `FAIL`, surface the exact error line to the user so they can act on it (permissions, wrong profile, bucket typo).
