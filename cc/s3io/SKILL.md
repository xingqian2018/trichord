---
name: s3io
description: Read, list, upload, download, copy, and delete S3 files (AWS S3, PBSS, GCS) via ~/Project/bashrc/s3_omni.py. Handles multi-profile paths like profile:bucket/prefix. Use whenever the user asks to ls/dl/ul/cp/rm anything on S3/GCS.
user_invocable: true
allowed-tools:
  - Bash
  - Read
  - Write
---

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
| `vim` | `vim <s3_file>` | Download file to a tmp path and open in vim (read-mostly; does NOT auto re-upload). |
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
8. **When the user says "open that file":** use `vim` for text, `dl` + local tool for binary/images. Remember `vim` does not auto re-upload — warn the user if they edit and expect a write-back.

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
| "open <s3-file> in vim" | `python ~/Project/bashrc/s3_omni.py vim <s3-file>` |

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
