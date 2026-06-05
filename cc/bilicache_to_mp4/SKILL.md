---
name: bilicache_to_mp4
description: Merge Bilibili offline cache folders (entry.json + audio.m4s + video.m4s) into MP4 files named after video titles, using ffmpeg stream-copy (no transcode). Handles a root folder containing multiple c_<hash> cache dirs.
user_invocable: true
allowed-tools:
  - Bash
---

## Purpose

Scan `~/Downloads/bilibili_cache` for Bilibili offline cache entries and merge each one
into a single `.mp4` under `~/Downloads/bilibili_media`. No re-encoding — pure stream copy.

Input layout:

```
~/Downloads/bilibili_cache/
  <folder>/
    c_<hash>/
      entry.json
      <subfolder>/
        audio.m4s
        video.m4s
```

Output mirrors the same relative path under the output root:

```
~/Downloads/bilibili_media/<folder>/c_<hash>/<title>.mp4
```

If the `.mp4` already exists it is skipped. Output directories are created automatically.

---

## Script

The logic lives in this skill's own `merge.py`. Always invoke it with an absolute path:

```
python3 /home/xingqianx/Project/trichord/cc/bilicache_to_mp4/merge.py <root_folder>
```

`merge.py` uses `os.system()` to call `ffmpeg` — no Python ffmpeg package required.

---

## Prerequisites

`ffmpeg` must be on PATH. Check with `ffmpeg -version`. If missing, tell the user:

> `ffmpeg` is not installed. Install it with:
> ```
> sudo apt install ffmpeg
> conda install -c conda-forge ffmpeg
> ```

---

## Step-by-step execution

### Step 1 — verify ffmpeg

```bash
ffmpeg -version
```

If that fails, show the install message above and stop.

### Step 2 — run merge.py

```bash
python3 /home/xingqianx/Project/trichord/cc/bilicache_to_mp4/merge.py
```

Stream the output to the user as it runs.

### Step 3 — report summary

After the script finishes, count lines containing `[done]`, `[skip]`, and `[fail]` in the output and show a one-line summary:

```
Merged: X  |  Skipped: Y  |  Failed: Z
```

If any `[fail]` lines appeared, show them verbatim so the user can act on them.

---

## What this skill does NOT do

- Does not delete or move the original `.m4s` files.
- Does not re-encode — quality is identical to the source streams.
- Does not handle DRM-encrypted streams.
