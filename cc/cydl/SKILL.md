---
name: cydl
description: Download a YouTube video to ~/Music/cydl/ using the bundled yt-dlp binary. Files are saved as "Title [VideoID].ext". Call with a YouTube URL or without one — the skill will ask.
user_invocable: true
allowed-tools:
  - Bash
---

## Purpose

Download a single YouTube video (or any yt-dlp-supported URL) to the default folder `~/Music/cydl/` with a clean, human-readable filename that embeds the YouTube video ID in square brackets.

---

## Binary

The `yt-dlp` executable lives inside this skill's own folder — always call it by absolute path:

```
/home/xingqian/Project/bashrc/cc_skills/cydl/yt-dlp
```

Do **not** use the `ydl` / `ydl_mp3` shell aliases or any system-wide `yt-dlp`. Use only the bundled binary above.

---

## Default download folder

```
~/Music/cydl/
```

Create it when it doesn't exist: `mkdir -p ~/Music/cydl/`

---

## Output filename template

All downloads use this yt-dlp template:

```
%(title)s [%(id)s].%(ext)s
```

Examples of resulting filenames:
- `Never Gonna Give You Up [dQw4w9WgXcQ].webm`
- `Bohemian Rhapsody (Official Video) [fJ9rUzIMcZQ].mp4`

---

## Step-by-step execution

### Step 1 — obtain a valid YouTube URL

If the user invoked `/cydl <url>` with a URL already provided, use that directly and skip to Step 2.

If no URL was given, ask the user:

> Please paste a valid YouTube URL (e.g. `https://www.youtube.com/watch?v=...` or `https://youtu.be/...`).

**Validate** the input: it must contain `youtube.com/watch` or `youtu.be/`. If it does not match, reply:

> That doesn't look like a valid YouTube link. Please provide a URL like `https://www.youtube.com/watch?v=XXXXXXXXXXX`.

Keep asking until a valid URL is supplied.

### Step 2 — run yt-dlp

```bash
mkdir -p ~/Music/cydl/
/home/xingqian/Project/bashrc/cc_skills/cydl/yt-dlp \
  --output "%(title)s [%(id)s].%(ext)s" \
  --paths ~/Music/cydl/ \
  --no-playlist \
  "<URL>"
```

Flag notes:
- `--output "%(title)s [%(id)s].%(ext)s"` — smart filename: title + `[videoID]` + extension.
- `--paths ~/Music/cydl/` — destination folder.
- `--no-playlist` — download the single video only, even if the URL contains a playlist parameter.
- yt-dlp automatically selects the best available quality (video + audio merged).

### Step 3 — report the result

**On success**, parse yt-dlp's output for the final filename:
- Look for a line like `[download] Destination: ...` or `[Merger] Merging formats into "..."`.
- Report:
  ```
  ✅ Downloaded: <filename>
  📁 Saved to:   ~/Music/cydl/<filename>
  ```

**On failure**, show the last several lines of yt-dlp stderr verbatim so the user can act on the error (geo-block, private video, age-gate, network issue, etc.).

---

## Optional overrides

| User says | What to do |
|---|---|
| "download to ~/Downloads/" | Replace `--paths ~/Music/cydl/` with `--paths ~/Downloads/` |
| "audio only" / "mp3" | Add `--extract-audio --audio-format mp3 --audio-quality 0` |
| "best quality / 4K" | yt-dlp already picks best by default; no change needed |
| "whole playlist" | Drop `--no-playlist`; keep the same output template |
| specific resolution (e.g. "720p") | Add `-f "bestvideo[height<=720]+bestaudio/best[height<=720]"` |

---

## What this skill does NOT do

- Does not modify or delete existing files in `~/Music/cydl/`.
- Does not upload or stream the video.
- Does not run in the background — download runs in the foreground so progress is visible.
