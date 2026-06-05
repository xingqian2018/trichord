import json
import os
from pathlib import Path

INPUT_ROOT = Path("~/Downloads/bilibili_cache").expanduser()
OUTPUT_ROOT = Path("~/Downloads/bilibili_media").expanduser()

ILLEGAL = r'/\:*?"<>|'


def sanitize(title):
    for ch in ILLEGAL:
        title = title.replace(ch, "_")
    return title.strip()


def find_file(root, name):
    for p in Path(root).rglob(name):
        return p
    return None


def merge_one(c_folder):
    c_folder = Path(c_folder)
    entry = c_folder / "entry.json"
    if not entry.exists():
        print(f"  [skip] no entry.json in {c_folder}")
        return

    title = sanitize(json.loads(entry.read_text(encoding="utf-8"))["title"])

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_ROOT / f"{title}.mp4"

    if output.exists():
        print(f"  [skip] already exists: {output}")
        return

    video = find_file(c_folder, "video.m4s")
    audio = find_file(c_folder, "audio.m4s")

    if not video or not audio:
        print(f"  [skip] missing streams in {c_folder} (video={video}, audio={audio})")
        return

    print(f"  [merge] {title}")
    cmd = (
        f'ffmpeg -y -i "{video}" -i "{audio}" '
        f'-c:v copy -c:a copy "{output}" -loglevel error'
    )
    ret = os.system(cmd)
    if ret == 0:
        print(f"  [done]  {output}")
    else:
        print(f"  [fail]  ffmpeg exited {ret} for {c_folder}")
        if output.exists():
            output.unlink()


def main():
    if not INPUT_ROOT.is_dir():
        print(f"input folder not found: {INPUT_ROOT}")
        return

    c_folders = sorted(INPUT_ROOT.rglob("c_*"))
    c_folders = [p for p in c_folders if p.is_dir() and (p / "entry.json").exists()]

    if not c_folders:
        print(f"no c_* cache folders found under {INPUT_ROOT}")
        return

    print(f"found {len(c_folders)} cache folder(s)")
    print(f"  input:  {INPUT_ROOT}")
    print(f"  output: {OUTPUT_ROOT}")
    for cf in c_folders:
        merge_one(cf)


if __name__ == "__main__":
    main()
