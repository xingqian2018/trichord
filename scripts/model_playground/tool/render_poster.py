import json
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.base import Tool, ToolCallError, migrate_files  # noqa: E402
from tool.generate_image import ASPECT_RATIO_TO_SIZE  # noqa: E402
from utils import image_conversion, put  # noqa: E402


CSS_DIM_RE = re.compile(r"(width|height):\s*(\d+)px", re.IGNORECASE)
LOCAL_IMG_SRC_RE = re.compile(r'(<img\b[^>]*\bsrc=["\'])([^"\']+)(["\'])', re.IGNORECASE)
FALLBACK_CANVAS_SIZE = (1080, 1350)


PLAYWRIGHT_BROWSERS_ROOT = Path.home() / "Software" / "playwright-browsers"
PLAYWRIGHT_CHROME_PATTERNS = [
    "chromium_headless_shell-*/chrome-headless-shell-linux*/chrome-headless-shell",
    "chromium_headless_shell-*/chrome-linux*/headless_shell",
    "chromium-*/chrome-linux*/chrome",
]


def get_playwright_chrome_candidates() -> list[str]:
    return [str(p) for pattern in PLAYWRIGHT_CHROME_PATTERNS for p in sorted(PLAYWRIGHT_BROWSERS_ROOT.glob(pattern), reverse=True)]


CHROME_CANDIDATES = [
    *get_playwright_chrome_candidates(),
    str(Path.home() / "Software" / "chrome" / "google-chrome"),
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
]


def infer_canvas_size(html_code: str) -> tuple[int, int]:
    best = None
    for rule in html_code.split("}"):
        dims = dict((prop.lower(), int(px)) for prop, px in CSS_DIM_RE.findall(rule))
        if "width" in dims and "height" in dims:
            area = dims["width"] * dims["height"]
            if best is None or area > best[0]:
                best = (area, dims["width"], dims["height"])
    if best is None:
        return FALLBACK_CANVAS_SIZE
    return best[1], best[2]


def inject_reset_css(html_code: str, width: int, height: int) -> str:
    reset = (
        "<style>"
        "html,body{margin:0!important;padding:0!important;"
        f"width:{width}px!important;height:{height}px!important;overflow:hidden!important;}}"
        "</style>"
    )
    match = re.search(r"<head[^>]*>", html_code, re.IGNORECASE)
    if match:
        return html_code[: match.end()] + reset + html_code[match.end() :]
    match = re.search(r"<html[^>]*>", html_code, re.IGNORECASE)
    if match:
        return html_code[: match.end()] + f"<head>{reset}</head>" + html_code[match.end() :]
    return reset + html_code


def inline_local_images(html_code: str) -> str:
    def replace(match):
        src = match.group(2)
        path = Path(src[7:] if src.startswith("file://") else src)
        if not path.is_absolute() or not path.is_file():
            return match.group(0)
        data_url = image_conversion(str(path), src_fmt="url", dst_fmt="data_url")
        return match.group(1) + data_url + match.group(3)

    return LOCAL_IMG_SRC_RE.sub(replace, html_code)


def find_chrome_binary() -> str | None:
    for name in CHROME_CANDIDATES:
        path = name if Path(name).is_file() else shutil.which(name)
        if path:
            return path
    return None


def aspect_ratio_bucket(width: int, height: int, tolerance: float = 0.03) -> Optional[str]:
    for aspect_ratio in ASPECT_RATIO_TO_SIZE:
        target_w, target_h = (int(x) for x in aspect_ratio.split(":"))
        if abs(width / height - target_w / target_h) <= tolerance * (target_w / target_h):
            return aspect_ratio
    return None


def target_size(width: int, height: int) -> tuple[int, int]:
    bucket = aspect_ratio_bucket(width, height)
    if bucket is None:
        raise RuntimeError(f"canvas {width}x{height} matches none of the aspect ratio tiers {list(ASPECT_RATIO_TO_SIZE)}")
    target_width, target_height = ASPECT_RATIO_TO_SIZE[bucket].split("x")
    return int(target_width), int(target_height)


def render_html_to_png(html_code: str, png_path: Path) -> Path:
    chrome_bin = find_chrome_binary()
    if chrome_bin is None:
        raise RuntimeError(f"No headless Chrome/Chromium binary found (tried {CHROME_CANDIDATES}).")
    width, height = infer_canvas_size(html_code)
    target_width, target_height = target_size(width, height)
    scale = target_width / width
    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "poster.html"
        html_path.write_text(inject_reset_css(inline_local_images(html_code), width, height))
        cmd = [
            chrome_bin,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-crash-reporter",
            f"--user-data-dir={Path(tmpdir) / 'profile'}",
            f"--window-size={width},{height}",
            f"--force-device-scale-factor={scale}",
            f"--screenshot={png_path}",
            f"file://{html_path}",
        ]
        home = Path(tmpdir) / "home"
        home.mkdir()
        env = {**os.environ, "HOME": str(home), "XDG_CONFIG_HOME": str(home / ".config"), "XDG_CACHE_HOME": str(home / ".cache")}
        result = subprocess.run(cmd, capture_output=True, timeout=120, env=env)
        if result.returncode != 0 or not png_path.exists():
            raise RuntimeError(f"Chrome render failed: {result.stderr.decode(errors='replace')}")
    with Image.open(png_path) as image:
        if image.size != (target_width, target_height):
            image.resize((target_width, target_height), Image.LANCZOS).save(png_path, format="PNG")
    return png_path



class render_poster(Tool):
    schema = {
        "type": "function",
        "function": {
            "name": "render_poster",
            "description": (
                "Render a complete HTML poster document to a PNG screenshot (about 2048 px on the long side) with "
                "headless Chrome and save it to disk. Returns the absolute path of the PNG."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "html": {"type": "string", "description": "The complete HTML poster document to render"},
                    "output_path": {"type": "string", "description": "Absolute path to save the rendered PNG as, e.g. /tmp/poster_agent/poster_v1.png"},
                    "aspect_ratio": {"type": "string", "enum": list(ASPECT_RATIO_TO_SIZE), "description": "Target aspect ratio of the poster; the HTML canvas must match it"},
                },
                "required": ["html", "output_path", "aspect_ratio"],
            },
        },
    }

    def __init__(self, scratch_root: Optional[str] = None):
        super().__init__()
        self.scratch_root = Path(scratch_root) if scratch_root else None
        self.image_path_history: list[str] = []

    def run_core(self, args: dict) -> str:
        out = Path(args["output_path"])
        if self.scratch_root is not None and (not out.is_absolute() or not out.resolve().is_relative_to(self.scratch_root)):
            raise ToolCallError(f"the target save path {out} is not available; use an absolute path under {self.scratch_root}/")
        if out.exists():
            raise ToolCallError(f"the target save path {out} already exists; overwriting is not allowed, choose a new filename")
        html = args["html"]
        if not html or not html.strip():
            raise ToolCallError("html is empty; pass the complete HTML poster document")
        if infer_canvas_size(html) == FALLBACK_CANVAS_SIZE and not CSS_DIM_RE.search(html):
            raise ToolCallError("html declares no canvas size; the root poster element must set explicit width and height in px")
        aspect_ratio = args["aspect_ratio"]
        if aspect_ratio not in ASPECT_RATIO_TO_SIZE:
            raise ToolCallError(f"aspect_ratio must be one of {list(ASPECT_RATIO_TO_SIZE)}, got {aspect_ratio!r}")
        width, height = infer_canvas_size(html)
        if aspect_ratio_bucket(width, height) != aspect_ratio:
            raise ToolCallError(f"the HTML canvas is {width}x{height}, which is not approximately {aspect_ratio}; set the root element width and height so it can render a {aspect_ratio} canvas with minor resize.")
        path = str(render_html_to_png(html, out))
        self.image_path_history.append(path)
        return f"render_poster saved PNG to {path}"

    def save_history_core(self, path: str) -> None:
        setting = {
            "tool": type(self).__name__,
            "scratch_root": str(self.scratch_root) if self.scratch_root else None,
            "aspect_ratio_to_size": ASPECT_RATIO_TO_SIZE,
            "chrome_binary": find_chrome_binary(),
        }
        put(json.dumps(setting, ensure_ascii=False, indent=4).encode("utf-8"), osp.join(path, "setting.json"))
        if self.result_text is not None:
            put(self.result_text.encode("utf-8"), osp.join(path, "result.txt"))
        migrate_files(path, self.image_path_history)
