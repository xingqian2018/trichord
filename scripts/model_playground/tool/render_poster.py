import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.base import Tool, migrate_files  # noqa: E402
from utils import image_conversion  # noqa: E402


CSS_DIM_RE = re.compile(r"(width|height):\s*(\d+)px", re.IGNORECASE)
LOCAL_IMG_SRC_RE = re.compile(r'(<img\b[^>]*\bsrc=["\'])([^"\']+)(["\'])', re.IGNORECASE)
FALLBACK_CANVAS_SIZE = (1080, 1350)
RENDER_TARGET_AREA = 2048 * 2048
CHROME_CANDIDATES = ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]


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
        path = shutil.which(name)
        if path:
            return path
    return None


def render_html_to_png(html_code: str, png_path: Path, target_area: int = RENDER_TARGET_AREA) -> Path:
    chrome_bin = find_chrome_binary()
    if chrome_bin is None:
        raise RuntimeError(f"No headless Chrome/Chromium binary found (tried {CHROME_CANDIDATES}).")
    width, height = infer_canvas_size(html_code)
    scale = (target_area / (width * height)) ** 0.5
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
            f"--window-size={width},{height}",
            f"--force-device-scale-factor={scale}",
            f"--screenshot={png_path}",
            f"file://{html_path}",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0 or not png_path.exists():
            raise RuntimeError(f"Chrome render failed: {result.stderr.decode(errors='replace')}")
    return png_path



class render_poster(Tool):
    schema = {
        "type": "function",
        "function": {
            "name": "render_poster",
            "description": (
                "Render a complete HTML poster document to a PNG screenshot (about 2048 px on the long side) with "
                "headless Chrome and save it to disk. Returns the absolute path of the PNG, which can be passed to "
                "poster_visual_critic_agent. Call it on the finished HTML before finalizing the poster."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "html": {"type": "string", "description": "The complete HTML poster document to render"},
                    "output_path": {"type": "string", "description": "Absolute path to save the rendered PNG as, e.g. /tmp/poster_agent/poster_v1.png"},
                },
                "required": ["html", "output_path"],
            },
        },
    }

    def __init__(self, scratch_root: Optional[str] = None):
        super().__init__()
        self.scratch_root = Path(scratch_root) if scratch_root else None
        self.images: list[str] = []
        self.html_path: str | None = None

    def render(self, html: str, output_path: str) -> str:
        png_path = render_html_to_png(html, Path(output_path))
        Path(png_path).with_suffix(".html").write_text(html)
        return str(png_path)

    def run(self, args: dict) -> str:
        out = Path(args["output_path"])
        if self.scratch_root is not None and (not out.is_absolute() or not out.resolve().is_relative_to(self.scratch_root)):
            raise ValueError(f"the target save path {out} is not available; use an absolute path under {self.scratch_root}/")
        path = self.render(args["html"], str(out))
        html_path = str(Path(path).with_suffix(".html"))
        self.images.append(path)
        self.html_path = html_path
        return f"[render_poster saved PNG to {path}; the exact HTML that was rendered is saved to {html_path}]"

    def produced_files(self) -> list[str]:
        return [*self.images, *([self.html_path] if self.html_path else [])]

    def save_history(self, path: str | Path) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        if self.result_text is not None:
            (target / "result.txt").write_text(self.result_text)
        moved = migrate_files(target, self.produced_files())
        self.images[:] = [moved.get(x, x) for x in self.images]
        if self.html_path:
            self.html_path = moved.get(self.html_path, self.html_path)
