import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import SCRATCH_ROOT, render_html_to_png, resolve_output_path  # noqa: E402

OUTPUT_DIR = SCRATCH_ROOT / "rendered_posters"


class render_poster:
    SCHEMA = {
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
                    "output_path": {"type": "string", "description": "Filename to save the rendered PNG as, e.g. poster_v1.png"},
                },
                "required": ["html", "output_path"],
            },
        },
    }

    def render(self, html: str, output_path: str) -> str:
        png_path = render_html_to_png(html, Path(output_path))
        Path(png_path).with_suffix(".html").write_text(html)
        return str(png_path)

    def run(self, args: dict) -> dict:
        out = resolve_output_path(args.get("output_path"), OUTPUT_DIR, ".png")
        path = self.render(args["html"], str(out))
        html_path = str(Path(path).with_suffix(".html"))
        return {
            "text": f"[render_poster saved PNG to {path}; the exact HTML that was rendered is saved to {html_path}]",
            "images": [path],
            "html_path": html_path,
        }
