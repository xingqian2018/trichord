import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.base import Tool, migrate_files  # noqa: E402
from utils import SCRATCH_ROOT, render_html_to_png, resolve_output_path  # noqa: E402

OUTPUT_DIR = SCRATCH_ROOT / "rendered_posters"


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
                    "output_path": {"type": "string", "description": "Filename to save the rendered PNG as, e.g. poster_v1.png"},
                },
                "required": ["html", "output_path"],
            },
        },
    }

    def __init__(self):
        super().__init__()
        self.images: list[str] = []
        self.html_path: str | None = None

    def render(self, html: str, output_path: str) -> str:
        png_path = render_html_to_png(html, Path(output_path))
        Path(png_path).with_suffix(".html").write_text(html)
        return str(png_path)

    def run(self, args: dict) -> str:
        out = resolve_output_path(args.get("output_path"), OUTPUT_DIR, ".png")
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
