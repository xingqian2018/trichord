import re
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.base import Agent  # noqa: E402
from tool.base import Tool  # noqa: E402
from tool.generate_image import generate_image  # noqa: E402
from agent.poster_visual_critic_agent import poster_visual_critic_agent  # noqa: E402
from tool.render_poster import render_poster  # noqa: E402
from utils import ALLOWED_OUTPUT_ROOT  # noqa: E402

IMG_SRC_RE = re.compile(r'(<img\b[^>]*\bsrc=["\'])([^"\']+)(["\'])', re.IGNORECASE)
DEFAULT_MODEL_NAME = "kimi-k3"
PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt"
DEFAULT_SYSTEM_PROMPT = (PROMPT_DIR / "agentic_poster_layout_t2i_required_lite.md").read_text().strip()


def produced_files(tool: Optional[Tool]) -> list[str]:
    getter = getattr(tool, "produced_files", None)
    return list(getter()) if callable(getter) else []


class poster_generation(Agent):
    schema = {
        "type": "agent",
        "agent": {
            "type": "agent",
            "name": "<placeholder>",
            "description": (
                "End-to-end poster designer. Given a brief, it writes the poster as HTML, generates any needed imagery, "
                "renders the page to a PNG, has the result visually reviewed, and iterates until the poster is ready. "
                "Returns the final HTML plus the absolute paths of the rendered poster image and any generated assets."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The brief: subject, audience, tone, required text, and the target aspect ratio (1:1, 4:3, 3:4, 16:9, 9:16)",
                    },
                    "image_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional absolute paths of reference images (logos, photos, prior drafts) the designer should look at",
                    },
                },
                "required": ["prompt"],
            },
        },
    }

    def __init__(self, agent_name: str, model_name: str = DEFAULT_MODEL_NAME, system_prompt: str = DEFAULT_SYSTEM_PROMPT, **kwargs: Any):
        super().__init__(
            agent_name,
            model_name,
            system_prompt,
            tool_registry={cls.__name__: cls for cls in (generate_image, render_poster)},
            agent_registry={poster_visual_critic_agent.__name__: poster_visual_critic_agent},
            **kwargs,
        )
        self.last_rendered_html_path: Optional[str] = None
        self.cached_html_path: Optional[str] = None
        self.cached_image_path: Optional[str] = None

    def run(self, args: dict[str, Any]) -> str:
        images = list(args.get("images") or [])
        images += [Image.open(p) for p in args.get("image_paths") or []]
        return super().run({"prompt": args.get("prompt"), "images": images})

    def add_tool_result(self, call_id: str, system_id: str, text: str, extended_text: Optional[str] = None) -> None:
        super().add_tool_result(call_id, system_id, text, extended_text)
        tool = self.tool_instances.get(system_id)
        if getattr(tool, "html_path", None):
            self.last_rendered_html_path = tool.html_path
        elif isinstance(tool, poster_visual_critic_agent) and tool.files:
            self.cached_image_path = tool.files[-1]
            self.cached_html_path = str(Path(self.cached_image_path).with_suffix(".html"))

    def save_history(self, path: str | Path) -> None:
        root = Path(path)
        before = {cid: produced_files(h) for cid, h in self.tool_instances.items()}
        super().save_history(root)
        moved: dict[str, str] = {}
        for cid, handler in self.tool_instances.items():
            moved.update({old: new for old, new in zip(before[cid], produced_files(handler)) if old != new})

        for attr in ("last_rendered_html_path", "cached_html_path", "cached_image_path"):
            value = getattr(self, attr)
            if value in moved:
                setattr(self, attr, moved[value])

        self.save_derived(root / "derived", moved)

        allowed = str(ALLOWED_OUTPUT_ROOT.resolve()) + "/"
        for old in moved:
            if old.startswith(allowed):
                Path(old).unlink(missing_ok=True)

    def save_derived(self, derived: Path, moved: dict[str, str]) -> None:
        if not self.cached_html_path or not Path(self.cached_html_path).is_file():
            return
        derived.mkdir(parents=True, exist_ok=True)
        assets = derived / "assets"

        def relink(match):
            src = match.group(2)
            local = Path(src[7:] if src.startswith("file://") else src)
            candidate = Path(moved.get(str(local), str(local)))
            if not candidate.is_absolute() or not candidate.is_file():
                return match.group(0)
            assets.mkdir(exist_ok=True)
            dst = assets / candidate.name
            if not dst.exists():
                shutil.copy2(candidate, dst)
            return f"{match.group(1)}assets/{dst.name}{match.group(3)}"

        (derived / "poster.html").write_text(IMG_SRC_RE.sub(relink, Path(self.cached_html_path).read_text()))
        if self.cached_image_path and Path(self.cached_image_path).is_file():
            shutil.copy2(self.cached_image_path, derived / "poster.png")
