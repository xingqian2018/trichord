import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.base import Agent  # noqa: E402
from utils import image_conversion  # noqa: E402
from tool.generate_image import generate_image  # noqa: E402
from agent.poster_visual_critic import poster_visual_critic_agent  # noqa: E402
from tool.render_poster import render_poster  # noqa: E402

IMG_SRC_RE = re.compile(r'(<img\b[^>]*\bsrc=["\'])([^"\']+)(["\'])', re.IGNORECASE)
DEFAULT_MODEL_NAME = "kimi-k3"
PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt"
DEFAULT_SYSTEM_PROMPT = (PROMPT_DIR / "agentic_poster_layout_t2i_required_lite.md").read_text().strip()


class poster_generation(Agent):
    schema = {
        "type": "agent",
        "agent": {
            "name": "<placeholder>",
            "duty": "poster_generation",
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
                    "image_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional reference images (logos, photos, prior drafts) the designer should look at: absolute paths, file://, http(s), or data: URLs",
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
            tool_registry=[generate_image, render_poster],
            agent_registry=[poster_visual_critic_agent],
            **kwargs,
        )
        self.cached_html: Optional[str] = None
        self.cached_image: Optional[Image.Image] = None

    def append(self, msg: dict[str, Any], extended: dict[str, Any]) -> None:
        super().append(msg, extended)
        if extended.get("role") != "assistant":
            return
        for call in extended.get("tool_calls") or []:
            if call["type"] == "function" and call["function"]["name"] == render_poster.schema["function"]["name"]:
                self.cached_html = json.loads(call["function"]["arguments"]).get("html")
            elif call["type"] == "agent" and poster_visual_critic_agent.schema["agent"]["duty"] in call["agent"].values():
                image_url = json.loads(call["agent"]["arguments"]).get("image_url")
                if image_url:
                    try:
                        self.cached_image = image_conversion(image_url, dst_fmt="pil")
                    except Exception:
                        pass

    def save_history(self, path: str | Path) -> None:
        super().save_history(path)
        root = Path(path) / self.agent_id.replace("(", "").replace(")", "")
        self.save_derived(root / "derived")

    def save_derived(self, derived: Path) -> None:
        if self.cached_html is None:
            return
        derived.mkdir(parents=True, exist_ok=True)
        assets = derived / "assets"

        def relink(match):
            src = match.group(2)
            candidate = Path(src[7:] if src.startswith("file://") else src)
            if not candidate.is_absolute() or not candidate.is_file():
                return match.group(0)
            assets.mkdir(exist_ok=True)
            dst = assets / candidate.name
            if not dst.exists():
                shutil.copy2(candidate, dst)
            return f"{match.group(1)}assets/{dst.name}{match.group(3)}"

        (derived / "poster.html").write_text(IMG_SRC_RE.sub(relink, self.cached_html))
        if self.cached_image is not None:
            self.cached_image.save(derived / "poster.png")
