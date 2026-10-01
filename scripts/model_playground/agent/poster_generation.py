import difflib
import json
import os.path as osp
import re
import sys
from pathlib import Path
from typing import Any, Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.base import Agent  # noqa: E402
from utils import image_conversion, put  # noqa: E402
from tool.base import ToolCallError  # noqa: E402
from tool.generate_image import ASPECT_RATIO_TO_SIZE, generate_image  # noqa: E402
from agent.poster_visual_critic import poster_visual_critic_agent  # noqa: E402
from tool.render_poster import render_poster  # noqa: E402

IMG_SRC_RE = re.compile(r'(<img\b[^>]*\bsrc=["\'])([^"\']+)(["\'])', re.IGNORECASE)
HTML_BLOCK_RE = re.compile(r"```\s*html\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
HTML_MATCH_THRESHOLD = 0.98
HTML_MISMATCH_WARNING = (
    "The HTML in your final answer is not the HTML you last sent to render_poster, so it was never rendered or reviewed. "
    "Deliver exactly the HTML you last rendered, or render and review the new version first."
)
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
                    "aspect_ratio": {"type": "string", "enum": list(ASPECT_RATIO_TO_SIZE), "description": "Target aspect ratio of the poster"},
                    "image_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional reference images (logos, photos, prior drafts) the designer should look at: absolute paths, file://, http(s), or data: URLs",
                    },
                },
                "required": ["prompt", "aspect_ratio"],
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

    def run_core(self, args: dict[str, Any]) -> str:
        if args.get("prompt") is not None:
            aspect_ratio = args.get("aspect_ratio")
            if aspect_ratio not in ASPECT_RATIO_TO_SIZE:
                raise ToolCallError(f"aspect_ratio must be one of {list(ASPECT_RATIO_TO_SIZE)}, got {aspect_ratio!r}")
            args = {**args, "prompt": f"{args['prompt']}\n\nTarget aspect ratio: {aspect_ratio}"}
        super().run_core(args)
        if self.check_final_html() == "mismatch":
            self.add_user_message(HTML_MISMATCH_WARNING)
            super().run_core({})
            self.check_final_html()
        return self.messages[-1]["content"]

    def check_final_html(self) -> str:
        content = self.messages[-1].get("content") or ""
        blocks = HTML_BLOCK_RE.findall(content)
        if self.cached_html is None or not blocks:
            return "ok"
        delivered = blocks[-1]
        normalize = lambda s: re.sub(r"\s+", " ", s).strip()
        ratio = difflib.SequenceMatcher(None, normalize(delivered), normalize(self.cached_html)).ratio()
        if ratio == 1.0:
            return "ok"
        if ratio < HTML_MATCH_THRESHOLD:
            return "mismatch"
        corrected = content.replace(delivered, self.cached_html.strip())
        self.messages[-1]["content"] = corrected
        self.messages_extended[-1]["content"] = corrected
        return "corrected"

    def append(self, msg: dict[str, Any], extended: dict[str, Any]) -> None:
        super().append(msg, extended)
        if extended.get("role") != "assistant":
            return
        for call in extended.get("tool_calls") or []:
            try:
                args = json.loads(call["function"]["arguments"] if call["type"] == "function" else call["agent"]["arguments"])
            except json.JSONDecodeError:
                continue
            if call["type"] == "function" and call["function"]["name"] == render_poster.schema["function"]["name"]:
                self.cached_html = args.get("html")
            elif call["type"] == "agent" and call["agent"]["duty"] == poster_visual_critic_agent.schema["agent"]["duty"]:
                image_url = args.get("image_url")
                if image_url:
                    try:
                        self.cached_image = image_conversion(image_url, dst_fmt="pil")
                    except Exception:
                        pass

    def save_history_core(self, path: str) -> None:
        root = osp.join(path, self.agent_id.replace("(", "").replace(")", ""))
        self.save_derived(osp.join(root, "derived"))
        super().save_history_core(path)

    def save_derived(self, derived: str) -> None:
        if self.cached_html is None:
            return

        def relink(match):
            src = match.group(2)
            candidate = Path(src[7:] if src.startswith("file://") else src)
            if not candidate.is_absolute() or not candidate.is_file():
                return match.group(0)
            put(candidate.read_bytes(), osp.join(derived, "assets", candidate.name))
            return f"{match.group(1)}assets/{candidate.name}{match.group(3)}"

        put(IMG_SRC_RE.sub(relink, self.cached_html).encode("utf-8"), osp.join(derived, "poster.html"))
        if self.cached_image is not None:
            put(image_conversion(self.cached_image, dst_fmt="bytes"), osp.join(derived, "poster.png"))
