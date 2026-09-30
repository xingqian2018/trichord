import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.agent import EMPTY_REPLY, Agent  # noqa: E402

DEFAULT_MODEL_NAME = "kimi-k3"
PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt"
SYSTEM_PROMPT = (PROMPT_DIR / "poster_visual_critic.md").read_text().strip()


class poster_visual_critic_agent(Agent):
    SCHEMA = {
        "type": "function",
        "function": {
            "name": "poster_visual_critic_agent",
            "description": (
                "Visual reviewer for a poster or generated poster image. Returns natural-language feedback: an overall take "
                "(ready to ship, ready with minor polish, or needs another pass), what is working, and for each issue "
                "where it is, why it matters, and a concrete suggestion. Only 'needs another pass' means the poster must be "
                "redone; it is reserved for major problems. Use it on a rendered poster image path before finalizing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The brief: what the poster is for, its intended audience, and what to focus the critique on",
                    },
                    "image_path": {
                        "type": "string",
                        "description": "Absolute path of the image to critique, as returned by generate_image or a render",
                    },
                },
                "required": ["prompt", "image_path"],
            },
        },
    }

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        super().__init__(
            model_name,
            SYSTEM_PROMPT,
            tools=None,
            max_tokens=4096,
            temperature=0.0,
            num_max_retry=1,
            timeout=180,
        )

    def critique(self, prompt: str, image: Image.Image) -> str:
        self.reset()
        self.add_user_message(prompt, images=[image])
        idx = self.step()
        content = self.messages[idx]["content"]
        if content == EMPTY_REPLY:
            raise RuntimeError(f"Critic returned no content (finish_reason={self.meta[idx].get('finish_reason')})")
        return content.strip()

    def run(self, args: dict) -> dict:
        text = self.critique(args["prompt"], Image.open(args["image_path"]))
        assistant = {"role": "assistant", "content": text}
        if self.meta[-1].get("reasoning"):
            assistant["thinking"] = self.meta[-1]["reasoning"]
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": args["prompt"]}, {"type": "image", "path": args["image_path"]}]},
            assistant,
        ]
        return {"text": text, "images": [], "chat": chat}
