import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.base import Agent  # noqa: E402
from tool.base import migrate_files  # noqa: E402

DEFAULT_MODEL_NAME = "kimi-k3"
PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt"
SYSTEM_PROMPT = (PROMPT_DIR / "poster_visual_critic.md").read_text().strip()


class poster_visual_critic_agent(Agent):
    schema = {
        "type": "agent",
        "agent": {
            "type": "agent",
            "name": "<placeholder>",
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

    def __init__(self, agent_name: str, model_name: str = DEFAULT_MODEL_NAME, **kwargs):
        super().__init__(
            agent_name,
            model_name,
            SYSTEM_PROMPT,
            max_tokens=4096,
            temperature=0.0,
            num_max_retry=1,
            timeout=180,
            **kwargs,
        )
        self.files: list[str] = []

    def critique(self, prompt: str, image: Image.Image) -> str:
        self.add_user_message(prompt, images=[image])
        self.step()
        content = self.messages[-1]["content"]
        if not content:
            raise RuntimeError("Critic returned no content")
        return content.strip()

    def run(self, args: dict) -> str:
        self.files.append(args["image_path"])
        return self.critique(args["prompt"], Image.open(args["image_path"]))

    def produced_files(self) -> list[str]:
        return list(self.files)

    def save_history(self, path: str | Path) -> None:
        super().save_history(path)
        moved = migrate_files(Path(path), self.files)
        self.files[:] = [moved.get(x, x) for x in self.files]
