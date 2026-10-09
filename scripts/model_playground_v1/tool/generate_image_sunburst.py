import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.generate_image import generate_image  # noqa: E402

DEFAULT_MODEL_NAME = "openai/openai/gpt-image-2.5-sunburst@nvidiak"


class generate_image_sunburst(generate_image):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, scratch_root: Optional[str] = None):
        super().__init__(model_name=model_name, scratch_root=scratch_root)
