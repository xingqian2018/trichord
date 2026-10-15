import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.generate_image import generate_image  # noqa: E402

DEFAULT_MODEL_NAME = "LeptonTR/Cosmos3-Super-Text2Image@lepton_cosmos3_t2i"
COSMOS3_ASPECT_RATIO_TO_SIZE = {
    "1:1": "1024x1024",
    "4:3": "1184x880",
    "3:4": "880x1184",
    "16:9": "1360x768",
    "9:16": "768x1360",
}


# vLLM-Omni per-request override: skips both the prompt check (blocklist + Qwen3Guard) and the face pixelation of the output
COSMOS3_EXTRA_PARAMS = {"guardrails": False}


class generate_image_cosmos3(generate_image):
    aspect_ratio_to_size = COSMOS3_ASPECT_RATIO_TO_SIZE
    gateway_timeout = 1800

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, scratch_root: Optional[str] = None):
        super().__init__(model_name=model_name, scratch_root=scratch_root)
        build_request = self.gateway.build_request

        def build_request_cosmos3(*args, **kwargs) -> dict:
            request = build_request(*args, **kwargs)
            request["extra_body"] = {**request.get("extra_body", {}), "extra_params": dict(COSMOS3_EXTRA_PARAMS)}
            return request

        self.gateway.build_request = build_request_cosmos3

    def setting(self) -> dict:
        return {**super().setting(), "extra_params": COSMOS3_EXTRA_PARAMS}
