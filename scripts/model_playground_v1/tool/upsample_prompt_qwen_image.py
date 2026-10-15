import json
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slurm_pool import SlurmPool  # noqa: E402
from tool.generate_image_qwen_image import WH_RATIO_TO_SIZE  # noqa: E402
from tool.tool import Tool, ToolCallError  # noqa: E402

QWEN_IMAGE_PE_MODEL = "Qwen/Qwen-Image-2.1-PE-T2I"
PE_SYSTEM_PROMPT = (Path(__file__).resolve().parents[1] / "prompt" / "t2i_generation_qwen2p1_prompt_enhancement.md").read_text().strip()
PE_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "max_tokens": 8192, "extra_body": {"top_k": 20}}
PE_MAX_ATTEMPTS = 3
DEFAULT_ASPECT_RATIO = "1:1"


def parse_rewrite(text: str) -> dict[str, str]:
    answer = text.rpartition("</think>")[2].strip()
    result = json.loads(answer[answer.find("{"): answer.rfind("}") + 1])
    if not isinstance(result.get("rewritten_prompt"), str) or not result["rewritten_prompt"].strip():
        raise ValueError("rewriter returned no rewritten_prompt")
    return result


class upsample_prompt_qwen_image(Tool):
    """Qwen-Image-2.1-PE-T2I as a tool: a short image request in, a detailed description and a suggested aspect ratio out."""

    schema = {
        "type": "function",
        "function": {
            "name": "upsample_prompt",
            "description": (
                "Expand a short image request into a detailed description of the finished image and suggest an aspect ratio. "
                "Returns JSON with upsampled_prompt and aspect_ratio."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The short image request to expand"},
                },
                "required": ["prompt"],
            },
        },
    }
    allow_parallel_calls = False

    def __init__(self):
        super().__init__()
        self.rewriter = SlurmPool("QPE", "SLURM_QWEN_IMAGE_PE_API_KEY", timeout=600)
        self.output: Optional[dict[str, Any]] = None

    def rewrite(self, prompt: str) -> dict[str, str]:
        def query(client: Any) -> str:
            response = client.chat.completions.create(
                model=QWEN_IMAGE_PE_MODEL,
                messages=[{"role": "system", "content": PE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                **PE_SAMPLING,
            )
            return response.choices[0].message.content or ""

        error: Optional[Exception] = None
        for _ in range(PE_MAX_ATTEMPTS):
            try:
                return parse_rewrite(self.rewriter.call(query))
            except (json.JSONDecodeError, ValueError) as e:
                error = e
        raise RuntimeError(f"prompt rewriter gave no valid JSON after {PE_MAX_ATTEMPTS} attempts: {error}")

    def run_core(self, args: dict[str, Any]) -> str:
        prompt = (args.get("prompt") or "").strip()
        if not prompt:
            raise ToolCallError("missing required argument 'prompt'; call upsample_prompt again with the image request")
        rewrite = self.rewrite(prompt)
        ratio = rewrite.get("wh_ratio") if rewrite.get("wh_ratio") in WH_RATIO_TO_SIZE else DEFAULT_ASPECT_RATIO
        self.output = {"prompt": prompt, "upsampled_prompt": rewrite["rewritten_prompt"], "aspect_ratio": ratio}
        return json.dumps({"upsampled_prompt": rewrite["rewritten_prompt"], "aspect_ratio": ratio}, ensure_ascii=False)

    def setting(self) -> dict[str, Any]:
        return {**super().setting(), "rewriter_model": QWEN_IMAGE_PE_MODEL}
