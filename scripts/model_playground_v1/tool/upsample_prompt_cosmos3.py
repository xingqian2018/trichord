import json
import os.path as osp
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool.agent import Agent  # noqa: E402
from tool.generate_image_cosmos3_with_strict_prompt_format_slurm import ASPECT_RATIO_TO_SIZE, TEMPLATE_KEYS, generate_image_cosmos3_with_strict_prompt_format_slurm  # noqa: E402
from tool.tool import Tool, ToolCallError  # noqa: E402

DEFAULT_MODEL_NAME = "kimi-k3@k3slurm"
UPSAMPLER_SYSTEM_PROMPT = (Path(__file__).resolve().parents[1] / "prompt" / "cosmos3_prompt_upsampling_v0.md").read_text().strip()
MAX_STEPS = 4
NUDGE_MESSAGE = "You replied with text. Call submit_prompt now through a function call; do not write the call as text."


class submit_prompt_cosmos3(Tool):
    """The upsampler's only tool: takes the structured JSON and checks it against the Cosmos3 T2I template (same checks as
    generate_image_cosmos3_with_strict_prompt_format_slurm); a violation goes back to the upsampler as a tool error."""

    schema = {
        "type": "function",
        "function": {
            "name": "submit_prompt",
            "description": "Submit the structured JSON description of the image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "object", "description": "The structured image description, with every top-level key of the template"},
                },
                "required": ["prompt"],
            },
        },
    }
    allow_parallel_calls = False

    def __init__(self):
        super().__init__()
        self.output: Optional[dict[str, Any]] = None

    def run_core(self, args: dict[str, Any]) -> str:
        prompt = args.get("prompt")
        if isinstance(prompt, str):
            try:
                prompt = json.loads(prompt)
            except json.JSONDecodeError:
                raise ToolCallError("prompt must be a JSON object following the template, not free text")
        if not isinstance(prompt, dict):
            raise ToolCallError("prompt must be a JSON object following the template")
        found = generate_image_cosmos3_with_strict_prompt_format_slurm.problems(prompt)
        if found:
            raise ToolCallError("the structured prompt does not follow the template: " + "; ".join(found) + ". Fix it and call submit_prompt again.")
        self.output = {k: prompt[k] for k in TEMPLATE_KEYS}
        return "submit_prompt accepted the structured prompt"


class cosmos3_prompt_upsampler(Agent):
    """Kimi-K3 as the Cosmos3 prompt upsampler (prompt/cosmos3_prompt_upsampling_v0.md, the official upsampling rules and
    T2I template): one request in, one template-valid structured JSON out through submit_prompt (tool_choice="required")."""

    tool_registry_classes = [submit_prompt_cosmos3]

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, system_prompt: str = UPSAMPLER_SYSTEM_PROMPT, **kwargs: Any):
        super().__init__(model_name, system_prompt, tool_registry=list(type(self).tool_registry_classes), **kwargs)

    def submit_tool(self) -> submit_prompt_cosmos3:
        return self.tool_instances[submit_prompt_cosmos3.schema["function"]["name"]]

    def build_request(self) -> dict[str, Any]:
        return {**super().build_request(), "tool_choice": "required"}

    def run_core(self, args: dict[str, Any]) -> str:
        message = args["prompt"] if not args.get("aspect_ratio") else f"{args['prompt']}\n\nAspect ratio: {args['aspect_ratio']}"
        self.add_user_message(message)
        for _ in range(MAX_STEPS):
            finish_reason = self.step()
            if finish_reason == "tool_calls":
                self.execute_tool_calls()
            else:
                self.add_user_message(NUDGE_MESSAGE)
            if self.submit_tool().output is not None:
                return json.dumps(self.submit_tool().output, ensure_ascii=False)
        raise RuntimeError(f"{self.model_name} gave no template-valid structured prompt within {MAX_STEPS} steps")


class upsample_prompt_cosmos3(Tool):
    """Cosmos3 prompt upsampling as a tool: a short image request (and optionally the aspect ratio of the frame) in, the
    template-valid Cosmos3 structured JSON out, written by a separate Kimi-K3 conversation (cosmos3_prompt_upsampler) that
    does not see the caller's context. Callers can expose it to an agent as upsample_prompt or call upsample() directly.
    The upsampler conversation of each call is saved under call/<id>/upsampler/."""

    schema = {
        "type": "function",
        "function": {
            "name": "upsample_prompt",
            "description": (
                "Expand a short image request into the dense structured JSON description that Cosmos3 renders from. "
                "Returns the structured JSON."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The short image request to expand"},
                    "aspect_ratio": {"type": "string", "enum": list(ASPECT_RATIO_TO_SIZE), "description": "Aspect ratio of the frame, if known"},
                },
                "required": ["prompt"],
            },
        },
    }
    allow_parallel_calls = False

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, reasoning_effort: Optional[str] = None):
        super().__init__()
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.upsamplers: list[tuple[str, cosmos3_prompt_upsampler]] = []
        self.output: Optional[dict[str, Any]] = None

    def upsample(self, request: str, aspect_ratio: Optional[str] = None, key: Optional[str] = None) -> dict[str, Any]:
        upsampler = cosmos3_prompt_upsampler(model_name=self.model_name, reasoning_effort=self.reasoning_effort)
        self.upsamplers.append((key or f"direct_{len(self.upsamplers)}", upsampler))
        upsampler.run({"prompt": request, "aspect_ratio": aspect_ratio})
        if upsampler.submit_tool().output is None:
            raise RuntimeError(upsampler.result_text or "the upsampler gave no structured prompt")
        return upsampler.submit_tool().output

    def run_core(self, args: dict[str, Any]) -> str:
        request = (args.get("prompt") or "").strip()
        if not request:
            raise ToolCallError("missing required argument 'prompt'; call upsample_prompt again with the image request")
        ratio = args.get("aspect_ratio")
        if ratio is not None and ratio not in ASPECT_RATIO_TO_SIZE:
            raise ToolCallError(f"aspect_ratio must be one of {list(ASPECT_RATIO_TO_SIZE)}, got {ratio!r}")
        record = getattr(self.current_call, "record", None)
        structured = self.upsample(request, ratio, key=record["tool_call_id"] if record else None)
        self.output = {"prompt": request, "aspect_ratio": ratio, "upsampled_prompt": structured}
        return json.dumps(structured, ensure_ascii=False)

    def setting(self) -> dict[str, Any]:
        return {**super().setting(), "upsampler_model": self.model_name, "reasoning_effort": self.reasoning_effort}

    def save_history_core(self, root: str) -> None:
        super().save_history_core(root)
        for key, upsampler in self.upsamplers:
            upsampler.save_history(osp.join(root, "call", key, "upsampler"))
