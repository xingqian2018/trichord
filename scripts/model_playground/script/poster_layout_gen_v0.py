import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger  # noqa: E402
from tqdm import tqdm  # noqa: E402

from tool.generate_image import generate_image  # noqa: E402
from tool.poster_visual_critic_agent import poster_visual_critic_agent  # noqa: E402
from tool.render_poster import render_poster  # noqa: E402
from utils import GATEWAY_CONFIG, UnifiedGatewayLLM, render_html_to_png, resolve_model_string  # noqa: E402

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompt"
DEFAULT_SYSTEM_PROMPT_PRESET = "agentic_poster_layout_t2i_required"

HTML_BLOCK_RE = re.compile(r"```\s*html\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
GENERIC_FENCE_RE = re.compile(r"```(\w*)\s*\n?(.*?)```", re.DOTALL)
HTML_DOC_RE = re.compile(r"<!doctype html|<html[\s>]", re.IGNORECASE)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True, help="one json per line: {id?, prompt}")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-6-astra@nvidia")
    parser.add_argument("--image_model_name", type=str, default="nano-banana-2.0")
    parser.add_argument("--critic_model_name", type=str, default=poster_visual_critic_agent.__init__.__defaults__[0])
    parser.add_argument("--system_prompt_preset", type=str, default=DEFAULT_SYSTEM_PROMPT_PRESET)
    parser.add_argument("--max_concurrency", type=int, default=8)
    parser.add_argument("--max_turns", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max_try", type=int, default=2)
    parser.add_argument("--no_render_png", action="store_true")
    return parser.parse_args()


def load_samples(input_jsonl: str) -> list[dict]:
    samples = []
    for line_idx, line in enumerate(Path(input_jsonl).read_text().splitlines()):
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = str(record.get("id", f"{line_idx:05d}"))
        samples.append({"id": sample_id, "prompt": record["prompt"], "try_num": 0})
    return samples


def extract_html_blocks(text: str) -> list[str]:
    blocks = [b.strip() for b in HTML_BLOCK_RE.findall(text)]
    if blocks:
        return blocks
    for lang, body in GENERIC_FENCE_RE.findall(text):
        if lang.lower() != "html" and HTML_DOC_RE.search(body):
            blocks.append(body.strip())
    return blocks







class PosterLayoutGenerator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.system_prompt = (PROMPT_DIR / f"{args.system_prompt_preset}.md").read_text()
        self.model_str = resolve_model_string(args.model_name)
        self.llm = UnifiedGatewayLLM(GATEWAY_CONFIG, num_concurrency=args.max_concurrency, num_max_retry=2, timeout=args.timeout)
        self.image_tool = generate_image(args.image_model_name)
        self.image_tool.gateway.num_concurrency = args.max_concurrency
        self.render_tool = render_poster()
        self.critic_tool = poster_visual_critic_agent(args.critic_model_name)
        self.critic_tool.gateway.num_concurrency = args.max_concurrency
        self.tools_def = [generate_image.SCHEMA, render_poster.SCHEMA, poster_visual_critic_agent.SCHEMA]
        self.sem = asyncio.Semaphore(args.max_concurrency)

    def build_request(self, messages: list[dict]) -> dict:
        request = {
            "model": self.model_str,
            "messages": [{"role": "system", "content": self.system_prompt}, *messages],
            "max_tokens": self.args.max_tokens,
            "stream": True,
            "tools": self.tools_def,
        }
        if self.args.temperature is not None:
            request["temperature"] = self.args.temperature
        return request

    def execute_tool_call(self, name: str, arguments_json: str, sample_dir: Path) -> dict:
        tool_args = json.loads(arguments_json) if arguments_json else {}
        if name == "generate_image":
            filename = Path(tool_args.get("output_path", "")).name or f"image_{int(time.time() * 1000)}.png"
            path = self.image_tool.generate(tool_args["prompt"], tool_args.get("aspect_ratio", "1:1"), str(sample_dir / "images" / filename))
            return {"text": f"[generate_image saved to {path}]", "images": [path]}
        if name == "render_poster":
            filename = Path(tool_args.get("output_path", "")).name or f"render_{int(time.time() * 1000)}.png"
            path = self.render_tool.render(tool_args["html"], str(sample_dir / "renders" / filename))
            return {"text": f"[render_poster saved to {path}]", "images": [path]}
        if name == "poster_visual_critic_agent":
            return self.critic_tool.run(tool_args)
        return {"text": f"Tool call '{name}' failed: unknown tool", "images": []}

    async def single_process(self, sample: dict) -> dict:
        sample_dir = Path(self.args.output_dir) / sample["id"]
        image_dir = sample_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        log_path = sample_dir / "log.jsonl"
        log_file = log_path.open("a")

        def log_event(event: dict):
            log_file.write(json.dumps({"time": time.time(), "try_num": sample["try_num"], **event}, ensure_ascii=False) + "\n")
            log_file.flush()

        messages = [{"role": "user", "content": sample["prompt"]}]
        result = {"id": sample["id"], "status": "failed", "turns": 0, "images": [], "html_path": None, "png_path": None}
        log_event({"event": "start", "model": self.model_str, "prompt": sample["prompt"]})

        async with self.sem:
            for turn in range(self.args.max_turns):
                response = await self.llm.query_core(self.build_request(messages))
                log_event({"event": "response", "turn": turn, **response})
                result["turns"] = turn + 1
                if response["finish_reason"] is None and response["content"] is None and response["tool_calls"] is None:
                    result["error"] = "llm query failed"
                    break

                tool_calls = json.loads(response["tool_calls"]) if response["tool_calls"] else []
                if not tool_calls:
                    messages.append({"role": "assistant", "content": response["content"] or ""})
                    html_blocks = extract_html_blocks(response["content"] or "")
                    if not html_blocks:
                        result["error"] = "final response has no html block"
                        break
                    html_path = sample_dir / "poster.html"
                    html_path.write_text(html_blocks[-1])
                    result["html_path"] = str(html_path)
                    result["status"] = "done"
                    break

                for i, call in enumerate(tool_calls):
                    call["id"] = call["id"] or f"call_{turn}_{i}"
                messages.append({
                    "role": "assistant",
                    "content": response["content"] or "",
                    "tool_calls": [
                        {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                        for c in tool_calls
                    ],
                })
                for call in tool_calls:
                    try:
                        tool_result = await asyncio.to_thread(self.execute_tool_call, call["name"], call["arguments"], sample_dir)
                    except Exception as e:
                        tool_result = {"text": f"Tool call '{call['name']}' failed: {type(e).__name__}: {e}", "images": []}
                    log_event({"event": "tool_result", "turn": turn, "tool_call_id": call["id"], "name": call["name"], **tool_result})
                    result["images"].extend(tool_result["images"])
                    messages.append({"role": "tool", "tool_call_id": call["id"], "content": tool_result["text"]})
            else:
                result["error"] = f"exceeded max_turns={self.args.max_turns}"

            if result["status"] == "done" and not self.args.no_render_png:
                png_path = sample_dir / "poster.png"
                try:
                    await asyncio.to_thread(render_html_to_png, Path(result["html_path"]).read_text(), png_path)
                    result["png_path"] = str(png_path)
                except Exception as e:
                    result["render_error"] = f"{type(e).__name__}: {e}"
                    logger.warning(f"[{sample['id']}] png render failed: {result['render_error']}")

        (sample_dir / "messages.json").write_text(json.dumps(messages, indent=2, ensure_ascii=False))
        (sample_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
        log_event({"event": "end", **result})
        log_file.close()
        return result

    async def run_batch(self, samples: list[dict]) -> list[dict]:
        pbar = tqdm(total=len(samples), desc=f"try_num={samples[0]['try_num']}")

        async def one(sample: dict) -> dict:
            result = await self.single_process(sample)
            pbar.update(1)
            return result

        results = await asyncio.gather(*(one(s) for s in samples))
        pbar.close()
        return results


def already_done(sample: dict, output_dir: str) -> bool:
    return (Path(output_dir) / sample["id"] / "poster.html").exists()


def main() -> None:
    args = parse_arguments()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    samples_todo = sorted(load_samples(args.input_jsonl), key=lambda s: s["id"])
    num_total = len(samples_todo)
    samples_todo = [s for s in samples_todo if not already_done(s, args.output_dir)]
    logger.info(f"{num_total} samples total, {num_total - len(samples_todo)} already done, {len(samples_todo)} to run")

    generator = PosterLayoutGenerator(args)

    async def run_all_tries(samples_todo: list[dict]) -> tuple[list[dict], list[dict]]:
        samples_done = []
        while True:
            samples_exhausted = [s for s in samples_todo if s["try_num"] >= args.max_try]
            samples_todo = sorted((s for s in samples_todo if s["try_num"] < args.max_try), key=lambda s: (s["try_num"], s["id"]))
            if not samples_todo:
                return samples_done, samples_exhausted
            results = await generator.run_batch(samples_todo)
            samples_error = []
            for sample, result in zip(samples_todo, results):
                if result["status"] == "done":
                    samples_done.append(result)
                else:
                    sample["try_num"] += 1
                    samples_error.append(sample)
                    logger.warning(f"[{sample['id']}] failed (try {sample['try_num']}/{args.max_try}): {result.get('error')}")
            samples_todo = samples_error

    samples_done, samples_failed = asyncio.run(run_all_tries(samples_todo))
    logger.info(f"done: {len(samples_done)}, failed after {args.max_try} tries: {len(samples_failed)}")
    generator.llm.close()
    generator.image_tool.gateway.close()
    generator.critic_tool.gateway.close()


if __name__ == "__main__":
    main()
