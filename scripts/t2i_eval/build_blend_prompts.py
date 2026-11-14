# -----------------------------------------------------------------------------
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact Cosmos Lab at cosmoslab@exchange.nvidia.com.
# -----------------------------------------------------------------------------

"""Build taxonomy-stratified T2I eval prompt blends (arena1000, cosmos500).

Output: <out_dir>/<name>.json = {"blend", "benchmark": [
    {id, taxonomy, prompt, prompt_upsampled, dims, writer_model, target_words, dims_votes, dims_facets}]}

arena: model_playground_v1/taxonomy/t2i_arena_v0.yaml, no category filter. Every L1 domain gets
    --min_per_l1 prompts, the rest is split by L1 weight (largest remainder); inside a domain each
    leaf is drawn L2-by-weight, then uniformly level by level (L3-L5 carry no weights).
cosmos: taxonomy/t2i_cosmos_v0.yaml. L1 carries the ratio ("name|weight"), split exactly by largest
    remainder; L2 categories split evenly inside their L1; each L2 value is the category guideline.

Prompt lengths follow Qwen-Image-Bench: target word counts are evenly spaced quantiles of the
official prompt_en word counts (taxonomy/qwen_image_bench_prompt_en_words.json), shuffled.

Stages (resumable through <out_dir>/<name>_state.json): plan -> write prompts (writer models in
rotation) -> near-duplicate rewrite -> L1 dims by dims_labeler (facet selection, 3-model vote) -> final json.
Gateway key: credentials/gateway.json "NVIDIA_GATEWAY".

cd ~/Project/trichord/scripts/t2i_eval
python build_blend_prompts.py --blend arena --num 1000 --out_dir blend
python build_blend_prompts.py --blend cosmos --num 500 --out_dir blend
"""

import argparse
import json
import random
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dims_labeler import DimsLabeler  # noqa: E402
from utils import GATEWAY_CONFIG, UnifiedGatewayLLM  # noqa: E402

WRITER_MODELS = ("opus-5.5", "kimi-k3", "gpt-6-astra")

LENGTH_REQUEST_TOLERANCE = 0.1
LENGTH_ACCEPT_TOLERANCE = 0.3
LENGTH_MIN_HALF_WIDTH = 2

DUP_JACCARD = 0.6
MAX_ROUNDS = 5

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ARENA_TAXONOMY = SCRIPT_DIR.parent / "model_playground_v1" / "taxonomy" / "t2i_arena_v0.yaml"
DEFAULT_COSMOS_TAXONOMY = SCRIPT_DIR / "taxonomy" / "t2i_cosmos_v0.yaml"
DEFAULT_LENGTH_REFERENCE = SCRIPT_DIR / "taxonomy" / "qwen_image_bench_prompt_en_words.json"

WRITER_SYSTEM_PROMPT = """You write prompts for a professional text-to-image evaluation set, written by test designers who probe what image models get right and wrong.

For every prompt:
- Describe one concrete image that clearly belongs to the given category: specific subjects with their key attributes (count, material, size, shape, state), what they are doing, how they relate and where they are placed, and the setting. A specific subject, never the category itself.
- Write plainly and concretely, mostly in present-tense statements ("The ice contains air bubbles", "Flow path follows gravity"). Prefer precise nouns and verbs over decorative adjectives.
- Write like a test designer's spec rather than a story: open with the scene type, the setting, or the main subject with "The" (for example "Rainy night street market, ...", "The interior of a ...", "Close-up of ..."); open with "A" or "An" only occasionally. Lean on "with ..." phrases and comma-separated noun phrases; an occasional em dash or parenthetical is fine.
- Use numbers only when the count itself is being tested.
- When you state expectations, use the vocabulary of image evaluation and vary the wording: "overall ...", "readable ...", "... without ...", "... yet ...", "realistic", "natural", "clear structure", "correct light and shadow logic", "sharp edges", "accurate proportions", "no excessive noise or distortion". Do not repeat the same expectation phrasing from prompt to prompt.
- Mention a medium, art style, camera, colour or lighting only when it matters for this image; most prompts mention at most one of these. No palette lists.
- If the image carries text, give the exact text in double quotes.
- Follow the requested prompt form and word-count range.
- Avoid real, identifiable private individuals (public figures, real brands and copyrighted characters only when the category explicitly asks for them), sexual content, gore, and anything hateful or demeaning. Leave out technical generation settings (resolution, steps, seeds, "8k", "masterpiece").

Across the set, vary subjects, people, places, cultures, eras and scale.

Work like a Chinese test designer: first compose the prompt in natural Chinese, following all the rules above, then translate it into English faithfully, keeping the Chinese sentence structure, clause order and punctuation style rather than polishing it into idiomatic English. Text that must appear in the image is English text in double quotes, unless you are told to include Chinese text. The word-count range applies to the English translation.

Respond with a JSON object only, no markdown."""

PROMPT_FORMS = (
    ("plain", 0.35, "a plain description of the image in one to three present-tense sentences, with no explicit requirements."),
    ("spec", 0.35, "a description of the image followed by one or two expectations, phrased in a varied way (e.g. 'Require ...', 'Requirements: ...', '... should ...', '... is clearly ...', '... strictly follows ...'); use the word 'must' at most once."),
    ("header", 0.15, "a short label for the kind of image followed by a colon (e.g. 'Product hero shot:', 'Single frame of stop-motion animation:', 'Museum label illustration:'), then clipped, note-like clauses separated by semicolons, ending with the quality expectations."),
    ("challenge", 0.15, "a prompt built around one specific generation challenge that suits the category: something conspicuously absent (negation), an exact list of items with their attributes, an unusual attribute binding, counter-intuitive physics, or a precise spatial arrangement."),
)

CHINESE_TEXT_RATE = 0.03

CHINESE_TEXT_INSTRUCTION = (
    "\nThis prompt must ask for one short piece of Chinese text (2 to 8 Chinese characters) to be rendered in the "
    "image, given in double quotes; everything else stays in English."
)

ARENA_USER_TEMPLATE = """Category path (broad to specific): {path}
Prompt form: {form}
Write 1 prompt of {lo}-{hi} words.{chinese}

Output: {{"prompt_cn": "...", "prompt": "<the faithful English translation>"}}"""

COSMOS_USER_TEMPLATE = """Category: {group} > {subcategory}
Category guideline (the prompt must satisfy it): {description}
Prompt form: {form}
Write 1 prompt of {lo}-{hi} words.{chinese}
{avoid}
Output: {{"prompt_cn": "...", "prompt": "<the faithful English translation>"}}"""

_NON_LATIN_SCRIPTS = ("CJK", "HIRAGANA", "KATAKANA", "HANGUL", "CYRILLIC", "ARABIC", "HEBREW", "DEVANAGARI", "THAI")


def split_weight(key: str) -> tuple[str, Optional[float]]:
    name, sep, weight = str(key).rpartition("|")
    if sep and name:
        try:
            return name, float(weight)
        except ValueError:
            pass
    return str(key), None


def load_arena_tree(path: Path) -> dict:
    def convert(tree: Optional[dict]) -> dict:
        nodes = {}
        for key, child in (tree or {}).items():
            name, weight = split_weight(key)
            nodes[name] = {"weight": weight, "children": convert(child) if child else {}}
        return nodes

    return convert(yaml.safe_load(path.read_text()))


def largest_remainder(total: int, weights: list[float]) -> list[int]:
    weight_sum = sum(weights)
    quotas = [total * w / weight_sum for w in weights]
    counts = [int(q) for q in quotas]
    order = sorted(range(len(weights)), key=lambda i: (-(quotas[i] - counts[i]), i))
    for i in order[: total - sum(counts)]:
        counts[i] += 1
    return counts


def arena_allocation(tree: dict, num: int, min_per_l1: int) -> dict[str, int]:
    names = list(tree)
    weights = [tree[n]["weight"] for n in names]
    if any(w is None for w in weights):
        raise ValueError("every arena L1 node must carry a weight")
    remaining = num - min_per_l1 * len(names)
    if remaining < 0:
        raise ValueError(f"num={num} is below min_per_l1 * {len(names)} L1 domains")
    extra = largest_remainder(remaining, weights)
    return {n: min_per_l1 + e for n, e in zip(names, extra)}


def draw_arena_leaf(rng: random.Random, l1_name: str, l1_node: dict) -> list[str]:
    path = [l1_name]
    children = l1_node["children"]
    while children:
        names = list(children)
        weights = [children[n]["weight"] for n in names]
        if all(w is not None for w in weights):
            name = rng.choices(names, weights=weights)[0]
        else:
            name = rng.choice(names)
        path.append(name)
        children = children[name]["children"]
    return path


def plan_arena(tree: dict, num: int, min_per_l1: int, rng: random.Random) -> list[dict]:
    allocation = arena_allocation(tree, num, min_per_l1)
    items = []
    for l1_name, count in allocation.items():
        seen: set[tuple[str, ...]] = set()
        while len(seen) < count:
            leaf = tuple(draw_arena_leaf(rng, l1_name, tree[l1_name]))
            if leaf not in seen:
                seen.add(leaf)
                items.append({"taxonomy": list(leaf)})
    return items


def load_cosmos_taxonomy(path: Path) -> list[tuple[str, float, list[tuple[str, str]]]]:
    groups = []
    for key, categories in yaml.safe_load(path.read_text()).items():
        name, weight = split_weight(key)
        if weight is None or not isinstance(categories, dict) or not categories:
            raise ValueError(f"cosmos taxonomy {path}: L1 {key!r} needs a weight and non-empty categories")
        if not all(isinstance(g, str) and g.strip() for g in categories.values()):
            raise ValueError(f"cosmos taxonomy {path}: every L2 category under {name!r} needs a guideline")
        groups.append((name, weight, [(cat, " ".join(g.split())) for cat, g in categories.items()]))
    return groups


def plan_cosmos(groups: list[tuple[str, float, list[tuple[str, str]]]], num: int) -> list[dict]:
    items = []
    for (group, _, categories), group_count in zip(groups, largest_remainder(num, [w for _, w, _ in groups])):
        for (cat, guideline), cat_count in zip(categories, largest_remainder(group_count, [1.0] * len(categories))):
            items.extend({"taxonomy": [group, cat], "description": guideline} for _ in range(cat_count))
    return items


def assign_target_words(items: list[dict], reference: list[int], rng: random.Random) -> None:
    ref = sorted(reference)
    targets = [ref[min(len(ref) - 1, int((i + 0.5) * len(ref) / len(items)))] for i in range(len(items))]
    rng.shuffle(targets)
    for item, target in zip(items, targets):
        item["target_words"] = target


def word_bounds(target: int, tolerance: float) -> tuple[int, int]:
    half = max(LENGTH_MIN_HALF_WIDTH, round(target * tolerance))
    return max(1, target - half), target + half


def extract_json(text: str) -> Optional[Any]:
    if not text:
        return None
    text = text[text.rfind("</think>") + len("</think>") :] if "</think>" in text else text
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except ValueError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except ValueError:
                return None
    return None


def is_english(text: str) -> bool:
    for ch in text:
        if ch.isalpha() and any(script in unicodedata.name(ch, "") for script in _NON_LATIN_SCRIPTS):
            return False
    return True


def valid_prompt(prompt: Any, target: int, chinese_text: bool) -> bool:
    if not isinstance(prompt, str) or not prompt.strip():
        return False
    lo, hi = word_bounds(target, LENGTH_ACCEPT_TOLERANCE)
    if not lo <= len(prompt.split()) <= hi:
        return False
    if not chinese_text:
        return is_english(prompt)
    quoted = "".join(re.findall(r'"([^"]*)"', prompt))
    return not is_english(quoted) and is_english(re.sub(r'"[^"]*"', "", prompt))


def assign_forms_and_chinese(items: list[dict], rng: random.Random) -> None:
    counts = largest_remainder(len(items), [weight for _, weight, _ in PROMPT_FORMS])
    forms = [name for (name, _, _), count in zip(PROMPT_FORMS, counts) for _ in range(count)]
    rng.shuffle(forms)
    chinese = set(rng.sample(range(len(items)), round(CHINESE_TEXT_RATE * len(items))))
    for idx, (item, form) in enumerate(zip(items, forms)):
        item["prompt_form"] = form
        item["chinese_text"] = idx in chinese


class BlendBuilder:
    def __init__(self, blend: str, state_path: Path, num_concurrency: int, timeout: int):
        self.blend = blend
        self.state_path = state_path
        self.vlm = UnifiedGatewayLLM(
            gateway_configs={"nvidia": GATEWAY_CONFIG["nvidia"]}, num_concurrency=num_concurrency, timeout=timeout
        )

    def save(self, items: list[dict]) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.state_path)

    def request(self, model: str, system_prompt: str, user_prompt: str) -> dict:
        return self.vlm.build_request(
            model_name=model, system_prompt=system_prompt, content_prompt=user_prompt, temperature=1.0, max_tokens=16384
        )

    def pending_batch(self, items: list[dict]) -> list[int]:
        pending = [i for i, it in enumerate(items) if not it.get("prompt")]
        if self.blend == "arena":
            return pending
        first_per_category: dict[str, int] = {}
        for i in pending:
            first_per_category.setdefault(items[i]["taxonomy"][1], i)
        return list(first_per_category.values())

    def item_prompt(self, items: list[dict], i: int) -> str:
        item = items[i]
        lo, hi = word_bounds(item["target_words"], LENGTH_REQUEST_TOLERANCE)
        form = dict((name, desc) for name, _, desc in PROMPT_FORMS)[item["prompt_form"]]
        chinese = CHINESE_TEXT_INSTRUCTION if item["chinese_text"] else ""
        if self.blend == "arena":
            return ARENA_USER_TEMPLATE.format(path=" > ".join(item["taxonomy"]), form=form, lo=lo, hi=hi, chinese=chinese)
        siblings = [it["prompt"] for it in items if it.get("prompt") and it["taxonomy"] == item["taxonomy"]]
        if item.get("avoid"):
            siblings.append(item["avoid"])
        avoid = (
            "Already written for this category; pick a clearly different subject and scene:\n"
            + "\n".join(f"- {p}" for p in siblings)
            + "\n"
            if siblings
            else ""
        )
        return COSMOS_USER_TEMPLATE.format(
            group=item["taxonomy"][0],
            subcategory=item["taxonomy"][1],
            description=item["description"],
            form=form,
            lo=lo,
            hi=hi,
            chinese=chinese,
            avoid=avoid,
        )

    def write_round(self, items: list[dict], round_idx: int) -> int:
        batch = self.pending_batch(items)
        models = [WRITER_MODELS[(i + round_idx) % len(WRITER_MODELS)] for i in batch]
        requests = [self.request(m, WRITER_SYSTEM_PROMPT, self.item_prompt(items, i)) for m, i in zip(models, batch)]
        responses = self.vlm.query(requests, pbar_desc=f"{self.blend} write round {round_idx}")
        written = 0
        for i, model, response in zip(batch, models, responses):
            parsed = extract_json(response)
            prompt = parsed.get("prompt") if isinstance(parsed, dict) else None
            if valid_prompt(prompt, items[i]["target_words"], items[i]["chinese_text"]):
                items[i]["prompt"] = " ".join(prompt.split())
                items[i]["writer_model"] = model
                items[i].pop("avoid", None)
                written += 1
        self.save(items)
        logger.info(f"[{self.blend}] write round {round_idx}: {len(batch)} calls, {written} written")
        return written

    def write_until_complete(self, items: list[dict]) -> None:
        round_idx, stalled = 0, 0
        while any(not it.get("prompt") for it in items):
            stalled = 0 if self.write_round(items, round_idx) else stalled + 1
            if stalled >= MAX_ROUNDS:
                raise RuntimeError(f"no prompt written in {MAX_ROUNDS} consecutive rounds")
            round_idx += 1

    def dedup(self, items: list[dict]) -> int:
        token_sets = [set(re.findall(r"[a-z0-9']+", it["prompt"].lower())) for it in items]
        flagged = 0
        for j in range(len(items)):
            for i in range(j):
                union = token_sets[i] | token_sets[j]
                if union and len(token_sets[i] & token_sets[j]) / len(union) >= DUP_JACCARD:
                    items[j]["avoid"] = items[i]["prompt"]
                    items[j]["prompt"] = ""
                    token_sets[j] = set()
                    flagged += 1
                    break
        return flagged

    def label_dims(self, items: list[dict]) -> None:
        pending = [it for it in items if "dims" not in it]
        if not pending:
            return
        labeler = DimsLabeler(llm=self.vlm)
        for it, labeled in zip(pending, labeler.label([it["prompt"] for it in pending])):
            it.update(labeled)
        self.save(items)

    def close(self) -> None:
        self.vlm.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build taxonomy-stratified T2I eval prompt blends")
    parser.add_argument("--blend", choices=["arena", "cosmos"], required=True)
    parser.add_argument("--num", type=int, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--arena_taxonomy", type=str, default=str(DEFAULT_ARENA_TAXONOMY))
    parser.add_argument("--cosmos_taxonomy", type=str, default=str(DEFAULT_COSMOS_TAXONOMY))
    parser.add_argument("--length_reference", type=str, default=str(DEFAULT_LENGTH_REFERENCE))
    parser.add_argument("--min_per_l1", type=int, default=3)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_concurrency", type=int, default=48)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--plan_only", action="store_true")
    args = parser.parse_args()

    name = args.name or f"{args.blend}{args.num}"
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / f"{name}_state.json"

    taxonomy_arg = args.arena_taxonomy if args.blend == "arena" else args.cosmos_taxonomy
    taxonomy_source = str(Path(taxonomy_arg).expanduser().resolve())

    if state_path.exists():
        items = json.loads(state_path.read_text(encoding="utf-8"))
        logger.info(f"Resuming {len(items)} items from {state_path}")
    else:
        rng = random.Random(args.seed)
        if args.blend == "arena":
            items = plan_arena(load_arena_tree(Path(taxonomy_source)), args.num, args.min_per_l1, rng)
        else:
            items = plan_cosmos(load_cosmos_taxonomy(Path(taxonomy_source)), args.num)
        reference = json.loads(Path(args.length_reference).expanduser().read_text())["word_counts"]
        assign_target_words(items, reference, rng)
        assign_forms_and_chinese(items, rng)
        for idx, item in enumerate(items):
            item["id"] = f"{name}_{idx:04d}"
        if len(items) != args.num:
            raise RuntimeError(f"planned {len(items)} items, expected {args.num}")

    if args.plan_only:
        plan_path = out_dir / f"{name}_plan.json"
        plan_path.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
        logger.info(f"Wrote plan ({len(items)} items) to {plan_path}")
        return

    builder = BlendBuilder(args.blend, state_path, args.num_concurrency, args.timeout)
    builder.save(items)
    try:
        builder.write_until_complete(items)
        for dedup_round in range(3):
            flagged = builder.dedup(items)
            logger.info(f"[{name}] dedup round {dedup_round}: {flagged} near-duplicate(s) to rewrite")
            if not flagged:
                break
            for it in items:
                if not it["prompt"]:
                    it.pop("dims", None)
                    it.pop("dims_votes", None)
                    it.pop("dims_facets", None)
            builder.save(items)
            builder.write_until_complete(items)
        builder.label_dims(items)
    finally:
        builder.close()

    benchmark = [
        {
            "id": it["id"],
            "taxonomy": it["taxonomy"],
            "prompt": it["prompt"],
            "prompt_upsampled": "",
            "dims": it["dims"],
            "writer_model": it["writer_model"],
            "target_words": it["target_words"],
            "prompt_form": it["prompt_form"],
            "dims_votes": it["dims_votes"],
            "dims_facets": it["dims_facets"],
        }
        for it in items
    ]
    final_path = out_dir / f"{name}.json"
    final_path.write_text(
        json.dumps({"blend": name, "benchmark": benchmark}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Wrote {len(benchmark)} prompts to {final_path}")


if __name__ == "__main__":
    main()
