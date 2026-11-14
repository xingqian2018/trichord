import argparse
import json
from pathlib import Path

from dims_labeler import MODELS, RULES, DimsLabeler, SCHEMES

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE = SCRIPT_DIR / "taxonomy" / "qwen_image_bench_dims_reference.json"
LABELS = {"rwf": "real_world_fidelity", "cg": "creative_generation"}


def load_split(split: str) -> list[dict]:
    rows = json.loads(REFERENCE.read_text(encoding="utf-8"))["rows"]
    if split == "tune":
        return [r for r in rows if r["id"] % 2 == 1]
    if split == "heldout":
        return [r for r in rows if r["id"] % 2 == 0]
    return rows


def metrics(gold: list[bool], pred: list[bool]) -> dict:
    n = len(gold)
    tp = sum(g and p for g, p in zip(gold, pred))
    tn = sum(not g and not p for g, p in zip(gold, pred))
    fp = sum(not g and p for g, p in zip(gold, pred))
    fn = sum(g and not p for g, p in zip(gold, pred))
    po = (tp + tn) / n
    pe = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (n * n)
    return {
        "gold_rate": (tp + fn) / n,
        "pred_rate": (tp + fp) / n,
        "acc": po,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "kappa": (po - pe) / (1 - pe) if pe < 1 else 0.0,
    }


def evaluate(rows: list[dict], labeler: DimsLabeler, scheme: str) -> dict:
    prompts = [r["prompt_en"] for r in rows]
    result = {}
    for model in MODELS:
        votes = [labeler.votes_for(p, scheme, (model,))[model] for p in prompts]
        result[model] = {k: metrics([r[full] for r in rows], [v[full] for v in votes]) for k, full in LABELS.items()}
    all_votes = [labeler.votes_for(p, scheme, MODELS) for p in prompts]
    for l1, full in LABELS.items():
        for rule, fn in RULES.items():
            pred = [fn(sum(v[full] for v in vs.values()), len(vs)) for vs in all_votes]
            result.setdefault(f"vote_{rule}", {})[l1] = metrics([r[full] for r in rows], pred)
    return result


def format_table(title: str, result: dict) -> str:
    lines = [f"### {title}", "", "| 投票 | 标签 | 官方% | 预测% | acc | precision | recall | kappa |", "|---|---|---|---|---|---|---|---|"]
    for name, per in result.items():
        for l1, m in per.items():
            lines.append(
                f"| {name} | {l1.upper()} | {m['gold_rate']*100:.1f} | {m['pred_rate']*100:.1f} | {m['acc']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['kappa']:.3f} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=list(SCHEMES), required=True)
    parser.add_argument("--split", choices=["tune", "heldout", "all"], default="tune")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num_concurrency", type=int, default=16)
    parser.add_argument("--cache", type=str, default=str(SCRIPT_DIR / "blend" / "dims_calib_cache.jsonl"))
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    rows = load_split(args.split)
    if args.limit:
        rows = rows[: args.limit]
    labeler = DimsLabeler(scheme=args.scheme, cache_path=Path(args.cache), num_concurrency=min(args.num_concurrency, 16))
    labeler.collect([r["prompt_en"] for r in rows])
    result = evaluate(rows, labeler, args.scheme)
    print(format_table(f"{args.scheme} / {args.split} (n={len(rows)})", result))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    labeler.close()


if __name__ == "__main__":
    main()
