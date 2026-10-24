#!/usr/bin/env python3
"""reports_xx/collect_doc_eval.py - one JSON per evaluated checkpoint from the per-split doc-isolated evals of reports_xx/run_doc_eval.sh.

    python3 reports_xx/collect_doc_eval.py [--out DIR]       (default DIR: Project/trichord/dense_pretrain_result)

Reads <WORK_DIR>/<CHANGE>-<rung>-<job>/eval/val_splits_iter<N>_doc_isolated.json (the newest finished job per CHANGE/rung) and writes
<DIR>/<CHANGE>_<rung>.json with:
  metrics  dpl's derived metrics (val_metrics.derived_metrics: <collection>_tw / _macro, 01_val_v2_overall_tw, 00_overall_tw / _macro),
           each {loss, ppl, bpb}; 00_overall_* pool val_v2, uncheatable, codefresh, v3small, private_internal, pile_test (pile_10k apart)
  suites   per suite: token-weighted loss over the scored tokens + every split {loss, tokens}
plus the checkpoint, iteration and eval job. Nothing else is written.
"""
import argparse
import glob
import json
import math
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code", "scaling_ladder"))
import val_metrics  # noqa: E402

WORK_DIR = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/ladder_doc_eval"
OUT_DIR = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/Project/trichord/dense_pretrain_result"   # source of truth (in trichord's git)
SHARE_DIR = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/dense_pretrain_result"   # group-readable copy
REPO_DIR = os.path.join(HERE, "..", "results_xx")   # copy in the ladder repo (git), next to Boxiang's results/
DIR_RE = re.compile(r"^(?P<change>.+)-(?P<rung>q\d)-(?P<job>\d+)$")


def latest_evals():
    found = {}
    for d in glob.glob(os.path.join(WORK_DIR, "*-q[0-9]-*")):
        m = DIR_RE.match(os.path.basename(d))
        files = sorted(glob.glob(os.path.join(d, "eval", "val_splits_iter*_doc_isolated.json")))
        if not m or not files:
            continue
        key = (m["change"], m["rung"])
        if key not in found or int(m["job"]) > int(found[key][1]):
            found[key] = (d, m["job"], files[-1])
    return found


def ckpt_of(eval_dir):
    """The --load of the eval argv (promoted dir or our symlink stage) and its realpath."""
    argv = open(os.path.join(eval_dir, "argv_doc.txt")).read().split()
    load = argv[argv.index("--load") + 1]
    return load, os.path.realpath(os.path.join(load, "iter_%07d" % int(open(os.path.join(load, "latest_checkpointed_iteration.txt")).read())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--repo", default=REPO_DIR, help="after writing to --out (trichord), copy every JSON there to this repo dir ('' to skip)")
    a = ap.parse_args()
    outs = [a.out]
    os.makedirs(a.out, exist_ok=True)
    index = []
    for (change, rung), (eval_dir, job, path) in sorted(latest_evals().items()):
        rec = json.load(open(path))
        names = list(rec["suites"])
        suites = {s.name: s for s in val_metrics.load_suites(names, val_metrics.DEFAULT_SUITES_DIR)}
        derived = val_metrics.derived_metrics({n: rec["suites"][n]["loss"] for n in names}, suites, doc_isolated=True)
        metrics = {}
        for k, v in derived.items():                       # "val_loss/02_uncheatable_tw" -> metrics["02_uncheatable_tw"]["loss"]
            section, key = k.split("/", 1)
            metrics.setdefault(key, {})[section.replace("val_", "")] = v
        out_suites = {}
        for n in names:
            s = rec["suites"][n]
            tok = sum(s["tokens"])
            out_suites[n] = {"loss_tw": sum(l * t for l, t in zip(s["loss"], s["tokens"])) / tok, "tokens": tok,
                             "splits": {lab: {"loss": l, "tokens": t} for lab, l, t in zip(s["labels"], s["loss"], s["tokens"])}}
        # tsqw (Xingqian, 2026-10-23): all 7 collections (val_v2, uncheatable, codefresh, v3small, private_internal, pile_test, pile_10k;
        # unlike 00_overall_*, pile_10k is IN), each its token-weighted loss, weighted by sqrt(scored tokens) - between tw and macro
        tsqw_cols = {"01_val_v2_overall_tw": ("cosmos_val_v2_cc", "cosmos_val_v2_python", "cosmos_val_v2_arxiv"), "02_uncheatable_tw": ("uncheatable",),
                     "03_codefresh_tw": ("codefresh",), "04_v3small_tw": ("v3small",), "04_private_internal_tw": ("private_internal",),
                     "05_pile_test_tw": ("pile_test",), "05_pile_10k_tw": ("pile_10k",)}   # 7 collections incl. pile_10k (user)
        num = den = 0.0
        for key, prefixes in tsqw_cols.items():
            n_tok = sum(v["tokens"] for k, v in out_suites.items() if k.startswith(prefixes))
            if key in metrics and n_tok:
                num += math.sqrt(n_tok) * metrics[key]["loss"]; den += math.sqrt(n_tok)
        if den:
            metrics["00_overall_tsqw"] = {"loss": num / den, "ppl": math.exp(num / den)}
        load, ckpt = ckpt_of(eval_dir)
        out = {"change": change, "rung": rung, "iteration": rec["iteration"], "protocol": rec["protocol"],
               "checkpoint": ckpt, "eval_job": int(job), "eval_dir": eval_dir, "evaluated_at": rec["time"],
               "metrics": dict(sorted(metrics.items())), "suites": out_suites}
        for d in outs:
            dst = os.path.join(d, f"{change}_{rung}.json")
            with open(dst, "w") as f:
                json.dump(out, f, indent=1)
                f.write("\n")
        index.append(out)
        print(f"{change:16s} {rung}  val_v2 {metrics['01_val_v2_overall_tw']['loss']:.4f}  overall_tw {metrics.get('00_overall_tw', {}).get('loss', float('nan')):.4f}  -> {dst}")


    for dst_dir in [d for d in (SHARE_DIR, a.repo) if d]:   # then sync trichord -> shared dir and repo: copy every JSON (no deletes)
        os.makedirs(dst_dir, exist_ok=True)
        srcs = sorted(glob.glob(os.path.join(a.out, "*.json")))
        for src in srcs:
            shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
        print(f"copied {len(srcs)} JSON files {a.out} -> {os.path.abspath(dst_dir)}")


INDEX_COLS = [("01_val_v2_overall_tw", "val_v2"), ("02_uncheatable_tw", "uncheatable"), ("03_codefresh_tw", "codefresh"),
              ("04_private_internal_tw", "private_internal"), ("00_overall_tw", "overall_tw")]


def write_index(repo, recs):
    """results_xx/README.md: what the JSONs are + an index table (regenerated on every run)."""
    head = """# results_xx/ - Xingqian's doc-isolated per-split eval of the ladder checkpoints

One JSON per evaluated checkpoint, `<Change ID>_<rung>.json` (Change ID = the folder name under `results/`). Generated by
`reports_xx/collect_doc_eval.py` from the eval-only runs of `reports_xx/run_doc_eval.sh`; do not edit by hand. Same files as the shared
copy `users/xingqianx/dense_pretrain_result/`.

- **Protocol**: document-isolated (every document scored as its own padded sample), `sh/eval_ckpt.sh <ckpt> doc` with `VAL_PER_SPLIT=1`:
  the 9 decontaminated suites (val_v2 cc / python / arxiv, uncheatable, codefresh, v3small, **private_internal**, pile_test, pile_10k),
  115 splits. Every suite except private_internal matches the doc-isolated column of `results/<Change ID>/ckpts/<rung>/eval/RESULTS.md`
  within 6e-5.
- **`metrics`**: dpl's definitions (`val_metrics.derived_metrics`), each `{loss, ppl, bpb}`: `<collection>_tw` = token-weighted,
  `_macro` = mean over splits, `01_val_v2_overall_tw`, `00_overall_tw` (val_v2 + uncheatable + codefresh + v3small + private_internal +
  pile_test pooled, token-weighted) and `00_overall_macro` (mean of those six); pile_10k is reported apart.
- **`suites`**: per suite the token-weighted loss over the scored tokens and every split `{loss, tokens}`.
- Also: `checkpoint` (resolved path), `iteration`, `eval_job`, `eval_dir`, `evaluated_at`.
- private_internal is a NON-PUBLIC set (data ACL: runlongl / boxiangw / xingqianx); these files hold only its loss numbers.

## Index (nats / token, lower is better)

"""
    rows = ["| Change ID | rung | iteration | " + " | ".join(n for _, n in INDEX_COLS) + " | eval job |",
            "|---|---|---:|" + "---:|" * len(INDEX_COLS) + "---:|"]
    for r in sorted(recs, key=lambda r: (r["change"], r["rung"])):
        m = r["metrics"]
        rows.append(f"| {r['change']} | {r['rung']} | {r['iteration']} | " + " | ".join(f"{m[k]['loss']:.4f}" if k in m else "·" for k, _ in INDEX_COLS)
                    + f" | {r['eval_job']} |")
    with open(os.path.join(repo, "README.md"), "w") as f:
        f.write(head + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
