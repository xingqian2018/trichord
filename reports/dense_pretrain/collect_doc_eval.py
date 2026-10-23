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
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "code", "scaling_ladder"))
import val_metrics  # noqa: E402

WORK_DIR = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/ladder_doc_eval"
OUT_DIR = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/Project/trichord/dense_pretrain_result"
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
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
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
        load, ckpt = ckpt_of(eval_dir)
        out = {"change": change, "rung": rung, "iteration": rec["iteration"], "protocol": rec["protocol"],
               "checkpoint": ckpt, "eval_job": int(job), "eval_dir": eval_dir, "evaluated_at": rec["time"],
               "metrics": dict(sorted(metrics.items())), "suites": out_suites}
        dst = os.path.join(a.out, f"{change}_{rung}.json")
        with open(dst, "w") as f:
            json.dump(out, f, indent=1)
            f.write("\n")
        print(f"{change:16s} {rung}  val_v2 {metrics['01_val_v2_overall_tw']['loss']:.4f}  overall_tw {metrics.get('00_overall_tw', {}).get('loss', float('nan')):.4f}  -> {dst}")


if __name__ == "__main__":
    main()
