#!/usr/bin/env python3
"""reports_xx/collect_prod_isodoc.py - doc-isolated (iso-doc) val loss of the PRODUCTION run's checkpoints -> results_xx/prod_isodoc/.

    python3 reports_xx/collect_prod_isodoc.py [--out DIR]

Read-only on the sources. One JSON per checkpoint and eval window, `<RUN>[_W32K]_iter<NNNNNNN>.json` (no suffix = the 8K window, the
window of record of every other doc-isolated number; _W32K = the 32K window, the length the production run trains at).

Sources (user 2026-11-13 "prod iso-doc"): the training-side doc evals of the 59B run `ultra59b_muonh_kd_qbk3`, written as
`<root>/ultra59b-iter<N>-<job>/results.json` by boxiangw's launch/eval59/doc_eval_59b.sbatch (the ladder's pretrain_doceval.py + doc_eval.py
on the production Megatron tree; the same scoring code as reports_xx/run_doc_eval.sh, doc_eval.py is byte-identical):
  - boxiangw/cosmos_ultra_pre/runs/eval59/doc_eval/          (his auto pipeline, loss-parity reference of the downstream eval)
  - xingqianx/runs/cosmos_eval_downstream/_doc_eval59/       (the same script submitted from our account; runs/cosmos_eval is a link
                                                              to cosmos_eval_downstream since the 2026-11-13 reorganisation)
These evals score POOLED sets (`val_decon_pooled/<set>`), one loss per set and no per-split numbers. A pooled set holds exactly the
documents of one of our collections, so its loss is that collection's token-weighted loss and is stored under the ladder's metric key
(val_v2_decon -> 01_val_v2_overall_tw, v3small_decon -> 04_v3small_tw, ...). The collector checks this: the set's predicted-token and
document counts must equal the collection's in the suite yamls, otherwise the set is refused. bpb = loss * tokens / (bytes * ln 2),
the definition of val_metrics.derived_metrics.

When the same checkpoint and window were evaluated more than once, the newest complete result is used and the others are listed under
`other_evals`. Results that are incomplete or carry problems are skipped and printed.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(HERE, "..", "code", "scaling_ladder"))
import val_metrics  # noqa: E402

BX = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_data/users/boxiangw/cosmos_ultra_pre"
U = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx"
OUT_DIR = os.path.join(HERE, "..", "results_xx", "prod_isodoc")
RUNS = [dict(run="ULTRA59B_MUONH_KD_QBK3", display="Ultra 59B (MuonH + KD, QBK3)", prefix="ultra59b-iter",
             view=BX + "/runs/eval59/view/ultra59b_muonh_kd_qbk3",
             roots=[BX + "/runs/eval59/doc_eval", U + "/runs/cosmos_eval_downstream/_doc_eval59"])]
# pooled set -> (the ladder's metric key, suite-name stems of the collection)
SETS = {"val_v2_decon": ("01_val_v2_overall_tw", ("cosmos_val_v2_cc", "cosmos_val_v2_python", "cosmos_val_v2_arxiv")),
        "uncheatable_decon": ("02_uncheatable_tw", ("uncheatable",)),
        "codefresh_decon": ("03_codefresh_tw", ("codefresh",)),
        "v3small_decon": ("04_v3small_tw", ("v3small",)),
        "private_internal_decon": ("04_private_internal_tw", ("private_internal",)),
        "pile_test_decon": ("05_pile_test_tw", ("pile_test",)),
        "pile_10k_decon": ("05_pile_10k_tw", ("pile_10k",))}
SUITES = ["cosmos_val_v2_cc_decon_qwen3", "cosmos_val_v2_python_decon_qwen3", "cosmos_val_v2_arxiv_decon_qwen3", "uncheatable_decon_qwen3",
          "codefresh_decon_qwen3", "v3small_decon_qwen3", "private_internal_decon_qwen3", "pile_test_decon_qwen3", "pile_10k_decon_qwen3"]


def collections():
    """metric key -> (predicted tokens, documents, bytes) of the collection, from the suite yamls (doc-isolated: n_tokens - 2 n_docs)."""
    suites = val_metrics.load_suites(SUITES, val_metrics.DEFAULT_SUITES_DIR)
    out = {}
    for key, stems in SETS.values():
        sp = [s for su in suites if su.name.startswith(stems) for s in su.splits]
        out[key] = (sum(s.n_tokens - 2 * s.n_docs for s in sp), sum(s.n_docs for s in sp), sum(s.n_bytes for s in sp))
    return out


def train_args(view):
    """What the run trains with, from the view's argv.txt / run.json (context for the report; None where absent)."""
    try:
        a = open(os.path.join(view, "argv.txt")).read().split()
        rj = json.load(open(os.path.join(view, "run.json")))
    except OSError:
        return None
    get = lambda k, f=str: f(a[a.index(k) + 1]) if k in a else None
    kd = get("--kd-cache-dir")
    return {"total_iterations": rj.get("iters"), "seq_length": get("--seq-length", int), "global_batch_size": get("--global-batch-size", int),
            "lr": get("--lr", float), "num_layers": get("--num-layers", int), "hidden_size": get("--hidden-size", int),
            "num_experts": get("--num-experts", int), "moe_router_topk": get("--moe-router-topk", int),
            "kd_alpha": get("--kd-alpha", float), "kd_teacher": os.path.basename(os.path.dirname(kd)) if kd else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    coll = collections()
    ln2 = math.log(2)
    n = 0
    for spec in RUNS:
        train = train_args(spec["view"])
        found = {}                                           # (iteration, window) -> [result records, newest first]
        for root in spec["roots"]:
            if not os.path.isdir(root):                      # a moved / renamed source must not silently drop its checkpoints
                print(f"WARNING: source directory missing, its results are NOT collected: {root}")
            for d in sorted(glob.glob(os.path.join(root, spec["prefix"] + "*"))):
                m = re.fullmatch(re.escape(spec["prefix"]) + r"(\d+)-(\d+)", os.path.basename(d))
                p = os.path.join(d, "results.json")
                if not m or not os.path.exists(p):
                    continue
                r = json.load(open(p))
                if not r.get("complete") or r.get("problems") or r.get("extra", {}).get("mode") != "doc":
                    print(f"SKIPPED (incomplete / problems / not a doc eval): {d}: complete={r.get('complete')} problems={r.get('problems')}")
                    continue
                if int(r["iteration"]) != int(m.group(1)):
                    print(f"SKIPPED (iteration {r['iteration']} != directory name): {d}")
                    continue
                r["_dir"], r["_by"] = d, d.split("/users/")[1].split("/")[0]
                found.setdefault((int(r["iteration"]), int(r["extra"]["seq"])), []).append(r)
        for (it, win), recs in sorted(found.items()):
            recs.sort(key=lambda r: (r["date"], int(r["job_id"])), reverse=True)
            r = recs[0]
            metrics, refused = {}, []
            for s, loss in r["doc"].items():
                st = r["doc_stats"][s]
                if s not in SETS:
                    refused.append(f"{s}: not one of our collections")
                    continue
                key = SETS[s][0]
                tok, docs, nbytes = coll[key]
                if st["predicted_tokens"] != tok or st["docs"] != docs:
                    refused.append(f"{s}: {st['predicted_tokens']} tokens / {st['docs']} docs, the collection {key} has {tok} / {docs}")
                    continue
                metrics[key] = {"loss": loss, "ppl": math.exp(loss), "bpb": loss * tok / (nbytes * ln2), "tokens": tok, "docs": docs, "set": s}
            for x in refused:
                print(f"REFUSED set of iter {it} window {win}: {x}")
            if not metrics:
                continue
            name = f"{spec['run']}{'' if win == 8192 else '_W%dK' % (win // 1024)}_iter{it:07d}"
            out = {"run": spec["run"], "display_name": spec["display"], "iteration": it, "window": win, "protocol": "doc_isolated",
                   "pooled_sets": True, "checkpoint": os.path.join(r["loaded"][0][0], f"iter_{it:07d}") if r.get("loaded") else None,
                   "train": train, "metrics": dict(sorted(metrics.items())),
                   "not_scored": sorted(k for k, _ in SETS.values() if k not in metrics), "refused_sets": refused,
                   "source": {"eval_dir": r["_dir"], "eval_job": int(r["job_id"]), "evaluated_at": r["date"], "by": r["_by"],
                              "script": r["extra"].get("by"), "megatron": r.get("megatron"), "megatron_commit": r.get("megatron_commit"),
                              "ladder_dir": r.get("ladder_dir")},
                   "other_evals": [{"eval_dir": o["_dir"], "eval_job": int(o["job_id"]), "evaluated_at": o["date"], "by": o["_by"], "doc": o["doc"]}
                                   for o in recs[1:]]}
            with open(os.path.join(a.out, name + ".json"), "w") as f:
                json.dump(out, f, indent=1)
                f.write("\n")
            n += 1
            print(f"{name}: " + "  ".join(f"{k} {v['loss']:.4f} (bpb {v['bpb']:.4f})" for k, v in out["metrics"].items())
                  + f"  [{r['_by']} job {r['job_id']}, {r['date']}]" + (f"  (+{len(recs) - 1} older eval)" if len(recs) > 1 else ""))
    print(f"wrote {n} JSON files -> {os.path.abspath(a.out)}")


if __name__ == "__main__":
    main()
