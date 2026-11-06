#!/usr/bin/env python3
"""reports_xx/nemo6/collect_nemo6.py [NAME ...] - one compact JSON per model from the raw nemo6 runs (user 11-04 storage plan).

Raw (Lustre, not in git):  users/xingqianx/runs/nemo6_eval/<NAME>/  (lm_eval results_*.json + samples_*.jsonl, logs, manifests)
Summary <NAME>.json, written in this order (same convention as collect_doc_eval.py; copy / overwrite only, never delete):
  1. Project/trichord/dense_pretrain_result/nemo6/       source of truth (trichord git)
  2. users/xingqianx/dense_pretrain_result/nemo6/         group-readable copy
  3. <this repo>/results_xx/nemo6/                        read by the eval report
Content: every report metric of the config (value + stderr, 0-1 scale as lm_eval writes it; chrF++ too), grouped by task_group, with
n-shot / n-samples / run minutes per task; complete flag + missing list; meta = model, snapshot, serving (image, TP/PP/DP, flags),
harness commit, config sha256, client patches, judge model / endpoint (no secrets), Slurm jobs. Default NAMEs: every raw run dir.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time

import yaml

U = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx"
RAW = f"{U}/runs/nemo6_eval"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = f"{U}/Project/trichord/dense_pretrain_result/nemo6"        # source of truth
SHARE_DIR = f"{U}/dense_pretrain_result/nemo6"                         # group-readable copy
REPO_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "results_xx", "nemo6"))
RT = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_data/users/boxiangw/cosmos_lightning/lm-evaluation-harness/temp/nemo5_muon_eval"
JUDGE_ENV = f"{U}/private/nemo6_judge.env"
# display name, post-trained? (same names as gen_eval_html.py REF_MODELS)
MODELS = {"QWEN3_30B_A3B_BASE": ("Qwen3-30B-A3B-Base", False), "QWEN3_235B_A22B": ("Qwen3-235B-A22B", True),
          "QWEN38_FLASH_NEXT": ("Qwen3.8-Flash-Next (FP8)", True), "QWEN38_27B": ("Qwen3.8-27B", True),
          "QWEN38_MAX": ("Qwen3.8-Max (2.4T-A95B, FP8)", True), "DEEPSEEK_V4_FLASH_BASE": ("DeepSeek-V4-Flash-Base", False),
          "MIMO_V25_PRO_BASE": ("MiMo-V2.5-Pro-Base", False), "MIMO_V26_PRO_RL": ("MiMo-V2.6-Pro-RL", True),
          "KIMI_K3": ("Kimi-K3", True), "MISTRAL_LARGE4": ("Mistral-Large-4 (1T-A52B, FP8)", True),
          "RIPPLE_65B_BASE": ("Ripple-65B-Base-Preview (dense 65B)", False)}


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None


def load_results(d):
    """task -> (metrics dict, n-shot, n-samples, file); the newest full (limit None) results file wins per task."""
    out = {}
    for f in sorted(glob.glob(f"{d}/**/results_*.json", recursive=True), key=os.path.getmtime):
        r = json.load(open(f))
        if r.get("config", {}).get("limit") is not None:
            continue
        for t, m in r.get("results", {}).items():
            out[t] = (m, r.get("n-shot", {}).get(t), (r.get("n-samples", {}) or {}).get(t), os.path.basename(f))
    return out


# tasks whose `until` holds a whitespace-only stop ("\n" / "\n\n"): a model that answers "\n\n..." after the few-shot "A:" (post-trained
# refs, 11-05) gets an empty, zero-scored answer. empty_frac = share of empty responses (first EMPTY_N lines of each sample file).
WS_TASKS = ("trivia_qa", "naturalqs_5shot", "popqa_5shot", "simpleqa_5shot", "simpleqa_verified_25shot", "facts_parametric_25shot",
            "mmlu_str", "mmlu_redux_str", "wmt24pp_en_xx")
EMPTY_N = 400


def empty_frac(d, task):
    latest = {}
    for f in sorted(set(glob.glob(f"{d}/*/samples_{task}_*.jsonl"))):
        sub = os.path.basename(f)[len("samples_"):].rsplit("_20", 1)[0]
        if sub != task and not sub.startswith(task + "_"):
            continue
        if task == "mmlu_str" and sub.startswith("mmlu_redux"):
            continue
        if sub not in latest or os.path.getmtime(f) > os.path.getmtime(latest[sub]):
            latest[sub] = f
    n = e = 0
    for f in latest.values():
        with open(f) as fh:
            for i, line in enumerate(fh):
                if i >= EMPTY_N:
                    break
                for x in json.loads(line).get("resps", []):
                    s = x[0] if isinstance(x, list) and x else x
                    if isinstance(s, str):
                        n += 1
                        e += not s.strip()
    return round(e / n, 4) if n else None


def task_minutes(d):
    mins = {}
    p = f"{d}/task_times.tsv"
    if os.path.exists(p):
        for line in open(p):
            job, task, rc, m = line.rstrip("\n").split("\t")
            if rc == "0":
                mins[task] = float(m)
    return mins


def collect(name, cfg_path):
    d = f"{RAW}/{name}"
    cfg = yaml.safe_load(open(cfg_path))
    res, mins = load_results(d), task_minutes(d)
    groups, missing, n_metrics = {}, [], 0
    for t in cfg["tasks"]:
        tn = t["task_name"]
        rows = []
        for rm in t["report_metric"]:
            alias, metric = rm["alias"], rm["metric"]
            key = alias if alias in res and metric in res[alias][0] else tn      # sampled GPQA / MMLU-Pro: alias != result key
            m, nshot, nsamp, f = res.get(key, ({}, None, None, None))
            name_, _, filt = metric.partition(",")
            row = {"task": tn, "alias": alias, "result_key": key, "metric": metric, "value": m.get(metric),
                   "stderr": m.get(f"{name_}_stderr,{filt}"), "n_shot": nshot, "n_samples": nsamp, "results_file": f}
            n_metrics += 1
            if not isinstance(row["value"], (int, float)):
                missing.append(f"{alias} {metric}")
            rows.append(row)
        entry = {"task": tn, "minutes": mins.get(tn), "metrics": rows}
        if tn in WS_TASKS and tn in res:
            entry["empty_frac"] = empty_frac(d, tn)
        groups.setdefault(t["task_group"], []).append(entry)
    mans = [json.load(open(p)) for p in sorted(glob.glob(f"{d}/manifest_*.json"), key=os.path.getmtime)]
    main = next((m for m in reversed(mans) if m.get("model")), {})
    judge = {}
    if os.path.exists(JUDGE_ENV):                                          # non-secret fields only
        for line in open(JUDGE_ENV):
            k, _, v = line.strip().partition("=")
            if k in ("PARAMETRIC_QA_JUDGE_PROVIDER", "PARAMETRIC_QA_JUDGE_BASE_URL", "PARAMETRIC_QA_JUDGE_MODEL", "PARAMETRIC_QA_JUDGE_CONCURRENCY"):
                judge[k] = v
    model = main.get("model", "")
    disp, post = MODELS.get(name, (name, None))
    return {
        "name": name, "display_name": disp, "post_trained": post, "suite": os.path.basename(cfg_path).replace(".yml", ""),
        "complete": not missing, "n_tasks": len(cfg["tasks"]), "n_metrics": n_metrics, "missing": missing,
        "scale": "0-1 as written by lm_eval (chrF++ / 100 too); report x100",
        "groups": groups,
        "meta": {"model_path": model, "snapshot": model.rstrip("/").split("/")[-1] if "/snapshots/" in model else None,
                 "serving": {k: main.get(k) for k in ("image", "tp", "pp", "dp", "nn", "mml", "mns", "serve_extra") if k in main},
                 "serving_fixed": "vllm serve --generation-config vllm --no-enable-prefix-caching; client local-completions, max_length 2048 (harness default)",
                 "client_image": main.get("client_image") or main.get("cimage"),
                 "harness_commit": open(f"{RT}/SOURCE_COMMIT").read().strip() if os.path.exists(f"{RT}/SOURCE_COMMIT") else None,
                 "config": cfg_path, "config_sha256": sha(cfg_path),
                 "client_patches": {"file": f"{HERE}/client_compat/nemo6_patches.py", "sha256": sha(f"{HERE}/client_compat/nemo6_patches.py"),
                                    "what": "stop<=4 sent to vLLM (client cuts on all until strings); LCB exec 32 workers + 16 GB RLIMIT_DATA"},
                 "judge": judge, "slurm_jobs": sorted({m.get("slurm_job") for m in mans if m.get("slurm_job")}),
                 "raw_dir": d, "task_minutes_total": round(sum(v for v in mins.values()), 1),
                 "collected": time.strftime("%F %T")},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--config", default=f"{HERE}/nemo6_report.yml")
    ap.add_argument("--check", action="store_true", help="no writes: exit 0 iff every NAME has every report metric (nemo6_ref.sbatch skip)")
    a = ap.parse_args()
    if a.check:
        bad = [n for n in a.names if not os.path.isdir(f"{RAW}/{n}") or not collect(n, a.config)["complete"]]
        print("complete" if not bad else f"incomplete: {' '.join(bad)}")
        return 0 if a.names and not bad else 1
    names = a.names or sorted(n for n in os.listdir(RAW) if os.path.isdir(f"{RAW}/{n}") and "." not in n   # skip e.g. <NAME>.superseded_*
                              and glob.glob(f"{RAW}/{n}/**/results_*.json", recursive=True))
    os.makedirs(OUT_DIR, exist_ok=True)
    for n in names:
        s = collect(n, a.config)
        p = f"{OUT_DIR}/{n}.json"
        with open(p + ".tmp", "w") as f:
            json.dump(s, f, indent=1)
        os.replace(p + ".tmp", p)
        print(f"{n:24s} complete={s['complete']}  metrics {s['n_metrics'] - len(s['missing'])}/{s['n_metrics']}  "
              f"task-min {s['meta']['task_minutes_total']}  -> {p}")
        if s["missing"]:
            print("   missing:", ", ".join(s["missing"][:8]) + (" ..." if len(s["missing"]) > 8 else ""))
    for dst in (SHARE_DIR, REPO_DIR):                       # trichord first (above), then copies; copy / overwrite only
        os.makedirs(dst, exist_ok=True)
        for src in glob.glob(f"{OUT_DIR}/*.json"):
            shutil.copy2(src, f"{dst}/{os.path.basename(src)}")
    print(f"copied -> {SHARE_DIR}, {REPO_DIR}")


if __name__ == "__main__":
    sys.exit(main())
