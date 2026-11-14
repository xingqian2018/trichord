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
          "QWEN3_235B_A22B_THINKING_2507": ("Qwen3-235B-A22B-Thinking-2507", True),   # user 11-10: the 235B benchmark of the downstream tab (job 57497)
          "QWEN38_FLASH_NEXT": ("Qwen3.8-Flash-Next (FP8)", True), "QWEN38_27B": ("Qwen3.8-27B", True),
          "QWEN38_MAX": ("Qwen3.8-Max (2.4T-A95B, FP8)", True), "DEEPSEEK_V4_FLASH_BASE": ("DeepSeek-V4-Flash-Base", False),
          "MIMO_V25_PRO_BASE": ("MiMo-V2.5-Pro-Base", False), "MIMO_V26_PRO_RL": ("MiMo-V2.6-Pro-RL", True),
          "KIMI_K3": ("Kimi-K3", True), "MISTRAL_LARGE4": ("Mistral-Large-4 (1T-A52B, FP8)", True),
          "RIPPLE_65B_BASE": ("Ripple-65B-Base-Preview (dense 65B)", False),
          # our ladder checkpoints (nemo6_ladder.sbatch, Megatron text-generation server; user 11-05 "全量测评" on KD 2.5 · PROD1_DPSK · 新数据 q3)
          "PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K_q3": ("KD 2.5 · PROD1_DPSK · new data (q3, iter 70438)", False),
          # 11-08 user "把q4 q5做全量测评": PROD1_DPSK CE (650B data) q4 / q5 finals (jobs 53746 / 53747)
          "PROD1_DPSK_q4": ("CE · PROD1_DPSK (q4, iter 120500)", False), "PROD1_DPSK_q5": ("CE · PROD1_DPSK (q5, iter 189968)", False),
          # 11-08 user "把我们一系列东西的评测都发了 … 主要是 downstream": every finished q3+ PROD1 arm, client fix v2 (jobs 54277-54287)
          "PROD1_DPSK_q3": ("CE · PROD1_DPSK (q3, iter 70438)", False),
          "PROD1_DPSK_LATENT2_q3": ("CE · LATENT2 (q3, iter 70438)", False),
          "PROD1_DPSK_LATENT2_q4": ("CE · LATENT2 (q4, iter 120500)", False),
          "PROD1_DPSK_E2X_q3": ("CE · DPSK_E2X 8+1/256 (q3, iter 70438)", False),
          "PROD1_DPSK_LATENT2_E2X_q3": ("CE · LATENT2_E2X 16+1/512 (q3, iter 70438)", False),
          "PROD1_DPSK_LATENT2_E2X_q4": ("CE · LATENT2_E2X 16+1/512 (q4, iter 120500)", False),
          "PROD1_DPSK_RS1_q3": ("CE · RS1 (q3, iter 70438)", False),
          "PROD1_DPSK_GAUX_q3": ("CE · GAUX (q3, iter 70438)", False),
          "PROD1_DPSK_GAUX_NOBIAS_q3": ("CE · GAUX_NOBIAS (q3, iter 70438)", False),
          "PROD1_DPSK_CLAMP20_q3": ("CE · CLAMP20 (q3, iter 70438)", False),
          "PROD1_DPSK_CLAMP7_q3": ("CE · CLAMP7 (q3, iter 70438)", False),
          # 11-08 user "我关心 QBK3 和 E2X 的对比": router pair at q1 / q2 (jobs 55248-55251)
          "PROD1_DPSK_LATENT2_E2X_q1": ("CE · LATENT2_E2X 16+1/512 (q1, iter 15824)", False),
          "PROD1_DPSK_LATENT2_E2X_q2": ("CE · LATENT2_E2X 16+1/512 (q2, iter 36606)", False),
          "PROD1_DPSK_LATENT2_E2X_QBK3_q1": ("CE · LATENT2_E2X + QBK3 router (q1, iter 15824)", False),
          "PROD1_DPSK_LATENT2_E2X_QBK3_q2": ("CE · LATENT2_E2X + QBK3 router (q2, iter 36606)", False),
          # 11-10 user "把我们的qbx q3 derisk，把那几个重要的downstream给测了": data derisk round 2 arm B q3, the 15 key-benchmark tasks only (job 57294)
          "PROD1_DPSK_LATENT2_E2X_QBX_KD_A0P8_S64Y_MIMO25_PPFULL_q3": ("KD 2.5 · LATENT2_E2X + QBX router · prepack data (q3, iter 70438)", False),
          # 11-10 user "QBK3 run 也发，系统检查这几个 MoE 的都发完了": router arms at q3 / q2 + QBK3_WU q1 (jobs 57335-57341)
          "PROD1_DPSK_LATENT2_E2X_QB_q3": ("CE · LATENT2_E2X + QB router (q3, iter 70438)", False),
          "PROD1_DPSK_LATENT2_E2X_QBX_q3": ("CE · LATENT2_E2X + QBX router (q3, iter 70438)", False),
          "PROD1_DPSK_LATENT2_E2X_QBK3_q3": ("CE · LATENT2_E2X + QBK3 router (q3, iter 70438)", False),
          "PROD1_DPSK_LATENT2_E2X_QBK3_WU_q1": ("CE · LATENT2_E2X + QBK3 router + LR warmup (q1, iter 15824)", False),
          "PROD1_DPSK_LATENT2_E2X_QB_q2": ("CE · LATENT2_E2X + QB router (q2, iter 36606)", False),
          "PROD1_DPSK_LATENT2_E2X_QBX_q2": ("CE · LATENT2_E2X + QBX router (q2, iter 36606)", False),
          "PROD1_DPSK_LATENT2_E2X_BIASONLY_q2": ("CE · LATENT2_E2X, bias only / no seq-aux (q2, iter 36606)", False)}

MEGATRON_SERVING = ("Megatron tools/run_dynamic_text_generation_server.py from the run's own Megatron tree, training argv (ladder_server_argv.py: "
                    "flash attention, per-layer SWA / NoPE / sinks as trained, EP 4), --default-temperature 1.0 --default-top-p 1.0 "
                    "--default-top-k 0 (= vllm --generation-config vllm); preflight megatron_api_check.py incl. doc eval parity; "
                    "client local-completions, max_length 2048 (harness default)")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.exists(path) else None


# 11-10 decision (user + Runlong): MATH-500 / Minerva for every model at the longest generation of a 32K context; results generated with a
# smaller cap (ADLR's 1024) are dropped from every summary ("-" in the report until the 32K run exists). Same table in run_tasks.py
MIN_GEN_TOKS = {"math_500_sampled": 31232, "minerva_math_nemo": 30720}


def task_cap(r, t):
    """max_gen_toks a results file used for task t (a group: the smallest over its subtasks; 0 = not recorded)."""
    cfgs = r.get("configs", {})
    if t in cfgs:
        return int(((cfgs[t] or {}).get("generation_kwargs") or {}).get("max_gen_toks") or 0)
    subs = (r.get("group_subtasks") or {}).get(t) or []
    caps = [int(((cfgs.get(x) or {}).get("generation_kwargs") or {}).get("max_gen_toks") or 0) for x in subs]
    return min(caps) if caps else 0


def load_results(d):
    """task -> (metrics dict, n-shot, n-samples, file); the newest full (limit None) results file wins per task."""
    out = {}
    for f in sorted(glob.glob(f"{d}/**/results_*.json", recursive=True), key=os.path.getmtime):
        r = json.load(open(f))
        if r.get("config", {}).get("limit") is not None:
            continue
        low = {t for t in MIN_GEN_TOKS if t in r.get("results", {}) and task_cap(r, t) < MIN_GEN_TOKS[t]}
        low |= {x for t in low for x in (r.get("group_subtasks") or {}).get(t, [])}
        for t, m in r.get("results", {}).items():
            if t in low:
                continue      # MATH-500 / Minerva at ADLR's 1024-token cap: superseded, "-" until the 32K run exists
            out[t] = (m, r.get("n-shot", {}).get(t), (r.get("n-samples", {}) or {}).get(t), os.path.basename(f))
    return out


# tasks whose `until` holds a whitespace-only stop ("\n" / "\n\n"): a model that answers "\n\n..." after the few-shot "A:" (post-trained
# refs, 11-05) gets an empty, zero-scored answer. empty_frac = share of empty responses (first EMPTY_N lines of each sample file).
WS_TASKS = ("trivia_qa", "naturalqs_5shot", "popqa_5shot", "simpleqa_5shot", "simpleqa_verified_25shot", "facts_parametric_25shot",
            "mmlu_str", "mmlu_redux_str", "wmt24pp_en_xx")
# empty_frac is also reported for BigCodeBench (11-06): the harness never runs its unittest TestCases (doc["test"] only defines the class)
# and passes every program that exits 0, so an EMPTY answer scores as a pass (ladder q3 instruct: 1114 / 1140 empty, "76 %").
# 11-06 (user: "有疑问的数字" must be "-"): empty_frac is computed for EVERY generation task (an audit found >= 20 % empty answers outside
# the list above, e.g. Qwen3.8-Flash-Next GPQA-Diamond 53 %, AGIEval 31 %; MiMo-V2.6 HLE 21 %). Generation task = a task whose report
# metrics are not only acc / acc_norm (those are multiple choice, nothing generated). Group tasks: samples files of their subtasks.
EMPTY_TASKS = WS_TASKS + ("bigcodebench_complete_greedy", "bigcodebench_instruct_greedy")   # kept for reference; see is_generation
SUBTASK_PREFIX = {"adlr_mgsm": "mgsm_native_cot_", "agieval_en_cot": "agieval_", "minerva_math_nemo": "minerva_math_"}
EMPTY_N = 400


# An answer that is empty, or that starts with an end-of-document marker (the model ended the document right after the prompt and the rest
# is an unrelated new document, e.g. DeepSeek-V4-Flash-Base on BigCodeBench instruct: "<｜end▁of▁file｜>\n<｜begin▁of▁file▁name｜>..."),
# is void: counted in empty_frac (11-06).
EOD_MARKERS = ("<｜end▁of▁file｜>", "<｜end▁of▁sentence｜>", "<|endoftext|>", "<|end_of_text|>", "<|im_end|>", "</s>")


def is_void(s):
    t = s.strip()
    return not t or t.startswith(EOD_MARKERS)


def is_generation(cfg_task):
    return not {rm["metric"].split(",")[0] for rm in cfg_task["report_metric"]} <= {"acc", "acc_norm"}


def empty_frac(d, task, all_tasks=()):
    latest = {}
    pre = SUBTASK_PREFIX.get(task, task + "_")
    longer = [o for o in all_tasks if o != task and o.startswith(task)]   # e.g. gpqa_diamond_cot_n_shot vs ..._sampled
    for f in sorted(set(glob.glob(f"{d}/*/samples_*.jsonl"))):
        sub = os.path.basename(f)[len("samples_"):].rsplit("_20", 1)[0]
        if sub != task and not sub.startswith(pre):
            continue
        if any(sub == o or sub.startswith(o + "_") for o in longer):
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
                    for s in (x if isinstance(x, list) else [x]):          # every sample of a repeats>1 task, not only the first
                        if isinstance(s, str):
                            n += 1
                            e += is_void(s)
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
        if is_generation(t) and (tn in res or any(r["alias"] in res for r in t["report_metric"])):
            entry["empty_frac"] = empty_frac(d, tn, [x["task_name"] for x in cfg["tasks"]])
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
                 "serving": {k: main.get(k) for k in ("image", "tp", "pp", "dp", "nn", "mml", "mns", "serve_extra", "server", "mlm", "ckiter",
                                                      "maxseq", "buffer_gb", "stop_token_ids", "drop_seed") if k in main},
                 "serving_fixed": (MEGATRON_SERVING if "megatron" in str(main.get("server", "")) else
                                   "vllm serve --generation-config vllm --no-enable-prefix-caching; client local-completions, max_length 2048 (harness default)"),
                 "client_image": main.get("client_image") or main.get("cimage"),
                 "harness_commit": open(f"{RT}/SOURCE_COMMIT").read().strip() if os.path.exists(f"{RT}/SOURCE_COMMIT") else None,
                 "config": cfg_path, "config_sha256": sha(cfg_path),
                 "client_patches": {"file": f"{HERE}/client_compat/nemo6_patches.py", "sha256": sha(f"{HERE}/client_compat/nemo6_patches.py"),
                                    "what": "stop<=4 sent to vLLM (client cuts on all until strings); LCB exec 32 workers + 16 GB RLIMIT_DATA"
                                            + ("; ladder: stop_token_ids 151643 + no seed" if main.get("drop_seed") else "")},
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
