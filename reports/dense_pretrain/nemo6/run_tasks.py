#!/usr/bin/env python3
"""reports_xx/nemo6/run_tasks.py - run the nemo6 benchmark suite (nemo6_report.yml, 40 tasks) against ONE running OpenAI-compatible
server (vLLM / SGLang), several task lanes in parallel so the server stays busy (user 11-04: external references on the full suite).

The lm_eval command per task is the one nvbench/launch_evals.py builds for the vLLM engine (ADLR harness, Boxiang's frozen copy @17abcb7:
`--model local-completions`, base_url, model, tokenizer, tokenizer_backend=huggingface, max_concurrent_per_client), with the task's
YAML batch_size / num_fewshot / overrides / use_cache rules unchanged. Differences, all operational:
  * lanes: --lanes lm_eval processes at once, longest tasks first (Boxiang's 5B timings), judge tasks last;
  * --conc = max_concurrent_per_client per lane (the frozen launcher's `_dyn` preset would force batch_size 1 on every task);
  * a failed task does not stop the others (exit code 3 at the end); finished tasks are skipped (same load_results as the launcher);
  * judge credentials (simpleqa_verified_25shot / facts_parametric_25shot) are read from --judge-env (mode 600) for those two tasks only;
    a missing / empty key skips them (rerun later: finished tasks are skipped).
Run inside Boxiang's client image with his venv (see nemo6_ref.sbatch)."""
import argparse
import copy
import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import yaml

SRC = os.environ["LME_SRC_DIR"]
sys.path.insert(0, f"{SRC}/nvbench")
from assemble_results import load_results, tabulate_results  # noqa: E402
from launch_evals import vllm_model_args  # noqa: E402

JUDGE_TASKS = {"simpleqa_verified_25shot", "facts_parametric_25shot"}
RESUME_CACHE = os.environ.get("NEMO6_RESUME_CACHE", "1") == "1"   # see task_cmd
# tasks that execute generated programs at scoring time: at most one at a time (pilot 39048: LCB scoring OOM-killed the client)
CODE_TASKS = {"lcb_codegen", "lcb_codegen_sampled", "humaneval_sampled", "mbpp_sanitized_3shot_sampled",
              "bigcodebench_complete_greedy", "bigcodebench_instruct_greedy"}
# minutes per task for Boxiang's 5B (Megatron server, c8; muon_lr200_s0 iter 6000): used only to order the queue
EST = {"mmlu_pro_0_shot_sampled": 540, "wmt24pp_en_xx": 390, "hle_text_sampled": 300, "lcb_codegen_sampled": 54,
       "supergpqa_main_cot_n_shot": 49, "math_500_sampled": 29, "gpqa_diamond_cot_n_shot_sampled": 21,
       "mbpp_sanitized_3shot_sampled": 16.5, "mmlu_pro_5_shot_base": 14.6, "mmlu_str": 10.1, "minerva_math_nemo": 9.7,
       "lcb_codegen": 7.3, "hellaswag": 7.1, "humaneval_sampled": 7.1, "trivia_qa": 6.8, "mmlu_redux_str": 6.1,
       "adlr_include": 5.9, "agieval_en_cot": 5.2, "adlr_global_mmlu": 5.1, "popqa_5shot": 4.7, "hmmt_sampled": 4.1}


def task_cmd(task, out, model, host, port, conc, tokenizer):
    """Same construction as launch_evals.main() (harness ordering: the task's YAML batch_size wins)."""
    t = copy.deepcopy(task)
    name = t.pop("task_name")
    bs = t.pop("batch_size", None)
    use_cache = t.pop("use_cache", True)
    overrides = t.pop("overrides", {})
    gk = t.pop("generation_kwargs", None)
    use_cache &= t.pop("cache_requests", False)
    for k in ("report_metric", "task_group"):
        t.pop(k, None)
    eff = bs or 1
    # resumability (11-05: big refs need 10-40 h, normal QOS preempts after 4h05, 24 h limit): response cache for every task whose
    # repeats fit in one batch. Greedy requests: lm_eval's per-request cache. Sampled requests bypass that one (CachingLM skips
    # do_sample) and only hit the client's per-BATCH cache (load_balancer, key = md5 of the batch): a resumed run reuses finished
    # batches' genuine samples and samples the rest. Requests are sorted by context, so with repeats <= batch_size a batch never
    # repeats another batch's content; hmmt_sampled (repeats 32 > batch 16: two identical batches per question) keeps no cache.
    reps = int((overrides or {}).get("repeats", 1)) if isinstance(overrides, dict) else 1
    if RESUME_CACHE and reps <= eff:
        use_cache = True
    margs = vllm_model_args(host, port, model, eff, max_concurrent_per_client=conc, tokenizer=tokenizer)
    s = "".join(f"--{k}={v} " for k, v in t.items())
    if use_cache:
        s += f"--use_cache {out}/cache_{name}_ "
    od = dict(overrides) if isinstance(overrides, dict) else {}
    if isinstance(gk, dict):
        od.update({f"generation_kwargs.{k}": v for k, v in gk.items()})
    if od:
        s += '--override "' + " ".join(f"{k}={v}" for k, v in od.items()) + '"'
    return name, f"lm_eval {margs} --tasks {name} --batch_size {eff} {s} --log_samples --output_path {out} "


def judge_env(path):
    """KEY=VALUE lines of a mode-600 file (not shell-sourced). None (with a reason) when unusable."""
    p = Path(path)
    if not p.is_file():
        return None, f"{path} missing"
    if stat.S_IMODE(p.stat().st_mode) & 0o077:
        return None, f"{path} must be chmod 600"
    env = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    prov = env.get("PARAMETRIC_QA_JUDGE_PROVIDER", "")
    key = {"nvidia": env.get("NVIDIA_API_KEY") or env.get("NVIDIA_INFERENCE_API_KEY"),
           "openrouter": env.get("OPENROUTER_API_KEY")}.get(prov)
    if not key:
        return None, f"no API key for provider {prov!r} in {path}"
    return env, ""


def healthy(host, port):
    for path in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(f"http://{host}:{port}{path}", timeout=10) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", required=True)
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--conc", type=int, default=0, help="fixed max_concurrent_per_client; 0 = from --prompt-budget")
    ap.add_argument("--prompt-budget", type=int, default=2048,
                    help="prompts in flight per lane: conc = clamp(budget / task batch_size, 4, 128)")
    ap.add_argument("--tasks", default="", help="comma list of task names or YAML indices (default: all)")
    ap.add_argument("--judge-env", default="")
    ap.add_argument("--manifest", default="{}", help="JSON written into <out>/manifest.json")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    tasks = cfg["tasks"]
    if a.tasks:
        sel = set(a.tasks.split(","))
        tasks = [t for i, t in enumerate(tasks) if t["task_name"] in sel or str(i) in sel]
    out = Path(a.results_dir) / a.model_name
    (out / "logs").mkdir(parents=True, exist_ok=True)
    man = json.loads(a.manifest)
    man.update(config=a.config, config_sha256=hashlib.sha256(open(a.config, "rb").read()).hexdigest(), lme_src=SRC,
               tokenizer=a.tokenizer, lanes=a.lanes, conc=a.conc, slurm_job=os.environ.get("SLURM_JOB_ID"),
               started=time.strftime("%F %T"))
    (out / f"manifest_{os.environ.get('SLURM_JOB_ID', 'local')}.json").write_text(json.dumps(man, indent=1))

    done = {k[0] for k in load_results(out).keys()}
    todo = [t for t in tasks if t["task_name"] not in done]
    print(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already done, {len(todo)} to run; lanes {a.lanes} conc {a.conc}", flush=True)
    todo.sort(key=lambda t: (t["task_name"] in JUDGE_TASKS, -EST.get(t["task_name"], 3)))
    lock = threading.Condition()
    status = {}
    code_running = [False]
    times = out / "task_times.tsv"

    def take():
        """Next task in queue order; a code-execution task only when no other one is running (else wait for it)."""
        with lock:
            while todo:
                for i, t in enumerate(todo):
                    if t["task_name"] not in CODE_TASKS or not code_running[0]:
                        if t["task_name"] in CODE_TASKS:
                            code_running[0] = True
                        return todo.pop(i)
                lock.wait(60)
            return None

    def release(name):
        if name in CODE_TASKS:
            with lock:
                code_running[0] = False
                lock.notify_all()

    def lane(li):
        while True:
            t = take()
            if t is None:
                return
            conc = a.conc or max(4, min(128, a.prompt_budget // int(t.get("batch_size") or 1)))
            name, cmd = task_cmd(t, out, a.model_name, a.host, a.port, conc, a.tokenizer)
            try:
                run_one(li, t, name, cmd)
            finally:
                release(name)

    def run_one(li, t, name, cmd):
        if True:
            env = os.environ.copy()
            env["HF_ALLOW_CODE_EVAL"] = "1"
            if name in JUDGE_TASKS:
                je, why = judge_env(a.judge_env) if a.judge_env else (None, "no --judge-env")
                if je is None:
                    with lock:
                        status[name] = "skipped (judge: " + why + ")"
                    print(f"[lane {li}] SKIP {name}: judge credentials unusable ({why})", flush=True)
                    return
                env.update(je)
            if not healthy(a.host, a.port):
                with lock:
                    status[name] = "not run (server down)"
                print(f"[lane {li}] server not healthy, not starting {name}", flush=True)
                return
            idx = next(i for i, x in enumerate(cfg["tasks"]) if x["task_name"] == name)
            log = out / "logs" / f"{idx:02d}_{name}.log"
            print(f"[lane {li}] {time.strftime('%T')} start {name}  (log {log})", flush=True)
            t0 = time.time()
            with open(log, "a") as f:
                f.write(f"\n### {time.strftime('%F %T')} job {os.environ.get('SLURM_JOB_ID')}\n### {cmd}\n")
                f.flush()
                rc = subprocess.run(shlex.split(cmd), env=env, stdout=f, stderr=subprocess.STDOUT).returncode
            dt = (time.time() - t0) / 60
            with lock:
                status[name] = "ok" if rc == 0 else f"FAILED rc={rc}"
                with open(times, "a") as f:
                    f.write(f"{os.environ.get('SLURM_JOB_ID')}\t{name}\t{rc}\t{dt:.1f}\n")
            print(f"[lane {li}] {time.strftime('%T')} {'done' if rc == 0 else 'FAILED'} {name}  {dt:.1f} min", flush=True)

    th = [threading.Thread(target=lane, args=(i,), daemon=True) for i in range(a.lanes)]
    for x in th:
        x.start()
    for x in th:
        x.join()
    print("\n=== status ===")
    for k, v in status.items():
        print(f"  {k:36s} {v}")
    res = load_results(out, cfg)
    try:
        tabulate_results(res)
    except Exception as e:  # tabulation is cosmetic
        print(f"tabulate_results failed: {e}")
    sys.exit(0 if all(v == "ok" for v in status.values()) else 3)   # 3 = some task failed / skipped (sbatch: no relaunch)


if __name__ == "__main__":
    main()
