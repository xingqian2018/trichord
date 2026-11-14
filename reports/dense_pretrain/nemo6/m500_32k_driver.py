#!/usr/bin/env python3
"""reports_xx/nemo6/m500_32k_driver.py [--dry] - background driver of the MATH-500 + Minerva 32K re-run. Runs in tmux `m500_32k`.

Decision 11-10 (user + Runlong, "所有baseline和我们的模型，在测试math500/minerva上面，统一使用32K（最长生成）", "就是不要rollback"):
MATH-500 and Minerva MATH for every model at the longest generation a 32K context allows, instead of ADLR's 1024-token cap (which cut
5-19 % of our answers and ~11-12 % of Qwen3-235B's). nemo6_report.yml: math_500_sampled max_gen_toks 31232, minerva_math_nemo 30720
(+ the longest 4-shot prompt, 1354 / 1899 Qwen3 tokens, <= 32768); servers default to MAXSEQ / MML 32768. collect_nemo6.py / run_tasks.py
ignore any result of these two tasks generated with a smaller cap (MIN_GEN_TOKS), so the old scores are out of every summary archive
and the report ("-") until the 32K result of that model exists.

Every model that had these scores gets a job running only its missing tasks (TASKS=<missing>, NEMO6_RESUME_CACHE=0 so no cached
1024-token response can be reused): our ladder checkpoints via nemo6_ladder.sbatch, the references via launch_refs.sh (Qwen3-30B-A3B-Base,
not in its table, directly via nemo6_ref.sbatch). Report-visible rows first. Gates: ladder jobs wait for the ladder smoke (PROD1_DPSK_q5,
job 57509: MATH-500 at 32K), reference jobs for the reference smoke (QWEN3_30B_A3B_BASE), i.e. one finished 32K MATH-500 result per server
type. At most MAX_INFLIGHT jobs queued or running at once. A job that leaves the queue without finishing any missing task counts as a
failure and is resubmitted with its nodes excluded (standing rule: infra failures are retried at once), at most MAX_FAIL failures per
model. Thinking-2507: its full-suite job 57511 is watched (it reads the 32K config when it starts); missing tasks are submitted after it.

When a model finishes a task: collect_nemo6.py <NAME> (-> trichord / shared / repo summaries) + gen_eval_html.py (report + md mirror).
Archive: once a model has both 32K results and none of its jobs is queued, its old smaller-cap raw files of the two tasks (results file
holding only that task / its subtasks, every samples file of the same timestamp, the task's cache db) are MOVED (never deleted) to
runs/nemo6_eval/_superseded.math500_1024/<NAME>/ (the "." keeps collect_nemo6.py from reading it as a model).
State: runs/nemo6_eval/sbatch/m500_32k_state_v2.json; log: runs/nemo6_eval/sbatch/m500_32k_driver.log.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

U = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx"
REPO = f"{U}/Project/cosmos_llm_scaling_ladder"
N6 = f"{REPO}/reports_xx/nemo6"
RAW = f"{U}/runs/nemo6_eval"
STATE = f"{RAW}/sbatch/m500_32k_state_v2.json"
LOG = f"{RAW}/sbatch/m500_32k_driver.log"
SUPER = f"{RAW}/_superseded.math500_1024"
TASKS = {"math_500_sampled": 31232, "minerva_math_nemo": 30720}     # = MIN_GEN_TOKS of collect_nemo6.py / run_tasks.py
MAX_FAIL = 3
MAX_INFLIGHT = 12
POLL = 300
BAD_NODES = ["nvl72d135-t03", "nvl72d286-t03", "nvl72d220-t16", "nvl72d212-t15", "nvl72d379-t07"]   # d379-t07: Lustre client hung 11-10 (job 57509)
QOS = ["--qos=normal_quota", "--reservation=sla_res_quota"]
LD = f"{U}/runs/ladder_doc_eval"
SL = f"{U}/runs/scaling_ladder"
H = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/shared/huggingface_cache/hub"

# (NAME, RUN_DIR, PARITY_FROM) of our checkpoints, read 11-10 from their nemo6 job logs (m500_32k_ladder_rows.json: NAME, run, iter, parity, log)
LADDER = [
    ("PROD1_DPSK_q5", f"{SL}/prod_v1/qwen3_tpp200_prod1_dpsk-q5-tpp200-s1", f"{LD}/PROD1_DPSK__P1of3-q5-52824"),
    ("PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K_q3", f"{SL}/prod_v1_packplan/qwen3_e9r500_32k_prod1_dpsk_kd_a0p8_s64y_mimo25_packplan-q3-tpp200-s1",
     f"{LD}/PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K__P1of3-q3-42580"),
]
_rows = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m500_32k_ladder_rows.json")))
LADDER += [(n, rd, par) for n, rd, _it, par, _log in _rows if n not in {x[0] for x in LADDER}]
FIRST = ["PROD1_DPSK_q5", "PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K_q3", "PROD1_DPSK_q4",
         "PROD1_DPSK_LATENT2_E2X_QBX_KD_A0P8_S64Y_MIMO25_PPFULL_q3", "PROD1_DPSK_LATENT2_E2X_q3", "PROD1_DPSK_LATENT2_E2X_q4",
         "PROD1_DPSK_q3", "PROD1_DPSK_LATENT2_q3", "PROD1_DPSK_LATENT2_q4", "PROD1_DPSK_E2X_q3"]
LADDER.sort(key=lambda r: (FIRST.index(r[0]) if r[0] in FIRST else len(FIRST), r[0]))
REFS = ["QWEN3_30B_A3B_BASE", "QWEN3_235B_A22B_THINKING_2507", "QWEN3_235B_A22B", "MIMO_V25_PRO_BASE", "MIMO_V26_PRO_RL",
        "DEEPSEEK_V4_FLASH_BASE", "KIMI_K3", "MISTRAL_LARGE4", "QWEN38_27B", "QWEN38_FLASH_NEXT", "QWEN38_MAX"]
# jobs already submitted before this driver: the two smokes (57510 was cancelled during the short rollback -> resubmitted by the driver)
# and Thinking-2507's full-suite job
INITIAL = {"PROD1_DPSK_q5": 57509, "QWEN3_30B_A3B_BASE": 57510, "QWEN3_235B_A22B_THINKING_2507": 57511}
SMOKE = {"ladder": "PROD1_DPSK_q5", "ref": "QWEN3_30B_A3B_BASE"}
DRY = "--dry" in sys.argv


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def sh(cmd, env=None, cwd=REPO):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, env=env, cwd=cwd)
    return r.returncode, r.stdout, r.stderr


def results_files(name):
    return sorted(glob.glob(f"{RAW}/{name}/**/results_*.json", recursive=True), key=os.path.getmtime)


def task_cap(r, t):
    """max_gen_toks a results file used for task t (a group: its smallest subtask cap; 0 = not recorded)"""
    cfgs = r.get("configs", {})
    subs = [t] if t in cfgs else ((r.get("group_subtasks") or {}).get(t) or [])
    caps = [int(((cfgs.get(x) or {}).get("generation_kwargs") or {}).get("max_gen_toks") or 0) for x in subs]
    return min(caps) if caps else 0


def load(f):
    try:
        return json.load(open(f))
    except Exception:
        return {}


def done32k(name, task):
    for f in results_files(name):
        r = load(f)
        if task in r.get("results", {}) and r.get("config", {}).get("limit") is None and task_cap(r, task) >= TASKS[task]:
            return f
    return None


def missing(name):
    return [t for t in TASKS if not done32k(name, t)]


def queued(job):
    rc, out, _ = sh(["squeue", "-h", "-j", str(job), "-o", "%T"])
    # COMPLETING = ended (a cancelled job on a node whose Lustre client hangs can stay COMPLETING for hours, e.g. 57509 on nvl72d379-t07)
    return rc == 0 and out.strip() not in ("", "COMPLETED", "COMPLETING", "FAILED", "CANCELLED", "TIMEOUT")


def job_end(job):
    _, out, _ = sh(["sacct", "-X", "-n", "-P", "-j", str(job), "-o", "State,NodeList,ExitCode"])
    line = (out.strip().splitlines() or ["?||"])[-1].split("|")
    nodes = []
    if len(line) > 1 and line[1] and line[1] != "None assigned":
        _, hn, _ = sh(["scontrol", "show", "hostnames", line[1]])
        nodes = hn.split()
    return line[0], nodes, line[2] if len(line) > 2 else ""


def busy(name):
    """any job of this NAME (from its manifests) still queued?"""
    for m in glob.glob(f"{RAW}/{name}/manifest_*.json"):
        j = re.search(r"manifest_(\d+)\.json$", m)
        if j and queued(j.group(1)):
            return True
    return False


def submit(name, kind, tasks, exclude):
    ex = ",".join(sorted(set(BAD_NODES) | set(exclude)))
    env = dict(os.environ, TASKS=",".join(tasks), NEMO6_RESUME_CACHE="0", NAME=name)
    if kind == "ladder":
        _, rd, par = next(r for r in LADDER if r[0] == name)
        env.update(RUN_DIR=rd, PARITY_FROM=par)
        cmd = ["sbatch", "--parsable", "-N", "1", "-t", "12:00:00", *QOS, f"--exclude={ex}", "--export=ALL", f"{N6}/nemo6_ladder.sbatch"]
    elif name == "QWEN3_30B_A3B_BASE":
        env.update(MODEL=f"{H}/models--Qwen--Qwen3-30B-A3B-Base/snapshots/1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9",
                   IMAGE=f"{U}/containers/vllm-openai+v0.31.0.sqsh", TP="1", PP="1", MNS="512")
        cmd = ["sbatch", "--parsable", "-N", "1", "-t", "12:00:00", *QOS, f"--exclude={ex}", "--export=ALL", f"{N6}/nemo6_ref.sbatch"]
    else:
        env.pop("NAME")
        env.update(SBATCH_QOS="normal_quota", SBATCH_RESERVATION="sla_res_quota", EXCLUDE=ex)
        cmd = ["bash", f"{N6}/launch_refs.sh", name]
    if DRY:
        log(f"[dry] {name} TASKS={','.join(tasks)}: {' '.join(cmd)}")
        return None
    rc, out, err = sh(cmd, env=env)
    m = re.match(r"(\d+)", out.strip()) if cmd[0] == "sbatch" else re.search(r"job (\d+)", out)
    if rc != 0 or not m:
        log(f"SUBMIT FAILED {name}: rc {rc} {out.strip()[-300:]} {err.strip()[-300:]}")
        return None
    return int(m.group(1))


def archive_old(name):
    """move the old (< 32K cap) raw files of the two tasks aside; returns the number of files moved"""
    moved = 0
    dst = f"{SUPER}/{name}"

    def mv(p):
        nonlocal moved
        rel = os.path.relpath(p, f"{RAW}/{name}")
        os.makedirs(os.path.dirname(f"{dst}/{rel}"), exist_ok=True)
        shutil.move(p, f"{dst}/{rel}")
        moved += 1

    for f in results_files(name):
        r = load(f)
        res = set(r.get("results", {}))
        for t in TASKS:
            own = {t} | set((r.get("group_subtasks") or {}).get(t) or [])
            if t in res and res <= own and task_cap(r, t) < TASKS[t]:
                ts = os.path.basename(f)[len("results_"):-len(".json")]
                for p in [f] + glob.glob(os.path.join(os.path.dirname(f), f"samples_*_{ts}.jsonl")):
                    mv(p)
                break
    for t in TASKS:
        for c in glob.glob(f"{RAW}/{name}/cache_{t}_*.db"):
            mv(c)
    return moved


def refresh(names):
    for n in names:
        rc, out, _ = sh(["python3", f"{N6}/collect_nemo6.py", n])
        log(f"collect {n}: rc {rc} {' '.join(l for l in out.splitlines() if 'complete=' in l)[:160]}")
    rc, _, _ = sh(["python3", f"{REPO}/reports_xx/gen_eval_html.py"], cwd=f"{REPO}/reports_xx")
    log(f"report regenerated: rc {rc}")


def main():
    st = json.load(open(STATE)) if os.path.exists(STATE) else {}
    models = [(n, "ladder") for n, _, _ in LADDER] + [(n, "ref") for n in REFS]
    for n, _ in models:
        st.setdefault(n, {"jobs": [INITIAL[n]] if n in INITIAL else [], "fails": 0, "exclude": [], "done": False, "archived": False,
                          "gave_up": False, "missing_at_submit": list(TASKS)})
    log(f"start (v2: MATH-500 + Minerva): {len(models)} models ({sum(k == 'ladder' for _, k in models)} ladder, "
        f"{sum(k == 'ref' for _, k in models)} refs), dry={DRY}")
    while True:
        changed = []
        inflight = sum(1 for n, _ in models if st[n]["jobs"] and queued(st[n]["jobs"][-1]))
        for n, kind in models:
            s = st[n]
            if s["done"]:
                if not s["archived"] and not busy(n):
                    k = 0 if DRY else archive_old(n)
                    s["archived"] = True
                    log(f"archived {n}: {k} old 1024-token raw files -> {SUPER}/{n}")
                continue
            miss = missing(n)
            newly = [t for t in s.get("seen_missing", list(TASKS)) if t not in miss]
            if newly:
                changed.append(n)
                log(f"32K result {n}: {', '.join(newly)}")
            s["seen_missing"] = miss
            if not miss:
                s["done"] = True
                continue
            last = s["jobs"][-1] if s["jobs"] else None
            if last and queued(last):
                continue
            if last and s.get("seen_end") != last:
                state, nodes, ec = job_end(last)
                s["seen_end"] = last
                progress = [t for t in s["missing_at_submit"] if t not in miss]
                if progress:
                    log(f"job {last} of {n} ended {state}: finished {', '.join(progress)}; still missing {', '.join(miss)}")
                else:
                    s["fails"] += 1
                    s["exclude"] = sorted(set(s["exclude"]) | set(nodes))
                    log(f"job {last} of {n} ended {state} {ec} on {','.join(nodes) or '-'} without a 32K result (failure {s['fails']})")
            if s["gave_up"]:
                continue
            if s["fails"] >= MAX_FAIL:
                s["gave_up"] = True
                log(f"GAVE UP {n} after {s['fails']} failures (jobs {s['jobs']}) - needs a look")
                continue
            if not DRY and n not in SMOKE.values() and not done32k(SMOKE[kind], "math_500_sampled"):
                continue                                           # gate: one 32K MATH-500 result of this server type first
            if not DRY and inflight >= MAX_INFLIGHT:
                continue
            j = submit(n, kind, miss, s["exclude"])
            if j:
                s["jobs"].append(j)
                s["missing_at_submit"] = miss
                inflight += 1
                log(f"submitted {n} ({kind}) TASKS={','.join(miss)}: job {j}, exclude extra {s['exclude'] or '-'}")
            elif not DRY:
                s["fails"] += 1
        if not DRY:
            json.dump(st, open(STATE + ".tmp", "w"), indent=1)
            os.replace(STATE + ".tmp", STATE)
        if changed:
            refresh(changed)
        left = [n for n, _ in models if not st[n]["done"] and not st[n]["gave_up"]]
        unarch = [n for n, _ in models if st[n]["done"] and not st[n]["archived"]]
        if DRY or (not left and not unarch):
            log(f"{'dry run' if DRY else 'ALL DONE'}: done {sum(st[n]['done'] for n, _ in models)}, gave up "
                f"{[n for n, _ in models if st[n]['gave_up']]}")
            if not DRY:
                refresh([])
            return
        log(f"tick: {len(left)} left, {inflight} in flight, {len(unarch)} to archive")
        time.sleep(POLL)


if __name__ == "__main__":
    main()
