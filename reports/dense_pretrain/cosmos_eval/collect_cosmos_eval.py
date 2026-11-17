#!/usr/bin/env python3
"""reports_xx/cosmos_eval/collect_cosmos_eval.py - read-only collector of the unified-harness full evaluation (user 11-13: "新的 baseline
全量测评结果在这里，以这个版本为准") into results_xx/cosmos_eval/<MODEL>.json, the form reports_xx/cosmos_eval_section.py renders.

    python3 collect_cosmos_eval.py            every listed model + every evaluated checkpoint of the tracked production run
    python3 collect_cosmos_eval.py NAME ...   only these

Source: <root>/<DIR>/<DIR>/results_*.json written by lm-evaluation-harness (branch ultra-prod, nvbench/launch_cosmos_reference.sbatch
/ launch_cosmos_ladder.sbatch), <DIR>/identity.json and <DIR>/gen_health.json. Nothing is run or submitted here, and nothing is written into
a source directory.

Task list = reports_xx/cosmos_eval/cosmos_ultra_ladder.yml, a copy of the harness's nvbench/cosmos_ultra_ladder.yml (the same bytes at
16f7620f / bc4d42ba / 66427095 / cfcd736a / 6bf2cd1a); its sha256 must equal the config_sha256 of every main run's identity.json (checked).

A listed model (MODELS, OURS) = an ordered list of result directories under runs/cosmos_eval_downstream, later ones override earlier ones task by task:
  <NAME>           the whole list at harness 16f7620f
  MATHFIX_<NAME>   the five math tasks again at cfcd736a ("Solution:" + space; with the newline layout of the main run models ended the
                   document at once), config mathfix.yml: these replace the main run's math tasks
  MiMo-V2.5-Pro-Base: only EOS_MIMO_V25_PRO_BASE (whole list, harness 6bf2cd1a, stop_token_ids 151643). MIMO_V25_PRO_BASE and
                   MATHFIX_MIMO_V25_PRO_BASE carry a VOID.txt (the server did not stop at the end-of-document token) and are never read.
A directory with a VOID.txt is refused.

Our own checkpoints (OURS, 11-13: the downstream tab is rewritten on this protocol) are collected the same way, with "ours": true in the JSON.
A model that lists a MATHFIX directory takes its five math tasks ONLY from it: while that re-run has no result for a task the task is
"pending" (value null), never the main run's value, which was produced with the earlier math prompt.

The tracked production run (TRACK, user 11-13: "这个我们主要收集的是这个prod测试，我们要跟踪这个") is not listed: every result directory
<TRACK>_iter<N> (and MATHFIX_<TRACK>_iter<N>) under TRACK_ROOTS is found by itself, ours and boxiangw's (his eval59 pipeline evaluates the
checkpoints at about every 10 % of training; it re-runs steps on newer harness commits under a new root). One JSON per step,
<TRACK>_iter<N 7 digits>.json, with "track": {run, step, total_steps, phase1_end}, "running" and "stopped". A task of a step comes from
the NEWEST result among the step's directories whose task definition is the report's.
running = an unfinished directory (no gen_health.json) that a job still writes: for our own directories a queued job names it as its
results directory (organize_cosmos_eval.queued_result_dirs(); only when the queue cannot be read: it changed within RUNNING_HOURS); for
boxiangw's directories, whose jobs we do not read: it changed within RUNNING_HOURS. stopped = not running, with an unfinished directory
and tasks without a score (the job was cancelled or failed, e.g. iter 24000 on 11-13 14:19 when the user had every eval job stopped):
the scores it has are kept, the other tasks are "pending": "stopped".

"The report's task definition" = the definition signature of the PILOT model's source for that task: a hash of the resolved task configs,
the task version and the shot counts that lm-eval recorded in the results file (the rule of boxiangw's launch/eval59/curve/build_curve.py,
so both trackers call the same scores comparable). It is checked for EVERY model: a score with another signature is withdrawn (null,
"pending": "definition") and reported on stdout. Scoring code outside the task configs is not in the hash; the harness commit is recorded.
"""
import glob
import hashlib
import json
import os
import re
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
# The run directories (one per job name, as the harness wrote them) live in runs/cosmos_eval_downstream/_runs since 2026-11-13 (user: "reorg now");
# runs/cosmos_eval_downstream/<MODEL>/ is the per-model view (summary, breakdowns, slurm) that organize_cosmos_eval.py builds from the JSONs written here.
SRC = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/cosmos_eval_downstream/_runs"
CFG = os.path.join(HERE, "cosmos_ultra_ladder.yml")
OUT = os.path.join(HERE, "..", "..", "results_xx", "cosmos_eval")
MATH = ["math_500_sampled", "hmmt_sampled", "minerva_math_nemo", "aime24", "aime25"]   # the tasks of mathfix.yml
# name (= the report's reference-model code) -> (display name, post-trained, [result dirs, later overrides earlier])
MODELS = {
    "QWEN3_30B_A3B_BASE": ("Qwen3-30B-A3B-Base", False, ["R1114_QWEN3_30B_A3B_BASE"]),
    "QWEN3_30B_A3B_INSTRUCT_2507": ("Qwen3-30B-A3B-Instruct-2507", True, ["QWEN3_30B_A3B_INSTRUCT_2507"]),   # 11-16 (user 11-15 "请补充测试downstream 66个分数",
    "QWEN3_30B_A3B_THINKING_2507": ("Qwen3-30B-A3B-Thinking-2507", True, ["QWEN3_30B_A3B_THINKING_2507"]),   #   "纯文本的两个"): jobs 70056 / 70057, judge replays 70227 / 70228
    "QWEN3_235B_A22B": ("Qwen3-235B-A22B", True, ["R1114_QWEN3_235B_A22B"]),
    "QWEN38_27B": ("Qwen3.8-27B", True, ["R1114_QWEN38_27B"]),
    "QWEN38_FLASH_NEXT": ("Qwen3.8-Flash-Next (FP8)", True, ["R1114_QWEN38_FLASH_NEXT"]),
    "QWEN38_MAX": ("Qwen3.8-Max (2.4T-A95B, FP8)", True, ["R1114_QWEN38_MAX"]),
    "MIMO_V25_PRO_BASE": ("MiMo-V2.5-Pro-Base", False, ["R1114_MIMO_V25_PRO_BASE"]),
    "MIMO_V26_PRO_RL": ("MiMo-V2.6-Pro-RL", True, ["R1114_MIMO_V26_PRO_RL"]),
}
# 2026-11-15 (user: "更新报告", after being told of them): the reference models' source is their full re-run of 11-14 (R1114_<name>, all 36
# tasks at harness 6bf2cd1a, jobs 67142-67148). Before: ["<name>", "MATHFIX_<name>"] (harness 16f7620f, math tasks re-run at cfcd736a) and
# ["EOS_MIMO_V25_PRO_BASE"]; those directories stay in baseline/<name>/runs/, so putting the old lists back restores the old scores.
# our ladder checkpoints: name = <Change ID>_<rung> (the code the report uses everywhere)
OURS = {
    "PROD1_DPSK_LATENT2_E2X_q3": ("PROD1_DPSK_LATENT2_E2X · q3", False, ["ALIGN_E2X_q3_16f7620f", "MATHFIX_ALIGN_E2X_q3_16f7620f"]),
    "PROD1_DPSK_LATENT2_E2X_QBK3_q3": ("PROD1_DPSK_LATENT2_E2X_QBK3 · q3", False, ["ALIGN_QBK3_q3_bc4d42ba", "MATHFIX_ALIGN_QBK3_q3_bc4d42ba"]),
}
# 11-17 (Runlong 00:4x "尝试几个加权平均的版本，都测试下，都发"): the 59B checkpoint averages of tmp/ablate59/merge59.sh are ours too, one row
# each (MERGE59_<wave>_<recipe>_iter<newest step>; wave M94 = newest checkpoint 94,000, M97 = the final 97,330). Only directories that
# already hold a results file (an empty one would stop the collection, see _layer).
MERGE_RECIPE = {"U2": "uniform, last 2", "U3": "uniform, last 3", "U5": "uniform, last 5", "U8": "uniform, last 8",
                "LIN5": "linear 5:4:3:2:1, last 5", "EXP6": "exp 0.67, last 6", "EXPr05": "exp 0.5, last 4",
                "POLYH": "Polyak, 2nd half of the decay", "POLY": "Polyak, all of phase 2",
                # 11-17 (Runlong "新的 ckpt 融合测试也发"): Boxiang's dense end checkpoints (every 250 steps, 95,250-97,250, read only)
                "DU3": "dense: uniform, last 3 (250-step)", "DU5": "dense: uniform, last 5", "DU10": "dense: uniform, last 10 (~2,080 steps)",
                "DLIN5": "dense: last 5, 5:4:3:2:1", "DEXPT500": "dense: exp(-d/500 steps)", "DEXPT1000": "dense: exp(-d/1000 steps)"}
for _d in sorted(glob.glob(os.path.join(SRC, "MERGE59_*_iter*"))):
    _n = os.path.basename(_d)
    _m = re.fullmatch(r"MERGE59_(M\d+)_(\w+?)_iter0*(\d+)", _n)
    if _m and glob.glob(os.path.join(_d, _n, "results_*.json")):
        OURS[_n] = (f"59B avg {_m.group(1)}_{_m.group(2)} ({MERGE_RECIPE.get(_m.group(2), _m.group(2))}; newest iter {int(_m.group(3)):,})", False, [_n])
MODELS.update(OURS)
# 11-17 (Runlong "recursive flint 的分没有测？"): RecursiveSI/flint, served by SGLang (no vLLM support; ~/tmp/flint/ds/
# launch_cosmos_reference_sglang.sbatch, job 75598); a post-trained coding-agent model. Added once it has results.
if glob.glob(os.path.join(SRC, "R1117_FLINT", "R1117_FLINT", "results_*.json")):
    MODELS["FLINT"] = ("Flint (Recursive, 40B-A2.5B)", True, ["R1117_FLINT"])
PILOT = "QWEN3_30B_A3B_BASE"                               # its sources define the task definitions every score must share
# the production run we track: training length and the end of its phase 1 as in boxiangw's build_curve.py (TOTAL_STEPS, PHASE1_END)
TRACK = "ULTRA59B_MUONH_KD_QBK3"
TRACK_TOTAL, TRACK_PHASE1 = 97330, 58000
TRACK_BX = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_data/users/boxiangw/cosmos_ultra_pre/runs/eval59"
TRACK_ROOTS = [("xingqianx", SRC)] + [("boxiangw", d) for d in sorted(glob.glob(os.path.join(TRACK_BX, "cosmos_eval*")))]
RUNNING_HOURS = 6                                          # an unfinished directory that changed within this time = still running (see the docstring)
_QUEUED = []                                               # [result directories of this user's queued eval jobs, or None], read once


def _cfg():
    return yaml.safe_load(open(CFG))["tasks"], hashlib.sha256(open(CFG, "rb").read()).hexdigest()


def _leaves(task, groups):
    sub = groups.get(task)
    return [x for s in sub for x in _leaves(s, groups)] if sub else [task]


def _signature(key, r):
    """definition signature of one results key (boxiangw's build_curve.py signature(), same bytes hashed)."""
    ls = sorted(set(_leaves(key, r.get("group_subtasks") or {})))
    blob = json.dumps({"cfg": [r.get("configs", {}).get(x) for x in ls], "ver": r.get("versions", {}).get(key),
                       "nshot": [r.get("n-shot", {}).get(x) for x in ls]}, sort_keys=True, default=str)
    blob = re.sub(r"0x[0-9a-fA-F]+", "0x", blob)           # reprs of functools.partial objects carry an address
    return hashlib.sha1(blob.encode()).hexdigest()[:10]


def _layer(dn, root=SRC, owner="xingqianx", aliases=(), allow_empty=False):
    """one result directory -> its results by key (file, row, git_hash, version, signature, date), groups, n-shot, n-samples, health,
    identity, whether it is finished (gen_health.json) and when it last changed."""
    d = os.path.join(root, dn)
    if os.path.exists(os.path.join(d, "VOID.txt")):
        sys.exit(f"{dn}: VOID.txt present - not a valid source")
    res, groups, nshot, nsamp = {}, {}, {}, {}
    files = sorted(glob.glob(os.path.join(d, dn, "results_*.json")))
    if not files and not (allow_empty or dn.startswith("MATHFIX_")):
        sys.exit(f"{dn}: no results_*.json")
    for f in files:                                        # sorted by timestamp: a later file wins for the same key
        try:
            r = json.load(open(f))
        except (OSError, ValueError):
            continue                                       # a file being written right now: the next collection reads it
        if "results" not in r or (r.get("config") or {}).get("limit") is not None:
            continue                                       # a --limit run is a test, not a score
        date = float(r.get("date") or os.path.getmtime(f))
        for k, row in r["results"].items():
            res[k] = (os.path.basename(f), row, r.get("git_hash"), r.get("versions", {}).get(k),
                      _signature(k, r) if k in aliases else None, date)
        groups.update({k: v for k, v in r.get("group_subtasks", {}).items() if v})
        nshot.update(r.get("n-shot", {}))
        nsamp.update(r.get("n-samples", {}))
    hp = os.path.join(d, "gen_health.json")
    health = json.load(open(hp)) if os.path.exists(hp) else {}
    ident = json.load(open(os.path.join(d, "identity.json")))
    mt = max(os.path.getmtime(p) for p in (d, os.path.join(d, dn), os.path.join(d, "slurm")) if os.path.exists(p))
    return {"dn": dn, "label": dn if root == SRC else f"{owner}:{os.path.basename(root)}/{dn}", "path": d, "owner": owner,
            "fix": dn.startswith("MATHFIX_"), "res": res, "grp": groups, "nshot": nshot, "nsamp": nsamp, "health": health, "ident": ident,
            "finished": os.path.exists(hp), "mtime": mt}


def _health(leaves, health):
    """responses-weighted share of empty / unanswered / capped responses over the generation subtasks; None for likelihood tasks."""
    rows = [health[x] for x in leaves if x in health and "responses" in health[x]]
    n = sum(r["responses"] for r in rows)
    if not n:
        return {}
    out = {k: sum((r.get(k) or 0.0) * r["responses"] for r in rows) / n for k in ("empty", "no_answer", "at_cap")}
    out["responses"] = n
    out["flags"] = sorted({f for r in rows for f in (r.get("flags") or [])})
    return out


def _listed(t, layers):
    """the source of a task for a listed model: the last layer that ran it; math tasks from the MATHFIX layer only when there is one."""
    has_fix = any(L["fix"] for L in layers)
    src = None
    for L in layers:
        if L["fix"] != (has_fix and t["task_name"] in MATH):   # math tasks: the MATHFIX layer only; the other tasks: never from it
            continue
        if any(rm["alias"] in L["res"] for rm in t["report_metric"]):
            src = L
    return src, ("mathfix" if not src and has_fix and t["task_name"] in MATH else None)


_REF = {}


def _ref_sigs(cfg, aliases):
    """alias -> the definition signature of the PILOT model's source: the report's definition of the task."""
    if not _REF:
        layers = [_layer(dn, aliases=aliases) for dn in MODELS[PILOT][2]]
        for t in cfg:
            src, _ = _listed(t, layers)
            for rm in t["report_metric"]:
                if src and rm["alias"] in src["res"]:
                    _REF[rm["alias"]] = src["res"][rm["alias"]][4]
        lack = sorted({rm["alias"] for t in cfg for rm in t["report_metric"]} - set(_REF))
        if lack:
            sys.exit(f"{PILOT}: no result (so no reference definition) for {lack}")
    return _REF


def _assemble(name, disp, post, ours, layers, choose, extra=None):
    cfg, sha = _cfg()
    ref = _ref_sigs(cfg, {rm["alias"] for t in cfg for rm in t["report_metric"]})
    groups, missing, n_met, differs = {}, [], 0, []
    for t in cfg:
        tn = t["task_name"]
        src, pend = choose(t, ref)
        if src:                                            # the score must be of the report's task definition
            bad = [rm["alias"] for rm in t["report_metric"] if rm["alias"] in src["res"] and src["res"][rm["alias"]][4] != ref[rm["alias"]]]
            if bad:
                differs.append(f"{tn} ({src['label']})")
                src, pend = None, "definition"
        mets = []
        for rm in t["report_metric"]:
            n_met += 1
            row = src["res"].get(rm["alias"]) if src else None
            v = row[1].get(rm["metric"]) if row else None
            if not isinstance(v, (int, float)):
                missing.append(f"{rm['alias']} {rm['metric']}")
                v = None
            base, _, filt = rm["metric"].partition(",")
            mets.append({"task": tn, "alias": rm["alias"], "metric": rm["metric"], "value": v,
                         "stderr": (row[1].get(f"{base}_stderr,{filt}") if row else None),
                         "version": row[3] if row else None, "results_file": row[0] if row else None})
        entry = {"task": tn, "metrics": mets, "source_dir": src["label"] if src else None}
        if not src and pend:
            entry["pending"] = pend        # mathfix: the math re-run has no result yet; running: the evaluation is still going; definition
        if src:
            res, grp, nshot, nsamp = src["res"], src["grp"], src["nshot"], src["nsamp"]
            leaves = sorted({x for rm in t["report_metric"] for x in _leaves(rm["alias"], grp)})
            ns = sorted({nshot[x] for x in leaves if isinstance(nshot.get(x), int)})
            entry["n_shot"] = ns[0] if len(ns) == 1 else None
            entry["n_docs"] = sum((nsamp.get(x) or {}).get("effective", 0) for x in _leaves(tn, grp)) or None
            entry["harness"] = src["ident"].get("harness_commit", "")[:8]
            entry["git_hash"] = sorted({res[rm["alias"]][2] for rm in t["report_metric"] if rm["alias"] in res})
            h = _health(leaves, src["health"])
            entry["health"] = h
            entry["empty_frac"] = h.get("empty")
        groups.setdefault(t["task_group"], []).append(entry)
    out = {"name": name, "display_name": disp, "post_trained": post, "ours": ours, "suite": "cosmos_ultra_ladder", "config_sha256": sha,
           "complete": not missing, "n_tasks": len(cfg), "n_metrics": n_met, "missing": missing,
           "scale": "0-1 as written by lm_eval; report x100", "groups": groups,
           "meta": {"source": SRC, "dirs": [L["label"] for L in layers], "identity": {L["label"]: L["ident"] for L in layers}}}
    out.update(extra or {})
    if differs:
        print(f"  !! {name}: withdrawn, task definition differs from {PILOT}'s: {differs}")
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, name + ".json"), "w"), indent=1, ensure_ascii=False)
    return out


def collect(name):
    disp, post, dirs = MODELS[name]
    cfg, sha = _cfg()
    aliases = {rm["alias"] for t in cfg for rm in t["report_metric"]}
    layers = [_layer(dn, aliases=aliases) for dn in dirs]
    for L in layers:
        if not L["fix"] and L["ident"].get("config_sha256") != sha:
            sys.exit(f"{L['dn']}: config_sha256 {L['ident'].get('config_sha256')} != our copy of the task list {sha}")
    return _assemble(name, disp, post, name in OURS, layers, lambda t, ref: _listed(t, layers))


def track_steps():
    """step -> [(owner, root, dir name)]: the tracked run's result directories that ran the report's task list (or a MATHFIX re-run)."""
    _, sha = _cfg()
    steps = {}
    for owner, root in TRACK_ROOTS:
        for dn in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            m = re.fullmatch(r"(MATHFIX_)?" + re.escape(TRACK) + r"_iter(\d+)", dn)
            d = os.path.join(root, dn)
            if not m or not os.path.isdir(d) or os.path.exists(os.path.join(d, "VOID.txt")):
                continue
            try:
                ident = json.load(open(os.path.join(d, "identity.json")))
            except (OSError, ValueError):
                continue
            if not m.group(1) and ident.get("config_sha256") != sha:
                continue                                   # evaluated with another version of the task list: not this protocol
            steps.setdefault(int(m.group(2)), []).append((owner, root, dn))
    return steps


def _queued():
    """real paths of the result directories that queued eval jobs of this user write; None when the queue cannot be read."""
    if not _QUEUED:
        try:
            import organize_cosmos_eval
            _QUEUED.append(organize_cosmos_eval.queued_result_dirs())
        except (Exception, SystemExit):
            _QUEUED.append(None)
    return _QUEUED[0]


def _is_running(L):
    if L["finished"]:
        return False
    if L["owner"] == "xingqianx" and _queued() is not None:
        return os.path.realpath(L["path"]) in _queued()   # our own jobs: the queue knows
    return time.time() - L["mtime"] < RUNNING_HOURS * 3600


def collect_track(step, dirs):
    """one checkpoint of the tracked run; None when none of its directories holds a score of the report's task definitions."""
    cfg, _ = _cfg()
    aliases = {rm["alias"] for t in cfg for rm in t["report_metric"]}
    layers = [_layer(dn, root, owner, aliases, allow_empty=True) for owner, root, dn in dirs]
    running = any(_is_running(L) for L in layers)
    halted = not running and any(not L["finished"] for L in layers)   # an evaluation that ended before its last task

    def choose(t, ref):
        best = None
        for L in layers:
            have = [rm["alias"] for rm in t["report_metric"] if rm["alias"] in L["res"]]
            if not have or any(L["res"][a][4] != ref[a] for a in have):
                continue                                   # not run here, or run with another definition of the task (e.g. the earlier math prompt)
            d = max(L["res"][a][5] for a in have)
            if best is None or d > best[0]:                # the newest result wins (as build_curve.py)
                best = (d, L)
        return (best[1], None) if best else (None, "running" if running else "stopped" if halted else None)

    name = f"{TRACK}_iter{step:07d}"
    out = _assemble(name, f"{TRACK} · iter {step}", False, True, layers, choose,
                    {"track": {"run": TRACK, "step": step, "total_steps": TRACK_TOTAL, "phase1_end": TRACK_PHASE1}, "running": running,
                     "stopped": halted})
    if out["stopped"] and not out["missing"]:              # every task has a score from another directory: nothing is missing
        out["stopped"] = False
        json.dump(out, open(os.path.join(OUT, name + ".json"), "w"), indent=1, ensure_ascii=False)
    if len(out["missing"]) == out["n_metrics"] and not running:
        os.remove(os.path.join(OUT, name + ".json"))       # nothing of this protocol and nothing coming: not a checkpoint of the report
        return None
    return out


def _line(o):
    bad = [(t["task"], round(t["empty_frac"], 2)) for ts in o["groups"].values() for t in ts if (t.get("empty_frac") or 0) >= 0.2]
    return (f"{o['name']}: {o['n_metrics'] - len(o['missing'])} / {o['n_metrics']} metrics, {o['n_tasks']} tasks, dirs {o['meta']['dirs']}"
            + ("; RUNNING" if o.get("running") else "; STOPPED before the end" if o.get("stopped") else "")
            + (f"; missing {o['missing']}" if o["missing"] and not (o.get("running") or o.get("stopped")) else "")
            + (f"; >= 20 % empty answers: {bad}" if bad else ""))


if __name__ == "__main__":
    want = sys.argv[1:]
    try:                                                   # a finished run that a job wrote at the old top level goes to _runs/ first
        import organize_cosmos_eval
        organize_cosmos_eval.sweep()
    except (Exception, SystemExit) as e:                   # the directory layout must never stop a collection
        organize_cosmos_eval = None
        print(f"organize_cosmos_eval: sweep failed, collecting from _runs/ as it is: {e!r}")
    steps = {f"{TRACK}_iter{s:07d}": (s, d) for s, d in sorted(track_steps().items())}
    unknown = [n for n in want if n not in MODELS and n not in steps]
    if unknown:
        sys.exit(f"unknown model(s) {unknown}: not in MODELS and not an evaluated checkpoint of {TRACK}")
    for name in (want or MODELS):
        if name in MODELS:
            print(_line(collect(name)))
    for name, (s, d) in steps.items():
        if not want or name in want:
            o = collect_track(s, d)
            print(_line(o) if o else f"{name}: no score of this protocol - skipped")
    if organize_cosmos_eval:
        try:                                               # runs/cosmos_eval_downstream/<MODEL>/: summary + breakdowns + slurm, from the JSONs written above
            organize_cosmos_eval.main(want, do_sweep=False)
        except (Exception, SystemExit) as e:
            print(f"organize_cosmos_eval: per-model directories not rebuilt: {e!r}")
