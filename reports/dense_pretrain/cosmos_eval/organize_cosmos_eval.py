#!/usr/bin/env python3
"""reports_xx/cosmos_eval/organize_cosmos_eval.py - keeps runs/cosmos_eval_downstream in the layout the user set on 2026-11-13:

    runs/cosmos_eval_downstream/
        baseline/                   external reference models        (user 11-13: "再拆下，baseline 和 ours_prod 和 ours_ladder",
        ours_ladder/                our ladder checkpoints            the same three groups as runs/cosmos_eval_doc_isolate)
        ours_prod/                  checkpoints of our production run
            index.md                the models of the group, one line each
            <model or checkpoint>/  one directory per model: the names of results_xx/cosmos_eval/<NAME>.json
                summary.md          every score of the model, as a table
                summary.json        the same, as collect_cosmos_eval.py wrote it (a copy of results_xx/cosmos_eval/<NAME>.json)
                identity.json       what was evaluated (link to the main run's identity.json)
                breakdowns/<task>/  exists only when the task is finished (every score there). No directory = not
                                    tested, or stopped half way: its answers so far are in the run's response cache
                                    and a resubmitted job goes on from them (user 2026-11-13).
                                    summary.json: the task's scores (accuracy ...), per subtask too; for a run with
                                    final records also which few-shot examples the questions were given ("examples")
                                    a multiple-choice task: answers.json, which question -> which option (compact_choices.py)
                                    a generation task: results.json, samples_<subtask>.jsonl, generations_<subtask>.jsonl.gz
                                    of the run that supplied the score. All links into _runs/ except summary.json
                slurm/<run>/        logs and preflight records of the runs behind the model: links into runs/
                runs/<run name>/    the raw directories of the runs behind the model, as the harness wrote them (one per job
                                    name, also the math re-run). The only real data of the folder: breakdowns/, slurm/ and
                                    identity.json are links into it, so the folder can be copied as one piece (user
                                    2026-11-13: "把_runs/里面对应的文件，放在对应的model 测试文件夹里面"; he chose "the whole
                                    raw directory moves in").
        _runs/<run name>            a link to <group>/<model>/runs/<run name>: the path jobs write to (EVAL_ROOT=<root>/_runs)
                                    and the collector reads. A real directory here = a run a queued job still writes to,
                                    or one that no model of the report lists yet.
        _doc_eval59/ _slurm/        not touched

    python3 organize_cosmos_eval.py [--dry-run] [NAME ...]

Two steps, both safe to repeat:
  sweep   a run directory that a job wrote at the top level (the old EVAL_ROOT) is moved to _runs/ as soon as no queued
          job writes to it. While a job writes to it, it stays, and _runs/<name> is a link to it, so that readers of
          _runs/ see it. Every move is one rename, appended to _runs/MOVES.log ("<time> <from> -> <to>"); to undo,
          rename back.
  adopt   a raw run directory in _runs/ that a model of the report lists and no queued job writes to is moved into that
          model's folder (runs/<run>), and _runs/<run> becomes a link to it. One rename, logged in _runs/MOVES.log.
  build   the per-model directories are rebuilt from results_xx/cosmos_eval/*.json (all of them, or NAME ...): the
          links and the summary files are made again; runs/ is never touched.
  clean   the response cache of a task is deleted as soon as the task's score is out (user 2026-11-13: "某个正式breakdown
          分数出来了后，就可以删除相应的cache了"), also in a run whose job is still working on other tasks. Only in runs
          that a model of the report lists: other directories in _runs/ are never touched. A cache is only
          what a stopped task resumes from; once the task has its score and its per-question record it is a second copy
          of the answers. Kept: the cache of a task without a score. Logged in _runs/DELETED.log.
  final   new runs only (results stamped from finalize_run.FINAL_FROM on; user 2026-11-13: "过去的结果就不用更改了，把测试代码
          之后的最终存储（就是cache变folder那一步）改成存example ID就行"): once no job writes into a run, the per-question
          records of its finished tasks become final records - question ids, example ids, answers, scores, no prompt and
          no dataset record. The work needs the harness environment and minutes of CPU, so it runs as a CPU job that this
          step starts when something waits (finalize_run.py); that job rebuilds the per-model directories it touched.

collect_cosmos_eval.py calls sweep() before it reads _runs/ and main(..., do_sweep=False) after it wrote the JSONs.
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/cosmos_eval_downstream"
RUNS = os.path.join(ROOT, "_runs")
JSONS = os.path.normpath(os.path.join(HERE, "..", "..", "results_xx", "cosmos_eval"))
# source labels "boxiangw:<root name>/<run>" of collect_cosmos_eval.py resolve here (read only: links point into it)
FOREIGN = {"boxiangw": "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_data/users/boxiangw/cosmos_ultra_pre/runs/eval59"}
SUMMARY_FILES = ("summary.json", "summary.md")
TASK_FILES = ("summary.json",)                         # the real (not linked) files of a breakdowns/<task>/ directory
# group of a model, from its JSON: a checkpoint of the tracked production run ("track"), another checkpoint of ours ("ours"), else a reference
CATEGORIES = {"baseline": "External reference models.",
              "ours_ladder": "Our ladder checkpoints (`<Change ID>_<rung>`).",
              "ours_prod": "Checkpoints of our production run (`<run>_iter<N>`)."}


def category(o):
    return "ours_prod" if o.get("track") else "ours_ladder" if o.get("ours") else "baseline"


# ------------------------------------------------------------------------------------------------ sweep
def is_view(d):
    """a per-model directory built here (the harness never writes a summary.json at the top of a run directory)."""
    return os.path.isfile(os.path.join(d, "summary.json")) and not os.path.islink(d)


def raw_runs_at_top():
    out = []
    for n in sorted(os.listdir(ROOT)):
        d = os.path.join(ROOT, n)
        if n.startswith(("_", ".")) or n in CATEGORIES or os.path.islink(d) or not os.path.isdir(d) or is_view(d):
            continue
        out.append(n)
    return out


def queued_result_dirs():
    """result directories that a queued job of this user writes: the "results <dir>" line the eval jobs print at their start.
    None when the queue cannot be read (then nothing unfinished is moved)."""
    try:
        q = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%Z"], capture_output=True, text=True, timeout=120)
        if q.returncode:
            return None
        out = set()
        for line in q.stdout.splitlines():
            jid, _, workdir = line.partition("|")
            if "lm-evaluation-harness" not in workdir or jid.split("_")[0] == os.environ.get("SLURM_JOB_ID"):
                continue                                     # not a harness job, or the job this very process runs in
            s = subprocess.run(["scontrol", "show", "job", "-o", jid.split("_")[0]], capture_output=True, text=True, timeout=120).stdout
            m = re.search(r"\bStdOut=(\S+)", s)
            if not m or not os.path.exists(m.group(1)):
                return None                                  # a harness job whose target is unknown: move nothing unfinished
            with open(m.group(1), errors="replace") as f:
                head = f.read(200000)
            found = re.findall(r"\bresults (/\S+)", head)
            if not found:
                return None
            out.update(os.path.realpath(p) for p in found)
        return out
    except (OSError, subprocess.SubprocessError):
        return None


def in_use(d, queued):
    """a queued job writes to it. When the queue cannot be read (None), everything counts as in use and nothing is moved.
    A stopped run is not in use: it goes to _runs/ at once and is resumed there (EVAL_ROOT=<root>/_runs)."""
    return queued is None or os.path.realpath(d) in queued


def sweep(dry=False, log=print):
    """move finished run directories from the top level into _runs/; link the ones still in use. Returns the names moved."""
    names = raw_runs_at_top()
    if not names:
        return []
    queued = queued_result_dirs()
    if not dry:
        os.makedirs(RUNS, exist_ok=True)
    moved = []
    for n in names:
        src, dst = os.path.join(ROOT, n), os.path.join(RUNS, n)
        if in_use(src, queued):
            if not os.path.lexists(dst):
                log(f"  in use, stays at the top level; _runs/{n} -> ../{n}")
                if not dry:
                    os.symlink(os.path.join("..", n), dst)
            continue
        if os.path.islink(dst) and os.path.realpath(dst) == os.path.realpath(src):
            if not dry:
                os.unlink(dst)                               # the link of its running time
        elif os.path.lexists(dst):
            log(f"  !! {n}: _runs/{n} already exists - not moved")
            continue
        log(f"  move {n} -> _runs/{n}")
        if not dry:
            os.rename(src, dst)
            with open(os.path.join(RUNS, "MOVES.log"), "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {src} -> {dst}\n")
        moved.append(n)
    return moved


# ------------------------------------------------------------------------------------------------ adopt
def _link_run(dst, dry=False):
    """_runs/<run> -> the run's directory inside its model folder."""
    src = os.path.join(RUNS, os.path.basename(dst))
    if os.path.islink(src) and os.path.realpath(src) != os.path.realpath(dst):
        if not dry:
            os.unlink(src)
    if not os.path.lexists(src) and not dry:
        os.symlink(os.path.relpath(os.path.realpath(dst), os.path.realpath(RUNS)), src)


def adopt(names=None, dry=False, log=print):
    """the raw run directories go into the folder of the model they belong to. Returns how many were moved."""
    names = list(names) if names else sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(JSONS, "*.json")))
    queued, asked, moved = None, False, 0
    for name in names:
        try:
            with open(os.path.join(JSONS, name + ".json")) as f:
                o = json.load(f)
        except (OSError, ValueError):
            continue
        cat = category(o)
        view = os.path.join(ROOT, cat, name)
        for c in CATEGORIES:                                 # the model changed its group: the folder moves as a whole, with its runs
            other = os.path.join(ROOT, c, name)
            if c != cat and os.path.isdir(os.path.join(other, "runs")) and not os.path.lexists(view):
                log(f"  {name}: {c}/ -> {cat}/")
                if not dry:
                    os.makedirs(os.path.dirname(view), exist_ok=True)
                    os.rename(other, view)
        for label in o["meta"]["dirs"]:
            if ":" in label:
                continue                                     # another person's directory: never moved
            src, dst = os.path.join(RUNS, label), os.path.join(view, "runs", label)
            if os.path.isdir(dst) and not os.path.islink(dst):
                if os.path.isdir(src) and not os.path.islink(src):
                    log(f"  !! {label}: a directory in _runs/ and one in {cat}/{name}/runs/ - nothing moved")
                else:
                    _link_run(dst, dry)                      # already there: only make sure the link is right
                continue
            if os.path.islink(src) or not os.path.isdir(src):
                continue
            if not asked:
                queued, asked = queued_result_dirs(), True
            if in_use(src, queued):
                log(f"  {label}: a queued job writes into it (or the queue cannot be read): stays in _runs/")
                continue
            log(f"  move _runs/{label} -> {cat}/{name}/runs/{label}")
            if not dry:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                os.rename(src, dst)
                _link_run(dst)
                with open(os.path.join(RUNS, "MOVES.log"), "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {src} -> {dst}  (and {src} is now a link to it)\n")
            moved += 1
    return moved


# ------------------------------------------------------------------------------------------------ build
def run_dir(label):
    """absolute directory of a source label of collect_cosmos_eval.py: "<run>" (ours) or "<owner>:<root name>/<run>"."""
    if ":" in label:
        owner, rest = label.split(":", 1)
        return os.path.join(FOREIGN[owner], rest), False
    return os.path.join(RUNS, label), True


def _fmt(v, scale=100.0):
    return "-" if not isinstance(v, (int, float)) else f"{v * scale:.2f}"


def summary_md(o):
    done = o["n_metrics"] - len(o["missing"])
    lines = [f"# {o['display_name']}", "",
             f"`{o['name']}` | task list `{o['suite']}` | {done} of {o['n_metrics']} metrics ({o['n_tasks']} tasks)"
             + (" | **still running**" if o.get("running") else " | **stopped before the end**" if o.get("stopped") else "") + f" | written {time.strftime('%Y-%m-%d %H:%M')}", "",
             "Scores are x100. `-` = no valid score (see note). Runs: " + ", ".join(f"`{d}`" for d in o["meta"]["dirs"]) + ".",
             "Per task: `breakdowns/<task>/`. Same numbers as `summary.json`.", ""]
    for group, entries in o["groups"].items():
        lines += [f"## {group}", "", "| task | metric | score | +/- | shots | docs | run | harness | note |", "|---|---|---:|---:|---:|---:|---|---|---|"]
        for e in entries:
            h = e.get("health") or {}
            note = []
            if e.get("pending"):
                note.append({"mathfix": "waiting for the math re-run", "running": "evaluation still running",
                             "stopped": "the evaluation was stopped before this task finished",
                             "definition": "withdrawn: task definition differs from the reference run"}.get(e["pending"], e["pending"]))
            for key, text in (("empty", "empty answers"), ("no_answer", "no answer"), ("at_cap", "at the token cap")):
                if (h.get(key) or 0) >= 0.2:
                    note.append(f"{h[key] * 100:.0f} % {text}")
            for i, m in enumerate(e["metrics"]):
                name = m["alias"] if m["alias"] == e["task"] else f"{e['task']} / {m['alias']}"
                lines.append(f"| {name if i == 0 or m['alias'] != e['task'] else ''} | `{m['metric']}` | {_fmt(m['value'])} | {_fmt(m.get('stderr'))} | "
                             f"{e.get('n_shot') if e.get('n_shot') is not None else ''} | {e.get('n_docs') or ''} | "
                             f"{e.get('source_dir') or ''} | {e.get('harness') or ''} | {'; '.join(note) if i == 0 else ''} |")
        lines.append("")
    return "\n".join(lines)


def task_summary(task, aliases, label, results_path):
    """breakdowns/<task>/summary.json: the scores of one task as the harness wrote them (0-1 scale), per subtask too.
    When the task has its final record (finalize_run.py), also which few-shot examples the questions were given."""
    with open(results_path) as f:
        d = json.load(f)
    res, nsamp, nshot = d.get("results") or {}, d.get("n-samples") or {}, d.get("n-shot") or {}
    leaves = sorted(d.get("configs") or {})

    def row(k):
        return {m: v for m, v in (res.get(k) or {}).items() if m != "alias" and isinstance(v, (int, float))}

    shots = sorted({nshot[x] for x in leaves if isinstance(nshot.get(x), int)})
    out = {"task": task, "run": label, "results_file": os.path.basename(results_path), "harness": d.get("git_hash"),
           "scale": "0-1 as written by lm_eval", "n_shot": shots[0] if len(shots) == 1 else None,
           "questions": sum((nsamp.get(x) or {}).get("effective", 0) for x in leaves) or None,
           "metrics": row(aliases[0]) if len(aliases) == 1 else {a: row(a) for a in aliases}}
    parts = {k: {"questions": (nsamp.get(k) or {}).get("effective"), "metrics": row(k)} for k in res if k not in aliases}
    final = os.path.join(os.path.dirname(results_path), f"final_{task}_{os.path.basename(results_path)[len('results_'):]}")
    if os.path.exists(final):
        with open(final) as f:
            fin = json.load(f)
        out["record"] = fin["record"]
        out["prompts_checked"] = fin["prompts_checked"]
        ex = {k: v["examples"] for k, v in fin["subtasks"].items()}
        if len(ex) == 1 and not parts:
            out["examples"] = next(iter(ex.values()))
        else:
            for k, v in ex.items():
                parts.setdefault(k, {})["examples"] = v
    if parts:
        out["subtasks"] = parts
    return out


def _link(target, new, view, rel, own, problems):
    """the link `rel` of the model folder `view`, written under `new` (it is swapped in afterwards). A target of ours is
    named relative to the link's final place, through real paths: a file of a run inside the folder gives a link that
    stays inside the folder."""
    if not os.path.exists(target):
        problems.append(f"missing {target}")
        return 0
    link = os.path.join(new, rel)
    os.makedirs(os.path.dirname(link), exist_ok=True)
    final_dir = os.path.dirname(os.path.join(os.path.realpath(view), rel))
    os.symlink(os.path.relpath(os.path.realpath(target), final_dir) if own else target, link)
    return 1


def _check_view(path):
    """a built part of a per-model directory (breakdowns/, slurm/, a left-over) may be removed only when it holds nothing
    but links and the summary files. runs/ is never passed here."""
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            p = os.path.join(dirpath, f)
            if not (os.path.islink(p) or f in SUMMARY_FILES or (f == "finish.flag" and os.path.getsize(p) == 0)):   # the empty flag of 2026-11-13
                sys.exit(f"{p} is a real file inside a built part of a per-model directory: not rebuilding {path}")


def _remove_view(path):
    _check_view(path)
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        for n in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            os.unlink(os.path.join(dirpath, n))
        for d in dirnames:
            if not os.path.islink(os.path.join(dirpath, d)) and os.path.isdir(os.path.join(dirpath, d)):
                os.rmdir(os.path.join(dirpath, d))
    os.rmdir(path)


def _ours(view):
    """a per-model directory built here, or the start of one (only its runs/ so far)."""
    return is_view(view) or (os.path.isdir(view) and not os.path.islink(view) and set(os.listdir(view)) <= {"runs"})


def build(name, dry=False, log=print):
    """rebuild the links and summary files of runs/cosmos_eval_downstream/<group>/<name>/ from results_xx/cosmos_eval/<name>.json.
    Returns (links made, problems). The new parts are made in <name>/.new_<pid>/ and swapped in one by one; runs/ (the raw
    data) is not touched. A run that is still in _runs/ (in use) is linked there."""
    src_json = os.path.join(JSONS, name + ".json")
    with open(src_json) as f:
        raw = f.read()
    o = json.loads(raw)
    cat = category(o)
    view = os.path.join(ROOT, cat, name)
    if os.path.lexists(view) and not _ours(view):
        log(f"  !! {cat}/{name} exists and is not a per-model directory built here - left alone")
        return 0, []
    new = os.path.join(view, f".new_{os.getpid()}")
    n, problems, plan, summaries = 0, [], [], []              # plan: (target, path inside the folder, ours)
    for entries in o["groups"].values():
        for e in entries:
            if not e.get("source_dir"):
                continue
            run, own = run_dir(e["source_dir"])
            dn = os.path.basename(run)
            files = sorted({m["results_file"] for m in e["metrics"] if m.get("results_file")})
            if e.get("pending") or not all(isinstance(m.get("value"), (int, float)) for m in e["metrics"]):
                continue                                             # not finished: no directory (it resumes from its cache)
            bd = os.path.join("breakdowns", e["task"])
            aliases = sorted({m["alias"] for m in e["metrics"] if m.get("results_file")})
            if len(files) == 1 and os.path.exists(os.path.join(run, dn, files[0])):
                summaries.append((os.path.join(bd, "summary.json"), e["task"], aliases, e["source_dir"], os.path.join(run, dn, files[0])))
            choices = [os.path.join(run, dn, f"choices_{e['task']}_{rf[len('results_'):-len('.json')]}.json") for rf in files]
            if len(files) == 1 and os.path.exists(choices[0]):
                plan.append((choices[0], os.path.join(bd, "answers.json"), own))    # a multiple-choice task: nothing else is needed
                continue
            for rf in files:
                plan.append((os.path.join(run, dn, rf), os.path.join(bd, "results.json" if len(files) == 1 else rf), own))
                stamp = rf[len("results_"):-len(".json")]
                samples = sorted(glob.glob(os.path.join(glob.escape(os.path.join(run, dn)), f"samples_*_{glob.escape(stamp)}.jsonl")))
                if not samples:
                    problems.append(f"{e['task']}: no samples file next to {rf}")
                for s in samples:
                    leaf = os.path.basename(s)[len("samples_"):-len(f"_{stamp}.jsonl")]
                    plan.append((s, os.path.join(bd, f"samples_{leaf}.jsonl"), own))
                    g = os.path.join(run, "generations", leaf + ".jsonl.gz")
                    if os.path.exists(g):
                        plan.append((g, os.path.join(bd, f"generations_{leaf}.jsonl.gz"), own))
    ident = None
    for label in o["meta"]["dirs"]:
        run, own = run_dir(label)
        if own and os.path.isdir(os.path.join(run, "slurm")):
            plan.append((os.path.join(run, "slurm"), os.path.join("slurm", label), True))
        if own and ident is None and not label.startswith("MATHFIX_") and os.path.exists(os.path.join(run, "identity.json")):
            ident = os.path.join(run, "identity.json")
    if ident:
        plan.append((ident, "identity.json", True))
    if dry:
        missing = [t for t, _, _ in plan if not os.path.exists(t)]
        log(f"  {name}: {len(plan)} links ({len(plan) - len(missing)} targets exist)" + (f"; {len(problems) + len(missing)} problems" if problems or missing else ""))
        return len(plan), problems + [f"missing {t}" for t in missing]
    os.makedirs(view, exist_ok=True)                         # creates the group directory as well
    for left in glob.glob(os.path.join(glob.escape(view), ".new_*")) + glob.glob(os.path.join(glob.escape(view), ".old_*")):
        _remove_view(left)                                   # what an interrupted rebuild left behind: links and summaries only
    os.makedirs(new)
    for target, rel, own in plan:
        n += _link(target, new, view, rel, own, problems)
    for rel, task, aliases, label, results_path in summaries:
        os.makedirs(os.path.dirname(os.path.join(new, rel)), exist_ok=True)
        with open(os.path.join(new, rel), "w") as f:
            json.dump(task_summary(task, aliases, label, results_path), f, indent=1, ensure_ascii=False)
    with open(os.path.join(new, "summary.json"), "w") as f:
        f.write(raw)
    with open(os.path.join(new, "summary.md"), "w") as f:
        f.write(summary_md(o) + "\n")
    for part in ("breakdowns", "slurm"):                     # swap the new parts in; runs/ stays where it is
        cur, old = os.path.join(view, part), os.path.join(view, f".old_{os.getpid()}_{part}")
        if os.path.lexists(cur):
            _check_view(cur)
            os.rename(cur, old)
        if os.path.isdir(os.path.join(new, part)):
            os.rename(os.path.join(new, part), cur)
        if os.path.lexists(old):
            _remove_view(old)
    for f in ("identity.json", "summary.json", "summary.md"):
        if os.path.lexists(os.path.join(new, f)):
            os.replace(os.path.join(new, f), os.path.join(view, f))
        elif os.path.islink(os.path.join(view, f)):
            os.unlink(os.path.join(view, f))
    os.rmdir(new)
    # the same model in another place: the flat layout of before the groups, or another group (without runs of its own)
    for other in [os.path.join(ROOT, name)] + [os.path.join(ROOT, c, name) for c in CATEGORIES if c != cat]:
        if is_view(other):
            if os.path.isdir(os.path.join(other, "runs")):
                log(f"  !! {os.path.relpath(other, ROOT)} holds runs of its own - left alone")
            else:
                _remove_view(other)
    return n, problems


def listed_runs():
    """names of our own run directories that a model of the report draws on (meta.dirs of results_xx/cosmos_eval/*.json).
    Only these are cleaned, finalized and moved: _runs/ is also written by other people's jobs (probe runs, ablations),
    and their directories are not ours to change (2026-11-14: the caches of 32 LOOPPROBE_* runs were deleted by the
    cache rule before this limit existed)."""
    out = set()
    for p in glob.glob(os.path.join(JSONS, "*.json")):
        try:
            with open(p) as f:
                out.update(d for d in json.load(f)["meta"]["dirs"] if ":" not in d)
        except (OSError, ValueError, KeyError):
            continue
    return out


def finished_tasks(run_dir):
    """tasks of a run that are finished: a results file has their score and the per-question record written with it is
    there (the samples file of every subtask, or the compact choices file of a multiple-choice task)."""
    name, done = os.path.basename(run_dir), set()
    for f in glob.glob(os.path.join(glob.escape(os.path.join(run_dir, name)), "results_*.json")):
        try:
            with open(f) as fh:
                r = json.load(fh)
        except (OSError, ValueError):
            continue
        if (r.get("config") or {}).get("limit") is not None:
            continue
        stamp = os.path.basename(f)[len("results_"):-len(".json")]
        d = os.path.dirname(f)
        record = all(os.path.exists(os.path.join(d, f"samples_{leaf}_{stamp}.jsonl")) for leaf in r.get("configs") or {})
        record = record or bool(glob.glob(os.path.join(glob.escape(d), f"choices_*_{glob.escape(stamp)}.json")))
        if record:
            done.update(r.get("results") or {})
    return done


def clean_caches(dry=False, log=print):
    """delete the response cache of every task whose score is out, task by task, whether or not the run's job is still
    working on its other tasks (each task is its own lm_eval process: a finished task's cache is closed). Returns the
    bytes freed."""
    freed = 0
    for run in sorted(listed_runs()):                        # only the runs behind the report's models
        d = os.path.realpath(os.path.join(RUNS, run))
        if not os.path.isdir(d):
            continue
        done = finished_tasks(d)
        files = [c for c in sorted(glob.glob(os.path.join(glob.escape(d), "cache_*.db")))
                 if re.sub(r"__rank\d+\.db$", "", os.path.basename(c)[len("cache_"):]) in done]
        if not files:
            continue
        size = sum(os.path.getsize(c) for c in files)
        freed += size
        log(f"  {run}: {'would remove' if dry else 'removed'} the caches of {len(files)} finished tasks, {size / 1e6:.1f} MB")
        if not dry:
            for c in files:
                os.unlink(c)
                for extra in (c + "-journal", c + "-wal", c + "-shm"):
                    if os.path.exists(extra):
                        os.unlink(extra)
            with open(os.path.join(RUNS, "DELETED.log"), "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {run}: response caches of {len(files)} finished tasks, {size / 1e6:.1f} MB "
                        f"(rule: a finished task keeps no cache; organize_cosmos_eval.clean_caches)\n")
    return freed


def write_index(cat):
    """<group>/index.md: the models of the group, one line each, from their summary.json."""
    d = os.path.join(ROOT, cat)
    os.makedirs(d, exist_ok=True)
    rows = []
    for name in sorted(os.listdir(d)):
        if name.startswith(".") or not is_view(os.path.join(d, name)):
            continue
        with open(os.path.join(d, name, "summary.json")) as f:
            o = json.load(f)
        done = o["n_metrics"] - len(o["missing"])
        status = "running" if o.get("running") else "stopped before the end" if o.get("stopped") else "complete" if o.get("complete") else "incomplete"
        rows.append(f"| [{name}]({name}/summary.md) | {o['display_name']} | {done} / {o['n_metrics']} | {status} | {', '.join(o['meta']['dirs'])} |")
    lines = [f"# {cat}", "", CATEGORIES[cat] + " Downstream evaluation, task list `cosmos_ultra_ladder`.", ""]
    lines += (["| directory | model | metrics | status | runs |", "|---|---|---:|---|---|"] + rows) if rows else ["No model yet."]
    with open(os.path.join(d, "index.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(rows)


def main(names=None, dry=False, log=print, do_sweep=True, do_clean=True, do_final=True, do_adopt=True):
    log("organize runs/cosmos_eval_downstream" + (" (dry run: nothing is changed)" if dry else ""))
    if do_sweep:
        sweep(dry, log)
    names = list(names) if names else sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(JSONS, "*.json")))
    if do_adopt:
        adopt(names, dry, log)
    for name in names:
        if not os.path.exists(os.path.join(JSONS, name + ".json")):
            log(f"  {name}: no results_xx/cosmos_eval/{name}.json - skipped")
            continue
        n, problems = build(name, dry, log)
        if not dry and n:
            log(f"  {name}: {n} links" + (f"; {len(problems)} problems" if problems else ""))
        for p in problems[:6]:
            log(f"      !! {p}")
    if not dry:
        log("  " + ", ".join(f"{cat}: {write_index(cat)}" for cat in CATEGORIES))
    if do_clean:
        freed = clean_caches(dry, log)
        if freed:
            log(f"  caches of finished tasks: {freed / 1e9:.2f} GB {'would be ' if dry else ''}freed")
    if do_final:
        try:
            import finalize_run
            finalize_run.submit(log, dry)
        except Exception as e:                               # the final records are an extra: they must not stop a collection
            log(f"  !! final records: not started ({e!r})")
    places = [(ROOT, "")] + [(os.path.join(ROOT, c), c + "/") for c in CATEGORIES if os.path.isdir(os.path.join(ROOT, c))]
    stale = [pre + n for d, pre in places for n in sorted(os.listdir(d))
             if not n.startswith(".") and is_view(os.path.join(d, n)) and not os.path.exists(os.path.join(JSONS, n + ".json"))]
    if stale:
        log(f"  per-model directories without a JSON in results_xx/cosmos_eval (left as they are): {stale}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="models to rebuild (default: every JSON in results_xx/cosmos_eval)")
    ap.add_argument("--dry-run", action="store_true", help="print what would be moved and linked, change nothing")
    ap.add_argument("--no-sweep", action="store_true", help="do not move run directories from the top level")
    ap.add_argument("--no-clean", action="store_true", help="do not delete the caches of finished tasks")
    ap.add_argument("--no-final", action="store_true", help="do not start the final records of new runs (finalize_run.py)")
    ap.add_argument("--no-adopt", action="store_true", help="do not move raw run directories from _runs/ into their model's folder")
    a = ap.parse_args()
    main(a.names, a.dry_run, do_sweep=not a.no_sweep, do_clean=not a.no_clean, do_final=not a.no_final, do_adopt=not a.no_adopt)
