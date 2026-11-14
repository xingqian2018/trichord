#!/usr/bin/env python3
"""reports_xx/cosmos_eval/finalize_run.py - the final per-question record of the finished tasks of a NEW run: ids instead of
copies (user 2026-11-13: "过去的结果就不用更改了，把测试代码之后的最终存储（就是cache变folder那一步）改成存example ID就行").

    python3 finalize_run.py --list                 what waits; nothing is changed
    python3 finalize_run.py [--workers N] [RUN ..] write the final records (CPU node; minutes per run)
    python3 finalize_run.py --submit               start the line above as a CPU job when something waits and no such job
                                                   is queued. organize_cosmos_eval.py does this at the end of every pass.

The harness writes, per question, the whole dataset record, the whole prompt (question + few-shot examples) and the
answers. The final record keeps what the run produced and names the rest by id:

    which question    doc_id = its row number in the test split, plus its short tags (question_id, difficulty, subject ..)
    which examples    example_ids = row numbers in the task's example pool (a dataset split, or the list written in the
                      task definition); breakdowns/<task>/summary.json says which pool
    what came out     the model's answers, the extracted answer / verdict, every score, the generation settings
    checks            doc_hash, prompt_hash, target_hash as the harness wrote them

  a multiple-choice task  -> choices_<task>_<stamp>.json (compact_choices.py) with "examples" per subtask; its samples
                             files are deleted
  a generation task       -> samples_<subtask>_<stamp>.jsonl is replaced by rows of the form above ("record": "final")

A prompt is dropped only after it was proven to follow from the ids: with the harness tree of the run's commit the
prompts are built again (fewshot_ids.py, no model involved) and every rebuilt prompt must have the hash the run stored
for that question. A task where one question differs, or whose scores cannot be recomputed from the new record, stays
exactly as the harness wrote it and is reported.

Only results files stamped from FINAL_FROM on are touched: everything older is a past result and stays as it is.
Only runs that a model of the report lists (meta.dirs of results_xx/cosmos_eval/*.json) are touched: _runs/ also holds
directories of other people's jobs (probe runs, ablations), which are left exactly as they are.
A run is handled once no queued job writes into it: at its end an eval job reads the full records (cosmos_eval/
gen_health.py, export_generations.py). A later job that resumes the run still works on final records: gen_health.py
reads the answers and the generation settings, which are kept; export_generations.py skips a record without a prompt.
Each task finalized leaves final_<task>_<stamp>.json next to its results file: what was done, the example pool, sizes.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compact_choices  # noqa: E402
import compact_samples  # noqa: E402
import organize_cosmos_eval as org  # noqa: E402

FINAL_FROM = "2026-11-13T21-30-00"       # results stamped earlier are past results (user 2026-11-13: not changed)
RUNS = org.RUNS
USERS = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users"
HARNESS_REPO = os.environ.get("COSMOS_EVAL_HARNESS_REPO", f"{USERS}/xingqianx/Project/lm-evaluation-harness")
TREES = os.environ.get("COSMOS_EVAL_FINALIZE_DIR", f"{USERS}/xingqianx/tmp/cosmos_eval_finalize")    # harness trees per commit, example ids
CLIENT_BIN = os.environ.get("COSMOS_EVAL_CLIENT_BIN", f"{USERS}/boxiangw/envs/lmeval/bin")            # the harness's client environment
HF_HOME = os.environ.get("COSMOS_EVAL_HF_HOME", f"{USERS}/xingqianx/data/nemo6_hf_cache")
JOB_NAME = "evalfinal"
MARK = "final"                            # "record": "final" in every row of a final samples file


# ------------------------------------------------------------------------------------------------ what waits
def marker(run_dir, task, stamp):
    return os.path.join(run_dir, os.path.basename(run_dir), f"final_{task}_{stamp}.json")


def tasks_of(run_dir):
    """(task, stamp, results dict, kind) of every results file of the run stamped from FINAL_FROM on."""
    for f in sorted(glob.glob(os.path.join(glob.escape(os.path.join(run_dir, os.path.basename(run_dir))), "results_*.json"))):
        stamp = os.path.basename(f)[len("results_"):-len(".json")]
        if stamp < FINAL_FROM:
            continue
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        cfgs = d.get("configs") or {}
        if not cfgs or "results" not in d or (d.get("config") or {}).get("limit") is not None:
            continue
        kinds = {c.get("output_type") for c in cfgs.values()}
        kind = "choice" if kinds == {"multiple_choice"} else "generation" if kinds == {"generate_until"} else "other"
        yield compact_choices.top_task(d), stamp, d, kind


def waiting(runs=None, log=None):
    """tasks of new runs whose record is still the harness's: [{run, run_dir, task, stamp, kind, files, bytes}]."""
    queued = org.queued_result_dirs()
    out = []
    listed = org.listed_runs()                               # only the runs behind the report's models: _runs/ also holds other people's
    for run in sorted(set(runs) & listed) if runs else sorted(listed):
        d = os.path.realpath(os.path.join(RUNS, run))
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "identity.json")):
            continue
        new = list(tasks_of(d))
        if not new:
            continue
        if org.in_use(d, queued):
            if log:
                log(f"  {run}: a queued job writes into it (or the queue cannot be read): later")
            continue
        for task, stamp, res, kind in new:
            files = [os.path.join(d, os.path.basename(d), f"samples_{leaf}_{stamp}.jsonl") for leaf in sorted(res["configs"])]
            if task is not None and os.path.exists(marker(d, task, stamp)) and not (kind == "choice" and any(os.path.exists(f) for f in files)):
                continue                                     # final (a choice task is final once its samples files are gone)
            if task is None or kind == "other" or not all(os.path.exists(f) for f in files):
                if log:
                    log(f"  {run} / {task or 'results_' + stamp}: " + ("no single task in the results file" if task is None else
                                                                        "neither multiple choice nor generation" if kind == "other" else
                                                                        "no complete per-question record") + " - left as the harness wrote it")
                continue
            out.append({"run": os.path.basename(d), "run_dir": d, "task": task, "stamp": stamp, "kind": kind, "files": files,
                        "bytes": sum(os.path.getsize(f) for f in files)})
    return out


# ------------------------------------------------------------------------------------------------ example ids
_TREE_LOCK = threading.Lock()


def harness_tree(commit):
    """the harness as it was at `commit` (lm_eval + nvbench), unpacked once per commit. Returns (path, problem)."""
    with _TREE_LOCK:                                         # the example-id workers are threads: one unpacks, the others wait
        return _harness_tree(commit)


def _harness_tree(commit):
    tree = os.path.join(TREES, f"harness_{commit[:12]}")
    if os.path.exists(os.path.join(tree, ".complete")):
        return tree, None
    if subprocess.run(["git", "-C", HARNESS_REPO, "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True).returncode:
        return None, f"commit {commit[:12]} is not in {HARNESS_REPO}"
    tmp = f"{tree}.tmp{os.getpid()}"
    os.makedirs(tmp)
    ar = subprocess.Popen(["git", "-C", HARNESS_REPO, "archive", commit, "lm_eval", "nvbench"], stdout=subprocess.PIPE)
    rc = subprocess.run(["tar", "-x", "-C", tmp], stdin=ar.stdout).returncode
    ar.stdout.close()
    if ar.wait() or rc:
        return None, f"git archive of {commit[:12]} failed"
    if not os.path.exists(tree):
        os.rename(tmp, tree)
    with open(os.path.join(tree, ".complete"), "w") as f:
        f.write(commit + "\n")
    return tree, None


def example_ids(item):
    """runs fewshot_ids.py for one task in the client environment. Returns (ids dict or None, problem or None)."""
    try:
        with open(os.path.join(item["run_dir"], "identity.json")) as f:
            ident = json.load(f)
    except (OSError, ValueError) as e:
        return None, f"identity.json: {e!r}"
    commit = ident.get("harness_commit")
    if not commit:
        return None, "identity.json names no harness commit"
    tree, problem = harness_tree(commit)
    if problem:
        return None, problem
    out_dir = os.path.join(TREES, "ids", item["run"])
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{item['task']}_{item['stamp']}.json")
    with open(os.path.join(item["run_dir"], item["run"], f"results_{item['stamp']}.json")) as f:
        seed = (json.load(f).get("config") or {}).get("fewshot_seed")
    cmd = [os.path.join(CLIENT_BIN, "python"), os.path.join(HERE, "fewshot_ids.py"), "--run", item["run_dir"], "--task", item["task"],
           "--stamp", item["stamp"], "--seed", str(seed if isinstance(seed, int) else 1234), "--out", out]
    config = ident.get("config") or ""
    task_list = config if os.path.isabs(config) else os.path.join(tree, config)
    if config and os.path.exists(task_list):
        cmd += ["--task-list", task_list]
    env = dict(os.environ, PATH=CLIENT_BIN + os.pathsep + os.environ.get("PATH", ""), PYTHONPATH=tree, PYTHONNOUSERSITE="1",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", HF_HOME=HF_HOME, TOKENIZERS_PARALLELISM="false")
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=tree, env=env)
    if p.returncode or not os.path.exists(out):
        return None, "prompts could not be rebuilt: " + (p.stderr.strip().splitlines() or ["no output"])[-1][:300]
    with open(out) as f:
        ids = json.load(f)
    for leaf, v in ids["leaves"].items():
        if not v.get("doc_ids_are_row_numbers"):
            return None, f"{leaf}: the question ids are not 0..{v['questions'] - 1}"
        if v["verified"] != v["questions"] or v["ids_not_in_pool"]:
            return None, (f"{leaf}: the rebuilt prompt has the stored hash for {v['verified']} of {v['questions']} questions"
                          + (f", {v['ids_not_in_pool']} examples not found in the pool" if v["ids_not_in_pool"] else ""))
    return dict(ids, harness_commit=commit), None


def per_question(v):
    return [list(v["same_for_all"])] * v["questions"] if v["same_for_all"] is not None else v["per_question"]


def examples_block(v, commit, ids_in=None):
    """where the examples of a subtask come from and which ones were used. `ids_in`: a pointer instead of the ids of
    every question (the generation tasks keep them in their rows)."""
    pool = v["pool"]
    used = per_question(v)
    shots = sorted({len(q) for q in used})
    out = {"shots": shots[0] if len(shots) == 1 else shots}
    if pool["kind"] == "none" or shots == [0]:
        out["from"] = None
    elif pool["kind"] == "list in the task file":
        out.update({"from": f"the examples written in the task definition (harness {commit[:12]}), {pool['size']} of them",
                    "id": "position in that list, from 0"})
    else:
        out.update({"from": f"{pool['split']} split of {pool['dataset_path']}" + (f" ({pool['dataset_name']})" if pool["dataset_name"] else "")
                            + f", {pool['size']} rows after the task's preprocessing",
                    "id": "row number in that split, from 0"})
    if v["same_for_all"] is not None:
        out["same_for_all_questions"] = v["same_for_all"]
    else:
        out["per_question"] = ids_in or v["per_question"]
    return out


# ------------------------------------------------------------------------------------------------ the final records
def final_row(r, ids):
    """the compact row of compact_samples.py (short doc fields, short target, generation settings without the prompt,
    answers, verdicts, hashes, metrics) plus the example ids."""
    return compact_samples.compact_row(r, MARK, {"example_ids": ids})


def _digest(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True, ensure_ascii=False).encode()).digest()


def final_samples(job):
    """one samples file -> <file>.final.tmp, checked. job = (file, results file, subtask, ids file). Returns a record
    with "problems" when a check fails (then no .final.tmp is left)."""
    path, results_path, leaf, ids_path = job
    tmp = path + ".final.tmp"
    rec = {"file": path, "leaf": leaf, "bytes_before": os.path.getsize(path), "mtime_before": os.path.getmtime(path)}
    try:
        with open(ids_path) as f:
            ids = per_question(json.load(f)["leaves"][leaf])
        n, order, h1 = 0, [], hashlib.sha256()
        with open(path) as f, open(tmp, "w") as g:
            for line in f:
                r = json.loads(line)
                c = final_row(r, ids[r["doc_id"]])
                h1.update(_digest(c))
                order.append(c["doc_id"])
                g.write(json.dumps(c, ensure_ascii=False) + "\n")
                n += 1
        h2, order2, sums, partial = hashlib.sha256(), [], {}, set()
        with open(tmp) as g:
            for line in g:
                c = json.loads(line)
                h2.update(_digest(c))
                order2.append(c["doc_id"])
                for k, v in c.items():
                    if k in compact_samples.BASE or k in ("record", "example_ids"):
                        continue
                    if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or v < 0:
                        partial.add(k)                       # not a plain score of every question (a tuple; NaN or -1 = not in this subset)
                    else:
                        sums[k] = sums.get(k, 0.0) + v
        problems = []
        if order != order2:
            problems.append("rows or their order differ after writing")
        if h1.digest() != h2.digest():
            problems.append("fields differ after writing")
        if len(set(order)) != len(ids):
            problems.append(f"{len(set(order))} questions in the file, example ids for {len(ids)}")
        with open(results_path) as f:
            res = (json.load(f).get("results") or {}).get(leaf) or {}
        checked = 0
        for k, s in sums.items():
            vals = [v for key, v in res.items() if key.startswith(k + ",") and "_stderr" not in key and isinstance(v, (int, float))]
            if k in partial or not vals:
                continue
            checked += 1
            if abs(s / n - sum(vals) / len(vals)) > 1e-9:
                problems.append(f"{k}: mean {s / n} in the rows, {sum(vals) / len(vals)} in the results file")
        rec.update(rows=n, questions=len(set(order)), metrics_checked=checked, bytes_after=os.path.getsize(tmp))
        if problems:
            rec["problems"] = problems
            os.unlink(tmp)
        return rec
    except Exception as e:                                   # one broken file must not stop the others
        if os.path.exists(tmp):
            os.unlink(tmp)
        rec["problems"] = [repr(e)]
        return rec


def _write_marker(item, ids, before, after, subtasks):
    rec = {"task": item["task"], "run": item["run"], "results_file": f"results_{item['stamp']}.json", "harness": ids["harness_commit"],
           "finalized": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": "multiple choice" if item["kind"] == "choice" else "generation",
           "record": (f"choices_{item['task']}_{item['stamp']}.json" if item["kind"] == "choice" else f"samples_<subtask>_{item['stamp']}.jsonl")
                     + ": question ids, example ids, answers, scores. Prompts and dataset records are not stored.",
           "prompts_checked": "every prompt was built again from the ids with the harness at this commit and has the hash the run stored: "
                              f"{sum(v['verified'] for v in ids['leaves'].values())} of {sum(v['questions'] for v in ids['leaves'].values())} questions",
           "fewshot_seed": ids["seed"], "bytes_before": before, "bytes_after": after, "subtasks": subtasks}
    path = marker(item["run_dir"], item["task"], item["stamp"])
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    os.replace(path + ".tmp", path)


def finalize_choice(item, ids):
    """a multiple-choice task: the choices file with the example ids; then its samples files go. Returns (bytes before, after, problem)."""
    d, task, stamp = item["run_dir"], item["task"], item["stamp"]
    with open(os.path.join(d, item["run"], f"results_{stamp}.json")) as f:
        res = json.load(f)

    def record():
        rec, sources, problems = compact_choices.compact(d, task, stamp, res)
        if not problems and set(rec["subtasks"]) != set(ids["leaves"]):
            problems = ["subtasks of the record and of the example ids differ"]
        for leaf, sub in rec["subtasks"].items():
            v = ids["leaves"].get(leaf)
            if v is None or v["questions"] != sub["questions"]:
                problems = problems or [f"{leaf}: {sub['questions']} questions, example ids for {v and v['questions']}"]
                continue
            sub["examples"] = examples_block(v, ids["harness_commit"])
        rec["how_to_read"] += (" examples = the few-shot examples of the questions: where they come from and their ids, one list for all questions"
                               " or one list per question. The prompts are not stored: they follow from the question ids and example ids.")
        return rec, sources, problems

    rec, sources, problems = record()
    if problems:
        return 0, 0, problems[0]
    out = os.path.join(d, item["run"], f"choices_{task}_{stamp}.json")
    text = compact_choices.dumps(rec)
    if json.loads(text) != rec:
        return 0, 0, "the choices record does not survive writing"
    with open(out + ".tmp", "w") as f:
        f.write(text)
    os.replace(out + ".tmp", out)
    again, sources, problems = record()                      # from the samples once more, against the file on disk
    with open(out) as f:
        on_disk = json.load(f)
    if problems or on_disk != again:
        return 0, 0, "the choices file on disk does not equal the record recomputed from the samples: samples kept"
    before = sum(os.path.getsize(f) for f in sources)
    _write_marker(item, ids, before, os.path.getsize(out),   # first the note, then the samples go: a note with samples left is redone
                  {leaf: {"questions": sub["questions"], "examples": examples_block(ids["leaves"][leaf], ids["harness_commit"],
                                                                                 f"examples.per_question of the subtask in {os.path.basename(out)}")}
                   for leaf, sub in rec["subtasks"].items()})
    for f in sources:
        os.unlink(f)
    return before, os.path.getsize(out), None


def run(runs=None, workers=6, log=print):
    items = waiting(runs, log)
    if not items:
        log("no task of a new run waits for its final record")
        return 0
    log(f"{len(items)} tasks of {len({i['run'] for i in items})} runs, {sum(i['bytes'] for i in items) / 1e9:.2f} GB  [{time.strftime('%H:%M:%S')}]")
    os.makedirs(TREES, exist_ok=True)
    for commit in sorted({json.load(open(os.path.join(i["run_dir"], "identity.json"))).get("harness_commit") or "" for i in items}):
        if commit:
            harness_tree(commit)                             # unpack once, before the workers start
    with ThreadPoolExecutor(workers) as ex:
        got = list(ex.map(example_ids, items))
    ready, bad = [], 0
    for item, (ids, problem) in zip(items, got):
        if problem:
            bad += 1
            log(f"  !! {item['run']} / {item['task']}: {problem} - left as the harness wrote it")
        else:
            ready.append((item, ids))
    log(f"  example ids: {len(ready)} tasks with every prompt rebuilt and equal, {bad} not  [{time.strftime('%H:%M:%S')}]")
    done = {}                                                # run -> [tasks, bytes before, bytes after]
    jobs, by_task, task_of = [], {}, {}
    for item, ids in ready:
        if item["kind"] == "choice":
            before, after, problem = finalize_choice(item, ids)
            if problem:
                bad += 1
                log(f"  !! {item['run']} / {item['task']}: {problem} - left as the harness wrote it")
                continue
            r = done.setdefault(item["run"], [0, 0, 0])
            r[0], r[1], r[2] = r[0] + 1, r[1] + before, r[2] + after
            continue
        ids_path = os.path.join(TREES, "ids", item["run"], f"{item['task']}_{item['stamp']}.json")
        results_path = os.path.join(item["run_dir"], item["run"], f"results_{item['stamp']}.json")
        key = (item["run"], item["task"], item["stamp"])
        by_task[key] = (item, ids, [])
        for f in item["files"]:
            leaf = os.path.basename(f)[len("samples_"):-len(f"_{item['stamp']}.jsonl")]
            jobs.append((f, results_path, leaf, ids_path))
            task_of[f] = key
    jobs.sort(key=lambda j: os.path.getsize(j[0]), reverse=True)
    if jobs:
        with Pool(min(workers, len(jobs))) as pool:
            for rec in pool.imap_unordered(final_samples, jobs):
                by_task[task_of[rec["file"]]][2].append(rec)
    for (run_name, task, stamp), (item, ids, recs) in by_task.items():
        problems = [f"{r['leaf']}: {r['problems'][0]}" for r in recs if r.get("problems")]
        changed = [r for r in recs if not r.get("problems") and os.path.getsize(r["file"]) != r["bytes_before"]]
        if problems or changed or len(recs) != len(item["files"]):
            bad += 1
            for r in recs:
                if os.path.exists(r["file"] + ".final.tmp"):
                    os.unlink(r["file"] + ".final.tmp")
            log(f"  !! {run_name} / {task}: {(problems or ['a samples file changed meanwhile'])[0]} - left as the harness wrote it")
            continue
        subtasks = {r["leaf"]: {"questions": r["questions"], "rows": r["rows"], "metric_means_checked": r["metrics_checked"],
                                "examples": examples_block(ids["leaves"][r["leaf"]], ids["harness_commit"],
                                                           f"example_ids of each row of samples_{r['leaf']}_{stamp}.jsonl")}
                    for r in sorted(recs, key=lambda r: r["leaf"])}
        before, after = sum(r["bytes_before"] for r in recs), sum(r["bytes_after"] for r in recs)
        for r in recs:
            os.replace(r["file"] + ".final.tmp", r["file"])
        _write_marker(item, ids, before, after, subtasks)    # the note last: a task without it is done again, which changes nothing
        r = done.setdefault(run_name, [0, 0, 0])
        r[0], r[1], r[2] = r[0] + 1, r[1] + before, r[2] + after
    if done:
        with open(os.path.join(RUNS, "DELETED.log"), "a") as f:
            f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} final per-question records of new runs (finalize_run.py): prompts and dataset records "
                    f"replaced by question ids and example ids, after every prompt was rebuilt and compared with its stored hash:\n")
            for run_name, (n, b, a) in sorted(done.items()):
                f.write(f"  {run_name:44s} {n:3d} tasks  {b / 1e9:7.2f} GB -> {a / 1e6:7.1f} MB\n")
                log(f"  {run_name}: {n} tasks final, {b / 1e9:.2f} GB -> {a / 1e6:.1f} MB")
        names = []                                           # the per-model directories that show these runs
        for p in sorted(glob.glob(os.path.join(org.JSONS, "*.json"))):
            try:
                with open(p) as f:
                    if set(json.load(f)["meta"]["dirs"]) & set(done):
                        names.append(os.path.basename(p)[:-5])
            except (OSError, ValueError, KeyError):
                continue
        if names:
            org.main(names, log=log, do_sweep=False, do_final=False)
    log(f"final: {sum(v[0] for v in done.values())} tasks, left as written: {bad}  [{time.strftime('%H:%M:%S')}]")
    return 1 if bad else 0


def submit(log=print, dry=False):
    """start run() as a CPU job when something waits and no such job is queued. Returns the job id, or None."""
    items = waiting()
    if not items:
        return None
    q = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-n", JOB_NAME, "-o", "%i"], capture_output=True, text=True, timeout=120)
    if q.returncode or q.stdout.split():
        log(f"  final records: {len(items)} tasks wait; " + ("the queue cannot be read" if q.returncode else f"job {q.stdout.split()[0]} is on it"))
        return None
    log(f"  final records: {len(items)} tasks of {len({i['run'] for i in items})} new runs wait ({sum(i['bytes'] for i in items) / 1e9:.2f} GB)"
        + (" - a CPU job would be started" if dry else ""))
    if dry:
        return None
    logs = os.path.join(org.ROOT, "_slurm")
    os.makedirs(logs, exist_ok=True)
    # CPU partition, QOS cpu-dm: the batch CPU queue without a small per-user job limit; costs no fairshare
    s = subprocess.run(["sbatch", "--parsable", "-A", "cosmos_base_training", "-p", "cpu", "--qos=cpu-dm", "-c", "8", "--mem=64G", "-t", "04:00:00",
                        "-J", JOB_NAME, "-D", HERE, "-o", os.path.join(logs, "finalize_%j.log"),
                        "--wrap", f"python3 -u {os.path.join(HERE, 'finalize_run.py')} --workers 8"], capture_output=True, text=True, timeout=120)
    if s.returncode:
        log(f"  !! final records: the CPU job was not accepted: {s.stderr.strip()[:200]}")
        return None
    log(f"  final records: CPU job {s.stdout.strip()} started, log {logs}/finalize_{s.stdout.strip()}.log")
    return s.stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directory names under _runs/ (default: all)")
    ap.add_argument("--list", action="store_true", help="list what waits, change nothing")
    ap.add_argument("--submit", action="store_true", help="start the work as a CPU job when something waits")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    if a.list:
        items = waiting(a.runs or None, print)
        for i in items:
            print(f"{i['run']:44s} {i['task']:34s} {i['kind']:10s} {len(i['files']):3d} files {i['bytes'] / 1e6:10.1f} MB")
        print(f"{len(items)} tasks wait for their final record, {sum(i['bytes'] for i in items) / 1e9:.2f} GB")
        return 0
    if a.submit:
        submit()
        return 0
    return run(a.runs or None, a.workers)


if __name__ == "__main__":
    sys.exit(main())
