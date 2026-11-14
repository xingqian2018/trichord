#!/usr/bin/env python3
"""reports_xx/cosmos_eval/compact_samples.py - per-question records of the generation tasks reduced to what a run produced:
which question (by id), what the model wrote, the verdict, the scores (user 2026-11-13: "就是一个ID的记录" - record ids,
do not copy the dataset; plan confirmed with "好的就按照这个计划执行").

    python3 compact_samples.py --prepare [--workers N] [RUN ...]   write <file>.compact next to every samples_*.jsonl and check it
    python3 compact_samples.py --list                              the prepared files per run, sizes before and after
    python3 compact_samples.py --apply                             replace each original by its checked .compact

The harness writes, for every question, the whole dataset record and the whole prompt into the samples file. Both follow
from the benchmark: the record by the question's id, the prompt from the record, the task version and the few-shot seed
(all in the results file). For LiveCodeBench the record carries the hidden test cases, megabytes per problem.

What a compact row keeps, per question:
    doc_id                 the question's index in the dataset
    doc                    only its short fields: ids and tags (question_id, task_id, platform, difficulty, category,
                           subject, level ...), i.e. strings up to 80 characters without a line break, numbers, booleans
    target                 the gold answer when it is short (up to 100 characters)
    arguments              the generation settings of the request (cap, temperature, stop strings ...); its prompt is
                           null. The harness's health check (cosmos_eval/gen_health.py) reads the cap here, so it gives
                           the same numbers on a compact file; export_generations.py skips a record without a prompt.
    resps                  everything the model wrote, unchanged
    filtered_resps         the extracted answer / the verdict per sample, unchanged, except that the code tasks' verdict
                           loses its "program" field (the tested program = prompt + answer + tests, a copy again)
    doc_hash prompt_hash target_hash   the harness's hashes of the full record, prompt and target: a rebuilt prompt or a
                           dataset record can be checked against them
    every metric column    exact_match, pass@k, judge_accuracy ...
What it drops: the long fields of doc (question text, options, solutions, test cases), the prompt, a long target.
The same row, with the ids of the few-shot examples added, is what finalize_run.py writes for new runs ("record": "final").

Checks, per file, before a .compact counts as prepared (written to <file>.compact.ok):
  - same number of rows, same doc_id order;
  - read back, every kept field equals the original's (hash over the kept content of all rows);
  - every metric that is a plain score in all rows has the same mean as the results file of the run. A metric that
    covers only part of the questions (the harness marks the others with NaN or -1) is left out of this check; its
    column is still compared field by field.
--apply replaces only files with a .compact.ok whose original is unchanged since, and refuses while a cosmos_* eval job
is in the queue. It also removes the .slim files of an earlier LiveCodeBench-only tool (deleted 2026-11-14).
"""
import argparse
import ast
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from multiprocessing import Pool

ROOT = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/cosmos_eval_downstream"
RUNS = os.path.join(ROOT, "_runs")
BASE = ("doc_id", "doc", "target", "arguments", "resps", "filtered_resps", "doc_hash", "prompt_hash", "target_hash")
MARK = "compact"                      # a row carries "record": "compact" so that a compact file is recognisable


def _short(v):
    return isinstance(v, (int, float, bool)) or (isinstance(v, str) and len(v) <= 80 and "\n" not in v)


def _verdict(x):
    """filtered_resps entry without the copy of the tested program (code tasks store str(dict) with a "program" key)."""
    if isinstance(x, list):
        return [_verdict(y) for y in x]
    if isinstance(x, str) and x.startswith("{") and "'program'" in x[:20]:
        try:
            d = ast.literal_eval(x)
        except (ValueError, SyntaxError, MemoryError):
            return x
        if isinstance(d, dict) and "program" in d:
            d.pop("program")
            return repr(d)
    return x


def settings(arguments):
    """the request without its prompt: the generation settings stay, the prompt becomes null."""
    out = {}
    for name, args in (arguments or {}).items():
        if isinstance(args, dict) and isinstance(args.get("arg_1"), dict):
            out[name] = {"arg_0": None, "arg_1": args["arg_1"]}
    return out


def compact_row(r, mark=MARK, extra=None):
    """`extra`: fields to add after the target (finalize_run.py: the example ids). doc_id comes first: gen_health.py
    finds the question of a line by that."""
    doc = r.get("doc")
    out = {"doc_id": r["doc_id"], "record": mark,
           "doc": {k: v for k, v in doc.items() if _short(v)} if isinstance(doc, dict) else {}}
    t = r.get("target")
    if isinstance(t, (int, float, bool)) or (isinstance(t, str) and len(t) <= 100):
        out["target"] = t
    out.update(extra or {})
    out["arguments"] = settings(r.get("arguments"))
    out["resps"] = r["resps"]
    out["filtered_resps"] = _verdict(r["filtered_resps"])
    for k in ("doc_hash", "prompt_hash", "target_hash"):
        if k in r:
            out[k] = r[k]
    for k, v in r.items():
        if k not in BASE and k not in out:
            out[k] = v
    return out


def _kept_hash(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True, ensure_ascii=False).encode()).digest()


def originals(runs=None):
    out = []
    for run in runs or sorted(os.listdir(RUNS)):
        d = os.path.realpath(os.path.join(RUNS, run))        # _runs/<run> is a link into the model's folder (or a run in progress)
        if os.path.isdir(d):
            out += sorted(glob.glob(os.path.join(glob.escape(os.path.join(d, run)), "samples_*.jsonl")))
    return out


def is_compact(path):
    with open(path) as f:
        head = f.read(64)
    return '"record": "compact"' in head or '"record": "final"' in head     # final = written by finalize_run.py


def prepare(path):
    out_path, tmp = path + ".compact", path + ".compact.tmp"
    m = re.match(r"samples_(.+)_(\d{4}-\d\d-\d\dT[\d.-]+)\.jsonl$", os.path.basename(path))
    leaf, stamp = m.group(1), m.group(2)
    rec = {"file": path, "leaf": leaf, "bytes_before": os.path.getsize(path), "mtime_before": os.path.getmtime(path)}
    try:
        n, ids, h1, prompts = 0, [], hashlib.sha256(), {}
        with open(path) as f, open(tmp, "w") as g:
            for line in f:
                c = compact_row(json.loads(line))
                h1.update(_kept_hash(c))
                ids.append(c["doc_id"])
                prompts.setdefault(c["doc_id"], c.get("prompt_hash", ""))
                g.write(json.dumps(c, ensure_ascii=False) + "\n")
                n += 1
        h2, ids2, sums, bad = hashlib.sha256(), [], {}, set()
        with open(tmp) as g:
            for line in g:
                c = json.loads(line)
                h2.update(_kept_hash(c))
                ids2.append(c["doc_id"])
                for k, v in c.items():
                    if k in BASE or k == "record":
                        continue
                    if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or v < 0:
                        bad.add(k)                           # not a plain per-question score (a tuple, "not in this subset" = NaN or -1)
                    else:
                        sums[k] = sums.get(k, 0.0) + v
        problems = []
        if len(ids2) != n or ids != ids2:
            problems.append("rows or their order differ after writing")
        if h1.digest() != h2.digest():
            problems.append("kept fields differ after writing")
        with open(os.path.join(os.path.dirname(path), f"results_{stamp}.json")) as f:
            res = (json.load(f).get("results") or {}).get(leaf) or {}
        checked = 0
        for k, s in sums.items():
            if k in bad:
                continue
            vals = [v for key, v in res.items() if key.startswith(k + ",") and "_stderr" not in key and isinstance(v, (int, float))]
            if not vals:
                continue
            checked += 1
            if abs(s / n - sum(vals) / len(vals)) > 1e-9:
                problems.append(f"{k}: mean {s / n} in the rows, {sum(vals) / len(vals)} in the results file")
        rec.update(rows=n, questions=len(prompts), metrics_checked=checked, bytes_after=os.path.getsize(tmp),
                   prompts_sha256=hashlib.sha256("".join(prompts[i] for i in sorted(prompts)).encode()).hexdigest())
        if problems:
            rec["problems"] = problems
            os.unlink(tmp)
            return rec
        os.replace(tmp, out_path)
        with open(out_path + ".ok", "w") as f:
            json.dump(rec, f)
        return rec
    except Exception as e:                                   # one broken file must not stop the others
        if os.path.exists(tmp):
            os.unlink(tmp)
        rec["problems"] = [repr(e)]
        return rec


def prepared():
    out = []
    for path in originals():
        ok = path + ".compact.ok"
        if os.path.exists(ok) and os.path.exists(path + ".compact"):
            with open(ok) as f:
                rec = json.load(f)
            if rec.get("bytes_before") == os.path.getsize(path) and rec.get("mtime_before") == os.path.getmtime(path):
                out.append(rec)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--list", action="store_true")
    g.add_argument("--apply", action="store_true")
    ap.add_argument("runs", nargs="*", help="run directory names under _runs/ (default: all)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if a.prepare:
        files = [f for f in originals(a.runs or None) if not is_compact(f)]
        files.sort(key=os.path.getsize, reverse=True)
        print(f"{len(files)} files to prepare, {sum(os.path.getsize(f) for f in files) / 1e9:.1f} GB  [{time.strftime('%H:%M:%S')}]", flush=True)
        bad = done = 0
        with Pool(min(a.workers, max(len(files), 1))) as pool:
            for rec in pool.imap_unordered(prepare, files):
                done += 1
                if rec.get("problems"):
                    bad += 1
                    print(f"  !! {'/'.join(rec['file'].split('/')[-3::2])}: {rec['problems'][:3]}", flush=True)
                elif rec["bytes_before"] > 1e9 or done % 100 == 0:
                    print(f"  ok {done}/{len(files)} {'/'.join(rec['file'].split('/')[-3::2])[:90]}: {rec['bytes_before'] / 1e6:.0f} MB -> "
                          f"{rec['bytes_after'] / 1e6:.1f} MB, {rec['metrics_checked']} metrics checked", flush=True)
        print(f"prepared {len(files) - bad}, problems {bad}  [{time.strftime('%H:%M:%S')}]")
        return 1 if bad else 0
    recs = prepared()
    by_run = {}
    for rec in recs:
        r = by_run.setdefault(rec["file"].split("/")[-3], [0, 0, 0, 0])
        r[0] += 1
        r[1] += rec["bytes_before"]
        r[2] += rec["bytes_after"]
        r[3] += rec["metrics_checked"]
    if a.list:
        for run, (n, b, c, m) in sorted(by_run.items()):
            print(f"{run:44s} {n:3d} files  {b / 1e9:7.2f} GB -> {c / 1e6:7.1f} MB   ({m} metric means checked)")
        waiting = [f for f in originals() if not is_compact(f)]
        print(f"{len(recs)} files prepared: {sum(r['bytes_before'] for r in recs) / 1e9:.2f} GB -> {sum(r['bytes_after'] for r in recs) / 1e9:.2f} GB; "
              f"samples files not compact yet: {len(waiting)}")
        return 0
    q = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%j"], capture_output=True, text=True, timeout=120)
    busy = sorted({n for n in q.stdout.split() if n.startswith("cosmos_")})
    if q.returncode or busy:
        sys.exit(f"not applying: {'the queue cannot be read' if q.returncode else 'eval jobs in the queue: ' + ', '.join(busy)}")
    freed = 0
    with open(os.path.join(RUNS, "DELETED.log"), "a") as log:
        log.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} per-question records of the generation tasks made compact (compact_samples.py): the "
                  f"dataset record and the prompt are no longer copied into every row; the model's answers, verdicts, hashes and metrics are unchanged:\n")
        for rec in recs:
            path = rec["file"]
            os.replace(path + ".compact", path)
            os.unlink(path + ".compact.ok")
            for old in (path + ".slim", path + ".slim.ok"):
                if os.path.exists(old):
                    os.unlink(old)
            freed += rec["bytes_before"] - rec["bytes_after"]
        for run, (n, b, c, m) in sorted(by_run.items()):
            log.write(f"  {run:44s} {n:3d} files  {b / 1e9:7.2f} GB -> {c / 1e6:7.1f} MB\n")
    print(f"replaced {len(recs)} files, {freed / 1e9:.2f} GB freed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
