#!/usr/bin/env python3
"""reports_xx/cosmos_eval/strip_test_cases.py - LiveCodeBench per-question files without the copy of the hidden test cases
(user 2026-11-13: "我们先处理废物数据").

    python3 strip_test_cases.py --prepare [--workers N]   write <file>.slim next to every samples_lcb_codegen*.jsonl and check it
    python3 strip_test_cases.py --list                    the prepared files, with sizes before and after
    python3 strip_test_cases.py --apply                   replace each original by its checked .slim (the test cases are gone)

The harness writes, for every question, the whole dataset record into the samples file. For LiveCodeBench that record
holds the hidden test cases of the problem, megabytes of text for some problems: 4.3 GB per file, two files per run, 99 %
of it test cases. They are a copy of what the dataset has; the model's programs and their verdicts are a few megabytes.

A .slim file is the same file with `doc.private_test_cases` replaced by one line that names its length and sha256.
Everything else of every row is kept: the prompt, the model's programs, the verdict per program and per test, the metrics.

Checks, per file, before a .slim counts as prepared (written to <file>.slim.ok):
  - same number of rows;
  - every row is identical to the original row once the test cases are left out of both (compared by hash);
  - pass@k recomputed from the .slim rows equals the results file of the run.
--apply replaces only files with a .slim.ok and refuses while a cosmos_* eval job is in the queue.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from multiprocessing import Pool

ROOT = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/cosmos_eval_downstream"
RUNS = os.path.join(ROOT, "_runs")
FIELD = "private_test_cases"
MARK = "<hidden test cases removed from this copy:"


def originals():
    return sorted(f for f in glob.glob(os.path.join(RUNS, "*", "*", "samples_lcb_codegen*.jsonl")) if not os.path.islink(os.path.dirname(os.path.dirname(f))))


def _row_hash(r):
    doc = {k: v for k, v in r["doc"].items() if k != FIELD}
    return hashlib.sha256(json.dumps({**r, "doc": doc}, sort_keys=True, ensure_ascii=False).encode()).digest()


def prepare(path):
    """write <path>.slim and check it; returns a record (ok or the problem)."""
    slim, tmp = path + ".slim", path + ".slim.tmp"
    m = re.match(r"samples_(.+)_(\d{4}-\d\d-\d\dT[\d.-]+)\.jsonl$", os.path.basename(path))
    task, stamp = m.group(1), m.group(2)
    rec = {"file": path, "task": task, "bytes_before": os.path.getsize(path)}
    try:
        n = removed = already = 0
        h_before = hashlib.sha256()
        with open(path) as f, open(tmp, "w") as g:
            for line in f:
                r = json.loads(line)
                h_before.update(_row_hash(r))
                v = r["doc"].get(FIELD)
                if isinstance(v, str) and v.startswith(MARK):
                    already += 1
                elif v is not None:
                    text = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                    removed += len(text)
                    r["doc"][FIELD] = (f"{MARK} {len(text)} characters, sha256 {hashlib.sha256(text.encode()).hexdigest()}; "
                                       f"they are in the dataset under question_id {r['doc'].get('question_id')}>")
                g.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
        # read the slim file back: same rows apart from the test cases, and the same scores
        h_after, sums, m_rows = hashlib.sha256(), {}, 0
        with open(tmp) as g:
            for line in g:
                r = json.loads(line)
                h_after.update(_row_hash(r))
                m_rows += 1
                for k, v in r.items():
                    if re.fullmatch(r"pass@\d+\[\d+\]", k) and isinstance(v, (int, float)):
                        sums[k] = sums.get(k, 0.0) + v
        problems = []
        if m_rows != n:
            problems.append(f"{m_rows} rows read back, {n} written")
        if h_before.digest() != h_after.digest():
            problems.append("rows differ beyond the test cases")
        res_file = os.path.join(os.path.dirname(path), f"results_{stamp}.json")
        res = (json.load(open(res_file)).get("results") or {}).get(task) or {}
        checked = {}
        for k, s in sums.items():
            want = res.get(f"{k},score")
            checked[k] = s / n
            if not isinstance(want, (int, float)) or abs(s / n - want) > 1e-9:
                problems.append(f"{k}: {s / n} from the rows, {want} in {os.path.basename(res_file)}")
        if not sums:
            problems.append("no pass@k column found")
        rec.update(rows=n, characters_removed=removed, rows_already_slim=already, scores=checked, bytes_after=os.path.getsize(tmp))
        if problems:
            rec["problems"] = problems
            os.unlink(tmp)
            return rec
        os.replace(tmp, slim)
        with open(slim + ".ok", "w") as f:
            json.dump(rec, f, indent=1)
        return rec
    except Exception as e:                                   # one broken file must not stop the others
        if os.path.exists(tmp):
            os.unlink(tmp)
        rec["problems"] = [repr(e)]
        return rec


def prepared():
    out = []
    for path in originals():
        ok = path + ".slim.ok"
        if os.path.exists(ok) and os.path.exists(path + ".slim"):
            rec = json.load(open(ok))
            if rec.get("bytes_before") == os.path.getsize(path):           # the original is still the file that was checked
                out.append(rec)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--list", action="store_true")
    g.add_argument("--apply", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    if a.prepare:
        files = [f for f in originals() if os.path.getsize(f) > 50e6]      # a file already replaced is small: nothing to do
        print(f"{len(files)} files to prepare, {sum(os.path.getsize(f) for f in files) / 1e9:.1f} GB  [{time.strftime('%H:%M:%S')}]", flush=True)
        bad = 0
        with Pool(min(a.workers, max(len(files), 1))) as pool:
            for rec in pool.imap_unordered(prepare, files):
                name = "/".join(rec["file"].split("/")[-3::2])
                if rec.get("problems"):
                    bad += 1
                    print(f"  !! {name}: {rec['problems']}", flush=True)
                else:
                    print(f"  ok {name}: {rec['rows']} rows, {rec['bytes_before'] / 1e9:.2f} GB -> {rec['bytes_after'] / 1e6:.1f} MB, "
                          f"scores {{{', '.join(f'{k} {v:.4f}' for k, v in rec['scores'].items())}}}", flush=True)
        print(f"prepared {len(files) - bad}, problems {bad}  [{time.strftime('%H:%M:%S')}]")
        return 1 if bad else 0
    recs = prepared()
    if a.list:
        for rec in recs:
            print(f"{'/'.join(rec['file'].split('/')[-3::2]):75s} {rec['bytes_before'] / 1e9:6.2f} GB -> {rec['bytes_after'] / 1e6:7.1f} MB")
        print(f"{len(recs)} files: {sum(r['bytes_before'] for r in recs) / 1e9:.2f} GB -> {sum(r['bytes_after'] for r in recs) / 1e9:.2f} GB")
        return 0
    q = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%j"], capture_output=True, text=True, timeout=120)
    busy = sorted({n for n in q.stdout.split() if n.startswith("cosmos_")})
    if q.returncode or busy:
        sys.exit(f"not applying: {'the queue cannot be read' if q.returncode else 'eval jobs in the queue: ' + ', '.join(busy)}")
    freed = 0
    with open(os.path.join(RUNS, "DELETED.log"), "a") as log:
        log.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} hidden test cases removed from the LiveCodeBench samples files (doc.private_test_cases, a copy "
                  f"of the dataset; every other field of every row unchanged, pass@k re-checked against the results files):\n")
        for rec in recs:
            path = rec["file"]
            os.replace(path + ".slim", path)
            os.unlink(path + ".slim.ok")
            freed += rec["bytes_before"] - rec["bytes_after"]
            log.write(f"  {'/'.join(path.split('/')[-3::2]):75s} {rec['bytes_before'] / 1e9:6.2f} GB -> {rec['bytes_after'] / 1e6:7.1f} MB\n")
    print(f"replaced {len(recs)} files, {freed / 1e9:.2f} GB freed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
