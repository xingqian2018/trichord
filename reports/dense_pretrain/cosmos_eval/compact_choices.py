#!/usr/bin/env python3
"""reports_xx/cosmos_eval/compact_choices.py - one small JSON per multiple-choice task of a run: which question, which option
the model chose (user 2026-11-13: "这类选择题测试，咱们只是留一个 json，只是记录下，哪个题目，选了哪个").

    python3 compact_choices.py                 check every run in runs/cosmos_eval_downstream/_runs, write nothing
    python3 compact_choices.py --write         also write <run>/<run>/choices_<task>_<stamp>.json
    python3 compact_choices.py --write RUN ... only these runs
    python3 compact_choices.py --redundant     list the files a verified choices file makes redundant, with sizes
    python3 compact_choices.py --delete-redundant [RUN ...]
                                               delete them (user 2026-11-13 17:40, on the list of 97.92 GB: "可以"). Per task the
                                               record is recomputed from the samples once more and must equal the choices file
                                               on disk; only then its samples files and its cache go. Logged in _runs/DELETED.log.
                                               Refused while a cosmos_* eval job is in the queue.

A multiple-choice task is scored without any generated text: the same prompt is sent once per option, the server returns
the log-likelihood of each option, and the option with the highest one is the model's answer. The harness keeps, per
question, the prompt four times (samples_*.jsonl) and the server's per-token answer (cache_*.db). What the score needs is
far less, and that is what a choices file holds, per subtask, one entry per question in dataset order (doc_id):

    chosen        "BBCA..."    the option with the highest log-likelihood; A = first option, B = second ...
    gold          "BBCA..."    the correct option (absent for truthfulqa_mc2, which has several true answers)
    loglik        [[...], ...] the log-likelihood of every option, as returned (ties and letter preference can be re-read)
    correct_norm  "1011..."    tasks that report acc_norm: whether the length-normalised choice was right
    score         [...]        truthfulqa_mc2: the per-question mc2 score
    prompts_sha256             one hash over the prompt hashes of all questions: a rebuilt prompt set can be checked against it

Before a file is written, every per-question value the harness stored is recomputed from the
compact record and must be equal: acc from chosen == gold, acc_ties_shared from the ties in loglik, the subtask accuracy
of the results file from the mean. A task that fails a check gets no file.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/cosmos_eval_downstream"
RUNS = os.path.join(ROOT, "_runs")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def top_task(d):
    """the task a results file was written for: the group that is in no other group, or the single subtask."""
    groups = {k: v for k, v in (d.get("group_subtasks") or {}).items() if v}
    children = {c for v in groups.values() for c in v}
    tops = [g for g in groups if g not in children]
    if len(tops) == 1:
        return tops[0]
    leaves = list(d.get("configs") or {})
    return leaves[0] if len(leaves) == 1 else None


def mc_results(run_dir):
    """(task, stamp, results dict) of every results file of the run whose subtasks are all multiple choice."""
    run = os.path.basename(run_dir)
    out = {}
    for f in sorted(glob.glob(os.path.join(run_dir, run, "results_*.json"))):
        try:
            d = json.load(open(f))
        except (OSError, ValueError):
            continue
        cfgs = d.get("configs") or {}
        if not cfgs or (d.get("config") or {}).get("limit") is not None:
            continue
        if any(c.get("output_type") != "multiple_choice" for c in cfgs.values()):
            continue
        task = top_task(d)
        if task:
            out[task] = (os.path.basename(f)[len("results_"):-len(".json")], d)      # a later file of the same task wins
    return [(t, s, d) for t, (s, d) in sorted(out.items())]


def gold_index(row, n_options, chosen):
    t = row.get("target")
    if isinstance(t, int):
        return t
    if isinstance(t, str) and t.isdigit():
        return int(t)
    if isinstance(t, str) and len(t) == 1 and t in LETTERS:
        return LETTERS.index(t)
    if n_options == 2 and row.get("acc") in (0.0, 1.0):       # winogrande: the target is the text of the right option
        return chosen if row["acc"] == 1.0 else 1 - chosen
    return None


def compact(run_dir, task, stamp, d):
    """the choices record of one task, the samples files it was read from, and the problems found while checking it."""
    run = os.path.basename(run_dir)
    problems, sources, subtasks = [], [], {}
    for leaf in sorted(d["configs"]):
        f = os.path.join(run_dir, run, f"samples_{leaf}_{stamp}.jsonl")
        if not os.path.exists(f):
            problems.append(f"{leaf}: no samples file")
            continue
        sources.append(f)
        rows = sorted((json.loads(line) for line in open(f)), key=lambda r: r["doc_id"])
        if [r["doc_id"] for r in rows] != list(range(len(rows))):
            problems.append(f"{leaf}: doc ids are not 0..{len(rows) - 1}")
            continue
        mc2 = leaf.startswith("truthfulqa_mc2")
        chosen, gold, loglik, norm, score = [], [], [], [], []
        for r in rows:
            lls = [float(o[0][0]) for o in r["resps"]]
            c = max(range(len(lls)), key=lambda i: (lls[i], -i))                       # the first of equal maxima, as numpy.argmax
            g = None if mc2 else gold_index(r, len(lls), c)
            chosen.append(c)
            loglik.append(lls)
            if mc2:
                score.append(r["acc"])
            else:
                if g is None or not 0 <= g < len(lls):
                    problems.append(f"{leaf} doc {r['doc_id']}: correct option not readable from target {r.get('target')!r}")
                    break
                gold.append(g)
                if (c == g) != (r.get("acc") == 1.0):
                    problems.append(f"{leaf} doc {r['doc_id']}: chosen {c}, gold {g}, but the harness stored acc {r.get('acc')}")
                if "acc_ties_shared" in r:
                    tied = [i for i, v in enumerate(lls) if v == lls[c]]
                    credit = 1.0 / len(tied) if g in tied else 0.0
                    if abs(credit - r["acc_ties_shared"]) > 1e-9:
                        problems.append(f"{leaf} doc {r['doc_id']}: tie credit {credit} != stored {r['acc_ties_shared']}")
            if "acc_norm" in r:
                norm.append("1" if r["acc_norm"] == 1.0 else "0")
        if len(chosen) != len(rows) or (not mc2 and len(gold) != len(rows)):
            continue
        res = (d.get("results") or {}).get(leaf) or {}
        mean = (sum(score) / len(score)) if mc2 else sum(c == g for c, g in zip(chosen, gold)) / len(rows)
        if isinstance(res.get("acc,none"), (int, float)) and abs(mean - res["acc,none"]) > 1e-9:
            problems.append(f"{leaf}: accuracy from the compact record {mean} != results file {res['acc,none']}")
        if norm and isinstance(res.get("acc_norm,none"), (int, float)) and abs(norm.count("1") / len(norm) - res["acc_norm,none"]) > 1e-9:
            problems.append(f"{leaf}: acc_norm from the compact record != results file {res['acc_norm,none']}")
        sub = {"questions": len(rows), "chosen": "".join(LETTERS[c] for c in chosen)}
        if not mc2:
            sub["gold"] = "".join(LETTERS[g] for g in gold)
        if norm:
            sub["correct_norm"] = "".join(norm)
        if mc2:
            sub["score"] = score
        sub["loglik"] = loglik
        sub["prompts_sha256"] = hashlib.sha256("".join(r.get("prompt_hash", "") for r in rows).encode()).hexdigest()
        subtasks[leaf] = sub
    record = {"task": task, "run": run, "results_file": f"results_{stamp}.json", "harness": d.get("git_hash"),
              "questions": sum(s["questions"] for s in subtasks.values()),
              "how_to_read": "per subtask, one entry per question in dataset order (doc_id 0, 1, 2 ...). chosen = the option with the "
                             "highest log-likelihood (A = first option); gold = the correct option; loglik = the log-likelihood of every "
                             "option; correct_norm = 1 where the length-normalised choice was right; score = per-question mc2 score.",
              "subtasks": subtasks}
    return record, sources, problems


def dumps(record):
    """JSON with one line per field of a subtask, so that a file of 14,000 questions stays readable."""
    head = {k: v for k, v in record.items() if k != "subtasks"}
    out = ["{"] + [f" {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}," for k, v in head.items()] + [' "subtasks": {']
    names = list(record["subtasks"])
    for i, name in enumerate(names):
        sub = record["subtasks"][name]
        out.append(f"  {json.dumps(name)}: {{")
        keys = list(sub)
        for j, k in enumerate(keys):
            out.append(f"   {json.dumps(k)}: {json.dumps(sub[k], separators=(',', ':'))}" + ("," if j < len(keys) - 1 else ""))
        out.append("  }" + ("," if i < len(names) - 1 else ""))
    out += [" }", "}"]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directory names under _runs/ (default: all)")
    ap.add_argument("--write", action="store_true", help="write the choices files of the tasks that pass every check")
    ap.add_argument("--redundant", action="store_true", help="list, per run, the samples and cache files covered by a choices file on disk")
    ap.add_argument("--delete-redundant", action="store_true", help="delete those files after one more check against the choices file")
    a = ap.parse_args(argv)
    if a.delete_redundant:
        q = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%j"], capture_output=True, text=True, timeout=120)
        busy = sorted({n for n in q.stdout.split() if n.startswith("cosmos_")})
        if q.returncode or busy:
            sys.exit(f"not deleting: {'the queue cannot be read' if q.returncode else 'eval jobs in the queue: ' + ', '.join(busy)}")
    runs = a.runs or sorted(n for n in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, n)))
    total_q = total_bytes = n_ok = n_bad = red_bytes = 0
    for run in runs:
        run_dir = os.path.realpath(os.path.join(RUNS, run))
        tasks = mc_results(run_dir)
        if not tasks:
            continue
        line, red = [], []
        for task, stamp, d in tasks:
            out_path = os.path.join(run_dir, run, f"choices_{task}_{stamp}.json")
            if a.redundant:
                if os.path.exists(out_path):
                    files = [os.path.join(run_dir, run, f"samples_{leaf}_{stamp}.jsonl") for leaf in d["configs"]]
                    files += glob.glob(os.path.join(glob.escape(run_dir), f"cache_{glob.escape(task)}__rank*.db"))
                    red += [(f, os.path.getsize(f)) for f in files if os.path.exists(f)]
                continue
            record, sources, problems = compact(run_dir, task, stamp, d)
            if os.path.exists(out_path) and not sources:
                continue                                     # already compact: the samples are gone, the choices file is the record
            if a.delete_redundant:
                if problems or not os.path.exists(out_path) or json.load(open(out_path)) != record:
                    n_bad += 1
                    print(f"  !! {run} / {task}: not deleted ({problems[0] if problems else 'no choices file equal to the samples'})")
                    continue
                files = sources + glob.glob(os.path.join(glob.escape(run_dir), f"cache_{glob.escape(task)}__rank*.db"))
                size = sum(os.path.getsize(f) for f in files)
                for f in files:
                    os.unlink(f)
                n_ok += 1
                red.append((task, size))
                red_bytes += size
                total_q += len(files)
                continue
            if problems:
                n_bad += 1
                print(f"  !! {run} / {task}: {len(problems)} problems, no file. First: {problems[0]}")
                continue
            n_ok += 1
            text = dumps(record)
            assert json.loads(text) == record
            total_q += record["questions"]
            total_bytes += len(text)
            line.append(f"{task} {record['questions']}")
            if a.write:
                tmp = out_path + ".tmp"
                with open(tmp, "w") as f:
                    f.write(text)
                os.replace(tmp, out_path)
        if a.delete_redundant:
            if red:
                size = sum(s for _, s in red)
                print(f"{run}: deleted the samples and caches of {len(red)} tasks, {size / 1e9:.2f} GB")
                with open(os.path.join(RUNS, "DELETED.log"), "a") as f:
                    f.write(f"  {run:44s} {size / 1e9:6.2f} GB  samples_<subtask>_<stamp>.jsonl and cache_<task>__rank0.db of {len(red)} multiple-choice tasks\n")
        elif a.redundant:
            size = sum(s for _, s in red)
            red_bytes += size
            print(f"{run}: {len(red)} files, {size / 1e9:.2f} GB")
        else:
            print(f"{run}: {len(line)} tasks ok" + (f" ({', '.join(line)})" if len(runs) <= 3 else ""))
    if a.delete_redundant:
        print(f"deleted {total_q} files, {red_bytes / 1e9:.2f} GB, of {n_ok} tasks; {n_bad} tasks kept because a check failed")
    elif a.redundant:
        print(f"redundant in all: {red_bytes / 1e9:.2f} GB")
    else:
        print(f"{n_ok} tasks verified, {n_bad} with problems; {total_q} questions; choices files {total_bytes / 1e6:.1f} MB in all"
              + ("; written" if a.write else "; nothing written (use --write)") + f"  [{time.strftime('%H:%M')}]")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
