#!/usr/bin/env python3
"""reports_xx/cosmos_eval/fewshot_ids.py - which few-shot examples every question of a task was given, as ids
(user 2026-11-13: the final record stores example ids, not the prompt text).

Run it in the harness's client environment, with PYTHONPATH pointing at the harness tree OF THE RUN'S COMMIT (the prompts
are code: another commit can build other prompts):

    PYTHONPATH=<harness tree> python fewshot_ids.py --run <run dir> --task <task> [--stamp <results stamp>]
                                                    [--task-list <yml> | --num-fewshot N --override "k=v ..."] --out <json>

--task-list takes the task's num_fewshot and overrides from the run's task list, the way nvbench/launch_evals.py does.

finalize_run.py calls it for every finished task of a new run; by hand it answers the same question for any run.

The harness does not write down which examples it used; it only stores the finished prompt. So the prompts are built
again exactly as the run built them (same task code, same overrides, few-shot seed 1234, no model involved), the examples
the sampler renders are noted, and every rebuilt prompt is compared with the prompt hash the run stored for that
question. Only where the hashes agree are the ids right; the output says for how many questions they do.

An example id = the row number of the example in the task's few-shot pool, counted from 0, the same way a question's
doc_id is its row number in the test split. The pool is a split of the dataset (after the task's own preprocessing) or,
for tasks with hand-written examples, the list in the task file; the output names it ("pool").
Tasks that use no sampler (0-shot, or an example built into the prompt, as LiveCodeBench) get empty lists.

Output (JSON): {"task", "seed", "leaves": {<subtask>: {"questions", "num_fewshot", "pool": {...}, "verified",
"same_for_all": [ids] | null, "per_question": [[ids], ...] | null}}}. per_question is in doc_id order.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

DOC_ID_RX = re.compile(r'\{"doc_id": (\d+)')
HASH_KEY = '"prompt_hash": "'


def _id_and_hash(line):
    """(doc_id, prompt_hash) of one row of a samples file without parsing it (a LiveCodeBench row is megabytes): the id
    is the first field, the hash a top-level field near the end. Any doubt -> the row is parsed."""
    m = DOC_ID_RX.match(line)
    i = line.rfind(HASH_KEY)
    if m and i > 0:
        h = line[i + len(HASH_KEY):i + len(HASH_KEY) + 64]
        if len(h) == 64 and all(c in "0123456789abcdef" for c in h) and line[i + len(HASH_KEY) + 64:i + len(HASH_KEY) + 65] == '"':
            return int(m.group(1)), h
    r = json.loads(line)
    return r["doc_id"], r["prompt_hash"]


def _key(doc):
    return hashlib.sha1(json.dumps(doc, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


def stored_hashes(run, leaf, top, stamp=None):
    """prompt hash per doc_id from the run's per-question record of the subtask (of the results file `stamp`, or of all
    of them), or the one hash over all of them that a compact multiple-choice record keeps.
    Returns (dict or None, combined sha256 or None)."""
    name = os.path.basename(run.rstrip("/"))
    d = os.path.join(run, name)
    per = {}
    for f in sorted(glob.glob(os.path.join(glob.escape(d), f"samples_{leaf}_{glob.escape(stamp) if stamp else '2*'}.jsonl"))):
        with open(f) as fh:
            for line in fh:
                i, h = _id_and_hash(line)
                per[i] = h                                       # a later file of the same subtask wins
    if per:
        return per, None
    for f in sorted(glob.glob(os.path.join(glob.escape(d), f"choices_{top}_{glob.escape(stamp) if stamp else '2*'}.json")), reverse=True):
        with open(f) as fh:
            sub = (json.load(fh).get("subtasks") or {}).get(leaf)
        if sub and sub.get("prompts_sha256"):
            return None, sub["prompts_sha256"]
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory (_runs/<name>)")
    ap.add_argument("--task", required=True, help="task name as in the task list")
    ap.add_argument("--stamp", default=None, help="time stamp of the task's results file (default: the newest record of each subtask)")
    ap.add_argument("--task-list", default=None, help="the run's task list (yml): num_fewshot and overrides of the task are read from it")
    ap.add_argument("--num-fewshot", type=int, default=None, help="the task list's num_fewshot, when it sets one")
    ap.add_argument("--override", default="", help="the task list's overrides, as launch_evals.py passes them")
    ap.add_argument("--seed", type=int, default=1234, help="few-shot seed of the run (results file: fewshot_seed)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from lm_eval.evaluator import _parse_config_overrides
    from lm_eval.evaluator_utils import get_task_list
    from lm_eval.tasks import TaskManager, get_task_dict
    from lm_eval.utils import hash_string

    if a.task_list:                                             # as nvbench/launch_evals.py turns a task-list entry into lm_eval arguments
        import yaml

        with open(a.task_list) as f:
            entry = next((t for t in yaml.safe_load(f)["tasks"] if t.get("task_name") == a.task), None)
        if entry is not None:
            a.num_fewshot = entry.get("num_fewshot")
            over = dict(entry.get("overrides") or {})
            over.update({f"generation_kwargs.{k}": v for k, v in (entry.get("generation_kwargs") or {}).items()})
            a.override = " ".join(f"{k}={v}" for k, v in over.items())
    ov = _parse_config_overrides(a.override or None, None, a.num_fewshot)
    task_dict = get_task_dict([a.task], TaskManager(), config_overrides=ov)
    leaves = {}
    for out in get_task_list(task_dict):
        task, leaf = out.task, out.task_name
        task.set_fewshot_seed(a.seed)
        sampler = getattr(task, "sampler", None)
        pool = list(getattr(sampler, "docs", None) or []) if sampler is not None else []
        by_obj = {id(d): i for i, d in enumerate(pool)}             # the sampler hands out these very objects
        by_text = {}
        for i, d in enumerate(pool):
            by_text.setdefault(_key(d), i)

        def row(doc):
            i = by_obj.get(id(doc))
            return i if i is not None else by_text.get(_key(doc))    # a copy of a pool document: by content

        calls, rendered, sampled = [], [], []
        if sampler is not None:
            def doc_to_text(doc, *args, _o=sampler.doc_to_text, **kw):   # every example the sampler renders
                rendered.append(row(doc))
                return _o(doc, *args, **kw)

            def sample(n, _o=sampler.sample):                         # fallback for a sampler that renders by itself
                got = _o(n)
                sampled.append([row(d) for d in got])
                return got

            sampler.doc_to_text, sampler.sample = doc_to_text, sample

            def fewshot_context(doc, num_fewshot, *args, _o=task.fewshot_context, **kw):
                del rendered[:], sampled[:]
                text = _o(doc, num_fewshot, *args, **kw)
                ids = list(rendered)
                if not ids and sampled:
                    me = row(doc)
                    ids = [i for call in sampled for i in call if i != me][:num_fewshot]
                calls.append(ids)
                return text

            task.fewshot_context = fewshot_context
        task.build_all_requests(limit=None, rank=0, world_size=1, cache_requests=False, rewrite_requests_cache=False,
                                system_instruction=None, apply_chat_template=False, fewshot_as_multiturn=False,
                                chat_template=None, tokenizer_name="")
        first = {}
        for inst in task.instances:                                # in the order the prompts were built
            first.setdefault(inst.doc_id, inst)
        doc_ids = list(first)
        per_q = calls if sampler is not None and len(calls) == len(doc_ids) else [[] for _ in doc_ids]
        order = sorted(range(len(doc_ids)), key=lambda j: doc_ids[j])
        hashes = {doc_ids[j]: hash_string(first[doc_ids[j]].arguments[0]) for j in order}
        per, combined = stored_hashes(a.run, leaf, a.task, a.stamp)
        if per is not None:
            verified = sum(1 for i, h in hashes.items() if per.get(i) == h)
            how = "per question"
        elif combined is not None:
            ok = hashlib.sha256("".join(hashes[i] for i in sorted(hashes)).encode()).hexdigest() == combined
            verified, how = (len(hashes) if ok else 0), "one hash over all questions"
        else:
            verified, how = None, "no stored hash found"
        ids = [per_q[j] for j in order]
        cfg = task.config
        fixed = bool(cfg.fewshot_config and cfg.fewshot_config.get("samples") is not None)
        unknown = sum(1 for q in ids for i in q if i is None)
        # the split the harness takes the examples from (ConfigurableTask.fewshot_docs): fewshot_split, else training, validation, test
        split = None if fixed or not pool else cfg.fewshot_split or cfg.training_split or cfg.validation_split or cfg.test_split
        leaves[leaf] = {
            "questions": len(ids), "num_fewshot": cfg.num_fewshot,
            "pool": {"kind": "none" if not pool else "list in the task file" if fixed else "dataset split", "size": len(pool),
                     "split": split, "dataset_path": cfg.dataset_path, "dataset_name": cfg.dataset_name},
            "verified": verified, "verified_how": how, "ids_not_in_pool": unknown,
            "doc_ids_are_row_numbers": sorted(doc_ids) == list(range(len(doc_ids))),
            "same_for_all": ids[0] if ids and all(q == ids[0] for q in ids) else None,
            "per_question": None if ids and all(q == ids[0] for q in ids) else ids,
        }
    with open(a.out, "w") as f:
        json.dump({"task": a.task, "stamp": a.stamp, "seed": a.seed, "override": a.override, "num_fewshot": a.num_fewshot, "leaves": leaves},
                  f, separators=(",", ":"))
    n = sum(v["questions"] for v in leaves.values())
    v = sum(v["verified"] or 0 for v in leaves.values())
    print(f"{a.task}: {len(leaves)} subtasks, {n} questions, prompt hash verified for {v}; "
          f"ids not in the pool: {sum(v_['ids_not_in_pool'] for v_ in leaves.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
