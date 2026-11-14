#!/usr/bin/env python3
"""reports_xx/organize_doc_isolated.py - builds runs/cosmos_eval_doc_isolate/, the home of the doc-isolated (iso-doc) results
(user 2026-11-13: "我们有一个新的文件夹，专门收录iso_doc的测试结果 ~/runs/cosmos_eval_doc_isolated/", "一个baseline 一个ours",
"ours -> ours_ladder / ours_prod"; to session Downstream测试: "新建立一个 cosmos_eval_doc_isolate", which created the directory with an
empty _runs/). Same idea as runs/cosmos_eval_downstream: one directory per model, open it and read summary.md.

    python3 reports_xx/organize_doc_isolated.py [--out DIR] [--only baseline|ours_ladder|ours_prod]

    baseline/<MODEL>/                       external reference models (result files <MODEL>[_W32K]_ref.json)
    ours_ladder/<Change ID>_<rung>/         scaling-ladder checkpoints (<Change ID>[_W32K]_<rung>.json)
    ours_prod/<RUN>_iter<NNNNNNN>/          checkpoints of a production run (<RUN>[_W32K]_it<N>.json, and the pooled-set evals)
        summary.md        the numbers of the model: every metric, loss and bpb, one column pair per eval window
        summary.json      the same numbers
        identity.json     what was evaluated (checkpoint, iteration, training arguments) and by which eval job per window
        breakdowns/<window>.json   the complete result of that window: the metrics + every suite and split (115 splits)
        runs/<name>       links to the raw eval directories (arguments, logs, per-split raw file)
    <category>/index.md   one line per model: overall_tsqw and val_v2 per window

A model's 8K and 32K results are two result files (the 32K one has _W32K in its Change ID); here they are one directory. The window
of a result is the --seq-length its eval ran with (the result file's train.seq_length), so a model evaluated at 16K shows "16k".
Result files are COPIED from the source of truth (trichord/dense_pretrain_result, written by collect_doc_eval.py; pooled production
evals from results_xx/prod_isodoc, written by collect_prod_isodoc.py). Raw eval directories are only linked, never moved.
Rebuilding is safe: a model directory is regenerated from the result files; nothing else under --out is touched.
"""
import argparse
import glob
import json
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
U = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx"
SRC = U + "/Project/trichord/dense_pretrain_result"
POOLED = os.path.join(HERE, "..", "results_xx", "prod_isodoc")
RAW = U + "/runs/ladder_doc_eval"
OUT = U + "/runs/cosmos_eval_doc_isolate"   # runs/cosmos_eval_doc_isolated is a link to it (the user used both spellings)
NAME_RE = re.compile(r"^(?P<change>.+)_(?P<rung>q\d|ref|it\d+)$")
PART_RE = re.compile(r"^(?P<base>.+)__P(?P<i>\d)of(?P<n>\d)-(?P<rung>q\d|ref|it\d+)-(?P<job>\d+)$")
CATS = ["baseline", "ours_ladder", "ours_prod"]
ROWS = [("00_overall_tsqw", "overall_tsqw"), ("00_overall_tw", "overall_tw"), ("00_overall_macro", "overall_macro"),
        ("01_val_v2_overall_tw", "val_v2"), ("01_val_v2_cc", "val_v2 / cc"), ("01_val_v2_python", "val_v2 / python"), ("01_val_v2_arxiv", "val_v2 / arxiv"),
        ("02_uncheatable_tw", "uncheatable"), ("02_uncheatable_macro", "uncheatable (macro)"), ("03_codefresh_tw", "codefresh"),
        ("04_v3small_tw", "v3small"), ("04_v3small_macro", "v3small (macro)"), ("04_private_internal_tw", "private_internal"),
        ("05_pile_test_tw", "pile_test"), ("05_pile_test_macro", "pile_test (macro)"), ("05_pile_10k_tw", "pile_10k")]
INDEX = [("00_overall_tsqw", "overall_tsqw"), ("01_val_v2_overall_tw", "val_v2")]
WHAT = {"full": "full suite: 9 suites, 115 splits", "pooled": "pooled sets: one loss per collection, no split detail (boxiangw's doc_eval_59b.sbatch)"}


def wlabel(seq):
    return f"{seq // 1024}k"


def wkey(label):
    return int(label[:-1])


def models():
    """{(category, directory name): {"change", "rung", "windows": {label: {kind, file, d}}}} from the result files."""
    out = {}

    def add(cat, name, change, rung, label, kind, path, d):
        """One result per model and window: a full result beats a pooled one; between two of a kind the newer eval wins."""
        rank = (1 if kind == "full" else 0, d["evaluated_at"] if kind == "full" else d["source"]["evaluated_at"])
        m = out.setdefault((cat, name), {"change": change, "rung": rung, "windows": {}})
        old = m["windows"].get(label)
        if old and old["kind"] == kind:
            print(f"NOTE {cat}/{name} {label}: two results ({os.path.basename(old['file'])}, {os.path.basename(path)}); the newer eval is used")
        if old and old["rank"] >= rank:
            return
        m["windows"][label] = {"kind": kind, "file": path, "d": d, "rank": rank}

    for p in sorted(glob.glob(os.path.join(SRC, "*.json"))):
        nm = NAME_RE.match(os.path.basename(p)[:-5])
        if not nm:
            continue
        d = json.load(open(p))
        change, rung = d["change"], d["rung"]
        seq = (d.get("train") or {}).get("seq_length") or (32768 if "_W32K" in change else 8192)
        base = change.replace("_W32K", "", 1)
        if rung == "ref":
            cat, name = "baseline", base
        elif rung.startswith("it"):
            cat, name = "ours_prod", f"{base}_iter{int(rung[2:]):07d}"
        else:
            cat, name = "ours_ladder", f"{base}_{rung}"
        add(cat, name, base, rung, wlabel(seq), "full", p, d)
    for p in sorted(glob.glob(os.path.join(POOLED, "*_iter*.json"))):
        d = json.load(open(p))
        add("ours_prod", f"{d['run']}_iter{d['iteration']:07d}", d["run"], f"it{d['iteration']}", wlabel(d["window"]), "pooled", p, d)
    return out


def raw_dirs(w):
    """The raw eval directories behind one window's result: the recorded one, plus the other parts of a 3-part eval (newest job each)."""
    d = w["d"]
    first = d["eval_dir"] if w["kind"] == "full" else d["source"]["eval_dir"]
    dirs = [first]
    pm = PART_RE.match(os.path.basename(first))
    if pm:
        for i in range(1, int(pm["n"]) + 1):
            if i == int(pm["i"]):
                continue
            cand = [c for c in glob.glob(os.path.join(os.path.dirname(first), f"{pm['base']}__P{i}of{pm['n']}-{pm['rung']}-*"))
                    if glob.glob(os.path.join(c, "eval", "val_splits_iter*_doc_isolated.json"))]
            if cand:
                dirs.append(max(cand, key=lambda c: int(c.rsplit("-", 1)[1])))
    return [x for x in dirs if os.path.isdir(x)]


def metrics_of(w):
    return {k: {s: v[s] for s in ("loss", "ppl", "bpb") if s in v} for k, v in w["d"]["metrics"].items()}


def build(out_dir, cat, name, m):
    dst = os.path.join(out_dir, cat, name)
    for sub in ("breakdowns", "runs"):
        shutil.rmtree(os.path.join(dst, sub), ignore_errors=True)
    os.makedirs(os.path.join(dst, "breakdowns"), exist_ok=True)
    os.makedirs(os.path.join(dst, "runs"), exist_ok=True)
    labels = sorted(m["windows"], key=wkey)
    any_d = m["windows"][labels[0]]["d"]
    ident = {"model": name, "category": cat, "change_id": m["change"], "rung": None if cat == "baseline" else m["rung"],
             "iteration": any_d.get("iteration"), "checkpoint": any_d.get("checkpoint"), "train": any_d.get("train"),
             "protocol": "doc_isolated: every validation document is its own padded sample, no context from other documents; "
                         "long documents are cut into consecutive chunks of the window", "evals": {}}
    summ = {"model": name, "category": cat, "windows": {}}
    for lab in labels:
        w = m["windows"][lab]
        d = w["d"]
        shutil.copy2(w["file"], os.path.join(dst, "breakdowns", lab + ".json"))
        src = d if w["kind"] == "full" else d["source"]
        ident["evals"][lab] = {"what": WHAT[w["kind"]], "result_id": os.path.basename(w["file"])[:-5], "eval_job": src["eval_job"],
                               "evaluated_at": src["evaluated_at"], "eval_dir": src["eval_dir"], "result_file_copied_from": os.path.realpath(w["file"])}
        if w["kind"] == "pooled":
            ident["evals"][lab].update({k: src.get(k) for k in ("by", "script", "megatron_commit", "ladder_dir")})
        summ["windows"][lab] = {"what": WHAT[w["kind"]], "eval_job": src["eval_job"], "evaluated_at": src["evaluated_at"], "metrics": metrics_of(w)}
        for rd in raw_dirs(w):
            link = os.path.join(dst, "runs", os.path.basename(rd))
            if not os.path.lexists(link):
                os.symlink(rd, link)
    for fn, obj in (("identity.json", ident), ("summary.json", summ)):
        with open(os.path.join(dst, fn), "w") as f:
            json.dump(obj, f, indent=1)
            f.write("\n")
    f4 = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "-"
    L = [f"# {name}", "", f"Doc-isolated val loss ({cat}). Lower is better; loss in nats per token, bpb = bits per byte; \"-\" = not evaluated.", ""]
    L += [f"- checkpoint: `{ident['checkpoint']}`" + (f", iteration {ident['iteration']:,}" if isinstance(ident["iteration"], int) and ident["iteration"] else "")]
    for lab in labels:
        e = ident["evals"][lab]
        L.append(f"- {lab} window: {e['what']}; job {e['eval_job']}, {e['evaluated_at'].replace('T', ' ')[:16]}; complete result: `breakdowns/{lab}.json`")
    L += ["", "| metric | " + " | ".join(f"{lab} loss | {lab} bpb" for lab in labels) + " |", "|---|" + "---:|---:|" * len(labels)]
    for k, n in ROWS:
        vals = [summ["windows"][lab]["metrics"].get(k, {}) for lab in labels]
        if any(vals):
            L.append(f"| {n} | " + " | ".join(f"{f4(v.get('loss'))} | {f4(v.get('bpb'))}" for v in vals) + " |")
    with open(os.path.join(dst, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    return summ


README = """# runs/cosmos_eval_doc_isolate

Doc-isolated ("iso-doc") validation loss: every validation document is scored as its own sample, with no context from other
documents. 9 decontaminated suites, 115 splits, 7 collections; windows 8K (comparable with everything) and 32K. Layout since 2026-11-13.

```
baseline/<model>/                  external reference models
ours_ladder/<Change ID>_<rung>/    scaling-ladder checkpoints
ours_prod/<run>_iter<N>/           checkpoints of a production run
    summary.md                     every metric of the model, loss and bpb, one column pair per window. Open this.
    summary.json                   the same numbers
    identity.json                  what was evaluated and by which eval job
    breakdowns/<window>.json       the complete result of one window: metrics + every suite and split
    runs/<name>                    links to the raw eval directories (arguments, logs)
<category>/index.md                one line per model: overall_tsqw and val_v2 per window
_runs/<run name>/                  for the raw output of new eval jobs. Empty so far: jobs still write to runs/ladder_doc_eval/
                                   (whether they move here is not decided)
```

- `runs/cosmos_eval_doc_isolated` is a link to this directory. The downstream evaluation is next to it: `runs/cosmos_eval_downstream`.
- The headline metric is `overall_tsqw`: the 7 collections' token-weighted losses, weighted by the square root of their token counts.
- A model directory is rebuilt from the result files; deleting it loses nothing. The result files are copies of
  `Project/trichord/dense_pretrain_result/*.json` (and `results_xx/prod_isodoc/*.json` for pooled production evals).
- The raw eval directories were not moved: they are still under `runs/ladder_doc_eval/` (production pooled evals: `runs/cosmos_eval_downstream/_doc_eval59/`
  and boxiangw's `runs/eval59/doc_eval/`).
- `ours_prod` checkpoints may have only a "pooled" result: two collections (val_v2, v3small), one loss each, 8K only.
- Nine full-suite jobs of ULTRA59B_MUONH_KD_QBK3 (iterations 10000 / 20000 / 24000, 8K + 32K) were cancelled on 2026-11-13 14:14,
  minutes after their start, when the result directories were reorganised; they have not been resubmitted. Their records:
  `runs/ladder_doc_eval/sbatch/stopped_2026-11-13/`.

Built by `cosmos_llm_scaling_ladder/reports_xx/organize_doc_isolated.py`. Refresh after new results:
`python3 collect_doc_eval.py && python3 collect_prod_isodoc.py && python3 organize_doc_isolated.py` (in `reports_xx/`).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--only", choices=CATS, default=None)
    a = ap.parse_args()
    ms = models()
    f4 = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "-"
    for cat in CATS:
        if a.only and cat != a.only:
            continue
        os.makedirs(os.path.join(a.out, cat), exist_ok=True)
        rows, labels = [], set()
        for (c, name), m in sorted(ms.items()):
            if c != cat:
                continue
            s = build(a.out, cat, name, m)
            rows.append((name, s))
            labels |= set(s["windows"])
        labels = sorted(labels, key=wkey)
        L = [f"# {cat}: {len(rows)} models", "", "Loss in nats per token (lower is better); \"-\" = not evaluated at that window. Details: `<model>/summary.md`.", "",
             "| model | " + " | ".join(f"{n} {lab}" for lab in labels for _, n in INDEX) + " |", "|---|" + "---:|" * (len(labels) * len(INDEX))]
        for name, s in rows:
            L.append(f"| {name} | " + " | ".join(f4(s["windows"].get(lab, {}).get("metrics", {}).get(k, {}).get("loss")) for lab in labels for k, _ in INDEX) + " |")
        with open(os.path.join(a.out, cat, "index.md"), "w") as f:
            f.write("\n".join(L) + "\n")
        print(f"{cat}: {len(rows)} model directories, windows {labels}")
    with open(os.path.join(a.out, "README.md"), "w") as f:
        f.write(README)
    print("->", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
