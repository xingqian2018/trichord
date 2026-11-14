#!/usr/bin/env python3
"""reports_xx/stage_prod_ckpt.py - stage ONE checkpoint of a production-script run as a ladder checkpoint directory, so that
reports_xx/run_doc_eval.sh (sh/eval_ckpt.sh + tools/eval_argv.py, the 115-split doc eval, both windows from one model load) can
evaluate it like any ladder checkpoint (user 2026-11-13: "全部重新跑一个isodoc测试", steps 10,000 / 20,000 / 24,000 of the 59B run).

    python3 reports_xx/stage_prod_ckpt.py --view <run view> --iter N --change <Change ID>     # prints the staged directory

A production run has no ladder checkpoint directory: its argv / Megatron tree are in a "run view" (boxiangw's launch/eval59/make_view.py:
argv.txt, run.json, checkpoints -> the run's checkpoint dir). This writes, under OUR work dir only,
  <ladder_doc_eval>/stage/<Change ID>/ckpts/it<N>/
      argv.txt                           the view's training argv with the training-only flags removed (below)
      run.json                           the view's run.json (the Megatron tree eval_ckpt.sh uses) + what this stage was built from
      latest_checkpointed_iteration.txt  N  (eval_argv.py passes --load <this dir>, Megatron loads the "latest" = iteration N)
      iter_<N>                           symlink to the run's checkpoint of that iteration (read-only use)
      per_split.json                     {"train": null, "valid": [], "test": null}

The argv filter is the deny-list of boxiangw's launch/eval59/render_doc_argv.py (the script behind every existing doc eval of this run),
minus what tools/eval_argv.py sets itself: dropped are --pack-plan*, --wandb-*, --phase-transition-iterations, --exit-duration-in-mins,
--exit-interval, --ckpt-step; --context-parallel-size is set to 1; --full-validation --multiple-validation-sets are added (the production
entry validates differently; eval_argv.py requires both). --hyperp-* / --kd-* are left for eval_argv.py to drop. The per-split JSON has
NO training blend: an eval-only run needs no training dataset, and building the index of the production blend takes hours and thousands
of cache files (the cause of the 11-07 / 11-09 prepack eval failures). --data-cache-path points into the stage; run_doc_eval.sh replaces
it per job (EVAL_CACHE_PER_JOB=1). Nothing is written into the run view, the run directory or the ladder tree.
"""
import argparse
import json
import os
import sys

WORK_DIR = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/ladder_doc_eval"
DROP_PREFIX = ("--pack-plan", "--wandb-")
DROP_EXACT = {"--phase-transition-iterations", "--exit-duration-in-mins", "--exit-interval", "--ckpt-step",
              "--full-validation", "--multiple-validation-sets", "--validation-set-names", "--per-split-data-args-path", "--data-cache-path"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", required=True)
    ap.add_argument("--iter", type=int, required=True)
    ap.add_argument("--change", required=True, help="Change ID of the 8K result, e.g. ULTRA59B_MUONH_KD_QBK3")
    ap.add_argument("--work-dir", default=WORK_DIR)
    a = ap.parse_args()
    view = os.path.abspath(a.view)
    src = os.path.join(view, "checkpoints", f"iter_{a.iter:07d}")
    if not os.path.isfile(os.path.join(src, ".metadata")):
        sys.exit(f"no torch_dist checkpoint at {src}")
    dst = os.path.join(os.path.abspath(a.work_dir), "stage", a.change, "ckpts", f"it{a.iter}")
    os.makedirs(dst, exist_ok=True)
    toks = open(os.path.join(view, "argv.txt")).read().split()
    out, dropped, i = [], [], 0
    while i < len(toks):
        t, j = toks[i], i + 1
        while j < len(toks) and not toks[j].startswith("--"):
            j += 1
        if t in DROP_EXACT or t.startswith(DROP_PREFIX):
            dropped.append(" ".join(toks[i:j]))
        else:
            out.extend(toks[i:j])
        i = j
    for flag in ("--tensor-model-parallel-size", "--pipeline-model-parallel-size"):
        if flag in out and out[out.index(flag) + 1] != "1":
            sys.exit(f"{flag} {out[out.index(flag) + 1]}: only 1 is supported here")
    if "--context-parallel-size" in out:
        out[out.index("--context-parallel-size") + 1] = "1"
    per_split = os.path.join(dst, "per_split.json")
    with open(per_split, "w") as f:
        json.dump({"train": None, "valid": [], "test": None}, f, indent=1)
        f.write("\n")
    out += ["--full-validation", "--multiple-validation-sets", "--per-split-data-args-path", per_split,
            "--data-cache-path", os.path.join(dst, "data_cache_unused")]
    with open(os.path.join(dst, "argv.txt"), "w") as f:
        f.write("\n".join(out) + "\n")
    rj = json.load(open(os.path.join(view, "run.json")))
    rj["staged_from"] = {"view": view, "iteration": a.iter, "by": "reports_xx/stage_prod_ckpt.py", "argv_dropped": dropped}
    with open(os.path.join(dst, "run.json"), "w") as f:
        json.dump(rj, f, indent=1)
        f.write("\n")
    with open(os.path.join(dst, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write(str(a.iter))
    link = os.path.join(dst, f"iter_{a.iter:07d}")
    if os.path.islink(link):
        os.unlink(link)
    os.symlink(os.path.realpath(src), link)
    print(dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
