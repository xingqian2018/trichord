#!/usr/bin/env python3
"""reports_xx/find_finished_runs.py - print `CHANGE/rung=run:<run name>` for every run of Boxiang's qwen3_tpp200 group that reached its
final iteration (latest checkpoint == --train-iters, iter dir present), promoted or not. Used by auto_doc_eval.sh; run_doc_eval.sh accepts
the lines as they are. Change ID mapping: v1 -> BASE, muon -> OPT_MUON, muonh -> OPT_MUONH, muonh_kd_a0p8_s64 -> KD_MUONH_A0P8_S64,
muonh_lr1p2x -> MUONH_LR1P2X; budget -iters<N>- -> _IT<N>, -tpp<N>- (N != 200) -> _TPP<N>; seed s<k> (k != 1) -> _S<k>.
"""
import glob
import os
import re

RUNS = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/boxiangw/runs/scaling_ladder/qwen3_tpp200"
NAME = re.compile(r"^qwen3_tpp200_(.+?)-(q\d)-(tpp\d+|iters\d+)-s(\d+)$")


def change_id(exp, budget, seed):
    base = {"v1": "BASE", "muon": "OPT_MUON", "muonh": "OPT_MUONH"}.get(exp)
    if base is None:
        if exp.startswith("muonh_kd_"):
            base = "KD_MUONH_" + exp[len("muonh_kd_"):].upper()
        elif exp.startswith("muonh_"):
            base = "MUONH_" + exp[len("muonh_"):].upper()
        else:
            return None
    if budget.startswith("iters"):
        base += "_IT" + budget[5:]
    elif budget != "tpp200":
        base += "_" + budget.upper()
    if seed != "1":
        base += "_S" + seed
    return base


for d in sorted(glob.glob(os.path.join(RUNS, "qwen3_tpp200_*"))):
    m = NAME.match(os.path.basename(d))
    if not m:
        continue
    try:
        argv = open(os.path.join(d, "argv.txt")).read().split()
        train_iters = int(argv[argv.index("--train-iters") + 1])
        it = int(open(os.path.join(d, "checkpoints", "latest_checkpointed_iteration.txt")).read())
    except (OSError, ValueError):
        continue
    if it != train_iters or not os.path.isdir(os.path.join(d, "checkpoints", f"iter_{it:07d}")):
        continue
    cid = change_id(m.group(1), m.group(3), m.group(4))
    if cid:
        print(f"{cid}/{m.group(2)}=run:{os.path.basename(d)}")
