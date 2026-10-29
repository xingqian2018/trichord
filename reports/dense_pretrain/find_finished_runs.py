#!/usr/bin/env python3
"""reports_xx/find_finished_runs.py - print `CHANGE/rung=run:<run name>` for every run of Boxiang's qwen3_tpp200 group that reached its
final iteration (latest checkpoint == --train-iters, iter dir present), promoted or not. Used by auto_doc_eval.sh; run_doc_eval.sh accepts
the lines as they are. Change ID mapping: v1 -> BASE, muon -> OPT_MUON, muonh -> OPT_MUONH, muonh_kd_a0p8_s64 -> KD_MUONH_A0P8_S64,
muonh_lr1p2x -> MUONH_LR1P2X, v1_dsv3_onetree -> BASE_DSV3_ONETREE; budget -iters<N>- -> _IT<N>, -tpp<N>- (N != 200) -> _TPP<N>; seed s<k> (k != 1) -> _S<k>.
"""
import glob
import os
import re

RUNS = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/boxiangw/runs/scaling_ladder/qwen3_tpp200"
PROMOTED = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos_llm_scaling_ladder/results"
NAME = re.compile(r"^qwen3_tpp200_(.+?)-(q\d)-(tpp\d+|iters\d+)-s(\d+)$")


def change_id(exp, budget, seed):
    base = {"v1": "BASE", "muon": "OPT_MUON", "muonh": "OPT_MUONH"}.get(exp)
    lr = re.match(r"^(muon_)?(lr\d+p?\d*x)$", exp)          # AdamW / Muon LR checks: lr0p5x -> BASE_LR0P5X, muon_lr2x -> OPT_MUON_LR2X
    if base is None and lr:
        base = ("OPT_MUON_" if lr.group(1) else "BASE_") + lr.group(2).upper()
    if base is None:
        if exp.startswith("muonh_kd_"):
            base = "KD_MUONH_" + exp[len("muonh_kd_"):].upper()
        elif exp.startswith("muonh_"):
            base = "MUONH_" + exp[len("muonh_"):].upper()
        elif exp.startswith("v1_"):                       # AdamW BASE variants: v1_dsv3_onetree -> BASE_DSV3_ONETREE (the router ablation C)
            base = "BASE_" + exp[len("v1_"):].upper()
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
    real = os.path.realpath(os.path.join(d, "checkpoints", f"iter_{it:07d}"))
    pm = re.match(r"^" + re.escape(PROMOTED) + r"/([^/]+)/ckpts/(q\d)/", real + "/")
    cid = pm.group(1) if pm else change_id(m.group(1), m.group(3), m.group(4))   # promoted: Boxiang's own Change ID (e.g. muonh_swa -> SWA)
    if cid:
        print(f"{cid}/{m.group(2)}=run:{os.path.basename(d)}")
