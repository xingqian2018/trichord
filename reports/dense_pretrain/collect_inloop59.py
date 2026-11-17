#!/usr/bin/env python3
"""reports_xx/collect_inloop59.py - the in-loop validation loss of the 59B production run, for the report's "59B val loss" tab
(Runlong 11-16 19:4x "网页你也要自动更新结果": the in-loop readings were reported in chat but never on the page).

Source: Megatron's "validation loss at iteration N | lm loss value: X" lines in boxiangw's training logs (read only):
  cosmos_ultra_pre/runs/ultra59b_muonh_kd_qbk3/logs/phase{1,2}/*.log (env.log skipped). A resumed job validates some iterations again:
  the reading of the later log (by modification time) wins. Each reading is ONE random draw of Boxiang's packed validation set
  (sigma about 0.006 per reading at this batch size, see the report text), not the doc-isolated eval of the other charts.
Writes results_xx/prod_isodoc/inloop_val.json: {"run", "total_steps", "phase2_start", "points": [[iteration, loss, log name], ...],
"collected_at"}. Run by ~/tmp/nemo6_ladder/tools/update_report.sh before gen_eval_html.py.
"""
import glob
import json
import os
import re
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RD = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_data/users/boxiangw/cosmos_ultra_pre/runs/ultra59b_muonh_kd_qbk3/logs"
OUT = os.path.join(HERE, "..", "results_xx", "prod_isodoc", "inloop_val.json")
VAL = re.compile(r"validation loss at iteration (\d+) \| lm loss value: ([0-9.Ee+-]+)")


def main():
    pts = {}
    logs = sorted((p for p in glob.glob(os.path.join(RD, "phase*", "*.log")) if not p.endswith("env.log")), key=os.path.getmtime)
    for p in logs:                                         # oldest first: a later log's reading of the same iteration wins
        with open(p, errors="replace") as fh:
            for ln in fh:
                if "validation loss at iteration" in ln:
                    m = VAL.search(ln)
                    if m:
                        pts[int(m.group(1))] = (float(m.group(2)), os.path.basename(p))
    if not pts:
        raise SystemExit("no in-loop validation line found")
    out = {"run": "ULTRA59B_MUONH_KD_QBK3", "total_steps": 97330, "phase2_start": 58000,
           "points": [[it, v, name] for it, (v, name) in sorted(pts.items())], "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M")}
    tmp = f"{OUT}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, OUT)
    print(f"{len(pts)} in-loop readings, last iteration {max(pts):,} -> {OUT}")


if __name__ == "__main__":
    main()
