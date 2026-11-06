#!/usr/bin/env python3
"""reports_xx/gen_eval_md.py - reports_xx/qwen3_scaling_ladder_eval.md, the agent-readable Markdown mirror of the eval report.

The Markdown is for agents (user 11-04): the zh HTML converted to text by html_to_md.py (every chart becomes a table rebuilt from
its hover tooltips, every toggle pane is expanded), plus an appendix of raw numbers from trichord/dense_pretrain_result/*.json.

    python3 reports_xx/collect_doc_eval.py && python3 reports_xx/gen_eval_md.py      # HTML (zh + en) first, then the Markdown
    python3 reports_xx/gen_eval_md.py --md-only                                      # Markdown from the current HTML

gen_eval_html.py calls `gen_eval_md.py --md-only` at the end of its own run, so the Markdown follows every HTML rebuild.
"""
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import html_to_md  # noqa: E402

RES = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/Project/trichord/dense_pretrain_result"
HTML = os.path.join(HERE, "qwen3_scaling_ladder_eval.html")
OUT = os.path.join(HERE, "qwen3_scaling_ladder_eval.md")
RUNGS = ["q0", "q1", "q2", "q3", "q4", "q5"]
IDX_RUNGS = ["q0", "q1", "q2", "q3", "q4", "q5", "q6"]
# (Change ID, optimizer, loss type, alpha) - row order inside a rung
ARMS = [("BASE", "AdamW", "CE", "—"), ("OPT_MUON", "Muon", "CE", "—"), ("OPT_MUONH", "MuonH", "CE", "—"),
        ("KD_MUONH_A0P7", "MuonH", "KD", "0.7"), ("KD_MUONH_A0P8", "MuonH", "KD", "0.8"), ("KD_MUONH_A0P9", "MuonH", "KD", "0.9")]
COLS = [("01_val_v2_overall_tw", "val_v2"), ("02_uncheatable_tw", "uncheatable"), ("03_codefresh_tw", "codefresh"), ("04_v3small_tw", "v3small"),
        ("04_private_internal_tw", "private_internal"), ("05_pile_test_tw", "pile_test"), ("05_pile_10k_tw", "pile_10k"),
        ("00_overall_tw", "overall_tw"), ("00_overall_macro", "overall_macro"), ("00_overall_tsqw", "overall_tsqw")]
SIZES = {"q0": ("12 × 512", "48M", "0.63B", "9.6B"), "q1": ("18 × 768", "154M", "1.82B", "30.8B"), "q2": ("24 × 1024", "356M", "4.06B", "71.1B"),
         "q3": ("30 × 1280", "683M", "7.71B", "136.6B"), "q4": ("36 × 1536", "1.17B", "13.1B", "233.6B"), "q5": ("42 × 1792", "1.84B", "20.6B", "368.0B"),
         "q6": ("48 × 2048", "2.73B", "—", "—")}

PREFACE = """# Qwen3-MoE scaling ladder 评测报告 —— Markdown 镜像（给 agent 读）

_由 `reports_xx/gen_eval_md.py` 生成于 {ts}：正文 = `qwen3_scaling_ladder_eval.html`（中文版，HTML 生成于 {hts}）经 `reports_xx/html_to_md.py` 逐段转换，
附录 = `trichord/dense_pretrain_result/*.json` 的原始数。不要手改：改 `gen_eval_html.py` 后重跑它（它会自动重写本文件）。_

读法：
- 内容和 HTML 一一对应、顺序相同。HTML 的每个页签 = 下面一个 `## 页签：…` 大节（doc-isolated = 主评测（逐项叠加的架构变化 + 数据 derisk）；doc-isolated老实验 = 其余单项实验（优化器、KD 对 CE、尾部估算方式、KD teacher、文档 mask、SWA、router、负熵均衡、第 0 层 dense、Latent MoE、位置编码），同一套评测；外部参考；LR 拟合实验；packed = Boxiang 训练时自带的验证，绝对值比 doc-isolated 高约 0.013–0.016，两者的绝对值不能互比）。
- 每张图都换成了数据表，数取自图上的悬停提示：表头是 rung（或横轴上的点），每行是图里的一条线；横轴点多时（LR 扫描、训练步数）行列互换，router 健康曲线均匀抽 12 个点（含首尾）。`·` = 该点没有数据。
- HTML 里靠按钮切换的内容全部展开，每块前面标明是哪个按钮：〔指标：…〕（overall_tsqw / overall_tw / overall_macro / 单个验证集）、〔视图：…〕（窗口、统计量等）、〔窗口：…〕。
- 单位 nats/token（另注 bpb、ppl 的除外）；图里的数大多是“该线 − 0 线”的 Δloss，负数 = 比 0 线好；各节的 0 线写在该节说明里，不同节的 0 线常常不同。
- 默认主指标 overall_tsqw（7 个集合按 √token 加权）；每个 checkpoint 的完整结果（115 个 split、bpb、ppl）在 `trichord/dense_pretrain_result/<Change ID>_<rung>.json`，附录 B 是全部 checkpoint 的 overall_tsqw 索引。
"""


def load():
    out = {}
    for p in glob.glob(os.path.join(RES, "*.json")):
        try:
            d = json.load(open(p))
            out[(d["change"], d["rung"])] = d
        except (OSError, ValueError, KeyError):
            continue
    return out


def loss(d, k):
    m = d["metrics"].get(k) if d else None
    return m["loss"] if isinstance(m, dict) else m


def appendix(R):
    L = ["## 附录 A：核心 6 组的绝对 loss（doc-isolated，8K 窗口）", "",
         "Boxiang ladder 的主线：AdamW / Muon / MuonH 三个 CE 优化器，和 MuonH 上 α 0.7 / 0.8 / 0.9 的 KD（teacher MiMo-V2.6-Pro-RL，tail = B64，ArchScale 老 LR，"
         "650B 数据，无文档 mask）。KD 的 CE 对照 = OPT_MUONH（同 recipe，α = 0）。`·` = 还没有结果。", "",
         "| rung | Change ID | 优化器 | loss | α | " + " | ".join(c[1] for c in COLS) + " |", "|---|---|---|---|---|" + "---:|" * len(COLS)]
    for q in RUNGS:
        for ch, opt, lt, a in ARMS:
            d = R.get((ch, q))
            if d is None and q in ("q4", "q5") and ch != "BASE":
                continue
            L.append(f"| {q} | `{ch}` | {opt} | {lt} | {a} | " + " | ".join(f"{v:.4f}" if (v := loss(d, k)) is not None else "·" for k, _ in COLS) + " |")
    L += ["", "模型大小（MoE，每层 128 个 expert 选 8 个）：", "", "| rung | 层数 × hidden | 激活参数（不含 embedding） | 总参数（含 embedding） | 主线训练 token（TPP 200） |",
          "|---|---|---:|---:|---:|"] + [f"| {q} | {s[0]} | {s[1]} | {s[2]} | {s[3]} |" for q, s in SIZES.items()]
    ids = sorted({c for c, q in R if q in IDX_RUNGS})
    L += ["", "## 附录 B：全部 checkpoint 的 overall_tsqw（doc-isolated）", "",
          f"`trichord/dense_pretrain_result` 里的全部 {len(ids)} 个 Change ID。后缀约定：`_S2` / `_S3` = 第 2、3 个 seed（无后缀 = seed 1）；"
          "`_W32K` = 同一 checkpoint 按 32K 窗口评（无后缀 = 8K 窗口）；`_PP32K` = 80B pack plan、32K 训练的新数据管线；`_IT4577` 等 = 改了训练步数（LR 格子）。"
          "Change ID 的含义以正文各节的说明为准。", "",
          "| Change ID | " + " | ".join(IDX_RUNGS) + " |", "|---|" + "---:|" * len(IDX_RUNGS)]
    for c in ids:
        L.append(f"| `{c}` | " + " | ".join(f"{v:.4f}" if (v := loss(R.get((c, q)), "00_overall_tsqw")) is not None else "" for q in IDX_RUNGS) + " |")
    return "\n".join(L) + "\n"


def write_md():
    R = load()
    hts = datetime.fromtimestamp(os.path.getmtime(HTML)).strftime("%Y-%m-%d %H:%M")
    body = html_to_md.convert(HTML)
    md = PREFACE.format(ts=f"{datetime.now():%Y-%m-%d %H:%M}", hts=hts) + "\n" + body + "\n" + appendix(R)
    tmp = OUT + ".tmp"
    open(tmp, "w").write(md)
    os.replace(tmp, OUT)                                     # atomic: readers never see a half-written file
    print(f"wrote {OUT} ({len(md) // 1024} KB, {md.count(chr(10))} lines; html {hts}; {len(R)} checkpoints)")


def main():
    if "--md-only" not in sys.argv[1:]:
        env = dict(os.environ, EVAL_MD_SKIP="1")             # gen_eval_html.py would call back into --md-only; we write the md below
        subprocess.run([sys.executable, os.path.join(HERE, "gen_eval_html.py")], check=True, env=env)
    write_md()


if __name__ == "__main__":
    main()
