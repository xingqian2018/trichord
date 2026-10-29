#!/usr/bin/env python3
"""reports_xx/gen_eval_md.py - reports_xx/qwen3_scaling_ladder_eval.md (Chinese) from the per-checkpoint JSONs of collect_doc_eval.py.

    python3 reports_xx/collect_doc_eval.py && python3 reports_xx/gen_eval_md.py
"""
import glob
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/Project/trichord/dense_pretrain_result"
OUT = os.path.join(HERE, "qwen3_scaling_ladder_eval.md")
RUNGS = ["q0", "q1", "q2", "q3", "q4", "q5"]
# (Change ID, optimizer, loss type, alpha) - row order inside a rung
ARMS = [("BASE", "AdamW", "CE", "—"), ("OPT_MUON", "Muon", "CE", "—"), ("OPT_MUONH", "MuonH", "CE", "—"),
        ("KD_MUONH_A0P7", "MuonH", "KD", "0.7"), ("KD_MUONH_A0P8", "MuonH", "KD", "0.8"), ("KD_MUONH_A0P9", "MuonH", "KD", "0.9")]
CE_REF = "OPT_MUONH"   # the KD arm's pair: same recipe, data order and layout, alpha = 0
COLS = [("01_val_v2_overall_tw", "val_v2"), ("02_uncheatable_tw", "uncheatable"), ("03_codefresh_tw", "codefresh"), ("04_v3small_tw", "v3small"),
        ("04_private_internal_tw", "private_internal"), ("05_pile_test_tw", "pile_test"), ("05_pile_10k_tw", "pile_10k"),
        ("00_overall_tw", "overall_tw"), ("00_overall_macro", "overall_macro")]
SIZES = {"q0": ("12 × 512", "48M", "0.63B", "0.20B", "9.6B"), "q1": ("18 × 768", "154M", "1.82B", "0.39B", "30.8B"),
         "q2": ("24 × 1024", "356M", "4.06B", "0.67B", "71.1B"), "q3": ("30 × 1280", "683M", "7.71B", "1.07B", "136.6B"),
         "q4": ("36 × 1536", "1.17B", "13.1B", "1.64B", "233.6B"),
         "q5": ("42 × 1792", "1.84B", "20.6B", "2.39B", "368.0B")}


def load():
    out = {}
    for p in glob.glob(os.path.join(RES, "*.json")):
        d = json.load(open(p))
        out[(d["change"], d["rung"])] = d
    return out


def fmt_d(x):
    """Signed delta with a real minus sign (render_html.py colours U+2212 negatives green)."""
    return f"{'−' if x < 0 else '+'}{abs(x):.3f}"


def row_head(q, opt, lt, a):
    return f"| {q} | {opt} | {lt} | {a} |"


def main():
    R = load()
    L = []
    w = L.append
    w("# Qwen3-MoE scaling ladder — 评测结果（doc-isolated，含 private_internal）")
    w("")
    w(f"_生成时间 {datetime.now():%Y-%m-%d %H:%M}，由 `reports_xx/gen_eval_md.py` 从 `trichord/dense_pretrain_result/*.json` 生成，不要手改。_")
    w("")
    w("## 1. 说明")
    w("")
    w("- Boxiang 的 ladder 最终 checkpoint，我们自己重新评测：**doc-isolated**，9 个 decon suite（**含 private_internal**），dpl 的指标定义。")
    w("- 单位 nats/token，越低越好。`overall_tw` = val_v2、uncheatable、codefresh、v3small、private_internal、pile_test 合在一起按 token 加权；"
      "`overall_macro` = 这六类的平均；pile_10k 不进 overall。")
    w(f"- loss type：CE = 交叉熵；KD = (1 − α)·CE + α·KL(teacher ‖ student)，teacher MiMo-V2.6-Pro-RL。KD 的 CE 对照 = **{CE_REF}**（同 recipe，α = 0）。")
    w("- 已核对：除 private_internal 外，各 suite 与 Boxiang 的 doc-isolated 结果一致（差 ≤ 0.00005）。每个模型的完整结果在 `trichord/dense_pretrain_result/<Change ID>_<rung>.json`。")
    w("")
    w("## 2. 结果（loss）")
    w("")
    w("· = 还没有结果。")
    w("")
    w("| rung | 优化器 | loss type | α | " + " | ".join(c[1] for c in COLS) + " |")
    w("|---|---|---|---|" + "---:|" * len(COLS))
    for q in RUNGS:
        for ch, opt, lt, a in ARMS:
            d = R.get((ch, q))
            if d is None and (q == "q4" and ch != "BASE"):
                continue
            cells = [f"{d['metrics'][k]['loss']:.4f}" if d and k in d["metrics"] else "·" for k, _ in COLS]
            w(row_head(q, opt, lt, a) + " " + " | ".join(cells) + " |")
    w("")
    w(f"## 3. KD − CE（相对 {CE_REF}，负数 = KD 更好）")
    w("")
    w("| rung | loss type | α | " + " | ".join(c[1] for c in COLS) + " |")
    w("|---|---|---|" + "---:|" * len(COLS))
    for q in RUNGS:
        ce = R.get((CE_REF, q))
        if ce is None:
            continue
        for ch, opt, lt, a in ARMS:
            if lt != "KD":
                continue
            d = R.get((ch, q))
            cells = [fmt_d(d['metrics'][k]['loss'] - ce['metrics'][k]['loss']) if d else "·" for k, _ in COLS]
            w(f"| {q} | {lt} | {a} | " + " | ".join(cells) + " |")
    w("")
    w("## 4. 模型大小")
    w("")
    w("| rung | 层数 × hidden | 激活参数（不含 embedding） | 总参数（含 embedding） | 训练 token |")
    w("|---|---|---:|---:|---:|")
    for q in RUNGS:
        sz = SIZES[q]
        w(f"| {q} | {sz[0]} | {sz[1]} | {sz[2]} | {sz[4]} |")
    w("")
    w("MoE，每层 128 个 expert、每个 token 激活 8 个。")
    refs = [(n, R[(c, "ref")]) for c, n in (("MIMO_V25_PRO_BASE", "MiMo-V2.5-Pro-Base"), ("MIMO_V26_PRO_RL", "MiMo-V2.6-Pro-RL")) if (c, "ref") in R]
    if refs:
        w("")
        w("## 5. 外部参考：MiMo（同一 doc-isolated 协议）")
        w("")
        w("同样的 115 个 split、同样的切分（vLLM prompt_logprobs，`reports_xx/mimo_doc_eval.py`）；MiMo 的 token id 与 ladder 的 Qwen3 tokenizer 相同，loss 可直接比较。"
          "V2.6-Pro-RL 是官方 mxfp4 权重（也是 KD 的 teacher），V2.5-Pro-Base 是 fp8。")
        w("")
        w("| 模型 | " + " | ".join(c[1] for c in COLS) + " | overall_tsqw |")
        w("|---|" + "---:|" * (len(COLS) + 1))
        for n, d in refs:
            m = d["metrics"]
            w(f"| {n} | " + " | ".join(f"{m[k]['loss']:.4f}" if k in m else "·" for k, _ in COLS + [("00_overall_tsqw", "")]) + " |")
    open(OUT, "w").write("\n".join(L) + "\n")
    print(f"wrote {OUT} ({len(R)} checkpoints)")
    import subprocess
    subprocess.run([sys.executable, os.path.join(HERE, "gen_eval_html.py")], check=True)   # the visual zh + en HTML


if __name__ == "__main__":
    main()
