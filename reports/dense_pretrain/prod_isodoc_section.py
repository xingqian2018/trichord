#!/usr/bin/env python3
"""reports_xx/prod_isodoc_section.py - the report's tab "prod iso-doc" (user 2026-11-13: "请+一个iso-doc测试tab", "就是prod iso-doc",
"记得测试应该是分为8k vs 32k"): doc-isolated val loss of the PRODUCTION run's checkpoints, by training step, at the 8K and the 32K window.

Two kinds of result per checkpoint and window, the same quantity where both exist (same documents, same tokens, same scoring code):
  full    results_xx/<RUN>[_W32K]_it<N>.json   the 115-split suite of run_doc_eval.sh (stage_prod_ckpt.py + collect_doc_eval.py): the 7
                                              collections and overall_tsqw, both windows from one model load
  pooled  results_xx/prod_isodoc/<RUN>[_W32K]_iter<N>.json   boxiangw's loss-parity doc evals (collect_prod_isodoc.py): pooled sets, so
                                              only the collections he scored (val_v2, v3small) and no overall_tsqw
A full result replaces the pooled one of the same checkpoint and window. Reference rows come from the ordinary doc-isolated results
(R of gen_eval_html.load()). The chart buttons are the seven classic metric options; the table lists every collection, with the columns
of the reference tab's table (TABLE_COLS; model size and vocab size first). A checkpoint / window without a score shows "-".

    from prod_isodoc_section import prod_tab
    prod_tab(lang, esc, R, ladder_size, sizes, vocab) -> (tab label, pane HTML), or None when there is no result yet
"""
import glob
import itertools
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results_xx", "prod_isodoc")   # pooled sets
FULL = os.path.join(HERE, "..", "results_xx")                     # 115-split suite
RUN = "ULTRA59B_MUONH_KD_QBK3"
SHORT = "59B"
COL = "--text-primary"   # 11-16 16:5x (user "我们的线用白色就好，不用区分颜色"): white in dark, black in light (blue --c1 before); = the latest checkpoint in prod downstream
WINS =[(8192, "8K", ""), (32768, "32K", "_W32K")]     # (window, label, Change-ID suffix of the full and of the reference results)
# the classic metric options, as METRICS of gen_eval_html.py (user 11-13: "我们不是经典的 overall_tsqw overall_tw overall_macro val_v2
# uncheatable codefresh private_internal 选项吗"): always all seven; a metric a checkpoint has no score for shows "-"
METRICS = [("00_overall_tsqw", "overall_tsqw"), ("00_overall_tw", "overall_tw"), ("00_overall_macro", "overall_macro"), ("01_val_v2_overall_tw", "val_v2"),
           ("02_uncheatable_tw", "uncheatable"), ("03_codefresh_tw", "codefresh"), ("04_private_internal_tw", "private_internal")]
# the table's columns (user 11-13 20:55: "这个要列出全部的，就是 model size / vocab size / overall_tsqw / overall_tw / overall_macro / val_v2 /
# uncheatable / codefresh / v3small / private_internal / pile_test / pile_10k"): REF_COLS of gen_eval_html.py, i.e. the columns of the
# 外部参考 tab's doc-isolated table, after "model size" and "vocab size". The chart buttons stay the seven classic options above.
TABLE_COLS = [("00_overall_tsqw", "overall_tsqw"), ("00_overall_tw", "overall_tw"), ("00_overall_macro", "overall_macro"), ("01_val_v2_overall_tw", "val_v2"),
              ("02_uncheatable_tw", "uncheatable"), ("03_codefresh_tw", "codefresh"), ("04_v3small_tw", "v3small"),
              ("04_private_internal_tw", "private_internal"), ("05_pile_test_tw", "pile_test"), ("05_pile_10k_tw", "pile_10k")]
VOCAB = 151936             # the production run's padded vocab (its argv: --padded-vocab-size 151936), the Qwen3 embedding rows
# a reference whose run record is not in the shared results: model size and vocab size from its architecture twin (the KD arm of data
# derisk round 2 is the LATENT2_E2X architecture; KD changes the loss, not the model)
SIZE_AS = {"PROD1_DPSK_LATENT2_E2X_KD_A0P8_S64Y_MIMO25_PPFULL": "PROD1_DPSK_LATENT2_E2X"}
# reference rows: (Change ID, rung, text key, colour, ticked by default). All share the Qwen3 token ids, so their loss per token is
# comparable with the production run's. Since 11-16 (user: "和 prod downstream 一样 / 把哪个chart 放在最先面 / 然后紧跟着的是实验列表，可以checkbox
# 的那种，这样方便我们追比其他模型 / 最后才是 数据表格 / 数据表格可以不用根据checkbox变化") every reference has a tick box in "实验列表" that shows / hides
# its dashed line; the chart's y range follows the ticked ones (one drawing per ticked set that changes the range). Ticked by default = the
# two lines drawn before (PROD1_DPSK q5, Qwen3-30B-A3B-Base). Colours: the ladder references = aqua steps (q5 keeps --c3 = --g08), the KD
# teacher orange (the next categorical slot), Qwen3-30B-A3B-Base grey as before; the production run is blue (COL).
# 11-16 16:5x (user "prod iso-doc实验列表，色系用square，而且色系要能统一"): the external models take their colour from
# cosmos_eval_section.PROD_FAM, the same as in prod downstream (Qwen3-30B blues, Qwen3.8 / 3.5 oranges, teacher violet); ladder aqua kept.
REFS = [("PROD1_DPSK", "q5", "r_p5", "--c3", True),
        ("PROD1_DPSK_LATENT2_E2X", "q4", "r_ce", "--g07", False),
        ("PROD1_DPSK_LATENT2_E2X_KD_A0P8_S64Y_MIMO25_PPFULL", "q3", "r_kd", "--g09", False),
        ("QWEN3_30B_A3B_BASE", "ref", "r_qwen", "--muted", True),
        # 11-16 (user, on the Qwen3-30B-A3B-Base line: "把instruct也拉上来 prod iso-doc", then "thinking instruct都要"): the two post-trained
        # text models of the same size, doc-isolated since 11-16 (jobs 71090 / 71091). Not ticked by default since 11-16 16:44 (user: "我们prod
        # isolate，不要默认选择 instruct和thinking"; ticked 15:05 - 16:44). Colours: one family = one 色系
        # (user 11-16 16:12 "而且把色系稍微调节下"; amber and pink until then): the Qwen3-30B three are greys, Base keeps its grey and stays
        # the strongest, the two post-trained ones recede toward the surface (--pg2, --pg3 in PQ_CSS), as --db2 / --db3 do in prod downstream.
        ("QWEN3_30B_A3B_INSTRUCT_2507", "ref", "r_qwen_i", "--pg2", False),
        ("QWEN3_30B_A3B_THINKING_2507", "ref", "r_qwen_t", "--pg3", False),
        # 11-16 (user: "我们有Qwen3.5和3.8的分数吗，也最好再prod里面变成指标", "两边都是，可选的，不是default的指标"): not ticked by default.
        # One family = one hue (violet, the palette's slot 7), stepped by model size: --pq1 .. --pq4 (PQ_CSS). An entry without a result
        # (Qwen3.5 until its eval, job 71603, is collected) is left out of the list, the chart and the tables.
        ("QWEN38_27B", "ref", "r_q38_27", "--pq1", False),
        ("QWEN38_FLASH_NEXT", "ref", "r_q38_fn", "--pq2", False),
        ("QWEN35_397B_A17B", "ref", "r_q35", "--pq3", False),
        ("QWEN38_MAX", "ref", "r_q38_max", "--pq4", False),
        # 11-17 (user "nemo 3.5 nano", then "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16": the iso-doc eval, job 76578). That it is
        # also a line here is my inference from the Qwen3.5 / 3.8 request (told to the user): optional, not ticked. Own tokenizer
        # (131,072 rows, the Nemotron-3 one) -> bpb only. Its colour is PROD_FAM's --dn.
        ("NEMOTRON35_LIGHTNING_30B_BASE", "ref", "r_nemo35l", "--dn", False),
        ("MIMO_V25_PRO_BASE", "ref", "r_teacher", "--c2", False)]
# References with their own tokenizer (Qwen3.5 / Qwen3.8: 248,320 rows, one BPE vocab for both). The same text is a different number
# of tokens (about 2 % fewer bytes per token than Qwen3 overall, 7 % on codefresh), so their loss per token is NOT comparable with the
# production run's: Qwen3.8-27B at 8K has overall_tsqw 1.2678 per token against 1.2761 of step 86,000, but 0.5030 bpb against 0.4970.
# The chart draws their line only under bpb; the tables list their own per-token loss as the 外部参考 tab does (the vocab column tells).
OWN_TOK = {"QWEN38_27B", "QWEN38_FLASH_NEXT", "QWEN35_397B_A17B", "QWEN38_MAX", "NEMOTRON35_LIGHTNING_30B_BASE"}
SIZE_OWN = {"QWEN35_397B_A17B": "397B-A17B"}             # until gen_eval_html.py's REF_SIZE / REF_VOCAB know the model
VOCAB_OWN = {"QWEN35_397B_A17B": 248320}
# the violet ramp of the Qwen3.5 / Qwen3.8 family, lightest = smallest model; validate_palette.js --ordinal passes in both modes
# (light on #fcfcfb: light end 2.16:1; dark on #1a1a19: dark end 2.96:1; adjacent dL >= 0.06).
# --pg2 / --pg3: the grey steps of the Qwen3-30B family after Base's --muted (#898781 in both modes), +-0.07 L each at its hue and
# chroma; --ordinal passes too (light end 2.07:1, dark end 2.69:1).
PQ_CSS = ('<style>.viz-root{--pq1:#a9a5fd;--pq2:#8a7ff4;--pq3:#6e60d2;--pq4:#5341b1;--pg2:#9e9c96;--pg3:#b4b2ac}'
          '@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) '
          '.viz-root{--pq1:#bdbcfd;--pq2:#9c95fe;--pq3:#7e72e7;--pq4:#6353c5;--pg2:#74726d;--pg3:#605e59}}</style>')
# since 11-16 16:5x the colours of REFS' external models come from cosmos_eval_section.PROD_FAM (shared with prod downstream); PQ_CSS is unused
from cosmos_eval_section import PROD_FAM, PROD_FAM_CSS, RK_CSS, rk_block, BCSS, AVG_CSS
REFS = [(ch, q, key, PROD_FAM.get(ch, c) if q == "ref" else c, on) for ch, q, key, c, on in REFS]
STATS = [("loss", "val_loss"), ("bpb", "bpb")]
# the data table folds the checkpoints before this step under a toggle row (user 11-17 "iso doc也要fold", right after "表格里面folder的老
# checkpoint放到 < 70000" for 59B downstream: the same step as gen_eval_html.py's DOWN_FOLD, the same markup and page script .dm-fold-btn)
FOLD = 80000                                             # 11-17 "算了 fold < 80000" (70000 for a few minutes); = DOWN_FOLD
# The pooled-set evals (boxiangw's loss-parity doc evals, results_xx/prod_isodoc/<RUN>_iter<N>.json) are NOT shown (user 11-13: "这个是boxiang的
# 分数对吗，这个暂时不要放上来"): the tab holds only the scores of OUR full-suite evals. The files stay on disk; True shows them again.
SHOW_POOLED = False
TEACHER = {"mimo-v2.5-pro-base": "MiMo-V2.5-Pro-Base", "mimo-v2.6-pro-rl": "MiMo-V2.6-Pro-RL"}   # teacher-cache directory name -> model name
TX = {"zh": {
    "what": "生产 run {run}（{size}；{L} 层，hidden {h}，{E} 个 expert 选 {k}；KD teacher {teacher}，α {alpha}；训练长度 {seq}，共 {total} 步）各 checkpoint 的 doc-isolated val loss。",
    "how": "测法和 doc-isolated 页相同：每篇验证文档单独成一个样本，不带别的文档的上下文，打分代码是同一份。窗口分两种：8K 和所有旧结果可比；32K 和这个 run 的训练长度一致。",
    "kinds": "结果有两种。全量：我们的 115 个子集、7 个集合，有 overall_tsqw，8K 和 32K 出自同一次模型加载。合并集合：Boxiang 的流程给 downstream 评测做 loss 对齐用的评测，只有 val_v2 和 v3small，一个集合一个 loss。两者的文档数和 token 数相同，同一个集合的数值是同一个量，可以和 doc-isolated 页直接对比。全量评测分三个部分并行跑，先跑完的集合先显示，还没跑完的显示 “-”。",
    "w_line": "{w} 窗口：{parts}。", "w_full": "全量 = {its}", "w_part": "全量已跑完一部分 = {its}", "w_pooled": "只补测 val_v2 / v3small = {its}", "w_none": "还没有结果，下面显示 “-”",
    "h_curve": "val loss / bpb 随训练步数的变化", "ylab_bpb": "bpb（bits / byte）",
    "curve_note": "用上面的按钮换指标，用卡片里的 8K / 32K 页签换窗口（图和下面的表一起切换）。横轴是训练步数，下面一行是占全程的比例。虚线是参考：PROD1_DPSK q5（ladder 上训练完成的 checkpoint）和外部基线 Qwen3-30B-A3B-Base，取同一窗口的数值。",
    "ylab": "val loss（nats / token）", "no_win": "{w} 窗口还没有这个指标的结果。", "step": "step", "ckpt": "checkpoint", "prog": "训练进度", "src": "来源",
    "win": "{w} 窗口", "full": "全量", "part": "全量（已跑完的部分）", "pooled": "合并集合", "ours": "我们", "ref_h": "参考",
    "r_kd": "ladder E2X + KD q3（teacher MiMo-V2.5）", "r_ce": "ladder E2X q4（CE）",
    "r_qwen": "Qwen3-30B-A3B-Base", "r_qwen_i": "Qwen3-30B-A3B-Instruct-2507", "r_qwen_t": "Qwen3-30B-A3B-Thinking-2507", "r_teacher": "MiMo-V2.5-Pro-Base（KD teacher）",
    "r_q38_27": "Qwen3.8-27B", "r_q38_fn": "Qwen3.8-Flash-Next", "r_q35": "Qwen3.5-397B-A17B", "r_q38_max": "Qwen3.8-Max", "r_nemo35l": "Mimi 3.5 Small",   # 11-17 13:3x (user: "麻烦名字改成mimi lighting", then "mimi lighting -> Mimi 3.5 Lighting", then "Mimi 3.5 Lighting -> Mimi 3.5 Small")
    "l_both": "val_loss、bpb", "l_bpb": "bpb",
    "own_tok": "有些外部模型用自己的 tokenizer（实验列表里“可比的指标”一列只写 bpb 的那些）：同一段文本切出的 token 数和 Qwen3 的不同，每 token 的 val loss 不能和 prod 直接比。所以它们的虚线只在 bpb 下画；表格里它们的 val loss 是按各自 token 计的数值，和外部参考页相同。",
    "r_p5": "PROD1_DPSK · q5", "c_p5": "PROD1_DPSK · q5", "c_qwen": "Qwen3-30B-A3B-Base",
    "h_list": "实验列表", "l_head": ["显示", "颜色", "模型", "大小", "可比的指标"],
    # 11-16 review ("主要两个可视化就是 val loss 和 downstream 放在最前面"): tab label, sub line, lead sentence, tiles, gap grid, table, notes
    "tab": "59B val loss",
    "sub": "{run}（{size}）各 checkpoint 的 val loss：每篇验证文档单独打分（doc-isolated）。8K 窗口和所有旧结果可比，32K = 这个 run 的训练长度。页顶的按钮切换窗口和 val_loss / bpb，整页一起切换。",
    "lead": "{wl} · {at}，和 Qwen3-30B-A3B-Base 比：{parts}。", "lead_p": "{n} {v}（{cmp} {d}）", "lower": "低", "higher": "高", "lead_sep": "；",
    "lead_at": "（{at}）", "at": "第 {step} 步（{pct}）", "at2": "第 {step} 步，{pct}",
    "t_prog": "训练进度（最新已评测）", "t_prog_s": "第 {step} 步，共 {total} 步",
    "better": "低于 Qwen3-30B-A3B-Base（{q}），更好", "worse": "高于 Qwen3-30B-A3B-Base（{q}），更差", "t_prev": "上一个点（{at2}）Δ {d}",
    "level": "和 Qwen3-30B-A3B-Base（{q}）持平（差距 < {th}）",
    "h_gap": "和 Qwen3-30B-A3B-Base 的差距（59B − Base，负数 = 59B 更好）",
    "gap_note": "每格一个验证集合，纵轴各格相同；蓝色虚线 = 0（和 Base 持平）；竖虚线 = 第 58,000 步（59.6 %）换成 phase 2 数据，各集合的 loss 在第 58,000 → 60,000 步（59.6 → 61.6 %）先升一下再继续降。只画第 40,000 步（41.1 %）以后的点：更早的差距大，会把纵轴撑开（早期见上面的曲线）。标题里的数 = 最新一点的差距。",
    "gap_prev": "上一个点（{at2}）{d}", "phase2": "phase 2 数据",
    "h_inloop": "训练中的 in-loop val（训练日志，每 1,000 步一个读数）",
    "inloop_note": "Megatron 训练时每 1,000 步打印的 validation loss（Boxiang 的打包验证集，带上下文，不是上面的 doc-isolated 测法，绝对值不能和上面直接比）。"
                   "每个读数是验证集的一次随机抽样，单个读数的噪声 σ 约 0.006：灰点 = 单个读数，线 = 同一数据阶段内最近 5 个读数的均值；竖虚线 = 第 58,000 步（59.6 %）换成 phase 2 数据。只画第 40,000 步（41.1 %）以后。",
    "t_inloop": "in-loop val（训练日志）", "t_inloop_s": "{at}；最近 5 个读数均值 {m}（单个读数 σ ≈ 0.006）", "inloop_mean": "最近 5 个均值",
    "merge_row": "59B 平均 {name}（{recipe}；最新点第 {step} 步）", "merge_h": "checkpoint 平均",
    "h_avg": "59B checkpoint 平均 vs Qwen3-30B-A3B-Base", "avg_note": "只测了 8K 窗口的 val_v2 / v3small（val_loss），不随页顶的按钮切换。Δ 负数 = 更低、更好。",
    "avg_wave": "{w}：最新 checkpoint 第 {s} 步", "avg_plain": "第 {s} 步（未平均）", "avg_plain_d": "这一组最新的 checkpoint 本身，作对照",
    "avg_head": ["模型", "平均方式", "val_v2", "Δ vs 未平均", "Δ vs Base", "v3small", "Δ vs 未平均", "Δ vs Base"],
    "h_table": "数据表（最新在上）", "h_sum": "最新数值 vs Qwen3-30B-A3B-Base", "d_row": "Δ 第 {step} 步 − Qwen3-30B-A3B-Base", "pooled_lab": "（仅 val_v2 / v3small）",
    "fold_open": "▸ 展开第 {s} 步之前的 {n} 个 checkpoint", "fold_close": "▾ 收起",
    "notes": "说明：测法、窗口、各窗口已测的步数",
    "p2_note": "第 58,000 步（59.6 %）起训练数据换成 phase 2 的混合（job 67288），所以各集合的 loss 在第 58,000 → 60,000 步（59.6 → 61.6 %）先升（overall_tsqw 约 +0.014）再继续降，不是训练出了问题。",
    "ours_only": "结果来自我们自己的评测：全量 = 115 个子集、7 个集合，8K 和 32K 出自同一次模型加载，分三个部分并行跑，先跑完的集合先显示，还没跑完的显示 “-”。"
                 "另有我们单独补测的 val_v2 和 v3small（只在 8K，和全量里的同一集合是同一个量），表里标“仅 val_v2 / v3small”，其余列为 “-”。Boxiang 流程的同类结果不显示。"},
      "en": {
    "what": "Doc-isolated val loss of the checkpoints of the production run {run} ({size}; {L} layers, hidden {h}, top-{k} of {E} experts; KD teacher {teacher}, α {alpha}; trained at {seq}, {total} steps in all).",
    "how": "Same protocol as the doc-isolated tab: every validation document is its own sample with no context from other documents, scored by the same code. Two windows: 8K is comparable with every earlier result; 32K matches the length this run trains at.",
    "kinds": "There are two kinds of result. Full: our 115 subsets in 7 collections, with overall_tsqw, 8K and 32K from one model load. Pooled: the evals Boxiang's pipeline runs as loss-parity reference of the downstream eval, val_v2 and v3small only, one loss per collection. Both score the same documents and tokens, so a collection's value is the same quantity in either and compares directly with the doc-isolated tab. A full eval runs as three parallel parts: collections that are finished are shown, the others show “-” until their part is done.",
    "w_line": "{w} window: {parts}.", "w_full": "full = {its}", "w_part": "full, parts finished so far = {its}", "w_pooled": "val_v2 / v3small only = {its}", "w_none": "no result yet, shown as “-” below",
    "h_curve": "Val loss / bpb vs. training step", "ylab_bpb": "bpb (bits / byte)",
    "curve_note": "The buttons above switch the metric; the 8K / 32K tabs in the card switch the window (the chart and the table below together). The x axis is the training step, with the share of the whole run below it. Dashed lines are references at the same window: PROD1_DPSK q5 (a finished ladder checkpoint) and the external baseline Qwen3-30B-A3B-Base.",
    "ylab": "val loss (nats / token)", "no_win": "No result for this metric at the {w} window yet.", "step": "step", "ckpt": "checkpoint", "prog": "training progress", "src": "source",
    "win": "{w} window", "full": "full", "part": "full (parts finished so far)", "pooled": "pooled", "ours": "ours", "ref_h": "References",
    "r_kd": "ladder E2X + KD q3 (teacher MiMo-V2.5)", "r_ce": "ladder E2X q4 (CE)",
    "r_qwen": "Qwen3-30B-A3B-Base", "r_qwen_i": "Qwen3-30B-A3B-Instruct-2507", "r_qwen_t": "Qwen3-30B-A3B-Thinking-2507", "r_teacher": "MiMo-V2.5-Pro-Base (KD teacher)",
    "r_q38_27": "Qwen3.8-27B", "r_q38_fn": "Qwen3.8-Flash-Next", "r_q35": "Qwen3.5-397B-A17B", "r_q38_max": "Qwen3.8-Max", "r_nemo35l": "Mimi 3.5 Small",   # 11-17 13:3x (user: "麻烦名字改成mimi lighting", then "mimi lighting -> Mimi 3.5 Lighting", then "Mimi 3.5 Lighting -> Mimi 3.5 Small")
    "l_both": "val_loss, bpb", "l_bpb": "bpb",
    "own_tok": "Some external models have their own tokenizer (those whose \"comparable on\" column in the experiment list says bpb only): the same text is a different number of tokens than with Qwen3's, so their val loss per token does not compare with the production run's. Their dashed lines are drawn under bpb only; their val loss in the tables is per token of their own, as in the external reference tab.",
    "r_p5": "PROD1_DPSK · q5", "c_p5": "PROD1_DPSK · q5", "c_qwen": "Qwen3-30B-A3B-Base",
    "h_list": "Experiment list", "l_head": ["show", "colour", "model", "size", "comparable on"],
    "tab": "59B val loss",
    "sub": "Val loss of the checkpoints of {run} ({size}): every validation document scored on its own (doc-isolated). The 8K window compares with every earlier result; 32K = the length this run trains at. The buttons at the top switch the window and val_loss / bpb for the whole page.",
    "lead": "{wl} · {at}, vs Qwen3-30B-A3B-Base: {parts}.", "lead_p": "{n} {v} ({d} {cmp})", "lower": "below", "higher": "above", "lead_sep": "; ",
    "lead_at": " ({at})", "at": "step {step} ({pct})", "at2": "step {step}, {pct}",
    "t_prog": "Training progress (latest evaluated)", "t_prog_s": "step {step} of {total}",
    "better": "below Qwen3-30B-A3B-Base ({q}): better", "worse": "above Qwen3-30B-A3B-Base ({q}): worse", "t_prev": "previous point ({at2}): Δ {d}",
    "level": "level with Qwen3-30B-A3B-Base ({q}; gap < {th})",
    "h_gap": "Gap to Qwen3-30B-A3B-Base (59B − Base; negative = 59B better)",
    "gap_note": "One panel per validation collection, same y axis in all; the blue dashed line = 0 (level with Base); the vertical dotted line = the switch to the phase-2 data at step 58,000 (59.6 %), after which every collection's loss rises once (steps 58,000 → 60,000, 59.6 → 61.6 %) and then keeps falling. Only steps from 40,000 (41.1 %) on: the early gaps are large and would stretch the y axis (see the curves above). The number in each title = the latest gap.",
    "gap_prev": "previous ({at2}) {d}", "phase2": "phase-2 data",
    "h_inloop": "In-loop val during training (training log, one reading per 1,000 steps)",
    "inloop_note": "The validation loss Megatron prints every 1,000 steps (Boxiang's packed validation set, with context; not the doc-isolated protocol above, so the levels do not compare). "
                   "Each reading is one random draw of the validation set, noise sigma about 0.006 per reading: grey dots = single readings, line = mean of the last 5 readings within the same data phase; dotted line = the switch to the phase-2 data at step 58,000 (59.6 %). From step 40,000 (41.1 %) on.",
    "t_inloop": "In-loop val (training log)", "t_inloop_s": "{at}; mean of the last 5 readings {m} (one reading sigma ≈ 0.006)", "inloop_mean": "mean of last 5",
    "merge_row": "59B average {name} ({recipe}; newest step {step})", "merge_h": "checkpoint averages",
    "h_avg": "59B checkpoint averages vs Qwen3-30B-A3B-Base", "avg_note": "Only val_v2 / v3small at the 8K window were evaluated (val_loss); the buttons at the top do not switch this table. Negative Δ = lower = better.",
    "avg_wave": "{w}: newest checkpoint step {s}", "avg_plain": "step {s} (not averaged)", "avg_plain_d": "the newest checkpoint of the group itself, for reference",
    "avg_head": ["model", "averaging", "val_v2", "Δ vs un-averaged", "Δ vs Base", "v3small", "Δ vs un-averaged", "Δ vs Base"],
    "h_table": "Data table (latest first)", "h_sum": "Latest values vs Qwen3-30B-A3B-Base", "d_row": "Δ step {step} − Qwen3-30B-A3B-Base", "pooled_lab": " (val_v2 / v3small only)",
    "fold_open": "▸ show the {n} checkpoints before step {s}", "fold_close": "▾ hide",
    "notes": "Notes: protocol, windows, evaluated steps",
    "p2_note": "From step 58,000 (59.6 %) the training data is the phase-2 blend (job 67288): every collection's loss rises from step 58,000 to 60,000 (59.6 → 61.6 %; overall_tsqw about +0.014) and then keeps falling; this is not a training problem.",
    "ours_only": "Only our own evals are shown. Full = 115 subsets in 7 collections, 8K and 32K from one model load, run as three parallel parts: finished collections are shown, the others show “-”. "
                 "Our separate val_v2 / v3small evals (8K only; the same quantity as those collections of the full suite) fill steps the full suite skipped; they are marked “val_v2 / v3small only”, the other columns show “-”. Boxiang's pipeline results of this kind are not shown."}}


def load():
    """{window: {iteration: {"metrics": {key: {loss, bpb, ..}}, "kind": full | part | pooled, "also": [kinds that filled gaps], "job",
    "date", "by", "train"}}} of RUN."""
    found = []                                            # (rank, window, iteration, entry); rank 0 pooled, 1 parts of a full eval, 2 full
    for p in glob.glob(os.path.join(RESULTS, RUN + "*_iter*.json")):
        d = json.load(open(p))
        if d.get("run") == RUN:
            s, kind = d["source"], d.get("kind", "pooled")
            if kind == "pooled" and not SHOW_POOLED and s.get("by") != "xingqianx":   # 11-16 review: our own val_v2 / v3small evals of the
                continue                                                             # steps the full suite skipped are shown; Boxiang's stay hidden
            found.append((0 if kind == "pooled" else 1, int(d["window"]), int(d["iteration"]),
                          {"metrics": d["metrics"], "kind": "part" if kind == "partial" else kind, "job": s["eval_job"], "date": s["evaluated_at"],
                           "by": s["by"], "train": d.get("train")}))
    for win, _, suf in WINS:
        for p in glob.glob(os.path.join(FULL, f"{RUN}{suf}_it[0-9]*.json")):
            d = json.load(open(p))
            if d.get("change") == RUN + suf:
                found.append((2, win, int(d["iteration"]), {"metrics": d["metrics"], "kind": "full", "job": d["eval_job"], "date": d["evaluated_at"],
                                                           "by": "xingqianx", "train": None}))
    out = {}
    for rank, win, it, e in sorted(found, key=lambda x: x[:3]):   # ascending rank: the better source overrides, the others fill what it lacks
        old = out.setdefault(win, {}).get(it)
        if old:
            e = {**e, "metrics": {**old["metrics"], **e["metrics"]}, "train": e["train"] or old["train"],
                 "also": [x for x in old.get("also", []) + [old["kind"]] if x != e["kind"]]}
        out[win][it] = e
    return out


PHASE2 = 58000          # the training data switches to the phase-2 blend at this step (job 67288): every curve rises once from 58k to 60k
GAP_REF = "QWEN3_30B_A3B_BASE"                           # the default comparison of the tiles, the gap grid and the table's delta row
# the gap grid (11-16 review): 59B minus Qwen3-30B-A3B-Base per collection, at the same window and statistic
GAP_KEYS = [("00_overall_tsqw", "overall_tsqw"), ("01_val_v2_overall_tw", "val_v2"), ("04_v3small_tw", "v3small"), ("02_uncheatable_tw", "uncheatable"),
            ("03_codefresh_tw", "codefresh"), ("04_private_internal_tw", "private_internal"), ("05_pile_test_tw", "pile_test"), ("00_overall_macro", "overall_macro")]
TILE_KEYS = [("00_overall_tsqw", "overall_tsqw"), ("01_val_v2_overall_tw", "val_v2"), ("04_v3small_tw", "v3small")]
GAP_FROM = 40000                                         # the gap grid starts here (11-16 review: step 10,000's large gaps flattened the recent ones)
LEVEL = {"loss": 0.002, "bpb": 0.001}                    # a tile calls |59B - Base| below this "level" (持平), not better / worse
M = (56, 24, 16, 42)                                     # margins of the detail chart (left, right, top, bottom); the overlay uses the same
GM = (44, 10, 12, 22)                                    # margins of a gap panel
VIEW_CSS = ("<style>.pi-ctl{position:sticky;top:var(--nav-h,38px);z-index:1;background:var(--page);display:flex;flex-wrap:wrap;gap:6px 18px;align-items:center;"
            "padding:8px 0;margin:0 0 4px}.pi-ctl .rtoggle{margin:0}"
            ".gap-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px 18px;margin:4px 0}"
            "@media (max-width:900px){.gap-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media (max-width:560px){.gap-grid{grid-template-columns:1fr}}"
            "[data-metric]{overflow-x:auto}[data-metric] .rk-wrap{min-width:600px}"
            ".gap-t{font-size:12.5px;font-weight:600;line-height:1.3}.gap-t span{font-weight:400;color:var(--text-secondary)}"
            ".gap-grid svg,.rk-v svg{display:block}tr.drow td{color:var(--text-secondary);font-style:italic}"
            "details.notes>summary{cursor:pointer;color:var(--text-secondary);font-size:13px}"
            ".tile .dl{font-size:13px;font-weight:600;margin:2px 0}.tile .dl span{font-weight:400;color:var(--text-secondary)}"
            ".lead{font-size:15px;margin:6px 0 12px}</style>")


# 11-17 checkpoint averages (Runlong "尝试几个加权平均的版本，都测试下，都发"; tmp/ablate59/merge59.sh): their doc evals (8K, val_v2 / v3small)
MERGE_DOC = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/tmp/ablate59/doc_eval"
MERGE_RECIPE = {"zh": {"U2": "最近 2 个等权", "U3": "最近 3 个等权", "U5": "最近 5 个等权", "U8": "最近 8 个等权", "LIN5": "最近 5 个，5:4:3:2:1",
                       "EXP6": "最近 6 个，0.67 指数衰减", "EXPr05": "最近 4 个，0.5 指数衰减", "POLYH": "Polyak，退火后半段等权",
                       "POLY": "Polyak，phase 2 全部等权", "DU3": "密集点（每 250 步）：最近 3 个等权", "DU5": "密集点：最近 5 个等权",
                       "DU10": "密集点：最近 10 个等权（约 2,080 步）", "DLIN5": "密集点：最近 5 个，5:4:3:2:1",
                       "DEXPT500": "密集点：exp(−距离 / 500 步)", "DEXPT1000": "密集点：exp(−距离 / 1,000 步)"},
                "en": {"U2": "uniform, last 2", "U3": "uniform, last 3", "U5": "uniform, last 5", "U8": "uniform, last 8", "LIN5": "last 5, 5:4:3:2:1",
                       "EXP6": "last 6, exp 0.67", "EXPr05": "last 4, exp 0.5", "POLYH": "Polyak, 2nd half of the decay", "POLY": "Polyak, all of phase 2",
                       "DU3": "dense (every 250 steps): uniform, last 3", "DU5": "dense: uniform, last 5", "DU10": "dense: uniform, last 10 (~2,080 steps)",
                       "DLIN5": "dense: last 5, 5:4:3:2:1", "DEXPT500": "dense: exp(-distance / 500 steps)", "DEXPT1000": "dense: exp(-distance / 1,000 steps)"}}


def _merges():
    """[(wave_recipe, newest step, recipe key, {set: loss})] of the averages' doc evals, newest wave first, best val_v2 first."""
    out = []
    for f in glob.glob(os.path.join(MERGE_DOC, "M*_*", "ultra59b-iter*", "results.json")):
        name = os.path.basename(os.path.dirname(os.path.dirname(f)))
        try:
            d = json.load(open(f))["doc"]
            it = int(os.path.basename(os.path.dirname(f)).split("-")[1][4:])
        except (OSError, ValueError, KeyError, IndexError):
            continue
        out.append((name, it, name.split("_", 1)[1], d))
    return sorted(out, key=lambda x: (-x[1], x[3].get("val_v2_decon", 9)))


INLOOP = os.path.join(HERE, "..", "results_xx", "prod_isodoc", "inloop_val.json")   # collect_inloop59.py (in-loop val from the training logs)
INLOOP_FROM, INLOOP_WIN = 40000, 5


def _inloop():
    """[(iteration, reading, mean of the last INLOOP_WIN readings of the same data phase)] or []."""
    try:
        d = json.load(open(INLOOP))
    except (OSError, ValueError):
        return []
    p2, out, ph = d.get("phase2_start") or PHASE2, [], []
    for it, v, *_ in d["points"]:
        ph = [x for x in ph if (x[0] >= p2) == (it >= p2)] + [(it, v)]   # the window restarts at the phase-2 switch
        ph = ph[-INLOOP_WIN:]
        out.append((it, v, sum(x[1] for x in ph) / len(ph)))
    return out


def _inloop_chart(esc, t, pts, total, w=872, h=300):
    """grey dots = single readings, a line in the run's ink = the running mean; x from INLOOP_FROM to the end of the run."""
    ml, mr, mt, mb = M
    pts = [p for p in pts if p[0] >= INLOOP_FROM]
    xmax = total or max(p[0] for p in pts)
    x0, tick = INLOOP_FROM, 10000
    lo, hi = min(min(p[1], p[2]) for p in pts), max(max(p[1], p[2]) for p in pts)
    pad = (hi - lo) * 0.06 or 0.005
    a, b, st, dec = _nice(lo - pad, hi + pad)
    X = lambda s: ml + (s - x0) / (xmax - x0) * (w - ml - mr)
    Y = lambda v: mt + (b - v) / (b - a) * (h - mt - mb)
    pct = lambda s: f"{100 * s / total:.1f} %" if total else ""
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="in-loop val">']
    for i in range(int(round((b - a) / st)) + 1):
        yv = a + i * st
        o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" stroke="var(--grid)" stroke-width="1"/>'
                 f'<text x="{ml - 6}" y="{Y(yv) + 4:.1f}" text-anchor="end">{yv:.{dec}f}</text>')
    for s in range(x0, xmax + 1, tick):
        o.append(f'<text x="{X(s):.1f}" y="{h - mb + 16}" text-anchor="middle">{s // 1000}k</text>'
                 f'<text x="{X(s):.1f}" y="{h - mb + 29}" text-anchor="middle" style="fill:var(--muted)">{pct(s)}</text>')
    o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}" stroke="var(--axis)"/>')
    o.append(f'<text x="12" y="{(h - mb + mt) / 2:.0f}" transform="rotate(-90 12 {(h - mb + mt) / 2:.0f})" text-anchor="middle">{esc(t["ylab"])}</text>')
    if x0 < PHASE2 < xmax:
        o.append(f'<line x1="{X(PHASE2):.1f}" x2="{X(PHASE2):.1f}" y1="{mt}" y2="{h - mb}" stroke="var(--muted)" stroke-dasharray="2 3"/>'
                 f'<text x="{X(PHASE2) + 5:.1f}" y="{mt + 10}" style="fill:var(--muted);font-size:10.5px">{esc(t["phase2"])}</text>')
    for seg in ([p for p in pts if p[0] < PHASE2], [p for p in pts if p[0] >= PHASE2]):   # the mean line, one piece per data phase
        if len(seg) > 1:
            o.append(f'<path d="{" ".join(("M" if i == 0 else "L") + f"{X(s):.1f},{Y(m):.1f}" for i, (s, _, m) in enumerate(seg))}" fill="none" '
                     f'stroke="var({COL})" stroke-width="2" stroke-linejoin="round"/>')
    for s, v, _ in pts:
        o.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="3.5" fill="var(--muted)" stroke="var(--surface-1)" stroke-width="1.5"/>')
    s, v, m = pts[-1]
    o.append(f'<text class="lbl" x="{X(s):.1f}" y="{Y(min(v, m)) + 18:.1f}" text-anchor="middle">{v:.4f}</text>')
    for i, (s, v, m) in enumerate(pts):
        a_ = X(s) - (X(s) - X(pts[i - 1][0])) / 2 if i else ml
        b_ = X(s) + (X(pts[i + 1][0]) - X(s)) / 2 if i + 1 < len(pts) else w - mr
        tip = (f"<b>{t['step']} {s:,} · {pct(s)}</b><br><span style=\"color:var(--muted)\">●</span> in-loop val: {v:.4f}"
               f"<br><span style=\"color:var({COL})\">●</span> {esc(t['inloop_mean'])}: {m:.4f}")
        o.append(f'<rect class="hit" x="{a_:.1f}" y="{mt}" width="{b_ - a_:.1f}" height="{h - mt - mb}" data-tip="{esc(tip)}"/>')
    o.append("</svg>")
    return "".join(o)


def _nice(lo, hi, n=5):
    """round y ticks covering [lo, hi] in about n intervals: (first, last, step, decimals)."""
    raw = max(hi - lo, 1e-9) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw * (1 - 1e-9))
    a, b = math.floor(lo / step + 1e-9) * step, math.ceil(hi / step - 1e-9) * step
    dec = max(0, -math.floor(math.log10(step) + 1e-9)) + (1 if abs(step / (10 ** math.floor(math.log10(step) + 1e-9)) - 2.5) < 1e-6 else 0)
    return a, b, step, dec


def _xaxis(total, xs):
    """x range = the whole run (11-16 review: the remaining steps stay visible) and the tick spacing."""
    xmax = total or max(10000, math.ceil(max(xs) * 1.06 / 5000) * 5000)
    return xmax, next(s for s in (5000, 10000, 20000, 50000) if xmax / s <= 10)


def _legend(esc, items):
    """items = [(name, colour var, dashed, code or None)], the report's legend markup; data-pi = code: shown while its box is ticked."""
    sw = lambda c, dashed: f"repeating-linear-gradient(90deg,var({c}) 0 4px,transparent 4px 7px)" if dashed else f"var({c})"
    return '<div class="legend" style="flex-wrap:wrap">' + "".join(f'<span{f" data-pi={chr(34)}{esc(k)}{chr(34)}" if k else ""}><i style="background:{sw(c, d)}"></i>{esc(n)}</span>'
                                                                 for n, c, d, k in items) + "</div>"


def _chart(esc, t, pts, refs, total, w=872, h=340, fit=None, ylab=None):
    """pts = [(step, value)] of the production run; refs = [(name, colour var, value, code)] drawn as dashed horizontal lines (data-pi = code:
    the tick boxes of "实验列表" show / hide them; their names and values are in the legend above the chart). fit = the reference values the
    y range must hold (default: all); a reference outside the range is not drawn. No hover layer: _hits() draws it once per view."""
    ml, mr, mt, mb = M
    xmax, tick = _xaxis(total, [s for s, _ in pts])
    ys = [v for _, v in pts] + ([v for _, _, v, _ in refs] if fit is None else list(fit))
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.06 or 0.01
    a, b, st, dec = _nice(lo - pad, hi + pad)
    refs = [r for r in refs if a <= r[2] <= b]
    X = lambda s: ml + s / xmax * (w - ml - mr)
    Y = lambda v: mt + (b - v) / (b - a) * (h - mt - mb)
    pct = lambda s: f"{100 * s / total:.1f} %" if total else ""
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for i in range(int(round((b - a) / st)) + 1):
        yv = a + i * st
        o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" stroke="var(--grid)" stroke-width="1"/>'
                 f'<text x="{ml - 6}" y="{Y(yv) + 4:.1f}" text-anchor="end">{yv:.{dec}f}</text>')
    for s in range(0, xmax + 1, tick):
        o.append(f'<text x="{X(s):.1f}" y="{h - mb + 16}" text-anchor="middle">{s // 1000}k</text>')
        if s and total:
            o.append(f'<text x="{X(s):.1f}" y="{h - mb + 29}" text-anchor="middle" style="fill:var(--muted)">{pct(s)}</text>')
    o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}" stroke="var(--axis)"/>')
    o.append(f'<text x="12" y="{(h - mb + mt) / 2:.0f}" transform="rotate(-90 12 {(h - mb + mt) / 2:.0f})" text-anchor="middle">{esc(ylab or t["ylab"])}</text>')
    if 0 < PHASE2 < xmax:
        o.append(f'<line x1="{X(PHASE2):.1f}" x2="{X(PHASE2):.1f}" y1="{mt}" y2="{h - mb}" stroke="var(--muted)" stroke-dasharray="2 3"/>'
                 f'<text x="{X(PHASE2) + 5:.1f}" y="{mt + 10}" style="fill:var(--muted);font-size:10.5px">{esc(t["phase2"])}</text>')
    for n, c, v, k in refs:
        o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var({c})" stroke-width="1.6" stroke-dasharray="5 4" data-pi="{esc(k)}"/>')
    d = " ".join(f"{'M' if i == 0 else 'L'}{X(s):.1f},{Y(v):.1f}" for i, (s, v) in enumerate(pts))
    o.append(f'<path d="{d}" fill="none" stroke="var({COL})" stroke-width="2" stroke-linejoin="round"/>')
    for s, v in pts:
        o.append(f'<circle cx="{X(s):.1f}" cy="{Y(v):.1f}" r="4" fill="var({COL})" stroke="var(--surface-1)" stroke-width="2"/>')
    ls, lv = pts[-1]
    o.append(f'<text class="lbl" x="{X(ls):.1f}" y="{Y(lv) - 10:.1f}" text-anchor="middle">{lv:.4f}</text>')
    o.append("</svg>")
    return "".join(o)


def _hits(esc, t, pts, refs, total, w=872, h=340):
    """the hover layer of a view, one band per checkpoint: the run and every reference with a value (data-pi rows follow the boxes)."""
    ml, mr, mt, mb = M
    xmax, _ = _xaxis(total, [s for s, _ in pts])
    X = lambda s: ml + s / xmax * (w - ml - mr)
    pct = lambda s: f"{100 * s / total:.1f} %" if total else ""
    o = [f'<svg class="rk-hit" viewBox="0 0 {w} {h}" aria-hidden="true">']
    for i, (s, v) in enumerate(pts):
        a = X(s) - (X(s) - X(pts[i - 1][0])) / 2 if i else ml
        b = X(s) + (X(pts[i + 1][0]) - X(s)) / 2 if i + 1 < len(pts) else w - mr
        rows = sorted([(SHORT, COL, v, None)] + list(refs), key=lambda r: r[2], reverse=True)
        tip = f"<b>{t['step']} {s:,} · {pct(s)}</b>" + "".join(
            (f'<span data-pi="{esc(k)}">' if k else "") + f'<br><span style="color:var({c})">●</span> {esc(n)}: {y:.4f}' + ("</span>" if k else "") for n, c, y, k in rows)
        o.append(f'<rect class="hit" x="{a:.1f}" y="{mt}" width="{b - a:.1f}" height="{h - mt - mb}" data-tip="{esc(tip)}"/>')
    o.append("</svg>")
    return "".join(o)


def _gap(esc, t, name, g, lo, hi, total, base_name, base_col, w=300, h=170):
    """one panel of the gap grid: g = [(step, gap, ours, base)]; lo / hi = the y range shared by every panel of the grid."""
    ml, mr, mt, mb = GM
    xmax, _ = _xaxis(total, [x[0] for x in g])
    x0, tick = GAP_FROM, 20000
    a, b, st, dec = _nice(lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.05, n=4)
    X = lambda s: ml + (s - x0) / (xmax - x0) * (w - ml - mr)
    Y = lambda v: mt + (b - v) / (b - a) * (h - mt - mb)
    pct = lambda s: f"{100 * s / total:.1f} %" if total else ""
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{esc(name)}">']
    for i in range(int(round((b - a) / st)) + 1):
        yv = a + i * st
        o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" stroke="var(--grid)" stroke-width="1"/>'
                 f'<text x="{ml - 5}" y="{Y(yv) + 3.5:.1f}" text-anchor="end" style="font-size:10px;fill:var(--muted)">{yv:+.{dec}f}</text>')
    for s in range(x0, xmax + 1, tick):
        o.append(f'<text x="{X(s):.1f}" y="{h - mb + 14}" text-anchor="middle" style="font-size:10px;fill:var(--muted)">{s // 1000}k</text>')
    o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{h - mb}" y2="{h - mb}" stroke="var(--axis)"/>')
    if x0 < PHASE2 < xmax:
        o.append(f'<line x1="{X(PHASE2):.1f}" x2="{X(PHASE2):.1f}" y1="{mt}" y2="{h - mb}" stroke="var(--muted)" stroke-dasharray="2 3"/>')
    o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var({base_col})" stroke-width="1.6" stroke-dasharray="5 4"/>')
    o.append(f'<path d="{" ".join(("M" if i == 0 else "L") + f"{X(s):.1f},{Y(d):.1f}" for i, (s, d, _, _) in enumerate(g))}" fill="none" '
             f'stroke="var({COL})" stroke-width="2" stroke-linejoin="round"/>')
    for s, d, _, _ in g:
        o.append(f'<circle cx="{X(s):.1f}" cy="{Y(d):.1f}" r="3.5" fill="var({COL})" stroke="var(--surface-1)" stroke-width="1.5"/>')
    s, d, _, _ = g[-1]
    left = X(s) > w - mr - 46
    o.append(f'<text x="{X(s) + (-7 if left else 7):.1f}" y="{Y(d) + (-8 if left else 4):.1f}" text-anchor="{"end" if left else "start"}" '
             f'style="fill:var(--text-primary);font-size:11px;font-weight:600">{d:+.4f}</text>')
    for i, (s, d, v, q) in enumerate(g):
        x0 = X(s) - (X(s) - X(g[i - 1][0])) / 2 if i else ml
        x1 = X(s) + (X(g[i + 1][0]) - X(s)) / 2 if i + 1 < len(g) else w - mr
        tip = (f"<b>{esc(name)} · {t['step']} {s:,} · {pct(s)}</b><br>Δ: {d:+.4f}<br><span style=\"color:var({COL})\">●</span> {SHORT}: {v:.4f}"
               f"<br><span style=\"color:var({base_col})\">●</span> {esc(base_name)}: {q:.4f}")
        o.append(f'<rect class="hit" x="{x0:.1f}" y="{mt}" width="{x1 - x0:.1f}" height="{h - mt - mb}" data-tip="{esc(tip)}"/>')
    o.append("</svg>")
    prev = (f' <span>{esc(t["gap_prev"].format(at2=t["at2"].format(step=f"{g[-2][0]:,}", pct=pct(g[-2][0])), d=f"{g[-2][1]:+.4f}"))}</span>'
            if len(g) > 1 else "")
    return f'<div><div class="gap-t">{esc(name)} <b>{d:+.4f}</b>{prev}</div>{"".join(o)}</div>'


def prod_tab(lang, esc, R, ladder_size=None, sizes=None, vocab=None):
    """vocab(Change ID, rung) -> vocab size of a reference row (gen_eval_html: REF_VOCAB / ladder_vocab), or None.
    Layout (11-16 review of "主要两个可视化就是 val loss 和 downstream 放在最前面"; before: user 11-16 "chart → 实验列表 → 表格"): one control bar
    (window 8K / 32K, val_loss / bpb) for the whole tab, then the lead sentence + tiles (latest values vs Qwen3-30B-A3B-Base), the gap grid
    (59B − Base per collection), the absolute curves with the reference lines (metric buttons), "实验列表" (tick boxes: which references the
    curves draw), the table (every row, latest first, not driven by the boxes), the notes folded."""
    t = TX[lang]
    res = load()
    if not res:
        return None
    sizes = sizes or {}
    tr = next((d["train"] for w in res.values() for d in w.values() if d.get("train")), None) or {}
    total = tr.get("total_iterations")
    cols = list(METRICS)
    sep = "、" if lang == "zh" else ", "
    kfmt = lambda n: f"{n // 1024}K" if n else "?"
    size = next((v for k, v in sizes.items() if k.startswith(RUN)), "")
    what = t["what"].format(run=RUN, size=size or "?", L=tr.get("num_layers", "?"), h=tr.get("hidden_size", "?"), E=tr.get("num_experts", "?"),
                            k=tr.get("moe_router_topk", "?"), teacher=TEACHER.get(tr.get("kd_teacher"), tr.get("kd_teacher") or "?"), alpha=tr.get("kd_alpha", "?"),
                            seq=kfmt(tr.get("seq_length")), total=f"{total:,}" if total else "?")
    notes = [t["sub"].format(run=RUN, size=size or "?"), what, t["how"], t["kinds"] if SHOW_POOLED else t["ours_only"], t["p2_note"]]
    for win, wl, _ in WINS:
        parts = []
        for kind in ("full", "part", "pooled"):
            its_k = sorted(it for it, d in res.get(win, {}).items() if d["kind"] == kind)
            if its_k:
                parts.append(t["w_" + kind].format(its=sep.join(t["at"].format(step=f"{i:,}", pct=f"{100 * i / total:.1f} %" if total else "?") for i in its_k)))
        notes.append(t["w_line"].format(w=wl, parts=("；" if lang == "zh" else "; ").join(parts) if parts else t["w_none"]))

    def ref_val(ch, q, suf, k, sec="loss"):
        m = R.get((ch + suf, q))
        return m[k].get(sec) if m and k in m else None

    def ref_size(ch, q):
        tw = SIZE_AS.get(ch, ch)
        return sizes.get(ch) or ((ladder_size(tw, q) or ("",))[0] if ladder_size and q != "ref" else SIZE_OWN.get(ch, "")), tw

    refs_on = [r for r in REFS if any((r[0] + suf, r[1]) in R for _, _, suf in WINS)]   # the references that have a result
    if any(r[0] in OWN_TOK for r in refs_on):
        notes.append(t["own_tok"])
    notes_card = (f'<details class="card notes"><summary>{esc(t["notes"])}</summary><ul class="f">'
                  + "".join(f"<li>{esc(x)}</li>" for x in notes) + "</ul></details>")
    code = lambda ch, q: f"{ch}_{q}"                      # a reference's key in the tick boxes / data-pi
    on0 = [on for *_, on in refs_on]
    hid = lambda i: "" if i == 0 else " hidden"
    pct = lambda s: f"{100 * s / total:.1f} %" if total else "?"
    at = lambda s: t["at"].format(step=f"{s:,}", pct=pct(s))
    base_name = t["r_qwen"]
    base_col = next((c for ch, q, _, c, _ in REFS if ch == GAP_REF and q == "ref"), "--muted")
    qref = lambda suf, k, sec: ref_val(GAP_REF, "ref", suf, k, sec)

    def series(win, k, sec):
        return [(it, d["metrics"][k][sec]) for it, d in sorted(res.get(win, {}).items()) if k in d["metrics"] and d["metrics"][k].get(sec) is not None]

    wid = {win: f"w{win // 1024}" for win, _, _ in WINS}
    il = _inloop()
    # ---- per window and statistic: lead sentence + tiles, and the gap grid
    heads, gaps = "", ""
    for i, (win, wl, suf) in enumerate(WINS):
        heads += f'<div data-win="{wid[win]}"{hid(i)}>'
        gaps += f'<div data-win="{wid[win]}"{hid(i)}>'
        last_it = max(res.get(win, {}) or [0])
        for si, (sec, _) in enumerate(STATS):
            tiles = [(t["t_prog"], pct(last_it) if last_it else "-", t["t_prog_s"].format(step=f"{last_it:,}", total=f"{total:,}" if total else "?"), None)]
            if il:                                       # the latest in-loop reading (11-16 19:4x: the page carries it too)
                tiles.append((t["t_inloop"], f"{il[-1][1]:.4f}", t["t_inloop_s"].format(at=at(il[-1][0]), m=f"{il[-1][2]:.4f}"), None))
            parts = []
            for k, n in TILE_KEYS:
                s = series(win, k, sec)
                if not s:
                    continue
                it, v = s[-1]
                q = qref(suf, k, sec)
                dl, sub = None, at(it)
                if q is not None:
                    d = v - q
                    dl = (f"Δ {d:+.4f}", t["level"].format(q=f"{q:.4f}", th=LEVEL[sec]) if abs(d) < LEVEL[sec] else t["better" if d < 0 else "worse"].format(q=f"{q:.4f}"))
                    parts.append(t["lead_p"].format(n=n, v=f"{v:.4f}", cmp=t["lower" if d < 0 else "higher"], d=f"{abs(d):.4f}")
                                 + (t["lead_at"].format(at=at(it)) if it != last_it else ""))
                    if len(s) > 1:
                        sub += " · " + t["t_prev"].format(at2=t["at2"].format(step=f"{s[-2][0]:,}", pct=pct(s[-2][0])), d=f"{s[-2][1] - q:+.4f}")
                tiles.append((f"{n} · {wl}" + (" · bpb" if sec == "bpb" else ""), f"{v:.4f}", sub, dl))
            lead = ""                                    # the lead sentence repeated the tiles (11-16 review): not shown
            heads += (f'<div data-ref="{sec}"{hid(si)}>{lead}<div class="tiles">'
                      + "".join(f'<div class="tile"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div>'
                                + (f'<div class="dl">{esc(dl[0])} <span>{esc(dl[1])}</span></div>' if dl else "") + f'<div class="s">{esc(s)}</div></div>'
                                for k, v, s, dl in tiles) + "</div></div>")
            gg = {}
            for k, n in GAP_KEYS:
                q = qref(suf, k, sec)
                g = [(it, v - q, v, q) for it, v in series(win, k, sec) if it >= GAP_FROM] if q is not None else []
                if g:
                    gg[n] = g
            if gg:
                allg = [x[1] for g in gg.values() for x in g] + [0.0]
                panels = "".join(_gap(esc, t, n, g, min(allg), max(allg), total, base_name, base_col) for n, g in gg.items())
                gaps += f'<div data-ref="{sec}"{hid(si)}><div class="gap-grid nomd">{panels}</div></div>'
            else:
                gaps += f'<div data-ref="{sec}"{hid(si)}><p class="note">{esc(t["no_win"].format(w=wl))}</p></div>'
        heads += "</div>"
        gaps += "</div>"

    # ---- the absolute curves: per window x metric x statistic one view; the references follow the tick boxes (rk_block)
    cell = lambda v: f"<td>{v:.4f}</td>" if v is not None else "<td>-</td>"
    its = sorted({it for w in res.values() for it in w})
    head = (f'<th class="l">{esc(t["ckpt"])}</th><th>{esc(t["prog"])}</th><th>model size</th><th>vocab size</th>'
            + "".join(f"<th>{esc(n)}</th>" for _, n in TABLE_COLS))
    txt = lambda v: f"<td>{esc(v)}</td>" if v else "<td>-</td>"
    charts, tables = "", ""
    for i, (win, wl, suf) in enumerate(WINS):
        charts += f'<div data-win="{wid[win]}"{hid(i)}>'
        for j, (k, n) in enumerate(cols):
            charts += f'<div data-metric="{k}"{hid(j)} style="max-width:960px;margin:0 auto"><div class="ct">{esc(n)} · {esc(t["win"].format(w=wl))}</div>'
            for si, (sec, _) in enumerate(STATS):        # val_loss / bpb (user 11-16: "prod iso-doc上面要有bpb的选项")
                charts += f'<div data-ref="{sec}"{hid(si)}>'
                pts = series(win, k, sec)
                if pts:
                    rv = [None if sec == "loss" and ch in OWN_TOK else ref_val(ch, q, suf, k, sec) for ch, q, _, _, _ in refs_on]   # OWN_TOK: bpb only
                    rv = [round(v, 6) if v is not None else None for v in rv]
                    refs = [(t[key], c, v, code(ch, q)) for (ch, q, key, c, _), v in zip(refs_on, rv) if v is not None]
                    charts += _legend(esc, [(SHORT, COL, False, None)] + [(f"{rn} {v:.4f}", c, True, rk) for rn, c, v, rk in refs])
                    ylab = t["ylab_bpb"] if sec == "bpb" else None
                    charts += rk_block(esc, "pi", rv, on0, lambda fit: _chart(esc, t, pts, refs, total, fit=fit, ylab=ylab), _hits(esc, t, pts, refs, total))
                else:
                    charts += f'<p class="note" style="margin:8px 0">{esc(t["no_win"].format(w=wl))}</p>'
                charts += "</div>"
            charts += "</div>"
        charts += "</div>"
        # ---- the tables: every row, latest first, not driven by the tick boxes (user 11-16: "数据表格可以不用根据checkbox变化"); a delta row
        # under the newest full checkpoint; model size / vocab size on the first row only
        tables += f'<div data-win="{wid[win]}"{hid(i)}>'
        for si, (sec, _) in enumerate(STATS):
            rows, first, drow = "", True, True
            # 11-17 (user "另外把仅有 val_v2 和 v3small测条目去掉"): a checkpoint that only has the in-between val_v2 / v3small eval (kind
            # "pooled", the rows labelled "仅 val_v2 / v3small") is not a row of the table; its points stay in the charts above
            in_table = lambda it: bool(res.get(win, {}).get(it)) and res[win][it]["kind"] != "pooled"
            nfold = sum(1 for it in its if it < FOLD and in_table(it))
            fold_s = f"{FOLD:,}" + ((f"（{pct(FOLD)}）" if lang == "zh" else f" ({pct(FOLD)})") if total else "")
            for it in sorted(its, reverse=True):
                d = res.get(win, {}).get(it)
                if not d or not in_table(it):
                    continue
                folded = it < FOLD
                if folded and nfold:                       # the toggle row, once, above the first folded checkpoint (rows are latest first)
                    lab_o = t["fold_open"].format(n=nfold, s=fold_s)
                    rows += (f'<tr><td colspan="{4 + len(TABLE_COLS)}" class="l"><button class="dm-fold-btn" data-open="{esc(lab_o)}" '
                             f'data-close="{esc(t["fold_close"])}" style="font:inherit;font-size:12px;cursor:pointer">{esc(lab_o)}</button></td></tr>')
                    nfold = 0
                tro = '<tr class="dm-fold" hidden>' if folded else "<tr>"
                lab = f'{SHORT} · {t["step"]} {it:,}' + (t["pooled_lab"] if d["kind"] == "pooled" else "")
                vals = [d["metrics"][k].get(sec) if k in d["metrics"] else None for k, _ in TABLE_COLS]
                rows += (f'{tro}<td class="l">{esc(lab)}</td><td>{pct(it) if total else "-"}</td>'
                         + txt(size if first else "") + txt(f"{VOCAB:,}" if first else "") + "".join(cell(v) for v in vals) + "</tr>")
                first = False
                if drow and d["kind"] != "pooled":
                    drow = False
                    qs = [qref(suf, k, sec) for k, _ in TABLE_COLS]
                    rows += (f'<tr class="drow{" dm-fold" if folded else ""}"{" hidden" if folded else ""}><td class="l">{esc(t["d_row"].format(step=f"{it:,}"))}</td><td>{pct(it) if total else ""}</td><td></td><td></td>'
                             + "".join(f"<td>{v - q:+.4f}</td>" if v is not None and q is not None else "<td>-</td>" for v, q in zip(vals, qs)) + "</tr>")
            first = True
            for ch, q, key, _, _ in refs_on:
                vals = [ref_val(ch, q, suf, k, sec) for k, _ in TABLE_COLS]
                if all(v is None for v in vals):
                    continue
                sz, tw = ref_size(ch, q)
                vc = (vocab(tw, q) if vocab else None) or VOCAB_OWN.get(ch)
                trc = ' class="sep"' if first else ""
                rows += (f'<tr{trc}><td class="l">{esc(t[key])}</td><td>{esc(t["ref_h"]) if first else ""}</td>'
                         + txt(sz) + txt(f"{vc:,}" if vc else "") + "".join(cell(v) for v in vals) + "</tr>")
                first = False
            tables += f'<div data-ref="{sec}"{hid(si)}><div style="overflow-x:auto"><table style="display:table;width:100%"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div></div>'
        tables += "</div>"
    # ---- one control bar for the whole tab (11-16 review: two unsynced toggle sets before); the page script scopes it to [data-sync]
    wtog = '<div class="rtoggle wtoggle">' + "".join(f'<button class="{"on" if i == 0 else ""}" data-w="{wid[win]}">{wl}</button>' for i, (win, wl, _) in enumerate(WINS)) + "</div>"
    stog = '<div class="rtoggle">' + "".join(f'<button class="{"on" if i == 0 else ""}" data-r="{sec}">{esc(n)}</button>' for i, (sec, n) in enumerate(STATS)) + "</div>"
    # order (user 11-16 18:0x "图永远放在最前面", 18:1x "overview 大字总结放在图的前面 表格还是放后面"): control bar, the overview tiles, the
    # curves, "实验列表" right under them (it drives the curves; 11-16 verify review), the gap grid, then the table and the notes
    out = PROD_FAM_CSS + RK_CSS + VIEW_CSS + f'<div data-sync="1"><div class="pi-ctl">{wtog}{stog}</div>' + heads
    out += f'<div class="msec"><h2 style="margin-top:8px">{esc(t["h_curve"])}</h2>'
    out += '<div class="toggle" style="position:static;margin:0 0 6px;flex-wrap:wrap">' + "".join(f'<button data-m="{k}" class="{"on" if j == 0 else ""}">{esc(n)}</button>' for j, (k, n) in enumerate(cols)) + "</div>"
    out += f'<div class="card nomd">{charts}</div></div>'   # md: the tables below carry every number (the chart tables were thinned)
    # ---- "实验列表": the production run (always drawn) and every reference, with its tick box, colour and size
    sample = lambda c: f'<i style="display:inline-block;width:12px;height:12px;border-radius:2px;background:var({c})"></i>'
    lrows = (f'<tr><td style="text-align:center"><input type="checkbox" checked disabled></td><td>{sample(COL)}</td>'
             f'<td class="l" style="white-space:nowrap">{esc(RUN)}</td><td class="l nt-d">{esc(size)}</td><td class="l nt-d">{esc(t["l_both"])}</td></tr>')
    for ch, q, key, c, on in refs_on:
        lrows += (f'<tr><td style="text-align:center"><input type="checkbox" class="pi-cb" value="{esc(code(ch, q))}"{" checked" if on else ""}></td>'
                  f'<td>{sample(c)}</td><td class="l" style="white-space:nowrap">{esc(t[key])}</td><td class="l nt-d">{esc(ref_size(ch, q)[0])}</td>'
                  f'<td class="l nt-d">{esc(t["l_bpb" if ch in OWN_TOK else "l_both"])}</td></tr>')
    out += (f'<h2>{esc(t["h_list"])}</h2><div class="card"><div style="overflow-x:auto"><table style="display:table;width:100%;font-size:12.5px"><thead><tr>'
            + "".join(f'<th class="l">{esc(x)}</th>' for x in t["l_head"]) + f"</tr></thead><tbody>{lrows}</tbody></table></div></div>")
    out += f'<h2>{esc(t["h_gap"])}</h2><p class="note">{esc(t["gap_note"])}</p><div class="card">{gaps}</div>'
    if il:
        out += (f'<h2>{esc(t["h_inloop"])}</h2><p class="note">{esc(t["inloop_note"])}</p>'
                f'<div class="card"><div style="max-width:960px;margin:0 auto">{_inloop_chart(esc, t, il, total)}</div></div>')
    out += f'<h2>{esc(t["h_table"])}</h2><div class="card">{tables}</div>'
    # ---- "59B checkpoint 平均": a chapter of its own, as in 59B downstream (user 11-17 "平均的 59B 在 59B val loss，也单独一个section吧"; rows
    # of the data table 11-17 morning, folded there for 20 minutes). The averages' doc evals (_merges): 8K val_v2 / v3small, val_loss only.
    # One group per wave (M97 = newest step 97,330), first the un-averaged checkpoint of that step, then the averages, best val_v2 first.
    mg = _merges()
    if mg:
        K2, K3 = "01_val_v2_overall_tw", "04_v3small_tw"
        own = lambda it, k: (((res.get(8192, {}).get(it) or {}).get("metrics") or {}).get(k) or {}).get("loss")
        b2, b3 = qref("", K2, "loss"), qref("", K3, "loss")
        dl = lambda v, q: f'<td class="b-d">{v - q:+.4f}</td>' if v is not None and q is not None else "<td>-</td>"
        arows = ""
        for wave in dict.fromkeys(x[0].split("_", 1)[0] for x in mg):
            grp = [x for x in mg if x[0].split("_", 1)[0] == wave]
            it_m = grp[0][1]
            p2, p3 = own(it_m, K2), own(it_m, K3)
            arows += (f'<tr class="b-cat"><td class="l" colspan="{len(t["avg_head"])}">{esc(t["avg_wave"].format(w=wave, s=f"{it_m:,}"))}</td></tr>'
                      f'<tr class="av-plain"><td class="l" style="white-space:nowrap">{esc(t["avg_plain"].format(s=f"{it_m:,}"))}</td>'
                      f'<td class="l nt-d">{esc(t["avg_plain_d"])}</td>' + cell(p2) + "<td></td>" + dl(p2, b2) + cell(p3) + "<td></td>" + dl(p3, b3) + "</tr>")
            for name, _, rk, dv in grp:
                v2, v3 = dv.get("val_v2_decon"), dv.get("v3small_decon")
                arows += (f'<tr><td class="l" style="white-space:nowrap">{esc(name)}</td><td class="l nt-d">{esc(MERGE_RECIPE[lang].get(rk, rk))}</td>'
                          + cell(v2) + dl(v2, p2) + dl(v2, b2) + cell(v3) + dl(v3, p3) + dl(v3, b3) + "</tr>")
        ahead = "".join(f'<th class="l">{esc(x)}</th>' if j < 2 else f"<th>{esc(x)}</th>" for j, x in enumerate(t["avg_head"]))
        out += (BCSS + AVG_CSS + f'<h2>{esc(t["h_avg"])}</h2><p class="note">{esc(t["avg_note"])}</p><div class="card"><div style="overflow-x:auto">'
                f'<table class="b-tab" style="display:table;width:100%"><thead><tr>{ahead}</tr></thead><tbody>{arows}</tbody></table></div></div>')
    out += notes_card + "</div>"
    return t["tab"], out
