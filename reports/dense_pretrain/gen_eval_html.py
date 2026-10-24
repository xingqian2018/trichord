#!/usr/bin/env python3
"""reports_xx/gen_eval_html.py - visual KD-vs-CE report (zh + en, self-contained HTML with inline SVG charts) from the per-checkpoint JSONs
of collect_doc_eval.py.

    python3 reports_xx/collect_doc_eval.py && python3 reports_xx/gen_eval_html.py
    -> reports_xx/qwen3_scaling_ladder_eval.html (Chinese), reports_xx/qwen3_scaling_ladder_eval_en.html (English)

Colours: the dataviz skill's reference palette (validated categorical order, blue sequential ramp, light/dark chart chrome).
"""
import glob
import html
import json
import math
import os
import re
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/Project/trichord/dense_pretrain_result"
RUNGS = ["q0", "q1", "q2", "q3", "q4", "q5"]
N_ACTIVE = {"q0": 48.0e6, "q1": 153.9e6, "q2": 355.5e6, "q3": 683.2e6, "q4": 1.17e9, "q5": 1.84e9}      # non-embedding active params
N_LABEL = {"q0": "48M", "q1": "154M", "q2": "356M", "q3": "683M", "q4": "1.17B", "q5": "1.84B"}
TOTAL = {"q0": "0.63B", "q1": "1.82B", "q2": "4.06B", "q3": "7.71B", "q4": "13.1B", "q5": "20.6B"}
TOKENS = {"q0": "9.6B", "q1": "30.8B", "q2": "71.1B", "q3": "136.6B", "q4": "233.6B", "q5": "368.0B"}
CE = "OPT_MUONH"
KD = {"0.7": "KD_MUONH_A0P7", "0.8": "KD_MUONH_A0P8", "0.9": "KD_MUONH_A0P9"}
ARMS = [("BASE", "AdamW", "CE", "—"), ("OPT_MUON", "Muon", "CE", "—"), ("OPT_MUONH", "MuonH", "CE", "—"),
        ("KD_MUONH_A0P7", "MuonH", "KD", "0.7"), ("KD_MUONH_A0P8", "MuonH", "KD", "0.8"), ("KD_MUONH_A0P9", "MuonH", "KD", "0.9")]
COLS = [("01_val_v2_overall_tw", "val_v2"), ("02_uncheatable_tw", "uncheatable"), ("03_codefresh_tw", "codefresh"),
        ("04_v3small_tw", "v3small"), ("04_private_internal_tw", "private_internal"), ("05_pile_test_tw", "pile_test"),
        ("05_pile_10k_tw", "pile_10k"), ("00_overall_tw", "overall_tw"), ("00_overall_macro", "overall_macro")]
HEAD = "00_overall_tw"
METRICS = [("00_overall_tw", "overall_tw"), ("00_overall_macro", "overall_macro"), ("01_val_v2_overall_tw", "val_v2"), ("02_uncheatable_tw", "uncheatable"), ("03_codefresh_tw", "codefresh"), ("04_private_internal_tw", "private_internal")]
# sequential blue ramp (reference palette), light -> dark
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]

T = {
 "zh": {
  "lang": "zh", "title": "Qwen3-MoE scaling ladder：KD 对 CE",
  "sub": "doc-isolated 评测（9 个 decon suite，含 private_internal），dpl 指标定义；数据生成于 {ts}",
  "nav": ["结论", "优化器（CE）", "KD（MuonH）", "热力图", "数据表", "说明"],
  "opt_h": "优化器对比：AdamW / Muon / MuonH（都是 CE）", "opt_note": "同一数据、同一 token 数，只换优化器（及其 LR 规则 / schedule）。横轴：激活参数（不含 embedding，对数坐标）。",
  "o2_h": "相对 AdamW 降低的 loss（越高越好）",
  "kd_h": "KD 对 CE：优化器固定为 MuonH", "kd_note": "CE 与 KD 只差 loss（α = 0 vs 0.7 / 0.8 / 0.9），优化器、数据顺序、layout 完全相同。",
  "k2_h": "CE − KD：各 α 降低的 loss（越高越好）",
  "adamw": "AdamW", "muon": "Muon", "vs_adamw": "Δloss (Muon − AdamW，越低越好)", "mean6": "六集平均",
  "v_h": "各种 KD loss 对比：同一 α 0.8、同一 MuonH", "v_note": "8 个 run 只差 KD 项里 teacher 分布的截断方式：α = 0.8、MuonH 优化器、seed 1、数据顺序全部相同；基线仍是 <b>CE</b>（MuonH，α = 0）；<b>B64</b>（top-64 + 剩余 tail 合成一个 bucket）就是 KD α 0.8 本身，作为一条线一起画出来，各变体看是否比 B64 更低。负数 = 比 CE 好。",
  "v_chart": "Δloss（KD 变体 − CE），越低越好", "v_table": "CE 的 loss，以及 B64 和各变体相对 CE 的差", "v_gate": "晋级线 −0.008",
  "vdesc": {"B64": "top-64 + bucket（= KD α 0.8）", "S64Y": "top-64 + 采样 tail + y exact", "S64": "top-64 + 采样 tail（4 draws）", "S128": "top-128 + 采样 tail", "B64Y": "top-64 bucket + y exact",
            "B128": "top-128 + bucket", "DD64": "top-64 + data draw", "DROP64": "top-64，丢掉 tail（对照）"},
  "lr_h": "MuonH 学习率测试：同一 MuonH、CE", "lr_note": "只改 MuonH 的峰值 LR（HyperP 规则给出的 LR × 0.5 / 0.7 / 1.4 / 2），其余相同；1× = 主线 MuonH（CE）。每个 rung 一条线；最低点就是这个 rung 上的最优 LR。",
  "lr_chart": "Δloss（该 LR − 1×），越低越好", "lr_table": "1× 的 loss，以及其他 LR 相对 1× 的差", "lr_x": "峰值 LR / HyperP 规则（对数坐标）",
  "find_v": "各种 KD loss（同一 α 0.8、MuonH）：采样 tail 的变体一直比 B64 好，但领先随规模缩小（overall_tw，q0 → q2：{sampled}）；bucket 类变体 ≈ 0（{bucket}）。",
  "find_lr": "MuonH 学习率（tsqw，抛物线拟合的最优倍数 m*，* = 落在扫描边界）：{best}。",
  "vn_title": "命名：字母 = tail 的处理方式 · 数字 = top-k 的大小 · 末尾 Y = 是否把 y 单独精确算", "vn_head": ["部分", "含义"],
  "vn_rows": [["B", "Bucket：top-k 以外的 tail 合成一个整体，只要求 student 在 tail 上的总概率对"],
              ["S", "Sampled：tail 用 cache 里预存的 4 个采样 token 来估计，结果无偏"],
              ["DD", "Data Draw：不用 teacher 的 tail，改用语料里真实的下一个 token 近似 tail"],
              ["DROP", "丢掉 tail：teacher 只在 top-k 内重新归一化（对照组）"],
              ["64 / 128", "精确计算的 top-k 有多少个 token"],
              ["Y", "y exact：当真实的下一个 token y 落在 top-k 以外时，把 y 从 tail 里拆出来单独精确算"]],
  "lri_h": "固定训练 token（9.6B），只加深模型", "lri_note": "q1 / q2 / q3 都只训 4,577 步（= q0 的 9.6B token）；对照 = 同预算的 1× run（runlong 的 (L, T) 实验）。<b>注意</b>：token 固定意味着模型越大越欠训练（TPP：q1 62、q2 27、q3 14，主线是 200），而训练越短最优 LR 通常越大，所以这里“越深越适合大 LR”混着欠训练的影响，<b>不能单独归因于深度</b>。这些 loss 不能和上图主线（TPP 200）比较。",
  "lri_chart": "Δloss（该 LR − 同预算 1×），越低越好",
  "g_metric": "本节固定只看 <b>{m}</b>（token-sqrt-weighted：全部 7 个集合，即 val_v2、uncheatable、codefresh、v3small、private_internal、pile_test、<b>pile_10k</b>，各自的 token 加权 loss 再按 √token 数加权平均；权重约 uncheatable 31%、pile_test 23%、val_v2 15%、codefresh 13%、pile_10k 8%、private_internal 5%、v3small 5%，介于 tw 和 macro 之间，避免最长或最短的集合主导；和 overall_tw / macro 不同，pile_10k 也算在内），不随上方的指标切换。",
  "g_h": "MuonH 学习率：每个格子找最优 LR（同一 MuonH、CE）", "g_note": "每个格子 = 一个固定的（模型大小，训练 token 数）。格子里按 HyperP 公式算出的 LR 的 0.5 / 0.7 / 1 / 1.2 / 1.4 / 2 倍各训一个模型，纵轴 = 相对该格 1× 的 loss 差（越低越好）；格子里的大数字 = 拟合出的最优倍数 <b>m*</b>（m* &gt; 1 表示公式算低了），下面是对应的绝对 LR <b>η*</b>（括号里是该格 1× 的 LR，即 HyperP 公式按层数和训练 token 数算出的 LR）和最优点比 1× 好多少；主线 TPP 200 的格子沿对角线排列；<b>鼠标移到格子上显示该格的 LR 测试曲线</b>（虚线 = 在 log(倍数) 上拟合的抛物线，橙线 = m*）。同一列看深度、q0 一行看训练量。<b>不同格子的训练量不同，只在格子内部比较。</b>空白 = 不在设计里。",
  "g_cols": {"9.6": "9.6B token", "30": "~30B token", "71": "71.1B token", "86": "86.4B token", "137": "136.6B token"}, "g_layers": "层",
  "g_kind": {"main": "主线 TPP {tpp}", "main_x": "主线 TPP {tpp}（不在 TODO 3 里）", "tpp": "{tok}，TPP {tpp}", "it": "{tok}，TPP {tpp}（IT4577）", "it2": "{tok}，TPP {tpp}"},
  "g_pending": "待测", "g_fewpts": "点数不足", "g_edge": "（边界）", "g_npts": "{n} 个点",
  "sd_h": "Seed 噪声：同一配置换 seed（doc-isolated）", "sd_note": "同一模型、同一 recipe，只换 seed（seed k 的数据顺序在不同 arm 之间相同，所以 Muon − AdamW 可以按 seed 配对）。<b>极差</b> = 三个 seed 最大减最小，<b>sd</b> = 标准差。用来判断其他小节里的差值是否超出噪声。",
  "sd_pair": "Muon − AdamW（按 seed 配对）", "sd_range": "极差",
  "tab_doc": "doc-isolated（xingqian的评测）", "tab_packed": "packed（Boxiang 训练时验证）",
  "p_notes": ["数据来自 Boxiang 的 <code>results/&lt;Change ID&gt;/scores.json</code>：训练结束时自带的 <b>packed</b> 验证（文档首尾拼接、不加 attention mask），我们不重新测。",
              "只有他的 6 个合并验证集（val_v2 不拆 split，没有 private_internal）；“六集平均”= 这 6 个的简单平均（Boxiang 的定义）。",
              "packed 比 doc-isolated 高约 0.013–0.016，<b>两个 tab 的绝对值不能互相比较</b>，只在同一 tab 内比差值。"],
  "tiles": ["最大 rung 上 KD 降低的 overall loss", "最大 rung 的等效 CE 模型规模", "private_internal 上 KD 降低的 loss", "三个 α 之间的最大差"],
  "tile_sub": ["{rung}，α 0.8，相对 CE（MuonH）", "KD {rung} ≈ {mult}× 激活参数的 CE 模型", "{rung}，α 0.8；训练语料之外的私有集", "同一 rung、同一指标；噪声约 0.008"],
  "find_h": "结论",
  "finds": [
   "KD 在<b>所有 rung、所有验证集、所有 α</b> 上都比 CE 低（{n_cells} 个格子全部为负）。",
   "增益随规模变大：overall_tw 上 KD 降低的 loss 为 {gains}（q0 → q{last}）。",
   "等效规模：KD 模型的 loss 相当于 <b>{mults}</b> 倍激活参数的 CE 模型（在 CE 的 loss–log N 曲线上插值），倍数也在随规模变大。",
   "增益最大的是 uncheatable / codefresh，最小的是 val_v2 和 private_internal；private_internal 是训练语料之外的私有集，KD 在上面同样稳定变好。",
   "α 0.7 / 0.8 / 0.9 基本分不开（最大差 {amax:.3f}），选 α 不影响结论。",
   "对比优化器（都是 CE）：MuonH 相对 AdamW 降低的 overall_tw loss 随规模<b>缩小</b>（{optg}），而 KD 的增益随规模<b>变大</b>——KD 带来的收益比换优化器更能保持到大模型。"],
  "c1_h": "overall loss 随模型规模变化", "c1_note": "横轴：激活参数（不含 embedding，对数坐标）；纵轴：overall_tw loss（越低越好）。",
  "c2_h": "KD 降低的 loss 随规模变化", "c2_note": "纵轴 = CE − KD 的 overall_tw（越高表示 KD 帮助越大）。",
  "c3_h": "各验证集上 KD 降低的 loss（α 0.8）", "c3_note": "格子 = CE − KD；颜色越深表示 KD 帮助越大。· = 还没有结果。",
  "t1_h": "全部结果（loss，nats/token）", "t2_h": "KD − CE（负数 = KD 更好）",
  "cols_head": ["rung", "优化器", "loss type", "α"], "pending": "· = 还没有结果",
  "notes_h": "说明",
  "notes": [
   "模型：Boxiang 的 ladder 最终 checkpoint，我们自己重新评测（document-isolated：每个文档单独成一个样本）。",
   "overall_tw = val_v2、uncheatable、codefresh、v3small、private_internal、pile_test 六类所有 split 合起来按 token 加权；overall_macro = 六类的平均；pile_10k 不进 overall。单位 nats/token。",
   "CE = 只有交叉熵；KD = (1 − α)·CE + α·KL(teacher top-64 + tail ‖ student)，teacher 为 MiMo-V2.6-Pro-RL。KD 的 CE 对照是同一 recipe 的 MuonH 模型（数据顺序、layout 相同，α = 0）。",
   "已核对：除 private_internal 外，各 suite 与 Boxiang 的 doc-isolated 结果一致（差 ≤ 0.00005）。",
   "每个模型的完整结果（含 115 个 split）：trichord/dense_pretrain_result/&lt;Change ID&gt;_&lt;rung&gt;.json。"],
  "ce": "MuonH", "kd": "KD α 0.8", "alpha": "KD α {a}", "size": "激活参数", "loss": "loss", "gain": "KD 降低的 loss",
  "table_view": "数据见下方表格。",
  "tok_h": "各验证集的 token 数", "tok_note": "overall_tw 由这六类按 token 加权，所以 token 多的集合（uncheatable、pile_test）权重大；overall_macro 则六类等权。pile_10k（{p10k} token）不进 overall，不画在图里。每个模型评测的 token 完全相同。",
  "tok_cols": ["验证集", "split 数", "token", "占比"], "tok_total": "合计"},
 "en": {
  "lang": "en", "title": "Qwen3-MoE scaling ladder: KD vs CE",
  "sub": "Document-isolated eval (9 decon suites incl. private_internal), dpl metric definitions; generated {ts}",
  "nav": ["Findings", "Optimizers (CE)", "KD (MuonH)", "Heatmap", "Tables", "Notes"],
  "opt_h": "Optimizers: AdamW / Muon / MuonH (all CE)", "opt_note": "Same data and token count; only the optimizer (with its LR rule / schedule) changes. x: active non-embedding parameters (log scale).",
  "o2_h": "Loss reduction vs AdamW (higher is better)",
  "kd_h": "KD vs CE with the optimizer fixed to MuonH", "kd_note": "CE and KD differ only in the loss (α = 0 vs 0.7 / 0.8 / 0.9); optimizer, data order and layout are identical.",
  "k2_h": "CE − KD: loss reduction per α (higher is better)",
  "adamw": "AdamW", "muon": "Muon", "vs_adamw": "Δloss (Muon − AdamW, lower is better)", "mean6": "mean of 6",
  "v_h": "KD loss variants compared: same α 0.8, same MuonH", "v_note": "The 8 runs differ only in how the teacher distribution is truncated in the KD term: α = 0.8, MuonH optimizer, seed 1 and data order are identical; the baseline is still <b>CE</b> (MuonH, α = 0); <b>B64</b> (top-64 + the rest of the tail as one bucket) is KD α 0.8 itself and is drawn as a line too, so each variant can be read against it. Negative = better than CE.",
  "v_chart": "Δloss (KD variant − CE), lower is better", "v_table": "CE loss, and B64 and each variant's difference from CE", "v_gate": "gate −0.008",
  "vdesc": {"B64": "top-64 + bucket (= KD α 0.8)", "S64Y": "top-64 + sampled tail + y exact", "S64": "top-64 + sampled tail (4 draws)", "S128": "top-128 + sampled tail", "B64Y": "top-64 bucket + y exact",
            "B128": "top-128 + bucket", "DD64": "top-64 + data draw", "DROP64": "top-64, tail dropped (control)"},
  "lr_h": "MuonH learning-rate test: same MuonH, CE", "lr_note": "Only MuonH's peak LR changes (the HyperP law's LR × 0.5 / 0.7 / 1.4 / 2); everything else is identical; 1× = the main MuonH (CE) run. One line per rung; its lowest point is that rung's best LR.",
  "lr_chart": "Δloss (this LR − 1×), lower is better", "lr_table": "1× loss, and every other LR's difference from 1×", "lr_x": "peak LR / HyperP law (log scale)",
  "find_v": "KD loss variants (same α 0.8, MuonH): the sampled-tail variants beat B64 consistently, but the lead shrinks with scale (overall_tw, q0 → q2: {sampled}); the bucket variants are ≈ 0 ({bucket}).",
  "find_lr": "MuonH learning rate (tsqw, parabola-fitted best multiple m*, * = at the edge of the sweep): {best}.",
  "vn_title": "Naming: letters = tail treatment · number = top-k size · trailing Y = y computed exactly", "vn_head": ["Part", "Meaning"],
  "vn_rows": [["B", "Bucket: the tail outside the top-k is merged into one outcome; only the student's total tail mass has to match"],
              ["S", "Sampled: the tail is estimated from the 4 tokens pre-sampled in the cache (unbiased)"],
              ["DD", "Data Draw: instead of the teacher's tail, the corpus's actual next token approximates it"],
              ["DROP", "Tail dropped: the teacher is renormalised over the top-k only (control)"],
              ["64 / 128", "how many top-k tokens are computed exactly"],
              ["Y", "y exact: when the actual next token y falls outside the top-k, y is taken out of the tail and computed exactly"]],
  "lri_h": "Fixed training tokens (9.6B), depth only", "lri_note": "q1 / q2 / q3 all trained 4,577 iterations (= q0's 9.6B tokens); reference = the 1× run of the same budget (Runlong's (L, T) design). <b>Caveat</b>: fixed tokens means the bigger the model, the more under-trained (TPP: q1 62, q2 27, q3 14 vs 200 on the main ladder), and shorter training usually favours a larger LR - so a larger best LR at depth here is confounded with under-training and <b>cannot be attributed to depth alone</b>. These losses are not comparable with the main TPP 200 runs above.",
  "lri_chart": "Δloss (this LR − 1× of the same budget), lower is better",
  "g_metric": "This section always uses <b>{m}</b> (token-sqrt-weighted: all 7 collections - val_v2, uncheatable, codefresh, v3small, private_internal, pile_test, <b>pile_10k</b> - each at its token-weighted loss, averaged with weights ∝ √(scored tokens); weights ≈ uncheatable 31 %, pile_test 23 %, val_v2 15 %, codefresh 13 %, pile_10k 8 %, private_internal 5 %, v3small 5 % - between tw and macro, so neither the longest nor the shortest collection dominates; unlike overall_tw / macro, pile_10k is included); it does not follow the metric toggle above.",
  "g_h": "MuonH learning rate: the best LR in each cell (same MuonH, CE)", "g_note": "Each cell = one fixed (model size, training tokens). In a cell, one model per 0.5 / 0.7 / 1 / 1.2 / 1.4 / 2 × the HyperP-law LR; y = loss difference to the cell's 1× (lower is better); the big number = the fitted best multiple <b>m*</b> (m* &gt; 1 means the law is too low), below it the absolute LR <b>η*</b> (in parentheses the cell's 1× LR, i.e. the HyperP-law LR for its depth and training tokens) and how much the optimum beats 1×; the main-ladder TPP 200 cells run along the diagonal; <b>hover a cell to see its LR sweep curve</b> (dashed = parabola in log(multiple), orange = m*). Read a column for depth, the q0 row for training length. <b>Cells have different budgets: compare only within a cell.</b> Blank = not in the design.",
  "g_cols": {"9.6": "9.6B tokens", "30": "~30B tokens", "71": "71.1B tokens", "86": "86.4B tokens", "137": "136.6B tokens"}, "g_layers": "layers",
  "g_kind": {"main": "main ladder, TPP {tpp}", "main_x": "main ladder, TPP {tpp} (not in TODO 3)", "tpp": "{tok}, TPP {tpp}", "it": "{tok}, TPP {tpp} (IT4577)", "it2": "{tok}, TPP {tpp}"},
  "g_pending": "pending", "g_fewpts": "too few points", "g_edge": "(edge)", "g_npts": "{n} points",
  "sd_h": "Seed noise: the same configuration with different seeds (doc-isolated)", "sd_note": "Same model and recipe, only the seed changes (seed k has the same data order across arms, so Muon − AdamW is paired by seed). <b>Range</b> = max − min over the three seeds, <b>sd</b> = standard deviation. Use it to judge whether the differences in the other sections exceed the noise.",
  "sd_pair": "Muon − AdamW (paired by seed)", "sd_range": "range",
  "tab_doc": "doc-isolated (xingqian's eval)", "tab_packed": "packed (Boxiang's training-time validation)",
  "p_notes": ["From Boxiang's <code>results/&lt;Change ID&gt;/scores.json</code>: the <b>packed</b> final validation of each training run (documents concatenated, no attention mask); we do not re-measure it.",
              "Only his 6 pooled sets (val_v2 not split, no private_internal); “mean of 6” = the simple mean of those 6 (his definition).",
              "Packed is ~0.013–0.016 above doc-isolated: <b>do not compare absolute values across the two tabs</b>, only deltas within a tab."],
  "tiles": ["Overall loss reduction from KD at the largest rung", "Equivalent CE model size at the largest rung", "KD loss reduction on private_internal", "Largest gap between the three alphas"],
  "tile_sub": ["{rung}, α 0.8, vs CE (MuonH)", "KD {rung} ≈ a CE model with {mult}× the active params", "{rung}, α 0.8; a private set outside the training corpus", "same rung and metric; noise ≈ 0.008"],
  "find_h": "Findings",
  "finds": [
   "KD beats CE at <b>every rung, on every validation set, for every α</b> (all {n_cells} cells negative).",
   "The gain grows with scale: KD lowers overall_tw loss by {gains} (q0 → q{last}).",
   "Equivalent size: KD models match CE models with <b>{mults}</b>× the active parameters (interpolated on CE's loss vs log N curve), and the multiplier grows with scale.",
   "The gain is largest on uncheatable / codefresh and smallest on val_v2 and private_internal; private_internal is a private set outside the training corpus, and KD improves it consistently too.",
   "α 0.7 / 0.8 / 0.9 are indistinguishable (largest gap {amax:.3f}); the choice of α does not change the conclusion.",
   "Optimizers (all CE): MuonH's overall_tw gain over AdamW <b>shrinks</b> with scale ({optg}), while KD's gain <b>grows</b> - the KD benefit holds up at scale better than the optimizer change."],
  "c1_h": "Overall loss vs model size", "c1_note": "x: active non-embedding parameters (log scale); y: overall_tw loss (lower is better).",
  "c2_h": "KD loss reduction vs model size", "c2_note": "y = CE − KD on overall_tw (higher = KD helps more).",
  "c3_h": "KD loss reduction per validation set (α 0.8)", "c3_note": "cell = CE − KD; darker = KD helps more. · = no result yet.",
  "t1_h": "All results (loss, nats/token)", "t2_h": "KD − CE (negative = KD better)",
  "cols_head": ["rung", "optimizer", "loss type", "α"], "pending": "· = no result yet",
  "notes_h": "Notes",
  "notes": [
   "Models: the final checkpoints of Boxiang's ladder, re-evaluated by us (document-isolated: every document scored as its own sample).",
   "overall_tw pools every split of val_v2, uncheatable, codefresh, v3small, private_internal and pile_test, token-weighted; overall_macro = mean of those six; pile_10k is reported apart. Units: nats/token.",
   "CE = cross-entropy only; KD = (1 − α)·CE + α·KL(teacher top-64 + tail ‖ student), teacher MiMo-V2.6-Pro-RL. KD's CE reference is the MuonH model of the same recipe (same data order and layout, α = 0).",
   "Checked: every suite except private_internal matches Boxiang's doc-isolated results (≤ 0.00005).",
   "Full per-model results (incl. 115 splits): trichord/dense_pretrain_result/&lt;Change ID&gt;_&lt;rung&gt;.json."],
  "ce": "MuonH", "kd": "KD α 0.8", "alpha": "KD α {a}", "size": "active params", "loss": "loss", "gain": "KD loss reduction",
  "table_view": "Data in the tables below.",
  "tok_h": "Tokens per validation set", "tok_note": "overall_tw weights these six sets by tokens, so the large ones (uncheatable, pile_test) dominate it; overall_macro weights the six equally. pile_10k ({p10k} tokens) is not in overall and not drawn. Every model is scored on exactly the same tokens.",
  "tok_cols": ["validation set", "splits", "tokens", "share"], "tok_total": "total"},
}

CSS = """
.viz-root{color-scheme:light;--page:#f9f9f7;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--muted:#898781;
 --grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#006300;
 --o-adamw:#6250d6;--o-muon:#1baf7a;--a07:#f4a27f;--a08:#eb6834;--a09:#b4471d;
 --v1:#2a78d6;--v2:#1baf7a;--v3:#eda100;--v4:#e87ba4;--v5:#008300;--v6:#6250d6;--v7:#e34948;
 --c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--c4:#eda100;--c5:#e87ba4;--c6:#008300}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;
 --text-primary:#f0efec;--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#0ca30c;
 --o-adamw:#9085e9;--o-muon:#199e70;--a07:#f0a47f;--a08:#d95926;--a09:#a8461f;
 --v1:#3987e5;--v2:#199e70;--v3:#c98500;--v4:#d55181;--v5:#008300;--v6:#9085e9;--v7:#e66767;
 --c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#c98500;--c5:#d55181;--c6:#008300}}
.pie{display:flex;flex-wrap:wrap;align-items:center;gap:24px}.pie td i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.card.vn{max-width:720px;padding:8px 12px;margin-left:auto;margin-right:auto}.vn table{margin:0 auto;width:fit-content}.vn .ct{text-align:center}.vn .ct{font-size:12px;font-weight:500;color:var(--text-secondary)}.vn table{font-size:11.5px}.vn td,.vn th{padding:2px 8px}
body{margin:0;background:var(--page)}
.viz-root{font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text-primary);background:var(--page)}
nav{position:sticky;top:0;z-index:2;background:var(--page);border-bottom:1px solid var(--border);padding:8px 28px;font-size:13px}
nav a{color:var(--text-secondary);text-decoration:none;margin-right:16px}nav a:hover{color:var(--text-primary)}
main{max-width:1440px;margin:0 auto;padding:20px 28px 64px}
h1{font-size:24px;margin:8px 0 2px}.sub{color:var(--text-secondary);margin:0 0 20px}
h2{font-size:18px;margin:34px 0 6px}.note{color:var(--text-secondary);font-size:13px;margin:0 0 10px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile .k{color:var(--text-secondary);font-size:13px}.tile .v{font-size:30px;font-weight:600;margin:4px 0}.tile .s{color:var(--muted);font-size:12px}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin:10px 0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(600px,1fr));gap:12px}.card.one{max-width:880px;margin-left:auto;margin-right:auto}
ul.f{margin:6px 0 0 0;padding-left:20px}ul.f li{margin:5px 0}
svg text{fill:var(--text-secondary);font-size:11px;font-variant-numeric:tabular-nums}svg .lbl{fill:var(--text-primary);font-size:12px}
.legend{display:flex;justify-content:center;gap:16px;font-size:12px;color:var(--text-secondary);margin:2px 0 6px}.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
table{border-collapse:collapse;font-size:12.5px;display:block;overflow-x:auto;max-width:100%}
th,td{border-bottom:1px solid var(--grid);padding:5px 9px;white-space:nowrap;text-align:right;font-variant-numeric:tabular-nums}
th{color:var(--text-secondary);font-weight:600;border-bottom:1px solid var(--axis)}td.l,th.l{text-align:left}
tr.kd td{background:color-mix(in srgb,var(--s2) 6%,transparent)}tr.sep td{border-top:1px solid var(--axis)}
td.neg{color:var(--good)}
#tip{position:fixed;pointer-events:none;background:var(--surface-1);color:var(--text-primary);border:1px solid var(--border);border-radius:6px;
 padding:5px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.15);display:none;z-index:9;white-space:nowrap}
.hit{fill:transparent;cursor:default}
.toggle{position:sticky;top:38px;z-index:1;margin:22px 0 0;display:flex;gap:6px}.toggle button{font:inherit;font-size:13px;padding:4px 12px;border-radius:999px;border:1px solid var(--border);background:var(--surface-1);color:var(--text-secondary);cursor:pointer}.toggle button.on{background:var(--text-primary);color:var(--surface-1)}
.ct{font-weight:600;font-size:13px;margin-bottom:2px}
.lrgrid{display:table;border-collapse:separate;border-spacing:6px}.lrgrid td,.lrgrid th{border:none;white-space:normal;vertical-align:top}.lrgrid thead th{text-align:center;font-size:13px}.lrgrid th.grow{text-align:left;font-size:14px;color:var(--text-primary);padding-top:30px}.lrgrid th.grow span{font-size:11px;color:var(--muted);font-weight:400}.lrgrid td.gc{width:180px;height:112px;position:relative;background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:8px 10px;cursor:default}.lrgrid td.gc:hover{border-color:var(--s1)}.mbig{font-size:26px;font-weight:600;margin:4px 0 0}.mbig small{font-size:11px;font-weight:400;color:var(--muted)}.mbig.muted{color:var(--muted)}.msub{font-size:11px;color:var(--muted)}.mlr{font-size:12px;margin:1px 0}.mlr span{color:var(--muted);font-size:11px}.pend{flex-direction:column}.pend span{font-size:11px}.lrgrid .pop{display:none;position:absolute;left:50%;top:100%;transform:translateX(-50%);z-index:5;width:300px;background:var(--surface-1);border:1px solid var(--border);border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,.18);padding:6px}.lrgrid td.gc:hover .pop{display:block}.lrgrid td.gc:hover{z-index:20}.lrgrid td.gc.up .pop{top:auto;bottom:100%}.lrgrid td.gx{width:180px;height:112px;background:repeating-linear-gradient(135deg,transparent 0 6px,var(--grid) 6px 7px);border-radius:8px;opacity:.6}.glab{font-size:11px;color:var(--text-secondary)}.mstar{font-size:12.5px;margin:2px 0}.mstar span{float:right;color:var(--muted);font-size:11px}.mstar.muted,.pend{color:var(--muted);font-size:12px}.pend{height:80px;display:flex;align-items:center;justify-content:center}
nav .tab{font:inherit;font-size:13px;padding:5px 14px;margin-right:6px;border-radius:999px;border:1px solid var(--border);background:var(--surface-1);color:var(--text-secondary);cursor:pointer}nav .tab.on{background:var(--text-primary);color:var(--surface-1)}
"""
JS = """
document.querySelectorAll('.toggle button').forEach(b=>b.addEventListener('click',()=>{const m=b.dataset.m,pane=b.closest('.pane');
 pane.querySelectorAll('.toggle button').forEach(x=>x.classList.toggle('on',x===b));
 pane.querySelectorAll('[data-metric]').forEach(d=>d.hidden=d.dataset.metric!==m);}));
document.querySelectorAll('nav .tab').forEach(b=>b.addEventListener('click',()=>{
 document.querySelectorAll('nav .tab').forEach(x=>x.classList.toggle('on',x===b));
 document.querySelectorAll('.pane').forEach(p=>p.hidden=p.dataset.pane!==b.dataset.tab);window.scrollTo(0,0);}));
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(e=>{
 e.addEventListener('mousemove',ev=>{tip.innerHTML=e.dataset.tip;tip.style.display='block';tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px'});
 e.addEventListener('mouseleave',()=>{tip.style.display='none'});});
"""


def load():
    R = {}
    for p in glob.glob(os.path.join(RES, "*.json")):
        d = json.load(open(p))
        R[(d["change"], d["rung"])] = d["metrics"]
    return R


def load_tokens():
    """{set name: (tokens, splits)} of the doc-isolated eval, from one checkpoint JSON (every checkpoint scores the same tokens).
    val_v2's three suites (cc / python / arxiv) are pooled into one set, as in overall_tw."""
    d = json.load(open(sorted(glob.glob(os.path.join(RES, "*.json")))[0]))
    out = {}
    for name, s in d["suites"].items():
        k = "val_v2" if name.startswith("cosmos_val_v2") else name.split("_decon")[0]
        tok, n = out.get(k, (0, 0))
        out[k] = (tok + s["tokens"], n + len(s["splits"]))
    return out


def fmt_tok(n):
    return f"{n / 1e6:.1f}M"


def token_pie(t):
    """Donut of the six overall_tw sets' token counts (categorical slots 1-6 in fixed order) + the same numbers as a table."""
    tok = load_tokens()
    sets = [(n, tok[n]) for _, n in COLS[:7] if n in tok and n != "pile_10k"]
    total = sum(v for _, (v, _) in sets)
    r1, r0, c = 110, 66, 120
    o = [f'<svg viewBox="0 0 {2*c} {2*c}" width="{2*c}" role="img">']
    a = -math.pi / 2
    for i, (n, (v, ns)) in enumerate(sets):
        b = a + 2 * math.pi * v / total
        big = int(b - a > math.pi)
        p = lambda r, x: (c + r * math.cos(x), c + r * math.sin(x))
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p(r1, a), p(r1, b), p(r0, b), p(r0, a)
        tip = f"<b>{esc(n)}</b><br>{fmt_tok(v)} tokens · {v / total:.1%}<br>{ns} split"
        o.append(f'<path d="M{x1:.2f},{y1:.2f} A{r1},{r1} 0 {big} 1 {x2:.2f},{y2:.2f} L{x3:.2f},{y3:.2f} A{r0},{r0} 0 {big} 0 {x4:.2f},{y4:.2f} Z" '
                 f'fill="var(--c{i+1})" stroke="var(--surface-1)" stroke-width="2" data-tip="{esc(tip)}"/>')
        a = b
    o.append(f'<text x="{c}" y="{c-2}" text-anchor="middle" class="lbl" style="font-size:20px;font-weight:600">{fmt_tok(total)}</text>'
             f'<text x="{c}" y="{c+16}" text-anchor="middle">tokens</text></svg>')
    rows = "".join(f'<tr><td class="l"><i style="background:var(--c{i+1})"></i>{esc(n)}</td><td>{ns}</td><td>{fmt_tok(v)}</td><td>{v / total:.1%}</td></tr>'
                   for i, (n, (v, ns)) in enumerate(sets))
    rows += (f'<tr class="sep"><td class="l">{esc(t["tok_total"])}</td><td>{sum(ns for _, (_, ns) in sets)}</td>'
             f'<td>{fmt_tok(total)}</td><td>100%</td></tr>')
    head = "".join(f'<th{" class=\"l\"" if j == 0 else ""}>{esc(h)}</th>' for j, h in enumerate(t["tok_cols"]))
    note = t["tok_note"].format(p10k=fmt_tok(tok["pile_10k"][0]) if "pile_10k" in tok else "·")
    return (f'<h2>{esc(t["tok_h"])}</h2><p class="note">{esc(note)}</p>'
            f'<div class="card pie">{"".join(o)}<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')


def val(R, ch, q, k=HEAD):
    m = R.get((ch, q))
    return m[k]["loss"] if m and k in m else None


def equiv_mult(R, q):
    """N_eq / N of KD(α0.8) at rung q: interpolate CE's loss vs log N piecewise-linearly (no extrapolation)."""
    kd = val(R, KD["0.8"], q)
    pts = [(math.log(N_ACTIVE[r]), val(R, CE, r)) for r in RUNGS if val(R, CE, r) is not None]
    if kd is None or len(pts) < 2:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y1 <= kd <= y0:
            x = x0 + (y0 - kd) / (y0 - y1) * (x1 - x0)
            return math.exp(x) / N_ACTIVE[q]
    return None


def esc(s):
    return html.escape(str(s), quote=True)


def line_chart(series, xs, ylab, fmt, w=760, h=360, y0=None, labels=True):
    """series = [(name, colour var, [(x, y, tip)], dashed)] on a log-x axis."""
    ml, mr, mt, mb = 52, 96, 12, 38
    allx = [p[0] for _, _, pts, _ in series for p in pts] or [1, 2]
    ally = [p[1] for _, _, pts, _ in series for p in pts] or [0, 1]
    lx0, lx1 = math.log(min(xs.values())) - 0.15, math.log(max(xs.values())) + 0.15
    ylo, yhi = min(ally), max(ally)
    if y0 is not None:
        ylo = min(ylo, y0)
    pad = (yhi - ylo) * 0.08 or 0.01
    ylo, yhi = ylo - (0 if y0 is not None else pad), yhi + pad
    X = lambda x: ml + (math.log(x) - lx0) / (lx1 - lx0) * (w - ml - mr)
    Y = lambda y: mt + (yhi - y) / (yhi - ylo) * (h - mt - mb)
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for i in range(5):                                   # recessive grid + y ticks
        yv = ylo + (yhi - ylo) * i / 4
        o.append(f'<line x1="{ml}" x2="{w-mr}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" stroke="var(--grid)" stroke-width="1"/>')
        o.append(f'<text x="{ml-6}" y="{Y(yv)+4:.1f}" text-anchor="end">{fmt(yv)}</text>')
    for q, x in xs.items():
        o.append(f'<text x="{X(x):.1f}" y="{h-mb+16}" text-anchor="middle">{q}</text>')
        o.append(f'<text x="{X(x):.1f}" y="{h-mb+29}" text-anchor="middle" style="fill:var(--muted)">{N_LABEL[q]}</text>')
    o.append(f'<line x1="{ml}" x2="{w-mr}" y1="{h-mb}" y2="{h-mb}" stroke="var(--axis)"/>')
    o.append(f'<text x="12" y="{(h-mb+mt)/2:.0f}" transform="rotate(-90 12 {(h-mb+mt)/2:.0f})" text-anchor="middle">{esc(ylab)}</text>')
    for name, col, pts, dashed in series:
        if not pts:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y, _) in enumerate(pts))
        o.append(f'<path d="{d}" fill="none" stroke="var({col})" stroke-width="2"{" stroke-dasharray=\"5 4\"" if dashed else ""} stroke-linejoin="round"/>')
        for x, y, t in pts:
            o.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="var({col})" stroke="var(--surface-1)" stroke-width="2"/>')
        lx, ly, _ = pts[-1]                              # direct label at the line end (off when lines overlap; the legend names them)
        if labels:
            o.append(f'<text class="lbl" x="{X(lx)+9:.1f}" y="{Y(ly)+4:.1f}">{esc(name)}</text>')
    qx = sorted((x, q) for q, x in xs.items())            # one hover band per rung: tooltip lists every series at that x
    for i, (x, q) in enumerate(qx):
        a = X(x) - (X(x) - X(qx[i - 1][0])) / 2 if i else ml
        b = X(x) + (X(qx[i + 1][0]) - X(x)) / 2 if i + 1 < len(qx) else w - mr
        rows = [(n, col, y) for n, col, pts, _ in series for px, y, _ in pts if px == x]
        if not rows:
            continue
        tip = f"<b>{q} · {N_LABEL[q]}</b>" + "".join(
            f'<br><span style="color:var({col})">●</span> {esc(n)}: {y:.4f}' for n, col, y in sorted(rows, key=lambda r: r[2], reverse=True))   # largest on top
        o.append(f'<rect class="hit" x="{a:.1f}" y="{mt}" width="{b-a:.1f}" height="{h-mt-mb}" data-tip="{esc(tip)}"/>')
    o.append("</svg>")
    return "".join(o)


def legend(items):
    return '<div class="legend">' + "".join(f'<span><i style="background:var({c})"></i>{esc(n)}</span>' for n, c in items) + "</div>"


def heatmap(R, t):
    rows = list(COLS)
    qs = [q for q in RUNGS if val(R, CE, q) is not None]
    vals = [val(R, CE, q, k) - val(R, KD["0.8"], q, k) for k, _ in rows for q in qs if val(R, KD["0.8"], q, k) is not None]
    vmax = max(vals) if vals else 1
    cw, ch, ml, mt = 92, 30, 130, 26
    w, h = ml + cw * len(qs) + 4, mt + ch * len(rows) + 4
    o = [f'<svg viewBox="0 0 {w} {h}" width="{w}" role="img">']
    for j, q in enumerate(qs):
        o.append(f'<text x="{ml + j*cw + cw/2:.0f}" y="16" text-anchor="middle">{q} · {N_LABEL[q]}</text>')
    for i, (k, name) in enumerate(rows):
        y = mt + i * ch
        o.append(f'<text x="{ml-8}" y="{y + ch/2 + 4:.0f}" text-anchor="end" class="lbl">{esc(name)}</text>')
        for j, q in enumerate(qs):
            x = ml + j * cw
            a, b = val(R, CE, q, k), val(R, KD["0.8"], q, k)
            if b is None:
                o.append(f'<rect x="{x+1}" y="{y+1}" width="{cw-2}" height="{ch-2}" rx="4" fill="var(--grid)"/>'
                         f'<text x="{x+cw/2:.0f}" y="{y+ch/2+4:.0f}" text-anchor="middle">·</text>')
                continue
            g = a - b
            step = min(len(BLUE) - 1, max(0, round(g / vmax * (len(BLUE) - 1))))
            ink = "#ffffff" if step >= 6 else "#0b0b0b"
            tipt = f"{esc(name)} · {q}<br>CE {a:.4f} → KD {b:.4f}<br>{esc(t['gain'])} {g:.4f}"
            o.append(f'<rect x="{x+1}" y="{y+1}" width="{cw-2}" height="{ch-2}" rx="4" fill="{BLUE[step]}" data-tip="{tipt}"/>'
                     f'<text x="{x+cw/2:.0f}" y="{y+ch/2+4:.0f}" text-anchor="middle" style="fill:{ink};pointer-events:none">{g:.3f}</text>')
    o.append("</svg>")
    return "".join(o)


PACKED_ROOT = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos_llm_scaling_ladder/results"
PSETS = [("val_v2_decon", "01_val_v2_overall_tw"), ("uncheatable_decon", "02_uncheatable_tw"), ("codefresh_decon", "03_codefresh_tw"),
         ("v3small_decon", "04_v3small_tw"), ("pile_test_decon", "05_pile_test_tw"), ("pile_10k_decon", "05_pile_10k_tw")]


VBASE = "KD_MUONH_A0P8"      # TODO 2 baseline B64 = top-64 + one tail bucket; every variant: alpha 0.8, MuonH, seed 1, same data order
VARIANTS = [("KD_MUONH_A0P8_S64Y", "S64Y"), ("KD_MUONH_A0P8_S64", "S64"), ("KD_MUONH_A0P8_S128", "S128"), ("KD_MUONH_A0P8_B64Y", "B64Y"),
            ("KD_MUONH_A0P8_B128", "B128"), ("KD_MUONH_A0P8_DD64", "DD64"), ("KD_MUONH_A0P8_DROP64", "DROP64")]
VRUNGS = ["q0", "q1", "q2"]
LRS = [("MUONH_LR0P5X", 0.5), ("MUONH_LR0P7X", 0.7), (None, 1.0), ("MUONH_LR1P2X", 1.2), ("MUONH_LR1P4X", 1.4), ("MUONH_LR2X", 2.0)]   # None = OPT_MUONH (the HyperP law, 1x)
LRS_IT = [("MUONH_LR0P5X_IT4577", 0.5), ("MUONH_LR0P7X_IT4577", 0.7), (None, 1.0), ("MUONH_LR1P2X_IT4577", 1.2), ("MUONH_LR1P4X_IT4577", 1.4), ("MUONH_LR2X_IT4577", 2.0)]
BASE_IT = "MUONH_LR1X_IT4577"   # fixed-T cell: every rung trained 4,577 iterations (9.6B tokens); None above = this 1x run
LRCOL = {"q0": "#86b6ef", "q1": "#3987e5", "q2": "#1c5cab", "q3": "#104281"}
VCOL = {"q0": "#86b6ef", "q1": "#3987e5", "q2": "#1c5cab"}   # ordinal blue ramp (reference palette steps 250 / 400 / 550)


RUNS = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/boxiangw/runs/scaling_ladder/qwen3_tpp200"
FINAL_RE = re.compile(r"validation-(\S+) loss at iteration (\d+) on validation set \| lm loss value: ([0-9.Ee+-]+)")


def log_final(change, q):
    """Packed final validation of a run from its training logs (the post-training pass at the highest iteration)."""
    mb = re.match(r"^(.*)_(IT|TPP)(\d+)$", change)
    if mb:                                            # MUONH_LR1X_IT4577 -> ..._lr1x-q1-iters4577-s1; MUONH_LR1X_TPP600 -> ..._lr1x-q0-tpp600-s1
        run = "qwen3_tpp200_" + mb.group(1).lower() + f"-{q}-" + ("iters" if mb.group(2) == "IT" else "tpp") + mb.group(3) + "-s1"
    else:
        run = "qwen3_tpp200_" + change.lower().replace("kd_muonh_", "muonh_kd_") + f"-{q}-tpp200-s1"
    vals, it = {}, -1
    for f in sorted(glob.glob(os.path.join(RUNS, run, "logs", "*.log"))):
        try:
            txt = open(f, errors="replace").read()
        except OSError:
            continue
        for m in FINAL_RE.finditer(txt):
            n = int(m.group(2))
            if n > it:
                vals, it = {}, n
            if n == it:
                vals[m.group(1)] = float(m.group(3))
    return vals


def load_packed():
    """Boxiang's packed final validation (scores.json, his 6 pooled sets) in the same {key: {loss}} shape; mean6 = his six-set mean."""
    R = {}
    for ch in [a[0] for a in ARMS] + [v[0] for v in VARIANTS] + [c for c, _ in LRS if c] + [c for c, _ in LRS_IT if c] + [BASE_IT] + sorted({cell_change(sfx, tag) for sfx, *_ in GRID_CELLS.values() for _, tag in MULTS}):
        p = os.path.join(PACKED_ROOT, ch, "scores.json")
        try:
            rungs = json.load(open(p)).get("rungs", {})
        except (OSError, ValueError):                   # missing, or mode 600 (Boxiang only)
            rungs = {}
        for q in RUNGS:                                 # rungs without a filed final: his run logs (finished but not promoted yet)
            if not all(k in ((rungs.get(q) or {}).get("val") or {}) for k, _ in PSETS):
                lf = log_final(ch, q)
                if lf:
                    rungs[q] = {"val": lf}
        for q, r in rungs.items():
            v = r.get("val") or {}
            if q in RUNGS and all(k in v for k, _ in PSETS):
                m = {key: {"loss": v[k]} for k, key in PSETS}
                m["mean6"] = {"loss": sum(v[k] for k, _ in PSETS) / len(PSETS)}
                R[(ch, q)] = m
    return R


def tables(R, t, cols):
    head = "".join(f'<th class="l">{h}</th>' for h in t["cols_head"]) + "".join(f"<th>{esc(n)}</th>" for _, n in cols)
    rows1, rows2 = [], []
    for q in RUNGS:
        first = True
        for ch, opt, lt, a in ARMS:
            m = R.get((ch, q))
            if m is None and (q == "q4" or lt == "CE"):
                continue
            cells = "".join(f"<td>{m[k]['loss']:.4f}</td>" if m and k in m else "<td>·</td>" for k, _ in cols)
            cls = ("kd" if lt == "KD" else "") + (" sep" if first else "")
            rows1.append(f'<tr class="{cls}"><td class="l">{q}</td><td class="l">{opt}</td><td class="l">{lt}</td><td class="l">{a}</td>{cells}</tr>')
            first = False
        ce = R.get((CE, q))
        if ce is None:
            continue
        first = True
        for a, ch in KD.items():
            m = R.get((ch, q))
            cells = ""
            for k, _ in cols:
                if m and k in m:
                    d = m[k]["loss"] - ce[k]["loss"]
                    cells += f'<td class="{"neg" if d < 0 else ""}">{"−" if d < 0 else "+"}{abs(d):.3f}</td>'
                else:
                    cells += "<td>·</td>"
            rows2.append(f'<tr class="{"sep" if first else ""}"><td class="l">{q}</td><td class="l">MuonH</td><td class="l">KD</td><td class="l">{a}</td>{cells}</tr>')
            first = False
    return (f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(rows1)}</tbody></table>',
            f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(rows2)}</tbody></table>')


def chart_sections(R, t, metrics):
    """Metric toggle + the optimizer (CE) delta chart and the KD-with-MuonH delta chart, one SVG per metric (deltas only, no absolute loss)."""
    qs_kd = [q for q in RUNGS if val(R, KD["0.8"], q, metrics[0][0]) is not None and val(R, CE, q, metrics[0][0]) is not None]
    charts = {}
    for mkey, mname in metrics:
        def pts_of(ch, f, name):
            return [(N_ACTIVE[q], f(q, ch), f"{esc(name)} · {q} ({N_LABEL[q]})<br>{mname} {f(q, ch):.4f}")
                    for q in RUNGS if val(R, ch, q, mkey) is not None and f(q, ch) is not None]
        xs_ce = {q: N_ACTIVE[q] for q in RUNGS if val(R, "BASE", q, mkey) is not None}
        opt = [(t["muon"], "--o-muon", "OPT_MUON"), (t["ce"], "--s1", CE)]
        dv = lambda q, ch: (val(R, ch, q, mkey) - val(R, "BASE", q, mkey)) if val(R, "BASE", q, mkey) is not None else None
        o2 = legend([(n, c) for n, c, _ in opt]) + line_chart(
            [(n, c, pts_of(ch, dv, n), False) for n, c, ch in opt],
            {q: x for q, x in xs_ce.items() if val(R, "OPT_MUON", q, mkey) is not None}, t["vs_adamw"], lambda v: f"{v:.3f}")
        kds = [(t["alpha"].format(a=a), f"--a{a.replace('.', '')}", ch) for a, ch in KD.items()]
        gk = lambda q, ch: (val(R, ch, q, mkey) - val(R, CE, q, mkey)) if val(R, CE, q, mkey) is not None else None
        k2 = legend([(n, c) for n, c, _ in kds]) + line_chart(
            [(n, c, pts_of(ch, gk, n), False) for n, c, ch in kds], {q: N_ACTIVE[q] for q in qs_kd}, "Δloss (KD − CE)", lambda v: f"{v:.3f}",
            labels=False)
        charts[mkey] = (o2, k2)

    def block(i):
        return "".join(f'<div data-metric="{m}"{"" if j == 0 else " hidden"}>{charts[m][i]}</div>' for j, (m, _) in enumerate(metrics))
    toggle = '<div class="toggle">' + "".join(f'<button data-m="{m}" class="{"on" if j == 0 else ""}">{esc(n)}</button>' for j, (m, n) in enumerate(metrics)) + "</div>"
    return f"""{toggle}
<h2>{esc(t['opt_h'])}</h2><p class="note">{esc(t['opt_note'])}</p>
<div class="card one"><div class="ct">{esc(t['o2_h'])}</div>{block(0)}</div>
<h2>{esc(t['kd_h'])}</h2><p class="note">{esc(t['kd_note'])}</p>
<div class="card one"><div class="ct">{esc(t['k2_h'])}</div>{block(1)}</div>"""


def variant_dot(R, t, mkey, mname):
    """Dot plot: one row per variant, x = variant - B64 on this metric, one dot per rung (ordinal blue)."""
    rows = [(ch, short) for ch, short in VARIANTS if any(val(R, ch, q, mkey) is not None and val(R, VBASE, q, mkey) is not None for q in VRUNGS)]
    ds = [val(R, ch, q, mkey) - val(R, VBASE, q, mkey) for ch, _ in rows for q in VRUNGS
          if val(R, ch, q, mkey) is not None and val(R, VBASE, q, mkey) is not None]
    if not ds:
        return f'<p class="note">{esc(t["pending"])}</p>'
    lo, hi = min(ds + [-0.008]), max(ds + [0.0])
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad
    ml, mr, mt, rh = 250, 20, 26, 34
    w, h = 760, mt + rh * len(rows) + 30
    X = lambda v: ml + (v - lo) / (hi - lo) * (w - ml - mr)
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for i in range(6):
        v = lo + (hi - lo) * i / 5
        o.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{mt-6}" y2="{h-30}" stroke="var(--grid)"/>'
                 f'<text x="{X(v):.1f}" y="{h-12}" text-anchor="middle">{"−" if v < 0 else "+"}{abs(v):.3f}</text>')
    o.append(f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="{mt-10}" y2="{h-30}" stroke="var(--axis)" stroke-width="1.5"/>'
             f'<text x="{X(0)+4:.1f}" y="{mt-12}" style="fill:var(--text-secondary)">B64 = 0</text>')
    o.append(f'<line x1="{X(-0.008):.1f}" x2="{X(-0.008):.1f}" y1="{mt-10}" y2="{h-30}" stroke="var(--muted)" stroke-dasharray="4 4"/>'
             f'<text x="{X(-0.008)-4:.1f}" y="{mt-12}" text-anchor="end" style="fill:var(--muted)">{esc(t["v_gate"])}</text>')
    for i, (ch, short) in enumerate(rows):
        y = mt + i * rh + rh / 2
        o.append(f'<text class="lbl" x="8" y="{y+4:.1f}">{esc(short)}</text><text x="62" y="{y+4:.1f}">{esc(t["vdesc"][short])}</text>')
        o.append(f'<line x1="{ml}" x2="{w-mr}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)" stroke-dasharray="1 3"/>')
        for q in VRUNGS:
            a, b = val(R, ch, q, mkey), val(R, VBASE, q, mkey)
            if a is None or b is None:
                continue
            d = a - b
            tip = f"{esc(short)} · {q} ({N_LABEL[q]})<br>{esc(mname)}: {short} {a:.4f} − B64 {b:.4f} = {d:+.4f}"
            o.append(f'<circle cx="{X(d):.1f}" cy="{y:.1f}" r="6" fill="{VCOL[q]}" stroke="var(--surface-1)" stroke-width="2"/>'
                     f'<circle class="hit" cx="{X(d):.1f}" cy="{y:.1f}" r="12" data-tip="{tip}"/>')
    o.append("</svg>")
    return '<div class="legend">' + "".join(f'<span><i style="background:{VCOL[q]};height:10px;width:10px;border-radius:50%"></i>{q} · {N_LABEL[q]}</span>' for q in VRUNGS) + "</div>" + "".join(o)


def variant_curve(R, t, mkey, mname):
    """Same form as the KD-vs-CE chart: x = active params (log), y = KD variant - CE (MuonH, alpha 0); B64 = KD alpha 0.8 itself."""
    series = []
    for i, (ch, short, col) in enumerate([(VBASE, "B64", "--a08")] + [(ch, sh, f"--v{j+1}") for j, (ch, sh) in enumerate(VARIANTS)]):
        pts = []
        for q in RUNGS:
            a, b = val(R, ch, q, mkey), val(R, CE, q, mkey)
            if a is not None and b is not None:
                pts.append((N_ACTIVE[q], a - b, f"{esc(short)} · {q} ({N_LABEL[q]})<br>{esc(mname)}: {short} {a:.4f} − CE {b:.4f} = {a - b:+.4f}"))
        if pts:
            series.append((f"{short} · {t['vdesc'][short]}", col, pts, False))
    if not series:
        return f'<p class="note">{esc(t["pending"])}</p>'
    xs = {q: N_ACTIVE[q] for q in RUNGS if any(val(R, ch, q, mkey) is not None for ch in [VBASE] + [v[0] for v in VARIANTS]) and val(R, CE, q, mkey) is not None}
    return legend([(n, c) for n, c, _, _ in series]) + line_chart(series, xs, "Δloss (KD − CE)", lambda v: f"{v:+.3f}", labels=False)


def variant_table(R, t, cols):
    head = '<th class="l">loss</th><th class="l">rung</th>' + "".join(f"<th>{esc(n)}</th>" for _, n in cols)
    rows = []
    for ch, short in [(CE, "CE")] + [(VBASE, "B64")] + VARIANTS:
        first = True
        for q in VRUNGS:
            m, b = R.get((ch, q)), R.get((CE, q))
            if m is None or b is None:
                continue
            if ch == CE:
                cells = "".join(f"<td>{m[k]['loss']:.4f}</td>" if k in m else "<td>·</td>" for k, _ in cols)
            else:
                cells = ""
                for k, _ in cols:
                    if k in m and k in b:
                        d = m[k]["loss"] - b[k]["loss"]
                        cells += f'<td class="{"neg" if d < 0 else ""}">{"−" if d < 0 else "+"}{abs(d):.4f}</td>'
                    else:
                        cells += "<td>·</td>"
            rows.append(f'<tr class="{"sep" if first else ""}{" kd" if ch == VBASE else ""}"><td class="l">{esc(short)}</td><td class="l">{q}</td>{cells}</tr>')
            first = False
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def variant_section(R, t, metrics, cols):
    blocks = "".join(f'<div data-metric="{m}"{"" if j == 0 else " hidden"}>{variant_curve(R, t, m, n)}</div>' for j, (m, n) in enumerate(metrics))
    names = ('<table><thead><tr>' + "".join(f'<th class="l">{esc(h)}</th>' for h in t["vn_head"]) + '</tr></thead><tbody>'
             + "".join(f'<tr><td class="l"><b>{esc(a)}</b></td><td class="l" style="white-space:normal">{esc(b)}</td></tr>' for a, b in t["vn_rows"])
             + '</tbody></table>')
    return f"""<h2>{esc(t['v_h'])}</h2><p class="note">{t['v_note']}</p>
<div class="card vn"><div class="ct">{esc(t['vn_title'])}</div>{names}</div>
<div class="card one"><div class="ct">{esc(t['v_chart'])}</div>{blocks}</div>
<div class="card"><div class="ct">{esc(t['v_table'])}</div>{variant_table(R, t, cols)}</div>"""


# MuonH LR grid (Runlong.md TODO 3): the user's predefined table - rows = model, columns = training-token budget; each "cell" is one
# (L, T) point where the best LR multiple m* is found from a 5-point sweep (x0.7 / 1 / 1.2 / 1.4 / 2, + x0.5 where run).
GRID_ROWS = [("q0", 12), ("q1", 18), ("q2", 24), ("q3", 30)]
GRID_COLS = ["9.6", "30", "71", "86", "137"]
MULTS = [(0.5, "0P5X"), (0.7, "0P7X"), (1.0, "1X"), (1.2, "1P2X"), (1.4, "1P4X"), (2.0, "2X")]
# (rung, column) -> (Change-ID suffix, token label, TPP, label key); suffix "" = the main ladder (1x = OPT_MUONH)
GRID_CELLS = {
    ("q0", "9.6"): ("", "9.6B", 200, "main"), ("q0", "30"): ("_TPP600", "28.8B", 600, "tpp"), ("q0", "86"): ("_TPP1800", "86.4B", 1800, "tpp"),
    ("q3", "137"): ("", "136.6B", 200, "main_x"),
    ("q1", "9.6"): ("_IT4577", "9.6B", 62, "it"), ("q1", "30"): ("", "30.8B", 200, "main"),
    ("q2", "9.6"): ("_IT4577", "9.6B", 27, "it"), ("q2", "30"): ("_IT14685", "30.8B", 87, "it2"), ("q2", "71"): ("", "71.1B", 200, "main"),
    ("q3", "9.6"): ("_IT4577", "9.6B", 14, "it"), ("q3", "30"): ("_IT14685", "30.8B", 45, "it2"),
}


GRID_ITERS = {("q0", "9.6"): 4577, ("q0", "30"): 13730, ("q0", "86"): 41189, ("q1", "9.6"): 4577, ("q1", "30"): 14685,
              ("q2", "9.6"): 4577, ("q2", "30"): 14685, ("q2", "71"): 33906, ("q3", "9.6"): 4577, ("q3", "30"): 14685, ("q3", "137"): 65165}


def law_lr(L, iters):
    """HyperP law: eta0 0.016 x sqrt(8 / L) x (T / 1.04e10)^-0.32, T = iters x 2^21 tokens (= the cell's 1x LR; matches every run's --lr)."""
    return 0.016 * math.sqrt(8 / L) * (iters * 2 ** 21 / 1.04e10) ** -0.32


def cell_change(suffix, tag):
    if suffix == "" and tag == "1X":
        return CE
    return f"MUONH_LR{tag}{suffix}"


def cell_points(R, rung, suffix, mkey):
    """[(multiple, loss - loss(1x of this cell))] for the multiples that have a result; [] if the cell's 1x is missing."""
    b = val(R, cell_change(suffix, "1X"), rung, mkey)
    if b is None:
        return []
    return [(m, val(R, cell_change(suffix, tag), rung, mkey) - b) for m, tag in MULTS if val(R, cell_change(suffix, tag), rung, mkey) is not None]


def fit_mstar(pts):
    """Parabola in log(multiple) through the points (>= 3, with points on both sides of the minimum): (m*, loss at m* - loss at 1x, edge flag)."""
    if len(pts) < 3:
        return None
    xs = [math.log(m) for m, _ in pts]; ys = [y for _, y in pts]
    n = len(xs); sx = [sum(x ** k for x in xs) for k in range(5)]; sxy = [sum(y * x ** k for x, y in zip(xs, ys)) for k in range(3)]
    A = [[sx[4], sx[3], sx[2]], [sx[3], sx[2], sx[1]], [sx[2], sx[1], n]]; bvec = [sxy[2], sxy[1], sxy[0]]
    def det(M): return (M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1]) - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0]) + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))
    D = det(A)
    if abs(D) < 1e-18:
        return None
    coef = []
    for i in range(3):
        Mi = [row[:] for row in A]
        for r in range(3):
            Mi[r][i] = bvec[r]
        coef.append(det(Mi) / D)
    a, b, c = coef
    if a <= 0:
        return None
    x0 = -b / (2 * a)
    edge = not (min(xs) < x0 < max(xs))
    x0c = min(max(x0, min(xs)), max(xs))
    return math.exp(x0c), a * x0c * x0c + b * x0c + c, edge


def mini_lr(R, t, rung, suffix, mkey, mname, w=250, h=150, lr1=None):
    pts = cell_points(R, rung, suffix, mkey)
    if len(pts) <= 1:                                   # nothing beyond the cell's own 1x yet
        return f'<div class="pend">{esc(t["g_pending"])}' + (f'<span>1× = {lr1:.4f}</span>' if lr1 else "") + '</div>'
    w, h = 300, 190
    ml, mr, mt, mb = 40, 8, 8, 30
    lx0, lx1 = math.log(0.45), math.log(2.2)
    fit = fit_mstar(pts)
    ys = [y for _, y in pts] + [0.0]
    if fit:
        ys.append(fit[1])
    lo, hi = min(ys), max(ys); pad = (hi - lo) * 0.12 or 0.005; lo, hi = lo - pad, hi + pad
    X = lambda m: ml + (math.log(m) - lx0) / (lx1 - lx0) * (w - ml - mr)
    Y = lambda y: mt + (hi - y) / (hi - lo) * (h - mt - mb)
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for v in (lo + (hi - lo) * k / 3 for k in range(4)):
        o.append(f'<line x1="{ml}" x2="{w-mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--grid)"/><text x="{ml-4}" y="{Y(v)+3:.1f}" text-anchor="end" style="font-size:9px">{v:+.3f}</text>')
    o.append(f'<line x1="{ml}" x2="{w-mr}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var(--axis)"/>')
    for m, _ in MULTS:
        o.append(f'<text x="{X(m):.1f}" y="{h-14}" text-anchor="middle" style="font-size:9px">{m:g}×</text>')
        if lr1:
            o.append(f'<text x="{X(m):.1f}" y="{h-3}" text-anchor="middle" style="font-size:8px;fill:var(--muted)">{m * lr1:.4f}</text>')
    if fit:                                               # fitted parabola + m* marker
        xs = [math.log(m) for m, _ in pts]
        a_lo, a_hi = min(xs), max(xs)
        curve = []
        for k in range(41):
            x = a_lo + (a_hi - a_lo) * k / 40
            pts_x = [(math.log(m), y) for m, y in pts]
        # re-evaluate the parabola from three fitted values: refit coefficients locally
        n = len(xs); sx = [sum(x ** k for x in xs) for k in range(5)]; sxy = [sum(y * math.log(m) ** k for m, y in pts) for k in range(3)]
        A = [[sx[4], sx[3], sx[2]], [sx[3], sx[2], sx[1]], [sx[2], sx[1], n]]
        import itertools
        def solve(A, b):
            M = [A[i][:] + [b[i]] for i in range(3)]
            for i in range(3):
                piv = max(range(i, 3), key=lambda r: abs(M[r][i])); M[i], M[piv] = M[piv], M[i]
                for r in range(3):
                    if r != i:
                        f = M[r][i] / M[i][i]
                        M[r] = [x - f * y for x, y in zip(M[r], M[i])]
            return [M[i][3] / M[i][i] for i in range(3)]
        qa, qb, qc = solve(A, [sxy[2], sxy[1], sxy[0]])
        d = " ".join(f"{'M' if k == 0 else 'L'}{X(math.exp(a_lo + (a_hi - a_lo) * k / 40)):.1f},{Y(qa * (a_lo + (a_hi - a_lo) * k / 40) ** 2 + qb * (a_lo + (a_hi - a_lo) * k / 40) + qc):.1f}" for k in range(41))
        o.append(f'<path d="{d}" fill="none" stroke="var(--s1)" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.8"/>')
        mstar, ystar, edge = fit
        o.append(f'<line x1="{X(mstar):.1f}" x2="{X(mstar):.1f}" y1="{mt}" y2="{h-mb}" stroke="var(--a08)" stroke-width="1.5"/>')
    for m, y in pts:
        tip = (f"{esc(rung)} · {m:g}× (η = {m * lr1:.5f})" if lr1 else f"{esc(rung)} · {m:g}×") + f"<br>{esc(mname)}: {y:+.4f} vs 1×"
        o.append(f'<circle cx="{X(m):.1f}" cy="{Y(y):.1f}" r="3.5" fill="var(--s1)" stroke="var(--surface-1)" stroke-width="1.5"/>'
                 f'<circle class="hit" cx="{X(m):.1f}" cy="{Y(y):.1f}" r="10" data-tip="{tip}"/>')
    o.append("</svg>")
    npts = t["g_npts"].format(n=len(pts))
    if fit:
        mstar, ystar, edge = fit
        head = (f'<div class="mbig">{mstar:.2f}×{"<small> " + esc(t["g_edge"]) + "</small>" if edge else ""}</div>'
                + (f'<div class="mlr">η* = <b>{mstar * lr1:.4f}</b> <span>(1× = {lr1:.4f})</span></div>' if lr1 else "")
                + f'<div class="msub">{ystar:+.4f} vs 1× · {esc(npts)}</div>')
    else:
        head = (f'<div class="mbig muted">—</div>' + (f'<div class="mlr"><span>1× = {lr1:.4f}</span></div>' if lr1 else "")
                + f'<div class="msub">{esc(t["g_fewpts"])} · {esc(npts)}</div>')
    return head + '<div class="pop">' + "".join(o) + "</div>"


SEED_ARMS = [("AdamW", ["BASE", "BASE_S2", "BASE_S3"]), ("Muon", ["OPT_MUON", "OPT_MUON_S2", "OPT_MUON_S3"])]
SEED_METRICS = [("00_overall_tsqw", "tsqw"), ("01_val_v2_overall_tw", "val_v2"), ("00_overall_tw", "overall_tw")]


def seed_section(R, t):
    """Seed noise: the same model trained with seeds 1-3 (same data per seed index), and the Muon - AdamW difference paired by seed."""
    rows = []
    for q in RUNGS:
        first = True
        for arm, ids in SEED_ARMS:
            if sum(R.get((c, q)) is not None for c in ids) < 2:
                continue
            cells = []
            for k, _ in SEED_METRICS:
                xs = [val(R, c, q, k) for c in ids]
                got = [x for x in xs if x is not None]
                sd = (sum((x - sum(got) / len(got)) ** 2 for x in got) / (len(got) - 1)) ** 0.5 if len(got) > 1 else None
                cells.append(" / ".join(f"{x:.4f}" if x is not None else "·" for x in xs) + f'</td><td><b>{max(got) - min(got):.4f}</b></td><td>{sd:.4f}')
            rows.append(f'<tr class="{"sep" if first else ""}"><td class="l">{q}</td><td class="l">{arm}</td><td>' + "</td><td>".join(cells) + "</td></tr>")
            first = False
        ds = {k: [val(R, a, q, k) - val(R, b, q, k) for a, b in zip(SEED_ARMS[1][1], SEED_ARMS[0][1])
                  if val(R, a, q, k) is not None and val(R, b, q, k) is not None] for k, _ in SEED_METRICS}
        if len(ds[SEED_METRICS[0][0]]) >= 2:
            cells = []
            for k, _ in SEED_METRICS:
                d = ds[k]; sd = (sum((x - sum(d) / len(d)) ** 2 for x in d) / (len(d) - 1)) ** 0.5
                cells.append(" / ".join(f"{'−' if x < 0 else '+'}{abs(x):.4f}" for x in d) + f'</td><td><b>{max(d) - min(d):.4f}</b></td><td>{sd:.4f}')
            rows.append(f'<tr class="kd"><td class="l">{q}</td><td class="l">{esc(t["sd_pair"])}</td><td>' + "</td><td>".join(cells) + "</td></tr>")
    if not rows:
        return ""
    head = (f'<tr><th class="l">rung</th><th class="l"></th>' + "".join(f'<th>{esc(n)}: s1 / s2 / s3</th><th>{esc(t["sd_range"])}</th><th>sd</th>' for _, n in SEED_METRICS) + "</tr>")
    return f"""<h2>{esc(t['sd_h'])}</h2><p class="note">{t['sd_note']}</p>
<div class="card"><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>"""


def grid_section(R, t, metrics):
    def table(mkey, mname):
        head = '<tr><th></th>' + "".join(f'<th>{esc(t["g_cols"][c])}</th>' for c in GRID_COLS) + "</tr>"
        rows = []
        for rung, L in GRID_ROWS:
            cells = []
            for c in GRID_COLS:
                spec = GRID_CELLS.get((rung, c))
                if spec is None:
                    cells.append('<td class="gx"></td>')
                    continue
                suffix, tok, tpp, kind = spec
                lab = t["g_kind"][kind].format(tok=tok, tpp=tpp)
                lr1 = law_lr(L, GRID_ITERS[(rung, c)])
                up = " up" if rung in (GRID_ROWS[-1][0], GRID_ROWS[-2][0]) else ""   # bottom rows: pop the curve upward
                cells.append(f'<td class="gc{up}"><div class="glab">{esc(lab)}</div>{mini_lr(R, t, rung, suffix, mkey, mname, lr1=lr1)}</td>')
            rows.append(f'<tr><th class="grow">{rung}<br><span>{L} {esc(t["g_layers"])} · {N_LABEL[rung]}</span></th>{"".join(cells)}</tr>')
        return f'<table class="lrgrid"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'
    m, n = metrics                                        # ONE fixed metric for this section (not tied to the toggle above)
    return f"""<h2>{esc(t['g_h'])}</h2><p class="note">{t['g_metric'].format(m=esc(n))} {t['g_note']}</p>
<div class="card" style="overflow:visible">{table(m, n)}</div>"""



def lr_chart(R, t, mkey, mname, w=760, h=340, lrs=None, base=None):
    """x = LR multiple of the HyperP law (log), y = loss - loss(1x), one line per rung (ordinal blue)."""
    LRS_, B = lrs or LRS, base or CE
    ch_of = lambda c: c or B
    qs = [q for q in RUNGS if val(R, B, q, mkey) is not None and sum(val(R, ch_of(c), q, mkey) is not None for c, _ in LRS_) >= 3]
    if not qs:
        return f'<p class="note">{esc(t["pending"])}</p>'
    pts = {q: [(m, val(R, ch_of(c), q, mkey) - val(R, B, q, mkey)) for c, m in LRS_ if val(R, ch_of(c), q, mkey) is not None] for q in qs}
    ys = [y for v in pts.values() for _, y in v] + [0.0]
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.08 or 0.01
    lo, hi = lo - pad, hi + pad
    ml, mr, mt, mb = 60, 30, 14, 40
    X = lambda m: ml + (math.log(m) - math.log(0.45)) / (math.log(2.2) - math.log(0.45)) * (w - ml - mr)
    Y = lambda y: mt + (hi - y) / (hi - lo) * (h - mt - mb)
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        o.append(f'<line x1="{ml}" x2="{w-mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--grid)"/><text x="{ml-6}" y="{Y(v)+4:.1f}" text-anchor="end">{v:+.3f}</text>')
    o.append(f'<line x1="{ml}" x2="{w-mr}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var(--axis)" stroke-width="1.5"/>')
    for _, m in LRS_:
        o.append(f'<text x="{X(m):.1f}" y="{h-mb+18}" text-anchor="middle">{m:g}×</text>')
    o.append(f'<text x="{(ml+w-mr)/2:.0f}" y="{h-6}" text-anchor="middle" style="fill:var(--muted)">{esc(t["lr_x"])}</text>')
    o.append(f'<text x="14" y="{(h-mb+mt)/2:.0f}" transform="rotate(-90 14 {(h-mb+mt)/2:.0f})" text-anchor="middle">Δloss (LR − 1×)</text>')
    for q in qs:
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(m):.1f},{Y(y):.1f}" for i, (m, y) in enumerate(pts[q]))
        o.append(f'<path d="{d}" fill="none" stroke="{LRCOL[q]}" stroke-width="2"/>')
        for m, y in pts[q]:
            o.append(f'<circle cx="{X(m):.1f}" cy="{Y(y):.1f}" r="4" fill="{LRCOL[q]}" stroke="var(--surface-1)" stroke-width="2"/>')
    for i, (_, m) in enumerate(LRS_):                    # hover band per LR: every rung at that LR, largest on top
        a = (X(m) + X(LRS_[i-1][1])) / 2 if i else ml
        b = (X(m) + X(LRS_[i+1][1])) / 2 if i + 1 < len(LRS_) else w - mr
        rows = sorted(((q, y) for q in qs for mm, y in pts[q] if mm == m), key=lambda r: r[1], reverse=True)
        if rows:
            tip = f"<b>{m:g}× · {esc(mname)}</b>" + "".join(f'<br><span style="color:{LRCOL[q]}">●</span> {q} ({N_LABEL[q]}): {y:+.4f}' for q, y in rows)
            o.append(f'<rect class="hit" x="{a:.1f}" y="{mt}" width="{b-a:.1f}" height="{h-mt-mb}" data-tip="{esc(tip)}"/>')
    o.append("</svg>")
    return '<div class="legend">' + "".join(f'<span><i style="background:{LRCOL[q]}"></i>{q} · {N_LABEL[q]}</span>' for q in qs) + "</div>" + "".join(o)


def lr_table(R, t, cols, lrs=None, base=None):
    LRS_, B = lrs or LRS, base or CE
    head = '<th class="l">LR</th><th class="l">rung</th>' + "".join(f"<th>{esc(n)}</th>" for _, n in cols)
    rows = []
    for q in RUNGS:
        b = R.get((B, q))
        if b is None or not any(R.get((c, q)) for c, _ in LRS_ if c):
            continue
        first = True
        for c, m in LRS_:
            r = R.get((c or B, q))
            if r is None:
                continue
            if c is None:
                cells = "".join(f"<td>{r[k]['loss']:.4f}</td>" if k in r else "<td>·</td>" for k, _ in cols)
            else:
                cells = "".join((f'<td class="{"neg" if r[k]["loss"] < b[k]["loss"] else ""}">{"−" if r[k]["loss"] < b[k]["loss"] else "+"}{abs(r[k]["loss"] - b[k]["loss"]):.4f}</td>')
                                if k in r and k in b else "<td>·</td>" for k, _ in cols)
            rows.append(f'<tr class="{"sep" if first else ""}{" kd" if c is None else ""}"><td class="l">{m:g}×</td><td class="l">{q}</td>{cells}</tr>')
            first = False
    return f'<table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def lr_section(R, t, metrics, cols):
    blocks = "".join(f'<div data-metric="{m}"{"" if j == 0 else " hidden"}>{lr_chart(R, t, m, n)}</div>' for j, (m, n) in enumerate(metrics))
    it_blocks = "".join(f'<div data-metric="{m}"{"" if j == 0 else " hidden"}>{lr_chart(R, t, m, n, lrs=LRS_IT, base=BASE_IT)}</div>' for j, (m, n) in enumerate(metrics))
    return f"""<h2>{esc(t['lr_h'])}</h2><p class="note">{t['lr_note']}</p>
<div class="card one"><div class="ct">{esc(t['lr_chart'])}</div>{blocks}</div>
<div class="card"><div class="ct">{esc(t['lr_table'])}</div>{lr_table(R, t, cols)}</div>
<h3>{esc(t['lri_h'])}</h3><p class="note">{t['lri_note']}</p>
<div class="card one"><div class="ct">{esc(t['lri_chart'])}</div>{it_blocks}</div>
<div class="card"><div class="ct">{esc(t['lr_table'])}</div>{lr_table(R, t, cols, lrs=LRS_IT, base=BASE_IT)}</div>"""


def build(lang, R, RP):
    t = T[lang]
    qs_kd = [q for q in RUNGS if val(R, KD["0.8"], q) is not None and val(R, CE, q) is not None]
    last = qs_kd[-1]
    gain_last = val(R, CE, last) - val(R, KD["0.8"], last)
    mults = {q: equiv_mult(R, q) for q in qs_kd}
    mult_last_q = max((q for q in qs_kd if mults[q]), key=RUNGS.index)
    priv = val(R, CE, last, "04_private_internal_tw") - val(R, KD["0.8"], last, "04_private_internal_tw")
    amax, n_cells, n_neg = 0.0, 0, 0
    for q in qs_kd:
        for k, _ in COLS:
            vs = [val(R, KD[a], q, k) for a in KD if val(R, KD[a], q, k) is not None]
            if len(vs) > 1:
                amax = max(amax, max(vs) - min(vs))
            for v in vs:
                n_cells += 1
                n_neg += v < val(R, CE, q, k)
    gains = " → ".join(f"{val(R, CE, q) - val(R, KD['0.8'], q):.3f}" for q in qs_kd)
    mult_s = " → ".join(f"{mults[q]:.2f}" for q in qs_kd if mults[q])
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    tiles = [(t["tiles"][0], f"{gain_last:.3f}", t["tile_sub"][0].format(rung=last)),
             (t["tiles"][1], f"{mults[mult_last_q]:.2f}×", t["tile_sub"][1].format(rung=mult_last_q, mult=f"{mults[mult_last_q]:.2f}")),
             (t["tiles"][2], f"{priv:.3f}", t["tile_sub"][2].format(rung=last)),
             (t["tiles"][3], f"{amax:.3f}", t["tile_sub"][3])]
    tiles_html = '<div class="tiles">' + "".join(f'<div class="tile"><div class="k">{esc(k)}</div><div class="v">{v}</div><div class="s">{esc(s)}</div></div>' for k, v, s in tiles) + "</div>"
    finds = [t["finds"][0].format(n_cells=n_cells), t["finds"][1].format(gains=gains, last=last[1]), t["finds"][2].format(mults=mult_s),
             t["finds"][3], t["finds"][4].format(amax=amax),
             t["finds"][5].format(optg=" → ".join(f"{val(R, 'BASE', q) - val(R, CE, q):.3f}" for q in RUNGS
                                                   if val(R, 'BASE', q) is not None and val(R, CE, q) is not None) + f" (q0 → q{max(q for q in RUNGS if val(R, CE, q) is not None)[1]})")]
    if n_neg != n_cells:
        finds[0] += f" ({n_neg}/{n_cells})"
    # KD loss variants: sampled tails vs B64 per rung (overall_tw); LR test: best multiple per rung
    sv = [(sh, " → ".join(f"{val(R, ch, q) - val(R, VBASE, q):+.4f}" for q in VRUNGS if val(R, ch, q) is not None and val(R, VBASE, q) is not None))
          for ch, sh in VARIANTS if sh in ("S64", "S64Y", "S128")]
    bk = [(sh, " → ".join(f"{val(R, ch, q) - val(R, VBASE, q):+.4f}" for q in VRUNGS if val(R, ch, q) is not None and val(R, VBASE, q) is not None))
          for ch, sh in VARIANTS if sh in ("B64Y", "B128")]
    if any(v for _, v in sv):
        finds.append(t["find_v"].format(sampled="；".join(f"{a} {b}" for a, b in sv if b) if lang == "zh" else "; ".join(f"{a} {b}" for a, b in sv if b),
                                         bucket="；".join(f"{a} {b}" for a, b in bk if b) if lang == "zh" else "; ".join(f"{a} {b}" for a, b in bk if b)))
    lr_best = []
    for (rung, c), (suffix, tok, tpp, kind) in GRID_CELLS.items():
        f = fit_mstar(cell_points(R, rung, suffix, "00_overall_tsqw"))
        if f:
            lr_best.append(f"{rung}@{tok}（TPP {tpp}）{f[0]:.2f}×{'*' if f[2] else ''}（η* {f[0] * law_lr(dict(GRID_ROWS)[rung], GRID_ITERS[(rung, c)]):.4f}）")
    if lr_best:
        finds.append(t["find_lr"].format(best="；".join(lr_best) if lang == "zh" else "; ".join(lr_best).replace("（", " (").replace("）", ")")))
    t1, _ = tables(R, t, COLS)
    pcols = [(key, n) for k, key in PSETS for kk, n in COLS if kk == key] + [("mean6", t["mean6"])]
    p1, p2 = tables(RP, t, pcols)
    PMETRICS = [("01_val_v2_overall_tw", "val_v2"), ("02_uncheatable_tw", "uncheatable"), ("03_codefresh_tw", "codefresh"), ("mean6", t["mean6"])]
    doc = f"""{tiles_html}
<h2>{esc(t['find_h'])}</h2><div class="card"><ul class="f">{''.join(f'<li>{f}</li>' for f in finds)}</ul></div>
{token_pie(t)}
{chart_sections(R, t, METRICS)}
{variant_section(R, t, METRICS, COLS)}
{grid_section(R, t, ("00_overall_tsqw", "tsqw"))}
<h2>{esc(t['t1_h'])}</h2><p class="note">{esc(t['pending'])}</p><div class="card">{t1}</div>
<h2>{esc(t['notes_h'])}</h2><div class="card"><ul class="f">{''.join(f'<li>{n}</li>' for n in t['notes'])}</ul></div>"""
    packed = f"""<div class="card"><ul class="f">{''.join(f'<li>{n}</li>' for n in t['p_notes'])}</ul></div>
{chart_sections(RP, t, PMETRICS)}
{variant_section(RP, t, PMETRICS, pcols)}
<h2>{esc(t['t2_h'])}</h2><div class="card">{p2}</div>
<h2>{esc(t['t1_h'])}</h2><p class="note">{esc(t['pending'])}</p><div class="card">{p1}</div>"""
    tabs = "".join(f'<button class="tab{" on" if i == 0 else ""}" data-tab="{k}">{esc(n)}</button>' for i, (k, n) in enumerate((("doc", t["tab_doc"]), ("packed", t["tab_packed"]))))
    body = f"""<div id="tip"></div><nav>{tabs}</nav><main>
<h1>{esc(t['title'])}</h1><p class="sub">{esc(t['sub'].format(ts=ts))}</p>
<section class="pane" data-pane="doc">{doc}</section>
<section class="pane" data-pane="packed" hidden>{packed}</section>
</main><script>{JS}</script>"""
    return (f'<!doctype html><html lang="{t["lang"]}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{esc(t["title"])}</title><style>{CSS}</style></head><body class="viz-root">{body}</body></html>\n')


def main():
    R, RP = load(), load_packed()
    for lang, name in (("zh", "qwen3_scaling_ladder_eval.html"), ("en", "qwen3_scaling_ladder_eval_en.html")):
        p = os.path.join(HERE, name)
        open(p, "w").write(build(lang, R, RP))
        print("wrote", p, f"(doc {len(R)}, packed {len(RP)} checkpoints)")


if __name__ == "__main__":
    main()
