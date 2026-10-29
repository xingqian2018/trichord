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
N_ACTIVE = {"q0": 48.0e6, "q1": 153.9e6, "q2": 355.5e6, "q3": 683.2e6, "q4": 1.17e9, "q5": 1.84e9, "q6": 2.73e9}      # non-embedding active params
N_LABEL = {"q0": "48M", "q1": "154M", "q2": "356M", "q3": "683M", "q4": "1.17B", "q5": "1.84B", "q6": "2.73B"}
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
METRICS = [("00_overall_tsqw", "overall_tsqw"), ("00_overall_tw", "overall_tw"), ("00_overall_macro", "overall_macro"), ("01_val_v2_overall_tw", "val_v2"), ("02_uncheatable_tw", "uncheatable"), ("03_codefresh_tw", "codefresh"), ("04_private_internal_tw", "private_internal")]
# sequential blue ramp (reference palette), light -> dark
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]

T = {
 "zh": {
  "lang": "zh", "title": "Qwen3-MoE scaling ladder：KD 对 CE",
  "sub": "doc-isolated 评测（9 个 decon suite，含 private_internal），dpl 指标定义；数据生成于 {ts}",
  "nav": ["结论", "优化器（CE）", "KD（MuonH）", "热力图", "数据表", "说明"],
  "muon_fit": "Muon（新拟合 LR）", "ce_fit": "MuonH（新拟合 LR，MUONH_FIT）", "opt_h": "优化器对比：AdamW / Muon / MuonH（都是 CE）", "opt_note": "虚线 = Boxiang 按新拟合的 LR 公式跑的同一优化器（MuonH：fit v2 η = 0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259，目前有 q2 / q3；Muon 没有按新公式跑）。同一数据、同一 token 数，只换优化器（及其 LR 规则 / schedule）。横轴：激活参数（不含 embedding，对数坐标）。",
  "o2_h": "相对 AdamW 的 Δloss（优化器 − AdamW，越负越好）",
  "kd_h": "KD 对 CE：优化器固定为 MuonH", "kd_note": "CE 与 KD 只差 loss（α = 0 vs 0.7 / 0.8 / 0.9），优化器、数据顺序、layout 完全相同。",
  "k2_h": "KD − CE：各 α 的 Δloss（越负越好）",
  "adamw": "AdamW", "muon": "Muon", "vs_adamw": "Δloss（优化器 − AdamW，越负越好）", "mean6": "六集平均",
  "v_h": "各种 KD loss 对比：同一 α 0.8、同一 MuonH", "v_note": "8 个 run 只差 KD 项里 teacher 分布的截断方式：α = 0.8、MuonH 优化器、seed 1、数据顺序全部相同；基线仍是 <b>CE</b>（MuonH，α = 0）；<b>B64</b>（top-64 + 剩余 tail 合成一个 bucket）就是 KD α 0.8 本身，作为一条线一起画出来，各变体看是否比 B64 更低。负数 = 比 CE 好。",
  "v_chart": "Δloss（KD 变体 − CE），越低越好", "v_table": "CE 的 loss，以及 B64 和各变体相对 CE 的差", "v_gate": "晋级线 −0.008",
  "vdesc": {"B64": "top-64 + bucket（= KD α 0.8）", "S64Y": "top-64 + 采样 tail + y exact", "S64": "top-64 + 采样 tail（4 draws）", "S128": "top-128 + 采样 tail", "S128Y": "top-128 + 采样 tail + y exact", "B64Y": "top-64 bucket + y exact",
            "B128": "top-128 + bucket", "DD64": "top-64 + data draw", "DROP64": "top-64，丢掉 tail（对照）"},
  "lr_h": "MuonH 学习率测试：同一 MuonH、CE", "lr_note": "只改 MuonH 的峰值 LR（HyperP 规则给出的 LR × 0.5 / 0.7 / 1.4 / 2），其余相同；1× = 主线 MuonH（CE）。每个 rung 一条线；最低点就是这个 rung 上的最优 LR。",
  "lr_chart": "Δloss（该 LR − 1×），越低越好", "lr_table": "1× 的 loss，以及其他 LR 相对 1× 的差", "lr_x": "峰值 LR / HyperP 规则（对数坐标）",
  "find_v": "各种 KD loss（同一 α 0.8、MuonH）：采样 tail 的变体一直比 B64 好，但领先随规模缩小（overall_tw，q0 → q2：{sampled}）；bucket 类变体 ≈ 0（{bucket}）。",
  "find_lr": "MuonH 学习率（overall_tsqw，抛物线拟合的最优倍数 m*，* = 顶点在已测倍数之外、为外推估计）：{best}。",
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
  "g_h": "MuonH 学习率：每个格子找最优 LR（同一 MuonH、CE）", "g_note": "每个格子 = 一个固定的（模型大小，训练 token 数）。格子里按 HyperP 公式算出的 LR 的 0.5 / 0.7 / 1 / 1.2 / 1.4 / 2（/ 2.8 / 4）倍各训一个模型，纵轴 = 相对该格 1× 的 loss 差（越低越好）；格子里的大数字 = 拟合出的最优倍数 <b>m*</b>（抛物线顶点，可能外推到已测倍数之外；m* &gt; 1 表示公式算低了），下面是对应的绝对 LR <b>η*</b>（括号里是该格 1× 的 LR，即 HyperP 公式按层数和训练 token 数算出的 LR）和最优点比 1× 好多少；主线 TPP 200 的格子沿对角线排列；<b>鼠标移到格子上显示该格的 LR 测试曲线</b>（虚线 = 在 log(倍数) 上拟合的抛物线，橙线 = m*）。同一列看深度、q0 一行看训练量。<b>不同格子的训练量不同，只在格子内部比较。</b>空白 = 不在设计里。",
  "g_cols": {"9.6": "9.6B token", "30": "~30B token", "71": "71.1B token", "86": "86.4B token", "137": "136.6B token"}, "g_layers": "层",
  "g_kind": {"main": "主线 TPP {tpp}", "main_x": "主线 TPP {tpp}（不在 TODO 3 里）", "tpp": "{tok}，TPP {tpp}", "it": "{tok}，TPP {tpp}（IT4577）", "it2": "{tok}，TPP {tpp}"},
  "g_ref": "所有倍数都相对 ArchScale 原版公式 η = 0.016 × (L/8)^−0.5 × (T/10.4B)^−0.32，由每个 run 实际的峰值 LR 计算（不看 Change ID 的名字），不随 ladder 改用的新公式变化。", "g_ref_muon": "倍数相对各 rung 现用的 Muon LR（固定值），由每个 run 实际的峰值 LR 计算。", "g_mm": "实际倍数与 Change ID 名字不符：", "g_pending": "待测", "g_fewpts": "点数不足", "g_edge": "（外推）", "g_npts": "{n} 个点", "g_nseed": "（{s} 个 seed）", "g_seedal": "按 seed 偏移对齐",
  "sd_h": "Seed 噪声：同一配置换 seed（doc-isolated）", "sd_note": "同一模型、同一 recipe，只换 seed（seed k 的数据顺序在不同 arm 之间相同，所以 Muon − AdamW 可以按 seed 配对）。<b>极差</b> = 三个 seed 最大减最小，<b>sd</b> = 标准差。用来判断其他小节里的差值是否超出噪声。",
  "sd_pair": "Muon − AdamW（按 seed 配对）", "sd_range": "极差",
  "arch_y": "Δloss（改动 − 对照，越负越好）", "arch_chart": "每种改动相对各自对照的 Δloss（单个 seed；全部是新 LR 公式；鼠标移到某个 rung 看所有改动的数值）", "arch_h": "滑动窗口注意力 SWA（同一 MuonH、CE、新 LR 公式 fit v2；128 选 8 MoE，softmax router，DSV3 底座一行除外；q0–q3，单个 seed）", "arch_note": "各种改动的具体设定见下表；图里的数值 = 改动 − 对照（doc-isolated，负数 = 改动更好），对照和改动都用新 LR 公式（fit v2）。",
  "arch_name": {"SWA": "SWA − MuonH（旧 LR 公式）", "SWA_FIT": "SWA − 无 SWA（MUONH_FIT）", "SWA_FIT_ROPE": "SWA，全注意力层用 RoPE − MUONH_FIT", "SWA_FIT_W4K": "SWA，窗口 4096 − MUONH_FIT", "SWA_FIT_W1K": "SWA，窗口 1024 − MUONH_FIT", "SWA_FIT_SINK": "SWA + 可学习 attention sink（gpt-oss 式）− MUONH_FIT", "SWA_FIT_SINK_W1K": "SWA + sink + 窗口 1024 − MUONH_FIT", "SWA_FIT_DSV3": "SWA（DSV3 router 底座）− 同底座无 SWA（MUONH_FIT_DSV3）", "MUONH_SPLITFC1": "MuonH 拆分 fc1（gate / up 各自更新）− MuonH"},
  "mg_g_h": "Muon 学习率：同样的格子（同一 Muon、CE）", "mg_g_note": "Boxiang 10-24 决定的 Muon 全因子网格：格子与上面 MuonH 相同，倍数 0.25 / 0.5 / 1 / 2 / 4×（间隔更粗，因为 Muon 的最优点位置未知）。1× = ladder 现用的 Muon LR（AdamW 规则，只随模型大小变，<b>不随训练 token 变</b>）。读法同上：大数字 = 抛物线顶点 m*，η* = 绝对 LR，鼠标移到格子上看曲线。目的：Muon 也用各自最优 LR 时，MuonH 是否仍然领先。", "mg_lf_h": "拟合：Muon 最优 LR 随激活参数量 N 和训练量的关系（与 Muon 现用的 adamw_active 规则同一写法）", "mg_lf_arch": "Muon / AdamW 现用 LR（ladder 的 adamw_active 规则，不是 ArchScale；每个格子的 1×）", "mg_lf_box": "Boxiang 的 Muon 拟合 v3（10-26 记录；val_v2、9 格 43 点；未用于实验；有效范围 L 12–30、T 9.6–137B）", "mg_lf_axN": "激活参数 N", "mg_noT": "（只随激活参数量 N 变，不含 T 项）",
  "ref_dag": "† 用自己的 tokenizer：val_loss / ppl 按它自己的 token 计，和按 Qwen3 token 计的其他行不可比（只看 bpb）。bpb = 总 NLL ÷（原文 UTF-8 字节数 × ln 2），分母与 ladder 完全相同；文本为 ~/data/eval_ppl_decon 的原文，切分规则同 ladder（每篇第一个 token 不计分，BOS 只作上下文）。", "ref_h": "外部参考：MiMo / DeepSeek / Kimi（同一 doc-isolated 协议）", "ref_gap": "我们最好的 − MiMo-V2.5-Pro-Base", "ref_best": "我们最好的", "ref_note": "MiMo-V2.5-Pro-Base（fp8）和 MiMo-V2.6-Pro-RL（官方 mxfp4 权重；也是 KD 的 teacher）用和 ladder 完全相同的 doc-isolated 115 个 split 打分（vLLM prompt_logprobs，每个 split 的预测 token 数与 ladder 一致）；MiMo 的 token id 与 ladder 的 Qwen3 tokenizer 相同，所以每 token 的 loss 可以直接比较。下面是 loss 绝对值（越低越好）；ladder 只列我们最好的一个 checkpoint（所有已测 checkpoint 中 overall_tsqw 最低）；最后一行 = 它减 MiMo-Base（ppl 为比值）。上方按钮切换 val_loss / ppl / bpb。RL 版本在预训练文本上比 Base 高约 0.05，属正常（RL 后训练 + 量化）。",
  "aw_h": "LR 公式的假设检查：深度 × 宽度，对角线用作 ArchScale 锚点（同一 MuonH、CE、训练量 9.6B）", "aw_note": "Runlong 计划 TODO 3 第 27、30、35 步：训练量固定 9.6B（4,577 步，GBS 2M），在不同层数 L 和宽度 h 上各扫一遍 LR（ArchScale 公式的倍数），看最优倍数 m* 是否只随深度变（公式假设宽度无关）。对角线 = ladder 各 rung（q0–q6）；其余格子 = 额外的深度 / 宽度点。指标 {m}；格子读法同上（悬停看曲线）。注意 9.6B 对大模型是严重欠训练（q6 TPP 3.5）。", "aw_ladder": "ladder {q}", "aw_extra": "额外格子", "aw_unstable": "router 崩溃（负载均衡 loss 9–16），不是 LR 点", "aw_no1x": "（无 1×：相对最小倍数）", "ac_h": "其余检查：batch、LR floor、按参数组分 LR", "ac_head": ["检查", "实验", "对照", "读法"], "ac_batch": "batch（q1，L 18，9.6B）", "ac_batch_r": "Runlong 预先写下的预测：公式按 token 计 1.45 / 按步数计 1.81 / B^0.56 2.14 / sqrt 噪声 2.56", "ac_floor0": "LR floor（q0，L 12）", "ac_floor_r": "MuonH 无 warmup、从头线性衰减到峰值的 X%（不是 WSD）；10% = ArchScale 自己的设置，ladder 用 1%", "ac_floor3": "LR floor（q3，L 30）", "ac_floor3_r": "相对同 floor 的 1×；负 = 比 1× 好", "ac_split": "按参数组分 LR（q3，L 30）", "ac_split_r": "相对 1×：A = 只有 AdamW 组（embedding、1-D）1.74×；B = hidden 矩阵 + LM head + router 1.74×；C = 只有 hidden 矩阵 1.74×",
  "rt_set_head": ["组", "Change ID", "主体优化器", "router 打分", "负载均衡", "gate（router 权重）优化器", "router LR", "LR 规则 / schedule", "对照", "已跑"], "rt_set_rows": [["对照 ①", "MUONH_FIT", "MuonH", "softmax", "aux loss 1e-3", "AdamH（球面）", "×1", "新拟合公式；无 warmup，线性衰减到 1%，wd 0", "—", "q0 ×3 seed、q1–q3"], ["<b>A</b>", "MUONH_FIT_DSV3", "MuonH", "<b>sigmoid</b>", "<b>expert bias</b> + seq aux 1e-4", "AdamH", "×1", "同 ①", "①", "q0 ×3、q1–q3"], ["<b>B2</b>", "…_DSV3_RTADAMW_RT0P19", "MuonH", "sigmoid", "同 A", "<b>AdamW</b>（wd 0）", "<b>×0.19</b>", "同 ①", "A", "q0 ×3、q1"], ["<b>B3</b>", "…_DSV3_RT0P19", "MuonH", "sigmoid", "同 A", "AdamH", "<b>×0.19</b>", "同 ①", "A", "q0 ×3"], ["对照 ②", "BASE", "AdamW", "softmax", "aux loss 1e-3", "AdamW", "×1", "adamw_active；WSD（5% warmup，最后 20% 衰减），wd 0.1", "—", "q0 ×3、q1–q5"], ["<b>C</b>", "BASE_DSV3_ONETREE", "AdamW", "<b>sigmoid</b>", "<b>expert bias</b> + seq aux 1e-4", "AdamW", "×1", "同 ②", "②", "q0 ×3、q1、q2"], ["S", "MUONH_FIT_DSV3_LR0P7X / 1P4X", "同 A，整体 LR ×0.7 / ×1.4", "", "", "", "", "", "A", "q0"], ["S_C", "BASE_DSV3_ONETREE_LR0P5X / 2X", "同 C，整体 LR ×0.5 / ×2", "", "", "", "", "", "C", "q0"]], "rt_set_l": "MuonH 主体（对照 ①：MUONH_FIT，softmax router）", "rt_set_r": "AdamW 主体（对照 ②：BASE，softmax router）", "rt_set_note": "所有组：128 个 expert 选 8 个、无 shared expert、batch 2M token；每组只改加粗的项：A、C 相对本表的对照，B2、B3 在 A 的基础上再改 gate。router LR ×1 = 与主体同一 LR（新公式在 q0 为 0.0164），×0.19 ≈ 3.1e-3。C 的 ONETREE = 和 A 同一份代码树，所以 C − BASE 是跨代码树的比较。", "rt_chart_h": "结果：实验组 − 对照组的 loss 差（overall_tsqw，越负越好）。q0 有 3 个 seed：空心点 = 每个 seed 各自的配对差，实心点和折线 = 3 个 seed 的平均；其余 rung 是 1 个 seed", "rt_ylab": "Δloss（实验组 − 对照组）", "rt_lr_note": "LR 检查（q0）：MuonH + DSV3 的最优倍数是新公式的 {s} 倍，AdamW + DSV3 是 adamw_active 的 {c} 倍，两者的现用 LR 都接近最优。",
  "rt_hm_h": "每层负载均衡（训练日志，seed 1）", "rt_hm_note": "来自 Boxiang 的 router 监测 <code>router_health.py</code>（<code>LADDER_ROUTER_HEALTH=100</code>，每 100 步一次）。每个小图：纵轴 = 层（上面是第 0 层），横轴 = 训练步数（最多 60 列，相邻的报告取平均），颜色越深数值越大；上方是训练结束时各层的平均值；悬停看具体数值。各指标：<b>MaxVio</b> = 一整个训练 step（所有 micro-batch、所有 rank）里每层每个 expert 实际分到的 token 数，最忙的 expert ÷ 平均值 − 1（DeepSeek-V3 的指标；0 = 完全均衡，2.86 = 最忙的分到平均值的 3.86 倍；色标上限 4）；<b>dead</b> = 分到的 token 不到平均值 10% 的 expert，占 128 个的百分比；<b>|bias| 最大</b> = DSV3 每个 expert 的 bias 的最大绝对值（Boxiang 预设健康上限 1）；<b>bias 极差</b> = 同一层 bias 的最大 − 最小——top-8 选择只看 bias 之间的差，所有 expert 一起抬高（|bias| 大但极差小）并不改变路由；<b>logit 饱和</b> = router logit |z| &gt; 4 的比例，<b>最大 |logit|</b> = 最大的 |z|，两者都只看 rank 0 的第一个 micro-batch，且 |z| &gt; 4 包括 z &lt; −4（明确拒绝某个 expert，正常）；<b>gate 更新幅度</b> = router 权重这一步的变化量 ÷ 权重大小（Boxiang 只作参考，不作判据）。softmax router 没有 bias；BASE（softmax AdamW）和 MUONH_FIT 的 q2 / q3 没有开监测。<b>注意：</b>训练日志把每一行截断在约 1000 个字符，逐层那一行写不完，DSV3 各组靠后的层缺数据（灰色；q0 缺 1 层、q1 缺 4 层、q3 缺 16 层），这在源头就丢了，补不回来。", "rt_hm_vio": "MaxVio", "rt_hm_dead": "dead (%)", "rt_hm_bmax": "|bias| 最大", "rt_hm_bspread": "bias 极差", "rt_hm_sat": "logit 饱和 (%)", "rt_hm_zmax": "最大 |logit|", "rt_hm_upd": "gate 更新幅度", "rt_hm_nobias": "softmax 没有 bias", "rt_hm_miss": "{n} 层缺数据", "rt_hm_na": "无监测数据", "rt_hm_end": "结束时层均值", "rt_hm_layers": "层 ↓ · 步数 →", "rt_hm_soft": "① softmax（MUONH_FIT）", "rt_hm_a": "A：DSV3", "rt_hm_b2": "B2：AdamW gate ×0.19", "rt_hm_b3": "B3：AdamH gate ×0.19", "rt_hm_c": "C：AdamW + DSV3",
  "rt_st_h": "router 健康统计（训练日志，seed 1）", "rt_st_note": "来自 Boxiang 的 router 监测 <code>router_health.py</code>（每 100 步一次）的汇总行，覆盖所有层。每个 rung 一张图，横轴 = 训练步数，每组一条线；上方按钮切换指标，鼠标悬停看各组数值。BASE（softmax AdamW）和 MUONH_FIT 的 q2 / q3 没有开监测，所以没有这两条线。", "rt_st_name": {"vio_mean": "MaxVio（层平均）", "vio_max": "MaxVio（最差层）", "dead": "dead expert (%)", "bmax": "|bias| 最大", "sat": "logit 饱和 (%)", "satpos": "z > +4 (%)", "pinned": "top-8 pinned (%)", "zmax": "最大 |logit|", "upd": "gate 更新幅度"}, "rt_st_def": {"vio_mean": "每层：一整个训练 step（所有 micro-batch、所有 rank）里分到 token 最多的 expert ÷ 平均值 − 1（DeepSeek-V3 的 MaxVio，0 = 完全均衡）；这里是所有层的平均。", "vio_max": "同上，取最不均衡的那一层。", "dead": "分到的 token 不到平均值 10% 的 expert 数，占（层数 × 128）的百分比。", "bmax": "DSV3 每个 expert 的 bias 在所有层里的最大绝对值；红色虚线 = Boxiang 预设的健康上限 1。注意 top-8 选择只看同一层 bias 之间的差，所有 expert 一起抬高并不改变路由（例如 A 的 q3 第 0 层 bias 在 +11.2 到 +11.7 之间）。softmax 没有 bias。", "sat": "router logit |z| &gt; 4 的比例（各层平均；只看 rank 0 的第一个 micro-batch）。包括 z &lt; −4（明确拒绝某个 expert，正常）。", "satpos": "z &gt; +4 的比例：分数被钉在 1，选中的 expert 之间失去区分度（坏）；红色虚线 = 预设上限 5%。只有后期日志才有。", "pinned": "选中的 8 个 expert 里分数卡在 1 附近的比例；红色虚线 = 预设上限 5%。只有后期日志才有。", "zmax": "所有层里最大的 |logit|（rank 0 的第一个 micro-batch）。", "upd": "router 权重这一步的变化量 ÷ 权重大小，各层平均（Boxiang 只作参考，不作判据）。"},
  "rt_h": "Router 选择：softmax 还是 DeepSeek-V3 式（MuonH / AdamW，新 LR 公式，128 选 8，q0 三个 seed）", "rt_note": "Boxiang 10-26 的对照实验（设计：reports/dsv3_router_experiment_design_2026-10-26.md）。把 softmax + aux loss 的 router 换成 DeepSeek-V3 式 router（sigmoid 打分 + 每个 expert 一个 bias 管负载均衡 + seq aux 1e-4），expert 数（128 选 8）等其余不变；MuonH 各组都用新拟合的 LR 公式（MUONH_FIT），AdamW 组用 BASE 自己的 LR 规则。数值 = <b>实验组 − 对照组</b>（doc-isolated，负 = 实验组更好），q0 三个 seed 逐 seed 配对后取均值 ± 标准误（括号里是配对的 seed 数）；预先定的判定线：|Δ| &lt; 0.003 视为没有差别，单 seed ±0.008。", "rt_cmp": "比较", "rt_health": "router 健康（训练日志）", "rt_h_ok": "正常", "rt_h_bad": "病态", "rt_a": "A：DSV3 router − softmax（MuonH，gate AdamH）", "rt_b2": "B2：gate 改 AdamW、router LR ×0.19 − A", "rt_b3": "B3：gate AdamH、router LR ×0.19 − A", "rt_c": "C：DSV3 router − softmax（AdamW 基线）", "rt_lead": "MuonH 领先 AdamW 的变化：(A − C) − (MUONH_FIT − BASE)", "rt_health_note": "健康状态取自 Boxiang 的 router 监测（DECISIONS 10-26 19:1x–19:2x）：A 在 q1–q3 上 expert bias 涨到 3–5（预设上限 1）、logit 失控到 60–87，负载均衡全靠 bias 撑——按预设规则不能直接采用；B2（AdamW gate）约 10% 的选中 expert 分数卡在 1 附近、bias 超过 1、logit 持续增大；B3（AdamH gate、router LR ×0.19）和 C 都健康。两种病态原因不同：A 的 gate 在 router LR ×1 下自己学不会均衡，bias 越涨越大，router LR 降到 ×0.19（B3）就健康了；B2 和 B3 的 router LR 相同，只有用 AdamW gate 的 B2 把选中的 expert 分数卡在 1，所以“卡死”来自 AdamW gate 本身（DECISIONS 10-26 17:29）。", "rt_hr_h": "更大的 rung（单 seed）", "rt_hr_pending": "q1–q3 还在训练 / 评测。", "rt_lr_h": "LR 检查（q0，overall_tsqw，相对该组 1× 三个 seed 的均值）", "rt_lr_head": ["组", "各倍数 Δloss", "最优倍数（相对该组公式）", "1× 峰值 LR", "该组的 LR 公式"], "rt_s": "S：MuonH + DSV3 router", "rt_s_lab": "A 的 LR ×0.7 / ×1 / ×1.4（横轴 = ArchScale 原版的倍数）", "rt_sc_lab": "C 的 LR ×0.5 / ×1 / ×2（横轴 = adamw_active 规则的倍数）", "rt_vs": "现用 LR（3 seed 平均）", "rt_lr_h2": "LR 检查：换成 DSV3 router 后现用 LR 还是不是最优（q0，overall_tsqw；只扫整体 LR，不是新实验组）。格子读法同 LR 网格：大数字 = 抛物线顶点 m*，鼠标移到格子上看曲线；纵轴 = 相对现用 LR（3 个 seed 平均）的 loss 差", "rt_lr_head2": ["LR 检查（q0）", "LR 扫描", "读法"], "rt_s_r": "最优 {m:.2f}×（ArchScale 原版），现用新公式 = {c:.2f}×：最优点是现用 LR 的 {r:.2f} 倍，在最优点 loss 只比现用低 {g:.4f}，现用 LR 接近最优。", "rt_sc_r": "最优 {m:.2f}×（adamw_active 规则），现用 = {c:.2f}×：最优点是现用 LR 的 {r:.2f} 倍，在最优点 loss 只比现用低 {g:.4f}，现用 LR 接近最优。", "rt_sc": "S_C：AdamW + DSV3 router", "rt_s_law": "MUONH_FIT 新公式", "rt_sc_law": "AdamW 的 adamw_active 规则",
  "dg_h": "Runlong 逐 rung 的 LR 是怎么来的（只用对角线：ladder q0–q6 的 9.6B 格子）", "dg_how": "Runlong 的生产 LR 表（Runlong.md TODO 8 A.2，step 7；目前只在 TODO 9 的 q4 一对实验上用了 5.36e-3，其余等 GO）。每一行：<b>① 该 rung 在 9.6B（4,577 步）格子实测的最优倍数 m*</b>（上表对角线，Boxiang 在 val_v2 上测）× <b>② 训练量修正 (主线步数 / 4,577)^0.03</b>（c = 0.03，范围 0–0.06，取自 LR 格子 token 方向的拟合，等效 T 指数 −0.32 + 0.03 = −0.29）= <b>③ 相对 ArchScale 原版的倍数</b>；再乘 <b>④ ArchScale 原版公式在主线训练量（TPP 200）下的 LR</b>，得到 <b>⑤ 峰值 LR</b>。⑤ 与他公布的数字相差都在 0.5% 以内。最后一列是我们自己（overall_tsqw）测的 9.6B m*，可与 ① 对照。", "dg_dhead": ["rung", "层数 L", "主线训练量 T", "主线步数", "① 9.6B 锚点 m*", "② (步数/4577)<sup>0.03</sup>", "③ 倍数 = ①×②", "④ ArchScale LR（主线 T）", "⑤ 峰值 LR = ③×④", "Runlong 公布", "我们的 9.6B m*"], "dg_note": "只用上表对角线上的 7 个 ladder 格子（宽度随层数一起变，h ≈ 43·L），训练量全是 9.6B，所以只能拟合<b>沿 ladder 放大</b>这一个方向，这里的指数不是纯深度指数，也定不出 T 指数。蓝点 = 我们实测的最优 LR（m* × ArchScale 原版 LR）；青色 = Runlong 逐 rung 生产 LR 表的锚点（Boxiang 在 val_v2 上测的 9.6B m*）。", "dg_ours": "我们拟合的（对角线，overall_tsqw）", "dg_ours_note": "（相对 ArchScale 的倍数 ∝ (L/8)^{b:+.3f}）", "dg_runlong": "Runlong 的逐 rung LR", "dg_short": "m*<sub>9.6B</sub>(rung) = 上表对角线上该 rung 实测的最优倍数（没有拟合）；0.03 是训练量的经验值。q4 已按此用 5.36e-3。", "dg_runlong_note": "；锚点 m*（q0–q6）= {anc}；c ≈ 0.03 [0, 0.06]；没有拟合，锚点直接取各 rung 实测值（未正式采用，q4 已用 5.36e-3）", "dg_pts": "格子实测最优 LR（空心 = 外推）", "dg_x": "层数 L（ladder rung；宽度随 L 一起变）", "dg_tip_o": "{q} 我们实测：m* {m:.2f}× → η* {v:.5f}（{mn}）", "dg_tip_r": "{q} Runlong 锚点：m* {m:.2f}× → η {v:.5f}（9.6B）",
  "ck_head": ["检查", "对照", "实验", "结论"], "ck_note": "每格和上面的 LR 格子一样：一个 LR 扫描，大数字 = 抛物线顶点 m*（相对 ArchScale 原版公式，按实际峰值 LR 计算），鼠标移到格子上看曲线。", "ck_floor1": "衰减到峰值的 1%（ladder）", "ck_floor10": "衰减到峰值的 10%（ArchScale）", "ck_split_ctl": "全部参数一起改 LR", "ck_split_exp": "只给一组参数 1.74×（菱形）", "ck_split_a": "A：只给 AdamW 组（embedding、1-D）", "ck_split_b": "B：hidden 矩阵 + 输出头 + router", "ck_split_c": "C：只给 hidden 矩阵", "ck_batch_r": "batch 翻倍、步数减半时最优倍数从 1.48 升到 1.86，最接近 Runlong 预先写下的“LR 按步数算”（1.81）→ 换 batch 时按步数调 LR。", "ck_floor_r": "衰减到 10% 时 ArchScale 原版几乎正好（m* ≈ 1）；ladder 衰减到 1%，需要高约 20% 的峰值 LR。", "ck_floor3_r": "q3 上同样：10% floor 时最优点往低移。", "ck_split_r": "只给 hidden 矩阵加 LR（C）就拿到几乎全部收益；只给 embedding / 1-D 参数（A）没有用。",
  "ne_h": "Router 负载均衡损失：换成负熵损失（MuonH，DeepSeek-V3 式 router，新 LR 公式；Runlong TODO 9）", "ne_note": "Runlong 当前第一优先级（arXiv 2605.15403）。把 router 的负载均衡损失（DSV3 底座：seq aux 1e-4；softmax 底座：Switch aux 1e-3）换成论文的负熵均衡损失：每层对 router 概率做 EMA（η 0.65），损失 = α·E·Σ p_e·(log m_e + 1)。DSV3 底座保留 expert bias。数值 = <b>实验组 − 对照组</b>（doc-isolated，负 = 更好），目前都是 q0 单个 seed（判定线 ±0.008；q0 三个 seed 之间的配对差约 ±0.001）。Runlong 的决定规则：在 q2 读数时，三个 seed + q1 + q2 同号、|Δ| ≥ 0.003、≥ 3 个标准误、健康不变差，才进底座。", "ne_base": "底座", "ne_pair": "对照", "ne_dsv3": "DSV3 底座（保留 bias）", "ne_nobias": "DSV3 底座，去掉 bias", "ne_soft": "softmax 底座（论文原样）",
  "rt_two_h": "两种 router 的区别", "rt_two": "<p><b>softmax router（原来的，对照组）</b></p><ul><li>对 128 个 expert 的 logit 做 softmax，取分数最高的 8 个。</li><li>负载均衡靠 aux loss（Switch 式，系数 1e-3）：某些 expert 被选得太多时，额外加一项惩罚，让 router 把 token 分散开。</li></ul><p><b>DSV3 router（DeepSeek-V3 式）</b></p><ul><li>每个 expert 的分数单独过 sigmoid，各自在 0–1 之间，互相不竞争。</li><li>选 expert 时用“分数 + bias”排序。每个 expert 有一个 bias，它不是训练出来的参数，而是每步按规则调整：负载太多的 expert，bias 减 0.001；太少的，bias 加 0.001。</li><li><b>bias 只影响选谁，不影响被选中 expert 的输出权重</b>：选出 8 个之后，权重用它们原始的 sigmoid 分数（不含 bias）在这 8 个里归一化（Megatron <code>moe_utils.py</code> L953–959）。</li><li>另外保留一个很小的 aux loss（seq aux 1e-4），按每条序列计算。</li><li>设计思路：主要靠 bias 做负载均衡，几乎不用 aux loss，避免 aux loss 干扰主任务。</li></ul>",
  "rt_gate_h": "gate（router 权重）用什么优化器、多大 LR", "rt_gate": "<p>gate 就是算 logit 的那个权重矩阵 W<sub>g</sub>（128 × 宽度）。</p><ul><li><b>AdamH</b>：MuonH 默认的做法，用 Adam 更新，但把权重约束在固定半径的球面上（hyperball），权重范数不会变大。</li><li><b>AdamW</b>：普通 AdamW，权重范数可以自由变化（B2 里 wd 设为 0）。</li><li><b>router LR ×1</b>：gate 和模型主体用同一个 LR（新公式下 q0 是 0.0164）。</li><li><b>router LR ×0.19</b>：gate 的 LR 降到约 3.1e-3，大约是 AdamW 基线自己的 LR 水平。之前的探针 B1 显示，AdamW gate 用 ×1 的 LR 会让 logit 饱和，所以才降到 0.19。</li></ul>",
  "arch_thead": ["改动", "Change ID", "具体改了什么", "对照", "LR 公式（按实际峰值 LR 判断）", "已有结果"], "arch_law": {"fit": "新公式（fit v2）", "arch": "旧公式（ArchScale 原版）", "other": "其他"},
  "arch_short": {"SWA_FIT": "SWA（基础版）", "SWA_FIT_ROPE": "SWA，全注意力层用 RoPE", "SWA_FIT_W4K": "SWA，窗口 4096", "SWA_FIT_W1K": "SWA，窗口 1024", "SWA_FIT_SINK": "SWA + attention sink", "SWA_FIT_SINK_W1K": "SWA + sink + 窗口 1024", "SWA_FIT_DSV3": "SWA（DSV3 router 底座）", "MUONH_SPLITFC1": "MuonH 拆分 fc1"},
  "arch_desc": {"SWA_FIT": "每 6 层中 5 层滑动窗口注意力（窗口 2048，RoPE）+ 1 层全注意力（NoPE）", "SWA_FIT_ROPE": "同基础版，但全注意力层也用 RoPE", "SWA_FIT_W4K": "同基础版，窗口 4096", "SWA_FIT_W1K": "同基础版，窗口 1024", "SWA_FIT_SINK": "同基础版 + 可学习 attention sink（gpt-oss 式：每层每个 query head 一个可学习的 sink logit）", "SWA_FIT_SINK_W1K": "attention sink + 窗口 1024", "SWA_FIT_DSV3": "基础版 SWA，但底座换成 DeepSeek-V3 式 router（对照也是 DSV3 底座）", "MUONH_SPLITFC1": "优化器改动（不是注意力）：expert 的 fc1 把 gate 和 up 拆成两个矩阵，各自做 Newton-Schulz 正交化、各自的缩放和球面半径（原来当作一个拼接矩阵）"},
  "rt_left_h": "MuonH 主体：各组 − 对照 ①（softmax router，MUONH_FIT）", "rt_right_h": "AdamW 主体：C − 对照 ②（softmax router，BASE）", "rt_l_a": "A：DSV3 router", "rt_l_b2": "B2：DSV3 + AdamW gate，router LR ×0.19", "rt_l_b3": "B3：DSV3 + AdamH gate，router LR ×0.19", "rt_l_c": "C：DSV3 router",
  "ol_h": "AdamW 的 LR 检查（只有 3 个点，不是格子；不在计划里）", "ol_opt": "优化器", "ol_note": "AdamW 和 Muon 都用 AdamW 的 LR 规则；这里只在个别 rung 上测了规则的 0.5× / 1× / 2×（TODO 3 step 5 的公平性检查），用来和上面 MuonH 的格子对照。数值 = 相对 1× 的 {m} 差（负数 = 比 1× 好）；m* = 3 点抛物线的最优倍数，η* = 对应的绝对 LR。",
  "lf_h": "拟合：最优 LR 随层数和训练量的关系（ArchScale 形式）", "lf_arch": "ArchScale 原版（= 每个格子的 1×，ladder 现用的 LR）", "lf_ours": "我们拟合的", "lf_fix": "我们拟合的，T 指数固定 −0.32", "lf_box": "Boxiang 的拟合 v2（10-25 冻结，MUONH_FIT 在用；val_v2、16 格）", "lf_pts": "格子实测最优 LR（空心 = 外推）", "lf_drag": "拖动旋转 · 悬停看格子", "lf_axL": "层数 L", "lf_axT": "训练 token T", "lf_note": "用格子里 {n} 个格子的最优绝对 LR，在 log 空间做最小二乘：log η* = log η0 + β·log(L/8) + γ·log(T/10.4B)。最优点取抛物线顶点；顶点在已测倍数之外的格子（{excl}）用外推估计，也参与拟合，表中标「外推」；外推超过已测范围 2 倍的不用。±为标准误差。",
  "lf_cells": "每个格子：实测 vs 拟合", "lf_meas": "实测", "lf_fitted": "拟合", "lf_res": "残差", "lf_ext": "外推（只作参考）", "lf_ratio": "拟合 / ArchScale",
  "lf_pred": ["q2 主线（TPP 200）", "q3 主线（TPP 200）", "q6（30B-A3B）", "235B 目标"],
  "lf_warn": "q2 / q3 主线是对格子已覆盖范围附近的预测；q6 和 235B 远超拟合范围（最深 30 层、最多 86B token），只说明方向，不能直接拿来用。另外目前单 seed，深度指数在 9.6B 和 30.8B 两列上不完全一致（约 −0.15 和 −0.06），可分离性还要更多格子确认。",
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
  "muon_fit": "Muon (new fitted LR)", "ce_fit": "MuonH (new fitted LR, MUONH_FIT)", "opt_h": "Optimizers: AdamW / Muon / MuonH (all CE)", "opt_note": "Dashed = the same optimizer at the new fitted LR law of Boxiang (MuonH: fit v2 η = 0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259, q2 / q3 so far; no Muon run at a fitted law). Same data and token count; only the optimizer (with its LR rule / schedule) changes. x: active non-embedding parameters (log scale).",
  "o2_h": "Δloss vs AdamW (optimizer − AdamW, more negative is better)",
  "kd_h": "KD vs CE with the optimizer fixed to MuonH", "kd_note": "CE and KD differ only in the loss (α = 0 vs 0.7 / 0.8 / 0.9); optimizer, data order and layout are identical.",
  "k2_h": "KD − CE: Δloss per α (more negative is better)",
  "adamw": "AdamW", "muon": "Muon", "vs_adamw": "Δloss (optimizer − AdamW, more negative is better)", "mean6": "mean of 6",
  "v_h": "KD loss variants compared: same α 0.8, same MuonH", "v_note": "The 8 runs differ only in how the teacher distribution is truncated in the KD term: α = 0.8, MuonH optimizer, seed 1 and data order are identical; the baseline is still <b>CE</b> (MuonH, α = 0); <b>B64</b> (top-64 + the rest of the tail as one bucket) is KD α 0.8 itself and is drawn as a line too, so each variant can be read against it. Negative = better than CE.",
  "v_chart": "Δloss (KD variant − CE), lower is better", "v_table": "CE loss, and B64 and each variant's difference from CE", "v_gate": "gate −0.008",
  "vdesc": {"B64": "top-64 + bucket (= KD α 0.8)", "S64Y": "top-64 + sampled tail + y exact", "S64": "top-64 + sampled tail (4 draws)", "S128": "top-128 + sampled tail", "S128Y": "top-128 + sampled tail + y exact", "B64Y": "top-64 bucket + y exact",
            "B128": "top-128 + bucket", "DD64": "top-64 + data draw", "DROP64": "top-64, tail dropped (control)"},
  "lr_h": "MuonH learning-rate test: same MuonH, CE", "lr_note": "Only MuonH's peak LR changes (the HyperP law's LR × 0.5 / 0.7 / 1.4 / 2); everything else is identical; 1× = the main MuonH (CE) run. One line per rung; its lowest point is that rung's best LR.",
  "lr_chart": "Δloss (this LR − 1×), lower is better", "lr_table": "1× loss, and every other LR's difference from 1×", "lr_x": "peak LR / HyperP law (log scale)",
  "find_v": "KD loss variants (same α 0.8, MuonH): the sampled-tail variants beat B64 consistently, but the lead shrinks with scale (overall_tw, q0 → q2: {sampled}); the bucket variants are ≈ 0 ({bucket}).",
  "find_lr": "MuonH learning rate (overall_tsqw, parabola-fitted best multiple m*, * = vertex outside the sampled multiples, extrapolated): {best}.",
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
  "g_h": "MuonH learning rate: the best LR in each cell (same MuonH, CE)", "g_note": "Each cell = one fixed (model size, training tokens). In a cell, one model per 0.5 / 0.7 / 1 / 1.2 / 1.4 / 2 (/ 2.8 / 4) × the HyperP-law LR; y = loss difference to the cell's 1× (lower is better); the big number = the fitted best multiple <b>m*</b> (the parabola vertex, possibly extrapolated beyond the sampled multiples; m* &gt; 1 means the law is too low), below it the absolute LR <b>η*</b> (in parentheses the cell's 1× LR, i.e. the HyperP-law LR for its depth and training tokens) and how much the optimum beats 1×; the main-ladder TPP 200 cells run along the diagonal; <b>hover a cell to see its LR sweep curve</b> (dashed = parabola in log(multiple), orange = m*). Read a column for depth, the q0 row for training length. <b>Cells have different budgets: compare only within a cell.</b> Blank = not in the design.",
  "g_cols": {"9.6": "9.6B tokens", "30": "~30B tokens", "71": "71.1B tokens", "86": "86.4B tokens", "137": "136.6B tokens"}, "g_layers": "layers",
  "g_kind": {"main": "main ladder, TPP {tpp}", "main_x": "main ladder, TPP {tpp} (not in TODO 3)", "tpp": "{tok}, TPP {tpp}", "it": "{tok}, TPP {tpp} (IT4577)", "it2": "{tok}, TPP {tpp}"},
  "g_ref": "Every multiple is relative to the ORIGINAL ArchScale law η = 0.016 × (L/8)^−0.5 × (T/10.4B)^−0.32, computed from each run's actual peak LR (not from its Change ID's name), so it does not move if the ladder adopts a new law.", "g_ref_muon": "Multiples are relative to each rung's current (fixed) Muon LR, computed from each run's actual peak LR.", "g_mm": "Actual multiple differs from the Change ID's name:", "g_pending": "pending", "g_fewpts": "too few points", "g_edge": "(extrapolated)", "g_npts": "{n} points", "g_nseed": " ({s} seeds)", "g_seedal": "aligned by seed offset",
  "sd_h": "Seed noise: the same configuration with different seeds (doc-isolated)", "sd_note": "Same model and recipe, only the seed changes (seed k has the same data order across arms, so Muon − AdamW is paired by seed). <b>Range</b> = max − min over the three seeds, <b>sd</b> = standard deviation. Use it to judge whether the differences in the other sections exceed the noise.",
  "sd_pair": "Muon − AdamW (paired by seed)", "sd_range": "range",
  "arch_y": "Δloss (change − control, more negative is better)", "arch_chart": "Δloss of each change vs its own control (one seed; all at the fitted LR law; hover a rung for all values)", "arch_h": "Sliding-window attention, SWA (same MuonH, CE, fitted LR law fit v2; 128-choose-8 MoE, softmax router except the DSV3-base row; q0-q3, one seed)", "arch_note": "The settings of each change are in the table below; values in the chart = change − control (doc-isolated, negative = the change is better); control and change both at the fitted LR law (fit v2).",
  "arch_name": {"SWA": "SWA − MuonH (old LR law)", "SWA_FIT": "SWA − no SWA (MUONH_FIT)", "SWA_FIT_ROPE": "SWA, RoPE in the full-attention layer − MUONH_FIT", "SWA_FIT_W4K": "SWA, window 4096 − MUONH_FIT", "SWA_FIT_W1K": "SWA, window 1024 − MUONH_FIT", "SWA_FIT_SINK": "SWA + learnable attention sinks (gpt-oss style) − MUONH_FIT", "SWA_FIT_SINK_W1K": "SWA + sinks + window 1024 − MUONH_FIT", "SWA_FIT_DSV3": "SWA (DSV3-router base) − same base without SWA (MUONH_FIT_DSV3)", "MUONH_SPLITFC1": "MuonH with split fc1 (gate / up stepped separately) − MuonH"},
  "mg_g_h": "Muon learning rate: the same cells (same Muon, CE)", "mg_g_note": "Boxiang's full Muon grid (decided 10-24): the MuonH cells above, multiples 0.25 / 0.5 / 1 / 2 / 4× (coarser, since Muon's optimum is unknown). 1× = the ladder's Muon LR (the AdamW rule: depends on model size only, <b>not on training tokens</b>). Read as above: big number = parabola vertex m*, η* = absolute LR, hover a cell for its curve. Purpose: does MuonH still lead when Muon also runs at its own best LR?", "mg_lf_h": "Fit: Muon best LR vs active parameters N and training tokens (same form as the Muon adamw_active rule)", "mg_lf_arch": "current Muon / AdamW LR (the ladder adamw_active rule, not ArchScale; every cell 1×)", "mg_lf_box": "Boxiang Muon fit v3 (on record 10-26; val_v2, 9 cells / 43 points; not used by any run; domain L 12-30, T 9.6-137B)", "mg_lf_axN": "active params N", "mg_noT": " (depends on active parameters N only, no T term)",
  "ref_dag": "† Own tokenizer: val_loss / ppl are per ITS token and not comparable with the rows per Qwen3 token (use bpb). bpb = total NLL / (UTF-8 bytes of the original text × ln 2), the same denominator as the ladder; text = the originals in ~/data/eval_ppl_decon, same chunking as the ladder (first token of a document not scored, BOS as context only).", "ref_h": "External reference: MiMo / DeepSeek / Kimi (same doc-isolated protocol)", "ref_gap": "our best − MiMo-V2.5-Pro-Base", "ref_best": "our best", "ref_note": "MiMo-V2.5-Pro-Base (fp8) and MiMo-V2.6-Pro-RL (the released mxfp4 weights; also the KD teacher), scored on exactly the ladder's doc-isolated 115 splits (vLLM prompt_logprobs; every split's predicted-token count equals the ladder's). MiMo's token ids are the ladder's Qwen3 ids, so per-token losses compare directly. Absolute losses (lower is better); only our best ladder checkpoint (lowest overall_tsqw of all evaluated checkpoints); last row = it minus MiMo-Base (ppl: ratio). The buttons switch val_loss / ppl / bpb. The RL model is ~0.05 above Base on pretraining text, as expected (RL post-training + quantisation).",
  "aw_h": "Checking the LR law's assumptions: depth × width, diagonal used as ArchScale anchors (same MuonH, CE, 9.6B tokens)", "aw_note": "Runlong.md TODO 3 steps 27, 30, 35: training tokens fixed at 9.6B (4,577 steps, GBS 2M); an LR sweep (multiples of the ArchScale law) at several depths L and widths h, to see whether the best multiple m* depends on depth only (the law assumes no width term). Diagonal = the ladder rungs (q0-q6); other cells = extra depth / width points. Metric {m}; read the cells as above (hover for the curve). Note: 9.6B is heavily under-trained for the large models (q6 TPP 3.5).", "aw_ladder": "ladder {q}", "aw_extra": "extra cell", "aw_unstable": "router collapse (load-balancing loss 9-16): not an LR point", "aw_no1x": " (no 1×: relative to the smallest multiple)", "ac_h": "Other checks: batch, LR floor, per-group LR", "ac_head": ["check", "experiment", "reference", "reading"], "ac_batch": "batch (q1, L 18, 9.6B)", "ac_batch_r": "Runlong's pre-registered predictions: law in tokens 1.45 / in steps 1.81 / B^0.56 2.14 / sqrt-noise 2.56", "ac_floor0": "LR floor (q0, L 12)", "ac_floor_r": "MuonH: no warmup, linear decay from the start to X% of the peak (not WSD); 10% = the ArchScale setting, the ladder uses 1%", "ac_floor3": "LR floor (q3, L 30)", "ac_floor3_r": "vs the same-floor 1×; negative = better than 1×", "ac_split": "per-group LR (q3, L 30)", "ac_split_r": "vs 1×: A = AdamW groups only (embedding, 1-D) at 1.74×; B = hidden matrices + LM head + router at 1.74×; C = hidden matrices only at 1.74×",
  "rt_set_head": ["arm", "Change ID", "main optimizer", "router score", "load balance", "gate (router weight) optimizer", "router LR", "LR rule / schedule", "partner", "rungs run"], "rt_set_rows": [["partner ①", "MUONH_FIT", "MuonH", "softmax", "aux loss 1e-3", "AdamH (sphere)", "×1", "fitted law; no warmup, linear decay to 1 %, wd 0", "—", "q0 ×3 seeds, q1-q3"], ["<b>A</b>", "MUONH_FIT_DSV3", "MuonH", "<b>sigmoid</b>", "<b>expert bias</b> + seq aux 1e-4", "AdamH", "×1", "as ①", "①", "q0 ×3, q1-q3"], ["<b>B2</b>", "…_DSV3_RTADAMW_RT0P19", "MuonH", "sigmoid", "as A", "<b>AdamW</b> (wd 0)", "<b>×0.19</b>", "as ①", "A", "q0 ×3, q1"], ["<b>B3</b>", "…_DSV3_RT0P19", "MuonH", "sigmoid", "as A", "AdamH", "<b>×0.19</b>", "as ①", "A", "q0 ×3"], ["partner ②", "BASE", "AdamW", "softmax", "aux loss 1e-3", "AdamW", "×1", "adamw_active; WSD (5 % warmup, last 20 % decay), wd 0.1", "—", "q0 ×3, q1-q5"], ["<b>C</b>", "BASE_DSV3_ONETREE", "AdamW", "<b>sigmoid</b>", "<b>expert bias</b> + seq aux 1e-4", "AdamW", "×1", "as ②", "②", "q0 ×3, q1, q2"], ["S", "MUONH_FIT_DSV3_LR0P7X / 1P4X", "as A, whole LR ×0.7 / ×1.4", "", "", "", "", "", "A", "q0"], ["S_C", "BASE_DSV3_ONETREE_LR0P5X / 2X", "as C, whole LR ×0.5 / ×2", "", "", "", "", "", "C", "q0"]], "rt_set_l": "MuonH body (control 1: MUONH_FIT, softmax router)", "rt_set_r": "AdamW body (control 2: BASE, softmax router)", "rt_set_note": "All arms: 128 experts choose 8, no shared expert, batch 2M tokens; each arm changes only the bold items: A and C relative to the control of their table, B2 and B3 change the gate on top of A. Router LR ×1 = the main LR (fitted law 0.0164 at q0), ×0.19 ≈ 3.1e-3. ONETREE = the same code tree as A, so C − BASE crosses code trees.", "rt_chart_h": "Results: loss difference arm − partner (overall_tsqw, more negative is better). q0 has 3 seeds: hollow points = each seed paired difference, filled point and line = the 3-seed mean; other rungs 1 seed", "rt_ylab": "Δloss (arm − partner)", "rt_lr_note": "LR checks (q0): MuonH + DSV3 has its best multiple at {s}× the fitted law, AdamW + DSV3 at {c}× adamw_active; both current LRs are near the optimum.",
  "rt_hm_h": "Per-layer load balance (training logs, seed 1)", "rt_hm_note": "From the router monitor of Boxiang, <code>router_health.py</code> (<code>LADDER_ROUTER_HEALTH=100</code>, every 100 steps). Each panel: y = layer (layer 0 on top), x = training step (at most 60 columns, neighbouring reports averaged); darker = larger; above each panel the layer mean at the end of training; hover for values. Metrics: <b>MaxVio</b> = tokens routed to each expert of each layer over a whole training step (all micro-batches, all ranks): the busiest expert / the mean - 1 (the DeepSeek-V3 measure; 0 = perfectly balanced, 2.86 = the busiest gets 3.86x the mean; colour scale capped at 4); <b>dead</b> = experts below 10 % of the mean load, as % of the 128; <b>|bias| max</b> = the largest absolute DSV3 per-expert bias (pre-registered health limit 1); <b>bias spread</b> = max - min of the bias within a layer - top-8 selection only sees bias differences, so a common shift of all experts (large |bias|, small spread) does not change routing; <b>logit saturation</b> = share of router logits with |z| &gt; 4 and <b>max |logit|</b> = the largest |z|, both on rank 0 first micro-batch only; |z| &gt; 4 includes z &lt; -4 (an expert confidently rejected, normal); <b>gate update</b> = the change of the router weight over the step / its norm (a flag, not a criterion). The softmax router has no bias; BASE (softmax AdamW) and MUONH_FIT q2 / q3 ran without the monitor. <b>Note:</b> the training logs cut every line at ~1000 characters, so the per-layer line is incomplete and the last layers of the DSV3 arms are missing (grey; q0 1 layer, q1 4, q3 16) - lost at the source.", "rt_hm_vio": "MaxVio", "rt_hm_dead": "dead (%)", "rt_hm_bmax": "|bias| max", "rt_hm_bspread": "bias spread", "rt_hm_sat": "logit saturation (%)", "rt_hm_zmax": "max |logit|", "rt_hm_upd": "gate update", "rt_hm_nobias": "softmax: no bias", "rt_hm_miss": "{n} layers missing", "rt_hm_na": "no monitor data", "rt_hm_end": "end, layer mean", "rt_hm_layers": "layer ↓ · step →", "rt_hm_soft": "① softmax (MUONH_FIT)", "rt_hm_a": "A: DSV3", "rt_hm_b2": "B2: AdamW gate ×0.19", "rt_hm_b3": "B3: AdamH gate ×0.19", "rt_hm_c": "C: AdamW + DSV3",
  "rt_st_h": "Router health statistics (training logs, seed 1)", "rt_st_note": "Summary lines (all layers) of the router monitor of Boxiang, <code>router_health.py</code> (every 100 steps). One chart per rung, x = training step, one line per arm; the buttons switch the statistic, hover for the values. BASE (softmax AdamW) and MUONH_FIT q2 / q3 ran without the monitor.", "rt_st_name": {"vio_mean": "MaxVio (layer mean)", "vio_max": "MaxVio (worst layer)", "dead": "dead experts (%)", "bmax": "|bias| max", "sat": "logit saturation (%)", "satpos": "z > +4 (%)", "pinned": "top-8 pinned (%)", "zmax": "max |logit|", "upd": "gate update"}, "rt_st_def": {"vio_mean": "Per layer: tokens routed to the busiest expert over a whole training step (all micro-batches, all ranks) / the mean - 1 (DeepSeek-V3 MaxVio, 0 = perfectly balanced); here the mean over layers.", "vio_max": "As above, the least balanced layer.", "dead": "Experts below 10 % of the mean load, as % of (layers × 128).", "bmax": "The largest absolute DSV3 per-expert bias over all layers; red dashed = the pre-registered health limit 1. Top-8 selection only sees bias differences within a layer, so a common shift of all experts does not change routing (e.g. A q3 layer 0: bias between +11.2 and +11.7). The softmax router has no bias.", "sat": "Share of router logits with |z| &gt; 4 (mean over layers; rank 0 first micro-batch). Includes z &lt; -4 (an expert confidently rejected, normal).", "satpos": "Share with z &gt; +4: scores pinned at 1, the selected experts lose discrimination (bad); red dashed = the 5 % limit. Later logs only.", "pinned": "Share of the selected 8 experts whose score is pinned near 1; red dashed = the 5 % limit. Later logs only.", "zmax": "The largest |logit| over all layers (rank 0 first micro-batch).", "upd": "Change of the router weight over the step / its norm, mean over layers (a flag, not a criterion)."},
  "rt_h": "Router choice: softmax or DeepSeek-V3 style (MuonH / AdamW, fitted LR law, 128-choose-8, q0 three seeds)", "rt_note": "Boxiang's controlled experiment of 10-26 (design: reports/dsv3_router_experiment_design_2026-10-26.md). The softmax + aux-loss router is replaced by the DeepSeek-V3 router (sigmoid scores + a per-expert bias for load balance + seq aux 1e-4); 128-choose-8 experts and everything else unchanged; the MuonH arms use the fitted LR law (MUONH_FIT), the AdamW arm BASE's own LR rule. Values = <b>arm − partner</b> (doc-isolated, negative = the arm is better); at q0 three seeds paired per seed, mean ± se (number of paired seeds in brackets); pre-registered lines: |Δ| &lt; 0.003 = no difference, single seed ±0.008.", "rt_cmp": "comparison", "rt_health": "router health (training log)", "rt_h_ok": "healthy", "rt_h_bad": "pathological", "rt_a": "A: DSV3 router − softmax (MuonH, AdamH gate)", "rt_b2": "B2: AdamW gate, router LR ×0.19 − A", "rt_b3": "B3: AdamH gate, router LR ×0.19 − A", "rt_c": "C: DSV3 router − softmax (AdamW baseline)", "rt_lead": "change of the MuonH lead over AdamW: (A − C) − (MUONH_FIT − BASE)", "rt_health_note": "Health from the router monitor of Boxiang (DECISIONS 10-26 19:1x-19:2x): A at q1-q3 has expert bias 3-5 (pre-registered limit 1) and runaway logits 60-87, load balanced only by the bias - not adoptable as is under the pre-registered rule; B2 (AdamW gate) pins ~10 % of the selected experts' scores near 1, bias above 1, logits rising; B3 (AdamH gate, router LR ×0.19) and C are healthy. The two pathologies have different causes: at router LR ×1 the AdamH gate of A does not learn the balance itself and the bias keeps growing, which router LR ×0.19 (B3) fixes; B2 and B3 share the router LR and only B2, with the AdamW gate, pins the selected scores near 1, so the pinning comes from the AdamW gate itself (DECISIONS 10-26 17:29).", "rt_hr_h": "Larger rungs (seed 1)", "rt_hr_pending": "q1-q3 still training / being evaluated.", "rt_lr_h": "LR checks (q0, overall_tsqw, vs the arm's 3-seed 1× mean)", "rt_lr_head": ["arm", "Δloss per multiple", "best multiple (of the arm's law)", "1× peak LR", "the arm's LR law"], "rt_s": "S: MuonH + DSV3 router", "rt_s_lab": "LR of A ×0.7 / ×1 / ×1.4 (x axis = multiples of the ArchScale original)", "rt_sc_lab": "LR of C ×0.5 / ×1 / ×2 (x axis = multiples of the adamw_active rule)", "rt_vs": "current LR (3-seed mean)", "rt_lr_h2": "LR check: is the current LR still optimal after the switch to the DSV3 router (q0, overall_tsqw; whole-LR sweeps only, not new arms). Read as the LR grids: big number = parabola vertex m*, hover a cell for its curve; y = loss difference vs the current LR (3-seed mean)", "rt_lr_head2": ["LR check (q0)", "LR sweep", "reading"], "rt_s_r": "best {m:.2f}× (ArchScale original), current fitted law = {c:.2f}×: the optimum is {r:.2f}× the current LR and only {g:.4f} below it in loss; the current LR is near optimal.", "rt_sc_r": "best {m:.2f}× (adamw_active rule), current = {c:.2f}×: the optimum is {r:.2f}× the current LR and only {g:.4f} below it in loss; the current LR is near optimal.", "rt_sc": "S_C: AdamW + DSV3 router", "rt_s_law": "MUONH_FIT fitted law", "rt_sc_law": "AdamW adamw_active rule",
  "dg_h": "How the per-rung LR of Runlong is derived (diagonal only: the 9.6B cells of ladder q0-q6)", "dg_how": "The production LR table of Runlong (Runlong.md TODO 8 A.2, step 7; so far used only by the TODO 9 q4 pair at 5.36e-3, the rest awaits GO). Per row: <b>① the rung best multiple m* measured on its 9.6B (4,577-step) cell</b> (the diagonal above, measured by Boxiang on val_v2) × <b>② the token correction (main-ladder steps / 4,577)^0.03</b> (c = 0.03, range 0-0.06, from the token-direction fits of the LR grid; effective T exponent −0.32 + 0.03 = −0.29) = <b>③ the multiple of the ArchScale original</b>; times <b>④ the ArchScale original LR at the main-ladder tokens (TPP 200)</b> gives <b>⑤ the peak LR</b>. ⑤ matches his published numbers within 0.5 %. Last column: our own 9.6B m* (overall_tsqw), to compare with ①.", "dg_dhead": ["rung", "layers L", "main tokens T", "main steps", "① 9.6B anchor m*", "② (steps/4577)<sup>0.03</sup>", "③ multiple = ①×②", "④ ArchScale LR (main T)", "⑤ peak LR = ③×④", "published", "our 9.6B m*"], "dg_note": "Only the 7 ladder cells on the diagonal of the table above (width grows with depth, h ≈ 43·L), all at 9.6B tokens: this fits the <b>along-ladder</b> direction only - not a pure depth exponent, and no token exponent. Blue = our measured best LR (m* × the ArchScale original LR); cyan = the anchors of the per-rung production LR table of Runlong (the 9.6B m* measured by Boxiang on val_v2).", "dg_ours": "our fit (diagonal, overall_tsqw)", "dg_ours_note": " (multiple of ArchScale ∝ (L/8)^{b:+.3f})", "dg_runlong": "per-rung LR of Runlong", "dg_short": "m*<sub>9.6B</sub>(rung) = the measured best multiple of that rung on the diagonal above (no fit); 0.03 is an empirical token value. q4 already runs at 5.36e-3 by it.", "dg_runlong_note": "; anchors m* (q0-q6) = {anc}; c ≈ 0.03 [0, 0.06]; no fit: the anchors are the measured values of each rung (not adopted yet; q4 already runs at 5.36e-3)", "dg_pts": "measured best LR per cell (hollow = extrapolated)", "dg_x": "layers L (ladder rung; width grows with L)", "dg_tip_o": "{q} ours: m* {m:.2f}× → η* {v:.5f} ({mn})", "dg_tip_r": "{q} Runlong anchor: m* {m:.2f}× → η {v:.5f} (9.6B)",
  "ck_head": ["check", "control", "experiment", "reading"], "ck_note": "Each cell as in the LR grid above: one LR sweep, big number = parabola vertex m* (multiple of the ORIGINAL ArchScale law, from the actual peak LR); hover a cell for its curve.", "ck_floor1": "decay to 1% of peak (ladder)", "ck_floor10": "decay to 10% of peak (ArchScale)", "ck_split_ctl": "all parameters at the multiple", "ck_split_exp": "one group at 1.74× (diamonds)", "ck_split_a": "A: AdamW groups only (embedding, 1-D)", "ck_split_b": "B: hidden matrices + head + router", "ck_split_c": "C: hidden matrices only", "ck_batch_r": "Doubling the batch (half the steps) moves m* from 1.48 to 1.86, closest to the pre-registered law-in-steps prediction of Runlong (1.81): scale the LR by steps when the batch changes.", "ck_floor_r": "With a 10% floor the ArchScale original is about right (m* ≈ 1); the ladder decays to 1% and wants ~20% higher peak LR.", "ck_floor3_r": "Same at q3: the 10% floor moves the optimum down.", "ck_split_r": "The hidden matrices alone (C) get almost all of the gain; the embedding / 1-D groups alone (A) get nothing.",
  "ne_h": "Router load-balancing loss: negative-entropy loss instead (MuonH, DeepSeek-V3-style router, fitted LR law; Runlong TODO 9)", "ne_note": "Priority 1 of Runlong now (arXiv 2605.15403). The router load-balancing loss (DSV3 base: seq aux 1e-4; softmax base: Switch aux 1e-3) is replaced by the paper's negative-entropy balancing loss: per layer an EMA (η 0.65) of the router probabilities m, loss = α·E·Σ p_e·(log m_e + 1). The DSV3 base keeps its expert bias. Values = <b>arm − pair</b> (doc-isolated, negative = better), q0 seed 1 so far (single-seed line ±0.008; paired q0 seed scatter ~±0.001). Decision rule of Runlong at the q2 reading: three seeds + q1 + q2 same sign, |Δ| ≥ 0.003, ≥ 3 se, health no worse.", "ne_base": "base", "ne_pair": "pair", "ne_dsv3": "DSV3 base (bias kept)", "ne_nobias": "DSV3 base, no bias", "ne_soft": "softmax base (paper-faithful)",
  "rt_two_h": "The two routers", "rt_two": "<p><b>softmax router (the original, control)</b></p><ul><li>Softmax over the logits of the 128 experts; the 8 highest scores are picked.</li><li>Load balance by an aux loss (Switch style, coefficient 1e-3): when some experts are picked too often, an extra penalty pushes the router to spread the tokens.</li></ul><p><b>DSV3 router (DeepSeek-V3 style)</b></p><ul><li>Each expert score goes through its own sigmoid, each in 0-1, not competing with the others.</li><li>Experts are picked by “score + bias”. Every expert has a bias that is not a trained parameter but adjusted by a rule every step: an overloaded expert gets bias − 0.001, an underloaded one + 0.001.</li><li><b>The bias only decides which experts are picked, not their output weights</b>: after the 8 are picked, their weights are their original sigmoid scores (without the bias), normalised over those 8 (Megatron <code>moe_utils.py</code> L953-959).</li><li>A small aux loss is kept as well (seq aux 1e-4), computed per sequence.</li><li>Design idea: balance mainly by the bias and hardly by an aux loss, so the aux loss does not interfere with the main task.</li></ul>",
  "rt_gate_h": "Which optimizer and LR for the gate (router weights)", "rt_gate": "<p>The gate is the weight matrix W<sub>g</sub> (128 × width) that computes the logits.</p><ul><li><b>AdamH</b>: the MuonH default - Adam updates, but the weights are kept on a sphere of fixed radius (hyperball); the weight norm cannot grow.</li><li><b>AdamW</b>: plain AdamW; the weight norm is free (wd 0 in B2).</li><li><b>router LR ×1</b>: the gate uses the same LR as the body (0.0164 at q0 under the fitted law).</li><li><b>router LR ×0.19</b>: the gate LR is lowered to ~3.1e-3, about the AdamW baseline own LR level. The earlier probe B1 showed that an AdamW gate at ×1 saturates the logits, hence 0.19.</li></ul>",
  "arch_thead": ["change", "Change ID", "what changes", "control", "LR law (from the actual peak LR)", "results at"], "arch_law": {"fit": "fitted law (fit v2)", "arch": "old law (ArchScale original)", "other": "other"},
  "arch_short": {"SWA_FIT": "SWA (base)", "SWA_FIT_ROPE": "SWA, RoPE in the full-attention layers", "SWA_FIT_W4K": "SWA, window 4096", "SWA_FIT_W1K": "SWA, window 1024", "SWA_FIT_SINK": "SWA + attention sinks", "SWA_FIT_SINK_W1K": "SWA + sinks + window 1024", "SWA_FIT_DSV3": "SWA (DSV3-router base)", "MUONH_SPLITFC1": "MuonH split fc1"},
  "arch_desc": {"SWA_FIT": "5 of every 6 layers sliding-window attention (window 2048, RoPE) + 1 full-attention layer (NoPE)", "SWA_FIT_ROPE": "as the base, but RoPE in the full-attention layers too", "SWA_FIT_W4K": "as the base, window 4096", "SWA_FIT_W1K": "as the base, window 1024", "SWA_FIT_SINK": "as the base + learnable attention sinks (gpt-oss style: one learnable sink logit per query head per layer)", "SWA_FIT_SINK_W1K": "attention sinks + window 1024", "SWA_FIT_DSV3": "base SWA on the DeepSeek-V3-style router base (control on the DSV3 base too)", "MUONH_SPLITFC1": "optimizer change (not attention): the expert fc1 is stepped as two matrices, gate and up, each with its own Newton-Schulz, scale and sphere radius (was one concatenated matrix)"},
  "rt_left_h": "MuonH body: each arm − control 1 (softmax router, MUONH_FIT)", "rt_right_h": "AdamW body: C − control 2 (softmax router, BASE)", "rt_l_a": "A: DSV3 router", "rt_l_b2": "B2: DSV3 + AdamW gate, router LR ×0.19", "rt_l_b3": "B3: DSV3 + AdamH gate, router LR ×0.19", "rt_l_c": "C: DSV3 router",
  "ol_h": "AdamW LR check (3 points only, not a grid; not in the plan)", "ol_opt": "optimizer", "ol_note": "AdamW and Muon both use the AdamW LR law; it was checked only at a few rungs at 0.5× / 1× / 2× (TODO 3 step 5, fairness), to set against the MuonH grid above. Values = {m} difference to 1× (negative = better than 1×); m* = best multiple of the 3-point parabola, η* = the absolute LR.",
  "lf_h": "Fit: best LR vs depth and training tokens (ArchScale form)", "lf_arch": "ArchScale original (= every cell's 1×, the LR the ladder uses)", "lf_ours": "Our fit", "lf_fix": "our fit, T exponent fixed at −0.32", "lf_box": "Boxiang's fit v2 (frozen 10-25, used by MUONH_FIT; val_v2, 16 cells)", "lf_pts": "measured best LR per cell (hollow = extrapolated)", "lf_drag": "drag to rotate · hover a point", "lf_axL": "layers L", "lf_axT": "training tokens T", "lf_note": "Least squares in log space over the best absolute LR of {n} grid cells: log η* = log η0 + β·log(L/8) + γ·log(T/10.4B). The optimum is the parabola vertex; cells whose vertex lies outside the sampled multiples ({excl}) use the extrapolated vertex and are included, marked as extrapolated; vertices more than 2× beyond the sweep are not used. ± = standard error.",
  "lf_cells": "Per cell: measured vs fitted", "lf_meas": "measured", "lf_fitted": "fitted", "lf_res": "residual", "lf_ext": "Extrapolation (indicative only)", "lf_ratio": "fitted / ArchScale",
  "lf_pred": ["q2 main (TPP 200)", "q3 main (TPP 200)", "q6 (30B-A3B)", "235B target"],
  "lf_warn": "q2 / q3 main are predictions near the covered range; q6 and 235B are far outside it (at most 30 layers and 86B tokens fitted) - direction only, not for use as is. One seed; the depth exponent differs between the 9.6B and 30.8B columns (about −0.15 vs −0.06), so separability needs more cells.",
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
 --grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#8a5cd6;--s5:#c49a00;--good:#006300;
 --o-adamw:#6250d6;--o-muon:#1baf7a;--a07:#f4a27f;--a08:#eb6834;--a09:#b4471d;
 --v1:#2a78d6;--v2:#1baf7a;--v3:#eda100;--v4:#e87ba4;--v5:#008300;--v6:#6250d6;--v7:#e34948;--v8:#52514e;
 --c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--c4:#eda100;--c5:#e87ba4;--c6:#008300}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;
 --text-primary:#f0efec;--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#9d75e0;--s5:#d4ab1e;--good:#0ca30c;
 --o-adamw:#9085e9;--o-muon:#199e70;--a07:#f0a47f;--a08:#d95926;--a09:#a8461f;
 --v1:#3987e5;--v2:#199e70;--v3:#c98500;--v4:#d55181;--v5:#008300;--v6:#9085e9;--v7:#e66767;--v8:#c3c2b7;
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
.rtoggle{display:flex;gap:6px;margin:0 0 10px}.rtoggle button{font:inherit;font-size:13px;padding:4px 12px;border-radius:999px;border:1px solid var(--border);background:var(--surface-1);color:var(--text-secondary);cursor:pointer}.rtoggle button.on{background:var(--text-primary);color:var(--surface-1)}tr.dim td{color:var(--muted);font-style:italic}.rt2 .legend{min-height:3.2em;align-content:flex-start}.rt2 svg{display:block;width:100%;height:auto}td.ckr{white-space:normal;max-width:300px;font-size:13px;vertical-align:middle}.rt-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:6px}.hmt td,.hmt th{padding:3px 4px;vertical-align:top}.hmt td.hm{min-width:120px}.hm-s{font-size:11px;color:var(--muted)}.hm-ax{font-size:10px;color:var(--muted);text-align:right}td.hm-na{color:var(--muted);font-size:11px;text-align:center;vertical-align:middle}.hm-bar{font-size:11px;margin:4px 0 8px}.hm-bar i{display:inline-block;width:14px;height:10px}
.lf .sq{font-size:20px;line-height:1;vertical-align:-2px}
.toggle{position:sticky;top:38px;z-index:1;margin:22px 0 0;display:flex;gap:6px}.toggle button{font:inherit;font-size:13px;padding:4px 12px;border-radius:999px;border:1px solid var(--border);background:var(--surface-1);color:var(--text-secondary);cursor:pointer}.toggle button.on{background:var(--text-primary);color:var(--surface-1)}
.ct{font-weight:600;font-size:13px;margin-bottom:2px}
h3{font-size:15px;margin:22px 0 4px}.lf td{font-size:15px;border-bottom:none;padding:6px 12px}.plane3d{margin-top:10px;max-width:760px}.plane3d svg{cursor:grab;touch-action:none}.plane3d svg:active{cursor:grabbing}
.lrgrid{display:table;border-collapse:separate;border-spacing:6px}.lrgrid td,.lrgrid th{border:none;white-space:normal;vertical-align:top}.lrgrid thead th{text-align:center;font-size:13px}.lrgrid th.grow{text-align:left;font-size:14px;color:var(--text-primary);padding-top:30px}.lrgrid th.grow span{font-size:11px;color:var(--muted);font-weight:400}.lrgrid td.gc{width:180px;height:112px;position:relative;background:var(--surface-1);border:1px solid var(--border);border-radius:8px;padding:8px 10px;cursor:default}.lrgrid td.gc:hover{border-color:var(--s1)}.mbig{font-size:26px;font-weight:600;margin:4px 0 0}.mbig small{font-size:11px;font-weight:400;color:var(--muted)}.mbig.muted{color:var(--muted)}.msub{font-size:11px;color:var(--muted)}.mlr{font-size:12px;margin:1px 0}.mlr span{color:var(--muted);font-size:11px}.pend{flex-direction:column}.pend span{font-size:11px}.lrgrid .pop{display:none;position:absolute;left:50%;top:100%;transform:translateX(-50%);z-index:5;width:300px;background:var(--surface-1);border:1px solid var(--border);border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,.18);padding:6px}.lrgrid td.gc:hover .pop{display:block}.lrgrid td.gc:hover{z-index:20}.lrgrid td.gc.up .pop{top:auto;bottom:100%}.lrgrid td.gx{width:180px;height:112px;background:repeating-linear-gradient(135deg,transparent 0 6px,var(--grid) 6px 7px);border-radius:8px;opacity:.6}.glab{font-size:11px;color:var(--text-secondary)}.mstar{font-size:12.5px;margin:2px 0}.mstar span{float:right;color:var(--muted);font-size:11px}.mstar.muted,.pend{color:var(--muted);font-size:12px}.pend{height:80px;display:flex;align-items:center;justify-content:center}
nav .tab{font:inherit;font-size:13px;padding:5px 14px;margin-right:6px;border-radius:999px;border:1px solid var(--border);background:var(--surface-1);color:var(--text-secondary);cursor:pointer}nav .tab.on{background:var(--text-primary);color:var(--surface-1)}
"""
JS = """
document.querySelectorAll('.plane3d').forEach(el=>{
 const D=JSON.parse(el.dataset.plane), svg=el.querySelector('svg'), NS='http://www.w3.org/2000/svg', W=760, H=460;
 const XA=D.x||{ref:8,lo:10,hi:48,ticks:[12,18,24,30,36,42,48],n:false};
 const L0=Math.log(XA.lo), L1=Math.log(XA.hi), T0=Math.log(8e9), T1=Math.log(160e9);
 const law=(p,L,T)=>p[0]*Math.pow(L/XA.ref,p[1])*Math.pow(T/1.04e10,p[2]);
 const N=10, grid=pl=>{const g=[];for(let i=0;i<=N;i++){const r=[];for(let j=0;j<=N;j++){const L=Math.exp(L0+(L1-L0)*i/N),T=Math.exp(T0+(T1-T0)*j/N);r.push([L,T,law(pl,L,T)]);}g.push(r);}return g;};
 const GA=grid(D.arch), GF=grid(D.fit), GX=D.fix?grid(D.fix):null, GB=D.box?grid(D.box):null;
 let zs=[];[GA,GF].concat(GX?[GX]:[]).concat(GB?[GB]:[]).forEach(g=>g.forEach(r=>r.forEach(p=>zs.push(Math.log(p[2])))));D.pts.forEach(p=>zs.push(Math.log(p.eta)));
 const Z0=Math.min(...zs), Z1=Math.max(...zs);
 const nx=L=>2*(Math.log(L)-L0)/(L1-L0)-1, ny=T=>2*(Math.log(T)-T0)/(T1-T0)-1, nz=e=>2*(Math.log(e)-Z0)/(Z1-Z0)-1;
 let az=-0.75, el_=0.45;
 const css=getComputedStyle(document.body), C=v=>css.getPropertyValue(v).trim()||'#888';
 function P(x,y,z){const ca=Math.cos(az),sa=Math.sin(az),xr=x*ca-y*sa,yr=x*sa+y*ca;const ce=Math.cos(el_),se=Math.sin(el_);
   return [W/2+xr*175, H/2+(-(z*0.8)*ce+yr*se)*175, yr*ce+z*0.8*se];}
 const mk=(tag,a)=>{const e=document.createElementNS(NS,tag);for(const k in a)e.setAttribute(k,a[k]);return e;};
 function fmtT(v){return v>=1e9?(v/1e9).toFixed(0)+'B':v;}
 function draw(){
  svg.innerHTML='';const items=[];
  const box=[[-1,-1],[1,-1],[1,1],[-1,1]];
  // floor frame + axes ticks
  for(let k=0;k<4;k++){const a=box[k],b=box[(k+1)%4],pa=P(a[0],a[1],-1),pb=P(b[0],b[1],-1);svg.appendChild(mk('line',{x1:pa[0],y1:pa[1],x2:pb[0],y2:pb[1],stroke:C('--axis')}));}
  XA.ticks.forEach(L=>{const p=P(nx(L),-1,-1);svg.appendChild(mk('text',{x:p[0],y:p[1]+14,'text-anchor':'middle'})).textContent=XA.n?(L>=1e9?(L/1e9).toFixed(2)+'B':Math.round(L/1e6)+'M'):L;});
  [10e9,30e9,100e9].forEach(T=>{const p=P(1,ny(T),-1);svg.appendChild(mk('text',{x:p[0]+14,y:p[1]+4})).textContent=fmtT(T);});
  const pe=P(-1,1,-1),pz=P(-1,1,1);svg.appendChild(mk('line',{x1:pe[0],y1:pe[1],x2:pz[0],y2:pz[1],stroke:C('--axis')}));
  [0.001,0.002,0.005,0.01,0.02].forEach(e=>{const z=nz(e);if(z<-1.05||z>1.05)return;const p=P(-1,1,z);svg.appendChild(mk('text',{x:p[0]-6,y:p[1]+4,'text-anchor':'end'})).textContent=e;});
  const lx=P(0,-1.25,-1),ly=P(1.3,0,-1),lz=P(-1,1,1.15);
  svg.appendChild(mk('text',{x:lx[0],y:lx[1]+20,'text-anchor':'middle',class:'lbl'})).textContent=D.lab.L;
  svg.appendChild(mk('text',{x:ly[0]+24,y:ly[1]+18,class:'lbl'})).textContent=D.lab.T;
  svg.appendChild(mk('text',{x:lz[0],y:lz[1]-6,'text-anchor':'middle',class:'lbl'})).textContent=D.lab.eta;
  // plane quads (painter's order)
  [[GA,'#9a9890',0.22],[GF,C('--a08'),0.30]].concat(GX?[[GX,C('--a08'),0.26]]:[]).concat(GB?[[GB,'#17becf',0.24]]:[]).forEach(([g,col,op])=>{for(let i=0;i<N;i++)for(let j=0;j<N;j++){
    const q=[g[i][j],g[i+1][j],g[i+1][j+1],g[i][j+1]].map(p=>P(nx(p[0]),ny(p[1]),nz(p[2])));
    items.push({d:q.reduce((s,p)=>s+p[2],0)/4,el:mk('polygon',{points:q.map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join(' '),fill:col,'fill-opacity':op,stroke:col,'stroke-opacity':0.55,'stroke-width':0.6})});}});
  // measured points + stem to our plane
  D.pts.forEach(p=>{const a=P(nx(p.X),ny(p.T),nz(p.eta)),b=P(nx(p.X),ny(p.T),nz(law(D.fit,p.X,p.T)));
    const g=mk('g',{});g.appendChild(mk('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:C('--s1'),'stroke-width':1.2}));
    const c=mk('circle',{cx:a[0],cy:a[1],r:5,fill:p.ex?C('--surface-1'):C('--s1'),stroke:C('--s1'),'stroke-width':2});
    c.dataset.tip=p.lab+'<br>η* '+p.eta.toFixed(4)+(p.ex?' ('+'extrap.)':'')+'<br>fit '+law(D.fit,p.X,p.T).toFixed(4)+' · ArchScale '+law(D.arch,p.X,p.T).toFixed(4)+(D.fix?' · T^-0.32 '+law(D.fix,p.X,p.T).toFixed(4):'')+(D.box?' · Boxiang v2 '+law(D.box,p.X,p.T).toFixed(4):'');
    c.addEventListener('mousemove',ev=>{tip.innerHTML=c.dataset.tip;tip.style.display='block';tip.style.left=(ev.clientX+12)+'px';tip.style.top=(ev.clientY+12)+'px'});
    c.addEventListener('mouseleave',()=>{tip.style.display='none'});
    g.appendChild(c);items.push({d:Math.min(a[2],b[2])-0.01,el:g});});
  items.sort((x,y)=>y.d-x.d).forEach(it=>svg.appendChild(it.el));
 }
 // rotate ONLY while a mouse button is held (click-and-drag); any missed pointerup is caught by checking e.buttons
 let drag=null;
 svg.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];try{svg.setPointerCapture(e.pointerId)}catch(_){};e.preventDefault();});
 svg.addEventListener('pointermove',e=>{if(!drag)return;if(e.buttons===0){drag=null;return;}
   az+=(e.clientX-drag[0])*0.01;el_=Math.max(-0.2,Math.min(1.4,el_+(e.clientY-drag[1])*0.01));drag=[e.clientX,e.clientY];draw();});
 ['pointerup','pointercancel','pointerleave','lostpointercapture'].forEach(ev=>svg.addEventListener(ev,()=>{drag=null}));
 draw();
});
document.querySelectorAll('.rtoggle button').forEach(b=>b.addEventListener('click',()=>{const card=b.closest('.card');
 card.querySelectorAll('.rtoggle button').forEach(x=>x.classList.toggle('on',x===b));
 card.querySelectorAll('[data-ref]').forEach(d=>d.hidden=d.dataset.ref!==b.dataset.r);}));
document.querySelectorAll('.toggle button').forEach(b=>b.addEventListener('click',()=>{const m=b.dataset.m,pane=b.closest('.msec')||b.closest('.pane');
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


TRAIN = {}                                               # (change, rung) -> the run's training args (peak LR, iters, L, h, GBS, seq)
MULT_MISMATCH = set()                                    # grid points whose actual LR multiple differs from their Change ID's name


def load():
    R = {}
    for p in glob.glob(os.path.join(RES, "*.json")):
        d = json.load(open(p))
        R[(d["change"], d["rung"])] = d["metrics"]
        TRAIN[(d["change"], d["rung"])] = d.get("train")
    for (ch, q), m in list(R.items()):                   # run-name IDs of SWA runs -> the promoted Change IDs (same runs)
        for a, b in (("MUONH_FIT_DSV3_SWA", "SWA_FIT_DSV3"), ("MUONH_FIT_SWA", "SWA_FIT")):
            if ch.startswith(a):
                R.setdefault((b + ch[len(a):], q), m); TRAIN.setdefault((b + ch[len(a):], q), TRAIN.get((ch, q)))
                break
    for (ch, q), m in list(R.items()):
        if ch.startswith("MUON_LR"):
            R.setdefault(("OPT_" + ch, q), m)
            TRAIN.setdefault(("OPT_" + ch, q), TRAIN.get((ch, q)))
        for a, b in (("MUONH_FIT_DSV3_SWA", "SWA_FIT_DSV3"), ("MUONH_FIT_SWA", "SWA_FIT"), ("MUONH_FIT_DSV3_NE_", "ROUTER_NE_")):   # run-name evals of
            if ch.startswith(a):                                                                                   # promoted Change IDs
                R.setdefault((b + ch[len(a):], q), m); TRAIN.setdefault((b + ch[len(a):], q), TRAIN.get((ch, q)))
                break
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


def line_chart(series, xs, ylab, fmt, w=760, h=360, y0=None, labels=True, zero=False, seeds=None):
    """series = [(name, colour var, [(x, y, tip)], dashed)] on a log-x axis. seeds = {(name, x): [y per seed]}: drawn as hollow
    circles in the series colour (the line and the filled point are the seed mean)."""
    seeds = seeds or {}
    ml, mr, mt, mb = 52, 96, 12, 38
    allx = [p[0] for _, _, pts, _ in series for p in pts] or [1, 2]
    ally = [p[1] for _, _, pts, _ in series for p in pts] + [y for v in seeds.values() for y in v] + ([0.0] if zero else []) or [0, 1]
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
    if zero:                                              # Δ = 0 reference (arm = its baseline)
        o.append(f'<line x1="{ml}" x2="{w-mr}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var(--axis)" stroke-width="1.5" stroke-dasharray="3 3"/>')
    o.append(f'<text x="12" y="{(h-mb+mt)/2:.0f}" transform="rotate(-90 12 {(h-mb+mt)/2:.0f})" text-anchor="middle">{esc(ylab)}</text>')
    for name, col, pts, dashed in series:
        if not pts:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y, _) in enumerate(pts))
        o.append(f'<path d="{d}" fill="none" stroke="var({col})" stroke-width="2"{" stroke-dasharray=\"5 4\"" if dashed else ""} stroke-linejoin="round"/>')
        for x, y, t in pts:
            for ys in seeds.get((name, x), []):           # individual seeds: hollow, same colour
                o.append(f'<circle cx="{X(x):.1f}" cy="{Y(ys):.1f}" r="4" fill="var(--surface-1)" stroke="var({col})" stroke-width="1.6"/>')
            o.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4.5" fill="var({col})" stroke="var(--surface-1)" stroke-width="2"/>')
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
            f'<br><span style="color:var({col})">●</span> {esc(n)}: {y:.4f}' + (f" (seeds " + " / ".join(f"{v:+.4f}" for v in seeds[(n, x)]) + ")" if len(seeds.get((n, x), [])) > 1 else "")
            for n, col, y in sorted(rows, key=lambda r: r[2], reverse=True))   # largest on top
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
            ("KD_MUONH_A0P8_B128", "B128"), ("KD_MUONH_A0P8_DD64", "DD64"), ("KD_MUONH_A0P8_DROP64", "DROP64"), ("KD_MUONH_A0P8_S128Y", "S128Y")]
VRUNGS = ["q0", "q1", "q2", "q3"]
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
    for ch in [a[0] for a in ARMS] + [v[0] for v in VARIANTS] + [c for c, _ in LRS if c] + [c for c, _ in LRS_IT if c] + [BASE_IT] + [a for a, _ in ARCH_ARMS] + sorted({cell_change(sfx, tag) for sfx, *_ in GRID_CELLS.values() for _, tag in MULTS}):
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


def mtoggle(metrics):
    """Per-section metric buttons (user: one toggle per ablation section, not one for the page); scoped by the enclosing .msec."""
    return '<div class="toggle">' + "".join(f'<button data-m="{m}" class="{"on" if j == 0 else ""}">{esc(n)}</button>' for j, (m, n) in enumerate(metrics)) + "</div>"


def chart_sections(R, t, metrics):
    """Metric toggle + the optimizer (CE) delta chart and the KD-with-MuonH delta chart, one SVG per metric (deltas only, no absolute loss)."""
    qs_kd = [q for q in RUNGS if val(R, KD["0.8"], q, metrics[0][0]) is not None and val(R, CE, q, metrics[0][0]) is not None]
    charts = {}
    for mkey, mname in metrics:
        def pts_of(ch, f, name):
            return [(N_ACTIVE[q], f(q, ch), f"{esc(name)} · {q} ({N_LABEL[q]})<br>{mname} {f(q, ch):.4f}")
                    for q in RUNGS if val(R, ch, q, mkey) is not None and f(q, ch) is not None]
        xs_ce = {q: N_ACTIVE[q] for q in RUNGS if val(R, "BASE", q, mkey) is not None}
        opt = [(t["muon"], "--o-muon", "OPT_MUON", False), (t["ce"], "--s1", CE, False)]
        # Boxiang's runs at the NEW fitted LR law (dashed, same colour as their optimizer), shown when they exist
        for ch, name, col in (("MUON_LRFIT", t["muon_fit"], "--o-muon"), ("OPT_MUON_LRFIT", t["muon_fit"], "--o-muon"), ("MUONH_FIT", t["ce_fit"], "--s1")):
            if any(val(R, ch, q, mkey) is not None for q in RUNGS) and not any(o[0] == name for o in opt):
                opt.append((name, col, ch, True))
        dv = lambda q, ch: (val(R, ch, q, mkey) - val(R, "BASE", q, mkey)) if val(R, "BASE", q, mkey) is not None else None
        o2 = legend([(n, c) for n, c, _, _ in opt]) + line_chart(
            [(n, c, pts_of(ch, dv, n), d) for n, c, ch, d in opt],
            {q: x for q, x in xs_ce.items() if val(R, "OPT_MUON", q, mkey) is not None}, t["vs_adamw"], lambda v: f"{v:.3f}")
        kds = [(t["alpha"].format(a=a), f"--a{a.replace('.', '')}", ch) for a, ch in KD.items()]
        gk = lambda q, ch: (val(R, ch, q, mkey) - val(R, CE, q, mkey)) if val(R, CE, q, mkey) is not None else None
        k2 = legend([(n, c) for n, c, _ in kds]) + line_chart(
            [(n, c, pts_of(ch, gk, n), False) for n, c, ch in kds], {q: N_ACTIVE[q] for q in qs_kd}, "Δloss (KD − CE)", lambda v: f"{v:.3f}",
            labels=False)
        charts[mkey] = (o2, k2)

    def block(i):
        return "".join(f'<div data-metric="{m}"{"" if j == 0 else " hidden"}>{charts[m][i]}</div>' for j, (m, _) in enumerate(metrics))
    return f"""<div class="msec"><h2>{esc(t['opt_h'])}</h2><p class="note">{esc(t['opt_note'])}</p>{mtoggle(metrics)}
<div class="card one"><div class="ct">{esc(t['o2_h'])}</div>{block(0)}</div></div>
<div class="msec"><h2>{esc(t['kd_h'])}</h2><p class="note">{esc(t['kd_note'])}</p>{mtoggle(metrics)}
<div class="card one"><div class="ct">{esc(t['k2_h'])}</div>{block(1)}</div></div>"""


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
        for q in VRUNGS:
            a, b = val(R, ch, q, mkey), val(R, CE, q, mkey)
            if a is not None and b is not None:
                pts.append((N_ACTIVE[q], a - b, f"{esc(short)} · {q} ({N_LABEL[q]})<br>{esc(mname)}: {short} {a:.4f} − CE {b:.4f} = {a - b:+.4f}"))
        if pts:
            series.append((f"{short} · {t['vdesc'][short]}", col, pts, False))
    if not series:
        return f'<p class="note">{esc(t["pending"])}</p>'
    xs = {q: N_ACTIVE[q] for q in VRUNGS if any(val(R, ch, q, mkey) is not None for ch in [VBASE] + [v[0] for v in VARIANTS]) and val(R, CE, q, mkey) is not None}
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
    return f"""<div class="msec"><h2>{esc(t['v_h'])}</h2><p class="note">{t['v_note']}</p>
<div class="card vn"><div class="ct">{esc(t['vn_title'])}</div>{names}</div>{mtoggle(metrics)}
<div class="card one"><div class="ct">{esc(t['v_chart'])}</div>{blocks}</div></div>"""


# MuonH LR grid (Runlong.md TODO 3): the user's predefined table - rows = model, columns = training-token budget; each "cell" is one
# (L, T) point where the best LR multiple m* is found from a 5-point sweep (x0.7 / 1 / 1.2 / 1.4 / 2, + x0.5 where run).
GRID_ROWS = [("q0", 12), ("q1", 18), ("q2", 24), ("q3", 30), ("q4", 36), ("q5", 42), ("q6", 48)]
GRID_COLS = ["9.6", "30", "71", "86", "137"]
MULTS = [(0.5, "0P5X"), (0.7, "0P7X"), (1.0, "1X"), (1.2, "1P2X"), (1.4, "1P4X"), (2.0, "2X"), (2.8, "2P8X"), (4.0, "4X")]
# (rung, column) -> (Change-ID suffix, token label, TPP, label key); suffix "" = the main ladder (1x = OPT_MUONH)
GRID_CELLS = {
    ("q0", "9.6"): ("", "9.6B", 200, "main"), ("q0", "30"): ("_TPP600", "28.8B", 600, "tpp"), ("q0", "86"): ("_TPP1800", "86.4B", 1800, "tpp"),
    ("q3", "137"): ("", "136.6B", 200, "main_x"),
    ("q1", "9.6"): ("_IT4577", "9.6B", 62, "it"), ("q1", "30"): ("", "30.8B", 200, "main"),
    ("q2", "9.6"): ("_IT4577", "9.6B", 27, "it"), ("q2", "30"): ("_IT14685", "30.8B", 87, "it2"), ("q2", "71"): ("", "71.1B", 200, "main"),
    ("q3", "9.6"): ("_IT4577", "9.6B", 14, "it"), ("q4", "9.6"): ("_IT4577", "9.6B", 8, "it"), ("q5", "9.6"): ("_IT4577", "9.6B", 5, "it"), ("q6", "9.6"): ("_IT4577", "9.6B", 3.5, "it"), ("q3", "30"): ("_IT14685", "30.8B", 45, "it2"),
}


GRID_ITERS = {("q0", "9.6"): 4577, ("q0", "30"): 13730, ("q0", "86"): 41189, ("q1", "9.6"): 4577, ("q1", "30"): 14685,
              ("q2", "9.6"): 4577, ("q2", "30"): 14685, ("q2", "71"): 33906, ("q3", "9.6"): 4577, ("q3", "30"): 14685, ("q3", "137"): 65165, ("q4", "9.6"): 4577, ("q5", "9.6"): 4577, ("q6", "9.6"): 4577}


def law_lr(L, iters):
    """HyperP law: eta0 0.016 x sqrt(8 / L) x (T / 1.04e10)^-0.32, T = iters x 2^21 tokens (= the cell's 1x LR; matches every run's --lr)."""
    return 0.016 * math.sqrt(8 / L) * (iters * 2 ** 21 / 1.04e10) ** -0.32


SEEDS = [("", 1), ("_S2", 2), ("_S3", 3)]           # seed-replicated cells (Runlong.md step 26: q0 and q2-IT4577): extra parabola points


def _muonh_change(suffix, tag, seed=""):
    if suffix == "" and tag == "1X":
        return CE + seed
    return f"MUONH_LR{tag}{suffix}{seed}"


def _muon_change(suffix, tag, seed=""):
    if suffix == "" and tag == "1X":
        return "OPT_MUON" + seed
    return f"OPT_MUON_LR{tag}{suffix}{seed}"


# Muon grid (Boxiang 2026-10-24 22:00, DECISIONS.md "Muon 全因子"): x0.25 / 0.5 / 1 / 2 / 4 of the ladder's Muon LR on the MuonH cells.
# Muon (like AdamW) uses an LR that depends on the rung only, not on T: 1x from the runs' argv.
MUON_LR1 = {"q0": 0.00308, "q1": 0.002088, "q2": 0.00158, "q3": 0.001271}
MUON_CELLS = {
    ("q0", "9.6"): ("", "9.6B", 200, "main"), ("q0", "30"): ("_TPP600", "28.8B", 600, "tpp"), ("q0", "86"): ("_TPP1800", "86.4B", 1800, "tpp"),
    ("q1", "9.6"): ("_IT4577", "9.6B", 62, "it"), ("q1", "30"): ("", "30.8B", 200, "main"),
    ("q2", "9.6"): ("_IT4577", "9.6B", 27, "it"), ("q2", "71"): ("", "71.1B", 200, "main"),
    ("q3", "9.6"): ("_IT4577", "9.6B", 14, "it"), ("q3", "137"): ("", "136.6B", 200, "main"),
}


def _power_fit(pairs):
    """eta = c (L/8)^b through [(L, eta)]: the Muon ladder rule written in the law's form (it has no T term)."""
    coef, _, _ = _lstsq([(1.0, math.log(L / 8)) for L, _ in pairs], [math.log(e) for _, e in pairs])
    return math.exp(coef[0]), coef[1]


GH = dict(key="h", rows=GRID_ROWS, cols=GRID_COLS, cells=GRID_CELLS, mults=MULTS, change=_muonh_change, xr=(0.45, 5.0),
          lr1=lambda rung, c: law_lr(dict(GRID_ROWS)[rung], GRID_ITERS[(rung, c)]), arch=lambda: (0.016, -0.5, -0.32), tx={})
GM = dict(key="m", rows=GRID_ROWS[:4], cols=GRID_COLS, cells=MUON_CELLS, mults=[(0.25, "0P25X"), (0.5, "0P5X"), (1.0, "1X"), (2.0, "2X"), (4.0, "4X")],
          change=_muon_change, xr=(0.2, 5.0), lr1=lambda rung, c: MUON_LR1[rung], ref="muon",
          arch=lambda: _power_fit([(L, MUON_LR1[q]) for q, L in GRID_ROWS[:4]]) + (0.0,), tx="mg")


def cell_change(suffix, tag, seed="", G=None):
    return (G or GH)["change"](suffix, tag, seed)


def archscale_lr(L, T):
    """ArchScale / HyperP ORIGINAL law (user: every multiple in the report is relative to THIS, whatever law a run was launched with)."""
    return 0.016 * (L / 8) ** -0.5 * (T / 1.04e10) ** -0.32


def actual_mult(G, ch, rung, m):
    """The run's actual peak LR / the reference LR (ArchScale original at the run's own L and T; Muon grid: the rung's ladder Muon LR).
    Falls back to the Change ID's nominal multiple m when the run's args are unknown; > 2 % disagreement is listed in the report."""
    tr = TRAIN.get((ch, rung))
    if not tr or not tr.get("lr") or not tr.get("num_layers") or not tr.get("train_iters"):
        return m
    ref = MUON_LR1.get(rung) if G.get("ref") == "muon" else \
        archscale_lr(tr["num_layers"], tr["train_iters"] * (tr.get("global_batch_size") or 256) * (tr.get("seq_length") or 8192))
    if not ref:
        return m
    a = tr["lr"] / ref
    if abs(a / m - 1) > 0.02:
        MULT_MISMATCH.add((ch, rung, m, round(a, 3)))
    return a


def cell_fit(R, rung, suffix, mkey, G=None):
    """Parabola in log(multiple) through every seed's points, ONE shape (a, b) and a free offset per seed (y = a x^2 + b x + c_seed):
    seeds shift the whole curve, so they are aligned by their offset, not by their own 1x (which may be missing).
    Returns (pts, fit, coef): pts = [(multiple, loss - loss(seed-1 1x) after the seed alignment, seed)]; fit = (m*, loss at m* - 1x,
    extrapolated flag) or None; coef = (a, b, c) of the seed-1 curve relative to its 1x. [] / None if the cell's seed-1 1x is missing."""
    G = G or GH
    raw = {k: [(actual_mult(G, cell_change(suffix, tag, sfx, G), rung, m), val(R, cell_change(suffix, tag, sfx, G), rung, mkey))
               for m, tag in G["mults"]] for sfx, k in SEEDS}
    raw = {k: [(m, y) for m, y in v if y is not None] for k, v in raw.items()}
    if G.get("pool"):   # one curve for all seeds, no per-seed offset (router LR checks: only the 1x has seeds 2 / 3); 0 = their mean 1x
        ones = [y for y in (val(R, cell_change(suffix, "1X", sfx, G), rung, mkey) for sfx, _ in SEEDS) if y is not None]
        if not ones or not raw[1]:
            return [], None, None
        b0 = sum(ones) / len(ones)
        pts = [(m, y - b0, k) for k in raw for m, y in raw[k]]
        xs = [math.log(m) for m, _, _ in pts]
        if len({round(x, 6) for x in xs}) < 3:
            return pts, None, None
        coef, _, _ = _lstsq([[x * x, x, 1.0] for x in xs], [y for _, y, _ in pts])
        a, b, c = coef
        cq = (a, b, c)
        if a <= 0:
            return pts, None, cq
        x0 = -b / (2 * a)
        if x0 < min(xs) - math.log(2) or x0 > max(xs) + math.log(2):
            return pts, None, cq
        return pts, (math.exp(x0), a * x0 * x0 + b * x0 + c, not (min(xs) < x0 < max(xs))), cq
    b0 = val(R, cell_change(suffix, "1X", "", G), rung, mkey)
    if b0 is None:
        if not G.get("rebase") or not raw[1]:
            return [], None, None
        b0 = raw[1][0][1]                                # no 1x yet: relative to the smallest sampled multiple (m* unaffected)
    use = [k for _, k in SEEDS if len(raw[k]) >= (1 if k == 1 else 2)]
    if 1 not in use:
        return [], None, None   # a seed with one point only fixes its own offset
    pts1 = [(m, y - b0, 1) for m, y in raw[1]]
    X, Y = [], []
    for i, k in enumerate(use):
        for m, y in raw[k]:
            x = math.log(m); X.append([x * x, x] + [1.0 if j == i else 0.0 for j in range(len(use))]); Y.append(y)
    xs = [r[1] for r in X]
    if len({round(x, 6) for x in xs}) < 3 or len(Y) < len(X[0]):   # 3 points = an exact parabola (allowed)
        return pts1, None, None
    coef, _, _ = _lstsq(X, Y)
    a, b, off = coef[0], coef[1], dict(zip(use, coef[2:]))
    pts = pts1 + [(m, y - off[k] + off[1] - b0, k) for k in use if k != 1 for m, y in raw[k]]
    cq = (a, b, off[1] - b0)
    if a <= 0:
        return pts, None, cq
    x0 = -b / (2 * a)
    if x0 < min(xs) - math.log(2) or x0 > max(xs) + math.log(2):
        return pts, None, cq                             # more than 2x beyond the sweep: not trusted
    edge = not (min(xs) < x0 < max(xs))                  # vertex outside the sampled multiples: an EXTRAPOLATED optimum (user: estimate it)
    return pts, (math.exp(x0), a * x0 * x0 + b * x0 + cq[2], edge), cq


def cell_points(R, rung, suffix, mkey):
    """[(multiple, loss - loss(1x of this cell))], seed 1 only; [] if the cell's 1x is missing."""
    return [(m, y) for m, y, k in cell_fit(R, rung, suffix, mkey)[0] if k == 1]


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
    if x0 < min(xs) - math.log(2) or x0 > max(xs) + math.log(2):
        return None
    return math.exp(x0), a * x0 * x0 + b * x0 + c, edge


def mini_lr(R, t, rung, suffix, mkey, mname, w=250, h=150, lr1=None, G=None, extra=None):
    G = G or GH
    pts, fit, cq = cell_fit(R, rung, suffix, mkey, G)
    if len(pts) <= 1:                                   # nothing beyond the cell's own 1x yet
        return f'<div class="pend">{esc(t["g_pending"])}' + (f'<span>1× = {lr1:.4f}</span>' if lr1 else "") + '</div>'
    w, h = 300, 190
    ml, mr, mt, mb = 40, 8, 8, 30
    lx0, lx1 = math.log(G["xr"][0]), math.log(G["xr"][1])
    ys = [y for _, y, _ in pts] + [0.0] + [e[1] for e in (extra or [])]
    if fit:
        ys.append(fit[1])
    lo, hi = min(ys), max(ys); pad = (hi - lo) * 0.12 or 0.005; lo, hi = lo - pad, hi + pad
    X = lambda m: ml + (math.log(m) - lx0) / (lx1 - lx0) * (w - ml - mr)
    Y = lambda y: mt + (hi - y) / (hi - lo) * (h - mt - mb)
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for v in (lo + (hi - lo) * k / 3 for k in range(4)):
        o.append(f'<line x1="{ml}" x2="{w-mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--grid)"/><text x="{ml-4}" y="{Y(v)+3:.1f}" text-anchor="end" style="font-size:9px">{v:+.3f}</text>')
    o.append(f'<line x1="{ml}" x2="{w-mr}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="var(--axis)"/>')
    for m, _ in G["mults"]:
        o.append(f'<text x="{X(m):.1f}" y="{h-14}" text-anchor="middle" style="font-size:9px">{m:g}×</text>')
        if lr1:
            o.append(f'<text x="{X(m):.1f}" y="{h-3}" text-anchor="middle" style="font-size:8px;fill:var(--muted)">{m * lr1:.4f}</text>')
    if cq:                                                # fitted parabola (seed-1 offset) + m* marker
        xs = [math.log(m) for m, _, _ in pts]; a_lo, a_hi = min(xs), max(xs)
        qx = [a_lo + (a_hi - a_lo) * k / 40 for k in range(41)]
        d = " ".join(f"{'M' if k == 0 else 'L'}{X(math.exp(x)):.1f},{Y(cq[0] * x * x + cq[1] * x + cq[2]):.1f}" for k, x in enumerate(qx))
        o.append(f'<path d="{d}" fill="none" stroke="var(--s1)" stroke-width="1.5" stroke-dasharray="4 3" opacity="0.8"/>')
    if fit:
        o.append(f'<line x1="{X(fit[0]):.1f}" x2="{X(fit[0]):.1f}" y1="{mt}" y2="{h-mb}" stroke="var(--a08)" stroke-width="1.5"/>')
    for m, y, k in sorted(pts, key=lambda p: -p[2]):     # seed 1 drawn last (on top)
        tip = (f"{esc(rung)} · {m:g}× (η = {m * lr1:.5f})" if lr1 else f"{esc(rung)} · {m:g}×") + f" · seed {k}<br>{esc(mname)}: {y:+.4f} vs {esc(G.get('vs', '1×'))}" + (f" ({esc(t['g_seedal'])})" if k != 1 and not G.get("pool") else "")
        o.append(f'<circle cx="{X(m):.1f}" cy="{Y(y):.1f}" r="{3.5 if k == 1 else 2.8}" fill="var(--s1)" fill-opacity="{1 if k == 1 else 0.45}" stroke="var(--s1)" stroke-width="1"/>'
                 f'<circle class="hit" cx="{X(m):.1f}" cy="{Y(y):.1f}" r="10" data-tip="{tip}"/>')
    for m, y, lab, col in (extra or []):                 # extra labelled single runs (e.g. per-group LR at 1.74x), diamonds
        cx, cy = X(m), Y(y)
        o.append(f'<path d="M{cx:.1f},{cy-6:.1f} L{cx+6:.1f},{cy:.1f} L{cx:.1f},{cy+6:.1f} L{cx-6:.1f},{cy:.1f} Z" fill="{col}" stroke="var(--surface-1)" stroke-width="1"/>'
                 f'<text x="{cx+8:.1f}" y="{cy+4:.1f}" style="font-size:10px;fill:{col}">{esc(lab.split(":")[0])}</text>'
                 f'<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="10" data-tip="{esc(lab)}: {y:+.4f} vs {esc(G.get('vs', '1×'))}"/>')
    o.append("</svg>")
    ns = len({k for _, _, k in pts})
    npts = t["g_npts"].format(n=len(pts)) + (t["g_nseed"].format(s=ns) if ns > 1 else "")
    if fit:
        mstar, ystar, edge = fit
        head = (f'<div class="mbig">{mstar:.2f}×{"<small> " + esc(t["g_edge"]) + "</small>" if edge else ""}</div>'
                + (f'<div class="mlr">η* = <b>{mstar * lr1:.4f}</b> <span>(1× = {lr1:.4f})</span></div>' if lr1 else "")
                + f'<div class="msub">{ystar:+.4f} vs {esc(G.get("vs", "1×"))} · {esc(npts)}</div>')
    else:
        head = (f'<div class="mbig muted">—</div>' + (f'<div class="mlr"><span>1× = {lr1:.4f}</span></div>' if lr1 else "")
                + f'<div class="msub">{esc(t["g_fewpts"])} · {esc(npts)}</div>')
    return head + '<div class="pop">' + "".join(o) + "</div>"


SEED_ARMS = [("AdamW", ["BASE", "BASE_S2", "BASE_S3"]), ("Muon", ["OPT_MUON", "OPT_MUON_S2", "OPT_MUON_S3"])]
SEED_METRICS = [("00_overall_tsqw", "overall_tsqw"), ("01_val_v2_overall_tw", "val_v2"), ("00_overall_tw", "overall_tw")]


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


ARCH_ARMS = [("SWA_FIT", "MUONH_FIT"),   # SWA only at the fitted LR law (user); the old-law SWA arm is dropped
              ("SWA_FIT_ROPE", "MUONH_FIT"), ("SWA_FIT_W4K", "MUONH_FIT"),
             ("SWA_FIT_W1K", "MUONH_FIT"), ("SWA_FIT_SINK", "MUONH_FIT"), ("SWA_FIT_SINK_W1K", "MUONH_FIT"),
             ("SWA_FIT_DSV3", "MUONH_FIT_DSV3")]   # MUONH_SPLITFC1 (optimizer change, old law) dropped from the SWA section (user title)   # (arm, its baseline): architecture arms on the MuonH stack


ARCH_COL = {"SWA": "--muted", "SWA_FIT": "--c1", "SWA_FIT_ROPE": "--c3", "SWA_FIT_W4K": "--s4", "SWA_FIT_W1K": "--c5", "SWA_FIT_SINK": "--c2",
            "SWA_FIT_SINK_W1K": "--a09", "SWA_FIT_DSV3": "--c6", "MUONH_SPLITFC1": "--o-adamw"}
ARCH_DASH = {"SWA", "MUONH_SPLITFC1"}                        # old-law / non-attention arms dashed


def _law_of(ch, q):
    """Which LR rule a run actually used, from its peak LR: fitted MuonH law (fit v2) or the ArchScale original."""
    tr = TRAIN.get((ch, q)) or {}
    if not tr.get("lr") or not tr.get("num_layers"):
        return None
    T = tr["train_iters"] * (tr.get("global_batch_size") or 256) * (tr.get("seq_length") or 8192)
    fit2 = 0.01725 * (tr["num_layers"] / 8) ** -0.1764 * (T / 1.04e10) ** -0.2591
    if abs(tr["lr"] / fit2 - 1) < 0.01:
        return "fit"
    if abs(tr["lr"] / archscale_lr(tr["num_layers"], T) - 1) < 0.01:
        return "arch"
    return "other"


def arch_table(R, t):
    rows = []
    for arm, base in ARCH_ARMS:
        qs = [q for q in RUNGS + ["q6"] if (arm, q) in R and (base, q) in R]
        if not qs:
            continue
        laws = {_law_of(c, q) for q in qs for c in (arm, base)} - {None}
        lawtxt = " / ".join(t["arch_law"][x] for x in sorted(laws)) if laws else "·"
        rows.append(f'<tr><td class="l"><span style="color:var({ARCH_COL.get(arm, "--muted")})">■</span> {esc(t["arch_name"][arm].split(" −")[0].split("（")[0] if False else t["arch_short"][arm])}</td>'
                    f'<td class="l"><code>{arm}</code></td><td class="l" style="white-space:normal;max-width:420px">{esc(t["arch_desc"][arm])}</td>'
                    f'<td class="l"><code>{base}</code></td><td class="l">{lawtxt}</td><td class="l">{", ".join(qs)}</td></tr>')
    head = "".join(f'<th class="l">{esc(x)}</th>' for x in t["arch_thead"])
    return f'<div class="card"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def arch_section(R, t, metrics):
    """Line chart (user): x = rung, y = arm - its baseline, one line per arm; own metric toggle, overall_tsqw first."""
    def chart(mkey, mname):
        series, xs = [], {}
        for arm, base in ARCH_ARMS:
            pts = []
            for q in RUNGS + ["q6"]:
                a, b = val(R, arm, q, mkey), val(R, base, q, mkey)
                if a is None or b is None:
                    continue
                xs[q] = N_ACTIVE[q]
                pts.append((N_ACTIVE[q], a - b, f"{esc(t['arch_name'][arm])} · {q}<br>{mname} {a - b:+.4f}"))
            if pts:
                series.append((t["arch_name"][arm], ARCH_COL.get(arm, "--muted"), pts, arm in ARCH_DASH))
        if not series:
            return ""
        return legend([(n, c) for n, c, _, _ in series]) + line_chart(series, xs, t["arch_y"], lambda v: f"{v:+.3f}", labels=False, zero=True)
    blocks = "".join(f'<div data-metric="{m}"{"" if j == 0 else " hidden"}>{chart(m, n)}</div>' for j, (m, n) in enumerate(metrics))
    return f"""<div class="msec"><h2>{esc(t['arch_h'])}</h2><p class="note">{t['arch_note']}</p>{arch_table(R, t)}{mtoggle(metrics)}
<div class="card one"><div class="ct">{esc(t['arch_chart'])}</div>{blocks}</div></div>"""



# Router ablation (Boxiang, 2026-10-26; reports/dsv3_router_experiment_design_2026-10-26.md): softmax + aux router vs the DeepSeek-V3 router
# (sigmoid + expert bias + seq aux 1e-4), all at the fitted MuonH law (MUONH_FIT) or, for AdamW, BASE's law. q0 x 3 seeds paired per seed.
RT_ROWS = [  # (label key, arm, partner) - partner per the design's pairing
    ("rt_a", "MUONH_FIT_DSV3", "MUONH_FIT"),
    ("rt_b2", "MUONH_FIT_DSV3_RTADAMW_RT0P19", "MUONH_FIT_DSV3"),
    ("rt_b3", "MUONH_FIT_DSV3_RT0P19", "MUONH_FIT_DSV3"),
    ("rt_c", "BASE_DSV3_ONETREE", "BASE"),
]
RT_HEALTH = {"rt_a": "bad", "rt_b2": "bad", "rt_b3": "ok", "rt_c": "ok"}   # training-log router health (Boxiang DECISIONS 2026-10-26 19:1x-19:2x)


def _seed_id(ch, k):
    return ch if k == 1 else f"{ch}_S{k}"


def _paired(R, arm, partner, q, mkey, partner2=None, arm2=None):
    """Per-seed differences arm - partner (seeds 1-3 where both exist); with arm2 / partner2: (arm - arm2) - (partner - partner2)."""
    out = []
    for k in (1, 2, 3):
        v = [val(R, _seed_id(c, k), q, mkey) for c in (arm, partner) + ((arm2, partner2) if arm2 else ())]
        if all(x is not None for x in v):
            out.append(v[0] - v[1] if not arm2 else (v[0] - v[2]) - (v[1] - v[3]))
    return out


def _fmt_paired(d, t):
    if not d:
        return '<td>·</td>'
    m = sum(d) / len(d)
    se = (math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)) / math.sqrt(len(d))) if len(d) > 1 else None
    cls = "neg" if m < 0 else "pos"
    return f'<td class="{cls}">{"−" if m < 0 else "+"}{abs(m):.4f}' + (f'<small> ±{se:.4f}</small>' if se is not None else '') + f'<small> ({len(d)})</small></td>'


# Router-health SUMMARY statistics from the training logs (Boxiang's router_health.py, LADDER_ROUTER_HEALTH=100: one "[router health] step N"
# line every 100 steps, aggregated over ALL layers - unlike the per-layer line, it is never cut by the log's ~1000-character line limit).
RUNS_DIR = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/boxiangw/runs/scaling_ladder/qwen3_tpp200"
RT_HEAT_ARMS = [("rt_hm_soft", "muonh_fit", "--s1"), ("rt_hm_a", "muonh_fit_dsv3", "--s2"), ("rt_hm_b2", "muonh_fit_dsv3_rtadamw_rt0p19", "--s4"),
                ("rt_hm_b3", "muonh_fit_dsv3_rt0p19", "--s3"), ("rt_hm_c", "v1_dsv3_onetree", "--s5")]
N_EXPERTS = 128
RT_STATS = [  # (key, regex on the summary line, transform, unit)
    ("vio_mean", r"MaxVio mean ([0-9.]+)", None, ""), ("vio_max", r"MaxVio mean [0-9.]+ max ([0-9.]+)", None, ""),
    ("dead", r"dead experts (\d+)", "dead", "%"), ("bmax", r"\|b\|max ([0-9.]+)", None, ""),
    ("sat", r"frac mean ([0-9.]+)%", None, "%"), ("satpos", r"z>\+4 mean ([0-9.]+)%", None, "%"), ("pinned", r"top-8 pinned ([0-9.]+)%", None, "%"),
    ("zmax", r"max \|z\| ([0-9.]+)", None, ""), ("upd", r"gate update/weight \(flag\): mean ([0-9.]+)", None, "")]


def router_summary_log(exp, q):
    """{step: {stat: value}} from the '[router health] step' summary lines of the run (seed 1), or {}."""
    out = {}
    for f in sorted(glob.glob(os.path.join(RUNS_DIR, f"qwen3_tpp200_{exp}-{q}-tpp200-s1", "logs", "*.log"))):
        try:
            for line in open(f, errors="replace"):
                if "[router health] step " not in line:
                    continue
                m = re.search(r"\[router health\] step (\d+) \| (\d+) MoE layers", line)
                if not m:
                    continue
                step, nl, rec = int(m.group(1)), int(m.group(2)), {}
                for key, rx, tr, _ in RT_STATS:
                    mm = re.search(rx, line)
                    if mm:
                        v = float(mm.group(1))
                        rec[key] = 100.0 * v / (nl * N_EXPERTS) if tr == "dead" else v
                out[step] = rec
        except OSError:
            pass
    return out


def router_trends(t):
    """One line chart per rung (x = training step, one line per arm), metric toggle; summary statistics over all layers."""
    data = {(q, key): router_summary_log(exp, q) for q in ("q0", "q1", "q2", "q3") for key, exp, _ in RT_HEAT_ARMS}
    rungs = [q for q in ("q0", "q1", "q2", "q3") if any(data[(q, k)] for k, _, _ in RT_HEAT_ARMS)]
    if not rungs:
        return ""
    unit = {k: u for k, _, _, u in RT_STATS}
    panes = []
    for j, (mk, _, _, _) in enumerate(RT_STATS):
        charts = []
        for q in rungs:
            series = [(t[key], col, sorted((st, r[mk]) for st, r in data[(q, key)].items() if mk in r)) for key, _, col in RT_HEAT_ARMS]
            series = [x for x in series if x[2]]
            if not series:
                continue
            xs = [st for _, _, p in series for st, _ in p]; ys = [v for _, _, p in series for _, v in p]
            x1 = max(xs) or 1; ylo, yhi = 0.0, max(ys) * 1.08 or 1.0
            w, h, ml, mr, mt, mb = 360, 200, 44, 8, 22, 26
            X = lambda v: ml + v / x1 * (w - ml - mr); Y = lambda v: mt + (yhi - v) / (yhi - ylo) * (h - mt - mb)
            o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img"><text x="{ml}" y="14" style="font-size:12px;font-weight:600">{q}</text>']
            for k in range(4):
                v = ylo + (yhi - ylo) * k / 3
                o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--grid)"/><text x="{ml - 4}" y="{Y(v) + 3:.1f}" text-anchor="end" style="font-size:9px">{v:.{3 if mk == "upd" else 2}g}</text>')
            o.append(f'<text x="{w - mr}" y="{h - 6}" text-anchor="end" style="font-size:9px">step {x1:,}</text><text x="{ml}" y="{h - 6}" style="font-size:9px">0</text>')
            if mk == "bmax":
                o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(1.0):.1f}" y2="{Y(1.0):.1f}" stroke="var(--bad)" stroke-dasharray="4 3"/>' if yhi > 1 else "")
            if mk in ("sat", "satpos", "pinned"):
                o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(5.0):.1f}" y2="{Y(5.0):.1f}" stroke="var(--bad)" stroke-dasharray="4 3"/>' if yhi > 5 else "")
            for name, col, pts in series:
                d = " ".join(f"{'M' if i == 0 else 'L'}{X(st):.1f},{Y(v):.1f}" for i, (st, v) in enumerate(pts))
                o.append(f'<path d="{d}" fill="none" stroke="var({col})" stroke-width="1.8"/>')
                st, v = pts[-1]
                o.append(f'<circle cx="{X(st):.1f}" cy="{Y(v):.1f}" r="3.5" fill="var({col})"/>')
            nb = 24                                              # hover bands: every arm at that step, largest first
            for bi in range(nb):
                s0, s1 = x1 * bi / nb, x1 * (bi + 1) / nb
                rows = []
                for name, col, pts in series:
                    near = [p for p in pts if s0 <= p[0] < s1 or (bi == nb - 1 and p[0] == x1)]
                    if near:
                        rows.append((near[-1][1], f'<span style="color:var({col})">●</span> {esc(name)}: {near[-1][1]:.{4 if mk == "upd" else 3}g}{unit[mk]}'))
                if rows:
                    tip = f"<b>{q} · step ~{int(s1):,}</b><br>" + "<br>".join(r for _, r in sorted(rows, key=lambda r: -r[0]))
                    o.append(f'<rect class="hit" x="{X(s0):.1f}" y="{mt}" width="{X(s1) - X(s0):.1f}" height="{h - mt - mb}" data-tip="{esc(tip)}"/>')
            o.append("</svg>")
            charts.append(f'<div>{"".join(o)}</div>')
        panes.append(f'<div data-ref="{mk}"{"" if j == 0 else " hidden"}><div class="rt-grid">{"".join(charts)}</div><p class="note">{t["rt_st_def"][mk]}</p></div>')
    tog = '<div class="rtoggle">' + "".join(f'<button data-r="{mk}" class="{"on" if j == 0 else ""}">{esc(t["rt_st_name"][mk])}</button>' for j, (mk, _, _, _) in enumerate(RT_STATS)) + "</div>"
    legend = '<div class="legend">' + "".join(f'<span><i style="background:var({col})"></i>{esc(t[key])}</span>' for key, _, col in RT_HEAT_ARMS) + "</div>"
    return f"""<h3>{esc(t['rt_st_h'])}</h3><p class="note">{t['rt_st_note']}</p>
<div class="card">{tog}{legend}{"".join(panes)}</div>"""


def router_section(R, t, cols):
    """Settings table (what each arm changes) + one dot chart of the results (overall_tsqw, arm - partner; q0 = 3-seed mean +- se,
    q1-q3 = seed 1); no numeric result table (user)."""
    # 1) settings
    # two tables, one per control (user), as the two charts below; Change ID, partner and rungs-run columns dropped (user)
    drop = {1, 8, 9}
    head = "".join(f'<th class="l">{esc(x)}</th>' for i, x in enumerate(t["rt_set_head"]) if i not in drop)
    def set_table(rows, title):
        body = "".join("<tr>" + "".join(f'<td class="l">{c}</td>' for i, c in enumerate(row) if i not in drop) + "</tr>" for row in rows)
        return f'<div class="card"><div class="ct">{esc(title)}</div><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
    rows = t["rt_set_rows"]
    settings = (set_table([rows[i] for i in (0, 1, 2, 3)], t["rt_set_l"]) + set_table([rows[i] for i in (4, 5)], t["rt_set_r"])
                + f'<p class="note">{t["rt_set_note"]}</p>')
    # 2) results: the same line-chart form as the optimizer / KD sections (x = model size, one line per comparison; q0 = 3-seed mean)
    mk = "00_overall_tsqw"
    # two charts side by side, one per control (user): MuonH arms - control 1 (MUONH_FIT), AdamW arm - control 2 (BASE)
    def chart_for(arms, partner):
        series, seeds = [], {}
        for key, arm, col in arms:
            pts = []
            for q in RUNGS:
                d = _paired(R, arm, partner, q, mk)
                if not d:
                    continue
                v = sum(d) / len(d)
                pts.append((N_ACTIVE[q], v, f"{esc(t[key])} · {q} ({N_LABEL[q]})<br>overall_tsqw {v:+.4f}" + (f"（{len(d)} seeds）" if len(d) > 1 else "")))
                if len(d) > 1:
                    seeds[(t[key], N_ACTIVE[q])] = d          # hollow per-seed points; filled point / line = their mean (user)
            if pts:
                series.append((t[key], col, pts, False))
        if not series:
            return f'<p class="note">{esc(t["pending"])}</p>'
        xs_ = {q: N_ACTIVE[q] for q in RUNGS if any(abs(p[0] - N_ACTIVE[q]) < 1 for _, _, pts, _ in series for p in pts)}
        return legend([(n, c) for n, c, _, _ in series]) + line_chart(series, xs_, t["rt_ylab"], lambda v: f"{v:.3f}", w=560, h=360, labels=False, zero=True, seeds=seeds)
    left = chart_for([("rt_l_a", "MUONH_FIT_DSV3", "--s2"), ("rt_l_b2", "MUONH_FIT_DSV3_RTADAMW_RT0P19", "--s4"), ("rt_l_b3", "MUONH_FIT_DSV3_RT0P19", "--s3")], "MUONH_FIT")
    right = chart_for([("rt_l_c", "BASE_DSV3_ONETREE", "--s5")], "BASE")
    chart = (f'<div class="rt2" style="display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px;align-items:stretch">'
             f'<div class="card" style="min-width:0;width:100%;margin:10px 0;box-sizing:border-box"><div class="ct" style="min-height:2.6em">{esc(t["rt_left_h"])}</div>{left}</div>'
             f'<div class="card" style="min-width:0;width:100%;margin:10px 0;box-sizing:border-box"><div class="ct" style="min-height:2.6em">{esc(t["rt_right_h"])}</div>{right}</div></div>')
    # 3) LR checks S / S_C: the LR-grid cell of the other sections (user: a grid with its curve, under the two charts). S in multiples of the
    #    ArchScale original at q0 (report rule: from the actual peak LR), S_C of the adamw_active rule; 0 = the 3-seed mean at the current LR
    def lr_check(base, tags, ref_lr, muon_ref):
        change = lambda suffix, tag, seed="": base + seed if tag == "1X" else f"{base}_LR{tag}{seed}"
        mults = [(round(TRAIN[(change("", g), "q0")]["lr"] / ref_lr, 2), g) for g in tags if (TRAIN.get((change("", g), "q0")) or {}).get("lr")]
        if len(mults) < 2:
            return None, None, f'<div class="pend">{esc(t["g_pending"])}</div>'
        G = dict(GH, mults=mults, change=change, pool=True, vs=t["rt_vs"], xr=(mults[0][0] / 1.35, mults[-1][0] * 1.35))
        if muon_ref:
            G["ref"] = "muon"                              # MUON_LR1 = the ladder's adamw_active LR per rung (AdamW and Muon share it)
        cur = dict((g, m) for m, g in mults).get("1X")
        return cur, cell_fit(R, "q0", "", mk, G)[1], mini_lr(R, t, "q0", "", mk, "overall_tsqw", lr1=ref_lr, G=G)
    def reading(key, cur, fit):
        if not fit or not cur:
            return t["g_fewpts"]
        return t[key].format(m=fit[0], c=cur, r=fit[0] / cur, g=-fit[1]) + (" " + t["g_edge"] if fit[2] else "")
    s_cur, s_fit, s_cell = lr_check("MUONH_FIT_DSV3", ["0P7X", "1X", "1P4X"], archscale_lr(12, 4577 * 256 * 8192), False)
    c_cur, c_fit, c_cell = lr_check("BASE_DSV3_ONETREE", ["0P5X", "1X", "2X"], MUON_LR1["q0"], True)
    lrrows = "".join(f'<tr><th class="grow">{esc(t[a])}</th><td class="gc up"><div class="glab">{esc(t[b])}</div>{c}</td><td class="l ckr">{esc(reading(k, cu, f))}</td></tr>'
                     for a, b, c, k, cu, f in (("rt_s", "rt_s_lab", s_cell, "rt_s_r", s_cur, s_fit), ("rt_sc", "rt_sc_lab", c_cell, "rt_sc_r", c_cur, c_fit)))
    lrhead = "".join(f"<th>{esc(x)}</th>" for x in t["rt_lr_head2"])
    lrgrid = (f'<p class="note">{esc(t["rt_lr_h2"])}</p><div class="card" style="overflow:visible"><table class="lrgrid"><thead><tr>{lrhead}</tr></thead>'
              f'<tbody>{lrrows}</tbody></table></div>')
    return f"""<h2>{esc(t['rt_h'])}</h2><p class="note">{t['rt_note']}</p>
<div class="card"><div class="ct">{esc(t['rt_two_h'])}</div>{t['rt_two']}</div>
<div class="card"><div class="ct">{esc(t['rt_gate_h'])}</div>{t['rt_gate']}</div>
{settings}
<p class="note">{esc(t['rt_chart_h'])}</p>{chart}
<p class="note">{t['rt_health_note']}</p>
{lrgrid}
{router_trends(t)}"""


# Router TODO 9 (Runlong.md, 2026-10-27): the load-balancing loss replaced by the phi-balancing negative-entropy loss (arXiv 2605.15403).
# ROUTER_NE_<alpha>[_NOBIAS|_SOFTMAX|_ETA<x>]; pair MUONH_FIT_DSV3 (DSV3 base, bias kept) or MUONH_FIT (_SOFTMAX), seed 1 at q0.
def ne_section(R, t, cols):
    arms = sorted({c for c, q in R if c.startswith("ROUTER_NE_")}, key=lambda c: ("SOFTMAX" in c, "NOBIAS" in c, "ETA" in c, c))
    if not arms:
        return ""
    alpha = lambda c: c.split("_")[2].replace("1EM", "1e-")
    rows = []
    for q in RUNGS:
        for c in arms:
            if (c, q) not in R:
                continue
            pair = "MUONH_FIT" if "SOFTMAX" in c else "MUONH_FIT_DSV3"
            base = t["ne_soft"] if "SOFTMAX" in c else (t["ne_nobias"] if "NOBIAS" in c else t["ne_dsv3"])
            extra = (" · η " + c.split("ETA")[1].replace("P", ".")) if "ETA" in c else ""
            cells = "".join(_fmt_paired(_paired(R, c, pair, q, k)[:1], t) for k, _ in cols)
            rows.append(f'<tr><td class="l">{q}</td><td class="l">{esc(c)}</td><td class="l">{esc(base)}{esc(extra)}</td><td>{alpha(c)}</td><td class="l">{pair}</td>{cells}</tr>')
    head = (f'<tr><th class="l">rung</th><th class="l">Change ID</th><th class="l">{esc(t["ne_base"])}</th><th>α</th><th class="l">{esc(t["ne_pair"])}</th>'
            + "".join(f"<th>{esc(n)}</th>" for _, n in cols) + "</tr>")
    return f"""<h3>{esc(t['ne_h'])}</h3><p class="note">{t['ne_note']}</p>
<div class="card"><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>"""


# Checks of the law's assumptions (Runlong.md TODO 3 steps 27-35), all at T = 9.6B (4,577 steps at GBS 2^21) unless noted:
# depth x width grid of LR sweeps (the ladder rungs sit on the diagonal), batch / floor / per-group-LR checks.
GA = dict(GH, mults=MULTS + [(5.6, "5P6X")], rebase=True)
AW_ROWS = [12, 18, 24, 30, 36, 42, 48, 60, 96]
AW_COLS = [512, 768, 1024, 1280, 1536, 1792, 2048, 4096]
AW_CELLS = {(12, 512): ("q0", ""), (18, 768): ("q1", "_IT4577"), (24, 1024): ("q2", "_IT4577"), (30, 1280): ("q3", "_IT4577"),
            (36, 1536): ("q4", "_IT4577"), (42, 1792): ("q5", "_IT4577"), (48, 2048): ("q6", "_IT4577"),
            (12, 1024): ("q2", "_L12H1024_IT4577"), (24, 512): ("q0", "_L24H512_IT4577"), (24, 1536): ("q4", "_L24H1536_IT4577"),
            (24, 2048): ("q4", "_L24H2048_IT4577"), (24, 4096): ("q6", "_L24H4096_IT4577"), (48, 1024): ("q2", "_L48H1024_IT4577"),
            (60, 1024): ("q2", "_L60H1024_IT4577"), (96, 1024): ("q2", "_L96H1024_IT4577")}


AW_UNSTABLE = {(24, 4096)}                               # router collapse (load-balancing loss 9-16 all run; DECISIONS 2026-10-26): not an LR point


# Diagonal fit (user 2026-10-28): only the ladder cells q0-q6 at 9.6B (h ~ 43 L, so the exponent is the along-ladder one).
# Runlong's step-7 rule (Runlong.md TODO 8 A.2 / TODO 3): multiplier(rung) = the rung's measured 9.6B m* x (T / 9.6B)^c, c ~ 0.03 [0, 0.06];
# anchors m* (Boxiang, val_v2 R1) q0-q5 1.19 / 1.45 / 1.59 / 1.66 / 1.75 / 1.81 (TODO 3 "what stands"); q6 1.82 = table 2.05 / (S6 / 4577)^0.03.
RUNLONG_TABLE = {"q0": 1.19, "q1": 1.48, "q2": 1.65, "q3": 1.80, "q4": 1.92, "q5": 2.01, "q6": 2.05}       # Runlong.md TODO 8 A.2 multipliers
RUNLONG_LR = {"q0": 1.59e-2, "q1": 1.11e-2, "q2": 8.26e-3, "q3": 6.51e-3, "q4": 5.36e-3, "q5": 4.49e-3, "q6": 3.77e-3}   # his peak LRs
MAIN_T = {"q0": 9.6e9, "q1": 30.8e9, "q2": 71.1e9, "q3": 136.6e9, "q4": 233.6e9, "q5": 368.0e9, "q6": 546.1e9}       # TPP-200 main-ladder tokens
_steps = lambda q: round(MAIN_T[q] / 2 ** 21)
RUNLONG_ANCHORS = {q: RUNLONG_TABLE[q] / (_steps(q) / 4577) ** 0.03 for q in RUNLONG_TABLE}   # implied 9.6B anchors (reproduce his LRs)
DIAG = [("q0", 12, ""), ("q1", 18, "_IT4577"), ("q2", 24, "_IT4577"), ("q3", 30, "_IT4577"), ("q4", 36, "_IT4577"), ("q5", 42, "_IT4577"), ("q6", 48, "_IT4577")]


def diag_fit_section(R, t, mkey="00_overall_tsqw", mname="overall_tsqw"):
    """Only the rule of Runlong (user 2026-10-28: no fit, no curve)."""
    return f"""<div class="card"><table class="lf"><tbody><tr><td class="l"><span class="sq" style="color:#17becf">■</span> {esc(t['dg_runlong'])}</td>
<td class="l">η(rung) = ArchScale(L, T) × m*<sub>9.6B</sub>(rung) × (T / 9.6B)<sup>0.03</sup></td></tr></tbody></table>
<p class="note">{t['dg_short']}</p></div>"""


def assume_section(R, t, mkey="00_overall_tsqw", mname="overall_tsqw"):
    head = '<tr><th>L \\ h</th>' + "".join(f"<th>h {c}</th>" for c in AW_COLS) + "</tr>"
    rows = []
    for i, L in enumerate(AW_ROWS):
        cells = []
        for h in AW_COLS:
            spec = AW_CELLS.get((L, h))
            if spec is None:
                cells.append('<td class="gx"></td>')
                continue
            rung, suf = spec
            ladder = suf in ("", "_IT4577")
            no1 = val(R, cell_change(suf, "1X", "", GA), rung, mkey) is None and bool(cell_fit(R, rung, suf, mkey, GA)[0])
            lab = (t["aw_ladder"].format(q=rung) if ladder else t["aw_extra"]) + (t["aw_no1x"] if no1 else "")
            if (L, h) in AW_UNSTABLE:
                lab = t["aw_unstable"]
            up = " up" if i >= len(AW_ROWS) - 3 else ""
            cells.append(f'<td class="gc{up}"><div class="glab">{esc(lab)}</div>{mini_lr(R, t, rung, suf, mkey, mname, lr1=law_lr(L, 4577), G=GA)}</td>')
        rows.append(f'<tr><th class="grow">L {L}</th>{"".join(cells)}</tr>')
    grid = f'<table class="lrgrid"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'
    # other checks as cells (user): control cell | experiment cell | reading
    def cell(rung, suf, L, lab, extra=None):
        return f'<td class="gc up"><div class="glab">{esc(lab)}</div>{mini_lr(R, t, rung, suf, mkey, mname, lr1=law_lr(L, 4577), G=GA, extra=extra)}</td>'
    v = lambda c, q: val(R, c, q, mkey)
    b3 = v("MUONH_LR1X_IT4577", "q3")
    split = [(1.74, v(c, "q3") - b3, t[k], col) for c, k, col in (("MUONH_LRSPLIT_A_IT4577", "ck_split_a", "#9a9890"), ("MUONH_LRSPLIT_B_IT4577", "ck_split_b", "var(--a08)"),
             ("MUONH_LRSPLIT_C_IT4577", "ck_split_c", "#17becf")) if v(c, "q3") is not None and b3 is not None]
    ck = [(t["ac_batch"], cell("q1", "_IT4577", 18, "GBS 2M"), cell("q1", "_B4M_IT2289", 18, "GBS 4M"), t["ck_batch_r"]),
          (t["ac_floor0"], cell("q0", "", 12, t["ck_floor1"]), cell("q0", "_FLOOR10", 12, t["ck_floor10"]), t["ck_floor_r"]),
          (t["ac_floor3"], cell("q3", "_IT4577", 30, t["ck_floor1"]), cell("q3", "_IT4577_FLOOR10", 30, t["ck_floor10"]), t["ck_floor3_r"]),
          (t["ac_split"], cell("q3", "_IT4577", 30, t["ck_split_ctl"]), cell("q3", "_IT4577", 30, t["ck_split_exp"], extra=split), t["ck_split_r"])]
    ckrows = "".join(f'<tr><th class="grow">{esc(a)}</th>{c1}{c2}<td class="l ckr">{esc(r)}</td></tr>' for a, c1, c2, r in ck)
    ckhead = "".join(f"<th>{esc(x)}</th>" for x in t["ck_head"])
    return f"""<h2>{esc(t['aw_h'])}</h2><p class="note">{t['aw_note'].format(m=esc(mname))}</p>
<div class="card" style="overflow:visible">{grid}</div>
{diag_fit_section(R, t, mkey, mname)}
<h3>{esc(t['ac_h'])}</h3><p class="note">{t['ck_note']}</p><div class="card" style="overflow:visible"><table class="lrgrid"><thead><tr>{ckhead}</tr></thead><tbody>{ckrows}</tbody></table></div>"""


# External reference models, scored with the SAME doc-isolated 115-split protocol (reports_xx/mimo_doc_eval.py, vLLM prompt_logprobs;
# MiMo token ids == the ladder's Qwen3 ids, so the per-token losses compare directly)
REF_MODELS = [("MIMO_V25_PRO_BASE", "MiMo-V2.5-Pro-Base", False), ("MIMO_V26_PRO_RL", "MiMo-V2.6-Pro-RL", False),
              ("DEEPSEEK_V4_FLASH_BASE", "DeepSeek-V4-Flash-Base", True), ("KIMI_K3", "Kimi-K3", True)]   # True = own tokenizer
REF_ARMS = [("BASE", "AdamW"), ("OPT_MUON", "Muon"), ("OPT_MUONH", "MuonH"), ("KD_MUONH_A0P8", "MuonH + KD α 0.8")]
REF_STATS = [("loss", "val_loss"), ("ppl", "ppl"), ("bpb", "bpb")]


def ref_section(R, t, cols):
    """Absolute val_loss / ppl / bpb (own toggle): the reference models next to each ladder arm's largest evaluated rung, + best
    ladder - MiMo-Base. Models with their own tokenizer: loss / ppl are per THEIR token (greyed, not comparable); bpb compares."""
    refs = [(n, R[(c, "ref")], f) for c, n, f in REF_MODELS if (c, "ref") in R]
    if not refs:
        return ""
    # ONLY our best ladder checkpoint (user): lowest overall_tsqw loss over every evaluated ladder checkpoint
    best = min(((c, q) for c, q in R if q != "ref" and "00_overall_tsqw" in R[(c, q)]), key=lambda k: R[k]["00_overall_tsqw"]["loss"])
    lad = [(f"{t['ref_best']}: {best[0]} · {best[1]}" + (f" ({N_LABEL[best[1]]})" if best[1] in N_LABEL else ""), R[best], False)]
    base = next(m for n, m, f in refs if not f)
    head = '<tr><th class="l"></th>' + "".join(f"<th>{esc(n)}</th>" for _, n in cols) + "</tr>"
    panes = []
    for j, (sec, sname) in enumerate(REF_STATS):
        fmt = (lambda v: f"{v:.3f}") if sec == "ppl" else (lambda v: f"{v:.4f}")
        cell = lambda m, k: fmt(m[k][sec]) if k in m and sec in m[k] else "·"
        rows = []
        for n, m, f in refs:
            dim = f and sec != "bpb"
            rows.append(f'<tr{" class=\"dim\"" if dim else ""}><td class="l"><b>{esc(n)}</b>{" †" if dim else ""}</td>'
                        + "".join(f"<td>{'' if dim else '<b>'}{cell(m, k)}{'' if dim else '</b>'}</td>" for k, _ in cols) + "</tr>")
        rows += [f'<tr><td class="l">{esc(n)}</td>' + "".join(f"<td>{cell(m, k)}</td>" for k, _ in cols) + "</tr>" for n, m, _ in lad]
        gap = []
        for k, _ in cols:
            v = [m[k][sec] for _, m, _ in lad if k in m and sec in m[k]]
            if not v or k not in base or sec not in base[k]:
                gap.append("<td>·</td>")
            elif sec == "ppl":
                gap.append(f'<td class="pos">{min(v) / base[k][sec]:.2f}×</td>')
            else:
                gap.append(f'<td class="pos">+{min(v) - base[k][sec]:.{3 if sec == "loss" else 4}f}</td>')
        rows.append(f'<tr><td class="l">{esc(t["ref_gap"])}{" (×)" if sec == "ppl" else ""}</td>{"".join(gap)}</tr>')
        panes.append(f'<div data-ref="{sec}"{"" if j == 0 else " hidden"}><table><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>')
    tog = '<div class="rtoggle">' + "".join(f'<button data-r="{sec}" class="{"on" if j == 0 else ""}">{esc(n)}</button>' for j, (sec, n) in enumerate(REF_STATS)) + "</div>"
    foot = f'<p class="note">{t["ref_dag"]}</p>' if any(f for _, _, f in refs) else ""
    return f"""<h2>{esc(t['ref_h'])}</h2><p class="note">{t['ref_note']}</p>
<div class="card">{tog}{"".join(panes)}{foot}</div>"""


OPT_LR = [("AdamW", "BASE", "BASE_LR0P5X", "BASE_LR2X")]
OPT_LR_ROOT = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos_llm_scaling_ladder/results"


def opt_lr_section(R, t, mkey, mname):
    """AdamW / Muon: 0.5x / 1x / 2x of their (AdamW) LR law at a rung -> parabola m* (3 points), absolute LR from the 1x run's argv."""
    rows = []
    for opt, one, half, two in OPT_LR:
        for q in RUNGS:
            b = val(R, one, q, mkey)
            pts = [(m, val(R, c, q, mkey) - b) for m, c in ((0.5, half), (1.0, one), (2.0, two)) if b is not None and val(R, c, q, mkey) is not None]
            if len(pts) < 2:
                continue
            lr1 = None
            try:
                a = open(os.path.join(OPT_LR_ROOT, one, "ckpts", q, "argv.txt")).read().split(); lr1 = float(a[a.index("--lr") + 1])
            except (OSError, ValueError, IndexError):
                pass
            fit = fit_mstar(pts)
            cell = lambda m: next((f'{"−" if y < 0 else "+"}{abs(y):.4f}' for mm, y in pts if mm == m), "·")
            mtxt = (f"<b>{fit[0]:.2f}×</b>{' ' + esc(t['g_edge']) if fit[2] else ''}" if fit else "—")
            etxt = (f"{fit[0] * lr1:.2e}" if fit and lr1 else "·")
            rows.append(f'<tr><td class="l">{opt}</td><td class="l">{q}</td><td>{cell(0.5)}</td><td>0</td><td>{cell(2.0)}</td><td>{mtxt}</td>'
                        f'<td>{etxt}</td><td>{f"{lr1:.2e}" if lr1 else "·"}</td></tr>')
    head = (f'<tr><th class="l">{esc(t["ol_opt"])}</th><th class="l">rung</th><th>0.5×</th><th>1×</th><th>2×</th><th>m*</th><th>η*</th><th>1× LR</th></tr>')
    body = "".join(rows) if rows else f'<tr><td class="l" colspan="8">{esc(t["g_pending"])}</td></tr>'
    return f"""<h2>{esc(t['ol_h'])}</h2><p class="note">{t['ol_note'].format(m=esc(mname))}</p>
<div class="card"><table><thead>{head}</thead><tbody>{body}</tbody></table></div>"""


def _lstsq(X, Y):
    """Ordinary least squares with standard errors (tiny k, pure python)."""
    k = len(X[0])
    def solve(A, b):
        M = [A[i][:] + [b[i]] for i in range(k)]
        for i in range(k):
            p = max(range(i, k), key=lambda r: abs(M[r][i])); M[i], M[p] = M[p], M[i]
            for r in range(k):
                if r != i:
                    f = M[r][i] / M[i][i]; M[r] = [a - f * bb for a, bb in zip(M[r], M[i])]
        return [M[i][k] / M[i][i] for i in range(k)]
    A = [[sum(x[i] * x[j] for x in X) for j in range(k)] for i in range(k)]
    coef = solve(A, [sum(x[i] * y for x, y in zip(X, Y)) for i in range(k)])
    res = [y - sum(c * xx for c, xx in zip(coef, x)) for x, y in zip(X, Y)]
    s2 = sum(r * r for r in res) / max(len(Y) - k, 1)
    se = [math.sqrt(max(s2 * solve(A, [1.0 if i == j else 0.0 for i in range(k)])[j], 0.0)) for j in range(k)]
    return coef, se, res


def fmt_exp(x):
    return f"{'−' if x < 0 else '+'}{abs(x):.3f}"


def law_fit_section(R, t, mkey="00_overall_tsqw", G=None):
    """ArchScale-form fit over the grid cells: log eta* = log eta0 + beta log(L/8) + gamma log(T/10.4B); edge cells excluded."""
    G = G or GH
    cells = []
    for (rung, c), (suffix, tok, tpp, kind) in G["cells"].items():
        f = cell_fit(R, rung, suffix, mkey, G)[1]
        if not f:
            continue
        L = dict(G["rows"])[rung]; it = GRID_ITERS[(rung, c)]
        X = N_ACTIVE[rung] if G is GM else L              # size variable: MuonH law in layers L, Muon (adamw_active rule) in active params N (user)
        cells.append((rung, tok, tpp, X, it * 2 ** 21, f[0] * G["lr1"](rung, c), f[2]))
    use = cells                                          # extrapolated vertices included (marked in the table)
    if len(use) < 4:
        return ""
    x0 = 1.08e8 if G is GM else 8
    coef, se, res = _lstsq([(1.0, math.log(x[3] / x0), math.log(x[4] / 1.04e10)) for x in use], [math.log(x[5]) for x in use])
    eta0, beta, gamma = math.exp(coef[0]), coef[1], coef[2]
    box = [0.01725, -0.1764, -0.2591] if G is GH else None   # Boxiang's frozen fit v2 (2026-10-25, MUONH_FIT exp file; val_v2, 16 cells)
    fix = None                                           # MuonH: the same fit with the token exponent FIXED at ArchScale's -0.32 (Runlong)
    if False:                                            # (fixed -0.32 plane removed, user 2026-10-26)
        cf, _, _ = _lstsq([(1.0, math.log(x[3] / 8)) for x in use], [math.log(x[5]) + 0.32 * math.log(x[4] / 1.04e10) for x in use])
        fix = [math.exp(cf[0]), cf[1], -0.32]
    fit = lambda L, T: eta0 * (L / x0) ** beta * (T / 1.04e10) ** gamma
    A0, Ab, Ag = G["arch"]()
    if G is GM:                                          # exact, in N: the current adamw_active rule and Boxiang's Muon fit v3 (DECISIONS 2026-10-26)
        A0, Ab, Ag = 2.35e-3, -1 / 3, 0.0
        box = [4.93e-3, -0.339, -0.361]
    arch = lambda L, T: A0 * (L / x0) ** Ab * (T / 1.04e10) ** Ag
    rows = "".join(f'<tr><td class="l">{q}</td><td class="l">{tok}</td><td>{tpp}</td><td>{eta:.4f}{" " + esc(t["g_edge"]) if ex else ""}</td><td>{fit(L, T):.4f}</td><td>{100 * (math.exp(r) - 1):+.1f}%</td></tr>'
                   for (q, tok, tpp, L, T, eta, ex), r in zip(use, res))
    pred = [(t["lf_pred"][0], 24, 33906 * 2 ** 21), (t["lf_pred"][1], 30, 65165 * 2 ** 21), (t["lf_pred"][2], 48, 546.1e9), (t["lf_pred"][3], 94, 4189e9)]
    prow = "".join(f'<tr><td class="l">{esc(n)}</td><td>{L}</td><td>{T / 1e9:,.0f}B</td><td>{arch(L, T):.5f}</td><td>{fit(L, T):.5f}</td><td><b>{fit(L, T) / arch(L, T):.2f}×</b></td></tr>' for n, L, T in pred)
    excl = ", ".join(f"{x[0]}@{x[1]}" for x in cells if x[6]) or "—"
    plane_json = esc(json.dumps({
        "fit": [eta0, beta, gamma], "arch": [A0, Ab, Ag], "fix": fix, "box": box,
        "pts": [{"X": x[3], "T": x[4], "eta": x[5], "ex": bool(x[6]), "lab": (f"{x[0]} · N {N_LABEL[x[0]]}" if G is GM else f"{x[0]} · {x[3]} {t['g_layers']}") + f" · T {x[4] / 1e9:.1f}B · TPP {x[2]}"} for x in use],
        "x": ({"ref": 1.08e8, "lo": 3.5e7, "hi": 1.0e9, "ticks": [N_ACTIVE[q] for q in ("q0", "q1", "q2", "q3")], "n": True} if G is GM else
              {"ref": 8, "lo": 10, "hi": 48, "ticks": [12, 18, 24, 30, 36, 42, 48], "n": False}),
        "lab": {"L": t["mg_lf_axN"] if G is GM else t["lf_axL"], "T": t["lf_axT"], "eta": "η*"}}))
    return f"""<h3>{esc(t['lf_h'])}</h3>
<div class="card"><table class="lf"><tbody>
<tr><td class="l"><span class="sq" style="color:#9a9890">■</span> {esc(t['lf_arch'])}</td><td class="l">{"η*(L, T) = 0.016 × (L/8)<sup>−0.5</sup> × (T/10.4B)<sup>−0.32</sup>" if G is GH else f"η = 2.35e-3 × (N<sub>active</sub> / 1.08e8)<sup>−1/3</sup>{esc(t['mg_noT'])}"}</td></tr>
<tr><td class="l"><span class="sq" style="color:var(--a08)">■</span> <b>{esc(t['lf_ours'])}</b></td><td class="l"><b>{f"η* = {eta0:.5f} × (N<sub>active</sub> / 1.08e8)<sup>{fmt_exp(beta)}</sup>" if G is GM else f"η*(L, T) = {eta0:.4f} × (L/8)<sup>{fmt_exp(beta)}</sup>"} × (T/10.4B)<sup>{fmt_exp(gamma)}</sup></b></td></tr>
{f"""<tr><td class="l"><span class="sq" style="color:var(--a08)">■</span> {esc(t['lf_fix'])}</td><td class="l">η*(L, T) = {fix[0]:.4f} × (L/8)<sup>{fmt_exp(fix[1])}</sup> × (T/10.4B)<sup>−0.32</sup></td></tr>""" if fix else ""}
{f"""<tr><td class="l"><span class="sq" style="color:#17becf">■</span> {esc(t['lf_box'])}</td><td class="l">{"η = 4.93e-3 × (N<sub>active</sub> / 1.08e8)<sup>−0.339</sup> × (T/10.4B)<sup>−0.361</sup>" if G is GM else f"η*(L, T) = {box[0]:.5f} × (L/8)<sup>{fmt_exp(box[1])}</sup> × (T/10.4B)<sup>{fmt_exp(box[2])}</sup>"}</td></tr>""" if box else ""}
</tbody></table>
<div class="plane3d" data-plane='{plane_json}'><div class="legend"><span><i style="background:#9a9890"></i>{esc(t['lf_arch'])}</span><span><i style="background:var(--a08)"></i>{esc(t['lf_ours'])}</span>{f'<span><i style="background:var(--a08)"></i>{esc(t["lf_fix"])}</span>' if fix else ""}{f'<span><i style="background:#17becf"></i>{esc(t["lf_box"])}</span>' if box else ""}<span><i style="background:var(--s1);border-radius:50%;width:9px;height:9px"></i>{esc(t['lf_pts'])}</span><span style="color:var(--muted)">{esc(t['lf_drag'])}</span></div><svg viewBox="0 0 760 460" width="100%"></svg></div></div>"""


def grid_section(R, t, metrics, G=None):
    G = G or GH
    if G["tx"]:                                           # Muon grid: its own heading / note / law labels
        t = {**t, **{k[len(G["tx"]) + 1:]: v for k, v in t.items() if k.startswith(G["tx"] + "_")}}
    def table(mkey, mname):
        head = '<tr><th></th>' + "".join(f'<th>{esc(t["g_cols"][c])}</th>' for c in G["cols"]) + "</tr>"
        rows = []
        for rung, L in G["rows"]:
            cells = []
            for c in G["cols"]:
                spec = G["cells"].get((rung, c))
                if spec is None:
                    cells.append('<td class="gx"></td>')
                    continue
                suffix, tok, tpp, kind = spec
                lab = t["g_kind"][kind].format(tok=tok, tpp=tpp)
                lr1 = G["lr1"](rung, c)
                up = " up" if rung in [r for r, _ in G["rows"][-3:]] else ""   # bottom rows: pop the curve upward
                cells.append(f'<td class="gc{up}"><div class="glab">{esc(lab)}</div>{mini_lr(R, t, rung, suffix, mkey, mname, lr1=lr1, G=G)}</td>')
            rows.append(f'<tr><th class="grow">{rung}<br><span>{L} {esc(t["g_layers"])} · {N_LABEL[rung]}</span></th>{"".join(cells)}</tr>')
        return f'<table class="lrgrid"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table>'
    m, n = metrics                                        # ONE fixed metric for this section (not tied to the toggle above)
    tab = table(m, n)                                     # (fills MULT_MISMATCH)
    mm = sorted(x for x in MULT_MISMATCH if any(x[0] == cell_change(sf, tg, sd, G) for (_, _), (sf, *_r) in G["cells"].items() for _, tg in G["mults"] for sd, _ in SEEDS))
    warn = (f'<p class="note"><b>{esc(t["g_mm"])}</b> ' + "; ".join(f"{c} {q}: {nm:g}× → {a:g}×" for c, q, nm, a in mm) + "</p>") if mm else ""
    return f"""<h2>{esc(t['g_h'])}</h2><p class="note">{t['g_metric'].format(m=esc(n))} {t['g_note']} <b>{t['g_ref_muon' if G.get('ref') == 'muon' else 'g_ref']}</b></p>{warn}
<div class="card" style="overflow:visible">{tab}</div>
{law_fit_section(R, t, m, G)}"""



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
        f = cell_fit(R, rung, suffix, "00_overall_tsqw")[1]
        if f:
            lr_best.append(f"{rung}@{tok}（TPP {tpp}）{f[0]:.2f}×{'*' if f[2] else ''}（η* {f[0] * law_lr(dict(GRID_ROWS)[rung], GRID_ITERS[(rung, c)]):.4f}）")
    if lr_best:
        finds.append(t["find_lr"].format(best="；".join(lr_best) if lang == "zh" else "; ".join(lr_best).replace("（", " (").replace("）", ")")))
    t1, _ = tables(R, t, COLS)
    pcols = [(key, n) for k, key in PSETS for kk, n in COLS if kk == key] + [("mean6", t["mean6"])]
    p1, p2 = tables(RP, t, pcols)
    PMETRICS = [("01_val_v2_overall_tw", "val_v2"), ("02_uncheatable_tw", "uncheatable"), ("03_codefresh_tw", "codefresh"), ("mean6", t["mean6"])]
    doc = f"""
<h2>{esc(t['find_h'])}</h2><div class="card"><ul class="f">{''.join(f'<li>{f}</li>' for f in finds)}</ul></div>
{token_pie(t)}
{chart_sections(R, t, METRICS)}
{variant_section(R, t, METRICS, COLS)}
{router_section(R, t, COLS + [("00_overall_tsqw", "overall_tsqw")])}
{ne_section(R, t, COLS + [("00_overall_tsqw", "overall_tsqw")])}
{grid_section(R, t, ("00_overall_tsqw", "overall_tsqw"))}
{grid_section(R, t, ("00_overall_tsqw", "overall_tsqw"), GM)}
{assume_section(R, t)}
{arch_section(R, t, METRICS)}
{ref_section(R, t, COLS + [("00_overall_tsqw", "overall_tsqw")])}
<h2>{esc(t['t1_h'])}</h2><p class="note">{esc(t['pending'])}</p><div class="card">{t1}</div>
<h2>{esc(t['notes_h'])}</h2><div class="card"><ul class="f">{''.join(f'<li>{n}</li>' for n in t['notes'])}</ul></div>"""
    packed = f"""<div class="card"><ul class="f">{''.join(f'<li>{n}</li>' for n in t['p_notes'])}</ul></div>
{chart_sections(RP, t, PMETRICS)}
{variant_section(RP, t, PMETRICS, pcols)}
{arch_section(RP, t, pcols)}
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
