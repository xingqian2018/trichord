#!/usr/bin/env python3
"""reports_xx/gen_eval_html.py - visual KD-vs-CE report (zh + en, self-contained HTML with inline SVG charts) from the per-checkpoint JSONs
of collect_doc_eval.py.

    python3 reports_xx/collect_doc_eval.py && python3 reports_xx/gen_eval_html.py
    -> reports_xx/qwen3_scaling_ladder_eval.html (Chinese), reports_xx/qwen3_scaling_ladder_eval_en.html (English)
    python3 reports_xx/gen_eval_html.py --only down,ref
    -> the same build, but only these tabs (data-pane keys: doc prod docold ref down fit packed) are replaced in the files on disk;
       every other tab keeps the bytes it has there. For the sessions that share the report: each one publishes its own tabs only.

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
REF_COLS = ([("00_overall_tsqw", "overall_tsqw"), ("00_overall_tw", "overall_tw"), ("00_overall_macro", "overall_macro")]   # reference table:
            + [c for c in COLS if not c[0].startswith("00_overall")])                                                     # overall first (user 11-04)
METRICS = [("00_overall_tsqw", "overall_tsqw"), ("00_overall_tw", "overall_tw"), ("00_overall_macro", "overall_macro"), ("01_val_v2_overall_tw", "val_v2"), ("02_uncheatable_tw", "uncheatable"), ("03_codefresh_tw", "codefresh"), ("04_private_internal_tw", "private_internal")]
# sequential blue ramp (reference palette), light -> dark
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95"]

T = {
 "zh": {
  "lang": "zh", "title": "59B 生产 run 跟踪 · Qwen3-MoE scaling ladder",
  "sub": "前两个页签 = 59B 生产 run（ULTRA59B_MUONH_KD_QBK3）的 val loss 和下游测评（下游每 2k 步一测；val loss 全量约每 4–6k 步，其间补测 val_v2 / v3small）；后面是外部模型和 ladder 研究的存档（doc-isolated 评测：9 个 decon suite，含 private_internal，dpl 指标定义）。数据生成于 {ts}",
  "nav_arch": "早期研究（另一个文件）：", "nav_arch_here": "早期研究：",
  "title_arch": "Qwen3-MoE scaling ladder · 早期研究存档",
  "sub_arch": "ladder 研究的存档：doc-isolated 评测（9 个 decon suite，含 private_internal，dpl 指标定义）、老实验、LR 拟合、packed。59B 生产 run 的 val loss 和下游测评在主页面 qwen3_scaling_ladder_eval.html。数据生成于 {ts}",
  "nav": ["结论", "优化器（CE）", "KD（MuonH）", "热力图", "数据表", "说明"],
  "muon_fit": "Muon（新拟合 LR）", "ce_fit": "MuonH（新拟合 LR，MUONH_FIT）", "opt_h": "优化器对比", "opt_note": "虚线 = Boxiang 按新拟合的 LR 公式跑的同一优化器（MuonH：fit v2 η = 0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259，目前有 q2 / q3；Muon 没有按新公式跑）。同一数据、同一 token 数，只换优化器（及其 LR 规则 / schedule）。横轴：激活参数（不含 embedding，对数坐标）。",
  "o2_h": "相对 AdamW 的 Δloss（优化器 − AdamW，越负越好）",
  "kd_h": "KD 对 CE", "kd_note": "CE 与 KD 只差 loss（α = 0 vs 0.7 / 0.8 / 0.9），优化器、数据顺序、layout 完全相同。",
  "k2_h": "KD − CE：各 α 的 Δloss（越负越好）",
  "adamw": "AdamW", "muon": "Muon", "vs_adamw": "Δloss（优化器 − AdamW，越负越好）", "mean6": "六集平均",
  "v_h": "尾部估算方式", "v_note": "8 个 run 只差 KD 项里 teacher 分布的截断方式：α = 0.8、MuonH 优化器、seed 1、数据顺序全部相同；基线仍是 <b>CE</b>（MuonH，α = 0）；<b>B64</b>（top-64 + 剩余 tail 合成一个 bucket）就是 KD α 0.8 本身，作为一条线一起画出来，各变体看是否比 B64 更低。负数 = 比 CE 好。",
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
  "arch_y": "Δloss（改动 − 对照，越负越好）", "arch_chart": "每种改动相对各自对照的 Δloss（全部是新 LR 公式；有 3 个 seed 的点 = 3 个 seed 按 seed 配对差的平均，其余是 1 个 seed；鼠标移到某个 rung 看所有改动的数值）", "arch_h": "滑动窗口注意力 SWA", "arch_c": {"MUONH_FIT": "softmax router：各改动 − MUONH_FIT（全注意力，softmax router）", "MUONH_FIT_DSV3": "DSV3 router：各改动 − MUONH_FIT_DSV3（全注意力，DSV3 router）"}, "arch_note": "设置：MuonH，新 LR 公式（fit v2），纯 CE（不用 KD），128 选 8 MoE；q0–q3，部分组 q0 有 3 个 seed。<br>左图：softmax router，0 线 = <b>MUONH_FIT</b>（全注意力）；右图：DSV3 router，0 线 = <b>MUONH_FIT_DSV3</b>（全注意力）。数值 = 改动 − 0 线（doc-isolated，负数 = 改动更好）；各改动的具体设定见下表。",
  "arch_name": {"MUONH_FIT_DSV3_SWS": "SWA + sink（DSV3 router 底座，SWS）− 同底座无 SWA（MUONH_FIT_DSV3）", "SWA_FIT_SINK_W512": "SWA + sink + 窗口 512 − MUONH_FIT", "SWA_FIT_SINK_W128": "SWA + sink + 窗口 128 − MUONH_FIT", "SWA_FIT_SINK_W512_R3": "SWA + sink + 窗口 512，局部:全局 3:1（全注意力层靠底，q3 没训）− MUONH_FIT", "SWA_FIT_SINK_W512_R3T": "SWA + sink + 窗口 512，3:1（全注意力层靠顶）− MUONH_FIT", "SWA": "SWA − MuonH（旧 LR 公式）", "SWA_FIT": "SWA − 无 SWA（MUONH_FIT）", "SWA_FIT_ROPE": "SWA，全注意力层用 RoPE − MUONH_FIT", "SWA_FIT_W4K": "SWA，窗口 4096 − MUONH_FIT", "SWA_FIT_W1K": "SWA，窗口 1024 − MUONH_FIT", "SWA_FIT_SINK": "SWA + 可学习 attention sink（gpt-oss 式）− MUONH_FIT", "SWA_FIT_SINK_W1K": "SWA + sink + 窗口 1024 − MUONH_FIT", "SWA_FIT_DSV3": "SWA（DSV3 router 底座）− 同底座无 SWA（MUONH_FIT_DSV3）", "MUONH_SPLITFC1": "MuonH 拆分 fc1（gate / up 各自更新）− MuonH"},
  "mg_g_h": "Muon 学习率：同样的格子（同一 Muon、CE）", "mg_g_note": "Boxiang 10-24 决定的 Muon 全因子网格：格子与上面 MuonH 相同，倍数 0.25 / 0.5 / 1 / 2 / 4×（间隔更粗，因为 Muon 的最优点位置未知）。1× = ladder 现用的 Muon LR（AdamW 规则，只随模型大小变，<b>不随训练 token 变</b>）。读法同上：大数字 = 抛物线顶点 m*，η* = 绝对 LR，鼠标移到格子上看曲线。目的：Muon 也用各自最优 LR 时，MuonH 是否仍然领先。", "mg_lf_h": "拟合：Muon 最优 LR 随激活参数量 N 和训练量的关系（与 Muon 现用的 adamw_active 规则同一写法）", "mg_lf_arch": "Muon / AdamW 现用 LR（ladder 的 adamw_active 规则，不是 ArchScale；每个格子的 1×）", "mg_lf_box": "Boxiang 的 Muon 拟合 v3（10-26 记录；val_v2、9 格 43 点；未用于实验；有效范围 L 12–30、T 9.6–137B）", "mg_lf_axN": "激活参数 N", "mg_noT": "（只随激活参数量 N 变，不含 T 项）",
  "ref_unsure": "（可能不对）",
  "ref_dag": "vocab size：模型 config 里的 vocab_size，即 embedding 行数（含 padding，所以比 tokenizer 实际的 token 数略大，如 Qwen3 是 151,936 行、151,665 个 token）；我们的 checkpoint 为训练时的 padded vocab。<br>model size：MoE 写「总参数-A激活参数」，dense 只写总参数，数字取自模型卡；我们的 checkpoint 含 embedding 和输出层（行名里的 1.17B 等是不含 embedding 的激活参数）；Qwen3.8-Flash-Next 另有 51B n-gram embedding 不在 125B 里。<br>† 用自己的 tokenizer：val_loss / ppl 按它自己的 token 计，和按 Qwen3 token 计的其他行不可比（只看 bpb）。bpb = 总 NLL ÷（原文 UTF-8 字节数 × ln 2），分母与 ladder 完全相同；文本为 ~/data/eval_ppl_decon 的原文，切分规则同 ladder（每篇第一个 token 不计分，BOS 只作上下文）。", "ref_h": "Pretrain doc-isolated 测试", "ref_gap": "我们最好的 − MiMo-V2.5-Pro-Base", "ref_best": "我们最好的", "ref_w32_h": "32K 窗口：长文档按 32K 切块", "ref_w32_note": "和 8K 窗口同样的模型、同样的设定，只把长文档的切块从 8192 改成 32760 个预测 token（Qwen3-30B-A3B-Base 的最大长度正好是 32768）。打分的 token 完全相同，只是超过 8K 的文档里，后面的 token 能看到更长的上文：""全部 234M 个预测 token 里有 36M（15%）受影响，val_v2 35%、pile 24%、uncheatable 9%、private_internal 4%，codefresh 没有超过 8K 的文档（不变）。和 8K 窗口的差见下面单独的表。ladder 模型都是 8K 训练，这一档没有它们。", "ref_w32_pending": "32K 窗口的评测正在跑（job 36018–36026，Mistral-Large-4 等它的 8K 版跑通后再加），结果出来后自动出现在这里。", "ref_d_h": "32K − 8K（bpb）", "ref_d_note": "bpb：32K 窗口 − 8K 窗口，负数 = 更长上文让 loss 变低。", "ref_win": [["w8", "window = 8K"], ["w32", "window = 32K"], ["wd", "32K − 8K"]], "ref_note": "MiMo-V2.5-Pro-Base（fp8）和 MiMo-V2.6-Pro-RL（官方 mxfp4 权重；也是 KD 的 teacher）用和 ladder 完全相同的 doc-isolated 115 个 split 打分（vLLM prompt_logprobs，每个 split 的预测 token 数与 ladder 一致）；MiMo 的 token id 与 ladder 的 Qwen3 tokenizer 相同，所以每 token 的 loss 可以直接比较。下面是 loss 绝对值（越低越好）；ladder 只列我们最好的一个 checkpoint（所有已测 checkpoint 中 overall_tsqw 最低）；最后一行 = 它减 MiMo-Base（ppl 为比值）。上方按钮切换 val_loss / ppl / bpb。RL 版本在预训练文本上比 Base 高约 0.05，属正常（RL 后训练 + 量化）。Qwen：Qwen3-30B-A3B-Base 和 Qwen3-235B-A22B（bf16）与 ladder 是同一个 Qwen3 tokenizer，loss 直接可比；235B 官方没有发布 Base，这里是后训练版（会偏高）。Qwen3.8-Flash 没有开源权重，用开源的 Qwen3.8-Flash-Next（官方 FP8，后训练版，新 tokenizer，只看 bpb）代替。Qwen3.8-27B 是 dense 模型（bf16，后训练版，没有 Base），tokenizer 与 Qwen3.8-Flash-Next 相同，只看 bpb。",
  "aw_h": "LR 公式的假设检查：深度 × 宽度，对角线用作 ArchScale 锚点（同一 MuonH、CE、训练量 9.6B）", "aw_note": "Runlong 计划 TODO 3 第 27、30、35 步：训练量固定 9.6B（4,577 步，GBS 2M），在不同层数 L 和宽度 h 上各扫一遍 LR（ArchScale 公式的倍数），看最优倍数 m* 是否只随深度变（公式假设宽度无关）。对角线 = ladder 各 rung（q0–q6）；其余格子 = 额外的深度 / 宽度点。指标 {m}；格子读法同上（悬停看曲线）。注意 9.6B 对大模型是严重欠训练（q6 TPP 3.5）。", "aw_ladder": "ladder {q}", "aw_extra": "额外格子", "aw_unstable": "router 崩溃（负载均衡 loss 9–16），不是 LR 点", "aw_no1x": "（无 1×：相对最小倍数）", "ac_h": "其余检查：batch、LR floor、按参数组分 LR", "ac_head": ["检查", "实验", "对照", "读法"], "ac_batch": "batch（q1，L 18，9.6B）", "ac_batch_r": "Runlong 预先写下的预测：公式按 token 计 1.45 / 按步数计 1.81 / B^0.56 2.14 / sqrt 噪声 2.56", "ac_floor0": "LR floor（q0，L 12）", "ac_floor_r": "MuonH 无 warmup、从头线性衰减到峰值的 X%（不是 WSD）；10% = ArchScale 自己的设置，ladder 用 1%", "ac_floor3": "LR floor（q3，L 30）", "ac_floor3_r": "相对同 floor 的 1×；负 = 比 1× 好", "ac_split": "按参数组分 LR（q3，L 30）", "ac_split_r": "相对 1×：A = 只有 AdamW 组（embedding、1-D）1.74×；B = hidden 矩阵 + LM head + router 1.74×；C = 只有 hidden 矩阵 1.74×",
  "rt_set_head": ["组", "Change ID", "主体优化器", "router 打分", "负载均衡", "gate（router 权重）优化器", "router LR", "LR 规则 / schedule", "对照", "已跑"], "rt_set_rows": [["对照 ①", "MUONH_FIT", "MuonH", "softmax", "aux loss 1e-3", "AdamH（球面）", "×1", "新拟合公式；无 warmup，线性衰减到 1%，wd 0", "—", "q0 ×3 seed、q1–q3"], ["<b>A</b>", "MUONH_FIT_DSV3", "MuonH", "<b>sigmoid</b>", "<b>expert bias</b> + seq aux 1e-4", "AdamH", "×1", "同 ①", "①", "q0 ×3、q1–q3"], ["<b>B2</b>", "…_DSV3_RTADAMW_RT0P19", "MuonH", "sigmoid", "同 A", "<b>AdamW</b>（wd 0）", "<b>×0.19</b>", "同 ①", "A", "q0 ×3、q1"], ["<b>B3</b>", "…_DSV3_RT0P19", "MuonH", "sigmoid", "同 A", "AdamH", "<b>×0.19</b>", "同 ①", "A", "q0 ×3"], ["对照 ②", "BASE", "AdamW", "softmax", "aux loss 1e-3", "AdamW", "×1", "adamw_active；WSD（5% warmup，最后 20% 衰减），wd 0.1", "—", "q0 ×3、q1–q5"], ["<b>C</b>", "BASE_DSV3_ONETREE", "AdamW", "<b>sigmoid</b>", "<b>expert bias</b> + seq aux 1e-4", "AdamW", "×1", "同 ②", "②", "q0 ×3、q1、q2"], ["S", "MUONH_FIT_DSV3_LR0P7X / 1P4X", "同 A，整体 LR ×0.7 / ×1.4", "", "", "", "", "", "A", "q0"], ["S_C", "BASE_DSV3_ONETREE_LR0P5X / 2X", "同 C，整体 LR ×0.5 / ×2", "", "", "", "", "", "C", "q0"]], "rt_set_l": "MuonH 主体（对照 ①：MUONH_FIT，softmax router）", "rt_set_r": "AdamW 主体（对照 ②：BASE，softmax router）", "rt_set_note": "所有组：128 个 expert 选 8 个、无 shared expert、batch 2M token；每组只改加粗的项：A、C 相对本表的对照，B2、B3 在 A 的基础上再改 gate。router LR ×1 = 与主体同一 LR（新公式在 q0 为 0.0164），×0.19 ≈ 3.1e-3。C 的 ONETREE = 和 A 同一份代码树，所以 C − BASE 是跨代码树的比较。", "rt_chart_h": "结果：实验组 − 对照组的 loss 差（overall_tsqw，越负越好）。q0 有 3 个 seed：空心点 = 每个 seed 各自的配对差，实心点和折线 = 3 个 seed 的平均；其余 rung 是 1 个 seed", "rt_ylab": "Δloss（实验组 − 对照组）", "rt_lr_note": "LR 检查（q0）：MuonH + DSV3 的最优倍数是新公式的 {s} 倍，AdamW + DSV3 是 adamw_active 的 {c} 倍，两者的现用 LR 都接近最优。",
  "rt_hm_h": "每层负载均衡（训练日志，seed 1）", "rt_hm_note": "来自 Boxiang 的 router 监测 <code>router_health.py</code>（<code>LADDER_ROUTER_HEALTH=100</code>，每 100 步一次）。每个小图：纵轴 = 层（上面是第 0 层），横轴 = 训练步数（最多 60 列，相邻的报告取平均），颜色越深数值越大；上方是训练结束时各层的平均值；悬停看具体数值。各指标：<b>MaxVio</b> = 一整个训练 step（所有 micro-batch、所有 rank）里每层每个 expert 实际分到的 token 数，最忙的 expert ÷ 平均值 − 1（DeepSeek-V3 的指标；0 = 完全均衡，2.86 = 最忙的分到平均值的 3.86 倍；色标上限 4）；<b>dead</b> = 分到的 token 不到平均值 10% 的 expert，占 128 个的百分比；<b>|bias| 最大</b> = DSV3 每个 expert 的 bias 的最大绝对值（Boxiang 预设健康上限 1）；<b>bias 极差</b> = 同一层 bias 的最大 − 最小——top-8 选择只看 bias 之间的差，所有 expert 一起抬高（|bias| 大但极差小）并不改变路由；<b>logit 饱和</b> = router logit |z| &gt; 4 的比例，<b>最大 |logit|</b> = 最大的 |z|，两者都只看 rank 0 的第一个 micro-batch，且 |z| &gt; 4 包括 z &lt; −4（明确拒绝某个 expert，正常）；<b>gate 更新幅度</b> = router 权重这一步的变化量 ÷ 权重大小（Boxiang 只作参考，不作判据）。softmax router 没有 bias；BASE（softmax AdamW）和 MUONH_FIT 的 q2 / q3 没有开监测。<b>注意：</b>训练日志把每一行截断在约 1000 个字符，逐层那一行写不完，DSV3 各组靠后的层缺数据（灰色；q0 缺 1 层、q1 缺 4 层、q3 缺 16 层），这在源头就丢了，补不回来。", "rt_hm_vio": "MaxVio", "rt_hm_dead": "dead (%)", "rt_hm_bmax": "|bias| 最大", "rt_hm_bspread": "bias 极差", "rt_hm_sat": "logit 饱和 (%)", "rt_hm_zmax": "最大 |logit|", "rt_hm_upd": "gate 更新幅度", "rt_hm_nobias": "softmax 没有 bias", "rt_hm_miss": "{n} 层缺数据", "rt_hm_na": "无监测数据", "rt_hm_end": "结束时层均值", "rt_hm_layers": "层 ↓ · 步数 →", "rt_hm_soft": "softmax（MUONH_FIT）", "rt_hm_a": "DSV3", "rt_hm_b2": "DSV3 + AdamW gate + router LR ×0.19", "rt_hm_b3": "DSV3 + router LR ×0.19", "rt_hm_c": "C：AdamW + DSV3",
  "rt_st_h": "router 健康统计", "rt_st_note": "来自 Boxiang 的 router 监测 <code>router_health.py</code>（每 100 步一次）的汇总行，覆盖所有层。每个 rung 一张图，横轴 = 训练步数，每组一条线；上方按钮切换指标，鼠标悬停看各组数值。MUONH_FIT 的 q2 / q3 没有开监测，所以没有这条线。", "rt_st_name": {"vio_mean": "MaxVio（层平均）", "vio_max": "MaxVio（最差层）", "dead": "dead expert (%)", "bmax": "|bias| 最大", "sat": "logit 饱和 (%)", "satpos": "z > +4 (%)", "pinned": "top-8 pinned (%)", "zmax": "最大 |logit|", "upd": "gate 更新幅度"}, "rt_st_def": {"vio_mean": "每层：一整个训练 step（所有 micro-batch、所有 rank）里分到 token 最多的 expert ÷ 平均值 − 1（DeepSeek-V3 的 MaxVio，0 = 完全均衡）；这里是所有层的平均。", "vio_max": "同上，取最不均衡的那一层。", "dead": "分到的 token 不到平均值 10% 的 expert 数，占（层数 × 128）的百分比。", "bmax": "DSV3 每个 expert 的 bias 在所有层里的最大绝对值；红色虚线 = Boxiang 预设的健康上限 1。注意 top-8 选择只看同一层 bias 之间的差，所有 expert 一起抬高并不改变路由（例如 A 的 q3 第 0 层 bias 在 +11.2 到 +11.7 之间）。softmax 没有 bias。", "sat": "router logit |z| &gt; 4 的比例（各层平均；只看 rank 0 的第一个 micro-batch）。包括 z &lt; −4（明确拒绝某个 expert，正常）。", "satpos": "z &gt; +4 的比例：分数被钉在 1，选中的 expert 之间失去区分度（坏）；红色虚线 = 预设上限 5%。只有后期日志才有。", "pinned": "选中的 8 个 expert 里分数卡在 1 附近的比例；红色虚线 = 预设上限 5%。只有后期日志才有。", "zmax": "所有层里最大的 |logit|（rank 0 的第一个 micro-batch）。", "upd": "router 权重这一步的变化量 ÷ 权重大小，各层平均（Boxiang 只作参考，不作判据）。"},
  "rt_h": "原来的内容（未删，之后逐步清理）：softmax 还是 DeepSeek-V3 式（MuonH / AdamW，新 LR 公式，128 选 8，q0 三个 seed）", "rt_note": "Boxiang 10-26 的对照实验（设计：reports/dsv3_router_experiment_design_2026-10-26.md）。把 softmax + aux loss 的 router 换成 DeepSeek-V3 式 router（sigmoid 打分 + 每个 expert 一个 bias 管负载均衡 + seq aux 1e-4），expert 数（128 选 8）等其余不变；MuonH 各组都用新拟合的 LR 公式（MUONH_FIT），AdamW 组用 BASE 自己的 LR 规则。数值 = <b>实验组 − 对照组</b>（doc-isolated，负 = 实验组更好），q0 三个 seed 逐 seed 配对后取均值 ± 标准误（括号里是配对的 seed 数）；预先定的判定线：|Δ| &lt; 0.003 视为没有差别，单 seed ±0.008。", "rt_cmp": "比较", "rt_health": "router 健康（训练日志）", "rt_h_ok": "正常", "rt_h_bad": "病态", "rt_a": "A：DSV3 router − softmax（MuonH，gate AdamH）", "rt_b2": "B2：gate 改 AdamW、router LR ×0.19 − A", "rt_b3": "B3：gate AdamH、router LR ×0.19 − A", "rt_c": "C：DSV3 router − softmax（AdamW 基线）", "rt_lead": "MuonH 领先 AdamW 的变化：(A − C) − (MUONH_FIT − BASE)", "rt_health_note": "健康状态取自 Boxiang 的 router 监测（DECISIONS 10-26 19:1x–19:2x）：DSV3 在 q1–q3 上 expert bias 涨到 3–5（预设上限 1）、logit 失控到 60–87，负载均衡全靠 bias 撑——按预设规则不能直接采用；DSV3 + AdamW gate + router LR ×0.19 约 10% 的选中 expert 分数卡在 1 附近、bias 超过 1、logit 持续增大；DSV3 + router LR ×0.19 健康。两种病态原因不同：DSV3 的 gate 在 router LR ×1 下自己学不会均衡，bias 越涨越大，router LR 降到 ×0.19 就健康了；两个 ×0.19 的组 router LR 相同，只有用 AdamW gate 的那组把选中的 expert 分数卡在 1，所以“卡死”来自 AdamW gate 本身（DECISIONS 10-26 17:29）。", "rt_hr_h": "更大的 rung（单 seed）", "rt_hr_pending": "q1–q3 还在训练 / 评测。", "rt_lr_h": "LR 检查（q0，overall_tsqw，相对该组 1× 三个 seed 的均值）", "rt_lr_head": ["组", "各倍数 Δloss", "最优倍数（相对该组公式）", "1× 峰值 LR", "该组的 LR 公式"], "rt_s": "S：MuonH + DSV3 router", "rt_rl": "router LR：MuonH + DSV3 router", "rt_rl_lab": "只改 gate 的 LR：×0.19 / ×0.5 / ×1 / ×2（横轴 = 主体 LR 的倍数，×1 = A）", "rt_rl_r": "最优 router LR ≈ 主体 LR 的 {m:.2f} 倍；现用 ×1 时 loss 比最优高 {g:.4f}。×1 和 ×0.19 各 3 个 seed，其余 1 个 seed（Runlong 10-29 加的）。", "rt_x_h": "DSV3 底座的其他消融（Runlong 10-29 加的；q0 单个 seed，相对 A = MUONH_FIT_DSV3；负数 = 更好）", "rt_x_head": ["改动", "Change ID", "相对 A seed 1（同数据顺序）", "相对 A 三个 seed 平均"], "rt_x_bias": "去掉 seq aux，只靠 bias 均衡（loss-free）", "rt_x_muonh": "gate 改用 MuonH（球面上），LR 同主体", "rt_s_lab": "A 的 LR ×0.7 / ×1 / ×1.4（横轴 = ArchScale 原版的倍数）", "rt_sc_lab": "C 的 LR ×0.5 / ×1 / ×2（横轴 = adamw_active 规则的倍数）", "rt_vs": "现用 LR（3 seed 平均）", "rt_lr_h2": "LR 检查：换成 DSV3 router 后现用 LR 还是不是最优（q0，overall_tsqw；只扫整体 LR，不是新实验组）。格子读法同 LR 网格：大数字 = 抛物线顶点 m*，鼠标移到格子上看曲线；纵轴 = 相对现用 LR（3 个 seed 平均）的 loss 差", "rt_lr_head2": ["LR 检查（q0）", "LR 扫描", "读法", "结论"], "rt_s_rc": "不用改 LR：换成 DSV3 后，现用新公式仍接近最优（只差 {pct:.0f}%，loss 差 {g:.4f}，在噪声内）。只测了 q0，大 rung 没验证。", "rt_rl_rc": "router LR 保持 ×1：最优约 ×{m:.2f}，×1 只差 {g:.4f}，在噪声内；降到 ×0.19 反而变差。只测了 q0。", "rt_s_r": "最优 {m:.2f}×（ArchScale 原版），现用新公式 = {c:.2f}×：最优点是现用 LR 的 {r:.2f} 倍，在最优点 loss 只比现用低 {g:.4f}，现用 LR 接近最优。", "rt_sc_r": "最优 {m:.2f}×（adamw_active 规则），现用 = {c:.2f}×：最优点是现用 LR 的 {r:.2f} 倍，在最优点 loss 只比现用低 {g:.4f}，现用 LR 接近最优。", "rt_sc": "S_C：AdamW + DSV3 router", "rt_s_law": "MUONH_FIT 新公式", "rt_sc_law": "AdamW 的 adamw_active 规则",
  "dg_h": "Runlong 逐 rung 的 LR 是怎么来的（只用对角线：ladder q0–q6 的 9.6B 格子）", "dg_how": "Runlong 的生产 LR 表（Runlong.md TODO 8 A.2，step 7；目前只在 TODO 9 的 q4 一对实验上用了 5.36e-3，其余等 GO）。每一行：<b>① 该 rung 在 9.6B（4,577 步）格子实测的最优倍数 m*</b>（上表对角线，Boxiang 在 val_v2 上测）× <b>② 训练量修正 (主线步数 / 4,577)^0.03</b>（c = 0.03，范围 0–0.06，取自 LR 格子 token 方向的拟合，等效 T 指数 −0.32 + 0.03 = −0.29）= <b>③ 相对 ArchScale 原版的倍数</b>；再乘 <b>④ ArchScale 原版公式在主线训练量（TPP 200）下的 LR</b>，得到 <b>⑤ 峰值 LR</b>。⑤ 与他公布的数字相差都在 0.5% 以内。最后一列是我们自己（overall_tsqw）测的 9.6B m*，可与 ① 对照。", "dg_dhead": ["rung", "层数 L", "主线训练量 T", "主线步数", "① 9.6B 锚点 m*", "② (步数/4577)<sup>0.03</sup>", "③ 倍数 = ①×②", "④ ArchScale LR（主线 T）", "⑤ 峰值 LR = ③×④", "Runlong 公布", "我们的 9.6B m*"], "dg_note": "只用上表对角线上的 7 个 ladder 格子（宽度随层数一起变，h ≈ 43·L），训练量全是 9.6B，所以只能拟合<b>沿 ladder 放大</b>这一个方向，这里的指数不是纯深度指数，也定不出 T 指数。蓝点 = 我们实测的最优 LR（m* × ArchScale 原版 LR）；青色 = Runlong 逐 rung 生产 LR 表的锚点（Boxiang 在 val_v2 上测的 9.6B m*）。", "dg_ours": "我们拟合的（对角线，overall_tsqw）", "dg_ours_note": "（相对 ArchScale 的倍数 ∝ (L/8)^{b:+.3f}）", "dg_runlong": "Runlong 的逐 rung LR", "dg_short": "m*<sub>9.6B</sub>(rung) = 上表对角线上该 rung 实测的最优倍数（没有拟合）；0.03 是训练量的经验值。q4 已按此用 5.36e-3。", "dg_runlong_note": "；锚点 m*（q0–q6）= {anc}；c ≈ 0.03 [0, 0.06]；没有拟合，锚点直接取各 rung 实测值（未正式采用，q4 已用 5.36e-3）", "dg_pts": "格子实测最优 LR（空心 = 外推）", "dg_x": "层数 L（ladder rung；宽度随 L 一起变）", "dg_tip_o": "{q} 我们实测：m* {m:.2f}× → η* {v:.5f}（{mn}）", "dg_tip_r": "{q} Runlong 锚点：m* {m:.2f}× → η {v:.5f}（9.6B）",
  "ck_head": ["检查", "对照", "实验", "结论"], "ck_note": "每格和上面的 LR 格子一样：一个 LR 扫描，大数字 = 抛物线顶点 m*（相对 ArchScale 原版公式，按实际峰值 LR 计算），鼠标移到格子上看曲线。", "ck_floor1": "衰减到峰值的 1%（ladder）", "ck_floor10": "衰减到峰值的 10%（ArchScale）", "ck_split_ctl": "全部参数一起改 LR", "ck_split_exp": "只给一组参数 1.74×（菱形）", "ck_split_a": "A：只给 AdamW 组（embedding、1-D）", "ck_split_b": "B：hidden 矩阵 + 输出头 + router", "ck_split_c": "C：只给 hidden 矩阵", "ck_batch_r": "batch 翻倍、步数减半时最优倍数从 1.48 升到 1.86，最接近 Runlong 预先写下的“LR 按步数算”（1.81）→ 换 batch 时按步数调 LR。", "ck_floor_r": "衰减到 10% 时 ArchScale 原版几乎正好（m* ≈ 1）；ladder 衰减到 1%，需要高约 20% 的峰值 LR。", "ck_floor3_r": "q3 上同样：10% floor 时最优点往低移。", "ck_split_r": "只给 hidden 矩阵加 LR（C）就拿到几乎全部收益；只给 embedding / 1-D 参数（A）没有用。",
  "ne_ch_h": "Router 负熵均衡 (phi balancing)", "ne_h": "全注意力底座：各组 − MUONH_FIT（MuonH，softmax router，CE）", "ne_zero": "0 线 = <b>MUONH_FIT</b>：MuonH，新 LR 公式（fit），纯 CE（不用 KD），128 选 8 MoE，全注意力，原版 softmax router + Switch aux loss 1e-3，gate 用 AdamH、LR ×1。", "ne_chart_l": "DSV3 底座 + 负熵 − MUONH_FIT（鼠标移到某个 rung 看数值）", "ne_chart_r": "softmax 底座 + 负熵 − MUONH_FIT（鼠标移到某个 rung 看数值）", "ne_y": "Δloss（该组 − MUONH_FIT）", "ne_l_dsv3": "DSV3（参照）", "ne_l_e6": "DSV3 + 负熵 α 1e-6", "ne_l_e5": "DSV3 + 负熵 α 1e-5", "ne_l_e4": "DSV3 + 负熵 α 1e-4", "ne_l_e3": "DSV3 + 负熵 α 1e-3", "ne_l_e5nb": "DSV3 + 负熵 α 1e-5，去掉 bias", "ne_l_e4nb": "DSV3 + 负熵 α 1e-4，去掉 bias", "ne_l_e5sm": "softmax + 负熵 α 1e-5", "ne_l_e4sm": "softmax + 负熵 α 1e-4", "nesws_h": "SWS 底座：各组 − DSV3 SWS", "nesws_note": "0 线 = <b>DSV3 SWS</b>（MUONH_FIT_DSV3_SWS）：MuonH，新 LR 公式（fit），纯 CE（不用 KD），128 选 8 MoE，DSV3 router（sigmoid 打分 + expert bias + seq aux 1e-4，gate 用 AdamH、LR ×1），SWA（每 6 层 5 层窗口 2048）+ attention sink。", "nesws_skip": "α 1e-2 没画：q0 +0.036，q1 训崩（+0.256）。", "ns_r1": "保留 bias：", "ns_r2": "变体：", "ns_e6": "α 1e-6", "ns_e5": "α 1e-5", "ns_e4": "α 1e-4", "ns_4e4": "α 4e-4", "ns_e3": "α 1e-3", "ns_e5nb": "1e-5 去掉 bias", "ns_e4nb": "1e-4 去掉 bias", "ns_e5nbf": "1e-5 去掉 bias + 按选中频率统计", "ns_e5batch": "1e-5 去掉 EMA", "ns_e5adamw": "1e-5 + AdamW gate ×0.19", "ns_e5g4": "1e-5 + bias γ 1e-4", "nesws_chart": "SWS 底座 + 负熵 − DSV3 SWS（鼠标移到某个 rung 看数值）", "nesws_x": "Δloss（该组 − DSV3 SWS）", "ne_note": "负熵均衡损失：每层对 router 概率做 EMA（η 0.65），得到每个 expert 长期分到的平均概率 m<sub>e</sub>（在整个 global batch 上统计，每个 step 更新一次）；损失 = α·E·Σ<sub>e</sub> p<sub>e</sub>·(log m<sub>e</sub> + 1)，p<sub>e</sub> = 当前这批 token 给 expert e 的平均概率，括号里那项不传梯度，梯度只走 p<sub>e</sub>。效果：长期被用得多的 expert（m<sub>e</sub> 大）概率被往下压，用得少的被往上拉，把平均分配推向均匀（熵最大）。它替换原来的负载均衡损失（DSV3 底座的 seq aux 1e-4、softmax 底座的 Switch aux 1e-3）；DSV3 底座的 expert bias 默认保留（标了“去掉 bias”的组除外）。和 aux loss 的区别：统计范围是整个 global batch 再做滑动平均，用的是概率而不是实际分到的 token 数，所以和 micro-batch 大小无关。（源自 arXiv 2605.15403）", "ne_base": "底座", "ne_pair": "对照", "ne_dsv3": "DSV3 底座（保留 bias）", "ne_nobias": "DSV3 底座，去掉 bias", "ne_soft": "softmax 底座（论文原样）", "ne_sws": "，SWS 底座（DSV3 + SWA + sink）", "ne_noema": "无 EMA（η 1）",
  "r0_h": "Router 选择", "r0_note": "0 线 = <b>MUONH_FIT</b>：MuonH，新 LR 公式（fit，q0 峰值 0.0164），<b>纯 CE（不用 KD）</b>，128 选 8 MoE，<b>全注意力</b>（不含 SWA / sink），原版 softmax router + Switch aux loss 1e-3，gate 用 AdamH、LR ×1。", "r0_chart": "各行 − MUONH_FIT（越靠左越好）。实心点 = 平均；有多个 seed 时空心点 = 每个 seed 各自的配对差；鼠标移到点上看数值和 Change ID", "r0_x": "Δloss（该行 − MUONH_FIT）", "r0_g_score": "① 换打分方式：softmax → DSV3 式", "r0_dsv3": "DSV3 式：sigmoid 打分 + expert bias + seq aux 1e-4", "r0_g_bal": "② DSV3 式 + 改负载均衡", "r0_biasonly": "只靠 bias（去掉 seq aux）", "r0_g_opt": "③ DSV3 式 + 改 gate 的优化器 / LR", "r0_rt019": "router LR ×0.19", "r0_rt05": "router LR ×0.5", "r0_rt2": "router LR ×2", "r0_rtmuonh": "gate 改用 MuonH", "r0_rtadamw": "gate 改用 AdamW，LR ×0.19", "r0_g_ne": "④ DSV3 式，seq aux 换成负熵均衡损失 α（Runlong TODO 9）", "r0_ne5nb": "α 1e-5，再去掉 bias", "r0_ne4nb": "α 1e-4，再去掉 bias", "r0_g_nesoft": "⑤ 保留 softmax，aux loss 换成负熵均衡损失 α", "sa_chart": "各组 − DSV3 SWS（鼠标移到某个 rung 看所有组的数值）", "sa_biasonly_d": "去掉 seq aux 1e-4，只靠 expert bias 均衡（DeepSeek-V3 的“无 aux loss”做法）。", "sa_nobias_d": "关掉 expert bias，只靠 seq aux 1e-4 均衡（和上一行互为镜像）。", "sa_nobal_d": "bias 和 aux 都去掉，完全不做负载均衡；作为下限参照，预计会出现 dead expert。", "sa_auxbatch_d": "按序列的 seq aux 换成按 micro-batch（32K token）统计的 Switch aux，系数仍是 1e-4；bias 保留。测的是统计范围（文献：micro-batch 最差，序列其次，global batch 最好）。", "sa_aux1em3_d": "seq aux 系数加大 10 倍（1e-4 → 1e-3），bias 保留。", "sa_g1em4_d": "expert bias 每步的调整量 γ 从 1e-3 降到 1e-4（调得慢 10 倍）。", "sa_g1em2_d": "γ 从 1e-3 升到 1e-2（调得快 10 倍；DeepSeek-V3 也扫过 γ）。", "sa_softmax_d": "打分从 sigmoid 换成 softmax（选出 top-8 后在这 8 个里重新归一化），bias 和 seq aux 保留（LongCat 式 softmax + bias）。", "sa_rs_d": "被选中 8 个 expert 的权重（加起来 = 1）整体乘 2.5 再相加，即 routed expert 的输出放大 2.5 倍（DeepSeek-V3 的取值）；不加 shared expert。", "sa_shared_d": "每层加 1 个 shared expert：所有 token 都过它，宽度和一个 routed expert 相同（3h/8，q0 是 192）；权重缩放保持 1。<b>不是等参数</b>，激活参数约多 7–8%。", "sa_shared_rs_d": "上面两项一起加：1 个 shared expert + routed scaling 2.5，也就是完整的 DeepSeek-V3 MoE 设置。<b>不是等参数</b>，激活参数约多 7–8%。", "sa_zloss_d": "对 router logit 加 z-loss 1e-3（ST-MoE / OLMoE 的做法）：惩罚 logit 的 log-sum-exp 的平方，防止 logit 变得太大；其余不变。", "sa_rt019_d": "gate（router 权重）仍用 AdamH，LR 降到主体 LR 的 0.19 倍。", "sa_rt05_d": "gate 用 AdamH，LR = 主体 LR × 0.5。", "sa_rt2_d": "gate 用 AdamH，LR = 主体 LR × 2。", "sa_rtmuonh_d": "gate 改用 MuonH 更新（在球面上），LR 同主体；GLM-4.5 是唯一公开在 router 上用 Muon 的模型。", "sa_rtadamw_d": "gate 改用普通 AdamW（wd 0），LR = 主体 LR × 0.19（AdamW 规则下 router 的 LR）；就是全注意力上 B2 那组的设置搬到 SWS 上。", "sa_thead": ["类别", "线", "名称", "具体改动（相对 DSV3 SWS）", "有数据的 rung"], "sa_table_h": "各组说明（图里每条线是什么）", "sa_none": "还在训练 / 评测", "sa_r_bal": "负载均衡：", "sa_r_gamma": "bias γ / 打分：", "sa_r_scale": "缩放 / shared / 正则：", "sa_r_opt": "gate 优化器 / LR：", "sa_biasonly": "去掉 seq aux（只靠 bias）", "sa_nobias": "去掉 bias（只靠 seq aux）", "sa_nobal": "都去掉", "sa_auxbatch": "aux 按 micro-batch", "sa_aux1em3": "seq aux ×10", "sa_g1em4": "γ 1e-4", "sa_g1em2": "γ 1e-2", "sa_softmax": "softmax 打分", "sa_rs": "routed scaling 2.5", "sa_shared": "+1 shared expert", "sa_shared_rs": "shared + scaling 2.5", "sa_zloss": "z-loss 1e-3", "sa_rt019": "router LR ×0.19", "sa_rt05": "router LR ×0.5", "sa_rt2": "router LR ×2", "sa_rtmuonh": "MuonH gate", "sa_rtadamw": "AdamW gate ×0.19", "lat_h": "Latent MoE", "lat_note": "前两条线的 0 线 = <b>PROD1_DPSK</b>（PROD v1）：SWA 512 局部:全局 5:1 + sink，DSV3 router，第 0 层 dense，8 + 1 shared expert，32K 训练 + 文档间 mask，MuonH，纯 CE<br>latent MoE（Kimi K3 的形式）：MoE 先把 h 维投影到 d = h/2，routed expert 在 d 维上算，RMSNorm 后再投影回 h；router 和 shared expert 仍在 h 维。<br>数值 = 该线 − 它的对照（对照见下表 Change ID 列：前两条和第四条是 PROD1_DPSK，第三条是 PROD1_DPSK_LATENT2；我们的 115 split doc-isolated 评测，seed 1）。这些 run 都是 32K 训练：每个 checkpoint 只 load 一次，同一个 job 出 8K 窗口和 32K 窗口两份分数（图上方 tab 切换）。", "lat_chart": "各线 − 各自的对照（越负越好；数值见图下的表）", "x_pending": "评测中：每个 checkpoint 一个 job（load 一次）同时出 8K / 32K 两个窗口，结果出来后自动画上。", "lat_l_l2": "latent MoE（等参数）", "lat_l_l2_d": "PROD1_DPSK_LATENT2：d = h/2，256 个 routed expert（宽 3h/8）选 16 个；总参数、routed FLOP、all-to-all 都和 0 线相同（只多两个投影，约每层 8%）", "lat_l_e2x": "latent MoE，expert ×2", "lat_l_e2x_d": "PROD1_DPSK_LATENT2_E2X：在上一条基础上把 routed expert 翻倍到 512 个，仍选 16 个。每个 token 的计算量不变，但 routed 参数 ×2（<b>不是等参数</b>）", "lat_l_x2l": "expert ×2（有 latent）", "lat_l_x2l_d": "PROD1_DPSK_LATENT2_E2X − <b>PROD1_DPSK_LATENT2</b>（这条线的 0 线不同）：两边都有 latent，只把 routed expert 从 256 个加到 512 个，仍选 16 个，看 expert ×2 单独的作用", "lat_l_x2": "expert ×2（无 latent）", "lat_l_x2_d": "PROD1_DPSK_E2X − PROD1_DPSK：不加 latent，routed expert 从 128 个加到 256 个，仍选 8 个；每个 token 计算量不变，routed 参数 ×2。和上一条对比：expert ×2 在有 / 没有 latent 时的作用", "rtq_h": "E2X 上的 router / 负载均衡（Runlong 的实验）", "rtq_note": "0 线 = <b>PROD1_DPSK_LATENT2_E2X</b>（latent d = h/2，512 个 routed expert 选 16 + 1 shared；sigmoid router + expert bias（更新率 1e-3）+ seq aux 1e-4；router 用 AdamH）。每条线只改 router 的一处，其余（步数、LR、数据、seed 1）都和 0 线相同。<br>数值 = 该线 − 0 线（我们的 115 split doc-isolated 评测；每个 checkpoint 加载一次，出 8K / 32K 两个窗口）。这些 run 11-05 / 11-06 才开始训练，没有 q0；每个 checkpoint 训完后自动评测、自动画上。", "rtq_chart": "各线 − PROD1_DPSK_LATENT2_E2X（越负越好；鼠标移到某个 rung 看数值）", "rtq_l_base": "0 线：PROD1_DPSK_LATENT2_E2X", "rtq_l_base_d": "bias + seq aux 1e-4", "rtq_l_qb": "QB：Quantile Balancing（分位数均衡）", "rtq_l_qb_d": "Megatron 自带（<code>--moe-router-load-balancing-type quantile_balancing</code>），代替 bias + seq aux：每个 expert 有一个 bias（qb_beta），用对偶坐标下降（dual coordinate descent）更新。每个 micro-batch 里按该 expert 的 routing margin 取分位数，使它正好收 T·k/E 个 token；各 micro-batch、各 DP rank 取平均并居中，每个 global batch 更新一次，用来路由下一个 batch（评测时不更新，只用存下的 bias）。aux loss 和 expert bias 都关掉", "rtq_l_qbx": "QBX：全局精确 quantile（Kimi K3 的做法）", "rtq_l_qbx_d": "同 QB，只是分位数在整个 global batch 上精确算（各 micro-batch、各 DP rank 的 margin 直方图相加后取第 C+1 大），不是取平均；用打了补丁的 Megatron", "rtq_l_bo": "只用 expert bias（去掉 seq aux）", "rtq_l_bo_d": "去掉 seq aux 1e-4，只靠 expert bias 均衡；是 QB 的干净对照（两者都没有 aux loss）", "rtq_l_g5": "bias 更新率 ×0.5", "rtq_l_g5_d": "expert bias 更新率 1e-3 → 5e-4（看 512 个 expert 下 bias 步长是否太大、来回过冲）", "rtq_l_g25": "bias 更新率 ×0.25", "rtq_l_g25_d": "expert bias 更新率 1e-3 → 2.5e-4", "rtq_l_adamw": "router 用 AdamW（LR ×0.19）", "rtq_l_adamw_d": "router 从 AdamH（Adam + 每个 router 矩阵一个 hyperball 球面）改成普通 AdamW，router LR ×0.19；看去掉整矩阵的 norm 约束后冷 expert 是否变少", "rtq_l_pin": "router 每行单独 hyperball（pin）", "rtq_l_pin_d": "router 权重的每一行（每个 expert）单独做 hyperball：每行投回自己的初始 norm，行与行之间不会互相拿走 norm；针对冷 expert（router 行变短、方向趋同）", "rtq_tl_h": "训练 loss（训练日志，早期参考）", "rtq_tl_note": "该线 − 0 线 E2X 的训练 lm loss：取该线最新记录的 step S，在 (S − 500, S] 内对两边各自求平均再相减（32K 训练 loss，不是评测；单 seed，噪声约 ±0.002；前 1–2 千步 loss 下降很快，差值会大得多，不能当结论）。训练中的 run 在报告每次重新生成时更新；结论以上面的评测为准。", "rtq_rt_h": "训练中的负载均衡（router health，训练日志）", "rtq_rt_note": "来自训练日志的 router health（每 100 步一行，所有层汇总），训练中的 run 会随报告更新。线的颜色和上面评测图的颜色一一对应，灰色虚线 = 0 线 E2X；鼠标移到图上看名字和数值。dead expert 的百分比按（层数 × 512）算。MaxVio 0 = 完全均衡。", "x_thead": ["线", "名称", "改动", "有数据的 rung"], "x_tab_h": "各条线（都减 0 线 PROD1_DPSK）", "rope_h": "位置编码：NoPE / RoPE / partial RoPE", "rope_note": "0 线 = <b>PROD1_DPSK</b>（PROD v1）：SWA 层用完整 RoPE（128 维全部旋转），全局层（每 6 层 1 层）NoPE；其余同上一章的 0 线<br>partial RoPE：每个 head 的 128 维里只有 32 维（1/4）加 RoPE（NeoX 排法，base 1e4），其余 96 维不带位置。<br>数值 = 该线 − PROD1_DPSK（我们的 115 split doc-isolated 评测，seed 1），8K / 32K 两个窗口同上一章。", "rope_chart": "各线 − PROD1_DPSK（越负越好；鼠标移到某个 rung 看数值）", "rope_l_p25": "SWA 层 1/4 RoPE", "rope_l_p25_d": "PROD1_DPSK_PROPE25：只改 SWA 层，RoPE 1 → 1/4；全局层仍是 NoPE（单变量）", "rope_l_p25g": "所有层 1/4 RoPE", "rope_l_p25g_d": "PROD1_DPSK_PROPE25G：SWA 层 RoPE 1 → 1/4，全局层 NoPE → 1/4 RoPE（改了两处；和上一条之差 = 全局层 NoPE vs 1/4 RoPE）", "rope_sm_h": "另一组：softmax 底座上全局层 RoPE vs NoPE（8K 训练，只有 8K 窗口；0 线不同，不画进图）", "rope_sm_row": "SWA 窗口 2048 局部:全局 5:1（无 sink），全局层 NoPE → 完整 RoPE", "sws_h": "Router 选择，SWS 架构，更杂的实验", "sws_note": "SWS = MUONH_FIT_DSV3_SWS：DSV3 式 router（sigmoid + expert bias γ 1e-3 + seq aux 1e-4，AdamH gate ×1）放在 SWA（每 6 层 5 层窗口 2048，全注意力层 NoPE）+ 可学习 attention sink 上，是 Runlong 10-29 晚起 router 消融的新底座。每组只改一项 router 设置，数值 = <b>该组 − SWS</b>（doc-isolated overall_tsqw，按 seed 配对；负数 = 更好）。灰带 = 预先定的“没有差别”区间 |Δ| &lt; 0.003；单个 seed 的噪声约 ±0.008。", "sws_chart": "各组 − SWS 底座（overall_tsqw，越靠左越好）。实心点 = 平均；有多个 seed 时空心点 = 每个 seed 各自的配对差；鼠标移到点上看数值和 Change ID", "sws_x": "Δloss（该组 − SWS）", "sws_pending": "还在训练 / 评测：", "sws_g_bal": "负载均衡方式", "sws_biasonly": "只靠 bias（去掉 seq aux）", "sws_nobias": "只靠 seq aux（去掉 bias）", "sws_nobal": "都不要（无均衡，下限参照）", "sws_auxbatch": "aux 改按 micro-batch 统计", "sws_aux1em3": "seq aux ×10（1e-3）", "sws_g_gamma": "bias 更新速度 γ（现用 1e-3）", "sws_g1em4": "γ 1e-4（×0.1）", "sws_g1em2": "γ 1e-2（×10）", "sws_g_score": "打分函数", "sws_softmax": "softmax 打分 + bias + seq aux", "sws_g_scale": "gate 权重缩放 / shared expert", "sws_rs": "routed scaling 2.5（DSV3 的值）", "sws_shared": "加 1 个 shared expert（非等参数）", "sws_shared_rs": "shared expert + scaling 2.5（完整 DSV3）", "sws_g_reg": "router 正则", "sws_zloss": "z-loss 1e-3", "sws_g_opt": "gate 的优化器 / LR", "sws_rt019": "router LR ×0.19", "sws_rt05": "router LR ×0.5", "sws_rt2": "router LR ×2", "sws_rtmuonh": "gate 改用 MuonH", "sws_rtadamw": "gate 改用 AdamW，LR ×0.19", "sws_g_ne": "负熵均衡损失（TODO 9）", "sws_ne6": "α 1e-6", "sws_ne5": "α 1e-5", "sws_ne4": "α 1e-4", "sws_ne4e4": "α 4e-4", "sws_ne5nb": "α 1e-5，去掉 bias", "sws_ne4nb": "α 1e-4，去掉 bias", "sws_ne3": "α 1e-3", "sws_ne2": "α 1e-2", "sws_ne5batch": "α 1e-5，去掉 EMA（每个 micro-batch 现算）", "sws_ne5adamw": "α 1e-5，gate AdamW ×0.19", "sws_ne5g4": "α 1e-5，bias 更新速度 γ 1e-4", "sws_ne5nbfreq": "α 1e-5，去掉 bias，统计用选中频率（freq，原为概率）", "kd25_h": "老师 MiMo-2.6 vs. MiMo-2.5", "kd25_note": "0 线 = <b>同一条 CE：OPT_MUONH</b>（MuonH 老 LR，650B 数据；q0 为 3 个 seed 的平均），三个色系都减这一条。每个点 = 该 KD 臂的 seed 平均 − OPT_MUONH；q0 的 3 个 seed 平均成一个点。<br>绿 vs 橙：LR、45B KD corpus v1（tc_small cache）、seed 和数据顺序完全相同，只有 teacher 不同，两者的距离 = 2.5 − 2.6。蓝 vs 绿：同是 MiMo-2.6，但 LR 和 KD 数据都不同（蓝色系用 650B 数据），所以两者的距离不只是 LR 的影响。蓝色系的 S128Y 只跑了 q3。", "kd25_pnote": "这里是 Boxiang 式的数值：训练结束时的 packed 验证（6 个 decon 集合，mean6 = 六个的简单平均），取自训练日志。", "kd25_nold": "MiMo-2.6 老 LR {v}", "kd25_n26": "MiMo-2.6 新 LR {v}", "kd25_n25": "MiMo-2.5 新 LR {v}", "kd25_y": "Δloss（KD − CE）", "kd25_chart": "各 KD 臂 − CE（鼠标移到某个 rung 看所有线的数值）", "kd25_fold": "<b>蓝色系</b>：MiMo-2.6 teacher，α 0.8，MuonH <b>老 LR</b>（ArchScale 原公式），650B 数据— S64 / S64Y / S128Y（由浅到深；S128Y 只有 q3）", "kd25_f26": "<b>绿色系</b>：MiMo-2.6 teacher，α 0.8，MuonH <b>新 LR 公式</b>（fit），tc_small 45B — S64 / S64Y / S128Y（由浅到深）", "kd25_f25": "<b>橙色系</b>：MiMo-2.5 teacher，α 0.8，MuonH <b>新 LR 公式</b>（fit），tc_small 45B — S64 / S64Y / S128Y（由浅到深）", "kd25_f0": "纵轴 = KD − CE：OPT_MUONH（MuonH 老 LR），三个色系都减这同一条；纵轴范围 = 当前数据的最小值到最大值。绿 / 橙色系（45B 语料）的 q3 用的是 TPP 100，和其他点不可比，不画。", "dm_h": "文档 mask：加 mask vs 不加 mask", "dm_note": "0 线 = <b>OPT_MUONH</b>：纯 CE，MuonH 老 LR<br>三条线都是 KD MiMo-2.6 teacher，α 0.8，S64Y，同一份 KD 数据（650B），softmax router，无 SWA。<b>加 mask</b> = 打包的 8K 序列里每个文档只看自己（Megatron --dataloader-inter-document-masking）。灰线和紫线是我们跑的一对，都用 MuonH 新 LR 公式（fit），只差 mask：两者的距离 = 文档 mask 的效果。蓝线是 Boxiang 原来用老 LR（ArchScale 原公式）跑的、不加 mask：蓝线和灰线只差 LR。每个点 = seed 平均 − OPT_MUONH（q0 有 3 个 seed 时平均成一个点）。注意：评测是 doc-isolated（每个文档单独评），和加 mask 训练时的条件一致。", "dm_chart": "各条线 − CE（鼠标移到某个 rung 看各条线的数值）", "dm_nb": "不加 mask，老 LR（Boxiang）", "dm_nb_d": "<b>不加 mask，老 LR</b>（Boxiang，KD_MUONH_A0P8_S64Y）：MuonH 老 LR（ArchScale 原公式），q0–q3 各 1 个 seed", "dm_n0": "不加 mask，新 LR", "dm_n1": "加 mask，新 LR", "dm_n0_d": "<b>不加 mask，新 LR</b>（我们跑的，KD_MUONH_A0P8_S64Y_FIT）：MuonH 新 LR 公式（fit），打包的 8K 序列里可以看到前面的其他文档", "dm_n1_d": "<b>加 mask，新 LR</b>（我们跑的，KD_MUONH_A0P8_S64Y_FIT_DOCMASK）：MuonH 新 LR 公式（fit），每个文档只看自己", "dm_f0": "纵轴 = 该线 − CE：OPT_MUONH（MuonH 老 LR），三条线都减这同一条；纵轴范围 = 当前数据的最小值到最大值", "kd25_pend": "（训练 / 评测中）", "kd25_table_h": "数值表（{m}；负数 = 比 CE 好，2.5 − 2.6 负数 = 2.5 更好）", "kd25_thead": ["rung", "变体", "2.6 − CE", "2.5 − CE", "2.5 − 2.6", "seed 数（2.6 / 2.5 / CE）"], "rt_two_h": "两种 router 的区别", "rt_two": "<p><b>softmax router</b></p><ul><li>对 128 个 expert 的 logit 做 softmax，取分数最高的 8 个。</li><li>负载均衡靠 aux loss（Switch 式，系数 1e-3）：某些 expert 被选得太多时，额外加一项惩罚，让 router 把 token 分散开。<ul><li><b>怎么做的</b>：每个 MoE 层，在每个 micro-batch（32,768 个 token）里统计两样东西：f<sub>i</sub> = expert i 实际分到的 token 比例，P<sub>i</sub> = 这些 token 给 expert i 的平均 softmax 概率。aux = 1e-3 × 128 × Σ<sub>i</sub> f<sub>i</sub>·P<sub>i</sub>，训练时相当于加到 loss 上一起反向传播。f<sub>i</sub> 只是计数、不求导，梯度只走 P<sub>i</sub>：分到 token 越多的 expert（f<sub>i</sub> 越大），它的概率被往下压得越多。</li></ul></li><li>micro-batch 大小（ladder 每个 rung 的布局表里定的，按显存选）决定每次 aux 统计多少 token：<table style='font-size:12px;margin:4px 0'><thead><tr><th>rung</th><th>节点</th><th>micro-batch</th><th>每次 aux 统计的 token 数</th><th>EP</th></tr></thead><tbody><tr><td>q0</td><td>4</td><td>4</td><td>32,768</td><td>1</td></tr><tr><td>q1</td><td>8</td><td>4</td><td>32,768</td><td>1</td></tr><tr><td>q2</td><td>16</td><td>4</td><td>32,768</td><td>1</td></tr><tr><td>q3</td><td>16</td><td>4</td><td>32,768</td><td>4</td></tr><tr><td>q4</td><td>32</td><td>2</td><td>16,384</td><td>8</td></tr><tr><td>q5</td><td>64</td><td>1</td><td>8,192</td><td>8</td></tr><tr><td>q6</td><td>64</td><td>1</td><td>8,192</td><td>8</td></tr></tbody></table><ul><li><b>EP（expert 并行）</b>：把 128 个 expert 分到几张 GPU 上存，每张只存 128 ÷ EP 个（q3：4 张卡各 32 个；q4–q6：8 张卡各 16 个）。每层 router 先在本卡给自己的 token 选好 8 个 expert，再用 all-to-all 通信把 token 发到存着这些 expert 的卡上计算，算完再发回。它只是为了省显存，不改变计算结果；router 的选择和 aux 统计都在 token 发出去之前、在本卡的 micro-batch 上完成，所以也不影响 aux loss。</li></ul></li></ul><p><b>DSV3 router</b></p><ul><li>每个 expert 的分数单独过 sigmoid，各自在 0–1 之间，互相不竞争。</li><li>选 expert 时用“分数 + bias”排序。每个 expert 有一个 bias，它不是训练出来的参数，而是每步按规则调整：负载太多的 expert，bias 减 0.001；太少的，bias 加 0.001。</li><li><b>bias 只影响选谁，不影响被选中 expert 的输出权重</b>：选出 8 个之后，权重用它们原始的 sigmoid 分数（不含 bias）在这 8 个里归一化（Megatron <code>moe_utils.py</code> L953–959）。</li><li>另外保留一个很小的 aux loss（seq aux 1e-4），按每条序列计算。<ul><li><b>怎么做的</b>：和上面 Switch aux 是同一个公式，只是统计范围换成<b>每一条 8K 训练序列</b>（打包后的一行，可能含几篇文档），系数小 10 倍。每条序列里：f<sub>i</sub> = expert i 分到的 token 比例，P<sub>i</sub> = 这条序列的 token 给 expert i 的平均分数（每个 token 的 128 个 sigmoid 分数先归一化成和为 1）。aux = 1e-4 × 128 × Σ<sub>i</sub> f<sub>i</sub>·P<sub>i</sub>，micro-batch 里 4 条序列各算一个再取平均；同样只有 P<sub>i</sub> 传梯度。因为按序列统计，它和 micro-batch 大小无关。</li></ul></li><li>设计思路：主要靠 bias 做负载均衡，几乎不用 aux loss，避免 aux loss 干扰主任务。</li></ul>",
  "rt_gate_h": "gate（router 权重）用什么优化器、多大 LR", "rt_gate": "<p>gate 就是算 logit 的那个权重矩阵 W<sub>g</sub>（128 × 宽度）。</p><ul><li><b>AdamH</b>：MuonH 默认的做法，用 Adam 更新，但把权重约束在固定半径的球面上（hyperball），权重范数不会变大。</li><li><b>AdamW</b>：普通 AdamW，权重范数可以自由变化（B2 里 wd 设为 0）。</li><li><b>router LR ×1</b>：gate 和模型主体用同一个 LR（新公式下 q0 是 0.0164）。</li><li><b>router LR ×0.19</b>：gate 的 LR 降到约 3.1e-3，大约是 AdamW 基线自己的 LR 水平。之前的探针 B1 显示，AdamW gate 用 ×1 的 LR 会让 logit 饱和，所以才降到 0.19。</li></ul>",
  "arch_thead": ["改动", "Change ID", "具体改了什么", "对照", "LR 公式（按实际峰值 LR 判断）", "已有结果"], "arch_law": {"fit": "新公式（fit v2）", "arch": "旧公式（ArchScale 原版）", "other": "其他"},
  "arch_short": {"MUONH_FIT_DSV3_SWS": "SWA + sink（DSV3 底座，SWS）", "SWA_FIT_SINK_W512": "SWA + sink + 窗口 512", "SWA_FIT_SINK_W128": "SWA + sink + 窗口 128", "SWA_FIT_SINK_W512_R3": "SWA + sink + 窗口 512，3:1（靠底，q3 没训）", "SWA_FIT_SINK_W512_R3T": "SWA + sink + 窗口 512，3:1（靠顶）", "SWA_FIT": "SWA（基础版）", "SWA_FIT_ROPE": "SWA，全注意力层用 RoPE", "SWA_FIT_W4K": "SWA，窗口 4096", "SWA_FIT_W1K": "SWA，窗口 1024", "SWA_FIT_SINK": "SWA + attention sink", "SWA_FIT_SINK_W1K": "SWA + sink + 窗口 1024", "SWA_FIT_DSV3": "SWA（DSV3 router 底座）", "MUONH_SPLITFC1": "MuonH 拆分 fc1"},
  "arch_desc": {"MUONH_FIT_DSV3_SWS": "DSV3 router 底座上的 SWA（窗口 2048，全注意力层 NoPE）+ 可学习 attention sink；Runlong 10-29 起 router 消融的新底座（SWS）", "SWA_FIT_SINK_W512": "attention sink + 窗口 512（窗口这条轴往下扫）", "SWA_FIT_SINK_W128": "attention sink + 窗口 128（窗口这条轴往下扫）", "SWA_FIT_SINK_W512_R3": "同窗口 512 + sink，但每 4 层 1 层全注意力（NoPE），局部:全局从 5:1 改成 3:1（比例这条轴；Boxiang 10-30）", "SWA_FIT_SINK_W512_R3T": "同 3:1，但全注意力层从顶往下数（最后一层一定是全注意力，和 5:1 一样）；R3 从底往上数，有的深度最顶两层都是 SWA。q0（12 层）和 q2（24 层）层数是 4 的倍数，两种数法排出来一样，两条线用同一个 run（q2 的 argv 与 R3 q2 的 golden 逐项相同）；只有 q1、q3 两者不同，R3 的 q3 没训", "SWA_FIT": "每 6 层中 5 层滑动窗口注意力（窗口 2048，RoPE）+ 1 层全注意力（NoPE）", "SWA_FIT_ROPE": "同基础版，但全注意力层也用 RoPE", "SWA_FIT_W4K": "同基础版，窗口 4096", "SWA_FIT_W1K": "同基础版，窗口 1024", "SWA_FIT_SINK": "同基础版 + 可学习 attention sink（gpt-oss 式：每层每个 query head 一个可学习的 sink logit）", "SWA_FIT_SINK_W1K": "attention sink + 窗口 1024", "SWA_FIT_DSV3": "基础版 SWA，但底座换成 DeepSeek-V3 式 router（对照也是 DSV3 底座）", "MUONH_SPLITFC1": "优化器改动（不是注意力）：expert 的 fc1 把 gate 和 up 拆成两个矩阵，各自做 Newton-Schulz 正交化、各自的缩放和球面半径（原来当作一个拼接矩阵）"},
  "rt_left_h": "MuonH 主体：各组 − 对照（softmax router，MUONH_FIT）", "rt_right_h": "AdamW 主体：C − 对照 ②（softmax router，BASE）", "rt_l_a": "DSV3", "rt_l_biasonly": "DSV3 去掉 seq aux", "rt_l_rtmuonh": "DSV3 + MuonH gate", "rt_l_b2": "DSV3 + AdamW gate + router LR ×0.19", "rt_l_b3": "DSV3 + router LR ×0.19", "rt_l_c": "C：DSV3 router",
  "ol_h": "AdamW 的 LR 检查（只有 3 个点，不是格子；不在计划里）", "ol_opt": "优化器", "ol_note": "AdamW 和 Muon 都用 AdamW 的 LR 规则；这里只在个别 rung 上测了规则的 0.5× / 1× / 2×（TODO 3 step 5 的公平性检查），用来和上面 MuonH 的格子对照。数值 = 相对 1× 的 {m} 差（负数 = 比 1× 好）；m* = 3 点抛物线的最优倍数，η* = 对应的绝对 LR。",
  "lf_h": "拟合：最优 LR 随层数和训练量的关系（ArchScale 形式）", "lf_arch": "ArchScale 原版（= 每个格子的 1×，ladder 现用的 LR）", "lf_ours": "我们拟合的", "lf_fix": "我们拟合的，T 指数固定 −0.32", "lf_box": "Boxiang 的拟合 v2（10-25 冻结，MUONH_FIT 在用；val_v2、16 格）", "lf_pts": "格子实测最优 LR（空心 = 外推）", "lf_drag": "拖动旋转 · 悬停看格子", "lf_axL": "层数 L", "lf_axT": "训练 token T", "lf_note": "用格子里 {n} 个格子的最优绝对 LR，在 log 空间做最小二乘：log η* = log η0 + β·log(L/8) + γ·log(T/10.4B)。最优点取抛物线顶点；顶点在已测倍数之外的格子（{excl}）用外推估计，也参与拟合，表中标「外推」；外推超过已测范围 2 倍的不用。±为标准误差。",
  "lf_cells": "每个格子：实测 vs 拟合", "lf_meas": "实测", "lf_fitted": "拟合", "lf_res": "残差", "lf_ext": "外推（只作参考）", "lf_ratio": "拟合 / ArchScale",
  "lf_pred": ["q2 主线（TPP 200）", "q3 主线（TPP 200）", "q6（30B-A3B）", "235B 目标"],
  "lf_warn": "q2 / q3 主线是对格子已覆盖范围附近的预测；q6 和 235B 远超拟合范围（最深 30 层、最多 86B token），只说明方向，不能直接拿来用。另外目前单 seed，深度指数在 9.6B 和 30.8B 两列上不完全一致（约 −0.15 和 −0.06），可分离性还要更多格子确认。",
  "tab_doc": "doc-isolated", "tab_docold": "老实验", "tab_ref": "外部参考", "tab_down": "59B downstream",
  "ce_lack": "还没有按这份清单测的外部模型（只有旧协议 nemo6 的分数，不在下面的表里）：{names}。",
  "ce_ours": "我们自己的 checkpoint 用同一份清单测，分数在“prod downstream”页。",
  "down_old": "只有旧协议（nemo6）分数的 {n} 个 checkpoint：按这份清单测过之后会自动出现在上面的列表里 — 点击展开",
  "tab_fit": "LR 拟合实验", "fit_tab_note": "最优 LR 的扫描与拟合（MuonH 网格、Muon 网格、深度 × 宽度假设检查）。数据同 doc-isolated tab（我们的评测）。", "tab_packed": "packed（Boxiang 训练时验证）",
  "p_notes": ["数据来自 Boxiang 的 <code>results/&lt;Change ID&gt;/scores.json</code>：训练结束时自带的 <b>packed</b> 验证（文档首尾拼接、不加 attention mask），我们不重新测。",
              "只有他的 6 个合并验证集（val_v2 不拆 split，没有 private_internal）；“六集平均”= 这 6 个的简单平均（Boxiang 的定义）。",
              "packed 比 doc-isolated 高约 0.013–0.016，<b>两个 tab 的绝对值不能互相比较</b>，只在同一 tab 内比差值。"],
  "tiles": ["最大 rung 上 KD 降低的 overall loss", "最大 rung 的等效 CE 模型规模", "private_internal 上 KD 降低的 loss", "三个 α 之间的最大差"],
  "tile_sub": ["{rung}，α 0.8，相对 CE（MuonH）", "KD {rung} ≈ {mult}× 激活参数的 CE 模型", "{rung}，α 0.8；训练语料之外的私有集", "同一 rung、同一指标；噪声约 0.008"],
  "d0_h": "第 0 层 dense（第一层不用 MoE）", "d0_note": "把第 0 层的 MoE 换成一个普通的 dense FFN（SwiGLU，宽 = 8 × expert），其余层不变；dense 实验都建在 DSV3 SWS 上。<br>0 线 = <b>MUONH_FIT</b>：MuonH，新 LR 公式（fit），纯 CE（不用 KD），128 选 8 MoE，全注意力，原版 softmax router。数值 = 该线 − MUONH_FIT（doc-isolated，按 seed 配对，q0 是 3 个 seed 的平均；负数 = 更好）。<br><b>负熵均衡 φ（phi balancing），系数 α = 1e-5</b>：每层对 router 概率做滑动平均（EMA），得到每个 expert 长期分到的平均概率 m<sub>e</sub>；损失 = α·E·Σ<sub>e</sub> p<sub>e</sub>·(log m<sub>e</sub> + 1)，p<sub>e</sub> = 当前这批 token 给 expert e 的平均概率（括号里那项不传梯度）。长期用得多的 expert 被往下压、用得少的被往上拉，把分配推向均匀（熵最大）。两种用法（各组成见下表）：<b>bias + 负熵</b> = 保留 DSV3 的 expert bias（γ = 1e-3），只把 seq aux 换成负熵；<b>只用负熵</b> = expert bias 和 seq aux 都去掉，只靠这一项均衡。统计范围是整个 global batch，和 micro-batch 大小无关（源自 arXiv 2605.15403；详见前面「Router 负熵均衡 (phi balancing)」一章）。<br><b>结论：不采用。</b>loss 只比 DSV3 低一点（下图两条负熵实线比蓝色实线低 0.002–0.006），在单 seed 噪声（约 ±0.008）以内；在完整 PROD 配置里几乎没差（PROD1_PHI − PROD1_DPSK：q0 0.000，q1 −0.001，q2 −0.002）；只用负熵的版本 router 不健康（见下面 router 健康图）。Runlong 11-03 定 DSV3 为默认，负熵的 run 全部停掉。", "d0_chart": "各线 − MUONH_FIT；虚线 = 全 MoE 参照（没有 dense 层），实线 = 第 0 层 dense（鼠标移到某个 rung 看所有线的数值）", "d0_y": "Δloss（该线 − MUONH_FIT）", "d0_l_dsv3": "DSV3 · 全 MoE · 全注意力", "d0_l_sws": "DSV3 · 全 MoE", "d0_l_d1": "DSV3 · 第 0 层 dense", "d0_l_d2": "DSV3 · 第 0、1 层 dense", "d0_l_fa0": "DSV3 · 第 0 层 dense（该层全注意力）", "d0_l_phib_moe": "bias + 负熵 · 全 MoE", "d0_l_phinb_moe": "只用负熵 · 全 MoE", "d0_l_phib": "bias + 负熵 · 第 0 层 dense", "d0_ltab": {"head": ["线", "注意力", "第 0 层", "sigmoid 打分，top-8，权重归一化", "expert bias（γ = 1e-3，每步按负载调）", "seq aux（α = 1e-4）", "负熵 φ（α = 1e-5）", "Change ID"], "rows": {"d0_l_dsv3": ["全注意力", "MoE", True, True, True, False], "d0_l_sws": ["SWS", "MoE", True, True, True, False], "d0_l_d1": ["SWS", "<b>dense</b>", True, True, True, False], "d0_l_d2": ["SWS", "<b>dense</b>（第 1 层也 dense）", True, True, True, False], "d0_l_fa0": ["SWS；第 0 层全注意力（NoPE）", "<b>dense</b>", True, True, True, False], "d0_l_phib_moe": ["SWS", "MoE", True, True, False, True], "d0_l_phib": ["SWS", "<b>dense</b>", True, True, False, True], "d0_l_phinb_moe": ["SWS", "MoE", True, False, False, True], "d0_l_phinb": ["SWS", "<b>dense</b>", True, False, False, True]}, "foot": "router 三种：<b>DSV3</b> = expert bias + seq aux；<b>bias + 负熵</b> = 把 DSV3 的 seq aux 换成负熵；<b>只用负熵</b> = expert bias 和 seq aux 都不要（只剩 sigmoid 打分，已不算 DSV3）。SWS = SWA 窗口 2048，局部:全局 5:1，加 attention sink。所有线相同：MuonH、拟合 LR、CE；128 个 expert 选 8 个（无 shared）；dense 层 = SwiGLU，宽 8 × expert（与 MoE 层激活宽度相同）；seed 1，同一数据顺序。同一种 router 的全 MoE（虚线）和第 0 层 dense（实线）用同一颜色；下面的 router 健康图用同样的颜色和线型。"}, "d0_l_phinb": "只用负熵 · 第 0 层 dense", "d0_missing": "还没有我们自己的评测、所以没画：", "d0_proc_h": "过程", "d0_steps": ["<b>起因（11-01）</b>：负熵均衡去掉 bias 那组多出来的 dead expert，主要集中在<b>第一个 MoE 层</b>（q3：60 个 dead 里有 32 个在第一层，第一层 top-8 expert 拿走约 40% 的 token）。参考外部模型：DeepSeek-V3 前 3 层是 dense，Kimi K2 / K3 前 1 层，MiMo V2 Flash / V2.5 / V2.6 Pro 前 1 层，Flint 前 2 层；Qwen3 和 MAI-Base-2 全部是 MoE。", "<b>设计（Runlong 11-01 15:4x：“把我们的 q1 q2 q3 第 0 层和第 0、1 层，dpsk 和我们的 phi 都发了”）</b>：第 0 层 dense（DENSE1）和第 0、1 层 dense（DENSE2），各配三种 router：DSV3（对照）、负熵 + bias、负熵去掉 bias；q1 / q2 / q3 各 1 个 seed，共 18 个 run。dense 层的宽度 = 8 × expert 宽度 = 3h（q1 2304 / q2 3072 / q3 3840），也就是和 MoE 层每个 token 实际用到的 FFN 宽度一样（DSV3 / Kimi / MiMo 的做法）；注意力不变，第 0 层仍是 SWA。底座 = DSV3 SWS。", "<b>1 层还是 2 层（11-02 05:2x）</b>：事先定的规则（Runlong 11-01 21:1x：“只发更好的”）：q1、q2 上同一 router 的 DENSE2 − DENSE1 取平均，|均值| ≥ 0.002 且 q2 基本同号就选 loss 低的。结果 DENSE2 − DENSE1 平均 +0.0025（六集合 +0.0030），q2 三组同号 → <b>选 1 层</b>，q4 只跑 DENSE1 的三组。", "<b>第 0 层用 SWA 还是全注意力（11-02 11:0x–12:xx）</b>：Runlong 先提出“第一层 dense 时也不能用 SWA”（参考 MiMo 第 0 层 = 全局注意力 + dense，Flint 第 0、1 层 dense + 全注意力），随后改成做消融：“第一层用 swa 还是 full，swa 就是 rope，full 就是 nope”。FA0 = 第 0 层改成全注意力 + NoPE（在原有的全局层之外多一层）。结果 FA0 比 SWA <b>差 0.006（q1）、0.007（q2）</b>，router 健康没有变好 → PROD v1 第 0 层用 SWA。", "<b>收尾（11-03 10:5x）</b>：router 评审后 DSV3 router 定为生产默认，Runlong：“phi 都停了吧” → 负熵的 dense q4 两组停掉（checkpoint 保留）。DSV3 + dense 第 0 层的 q4 搬到 slurm-2 继续排队；FA0 的 q3 在评测中。"], "d0_res_h": "结果：改动 − 对照（六个验证集平均，文档隔离，seed 1，负数 = 更好）", "d0_thead": ["实验", "改动 − 对照"], "d0_r_d1": "第 0 层 dense − 全 MoE（DSV3 router）", "d0_r_d2": "第 0、1 层 dense − 全 MoE", "d0_r_d2d1": "2 层 dense − 1 层 dense（正数 = 1 层更好）", "d0_r_fa0": "dense 第 0 层：全注意力 NoPE − SWA RoPE", "d0_r_phib": "dense 第 0 层上：负熵 + bias − DSV3", "d0_r_phinb": "dense 第 0 层上：负熵去掉 bias − DSV3", "d0_res_foot": "数值取自共享实验卡片（Boxiang 式 R1）。“·” = 还没有数据；q4 还在跑。单个 seed 的噪声约 ±0.008。", "d0_hl_h": "router 健康：全 MoE（虚线）对比 dense 第 0 层（实线），颜色和线型同上表", "d0_hk": ["val_v2 上的 dead expert 个数", "最忙 expert 的 MaxVio"], "d0_h_arm": "router", "d0_h_moe": "全 MoE", "d0_h_dense": "dense 第 0 层", "d0_h_ctrl": "DSV3", "d0_h_phib": "负熵 + bias", "d0_h_phinb": "负熵去掉 bias", "d0_hl_note": "与上面「router 健康统计」同一格式：来自训练日志（Boxiang 的 <code>router_health.py</code>，每 100 步汇总一次，覆盖所有 MoE 层，seed 1）。每个 rung 一张图，横轴 = 训练步数；上方按钮切换指标，鼠标悬停看数值。q4 的 dense 组还在训练，所以线比较短。", "d0_hl_foot": "dense 第 0 层主要救的是只用负熵那组：q1 / q2 / q3 训练结束时 dead expert 从 0.26 / 0.68 / 0.65% 降到 0，MaxVio（最差层）从 8.5 / 9.3 / 6.5 降到 3.4 / 2.4 / 4.4，但仍远高于 DSV3 的约 0.3。不过训练过程中，第 1 层（dense 之后的第一个 MoE 层）仍有 dead expert：q2 前半段约 11–12 个，q3 大部分时间 18–25 个，到训练末尾才降到 0。DSV3 和 bias + 负熵加不加 dense 都健康（dead 0，MaxVio 最差层 ≤ 0.36）。<br>横轴是训练步数，每个点是该步所有 MoE 层的汇总（层平均 / 最差层），不是按层画的。q4 浅橙色实线（bias + 负熵 · 第 0 层 dense）在第 ~44,200 步的尖峰来自最后两层（L34 / L35，MaxVio 约 8.5，L35 有 4 个 dead expert），约 200 步后恢复；同一步只用负熵的 dense run 也跳了一下（来自第 1 层），两个全 MoE run 没有。", "pp_h": "数据 derisk 和 2.5 vs 2.6", "pp_note": "<b>数据</b>：80B 的 KD 语料预先打包成 32K 的行（pack plan e9r500：一行由若干文档块拼成，块之间互不 attend），teacher cache = tc_80b，seed 1（plan 固定了数据顺序）。<br><b>实验</b>：MUONH_FIT 底座在同一份 plan 上训 CE / KD 2.6 / KD 2.5，检查新数据管线、KD 在新数据上的增益，以及 teacher 2.5 对 2.6。KD 都是 α 0.8（KD 0.8 + CE 0.2）、tail = S64Y。<br><b>每个 checkpoint 评两份</b>：8K 窗口（长文档按 8192 切块，和之前所有实验一样，可直接比较）和 32K 窗口（按 32768 切块，和训练长度一致）。", "pp_l_ce": "MUONH_FIT_PP32K", "pp_l_kd26": "KD 2.6 · MUONH_FIT · 新数据", "pp_l_kd25": "KD 2.5 · MUONH_FIT · 新数据", "pp_l_pd25": "KD 2.5 · PROD1_DPSK · 新数据", "pp_l_ba26": "KD 2.6 · 最佳架构 · 新数据", "pp_l_ba25": "KD 2.5 · 最佳架构 · 新数据", "pp_l_old26": "KD 2.6 · MUONH_FIT · 650B 数据（参照）", "pp_l_old25": "KD 2.5 · MUONH_FIT · 45B 语料（参照）", "pp_l_swa": "CE · SWA 512 + sink 5:1 · 650B 数据（参照）", "pp_l_pdce": "CE · PROD1_DPSK · 650B 数据（参照）", "pp_l_lat": "CE · PROD1_DPSK + latent MoE（等参数）· 650B 数据（参照）", "pp_l_lat2x": "CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照）", "pp_old32_h": "650B 参照在 32K 窗口下（不画进图）：32K 窗口分数 − MUONH_FIT（0 线同上图，按 seed 配对；正数 = 比 0 线差）", "pp_tab": {"head": ["线", "数据", "架构", "loss", "Change ID（8K 窗口）"], "rows": {"ce": ["80B pack plan，32K", "MUONH_FIT：softmax router、全注意力、文档 mask", "CE"], "kd26": ["80B pack plan，32K", "同上", "KD，teacher MiMo-2.6-Pro-RL"], "kd25": ["80B pack plan，32K", "同上", "KD，teacher MiMo-2.5-Pro-Base"], "pd25": ["80B pack plan，32K", "PROD1_DPSK（PROD v1：SWA 512 局部:全局 5:1 + sink、DSV3 router、第 0 层 dense、8 + 1 shared expert、文档 mask）", "KD，teacher MiMo-2.5-Pro-Base"], "ba26": ["80B pack plan，32K", "最佳架构 PROD1_DPSK_LATENT2", "KD，teacher MiMo-2.6-Pro-RL"], "ba25": ["80B pack plan，32K", "最佳架构 PROD1_DPSK_LATENT2", "KD，teacher MiMo-2.5-Pro-Base"], "old26": ["650B 数据，8K", "MUONH_FIT + 文档 mask", "KD，teacher MiMo-2.6-Pro-RL"], "old25": ["45B KD 语料，8K", "MUONH_FIT（无文档 mask）", "KD，teacher MiMo-2.5-Pro-Base"], "swa": ["650B 数据，8K", "滑动窗口 512 + attention sink，局部:全局 5:1（全局层 NoPE，局部层 RoPE 只看 512），无文档 mask", "CE"], "pdce": ["650B 数据，32K", "PROD1_DPSK，和「KD 2.5 · PROD1_DPSK · 新数据」同一架构；激活参数比 MUONH_FIT 多约 8%，训练 token 也多约 8%（TPP 200 按自己的激活参数算）", "CE"], "lat": ["650B 数据，32K", "PROD1_DPSK + latent MoE（等参数）：MoE 先把 h 维投影到 h/2，256 个 routed expert 选 16 个在 h/2 上算，再投影回 h；总参数、routed FLOP 都和 PROD1_DPSK 相同；文档 mask", "CE"], "lat2x": ["650B 数据，32K", "PROD1_DPSK + latent MoE（d = h/2），routed expert 512 个选 16 个：每个 token 的计算量和 PROD1_DPSK 基本相同，routed 参数 ×2（不是等参数）；文档 mask", "CE"]}, "foot": "所有 KD 组都是 α 0.8（KD 0.8 + CE 0.2），tail = S64Y（top-64 + 采样 tail 4 draws，y 精确），MuonH + 拟合 LR。32K 窗口的评测 Change ID 在后面加 _W32K。虚线 = 之前的同类 run（8K 训练）作参照：KD 2.6 用 650B 数据上的；KD 2.5 在 650B 数据上没有 run，用 45B KD 语料上的（它的 q3 是 TPP 100，不画；只有 8K 窗口）。PROD1_DPSK 和 latent MoE expert ×2 两条参照都是 650B 数据上 32K 训练（加文档 mask）的 CE，所以三个窗口 tab 都画。"}, "pp_y": "Δloss", "pp_none": "还没有评测结果。", "pp_a_h": "8K 窗口：各线 − MUONH_FIT", "pp_a_zero": "0 线 = <b>MUONH_FIT</b>：650B 数据、8K 训练、MuonH、拟合 LR、CE、softmax router、全注意力（和本报告其他章节同一条 0 线）。负数 = 更好。", "pp_b_h": "32K 窗口：各线 − MUONH_FIT", "pp_c_h": "32K 窗口：各线 − MUONH_FIT_PP32K（新数据 CE，32K 窗口）", "pp_c_zero": "0 线 = <b>MUONH_FIT_PP32K</b>：同一份 80B pack plan、32K 训练的 CE，也按 32K 窗口评，按 seed 配对。各线和它一样是 32K 窗口分数；负数 = 比新 CE 好。", "pp_b_zero": "0 线 = <b>MUONH_FIT</b>，和 8K 窗口 tab 同一条（它是 8K 训练，用的就是它的 8K 窗口分数，只作基准参考）；各线是 32K 窗口的分数。负数 = 更好。", "pp_win": [["w8", "8K 窗口"], ["w32", "32K 窗口"], ["w32ce", "32K 窗口 · 0 线 = 新 CE"]], "pp_got": "已有评测：", "ba_h": "架构变化", "ba_tab": {"head": ["大项", "小项", "当前最佳架构（我们按全部实验结果逐项取最好）", "当前PROD选择（Runlong）"], "groups": [["架构", [["Router 打分", "DSV3 式：sigmoid，top-8", "同左"], ["负载均衡", "DSV3：expert bias，γ = 1e-3；seq aux，α = 1e-4", "同左"], ["第 0 层", "dense SwiGLU（宽 9 × expert），注意力用 SWA", "同左"], ["MoE 层", "128 选 8 + 1 shared，routed scaling 2.5；+ latent 1/2（256 选 16）", "128 选 8 + 1 shared，routed scaling 2.5"], ["注意力", "SWA 512 + RoPE；每 6 层 1 层全注意力 + NoPE（5:1）；attention sink", "同左"], ["文档 mask", "文档间 attention mask", "同左"], ["训练长度", "8K", "32K"]]], ["优化器", [["优化器", "MuonH + AdamH（head、router）+ AdamW（向量），wd 0", "同左"], ["router LR", "主 LR × 1", "同左"], ["LR", "拟合公式（Boxiang fit v2，= MUONH_FIT）；无 warmup，线性衰减到 1%", "q0–q3 同左；q4 / q5 固定 5.36e-3 / 4.49e-3"], ["batch", "每步 2<sup>21</sup> token", "同左"]]], ["KD", [["loss", "KD（forward KL）", "纯 CE"], ["α", "0.8（KD 0.8 + CE 0.2）", "—"], ["teacher", "MiMo-2.5-Pro-Base", "—"], ["tail sampling", "S64Y：top-64 + 采样 tail（4 draws），y 精确", "—"]]]], "foot": "橙色 = 两边不一样。左列每一项取自各自的单变量实验，组合在一起还没跑过；其中 latent 1/2 和训练长度 8K 只有 q0 的证据（8K 是在我们 8K 长度的评测上，32K 对长文档的好处测不到）。负载均衡没有选负熵：它的 loss 只比 DSV3 低 0–0.007（单 seed 噪声约 ±0.008），在 PROD 设置下持平（PROD1_PHI − PROD1_DPSK ≈ 0），只用负熵（去掉 expert bias）的版本 router 不健康。"}, "ba_thead": ["组件", "实验", "改动 − 对照"], "ba_c_router": "① DSV3 router", "ba_r_dsv3": "softmax router → DSV3 router（全注意力）", "ba_r_phi": "PROD v1 里把均衡损失换成负熵（phi，α 1e-5）", "ba_c_dense0": "② 第 0 层 dense（SWA）", "ba_r_dense1": "第 0 层 MoE → dense（DSV3 SWS 上）", "ba_r_dense2": "第 0、1 两层都改 dense", "ba_r_fa0": "dense 第 0 层再把注意力从 SWA（RoPE）改成全注意力（NoPE）", "ba_c_shared": "③ 8 + 1 shared expert", "ba_r_shared": "加 1 个 shared expert（routed scaling 1；<b>不是等参数</b>，激活参数约多 7–8%）", "ba_r_shared_rs": "加 1 个 shared expert + routed scaling 2.5（<b>不是等参数</b>）", "ba_r_rs1": "PROD v1 里 routed scaling 2.5 → 1", "ba_c_latent": "latent MoE（1/2）", "ba_r_latent": "PROD v1 的 MoE 换成 latent MoE：先投影到 h/2 维，256 个 expert 选 16 个在 h/2 上算，再投影回 h（等参数、等 routed FLOP）", "ba_c_swa": "④ SWA 512 5:1 + sink", "ba_r_swa": "全注意力 → SWA 窗口 512、局部:全局 5:1、加 sink（softmax router）", "ba_c_ctx": "⑤ 32K + 文档间 mask", "ba_r_s32k": "训练长度 8K → 32K（不加 mask）", "ba_r_dm": "8K 加文档间 mask", "ba_r_s32kdm_vs32": "32K 再加文档间 mask", "ba_r_s32kdm_vs8": "32K + mask 对比 8K 不加 mask", "ba_c_all": "整套配方", "ba_r_all": "PROD v1 对比 32K 的 SWA 512 + sink（同时改了上面多项，不是单变量）", "ba_foot": "评测都在 8K 长度上做。“·” = 该 rung 还没有数据。PROD v1 系列、dense 第 0 层全注意力、文档 mask 这几组还没有跑我们自己的 115 split 评测，所以这里没有 overall_tsqw，也没有画进后面的图。", "find_h": "结论",
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
  "lang": "en", "title": "59B production run · Qwen3-MoE scaling ladder",
  "sub": "The first two tabs = val loss and downstream evals of the 59B production run (ULTRA59B_MUONH_KD_QBK3) (downstream every 2k steps; the full val-loss suite about every 4–6k steps, val_v2 / v3small in between); the others are the archive of external models and the ladder research (document-isolated eval: 9 decon suites incl. private_internal, dpl metric definitions). Generated {ts}",
  "nav_arch": "Earlier research (separate file):", "nav_arch_here": "Earlier research:",
  "title_arch": "Qwen3-MoE scaling ladder · research archive",
  "sub_arch": "Archive of the ladder research: document-isolated eval (9 decon suites incl. private_internal, dpl metric definitions), older experiments, LR fits, packed validation. The 59B production run's val loss and downstream evals are on the main page qwen3_scaling_ladder_eval_en.html. Generated {ts}",
  "nav": ["Findings", "Optimizers (CE)", "KD (MuonH)", "Heatmap", "Tables", "Notes"],
  "muon_fit": "Muon (new fitted LR)", "ce_fit": "MuonH (new fitted LR, MUONH_FIT)", "opt_h": "Optimizer comparison", "opt_note": "Dashed = the same optimizer at the new fitted LR law of Boxiang (MuonH: fit v2 η = 0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259, q2 / q3 so far; no Muon run at a fitted law). Same data and token count; only the optimizer (with its LR rule / schedule) changes. x: active non-embedding parameters (log scale).",
  "o2_h": "Δloss vs AdamW (optimizer − AdamW, more negative is better)",
  "kd_h": "KD vs CE", "kd_note": "CE and KD differ only in the loss (α = 0 vs 0.7 / 0.8 / 0.9); optimizer, data order and layout are identical.",
  "k2_h": "KD − CE: Δloss per α (more negative is better)",
  "adamw": "AdamW", "muon": "Muon", "vs_adamw": "Δloss (optimizer − AdamW, more negative is better)", "mean6": "mean of 6",
  "v_h": "Tail estimation", "v_note": "The 8 runs differ only in how the teacher distribution is truncated in the KD term: α = 0.8, MuonH optimizer, seed 1 and data order are identical; the baseline is still <b>CE</b> (MuonH, α = 0); <b>B64</b> (top-64 + the rest of the tail as one bucket) is KD α 0.8 itself and is drawn as a line too, so each variant can be read against it. Negative = better than CE.",
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
  "arch_y": "Δloss (change − control, more negative is better)", "arch_chart": "Δloss of each change vs its own control (all at the fitted LR law; where 3 seeds exist the point is the mean of the 3 per-seed paired differences, otherwise one seed; hover a rung for all values)", "arch_h": "Sliding-window attention (SWA)", "arch_c": {"MUONH_FIT": "softmax router: each change − MUONH_FIT (full attention, softmax router)", "MUONH_FIT_DSV3": "DSV3 router: each change − MUONH_FIT_DSV3 (full attention, DSV3 router)"}, "arch_note": "Setting: MuonH, fitted LR law (fit v2), plain CE (no KD), 128-choose-8 MoE; q0-q3, some arms with 3 seeds at q0.<br>Left: softmax router, zero line = <b>MUONH_FIT</b> (full attention); right: DSV3 router, zero line = <b>MUONH_FIT_DSV3</b> (full attention). Values = change − zero line (doc-isolated, negative = the change is better); the settings of each change are in the table below.",
  "arch_name": {"MUONH_FIT_DSV3_SWS": "SWA + sink (DSV3 router base, SWS) − same base without SWA (MUONH_FIT_DSV3)", "SWA_FIT_SINK_W512": "SWA + sink + window 512 − MUONH_FIT", "SWA_FIT_SINK_W128": "SWA + sink + window 128 − MUONH_FIT", "SWA_FIT_SINK_W512_R3": "SWA + sink + window 512, local:global 3:1 (global layers at the bottom, q3 not trained) − MUONH_FIT", "SWA_FIT_SINK_W512_R3T": "SWA + sink + window 512, 3:1 (global layers at the top) − MUONH_FIT", "SWA": "SWA − MuonH (old LR law)", "SWA_FIT": "SWA − no SWA (MUONH_FIT)", "SWA_FIT_ROPE": "SWA, RoPE in the full-attention layer − MUONH_FIT", "SWA_FIT_W4K": "SWA, window 4096 − MUONH_FIT", "SWA_FIT_W1K": "SWA, window 1024 − MUONH_FIT", "SWA_FIT_SINK": "SWA + learnable attention sinks (gpt-oss style) − MUONH_FIT", "SWA_FIT_SINK_W1K": "SWA + sinks + window 1024 − MUONH_FIT", "SWA_FIT_DSV3": "SWA (DSV3-router base) − same base without SWA (MUONH_FIT_DSV3)", "MUONH_SPLITFC1": "MuonH with split fc1 (gate / up stepped separately) − MuonH"},
  "mg_g_h": "Muon learning rate: the same cells (same Muon, CE)", "mg_g_note": "Boxiang's full Muon grid (decided 10-24): the MuonH cells above, multiples 0.25 / 0.5 / 1 / 2 / 4× (coarser, since Muon's optimum is unknown). 1× = the ladder's Muon LR (the AdamW rule: depends on model size only, <b>not on training tokens</b>). Read as above: big number = parabola vertex m*, η* = absolute LR, hover a cell for its curve. Purpose: does MuonH still lead when Muon also runs at its own best LR?", "mg_lf_h": "Fit: Muon best LR vs active parameters N and training tokens (same form as the Muon adamw_active rule)", "mg_lf_arch": "current Muon / AdamW LR (the ladder adamw_active rule, not ArchScale; every cell 1×)", "mg_lf_box": "Boxiang Muon fit v3 (on record 10-26; val_v2, 9 cells / 43 points; not used by any run; domain L 12-30, T 9.6-137B)", "mg_lf_axN": "active params N", "mg_noT": " (depends on active parameters N only, no T term)",
  "ref_unsure": " (possibly wrong)",
  "ref_dag": "vocab size: the model's config vocab_size, i.e. embedding rows (padding included, so a little above the tokenizer's token count, e.g. Qwen3 151,936 rows / 151,665 tokens); our checkpoint: its padded training vocab.<br>model size: MoE as total-Aactivated, dense as total, from the model cards; our checkpoint includes the embedding and output head (the 1.17B etc. in its name = non-embedding activated params); Qwen3.8-Flash-Next has a further 51B n-gram embedding outside its 125B.<br>† Own tokenizer: val_loss / ppl are per ITS token and not comparable with the rows per Qwen3 token (use bpb). bpb = total NLL / (UTF-8 bytes of the original text × ln 2), the same denominator as the ladder; text = the originals in ~/data/eval_ppl_decon, same chunking as the ladder (first token of a document not scored, BOS as context only).", "ref_h": "Pretrain doc-isolated eval", "ref_gap": "our best − MiMo-V2.5-Pro-Base", "ref_best": "our best", "ref_w32_h": "32K window: long documents chunked at 32K", "ref_w32_note": "The same models and settings as the 8K window; only the chunk length of long documents changes from 8192 to 32760 predictions (Qwen3-30B-A3B-Base's maximum length is exactly 32768). The scored tokens are identical; in documents longer than 8K the later tokens see a longer context: ""36M of the 234M predicted tokens (15 %) are affected - val_v2 35 %, pile 24 %, uncheatable 9 %, private_internal 4 %; codefresh has no document over 8K (unchanged). The difference to the 8K window is in its own table below. The ladder models are trained at 8K and are not in this view.", "ref_w32_pending": "The 32K-window evals are running (jobs 36018-36026; Mistral-Large-4 follows once its 8K run works); the results appear here automatically.", "ref_d_h": "32K − 8K (bpb)", "ref_d_note": "bpb: 32K window − 8K window; negative = the longer context lowers the loss.", "ref_win": [["w8", "window = 8K"], ["w32", "window = 32K"], ["wd", "32K − 8K"]], "ref_note": "MiMo-V2.5-Pro-Base (fp8) and MiMo-V2.6-Pro-RL (the released mxfp4 weights; also the KD teacher), scored on exactly the ladder's doc-isolated 115 splits (vLLM prompt_logprobs; every split's predicted-token count equals the ladder's). MiMo's token ids are the ladder's Qwen3 ids, so per-token losses compare directly. Absolute losses (lower is better); only our best ladder checkpoint (lowest overall_tsqw of all evaluated checkpoints); last row = it minus MiMo-Base (ppl: ratio). The buttons switch val_loss / ppl / bpb. The RL model is ~0.05 above Base on pretraining text, as expected (RL post-training + quantisation). Qwen: Qwen3-30B-A3B-Base and Qwen3-235B-A22B (bf16) use the ladder's Qwen3 tokenizer, so their losses compare directly; no Base of the 235B was released, this is the post-trained model (expect it higher). Qwen3.8-Flash has no open weights; the open Qwen3.8-Flash-Next (official FP8, post-trained, new tokenizer: bpb only) stands in for it. Qwen3.8-27B is a dense model (bf16, post-trained, no Base release) with the same tokenizer as Qwen3.8-Flash-Next: bpb only.",
  "aw_h": "Checking the LR law's assumptions: depth × width, diagonal used as ArchScale anchors (same MuonH, CE, 9.6B tokens)", "aw_note": "Runlong.md TODO 3 steps 27, 30, 35: training tokens fixed at 9.6B (4,577 steps, GBS 2M); an LR sweep (multiples of the ArchScale law) at several depths L and widths h, to see whether the best multiple m* depends on depth only (the law assumes no width term). Diagonal = the ladder rungs (q0-q6); other cells = extra depth / width points. Metric {m}; read the cells as above (hover for the curve). Note: 9.6B is heavily under-trained for the large models (q6 TPP 3.5).", "aw_ladder": "ladder {q}", "aw_extra": "extra cell", "aw_unstable": "router collapse (load-balancing loss 9-16): not an LR point", "aw_no1x": " (no 1×: relative to the smallest multiple)", "ac_h": "Other checks: batch, LR floor, per-group LR", "ac_head": ["check", "experiment", "reference", "reading"], "ac_batch": "batch (q1, L 18, 9.6B)", "ac_batch_r": "Runlong's pre-registered predictions: law in tokens 1.45 / in steps 1.81 / B^0.56 2.14 / sqrt-noise 2.56", "ac_floor0": "LR floor (q0, L 12)", "ac_floor_r": "MuonH: no warmup, linear decay from the start to X% of the peak (not WSD); 10% = the ArchScale setting, the ladder uses 1%", "ac_floor3": "LR floor (q3, L 30)", "ac_floor3_r": "vs the same-floor 1×; negative = better than 1×", "ac_split": "per-group LR (q3, L 30)", "ac_split_r": "vs 1×: A = AdamW groups only (embedding, 1-D) at 1.74×; B = hidden matrices + LM head + router at 1.74×; C = hidden matrices only at 1.74×",
  "rt_set_head": ["arm", "Change ID", "main optimizer", "router score", "load balance", "gate (router weight) optimizer", "router LR", "LR rule / schedule", "partner", "rungs run"], "rt_set_rows": [["partner ①", "MUONH_FIT", "MuonH", "softmax", "aux loss 1e-3", "AdamH (sphere)", "×1", "fitted law; no warmup, linear decay to 1 %, wd 0", "—", "q0 ×3 seeds, q1-q3"], ["<b>A</b>", "MUONH_FIT_DSV3", "MuonH", "<b>sigmoid</b>", "<b>expert bias</b> + seq aux 1e-4", "AdamH", "×1", "as ①", "①", "q0 ×3, q1-q3"], ["<b>B2</b>", "…_DSV3_RTADAMW_RT0P19", "MuonH", "sigmoid", "as A", "<b>AdamW</b> (wd 0)", "<b>×0.19</b>", "as ①", "A", "q0 ×3, q1"], ["<b>B3</b>", "…_DSV3_RT0P19", "MuonH", "sigmoid", "as A", "AdamH", "<b>×0.19</b>", "as ①", "A", "q0 ×3"], ["partner ②", "BASE", "AdamW", "softmax", "aux loss 1e-3", "AdamW", "×1", "adamw_active; WSD (5 % warmup, last 20 % decay), wd 0.1", "—", "q0 ×3, q1-q5"], ["<b>C</b>", "BASE_DSV3_ONETREE", "AdamW", "<b>sigmoid</b>", "<b>expert bias</b> + seq aux 1e-4", "AdamW", "×1", "as ②", "②", "q0 ×3, q1, q2"], ["S", "MUONH_FIT_DSV3_LR0P7X / 1P4X", "as A, whole LR ×0.7 / ×1.4", "", "", "", "", "", "A", "q0"], ["S_C", "BASE_DSV3_ONETREE_LR0P5X / 2X", "as C, whole LR ×0.5 / ×2", "", "", "", "", "", "C", "q0"]], "rt_set_l": "MuonH body (control 1: MUONH_FIT, softmax router)", "rt_set_r": "AdamW body (control 2: BASE, softmax router)", "rt_set_note": "All arms: 128 experts choose 8, no shared expert, batch 2M tokens; each arm changes only the bold items: A and C relative to the control of their table, B2 and B3 change the gate on top of A. Router LR ×1 = the main LR (fitted law 0.0164 at q0), ×0.19 ≈ 3.1e-3. ONETREE = the same code tree as A, so C − BASE crosses code trees.", "rt_chart_h": "Results: loss difference arm − partner (overall_tsqw, more negative is better). q0 has 3 seeds: hollow points = each seed paired difference, filled point and line = the 3-seed mean; other rungs 1 seed", "rt_ylab": "Δloss (arm − partner)", "rt_lr_note": "LR checks (q0): MuonH + DSV3 has its best multiple at {s}× the fitted law, AdamW + DSV3 at {c}× adamw_active; both current LRs are near the optimum.",
  "rt_hm_h": "Per-layer load balance (training logs, seed 1)", "rt_hm_note": "From the router monitor of Boxiang, <code>router_health.py</code> (<code>LADDER_ROUTER_HEALTH=100</code>, every 100 steps). Each panel: y = layer (layer 0 on top), x = training step (at most 60 columns, neighbouring reports averaged); darker = larger; above each panel the layer mean at the end of training; hover for values. Metrics: <b>MaxVio</b> = tokens routed to each expert of each layer over a whole training step (all micro-batches, all ranks): the busiest expert / the mean - 1 (the DeepSeek-V3 measure; 0 = perfectly balanced, 2.86 = the busiest gets 3.86x the mean; colour scale capped at 4); <b>dead</b> = experts below 10 % of the mean load, as % of the 128; <b>|bias| max</b> = the largest absolute DSV3 per-expert bias (pre-registered health limit 1); <b>bias spread</b> = max - min of the bias within a layer - top-8 selection only sees bias differences, so a common shift of all experts (large |bias|, small spread) does not change routing; <b>logit saturation</b> = share of router logits with |z| &gt; 4 and <b>max |logit|</b> = the largest |z|, both on rank 0 first micro-batch only; |z| &gt; 4 includes z &lt; -4 (an expert confidently rejected, normal); <b>gate update</b> = the change of the router weight over the step / its norm (a flag, not a criterion). The softmax router has no bias; BASE (softmax AdamW) and MUONH_FIT q2 / q3 ran without the monitor. <b>Note:</b> the training logs cut every line at ~1000 characters, so the per-layer line is incomplete and the last layers of the DSV3 arms are missing (grey; q0 1 layer, q1 4, q3 16) - lost at the source.", "rt_hm_vio": "MaxVio", "rt_hm_dead": "dead (%)", "rt_hm_bmax": "|bias| max", "rt_hm_bspread": "bias spread", "rt_hm_sat": "logit saturation (%)", "rt_hm_zmax": "max |logit|", "rt_hm_upd": "gate update", "rt_hm_nobias": "softmax: no bias", "rt_hm_miss": "{n} layers missing", "rt_hm_na": "no monitor data", "rt_hm_end": "end, layer mean", "rt_hm_layers": "layer ↓ · step →", "rt_hm_soft": "softmax (MUONH_FIT)", "rt_hm_a": "DSV3", "rt_hm_b2": "DSV3 + AdamW gate + router LR ×0.19", "rt_hm_b3": "DSV3 + router LR ×0.19", "rt_hm_c": "C: AdamW + DSV3",
  "rt_st_h": "Router health statistics", "rt_st_note": "Summary lines (all layers) of the router monitor of Boxiang, <code>router_health.py</code> (every 100 steps). One chart per rung, x = training step, one line per arm; the buttons switch the statistic, hover for the values. MUONH_FIT q2 / q3 ran without the monitor.", "rt_st_name": {"vio_mean": "MaxVio (layer mean)", "vio_max": "MaxVio (worst layer)", "dead": "dead experts (%)", "bmax": "|bias| max", "sat": "logit saturation (%)", "satpos": "z > +4 (%)", "pinned": "top-8 pinned (%)", "zmax": "max |logit|", "upd": "gate update"}, "rt_st_def": {"vio_mean": "Per layer: tokens routed to the busiest expert over a whole training step (all micro-batches, all ranks) / the mean - 1 (DeepSeek-V3 MaxVio, 0 = perfectly balanced); here the mean over layers.", "vio_max": "As above, the least balanced layer.", "dead": "Experts below 10 % of the mean load, as % of (layers × 128).", "bmax": "The largest absolute DSV3 per-expert bias over all layers; red dashed = the pre-registered health limit 1. Top-8 selection only sees bias differences within a layer, so a common shift of all experts does not change routing (e.g. A q3 layer 0: bias between +11.2 and +11.7). The softmax router has no bias.", "sat": "Share of router logits with |z| &gt; 4 (mean over layers; rank 0 first micro-batch). Includes z &lt; -4 (an expert confidently rejected, normal).", "satpos": "Share with z &gt; +4: scores pinned at 1, the selected experts lose discrimination (bad); red dashed = the 5 % limit. Later logs only.", "pinned": "Share of the selected 8 experts whose score is pinned near 1; red dashed = the 5 % limit. Later logs only.", "zmax": "The largest |logit| over all layers (rank 0 first micro-batch).", "upd": "Change of the router weight over the step / its norm, mean over layers (a flag, not a criterion)."},
  "rt_h": "Original content (kept, to be cleaned up step by step): softmax or DeepSeek-V3 style (MuonH / AdamW, fitted LR law, 128-choose-8, q0 three seeds)", "rt_note": "Boxiang's controlled experiment of 10-26 (design: reports/dsv3_router_experiment_design_2026-10-26.md). The softmax + aux-loss router is replaced by the DeepSeek-V3 router (sigmoid scores + a per-expert bias for load balance + seq aux 1e-4); 128-choose-8 experts and everything else unchanged; the MuonH arms use the fitted LR law (MUONH_FIT), the AdamW arm BASE's own LR rule. Values = <b>arm − partner</b> (doc-isolated, negative = the arm is better); at q0 three seeds paired per seed, mean ± se (number of paired seeds in brackets); pre-registered lines: |Δ| &lt; 0.003 = no difference, single seed ±0.008.", "rt_cmp": "comparison", "rt_health": "router health (training log)", "rt_h_ok": "healthy", "rt_h_bad": "pathological", "rt_a": "A: DSV3 router − softmax (MuonH, AdamH gate)", "rt_b2": "B2: AdamW gate, router LR ×0.19 − A", "rt_b3": "B3: AdamH gate, router LR ×0.19 − A", "rt_c": "C: DSV3 router − softmax (AdamW baseline)", "rt_lead": "change of the MuonH lead over AdamW: (A − C) − (MUONH_FIT − BASE)", "rt_health_note": "Health from the router monitor of Boxiang (DECISIONS 10-26 19:1x-19:2x): DSV3 at q1-q3 has expert bias 3-5 (pre-registered limit 1) and runaway logits 60-87, load balanced only by the bias - not adoptable as is under the pre-registered rule; DSV3 + AdamW gate + router LR ×0.19 pins ~10 % of the selected experts' scores near 1, bias above 1, logits rising; DSV3 + router LR ×0.19 is healthy. The two pathologies have different causes: at router LR ×1 the AdamH gate of DSV3 does not learn the balance itself and the bias keeps growing, which router LR ×0.19 fixes; the two ×0.19 arms share the router LR and only the one with the AdamW gate pins the selected scores near 1, so the pinning comes from the AdamW gate itself (DECISIONS 10-26 17:29).", "rt_hr_h": "Larger rungs (seed 1)", "rt_hr_pending": "q1-q3 still training / being evaluated.", "rt_lr_h": "LR checks (q0, overall_tsqw, vs the arm's 3-seed 1× mean)", "rt_lr_head": ["arm", "Δloss per multiple", "best multiple (of the arm's law)", "1× peak LR", "the arm's LR law"], "rt_s": "S: MuonH + DSV3 router", "rt_rl": "router LR: MuonH + DSV3 router", "rt_rl_lab": "gate LR only: ×0.19 / ×0.5 / ×1 / ×2 (x axis = multiples of the main LR, ×1 = A)", "rt_rl_r": "best router LR ≈ {m:.2f}× the main LR; the current ×1 is {g:.4f} above the optimum in loss. ×1 and ×0.19 have 3 seeds, the others 1 seed (added by Runlong 10-29).", "rt_x_h": "Other ablations on the DSV3 base (added by Runlong 10-29; q0 seed 1, vs A = MUONH_FIT_DSV3; negative = better)", "rt_x_head": ["change", "Change ID", "vs A seed 1 (same data order)", "vs A 3-seed mean"], "rt_x_bias": "seq aux removed, balance by the bias alone (loss-free)", "rt_x_muonh": "gate stepped by MuonH (on the hyperball), main LR", "rt_s_lab": "LR of A ×0.7 / ×1 / ×1.4 (x axis = multiples of the ArchScale original)", "rt_sc_lab": "LR of C ×0.5 / ×1 / ×2 (x axis = multiples of the adamw_active rule)", "rt_vs": "current LR (3-seed mean)", "rt_lr_h2": "LR check: is the current LR still optimal after the switch to the DSV3 router (q0, overall_tsqw; whole-LR sweeps only, not new arms). Read as the LR grids: big number = parabola vertex m*, hover a cell for its curve; y = loss difference vs the current LR (3-seed mean)", "rt_lr_head2": ["LR check (q0)", "LR sweep", "reading", "conclusion"], "rt_s_rc": "Keep the LR: under DSV3 the fitted law stays near optimal (off by only {pct:.0f}%, {g:.4f} in loss, within noise). q0 only; larger rungs not checked.", "rt_rl_rc": "Keep router LR ×1: the optimum is about ×{m:.2f}, ×1 is only {g:.4f} worse, within noise; going down to ×0.19 is worse. q0 only.", "rt_s_r": "best {m:.2f}× (ArchScale original), current fitted law = {c:.2f}×: the optimum is {r:.2f}× the current LR and only {g:.4f} below it in loss; the current LR is near optimal.", "rt_sc_r": "best {m:.2f}× (adamw_active rule), current = {c:.2f}×: the optimum is {r:.2f}× the current LR and only {g:.4f} below it in loss; the current LR is near optimal.", "rt_sc": "S_C: AdamW + DSV3 router", "rt_s_law": "MUONH_FIT fitted law", "rt_sc_law": "AdamW adamw_active rule",
  "dg_h": "How the per-rung LR of Runlong is derived (diagonal only: the 9.6B cells of ladder q0-q6)", "dg_how": "The production LR table of Runlong (Runlong.md TODO 8 A.2, step 7; so far used only by the TODO 9 q4 pair at 5.36e-3, the rest awaits GO). Per row: <b>① the rung best multiple m* measured on its 9.6B (4,577-step) cell</b> (the diagonal above, measured by Boxiang on val_v2) × <b>② the token correction (main-ladder steps / 4,577)^0.03</b> (c = 0.03, range 0-0.06, from the token-direction fits of the LR grid; effective T exponent −0.32 + 0.03 = −0.29) = <b>③ the multiple of the ArchScale original</b>; times <b>④ the ArchScale original LR at the main-ladder tokens (TPP 200)</b> gives <b>⑤ the peak LR</b>. ⑤ matches his published numbers within 0.5 %. Last column: our own 9.6B m* (overall_tsqw), to compare with ①.", "dg_dhead": ["rung", "layers L", "main tokens T", "main steps", "① 9.6B anchor m*", "② (steps/4577)<sup>0.03</sup>", "③ multiple = ①×②", "④ ArchScale LR (main T)", "⑤ peak LR = ③×④", "published", "our 9.6B m*"], "dg_note": "Only the 7 ladder cells on the diagonal of the table above (width grows with depth, h ≈ 43·L), all at 9.6B tokens: this fits the <b>along-ladder</b> direction only - not a pure depth exponent, and no token exponent. Blue = our measured best LR (m* × the ArchScale original LR); cyan = the anchors of the per-rung production LR table of Runlong (the 9.6B m* measured by Boxiang on val_v2).", "dg_ours": "our fit (diagonal, overall_tsqw)", "dg_ours_note": " (multiple of ArchScale ∝ (L/8)^{b:+.3f})", "dg_runlong": "per-rung LR of Runlong", "dg_short": "m*<sub>9.6B</sub>(rung) = the measured best multiple of that rung on the diagonal above (no fit); 0.03 is an empirical token value. q4 already runs at 5.36e-3 by it.", "dg_runlong_note": "; anchors m* (q0-q6) = {anc}; c ≈ 0.03 [0, 0.06]; no fit: the anchors are the measured values of each rung (not adopted yet; q4 already runs at 5.36e-3)", "dg_pts": "measured best LR per cell (hollow = extrapolated)", "dg_x": "layers L (ladder rung; width grows with L)", "dg_tip_o": "{q} ours: m* {m:.2f}× → η* {v:.5f} ({mn})", "dg_tip_r": "{q} Runlong anchor: m* {m:.2f}× → η {v:.5f} (9.6B)",
  "ck_head": ["check", "control", "experiment", "reading"], "ck_note": "Each cell as in the LR grid above: one LR sweep, big number = parabola vertex m* (multiple of the ORIGINAL ArchScale law, from the actual peak LR); hover a cell for its curve.", "ck_floor1": "decay to 1% of peak (ladder)", "ck_floor10": "decay to 10% of peak (ArchScale)", "ck_split_ctl": "all parameters at the multiple", "ck_split_exp": "one group at 1.74× (diamonds)", "ck_split_a": "A: AdamW groups only (embedding, 1-D)", "ck_split_b": "B: hidden matrices + head + router", "ck_split_c": "C: hidden matrices only", "ck_batch_r": "Doubling the batch (half the steps) moves m* from 1.48 to 1.86, closest to the pre-registered law-in-steps prediction of Runlong (1.81): scale the LR by steps when the batch changes.", "ck_floor_r": "With a 10% floor the ArchScale original is about right (m* ≈ 1); the ladder decays to 1% and wants ~20% higher peak LR.", "ck_floor3_r": "Same at q3: the 10% floor moves the optimum down.", "ck_split_r": "The hidden matrices alone (C) get almost all of the gain; the embedding / 1-D groups alone (A) get nothing.",
  "ne_ch_h": "Router negative-entropy balancing (phi balancing)", "ne_h": "Full-attention base: each arm − MUONH_FIT (MuonH, softmax router, CE)", "ne_zero": "Zero line = <b>MUONH_FIT</b>: MuonH, fitted LR law (fit), plain CE (no KD), 128-choose-8 MoE, full attention, the vanilla softmax router + Switch aux loss 1e-3, AdamH gate at LR ×1.", "ne_chart_l": "DSV3 base + negative entropy − MUONH_FIT (hover a rung for values)", "ne_chart_r": "softmax base + negative entropy − MUONH_FIT (hover a rung for values)", "ne_y": "Δloss (arm − MUONH_FIT)", "ne_l_dsv3": "DSV3 (reference)", "ne_l_e6": "DSV3 + neg-entropy α 1e-6", "ne_l_e5": "DSV3 + neg-entropy α 1e-5", "ne_l_e4": "DSV3 + neg-entropy α 1e-4", "ne_l_e3": "DSV3 + neg-entropy α 1e-3", "ne_l_e5nb": "DSV3 + neg-entropy α 1e-5, bias removed", "ne_l_e4nb": "DSV3 + neg-entropy α 1e-4, bias removed", "ne_l_e5sm": "softmax + neg-entropy α 1e-5", "ne_l_e4sm": "softmax + neg-entropy α 1e-4", "nesws_h": "SWS base: each arm − DSV3 SWS", "nesws_note": "Zero line = <b>DSV3 SWS</b> (MUONH_FIT_DSV3_SWS): MuonH, fitted LR law (fit), plain CE (no KD), 128-choose-8 MoE, DSV3 router (sigmoid scores + expert bias + seq aux 1e-4, AdamH gate at LR ×1), SWA (5 of 6 layers window 2048) + attention sinks.", "nesws_skip": "α 1e-2 not drawn: q0 +0.036, diverged at q1 (+0.256).", "ns_r1": "bias kept:", "ns_r2": "variants:", "ns_e6": "α 1e-6", "ns_e5": "α 1e-5", "ns_e4": "α 1e-4", "ns_4e4": "α 4e-4", "ns_e3": "α 1e-3", "ns_e5nb": "1e-5 bias removed", "ns_e4nb": "1e-4 bias removed", "ns_e5nbf": "1e-5 bias removed + selection-frequency statistic", "ns_e5batch": "1e-5 no EMA", "ns_e5adamw": "1e-5 + AdamW gate ×0.19", "ns_e5g4": "1e-5 + bias γ 1e-4", "nesws_chart": "SWS base + negative entropy − DSV3 SWS (hover a rung for values)", "nesws_x": "Δloss (arm − DSV3 SWS)", "ne_note": "Negative-entropy balance loss: every layer keeps an EMA (η 0.65) of the router probabilities, m<sub>e</sub> = the long-run mean probability of expert e (measured over the whole global batch, updated once per step); loss = α·E·Σ<sub>e</sub> p<sub>e</sub>·(log m<sub>e</sub> + 1), p<sub>e</sub> = the mean probability the current tokens give expert e; the bracket carries no gradient, only p<sub>e</sub> does. Effect: experts used a lot over time (large m<sub>e</sub>) are pushed down, rarely used ones pulled up, driving the mean routing toward uniform (maximum entropy). It replaces the original balance loss (seq aux 1e-4 on the DSV3 base, Switch aux 1e-3 on the softmax base); the DSV3 base keeps its expert bias (except the arms marked 'bias removed'). Unlike the aux losses it is measured over the whole global batch and smoothed, and uses probabilities rather than dispatched-token counts, so it does not depend on the micro-batch size. (From arXiv 2605.15403)", "ne_base": "base", "ne_pair": "pair", "ne_dsv3": "DSV3 base (bias kept)", "ne_nobias": "DSV3 base, no bias", "ne_soft": "softmax base (paper-faithful)", "ne_sws": ", SWS base (DSV3 + SWA + sink)", "ne_noema": "no EMA (η 1)",
  "r0_h": "Router choice", "r0_note": "Zero line = <b>MUONH_FIT</b>: MuonH, fitted LR law (fit, q0 peak 0.0164), <b>plain CE (no KD)</b>, 128-choose-8 MoE, <b>full attention</b> (no SWA / sink), the vanilla softmax router + Switch aux loss 1e-3, AdamH gate at LR ×1.", "r0_chart": "Each row − MUONH_FIT (further left is better). Filled dot = mean; with several seeds the hollow dots are the per-seed paired differences; hover a dot for values and Change ID", "r0_x": "Δloss (row − MUONH_FIT)", "r0_g_score": "① Score function: softmax → DSV3 style", "r0_dsv3": "DSV3 style: sigmoid scores + expert bias + seq aux 1e-4", "r0_g_bal": "② DSV3 style + another load balancing", "r0_biasonly": "bias only (seq aux removed)", "r0_g_opt": "③ DSV3 style + gate optimizer / LR", "r0_rt019": "router LR ×0.19", "r0_rt05": "router LR ×0.5", "r0_rt2": "router LR ×2", "r0_rtmuonh": "gate stepped by MuonH", "r0_rtadamw": "gate stepped by AdamW, LR ×0.19", "r0_g_ne": "④ DSV3 style, seq aux replaced by the negative-entropy balance loss α (Runlong TODO 9)", "r0_ne5nb": "α 1e-5, bias removed too", "r0_ne4nb": "α 1e-4, bias removed too", "r0_g_nesoft": "⑤ softmax kept, aux loss replaced by the negative-entropy balance loss α", "sa_chart": "Each arm − DSV3 SWS (hover a rung for all arms)", "sa_biasonly_d": "seq aux 1e-4 removed; the expert bias alone balances (DeepSeek-V3's aux-loss-free setting).", "sa_nobias_d": "expert bias off; seq aux 1e-4 alone balances (the mirror of the row above).", "sa_nobal_d": "bias and aux both off, no balancing at all: the lower reference, dead experts expected.", "sa_auxbatch_d": "the per-sequence aux replaced by a Switch aux over the micro-batch (32K tokens), still 1e-4; bias kept. Tests the statistic's scope (literature: micro-batch worst, sequence next, global batch best).", "sa_aux1em3_d": "seq aux ×10 (1e-4 → 1e-3), bias kept.", "sa_g1em4_d": "expert-bias step γ 1e-3 → 1e-4 (10× slower).", "sa_g1em2_d": "γ 1e-3 → 1e-2 (10× faster; DeepSeek-V3 also sweeps γ).", "sa_softmax_d": "scores softmax instead of sigmoid (renormalised over the top 8), bias and seq aux kept (LongCat-style softmax + bias).", "sa_rs_d": "the weights of the 8 selected experts (summing to 1) are multiplied by 2.5, i.e. the routed output is scaled 2.5× (DeepSeek-V3's value); no shared expert.", "sa_shared_d": "one shared expert per layer: every token goes through it, width of one routed expert (3h/8, 192 at q0); gate scaling 1. <b>Not iso-parameter</b>: about +7-8 % active parameters.", "sa_shared_rs_d": "both of the above: one shared expert + routed scaling 2.5, i.e. the full DeepSeek-V3 MoE package. <b>Not iso-parameter</b>: about +7-8 % active parameters.", "sa_zloss_d": "router z-loss 1e-3 (ST-MoE / OLMoE): penalises the squared log-sum-exp of the router logits to keep them small; nothing else changed.", "sa_rt019_d": "AdamH gate at 0.19× the main LR.", "sa_rt05_d": "AdamH gate at 0.5× the main LR.", "sa_rt2_d": "AdamH gate at 2× the main LR.", "sa_rtmuonh_d": "gate stepped by MuonH (on the hyperball) at the main LR; GLM-4.5 is the one open model with Muon on the router.", "sa_rtadamw_d": "gate on plain AdamW (wd 0) at 0.19× the main LR (the AdamW law's router LR): the full-attention B2 setting moved to the SWS base.", "sa_thead": ["group", "line", "name", "what changes (vs DSV3 SWS)", "rungs with data"], "sa_table_h": "What each line is", "sa_none": "training / being evaluated", "sa_r_bal": "balancing:", "sa_r_gamma": "bias γ / scores:", "sa_r_scale": "scaling / shared / regulariser:", "sa_r_opt": "gate optimizer / LR:", "sa_biasonly": "seq aux removed (bias only)", "sa_nobias": "bias removed (seq aux only)", "sa_nobal": "both removed", "sa_auxbatch": "aux per micro-batch", "sa_aux1em3": "seq aux ×10", "sa_g1em4": "γ 1e-4", "sa_g1em2": "γ 1e-2", "sa_softmax": "softmax scores", "sa_rs": "routed scaling 2.5", "sa_shared": "+1 shared expert", "sa_shared_rs": "shared + scaling 2.5", "sa_zloss": "z-loss 1e-3", "sa_rt019": "router LR ×0.19", "sa_rt05": "router LR ×0.5", "sa_rt2": "router LR ×2", "sa_rtmuonh": "MuonH gate", "sa_rtadamw": "AdamW gate ×0.19", "lat_h": "Latent MoE", "lat_note": "Zero line of the first two lines = <b>PROD1_DPSK</b> (PROD v1): SWA 512 local:global 5:1 + sinks, DSV3 router, dense layer 0, 8 + 1 shared expert, 32K training + inter-document mask, MuonH, pure CE<br>Latent MoE (the Kimi K3 form): the MoE projects h down to d = h/2, the routed experts work at d, RMSNorm, then project back to h; router and shared expert stay at h.<br>Values = line − its reference (see the Change ID column of the table below: PROD1_DPSK for lines 1, 2 and 4, PROD1_DPSK_LATENT2 for line 3; our 115-split doc-isolated eval, seed 1). All these runs are trained at 32K: each checkpoint is loaded once and the same job scores both the 8K and the 32K window (tabs above the chart).", "lat_chart": "Each line − its reference (more negative is better; numbers in the table under the chart)", "x_pending": "Evaluating: one job per checkpoint (loaded once) scores both the 8K and the 32K window; drawn automatically when the results land.", "lat_l_l2": "latent MoE (iso-parameter)", "lat_l_l2_d": "PROD1_DPSK_LATENT2: d = h/2, 256 routed experts (width 3h/8), top-16; total parameters, routed FLOPs and all-to-all equal to the zero line (plus two projections, ~8 % of a layer)", "lat_l_e2x": "latent MoE, experts ×2", "lat_l_e2x_d": "PROD1_DPSK_LATENT2_E2X: the line above with 512 routed experts, still top-16. Same compute per token, but routed parameters ×2 (<b>not iso-parameter</b>)", "lat_l_x2l": "experts ×2 (with latent)", "lat_l_x2l_d": "PROD1_DPSK_LATENT2_E2X − <b>PROD1_DPSK_LATENT2</b> (a different zero line): both latent, routed experts 256 → 512, still top-16: the effect of experts ×2 alone", "lat_l_x2": "experts ×2 (no latent)", "lat_l_x2_d": "PROD1_DPSK_E2X − PROD1_DPSK: no latent, routed experts 128 → 256, still top-8; same compute per token, routed parameters ×2. Against the line above: experts ×2 with and without latent", "rtq_h": "Router / load balancing on E2X (Runlong's arms)", "rtq_note": "Zero line = <b>PROD1_DPSK_LATENT2_E2X</b> (latent d = h/2, 512 routed experts top-16 + 1 shared; sigmoid router + expert bias (update rate 1e-3) + seq aux 1e-4; router on AdamH). Each line changes one thing in the router; everything else (iterations, LR, data, seed 1) equals the zero line.<br>Values = line − zero line (our 115-split doc-isolated eval; each checkpoint loaded once, scored at the 8K and 32K windows). These runs started 11-05 / 11-06 and have no q0; each checkpoint is evaluated and drawn automatically when it finishes.", "rtq_chart": "Each line − PROD1_DPSK_LATENT2_E2X (more negative is better; hover a rung for values)", "rtq_l_base": "zero line: PROD1_DPSK_LATENT2_E2X", "rtq_l_base_d": "bias + seq aux 1e-4", "rtq_l_qb": "QB: Quantile Balancing", "rtq_l_qb_d": "Megatron built-in (<code>--moe-router-load-balancing-type quantile_balancing</code>), replacing bias + seq aux: a per-expert bias (qb_beta) updated by dual coordinate descent. Per micro-batch, the quantile of each expert's routing margins admits exactly T·k/E tokens; averaged over micro-batches and DP ranks, recentred, updated once per global batch and used to route the next batch (frozen at eval). Aux loss and expert bias off", "rtq_l_qbx": "QBX: exact global quantile (the Kimi K3 form)", "rtq_l_qbx_d": "As QB, but the quantile is exact over the whole global batch (per-expert margin histograms summed over micro-batches and DP ranks, the (C+1)-th largest margin), not an average; patched Megatron", "rtq_l_bo": "expert bias only (no seq aux)", "rtq_l_bo_d": "seq aux 1e-4 removed, the expert bias is the only balancer; the clean control for QB (neither has an aux loss)", "rtq_l_g5": "bias update rate ×0.5", "rtq_l_g5_d": "expert-bias update rate 1e-3 → 5e-4 (is the bias step too coarse for 512 experts and overshooting?)", "rtq_l_g25": "bias update rate ×0.25", "rtq_l_g25_d": "expert-bias update rate 1e-3 → 2.5e-4", "rtq_l_adamw": "router on AdamW (LR ×0.19)", "rtq_l_adamw_d": "router from AdamH (Adam + one hyperball sphere per router matrix) to plain AdamW at router LR ×0.19: does dropping the matrix-level norm constraint shrink the cold-expert set?", "rtq_l_pin": "router rows as separate hyperballs (pin)", "rtq_l_pin_d": "every router row (one per expert) is its own hyperball, retracted onto its own initial norm, so no row takes norm from another; targets cold experts (shrunk, aligned router rows)", "rtq_tl_h": "Training loss (training logs, early read)", "rtq_tl_note": "Training lm loss, line − the E2X zero line: at the line's latest logged step S, each side averaged over (S − 500, S], then subtracted (32K training loss, not an eval; single seed, noise ~±0.002; in the first 1-2k steps the loss falls fast and the gap swings much more, not a result). Runs still training update whenever the report is regenerated; the doc evals above are the result.", "rtq_rt_h": "Load balance during training (router health, training logs)", "rtq_rt_note": "Router health from the training logs (one line every 100 steps, all layers); runs still training update with the report. Line colours match the eval chart above, grey dashed = the E2X zero line; hover for names and values. Dead experts in % of (layers × 512). MaxVio 0 = perfectly balanced.", "x_thead": ["line", "name", "change", "rungs with data"], "x_tab_h": "Lines (all minus the zero line PROD1_DPSK)", "rope_h": "Position encoding: NoPE / RoPE / partial RoPE", "rope_note": "Zero line = <b>PROD1_DPSK</b> (PROD v1): full RoPE on the SWA layers (all 128 dims rotate), NoPE on the global layers (1 in 6); otherwise as the previous chapter's zero line<br>Partial RoPE: only 32 of each head's 128 dims (1/4) rotate (NeoX layout, base 1e4); the other 96 carry no position.<br>Values = line − PROD1_DPSK (our 115-split doc-isolated eval, seed 1); 8K / 32K windows as in the previous chapter.", "rope_chart": "Each line − PROD1_DPSK (more negative is better; hover a rung for values)", "rope_l_p25": "SWA layers 1/4 RoPE", "rope_l_p25_d": "PROD1_DPSK_PROPE25: SWA layers only, RoPE 1 → 1/4; global layers stay NoPE (single variable)", "rope_l_p25g": "all layers 1/4 RoPE", "rope_l_p25g_d": "PROD1_DPSK_PROPE25G: SWA layers RoPE 1 → 1/4 and global layers NoPE → 1/4 RoPE (two changes; minus the line above = global layers NoPE vs 1/4 RoPE)", "rope_sm_h": "Another pair: global layers RoPE vs NoPE on the softmax base (8K-trained, 8K window only; different zero line, not in the chart)", "rope_sm_row": "SWA window 2048, local:global 5:1 (no sinks), global layers NoPE → full RoPE", "sws_h": "Router choice, SWS architecture, miscellaneous experiments", "sws_note": "SWS = MUONH_FIT_DSV3_SWS: the DeepSeek-V3 router (sigmoid + expert bias γ 1e-3 + seq aux 1e-4, AdamH gate ×1) on SWA (5 of 6 layers window 2048, NoPE global layers) + learnable attention sinks, the base of the router ablations from the evening of 10-29. Each arm changes one router item; values = <b>arm − SWS</b> (doc-isolated overall_tsqw, paired per seed; negative = better). Grey band = the pre-registered no-difference zone |Δ| &lt; 0.003; single-seed noise about ±0.008.", "sws_chart": "Each arm − the SWS base (overall_tsqw, further left is better). Filled dot = mean; with several seeds the hollow dots are the per-seed paired differences; hover a dot for values and Change ID", "sws_x": "Δloss (arm − SWS)", "sws_pending": "Still training / being evaluated: ", "sws_g_bal": "Load-balancing mechanism", "sws_biasonly": "bias only (seq aux removed)", "sws_nobias": "seq aux only (bias removed)", "sws_nobal": "neither (no balancing, lower reference)", "sws_auxbatch": "aux over the micro-batch", "sws_aux1em3": "seq aux ×10 (1e-3)", "sws_g_gamma": "Bias update rate γ (current 1e-3)", "sws_g1em4": "γ 1e-4 (×0.1)", "sws_g1em2": "γ 1e-2 (×10)", "sws_g_score": "Score function", "sws_softmax": "softmax scores + bias + seq aux", "sws_g_scale": "Gate scaling / shared expert", "sws_rs": "routed scaling 2.5 (DSV3 value)", "sws_shared": "one shared expert (not iso-param)", "sws_shared_rs": "shared expert + scaling 2.5 (full DSV3)", "sws_g_reg": "Router regulariser", "sws_zloss": "z-loss 1e-3", "sws_g_opt": "Gate optimizer / LR", "sws_rt019": "router LR ×0.19", "sws_rt05": "router LR ×0.5", "sws_rt2": "router LR ×2", "sws_rtmuonh": "gate stepped by MuonH", "sws_rtadamw": "gate stepped by AdamW, LR ×0.19", "sws_g_ne": "Negative-entropy balance loss (TODO 9)", "sws_ne6": "α 1e-6", "sws_ne5": "α 1e-5", "sws_ne4": "α 1e-4", "sws_ne4e4": "α 4e-4", "sws_ne5nb": "α 1e-5, bias removed", "sws_ne4nb": "α 1e-4, bias removed", "sws_ne3": "α 1e-3", "sws_ne2": "α 1e-2", "sws_ne5batch": "α 1e-5, EMA removed (fresh per micro-batch)", "sws_ne5adamw": "α 1e-5, gate AdamW ×0.19", "sws_ne5g4": "α 1e-5, bias update rate γ 1e-4", "sws_ne5nbfreq": "α 1e-5, bias removed, statistic = selection frequency (freq, was prob)", "kd25_h": "Teacher MiMo-2.6 vs. MiMo-2.5", "kd25_note": "Zero line = <b>one CE: OPT_MUONH</b> (MuonH old LR, 650B data; q0 = mean of its 3 seeds); all three families are minus this same CE. Each point = the KD arm's seed mean − OPT_MUONH; the 3 q0 seeds are averaged into one point. <br>Green vs orange: same LR, 45B KD corpus v1 (tc_small cache), seeds and data order, only the teacher differs, so their distance = 2.5 − 2.6. Blue vs green: both MiMo-2.6, but LR and KD data both differ (blue uses the 650B data), so their distance is not the LR effect alone. Blue S128Y has q3 only.", "kd25_pnote": "These are Boxiang-style numbers: the packed validation at the end of training (6 decon sets, mean6 = their plain mean), from the training logs.", "kd25_nold": "MiMo-2.6 old LR {v}", "kd25_n26": "MiMo-2.6 new LR {v}", "kd25_n25": "MiMo-2.5 new LR {v}", "kd25_y": "Δloss (KD − CE)", "kd25_chart": "Each KD arm − CE (hover a rung for all lines)", "kd25_fold": "<b>Blue family</b>: MiMo-2.6 teacher, α 0.8, MuonH <b>old LR</b> (original ArchScale law), 650B data — S64 / S64Y / S128Y (light to dark; S128Y q3 only)", "kd25_f26": "<b>Green family</b>: MiMo-2.6 teacher, α 0.8, MuonH <b>new LR law</b> (fit), tc_small 45B — S64 / S64Y / S128Y (light to dark)", "kd25_f25": "<b>Orange family</b>: MiMo-2.5 teacher, α 0.8, MuonH <b>new LR law</b> (fit), tc_small 45B — S64 / S64Y / S128Y (light to dark)", "kd25_f0": "y = KD − CE: OPT_MUONH (MuonH old LR); all three families minus this same CE; y range = current data min to max. The green / orange (45B corpus) q3 runs used TPP 100, not comparable with the other points, and are not drawn.", "dm_h": "Document mask: mask vs no mask", "dm_note": "Zero line = <b>OPT_MUONH</b>: pure CE, MuonH old LR<br>All three lines are KD MiMo-2.6 teacher, α 0.8, S64Y, the same KD data (650B), softmax router, no SWA. <b>Mask</b> = each document in a packed 8K sequence attends only to itself (Megatron --dataloader-inter-document-masking). Grey and purple are our pair, both at the fitted MuonH LR law, differing only in the mask: their distance = the effect of the mask. Blue is Boxiang's original run at the old LR (original ArchScale law), no mask: blue and grey differ only in the LR. Each point = seed mean − OPT_MUONH (where q0 has 3 seeds they are averaged into one point). Note: the eval is doc-isolated (every document scored alone), the same condition the masked run trains under.", "dm_chart": "Each line − CE (hover a rung for all values)", "dm_nb": "no mask, old LR (Boxiang)", "dm_nb_d": "<b>No mask, old LR</b> (Boxiang, KD_MUONH_A0P8_S64Y): MuonH old LR (original ArchScale law), q0-q3, 1 seed each", "dm_n0": "no mask, new LR", "dm_n1": "mask, new LR", "dm_n0_d": "<b>No mask, new LR</b> (ours, KD_MUONH_A0P8_S64Y_FIT): MuonH fitted LR law; a document in a packed 8K sequence sees the earlier documents", "dm_n1_d": "<b>Mask, new LR</b> (ours, KD_MUONH_A0P8_S64Y_FIT_DOCMASK): MuonH fitted LR law; each document attends only to itself", "dm_f0": "y = line − CE: OPT_MUONH (MuonH old LR); all three lines minus this same CE; y range = current data min to max", "kd25_pend": "(training / being evaluated)", "kd25_table_h": "Numbers ({m}; negative = better than CE; 2.5 − 2.6 negative = 2.5 better)", "kd25_thead": ["rung", "variant", "2.6 − CE", "2.5 − CE", "2.5 − 2.6", "seeds (2.6 / 2.5 / CE)"], "rt_two_h": "The two routers", "rt_two": "<p><b>softmax router</b></p><ul><li>Softmax over the logits of the 128 experts; the 8 highest scores are picked.</li><li>Load balance by an aux loss (Switch style, coefficient 1e-3): when some experts are picked too often, an extra penalty pushes the router to spread the tokens.<ul><li><b>How it works</b>: in every MoE layer and every micro-batch (32,768 tokens), two things are measured: f<sub>i</sub> = the fraction of tokens actually sent to expert i, P<sub>i</sub> = the mean softmax probability these tokens give expert i. aux = 1e-3 × 128 × Σ<sub>i</sub> f<sub>i</sub>·P<sub>i</sub>, in effect added to the training loss and back-propagated with it. f<sub>i</sub> is only a count with no gradient; the gradient flows through P<sub>i</sub>: the more tokens an expert gets (larger f<sub>i</sub>), the more its probability is pushed down.</li></ul></li><li>The micro-batch size (set in the ladder's per-rung layout table, chosen for memory) decides how many tokens each aux statistic covers:<table style='font-size:12px;margin:4px 0'><thead><tr><th>rung</th><th>nodes</th><th>micro-batch</th><th>tokens per aux statistic</th><th>EP</th></tr></thead><tbody><tr><td>q0</td><td>4</td><td>4</td><td>32,768</td><td>1</td></tr><tr><td>q1</td><td>8</td><td>4</td><td>32,768</td><td>1</td></tr><tr><td>q2</td><td>16</td><td>4</td><td>32,768</td><td>1</td></tr><tr><td>q3</td><td>16</td><td>4</td><td>32,768</td><td>4</td></tr><tr><td>q4</td><td>32</td><td>2</td><td>16,384</td><td>8</td></tr><tr><td>q5</td><td>64</td><td>1</td><td>8,192</td><td>8</td></tr><tr><td>q6</td><td>64</td><td>1</td><td>8,192</td><td>8</td></tr></tbody></table><ul><li><b>EP (expert parallelism)</b>: the 128 experts are stored across several GPUs, each holding 128 ÷ EP of them (q3: 4 GPUs with 32 each; q4-q6: 8 GPUs with 16 each). In every layer the router first picks the 8 experts for the tokens on its own GPU, then an all-to-all sends each token to the GPU holding its experts, which computes it and sends the result back. It only saves memory and does not change the result; the routing choice and the aux statistics are done on the GPU's own micro-batch before the tokens are sent, so it does not affect the aux loss either.</li></ul></li></ul><p><b>DSV3 router</b></p><ul><li>Each expert score goes through its own sigmoid, each in 0-1, not competing with the others.</li><li>Experts are picked by “score + bias”. Every expert has a bias that is not a trained parameter but adjusted by a rule every step: an overloaded expert gets bias − 0.001, an underloaded one + 0.001.</li><li><b>The bias only decides which experts are picked, not their output weights</b>: after the 8 are picked, their weights are their original sigmoid scores (without the bias), normalised over those 8 (Megatron <code>moe_utils.py</code> L953-959).</li><li>A small aux loss is kept as well (seq aux 1e-4), computed per sequence.<ul><li><b>How it works</b>: the same formula as the Switch aux above, but measured over <b>each 8K training sequence</b> (one packed row, possibly several documents), with a 10× smaller coefficient. Per sequence: f<sub>i</sub> = the fraction of its tokens sent to expert i, P<sub>i</sub> = the mean score its tokens give expert i (each token's 128 sigmoid scores first normalised to sum to 1). aux = 1e-4 × 128 × Σ<sub>i</sub> f<sub>i</sub>·P<sub>i</sub>, computed for each of the 4 sequences of a micro-batch and averaged; again only P<sub>i</sub> carries a gradient. Being per sequence, it does not depend on the micro-batch size.</li></ul></li><li>Design idea: balance mainly by the bias and hardly by an aux loss, so the aux loss does not interfere with the main task.</li></ul>",
  "rt_gate_h": "Which optimizer and LR for the gate (router weights)", "rt_gate": "<p>The gate is the weight matrix W<sub>g</sub> (128 × width) that computes the logits.</p><ul><li><b>AdamH</b>: the MuonH default - Adam updates, but the weights are kept on a sphere of fixed radius (hyperball); the weight norm cannot grow.</li><li><b>AdamW</b>: plain AdamW; the weight norm is free (wd 0 in B2).</li><li><b>router LR ×1</b>: the gate uses the same LR as the body (0.0164 at q0 under the fitted law).</li><li><b>router LR ×0.19</b>: the gate LR is lowered to ~3.1e-3, about the AdamW baseline own LR level. The earlier probe B1 showed that an AdamW gate at ×1 saturates the logits, hence 0.19.</li></ul>",
  "arch_thead": ["change", "Change ID", "what changes", "control", "LR law (from the actual peak LR)", "results at"], "arch_law": {"fit": "fitted law (fit v2)", "arch": "old law (ArchScale original)", "other": "other"},
  "arch_short": {"MUONH_FIT_DSV3_SWS": "SWA + sink (DSV3 base, SWS)", "SWA_FIT_SINK_W512": "SWA + sink + window 512", "SWA_FIT_SINK_W128": "SWA + sink + window 128", "SWA_FIT_SINK_W512_R3": "SWA + sink + window 512, 3:1 (bottom, no q3)", "SWA_FIT_SINK_W512_R3T": "SWA + sink + window 512, 3:1 (top)", "SWA_FIT": "SWA (base)", "SWA_FIT_ROPE": "SWA, RoPE in the full-attention layers", "SWA_FIT_W4K": "SWA, window 4096", "SWA_FIT_W1K": "SWA, window 1024", "SWA_FIT_SINK": "SWA + attention sinks", "SWA_FIT_SINK_W1K": "SWA + sinks + window 1024", "SWA_FIT_DSV3": "SWA (DSV3-router base)", "MUONH_SPLITFC1": "MuonH split fc1"},
  "arch_desc": {"MUONH_FIT_DSV3_SWS": "SWA (window 2048, NoPE global layers) + learnable attention sinks on the DSV3 router base; the new base of the router ablations from 10-29 (SWS)", "SWA_FIT_SINK_W512": "attention sink + window 512 (further down the window axis)", "SWA_FIT_SINK_W128": "attention sink + window 128 (further down the window axis)", "SWA_FIT_SINK_W512_R3": "as window 512 + sinks, but one full-attention (NoPE) layer in every 4: local:global 3:1 instead of 5:1 (the ratio axis; Boxiang 10-30)", "SWA_FIT_SINK_W512_R3T": "as 3:1, but the global layers counted down from the top (the last layer is always global, as in 5:1); R3 counts from the bottom, so at some depths its top two layers are SWA. At q0 (12 layers) and q2 (24 layers) the depth is a multiple of 4 and both countings give the same layout, so both lines use the same run (the q2 argv equals R3's q2 golden item by item); they differ only at q1 and q3, and R3 q3 was never trained", "SWA_FIT": "5 of every 6 layers sliding-window attention (window 2048, RoPE) + 1 full-attention layer (NoPE)", "SWA_FIT_ROPE": "as the base, but RoPE in the full-attention layers too", "SWA_FIT_W4K": "as the base, window 4096", "SWA_FIT_W1K": "as the base, window 1024", "SWA_FIT_SINK": "as the base + learnable attention sinks (gpt-oss style: one learnable sink logit per query head per layer)", "SWA_FIT_SINK_W1K": "attention sinks + window 1024", "SWA_FIT_DSV3": "base SWA on the DeepSeek-V3-style router base (control on the DSV3 base too)", "MUONH_SPLITFC1": "optimizer change (not attention): the expert fc1 is stepped as two matrices, gate and up, each with its own Newton-Schulz, scale and sphere radius (was one concatenated matrix)"},
  "rt_left_h": "MuonH body: each arm − control (softmax router, MUONH_FIT)", "rt_right_h": "AdamW body: C − control 2 (softmax router, BASE)", "rt_l_a": "DSV3", "rt_l_biasonly": "DSV3 without seq aux", "rt_l_rtmuonh": "DSV3 + MuonH gate", "rt_l_b2": "DSV3 + AdamW gate + router LR ×0.19", "rt_l_b3": "DSV3 + router LR ×0.19", "rt_l_c": "C: DSV3 router",
  "ol_h": "AdamW LR check (3 points only, not a grid; not in the plan)", "ol_opt": "optimizer", "ol_note": "AdamW and Muon both use the AdamW LR law; it was checked only at a few rungs at 0.5× / 1× / 2× (TODO 3 step 5, fairness), to set against the MuonH grid above. Values = {m} difference to 1× (negative = better than 1×); m* = best multiple of the 3-point parabola, η* = the absolute LR.",
  "lf_h": "Fit: best LR vs depth and training tokens (ArchScale form)", "lf_arch": "ArchScale original (= every cell's 1×, the LR the ladder uses)", "lf_ours": "Our fit", "lf_fix": "our fit, T exponent fixed at −0.32", "lf_box": "Boxiang's fit v2 (frozen 10-25, used by MUONH_FIT; val_v2, 16 cells)", "lf_pts": "measured best LR per cell (hollow = extrapolated)", "lf_drag": "drag to rotate · hover a point", "lf_axL": "layers L", "lf_axT": "training tokens T", "lf_note": "Least squares in log space over the best absolute LR of {n} grid cells: log η* = log η0 + β·log(L/8) + γ·log(T/10.4B). The optimum is the parabola vertex; cells whose vertex lies outside the sampled multiples ({excl}) use the extrapolated vertex and are included, marked as extrapolated; vertices more than 2× beyond the sweep are not used. ± = standard error.",
  "lf_cells": "Per cell: measured vs fitted", "lf_meas": "measured", "lf_fitted": "fitted", "lf_res": "residual", "lf_ext": "Extrapolation (indicative only)", "lf_ratio": "fitted / ArchScale",
  "lf_pred": ["q2 main (TPP 200)", "q3 main (TPP 200)", "q6 (30B-A3B)", "235B target"],
  "lf_warn": "q2 / q3 main are predictions near the covered range; q6 and 235B are far outside it (at most 30 layers and 86B tokens fitted) - direction only, not for use as is. One seed; the depth exponent differs between the 9.6B and 30.8B columns (about −0.15 vs −0.06), so separability needs more cells.",
  "tab_doc": "doc-isolated", "tab_docold": "Older experiments", "tab_ref": "External references", "tab_down": "59B downstream",
  "ce_lack": "External models not evaluated with this list yet (they only have scores of the earlier nemo6 protocol and are not in the tables below): {names}.",
  "ce_ours": "Our own checkpoints are evaluated with the same list; their scores are in the \"prod downstream\" tab.",
  "down_old": "{n} checkpoints that only have scores of the earlier protocol (nemo6): they appear in the list above once evaluated with this task list - click to expand",
  "tab_fit": "LR fitting", "fit_tab_note": "Best-LR sweeps and fits (MuonH grid, Muon grid, depth × width assumption check). Same data as the doc-isolated tab (our eval).", "tab_packed": "packed (Boxiang's training-time validation)",
  "p_notes": ["From Boxiang's <code>results/&lt;Change ID&gt;/scores.json</code>: the <b>packed</b> final validation of each training run (documents concatenated, no attention mask); we do not re-measure it.",
              "Only his 6 pooled sets (val_v2 not split, no private_internal); “mean of 6” = the simple mean of those 6 (his definition).",
              "Packed is ~0.013–0.016 above doc-isolated: <b>do not compare absolute values across the two tabs</b>, only deltas within a tab."],
  "tiles": ["Overall loss reduction from KD at the largest rung", "Equivalent CE model size at the largest rung", "KD loss reduction on private_internal", "Largest gap between the three alphas"],
  "tile_sub": ["{rung}, α 0.8, vs CE (MuonH)", "KD {rung} ≈ a CE model with {mult}× the active params", "{rung}, α 0.8; a private set outside the training corpus", "same rung and metric; noise ≈ 0.008"],
  "d0_h": "Dense layer 0 (the first layer is not MoE)", "d0_note": "The MoE of layer 0 is replaced by a plain dense FFN (SwiGLU, width 8 × expert); all other layers unchanged; the dense arms are built on DSV3 SWS.<br>Zero line = <b>MUONH_FIT</b>: MuonH, fitted LR law (fit), plain CE (no KD), 128-choose-8 MoE, full attention, the vanilla softmax router. Values = line − MUONH_FIT (doc-isolated, paired per seed, q0 = mean of 3 seeds; negative = better).<br><b>Negative-entropy balancing φ (phi balancing), coefficient α = 1e-5</b>: every layer keeps an EMA of the router probabilities, m<sub>e</sub> = the long-run mean probability of expert e; loss = α·E·Σ<sub>e</sub> p<sub>e</sub>·(log m<sub>e</sub> + 1), p<sub>e</sub> = the mean probability the current tokens give expert e (the bracket carries no gradient). Experts used a lot over time are pushed down and rarely used ones pulled up, driving the routing toward uniform (maximum entropy). Two uses (components in the table below): <b>bias + neg. entropy</b> = DSV3's expert bias (γ = 1e-3) kept, only the seq aux replaced by negative entropy; <b>neg. entropy only</b> = expert bias and seq aux both removed, this term alone. It is measured over the whole global batch, so it does not depend on the micro-batch size (from arXiv 2605.15403; see the 'Router negative-entropy balancing (phi balancing)' chapter above).<br><b>Verdict: not adopted.</b> The loss is only slightly below DSV3 (the two negative-entropy solid lines sit 0.002–0.006 below the blue solid line), within single-seed noise (about ±0.008); in the full PROD configuration there is almost no difference (PROD1_PHI − PROD1_DPSK: q0 0.000, q1 −0.001, q2 −0.002); with negative entropy only the router is unhealthy (router-health charts below). Runlong made DSV3 the default on 11-03 and stopped all negative-entropy runs.", "d0_chart": "Each line − MUONH_FIT; dashed = all-MoE references (no dense layer), solid = dense layer 0 (hover a rung for all lines)", "d0_y": "Δloss (line − MUONH_FIT)", "d0_l_dsv3": "DSV3 · all-MoE · full attention", "d0_l_sws": "DSV3 · all-MoE", "d0_l_d1": "DSV3 · dense layer 0", "d0_l_d2": "DSV3 · dense layers 0-1", "d0_l_fa0": "DSV3 · dense layer 0 (full attention there)", "d0_l_phib_moe": "bias + neg. entropy · all-MoE", "d0_l_phinb_moe": "neg. entropy only · all-MoE", "d0_l_phib": "bias + neg. entropy · dense layer 0", "d0_ltab": {"head": ["Line", "Attention", "Layer 0", "sigmoid scores, top-8, normalised weights", "expert bias (γ = 1e-3, adjusted every step by load)", "seq aux (α = 1e-4)", "negative entropy φ (α = 1e-5)", "Change ID"], "rows": {"d0_l_dsv3": ["full attention", "MoE", True, True, True, False], "d0_l_sws": ["SWS", "MoE", True, True, True, False], "d0_l_d1": ["SWS", "<b>dense</b>", True, True, True, False], "d0_l_d2": ["SWS", "<b>dense</b> (layer 1 dense too)", True, True, True, False], "d0_l_fa0": ["SWS; layer 0 full attention (NoPE)", "<b>dense</b>", True, True, True, False], "d0_l_phib_moe": ["SWS", "MoE", True, True, False, True], "d0_l_phib": ["SWS", "<b>dense</b>", True, True, False, True], "d0_l_phinb_moe": ["SWS", "MoE", True, False, False, True], "d0_l_phinb": ["SWS", "<b>dense</b>", True, False, False, True]}, "foot": "Three routers: <b>DSV3</b> = expert bias + seq aux; <b>bias + neg. entropy</b> = DSV3 with its seq aux replaced by negative entropy; <b>neg. entropy only</b> = neither expert bias nor seq aux (only the sigmoid scores remain; no longer DSV3). SWS = SWA window 2048, local:global 5:1, attention sinks. Common to all lines: MuonH, fitted LR, CE; 128 experts top-8 (no shared); dense layer = SwiGLU, width 8 × expert (same as a MoE layer's active width); seed 1, same data order. A router's all-MoE (dashed) and dense-layer-0 (solid) lines share a colour; the router-health charts below use the same colours and line styles."}, "d0_l_phinb": "neg. entropy only · dense layer 0", "d0_missing": "No eval of ours yet, so not drawn: ", "d0_proc_h": "Process", "d0_steps": ["<b>Why (11-01)</b>: the extra dead experts of the negative-entropy no-bias arm sat mostly in the <b>first MoE layer</b> (q3: 32 of its 60 dead experts; that layer's top 8 experts took ~40 % of the tokens). External references: DeepSeek-V3 starts with 3 dense layers, Kimi K2 / K3 with 1, MiMo V2 Flash / V2.5 / V2.6 Pro with 1, Flint with 2; Qwen3 and MAI-Base-2 are all-MoE.", "<b>Design (Runlong 11-01 15:4x)</b>: layer 0 dense (DENSE1) and layers 0-1 dense (DENSE2), each with three routers: DSV3 (control), negative entropy + bias, negative entropy without bias; q1 / q2 / q3, one seed each, 18 runs. Dense width = 8 × the expert width = 3h (q1 2304 / q2 3072 / q3 3840), i.e. the FFN width a token actually uses in an MoE layer (as DSV3 / Kimi / MiMo); attention unchanged, layer 0 stays SWA. Base = DSV3 SWS.", "<b>One layer or two (11-02 05:2x)</b>: pre-set rule (Runlong 11-01 21:1x, “launch only the better one”): mean of DENSE2 − DENSE1 over q1, q2 and the three routers; |mean| ≥ 0.002 with the q2 legs mostly the same sign → the lower-loss one. Result DENSE2 − DENSE1 mean +0.0025 (six-set +0.0030), q2 legs all the same sign → <b>one layer</b>; q4 runs only the three DENSE1 arms.", "<b>Layer 0 attention, SWA or full (11-02 11:0x-12:xx)</b>: Runlong first set “a dense first layer must not be SWA either” (MiMo: layer 0 = global attention + dense; Flint: layers 0-1 dense + full), then turned it into an ablation: “first layer SWA or full; SWA means RoPE, full means NoPE”. FA0 = layer 0 full attention + NoPE (one global layer added on top of the existing ones). Result: FA0 is <b>worse by 0.006 (q1) and 0.007 (q2)</b>, router health no better → PROD v1 keeps SWA on layer 0.", "<b>Wrap-up (11-03 10:5x)</b>: after the router review the DSV3 router became the production default; Runlong: “stop all phi” → the two negative-entropy dense q4 runs were stopped (checkpoints kept). The DSV3 + dense-layer-0 q4 run is queued again on slurm-2; the FA0 q3 is being evaluated."], "d0_res_h": "Results: change − control (mean of six validation sets, document-isolated, seed 1, negative = better)", "d0_thead": ["experiment", "change − control"], "d0_r_d1": "layer 0 dense − all-MoE (DSV3 router)", "d0_r_d2": "layers 0-1 dense − all-MoE", "d0_r_d2d1": "2 dense layers − 1 dense layer (positive = 1 layer better)", "d0_r_fa0": "dense layer 0: full attention NoPE − SWA RoPE", "d0_r_phib": "on dense layer 0: negative entropy + bias − DSV3", "d0_r_phinb": "on dense layer 0: negative entropy without bias − DSV3", "d0_res_foot": "Numbers from the shared result cards (Boxiang-style R1). “·” = no data yet; q4 still running. Single-seed noise about ±0.008.", "d0_hl_h": "Router health: all-MoE (dashed) vs dense layer 0 (solid); colours and line styles as in the table above", "d0_hk": ["dead experts on val_v2", "MaxVio of the busiest expert"], "d0_h_arm": "router", "d0_h_moe": "all-MoE", "d0_h_dense": "dense layer 0", "d0_h_ctrl": "DSV3", "d0_h_phib": "negative entropy + bias", "d0_h_phinb": "negative entropy, no bias", "d0_hl_note": "Same form as “Router health statistics” above: from the training logs (Boxiang's <code>router_health.py</code>, one summary every 100 steps over all MoE layers, seed 1). One chart per rung, x = training step; the buttons switch the statistic, hover for values. The q4 dense arms are still training, so their lines are shorter.", "d0_hl_foot": "Dense layer 0 mainly rescues the negative-entropy-only arm: at the end of q1 / q2 / q3 its dead experts go from 0.26 / 0.68 / 0.65 % to 0 and MaxVio (worst layer) from 8.5 / 9.3 / 6.5 to 3.4 / 2.4 / 4.4, still far above DSV3's ~0.3. During training, however, layer 1 (the first MoE layer after the dense one) still has dead experts: about 11–12 in the first half of q2, 18–25 for most of q3, reaching 0 only at the end. DSV3 and bias + negative entropy are healthy with or without the dense layer (dead 0, worst-layer MaxVio ≤ 0.36).<br>The x axis is the training step; each point summarises all MoE layers at that step (layer mean / worst layer), it is not per layer. The spike of the light-orange solid q4 line (bias + negative entropy · dense layer 0) at step ~44,200 comes from the last two layers (L34 / L35, MaxVio about 8.5, 4 dead experts in L35) and recovers within ~200 steps; at the same step the negative-entropy-only dense run also jumps (from layer 1), the two all-MoE runs do not.", "pp_h": "Data derisk and 2.5 vs 2.6", "pp_note": "<b>Data</b>: the 80B KD corpus pre-packed into 32K rows (pack plan e9r500: a row is several document blocks that do not attend to each other), teacher cache = tc_80b, seed 1 (the plan fixes the data order).<br><b>Experiment</b>: the MUONH_FIT base trained on the same plan with CE / KD 2.6 / KD 2.5, to check the new data pipeline, the KD gain on the new data, and teacher 2.5 vs 2.6. All KD: α 0.8 (KD 0.8 + CE 0.2), tail S64Y.<br><b>Every checkpoint is evaluated twice</b>: 8K window (long documents chunked at 8192, as in every earlier experiment, directly comparable) and 32K window (chunked at 32768, matching the training length).", "pp_l_ce": "MUONH_FIT_PP32K", "pp_l_kd26": "KD 2.6 · MUONH_FIT · new data", "pp_l_kd25": "KD 2.5 · MUONH_FIT · new data", "pp_l_pd25": "KD 2.5 · PROD1_DPSK · new data", "pp_l_ba26": "KD 2.6 · best arch · new data", "pp_l_ba25": "KD 2.5 · best arch · new data", "pp_l_old26": "KD 2.6 · MUONH_FIT · 650B data (reference)", "pp_l_old25": "KD 2.5 · MUONH_FIT · 45B corpus (reference)", "pp_l_swa": "CE · SWA 512 + sink 5:1 · 650B data (reference)", "pp_l_pdce": "CE · PROD1_DPSK · 650B data (reference)", "pp_l_lat": "CE · PROD1_DPSK + latent MoE (iso-param) · 650B data (reference)", "pp_l_lat2x": "CE · PROD1_DPSK + latent MoE, experts ×2 · 650B data (reference)", "pp_old32_h": "650B reference at the 32K window (not drawn): 32K-window score − MUONH_FIT (same zero line as the chart, seed-paired; positive = worse than the zero line)", "pp_tab": {"head": ["Line", "Data", "Architecture", "Loss", "Change ID (8K window)"], "rows": {"ce": ["80B pack plan, 32K", "MUONH_FIT: softmax router, full attention, document mask", "CE"], "kd26": ["80B pack plan, 32K", "same", "KD, teacher MiMo-2.6-Pro-RL"], "kd25": ["80B pack plan, 32K", "same", "KD, teacher MiMo-2.5-Pro-Base"], "pd25": ["80B pack plan, 32K", "PROD1_DPSK (PROD v1: SWA 512 local:global 5:1 + sinks, DSV3 router, dense layer 0, 8 + 1 shared expert, document mask)", "KD, teacher MiMo-2.5-Pro-Base"], "ba26": ["80B pack plan, 32K", "best architecture PROD1_DPSK_LATENT2", "KD, teacher MiMo-2.6-Pro-RL"], "ba25": ["80B pack plan, 32K", "best architecture PROD1_DPSK_LATENT2", "KD, teacher MiMo-2.5-Pro-Base"], "old26": ["650B data, 8K", "MUONH_FIT + document mask", "KD, teacher MiMo-2.6-Pro-RL"], "old25": ["45B KD corpus, 8K", "MUONH_FIT (no document mask)", "KD, teacher MiMo-2.5-Pro-Base"], "swa": ["650B data, 8K", "sliding window 512 + attention sink, local:global 5:1 (global layers NoPE, local layers RoPE over 512 only), no document mask", "CE"], "pdce": ["650B data, 32K", "PROD1_DPSK (the architecture of the KD 2.5 · PROD1_DPSK line); ~8 % more active params than MUONH_FIT and ~8 % more training tokens (TPP 200 on its own active params)", "CE"], "lat": ["650B data, 32K", "PROD1_DPSK + latent MoE (iso-param): the MoE projects h down to h/2, 256 routed experts pick 16 and run at h/2, then project back to h; total params and routed FLOPs equal PROD1_DPSK; document mask", "CE"], "lat2x": ["650B data, 32K", "PROD1_DPSK + latent MoE (d = h/2) with 512 routed experts picking 16: about the same compute per token as PROD1_DPSK, routed params ×2 (not iso-param); document mask", "CE"]}, "foot": "All KD arms: α 0.8 (KD 0.8 + CE 0.2), tail S64Y (top-64 + sampled tail of 4 draws, y exact), MuonH + fitted LR. The 32K-window evals carry the suffix _W32K. Dashed = earlier runs of the same kind (8K training) as references: KD 2.6 on the 650B data; KD 2.5 has no run on the 650B data, so the one on the 45B KD corpus (its q3 ran at TPP 100 and is not drawn; 8K window only). the PROD1_DPSK and latent MoE experts ×2 references are CE on the 650B data, trained at 32K (with the document mask), so they are drawn in all three window tabs."}, "pp_y": "Δloss", "pp_none": "No eval results yet.", "pp_a_h": "8K window: each line − MUONH_FIT", "pp_a_zero": "Zero line = <b>MUONH_FIT</b>: 650B data, 8K training, MuonH, fitted LR, CE, softmax router, full attention (the same zero line as the rest of the report). Negative = better.", "pp_b_h": "32K window: each line − MUONH_FIT", "pp_c_h": "32K window: each line − MUONH_FIT_PP32K (CE on the new data, 32K window)", "pp_c_zero": "Zero line = <b>MUONH_FIT_PP32K</b>: CE trained at 32K on the same 80B pack plan, also scored at the 32K window, seed-paired. The lines are 32K-window scores as well; negative = better than the new CE.", "pp_b_zero": "Zero line = <b>MUONH_FIT</b>, the same as in the 8K window tab (trained at 8K: its 8K-window score, a reference baseline only); the lines are 32K-window scores. Negative = better.", "pp_win": [["w8", "8K window"], ["w32", "32K window"], ["w32ce", "32K window · zero = new CE"]], "pp_got": "Evaluated so far: ", "ba_h": "Architecture changes", "ba_tab": {"head": ["Group", "Item", "Current best architecture (our pick, item by item, from all results)", "Current PROD choice (Runlong)"], "groups": [["Architecture", [["Router scores", "DSV3 style: sigmoid, top-8", "same"], ["Load balancing", "DSV3: expert bias, γ = 1e-3; seq aux, α = 1e-4", "same"], ["Layer 0", "dense SwiGLU (width 9 × expert), SWA attention", "same"], ["MoE layers", "128 top-8 + 1 shared, routed scaling 2.5; + latent 1/2 (256 top-16)", "128 top-8 + 1 shared, routed scaling 2.5"], ["Attention", "SWA 512 + RoPE; every 6th layer full attention + NoPE (5:1); attention sinks", "same"], ["Document mask", "inter-document attention masking", "same"], ["Training length", "8K", "32K"]]], ["Optimizer", [["Optimizer", "MuonH + AdamH (head, router) + AdamW (vectors), wd 0", "same"], ["Router LR", "main LR × 1", "same"], ["LR", "fitted law (Boxiang fit v2, = MUONH_FIT); no warmup, linear decay to 1 %", "q0–q3 same; q4 / q5 constant 5.36e-3 / 4.49e-3"], ["Batch", "2<sup>21</sup> tokens per step", "same"]]], ["KD", [["Loss", "KD (forward KL)", "plain CE"], ["α", "0.8 (KD 0.8 + CE 0.2)", "—"], ["Teacher", "MiMo-2.5-Pro-Base", "—"], ["Tail sampling", "S64Y: top-64 + sampled tail (4 draws), y exact", "—"]]]], "foot": "Orange = the two sides differ. Each item on the left comes from its own single-variable experiment; the combination has not been trained. Latent 1/2 and the 8K training length rest on q0 only (8K on our 8K-length eval, which cannot see the long-document benefit of 32K). Load balancing is not negative entropy: its loss is only 0–0.007 below DSV3 (single-seed noise about ±0.008), level in the PROD setting (PROD1_PHI − PROD1_DPSK ≈ 0), and with negative entropy only (no expert bias) the router is unhealthy."}, "ba_thead": ["component", "experiment", "change − control"], "ba_c_router": "① DSV3 router", "ba_r_dsv3": "softmax router → DSV3 router (full attention)", "ba_r_phi": "in PROD v1, balance loss → negative entropy (phi, α 1e-5)", "ba_c_dense0": "② layer 0 dense (SWA)", "ba_r_dense1": "layer 0 MoE → dense (on DSV3 SWS)", "ba_r_dense2": "layers 0 and 1 both dense", "ba_r_fa0": "dense layer 0, attention SWA (RoPE) → full attention (NoPE)", "ba_c_shared": "③ 8 + 1 shared expert", "ba_r_shared": "+1 shared expert (routed scaling 1; <b>not iso-parameter</b>, about +7-8 % active parameters)", "ba_r_shared_rs": "+1 shared expert + routed scaling 2.5 (<b>not iso-parameter</b>)", "ba_r_rs1": "in PROD v1, routed scaling 2.5 → 1", "ba_c_latent": "latent MoE (1/2)", "ba_r_latent": "PROD v1 MoE → latent MoE: project to h/2, 256 experts top-16 working at h/2, project back to h (iso-parameter, iso routed FLOP)", "ba_c_swa": "④ SWA 512 5:1 + sink", "ba_r_swa": "full attention → SWA window 512, local:global 5:1, sinks (softmax router)", "ba_c_ctx": "⑤ 32K + inter-document mask", "ba_r_s32k": "training length 8K → 32K (no mask)", "ba_r_dm": "8K with inter-document mask", "ba_r_s32kdm_vs32": "32K, mask added", "ba_r_s32kdm_vs8": "32K + mask vs 8K without mask", "ba_c_all": "whole recipe", "ba_r_all": "PROD v1 vs 32K SWA 512 + sink (several changes at once, not single-variable)", "ba_foot": "All evals at 8K. “·” = no data at that rung yet. The PROD v1 family, the dense-layer-0 full-attention arm and the document-mask arms have no 115-split eval of ours yet, so no overall_tsqw here and they are not in the charts below.", "find_h": "Findings",
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
# texts of the prepack KD section (E2X vs E2X + QBX, 11-06); kept here rather than in the long lines above
_PPF_IDS = ('<code>PROD1_DPSK_LATENT2_E2X_KD_A0P8_S64Y_MIMO25_PPFULL</code>', '<code>PROD1_DPSK_LATENT2_E2X_QBX_KD_A0P8_S64Y_MIMO25_PPFULL</code>')
PPF_TXT = {
 "zh": {
  "ppf_h": "数据 derisk round 2",   # renamed (user 11-07 "仍然叫做数据derisk round 2"); was "离线 prepack 数据上的 KD：E2X vs E2X + QBX"
  "ppf_note": "两个 arm 都是 E2X 架构（latent d = h/2，512 个 routed expert 选 16 + 1 shared），在 11-06 的离线 prepack 数据上做 KD："
              "pack plan <code>prepack_tcfull_snap1106_0731_q3e2x</code>（147.7B token，几乎不重复），teacher MiMo-V2.5-Pro-Base（tc_full cache），"
              "α 0.8、top-64、tail 采样 4 次（S64Y），doc mask。<b>A</b> = E2X 原版 router（DSV3 expert bias 1e-3 + seq aux 1e-4，也是 461B 生产默认）；"
              "<b>B</b> = E2X + QBX（quantile balancing，全局精确分位数；aux 0，无 expert bias；Megatron qbx2，255 bins）。"
              "两者命令行只差 router 的 4 个参数，数据和顺序、步数、LR、seed 1 都相同。<br>"
              "0 线和老实验 tab 里“数据 derisk 和 2.5 vs 2.6”一节相同：8K 窗口 = 各线 − <b>MUONH_FIT</b>；32K 窗口 = 各线的 32K 窗口分数 − MUONH_FIT（8K 训练，用它的 8K 分数）。"
              "实线 = KD 的 A / B，两条实线的距离 = KD 下 QBX 的效果（B − A）；虚线 = 参照（650B 数据上的 CE：E2X 和 PROD1_DPSK）。"
              "只有 q1–q3（11-06 16:40 开始训练）；每个 checkpoint 训完后自动评测（同一次加载出 8K / 32K）、自动画上。",
  "ppf_chart": "各线 − 各自的对照（越低越好）",
  "ppf_l_a": "A：KD 2.5 · E2X · prepack 数据", "ppf_l_b": "B：KD 2.5 · E2X + QBX · prepack 数据",
  "ppf_l_c": "C：A 的 q1 · 在 megatron-lm 上重跑",
  # legend table in the form of the data-derisk one (user 11-07 "legend表格里面也要模仿 数据 derisk 和 2.5 vs 2.6"); rows keyed by label key
  "ppf_tab": {"rows": {
      "ppf_l_a": ["prepack 数据 147.7B token（tc_full 11-06 快照），32K；每个 rung 按顺序读前一段：q1 33.2B、q2 76.8B、q3 147.7B（整份）", "PROD1_DPSK + latent MoE（d = h/2），routed expert 512 个选 16 个（= E2X）；router = DSV3 expert bias 1e-3 + seq aux 1e-4；文档 mask", "KD，teacher MiMo-2.5-Pro-Base"],
      "ppf_l_b": ["同上", "同上，只把 router 换成 QBX（quantile balancing，全局精确分位数；aux 0，无 expert bias）", "KD，teacher MiMo-2.5-Pro-Base"],
      "ppf_l_c": ["同 A（同一个 pack plan、同顺序），只有 q1", "同 A；代码换成生产的 megatron-lm（分支 xxdev，3B 生产脚本，原生 <code>--pack-plan</code>）：LayerWise + Megatron 的 hyperball，梯度归约 bf16", "同 A"],
      "pp_l_pdce": ["650B 数据，32K", "PROD1_DPSK（E2X 的基础架构：没有 latent MoE，expert 不加倍）；文档 mask", "CE"],
      "x_abs_zl": ["650B 数据，8K", "MUONH_FIT：128 选 8 MoE、softmax router、全注意力；MuonH、拟合 LR（前两个 tab 的 0 线）", "CE"]},
    "foot": "A、B 都是 α 0.8（KD 0.8 + CE 0.2），tail = S64Y（top-64 + 采样 tail 4 draws，y 精确），MuonH + 拟合 LR，seed 1；"
            "数据 = pack plan <code>prepack_tcfull_snap1106_0731_q3e2x</code>（4,508,032 条 × 32K = 147.7B token，padding 0.03%；按 q3 的 70,438 步 × 64 条做的），A、B 读同一份、顺序相同；"
            "几乎不重复：各数据源被读到的比例中位数 2.8%，99% 分位 18.7%，最多 1.05 遍。32K 窗口的评测 Change ID 在后面加 _W32K。"
            "虚线 = 参照：650B 数据上 32K 训练（加文档 mask）的 CE，一条和 A 同架构、同 router（E2X），一条是 PROD1_DPSK 本身，所有 tab 都画（和「数据 derisk 和 2.5 vs 2.6」里同样的线）。"},
  # arm C = A's q1 rerun on megatron-lm xxdev (user 11-10 "这个加到报告里 / 如果没有算isodoc测试，那先去测试"; job 56389 -> 56500, evals 57306-57308)
  "ppf_c_h": "C：A 的 q1 在 megatron-lm 上重跑（pack plan 迁移的 derisk）",
  "ppf_c_note": "把 A 的 q1（同一个 pack plan、同样的 KD 设定、seed 1）放到生产代码 megatron-lm 上重跑：分支 xxdev，原生 <code>--pack-plan</code> dataloader，3B 生产脚本，"
                "8 节点，job 56389 → 56500，15,824 步，11-10 03:49 跑完。和 A 的差别只是生产代码自己的选择：LayerWise + Megatron 的 hyperball、梯度归约 bf16（A 是 fp32）、"
                "没有 fast step、训练中只测 val v2。<br>"
                "<b>数据完全一致</b>：两边都打印了的 1,583 个日志点里，1,552 个 teacher tail mass 每一位相同，其余 31 个只在第 7 位有效数字上不同"
                "（相对误差 ≤ 4.5e-7，fp32 求和顺序），所以每一步读的数据行和 teacher 行都和 A 相同。KD coverage 一直是 1，token mismatch 一直是 0。<br>"
                "<b>训练 loss</b>：lm loss 全程平均比 A 低约 0.1%；最后 500 步平均 1.70087（A 1.70254）。训练中的 val v2 lm loss 第 15000 步 1.3700。<br>"
                "<b>doc-isolated 评测</b>（和上图同样的 115 个 split，job 57306–57308）：下表 7 个指标的 C − A 在 −0.0019 到 +0.0001 之间；"
                "按 9 个 suite 看，最大 +0.0005（arxiv），平均 −0.0009（8K）/ −0.0008（32K），C 低的 6 个、高的 3 个。迁移后的 q1 和 ladder 原 run 一致，C 略低。",
  "ppf_c_win": ["8K 窗口", "32K 窗口"], "ppf_c_head": ["指标", "C", "A", "C − A"],
  "ppf_tl_h": "训练 loss：B − A（同 step，训练中的早期参考）",
  "ppf_tl_note": "A 和 B 读同一个 pack plan、顺序相同，同一 step 看到的是同一批数据，所以同 step 相减比一般的单 seed 对比噪声小。"
                 "取两边都有记录的最新 step S，在 (S − 500, S] 内各自求平均再相减（B − A，负 = QBX 更好）。lm loss = 真实 token 的 CE；kd loss = 对 teacher 分布的蒸馏项。"
                 "这是训练 loss，不是评测；前 1–2 千步差值波动大，不能当结论。报告每次重新生成时更新；结论以上面的评测为准。",
  "ppf_tl_lm": "lm loss：B − A", "ppf_tl_kd": "kd loss：B − A",
  "ppf_rt_h": "Router health：A vs B（训练日志）",
  "ppf_st_h": "router health 各指标的说明",
  "ppf_st_note": "数据来自训练时挂的监测脚本 <code>router_health.py</code>，每 100 步记一次，覆盖所有 MoE 层。图上方的按钮就是下表的 9 个指标。",
  "ppf_st_head": ["指标", "怎么算", "怎么看"],
  "ppf_st_rows": {
    "vio_mean": ("每一层：在一整个训练 step（所有 micro-batch、所有 GPU）里数 512 个 expert 各收到多少 token，MaxVio = 最多的那个 ÷ 平均值 − 1（DeepSeek-V3 的定义）；再对所有 MoE 层取平均。",
                 "越小越均衡，0 = 完全均衡；1 表示最忙的 expert 收到平均值的 2 倍。"),
    "vio_max": ("同上，取最不均衡的那一层。", "越小越好；看有没有个别层特别不均衡。"),
    "dead": ("一个 step 里收到的 token 不到平均值 10% 的 expert 个数，占（层数 × 512）的百分比。", "应为 0；大于 0 说明有 expert 几乎没被用到。"),
    "bmax": ("A（E2X router）每个 expert 有一个 bias，只影响选哪些 expert；取所有层里绝对值最大的那个。",
             "预设健康上限 1（红色虚线）。同一层的 bias 一起抬高不改变路由，起作用的是层内 bias 之间的差。<b>B 没有这条线</b>：QBX 不用 expert bias，它的门槛 β 监测脚本没有记录。"),
    "sat": ("router logit z（= 输入 × router 权重，fp32，不含 bias / β）中 |z| &gt; 4 的比例，各层平均；只看 rank 0 的第一个 micro-batch。",
            "同时包括 z &lt; −4（明确不选某个 expert，正常，训练后期占大多数）和 z &gt; +4（见下一行），所以单看这个数分不出好坏。"),
    "satpos": ("同上，只数 z &gt; +4 的比例。", "sigmoid 分数被钉在 1 附近，被选中的 expert 之间失去区分度（坏）；预设上限 5%（红色虚线）。"),
    "pinned": ("每个 token 最大的 8 个 logit 里 z &gt; +4 的比例（固定看前 8 个，虽然 E2X 每个 token 选 16 个 expert）。", "越低越好；预设上限 5%（红色虚线）。"),
    "zmax": ("所有层里最大的 |z|（rank 0 的第一个 micro-batch）。", "看 logit 有没有一直变大、失控；持续上升是坏信号。"),
    "upd": ("router 权重这一步的变化量 ÷ 权重本身的大小（Frobenius 范数之比），各层平均。", "只作参考，不作判据；突然变大或接近 0 时再细看。"),
  },
  "ppf_rt_note": "来自训练日志的 router health（每 100 步一行，所有层汇总）。两条线 = KD 的 A / B，颜色和上面评测图一致；鼠标移到图上看名字和数值。"
                 "dead expert 的百分比按（层数 × 512）算。MaxVio 0 = 完全均衡。",
  "ppf_rt_a": "A：E2X + KD", "ppf_rt_b": "B：E2X + QBX + KD", "ppf_rt_ace": "E2X（CE）", "ppf_rt_bce": "E2X + QBX（CE）",
  "x_vt_loss": "Δloss（nats / token，和上图同一单位）", "x_vt_bpb": "Δbpb（bits / byte，按原文字节数换算，不受 tokenizer 影响）",
  "x_delta_tab": ["8K 窗口 · 差值", "32K 窗口 · 差值"], "x_abs_tab": "8K 窗口 · 绝对值", "x_abs_h": "8K 窗口：val_loss 绝对值（log-log）",
  "x_abs_zl": "0 线：MUONH_FIT · CE · 650B 数据",
  "x_abs_only": "只在绝对值 tab", "x_slope": "斜率",
  "x_abs32_tab": "32K 窗口 · 绝对值", "x_abs32_h": "32K 窗口：val_loss 绝对值（log-log）",
  "x_abs_zl32": "0 线：MUONH_FIT · CE · 650B 数据（8K 窗口分数）"},
 "en": {
  "ppf_h": "Data derisk round 2",
  "ppf_note": "Both arms are the E2X architecture (latent d = h/2, 512 routed experts top-16 + 1 shared), trained with KD on the 11-06 offline prepack: "
              "pack plan <code>prepack_tcfull_snap1106_0731_q3e2x</code> (147.7B tokens, almost no repetition), teacher MiMo-V2.5-Pro-Base (tc_full cache), "
              "α 0.8, top-64, tail sampled 4 times (S64Y), doc mask. <b>A</b> = the E2X router (DSV3 expert bias 1e-3 + seq aux 1e-4; also the 461B production default); "
              "<b>B</b> = E2X + QBX (quantile balancing with the exact global quantile; aux 0, no expert bias; Megatron qbx2, 255 bins). "
              "The command lines differ only in the router's 4 flags; data and order, iterations, LR and seed 1 are identical.<br>"
              "The zero line is that of the section “Data derisk and 2.5 vs 2.6” (older-experiments tab): 8K window = each line − <b>MUONH_FIT</b>; 32K window = each line's 32K-window score − MUONH_FIT (trained at 8K: its 8K score). "
              "Solid = the KD arms A / B, and the gap between them is QBX under KD (B − A); dashed = the references (CE on the 650B data: E2X and PROD1_DPSK). "
              "q1-q3 only (training since 11-06 16:40); each checkpoint is evaluated automatically when it finishes (one load, 8K and 32K) and drawn here.",
  "ppf_chart": "Each line − its reference (lower is better)",
  "ppf_l_a": "A: KD 2.5 · E2X · prepack data", "ppf_l_b": "B: KD 2.5 · E2X + QBX · prepack data",
  "ppf_l_c": "C: A's q1 · rerun on megatron-lm",
  "ppf_tab": {"rows": {
      "ppf_l_a": ["prepack data, 147.7B tokens (tc_full snapshot 11-06), 32K; each rung reads the plan from the start: q1 33.2B, q2 76.8B, q3 147.7B (all of it)", "PROD1_DPSK + latent MoE (d = h/2) with 512 routed experts picking 16 (= E2X); router = DSV3 expert bias 1e-3 + seq aux 1e-4; document mask", "KD, teacher MiMo-2.5-Pro-Base"],
      "ppf_l_b": ["same", "same, only the router replaced by QBX (quantile balancing with the exact global quantile; aux 0, no expert bias)", "KD, teacher MiMo-2.5-Pro-Base"],
      "ppf_l_c": ["as A (the same pack plan, same order), q1 only", "as A, on the production code megatron-lm (branch xxdev, the 3B production script, native <code>--pack-plan</code>): LayerWise + Megatron's hyperball, bf16 gradient reduction", "as A"],
      "pp_l_pdce": ["650B data, 32K", "PROD1_DPSK (the base of E2X: no latent MoE, experts not doubled); document mask", "CE"],
      "x_abs_zl": ["650B data, 8K", "MUONH_FIT: MoE 128 pick 8, softmax router, full attention; MuonH, fitted LR (the zero line of the first two tabs)", "CE"]},
    "foot": "A and B: α 0.8 (KD 0.8 + CE 0.2), tail S64Y (top-64 + sampled tail of 4 draws, y exact), MuonH + fitted LR, seed 1; "
            "data = pack plan <code>prepack_tcfull_snap1106_0731_q3e2x</code> (4,508,032 rows × 32K = 147.7B tokens, 0.03 % padding; sized for q3's 70,438 steps × 64 rows), A and B read the same plan in the same order; "
            "almost no repetition: the fraction of each source read has median 2.8 %, p99 18.7 %, max 1.05 passes. The 32K-window evals carry the suffix _W32K. "
            "Dashed = references: CE trained at 32K on the 650B data (+ document mask), one with A's architecture and router (E2X), one PROD1_DPSK itself, drawn in every tab (the same lines as in “Data derisk and 2.5 vs 2.6”)."},
  "ppf_c_h": "C: A's q1 rerun on megatron-lm (derisk of the pack-plan migration)",
  "ppf_c_note": "A's q1 (the same pack plan, the same KD settings, seed 1) rerun on the production code megatron-lm: branch xxdev, the native <code>--pack-plan</code> dataloader, the 3B production script, "
                "8 nodes, jobs 56389 → 56500, 15,824 steps, finished 11-10 03:49. It differs from A only by the production code's own choices: LayerWise + Megatron's hyperball, bf16 gradient reduction (A: fp32), "
                "no fast step, only val v2 during training.<br>"
                "<b>Same data</b>: of the 1,583 log points printed by both, 1,552 have bit-identical teacher tail mass, and the other 31 differ only in the 7th significant digit "
                "(relative ≤ 4.5e-7, fp32 summation order), so every step read the same data rows and teacher rows as A. KD coverage stayed 1 and the token mismatch 0.<br>"
                "<b>Training loss</b>: lm loss about 0.1 % below A on average over the run; mean of the last 500 steps 1.70087 (A 1.70254). Val v2 lm loss during training 1.3700 at step 15000.<br>"
                "<b>Doc-isolated evals</b> (the same 115 splits as the chart, jobs 57306-57308): C − A of the 7 metrics in the table below lies between −0.0019 and +0.0001; "
                "per suite (9) at most +0.0005 (arxiv), mean −0.0009 (8K) / −0.0008 (32K), C lower on 6, higher on 3. The migrated q1 matches the ladder run, C slightly lower.",
  "ppf_c_win": ["8K window", "32K window"], "ppf_c_head": ["metric", "C", "A", "C − A"],
  "ppf_tl_h": "Training loss: B − A (matched steps, early read during training)",
  "ppf_tl_note": "A and B read the same pack plan in the same order, so at a given step they see the same batch, and a matched-step difference is less noisy than a usual single-seed comparison. "
                 "At the latest step S logged by both, each side is averaged over (S − 500, S], then subtracted (B − A, negative = QBX better). lm loss = CE on the true tokens; kd loss = the distillation term against the teacher. "
                 "Training loss, not an eval; in the first 1-2k steps the gap swings a lot and is not a result. Updated whenever the report is regenerated; the doc evals above are the result.",
  "ppf_tl_lm": "lm loss: B − A", "ppf_tl_kd": "kd loss: B − A",
  "ppf_rt_h": "Router health: A vs B (training logs)",
  "ppf_st_h": "What the router-health statistics mean",
  "ppf_st_note": "From the monitor <code>router_health.py</code> hooked into training: one record every 100 steps, all MoE layers. The buttons above the chart are the 9 statistics of this table.",
  "ppf_st_head": ["statistic", "how it is computed", "how to read it"],
  "ppf_st_rows": {
    "vio_mean": ("Per layer: count the tokens each of the 512 experts receives over a whole training step (all micro-batches, all GPUs); MaxVio = the busiest expert / the mean − 1 (DeepSeek-V3's definition); then the mean over all MoE layers.",
                 "Smaller is more balanced, 0 = perfectly balanced; 1 means the busiest expert gets twice the mean."),
    "vio_max": ("As above, the least balanced layer.", "Smaller is better; shows whether a single layer is badly balanced."),
    "dead": ("Experts receiving under 10 % of the mean load in a step, as % of (layers × 512).", "Should be 0; above 0 means some experts are almost never used."),
    "bmax": ("A (E2X router) has one bias per expert that only decides which experts are picked; the largest absolute value over all layers.",
             "Pre-registered health limit 1 (red dashed). Shifting all biases of a layer together does not change routing; what matters is the spread within a layer. <b>B has no line here</b>: QBX uses no expert bias, and the monitor does not record its threshold β."),
    "sat": ("Share of router logits z (= input × router weight, fp32, without bias / β) with |z| &gt; 4, mean over layers; rank 0's first micro-batch only.",
            "Counts both z &lt; −4 (an expert clearly rejected: normal, most of them late in training) and z &gt; +4 (next row), so this number alone is neither good nor bad."),
    "satpos": ("As above, only z &gt; +4.", "Scores pinned near 1, the selected experts lose discrimination (bad); pre-registered limit 5 % (red dashed)."),
    "pinned": ("Share of each token's 8 largest logits with z &gt; +4 (always the top 8, although E2X picks 16 experts per token).", "Lower is better; pre-registered limit 5 % (red dashed)."),
    "zmax": ("The largest |z| over all layers (rank 0's first micro-batch).", "Watch for logits growing without bound; a steady rise is a bad sign."),
    "upd": ("Change of the router weight over the step / its size (ratio of Frobenius norms), mean over layers.", "A flag, not a criterion; look closer if it jumps or drops to ~0."),
  },
  "ppf_rt_note": "Router health from the training logs (one line every 100 steps, all layers). The two lines = the KD arms A / B, coloured as in the eval chart above; hover for names and values. "
                 "Dead experts in % of (layers × 512). MaxVio 0 = perfectly balanced.",
  "ppf_rt_a": "A: E2X + KD", "ppf_rt_b": "B: E2X + QBX + KD", "ppf_rt_ace": "E2X (CE)", "ppf_rt_bce": "E2X + QBX (CE)",
  "x_vt_loss": "Δloss (nats / token, the chart's unit)", "x_vt_bpb": "Δbpb (bits / byte of the original text, independent of the tokenizer)",
  "x_delta_tab": ["8K window · difference", "32K window · difference"], "x_abs_tab": "8K window · absolute", "x_abs_h": "8K window: absolute val_loss (log-log)",
  "x_abs_zl": "zero line: MUONH_FIT · CE · 650B data",
  "x_abs_only": "absolute tabs only", "x_slope": "slope",
  "x_abs32_tab": "32K window · absolute", "x_abs32_h": "32K window: absolute val_loss (log-log)",
  "x_abs_zl32": "zero line: MUONH_FIT · CE · 650B data (8K-window score)"}}
for _lang, _txt in PPF_TXT.items():
    T[_lang].update(_txt)

# QBK3 / QBK3_WU in the E2X router chapter (11-08; user "他们好像在训练qbk3神秘的"). Source: Runlong's exp files
# prod/exp/qwen3_tpp200_prod1_dpsk_latent2_e2x_qbk3[_wu].sh + share/megatron_qbk3_tests/REVIEW_REPORT.md
T["zh"].update({
  "rtq_l_qbk3": "QBK3：完全按 Kimi K3 原文的 quantile balancing",
  "rtq_l_qbk3_d": "按 K3 技术报告（2.3.3 节公式 13–14、附录 C / D）和官方 HF gate 实现。路由 = top-k(sigmoid(logit) + expert bias)，gate 权重不含 bias，走的是 Megatron 原本的 expert-bias 路径（和 BIASONLY 一样，没有 seq aux）。不同的是 bias 的更新：不再按 ±1e-3 的固定步长，而是每个 step 用整个 global batch 的分位数直接算出（1000 格直方图，TP × DP × CP 上一次 all-reduce，线性插值，居中，下一个 step 生效）。补丁 Megatron：<code>megatron-lm-80781ab0ab-qbk3</code>。",
  "rtq_l_qbk3wu": "QBK3 + 1% LR warmup",
  "rtq_l_qbk3wu_d": "QBK3 加 1% 线性 LR warmup（prod 配方本来没有 warmup；K3 / DSV3 都有）。看 warmup 能否去掉 QBK3 开头的瞬态和多出来的冷 expert。注意 warmup 改了所有参数的 LR，所以它和 QBK3 的 loss 差里混有 warmup 本身的效果；负载均衡 / 冷 expert 的读数是干净的。"})
T["en"].update({
  "rtq_l_qbk3": "QBK3: quantile balancing exactly as Kimi K3",
  "rtq_l_qbk3_d": "Implemented as in the K3 tech report (sec. 2.3.3 Eqs. 13–14, App. C / D) and the official HF gate. Routing = top-k(sigmoid(logit) + expert bias), gate weights without the bias, i.e. Megatron's existing expert-bias path (as BIASONLY, no seq aux). What changes is the bias update: no fixed ±1e-3 step; every step the bias is set from the quantile of the whole global batch (1000-bin histogram, one all-reduce over TP × DP × CP, linear interpolation, mean-centred, applied from the next step). Patched Megatron: <code>megatron-lm-80781ab0ab-qbk3</code>.",
  "rtq_l_qbk3wu": "QBK3 + 1% LR warmup",
  "rtq_l_qbk3wu_d": "QBK3 with a 1% linear LR warmup (the prod recipe has none; K3 / DSV3 warm up). Does a short warmup remove QBK3's early transient and its extra cold experts? Warmup changes the LR of every parameter, so its loss gap to QBK3 mixes in warmup's own effect; the balance / cold-expert readings are clean."})

# Progressive stack under the best-architecture table (user 11-05): strings for stack_section()
T["zh"].update({
  "st_h": "逐项叠加：每加一项，loss 降多少",
  "st_note": "全部是 CE、MuonH、650B 数据（corpus_v2）、seed 1，评测用 doc-isolated 8K 窗口。每一行比上一行多加一项：表里每一行列出全部设置，橙色底的格子 = 这一行相对上一行改了的地方；q0–q3 列 = 累计（这一行 − 第一行 OPT_MUONH，负数 = 更好），相邻两行之差就是这一项的效果。图的画法和优化器对比相同：横轴是激活参数（不含 embedding，对数坐标），纵轴是相对 MuonH 老 LR 的 Δloss，一条线一行，颜色和表里每行前的方块相同（每条线 = 叠到这一行为止的累计效果）。",
  "st_names": ["MuonH，老 LR（ArchScale 原版公式）", "+ 拟合 LR（fit v2）", "+ DSV3 router（sigmoid + expert bias + seq aux）",
               "+ SWA 窗口 2048（5:1，全注意力层 NoPE）+ attention sink", "+ 1 个 shared expert（8+1），routed scaling 2.5",
               "+ SWA 窗口 512、第 0 层 dense、32K 训练、文档 mask（= PROD1_DPSK）", "+ latent MoE（d = h/2，256 选 16，等参数）", "+ 专家数 ×2（latent，512 选 16）"],
  "st_short": ["老 LR", "+ 拟合 LR", "+ DSV3 router", "+ SWA 2048 + NoPE + sink", "+ shared expert", "+ SWA 512 / dense 0 / 32K / mask", "+ latent MoE", "+ 专家 ×2"],
  "st_thead": ["Change ID"],
  "st_aspects": [
    ("LR", [(0, "ArchScale 原版：0.016 · (L/8)<sup>−0.5</sup> · (T/10.4B)<sup>−0.32</sup>"), (1, "拟合 fit v2：0.01725 · (L/8)<sup>−0.176</sup> · (T/10.4B)<sup>−0.259</sup>")]),
    ("Router", [(0, "softmax 打分；负载均衡 aux loss 1e-3"), (2, "DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4")]),
    ("注意力（SWA / 位置编码 / sink）", [(0, "全注意力，RoPE；无 sink"), (3, "5:1：5 层 SWA 窗口 2048（RoPE）+ 1 层全注意力（NoPE）；可学习 sink"),
                                    (5, "5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink")]),
    ("MoE 层", [(0, "128 选 8，无 shared expert"), (4, "128 选 8 + 1 shared（3h/8）；routed scaling 2.5"),
               (6, "256 选 16（expert 在 d 维上，每个半大）+ 1 shared；routed scaling 2.5；总参数同上一行"), (7, "512 选 16 + 1 shared；routed scaling 2.5；routed 参数 ×2")]),
    ("Latent MoE", [(0, "无"), (6, "有：d = h/2（h → d 投影，expert 在 d 上算，RMSNorm 后投回 h；router 和 shared expert 仍在 h 上）")]),   # own column (user 11-05)
    ("第 0 层", [(0, "MoE（同其他层）"), (5, "dense SwiGLU（宽 9 × 3h/8），注意力 SWA")]),
    ("训练长度 / 文档 mask", [(0, "8K；文档之间不 mask"), (5, "32K（每步仍 2<sup>21</sup> token）；文档间 attention mask")]),
    ("训练量", [(0, "TPP 200（q0 4,577 步）"), (4, "TPP 200 按新的激活参数算（q0 4,915 步，+7.4%）")])], "st_cum": "累计", "st_step": "这一步", "st_y": "Δloss（相对 MuonH 老 LR，越负越好）",
  "st_chart": "相对 MuonH 老 LR 的 Δloss（越负越好）",
  "st_desc": ["起点。优化器 MuonH（hidden 矩阵含 expert）+ AdamH（head、router）+ AdamW（向量），wd 0，无 warmup，线性衰减到 1%；LR 用 ArchScale 原版公式 η = 0.016 · (L/8)<sup>−0.5</sup> · (T/10.4B)<sup>−0.32</sup>。架构：softmax router + aux loss 1e-3，128 选 8，无 shared expert；全注意力 + RoPE；8K 训练。",
              "只换 LR 公式：Boxiang 的拟合公式 fit v2，η = 0.01725 · (L/8)<sup>−0.176</sup> · (T/10.4B)<sup>−0.259</sup>（q0 0.0134 → 0.0164，q3 0.0036 → 0.0070，越深越大），其余不变。",
              "只换 router：打分 softmax → sigmoid（top-8）；负载均衡 aux loss 1e-3 → DeepSeek-V3 式 expert bias（γ = 1e-3，不走梯度）+ 序列级 aux loss 1e-4；gate 仍用 AdamH，router LR × 1；仍是 128 选 8、无 shared expert。",
              "注意力改成 5:1 混合 + sink，三项一起：① 每 6 层中 5 层滑动窗口注意力（窗口 2048，RoPE）；② 第 6 层全注意力且不用位置编码（NoPE）；③ gpt-oss 式可学习 attention sink：每层每个 query head 一个可学习的 sink logit（初始 0，放在 AdamW 组），softmax 时多一个可以“什么都不看”的位置。参数量、token 数不变（sink 只多几百个参数）。",
              "加 1 个 shared expert（宽度 = 一个 routed expert，3h/8），routed scaling 2.5（top-8 gate 权重 × 2.5，DeepSeek-V3 的值）。激活参数 +7–8%，TPP 200 按激活参数算，所以训练 token +7.4%（q0 4,577 → 4,915 步），LR 按公式略低。",
              "一次改四项：① SWA 窗口 2048 → 512；② 第 0 层 MoE 换成 dense SwiGLU（宽 9 × 3h/8），注意力用 SWA；③ 训练长度 8K → 32K（从第 0 步起，每步仍 2<sup>21</sup> token = 64 × 32K）；④ 文档间 attention mask。= Runlong 的 PROD v1（DSV3 router）。",
              "MoE 层换成 latent MoE（Kimi K3 式）：共享的 h → d 降维（d = h/2），256 个 routed expert（宽 3h/8）在 d 上计算，top-16，d 上 RMSNorm 后共享的 d → h 升维；router 和 shared expert 仍在 h 上。与上一行等参数、等 routed FLOP、等 all-to-all（降维 / 升维投影约多 8% 计算）。",
              "latent MoE 的 routed expert 数 256 → 512，top-16 不变：每个 token 的计算量和 all-to-all 不变，总参数 × 2（q1 非 embedding 1.61B → 3.14B），激活参数几乎不变（宽了的 router）。"],
  "st_notes": ["顺序跟着现有的单变量实验走：shared expert 只在 SWA + sink 底座上测过，所以 MUONH_FIT_DSV3_SWS_SHARED_RS2P5 排在 MUONH_FIT_DSV3_SWS 后面；NoPE 没有单独的实验（DSV3 底座上的 RoPE 对照没跑）。",
               "MUONH_FIT_DSV3_SWS 一次加了 SWA 2048、NoPE、sink 三项（只有 q0 的 SWA 无 sink 实验没有放进来）；PROD1_DPSK 一次加了四项（窗口 2048 → 512、第 0 层 dense、32K 训练、文档 mask），没有逐项拆开的实验。",
               "从 MUONH_FIT_DSV3_SWS_SHARED_RS2P5 起训练 token 多 7.4%（加了 shared expert 后激活参数变大，TPP 200 按激活参数算），LR 也随之略低；PROD1_DPSK_LATENT2_E2X 总参数 ×2（激活参数不变）。",
               "最后三行（PROD1_DPSK、PROD1_DPSK_LATENT2、PROD1_DPSK_LATENT2_E2X）是 32K 训练，这里用 8K 窗口评测，32K 带来的好处测不到。单 seed，单格噪声约 ±0.008。"]})
T["en"].update({
  "st_h": "Stacked one by one: how much each added item lowers the loss",
  "st_note": "All CE, MuonH, the 650B data (corpus_v2), seed 1, doc-isolated eval at the 8K window. Each row adds one item to the row above: every row lists the full setting, orange cells = what changes vs the row above; the q0–q3 columns = cumulative (this row − the first row, OPT_MUONH; negative = better); the difference between two adjacent rows is the effect of that item. The chart has the same form as the optimizer comparison: x = active parameters (non-embedding, log scale), y = Δloss vs MuonH at the old LR, one line per row in the colour of the square before that row in the table (each line = the cumulative effect up to that row).",
  "st_names": ["MuonH, old LR (original ArchScale law)", "+ fitted LR (fit v2)", "+ DSV3 router (sigmoid + expert bias + seq aux)",
               "+ SWA window 2048 (5:1, NoPE on the full-attention layers) + attention sink", "+ 1 shared expert (8+1), routed scaling 2.5",
               "+ SWA window 512, dense layer 0, 32K training, document mask (= PROD1_DPSK)", "+ latent MoE (d = h/2, 256 pick 16, iso-parameter)", "+ 2× experts (latent, 512 pick 16)"],
  "st_short": ["old LR", "+ fitted LR", "+ DSV3 router", "+ SWA 2048 + NoPE + sink", "+ shared expert", "+ SWA 512 / dense 0 / 32K / mask", "+ latent MoE", "+ experts ×2"],
  "st_thead": ["Change ID"],
  "st_aspects": [
    ("LR", [(0, "original ArchScale: 0.016 · (L/8)<sup>−0.5</sup> · (T/10.4B)<sup>−0.32</sup>"), (1, "fitted fit v2: 0.01725 · (L/8)<sup>−0.176</sup> · (T/10.4B)<sup>−0.259</sup>")]),
    ("Router", [(0, "softmax scores; balance aux loss 1e-3"), (2, "DSV3: sigmoid scores; expert bias (γ 1e-3) + seq aux 1e-4")]),
    ("Attention (SWA / position / sink)", [(0, "full attention, RoPE; no sink"), (3, "5:1: 5 SWA layers, window 2048 (RoPE) + 1 full layer (NoPE); learnable sinks"),
                                          (5, "5:1: 5 SWA layers, window 512 (RoPE) + 1 full layer (NoPE); learnable sinks")]),
    ("MoE layers", [(0, "128 pick 8, no shared expert"), (4, "128 pick 8 + 1 shared (3h/8); routed scaling 2.5"),
                    (6, "256 pick 16 (experts at d, half-size each) + 1 shared; routed scaling 2.5; total params as the row above"), (7, "512 pick 16 + 1 shared; routed scaling 2.5; routed params × 2")]),
    ("Latent MoE", [(0, "none"), (6, "yes: d = h/2 (h → d projection, experts at d, RMSNorm, then back to h; router and shared expert stay at h)")]),
    ("Layer 0", [(0, "MoE (as the other layers)"), (5, "dense SwiGLU (width 9 × 3h/8), SWA attention")]),
    ("Sequence length / document mask", [(0, "8K; no mask between documents"), (5, "32K (still 2<sup>21</sup> tokens per step); intra-document attention mask")]),
    ("Training tokens", [(0, "TPP 200 (q0 4,577 steps)"), (4, "TPP 200 on the new active params (q0 4,915 steps, +7.4 %)")])], "st_cum": "cumulative", "st_step": "this step", "st_y": "Δloss (vs MuonH at the old LR, more negative is better)",
  "st_chart": "Δloss vs MuonH at the old LR (more negative is better)",
  "st_desc": ["Start. Optimizer MuonH (hidden matrices incl. experts) + AdamH (head, router) + AdamW (vectors), wd 0, no warmup, linear decay to 1 %; LR = the original ArchScale law η = 0.016 · (L/8)<sup>−0.5</sup> · (T/10.4B)<sup>−0.32</sup>. Architecture: softmax router + aux loss 1e-3, 128 pick 8, no shared expert; full attention + RoPE; 8K training.",
              "Only the LR law changes: Boxiang's fit v2, η = 0.01725 · (L/8)<sup>−0.176</sup> · (T/10.4B)<sup>−0.259</sup> (q0 0.0134 → 0.0164, q3 0.0036 → 0.0070, larger the deeper); everything else unchanged.",
              "Only the router changes: scores softmax → sigmoid (top-8); balance aux loss 1e-3 → DeepSeek-V3 expert bias (γ = 1e-3, no gradient) + sequence-wise aux loss 1e-4; gate still AdamH, router LR × 1; still 128 pick 8, no shared expert.",
              "Attention becomes a 5:1 hybrid with sinks, three items together: (1) in every 6 layers, 5 sliding-window layers (window 2048, RoPE); (2) 1 full-attention layer without position encoding (NoPE); (3) gpt-oss-style learnable attention sinks: one learnable sink logit per query head per layer (init 0, in the AdamW group), an extra slot in the softmax that attends to nothing. Same parameters and tokens (the sinks add a few hundred parameters).",
              "Adds 1 shared expert (one routed expert's width, 3h/8) and routed scaling 2.5 (top-8 gate weights × 2.5, DeepSeek-V3's value). Active parameters +7–8 %; TPP 200 counts active parameters, so training tokens +7.4 % (q0 4,577 → 4,915 steps) and the law's LR is slightly lower.",
              "Four changes at once: (1) SWA window 2048 → 512; (2) layer 0's MoE replaced by a dense SwiGLU (width 9 × 3h/8) with SWA attention; (3) training length 8K → 32K (from step 0, still 2<sup>21</sup> tokens per step = 64 × 32K); (4) intra-document attention mask. = Runlong's PROD v1 (DSV3 router).",
              "The MoE layers become a latent MoE (Kimi K3 form): shared h → d down-projection (d = h/2), 256 routed experts (width 3h/8) at d, top-16, RMSNorm on d, then the shared d → h up-projection; router and shared expert stay at h. Iso-parameter, iso routed FLOPs and iso all-to-all with the row above (the projections add about 8 % compute).",
              "Latent MoE with 512 instead of 256 routed experts, top-16 unchanged: same per-token compute and all-to-all, 2 × the total parameters (q1 non-embedding 1.61B → 3.14B), active parameters almost unchanged (the wider router)."],
  "st_notes": ["The order follows the single-variable experiments that exist: the shared expert was only tested on the SWA + sink base, so MUONH_FIT_DSV3_SWS_SHARED_RS2P5 comes after MUONH_FIT_DSV3_SWS; NoPE has no experiment of its own (the RoPE control on the DSV3 base was never run).",
               "MUONH_FIT_DSV3_SWS adds SWA 2048, NoPE and the sinks together (the q0-only SWA-without-sink run is left out); PROD1_DPSK adds four at once (window 2048 → 512, dense layer 0, 32K training, document mask); there are no experiments that split them.",
               "From MUONH_FIT_DSV3_SWS_SHARED_RS2P5 on, training uses 7.4 % more tokens (the shared expert raises the active parameters and TPP 200 counts active parameters), with a slightly lower LR; PROD1_DPSK_LATENT2_E2X doubles the total parameters (active unchanged).",
               "The last three rows (PROD1_DPSK, PROD1_DPSK_LATENT2, PROD1_DPSK_LATENT2_E2X) are trained at 32K and evaluated here at the 8K window, so the 32K benefit is not measured. One seed; single-cell noise about ±0.008."]})

CSS = """
.viz-root{color-scheme:light;--page:#f9f9f7;--surface-1:#fcfcfb;--text-primary:#0b0b0b;--text-secondary:#52514e;--muted:#898781;
 --grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#8a5cd6;--s5:#c49a00;--good:#006300;
 --o-adamw:#6250d6;--o-muon:#1baf7a;--a07:#f4a27f;--a08:#eb6834;--a09:#b4471d;--k07:#9ec5f4;--k08:#5598e7;--k09:#256abf;--g07:#8fd9b6;--g08:#1baf7a;--g09:#0f7a53;
 --v1:#2a78d6;--v2:#1baf7a;--v3:#eda100;--v4:#e87ba4;--v5:#008300;--v6:#6250d6;--v7:#e34948;--v8:#52514e;
 --c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--c4:#eda100;--c5:#e87ba4;--c6:#008300}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--page:#0d0d0d;--surface-1:#1a1a19;
 --text-primary:#f0efec;--text-secondary:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#9d75e0;--s5:#d4ab1e;--good:#0ca30c;
 --o-adamw:#9085e9;--o-muon:#199e70;--a07:#f0a47f;--a08:#d95926;--a09:#a8461f;--k07:#86b6ef;--k08:#2a78d6;--k09:#184f95;--g07:#7fcfa9;--g08:#199e70;--g09:#0d6b4a;
 --v1:#3987e5;--v2:#199e70;--v3:#c98500;--v4:#d55181;--v5:#008300;--v6:#9085e9;--v7:#e66767;--v8:#c3c2b7;
 --c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#c98500;--c5:#d55181;--c6:#008300}}
.pie{display:flex;flex-wrap:wrap;align-items:center;gap:24px}.pie td i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.card.vn{max-width:720px;padding:8px 12px;margin-left:auto;margin-right:auto}.vn table{margin:0 auto;width:fit-content}.vn .ct{text-align:center}.vn .ct{font-size:12px;font-weight:500;color:var(--text-secondary)}.vn table{font-size:11.5px}.vn td,.vn th{padding:2px 8px}
body{margin:0;background:var(--page)}
.viz-root{font:14px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text-primary);background:var(--page)}
nav{position:sticky;top:0;z-index:2;background:var(--page);border-bottom:1px solid var(--border);padding:8px 28px;font-size:13px;overflow-x:auto;white-space:nowrap;scrollbar-width:none}nav::-webkit-scrollbar{display:none}
nav .tab.arch{font-size:12px;padding:3px 10px;color:var(--muted)}nav .tab.arch.on{color:var(--surface-1)}
nav a.tabl{display:inline-block;font-size:13px;padding:5px 14px;margin-right:6px;border-radius:999px;border:1px dashed var(--border);background:transparent;color:var(--text-secondary);text-decoration:none}
nav a.tabl.arch{font-size:12px;padding:3px 10px;color:var(--muted)}nav a.tabl:hover{color:var(--text-primary)}nav .nav-sep{color:var(--muted);font-size:12px;margin:0 6px 0 12px}
@media (max-width:700px){main{padding:14px 16px 48px}nav{padding:8px 16px}#tip{white-space:normal;max-width:90vw}.tiles{grid-template-columns:1fr 1fr}.tile .v{font-size:24px}}
nav a{color:var(--text-secondary);text-decoration:none;margin-right:16px}nav a:hover{color:var(--text-primary)}
main{max-width:1440px;margin:0 auto;padding:20px 28px 64px}
h1{font-size:24px;margin:8px 0 2px}.sub{color:var(--text-secondary);margin:0 0 20px}
h2{font-size:18px;margin:34px 0 6px}.note{color:var(--text-secondary);font-size:13px;margin:0 0 10px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tile .k{color:var(--text-secondary);font-size:13px}.tile .v{font-size:30px;font-weight:600;margin:4px 0}.tile .s{color:var(--muted);font-size:12px}
.rtset th,.rtset td{padding:4px 7px}.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px;margin:10px 0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(600px,1fr));gap:12px}.card.one{max-width:880px;margin-left:auto;margin-right:auto}.card.one.wide{max-width:none}
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
// reference tables (user 11-04): click a column header to sort the model rows by it (lower first; click again = reverse);
// rows without data-s (our best ladder checkpoint) stay at the bottom; missing values ("·", "—", "-") always last
document.querySelectorAll('table.rsort').forEach(tb=>{const ths=tb.querySelectorAll('thead th');
 ths.forEach((th,ci)=>{if(ci===0)return;th.style.cursor='pointer';th.title='click to sort / 点击排序';th.addEventListener('click',()=>{
  const first=tb.dataset.hi==='1'?-1:1,dir=th.dataset.dir?-parseInt(th.dataset.dir):first;   // data-hi: scores, first click = high to low
  ths.forEach(x=>{delete x.dataset.dir;x.querySelectorAll('.arr').forEach(a=>a.remove());});
  th.dataset.dir=String(dir);th.insertAdjacentHTML('beforeend',dir===1?'<span class="arr"> ▲</span>':'<span class="arr"> ▼</span>');
  const body=tb.querySelector('tbody'),rows=[...body.querySelectorAll('tr[data-s]')],rest=[...body.querySelectorAll('tr:not([data-s])')];
  const num=r=>{const c=r.children[ci];if(!c)return NaN;if(c.dataset.v!==undefined)return c.dataset.v===''?NaN:parseFloat(c.dataset.v);const v=parseFloat(c.textContent.replace('−','-').replace(/[^0-9.+\\-eE]/g,''));return v;};
  rows.sort((a,b)=>{const x=num(a),y=num(b);if(isNaN(x)&&isNaN(y))return 0;if(isNaN(x))return 1;if(isNaN(y))return -1;return dir*(x-y);});
  rows.concat(rest).forEach(r=>body.appendChild(r));});});});

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
document.querySelectorAll('.wtoggle button').forEach(b=>b.addEventListener('click',()=>{const card=b.closest('[data-sync]')||b.closest('.card');   // data-sync: one control bar for a whole tab (prod val loss)
 card.querySelectorAll('.wtoggle button').forEach(x=>x.classList.toggle('on',x===b));
 card.querySelectorAll('[data-win]').forEach(d=>d.hidden=d.dataset.win!==b.dataset.w);
 card.querySelectorAll('.stog').forEach(d=>d.style.display=b.dataset.w==='wd'?'none':'');
 card.querySelectorAll('.stog1').forEach(d=>d.style.display=b.dataset.w==='wd'?'':'none');}));
document.querySelectorAll('.rtoggle:not(.wtoggle):not(.stog1) button').forEach(b=>b.addEventListener('click',()=>{const card=b.closest('[data-sync]')||b.closest('.card');
 card.querySelectorAll('.rtoggle:not(.wtoggle):not(.stog1) button').forEach(x=>x.classList.toggle('on',x===b));
 card.querySelectorAll('[data-ref]').forEach(d=>d.hidden=d.dataset.ref!==b.dataset.r);}));
document.querySelectorAll('.toggle button').forEach(b=>b.addEventListener('click',()=>{const m=b.dataset.m,pane=b.closest('.msec')||b.closest('.pane');
 pane.querySelectorAll('.toggle button').forEach(x=>x.classList.toggle('on',x===b));
 pane.querySelectorAll('[data-metric]').forEach(d=>d.hidden=d.dataset.metric!==m);}));
// tabs; the URL hash names the open tab (#prod, #down, ...: links to a tab, and a reload keeps it)
function showTab(k){const b=document.querySelector('nav .tab[data-tab="'+k+'"]');if(!b)return;
 document.querySelectorAll('nav .tab').forEach(x=>x.classList.toggle('on',x===b));
 document.querySelectorAll('.pane').forEach(p=>p.hidden=p.dataset.pane!==k);}
document.querySelectorAll('nav .tab').forEach(b=>b.addEventListener('click',()=>{showTab(b.dataset.tab);try{history.replaceState(null,'','#'+b.dataset.tab)}catch(_){};window.scrollTo(0,0);}));
function openHash(){const k=location.hash.slice(1);if(!k)return;const a=document.querySelector('nav a.tabl[data-tabl="'+k+'"]');if(a){location.replace(a.href);return;}showTab(k);}
openHash();window.addEventListener('hashchange',openHash);
// y-range drawings (11-16 review): .rk-wrap data-rv = a chart's reference values in tick-box order; the drawing whose data-rk holds the key
// of the ticked set is shown ("lo,hi" = indices of its lowest / highest value, first on ties; "e" = nothing ticked)
function rkPick(kind,boxes){document.querySelectorAll('.rk-wrap[data-rkc="'+kind+'"]').forEach(w=>{let rv;try{rv=JSON.parse(w.dataset.rv)}catch(_){return;}let lo=-1,hi=-1;
 rv.forEach((v,i)=>{if(v===null||!boxes[i]||!boxes[i].checked)return;if(lo<0||v<rv[lo])lo=i;if(hi<0||v>rv[hi])hi=i;});
 const k=lo<0?'e':lo+','+hi;w.querySelectorAll(':scope>.rk-v').forEach(d=>{d.hidden=!d.dataset.rk.split(' ').includes(k);});});}
// downstream tab (user 11-09): tick boxes of "实验列表" show / hide each model's lines, points, tooltip rows and table rows
(function(){const cbs=[...document.querySelectorAll('.dm-cb')];if(!cbs.length)return;const st=document.createElement('style');document.head.appendChild(st);
 const upd=()=>{st.textContent=cbs.filter(c=>!c.checked).map(c=>{const v=CSS.escape(c.value);return 'section[data-pane="down"] [data-dm="'+v+'"],#tip [data-dm="'+v+'"]{display:none}';}).join(' ');
  const bb=cbs.filter(c=>c.dataset.b!==undefined).sort((a,b)=>a.dataset.b-b.dataset.b);rkPick('dm',bb);const bm=bb.map(c=>c.checked?'1':'0').join('');document.querySelectorAll('.tk-wrap').forEach(w=>{w.dataset.bm='b'+bm;});};   // step charts: the y range of the ticked baselines (user 11-16)
 cbs.forEach(c=>c.addEventListener('change',upd));
 document.querySelectorAll('.dm-all').forEach(b=>b.addEventListener('click',()=>{cbs.forEach(c=>{c.checked=b.dataset.v==='1';});upd();}));upd();})();
// prod iso-doc (user 11-16): tick boxes of its "实验列表" show / hide the reference lines; the drawing of the ticked set (its y range) is shown,
// and the end labels of the ticked lines are spaced at least 13 px apart
(function(){const cbs=[...document.querySelectorAll('.pi-cb')];if(!cbs.length)return;const st=document.createElement('style');document.head.appendChild(st);
 const upd=()=>{st.textContent=cbs.filter(c=>!c.checked).map(c=>{const v=CSS.escape(c.value);return 'section[data-pane="prod"] [data-pi="'+v+'"],#tip [data-pi="'+v+'"]{display:none}';}).join(' ');
  const on=new Set(cbs.filter(c=>c.checked).map(c=>c.value));const bm='b'+cbs.map(c=>c.checked?'1':'0').join('');rkPick('pi',cbs);
  document.querySelectorAll('.pi-wrap').forEach(w=>{w.dataset.bm=bm;});
  document.querySelectorAll('.pi-v svg').forEach(s=>{let prev=-1e9;[...s.querySelectorAll('text.pi-lab')].filter(t=>on.has(t.dataset.pi)).sort((x,y)=>x.dataset.y0-y.dataset.y0)
   .forEach(t=>{const y=Math.max(+t.dataset.y0,prev+13);t.setAttribute('y',(y+4).toFixed(1));prev=y;});});};
 cbs.forEach(c=>c.addEventListener('change',upd));upd();})();
// "实验列表": rows of the early checkpoints, folded (user 11-16)
document.querySelectorAll('.dm-fold-btn').forEach(b=>b.addEventListener('click',()=>{const rs=[...b.closest('table').querySelectorAll('tr.dm-fold')];
 const open=rs.length>0&&rs[0].hidden;rs.forEach(r=>{r.hidden=!open;});b.textContent=open?b.dataset.close:b.dataset.open;}));
const tip=document.getElementById('tip');
// one delegated handler for every hover target (11-16 review: 10k+ per-element listeners before)
let tipEl=null;const tipOff=()=>{tip.style.display='none';tipEl=null;};
document.addEventListener('mousemove',ev=>{const e=ev.target&&ev.target.closest?ev.target.closest('[data-tip]'):null;
 if(!e){if(tipEl)tipOff();return;}if(e!==tipEl){tip.innerHTML=e.dataset.tip;tipEl=e;}
 tip.style.display='block';tip.style.left=Math.max(8,Math.min(ev.clientX+12,window.innerWidth-tip.offsetWidth-8))+'px';
 tip.style.top=Math.max(8,Math.min(ev.clientY+12,window.innerHeight-tip.offsetHeight-8))+'px';});
document.documentElement.addEventListener('mouseleave',tipOff);window.addEventListener('scroll',tipOff,{passive:true});window.addEventListener('blur',tipOff);
// sticky bars below the nav: its real height (it can grow on narrow screens)
(function(){const nv=document.querySelector('nav');if(!nv)return;const f=()=>document.documentElement.style.setProperty('--nav-h',nv.offsetHeight+'px');f();window.addEventListener('resize',f);})();
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
    if ("SWA_FIT_SINK_W512_R3", "q0") in R:               # R3T q0 = R3 q0 (identical argv; the card reuses it), user 11-03
        R.setdefault(("SWA_FIT_SINK_W512_R3T", "q0"), R[("SWA_FIT_SINK_W512_R3", "q0")])
        TRAIN.setdefault(("SWA_FIT_SINK_W512_R3T", "q0"), TRAIN.get(("SWA_FIT_SINK_W512_R3", "q0")))
    if ("SWA_FIT_SINK_W512_R3T", "q2") in R:              # R3 q2 = R3T q2: L 24 % 4 == 0, so both countings give globals 4/8/../24; the R3T q2 run
        R.setdefault(("SWA_FIT_SINK_W512_R3", "q2"), R[("SWA_FIT_SINK_W512_R3T", "q2")])   # argv == swa/golden/..._r3-q2.json except run paths
        TRAIN.setdefault(("SWA_FIT_SINK_W512_R3", "q2"), TRAIN.get(("SWA_FIT_SINK_W512_R3T", "q2")))   # (checked 11-04, user "之前我记得你说有的")
    for (ch, q), m in list(R.items()):
        if ch.startswith("MUON_LR"):
            R.setdefault(("OPT_" + ch, q), m)
            TRAIN.setdefault(("OPT_" + ch, q), TRAIN.get((ch, q)))
        sws = re.match(r"^MUONH_FIT_DSV3_SWS_NE_(.+?)(_S\d+)?$", ch)        # SWS-base NE runs: muonh_fit_dsv3_sws_ne_1em5[_x][-s2] -> ROUTER_NE_1EM5[_X]_SWS[_S2]
        if sws:
            b = f"ROUTER_NE_{sws.group(1)}_SWS{sws.group(2) or ''}"
            R.setdefault((b, q), m); TRAIN.setdefault((b, q), TRAIN.get((ch, q)))
            continue
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


def line_chart(series, xs, ylab, fmt, w=760, h=360, y0=None, labels=True, zero=False, seeds=None, yrange=None, ystep=None, notes=None, tipfmt=None, ylog=False, endlabels=None):
    """series = [(name, colour var, [(x, y, tip)], dashed)] on a log-x axis. seeds = {(name, x): [y per seed]}: drawn as hollow
    circles in the series colour (the line and the filled point are the seed mean). ylog: log y as well (log-log; positive values only)."""
    seeds = seeds or {}
    ml, mr, mt, mb = 52, 96, 12, 38
    if endlabels:
        mr = 150                                         # room for the end-of-line labels
    allx = [p[0] for _, _, pts, _ in series for p in pts] or [1, 2]
    ally = [p[1] for _, _, pts, _ in series for p in pts] + [y for v in seeds.values() for y in v] + ([0.0] if zero else []) or [0, 1]
    lx0, lx1 = math.log(min(xs.values())) - 0.15, math.log(max(xs.values())) + 0.15
    ylo, yhi = min(ally), max(ally)
    if y0 is not None:
        ylo = min(ylo, y0)
    pad = (yhi - ylo) * 0.08 or 0.01
    ylo, yhi = ylo - (0 if y0 is not None else pad), yhi + pad
    if yrange:                                           # fixed y range (widened only if a point falls outside it)
        ylo, yhi = min(yrange[0], min(ally)), max(yrange[1], max(ally))
    X = lambda x: ml + (math.log(x) - lx0) / (lx1 - lx0) * (w - ml - mr)
    Y = lambda y: mt + (yhi - y) / (yhi - ylo) * (h - mt - mb)
    nt = round((yhi - ylo) / ystep) if ystep else 4
    ticks = [ylo + (yhi - ylo) * i / nt for i in range(nt + 1)]
    if ylog:   # log-log (Boxiang 11-07): padding in log space, ticks at round values (0.05 / 0.1 / 0.2 / 0.5 steps)
        lg = [math.log(v) for v in ally if v > 0]
        pd = (max(lg) - min(lg)) * 0.08 or 0.01
        ylo, yhi = math.exp(min(lg) - pd), math.exp(max(lg) + pd)
        Y = lambda y: mt + (math.log(yhi) - math.log(y)) / (math.log(yhi) - math.log(ylo)) * (h - mt - mb)
        st = next(s for s in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0) if (yhi - ylo) / s <= 8)
        ticks = [k * st for k in range(math.ceil(ylo / st), math.floor(yhi / st) + 1)]
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">']
    for yv in ticks:                                     # recessive grid + y ticks
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
    if endlabels:                                        # {series name: text} at the line end, series colour, pushed apart at the same x
        el = sorted([[X(p[-1][0]), Y(p[-1][1]), endlabels[n], c] for n, c, p, _ in series if p and n in endlabels], key=lambda r: (round(r[0]), r[1]))
        for i in range(1, len(el)):
            if abs(el[i][0] - el[i - 1][0]) < 4 and el[i][1] - el[i - 1][1] < 13:
                el[i][1] = el[i - 1][1] + 13
        for x, y, txt, c in el:
            o.append(f'<text x="{x + 8:.1f}" y="{y + 4:.1f}" style="fill:var({c});font-weight:600;paint-order:stroke;stroke:var(--surface-1);stroke-width:3px">{esc(txt)}</text>')
    for x, y, txt in notes or []:                          # annotations next to single points, e.g. "(TPP=100)"
        o.append(f'<text class="lbl" x="{X(x)+9:.1f}" y="{Y(y)+4:.1f}">{esc(txt)}</text>')
    qx = sorted((x, q) for q, x in xs.items())            # one hover band per rung: tooltip lists every series at that x
    for i, (x, q) in enumerate(qx):
        a = X(x) - (X(x) - X(qx[i - 1][0])) / 2 if i else ml
        b = X(x) + (X(qx[i + 1][0]) - X(x)) / 2 if i + 1 < len(qx) else w - mr
        rows = [(n, col, y) for n, col, pts, _ in series for px, y, _ in pts if px == x]
        if not rows:
            continue
        tip = f"<b>{q} · {N_LABEL[q]}</b>" + "".join(
            f'<br><span style="color:var({col})">●</span> {esc(n)}: {tipfmt(y) if tipfmt else f"{y:.4f}"}' + (f" (seeds " + " / ".join(f"{v:+.4f}" for v in seeds[(n, x)]) + ")" if len(seeds.get((n, x), [])) > 1 else "")
            for n, col, y in sorted(rows, key=lambda r: r[2], reverse=True))   # largest on top
        o.append(f'<rect class="hit" x="{a:.1f}" y="{mt}" width="{b-a:.1f}" height="{h-mt-mb}" data-tip="{esc(tip)}"/>')
    o.append("</svg>")
    return "".join(o)


def swatch_bg(c, dashed=False):
    """Legend swatch background: solid, or dashed like the chart line (user 11-03: dashed lines need dashed legend swatches)."""
    return f"repeating-linear-gradient(90deg,var({c}) 0 4px,transparent 4px 7px)" if dashed else f"var({c})"


def legend(items):
    """items = [(name, colour var)] or [(name, colour var, dashed)] or [(name, colour var, dashed, dot)]; dot = a series with a
    single point so far: a circle swatch (user 11-03 "用circle")."""
    dot = ";width:9px;height:9px;border-radius:50%"
    return '<div class="legend">' + "".join(f'<span><i style="background:{swatch_bg(it[1], len(it) > 2 and it[2] and not (len(it) > 3 and it[3]))}{dot if len(it) > 3 and it[3] else ""}"></i>{esc(it[0])}</span>' for it in items) + "</div>"


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


# KD teacher MiMo-2.6 vs MiMo-2.5 (user 2026-10-30, revised): both teacher families against ONE zero line = the CE of the 2.6 group
# (OPT_MUONH: MuonH, ArchScale LR, corpus v2; q0 = mean of its 3 seeds); every point = the arm's seed mean - that CE (no per-seed dots).
# 2.6 = Boxiang's KD_MUONH_A0P8_* (+ our q3 S128Y), 2.5 = our tc_small arms (fitted LR, 45B KD corpus). (variant, 2.6 ID, 2.5 ID)
# REVISED 2026-10-30 19:3x (user: "不能和没有新 fit LR 的做对比"): fitted MuonH LR only - zero line MUONH_FIT, 2.6 = the fit-LR tc_small twins
# (KD_MUONH_FIT_A0P8_*_TCS_MIMO26: same LR, 45B KD corpus, seeds and data order as the 2.5 arms -> only the teacher differs).
# REVISED 2026-10-30 21:5x (user: "两个 2.6 的结果…用三个不同色系"; "要写清楚，那些是老LR，那些是新的"): three families, each against the CE
# trained with the SAME LR law - blue = MiMo-2.6 old ArchScale LR (KD_MUONH_A0P8_*, corpus v2 KD data) - OPT_MUONH; green = MiMo-2.6
# new fitted LR, orange = MiMo-2.5 new fitted LR (tc_small twins). Labels name the LR law everywhere.
# REVISED 22:1x (user: "我们都是对比同一个老LR 2.6 的 CE，不比新LR的CE"): ALL three families minus the same CE = OPT_MUONH (old ArchScale LR).
KD25_CE = "MUONH_FIT"
KD25_VARIANTS = [(v, f"KD_MUONH_FIT_A0P8_{v}_TCS_MIMO26", f"KD_MUONH_FIT_A0P8_{v}_TCS_MIMO25") for v in ("S64", "S64Y", "S128Y")]
KD25_COL26 = {"S64": "--g07", "S64Y": "--g08", "S128Y": "--g09"}       # green family = MiMo-2.6, new fitted LR
KD25_COL25 = {"S64": "--a07", "S64Y": "--a08", "S128Y": "--a09"}       # orange family = MiMo-2.5, new LR
KD25_COLOLD = {"S64": "--k07", "S64Y": "--k08", "S128Y": "--k09"}      # blue family = MiMo-2.6, old ArchScale LR
KD25_FAMS = [  # (Change ID of variant v, colours, LR-matched CE, legend-name key, family-line key)
    ("KD_MUONH_A0P8_{v}", KD25_COLOLD, "OPT_MUONH", "kd25_nold", "kd25_fold"),
    ("KD_MUONH_FIT_A0P8_{v}_TCS_MIMO26", KD25_COL26, "OPT_MUONH", "kd25_n26", "kd25_f26"),
    ("KD_MUONH_FIT_A0P8_{v}_TCS_MIMO25", KD25_COL25, "OPT_MUONH", "kd25_n25", "kd25_f25"),
]
OUR_RUNS = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/scaling_ladder/qwen3_tpp200"
KD25_PACKED = ([("MUONH_FIT", "muonh_fit", "both"), ("OPT_MUONH", "muonh", "box"), ("KD_MUONH_A0P8_S128Y", "muonh_kd_a0p8_s128y", "ours")]
               + [(c26, "muonh_kd_a0p8_" + v.lower() + "_fit_tcs_mimo26", "ours") for v, c26, _ in KD25_VARIANTS]
               + [(c25, "muonh_kd_a0p8_" + v.lower() + "_fit_tcs_mimo25", "ours") for v, _, c25 in KD25_VARIANTS])   # (ID, run stem, where)


def _seed_vals(R, ch, q, mk):
    return [x for x in (val(R, _seed_id(ch, k), q, mk) for k in (1, 2, 3)) if x is not None]


def _smean(R, ch, q, mk):
    v = _seed_vals(R, ch, q, mk)
    return sum(v) / len(v) if v else None


# Doc-mask ablation (user 10-31: "逻辑是一样的…基线是 CE…mask vs not mask 就够了"): drawn like the KD teacher chart - each line = seed mean
# minus the same CE (OPT_MUONH), q0 seeds averaged into one point, y = data min .. max. Doc tab only: the arm's packed validation runs masked.
DM_CE = "OPT_MUONH"
DM_LINES = [("KD_MUONH_A0P8_S64Y", "--k08", "dm_nb"),                  # user 10-31: also Boxiang's old-LR S64Y (same KD cache, no mask) as a baseline
            ("KD_MUONH_A0P8_S64Y_FIT", "--v8", "dm_n0"), ("KD_MUONH_A0P8_S64Y_FIT_DOCMASK", "--s4", "dm_n1")]   # (Change ID, colour, name key)


def docmask_section(R, t, metrics):
    def chart(mk, mname):
        series, xs = [], {}
        for ch, col, lab in DM_LINES:
            pts = []
            for q in RUNGS:
                a, c = _smean(R, ch, q, mk), _smean(R, DM_CE, q, mk)
                if a is None or c is None:
                    continue
                y, n = a - c, len(_seed_vals(R, ch, q, mk)); xs[q] = N_ACTIVE[q]
                pts.append((N_ACTIVE[q], y, f"{esc(t[lab])} · {q}<br>{mname} {y:+.4f}（vs {DM_CE}）" + (f"（{n} seeds 平均）" if n > 1 else "")))
            if pts:
                series.append((t[lab], col, pts, False))
        if not series:
            return f'<p class="note">{esc(t["pending"])}</p>'
        return legend([(n, c, d) for n, c, _, d in series]) + line_chart(series, xs, t["kd25_y"], lambda v: f"{v:+.3f}", labels=False)
    if not any(_smean(R, ch, q, metrics[0][0]) is not None for ch, *_ in DM_LINES for q in RUNGS):
        return ""
    blocks = "".join(f'<div data-metric="{m}"{"" if j == 0 else " hidden"}>{chart(m, n)}</div>' for j, (m, n) in enumerate(metrics))
    fams = (f'<div class="kd25fam" style="font-size:12.5px;line-height:1.7;margin:2px 0 6px">'
            + "".join(f'<div><i style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;background:var({col})"></i>{t[lab + "_d"]}</div>' for _, col, lab in DM_LINES)
            + f'<div style="color:var(--muted)">{t["dm_f0"]}</div></div>')
    return f"""<div class="msec"><h2>{esc(t['dm_h'])}</h2><p class="note">{t['dm_note']}</p>{mtoggle(metrics)}
<div class="card one"><div class="ct">{esc(t['dm_chart'])}</div>{fams}{blocks}</div></div>"""


def kd25_section(R, t, metrics, head_mk, head_name, packed=False):
    V = [v for v, *_ in KD25_VARIANTS]
    def d(ch, ce, q, mk):
        a, c = _smean(R, ch, q, mk), _smean(R, ce, q, mk)
        return None if a is None or c is None else a - c
    def chart(mk, mname):
        series, xs, half, full = [], {}, [], []
        for pat, cols, ce, lab, _ in KD25_FAMS:
            for v in V:
                ch, name, pts = pat.format(v=v), t[lab].format(v=v), []
                for q in RUNGS:
                    y = d(ch, ce, q, mk)
                    if y is None:
                        continue
                    if q == "q3" and "_TCS_" in ch:   # tc_small q3 ran TPP 100: not drawn (user 11-03: "太confusing")
                        continue
                    xs[q] = N_ACTIVE[q]; n = len(_seed_vals(R, ch, q, mk))
                    pts.append((N_ACTIVE[q], y, f"{esc(name)} · {q}<br>{mname} {y:+.4f}（vs {ce}）" + (f"（{n} seeds 平均）" if n > 1 else "")))
                if pts:
                    series.append((name, cols[v], pts, False))
        if not series:
            return f'<p class="note">{esc(t["pending"])}</p>'
        return legend([(n, c, d) for n, c, _, d in series]) + line_chart(series, xs, t["kd25_y"], lambda v: f"{v:+.3f}", labels=False)   # user 10-31: y = data min .. max (no fixed range, no 0)
    if not any(d(pat.format(v=v), ce, q, metrics[0][0]) is not None for pat, _, ce, *_ in KD25_FAMS for v in V for q in RUNGS):
        return ""
    blocks = "".join(f'<div data-metric="{m}"{"" if j == 0 else " hidden"}>{chart(m, n)}</div>' for j, (m, n) in enumerate(metrics))
    sw = lambda cols: "".join(f'<i style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:2px;background:var({cols[v]})"></i>' for v in V)
    pend = lambda pat, ce: "" if any(d(pat.format(v=v), ce, q, metrics[0][0]) is not None for v in V for q in RUNGS) else f' <span style="color:var(--muted)">{t["kd25_pend"]}</span>'
    fams = (f'<div class="kd25fam" style="font-size:12.5px;line-height:1.7;margin:2px 0 6px">'
            + "".join(f'<div>{sw(cols)} {t[fk]}{pend(pat, ce)}</div>' for pat, cols, ce, _, fk in KD25_FAMS)
            + f'<div style="color:var(--muted)">{t["kd25_f0"]}</div></div>')
    note = t["kd25_note"] + (" " + t["kd25_pnote"] if packed else "")
    return f"""<div class="msec"><h2>{esc(t['kd25_h'])}</h2><p class="note">{note}</p>{mtoggle(metrics)}
<div class="card one"><div class="ct">{esc(t['kd25_chart'])}</div>{fams}{blocks}</div></div>"""   # numbers table removed (user 10-30)


def _final_vals(run_dir):
    """Packed final validation of one run dir (only once the run reached --train-iters)."""
    try:
        argv = open(os.path.join(run_dir, "argv.txt")).read().split()
        ti = int(argv[argv.index("--train-iters") + 1])
    except (OSError, ValueError):
        return {}
    vals = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "logs", "*.log"))):
        try:
            txt = open(f, errors="replace").read()
        except OSError:
            continue
        for m in FINAL_RE.finditer(txt):
            if int(m.group(2)) == ti:
                vals[m.group(1)] = float(m.group(3))
    return vals


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
    for ch, stem, where in KD25_PACKED:                # KD teacher section: CE of the 2.6 group (3 seeds at q0), 2.6 and 2.5 arms, every seed
        roots = {"box": [RUNS], "ours": [OUR_RUNS], "both": [RUNS, OUR_RUNS]}[where]
        for q in RUNGS:
            for k in (1, 2, 3):
                for root in roots:
                    v = _final_vals(os.path.join(root, f"qwen3_tpp200_{stem}-{q}-tpp200-s{k}"))
                    if all(x in v for x, _ in PSETS):
                        m = {key: {"loss": v[x]} for x, key in PSETS}
                        m["mean6"] = {"loss": sum(v[x] for x, _ in PSETS) / len(PSETS)}
                        R.setdefault((_seed_id(ch, k), q), m)
                        break
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
        o2 = legend([(n, c, d) for n, c, _, d in opt]) + line_chart(
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
    return legend([(n, c, d) for n, c, _, d in series]) + line_chart(series, xs, "Δloss (KD − CE)", lambda v: f"{v:+.3f}", labels=False)


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
    if G.get("nominal") or not tr or not tr.get("lr") or not tr.get("num_layers") or not tr.get("train_iters"):
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
             ("SWA_FIT_SINK_W512", "MUONH_FIT"), ("SWA_FIT_SINK_W128", "MUONH_FIT"), ("SWA_FIT_SINK_W512_R3", "MUONH_FIT"), ("SWA_FIT_SINK_W512_R3T", "MUONH_FIT"),   # window axis (+ 3:1 ratio axis, 10-30) on the sink hybrid (Boxiang 10-29), q0 + q1
             ("SWA_FIT_DSV3", "MUONH_FIT_DSV3"), ("MUONH_FIT_DSV3_SWS", "MUONH_FIT_DSV3")]   # MUONH_SPLITFC1 (optimizer change, old law) dropped from the SWA section (user title)   # (arm, its baseline): architecture arms on the MuonH stack


ARCH_COL = {"SWA": "--muted", "SWA_FIT": "--c1", "SWA_FIT_ROPE": "--c3", "SWA_FIT_W4K": "--s4", "SWA_FIT_W1K": "--c5", "SWA_FIT_SINK": "--c2",
            "SWA_FIT_SINK_W1K": "--a09", "SWA_FIT_SINK_W512": "--a07", "SWA_FIT_SINK_W128": "--c4", "SWA_FIT_SINK_W512_R3": "--g08", "SWA_FIT_SINK_W512_R3T": "--g09", "SWA_FIT_DSV3": "--c6", "MUONH_FIT_DSV3_SWS": "--s5", "MUONH_SPLITFC1": "--o-adamw"}
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


def arch_table(R, t, arms=ARCH_ARMS):
    rows = []
    for arm, base in arms:
        qs = [q for q in RUNGS + ["q6"] if (arm, q) in R and (base, q) in R]
        if not qs:
            continue
        laws = {_law_of(c, q) for q in qs for c in (arm, base)} - {None}
        lawtxt = " / ".join(t["arch_law"][x] for x in sorted(laws)) if laws else "·"
        rows.append(f'<tr><td class="l"><span style="color:var({ARCH_COL.get(arm, "--muted")})">■</span> {esc(t["arch_name"][arm].split(" −")[0].split("（")[0] if False else t["arch_short"][arm])}</td>'
                    f'<td class="l"><code>{arm}</code></td><td class="l" style="white-space:normal;max-width:420px">{esc(t["arch_desc"][arm])}</td>'
                    f'<td class="l"><code>{base}</code></td><td class="l">{lawtxt}</td><td class="l">{", ".join(q + (f" ×{len(_paired(R, arm, base, q, '00_overall_tsqw'))}" if len(_paired(R, arm, base, q, '00_overall_tsqw')) > 1 else "") for q in qs)}</td></tr>')
    head = "".join(f'<th class="l">{esc(x)}</th>' for x in t["arch_thead"])
    return f'<div class="card"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def arch_section(R, t, metrics, arms=ARCH_ARMS, key="arch"):
    """Line chart (user): x = rung, y = arm - its baseline, one line per arm; own metric toggle, overall_tsqw first."""
    def chart(mkey, mname, sub, w, h):
        series, xs, seeds = [], {}, {}
        for arm, base in sub:
            pts = []
            for q in RUNGS + ["q6"]:
                d = _paired(R, arm, base, q, mkey)              # per-seed paired differences (seeds 1-3 where arm and control both exist)
                if not d:
                    continue
                v = sum(d) / len(d)
                xs[q] = N_ACTIVE[q]
                pts.append((N_ACTIVE[q], v, f"{esc(t['arch_name'][arm])} · {q}<br>{mname} {v:+.4f}" + (f"（{len(d)} seeds）" if len(d) > 1 else "")))
                # no hollow per-seed points (user 11-03): only the filled seed mean is drawn
            if pts:
                series.append((t["arch_name"][arm], ARCH_COL.get(arm, "--muted"), pts, arm in ARCH_DASH))
        if not series:
            return ""
        return legend([(n, c, d) for n, c, _, d in series]) + line_chart(series, xs, t["arch_y"], lambda v: f"{v:+.3f}", w=w, h=h, labels=False, zero=True, seeds=seeds)
    bases = list(dict.fromkeys(b for _, b in arms))      # one chart per zero line (user 11-03: a different zero line gets its own chart)
    subs = [[(a, b) for a, b in arms if b == base] for base in bases]
    widths = [900, 450] if len(bases) == 2 else [1240] * len(bases)
    card = lambda base, body: (f'<div class="card" style="min-width:0;width:100%;margin:10px 0;box-sizing:border-box">'
                               f'<div class="ct">{esc(t["arch_c"].get(base, base))}</div>{body}</div>')
    grid = lambda m, n: (f'<div style="display:grid;grid-template-columns:{" ".join("minmax(0,2fr)" if k == 0 and len(bases) == 2 else "minmax(0,1fr)" for k in range(len(bases)))};gap:14px;align-items:stretch">'
                         + "".join(card(base, chart(m, n, sub, wd, 420)) for base, sub, wd in zip(bases, subs, widths)) + "</div>")
    blocks = "".join(f'<div data-metric="{m}"{"" if j == 0 else " hidden"}>{grid(m, n)}</div>' for j, (m, n) in enumerate(metrics))
    if key != "arch" and not any(_paired(R, a, b, q, metrics[0][0]) for a, b in arms for q in RUNGS):
        return ""
    return f"""<div class="msec"><h2>{esc(t[key + '_h'])}</h2><p class="note">{t[key + '_note']}</p>{arch_table(R, t, arms)}{mtoggle(metrics)}
{blocks}<p class="note">{esc(t[key + '_chart'])}</p></div>"""



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
OUR_RUNS_DIR = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/scaling_ladder/qwen3_tpp200"
PROD_RUNS_DIR = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/scaling_ladder/prod_v1"   # Runlong's PROD v1 runs (11-06)
PREPACK_RUNS_DIR = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/runs/scaling_ladder/prod_v1_prepack"   # E2X KD A / B(QBX) on the tc_full prepack (11-06)
RT_HEAT_ARMS = [("rt_hm_soft", "muonh_fit", "--v8"), ("rt_hm_a", "muonh_fit_dsv3", "--s2"), ("rt_hm_b2", "muonh_fit_dsv3_rtadamw_rt0p19", "--s4"),
                ("rt_hm_b3", "muonh_fit_dsv3_rt0p19", "--s3"), ("rt_l_biasonly", "muonh_fit_dsv3_biasonly", "--s5"),
                ("rt_l_rtmuonh", "muonh_fit_dsv3_rtmuonh", "--s1")]   # same names / colours as the loss chart above (user 11-02)   # AdamW arm C (v1_dsv3_onetree) dropped (user 11-02)
N_EXPERTS = 128
RT_STATS = [  # (key, regex on the summary line, transform, unit)
    ("vio_mean", r"MaxVio mean ([0-9.]+)", None, ""), ("vio_max", r"MaxVio mean [0-9.]+ max ([0-9.]+)", None, ""),
    ("dead", r"dead experts (\d+)", "dead", "%"), ("bmax", r"\|b\|max ([0-9.]+)", None, ""),
    ("sat", r"frac mean ([0-9.]+)%", None, "%"), ("satpos", r"z>\+4 mean ([0-9.]+)%", None, "%"), ("pinned", r"top-8 pinned ([0-9.]+)%", None, "%"),
    ("zmax", r"max \|z\| ([0-9.]+)", None, ""), ("upd", r"gate update/weight \(flag\): mean ([0-9.]+)", None, "")]


def _run_logs(exp, q):
    """Training logs of run <exp> at rung q (seed 1), oldest first: Boxiang's, ours, then Runlong's PROD v1 dirs."""
    if exp.startswith("qwen3_"):                         # a full run stem (prod_v1_prepack arms: qwen3_tcfull_prepack_32k_..., 11-06)
        return sorted(glob.glob(os.path.join(PREPACK_RUNS_DIR, f"{exp}-{q}-iters*-s1", "logs", "*.log")), key=os.path.getmtime)
    files = []
    for root, pat in ((RUNS_DIR, "tpp200"), (OUR_RUNS_DIR, "tpp200"), (PROD_RUNS_DIR, "iters*")):   # Runlong's later arms (e.g. biasonly, rtmuonh) live in our run dir (user 11-02); prod v1 runs: -iters<N>- (11-06)
        files = sorted(glob.glob(os.path.join(root, f"qwen3_tpp200_{exp}-{q}-{pat}-s1", "logs", "*.log")), key=os.path.getmtime)
        if files:
            break
    return files


def train_loss_log(exp, q, key="lm loss"):
    """{iteration: <key>} from the run's training log lines (a later segment overwrites an earlier one); key e.g. "lm loss", "kd loss"."""
    out = {}
    rx = re.compile(r"iteration\s+(\d+)/\s*\d+ \|.*?\| " + re.escape(key) + r": ([0-9.E+-]+)")
    for f in _run_logs(exp, q):
        try:
            for line in open(f, errors="replace"):
                m = rx.search(line)
                if m:
                    out[int(m.group(1))] = float(m.group(2))
        except OSError:
            pass
    return out


def router_summary_log(exp, q):
    """{step: {stat: value}} from the '[router health] step' summary lines of the run (seed 1), or {}."""
    out = {}
    files = _run_logs(exp, q)
    n_exp = 512 if ("latent2_e2x" in exp or "prepack_32k_e2x" in exp) else 256 if ("latent2" in exp or "dpsk_e2x" in exp) else N_EXPERTS   # dead % of (layers x this arm's experts)
    for f in files:
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
                        rec[key] = 100.0 * v / (nl * n_exp) if tr == "dead" else v
                out[step] = rec
        except OSError:
            pass
    return out


# range tabs of the router-health charts (user 11-08 "后 90% step / 前 10% step / 全 step")
RT_RANGE = {"zh": [("r90", "后 90% step"), ("r10", "前 10% step"), ("rall", "全 step")],
            "en": [("r90", "Last 90% of steps"), ("r10", "First 10% of steps"), ("rall", "All steps")]}


def router_trends(t, arms=None, rungs=("q0", "q1", "q2", "q3"), head=True, show_legend=True, early=None, big=False, outside=False, qtabs=False, defs=True):
    """One line chart per rung (x = training step, one line per arm), metric toggle; summary statistics over all layers.
    arms = [(name, run exp, colour var, dashed)]; default = the Router chapter arms. head=False: the card only (reused by the dense-layer-0 chapter).
    early = (fraction, [(win key, label) x 2]): a range tab row on top of the card, "last 1 - fraction" | "first fraction" of the steps (x and y rescaled;
    user 11-08 "step 的第一个tab是 后 90%，然后是 前10%";
    user 11-08 "把前10%的step单独列个tab ... tab放在chart上面"); the fraction is of the rung's last logged step over all arms. A third entry
    = all steps (user 11-08 "后 90% step / 前 10% step / 全 step"; RT_RANGE).
    qtabs: one rung at a time behind a q1 / q2 / q3 tab row at the top of the card ([data-ref] panes; needs outside=True so the statistic buttons
    do not share the .rtoggle; user 11-08 "在这个上面加个 q1 q2 q3 的tab").
    big: one chart per row at the val_loss chart size (centred .card.one, 760 x 360; user 11-08 "图像弄大，像我们的val_loss chart一样的大小").
    outside: the statistic buttons (MaxVio / dead expert / ...) sit above and outside the card like the val_loss metric toggle (.msec + .toggle,
    panes [data-metric]); the card keeps only the range tab (user 11-08 "maxvio / deadexpert，你放在图像上面吧 ... 而且在外面")."""
    arms = arms or [(t[key], exp, col, False) for key, exp, col in RT_HEAT_ARMS]
    data = {(q, name): router_summary_log(exp, q) for q in rungs for name, exp, _, _ in arms}
    rungs = [q for q in rungs if any(data[(q, n)] for n, _, _, _ in arms)]
    if not rungs:
        return ""
    unit = {k: u for k, _, _, u in RT_STATS}
    lastst = {q: max((st for n, _, _, _ in arms for st in data[(q, n)]), default=0) for q in rungs}
    def draw(mk, rng=None, qs=None):                    # rng = (lo, hi) fractions of the rung's last logged step: lo < step <= hi; qs = rungs drawn
        charts = []
        for q in (qs or rungs):
            L0, L1 = (lastst[q] * rng[0], lastst[q] * rng[1]) if rng else (None, None)
            series = [(name, col, sorted((st, r[mk]) for st, r in data[(q, name)].items() if mk in r and (L0 is None or (st > L0 or L0 == 0) and st <= L1)), dash)
                      for name, _, col, dash in arms]
            series = [x for x in series if x[2]]
            if not series:
                continue
            xs = [st for _, _, p, _ in series for st, _ in p]; ys = [v for _, _, p, _ in series for _, v in p]
            x1 = max(xs) or 1; x0 = min(xs) if rng and rng[0] > 0 else 0; ylo, yhi = 0.0, max(ys) * 1.08 or 1.0
            x1 = max(x1, x0 + 1)
            w, h, ml, mr, mt, mb = (760, 360, 52, 14, 24, 34) if big else (360, 200, 44, 8, 22, 26)
            fs = 11 if big else 9
            X = lambda v: ml + (v - x0) / (x1 - x0) * (w - ml - mr); Y = lambda v: mt + (yhi - v) / (yhi - ylo) * (h - mt - mb)
            o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img"><text x="{ml}" y="14" style="font-size:{fs + 3}px;font-weight:600">{q}</text>']
            for k in range(4):
                v = ylo + (yhi - ylo) * k / 3
                o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--grid)"/><text x="{ml - 4}" y="{Y(v) + 3:.1f}" text-anchor="end" style="font-size:{fs}px">{v:.{3 if mk == "upd" else 2}g}</text>')
            o.append(f'<text x="{w - mr}" y="{h - 6}" text-anchor="end" style="font-size:{fs}px">step {x1:,}</text><text x="{ml}" y="{h - 6}" style="font-size:{fs}px">{f"step {x0:,}" if x0 else "0"}</text>')
            if mk == "bmax":
                o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(1.0):.1f}" y2="{Y(1.0):.1f}" stroke="var(--bad)" stroke-dasharray="4 3"/>' if yhi > 1 else "")
            if mk in ("sat", "satpos", "pinned"):
                o.append(f'<line x1="{ml}" x2="{w - mr}" y1="{Y(5.0):.1f}" y2="{Y(5.0):.1f}" stroke="var(--bad)" stroke-dasharray="4 3"/>' if yhi > 5 else "")
            for name, col, pts, dash in series:
                d = " ".join(f"{'M' if i == 0 else 'L'}{X(st):.1f},{Y(v):.1f}" for i, (st, v) in enumerate(pts))
                o.append(f'<path d="{d}" fill="none" stroke="var({col})" stroke-width="1.8"{' stroke-dasharray="5 3"' if dash else ''}/>')
                st, v = pts[-1]
                o.append(f'<circle cx="{X(st):.1f}" cy="{Y(v):.1f}" r="3.5" fill="var({col})"/>')
            nb = 24                                              # hover bands: every arm at that step, largest first
            for bi in range(nb):
                s0, s1 = x0 + (x1 - x0) * bi / nb, x0 + (x1 - x0) * (bi + 1) / nb
                rows = []
                for name, col, pts, _ in series:
                    near = [p for p in pts if s0 <= p[0] < s1 or (bi == nb - 1 and p[0] == x1)]
                    if near:
                        rows.append((near[-1][1], f'<span style="color:var({col})">●</span> {esc(name)}: {near[-1][1]:.{4 if mk == "upd" else 3}g}{unit[mk]}'))
                if rows:
                    tip = f"<b>{q} · step ~{int(s1):,}</b><br>" + "<br>".join(r for _, r in sorted(rows, key=lambda r: -r[0]))
                    o.append(f'<rect class="hit" x="{X(s0):.1f}" y="{mt}" width="{X(s1) - X(s0):.1f}" height="{h - mt - mb}" data-tip="{esc(tip)}"/>')
            o.append("</svg>")
            charts.append(f'<div>{"".join(o)}</div>')
        return f'<div class="rt-grid"{' style="grid-template-columns:1fr"' if big else ""}>{"".join(charts)}</div>'
    qtabs = qtabs and outside and len(rungs) > 1
    wins = [(k, r) for (k, _), r in zip(early[1], [(early[0], 1.0), (0.0, early[0]), None])] if early else []   # last 1 - f | first f | all
    def ranged(mk, qs):
        if not wins:
            return draw(mk, qs=qs)
        return "".join(f'<div data-win="{k}"{"" if j == 0 else " hidden"}>{draw(mk, r, qs)}</div>' for j, (k, r) in enumerate(wins))
    panes = []
    for j, (mk, _, _, _) in enumerate(RT_STATS):
        body = ("".join(f'<div data-ref="{q}"{"" if i == 0 else " hidden"}>{ranged(mk, [q])}</div>' for i, q in enumerate(rungs)) if qtabs
                else ranged(mk, rungs))
        panes.append(f'<div data-{"metric" if outside else "ref"}="{mk}"{"" if j == 0 else " hidden"}>{body}{f'<p class="note">{t["rt_st_def"][mk]}</p>' if defs else ""}</div>')   # defs=False: a table explains the statistics instead
    tog = (mtoggle([(mk, t["rt_st_name"][mk]) for mk, _, _, _ in RT_STATS]) if outside
           else '<div class="rtoggle">' + "".join(f'<button data-r="{mk}" class="{"on" if j == 0 else ""}">{esc(t["rt_st_name"][mk])}</button>' for j, (mk, _, _, _) in enumerate(RT_STATS)) + "</div>")
    legend = ('<div class="legend">' + "".join(f'<span><i style="background:{swatch_bg(col, dash)}"></i>{esc(name)}</span>' for name, _, col, dash in arms) + "</div>") if show_legend else ""
    rtab = ('<div class="wtoggle rtoggle">' + "".join(f'<button data-w="{k}" class="{"on" if j == 0 else ""}">{esc(n)}</button>' for j, (k, n) in enumerate(early[1])) + "</div>") if early else ""
    qtab = ('<div class="rtoggle">' + "".join(f'<button data-r="{q}" class="{"on" if j == 0 else ""}">{q}</button>' for j, q in enumerate(rungs)) + "</div>") if qtabs else ""
    card = (f'<div class="msec">{tog}<div class="card{" one" if big else ""}">{qtab}{rtab}{legend}{"".join(panes)}</div></div>' if outside
            else f'<div class="card{" one" if big else ""}">{rtab}{tog}{legend}{"".join(panes)}</div>')
    if not head:
        return card
    return f"""<h3>{esc(t['rt_st_h'])}</h3><p class="note">{t['rt_health_note']}</p><p class="note">{t['rt_st_note']}</p>
{card}"""


def router_section(R, t, cols):
    """Settings table (what each arm changes) + one dot chart of the results (overall_tsqw, arm - partner; q0 = 3-seed mean +- se,
    q1-q3 = seed 1); no numeric result table (user)."""
    # 1) settings
    # two tables, one per control (user), as the two charts below; Change ID, partner and rungs-run columns dropped (user)
    drop = {1, 8, 9}
    head = "".join(f'<th class="l">{esc(x)}</th>' for i, x in enumerate(t["rt_set_head"]) if i not in drop)
    def set_table(rows, title):
        body = "".join("<tr>" + "".join(f'<td class="l">{c}</td>' for i, c in enumerate(row) if i not in drop) + "</tr>" for row in rows)
        return (f'<div class="card rtset" style="width:fit-content;max-width:100%;margin:10px auto;box-sizing:border-box"><div class="ct" style="text-align:center">{esc(title)}</div>'
                f'<table style="font-size:11.5px"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')   # smaller, centred (user)
    rows = t["rt_set_rows"]
    settings = ""                                         # both settings tables dropped (user 11-02: MuonH table, then AdamW - "we don't care about AdamW")
    # 2) results: the same line-chart form as the optimizer / KD sections (x = model size, one line per comparison; q0 = 3-seed mean)
    mk = "00_overall_tsqw"
    # two charts side by side, one per control (user): MuonH arms - control 1 (MUONH_FIT), AdamW arm - control 2 (BASE)
    def chart_for(arms, partner, mk=mk, mname="overall_tsqw"):
        series = []
        for key, arm, col in arms:
            pts = []
            for q in RUNGS:
                d = _paired(R, arm, partner, q, mk)
                if not d:
                    continue
                v = sum(d) / len(d)
                pts.append((N_ACTIVE[q], v, f"{esc(t[key])} · {q} ({N_LABEL[q]})<br>{mname} {v:+.4f}" + (f"（{len(d)} seeds）" if len(d) > 1 else "")))
            if pts:
                series.append((t[key], col, pts, False))
        if not series:
            return f'<p class="note">{esc(t["pending"])}</p>'
        xs_ = {q: N_ACTIVE[q] for q in RUNGS if any(abs(p[0] - N_ACTIVE[q]) < 1 for _, _, pts, _ in series for p in pts)}
        return legend([(n, c, d) for n, c, _, d in series]) + line_chart(series, xs_, t["rt_ylab"], lambda v: f"{v:+.3f}", labels=False, zero=True)   # q0 = the seed mean only, no hollow per-seed points (user 11-02)
    arms_l = [("rt_l_a", "MUONH_FIT_DSV3", "--s2"), ("rt_l_b2", "MUONH_FIT_DSV3_RTADAMW_RT0P19", "--s4"), ("rt_l_b3", "MUONH_FIT_DSV3_RT0P19", "--s3"),
              ("rt_l_biasonly", "MUONH_FIT_DSV3_BIASONLY", "--s5"), ("rt_l_rtmuonh", "MUONH_FIT_DSV3_RTMUONH", "--s1")]   # q0-only arms, were a side table vs A (user 11-02)
    blocks = "".join(f'<div data-metric="{m}"{"" if j == 0 else " hidden"}>{chart_for(arms_l, "MUONH_FIT", m, n)}</div>' for j, (m, n) in enumerate(METRICS))
    chart = (f'<div class="msec">{mtoggle(METRICS)}'                # metric buttons back (user 11-02); MuonH only, the AdamW pair (C − BASE) dropped
             f'<div class="card one"><div class="ct">{esc(t["rt_left_h"])}</div>{blocks}</div></div>')
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
    # router-LR scan on the DSV3 base (Runlong 10-29: RT0P19 / RT0P5 / x1 = A / RT2): only the gate's LR changes, nominal multiples of the law
    GR = dict(GH, mults=[(0.19, "0P19"), (0.5, "0P5"), (1.0, "1X"), (2.0, "2")], nominal=True, pool=True, vs=t["rt_vs"], xr=(0.14, 2.7),
              change=lambda suffix, tag, seed="": "MUONH_FIT_DSV3" + seed if tag == "1X" else f"MUONH_FIT_DSV3_RT{tag}{seed}")
    r_fit = cell_fit(R, "q0", "", mk, GR)[1]
    r_cell = mini_lr(R, t, "q0", "", mk, "overall_tsqw", lr1=0.0164, G=GR)
    concl = lambda key, cur, fit: t[key].format(m=fit[0], pct=abs(fit[0] / cur - 1) * 100, g=-fit[1]) if fit and cur else ""   # 结论 column (user 11-02)
    lrrows = "".join(f'<tr><th class="grow">{esc(t[a])}</th><td class="gc up"><div class="glab">{esc(t[b])}</div>{c}</td><td class="l ckr">{esc(reading(k, cu, f))}</td>'
                     f'<td class="l ckr">{esc(concl(k + "c", cu, f))}</td></tr>'
                     for a, b, c, k, cu, f in (("rt_s", "rt_s_lab", s_cell, "rt_s_r", s_cur, s_fit),
                                               ("rt_rl", "rt_rl_lab", r_cell, "rt_rl_r", 1.0, r_fit)))
    # other DSV3-base ablations (Runlong 10-29), q0 seed 1, vs A: seed-1 pair and vs A's 3-seed mean
    lrhead = "".join(f"<th>{esc(x)}</th>" for x in t["rt_lr_head2"])
    lrgrid = (f'<p class="note">{esc(t["rt_lr_h2"])}</p><div class="card" style="overflow:visible"><table class="lrgrid"><thead><tr>{lrhead}</tr></thead>'
              f'<tbody>{lrrows}</tbody></table></div>')
    return f"""<div class="card">{t['rt_two']}</div>
<div class="card"><div class="ct">{esc(t['rt_gate_h'])}</div>{t['rt_gate']}</div>
{settings}
{chart}
{lrgrid}
{router_trends(t)}"""


# Router TODO 9 (Runlong.md, 2026-10-27): the load-balancing loss replaced by the phi-balancing negative-entropy loss (arXiv 2605.15403).
# ROUTER_NE_<alpha>[_NOBIAS|_SOFTMAX|_ETA<x>]; pair MUONH_FIT_DSV3 (DSV3 base, bias kept) or MUONH_FIT (_SOFTMAX), seed 1 at q0.
NE_FULL = [  # (Change ID, label key, colour, dashed) - full-attention negative-entropy arms, all minus MUONH_FIT (user 11-02)
    ("MUONH_FIT_DSV3", "ne_l_dsv3", "--v8", True),
    ("ROUTER_NE_1EM6", "ne_l_e6", "--a07", False), ("ROUTER_NE_1EM5", "ne_l_e5", "--a08", False),
    ("ROUTER_NE_1EM4", "ne_l_e4", "--a09", False), ("ROUTER_NE_1EM3", "ne_l_e3", "--v7", False),
    ("ROUTER_NE_1EM5_NOBIAS", "ne_l_e5nb", "--g08", False), ("ROUTER_NE_1EM4_NOBIAS", "ne_l_e4nb", "--g09", False),
    ("ROUTER_NE_1EM5_SOFTMAX", "ne_l_e5sm", "--k08", False), ("ROUTER_NE_1EM4_SOFTMAX", "ne_l_e4sm", "--k09", False),
]


def ne_section(R, t, cols):
    """Full-attention negative-entropy arms as a line chart (was a table, user 11-02): every arm minus MUONH_FIT (MuonH, softmax router, CE),
    per-seed paired, q0 = the seed mean only; DSV3 itself dashed as a reference; metric buttons as in the Router chart."""
    def series_for(arms, mk, mname):
        series, xs = [], {}
        for ch, lab, col, dashed in arms:
            pts = []
            for q in RUNGS:
                d = _paired(R, ch, "MUONH_FIT", q, mk)
                if not d:
                    continue
                v = sum(d) / len(d); xs[q] = N_ACTIVE[q]
                pts.append((N_ACTIVE[q], v, f"{esc(t[lab])} · {q} ({N_LABEL[q]})<br>{mname} {v:+.4f}" + (f"（{len(d)} seeds）" if len(d) > 1 else "")))
            if pts:
                series.append((t[lab], col, pts, dashed))
        return series, xs
    left_arms = [a for a in NE_FULL if "SOFTMAX" not in a[0]]
    right_arms = [a for a in NE_FULL if "SOFTMAX" in a[0]]       # softmax + negative entropy on the right (user 11-02)
    def panel(series, xs, yr):
        if not series:
            return f'<p class="note">{esc(t["pending"])}</p>'
        return legend([(n, c, d) for n, c, _, d in series]) + line_chart(series, xs, t["ne_y"], lambda v: f"{v:+.3f}", w=560, h=360, labels=False, zero=True, yrange=yr)
    def pair(mk, mname):
        (ls, lx), (rs, rx) = series_for(left_arms, mk, mname), series_for(right_arms, mk, mname)
        xs = {**lx, **rx}                                         # same x axis on both sides
        ys = [p[1] for _, _, pts, _ in ls + rs for p in pts] + [0.0]
        pad = (max(ys) - min(ys)) * 0.08 or 0.01
        yr = (min(ys) - pad, max(ys) + pad)                       # same y range on both sides
        card = lambda h, body: (f'<div class="card" style="min-width:0;width:100%;margin:10px 0;box-sizing:border-box">'
                                f'<div class="ct" style="min-height:2.6em">{esc(t[h])}</div>{body}</div>')
        return (f'<div class="rt2" style="display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px;align-items:stretch">'
                + card("ne_chart_l", panel(ls, xs, yr)) + card("ne_chart_r", panel(rs, xs, yr)) + "</div>")
    if not any(_paired(R, ch, "MUONH_FIT", q, "00_overall_tsqw") for ch, *_ in NE_FULL[1:] for q in RUNGS):
        return ""
    blocks = "".join(f'<div data-metric="{m}"{"" if j == 0 else " hidden"}>{pair(m, n)}</div>' for j, (m, n) in enumerate(METRICS))
    return f"""<h3>{esc(t['ne_h'])}</h3><p class="note">{t['ne_zero']}</p><div class="msec">{mtoggle(METRICS)}{blocks}</div>"""

# Router ablations on the SWS base (Runlong, from 2026-10-29 21:5x): MUONH_FIT_DSV3_SWS = DSV3 router + SWA (5:1, window 2048, NoPE globals)
# + learnable attention sinks; every arm changes one router item and pairs with it per seed. (group key, [(arm, label key)])
SWS_BASE = "MUONH_FIT_DSV3_SWS"
SWS_GROUPS = [
    ("sws_g_bal", [("MUONH_FIT_DSV3_SWS_BIASONLY", "sws_biasonly"), ("MUONH_FIT_DSV3_SWS_NOBIAS", "sws_nobias"), ("MUONH_FIT_DSV3_SWS_NOBAL", "sws_nobal"),
                   ("MUONH_FIT_DSV3_SWS_AUXBATCH", "sws_auxbatch"), ("MUONH_FIT_DSV3_SWS_AUX1EM3", "sws_aux1em3")]),
    ("sws_g_gamma", [("MUONH_FIT_DSV3_SWS_GAMMA1EM4", "sws_g1em4"), ("MUONH_FIT_DSV3_SWS_GAMMA1EM2", "sws_g1em2")]),
    ("sws_g_score", [("MUONH_FIT_DSV3_SWS_SOFTMAXBIAS", "sws_softmax")]),
    ("sws_g_scale", [("MUONH_FIT_DSV3_SWS_RS2P5", "sws_rs"), ("MUONH_FIT_DSV3_SWS_SHARED", "sws_shared"), ("MUONH_FIT_DSV3_SWS_SHARED_RS2P5", "sws_shared_rs")]),
    ("sws_g_reg", [("MUONH_FIT_DSV3_SWS_ZLOSS1EM3", "sws_zloss")]),
    ("sws_g_opt", [("MUONH_FIT_DSV3_SWS_RT0P19", "sws_rt019"), ("MUONH_FIT_DSV3_SWS_RT0P5", "sws_rt05"), ("MUONH_FIT_DSV3_SWS_RT2", "sws_rt2"),
                   ("MUONH_FIT_DSV3_SWS_RTMUONH", "sws_rtmuonh"), ("MUONH_FIT_DSV3_SWS_RTADAMW_RT0P19", "sws_rtadamw")]),
    ("sws_g_ne", [("ROUTER_NE_1EM6_SWS", "sws_ne6"), ("ROUTER_NE_1EM5_SWS", "sws_ne5"), ("ROUTER_NE_1EM4_SWS", "sws_ne4"), ("ROUTER_NE_4EM4_SWS", "sws_ne4e4"),
                  ("ROUTER_NE_1EM3_SWS", "sws_ne3"), ("ROUTER_NE_1EM2_SWS", "sws_ne2"),
                  ("ROUTER_NE_1EM5_NOBIAS_SWS", "sws_ne5nb"), ("ROUTER_NE_1EM4_NOBIAS_SWS", "sws_ne4nb"),
                  ("ROUTER_NE_1EM5_BATCH_SWS", "sws_ne5batch"), ("ROUTER_NE_1EM5_RTADAMW_SWS", "sws_ne5adamw"),
                  ("ROUTER_NE_1EM5_GAMMA1EM4_SWS", "sws_ne5g4"), ("ROUTER_NE_1EM5_NOBIAS_FREQ_SWS", "sws_ne5nbfreq")]),
]


def dot_rows_chart(groups, rungs, t, xlab):
    """Dot chart: rows = arms grouped under bold axis headers, one panel per rung on a shared x scale; filled dot = mean of the per-seed
    paired differences, hollow = each seed; grey band = the pre-registered |delta| < 0.003 'no difference' zone; zero line solid."""
    CAP = 0.04                                          # beyond +-CAP: drawn at the edge with the value printed (e.g. a diverged run)
    c = lambda v: max(-CAP, min(CAP, v))
    vals = [c(x) for _, rows in groups for _, _, by in rows for d in by.values() for x in d + [sum(d) / len(d)]]
    lo, hi = min(vals + [-0.004]), max(vals + [0.004]); pad = (hi - lo) * 0.08; lo, hi = lo - pad, hi + pad
    W, LW, PG, RH, GH, TOP, BOT = 1000, 290, 14, 22, 24, 34, 30
    pw = (W - LW - PG * (len(rungs) - 1) - 10) / len(rungs)
    n_rows = sum(len(r) for _, r in groups); H = TOP + n_rows * RH + len(groups) * GH + BOT
    X = lambda i, v: LW + i * (pw + PG) + (v - lo) / (hi - lo) * pw
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img">']
    step = 0.01 if hi - lo > 0.03 else 0.005
    ticks = [k * step for k in range(int(lo / step) - 1, int(hi / step) + 2) if lo <= k * step <= hi]
    for i, q in enumerate(rungs):
        x0 = LW + i * (pw + PG)
        o.append(f'<text x="{x0 + pw / 2:.1f}" y="16" text-anchor="middle" style="font-size:12px;font-weight:600">{q}（{N_LABEL[q]}）</text>')
        o.append(f'<rect x="{X(i, -0.003):.1f}" y="{TOP - 6}" width="{X(i, 0.003) - X(i, -0.003):.1f}" height="{H - TOP - BOT + 6}" fill="var(--grid)" opacity="0.55"/>')
        o.append(f'<line x1="{X(i, 0):.1f}" x2="{X(i, 0):.1f}" y1="{TOP - 6}" y2="{H - BOT}" stroke="var(--axis)"/>')
        for v in ticks:
            o.append(f'<text x="{X(i, v):.1f}" y="{H - BOT + 14}" text-anchor="middle" style="font-size:9px;fill:var(--muted)">{v:+.3f}</text>')
    o.append(f'<text x="{LW + (W - LW) / 2:.1f}" y="{H - 3}" text-anchor="middle" style="font-size:10px;fill:var(--muted)">{esc(xlab)}</text>')
    y = TOP
    for gk, rows in groups:
        y += GH - 6
        o.append(f'<text x="4" y="{y:.1f}" style="font-size:12px;font-weight:600">{esc(t[gk])}</text>')
        y += 6
        for lab, arm, by in rows:
            yc = y + RH / 2
            o.append(f'<line x1="{LW - 4}" x2="{W - 10}" y1="{yc:.1f}" y2="{yc:.1f}" stroke="var(--grid)" stroke-dasharray="1 3"/>')
            o.append(f'<text x="14" y="{yc + 4:.1f}" style="font-size:11.5px">{esc(lab)}</text>')
            for i, q in enumerate(rungs):
                d = by.get(q)
                if not d:
                    continue
                m = sum(d) / len(d)
                if len(d) > 1:
                    for x in d:
                        o.append(f'<circle cx="{X(i, c(x)):.1f}" cy="{yc:.1f}" r="3.5" fill="none" stroke="var(--s1)" stroke-width="1.2" opacity="0.8"/>')
                o.append(f'<circle cx="{X(i, c(m)):.1f}" cy="{yc:.1f}" r="4.5" fill="var(--s1)" stroke="var(--surface-1)" stroke-width="1.5"/>')
                if abs(m) > CAP:                        # off-scale: value printed inside the panel next to the edge dot
                    o.append(f'<text x="{X(i, c(m)) + (-8 if m > 0 else 8):.1f}" y="{yc + 4:.1f}" text-anchor="{"end" if m > 0 else "start"}" '
                             f'style="font-size:10px;font-weight:600;fill:var(--s2)">{m:+.3f} →</text>')
                tip = (f"<b>{esc(lab)}</b> · {q}<br><code>{arm}</code><br>overall_tsqw {m:+.4f}"
                       + (f"（{len(d)} seeds：" + " / ".join(f"{x:+.4f}" for x in d) + "）" if len(d) > 1 else ""))
                o.append(f'<circle class="hit" cx="{X(i, c(m)):.1f}" cy="{yc:.1f}" r="11" data-tip="{esc(tip)}"/>')
            y += RH
    o.append("</svg>")
    return "".join(o)


def dot_section(R, t, groups_def, base, key, hl="h2"):
    """One dot_rows_chart section: every arm of groups_def minus the same base, paired per seed (texts t[key + '_h' / '_note' / ...])."""
    mk = "00_overall_tsqw"
    groups, pending = [], []
    for gk, arms in groups_def:
        rows = []
        for arm, lk in arms:
            by = {q: d for q in RUNGS if (d := _paired(R, arm, base, q, mk))}
            (rows.append((t[lk], arm, by)) if by else pending.append(t[lk]))
        if rows:
            groups.append((gk, rows))
    if not groups:
        return ""
    rungs = [q for q in RUNGS if any(q in by for _, rows in groups for _, _, by in rows)]
    chart = dot_rows_chart(groups, rungs, t, t[key + "_x"])
    pend = f'<p class="note">{esc(t["sws_pending"])}{"、".join(esc(x) for x in pending) if t["lang"] == "zh" else ", ".join(esc(x) for x in pending)}</p>' if pending else ""
    return f"""<{hl}>{esc(t[key + '_h'])}</{hl}><p class="note">{t[key + '_note']}</p>
<div class="card"><div class="ct">{esc(t[key + '_chart'])}</div>{chart}</div>{pend}"""


SWS_LINES = [  # (legend-table group key, [(Change ID, short label key, colour, dashed)]) - single-knob router ablations, all minus DSV3 SWS,
    # ONE chart (same zero line) with the legend as a table above it (user 11-02)
    ("sa_r_bal", [("MUONH_FIT_DSV3_SWS_BIASONLY", "sa_biasonly", "--a07", False), ("MUONH_FIT_DSV3_SWS_NOBIAS", "sa_nobias", "--a09", False),
                  ("MUONH_FIT_DSV3_SWS_NOBAL", "sa_nobal", "--v8", False), ("MUONH_FIT_DSV3_SWS_AUXBATCH", "sa_auxbatch", "--v3", False),
                  ("MUONH_FIT_DSV3_SWS_AUX1EM3", "sa_aux1em3", "--v7", False)]),
    ("sa_r_gamma", [("MUONH_FIT_DSV3_SWS_GAMMA1EM4", "sa_g1em4", "--v4", False), ("MUONH_FIT_DSV3_SWS_GAMMA1EM2", "sa_g1em2", "--s4", False),
                    ("MUONH_FIT_DSV3_SWS_SOFTMAXBIAS", "sa_softmax", "--s5", False)]),
    ("sa_r_scale", [("MUONH_FIT_DSV3_SWS_RS2P5", "sa_rs", "--g07", False), ("MUONH_FIT_DSV3_SWS_SHARED", "sa_shared", "--g08", False),
                    ("MUONH_FIT_DSV3_SWS_SHARED_RS2P5", "sa_shared_rs", "--g09", False), ("MUONH_FIT_DSV3_SWS_ZLOSS1EM3", "sa_zloss", "--v6", False)]),
    ("sa_r_opt", [("MUONH_FIT_DSV3_SWS_RT0P19", "sa_rt019", "--k07", False), ("MUONH_FIT_DSV3_SWS_RT0P5", "sa_rt05", "--k08", False),
                  ("MUONH_FIT_DSV3_SWS_RT2", "sa_rt2", "--k09", False), ("MUONH_FIT_DSV3_SWS_RTMUONH", "sa_rtmuonh", "--s1", True),
                  ("MUONH_FIT_DSV3_SWS_RTADAMW_RT0P19", "sa_rtadamw", "--k09", True)]),
]


def sws_section(R, t):
    """The scattered single-knob router ablations on the SWS base: ONE line chart (every arm minus DSV3 SWS) with a legend table
    above it (group / line / name / rungs with data), metric buttons (user 11-02; the NE group lives in ne_chapter)."""
    mk0 = "00_overall_tsqw"
    has = lambda ch, q, mk=mk0: bool(_paired(R, ch, SWS_BASE, q, mk))
    if not any(has(ch, q) for _, arms in SWS_LINES for ch, *_ in arms for q in RUNGS):
        return ""
    def chart(mk, mname):
        series, xs = [], {}
        for _, arms in SWS_LINES:
            for ch, lab, col, dashed in arms:
                pts = []
                for q in RUNGS:
                    d = _paired(R, ch, SWS_BASE, q, mk)
                    if not d:
                        continue
                    v = sum(d) / len(d); xs[q] = N_ACTIVE[q]
                    pts.append((N_ACTIVE[q], v, f"{esc(t[lab])} · {q} ({N_LABEL[q]})<br>{mname} {v:+.4f}" + (f"（{len(d)} seeds）" if len(d) > 1 else "")))
                if pts:
                    series.append((t[lab], col, pts, dashed))
        return line_chart(series, xs, t["nesws_x"], lambda v: f"{v:+.3f}", h=420, labels=False, zero=True) if series else f'<p class="note">{esc(t["pending"])}</p>'
    swatch = lambda col, dashed: (f'<svg width="26" height="10" style="vertical-align:middle"><line x1="1" x2="25" y1="5" y2="5" stroke="var({col})" stroke-width="2.5"'
                                  + (' stroke-dasharray="5 3"' if dashed else "") + "/></svg>")
    rows = []
    for gk, arms in SWS_LINES:
        for j, (ch, lab, col, dashed) in enumerate(arms):
            rungs = [q for q in RUNGS if has(ch, q)]
            rows.append("<tr>" + (f'<td class="l" rowspan="{len(arms)}" style="font-weight:600;vertical-align:top">{esc(t[gk].rstrip("：:"))}</td>' if j == 0 else "")
                        + f'<td>{swatch(col, dashed)}</td><td class="l" style="white-space:nowrap">{esc(t[lab])}</td>'
                        + f'<td class="l" style="white-space:normal;max-width:560px">{t[lab + "_d"]}</td><td class="l" style="white-space:nowrap">{", ".join(rungs) if rungs else esc(t["sa_none"])}</td></tr>')
    table = (f'<div class="card rtset" style="width:fit-content;max-width:100%;margin:10px auto;box-sizing:border-box">'   # no caption (user 11-04)
             f'<table style="font-size:12px"><thead><tr>' + "".join(f'<th class="l">{esc(x)}</th>' for x in t["sa_thead"])
             + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')   # its own card, above the chart (user 11-03)
    blocks = "".join(f'<div data-metric="{m}"{"" if j == 0 else " hidden"}>{chart(m, n)}</div>' for j, (m, n) in enumerate(METRICS))
    return f"""<h2>{esc(t['sws_h'])}</h2><p class="note">{t['nesws_note']}</p>{table}<div class="msec">{mtoggle(METRICS)}
<div class="card one"><div class="ct">{esc(t['sa_chart'])}</div>{blocks}</div></div>"""

NE_SWS = [  # (Change ID, short label key, colour, legend row) - negative-entropy arms on the SWS base, minus SWS; α 1e-2 left out (diverged at q1)
    ("ROUTER_NE_1EM6_SWS", "ns_e6", "--a07", 0), ("ROUTER_NE_1EM5_SWS", "ns_e5", "--a08", 0), ("ROUTER_NE_1EM4_SWS", "ns_e4", "--a09", 0),
    ("ROUTER_NE_4EM4_SWS", "ns_4e4", "--v3", 0), ("ROUTER_NE_1EM3_SWS", "ns_e3", "--v7", 0),
    ("ROUTER_NE_1EM5_NOBIAS_SWS", "ns_e5nb", "--g08", 1), ("ROUTER_NE_1EM4_NOBIAS_SWS", "ns_e4nb", "--g09", 1),
    ("ROUTER_NE_1EM5_NOBIAS_FREQ_SWS", "ns_e5nbf", "--g07", 1), ("ROUTER_NE_1EM5_BATCH_SWS", "ns_e5batch", "--k07", 1),
    ("ROUTER_NE_1EM5_RTADAMW_SWS", "ns_e5adamw", "--k09", 1), ("ROUTER_NE_1EM5_GAMMA1EM4_SWS", "ns_e5g4", "--v6", 1),
]

def nesws_section(R, t):
    """SWS-base negative-entropy arms as one centred line chart (no softmax arm on this base), minus SWS, metric buttons (user 11-02)."""
    def chart(mk, mname):
        series, xs = [], {}
        for ch, lab, col, _ in NE_SWS:
            pts = []
            for q in RUNGS:
                d = _paired(R, ch, SWS_BASE, q, mk)
                if not d:
                    continue
                v = sum(d) / len(d); xs[q] = N_ACTIVE[q]
                pts.append((N_ACTIVE[q], v, f"{esc(t[lab])} · {q} ({N_LABEL[q]})<br>{mname} {v:+.4f}" + (f"（{len(d)} seeds）" if len(d) > 1 else "")))
            if pts:
                series.append((t[lab], col, pts, False))
        if not series:
            return f'<p class="note">{esc(t["pending"])}</p>'
        have = {n for n, *_ in series}
        rows = "".join('<div class="legend" style="min-height:0;margin:1px 0">' + f'<span style="font-weight:600">{esc(t[rk])}</span>'
                       + "".join(f'<span><i style="background:var({col})"></i>{esc(t[lab])}</span>' for _, lab, col, r in NE_SWS if r == k and t[lab] in have)
                       + "</div>" for k, rk in enumerate(("ns_r1", "ns_r2")))   # two legend rows: α sweep / variants (user 11-02)
        return rows + line_chart(series, xs, t["nesws_x"], lambda v: f"{v:+.3f}", labels=False, zero=True)
    if not any(_paired(R, ch, SWS_BASE, q, "00_overall_tsqw") for ch, *_ in NE_SWS for q in RUNGS):
        return ""
    blocks = "".join(f'<div data-metric="{m}"{"" if j == 0 else " hidden"}>{chart(m, n)}</div>' for j, (m, n) in enumerate(METRICS))
    return f"""<h3>{esc(t['nesws_h'])}</h3><p class="note">{t['nesws_note']}</p><div class="msec">{mtoggle(METRICS)}
<div class="card one"><div class="ct">{esc(t['nesws_chart'])}</div>{blocks}<p class="note">{t['nesws_skip']}</p></div></div>"""


def ne_chapter(R, t, cols):
    """Own chapter for the negative-entropy balance loss (Runlong TODO 9, user 11-02): what it is + the full-attention arms (each vs its own
    control) + the SWS-base arms (vs SWS); nothing dropped, only moved here from the Router chapter and the SWS section."""
    full = ne_section(R, t, cols)
    sws = nesws_section(R, t)
    if not (full or sws):
        return ""
    return f"""<h2>{esc(t['ne_ch_h'])}</h2><p class="note">{t['ne_note']}</p>\n{full}\n{sws}"""


# Router chapter, cleaned up step by step (user 11-02): ONE zero line = MUONH_FIT (vanilla softmax router, Switch aux 1e-3, AdamH gate x1);
# full attention only (no SWA / sink), MuonH fitted law, CE. Rows ②-④ build on the DSV3 package, so their values include ①.
RT0_BASE = "MUONH_FIT"
RT0_GROUPS = [
    ("r0_g_score", [("MUONH_FIT_DSV3", "r0_dsv3")]),
    ("r0_g_bal", [("MUONH_FIT_DSV3_BIASONLY", "r0_biasonly")]),
    ("r0_g_opt", [("MUONH_FIT_DSV3_RT0P19", "r0_rt019"), ("MUONH_FIT_DSV3_RT0P5", "r0_rt05"), ("MUONH_FIT_DSV3_RT2", "r0_rt2"),
                  ("MUONH_FIT_DSV3_RTMUONH", "r0_rtmuonh"), ("MUONH_FIT_DSV3_RTADAMW_RT0P19", "r0_rtadamw")]),
    ("r0_g_ne", [("ROUTER_NE_1EM6", "sws_ne6"), ("ROUTER_NE_1EM5", "sws_ne5"), ("ROUTER_NE_1EM4", "sws_ne4"), ("ROUTER_NE_1EM3", "sws_ne3"),
                 ("ROUTER_NE_1EM5_NOBIAS", "r0_ne5nb"), ("ROUTER_NE_1EM4_NOBIAS", "r0_ne4nb")]),
    ("r0_g_nesoft", [("ROUTER_NE_1EM5_SOFTMAX", "sws_ne5"), ("ROUTER_NE_1EM4_SOFTMAX", "sws_ne4")]),
]


def rt0_section(R, t):
    """Chapter head only: title + the zero-line setting (the dot chart of RT0_GROUPS was dropped, user 11-02)."""
    return f"""<h2>{esc(t['r0_h'])}</h2><p class="note">{t['r0_note']}</p>"""


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
              ("QWEN3_30B_A3B_BASE", "Qwen3-30B-A3B-Base", False),
              ("QWEN3_30B_A3B_INSTRUCT_2507", "Qwen3-30B-A3B-Instruct-2507", False),   # 11-16 (user 11-15 "请补充测试downstream 66个分数", "纯文本的两个"): 66-metric
              ("QWEN3_30B_A3B_THINKING_2507", "Qwen3-30B-A3B-Thinking-2507", False),   #   scores (jobs 70056 / 70057); doc-isolated rows since 11-16 13:16 (jobs 71090 / 71091)
              ("QWEN3_VL_30B_A3B_INSTRUCT", "Qwen3-VL-30B-A3B-Instruct", False),   # 11-05 (user "追加 qwen3-30B-VL"): no VL Base exists; Qwen3 ids
              ("QWEN3_VL_30B_A3B_THINKING", "Qwen3-VL-30B-A3B-Thinking", False),   #   (tokenizer.json identical to Qwen3's), text-only vLLM
              ("QWEN3_235B_A22B", "Qwen3-235B-A22B (2504)", False),
              ("QWEN3_235B_A22B_INSTRUCT_2507", "Qwen3-235B-A22B-Instruct-2507", False),   # 11-04 (user "全部测下"): the 2507 split of the 2504; listed right under the 2504 (user 11-04 "把 Qwen3-235B（2507）都放在 2504 下面")
              ("QWEN3_235B_A22B_THINKING_2507", "Qwen3-235B-A22B-Thinking-2507", False),
              ("QWEN3_VL_235B_A22B_INSTRUCT", "Qwen3-VL-235B-A22B-Instruct", False),   # 11-05 (user "追加 qwen3-235B-VL-instruct/thinking")
              ("QWEN3_VL_235B_A22B_THINKING", "Qwen3-VL-235B-A22B-Thinking", False),
              ("DEEPSEEK_V4_FLASH_BASE", "DeepSeek-V4-Flash-Base", True), ("KIMI_K3", "Kimi-K3", True),
              ("KIMI_K31_PRO", "Kimi-K3.1-Pro (MXFP4)", True),   # 11-04 (user "kimik3.1出了也要测试"; "kimi 也放一起": right under K3): post-trained from K3, own tokenizer (differs from K3's)
              ("NEMOTRON3_ULTRA_550B_BASE", "Nemotron-3-Ultra-Base", True),   # 11-05 (user "nemo 最好模型 ... 也请测试"): the largest Nemotron
              ("NEMOTRON35_SUPER_VL_120B_BASE", "Nemotron-3.5-Super-VL-Base", True),   #   + the newest (3.5 Super VL, LM distilled from Ultra Base;
                                                                                      #   text-only NemotronH extract); own tokenizer (131072) -> bpb
              ("NEMOTRON35_LIGHTNING_30B_BASE", "Mimi 3.5 Small", True),   # name: user 11-17 13:3x to Doc-isolate测试 "麻烦名字改成mimi lighting", then "mimi lighting -> Mimi 3.5 Lighting", then "Mimi 3.5 Lighting -> Mimi 3.5 Small" (as typed; Nemotron-3.5-Lightning-Base before)
                # 11-17 (user to Doc-isolate测试 "nemo 3.5 nano", then the full name
                                                                                      #   NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Base-BF16): job 76578, 8K + 32K,
                                                                                      #   Ultra-Base's tokenizer and recipe; row appears once collected
              ("QWEN35_397B_A17B", "Qwen3.5-397B-A17B", True),   # 11-16 (user to Doc-isolate测试 "我们有Qwen3.5和3.8的分数吗"): job 71603, 8K + 32K; row appears once collected
              ("QWEN38_FLASH_NEXT", "Qwen3.8-Flash-Next (FP8)", True),
              ("QWEN38_27B", "Qwen3.8-27B", True),
              ("QWEN38_MAX", "Qwen3.8-Max (FP8)", True),          # 11-03 (user), job 35552, 4 nodes
              ("MISTRAL_LARGE4", "Mistral-Large-4 (FP8)", True),
              ("FLINT", "Flint", True),
              ("RIPPLE_65B_BASE", "Ripple-65B-Base-Preview", True)]   # 11-04 (user "能测下 Ripple-65B"), ricursiveAI, pretrain-only (12T, 8K), own tokenizer (100352), HF backend (mimo_doc_eval.py --backend hf)                         # 11-03 (user "flint记得测试"), RecursiveSI, post-trained, SGLang kit (no vLLM / HF code), no BOS, jobs 36553 / 36554   # 11-03 (user), post-trained (no base release), job 35810, 2 nodes   # True = own tokenizer; Qwen added 11-01 (user); Qwen3.8-27B (dense, post-trained, bf16) 11-03 (user)
# "model size" column (user 11-04: dense "XXB", MoE "xxB-AxxB"): total / activated params as the model cards state them (DeepSeek-V4-Flash has no
# card: 284.3B counted from its safetensors headers without the MTP layer, ~13B activated, = the 284B-A13B others quote). Qwen3.8-Flash-Next's card:
# 125B with 6B activated PLUS a 51B n-gram embedding (not in the 125B). Sort key = total params.
REF_SIZE = {"MIMO_V25_PRO_BASE": "1.02T-A42B", "MIMO_V26_PRO_RL": "1.02T-A42B", "QWEN3_30B_A3B_BASE": "30B-A3B",
            "QWEN3_235B_A22B": "235B-A22B", "QWEN3_235B_A22B_INSTRUCT_2507": "235B-A22B", "QWEN3_235B_A22B_THINKING_2507": "235B-A22B",
            "QWEN3_VL_30B_A3B_INSTRUCT": "30B-A3B", "QWEN3_VL_30B_A3B_THINKING": "30B-A3B", "QWEN3_VL_235B_A22B_INSTRUCT": "235B-A22B",
            "QWEN3_VL_235B_A22B_THINKING": "235B-A22B", "NEMOTRON3_ULTRA_550B_BASE": "550B-A55B", "NEMOTRON35_SUPER_VL_120B_BASE": "120B-A12B",
            "DEEPSEEK_V4_FLASH_BASE": "284B-A13B", "KIMI_K3": "2.8T-A104B", "KIMI_K31_PRO": "2.8T-A104B", "QWEN38_FLASH_NEXT": "125B-A6B",
            "QWEN38_27B": "27B", "QWEN38_MAX": "2.4T-A95B", "MISTRAL_LARGE4": "1T-A52B", "FLINT": "40B-A2.5B", "RIPPLE_65B_BASE": "65B",
            "QWEN3_30B_A3B_INSTRUCT_2507": "30B-A3B", "QWEN3_30B_A3B_THINKING_2507": "30B-A3B", "QWEN35_397B_A17B": "397B-A17B",
            "NEMOTRON35_LIGHTNING_30B_BASE": "30B-A3B"}


def _size_num(lbl):
    """'1.02T-A42B' -> 1.02e12 (total params; the sort key of the model size column)."""
    m = re.match(r"([0-9.]+)([BT])", lbl or "")
    return float(m.group(1)) * (1e12 if m.group(2) == "T" else 1e9) if m else None


def _fmt_params(n):
    v, u = (n / 1e12, "T") if n >= 1e12 else (n / 1e9, "B")
    return (f"{v:.0f}" if v >= 100 else f"{v:.1f}".rstrip("0").rstrip(".")) + u   # 20.3B, not 20B (user 11-09; as the nemo6 rows)


def _prod_run(change, q):
    """(run.json, argv list, Megatron total params or None) of a PROD v1 run (Runlong's dirs, e.g. qwen3_tpp200_prod1_dpsk-q5-tpp200-s1), or None.
    The ladder's total_nonemb_params counts every layer as MoE (PROD1 layer 0 is dense: +0.43B at q5), so the total comes from Megatron's own
    "Total number of parameters in billions" log line when present; the active count is exact (user 11-09 "size 和 vocab 写对")."""
    for d in sorted(glob.glob(os.path.join(PROD_RUNS_DIR, f"qwen3_tpp200_{change.lower()}-{q}-*-s1"))):
        try:
            rj = json.load(open(os.path.join(d, "run.json")))
            av = open(os.path.join(d, "argv.txt")).read().split()
        except (OSError, ValueError):
            continue
        tot = None
        for f in sorted(glob.glob(os.path.join(d, "logs", "*.log")), key=os.path.getmtime)[:3]:
            try:
                for line in open(f, errors="replace"):
                    m = re.search(r"Total number of parameters in billions: ([0-9.]+)", line)
                    if m:
                        tot = float(m.group(1)) * 1e9
                        break
            except OSError:
                pass
            if tot:
                break
        return rj, av, tot
    return None


def ladder_size(change, q):
    """Our ladder checkpoint as 'xxB-AxxB' with embedding + untied output head included (as the model cards count): the run's submission.json
    (total / active NON-embedding params) + argv.txt (hidden, padded vocab, untied). None when the run record is not in the shared results."""
    for d in sorted(glob.glob(os.path.join(OPT_LR_ROOT, change, "runs", f"*-{q}-*"))):
        try:
            sub = json.load(open(os.path.join(d, "submission.json")))
            av = open(os.path.join(d, "argv.txt")).read().split()
            hid = int(av[av.index("--hidden-size") + 1])
            voc = int(av[av.index("--padded-vocab-size") + 1] if "--padded-vocab-size" in av else av[av.index("--vocab-size") + 1])
            emb = hid * voc * (2 if "--untie-embeddings-and-output-weights" in av else 1)
            tot, act = sub["total_nonemb_params"] + emb, sub["active_nonemb_params"] + emb
        except (OSError, ValueError, KeyError, IndexError):
            continue
        return _fmt_params(tot) if tot == act else f"{_fmt_params(tot)}-A{_fmt_params(act)}", tot
    pr = _prod_run(change, q)                                 # PROD v1 runs (e.g. our best, PROD1_DPSK q5) are not in the shared results
    if pr:
        rj, av, mtot = pr
        try:
            hid = int(av[av.index("--hidden-size") + 1])
            voc = int(av[av.index("--padded-vocab-size") + 1] if "--padded-vocab-size" in av else av[av.index("--vocab-size") + 1])
            emb = hid * voc * (2 if "--untie-embeddings-and-output-weights" in av else 1)
            tot, act = int(mtot) if mtot else rj["total_nonemb_params"] + emb, rj["active_nonemb_params"] + emb
            return f"{_fmt_params(tot)}-A{_fmt_params(act)}", tot
        except (ValueError, KeyError, IndexError):
            pass
    return None


def size_td(lbl, tot=None):
    return f'<td data-v="{tot if tot is not None else (_size_num(lbl) or "")}">{esc(lbl)}</td>' if lbl else "<td>·</td>"


# "vocab size" column right of "model size" (user 11-05): the model's vocab_size (embedding rows, padding included) from its config.json
# (text_config.vocab_size for the VL checkpoints), read 11-05 from the snapshots in the shared HF cache; tokenizer token counts are a
# little lower where the embedding is padded (Qwen3: 151,936 rows / 151,665 tokens).
REF_VOCAB = {"MIMO_V25_PRO_BASE": 152576, "MIMO_V26_PRO_RL": 152576, "QWEN3_30B_A3B_BASE": 151936, "QWEN3_235B_A22B": 151936,
             "DEEPSEEK_V4_FLASH_BASE": 129280, "KIMI_K3": 163840, "QWEN38_FLASH_NEXT": 248320, "QWEN38_27B": 248320, "QWEN38_MAX": 248320,
             "QWEN35_397B_A17B": 248320,
             "MISTRAL_LARGE4": 204800, "FLINT": 150144, "RIPPLE_65B_BASE": 100352,
             "QWEN3_VL_30B_A3B_INSTRUCT": 151936, "QWEN3_VL_30B_A3B_THINKING": 151936, "QWEN3_235B_A22B_INSTRUCT_2507": 151936,
             "QWEN3_235B_A22B_THINKING_2507": 151936, "QWEN3_VL_235B_A22B_INSTRUCT": 151936, "QWEN3_VL_235B_A22B_THINKING": 151936,
             "KIMI_K31_PRO": 163840, "NEMOTRON3_ULTRA_550B_BASE": 131072, "NEMOTRON35_SUPER_VL_120B_BASE": 131072,
             "NEMOTRON35_LIGHTNING_30B_BASE": 131072,
             "QWEN3_30B_A3B_INSTRUCT_2507": 151936, "QWEN3_30B_A3B_THINKING_2507": 151936}   # (llm_config for Nemotron VL)


def vocab_td(v):
    return f'<td data-v="{v}">{v:,}</td>' if v else "<td>·</td>"


def ladder_vocab(change, q):
    """Padded vocab size of our ladder checkpoint (argv.txt --padded-vocab-size / --vocab-size), as in ladder_size()."""
    for d in sorted(glob.glob(os.path.join(OPT_LR_ROOT, change, "runs", f"*-{q}-*"))):
        try:
            av = open(os.path.join(d, "argv.txt")).read().split()
            return int(av[av.index("--padded-vocab-size") + 1] if "--padded-vocab-size" in av else av[av.index("--vocab-size") + 1])
        except (OSError, ValueError, IndexError):
            continue
    pr = _prod_run(change, q)
    if pr:
        av = pr[1]
        try:
            return int(av[av.index("--padded-vocab-size") + 1] if "--padded-vocab-size" in av else av[av.index("--vocab-size") + 1])
        except (ValueError, IndexError):
            pass
    return None


REF_UNSURE = {"FLINT"}   # user 11-04: mark the Flint baseline "(可能不对)" - its numbers may be wrong
REF_ARMS = [("BASE", "AdamW"), ("OPT_MUON", "Muon"), ("OPT_MUONH", "MuonH"), ("KD_MUONH_A0P8", "MuonH + KD α 0.8")]
REF_STATS = [("bpb", "bpb"), ("loss", "val_loss"), ("ppl", "ppl")]   # bpb first and the default (user 11-03)


SHOW_REF_GAP = False


def ref_section(R, t, cols):
    """Absolute val_loss / ppl / bpb (stat toggle) x eval window 8K / 32K (window toggle, user 11-03): the reference models + our best
    ladder checkpoint and its gap to MiMo-Base (8K only: the ladder is trained at 8K). 32K = <NAME>_W32K (chunks of 32760 predictions);
    each 32K cell also shows its change vs the 8K window. Own-tokenizer models: loss / ppl per THEIR token (greyed); bpb compares."""
    refs = [(n + (t["ref_unsure"] if c in REF_UNSURE else ""), c, f) for c, n, f in REF_MODELS if (c, "ref") in R]
    if not refs:
        return ""
    # ONLY our best ladder checkpoint (user): lowest overall_tsqw loss over every evaluated ladder checkpoint
    # 8K-window records only (user 11-09): a <ID>_W32K record is the same checkpoint at the 32K window and was picked here before, landing
    # 32K numbers in the 8K tab under a non-existent Change ID "PROD1_DPSK_W32K"; its _W32K record now fills the 32K tab
    best = min(((c, q) for c, q in R if q != "ref" and not c.endswith("_W32K") and "00_overall_tsqw" in R[(c, q)]), key=lambda k: R[k]["00_overall_tsqw"]["loss"])
    best32 = R.get((best[0] + "_W32K", best[1])) or {}
    lad = [(f"{t['ref_best']}: {best[0]} · {best[1]}" + (f" ({N_LABEL[best[1]]})" if best[1] in N_LABEL else ""), R[best], False)]
    head = '<tr><th class="l"></th><th>model size</th><th>vocab size</th>' + "".join(f"<th>{esc(n)}</th>" for _, n in cols) + "</tr>"
    tbl = ref_tbl(cols)
    lsz = ladder_size(*best)

    def panes(w32):
        rows_m = [(n, (R.get((c + "_W32K", "ref")) or {}) if w32 else R[(c, "ref")], f, R[(c, "ref")], c) for n, c, f in refs]
        if not rows_m:
            return f'<p class="note">{esc(t["ref_w32_pending"])}</p>'
        base = next((m for n, m, f, _, _ in rows_m if not f), None)
        out = []
        for j, (sec, sname) in enumerate(REF_STATS):
            fmt = (lambda v: f"{v:.3f}") if sec == "ppl" else (lambda v: f"{v:.4f}")
            cell = lambda m, k: fmt(m[k][sec]) if k in m and sec in m[k] else ("—" if w32 else "·")   # 32K not yet run: a dash (user 11-03)
            rows = []
            for n, m, f, m8, c in rows_m:
                dim = f and sec != "bpb"
                rows.append(f'<tr data-s{" class=\"dim\"" if dim else ""}><td class="l">{esc(n)}{" †" if dim else ""}</td>' + size_td(REF_SIZE.get(c)) + vocab_td(REF_VOCAB.get(c))
                            + "".join(f"<td>{cell(m, k)}</td>" for k, _ in cols) + "</tr>")   # references never bold; only ours (user 11-09)
            if w32:                                           # our best at the 32K window = its _W32K record ("-" where not evaluated)
                rows += [f'<tr><td class="l"><b>{esc(n)}</b></td>' + size_td(*(lsz or (None,))) + vocab_td(ladder_vocab(*best))
                         + "".join(f"<td><b>{fmt(best32[k][sec])}</b></td>" if k in best32 and sec in best32[k] else "<td>-</td>" for k, _ in cols) + "</tr>" for n, _, _ in lad]
            if not w32:
                rows += [f'<tr><td class="l"><b>{esc(n)}</b></td>' + size_td(*(lsz or (None,))) + vocab_td(ladder_vocab(*best)) + "".join(f"<td><b>{cell(m, k)}</b></td>" for k, _ in cols) + "</tr>" for n, m, _ in lad]
                gap = []
                for k, _ in cols:
                    v = [m[k][sec] for _, m, _ in lad if k in m and sec in m[k]]
                    if not v or base is None or k not in base or sec not in base[k]:
                        gap.append("<td>·</td>")
                    elif sec == "ppl":
                        gap.append(f'<td class="pos">{min(v) / base[k][sec]:.2f}×</td>')
                    else:
                        gap.append(f'<td class="pos">+{min(v) - base[k][sec]:.{3 if sec == "loss" else 4}f}</td>')
                if SHOW_REF_GAP:                                   # "our best - MiMo-V2.5-Pro-Base" row: off for now (user 11-03)
                    rows.append(f'<tr><td class="l">{esc(t["ref_gap"])}{" (×)" if sec == "ppl" else ""}</td><td></td>{"".join(gap)}</tr>')
            out.append(f'<div data-ref="{sec}"{"" if j == 0 else " hidden"}>{tbl}<thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>')
        return "".join(out)

    wtog = '<div class="wtoggle rtoggle">' + "".join(f'<button data-w="{k}" class="{"on" if j == 0 else ""}">{esc(n)}</button>' for j, (k, n) in enumerate(t["ref_win"])) + "</div>"
    stog = '<div class="rtoggle stog">' + "".join(f'<button data-r="{sec}" class="{"on" if j == 0 else ""}">{esc(n)}</button>' for j, (sec, n) in enumerate(REF_STATS)) + "</div>"
    stog += '<div class="rtoggle stog1" style="display:none"><button class="on">bpb</button></div>'   # the 32K - 8K tab: bpb only
    foot = ""   # vocab size / model size / † notes removed (user 11-09 "扔掉"); t["ref_dag"] kept unused
    return f"""<h2>{esc(t['ref_h'])}</h2>
<div class="card">{wtog}{stog}<div data-win="w8">{panes(False)}</div>
<div data-win="w32" hidden>{panes(True)}</div>
<div data-win="wd" hidden>{ref_w32_diff(R, t, cols)}</div>{foot}</div>"""


def ref_w32_diff(R, t, cols):
    """Its own table (user 11-03): bpb of the 32K window minus the 8K window, per reference model."""
    rows = [(n + (t["ref_unsure"] if c in REF_UNSURE else ""), R.get((c + "_W32K", "ref")) or {}, R[(c, "ref")], c) for c, n, _ in REF_MODELS if (c, "ref") in R]
    if not any(a for _, a, _, _ in rows):
        return f'<p class="note">{esc(t["ref_w32_pending"])}</p>'
    th = '<tr><th class="l"></th><th>model size</th><th>vocab size</th>' + "".join(f"<th>{esc(n)}</th>" for _, n in cols) + "</tr>"
    def td(a, b, k):
        if k not in a or k not in b or "bpb" not in a[k] or "bpb" not in b[k]:
            return "<td>-</td>" if k in b and "bpb" in b[k] else "<td>·</td>"   # 8K exists, 32K not yet: "-" (user 11-03)
        d = a[k]["bpb"] - b[k]["bpb"]
        return f'<td class="{"neg" if d < -5e-5 else "pos" if d > 5e-5 else ""}">{d:+.4f}</td>' if abs(d) >= 5e-5 else "<td>0</td>"
    trs = "".join(f'<tr data-s><td class="l">{esc(n)}</td>' + size_td(REF_SIZE.get(c)) + vocab_td(REF_VOCAB.get(c)) + "".join(td(a, b, k) for k, _ in cols) + "</tr>" for n, a, b, c in rows)
    best = min(((c, q) for c, q in R if q != "ref" and not c.endswith("_W32K") and "00_overall_tsqw" in R[(c, q)]), key=lambda k: R[k]["00_overall_tsqw"]["loss"])   # as in ref_section
    b32 = R.get((best[0] + "_W32K", best[1])) or {}
    trs += (f'<tr><td class="l"><b>{esc(t["ref_best"])}: {best[0]} · {best[1]}' + (f" ({N_LABEL[best[1]]})" if best[1] in N_LABEL else "") + "</b></td>"
            + size_td(*(ladder_size(*best) or (None,))) + vocab_td(ladder_vocab(*best)) + "".join(re.sub(r"^(<td[^>]*>)(.*)(</td>)$", r"\1<b>\2</b>\3", td(b32, R[best], k)) for k, _ in cols) + "</tr>")   # ours bold (user 11-09)
    return f'{ref_tbl(cols)}<thead>{th}</thead><tbody>{trs}</tbody></table>'                       # caption dropped (user 11-03)


def ref_tbl(cols):
    """Opening tag + colgroup shared by the reference tables, so their columns line up across the 8K / 32K tabs and the difference table."""
    w = 54 / len(cols)                                   # + the model size column (user 11-04) and vocab size (user 11-05), left of overall_tsqw
    return ('<table class="rsort" style="table-layout:fixed;width:100%"><colgroup><col style="width:27%"><col style="width:10%"><col style="width:9%">'
            + "".join(f'<col style="width:{w:.2f}%">' for _ in cols) + "</colgroup>")


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


# "Current best architecture" chapter (user 11-03): the PROD v1 recipe (Runlong 2026-11-02) and, per component, the single-variable
# experiment behind it. Numbers = the shared result cards' R1 val_doc (document-isolated, six decon sets, seed 1), arm − pair; our own
# 115-split evals of these runs (PROD1_*, DENSE*_FA0, *_DM) do not exist yet, so overall_tsqw is not available here.
CARDS_DIR = "/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos_llm_scaling_ladder/results"
S6 = ["val_v2_decon", "uncheatable_decon", "codefresh_decon", "v3small_decon", "pile_test_decon", "pile_10k_decon"]
BEST_ROWS = [  # (component key, [(arm, pair, label key)])
    ("ba_c_router", [("MUONH_FIT_DSV3", "MUONH_FIT", "ba_r_dsv3"), ("PROD1_PHI", "PROD1_DPSK", "ba_r_phi")]),
    ("ba_c_dense0", [("MUONH_FIT_DSV3_SWS_DENSE1", "MUONH_FIT_DSV3_SWS", "ba_r_dense1"), ("MUONH_FIT_DSV3_SWS_DENSE2", "MUONH_FIT_DSV3_SWS", "ba_r_dense2"),
                     ("MUONH_FIT_DSV3_SWS_DENSE1_FA0", "MUONH_FIT_DSV3_SWS_DENSE1", "ba_r_fa0")]),
    ("ba_c_shared", [("MUONH_FIT_DSV3_SWS_SHARED", "MUONH_FIT_DSV3_SWS", "ba_r_shared"), ("MUONH_FIT_DSV3_SWS_SHARED_RS2P5", "MUONH_FIT_DSV3_SWS", "ba_r_shared_rs"),
                     ("PROD1_DPSK_RS1", "PROD1_DPSK", "ba_r_rs1")]),
    ("ba_c_latent", [("PROD1_DPSK_LATENT2", "PROD1_DPSK", "ba_r_latent")]),
    ("ba_c_swa", [("SWA_FIT_SINK_W512", "MUONH_FIT", "ba_r_swa")]),
    ("ba_c_ctx", [("SWA_FIT_SINK_W512_S32K", "SWA_FIT_SINK_W512", "ba_r_s32k"), ("SWA_FIT_SINK_W512_DM", "SWA_FIT_SINK_W512", "ba_r_dm"),
                  ("SWA_FIT_SINK_W512_S32K_DM", "SWA_FIT_SINK_W512_S32K", "ba_r_s32kdm_vs32"), ("SWA_FIT_SINK_W512_S32K_DM", "SWA_FIT_SINK_W512", "ba_r_s32kdm_vs8")]),
    ("ba_c_all", [("PROD1_DPSK", "SWA_FIT_SINK_W512_S32K", "ba_r_all")]),
]


def _card_s6(change, q):
    """Six-set mean of a card's R1 val_doc at rung q (seed 1), or None."""
    try:
        r = (json.load(open(os.path.join(CARDS_DIR, change, "scores.json"))).get("rungs") or {}).get(q) or {}
    except (OSError, ValueError):
        return None
    v = r.get("val_doc") or {}
    return sum(v[k] for k in S6) / len(S6) if all(isinstance(v.get(k), (int, float)) for k in S6) else None


# Latent MoE and position-encoding (NoPE / RoPE / partial RoPE) arms on the PROD v1 base (user 11-04: two sections after the SWS one).
# Values = our 115-split doc evals (8K = <ID>, 32K = <ID>_W32K), one dual-window job per checkpoint (launch_prod1_dualwin.sh).
RTQ_ARMS = [("QB", "qb", "--c1"), ("QBX", "qbx", "--c2"), ("BIASONLY", "bo", "--c3"), ("G5EM4", "g5", "--c4"), ("G2P5EM4", "g25", "--c5"),
            ("RTADAMW", "adamw", "--c6"), ("RTPIN", "pin", "--v7"),
            ("QBK3", "qbk3", "--v6"), ("QBK3_WU", "qbk3wu", "--a09")]   # (Change ID suffix, label key, colour); same colours in the eval and router-health charts
X_VALTAB = {"lat"}   # ppf / rtq: value tables removed (user 11-07 "2个表格麻烦删除", 11-08 "这两个不要")
X_BPB = {"lat"}
X_ABS8K = {"ppf"}   # + a third tab: 8K window, absolute val_loss, no zero line (user 11-07 "加一个8k non delta ... 没有0线，按照val_loss绝对值来画")
X_PPZERO = {"ppf"}   # zero line of the data-derisk chapter (packplan_section): 8K − MUONH_FIT, 32K − MUONH_FIT (its 8K score); no 3rd tab (user 11-07)   # value tables also in bpb (user 11-06 "bpb", for the router / QBX experiments)
# E2X KD on the tc_full prepack, A = E2X router (DSV3 bias + seq aux), B = E2X + QBX (user 11-06; jobs 44918-44931; evals by prepack_autoeval.sh)
PPF_A, PPF_B = "PROD1_DPSK_LATENT2_E2X_KD_A0P8_S64Y_MIMO25_PPFULL", "PROD1_DPSK_LATENT2_E2X_QBX_KD_A0P8_S64Y_MIMO25_PPFULL"
PPF_C = "ULTRA_XXDEV_3B_KD_A0P8_S64Y_MIMO25_PPFULL"   # A's q1 rerun on megatron-lm xxdev (11-10; job 56500, evals 57306-57308 by run_doc_eval.sh)
PPF_STEM_A, PPF_STEM_B = "qwen3_tcfull_prepack_32k_e2x_kd_a0p8_s64y_mimo25", "qwen3_tcfull_prepack_32k_e2x_qbx_kd_a0p8_s64y_mimo25"
# QB explainer under the router section's title (user 11-06 "在title下面能否写一个介绍 qb 的文档"). Source of every statement: Megatron
# megatron-lm-80781ab0ab (moe_utils.qb_dual_update, router.quantile_balancing, finalize_model_grads._update_router_qb_beta,
# moe_utils.get_updated_expert_bias / topk_routing_with_score_function) + the QB / QBX run argv and Runlong's README of QBX.
_QB_CSS = 'style="max-width:1000px;margin:12px auto;text-align:left;line-height:1.65;font-size:13.5px"'
X_INTRO = {"rtq": {
"zh": f"""<div class="card" {_QB_CSS}>
<h3 style="margin-top:0">QB 是什么：Quantile Balancing（分位数均衡）</h3>
<p><b>一句话：</b>每个 expert 有一个门槛 β<sub>e</sub>，token 按「router 打分 − 门槛」排序选 top-k。门槛不是按固定步长慢慢调，而是每个 global batch 直接算出“恰好让每个 expert 收到平均份额 token”的值（一个分位数）。负载均衡全靠这个门槛：不用 aux loss，也不用 DSV3 的 expert bias。Megatron 自带，开关是 <code>--moe-router-load-balancing-type quantile_balancing</code>。</p>
<h4>和 0 线（E2X：DSV3 expert bias + seq aux）的区别</h4>
<table style="font-size:12.5px;width:100%"><thead><tr><th class="l"></th><th class="l">DSV3 expert bias（0 线）</th><th class="l">QB</th></tr></thead><tbody>
<tr><td class="l"><b>怎么调</b></td><td class="l" style="white-space:normal">每个 global batch 数一遍每个 expert 收到的 token：偏多的 bias −γ，偏少的 +γ（γ = 1e-3，固定步长，只看多还是少）</td><td class="l" style="white-space:normal">直接解出“让每个 expert 正好收 T·k/E 个 token”的门槛，一步到位，没有步长</td></tr>
<tr><td class="l"><b>加在哪</b></td><td class="l" style="white-space:normal">加在 sigmoid 之后的分数上</td><td class="l" style="white-space:normal">从 router 原始 logit 上减（sigmoid 之前）</td></tr>
<tr><td class="l"><b>gate 权重</b></td><td class="l" style="white-space:normal" colspan="2">两者相同：bias / 门槛只决定“选哪些 expert”，gate 权重仍是被选 expert 的 sigmoid 分数（归一化后 × routed scaling 2.5）</td></tr>
<tr><td class="l"><b>aux loss</b></td><td class="l">seq aux 1e-4</td><td class="l">必须关掉</td></tr>
<tr><td class="l"><b>超参数</b></td><td class="l">γ、seq aux 系数</td><td class="l" style="white-space:normal">基本没有（可选 EMA 系数，默认 0 = 不用，我们也没用）</td></tr>
<tr><td class="l"><b>评测 / 推理</b></td><td class="l">bias 固定</td><td class="l" style="white-space:normal">门槛存在 checkpoint 里，评测时固定不更新（只用过去 batch 算出的值，不会用到同一 batch 后面的 token）</td></tr>
</tbody></table>
<h4>算法（每层各自做）</h4>
<p>记号：s<sub>te</sub> = token t 对 expert e 的 router logit；k = 每个 token 选几个 expert；E = expert 数；T = 一个 micro-batch 的 token 数；C = T·k/E = 每个 expert 的目标 token 数。</p>
<ol style="margin:4px 0 8px 0">
<li><b>路由：</b>每个 token 选 s<sub>te</sub> − β<sub>e</sub> 最大的 k 个 expert（β 用上一个 global batch 算好的值）。</li>
<li><b>token 的门槛 α<sub>t</sub>：</b>同一排序里第 k+1 名的值，也就是没选上的 expert 里最好的那个。</li>
<li><b>expert 的新门槛：</b>对每个 expert，在这个 micro-batch 的 T 个 token 上取 s<sub>te</sub> − α<sub>t</sub> 的第 C+1 大值，作为新的 β<sub>e</sub>。按这个门槛，这批 token 里恰好有 C 个会选它。它是一个分位数，所以叫 quantile balancing。</li>
<li><b>每个 global batch 更新一次：</b>各 micro-batch 算出的 β 先取平均，再在所有 DP rank 之间取平均；减去 E 个 expert 的均值（居中：只有 expert 之间的差影响路由），写回 β。</li>
</ol>
<p>这是“每个 token 选 k 个、每个 expert 收 C 个”这个分配问题的<b>对偶坐标下降</b>（dual coordinate descent）：α 是 token 侧的对偶变量，β 是 expert 侧的对偶变量，每个 global batch 交替更新一次。</p>
<h4>放到我们的配置里</h4>
<p>E2X：E = 512，k = 16。一个 micro-batch = 1 条 32K 序列（TP / CP = 1，所以 T = 32,768），每个 expert 在这条序列里的目标 C = 32,768 × 16 / 512 = <b>1,024</b> 个 token。一个 global batch = 64 条序列（约 2.1M token），β 每个训练 step 更新一次。</p>
<h4>QBX：全局精确分位数（Kimi K3 的做法）</h4>
<p>QB 的 β 是“每条序列各自的分位数”的平均，而平均值不等于整个 batch 的分位数。QBX 把所有 micro-batch、所有 DP rank 的 margin（s − α）按 expert 做直方图并相加，在合并后的直方图上直接读整个 global batch 的第 C+1 大（这里 C = 整个 batch 的 token 数 × k / E）。选 expert 的方法、α、margin 都和 QB 完全相同，只有这一步聚合不同。代码是打了补丁的 Megatron；落在直方图窗口外的 expert 用平均值，并截到窗口边界。</p><p><b>QBX 的具体设置（现在在跑的 run）：</b>Megatron 用 <code>users/xingqianx/share/megatron-lm-80781ab0ab-qbx2</code>（80781ab0ab + 3 个文件的补丁：router.py、moe_utils.py、finalize_model_grads.py），由环境变量打开：<code>MOE_QB_EXACT=1</code>，窗口半宽 <code>MOE_QB_EXACT_WINDOW=2.0</code>（logit），<code>MOE_QB_EXACT_BINS=255</code>（每格 4/255 ≈ 0.016 logit），<code>MOE_QB_EXACT_LOG=100</code>。命令行参数和 QB 完全一样（quantile_balancing，aux loss 0，无 expert bias）。bins 取奇数：每个 token 都有一个 margin 正好等于中心值，偶数 bins 时它落在格子边界上，读数系统性偏高约 0.3 格；11-06 下午把 256 bins 的 run 删掉，按 255 bins 从头重跑。要求 TP = CP = 1；用 full activation recompute 时 router 会算两次（计数翻倍，分位数基本不变）。</p>
<h4>QBK3：完全按 Kimi K3 原文（11-07 开跑）</h4>
<p>QBX 仍是 QB 的形式：门槛从 sigmoid 之前的 logit 上减，直方图窗口 ±2 logit、255 格。QBK3 换成 K3 原文的形式：门槛就是 DSV3 的 expert bias，加在 sigmoid 之后的分数上（routing = top-k(sigmoid(s) + b)，gate 权重不含 b，没有 aux loss）。bias 不再按 ±1e-3 的符号步长更新，而是每个 step 用整个 global batch 的分位数直接算出：直方图范围 [b<sub>min</sub> − 1, b<sub>max</sub> + 1]，1000 格，TP × DP × CP 上一次 all-reduce，取第 ⌈mk/E⌉ 位并线性插值，居中，下一个 step 生效。补丁 Megatron：<code>users/xingqianx/share/megatron-lm-80781ab0ab-qbk3</code>（每个 job 启动时都核对补丁文件的 sha256；BINS = 1000、EMA = 0 写死在 exp 里）。Runlong 让 22 个 agent 做了 review，结论是没有严重问题（<code>share/megatron_qbk3_tests/REVIEW_REPORT.md</code>）。已知现象：没有 LR warmup 时，开头几十步 MaxVio 约 11–15（router 漂移，不是估计方法的问题）。所以另开了一条 <b>QBK3_WU</b>：同样的 QBK3，加 1% LR warmup。QBK3 跑 q1–q4，QBK3_WU 跑 q1–q2。</p>
<h4>这一组实验要回答什么</h4>
<ul style="margin:4px 0">
<li><b>QB / QBX：</b>能不能把 512 个 expert 的不均衡压下来（0 线训练前中期 MaxVio 约 1.5–2.5，有少量 dead expert；后期 LR 降下来后自己降到 0.4–0.6），同时 loss 不变差甚至更好。干净对照是 <b>BIASONLY</b>：同样没有 aux loss，只用 DSV3 bias。</li>
<li><b>其余几条从别的方向处理同一个问题：</b>bias 步长减小（×0.5、×0.25），以及 router 的优化器（AdamW、每行单独 hyperball）。</li>
<li><b>看哪里：</b>结论看下面的评测（各线 − 0 线，8K / 32K）。router health 曲线是训练中的早期参考，每小时更新。</li>
</ul>
<h4>限制</h4>
<ul style="margin:4px 0"><li>必须关掉 aux loss；不支持 router fusion，也不支持 group-limited routing；每个 micro-batch 的 token 数 × k 必须能被 E 整除。</li></ul>
</div>""",
"en": f"""<div class="card" {_QB_CSS}>
<h3 style="margin-top:0">What QB is: Quantile Balancing</h3>
<p><b>In one sentence:</b> every expert has a threshold β<sub>e</sub>, and a token picks its top-k experts by "router score − threshold". The thresholds are not nudged by a fixed step; once per global batch they are set to the value that would have given every expert exactly its average share of tokens (a quantile). Load balance comes from these thresholds alone: no aux loss and no DSV3 expert bias. Built into Megatron: <code>--moe-router-load-balancing-type quantile_balancing</code>.</p>
<h4>Compared with the zero line (E2X: DSV3 expert bias + seq aux)</h4>
<table style="font-size:12.5px;width:100%"><thead><tr><th class="l"></th><th class="l">DSV3 expert bias (zero line)</th><th class="l">QB</th></tr></thead><tbody>
<tr><td class="l"><b>Update</b></td><td class="l" style="white-space:normal">once per global batch, count each expert's tokens: overloaded experts bias −γ, underloaded +γ (γ = 1e-3, fixed step, sign only)</td><td class="l" style="white-space:normal">solves directly for the threshold that gives each expert exactly T·k/E tokens: one jump, no step size</td></tr>
<tr><td class="l"><b>Applied to</b></td><td class="l" style="white-space:normal">added to the scores after the sigmoid</td><td class="l" style="white-space:normal">subtracted from the raw router logits (before the sigmoid)</td></tr>
<tr><td class="l"><b>Gate weights</b></td><td class="l" style="white-space:normal" colspan="2">the same in both: the bias / threshold only decides WHICH experts; the gate weights are the selected experts' sigmoid scores (normalised, × routed scaling 2.5)</td></tr>
<tr><td class="l"><b>Aux loss</b></td><td class="l">seq aux 1e-4</td><td class="l">must be off</td></tr>
<tr><td class="l"><b>Hyperparameters</b></td><td class="l">γ, seq-aux coefficient</td><td class="l" style="white-space:normal">essentially none (an optional EMA coefficient, default 0 = off; not used here)</td></tr>
<tr><td class="l"><b>Eval / inference</b></td><td class="l">bias frozen</td><td class="l" style="white-space:normal">thresholds are saved in the checkpoint and frozen at eval (only values from past batches, never later tokens of the same batch)</td></tr>
</tbody></table>
<h4>Algorithm (per layer)</h4>
<p>Notation: s<sub>te</sub> = router logit of token t for expert e; k = experts per token; E = number of experts; T = tokens in a micro-batch; C = T·k/E = target tokens per expert.</p>
<ol style="margin:4px 0 8px 0">
<li><b>Route:</b> each token takes the k experts with the largest s<sub>te</sub> − β<sub>e</sub> (β from the previous global batch).</li>
<li><b>Token threshold α<sub>t</sub>:</b> the (k+1)-th value of the same ranking, i.e. the best expert the token did not pick.</li>
<li><b>New expert threshold:</b> for each expert, the (C+1)-th largest s<sub>te</sub> − α<sub>t</sub> over the micro-batch's T tokens becomes the new β<sub>e</sub>; under it, exactly C of these tokens would pick that expert. It is a quantile, hence the name.</li>
<li><b>Once per global batch:</b> the micro-batch βs are averaged, then averaged across DP ranks, re-centred (minus the mean over the E experts: only differences between experts matter), and written back.</li>
</ol>
<p>This is <b>dual coordinate descent</b> on the assignment problem "each token picks k, each expert takes C": α is the token-side dual, β the expert-side dual, alternated once per global batch.</p>
<h4>In our configuration</h4>
<p>E2X: E = 512, k = 16. A micro-batch is one 32K sequence (TP / CP = 1, so T = 32,768), so each expert's target within a sequence is C = 32,768 × 16 / 512 = <b>1,024</b> tokens. A global batch is 64 sequences (~2.1M tokens); β is updated once per training step.</p>
<h4>QBX: exact global quantile (the Kimi K3 form)</h4>
<p>QB's β is the average of per-sequence quantiles, and an average of quantiles is not the quantile of the whole batch. QBX sums per-expert histograms of the margins (s − α) over all micro-batches and DP ranks and reads the (C+1)-th largest of the whole global batch from the merged histogram (C = tokens in the global batch × k / E). Expert selection, α and the margins are identical to QB; only this aggregation differs. The code is a patched Megatron; experts outside the histogram window take the averaged value, clamped to the window edge.</p><p><b>QBX settings (the runs now training):</b> Megatron <code>users/xingqianx/share/megatron-lm-80781ab0ab-qbx2</code> (80781ab0ab + a 3-file patch: router.py, moe_utils.py, finalize_model_grads.py), switched on by environment variables: <code>MOE_QB_EXACT=1</code>, window half-width <code>MOE_QB_EXACT_WINDOW=2.0</code> (logits), <code>MOE_QB_EXACT_BINS=255</code> (bin 4/255 ≈ 0.016 logits), <code>MOE_QB_EXACT_LOG=100</code>. The command-line flags are exactly QB's (quantile_balancing, aux loss 0, no expert bias). The bin count is odd because every token contributes one margin exactly at the centre; with an even count it sits on a bin edge and the read is ~0.3 bin high. On 11-06 afternoon the 256-bin runs were deleted and restarted from scratch with 255 bins. Requires TP = CP = 1; full activation recompute runs the router twice (counts doubled, quantile essentially unchanged).</p>
<h4>QBK3: exactly as the Kimi K3 report (started 11-07)</h4>
<p>QBX keeps QB's form: the threshold is subtracted from the logits before the sigmoid, histogram window ±2 logits, 255 bins. QBK3 uses K3's own form: the threshold is the DSV3 expert bias, added to the scores after the sigmoid (routing = top-k(sigmoid(s) + b), gate weights without b, no aux loss). The bias is no longer moved by a ±1e-3 sign step; every step it is set from the quantile of the whole global batch: histogram over [b<sub>min</sub> − 1, b<sub>max</sub> + 1], 1000 bins, one all-reduce over TP × DP × CP, the ⌈mk/E⌉-th position with linear interpolation, mean-centred, applied from the next step. Patched Megatron: <code>users/xingqianx/share/megatron-lm-80781ab0ab-qbk3</code> (every job checks the sha256 of the patched files at start; BINS = 1000 and EMA = 0 are pinned in the exp file). Runlong had 22 agents review it; they found no critical or major defect (<code>share/megatron_qbk3_tests/REVIEW_REPORT.md</code>). Known behaviour: without LR warmup, MaxVio is ~11–15 in the first few dozen steps (router drift, not the estimator). Hence a second arm, <b>QBK3_WU</b>: the same QBK3 plus a 1% LR warmup. QBK3 runs q1–q4, QBK3_WU q1–q2.</p>
<h4>What this set of runs asks</h4>
<ul style="margin:4px 0">
<li><b>QB / QBX:</b> can they remove the 512-expert imbalance (zero line in the early and middle part of training: MaxVio ~1.5–2.5, a few dead experts; it falls to 0.4–0.6 on its own once the LR decays) with equal or better loss? The clean control is <b>BIASONLY</b>: no aux loss either, DSV3 bias only.</li>
<li><b>The other arms attack the same problem differently:</b> smaller bias steps (×0.5, ×0.25), and the router optimizer (AdamW, one hyperball per row).</li>
<li><b>Where to look:</b> the result is the doc eval below (each line − zero line, 8K / 32K). The router-health curves are early reads during training, refreshed hourly.</li>
</ul>
<h4>Limits</h4>
<ul style="margin:4px 0"><li>Aux loss must be off; no router fusion and no group-limited routing; tokens per micro-batch × k must be divisible by E.</li></ul>
</div>"""}}
TRAIN_ITERS_E2X = {"q0": 4915, "q1": 15824, "q2": 36606, "q3": 70438}   # PROD1 TPP 200 iterations (the E2X arms all use them)   # sections with a value table (line x rung, per window) under the chart (user 11-06)
X_SECTIONS = [  # (key, [(arm, pair, label key, colour)], extra rows [(arm, pair, label key)] shown as a table only)
    ("lat", [("PROD1_DPSK_LATENT2", "PROD1_DPSK", "lat_l_l2", "--g08"), ("PROD1_DPSK_LATENT2_E2X", "PROD1_DPSK", "lat_l_e2x", "--a08"),
         ("PROD1_DPSK_LATENT2_E2X", "PROD1_DPSK_LATENT2", "lat_l_x2l", "--s4"), ("PROD1_DPSK_E2X", "PROD1_DPSK", "lat_l_x2", "--v7")], []),   # + expert x2 with / without latent (user 11-06 "放到我们的report上面"; PROD1_DPSK_E2X evals 44281-92)
    ("rope", [("PROD1_DPSK_PROPE25", "PROD1_DPSK", "rope_l_p25", "--c1"), ("PROD1_DPSK_PROPE25G", "PROD1_DPSK", "rope_l_p25g", "--v7")],
     []),   # the softmax-base pair SWA_FIT_ROPE − SWA_FIT is not part of this experiment (user 11-04: removed)
    # Runlong's router / load-balancing arms on the E2X base (user 11-06 "runlong有了moe routing QB的实验"); evals by reports_xx/router_autoeval.sh
    ("rtq", [(f"PROD1_DPSK_LATENT2_E2X_{c}", "PROD1_DPSK_LATENT2_E2X", f"rtq_l_{k}", col) for c, k, col in RTQ_ARMS], []),
    # KD on the prepack (user 11-06 "runlong一把子关于qbx的实验，记得要测试"): A, B and their CE twins (dashed), every line − MUONH_FIT with the
    # data-derisk tabs (user 11-07 "请用 数据 derisk 和 2.5 vs 2.6 一样的0线"; was B − A / QBX under CE / each arm − its CE twin)
    ("ppf", [(PPF_A, "MUONH_FIT", "ppf_l_a", "--c1"), (PPF_B, "MUONH_FIT", "ppf_l_b", "--c2"),
             (PPF_C, "MUONH_FIT", "ppf_l_c", "--c3"),   # A's q1 on megatron-lm (user 11-10 "这个加到报告里")
             ("PROD1_DPSK_LATENT2_E2X", "MUONH_FIT", "pp_l_lat2x", "--s5", True),
             ("PROD1_DPSK", "MUONH_FIT", "pp_l_pdce", "--v4", True)], []),   # PROD1_DPSK CE in every tab (user 11-07 "这个你就都放进去把")   # reference = the derisk chapter's lat2x line, same label / colour (user 11-07)
]


def x_section(R, t, key):
    """Lines arm − pair from OUR 115-split doc evals (user 11-04: the report's metrics, not the cards' six-set R1), metric buttons + window
    tabs (8K = <ID>, 32K = <ID>_W32K; one dual-window job per checkpoint, reports_xx/launch_prod1_dualwin.sh)."""
    arms, extra = next((a, e) for k, a, e in X_SECTIONS if k == key)
    arms = [tuple(a) + (False,) * (5 - len(a)) for a in arms]   # (arm, pair, label key, colour, dashed)
    rungs = RUNGS[:5]
    # windows = (tab, arm suffix, partner rule); X_PPZERO: the data-derisk zero line (user 11-07), pair = MUONH_FIT
    pz = key in X_PPZERO
    WINS = ([("", lambda p: p), ("_W32K", lambda p: p)] if pz   # the third tab (32K, zero = MUONH_FIT_PP32K_W32K) dropped (user 11-07)
            else [("", lambda p: p), ("_W32K", lambda p: p + "_W32K")])
    wins = t["pp_win"][:len(WINS)]
    ab = key in X_ABS8K
    if ab:
        wins = list(wins) + [("w8abs", t["x_abs_tab"]), ("w32abs", t["x_abs32_tab"])]
    pfor =lambda pair, sfx, w=None: WINS[w][1](pair) if w is not None else pair + sfx   # partner Change ID at window w
    wof = lambda sfx: 0 if sfx == "" else 1                                             # legacy callers pass a suffix: window 0 / 1
    dd = lambda arm, pair, q, mk, sfx, w=None: (lambda d: sum(d) / len(d) if d else None)(
        _paired(R, arm + (WINS[w][0] if w is not None else sfx), pfor(pair, sfx, w if w is not None else (wof(sfx) if pz else None)), q, mk))

    def series_of(mk, mname, w):
        series, xs = [], {}
        for arm, pair, lab, col, dash in arms:
            pts = []
            for q in rungs:
                v = dd(arm, pair, q, mk, "", w)
                if v is None:
                    continue
                xs[q] = N_ACTIVE[q]
                pts.append((N_ACTIVE[q], v, f"{esc(t[lab])} · {q} ({N_LABEL[q]})<br>{mname} {v:+.4f}"))
            if pts:
                series.append((t[lab], col, pts, dash))
        return series, xs

    def chart(mk, mname, w):
        series, xs = series_of(mk, mname, w)
        yr = None
        if pz:   # one y range for the window tabs of a metric, as in the data-derisk chapter (8 % padding, 0 included)
            ys = [0.0] + [p[1] for ww in range(len(WINS)) for s in series_of(mk, mname, ww)[0] for p in s[2]]
            pad = (max(ys) - min(ys)) * 0.08 or 0.01
            yr = (min(ys) - pad, max(ys) + pad)
        return (line_chart(series, xs, "Δloss", lambda v: f"{v:+.3f}", labels=False, zero=True, yrange=yr) if series
                else f'<p class="note">{esc(t["x_pending"])}</p>')

    ABS_LINES = [("MUONH_FIT", None, "x_abs_zl", "--muted", False)] if ab else []   # absolute tabs only (PROD1_DPSK moved into every tab, user 11-07)

    def chart_abs(mk, mname, sfx=""):
        """X_ABS8K: each line's own loss at the 8K (sfx "") or 32K (sfx "_W32K") window, mean over its seeds, nothing subtracted, log-log
        (user 11-07; log-log from Boxiang); + the zero line MUONH_FIT itself in grey (always its 8K score: trained at 8K) and PROD1_DPSK CE
        q0-q5 (Runlong 11-07 "dpsk q4 q5")."""
        series, xs = [], {}
        extra_lines = [(a, p_, lab + ("32" if sfx and a == "MUONH_FIT" else ""), c, d) for a, p_, lab, c, d in ABS_LINES]
        for arm, pair, lab, col, dash in arms + extra_lines:
            pts = []
            for q in RUNGS:
                ch = arm if arm == "MUONH_FIT" else arm + sfx
                vs = [v for v in (val(R, _seed_id(ch, k), q, mk) for k in (1, 2, 3)) if v is not None]
                if not vs:
                    continue
                v = sum(vs) / len(vs)
                xs[q] = N_ACTIVE[q]
                pts.append((N_ACTIVE[q], v, f"{esc(t[lab])} · {q} ({N_LABEL[q]})<br>{mname} {v:.4f}" + (f" ({len(vs)} seeds)" if len(vs) > 1 else "")))
            if pts:
                series.append((t[lab], col, pts, dash))
        if not series:
            return f'<p class="note">{esc(t["x_pending"])}</p>'
        # fitted slope d log L / d log N (least squares over the line's own rungs), written next to the line end (user 11-07 "给一个 fited line的斜率",
        # "斜率直接标记在线旁边"); the slope table under the chart was dropped
        inv = {x: q for q, x in xs.items()}
        def slope(p):
            u = [math.log(x) for x, _, _ in p]; w_ = [math.log(y) for _, y, _ in p]; mu, mw = sum(u) / len(u), sum(w_) / len(w_)
            return sum((a - mu) * (b - mw) for a, b in zip(u, w_)) / sum((a - mu) ** 2 for a in u)
        el = {name: f"{t['x_slope']} {slope(p):+.3f}" for name, col, p, dash in series if len(p) > 1}   # no rung range in the label (user 11-07)
        return line_chart(series, xs, "val_loss (log)", lambda v: f"{v:.2f}", labels=False, ylog=True, endlabels=el)

    def ext(mk):
        if not extra:
            return ""
        cells = lambda arm, pair: "".join("<td>·</td>" if dd(arm, pair, q, mk, "") is None else
                                          f'<td class="{"neg" if dd(arm, pair, q, mk, "") < 0 else "pos"}">{dd(arm, pair, q, mk, ""):+.4f}</td>' for q in rungs[:4])
        return (f'<p class="note" style="margin:10px 0 4px">{esc(t[key + "_sm_h"])}</p><table style="font-size:12px"><thead><tr><th class="l"></th><th class="l">Change ID</th>'
                + "".join(f"<th>{q}</th>" for q in rungs[:4]) + "</tr></thead><tbody>"
                + "".join(f'<tr><td class="l" style="white-space:normal;max-width:420px">{esc(t[lab])}</td><td class="l"><code>{arm}</code> − <code>{pair}</code></td>{cells(arm, pair)}</tr>'
                          for arm, pair, lab in extra) + "</tbody></table>")

    sw = lambda col, dash=False: (f'<svg width="26" height="10" style="vertical-align:middle"><line x1="1" x2="25" y1="5" y2="5" stroke="var({col})" stroke-width="2.5"'
                                  f'{' stroke-dasharray="5 3"' if dash else ""}/></svg>')

    def dd_bpb(arm, pair, q, mk, sfx, w):
        """Same difference in bits per byte (the JSONs carry loss / ppl / bpb per metric; seed-1 runs here)."""
        out = []
        for k in (1, 2, 3):
            ma, mb = R.get((_seed_id(arm + WINS[w][0], k), q)), R.get((_seed_id(pfor(pair, sfx, w), k), q))
            if ma and mb and mk in ma and mk in mb and ma[mk].get("bpb") is not None and mb[mk].get("bpb") is not None:
                out.append(ma[mk]["bpb"] - mb[mk]["bpb"])
        return sum(out) / len(out) if out else None

    def vtab(mk, w):
        """The chart's numbers as a table (line x rung) at this window (user 11-06); + the same in bpb for X_BPB sections (user 11-06 "bpb")."""
        if key not in X_VALTAB:
            return ""
        out = ""
        for unit, f in (("loss", dd), ("bpb", dd_bpb)) if key in X_BPB else (("loss", dd),):
            rows = ""
            for arm, pair, lab, col, dash in arms:
                vs = [f(arm, pair, q, mk, "", w) for q in rungs[:4]]
                cells = "".join("<td>·</td>" if v is None else f'<td class="{"neg" if v < 0 else "pos"}">{v:+.4f}</td>' for v in vs)
                rows += (f'<tr><td>{sw(col, dash)}</td><td class="l" style="white-space:nowrap">{esc(t[lab])}</td>'
                         f'<td class="l"><code>{arm}{WINS[w][0]}</code> − <code>{pfor(pair, "", w)}</code></td>{cells}</tr>')
            cap = f'<p class="note" style="margin:10px 0 2px">{esc(t["x_vt_" + unit])}</p>' if key in X_BPB else ""
            out += (cap + '<table style="font-size:12px;margin-top:8px"><thead><tr><th></th><th class="l"></th><th class="l">Change ID</th>'
                    + "".join(f"<th>{q} · {N_LABEL[q]}</th>" for q in rungs[:4]) + f"</tr></thead><tbody>{rows}</tbody></table>")
        return out

    got = lambda arm, pair: ", ".join(q + lw for q in rungs for w, lw in ((0, " 8K"), (1, " 32K")) if dd(arm, pair, q, "00_overall_tsqw", "", w) is not None) or "·"
    rows = "".join(f'<tr><td>{sw(col, dash)}</td><td class="l" style="white-space:nowrap">{esc(t[lab])}</td><td class="l" style="white-space:normal;max-width:620px">{t[lab + "_d"]}</td>'
                   f'<td class="l" style="white-space:normal;max-width:220px">{got(arm, pair)}</td></tr>' for arm, pair, lab, col, dash in arms) if not pz else ""
    table = (f'<div class="card" style="width:fit-content;max-width:100%;margin:10px auto">'   # no caption (user 11-04: too wordy)
             f'<table style="font-size:12px"><thead><tr><th></th>' + "".join(f'<th class="l">{esc(h)}</th>' for h in t["x_thead"][1:]) + f'</tr></thead><tbody>{rows}</tbody></table></div>')
    if pz:   # legend table in the data-derisk form (user 11-07): line | data | architecture | loss | Change ID (8K window), + foot note
        tb = t[key + "_tab"]
        lrow = lambda lab: tb["rows"].get(lab) or t["pp_tab"]["rows"][lab[len("pp_l_"):]]   # pp_l_* lines reuse the derisk chapter's row
        table = ('<div class="card"><table style="font-size:12.5px"><thead><tr>' + "".join(f'<th class="l">{h}</th>' for h in t["pp_tab"]["head"]) + "</tr></thead><tbody>"
                 + "".join(f'<tr><td class="l" style="white-space:nowrap">{sw(col, dash)}<span style="margin-left:6px">{esc(t[lab])}</span>'
                           + (f'<br><small style="color:var(--muted);margin-left:32px">{esc(t["x_abs_only"])}</small>' if (arm, pair, lab, col, dash) in ABS_LINES else "") + "</td>"
                           + "".join(f'<td class="l" style="white-space:normal">{c}</td>' for c in lrow(lab))
                           + f'<td class="l"><code style="font-size:11px">{arm}</code></td></tr>' for arm, pair, lab, col, dash in arms + ABS_LINES)
                 + f'</tbody></table><p class="note">{tb["foot"]}</p></div>')
    order = ([len(WINS), len(WINS) + 1] + list(range(len(WINS)))) if ab else list(range(len(wins)))   # X_ABS8K: absolute tabs first (user 11-07)
    if ab:   # the difference tabs are named as such once the absolute ones sit next to them
        wins = [(k, t["x_delta_tab"][i]) if i < len(WINS) else (k, n) for i, (k, n) in enumerate(wins)]
    wt = '<div class="rtoggle">' + "".join(f'<button data-r="{wins[i][0]}" class="{"on" if j == 0 else ""}">{esc(wins[i][1])}</button>' for j, i in enumerate(order)) + "</div>"
    # X_PPZERO: each tab carries the data-derisk tab title + zero-line note (pp_a/b/c_h, pp_a/b/c_zero), no card title
    tabhead = lambda i: (f'<div class="ct">{esc(t["pp_" + "abc"[i] + "_h"])}</div><p class="note" style="margin:2px 0 6px">{t["pp_" + "abc"[i] + "_zero"]}</p>' if pz else "")
    absp = lambda m, n, s: (f'<div class="ct">{esc(t["x_abs" + s + "_h"])}</div>'   # no note under the title (user 11-07 "太啰嗦去掉把")
                            f'{chart_abs(m, n, "_W32K" if s else "")}')
    pane = lambda m, n, i: (absp(m, n, "") if ab and i == len(WINS) else absp(m, n, "32") if ab and i == len(WINS) + 1
                            else f'{tabhead(i)}{chart(m, n, i)}{vtab(m, i)}{ext(m) if i == 0 else ""}')   # X_ABS8K: the last two tabs (8K / 32K absolute)
    panes = lambda m, n: "".join(f'<div data-ref="{wins[i][0]}"{"" if j == 0 else " hidden"}>{pane(m, n, i)}</div>'
                                 for j, i in enumerate(order))
    card = (('<div class="card one">' if pz else f'<div class="card one"><div class="ct">{esc(t[key + "_chart"])}</div>') + wt
            + "".join(f'<div data-metric="{m}"{"" if j == 0 else " hidden"}>{panes(m, n)}</div>' for j, (m, n) in enumerate(METRICS)) + "</div>")
    intro = X_INTRO.get(key, {}).get(t["lang"], "")      # explainer right under the title (user 11-06, QB)
    return f"""<h2>{esc(t[key + '_h'])}</h2>{intro}<p class="note">{t[key + '_note']}</p>{table}<div class="msec">{mtoggle(METRICS)}{card}</div>"""


def rtq_section(R, t):
    """Router / load balancing on PROD1_DPSK_LATENT2_E2X (Runlong, 11-06): our doc evals (minus the E2X base) + in-run router health."""
    arms = [(t["rtq_l_base"], "prod1_dpsk_latent2_e2x", "--muted", True)] + [(t[f"rtq_l_{k}"], f"prod1_dpsk_latent2_e2x_{c.lower()}", col, False) for c, k, col in RTQ_ARMS]
    rt = router_trends(t, arms=arms, rungs=("q1", "q2", "q3"), head=False, early=(0.1, RT_RANGE[t["lang"]]), big=True, outside=True, qtabs=True,   # user 11-08: q1 / q2 / q3 tab + 后 90% / 前 10% / 全 step
                       show_legend=False)   # legend dropped (user 11-08 "legand 扔掉"); colours = the chart legend above, hover for names
    # training-loss table (line − zero line at matched steps) removed (user 11-08 "训练 loss（训练日志，早期参考）扔掉")
    return x_section(R, t, "rtq") + (f'<h3>{esc(t["rtq_rt_h"])}</h3><p class="note">{t["rtq_rt_note"]}</p>{rt}' if rt else "")


# Runlong's 6 router variants on top of QBX, q1 (user 11-07 "新跑完的 QBX"; evals 52804-52822, 11-08): own chapter "QBX 变体" with router health
# (user 11-08 "增加一章，叫做qbx变体，记得要有router health"), right after the E2X router chapter. Zero = QBX q1; bpb on the 7 metrics,
# 8K big + 32K small in each cell. Differences vs QBX read off each run's argv (megatron qbx2; RTPIN = code tree overlay_latent_rowball).
QBXV_ARMS = [("EMA5", "ema", "--c1"), ("RLR0P5", "rlr05", "--c2"), ("RLR2", "rlr2", "--c3"), ("RTADAMW", "adamw", "--c4"), ("RTMUONH", "muonh", "--c5"), ("RTPIN", "pin", "--c6")]   # (Change ID suffix, label key, colour)
QBXV_TXT = {
    "zh": {"h": "QBX 上的 router 变体（q1）", "zero": "QBX（AdamH，router LR ×1，β 每步直接换成最新分位数）",
           "note": "Runlong 在 QBX 上只改一处的 6 个变体，都只有 q1、seed 1。每格 = 该行 − QBX q1，单位 bpb，越小越好；大字是 8K 窗口，下面小字是 32K 窗口。最后一行是 E2X 本身（DSV3 bias + seq aux），用来对照 QBX 和 E2X 的差距。",
           "ema": "β 用 EMA 0.5 平滑（默认 0 = 每步直接替换）", "rlr05": "router LR ×0.5", "rlr2": "router LR ×2",
           "adamw": "router 改用 AdamW，LR ×0.19", "muonh": "router 改用 MuonH", "pin": "router 每行单独 hyperball（pin）",
           "e2x": "E2X（不用 QBX，参照）", "abs": "QBX q1 绝对值", "base": "0 线：QBX",
           "chap": "QBX 变体", "chap_note": "Runlong 在 PROD1_DPSK_LATENT2_E2X_QBX（E2X + QBX）上每次只改一处 router 设置的 6 个变体，都只跑了 q1、seed 1（154M 激活参数，15,824 步）。0 线 = QBX q1：router 用 AdamH、LR ×1，β 每个 global batch 直接换成最新的分位数（EMA 0）。QBX 本身是什么见上一章开头的介绍。",
           "eval_h": "评测结果（q1，各行 − QBX，bpb）",
           "rt_h": "训练中的负载均衡（router health，训练日志）",
           "rt_note": "来自训练日志的 router health（每 100 步一行，所有层汇总）。线的颜色和上面评测表每行前的色块一致：灰色虚线 = 0 线 QBX，粉色虚线 = E2X（不用 QBX）参照；鼠标移到图上看各线的名字和数值。dead expert 的百分比按（层数 × 512）算。MaxVio 0 = 完全均衡。",
           "end_h": "训练最后一步的 router health（q1，step {s:,}）", "rt_90": "后 90% step", "rt_10": "前 10% step"},
    "en": {"h": "Router variants on top of QBX (q1)", "zero": "QBX (AdamH, router LR ×1, β replaced by the latest quantile every step)",
           "note": "Runlong's 6 single-change variants of QBX, q1 and seed 1 only. Each cell = row − QBX q1 in bpb, lower is better; the large number is the 8K window, the small one below is 32K. The last row is E2X itself (DSV3 bias + seq aux), for the QBX-vs-E2X gap.",
           "ema": "β smoothed by EMA 0.5 (default 0 = replace every step)", "rlr05": "router LR ×0.5", "rlr2": "router LR ×2",
           "adamw": "router on AdamW, LR ×0.19", "muonh": "router on MuonH", "pin": "router rows as separate hyperballs (pin)",
           "e2x": "E2X (no QBX, reference)", "abs": "QBX q1 absolute", "base": "zero line: QBX",
           "chap": "QBX variants", "chap_note": "Runlong's 6 variants of PROD1_DPSK_LATENT2_E2X_QBX (E2X + QBX), each changing one router setting; q1 and seed 1 only (154M active parameters, 15,824 steps). Zero line = QBX q1: router on AdamH, LR ×1, β replaced by the latest quantile every global batch (EMA 0). What QBX is: see the introduction at the top of the previous chapter.",
           "eval_h": "Eval results (q1, row − QBX, bpb)",
           "rt_h": "Load balance during training (router health, training logs)",
           "rt_note": "Router health from the training logs (one line every 100 steps, all layers). Line colours match the swatches in front of each row of the eval table above: grey dashed = the QBX zero line, pink dashed = E2X (no QBX) for reference; hover for names and values. Dead experts in % of (layers × 512). MaxVio 0 = perfectly balanced.",
           "end_h": "Router health at the last training step (q1, step {s:,})", "rt_90": "Last 90% of steps", "rt_10": "First 10% of steps"}}


def qbxv_section(R, t):
    """Chapter "QBX 变体": eval table (row - QBX q1, bpb) + router health of the 6 variants, QBX and E2X (training logs, q1)."""
    tx = QBXV_TXT[t["lang"]]
    tab = qbx_variants(R, t)
    if not tab:
        return ""
    arms = ([(tx["base"], "prod1_dpsk_latent2_e2x_qbx", "--muted", True)]
            + [(f"{c} · {tx[k]}", f"prod1_dpsk_latent2_e2x_qbx_{c.lower()}", col, False) for c, k, col in QBXV_ARMS]
            + [(tx["e2x"], "prod1_dpsk_latent2_e2x", "--v4", True)])
    rt = router_trends(t, arms=arms, rungs=("q1",), head=False, early=(0.1, RT_RANGE[t["lang"]]),   # user 11-08 前 10% step tab; + 全 step (11-08)
                       big=True, show_legend=False, outside=True)   # statistic buttons outside, above the card (user 11-08)   # user 11-08: val_loss-chart size; legend removed ("...去掉"), colours = swatches of the eval table
    data = [(name, col, dash, router_summary_log(exp, "q1")) for name, exp, col, dash in arms]
    end = ""
    if all(d for *_, d in data):
        last = min(max(d) for *_, d in data)
        stats = [mk for mk, _, _, _ in RT_STATS if any(mk in d[max(d)] for *_, d in data)]
        unit = {k: u for k, _, _, u in RT_STATS}
        fmt = lambda mk, v: f"{v:.{4 if mk == 'upd' else 3}g}{unit[mk]}"
        rows = "".join(f'<tr><td class="l" style="white-space:nowrap"><i style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:7px;background:{swatch_bg(col, dash)}"></i>{esc(name)}</td>'
                       + "".join(f"<td>{fmt(mk, d[max(d)][mk])}</td>" if mk in d[max(d)] else "<td>·</td>" for mk in stats) + "</tr>" for name, col, dash, d in data)
        end = (f'<h3>{esc(tx["end_h"].format(s=last))}</h3><div class="card" style="width:fit-content;max-width:100%;margin:10px auto;overflow-x:auto">'
               f'<table style="font-size:12.5px"><thead><tr><th class="l"></th>' + "".join(f'<th>{esc(t["rt_st_name"][mk])}</th>' for mk in stats)
               + f'</tr></thead><tbody>{rows}</tbody></table></div>')
    return (f'<h2>{esc(tx["chap"])}</h2><p class="note">{tx["chap_note"]}</p>{tab}'
            + (f'<h3>{esc(tx["rt_h"])}</h3><p class="note">{tx["rt_note"]}</p>{rt}{end}' if rt else ""))


def qbx_variants(R, t):
    tx = QBXV_TXT[t["lang"]]
    ref = {w: R.get((f"PROD1_DPSK_LATENT2_E2X_QBX{w}", "q1")) for w in ("", "_W32K")}
    if not ref[""]:
        return ""
    def b(m, mk):
        return m[mk].get("bpb") if m and mk in m else None
    def cell(ch):
        out = []
        for mk, _ in METRICS:
            v = []
            for w in ("", "_W32K"):
                a, z = b(R.get((ch + w, "q1")), mk), b(ref[w], mk)
                v.append(None if a is None or z is None else a - z)
            if v[0] is None:
                out.append("<td>·</td>"); continue
            s = lambda d: f'<span class="{"neg" if d < 0 else "pos" if d > 0 else ""}">{d:+.4f}</span>'
            out.append(f'<td>{s(v[0])}<br><small>{s(v[1]) if v[1] is not None else "·"}</small></td>')
        return "".join(out)
    rows = [(f"{c} · {tx[k]}", f"PROD1_DPSK_LATENT2_E2X_QBX_{c}", col) for c, k, col in QBXV_ARMS if R.get((f"PROD1_DPSK_LATENT2_E2X_QBX_{c}", "q1"))]
    if not rows:
        return ""
    sw = lambda col, dash=False: f'<i style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:7px;background:{swatch_bg(col, dash)}"></i>'
    body = "".join(f'<tr><td class="l" style="white-space:nowrap">{sw(col)}{esc(n)}<br><small style="color:var(--muted)">{esc(ch)}</small></td>{cell(ch)}</tr>' for n, ch, col in rows)
    body += f'<tr><td class="l" style="white-space:nowrap;color:var(--muted)">{sw("--v4", True)}{esc(tx["e2x"])}<br><small>PROD1_DPSK_LATENT2_E2X</small></td>{cell("PROD1_DPSK_LATENT2_E2X")}</tr>'
    absr = "".join(f'<td>{b(ref[""], mk):.4f}<br><small>{b(ref["_W32K"], mk):.4f}</small></td>' if b(ref[""], mk) is not None and b(ref["_W32K"], mk) is not None else "<td>·</td>" for mk, _ in METRICS)
    body += f'<tr><td class="l" style="white-space:nowrap;color:var(--muted)">{sw("--muted", True)}{esc(tx["abs"])}<br><small>{esc(tx["zero"])}</small></td>{absr}</tr>'
    return (f'<h3>{esc(tx["eval_h"])}</h3><p class="note">{tx["note"]}</p><div class="card" style="width:fit-content;max-width:100%;margin:10px auto;overflow-x:auto">'
            f'<table style="font-size:12.5px"><thead><tr><th class="l"></th>' + "".join(f"<th>{esc(n)}</th>" for _, n in METRICS)
            + f'</tr></thead><tbody>{body}</tbody></table></div>')


def ppf_section(R, t):
    """Data derisk round 2 (E2X KD on the tc_full prepack, A (E2X router) vs B (E2X + QBX), 11-06): our doc evals + router health of A / B from
    the training logs. The matched-step training-loss table B − A was removed and the CE twins dropped from router health (user 11-07)."""
    rq = ("q1", "q2", "q3")
    arms = [(t["ppf_rt_a"], PPF_STEM_A, "--c1", False), (t["ppf_rt_b"], PPF_STEM_B, "--c2", False)]
    rt = router_trends(t, arms=arms, rungs=rq, head=False, early=(0.1, RT_RANGE[t["lang"]]), big=True, outside=True, qtabs=True,
                       show_legend=False, defs=False)   # the new router-health format of the E2X router chapter (user 11-08 "按照我们的新router health格式")
    # a table introducing the 9 statistics (user 11-08 "在 数据 derisk round 2 下面 写个表格来介绍下"); replaces the per-statistic note under the chart
    st = (f'<div class="card" style="width:fit-content;max-width:100%;margin:10px auto"><div class="ct" style="text-align:center">{esc(t["ppf_st_h"])}</div>'
          f'<p class="note" style="margin:2px 0 8px">{t["ppf_st_note"]}</p><table style="font-size:12.5px"><thead><tr>'
          + "".join(f'<th class="l">{esc(x)}</th>' for x in t["ppf_st_head"]) + "</tr></thead><tbody>"
          + "".join(f'<tr><td class="l" style="white-space:nowrap"><b>{esc(t["rt_st_name"][mk])}</b></td><td class="l" style="white-space:normal;max-width:430px">{t["ppf_st_rows"][mk][0]}</td>'
                    f'<td class="l" style="white-space:normal;max-width:360px">{t["ppf_st_rows"][mk][1]}</td></tr>' for mk, _, _, _ in RT_STATS)
          + "</tbody></table></div>")
    # arm C = A's q1 rerun on megatron-lm xxdev (user 11-10): what was checked + C vs A per metric, both windows
    ev = [R.get((ch, "q1")) for ch in (PPF_C, PPF_A, PPF_C + "_W32K", PPF_A + "_W32K")]
    cc = ""
    if all(ev):
        c8, a8, c32, a32 = ev
        cell = lambda c, a, mk: f'<td>{c[mk]["loss"]:.4f}</td><td>{a[mk]["loss"]:.4f}</td><td>{c[mk]["loss"] - a[mk]["loss"]:+.4f}</td>'
        cc = (f'<h3>{esc(t["ppf_c_h"])}</h3><p class="note">{t["ppf_c_note"]}</p>'
              '<div class="card" style="width:fit-content;max-width:100%;margin:10px auto"><table style="font-size:12.5px"><thead>'
              f'<tr><th></th><th colspan="3">{esc(t["ppf_c_win"][0])}</th><th colspan="3">{esc(t["ppf_c_win"][1])}</th></tr>'
              f'<tr><th class="l">{esc(t["ppf_c_head"][0])}</th>' + "".join(f"<th>{esc(x)}</th>" for x in t["ppf_c_head"][1:]) * 2 + "</tr></thead><tbody>"
              + "".join(f'<tr><td class="l">{esc(name)}</td>{cell(c8, a8, mk)}{cell(c32, a32, mk)}</tr>'
                        for mk, name in METRICS if all(mk in e for e in ev))
              + "</tbody></table></div>")
    return x_section(R, t, "ppf") + cc + (f'<h3>{esc(t["ppf_rt_h"])}</h3><p class="note">{t["ppf_rt_note"]}</p>{st}{rt}' if rt else "")


def best_arch_section(t):
    rungs = ["q0", "q1", "q2", "q3"]
    rows = []
    for ck, arms in BEST_ROWS:
        for j, (arm, pair, lk) in enumerate(arms):
            cells = []
            for q in rungs:
                a, b = _card_s6(arm, q), _card_s6(pair, q)
                if a is None or b is None:
                    cells.append("<td>·</td>")
                else:
                    d = a - b
                    cells.append(f'<td class="{"neg" if d < 0 else "pos"}">{d:+.4f}</td>')
            rows.append("<tr>" + (f'<td class="l" rowspan="{len(arms)}" style="font-weight:600;vertical-align:top;white-space:normal;max-width:220px">{t[ck]}</td>' if j == 0 else "")
                        + f'<td class="l" style="white-space:normal;max-width:430px">{t[lk]}</td><td class="l"><code>{arm}</code> − <code>{pair}</code></td>{"".join(cells)}</tr>')
    head = "".join(f'<th class="l">{esc(x)}</th>' for x in t["ba_thead"]) + "".join(f"<th>{q}</th>" for q in rungs)
    tb = t["ba_tab"]                                      # grouped table, no scores (user 11-03): group | item | best (ours) | PROD (Runlong)
    same = ("同左", "same", "—")
    trs = []
    for g, items in tb["groups"]:
        for j, (it, best, prod) in enumerate(items):
            diff = prod not in same[:2]
            trs.append("<tr>" + (f'<td class="l" rowspan="{len(items)}" style="font-weight:700;vertical-align:top;white-space:nowrap;border-right:1px solid var(--border)">{esc(g)}</td>' if j == 0 else "")
                       + f'<td class="l" style="white-space:nowrap;font-weight:600{";color:var(--a08)" if diff else ""}">{it}</td>'
                       + f'<td class="l" style="white-space:normal">{best}</td><td class="l" style="white-space:normal{";color:var(--muted)" if not diff else ""}">{prod}</td></tr>')
    return f"""<h2>{esc(t['ba_h'])}</h2><div class="card"><table style="font-size:13.5px;table-layout:fixed;width:100%"><colgroup><col style="width:9%"><col style="width:12%"><col style="width:46%"><col style="width:33%"></colgroup>
<thead><tr>{"".join(f'<th class="l">{h}</th>' for h in tb["head"])}</tr></thead><tbody>{"".join(trs)}</tbody></table><p class="note">{tb["foot"]}</p></div>"""



# Progressive stack under the best-architecture table (user 11-05 "progressive + 的各种trick，清晰的看出improvement"): each row adds one
# item to the row above, all CE / MuonH / the 650B corpus_v2 / seed 1, our 115-split doc evals at the 8K window. The order follows the
# single-variable pairs that exist (the user's list put the shared expert before SWA; it was only run on the SWA + sink base).
# SWA_FIT_DSV3 (SWA 2048 + NoPE without sink, q0 only) row removed (user 11-05)
STACK = ["OPT_MUONH", "MUONH_FIT", "MUONH_FIT_DSV3", "MUONH_FIT_DSV3_SWS", "MUONH_FIT_DSV3_SWS_SHARED_RS2P5",
         "PROD1_DPSK", "PROD1_DPSK_LATENT2", "PROD1_DPSK_LATENT2_E2X"]
STACK_RUNGS = ["q0", "q1", "q2", "q3"]


def _stack_vals(R, mkey):
    """{q: [(step, value, delta vs the nearest earlier step with a value, skipped a step?)]} on metric mkey."""
    out = {}
    for q in STACK_RUNGS:
        rows, prev = [], None
        for i, c in enumerate(STACK):
            v = val(R, c, q, mkey)
            if v is None:
                continue
            rows.append((i, v, None if prev is None else v - prev[1], prev is not None and prev[0] != i - 1))
            prev = (i, v)
        out[q] = rows
    return out


STACK_COL = ["--muted", "--c1", "--c2", "--c3", "--c4", "--c5", "--c6", "--s4"]   # step 1-8: categorical slots in fixed order


def stack_chart(R, t, mkey, mname):
    """Same form as the optimizer chart (user 11-05): x = active params (rung, log), y = step k - step 0 (MuonH, old LR), one line per step."""
    series, xs = [], {}
    for i, c in enumerate(STACK):
        if i == 0:
            continue
        pts = []
        for q in STACK_RUNGS:
            a, b = val(R, c, q, mkey), val(R, STACK[0], q, mkey)
            if a is None or b is None:
                continue
            xs[q] = N_ACTIVE[q]
            pts.append((N_ACTIVE[q], a - b, f"{c} · {q} ({N_LABEL[q]})<br>{esc(mname)} {a - b:+.4f}"))
        if pts:
            series.append((c, STACK_COL[i], pts, False))
    if not series:
        return f'<p class="note">{esc(t["pending"])}</p>'
    return line_chart(series, xs, t["st_y"], lambda v: f"{v:+.3f}", labels=False, zero=True)   # no legend (user 11-05): the table below carries the swatches


def _stack_cfg(t, i):
    """Row i's setting per aspect: each aspect is a list of (first row, text) spans."""
    return [next(txt for st, txt in reversed(spans) if st <= i) for _, spans in t["st_aspects"]]


def stack_table(R, t, mkey):
    sv = _stack_vals(R, mkey)
    base = {q: next((v for i, v, _, _ in sv[q] if i == 0), None) for q in STACK_RUNGS}
    trs = []
    for i, c in enumerate(STACK):
        cells = []
        for q in STACK_RUNGS:
            r = next(((v, d, sk) for j, v, d, sk in sv[q] if j == i), None)
            if r is None:
                cells.append("<td>·</td>")
            elif i == 0:
                cells.append(f'<td style="color:var(--muted)">{r[0]:.4f}</td>')
            else:
                v, d, sk = r   # cells = cumulative only (this row - OPT_MUONH), the per-step delta dropped (user 11-05 "这个改累积就好")
                cv = v - base[q] if base[q] is not None else None   # (not "c": that is the row's Change ID)
                cells.append(f'<td><span class="{"neg" if cv < 0 else "pos"}" style="font-weight:600">{cv:+.4f}</span>{"†" if sk else ""}</td>' if cv is not None else "<td>·</td>")
        sw = (f'<i style="display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:7px;vertical-align:-1px;background:var({STACK_COL[i]})"></i>'
              if i else '<i style="display:inline-block;width:10px;height:10px;margin-right:7px"></i>')   # colour square (user 11-05); row 0 = the zero line
        cfg = "".join(f'<td class="l" style="white-space:normal;min-width:110px;max-width:190px;font-size:11.5px;'
                      + ("background:color-mix(in srgb,var(--a08) 14%,transparent);font-weight:600" if i and a != _stack_cfg(t, i - 1)[k] else "color:var(--muted)" if i else "")
                      + f'">{a}</td>' for k, a in enumerate(_stack_cfg(t, i)))
        trs.append(f'<tr><td class="l" style="white-space:nowrap">{sw}<code style="font-weight:600">{c}</code></td>'
                   f'{cfg}{"".join(cells)}</tr>')   # changed aspects highlighted (user 11-05)
    head = ("".join(f'<th class="l">{esc(x)}</th>' for x in t["st_thead"]) + "".join(f'<th class="l">{esc(h)}</th>' for h, _ in t["st_aspects"])
            + "".join(f"<th>{q} · {N_LABEL[q]}</th>" for q in STACK_RUNGS))
    return f'<table style="font-size:12.5px"><thead><tr>{head}</tr></thead><tbody>{"".join(trs)}</tbody></table>'


def stack_section(R, t):
    panes = "".join(f'<div data-metric="{m}"{"" if j == 0 else " hidden"}><div class="card" style="width:fit-content;max-width:100%;margin:10px auto">{stack_table(R, t, m)}</div>'
                    f'<div class="card one">{stack_chart(R, t, m, n)}</div></div>'   # chart title removed (user 11-05 "title 去掉")   # table above the chart (user 11-05)
                    for j, (m, n) in enumerate(METRICS))
    notes = "".join(f"<li>{x}</li>" for x in t["st_notes"])
    return f"""<div class="msec"><h3>{esc(t['st_h'])}</h3><p class="note">{t['st_note']}</p>{mtoggle(METRICS)}{panes}<ul class="f note">{notes}</ul></div>"""


# 80B pack plan (32K training) chapter (user 11-03 "有新的实验，麻烦放在报告里"): data derisk (MUONH_FIT base: CE / KD 2.6 / KD 2.5 on the
# same plan) and the best architecture + KD, 2.6 vs 2.5. Every checkpoint is evaluated twice (user): 8K window = <ID>, 32K window = <ID>_W32K.
PP_LINES = [  # (key, Change ID of the 8K eval, label key, colour, dashed)
    ("ce", "MUONH_FIT_PP32K", "pp_l_ce", "--s1", False),
    ("kd26", "KD_MUONH_A0P8_S64Y_FIT_PP32K", "pp_l_kd26", "--g08", False),
    ("kd25", "KD_MUONH_A0P8_S64Y_FIT_MIMO25_PP32K", "pp_l_kd25", "--a08", False),
    ("pd25", "PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K", "pp_l_pd25", "--v7", False),   # KD 2.5 on PROD1_DPSK, same pack plan (launched 11-04 15:53; evals via pp_autoeval)
    # best architecture + KD 2.6 / 2.5 (PROD1_DPSK_LATENT2_KD_A0P8_S64Y[_MIMO25]_PP32K) not drawn (user 11-03: probably cancelled for good)
    ("old26", "KD_MUONH_A0P8_S64Y_FIT_DOCMASK", "pp_l_old26", "--g07", True),       # old data, 8K training: references (8K window only)
    ("old25", "KD_MUONH_FIT_A0P8_S64Y_TCS_MIMO25", "pp_l_old25", "--a07", True),   # 45B KD corpus (no 2.5 run on the 650B data); q3 dropped (TPP 100), user 11-03
    # ("swa", "SWA_FIT_SINK_W512", "pp_l_swa", "--c1", True),   # removed from the chapter (user 11-05 "这两个从数据 derisk 和 2.5 vs 2.6 去掉吧")
    ("pdce", "PROD1_DPSK", "pp_l_pdce", "--v4", True),   # 650B, 32K-trained (+ doc mask), CE, PROD1_DPSK itself: reference at both windows (user 11-05 "CE · PROD1_DPSK 650B 数据也画在chart里")
    # ("lat", "PROD1_DPSK_LATENT2", "pp_l_lat", "--s4", True),   # removed from the chapter (user 11-05 "这两个从数据 derisk 和 2.5 vs 2.6 去掉吧")
    ("lat2x", "PROD1_DPSK_LATENT2_E2X", "pp_l_lat2x", "--s5", True),   # same, routed experts x2 (512 pick 16; user 11-05 "你把E2X也加上吧"); q3 still training   # 650B, 32K-trained (+ doc mask), CE, PROD1_DPSK + latent MoE iso-param: reference at both windows (user 11-05 "latent MoE（等参数）也放进去")   # 650B, 8K-trained, CE, SWA 512 + sink 5:1 (global layers NoPE): reference that can be read at 32K (user 11-04; 32K evals 37992-38003)
]


def packplan_section(R, t):
    L = {k: (ch, lab, col, dash) for k, ch, lab, col, dash in PP_LINES}
    sw = lambda col, dash: (f'<svg width="28" height="10" style="vertical-align:middle;margin-right:6px"><line x1="1" y1="5" x2="27" y2="5" stroke="var({col})" '
                            f'stroke-width="2.6"{' stroke-dasharray="5 3"' if dash else ""}/></svg>')
    tb = t["pp_tab"]
    legend_tab = ('<div class="card"><table style="font-size:12.5px"><thead><tr>' + "".join(f'<th class="l">{h}</th>' for h in tb["head"]) + "</tr></thead><tbody>"
                  + "".join(f'<tr><td class="l" style="white-space:nowrap">{sw(col, dash)}{esc(t[lab])}</td>' + "".join(f'<td class="l" style="white-space:normal">{c}</td>' for c in tb["rows"][k])
                            + f'<td class="l"><code style="font-size:11px">{ch}</code></td></tr>' for k, ch, lab, col, dash in PP_LINES)
                  + f'</tbody></table><p class="note">{tb["foot"]}</p></div>')

    def chart(pairs, mk, mname, w=560, h=360, yr=None):
        """pairs = [(label, colour, dashed, arm, partner)]; x = rung, y = arm - partner (per-seed paired mean). yr = fixed y range."""
        series, xs = series_of(pairs, mk, mname)
        if not series:
            return f'<p class="note">{esc(t["pp_none"])}</p>'
        # no legend row (user 11-04 "leged 去掉了"): the table above the chart already lists the lines with their colours
        return line_chart(series, xs, t["pp_y"], lambda v: f"{v:+.3f}", w=w, h=h, labels=False, zero=True, yrange=yr)

    def shared_yr(mk, mname):
        """One y range for the three window tabs of a metric (user 11-04 "这三个可以 switch 的图的 y-axis 一样"), 8 % padding, 0 included."""
        ys = [0.0] + [p[1] for prs in (a_pairs, b_pairs, c_pairs) for s in series_of(prs, mk, mname)[0] for p in s[2]]
        pad = (max(ys) - min(ys)) * 0.08 or 0.01
        return (min(ys) - pad, max(ys) + pad)

    def series_of(pairs, mk, mname):
        series, xs = [], {}
        for name, col, dash, arm, partner in pairs:
            pts = []
            for q in RUNGS:
                if arm == "KD_MUONH_FIT_A0P8_S64Y_TCS_MIMO25" and q == "q3":   # its q3 ran at TPP 100, not 200 (user 11-03: drop the point)
                    continue
                d = _paired(R, arm, partner, q, mk)
                if not d:
                    continue
                v = sum(d) / len(d); xs[q] = N_ACTIVE[q]
                pts.append((N_ACTIVE[q], v, f"{esc(name)} · {q} ({N_LABEL[q]})<br>{mname} {v:+.4f}" + (f"（{len(d)} seeds）" if len(d) > 1 else "")))
            if pts:
                series.append((name, col, pts, dash))
        return series, xs

    a_pairs =[(t[L[k][1]], L[k][2], L[k][3], L[k][0], "MUONH_FIT") for k in ("ce", "kd26", "kd25", "pd25", "old26", "old25", "pdce", "lat2x")]
    b_pairs = [(t[L[k][1]], L[k][2], L[k][3], L[k][0] + "_W32K", "MUONH_FIT") for k in ("ce", "kd26", "kd25", "pd25", "pdce", "lat2x")]   # swa: 8K-trained but sliding-window, so readable at 32K (user 11-04);   old26 (650B KD 2.6 + mask, 8K-trained) at the 32K window: +0.17 q0 (beyond its training length), not drawn (11-03)   # zero = MUONH_FIT as is (its 8K score; user 11-03: a reference baseline)
    # third tab (user 11-04 "8K / 32K 后面加一个 tag，32K 用新跑的 CE 做 0 线"): 32K window, zero = MUONH_FIT_PP32K at the 32K window (seed-paired)
    c_pairs = [(t[L[k][1]], L[k][2], L[k][3], L[k][0] + "_W32K", L["ce"][0] + "_W32K") for k in ("kd26", "kd25", "pd25", "pdce", "lat2x")]
    # teacher chart "KD 2.5 - KD 2.6 (same pack plan)" removed 11-04 (user "不用这个"); the 2.5 / 2.6 lines stay in charts A / B
    card = lambda title, zero, body: (f'<div class="card" style="min-width:0;width:100%;margin:10px 0;box-sizing:border-box"><div class="ct">{esc(title)}</div>'
                                      f'<p class="note" style="margin:2px 0 6px">{zero}</p>{body}</div>')
    # 8K / 32K window as tabs of one card (user 11-04 "用不同的tab吧"; was two cards side by side): the rtoggle switches [data-ref] inside the card,
    # the metric toggle switches [data-metric], so the chosen window stays when the metric changes
    wt = '<div class="rtoggle">' + "".join(f'<button data-r="{k}" class="{"on" if j == 0 else ""}">{esc(n)}</button>' for j, (k, n) in enumerate(t["pp_win"])) + "</div>"
    def old32(mk):
        """650B KD 2.6 + mask (8K-trained) at the 32K window - MUONH_FIT, one row under the 32K chart instead of a line (user 11-04)."""
        cells, any_ = [], False
        for q in RUNGS[:4]:
            d = _paired(R, L["old26"][0] + "_W32K", "MUONH_FIT", q, mk)
            if d:
                any_ = True
            cells.append(f'<td><b>{sum(d) / len(d):+.3f}</b>' + (f'<span style="color:var(--muted);font-size:11px">{"（" if t["lang"] == "zh" else " ("}{len(d)} seeds{"）" if t["lang"] == "zh" else ")"}</span>' if len(d) > 1 else "") + "</td>"
                         if d else '<td style="color:var(--muted)">—</td>')
        if not any_:
            return ""
        return (f'<p class="note" style="margin:10px 0 4px">{t["pp_old32_h"]}</p><table style="font-size:12.5px"><thead><tr><th class="l"></th>'
                + "".join(f"<th>{q} ({N_LABEL[q]})</th>" for q in RUNGS[:4]) + f'</tr></thead><tbody><tr><td class="l">{esc(t[L["old26"][1]])}</td>{"".join(cells)}</tr></tbody></table>')
    panes = lambda m, n: "".join(f'<div data-ref="{k}"{"" if i == 0 else " hidden"}><div class="ct">{esc(h)}</div><p class="note" style="margin:2px 0 6px">{z}</p>'
                                 f'{chart(pairs, m, n, w=760, h=360, yr=shared_yr(m, n))}</div>'   # 650B-reference-at-32K table under the 32K chart removed (user 11-05 "删除？")
                                 for i, ((k, _), h, z, pairs) in enumerate(zip(t["pp_win"], (t["pp_a_h"], t["pp_b_h"], t["pp_c_h"]),
                                                                               (t["pp_a_zero"], t["pp_b_zero"], t["pp_c_zero"]), (a_pairs, b_pairs, c_pairs))))
    ab = ('<div class="card one">' + wt   # user 11-04: same size as the doc-mask chart (centred .card.one, max 880 px, 760 x 360)
          + "".join(f'<div data-metric="{m}"{"" if j == 0 else " hidden"}>{panes(m, n)}</div>' for j, (m, n) in enumerate(METRICS)) + "</div>")
    have = sorted({(c[:-5] if c.endswith("_W32K") else c, q, "32K" if c.endswith("_W32K") else "8K") for c, q in R
                   if any(c in (ch, ch + "_W32K") for _, ch, *_ in PP_LINES if not ch.startswith("KD_MUONH_A0P8_S64Y_FIT_D") and "TCS" not in ch)})
    got = ("；" if t["lang"] == "zh" else "; ").join(f"{t[L[k][1]]} {q} ({w})" for k in L for c, q, w in have if c == L[k][0]) or "—"
    return f"""<h2>{esc(t['pp_h'])}</h2><p class="note">{t['pp_note']}</p>
{legend_tab}<div class="msec">{mtoggle(METRICS)}{ab}</div>"""   # "已有评测：..." list removed (user 11-05 "删除")


# Dense first layer (layer 0 not MoE): its own chapter with the process and the results (user 11-03). Source of the process: the router-NE
# session notes (router_ne_xx/NOTES.md 2026-11-01 16:xx .. 2026-11-03 10:5x); numbers = the shared cards' R1 val_doc six-set mean (seed 1).
D0_PAIRS = [  # (label key, arm, pair)
    ("d0_r_d1", "MUONH_FIT_DSV3_SWS_DENSE1", "MUONH_FIT_DSV3_SWS"),
    ("d0_r_d2", "MUONH_FIT_DSV3_SWS_DENSE2", "MUONH_FIT_DSV3_SWS"),
    ("d0_r_d2d1", "MUONH_FIT_DSV3_SWS_DENSE2", "MUONH_FIT_DSV3_SWS_DENSE1"),
    ("d0_r_fa0", "MUONH_FIT_DSV3_SWS_DENSE1_FA0", "MUONH_FIT_DSV3_SWS_DENSE1"),
    ("d0_r_phib", "ROUTER_NE_1EM5_SWS_DENSE1", "MUONH_FIT_DSV3_SWS_DENSE1"),
    ("d0_r_phinb", "ROUTER_NE_1EM5_NOBIAS_SWS_DENSE1", "MUONH_FIT_DSV3_SWS_DENSE1"),
]
D0_HEALTH = [  # (label key, all-MoE run, dense-layer-0 run)
    ("d0_h_ctrl", "MUONH_FIT_DSV3_SWS", "MUONH_FIT_DSV3_SWS_DENSE1"),
    ("d0_h_phib", "ROUTER_NE_1EM5_SWS", "ROUTER_NE_1EM5_SWS_DENSE1"),
    ("d0_h_phinb", "ROUTER_NE_1EM5_NOBIAS_SWS", "ROUTER_NE_1EM5_NOBIAS_SWS_DENSE1"),
]
D0_EXP = {"MUONH_FIT_DSV3_SWS": "muonh_fit_dsv3_sws", "MUONH_FIT_DSV3_SWS_DENSE1": "muonh_fit_dsv3_sws_dense1",   # change id -> run dir name
          "ROUTER_NE_1EM5_SWS": "muonh_fit_dsv3_sws_ne_1em5", "ROUTER_NE_1EM5_SWS_DENSE1": "muonh_fit_dsv3_sws_ne_1em5_dense1",
          "ROUTER_NE_1EM5_NOBIAS_SWS": "muonh_fit_dsv3_sws_ne_1em5_nobias", "ROUTER_NE_1EM5_NOBIAS_SWS_DENSE1": "muonh_fit_dsv3_sws_ne_1em5_nobias_dense1"}


D0_LINES = [  # (Change ID, label key, colour, dashed) - all minus MUONH_FIT (vanilla softmax router, MuonH, fitted LR, CE), user 11-03
    # colour = router (DSV3 / negative entropy + bias / negative entropy without bias), dashed = all-MoE, solid = dense layer 0; the same colours
    # are used by the router-health charts below, so the table above the chart is the legend of both (user 11-03)
    ("MUONH_FIT_DSV3", "d0_l_dsv3", "--v8", True), ("MUONH_FIT_DSV3_SWS", "d0_l_sws", "--s1", True),
    ("MUONH_FIT_DSV3_SWS_DENSE1", "d0_l_d1", "--s1", False), ("MUONH_FIT_DSV3_SWS_DENSE2", "d0_l_d2", "--g09", False),
    ("MUONH_FIT_DSV3_SWS_DENSE1_FA0", "d0_l_fa0", "--v7", False),
    ("ROUTER_NE_1EM5_SWS", "d0_l_phib_moe", "--a07", True), ("ROUTER_NE_1EM5_SWS_DENSE1", "d0_l_phib", "--a07", False),
    ("ROUTER_NE_1EM5_NOBIAS_SWS", "d0_l_phinb_moe", "--a09", True), ("ROUTER_NE_1EM5_NOBIAS_SWS_DENSE1", "d0_l_phinb", "--a09", False),
]


def dense0_legend(t):
    """One row per line of the dense-layer-0 chart (user 11-03: what each line is), same order / colours / dashes as D0_LINES."""
    tb = t["d0_ltab"]
    sw = lambda col, dash: (f'<svg width="28" height="10" style="vertical-align:middle;margin-right:6px"><line x1="1" y1="5" x2="27" y2="5" stroke="var({col})" '
                            f'stroke-width="2.6"{' stroke-dasharray="5 3"' if dash else ""}/></svg>')
    cell = lambda c: (f'<td style="text-align:center;font-size:14px">{"✓" if c else '<span style="color:var(--muted)">✗</span>'}</td>' if isinstance(c, bool)
                      else f'<td class="l" style="white-space:normal">{c}</td>')    # router components as ✓ / ✗ (user 11-03)
    rows = "".join(f'<tr><td class="l" style="white-space:nowrap">{sw(col, dash)}{esc(t[lab])}</td>' + "".join(cell(c) for c in tb["rows"][lab])
                   + f'<td class="l"><code style="font-size:11px">{ch}</code></td></tr>' for ch, lab, col, dash in D0_LINES)
    return (f'<div class="card"><table style="font-size:12.5px"><thead><tr>{"".join(f'<th class="l" style="white-space:normal;vertical-align:bottom{";text-align:center;max-width:110px" if 3 <= j <= 6 else ""}">{h}</th>' for j, h in enumerate(tb["head"]))}</tr></thead>'
            f'<tbody>{rows}</tbody></table><p class="note">{tb["foot"]}</p></div>')


def dense0_chart(R, t):
    def chart(mk, mname):
        series, xs = [], {}
        for ch, lab, col, dashed in D0_LINES:
            pts = []
            for q in RUNGS:
                d = _paired(R, ch, "MUONH_FIT", q, mk)
                if not d:
                    continue
                v = sum(d) / len(d); xs[q] = N_ACTIVE[q]
                pts.append((N_ACTIVE[q], v, f"{esc(t[lab])} · {q} ({N_LABEL[q]})<br>{mname} {v:+.4f}" + (f"（{len(d)} seeds）" if len(d) > 1 else "")))
            if pts:
                series.append((t[lab], col, pts, dashed))
        if not series:
            return f'<p class="note">{esc(t["pending"])}</p>'
        return line_chart(series, xs, t["d0_y"], lambda v: f"{v:+.3f}", labels=False, zero=True)
    blocks = "".join(f'<div data-metric="{m}"{"" if j == 0 else " hidden"}>{chart(m, n)}</div>' for j, (m, n) in enumerate(METRICS))
    missing = [t[lab] for ch, lab, _, _ in D0_LINES if not any(_paired(R, ch, "MUONH_FIT", q, "00_overall_tsqw") for q in RUNGS)]
    pend = f'<p class="note">{esc(t["d0_missing"])}{"、".join(esc(x) for x in missing) if t["lang"] == "zh" else ", ".join(esc(x) for x in missing)}</p>' if missing else ""
    return f'<div class="msec">{mtoggle(METRICS)}<div class="card one"><div class="ct">{esc(t["d0_chart"])}</div>{blocks}{pend}</div></div>'


def _card_dead(change, q):
    """(dead experts, MaxVio max) on val_v2 from a card's packed router readout at rung q, or (None, None)."""
    try:
        r = (json.load(open(os.path.join(CARDS_DIR, change, "scores.json"))).get("rungs") or {}).get(q) or {}
    except (OSError, ValueError):
        return None, None
    m = (r.get("val_maxvio") or {}).get("val_v2_decon") or {}
    return m.get("dead_total"), m.get("maxvio_max")


def dense0_section(R, t):
    rungs = ["q1", "q2", "q3"]
    def cell(a, b, q):
        x, y = _card_s6(a, q), _card_s6(b, q)
        if x is None or y is None:
            return "<td>·</td>"
        return f'<td class="{"neg" if x - y < 0 else "pos"}">{x - y:+.4f}</td>'
    rows = "".join(f'<tr><td class="l" style="white-space:normal;max-width:420px">{t[lk]}</td><td class="l"><code>{a}</code> − <code>{b}</code></td>'
                   + "".join(cell(a, b, q) for q in rungs) + "</tr>" for lk, a, b in D0_PAIRS)
    res = (f'<table style="font-size:12.5px"><thead><tr>' + "".join(f'<th class="l">{esc(x)}</th>' for x in t["d0_thead"])
           + "".join(f"<th>{q}</th>" for q in rungs) + f"</tr></thead><tbody>{rows}</tbody></table>")
    line = {ch: (lab, col, dash) for ch, lab, col, dash in D0_LINES}
    # same form as the Router chapter's "router 健康统计" (user 11-03: align): training-log summaries, one chart per rung, stat toggle;
    # solid = dense layer 0, dashed = all-MoE, same colour = same router
    arms = [(t[line[ch][0]], D0_EXP[ch], line[ch][1], line[ch][2]) for _, moe, dense in D0_HEALTH for ch in (dense, moe)]
    health = router_trends(t, arms=arms, rungs=("q1", "q2", "q3", "q4"), head=False, show_legend=False)   # legend = the table above (user 11-03)
    return f"""<h2>{esc(t['d0_h'])}</h2><p class="note">{t['d0_note']}</p>
{dense0_legend(t)}
{dense0_chart(R, t)}
<h3>{esc(t['d0_hl_h'])}</h3><p class="note">{t['d0_hl_note']}</p>{health}<p class="note">{t['d0_hl_foot']}</p>"""

# tab "prod downstream" (named "downstream eval" until 11-14; rewritten 11-13, user: "新的是66个指标，不再是nemov6的75个，根据这个彻底改写 外部参考，和 downstream eval 两个tab页面"):
# "实验列表" + "全量测评重点" + "全量测评结果" for OUR checkpoints on the unified harness's task list (cosmos_eval_section.py; every
# results_xx/cosmos_eval/<code>.json with "ours": true, so a newly collected checkpoint appears by itself), next to the external baselines.
# 11-13 (user: "这个我们主要收集的是这个prod测试，我们要跟踪这个"): the page's main line is the production run ULTRA59B_MUONH_KD_QBK3:
# every evaluated checkpoint of the run is a row of the list and of the tables (collect_cosmos_eval.py finds them by itself: JSONs with
# "track"). Above the list: ce_steps(), the key tier against the training step, charts only (user 11-13 evening; the chapter with notes
# and tables that stood there was removed that afternoon on "这个删除").
# Rows: baselines, then the production run's checkpoints by step, then ladder checkpoints (PROD1_DPSK first, the other Change IDs A-Z,
# each by rung). Ticked by default: the production run's latest checkpoint only; the others are listed but not ticked.
# colours (user 11-09: colour squares in "实验列表", one colour per model): a Change ID with several rungs = one ramp light -> dark by rung
# (PROD1_DPSK blue, LATENT2 aqua, LATENT2_E2X orange); single-rung Change IDs = their own hue; the production run's checkpoints = a grey scale by step (_track_css); baselines blue; all lines solid
DOWN_RAMP = {"PROD1_DPSK": ["--k07", "--k08", "--k09"], "PROD1_DPSK_LATENT2": ["--g07", "--g09"], "PROD1_DPSK_LATENT2_E2X": ["--a07", "--a09"]}
DOWN_ONE = {"PROD1_DPSK_CLAMP7": "--c4", "PROD1_DPSK_CLAMP20": "--dd-br", "PROD1_DPSK_E2X": "--v6", "PROD1_DPSK_GAUX": "--v7",
            "PROD1_DPSK_GAUX_NOBIAS": "--c5", "PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K": "--c3", "PROD1_DPSK_RS1": "--dd-cy",
            "PROD1_DPSK_LATENT2_E2X_QBX_KD_A0P8_S64Y_MIMO25_PPFULL": "--c2",   # data derisk round 2, arm B (11-10)
            "PROD1_DPSK_LATENT2_E2X_QBK3": "--s4",                               # violet, next to LATENT2_E2X's orange
            "ULTRA59B_MUONH_KD_QBK3": "--c1"}                                    # a tracked run: this is the two ends' hue (see _track_css)
DOWN_PAL = ["--c6", "--s4", "--s5", "--v8"]           # fallback for new Change IDs
# --db2 / --db3: the 30B Instruct / Thinking baselines (user 11-16 "三个baseline（蓝色系）"): steps of Base's blue (--c1) at the same hue, receding
# toward the surface (light: L 0.575 / 0.66 / 0.745; dark: 0.622 / 0.53 / 0.44), so the default comparison (Base) keeps its colour and stays the
# strongest. Validated --ordinal in both modes; >= 13 dE from every grey of the checkpoints (the light end is gamut-limited at 0.131 chroma).
# --dq1..3: the Qwen3.8 baselines (user 11-16 "我们有Qwen3.5和3.8的分数吗，也最好再prod里面变成指标"; "两边都是，可选的，不是default的指标"):
# orange steps by model size, strongest = largest (violet next to these blues is ~1 dE under protan / deutan). Since 16:5x a 4-step ramp
# (Qwen3.5-397B between Flash-Next and Max, used by prod iso-doc) defined with the blues in cosmos_eval_section.PROD_FAM_CSS.
# 16:5x (user "色系要能统一"): the baseline colours (--db2 / --db3 / --dq1..4) moved to cosmos_eval_section.PROD_FAM(_CSS), shared with prod iso-doc
DOWN_CSS = ('<style>.viz-root{--dd-br:#8c5a2b;--dd-cy:#0093a8}@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) '
            '.viz-root{--dd-br:#c08a55;--dd-cy:#36bfd3}}</style>')
# baselines in the downstream tab (user 11-09 "把qwen 30b 和 235 B 的实验，当作 baseline 也拉进来"): (code, name, colour); blues, solid (grey until 11-13, when the production run's checkpoints became the grey scale)
# 11-10 (user "qwen3-235B咱们用thinking去做标杆在 downstream eval"): the 235B baseline = Thinking-2507 (job 57497), not the 2504 (still in the ref tab)
# 11-13: only the baselines that have a result on the new task list are shown (Thinking-2507 has none yet: it appears once collected)
# 11-16 (user "把qwen3-30b thinking 和 instruct 放在 prod downstream ... 我们可以有三个baseline（蓝色系）"; "可以不用default选base做比，然用户自己选，
# default可以是base"): + the 30B Instruct-2507 / Thinking-2507; every baseline has a tick box like the checkpoints, ticked by default: DOWN_BASE_ON
DOWN_BASE = [("QWEN3_30B_A3B_BASE", "Qwen3-30B-A3B-Base", "--c1"), ("QWEN3_30B_A3B_INSTRUCT_2507", "Qwen3-30B-A3B-Instruct-2507", "--db2"),
             ("QWEN3_30B_A3B_THINKING_2507", "Qwen3-30B-A3B-Thinking-2507", "--db3"), ("QWEN3_235B_A22B_THINKING_2507", "Qwen3-235B-A22B-Thinking-2507", "--k08"),
             ("QWEN38_27B", "Qwen3.8-27B", "--dq1"), ("QWEN38_FLASH_NEXT", "Qwen3.8-Flash-Next (FP8)", "--dq2"),   # 11-16: optional, unticked
             ("QWEN38_MAX", "Qwen3.8-Max (FP8)", "--dq4"),                                                          #   (Qwen3.5: no 66-metric score)
             ("MIMO_V25_PRO_BASE", "MiMo-V2.5-Pro-Base", "--dt")]   # 11-16 review: the KD teacher, optional (unticked); also a scoreboard column
# colours: cosmos_eval_section.PROD_FAM is the source (one colour per model in both prod tabs); downstream_tab applies it over these
DOWN_BASE_ON = {"QWEN3_30B_A3B_BASE"}
DOWN_FOLD = 80000                                        # "实验列表": checkpoints of the tracked run before this step are folded (user 11-16: < 50000; 11-17 "放到 < 70000", then "算了 fold < 80000")
# size of a checkpoint that is not a ladder run (no submission.json). ULTRA59B: total 58.81B = Megatron's "Total number of parameters in
# billions" in the run's log; activated = total - the routed experts not selected: 47 MoE layers x 512 experts x 3 x 1024 x 768 params,
# of which 16 / 512 are active -> 58.81B - 56.77B x 496 / 512 = 3.81B
DOWN_SIZE = {"ULTRA59B_MUONH_KD_QBK3": "58.8B-A3.8B"}


# ladder checkpoints that have scores but are not shown on the page. User 11-14: "PROD1_DPSK_LATENT2_E2X · q3 / PROD1_DPSK_LATENT2_E2X_QBK3 · q3
# 这个不用放了，不如放KD q3 和 PROD1_DPSK q5" (those two have no score of this protocol yet; they appear by themselves once collected as "ours")
DOWN_HIDE = {"PROD1_DPSK_LATENT2_E2X_q3", "PROD1_DPSK_LATENT2_E2X_QBK3_q3"}
# the 59B checkpoint averages (collect_cosmos_eval.py: MERGE59_<wave>_<recipe>_iter<newest step>, "ours"): a chapter of their own under the
# run's scoreboard, against the same baseline (cosmos_eval_section.ce_avg), and no rows in "实验列表" / the folded tables any more
# (user 11-17: "能否把 [59B avg ...] 在 prod downstream 里面，但开一个 section，baseline 还是 qwen3-30b-base")
DOWN_AVG_RE = re.compile(r"MERGE59_(M\d+)_(\w+?)_iter0*(\d+)$")


def _down_avg_groups(res, track):
    """[(wave, newest step, code of the tracked run's own checkpoint of that step or None, [(code, recipe, description)])], by step."""
    plain = {s: code for code, _, s in track}
    g = {}
    for code, d in res.items():
        m = DOWN_AVG_RE.match(code)
        if m and d.get("ours") and code not in DOWN_HIDE:
            desc = re.search(r"\((.*); newest iter", d.get("display_name") or "")
            g.setdefault((int(m.group(3)), m.group(1)), []).append((code, m.group(2), desc.group(1) if desc else ""))
    return [(w, s, plain.get(s), ms) for (s, w), ms in sorted(g.items())]


def _track_css(steps):
    """One grey per checkpoint of the tracked run (user 11-13: "这几个能搞个灰度变化吗"): the page's text colour mixed into its surface colour,
    30 % .. 100 % by training step relative to the latest evaluated checkpoint, so the checkpoints read as a grey scale and the latest one
    is the strongest line. Both colours are theme variables: light theme light grey -> black, dark theme dark grey -> white. Ticking boxes
    never repaints; a new latest checkpoint makes the earlier ones a little lighter. Returns ({step: css variable}, <style>)."""
    var = {s: f"--pu{s}" for s in steps}
    top = max(steps)
    css = "".join(f"{var[s]}:color-mix(in oklab,var(--text-primary) {round(30 + 70 * s / top)}%,var(--surface-1));" for s in steps)
    return var, "<style>.viz-root{" + css + "}</style>"
# "实验列表" (user 11-09: the legend as a table, own chapter above the key benchmarks): what each checkpoint is. The Change IDs read 11-09
# from the runs' argv (diff vs PROD1_DPSK q3) say what they change vs PROD1_DPSK; the ones shown on the new protocol are written out in full
# (PROD1_DPSK itself is not in the list yet); ULTRA59B read 11-13 from its argv (boxiangw/cosmos_ultra_pre/runs/eval59/view)
DOWN_DESC = {
    "zh": {"PROD1_DPSK": "PROD v1 底座：SWA 512（局部:全局 5:1）+ sink，DSV3 router（sigmoid + expert bias + seq aux 1e-4），第 0 层 dense，128 选 8 + 1 shared，routed scaling 2.5，32K 训练 + 文档间 mask，MuonH，纯 CE，650B 数据",
           "PROD1_DPSK_CLAMP7": "expert 的 SwiGLU 加截断（gpt-oss 的做法）：gate 支路上限 7，线性支路截断到 ±7",
           "PROD1_DPSK_CLAMP20": "expert 的 SwiGLU 加截断（gpt-oss 的做法）：gate 支路上限 20，线性支路截断到 ±20",
           "PROD1_DPSK_E2X": "routed expert 128 → 256，仍选 8 个（每个 token 计算量不变，routed 参数 ×2）",
           "PROD1_DPSK_GAUX": "seq aux 1e-4 → global aux 1e-4（负载统计从每条序列改为整个 global batch），expert bias 保留",
           "PROD1_DPSK_GAUX_NOBIAS": "global aux 1e-3，去掉 expert bias（只靠 aux loss 均衡）",
           "PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K": "架构同 PROD1_DPSK，训练改为 KD：teacher MiMo-2.5，α 0.8，S64Y；新数据（pack-plan 32K，q3 训练 148B token）",
           "PROD1_DPSK_LATENT2": "latent MoE：d = h/2，256 个 routed expert 选 16 个；总参数、routed FLOP、all-to-all 与 PROD1_DPSK 相同",
           "PROD1_DPSK_LATENT2_E2X": "PROD v1 底座（SWA 512，局部:全局 5:1 + sink；DSV3 router：sigmoid + expert bias + seq aux 1e-4；第 0 层 dense；32K 训练 + 文档间 mask；MuonH，纯 CE，650B 数据）上的 latent MoE：d = h/2，512 个 routed expert 选 16 + 1 shared",
           "PROD1_DPSK_LATENT2_E2X_QBK3": "同 PROD1_DPSK_LATENT2_E2X，router 换成 QBK3：按 Kimi K3 原文的 quantile balancing（expert bias 每个 step 由整个 global batch 的分位数直接算出，没有 seq aux）",
           "ULTRA59B_MUONH_KD_QBK3": "生产规模的 run（Boxiang 的 cosmos_ultra_pre；训练共 97,330 步，还在训练）：PROD1_DPSK_LATENT2_E2X_QBK3 的结构放大到 48 层、h 2048（latent d = h/2，512 个 routed expert 选 16 + 1 shared，第 0 层 dense，SWA 512 局部:全局 5:1，QBK3 router），MuonH，KD（teacher MiMo-V2.5-Pro-Base，α 0.75，top-64 + bucket 尾部）。大小：总参数取自 Megatron 日志，激活参数按结构算出",
           "PROD1_DPSK_RS1": "routed scaling 2.5 → 1",
           "PROD1_DPSK_LATENT2_E2X_QBX_KD_A0P8_S64Y_MIMO25_PPFULL": "数据 derisk round 2 的 B 臂：LATENT2_E2X（512 选 16）+ QBX router，KD（teacher MiMo-2.5，α 0.8，S64Y），prepack 数据（tc_full）",
           "QWEN3_30B_A3B_BASE": "外部 baseline：Qwen 官方 base 模型",
           "QWEN3_235B_A22B": "外部 baseline：Qwen 官方模型（2504 版，后训练）",
           "QWEN3_30B_A3B_INSTRUCT_2507": "外部 baseline：Qwen 官方 Instruct-2507（后训练，非思考模式）",
           "QWEN3_30B_A3B_THINKING_2507": "外部 baseline：Qwen 官方 Thinking-2507（后训练的推理模型）",
           "QWEN3_235B_A22B_THINKING_2507": "外部 baseline：Qwen 官方 Thinking-2507（后训练的推理模型，和其他外部模型一样用 base 式 few-shot 提示测）",
           "QWEN38_27B": "外部 baseline：Qwen 官方 Qwen3.8-27B（dense，后训练，没有发布 Base）",
           "QWEN38_FLASH_NEXT": "外部 baseline：Qwen 官方 Qwen3.8-Flash-Next（官方 FP8，后训练；Qwen3.8-Flash 没有开源权重，用它代替）",
           "QWEN38_MAX": "外部 baseline：Qwen 官方 Qwen3.8-Max（官方 FP8，后训练）",
           "MIMO_V25_PRO_BASE": "外部参考：59B 的 KD teacher MiMo-V2.5-Pro-Base（Xiaomi 的 base 模型）",
           "h": "实验列表", "all": "全选", "none": "全不选", "fold_open": "▸ 展开第 {s} 步之前的 {n} 个 checkpoint", "fold_close": "▾ 收起", "head": ["显示", "颜色", "模型", "说明", "大小", "已出分"], "pend": "还没出分：{tasks}", "track_row": "生产 run 的 checkpoint：iter {step} / {total}（{pct}）。评测：{src}{run}", "track_run": "；评测还在跑", "track_stop": "；评测中途停了（作业被取消或失败）"},
    "en": {"PROD1_DPSK": "PROD v1 base: SWA 512 (local:global 5:1) + sinks, DSV3 router (sigmoid + expert bias + seq aux 1e-4), dense layer 0, 128 pick 8 + 1 shared, routed scaling 2.5, 32K training + inter-document mask, MuonH, pure CE, 650B data",
           "PROD1_DPSK_CLAMP7": "SwiGLU clamp in the experts (the gpt-oss form): gate branch capped at 7, linear branch clipped to ±7",
           "PROD1_DPSK_CLAMP20": "SwiGLU clamp in the experts (the gpt-oss form): gate branch capped at 20, linear branch clipped to ±20",
           "PROD1_DPSK_E2X": "routed experts 128 → 256, still top-8 (same compute per token, routed parameters ×2)",
           "PROD1_DPSK_GAUX": "seq aux 1e-4 → global aux 1e-4 (load statistic over the whole global batch instead of each sequence), expert bias kept",
           "PROD1_DPSK_GAUX_NOBIAS": "global aux 1e-3, expert bias removed (balanced by the aux loss only)",
           "PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K": "PROD1_DPSK architecture trained with KD: teacher MiMo-2.5, α 0.8, S64Y; new data (pack-plan 32K, 148B training tokens at q3)",
           "PROD1_DPSK_LATENT2": "latent MoE: d = h/2, 256 routed experts top-16; total parameters, routed FLOPs and all-to-all as PROD1_DPSK",
           "PROD1_DPSK_LATENT2_E2X": "latent MoE on the PROD v1 base (SWA 512, local:global 5:1 + sinks; DSV3 router: sigmoid + expert bias + seq aux 1e-4; dense layer 0; 32K training + inter-document mask; MuonH, pure CE, 650B data): d = h/2, 512 routed experts top-16 + 1 shared",
           "PROD1_DPSK_LATENT2_E2X_QBK3": "PROD1_DPSK_LATENT2_E2X with the QBK3 router: quantile balancing exactly as the Kimi K3 report (the expert bias is computed every step from the quantiles of the whole global batch, no seq aux)",
           "ULTRA59B_MUONH_KD_QBK3": "production-scale run (Boxiang's cosmos_ultra_pre; 97,330 training steps, still training): the PROD1_DPSK_LATENT2_E2X_QBK3 architecture scaled to 48 layers, h 2048 (latent d = h/2, 512 routed experts top-16 + 1 shared, dense layer 0, SWA 512 local:global 5:1, QBK3 router), MuonH, KD (teacher MiMo-V2.5-Pro-Base, α 0.75, top-64 + bucket tail). Size: total parameters from the Megatron log, activated parameters computed from the architecture",
           "PROD1_DPSK_RS1": "routed scaling 2.5 → 1",
           "PROD1_DPSK_LATENT2_E2X_QBX_KD_A0P8_S64Y_MIMO25_PPFULL": "data derisk round 2, arm B: LATENT2_E2X (512 pick 16) + QBX router, KD (teacher MiMo-2.5, α 0.8, S64Y), prepack data (tc_full)",
           "QWEN3_30B_A3B_BASE": "external baseline: the official Qwen base model",
           "QWEN3_235B_A22B": "external baseline: the official Qwen model (2504, post-trained)",
           "QWEN3_30B_A3B_INSTRUCT_2507": "external baseline: the official Qwen Instruct-2507 (post-trained, non-thinking)",
           "QWEN3_30B_A3B_THINKING_2507": "external baseline: the official Qwen Thinking-2507 (post-trained reasoning model)",
           "QWEN3_235B_A22B_THINKING_2507": "external baseline: the official Qwen Thinking-2507 (post-trained reasoning model, tested with the same base-style few-shot prompts as the other external models)",
           "QWEN38_27B": "external baseline: the official Qwen3.8-27B (dense, post-trained, no Base release)",
           "QWEN38_FLASH_NEXT": "external baseline: the official Qwen3.8-Flash-Next (official FP8, post-trained; Qwen3.8-Flash has no open weights, this stands in for it)",
           "QWEN38_MAX": "external baseline: the official Qwen3.8-Max (official FP8, post-trained)",
           "MIMO_V25_PRO_BASE": "external reference: the 59B's KD teacher MiMo-V2.5-Pro-Base (Xiaomi's base model)",
           "h": "Experiment list", "all": "all", "none": "none", "fold_open": "▸ show the {n} checkpoints before step {s}", "fold_close": "▾ hide", "head": ["show", "colour", "model", "description", "size", "scored"], "pend": "no score yet: {tasks}", "track_row": "checkpoint of the production run: iter {step} of {total} ({pct}). Evaluated by {src}{run}", "track_run": "; the evaluation is still running", "track_stop": "; the evaluation stopped before the end (job cancelled or failed)"}}


def downstream_tab(R, lang):
    try:
        from cosmos_eval_section import ce_scores, ce_results, ce_counts, ce_steps, ce_board, ce_avg, PROD_FAM, PROD_FAM_CSS, PROD_TABS, PROD_SECTIONS
        res = ce_results()
        nt, nm, nk = ce_counts()
    except Exception as e:
        print("downstream tab skipped:", type(e).__name__, e)
        return ""
    ours = []                                            # (sort key, code, Change ID or run name, rung or None, step or None)
    for code, d in res.items():
        if not d.get("ours") or code in DOWN_HIDE or DOWN_AVG_RE.match(code):   # the averages: their own chapter (ce_avg below)
            continue
        m = re.match(r"(.+)_(q\d)$", code)
        if d.get("track"):                               # a checkpoint of the tracked production run
            ours.append(((0, False, d["track"]["run"], f'{d["track"]["step"]:09d}'), code, d["track"]["run"], None, d["track"]["step"]))
        elif m:
            ours.append(((1, m.group(1) != "PROD1_DPSK", m.group(1), m.group(2)), code, m.group(1), m.group(2), None))
        else:
            ours.append(((2, False, code, ""), code, re.sub(r"_iter\d+$", "", code), None, None))
    if not ours:
        return ""
    ours.sort()
    track = [(code, chg, step) for _, code, chg, _, step in ours if step is not None]
    colors, k, tcss = {}, 0, ""
    if track:
        tvar, tcss = _track_css([s for _, _, s in track])
        colors.update({code: tvar[s] for code, _, s in track})
    for chg in dict.fromkeys(c for _, _, c, _, step in ours if step is None):
        codes = [code for _, code, c, _, step in ours if c == chg and step is None]
        ramp = DOWN_RAMP.get(chg)
        for i, code in enumerate(codes):
            if ramp and 1 < len(codes) <= len(ramp):
                colors[code] = ramp[i if len(codes) == len(ramp) else (0 if i == 0 else len(ramp) - 1)]
            elif chg in DOWN_ONE and len(codes) == 1:
                colors[code] = DOWN_ONE[chg]
            elif ramp and len(codes) == 1:
                colors[code] = ramp[-1]                  # the only rung of a ramp Change ID: its dark end
            else:
                colors[code] = DOWN_PAL[k % len(DOWN_PAL)]; k += 1
    sizes = {code: ((ladder_size(chg, q) or ("",))[0] if q else "") or DOWN_SIZE.get(chg, "") for _, code, chg, q, _ in ours}
    base = [(c, n, PROD_FAM.get(c, col)) for c, n, col in DOWN_BASE if c in res]   # baselines first (user 11-09)
    colors.update({c: col for c, _, col in base}); sizes.update({c: REF_SIZE.get(c, "") for c, _, _ in base})
    models = [(c, n, False) for c, n, _ in base] + [(code, res[code].get("display_name") or code, False) for _, code, _, _, _ in ours]
    # chapter "实验列表" = the legend as a table (user 11-09): colour, model, what it is, size, metrics scored so far
    D = DOWN_DESC[lang]
    def track_row(code, step):                           # a checkpoint of the tracked run: progress, who evaluated it on which harness commit
        d = res[code]
        by = {}
        for lab, ident in (d.get("meta", {}).get("identity") or {}).items():
            by.setdefault(lab.split(":")[0] if ":" in lab else "xingqianx", []).append((ident.get("harness_commit") or "")[:8])
        src = "; ".join(f'{o} (harness {" + ".join(dict.fromkeys(h))})' for o, h in by.items())
        tot = d["track"]["total_steps"]
        return D["track_row"].format(step=f"{step:,}", total=f"{tot:,}", pct=f"{100 * step / tot:.1f} %", src=src, run=D["track_run"] if d.get("running") else D["track_stop"] if d.get("stopped") else "")
    desc = {**{c: D.get(c, "") for c, _, _ in base},
            **{code: (track_row(code, step) if step is not None else D.get(chg, "")) for _, code, chg, _, step in ours}}
    def scored(code):
        d = res[code]
        n = sum(isinstance(m.get("value"), (int, float)) for ts in d["groups"].values() for t in ts for m in t["metrics"])
        pend = [t["task"] for ts in d["groups"].values() for t in ts if t.get("pending")]
        tip = f' title="{esc(D["pend"].format(tasks=", ".join(pend)))}" style="text-decoration:underline dotted;text-underline-offset:3px"' if pend else ""
        return f"<td{tip}>{n} / {d.get('n_metrics') or nm}</td>"
    def sample(code):                                    # colour square (user 11-09)
        return f'<i style="display:inline-block;width:12px;height:12px;border-radius:2px;background:var({colors.get(code, "--text-secondary")})"></i>'
    bidx = {c: i for i, (c, _, _) in enumerate(base)}     # baselines: own tick boxes since 11-16 (before: always shown, user 11-09); data-b = position
    opt = {c for c, _, _ in base if c not in DOWN_BASE_ON}  # optional baselines: their box has data-opt; out of the curve's default scale (ce_scores)
    # ticked by default: only the LATEST checkpoint of the tracked run (user 11-13: "这个默认只是放最后的checkpoint成绩"; the chart block above
    # shows every checkpoint); without a tracked run: all
    with_score = [x for x in track if len(res[x[0]].get("missing") or []) < res[x[0]].get("n_metrics", 1)] or track   # a checkpoint whose evaluation has only started has no score yet
    heads = {f"{a} {m}" for _, _, _, ks in PROD_TABS for a, m in ks}
    done = [x for x in track if not heads & set(res[x[0]].get("missing") or [])] or with_score   # 11-16 review: the latest COMPLETE checkpoint (no headline metric pending)
    tset = {max(done, key=lambda x: x[2])[0]} if track else {code for _, code, _, _, _ in ours}   # ticked by default: only that one (user 11-13: the latest)
    # checkpoints of the tracked run before DOWN_FOLD (user 11-16 "可否folder < 50000的checkpoint"): folded rows under a toggle row
    fold = {code for code, _, s in track if s < DOWN_FOLD}
    tot0 = (res[track[0][0]].get("track") or {}).get("total_steps") if track else None
    fold_s = f"{DOWN_FOLD:,}" + ((f"（{100 * DOWN_FOLD / tot0:.1f} %）" if lang == "zh" else f" ({100 * DOWN_FOLD / tot0:.1f} %)") if tot0 else "")
    rows = "".join((f'<tr><td colspan="{len(D["head"])}" class="l"><button class="dm-fold-btn" data-open="{esc(D["fold_open"].format(n=len(fold), s=fold_s))}" '
                    f'data-close="{esc(D["fold_close"])}" style="font:inherit;font-size:12px;cursor:pointer">{esc(D["fold_open"].format(n=len(fold), s=fold_s))}</button></td></tr>'
                    if fold and code == min(fold, key=lambda c: res[c]["track"]["step"]) else "")
                   + (f'<tr class="dm-fold" hidden>' if code in fold else '<tr>') + '<td style="text-align:center">' + f'<input type="checkbox" class="dm-cb" value="{esc(code)}"' + (f' data-b="{bidx[code]}"' if code in bidx else "") + (" data-opt" if code in opt else "") + (" checked" if code in tset or code in DOWN_BASE_ON else "") + ">" + '</td>'   # show / hide (user 11-09)
                   f'<td>{sample(code)}</td><td class="l" style="white-space:nowrap">{esc(name)}</td>'
                   f'<td class="l" style="white-space:normal;max-width:620px">{esc(desc.get(code, ""))}</td>'
                   f'<td class="l nt-d" style="white-space:nowrap">{esc(sizes.get(code, ""))}</td>{scored(code)}</tr>' for code, name, _ in models)
    head = "".join(f'<th class="l">{esc(x)}</th>' for x in D["head"])
    sel = (f'<span style="white-space:nowrap"><button class="dm-all" data-v="1" style="font:inherit;font-size:12px;cursor:pointer">{esc(D["all"])}</button> '
           f'<button class="dm-all" data-v="0" style="font:inherit;font-size:12px;cursor:pointer">{esc(D["none"])}</button></span>')
    run = track[0][1] if track else ""
    out = DOWN_CSS + PROD_FAM_CSS + tcss + ('<style>.cv-v[data-cv="1"]{display:none}section[data-pane="down"]:has(.dm-cb[data-opt]:checked) .cv-v[data-cv="0"]{display:none}'
                             'section[data-pane="down"]:has(.dm-cb[data-opt]:checked) .cv-v[data-cv="1"]{display:block}</style>')
    # order (user 11-16 18:0x "图永远放在最前面", 18:1x "overview 大字总结放在图的前面 表格还是放后面"): the overview tiles (11-16 review),
    # the step charts (every task against the training step), "实验列表" (drives them), then the scoreboard table (latest score per task vs
    # Qwen3-30B-A3B-Base, noise call, best so far), then the folded table chapters
    board = ""
    if track:
        try:
            tiles, board = ce_board(lang, esc, [(code, s) for code, _, s in track], "QWEN3_30B_A3B_BASE", "Qwen3-30B-A3B-Base",
                                    extra=["MIMO_V25_PRO_BASE"], names={c: n for c, n, _ in base}, split=True) or ("", "")
            out += tiles
        except Exception as e:
            print("downstream scoreboard skipped:", type(e).__name__, e)
        try:
            out += ce_steps(lang, esc, run, [(code, s) for code, _, s in track], base, colors, on=DOWN_BASE_ON, tabs=PROD_TABS)   # 11-16: all 36 tasks, by category / importance
        except Exception as e:
            print("downstream step charts skipped:", type(e).__name__, e)
    out += (f'<h2>{esc(D["h"])}</h2><div class="card"><p class="note" style="margin:0 0 6px">{sel}</p><div style="overflow-x:auto">'
            f'<table style="display:table;width:100%;font-size:12.5px"><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')
    # (the folded line under the list that named the checkpoints with scores of the earlier protocol only, results_xx/nemo6, was removed
    # 11-17: user "只有旧协议（nemo6）分数的 24 个 checkpoint ... 这个不要了吧"; text key down_old is unused since)
    out += "</div>" + board
    try:                                                 # 11-17: the 59B checkpoint averages, a chapter of their own against the same baseline
        out += ce_avg(lang, esc, _down_avg_groups(res, track), "QWEN3_30B_A3B_BASE", "Qwen3-30B-A3B-Base")
    except Exception as e:
        print("downstream averages chapter skipped:", type(e).__name__, e)
    for focus in (True, False):                         # 11-16 review: both chapters are tables only and folded (the scoreboard and the step charts
        try:                                             # above carry the picture; the md mirror still has every table)
            out += ce_scores(lang, esc, models, sizes, focus=focus, colors=colors, legend=False, opt=opt,
                             tabs=PROD_TABS, sections=PROD_SECTIONS, curve=False, fold=True)   # Runlong 11-16: "把所有 bench 都放完，分类，按照你觉得的重要性排序"
        except Exception as e:
            print("downstream chapter skipped:", type(e).__name__, e)
    return out


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
    pcols = [(key, n) for k, key in PSETS for kk, n in COLS if kk == key] + [("mean6", t["mean6"])]
    _, p2 = tables(RP, t, pcols)                          # the all-results table is dropped (user); KD - CE stays
    PMETRICS = [("01_val_v2_overall_tw", "val_v2"), ("02_uncheatable_tw", "uncheatable"), ("03_codefresh_tw", "codefresh"), ("mean6", t["mean6"])]
    doc = f"""
{token_pie(t)}
<h2>{esc(t['ba_h'])}</h2>
{stack_section(R, t)}
{rtq_section(R, t)}
{qbxv_section(R, t)}
{ppf_section(R, t)}"""   # + router / load balancing on E2X (user 11-06 "runlong有了moe routing QB的实验") + KD on the prepack, E2X vs E2X + QBX (11-06)   # "说明" notes section removed (user 11-05)
    # older experiments in their own tab, right of doc-isolated (user 11-05): optimizer comparison, KD vs CE, KD loss variants, KD teacher 2.6 / 2.5
    docold = f"""{chart_sections(R, t, METRICS)}
{variant_section(R, t, METRICS, COLS)}
{kd25_section(R, t, METRICS, "00_overall_tsqw", "overall_tsqw")}
{docmask_section(R, t, METRICS)}
{arch_section(R, t, METRICS)}
{rt0_section(R, t)}
{router_section(R, t, COLS + [("00_overall_tsqw", "overall_tsqw")])}
{ne_chapter(R, t, COLS + [("00_overall_tsqw", "overall_tsqw")])}
{dense0_section(R, t)}
{sws_section(R, t)}
{x_section(R, t, "lat")}
{x_section(R, t, "rope")}
{packplan_section(R, t)}"""   # + data derisk / 2.5 vs 2.6 (user 11-10 "也放到老实验里去吧"); + doc mask (user 11-05 "文档 mask -> 老实验"); + SWA / router / phi / dense0 / SWS / latent / RoPE sections (user 11-05 "搬家 -> 老实验")
    packed = f"""<div class="card"><ul class="f">{''.join(f'<li>{n}</li>' for n in t['p_notes'])}</ul></div>
{chart_sections(RP, t, PMETRICS)}
{variant_section(RP, t, PMETRICS, pcols)}
{kd25_section(RP, t, [("mean6", t["mean6"])] + PMETRICS[:3], "mean6", t["mean6"], packed=True)}
{arch_section(RP, t, pcols)}
<h2>{esc(t['t2_h'])}</h2><div class="card">{p2}</div>"""
    fit = f"""<p class="note">{t['fit_tab_note']}</p>
{grid_section(R, t, ("00_overall_tsqw", "overall_tsqw"))}
{grid_section(R, t, ("00_overall_tsqw", "overall_tsqw"), GM)}
{assume_section(R, t)}"""                                  # LR-fitting experiments in their own tab (user 11-03)
    ref = ref_section(R, t, REF_COLS)                    # external reference models in their own tab (user 11-05), was a doc-tab section
    try:                                                 # full evaluation on the unified harness's task list (user 11-13: 66 metrics, "不再是
        from cosmos_eval_section import ce_chapter, ce_scores, ce_results   # nemov6的75个"): scope, key tier, results; runs/cosmos_eval_downstream via results_xx/cosmos_eval
        have = ce_results()
        lack = [n for c, n, _ in REF_MODELS if c not in have and os.path.exists(os.path.join(HERE, "..", "results_xx", "nemo6", c + ".json"))]
        more = (f'<p class="note">{esc(t["ce_lack"].format(names=("、" if lang == "zh" else ", ").join(lack)))}</p>' if lack else "") + f'<p class="note">{esc(t["ce_ours"])}</p>'
        ref += ce_chapter(lang, esc, more) + ce_scores(lang, esc, REF_MODELS, REF_SIZE, focus=True) + ce_scores(lang, esc, REF_MODELS, REF_SIZE)
    except Exception as e:
        print("cosmos_eval chapters skipped:", type(e).__name__, e)
    down = downstream_tab(R, lang)                       # tab "prod downstream": our checkpoints on the same task list (rewritten 11-13)
    try:                                                 # tab "prod iso-doc" (user 11-13 "请+一个iso-doc测试tab", "就是prod iso-doc", "分为8k vs 32k"):
        from prod_isodoc_section import prod_tab         # doc-isolated val loss of the production run's checkpoints (results_xx/prod_isodoc)
        prod = prod_tab(lang, esc, R, ladder_size, {**REF_SIZE, **DOWN_SIZE},   # (tab label, pane) or None while there is no result
                        vocab=lambda ch, q: REF_VOCAB.get(ch) if q == "ref" else ladder_vocab(ch, q))
    except Exception as e:
        prod = None
        print("prod iso-doc tab skipped:", type(e).__name__, e)
    # tabs (11-16 review of Runlong's "主要两个可视化就是 val loss 和 downstream 放在最前面"): one ordered table drives the nav, the DOM order and
    # the default tab = the first one with content (59B val loss). Earlier order (user 11-14): doc, prod iso-doc, prod downstream, 外部参考,
    # 老实验, ...; the ladder tabs now sit behind an "早期研究" separator, each still its own pane (pane ids unchanged: --only and the md
    # mirror key on them). A pane that failed to build carries EMPTY: keep_other_panes keeps the page's previous content for it.
    panes = {"prod": prod[1] if prod else EMPTY, "down": down or EMPTY, "ref": ref, "doc": doc, "docold": docold, "fit": fit, "packed": packed}
    labels = {"prod": prod[0] if prod else "59B val loss", "down": t["tab_down"], "ref": t["tab_ref"], "doc": t["tab_doc"], "docold": t["tab_docold"],
              "fit": t["tab_fit"], "packed": t["tab_packed"]}
    # 11-16 18:0x (user "网页太大了 很卡"): the archive tabs (10.5 MB of 14) go to their own file; each page shows its own tabs as buttons and
    # the other page's tabs as links (file#tab; the page script opens the tab named in the hash)
    out = {}
    for part in PARTS:
        own = [(k, g) for k, g in TAB_ORDER if PART_OF[g] == part]
        first = next((k for k, _ in own if EMPTY not in panes[k]), own[0][0])
        nav, prev = "", None
        for k, g in TAB_ORDER:
            if g == "arch" and prev != "arch":
                nav += f'<span class="nav-sep">{esc(t["nav_arch_here" if part == "arch" else "nav_arch"])}</span>'
            if PART_OF[g] == part:
                nav += f'<button class="tab{" arch" if g == "arch" else ""}{" on" if k == first else ""}" data-tab="{k}">{esc(labels[k])}</button>'
            else:
                nav += f'<a class="tabl{" arch" if g == "arch" else ""}" data-tabl="{k}" href="{FILES[PART_OF[g]][lang]}#{k}">{esc(labels[k])}</a>'
            prev = g
        sections = "\n".join(f'<section class="pane" data-pane="{k}"{"" if k == first else " hidden"}>{panes[k]}</section>' for k, _ in own)
        title = t["title"] if part == "main" else t["title_arch"]
        sub = t["sub"] if part == "main" else t["sub_arch"]
        body = f"""<div id="tip"></div><nav>{nav}</nav><main>
<h1>{esc(title)}</h1><p class="sub">{esc(sub.format(ts=ts))}</p>
{sections}
</main><script>{JS}</script>"""
        out[part] = (f'<!doctype html><html lang="{t["lang"]}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
                     f'<title>{esc(title)}</title><style>{CSS}</style></head><body class="viz-root">{body}</body></html>\n')
    return out


PANE_RE = re.compile(r'(<section class="pane" data-pane="(\w+)"[^>]*>)(.*?)(</section>)', re.S)
EMPTY = '<!--pane-empty--><p class="note">-</p>'          # the content of a tab whose builder failed (keep_other_panes keeps the old one)
TAB_ORDER = [("prod", "main"), ("down", "main"), ("ref", "ref"), ("doc", "arch"), ("docold", "arch"), ("fit", "arch"), ("packed", "arch")]
# two files (11-16 18:0x, user "网页太大了 很卡"): the main page (59B val loss, 59B downstream, 外部参考) and the archive of the ladder research
PARTS = ["arch", "main"]                                 # write order: the archive first (on its first build it takes its tabs from the one-file page)
PART_OF = {"main": "main", "ref": "main", "arch": "arch"}
FILES = {"main": {"zh": "qwen3_scaling_ladder_eval.html", "en": "qwen3_scaling_ladder_eval_en.html"},
         "arch": {"zh": "qwen3_scaling_ladder_eval_archive.html", "en": "qwen3_scaling_ladder_eval_archive_en.html"}}


def set_default_tab(page):
    """the first pane with content is open and its nav button is on (11-16 verify review: decided after the merge, so a tab whose builder
    failed but whose old content was kept stays the default)."""
    ks = [(m.group(2), "<!--pane-empty-->" in m.group(3)) for m in PANE_RE.finditer(page)]
    first = next((k for k, e in ks if not e), ks[0][0] if ks else None)
    if first is None:
        return page
    page = PANE_RE.sub(lambda m: f'<section class="pane" data-pane="{m.group(2)}"{"" if m.group(2) == first else " hidden"}>' + m.group(3) + m.group(4), page)
    return re.sub(r'<button class="(tab[^"]*?)( on)?" data-tab="(\w+)">',
                  lambda m: f'<button class="{m.group(1)}{" on" if m.group(3) == first else ""}" data-tab="{m.group(3)}">', page)


def keep_other_panes(new, old, only):
    """the page `new` with every tab that is not in `only` taken from the page `old` (the file on disk), byte for byte. Several sessions
    regenerate this report (11-13): with --only a session publishes its own tabs and nobody else's unfinished work. A tab that the old
    page does not have stays as built. A tab in `only` that built empty (EMPTY marker, or no content) keeps the old page's content too
    (11-16: a failing builder must not blank a tab). The opening <section> tag (its hidden flag = the default tab) always comes from new."""
    kept = {m.group(2): m.group(3) for m in PANE_RE.finditer(old)}

    def sub(m):
        k, body = m.group(2), m.group(3)
        if k not in kept:
            return m.group(0)
        if k in only and "<!--pane-empty-->" not in body and body.strip():
            return m.group(0)
        if k in only:
            print(f"tab {k} built empty: the page keeps its previous content")
        return m.group(1) + kept[k] + m.group(4)
    return PANE_RE.sub(sub, new)


def main():
    import sys
    only = None
    if "--only" in sys.argv[1:]:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    R, RP = load(), load_packed()
    import fcntl
    lock = os.path.expanduser("~/tmp") if os.path.isdir(os.path.expanduser("~/tmp")) else HERE
    if only is not None:
        unknown = only - {k for k, _ in TAB_ORDER}
        if unknown:
            sys.exit(f"--only: no such tab {sorted(unknown)}")
    for lang in ("zh", "en"):
        pages = build(lang, R, RP)
        for part in PARTS:
            p = os.path.join(HERE, FILES[part][lang])
            mine = {k for k, g in TAB_ORDER if PART_OF[g] == part}
            use = mine if only is None else only & mine
            # a file none of whose tabs is in --only is still rewritten with every tab kept byte for byte: its nav / script / style follow the code
            # 11-16 review: sessions that run at the same time merge one after the other (lock), and the file is replaced atomically, so no
            # reader or merge ever sees a half-written page. The archive's first build takes its tabs from the one-file page (main file).
            with open(os.path.join(lock, ".gen_eval_html.lock"), "w") as lk:
                fcntl.flock(lk, fcntl.LOCK_EX)
                src = p if os.path.exists(p) else os.path.join(HERE, FILES["main"][lang])
                page = pages[part]
                if os.path.exists(src):
                    page = keep_other_panes(page, open(src, encoding="utf-8").read(), use)
                page = set_default_tab(page)
                tmp = f"{p}.tmp{os.getpid()}"
                open(tmp, "w").write(page)
                os.replace(tmp, p)
            print("wrote", p, f"(doc {len(R)}, packed {len(RP)} checkpoints)" + (f"; only the tabs {sorted(use)} replaced" if only is not None else ""))
    if not os.environ.get("EVAL_MD_SKIP"):   # agent-readable Markdown mirror of the zh HTML (user 11-04); never fails the HTML build
        import subprocess, sys
        with open(os.path.join(lock, ".gen_eval_html.lock"), "w") as lk:   # under the lock: two sessions never write the md at once
            fcntl.flock(lk, fcntl.LOCK_EX)
            subprocess.run([sys.executable, os.path.join(HERE, "gen_eval_md.py"), "--md-only"], check=False)


if __name__ == "__main__":
    main()
