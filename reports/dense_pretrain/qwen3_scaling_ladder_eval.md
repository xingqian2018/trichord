# Qwen3-MoE scaling ladder 评测报告 —— Markdown 镜像（给 agent 读）

_由 `reports_xx/gen_eval_md.py` 生成于 2026-11-05 21:40：正文 = `qwen3_scaling_ladder_eval.html`（中文版，HTML 生成于 2026-11-05 21:40）经 `reports_xx/html_to_md.py` 逐段转换，
附录 = `trichord/dense_pretrain_result/*.json` 的原始数。不要手改：改 `gen_eval_html.py` 后重跑它（它会自动重写本文件）。_

读法：
- 内容和 HTML 一一对应、顺序相同。HTML 的每个页签 = 下面一个 `## 页签：…` 大节（doc-isolated = 主评测（逐项叠加的架构变化 + 数据 derisk）；doc-isolated老实验 = 其余单项实验（优化器、KD 对 CE、尾部估算方式、KD teacher、文档 mask、SWA、router、负熵均衡、第 0 层 dense、Latent MoE、位置编码），同一套评测；外部参考；LR 拟合实验；packed = Boxiang 训练时自带的验证，绝对值比 doc-isolated 高约 0.013–0.016，两者的绝对值不能互比）。
- 每张图都换成了数据表，数取自图上的悬停提示：表头是 rung（或横轴上的点），每行是图里的一条线；横轴点多时（LR 扫描、训练步数）行列互换，router 健康曲线均匀抽 12 个点（含首尾）。`·` = 该点没有数据。
- HTML 里靠按钮切换的内容全部展开，每块前面标明是哪个按钮：〔指标：…〕（overall_tsqw / overall_tw / overall_macro / 单个验证集）、〔视图：…〕（窗口、统计量等）、〔窗口：…〕。
- 单位 nats/token（另注 bpb、ppl 的除外）；图里的数大多是“该线 − 0 线”的 Δloss，负数 = 比 0 线好；各节的 0 线写在该节说明里，不同节的 0 线常常不同。
- 默认主指标 overall_tsqw（7 个集合按 √token 加权）；每个 checkpoint 的完整结果（115 个 split、bpb、ppl）在 `trichord/dense_pretrain_result/<Change ID>_<rung>.json`，附录 B 是全部 checkpoint 的 overall_tsqw 索引。

# Qwen3-MoE scaling ladder

doc-isolated 评测（9 个 decon suite，含 private_internal），dpl 指标定义；数据生成于 2026-11-05 21:40

## 页签：doc-isolated

### 各验证集的 token 数

overall_tw 由这六类按 token 加权，所以 token 多的集合（uncheatable、pile_test）权重大；overall_macro 则六类等权。pile_10k（7.4M token）不进 overall，不画在图里。每个模型评测的 token 完全相同。

| 验证集 | split 数 | token | 占比 |
|---|---|---|---|
| val_v2 | 3 | 26.6M | 11.7% |
| uncheatable | 55 | 114.9M | 50.6% |
| codefresh | 30 | 19.0M | 8.4% |
| v3small | 8 | 2.8M | 1.2% |
| private_internal | 2 | 3.2M | 1.4% |
| pile_test | 16 | 60.4M | 26.6% |
| 合计 | 114 | 226.9M | 100% |

### 架构变化

#### 逐项叠加：每加一项，loss 降多少

全部是 CE、MuonH、650B 数据（corpus_v2）、seed 1，评测用 doc-isolated 8K 窗口。每一行比上一行多加一项：表里每一行列出全部设置，橙色底的格子 = 这一行相对上一行改了的地方；q0–q3 列 = 累计（这一行 − 第一行 OPT_MUONH，负数 = 更好），相邻两行之差就是这一项的效果。图的画法和优化器对比相同：横轴是激活参数（不含 embedding，对数坐标），纵轴是相对 MuonH 老 LR 的 Δloss，一条线一行，颜色和表里每行前的方块相同（每条线 = 叠到这一行为止的累计效果）。

**〔指标：overall_tsqw〕**

| Change ID | LR | Router | 注意力（SWA / 位置编码 / sink） | MoE 层 | Latent MoE | 第 0 层 | 训练长度 / 文档 mask | 训练量 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `OPT_MUONH` | ArchScale 原版：0.016 · (L/8)^−0.5 · (T/10.4B)^−0.32 | softmax 打分；负载均衡 aux loss 1e-3 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | 2.1829 | 1.9094 | 1.7453 | 1.6314 |
| `MUONH_FIT` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | softmax 打分；负载均衡 aux loss 1e-3 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0050 | -0.0110 | -0.0173 | -0.0163 |
| `MUONH_FIT_DSV3` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0027 | -0.0134 | -0.0155 | -0.0146 |
| `MUONH_FIT_DSV3_SWS` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 2048（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0171 | -0.0197 | -0.0237 | -0.0241 |
| `MUONH_FIT_DSV3_SWS_SHARED_RS2P5` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 2048（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8 + 1 shared（3h/8）；routed scaling 2.5 | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0356 | -0.0465 | -0.0496 | -0.0490 |
| `PROD1_DPSK` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8 + 1 shared（3h/8）；routed scaling 2.5 | 无 | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0375 | -0.0527 | -0.0567 | -0.0560 |
| `PROD1_DPSK_LATENT2` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 256 选 16（expert 在 d 维上，每个半大）+ 1 shared；routed scaling 2.5；总参数同上一行 | 有：d = h/2（h → d 投影，expert 在 d 上算，RMSNorm 后投回 h；router 和 shared expert 仍在 h 上） | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0466 | -0.0667 | -0.0759 | -0.0747 |
| `PROD1_DPSK_LATENT2_E2X` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 512 选 16 + 1 shared；routed scaling 2.5；routed 参数 ×2 | 有：d = h/2（h → d 投影，expert 在 d 上算，RMSNorm 后投回 h；router 和 shared expert 仍在 h 上） | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0832 | -0.1022 | -0.1024 | · |

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MUONH_FIT_DSV3 | -0.0027 | -0.0134 | -0.0155 | -0.0146 |
| MUONH_FIT | -0.0050 | -0.0110 | -0.0173 | -0.0163 |
| MUONH_FIT_DSV3_SWS | -0.0171 | -0.0197 | -0.0237 | -0.0241 |
| MUONH_FIT_DSV3_SWS_SHARED_RS2P5 | -0.0356 | -0.0465 | -0.0496 | -0.0490 |
| PROD1_DPSK | -0.0375 | -0.0527 | -0.0567 | -0.0560 |
| PROD1_DPSK_LATENT2 | -0.0466 | -0.0667 | -0.0759 | -0.0747 |
| PROD1_DPSK_LATENT2_E2X | -0.0832 | -0.1022 | -0.1024 | · |

**〔指标：overall_tw〕**

| Change ID | LR | Router | 注意力（SWA / 位置编码 / sink） | MoE 层 | Latent MoE | 第 0 层 | 训练长度 / 文档 mask | 训练量 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `OPT_MUONH` | ArchScale 原版：0.016 · (L/8)^−0.5 · (T/10.4B)^−0.32 | softmax 打分；负载均衡 aux loss 1e-3 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | 2.1728 | 1.8960 | 1.7285 | 1.6120 |
| `MUONH_FIT` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | softmax 打分；负载均衡 aux loss 1e-3 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0047 | -0.0109 | -0.0176 | -0.0171 |
| `MUONH_FIT_DSV3` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0029 | -0.0137 | -0.0157 | -0.0150 |
| `MUONH_FIT_DSV3_SWS` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 2048（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0167 | -0.0197 | -0.0242 | -0.0247 |
| `MUONH_FIT_DSV3_SWS_SHARED_RS2P5` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 2048（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8 + 1 shared（3h/8）；routed scaling 2.5 | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0349 | -0.0468 | -0.0506 | -0.0507 |
| `PROD1_DPSK` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8 + 1 shared（3h/8）；routed scaling 2.5 | 无 | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0367 | -0.0537 | -0.0580 | -0.0577 |
| `PROD1_DPSK_LATENT2` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 256 选 16（expert 在 d 维上，每个半大）+ 1 shared；routed scaling 2.5；总参数同上一行 | 有：d = h/2（h → d 投影，expert 在 d 上算，RMSNorm 后投回 h；router 和 shared expert 仍在 h 上） | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0463 | -0.0676 | -0.0779 | -0.0771 |
| `PROD1_DPSK_LATENT2_E2X` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 512 选 16 + 1 shared；routed scaling 2.5；routed 参数 ×2 | 有：d = h/2（h → d 投影，expert 在 d 上算，RMSNorm 后投回 h；router 和 shared expert 仍在 h 上） | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0829 | -0.1039 | -0.1044 | · |

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MUONH_FIT_DSV3 | -0.0029 | -0.0137 | -0.0157 | -0.0150 |
| MUONH_FIT | -0.0047 | -0.0109 | -0.0176 | -0.0171 |
| MUONH_FIT_DSV3_SWS | -0.0167 | -0.0197 | -0.0242 | -0.0247 |
| MUONH_FIT_DSV3_SWS_SHARED_RS2P5 | -0.0349 | -0.0468 | -0.0506 | -0.0507 |
| PROD1_DPSK | -0.0367 | -0.0537 | -0.0580 | -0.0577 |
| PROD1_DPSK_LATENT2 | -0.0463 | -0.0676 | -0.0779 | -0.0771 |
| PROD1_DPSK_LATENT2_E2X | -0.0829 | -0.1039 | -0.1044 | · |

**〔指标：overall_macro〕**

| Change ID | LR | Router | 注意力（SWA / 位置编码 / sink） | MoE 层 | Latent MoE | 第 0 层 | 训练长度 / 文档 mask | 训练量 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `OPT_MUONH` | ArchScale 原版：0.016 · (L/8)^−0.5 · (T/10.4B)^−0.32 | softmax 打分；负载均衡 aux loss 1e-3 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | 2.2383 | 1.9692 | 1.8100 | 1.7002 |
| `MUONH_FIT` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | softmax 打分；负载均衡 aux loss 1e-3 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0060 | -0.0109 | -0.0174 | -0.0163 |
| `MUONH_FIT_DSV3` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0029 | -0.0132 | -0.0155 | -0.0146 |
| `MUONH_FIT_DSV3_SWS` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 2048（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0174 | -0.0200 | -0.0235 | -0.0242 |
| `MUONH_FIT_DSV3_SWS_SHARED_RS2P5` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 2048（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8 + 1 shared（3h/8）；routed scaling 2.5 | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0367 | -0.0463 | -0.0490 | -0.0481 |
| `PROD1_DPSK` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8 + 1 shared（3h/8）；routed scaling 2.5 | 无 | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0381 | -0.0517 | -0.0558 | -0.0552 |
| `PROD1_DPSK_LATENT2` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 256 选 16（expert 在 d 维上，每个半大）+ 1 shared；routed scaling 2.5；总参数同上一行 | 有：d = h/2（h → d 投影，expert 在 d 上算，RMSNorm 后投回 h；router 和 shared expert 仍在 h 上） | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0475 | -0.0658 | -0.0742 | -0.0733 |
| `PROD1_DPSK_LATENT2_E2X` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 512 选 16 + 1 shared；routed scaling 2.5；routed 参数 ×2 | 有：d = h/2（h → d 投影，expert 在 d 上算，RMSNorm 后投回 h；router 和 shared expert 仍在 h 上） | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0833 | -0.0995 | -0.0998 | · |

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MUONH_FIT_DSV3 | -0.0029 | -0.0132 | -0.0155 | -0.0146 |
| MUONH_FIT | -0.0060 | -0.0109 | -0.0174 | -0.0163 |
| MUONH_FIT_DSV3_SWS | -0.0174 | -0.0200 | -0.0235 | -0.0242 |
| MUONH_FIT_DSV3_SWS_SHARED_RS2P5 | -0.0367 | -0.0463 | -0.0490 | -0.0481 |
| PROD1_DPSK | -0.0381 | -0.0517 | -0.0558 | -0.0552 |
| PROD1_DPSK_LATENT2 | -0.0475 | -0.0658 | -0.0742 | -0.0733 |
| PROD1_DPSK_LATENT2_E2X | -0.0833 | -0.0995 | -0.0998 | · |

**〔指标：val_v2〕**

| Change ID | LR | Router | 注意力（SWA / 位置编码 / sink） | MoE 层 | Latent MoE | 第 0 层 | 训练长度 / 文档 mask | 训练量 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `OPT_MUONH` | ArchScale 原版：0.016 · (L/8)^−0.5 · (T/10.4B)^−0.32 | softmax 打分；负载均衡 aux loss 1e-3 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | 1.7580 | 1.5419 | 1.4141 | 1.3266 |
| `MUONH_FIT` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | softmax 打分；负载均衡 aux loss 1e-3 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0031 | -0.0078 | -0.0116 | -0.0100 |
| `MUONH_FIT_DSV3` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0013 | -0.0101 | -0.0111 | -0.0097 |
| `MUONH_FIT_DSV3_SWS` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 2048（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0146 | -0.0147 | -0.0166 | -0.0157 |
| `MUONH_FIT_DSV3_SWS_SHARED_RS2P5` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 2048（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8 + 1 shared（3h/8）；routed scaling 2.5 | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0279 | -0.0345 | -0.0352 | -0.0339 |
| `PROD1_DPSK` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8 + 1 shared（3h/8）；routed scaling 2.5 | 无 | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0300 | -0.0394 | -0.0396 | -0.0376 |
| `PROD1_DPSK_LATENT2` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 256 选 16（expert 在 d 维上，每个半大）+ 1 shared；routed scaling 2.5；总参数同上一行 | 有：d = h/2（h → d 投影，expert 在 d 上算，RMSNorm 后投回 h；router 和 shared expert 仍在 h 上） | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0367 | -0.0496 | -0.0527 | -0.0515 |
| `PROD1_DPSK_LATENT2_E2X` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 512 选 16 + 1 shared；routed scaling 2.5；routed 参数 ×2 | 有：d = h/2（h → d 投影，expert 在 d 上算，RMSNorm 后投回 h；router 和 shared expert 仍在 h 上） | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0685 | -0.0777 | -0.0751 | · |

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MUONH_FIT_DSV3 | -0.0013 | -0.0101 | -0.0111 | -0.0097 |
| MUONH_FIT | -0.0031 | -0.0078 | -0.0116 | -0.0100 |
| MUONH_FIT_DSV3_SWS | -0.0146 | -0.0147 | -0.0166 | -0.0157 |
| MUONH_FIT_DSV3_SWS_SHARED_RS2P5 | -0.0279 | -0.0345 | -0.0352 | -0.0339 |
| PROD1_DPSK | -0.0300 | -0.0394 | -0.0396 | -0.0376 |
| PROD1_DPSK_LATENT2 | -0.0367 | -0.0496 | -0.0527 | -0.0515 |
| PROD1_DPSK_LATENT2_E2X | -0.0685 | -0.0777 | -0.0751 | · |

**〔指标：uncheatable〕**

| Change ID | LR | Router | 注意力（SWA / 位置编码 / sink） | MoE 层 | Latent MoE | 第 0 层 | 训练长度 / 文档 mask | 训练量 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `OPT_MUONH` | ArchScale 原版：0.016 · (L/8)^−0.5 · (T/10.4B)^−0.32 | softmax 打分；负载均衡 aux loss 1e-3 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | 2.1961 | 1.9065 | 1.7269 | 1.6025 |
| `MUONH_FIT` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | softmax 打分；负载均衡 aux loss 1e-3 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0050 | -0.0113 | -0.0195 | -0.0200 |
| `MUONH_FIT_DSV3` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0033 | -0.0153 | -0.0176 | -0.0169 |
| `MUONH_FIT_DSV3_SWS` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 2048（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0155 | -0.0202 | -0.0266 | -0.0274 |
| `MUONH_FIT_DSV3_SWS_SHARED_RS2P5` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 2048（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8 + 1 shared（3h/8）；routed scaling 2.5 | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0331 | -0.0490 | -0.0554 | -0.0566 |
| `PROD1_DPSK` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8 + 1 shared（3h/8）；routed scaling 2.5 | 无 | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0352 | -0.0580 | -0.0630 | -0.0642 |
| `PROD1_DPSK_LATENT2` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 256 选 16（expert 在 d 维上，每个半大）+ 1 shared；routed scaling 2.5；总参数同上一行 | 有：d = h/2（h → d 投影，expert 在 d 上算，RMSNorm 后投回 h；router 和 shared expert 仍在 h 上） | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0467 | -0.0722 | -0.0849 | -0.0859 |
| `PROD1_DPSK_LATENT2_E2X` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 512 选 16 + 1 shared；routed scaling 2.5；routed 参数 ×2 | 有：d = h/2（h → d 投影，expert 在 d 上算，RMSNorm 后投回 h；router 和 shared expert 仍在 h 上） | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0835 | -0.1103 | -0.1112 | · |

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MUONH_FIT_DSV3 | -0.0033 | -0.0153 | -0.0176 | -0.0169 |
| MUONH_FIT | -0.0050 | -0.0113 | -0.0195 | -0.0200 |
| MUONH_FIT_DSV3_SWS | -0.0155 | -0.0202 | -0.0266 | -0.0274 |
| MUONH_FIT_DSV3_SWS_SHARED_RS2P5 | -0.0331 | -0.0490 | -0.0554 | -0.0566 |
| PROD1_DPSK | -0.0352 | -0.0580 | -0.0630 | -0.0642 |
| PROD1_DPSK_LATENT2 | -0.0467 | -0.0722 | -0.0849 | -0.0859 |
| PROD1_DPSK_LATENT2_E2X | -0.0835 | -0.1103 | -0.1112 | · |

**〔指标：codefresh〕**

| Change ID | LR | Router | 注意力（SWA / 位置编码 / sink） | MoE 层 | Latent MoE | 第 0 层 | 训练长度 / 文档 mask | 训练量 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `OPT_MUONH` | ArchScale 原版：0.016 · (L/8)^−0.5 · (T/10.4B)^−0.32 | softmax 打分；负载均衡 aux loss 1e-3 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | 1.7185 | 1.4132 | 1.2432 | 1.1307 |
| `MUONH_FIT` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | softmax 打分；负载均衡 aux loss 1e-3 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0078 | -0.0154 | -0.0213 | -0.0200 |
| `MUONH_FIT_DSV3` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | +0.0015 | -0.0134 | -0.0138 | -0.0143 |
| `MUONH_FIT_DSV3_SWS` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 2048（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0191 | -0.0213 | -0.0254 | -0.0292 |
| `MUONH_FIT_DSV3_SWS_SHARED_RS2P5` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 2048（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8 + 1 shared（3h/8）；routed scaling 2.5 | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0468 | -0.0552 | -0.0550 | -0.0510 |
| `PROD1_DPSK` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8 + 1 shared（3h/8）；routed scaling 2.5 | 无 | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0464 | -0.0584 | -0.0629 | -0.0631 |
| `PROD1_DPSK_LATENT2` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 256 选 16（expert 在 d 维上，每个半大）+ 1 shared；routed scaling 2.5；总参数同上一行 | 有：d = h/2（h → d 投影，expert 在 d 上算，RMSNorm 后投回 h；router 和 shared expert 仍在 h 上） | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0565 | -0.0769 | -0.0838 | -0.0805 |
| `PROD1_DPSK_LATENT2_E2X` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 512 选 16 + 1 shared；routed scaling 2.5；routed 参数 ×2 | 有：d = h/2（h → d 投影，expert 在 d 上算，RMSNorm 后投回 h；router 和 shared expert 仍在 h 上） | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0985 | -0.1157 | -0.1133 | · |

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MUONH_FIT_DSV3 | 0.0015 | -0.0134 | -0.0138 | -0.0143 |
| MUONH_FIT | -0.0078 | -0.0154 | -0.0213 | -0.0200 |
| MUONH_FIT_DSV3_SWS | -0.0191 | -0.0213 | -0.0254 | -0.0292 |
| PROD1_DPSK | -0.0464 | -0.0584 | -0.0629 | -0.0631 |
| MUONH_FIT_DSV3_SWS_SHARED_RS2P5 | -0.0468 | -0.0552 | -0.0550 | -0.0510 |
| PROD1_DPSK_LATENT2 | -0.0565 | -0.0769 | -0.0838 | -0.0805 |
| PROD1_DPSK_LATENT2_E2X | -0.0985 | -0.1157 | -0.1133 | · |

**〔指标：private_internal〕**

| Change ID | LR | Router | 注意力（SWA / 位置编码 / sink） | MoE 层 | Latent MoE | 第 0 层 | 训练长度 / 文档 mask | 训练量 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `OPT_MUONH` | ArchScale 原版：0.016 · (L/8)^−0.5 · (T/10.4B)^−0.32 | softmax 打分；负载均衡 aux loss 1e-3 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | 2.3556 | 2.1127 | 1.9718 | 1.8766 |
| `MUONH_FIT` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | softmax 打分；负载均衡 aux loss 1e-3 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0085 | -0.0114 | -0.0175 | -0.0166 |
| `MUONH_FIT_DSV3` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 全注意力，RoPE；无 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0036 | -0.0136 | -0.0156 | -0.0152 |
| `MUONH_FIT_DSV3_SWS` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 2048（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8，无 shared expert | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200（q0 4,577 步） | -0.0148 | -0.0198 | -0.0228 | -0.0233 |
| `MUONH_FIT_DSV3_SWS_SHARED_RS2P5` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 2048（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8 + 1 shared（3h/8）；routed scaling 2.5 | 无 | MoE（同其他层） | 8K；文档之间不 mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0315 | -0.0427 | -0.0478 | -0.0463 |
| `PROD1_DPSK` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 128 选 8 + 1 shared（3h/8）；routed scaling 2.5 | 无 | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0347 | -0.0480 | -0.0530 | -0.0534 |
| `PROD1_DPSK_LATENT2` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 256 选 16（expert 在 d 维上，每个半大）+ 1 shared；routed scaling 2.5；总参数同上一行 | 有：d = h/2（h → d 投影，expert 在 d 上算，RMSNorm 后投回 h；router 和 shared expert 仍在 h 上） | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0429 | -0.0598 | -0.0698 | -0.0704 |
| `PROD1_DPSK_LATENT2_E2X` | 拟合 fit v2：0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259 | DSV3：sigmoid 打分；expert bias（γ 1e-3）+ seq aux 1e-4 | 5:1：5 层 SWA 窗口 512（RoPE）+ 1 层全注意力（NoPE）；可学习 sink | 512 选 16 + 1 shared；routed scaling 2.5；routed 参数 ×2 | 有：d = h/2（h → d 投影，expert 在 d 上算，RMSNorm 后投回 h；router 和 shared expert 仍在 h 上） | dense SwiGLU（宽 9 × 3h/8），注意力 SWA | 32K（每步仍 2^21 token）；文档间 attention mask | TPP 200 按新的激活参数算（q0 4,915 步，+7.4%） | -0.0724 | -0.0865 | -0.0874 | · |

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MUONH_FIT_DSV3 | -0.0036 | -0.0136 | -0.0156 | -0.0152 |
| MUONH_FIT | -0.0085 | -0.0114 | -0.0175 | -0.0166 |
| MUONH_FIT_DSV3_SWS | -0.0148 | -0.0198 | -0.0228 | -0.0233 |
| MUONH_FIT_DSV3_SWS_SHARED_RS2P5 | -0.0315 | -0.0427 | -0.0478 | -0.0463 |
| PROD1_DPSK | -0.0347 | -0.0480 | -0.0530 | -0.0534 |
| PROD1_DPSK_LATENT2 | -0.0429 | -0.0598 | -0.0698 | -0.0704 |
| PROD1_DPSK_LATENT2_E2X | -0.0724 | -0.0865 | -0.0874 | · |

- 顺序跟着现有的单变量实验走：shared expert 只在 SWA + sink 底座上测过，所以 MUONH_FIT_DSV3_SWS_SHARED_RS2P5 排在 MUONH_FIT_DSV3_SWS 后面；NoPE 没有单独的实验（DSV3 底座上的 RoPE 对照没跑）。
- MUONH_FIT_DSV3_SWS 一次加了 SWA 2048、NoPE、sink 三项（只有 q0 的 SWA 无 sink 实验没有放进来）；PROD1_DPSK 一次加了四项（窗口 2048 → 512、第 0 层 dense、32K 训练、文档 mask），没有逐项拆开的实验。
- 从 MUONH_FIT_DSV3_SWS_SHARED_RS2P5 起训练 token 多 7.4%（加了 shared expert 后激活参数变大，TPP 200 按激活参数算），LR 也随之略低；PROD1_DPSK_LATENT2_E2X 总参数 ×2（激活参数不变）。
- 最后三行（PROD1_DPSK、PROD1_DPSK_LATENT2、PROD1_DPSK_LATENT2_E2X）是 32K 训练，这里用 8K 窗口评测，32K 带来的好处测不到。单 seed，单格噪声约 ±0.008。

### 数据 derisk 和 2.5 vs 2.6

**数据**：80B 的 KD 语料预先打包成 32K 的行（pack plan e9r500：一行由若干文档块拼成，块之间互不 attend），teacher cache = tc_80b，seed 1（plan 固定了数据顺序）。
**实验**：MUONH_FIT 底座在同一份 plan 上训 CE / KD 2.6 / KD 2.5，检查新数据管线、KD 在新数据上的增益，以及 teacher 2.5 对 2.6。KD 都是 α 0.8（KD 0.8 + CE 0.2）、tail = S64Y。
**每个 checkpoint 评两份**：8K 窗口（长文档按 8192 切块，和之前所有实验一样，可直接比较）和 32K 窗口（按 32768 切块，和训练长度一致）。

| 线 | 数据 | 架构 | loss | Change ID（8K 窗口） |
|---|---|---|---|---|
| CE · MUONH_FIT · 新数据 | 80B pack plan，32K | MUONH_FIT：softmax router、全注意力、文档 mask | CE | `MUONH_FIT_PP32K` |
| KD 2.6 · MUONH_FIT · 新数据 | 80B pack plan，32K | 同上 | KD，teacher MiMo-2.6-Pro-RL | `KD_MUONH_A0P8_S64Y_FIT_PP32K` |
| KD 2.5 · MUONH_FIT · 新数据 | 80B pack plan，32K | 同上 | KD，teacher MiMo-2.5-Pro-Base | `KD_MUONH_A0P8_S64Y_FIT_MIMO25_PP32K` |
| KD 2.5 · PROD1_DPSK · 新数据 | 80B pack plan，32K | PROD1_DPSK（PROD v1：SWA 512 局部:全局 5:1 + sink、DSV3 router、第 0 层 dense、8 + 1 shared expert、文档 mask） | KD，teacher MiMo-2.5-Pro-Base | `PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K` |
| KD 2.6 · MUONH_FIT · 650B 数据（参照） | 650B 数据，8K | MUONH_FIT + 文档 mask | KD，teacher MiMo-2.6-Pro-RL | `KD_MUONH_A0P8_S64Y_FIT_DOCMASK` |
| KD 2.5 · MUONH_FIT · 45B 语料（参照） | 45B KD 语料，8K | MUONH_FIT（无文档 mask） | KD，teacher MiMo-2.5-Pro-Base | `KD_MUONH_FIT_A0P8_S64Y_TCS_MIMO25` |
| CE · PROD1_DPSK · 650B 数据（参照） | 650B 数据，32K | PROD1_DPSK，和「KD 2.5 · PROD1_DPSK · 新数据」同一架构；激活参数比 MUONH_FIT 多约 8%，训练 token 也多约 8%（TPP 200 按自己的激活参数算） | CE | `PROD1_DPSK` |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | 650B 数据，32K | PROD1_DPSK + latent MoE（d = h/2），routed expert 512 个选 16 个：每个 token 的计算量和 PROD1_DPSK 基本相同，routed 参数 ×2（不是等参数）；文档 mask | CE | `PROD1_DPSK_LATENT2_E2X` |

所有 KD 组都是 α 0.8（KD 0.8 + CE 0.2），tail = S64Y（top-64 + 采样 tail 4 draws，y 精确），MuonH + 拟合 LR。32K 窗口的评测 Change ID 在后面加 _W32K。虚线 = 之前的同类 run（8K 训练）作参照：KD 2.6 用 650B 数据上的；KD 2.5 在 650B 数据上没有 run，用 45B KD 语料上的（它的 q3 是 TPP 100，不画；只有 8K 窗口）。PROD1_DPSK 和 latent MoE expert ×2 两条参照都是 650B 数据上 32K 训练（加文档 mask）的 CE，所以三个窗口 tab 都画；expert ×2 的 q3 还没有评测。

**〔指标：overall_tsqw〕**

**〔视图：8K 窗口〕**

**8K 窗口：各线 − MUONH_FIT**

0 线 = **MUONH_FIT**：650B 数据、8K 训练、MuonH、拟合 LR、CE、softmax router、全注意力（和本报告其他章节同一条 0 线）。负数 = 更好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · MUONH_FIT · 新数据 | -0.0041 | -0.0033 | 0.0005 | 0.0027 |
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0325 | -0.0417 | -0.0395 | -0.0397 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0483 | -0.0528 | -0.0517 | -0.0539 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0505 | -0.0556 | -0.0520 | -0.0561 |
| KD 2.5 · MUONH_FIT · 45B 语料（参照） | -0.0506 | -0.0529 | -0.0498 | · |
| KD 2.6 · MUONH_FIT · 650B 数据（参照） | -0.0575 | -0.0620 | -0.0591 | -0.0617 |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.0775 | -0.0870 | -0.0882 | -0.0897 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.0782 | -0.0912 | -0.0851 | · |

**〔视图：32K 窗口〕**

**32K 窗口：各线 − MUONH_FIT**

0 线 = **MUONH_FIT**，和 8K 窗口 tab 同一条（它是 8K 训练，用的就是它的 8K 窗口分数，只作基准参考）；各线是 32K 窗口的分数。负数 = 更好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · MUONH_FIT · 新数据 | -0.0280 | -0.0290 | -0.0253 | -0.0229 |
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0655 | -0.0729 | -0.0695 | -0.0686 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0717 | -0.0781 | -0.0772 | -0.0790 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0737 | -0.0810 | -0.0774 | -0.0810 |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.1105 | -0.1179 | -0.1180 | -0.1182 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.1108 | -0.1216 | -0.1143 | · |

**〔视图：32K 窗口 · 0 线 = 新 CE〕**

**32K 窗口：各线 − MUONH_FIT_PP32K（新数据 CE，32K 窗口）**

0 线 = **MUONH_FIT_PP32K**：同一份 80B pack plan、32K 训练的 CE，也按 32K 窗口评，按 seed 配对。各线和它一样是 32K 窗口分数；负数 = 比新 CE 好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0409 | -0.0439 | -0.0441 | -0.0458 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0437 | -0.0491 | -0.0519 | -0.0561 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0457 | -0.0520 | -0.0520 | -0.0581 |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.0825 | -0.0889 | -0.0926 | -0.0953 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.0862 | -0.0926 | -0.0889 | · |

**〔指标：overall_tw〕**

**〔视图：8K 窗口〕**

**8K 窗口：各线 − MUONH_FIT**

0 线 = **MUONH_FIT**：650B 数据、8K 训练、MuonH、拟合 LR、CE、softmax router、全注意力（和本报告其他章节同一条 0 线）。负数 = 更好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · MUONH_FIT · 新数据 | -0.0028 | -0.0024 | 0.0008 | 0.0031 |
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0320 | -0.0427 | -0.0404 | -0.0406 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0490 | -0.0547 | -0.0541 | -0.0565 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0508 | -0.0569 | -0.0537 | -0.0578 |
| KD 2.5 · MUONH_FIT · 45B 语料（参照） | -0.0513 | -0.0548 | -0.0518 | · |
| KD 2.6 · MUONH_FIT · 650B 数据（参照） | -0.0591 | -0.0643 | -0.0616 | -0.0643 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.0782 | -0.0930 | -0.0868 | · |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.0784 | -0.0897 | -0.0915 | -0.0930 |

**〔视图：32K 窗口〕**

**32K 窗口：各线 − MUONH_FIT**

0 线 = **MUONH_FIT**，和 8K 窗口 tab 同一条（它是 8K 训练，用的就是它的 8K 窗口分数，只作基准参考）；各线是 32K 窗口的分数。负数 = 更好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · MUONH_FIT · 新数据 | -0.0267 | -0.0279 | -0.0248 | -0.0222 |
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0640 | -0.0729 | -0.0694 | -0.0686 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0726 | -0.0797 | -0.0794 | -0.0813 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0742 | -0.0821 | -0.0789 | -0.0824 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.1097 | -0.1224 | -0.1149 | · |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.1104 | -0.1195 | -0.1201 | -0.1205 |

**〔视图：32K 窗口 · 0 线 = 新 CE〕**

**32K 窗口：各线 − MUONH_FIT_PP32K（新数据 CE，32K 窗口）**

0 线 = **MUONH_FIT_PP32K**：同一份 80B pack plan、32K 训练的 CE，也按 32K 窗口评，按 seed 配对。各线和它一样是 32K 窗口分数；负数 = 比新 CE 好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0401 | -0.0450 | -0.0446 | -0.0464 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0459 | -0.0518 | -0.0546 | -0.0591 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0474 | -0.0542 | -0.0541 | -0.0602 |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.0837 | -0.0916 | -0.0954 | -0.0983 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.0858 | -0.0945 | -0.0902 | · |

**〔指标：overall_macro〕**

**〔视图：8K 窗口〕**

**8K 窗口：各线 − MUONH_FIT**

0 线 = **MUONH_FIT**：650B 数据、8K 训练、MuonH、拟合 LR、CE、softmax router、全注意力（和本报告其他章节同一条 0 线）。负数 = 更好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · MUONH_FIT · 新数据 | -0.0049 | -0.0050 | -0.0003 | 0.0022 |
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0321 | -0.0408 | -0.0384 | -0.0389 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0480 | -0.0522 | -0.0497 | -0.0516 |
| KD 2.5 · MUONH_FIT · 45B 语料（参照） | -0.0495 | -0.0509 | -0.0476 | · |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0503 | -0.0553 | -0.0505 | -0.0546 |
| KD 2.6 · MUONH_FIT · 650B 数据（参照） | -0.0566 | -0.0609 | -0.0570 | -0.0596 |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.0762 | -0.0848 | -0.0849 | -0.0864 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.0773 | -0.0886 | -0.0824 | · |

**〔视图：32K 窗口〕**

**32K 窗口：各线 − MUONH_FIT**

0 线 = **MUONH_FIT**，和 8K 窗口 tab 同一条（它是 8K 训练，用的就是它的 8K 窗口分数，只作基准参考）；各线是 32K 窗口的分数。负数 = 更好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · MUONH_FIT · 新数据 | -0.0218 | -0.0232 | -0.0185 | -0.0159 |
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0554 | -0.0628 | -0.0595 | -0.0591 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0641 | -0.0699 | -0.0677 | -0.0692 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0663 | -0.0733 | -0.0684 | -0.0721 |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.0995 | -0.1066 | -0.1057 | -0.1063 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.1003 | -0.1100 | -0.1028 | · |

**〔视图：32K 窗口 · 0 线 = 新 CE〕**

**32K 窗口：各线 − MUONH_FIT_PP32K（新数据 CE，32K 窗口）**

0 线 = **MUONH_FIT_PP32K**：同一份 80B pack plan、32K 训练的 CE，也按 32K 窗口评，按 seed 配对。各线和它一样是 32K 窗口分数；负数 = 比新 CE 好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0389 | -0.0396 | -0.0409 | -0.0432 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0423 | -0.0468 | -0.0491 | -0.0534 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0445 | -0.0501 | -0.0498 | -0.0562 |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.0778 | -0.0834 | -0.0871 | -0.0904 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.0838 | -0.0869 | -0.0843 | · |

**〔指标：val_v2〕**

**〔视图：8K 窗口〕**

**8K 窗口：各线 − MUONH_FIT**

0 线 = **MUONH_FIT**：650B 数据、8K 训练、MuonH、拟合 LR、CE、softmax router、全注意力（和本报告其他章节同一条 0 线）。负数 = 更好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · MUONH_FIT · 新数据 | -0.0041 | -0.0026 | 0.0004 | 0.0030 |
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0270 | -0.0316 | -0.0280 | -0.0276 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0362 | -0.0373 | -0.0347 | -0.0366 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0387 | -0.0406 | -0.0379 | -0.0398 |
| KD 2.5 · MUONH_FIT · 45B 语料（参照） | -0.0389 | -0.0403 | -0.0366 | · |
| KD 2.6 · MUONH_FIT · 650B 数据（参照） | -0.0414 | -0.0432 | -0.0399 | -0.0424 |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.0622 | -0.0651 | -0.0643 | -0.0648 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.0654 | -0.0699 | -0.0636 | · |

**〔视图：32K 窗口〕**

**32K 窗口：各线 − MUONH_FIT**

0 线 = **MUONH_FIT**，和 8K 窗口 tab 同一条（它是 8K 训练，用的就是它的 8K 窗口分数，只作基准参考）；各线是 32K 窗口的分数。负数 = 更好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · MUONH_FIT · 新数据 | -0.0460 | -0.0467 | -0.0433 | -0.0412 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0776 | -0.0808 | -0.0782 | -0.0797 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0796 | -0.0843 | -0.0807 | -0.0827 |
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0844 | -0.0858 | -0.0797 | -0.0775 |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.1197 | -0.1187 | -0.1158 | -0.1144 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.1220 | -0.1225 | -0.1138 | · |

**〔视图：32K 窗口 · 0 线 = 新 CE〕**

**32K 窗口：各线 − MUONH_FIT_PP32K（新数据 CE，32K 窗口）**

0 线 = **MUONH_FIT_PP32K**：同一份 80B pack plan、32K 训练的 CE，也按 32K 窗口评，按 seed 配对。各线和它一样是 32K 窗口分数；负数 = 比新 CE 好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| KD 2.6 · MUONH_FIT · 新数据 | -0.0316 | -0.0341 | -0.0349 | -0.0385 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0336 | -0.0376 | -0.0374 | -0.0415 |
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0401 | -0.0391 | -0.0364 | -0.0363 |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.0737 | -0.0721 | -0.0724 | -0.0732 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.0777 | -0.0758 | -0.0705 | · |

**〔指标：uncheatable〕**

**〔视图：8K 窗口〕**

**8K 窗口：各线 − MUONH_FIT**

0 线 = **MUONH_FIT**：650B 数据、8K 训练、MuonH、拟合 LR、CE、softmax router、全注意力（和本报告其他章节同一条 0 线）。负数 = 更好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · MUONH_FIT · 新数据 | 0.0006 | -0.0009 | 0.0018 | 0.0045 |
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0302 | -0.0466 | -0.0435 | -0.0442 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0517 | -0.0613 | -0.0606 | -0.0633 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0524 | -0.0617 | -0.0583 | -0.0625 |
| KD 2.5 · MUONH_FIT · 45B 语料（参照） | -0.0538 | -0.0604 | -0.0570 | · |
| KD 2.6 · MUONH_FIT · 650B 数据（参照） | -0.0627 | -0.0708 | -0.0682 | -0.0708 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.0785 | -0.0990 | -0.0918 | · |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.0817 | -0.0985 | -0.1008 | -0.1027 |

**〔视图：32K 窗口〕**

**32K 窗口：各线 − MUONH_FIT**

0 线 = **MUONH_FIT**，和 8K 窗口 tab 同一条（它是 8K 训练，用的就是它的 8K 窗口分数，只作基准参考）；各线是 32K 窗口的分数。负数 = 更好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · MUONH_FIT · 新数据 | -0.0229 | -0.0245 | -0.0217 | -0.0192 |
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0564 | -0.0719 | -0.0687 | -0.0691 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0749 | -0.0844 | -0.0836 | -0.0863 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0755 | -0.0850 | -0.0814 | -0.0854 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.1043 | -0.1238 | -0.1165 | · |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.1080 | -0.1240 | -0.1261 | -0.1276 |

**〔视图：32K 窗口 · 0 线 = 新 CE〕**

**32K 窗口：各线 − MUONH_FIT_PP32K（新数据 CE，32K 窗口）**

0 线 = **MUONH_FIT_PP32K**：同一份 80B pack plan、32K 训练的 CE，也按 32K 窗口评，按 seed 配对。各线和它一样是 32K 窗口分数；负数 = 比新 CE 好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0368 | -0.0474 | -0.0471 | -0.0499 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0520 | -0.0598 | -0.0619 | -0.0671 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0526 | -0.0605 | -0.0598 | -0.0662 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.0848 | -0.0993 | -0.0948 | · |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.0851 | -0.0994 | -0.1044 | -0.1084 |

**〔指标：codefresh〕**

**〔视图：8K 窗口〕**

**8K 窗口：各线 − MUONH_FIT**

0 线 = **MUONH_FIT**：650B 数据、8K 训练、MuonH、拟合 LR、CE、softmax router、全注意力（和本报告其他章节同一条 0 线）。负数 = 更好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · MUONH_FIT · 新数据 | -0.0119 | -0.0087 | 0.0041 | 0.0074 |
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0386 | -0.0429 | -0.0416 | -0.0431 |
| KD 2.5 · MUONH_FIT · 45B 语料（参照） | -0.0567 | -0.0497 | -0.0477 | · |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0613 | -0.0642 | -0.0541 | -0.0598 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0618 | -0.0611 | -0.0591 | -0.0599 |
| KD 2.6 · MUONH_FIT · 650B 数据（参照） | -0.0756 | -0.0764 | -0.0725 | -0.0740 |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.0853 | -0.0890 | -0.0872 | -0.0894 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.0907 | -0.1002 | -0.0920 | · |

**〔视图：32K 窗口〕**

**32K 窗口：各线 − MUONH_FIT**

0 线 = **MUONH_FIT**，和 8K 窗口 tab 同一条（它是 8K 训练，用的就是它的 8K 窗口分数，只作基准参考）；各线是 32K 窗口的分数。负数 = 更好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · MUONH_FIT · 新数据 | -0.0119 | -0.0087 | 0.0041 | 0.0074 |
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0386 | -0.0429 | -0.0416 | -0.0431 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0613 | -0.0642 | -0.0541 | -0.0598 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0618 | -0.0611 | -0.0591 | -0.0599 |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.0853 | -0.0890 | -0.0872 | -0.0894 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.0907 | -0.1002 | -0.0920 | · |

**〔视图：32K 窗口 · 0 线 = 新 CE〕**

**32K 窗口：各线 − MUONH_FIT_PP32K（新数据 CE，32K 窗口）**

0 线 = **MUONH_FIT_PP32K**：同一份 80B pack plan、32K 训练的 CE，也按 32K 窗口评，按 seed 配对。各线和它一样是 32K 窗口分数；负数 = 比新 CE 好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0348 | -0.0343 | -0.0457 | -0.0505 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0494 | -0.0555 | -0.0582 | -0.0672 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0499 | -0.0525 | -0.0632 | -0.0673 |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.0734 | -0.0803 | -0.0912 | -0.0968 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.0869 | -0.0916 | -0.0961 | · |

**〔指标：private_internal〕**

**〔视图：8K 窗口〕**

**8K 窗口：各线 − MUONH_FIT**

0 线 = **MUONH_FIT**：650B 数据、8K 训练、MuonH、拟合 LR、CE、softmax router、全注意力（和本报告其他章节同一条 0 线）。负数 = 更好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · MUONH_FIT · 新数据 | -0.0076 | -0.0074 | -0.0038 | -0.0015 |
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0262 | -0.0366 | -0.0355 | -0.0368 |
| KD 2.5 · MUONH_FIT · 45B 语料（参照） | -0.0436 | -0.0426 | -0.0412 | · |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0438 | -0.0480 | -0.0442 | -0.0462 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0448 | -0.0489 | -0.0444 | -0.0485 |
| KD 2.6 · MUONH_FIT · 650B 数据（参照） | -0.0477 | -0.0525 | -0.0480 | -0.0513 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.0639 | -0.0752 | -0.0699 | · |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.0663 | -0.0752 | -0.0752 | -0.0784 |

**〔视图：32K 窗口〕**

**32K 窗口：各线 − MUONH_FIT**

0 线 = **MUONH_FIT**，和 8K 窗口 tab 同一条（它是 8K 训练，用的就是它的 8K 窗口分数，只作基准参考）；各线是 32K 窗口的分数。负数 = 更好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · MUONH_FIT · 新数据 | -0.0096 | -0.0093 | -0.0057 | -0.0033 |
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0280 | -0.0384 | -0.0373 | -0.0385 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0458 | -0.0498 | -0.0460 | -0.0479 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0467 | -0.0507 | -0.0461 | -0.0502 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.0657 | -0.0769 | -0.0716 | · |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.0682 | -0.0770 | -0.0770 | -0.0801 |

**〔视图：32K 窗口 · 0 线 = 新 CE〕**

**32K 窗口：各线 − MUONH_FIT_PP32K（新数据 CE，32K 窗口）**

0 线 = **MUONH_FIT_PP32K**：同一份 80B pack plan、32K 训练的 CE，也按 32K 窗口评，按 seed 配对。各线和它一样是 32K 窗口分数；负数 = 比新 CE 好。

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| CE · PROD1_DPSK · 650B 数据（参照） | -0.0221 | -0.0291 | -0.0316 | -0.0352 |
| KD 2.6 · MUONH_FIT · 新数据 | -0.0361 | -0.0405 | -0.0403 | -0.0446 |
| KD 2.5 · MUONH_FIT · 新数据 | -0.0371 | -0.0414 | -0.0405 | -0.0469 |
| KD 2.5 · PROD1_DPSK · 新数据 | -0.0586 | -0.0677 | -0.0713 | -0.0768 |
| CE · PROD1_DPSK + latent MoE，expert ×2 · 650B 数据（参照） | -0.0597 | -0.0676 | -0.0659 | · |

## 页签：doc-isolated老实验

### 优化器对比

虚线 = Boxiang 按新拟合的 LR 公式跑的同一优化器（MuonH：fit v2 η = 0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259，目前有 q2 / q3；Muon 没有按新公式跑）。同一数据、同一 token 数，只换优化器（及其 LR 规则 / schedule）。横轴：激活参数（不含 embedding，对数坐标）。

**相对 AdamW 的 Δloss（优化器 − AdamW，越负越好）**

**〔指标：overall_tsqw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| Muon | -0.0607 | -0.0281 | -0.0217 | -0.0163 | -0.0139 |
| MuonH | -0.0779 | -0.0385 | -0.0256 | -0.0188 | -0.0138 |
| MuonH（新拟合 LR，MUONH_FIT） | -0.0829 | -0.0495 | -0.0428 | -0.0351 | · |

**〔指标：overall_tw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| Muon | -0.0612 | -0.0286 | -0.0220 | -0.0169 | -0.0142 |
| MuonH | -0.0790 | -0.0393 | -0.0261 | -0.0188 | -0.0133 |
| MuonH（新拟合 LR，MUONH_FIT） | -0.0838 | -0.0503 | -0.0437 | -0.0359 | · |

**〔指标：overall_macro〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| Muon | -0.0607 | -0.0277 | -0.0211 | -0.0153 | -0.0133 |
| MuonH | -0.0764 | -0.0373 | -0.0240 | -0.0174 | -0.0129 |
| MuonH（新拟合 LR，MUONH_FIT） | -0.0824 | -0.0482 | -0.0414 | -0.0337 | · |

**〔指标：val_v2〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| Muon | -0.0474 | -0.0208 | -0.0164 | -0.0124 | -0.0109 |
| MuonH | -0.0646 | -0.0299 | -0.0220 | -0.0176 | -0.0158 |
| MuonH（新拟合 LR，MUONH_FIT） | -0.0677 | -0.0377 | -0.0336 | -0.0277 | · |

**〔指标：uncheatable〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| Muon | -0.0642 | -0.0308 | -0.0225 | -0.0182 | -0.0148 |
| MuonH | -0.0829 | -0.0416 | -0.0260 | -0.0171 | -0.0102 |
| MuonH（新拟合 LR，MUONH_FIT） | -0.0879 | -0.0530 | -0.0455 | -0.0372 | · |

**〔指标：codefresh〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| Muon | -0.0846 | -0.0351 | -0.0248 | -0.0162 | -0.0148 |
| MuonH | -0.0949 | -0.0424 | -0.0245 | -0.0141 | -0.0110 |
| MuonH（新拟合 LR，MUONH_FIT） | -0.1027 | -0.0578 | -0.0458 | -0.0342 | · |

**〔指标：private_internal〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| Muon | -0.0567 | -0.0240 | -0.0178 | -0.0118 | -0.0104 |
| MuonH | -0.0694 | -0.0295 | -0.0159 | -0.0087 | -0.0024 |
| MuonH（新拟合 LR，MUONH_FIT） | -0.0779 | -0.0408 | -0.0334 | -0.0253 | · |

### KD 对 CE

CE 与 KD 只差 loss（α = 0 vs 0.7 / 0.8 / 0.9），优化器、数据顺序、layout 完全相同。

**KD − CE：各 α 的 Δloss（越负越好）**

**〔指标：overall_tsqw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| KD α 0.7 | -0.0375 | -0.0420 | -0.0441 | -0.0486 | -0.0504 |
| KD α 0.8 | -0.0385 | -0.0419 | -0.0453 | -0.0492 | -0.0518 |
| KD α 0.9 | -0.0389 | -0.0432 | -0.0453 | -0.0486 | -0.0510 |

**〔指标：overall_tw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| KD α 0.7 | -0.0396 | -0.0448 | -0.0469 | -0.0516 | -0.0539 |
| KD α 0.8 | -0.0408 | -0.0446 | -0.0486 | -0.0524 | -0.0552 |
| KD α 0.9 | -0.0415 | -0.0463 | -0.0487 | -0.0522 | -0.0549 |

**〔指标：overall_macro〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| KD α 0.7 | -0.0357 | -0.0391 | -0.0410 | -0.0455 | -0.0466 |
| KD α 0.8 | -0.0365 | -0.0389 | -0.0420 | -0.0458 | -0.0480 |
| KD α 0.9 | -0.0365 | -0.0401 | -0.0419 | -0.0448 | -0.0468 |

**〔指标：val_v2〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| KD α 0.7 | -0.0261 | -0.0302 | -0.0308 | -0.0343 | -0.0360 |
| KD α 0.8 | -0.0272 | -0.0301 | -0.0315 | -0.0347 | -0.0372 |
| KD α 0.9 | -0.0283 | -0.0312 | -0.0309 | -0.0340 | -0.0362 |

**〔指标：uncheatable〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| KD α 0.7 | -0.0448 | -0.0521 | -0.0536 | -0.0596 | -0.0623 |
| KD α 0.8 | -0.0468 | -0.0514 | -0.0566 | -0.0604 | -0.0632 |
| KD α 0.9 | -0.0481 | -0.0551 | -0.0574 | -0.0613 | -0.0641 |

**〔指标：codefresh〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| KD α 0.7 | -0.0434 | -0.0438 | -0.0488 | -0.0543 | -0.0542 |
| KD α 0.8 | -0.0450 | -0.0475 | -0.0502 | -0.0557 | -0.0569 |
| KD α 0.9 | -0.0459 | -0.0476 | -0.0518 | -0.0558 | -0.0583 |

**〔指标：private_internal〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| KD α 0.7 | -0.0294 | -0.0318 | -0.0309 | -0.0380 | -0.0379 |
| KD α 0.8 | -0.0305 | -0.0305 | -0.0323 | -0.0364 | -0.0390 |
| KD α 0.9 | -0.0296 | -0.0335 | -0.0315 | -0.0354 | -0.0370 |

### 尾部估算方式

8 个 run 只差 KD 项里 teacher 分布的截断方式：α = 0.8、MuonH 优化器、seed 1、数据顺序全部相同；基线仍是 **CE**（MuonH，α = 0）；**B64**（top-64 + 剩余 tail 合成一个 bucket）就是 KD α 0.8 本身，作为一条线一起画出来，各变体看是否比 B64 更低。负数 = 比 CE 好。

**命名：字母 = tail 的处理方式 · 数字 = top-k 的大小 · 末尾 Y = 是否把 y 单独精确算**

| 部分 | 含义 |
|---|---|
| B | Bucket：top-k 以外的 tail 合成一个整体，只要求 student 在 tail 上的总概率对 |
| S | Sampled：tail 用 cache 里预存的 4 个采样 token 来估计，结果无偏 |
| DD | Data Draw：不用 teacher 的 tail，改用语料里真实的下一个 token 近似 tail |
| DROP | 丢掉 tail：teacher 只在 top-k 内重新归一化（对照组） |
| 64 / 128 | 精确计算的 top-k 有多少个 token |
| Y | y exact：当真实的下一个 token y 落在 top-k 以外时，把 y 从 tail 里拆出来单独精确算 |

**Δloss（KD 变体 − CE），越低越好**

**〔指标：overall_tsqw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| B64 · top-64 + bucket（= KD α 0.8） | -0.0385 | -0.0419 | -0.0453 | -0.0492 |
| S64Y · top-64 + 采样 tail + y exact | -0.0446 | -0.0466 | -0.0484 | -0.0529 |
| S64 · top-64 + 采样 tail（4 draws） | -0.0409 | -0.0471 | -0.0492 | -0.0526 |
| S128 · top-128 + 采样 tail | -0.0411 | -0.0460 | -0.0487 | -0.0530 |
| B64Y · top-64 bucket + y exact | -0.0386 | -0.0436 | -0.0449 | · |
| B128 · top-128 + bucket | -0.0396 | -0.0437 | -0.0456 | · |
| DD64 · top-64 + data draw | -0.0390 | -0.0418 | -0.0437 | · |
| DROP64 · top-64，丢掉 tail（对照） | -0.0408 | · | · | · |
| S128Y · top-128 + 采样 tail + y exact | · | · | · | -0.0526 |

**〔指标：overall_tw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| B64 · top-64 + bucket（= KD α 0.8） | -0.0408 | -0.0446 | -0.0486 | -0.0524 |
| S64Y · top-64 + 采样 tail + y exact | -0.0467 | -0.0491 | -0.0515 | -0.0560 |
| S64 · top-64 + 采样 tail（4 draws） | -0.0431 | -0.0498 | -0.0521 | -0.0559 |
| S128 · top-128 + 采样 tail | -0.0432 | -0.0486 | -0.0517 | -0.0560 |
| B64Y · top-64 bucket + y exact | -0.0410 | -0.0463 | -0.0479 | · |
| B128 · top-128 + bucket | -0.0416 | -0.0464 | -0.0485 | · |
| DD64 · top-64 + data draw | -0.0408 | -0.0439 | -0.0463 | · |
| DROP64 · top-64，丢掉 tail（对照） | -0.0430 | · | · | · |
| S128Y · top-128 + 采样 tail + y exact | · | · | · | -0.0558 |

**〔指标：overall_macro〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| B64 · top-64 + bucket（= KD α 0.8） | -0.0365 | -0.0389 | -0.0420 | -0.0458 |
| S64Y · top-64 + 采样 tail + y exact | -0.0424 | -0.0442 | -0.0456 | -0.0499 |
| S64 · top-64 + 采样 tail（4 draws） | -0.0391 | -0.0446 | -0.0466 | -0.0496 |
| S128 · top-128 + 采样 tail | -0.0392 | -0.0434 | -0.0459 | -0.0502 |
| B64Y · top-64 bucket + y exact | -0.0364 | -0.0406 | -0.0419 | · |
| B128 · top-128 + bucket | -0.0377 | -0.0408 | -0.0428 | · |
| DD64 · top-64 + data draw | -0.0374 | -0.0395 | -0.0412 | · |
| DROP64 · top-64，丢掉 tail（对照） | -0.0388 | · | · | · |
| S128Y · top-128 + 采样 tail + y exact | · | · | · | -0.0497 |

**〔指标：val_v2〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| B64 · top-64 + bucket（= KD α 0.8） | -0.0272 | -0.0301 | -0.0315 | -0.0347 |
| S64Y · top-64 + 采样 tail + y exact | -0.0311 | -0.0335 | -0.0334 | -0.0367 |
| S64 · top-64 + 采样 tail（4 draws） | -0.0287 | -0.0337 | -0.0344 | -0.0369 |
| S128 · top-128 + 采样 tail | -0.0292 | -0.0335 | -0.0336 | -0.0373 |
| B64Y · top-64 bucket + y exact | -0.0277 | -0.0311 | -0.0314 | · |
| B128 · top-128 + bucket | -0.0280 | -0.0315 | -0.0319 | · |
| DD64 · top-64 + data draw | -0.0269 | -0.0311 | -0.0305 | · |
| DROP64 · top-64，丢掉 tail（对照） | -0.0289 | · | · | · |
| S128Y · top-128 + 采样 tail + y exact | · | · | · | -0.0368 |

**〔指标：uncheatable〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| B64 · top-64 + bucket（= KD α 0.8） | -0.0468 | -0.0514 | -0.0566 | -0.0604 |
| S64Y · top-64 + 采样 tail + y exact | -0.0521 | -0.0561 | -0.0595 | -0.0642 |
| S64 · top-64 + 采样 tail（4 draws） | -0.0483 | -0.0569 | -0.0593 | -0.0643 |
| S128 · top-128 + 采样 tail | -0.0483 | -0.0555 | -0.0596 | -0.0640 |
| B64Y · top-64 bucket + y exact | -0.0469 | -0.0536 | -0.0555 | · |
| B128 · top-128 + bucket | -0.0470 | -0.0532 | -0.0561 | · |
| DD64 · top-64 + data draw | -0.0454 | -0.0499 | -0.0531 | · |
| DROP64 · top-64，丢掉 tail（对照） | -0.0484 | · | · | · |
| S128Y · top-128 + 采样 tail + y exact | · | · | · | -0.0639 |

**〔指标：codefresh〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| B64 · top-64 + bucket（= KD α 0.8） | -0.0450 | -0.0475 | -0.0502 | -0.0557 |
| S64Y · top-64 + 采样 tail + y exact | -0.0535 | -0.0515 | -0.0543 | -0.0587 |
| S64 · top-64 + 采样 tail（4 draws） | -0.0482 | -0.0529 | -0.0559 | -0.0570 |
| S128 · top-128 + 采样 tail | -0.0495 | -0.0508 | -0.0545 | -0.0582 |
| B64Y · top-64 bucket + y exact | -0.0448 | -0.0508 | -0.0517 | · |
| B128 · top-128 + bucket | -0.0477 | -0.0468 | -0.0520 | · |
| DD64 · top-64 + data draw | -0.0441 | -0.0465 | -0.0476 | · |
| DROP64 · top-64，丢掉 tail（对照） | -0.0496 | · | · | · |
| S128Y · top-128 + 采样 tail + y exact | · | · | · | -0.0578 |

**〔指标：private_internal〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| B64 · top-64 + bucket（= KD α 0.8） | -0.0305 | -0.0305 | -0.0323 | -0.0364 |
| S64Y · top-64 + 采样 tail + y exact | -0.0384 | -0.0386 | -0.0372 | -0.0421 |
| S64 · top-64 + 采样 tail（4 draws） | -0.0338 | -0.0392 | -0.0381 | -0.0422 |
| S128 · top-128 + 采样 tail | -0.0355 | -0.0370 | -0.0386 | -0.0434 |
| B64Y · top-64 bucket + y exact | -0.0297 | -0.0330 | -0.0330 | · |
| B128 · top-128 + bucket | -0.0316 | -0.0332 | -0.0341 | · |
| DD64 · top-64 + data draw | -0.0331 | -0.0342 | -0.0337 | · |
| DROP64 · top-64，丢掉 tail（对照） | -0.0312 | · | · | · |
| S128Y · top-128 + 采样 tail + y exact | · | · | · | -0.0424 |

### 老师 MiMo-2.6 vs. MiMo-2.5

0 线 = **同一条 CE：OPT_MUONH**（MuonH 老 LR，650B 数据；q0 为 3 个 seed 的平均），三个色系都减这一条。每个点 = 该 KD 臂的 seed 平均 − OPT_MUONH；q0 的 3 个 seed 平均成一个点。
绿 vs 橙：LR、45B KD corpus v1（tc_small cache）、seed 和数据顺序完全相同，只有 teacher 不同，两者的距离 = 2.5 − 2.6。蓝 vs 绿：同是 MiMo-2.6，但 LR 和 KD 数据都不同（蓝色系用 650B 数据），所以两者的距离不只是 LR 的影响。蓝色系的 S128Y 只跑了 q3。

**各 KD 臂 − CE（鼠标移到某个 rung 看所有线的数值）**

**蓝色系**：MiMo-2.6 teacher，α 0.8，MuonH **老 LR**（ArchScale 原公式），650B 数据— S64 / S64Y / S128Y（由浅到深；S128Y 只有 q3）

**绿色系**：MiMo-2.6 teacher，α 0.8，MuonH **新 LR 公式**（fit），tc_small 45B — S64 / S64Y / S128Y（由浅到深）

**橙色系**：MiMo-2.5 teacher，α 0.8，MuonH **新 LR 公式**（fit），tc_small 45B — S64 / S64Y / S128Y（由浅到深）

纵轴 = KD − CE：OPT_MUONH（MuonH 老 LR），三个色系都减这同一条；纵轴范围 = 当前数据的最小值到最大值。绿 / 橙色系（45B 语料）的 q3 用的是 TPP 100，和其他点不可比，不画。

**〔指标：overall_tsqw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MiMo-2.6 老 LR S64 | -0.0421 | -0.0471 | -0.0492 | -0.0526 |
| MiMo-2.6 老 LR S64Y | -0.0458 | -0.0466 | -0.0484 | -0.0529 |
| MiMo-2.6 老 LR S128Y | · | · | · | -0.0526 |
| MiMo-2.6 新 LR S64 | -0.0524 | -0.0605 | -0.0647 | · |
| MiMo-2.6 新 LR S64Y | -0.0518 | -0.0611 | -0.0651 | · |
| MiMo-2.6 新 LR S128Y | -0.0518 | -0.0615 | -0.0641 | · |
| MiMo-2.5 新 LR S64 | -0.0540 | -0.0651 | -0.0678 | · |
| MiMo-2.5 新 LR S64Y | -0.0546 | -0.0638 | -0.0671 | · |
| MiMo-2.5 新 LR S128Y | -0.0543 | -0.0642 | -0.0659 | · |

**〔指标：overall_tw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MiMo-2.6 老 LR S64 | -0.0441 | -0.0498 | -0.0521 | -0.0559 |
| MiMo-2.6 老 LR S64Y | -0.0478 | -0.0491 | -0.0515 | -0.0560 |
| MiMo-2.6 老 LR S128Y | · | · | · | -0.0558 |
| MiMo-2.6 新 LR S64 | -0.0535 | -0.0633 | -0.0675 | · |
| MiMo-2.6 新 LR S64Y | -0.0527 | -0.0632 | -0.0679 | · |
| MiMo-2.6 新 LR S128Y | -0.0530 | -0.0641 | -0.0668 | · |
| MiMo-2.5 新 LR S64 | -0.0546 | -0.0668 | -0.0701 | · |
| MiMo-2.5 新 LR S64Y | -0.0553 | -0.0657 | -0.0694 | · |
| MiMo-2.5 新 LR S128Y | -0.0549 | -0.0659 | -0.0681 | · |

**〔指标：overall_macro〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MiMo-2.6 老 LR S64 | -0.0406 | -0.0446 | -0.0466 | -0.0496 |
| MiMo-2.6 老 LR S64Y | -0.0440 | -0.0442 | -0.0456 | -0.0499 |
| MiMo-2.6 老 LR S128Y | · | · | · | -0.0497 |
| MiMo-2.6 新 LR S64 | -0.0517 | -0.0578 | -0.0625 | · |
| MiMo-2.6 新 LR S64Y | -0.0513 | -0.0590 | -0.0627 | · |
| MiMo-2.6 新 LR S128Y | -0.0508 | -0.0590 | -0.0621 | · |
| MiMo-2.5 新 LR S64 | -0.0538 | -0.0632 | -0.0657 | · |
| MiMo-2.5 新 LR S64Y | -0.0541 | -0.0618 | -0.0650 | · |
| MiMo-2.5 新 LR S128Y | -0.0539 | -0.0624 | -0.0640 | · |

**〔指标：val_v2〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MiMo-2.6 老 LR S64 | -0.0305 | -0.0337 | -0.0344 | -0.0369 |
| MiMo-2.6 老 LR S64Y | -0.0329 | -0.0335 | -0.0334 | -0.0367 |
| MiMo-2.6 老 LR S128Y | · | · | · | -0.0368 |
| MiMo-2.6 新 LR S64 | -0.0386 | -0.0438 | -0.0445 | · |
| MiMo-2.6 新 LR S64Y | -0.0383 | -0.0443 | -0.0452 | · |
| MiMo-2.6 新 LR S128Y | -0.0385 | -0.0441 | -0.0447 | · |
| MiMo-2.5 新 LR S64 | -0.0412 | -0.0488 | -0.0483 | · |
| MiMo-2.5 新 LR S64Y | -0.0416 | -0.0481 | -0.0481 | · |
| MiMo-2.5 新 LR S128Y | -0.0419 | -0.0485 | -0.0476 | · |

**〔指标：uncheatable〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MiMo-2.6 老 LR S64 | -0.0496 | -0.0569 | -0.0593 | -0.0643 |
| MiMo-2.6 老 LR S64Y | -0.0535 | -0.0561 | -0.0595 | -0.0642 |
| MiMo-2.6 老 LR S128Y | · | · | · | -0.0639 |
| MiMo-2.6 新 LR S64 | -0.0573 | -0.0714 | -0.0757 | · |
| MiMo-2.6 新 LR S64Y | -0.0563 | -0.0705 | -0.0764 | · |
| MiMo-2.6 新 LR S128Y | -0.0570 | -0.0716 | -0.0748 | · |
| MiMo-2.5 新 LR S64 | -0.0574 | -0.0731 | -0.0767 | · |
| MiMo-2.5 新 LR S64Y | -0.0579 | -0.0718 | -0.0765 | · |
| MiMo-2.5 新 LR S128Y | -0.0575 | -0.0723 | -0.0748 | · |

**〔指标：codefresh〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MiMo-2.6 老 LR S64 | -0.0507 | -0.0529 | -0.0559 | -0.0570 |
| MiMo-2.6 老 LR S64Y | -0.0561 | -0.0515 | -0.0543 | -0.0587 |
| MiMo-2.6 老 LR S128Y | · | · | · | -0.0578 |
| MiMo-2.6 新 LR S64 | -0.0616 | -0.0631 | -0.0678 | · |
| MiMo-2.6 新 LR S64Y | -0.0616 | -0.0678 | -0.0691 | · |
| MiMo-2.6 新 LR S128Y | -0.0589 | -0.0669 | -0.0677 | · |
| MiMo-2.5 新 LR S64 | -0.0607 | -0.0667 | -0.0687 | · |
| MiMo-2.5 新 LR S64Y | -0.0625 | -0.0652 | -0.0690 | · |
| MiMo-2.5 新 LR S128Y | -0.0606 | -0.0658 | -0.0651 | · |

**〔指标：private_internal〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MiMo-2.6 老 LR S64 | -0.0362 | -0.0392 | -0.0381 | -0.0422 |
| MiMo-2.6 老 LR S64Y | -0.0408 | -0.0386 | -0.0372 | -0.0421 |
| MiMo-2.6 老 LR S128Y | · | · | · | -0.0424 |
| MiMo-2.6 新 LR S64 | -0.0478 | -0.0518 | -0.0575 | · |
| MiMo-2.6 新 LR S64Y | -0.0476 | -0.0527 | -0.0567 | · |
| MiMo-2.6 新 LR S128Y | -0.0470 | -0.0509 | -0.0563 | · |
| MiMo-2.5 新 LR S64 | -0.0491 | -0.0560 | -0.0588 | · |
| MiMo-2.5 新 LR S64Y | -0.0502 | -0.0539 | -0.0587 | · |
| MiMo-2.5 新 LR S128Y | -0.0490 | -0.0550 | -0.0576 | · |

### 文档 mask：加 mask vs 不加 mask

0 线 = **OPT_MUONH**：纯 CE，MuonH 老 LR
三条线都是 KD MiMo-2.6 teacher，α 0.8，S64Y，同一份 KD 数据（650B），softmax router，无 SWA。**加 mask** = 打包的 8K 序列里每个文档只看自己（Megatron --dataloader-inter-document-masking）。灰线和紫线是我们跑的一对，都用 MuonH 新 LR 公式（fit），只差 mask：两者的距离 = 文档 mask 的效果。蓝线是 Boxiang 原来用老 LR（ArchScale 原公式）跑的、不加 mask：蓝线和灰线只差 LR。每个点 = seed 平均 − OPT_MUONH（q0 有 3 个 seed 时平均成一个点）。注意：评测是 doc-isolated（每个文档单独评），和加 mask 训练时的条件一致。

**各条线 − CE（鼠标移到某个 rung 看各条线的数值）**

**不加 mask，老 LR**（Boxiang，KD_MUONH_A0P8_S64Y）：MuonH 老 LR（ArchScale 原公式），q0–q3 各 1 个 seed

**不加 mask，新 LR**（我们跑的，KD_MUONH_A0P8_S64Y_FIT）：MuonH 新 LR 公式（fit），打包的 8K 序列里可以看到前面的其他文档

**加 mask，新 LR**（我们跑的，KD_MUONH_A0P8_S64Y_FIT_DOCMASK）：MuonH 新 LR 公式（fit），每个文档只看自己

纵轴 = 该线 − CE：OPT_MUONH（MuonH 老 LR），三条线都减这同一条；纵轴范围 = 当前数据的最小值到最大值

**〔指标：overall_tsqw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| 不加 mask，老 LR（Boxiang） | -0.0458 | -0.0466 | -0.0484 | -0.0529 |
| 不加 mask，新 LR | -0.0473 | -0.0594 | -0.0669 | -0.0674 |
| 加 mask，新 LR | -0.0615 | -0.0729 | -0.0763 | -0.0780 |

**〔指标：overall_tw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| 不加 mask，老 LR（Boxiang） | -0.0478 | -0.0491 | -0.0515 | -0.0560 |
| 不加 mask，新 LR | -0.0494 | -0.0624 | -0.0703 | -0.0714 |
| 加 mask，新 LR | -0.0630 | -0.0752 | -0.0792 | -0.0814 |

**〔指标：overall_macro〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| 不加 mask，老 LR（Boxiang） | -0.0440 | -0.0442 | -0.0456 | -0.0499 |
| 不加 mask，新 LR | -0.0460 | -0.0565 | -0.0641 | -0.0641 |
| 加 mask，新 LR | -0.0613 | -0.0717 | -0.0744 | -0.0759 |

**〔指标：val_v2〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| 不加 mask，老 LR（Boxiang） | -0.0329 | -0.0335 | -0.0334 | -0.0367 |
| 不加 mask，新 LR | -0.0340 | -0.0424 | -0.0458 | -0.0455 |
| 加 mask，新 LR | -0.0441 | -0.0510 | -0.0515 | -0.0525 |

**〔指标：uncheatable〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| 不加 mask，老 LR（Boxiang） | -0.0535 | -0.0561 | -0.0595 | -0.0642 |
| 不加 mask，新 LR | -0.0558 | -0.0711 | -0.0799 | -0.0828 |
| 加 mask，新 LR | -0.0668 | -0.0822 | -0.0877 | -0.0908 |

**〔指标：codefresh〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| 不加 mask，老 LR（Boxiang） | -0.0561 | -0.0515 | -0.0543 | -0.0587 |
| 不加 mask，新 LR | -0.0551 | -0.0685 | -0.0763 | -0.0736 |
| 加 mask，新 LR | -0.0815 | -0.0918 | -0.0938 | -0.0941 |

**〔指标：private_internal〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| 不加 mask，老 LR（Boxiang） | -0.0408 | -0.0386 | -0.0372 | -0.0421 |
| 不加 mask，新 LR | -0.0422 | -0.0509 | -0.0561 | -0.0580 |
| 加 mask，新 LR | -0.0542 | -0.0639 | -0.0655 | -0.0679 |

### 滑动窗口注意力 SWA

设置：MuonH，新 LR 公式（fit v2），纯 CE（不用 KD），128 选 8 MoE；q0–q3，部分组 q0 有 3 个 seed。
左图：softmax router，0 线 = **MUONH_FIT**（全注意力）；右图：DSV3 router，0 线 = **MUONH_FIT_DSV3**（全注意力）。数值 = 改动 − 0 线（doc-isolated，负数 = 改动更好）；各改动的具体设定见下表。

| 改动 | Change ID | 具体改了什么 | 对照 | LR 公式（按实际峰值 LR 判断） | 已有结果 |
|---|---|---|---|---|---|
| ■ SWA（基础版） | `SWA_FIT` | 每 6 层中 5 层滑动窗口注意力（窗口 2048，RoPE）+ 1 层全注意力（NoPE） | `MUONH_FIT` | 新公式（fit v2） | q0 ×3, q1, q2, q3 |
| ■ SWA，全注意力层用 RoPE | `SWA_FIT_ROPE` | 同基础版，但全注意力层也用 RoPE | `MUONH_FIT` | 新公式（fit v2） | q0, q1, q2, q3 |
| ■ SWA，窗口 4096 | `SWA_FIT_W4K` | 同基础版，窗口 4096 | `MUONH_FIT` | 新公式（fit v2） | q0, q1, q2, q3 |
| ■ SWA，窗口 1024 | `SWA_FIT_W1K` | 同基础版，窗口 1024 | `MUONH_FIT` | 新公式（fit v2） | q0 ×3, q1, q2, q3 |
| ■ SWA + attention sink | `SWA_FIT_SINK` | 同基础版 + 可学习 attention sink（gpt-oss 式：每层每个 query head 一个可学习的 sink logit） | `MUONH_FIT` | 新公式（fit v2） | q0 ×3, q1, q2, q3 |
| ■ SWA + sink + 窗口 1024 | `SWA_FIT_SINK_W1K` | attention sink + 窗口 1024 | `MUONH_FIT` | 新公式（fit v2） | q0, q1, q2, q3 |
| ■ SWA + sink + 窗口 512 | `SWA_FIT_SINK_W512` | attention sink + 窗口 512（窗口这条轴往下扫） | `MUONH_FIT` | 新公式（fit v2） | q0, q1, q2, q3 |
| ■ SWA + sink + 窗口 128 | `SWA_FIT_SINK_W128` | attention sink + 窗口 128（窗口这条轴往下扫） | `MUONH_FIT` | 新公式（fit v2） | q0, q1, q2, q3 |
| ■ SWA + sink + 窗口 512，3:1（靠底，q3 没训） | `SWA_FIT_SINK_W512_R3` | 同窗口 512 + sink，但每 4 层 1 层全注意力（NoPE），局部:全局从 5:1 改成 3:1（比例这条轴；Boxiang 10-30） | `MUONH_FIT` | 新公式（fit v2） | q0, q1, q2 |
| ■ SWA + sink + 窗口 512，3:1（靠顶） | `SWA_FIT_SINK_W512_R3T` | 同 3:1，但全注意力层从顶往下数（最后一层一定是全注意力，和 5:1 一样）；R3 从底往上数，有的深度最顶两层都是 SWA。q0（12 层）和 q2（24 层）层数是 4 的倍数，两种数法排出来一样，两条线用同一个 run（q2 的 argv 与 R3 q2 的 golden 逐项相同）；只有 q1、q3 两者不同，R3 的 q3 没训 | `MUONH_FIT` | 新公式（fit v2） | q0, q1, q2, q3 |
| ■ SWA（DSV3 router 底座） | `SWA_FIT_DSV3` | 基础版 SWA，但底座换成 DeepSeek-V3 式 router（对照也是 DSV3 底座） | `MUONH_FIT_DSV3` | 新公式（fit v2） | q0 ×3 |
| ■ SWA + sink（DSV3 底座，SWS） | `MUONH_FIT_DSV3_SWS` | DSV3 router 底座上的 SWA（窗口 2048，全注意力层 NoPE）+ 可学习 attention sink；Runlong 10-29 起 router 消融的新底座（SWS） | `MUONH_FIT_DSV3` | 新公式（fit v2） | q0 ×3, q1, q2, q3 |

**〔指标：overall_tsqw〕**

**softmax router：各改动 − MUONH_FIT（全注意力，softmax router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA − 无 SWA（MUONH_FIT） | -0.0047 | 0.0004 | -0.0028 | -0.0028 |
| SWA，全注意力层用 RoPE − MUONH_FIT | 0.0177 | -0.0016 | 0.0005 | 0.0003 |
| SWA，窗口 4096 − MUONH_FIT | 0.0019 | 0.0055 | -0.0030 | -0.0001 |
| SWA，窗口 1024 − MUONH_FIT | 0.0002 | -0.0048 | -0.0004 | -0.0033 |
| SWA + 可学习 attention sink（gpt-oss 式）− MUONH_FIT | -0.0129 | -0.0107 | -0.0067 | -0.0054 |
| SWA + sink + 窗口 1024 − MUONH_FIT | -0.0052 | -0.0098 | -0.0095 | -0.0106 |
| SWA + sink + 窗口 512 − MUONH_FIT | -0.0072 | -0.0131 | -0.0138 | -0.0142 |
| SWA + sink + 窗口 128 − MUONH_FIT | 0.0029 | -0.0096 | -0.0095 | -0.0117 |
| SWA + sink + 窗口 512，局部:全局 3:1（全注意力层靠底，q3 没训）− MUONH_FIT | -0.0129 | -0.0175 | -0.0117 | · |
| SWA + sink + 窗口 512，3:1（全注意力层靠顶）− MUONH_FIT | -0.0129 | -0.0159 | -0.0117 | -0.0130 |

**DSV3 router：各改动 − MUONH_FIT_DSV3（全注意力，DSV3 router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA（DSV3 router 底座）− 同底座无 SWA（MUONH_FIT_DSV3） | -0.0087 | · | · | · |
| SWA + sink（DSV3 router 底座，SWS）− 同底座无 SWA（MUONH_FIT_DSV3） | -0.0158 | -0.0063 | -0.0082 | -0.0095 |

**〔指标：overall_tw〕**

**softmax router：各改动 − MUONH_FIT（全注意力，softmax router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA − 无 SWA（MUONH_FIT） | -0.0049 | 0.0005 | -0.0033 | -0.0033 |
| SWA，全注意力层用 RoPE − MUONH_FIT | 0.0190 | -0.0013 | 0.0010 | 0.0009 |
| SWA，窗口 4096 − MUONH_FIT | 0.0018 | 0.0057 | -0.0036 | -0.0003 |
| SWA，窗口 1024 − MUONH_FIT | 0.0005 | -0.0050 | -0.0007 | -0.0037 |
| SWA + 可学习 attention sink（gpt-oss 式）− MUONH_FIT | -0.0124 | -0.0109 | -0.0065 | -0.0049 |
| SWA + sink + 窗口 1024 − MUONH_FIT | -0.0042 | -0.0094 | -0.0099 | -0.0108 |
| SWA + sink + 窗口 512 − MUONH_FIT | -0.0066 | -0.0137 | -0.0144 | -0.0147 |
| SWA + sink + 窗口 128 − MUONH_FIT | 0.0028 | -0.0104 | -0.0104 | -0.0125 |
| SWA + sink + 窗口 512，局部:全局 3:1（全注意力层靠底，q3 没训）− MUONH_FIT | -0.0128 | -0.0184 | -0.0121 | · |
| SWA + sink + 窗口 512，3:1（全注意力层靠顶）− MUONH_FIT | -0.0128 | -0.0168 | -0.0121 | -0.0139 |

**DSV3 router：各改动 − MUONH_FIT_DSV3（全注意力，DSV3 router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA（DSV3 router 底座）− 同底座无 SWA（MUONH_FIT_DSV3） | -0.0088 | · | · | · |
| SWA + sink（DSV3 router 底座，SWS）− 同底座无 SWA（MUONH_FIT_DSV3） | -0.0156 | -0.0060 | -0.0085 | -0.0097 |

**〔指标：overall_macro〕**

**softmax router：各改动 − MUONH_FIT（全注意力，softmax router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA − 无 SWA（MUONH_FIT） | -0.0034 | 0.0000 | -0.0023 | -0.0025 |
| SWA，全注意力层用 RoPE − MUONH_FIT | 0.0150 | -0.0026 | -0.0005 | -0.0004 |
| SWA，窗口 4096 − MUONH_FIT | 0.0023 | 0.0046 | -0.0022 | 0.0003 |
| SWA，窗口 1024 − MUONH_FIT | 0.0003 | -0.0048 | 0.0001 | -0.0025 |
| SWA + 可学习 attention sink（gpt-oss 式）− MUONH_FIT | -0.0130 | -0.0108 | -0.0068 | -0.0061 |
| SWA + sink + 窗口 1024 − MUONH_FIT | -0.0061 | -0.0106 | -0.0090 | -0.0105 |
| SWA + sink + 窗口 512 − MUONH_FIT | -0.0068 | -0.0124 | -0.0129 | -0.0135 |
| SWA + sink + 窗口 128 − MUONH_FIT | 0.0049 | -0.0081 | -0.0083 | -0.0105 |
| SWA + sink + 窗口 512，局部:全局 3:1（全注意力层靠底，q3 没训）− MUONH_FIT | -0.0110 | -0.0162 | -0.0108 | · |
| SWA + sink + 窗口 512，3:1（全注意力层靠顶）− MUONH_FIT | -0.0110 | -0.0150 | -0.0108 | -0.0121 |

**DSV3 router：各改动 − MUONH_FIT_DSV3（全注意力，DSV3 router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA（DSV3 router 底座）− 同底座无 SWA（MUONH_FIT_DSV3） | -0.0086 | · | · | · |
| SWA + sink（DSV3 router 底座，SWS）− 同底座无 SWA（MUONH_FIT_DSV3） | -0.0148 | -0.0067 | -0.0080 | -0.0096 |

**〔指标：val_v2〕**

**softmax router：各改动 − MUONH_FIT（全注意力，softmax router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA − 无 SWA（MUONH_FIT） | -0.0046 | 0.0013 | -0.0009 | -0.0006 |
| SWA，全注意力层用 RoPE − MUONH_FIT | 0.0241 | -0.0008 | 0.0012 | 0.0007 |
| SWA，窗口 4096 − MUONH_FIT | 0.0029 | 0.0061 | -0.0019 | 0.0004 |
| SWA，窗口 1024 − MUONH_FIT | -0.0002 | -0.0041 | -0.0001 | -0.0021 |
| SWA + 可学习 attention sink（gpt-oss 式）− MUONH_FIT | -0.0099 | -0.0076 | -0.0041 | -0.0034 |
| SWA + sink + 窗口 1024 − MUONH_FIT | -0.0055 | -0.0064 | -0.0059 | -0.0070 |
| SWA + sink + 窗口 512 − MUONH_FIT | -0.0065 | -0.0093 | -0.0087 | -0.0093 |
| SWA + sink + 窗口 128 − MUONH_FIT | 0.0025 | -0.0061 | -0.0053 | -0.0073 |
| SWA + sink + 窗口 512，局部:全局 3:1（全注意力层靠底，q3 没训）− MUONH_FIT | -0.0129 | -0.0134 | -0.0084 | · |
| SWA + sink + 窗口 512，3:1（全注意力层靠顶）− MUONH_FIT | -0.0129 | -0.0116 | -0.0084 | -0.0084 |

**DSV3 router：各改动 − MUONH_FIT_DSV3（全注意力，DSV3 router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA（DSV3 router 底座）− 同底座无 SWA（MUONH_FIT_DSV3） | -0.0078 | · | · | · |
| SWA + sink（DSV3 router 底座，SWS）− 同底座无 SWA（MUONH_FIT_DSV3） | -0.0140 | -0.0045 | -0.0055 | -0.0060 |

**〔指标：uncheatable〕**

**softmax router：各改动 − MUONH_FIT（全注意力，softmax router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA − 无 SWA（MUONH_FIT） | -0.0050 | 0.0010 | -0.0050 | -0.0048 |
| SWA，全注意力层用 RoPE − MUONH_FIT | 0.0230 | -0.0014 | 0.0020 | 0.0021 |
| SWA，窗口 4096 − MUONH_FIT | 0.0021 | 0.0062 | -0.0056 | -0.0011 |
| SWA，窗口 1024 − MUONH_FIT | 0.0021 | -0.0060 | -0.0015 | -0.0047 |
| SWA + 可学习 attention sink（gpt-oss 式）− MUONH_FIT | -0.0111 | -0.0111 | -0.0064 | -0.0038 |
| SWA + sink + 窗口 1024 − MUONH_FIT | -0.0005 | -0.0085 | -0.0107 | -0.0114 |
| SWA + sink + 窗口 512 − MUONH_FIT | -0.0023 | -0.0146 | -0.0162 | -0.0165 |
| SWA + sink + 窗口 128 − MUONH_FIT | 0.0054 | -0.0119 | -0.0123 | -0.0146 |
| SWA + sink + 窗口 512，局部:全局 3:1（全注意力层靠底，q3 没训）− MUONH_FIT | -0.0111 | -0.0212 | -0.0137 | · |
| SWA + sink + 窗口 512，3:1（全注意力层靠顶）− MUONH_FIT | -0.0111 | -0.0192 | -0.0137 | -0.0166 |

**DSV3 router：各改动 − MUONH_FIT_DSV3（全注意力，DSV3 router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA（DSV3 router 底座）− 同底座无 SWA（MUONH_FIT_DSV3） | -0.0093 | · | · | · |
| SWA + sink（DSV3 router 底座，SWS）− 同底座无 SWA（MUONH_FIT_DSV3） | -0.0141 | -0.0049 | -0.0089 | -0.0105 |

**〔指标：codefresh〕**

**softmax router：各改动 − MUONH_FIT（全注意力，softmax router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA − 无 SWA（MUONH_FIT） | -0.0054 | 0.0012 | -0.0022 | -0.0014 |
| SWA，全注意力层用 RoPE − MUONH_FIT | -0.0027 | -0.0047 | -0.0025 | -0.0021 |
| SWA，窗口 4096 − MUONH_FIT | 0.0008 | 0.0032 | -0.0009 | 0.0000 |
| SWA，窗口 1024 − MUONH_FIT | -0.0001 | -0.0026 | 0.0031 | -0.0014 |
| SWA + 可学习 attention sink（gpt-oss 式）− MUONH_FIT | -0.0194 | -0.0140 | -0.0113 | -0.0102 |
| SWA + sink + 窗口 1024 − MUONH_FIT | -0.0119 | -0.0166 | -0.0124 | -0.0129 |
| SWA + sink + 窗口 512 − MUONH_FIT | -0.0081 | -0.0131 | -0.0160 | -0.0167 |
| SWA + sink + 窗口 128 − MUONH_FIT | 0.0093 | -0.0065 | -0.0096 | -0.0120 |
| SWA + sink + 窗口 512，局部:全局 3:1（全注意力层靠底，q3 没训）− MUONH_FIT | -0.0102 | -0.0173 | -0.0149 | · |
| SWA + sink + 窗口 512，3:1（全注意力层靠顶）− MUONH_FIT | -0.0102 | -0.0147 | -0.0149 | -0.0144 |

**DSV3 router：各改动 − MUONH_FIT_DSV3（全注意力，DSV3 router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA（DSV3 router 底座）− 同底座无 SWA（MUONH_FIT_DSV3） | -0.0069 | · | · | · |
| SWA + sink（DSV3 router 底座，SWS）− 同底座无 SWA（MUONH_FIT_DSV3） | -0.0210 | -0.0079 | -0.0116 | -0.0149 |

**〔指标：private_internal〕**

**softmax router：各改动 − MUONH_FIT（全注意力，softmax router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA − 无 SWA（MUONH_FIT） | -0.0024 | -0.0006 | -0.0037 | -0.0048 |
| SWA，全注意力层用 RoPE − MUONH_FIT | 0.0163 | -0.0042 | -0.0022 | -0.0021 |
| SWA，窗口 4096 − MUONH_FIT | 0.0044 | 0.0050 | -0.0031 | -0.0011 |
| SWA，窗口 1024 − MUONH_FIT | 0.0029 | -0.0054 | -0.0029 | -0.0038 |
| SWA + 可学习 attention sink（gpt-oss 式）− MUONH_FIT | -0.0100 | -0.0101 | -0.0059 | -0.0062 |
| SWA + sink + 窗口 1024 − MUONH_FIT | -0.0038 | -0.0086 | -0.0082 | -0.0110 |
| SWA + sink + 窗口 512 − MUONH_FIT | -0.0004 | -0.0102 | -0.0111 | -0.0138 |
| SWA + sink + 窗口 128 − MUONH_FIT | 0.0150 | -0.0018 | -0.0048 | -0.0088 |
| SWA + sink + 窗口 512，局部:全局 3:1（全注意力层靠底，q3 没训）− MUONH_FIT | -0.0046 | -0.0130 | -0.0095 | · |
| SWA + sink + 窗口 512，3:1（全注意力层靠顶）− MUONH_FIT | -0.0046 | -0.0133 | -0.0095 | -0.0118 |

**DSV3 router：各改动 − MUONH_FIT_DSV3（全注意力，DSV3 router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA（DSV3 router 底座）− 同底座无 SWA（MUONH_FIT_DSV3） | -0.0092 | · | · | · |
| SWA + sink（DSV3 router 底座，SWS）− 同底座无 SWA（MUONH_FIT_DSV3） | -0.0146 | -0.0062 | -0.0072 | -0.0081 |

每种改动相对各自对照的 Δloss（全部是新 LR 公式；有 3 个 seed 的点 = 3 个 seed 按 seed 配对差的平均，其余是 1 个 seed；鼠标移到某个 rung 看所有改动的数值）

### Router 选择

0 线 = **MUONH_FIT**：MuonH，新 LR 公式（fit，q0 峰值 0.0164），**纯 CE（不用 KD）**，128 选 8 MoE，**全注意力**（不含 SWA / sink），原版 softmax router + Switch aux loss 1e-3，gate 用 AdamH、LR ×1。

**softmax router**

- 对 128 个 expert 的 logit 做 softmax，取分数最高的 8 个。
- 负载均衡靠 aux loss（Switch 式，系数 1e-3）：某些 expert 被选得太多时，额外加一项惩罚，让 router 把 token 分散开。
  - **怎么做的**：每个 MoE 层，在每个 micro-batch（32,768 个 token）里统计两样东西：f_i = expert i 实际分到的 token 比例，P_i = 这些 token 给 expert i 的平均 softmax 概率。aux = 1e-3 × 128 × Σ_i f_i·P_i，训练时相当于加到 loss 上一起反向传播。f_i 只是计数、不求导，梯度只走 P_i：分到 token 越多的 expert（f_i 越大），它的概率被往下压得越多。
- micro-batch 大小（ladder 每个 rung 的布局表里定的，按显存选）决定每次 aux 统计多少 token： rung 节点 micro-batch 每次 aux 统计的 token 数 EP q0 4 4 32,768 1 q1 8 4 32,768 1 q2 16 4 32,768 1 q3 16 4 32,768 4 q4 32 2 16,384 8 q5 64 1 8,192 8 q6 64 1 8,192 8
  - **EP（expert 并行）**：把 128 个 expert 分到几张 GPU 上存，每张只存 128 ÷ EP 个（q3：4 张卡各 32 个；q4–q6：8 张卡各 16 个）。每层 router 先在本卡给自己的 token 选好 8 个 expert，再用 all-to-all 通信把 token 发到存着这些 expert 的卡上计算，算完再发回。它只是为了省显存，不改变计算结果；router 的选择和 aux 统计都在 token 发出去之前、在本卡的 micro-batch 上完成，所以也不影响 aux loss。

**DSV3 router**

- 每个 expert 的分数单独过 sigmoid，各自在 0–1 之间，互相不竞争。
- 选 expert 时用“分数 + bias”排序。每个 expert 有一个 bias，它不是训练出来的参数，而是每步按规则调整：负载太多的 expert，bias 减 0.001；太少的，bias 加 0.001。
- **bias 只影响选谁，不影响被选中 expert 的输出权重**：选出 8 个之后，权重用它们原始的 sigmoid 分数（不含 bias）在这 8 个里归一化（Megatron `moe_utils.py` L953–959）。
- 另外保留一个很小的 aux loss（seq aux 1e-4），按每条序列计算。
  - **怎么做的**：和上面 Switch aux 是同一个公式，只是统计范围换成**每一条 8K 训练序列**（打包后的一行，可能含几篇文档），系数小 10 倍。每条序列里：f_i = expert i 分到的 token 比例，P_i = 这条序列的 token 给 expert i 的平均分数（每个 token 的 128 个 sigmoid 分数先归一化成和为 1）。aux = 1e-4 × 128 × Σ_i f_i·P_i，micro-batch 里 4 条序列各算一个再取平均；同样只有 P_i 传梯度。因为按序列统计，它和 micro-batch 大小无关。
- 设计思路：主要靠 bias 做负载均衡，几乎不用 aux loss，避免 aux loss 干扰主任务。

**gate（router 权重）用什么优化器、多大 LR**

gate 就是算 logit 的那个权重矩阵 W_g（128 × 宽度）。

- **AdamH**：MuonH 默认的做法，用 Adam 更新，但把权重约束在固定半径的球面上（hyperball），权重范数不会变大。
- **AdamW**：普通 AdamW，权重范数可以自由变化（B2 里 wd 设为 0）。
- **router LR ×1**：gate 和模型主体用同一个 LR（新公式下 q0 是 0.0164）。
- **router LR ×0.19**：gate 的 LR 降到约 3.1e-3，大约是 AdamW 基线自己的 LR 水平。之前的探针 B1 显示，AdamW gate 用 ×1 的 LR 会让 logit 饱和，所以才降到 0.19。

**MuonH 主体：各组 − 对照（softmax router，MUONH_FIT）**

**〔指标：overall_tsqw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3 | 0.0023 | -0.0024 | 0.0018 | 0.0017 |
| DSV3 + AdamW gate + router LR ×0.19 | 0.0051 | 0.0020 | · | · |
| DSV3 + router LR ×0.19 | 0.0074 | 0.0050 | · | · |
| DSV3 去掉 seq aux | -0.0003 | · | · | · |
| DSV3 + MuonH gate | 0.0029 | · | · | · |

**〔指标：overall_tw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3 | 0.0021 | -0.0028 | 0.0019 | 0.0021 |
| DSV3 + AdamW gate + router LR ×0.19 | 0.0050 | 0.0014 | · | · |
| DSV3 + router LR ×0.19 | 0.0073 | 0.0048 | · | · |
| DSV3 去掉 seq aux | -0.0002 | · | · | · |
| DSV3 + MuonH gate | 0.0026 | · | · | · |

**〔指标：overall_macro〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3 | 0.0032 | -0.0024 | 0.0019 | 0.0017 |
| DSV3 + AdamW gate + router LR ×0.19 | 0.0056 | 0.0024 | · | · |
| DSV3 + router LR ×0.19 | 0.0082 | 0.0050 | · | · |
| DSV3 去掉 seq aux | -0.0000 | · | · | · |
| DSV3 + MuonH gate | 0.0036 | · | · | · |

**〔指标：val_v2〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3 | 0.0011 | -0.0023 | 0.0005 | 0.0003 |
| DSV3 + AdamW gate + router LR ×0.19 | 0.0040 | 0.0015 | · | · |
| DSV3 + router LR ×0.19 | 0.0050 | 0.0029 | · | · |
| DSV3 去掉 seq aux | -0.0004 | · | · | · |
| DSV3 + MuonH gate | 0.0009 | · | · | · |

**〔指标：uncheatable〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3 | 0.0020 | -0.0040 | 0.0018 | 0.0031 |
| DSV3 + AdamW gate + router LR ×0.19 | 0.0045 | 0.0003 | · | · |
| DSV3 + router LR ×0.19 | 0.0073 | 0.0039 | · | · |
| DSV3 去掉 seq aux | 0.0007 | · | · | · |
| DSV3 + MuonH gate | 0.0030 | · | · | · |

**〔指标：codefresh〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3 | 0.0061 | 0.0020 | 0.0076 | 0.0057 |
| DSV3 + AdamW gate + router LR ×0.19 | 0.0098 | 0.0093 | · | · |
| DSV3 + router LR ×0.19 | 0.0152 | 0.0148 | · | · |
| DSV3 去掉 seq aux | 0.0017 | · | · | · |
| DSV3 + MuonH gate | 0.0107 | · | · | · |

**〔指标：private_internal〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3 | 0.0052 | -0.0023 | 0.0019 | 0.0014 |
| DSV3 + AdamW gate + router LR ×0.19 | 0.0075 | 0.0030 | · | · |
| DSV3 + router LR ×0.19 | 0.0095 | 0.0032 | · | · |
| DSV3 去掉 seq aux | 0.0017 | · | · | · |
| DSV3 + MuonH gate | 0.0055 | · | · | · |

LR 检查：换成 DSV3 router 后现用 LR 还是不是最优（q0，overall_tsqw；只扫整体 LR，不是新实验组）。格子读法同 LR 网格：大数字 = 抛物线顶点 m*，鼠标移到格子上看曲线；纵轴 = 相对现用 LR（3 个 seed 平均）的 loss 差

| LR 检查（q0） | LR 扫描 | 读法 | 结论 |
|---|---|---|---|
| S：MuonH + DSV3 router | A 的 LR ×0.7 / ×1 / ×1.4（横轴 = ArchScale 原版的倍数） · 1.28× · η* = 0.0172 (1× = 0.0134) · -0.0002 vs 现用 LR（3 seed 平均） · 5 个点（3 个 seed） | 最优 1.28×（ArchScale 原版），现用新公式 = 1.22×：最优点是现用 LR 的 1.05 倍，在最优点 loss 只比现用低 0.0002，现用 LR 接近最优。 | 不用改 LR：换成 DSV3 后，现用新公式仍接近最优（只差 5%，loss 差 0.0002，在噪声内）。只测了 q0，大 rung 没验证。 |
| router LR：MuonH + DSV3 router | 只改 gate 的 LR：×0.19 / ×0.5 / ×1 / ×2（横轴 = 主体 LR 的倍数，×1 = A） · 0.75× · η* = 0.0123 (1× = 0.0164) · -0.0007 vs 现用 LR（3 seed 平均） · 8 个点（3 个 seed） | 最优 router LR ≈ 主体 LR 的 0.75 倍；现用 ×1 时 loss 比最优高 0.0007。×1 和 ×0.19 各 3 个 seed，其余 1 个 seed（Runlong 10-29 加的）。 | router LR 保持 ×1：最优约 ×0.75，×1 只差 0.0007，在噪声内；降到 ×0.19 反而变差。只测了 q0。 |

**LR 扫描点：S：MuonH + DSV3 router / LR 扫描**（格子：A 的 LR ×0.7 / ×1 / ×1.4（横轴 = ArchScale 原版的倍数） · 1.28× · η* = 0.0172 (1× = 0.0134) · -0.0002 vs 现用 LR（3 seed 平均） · 5 个点（3 个 seed））

| 横轴 | overall_tsqw |
|---|---|
| q0 · 1.22356× (η = 0.01640) · seed 3 | +0.0008 vs 现用 LR（3 seed 平均） |
| q0 · 1.22356× (η = 0.01640) · seed 2 | +0.0015 vs 现用 LR（3 seed 平均） |
| q0 · 0.856494× (η = 0.01148) · seed 1 | +0.0112 vs 现用 LR（3 seed 平均） |
| q0 · 1.22356× (η = 0.01640) · seed 1 | -0.0023 vs 现用 LR（3 seed 平均） |
| q0 · 1.71224× (η = 0.02295) · seed 1 | +0.0055 vs 现用 LR（3 seed 平均） |

**LR 扫描点：router LR：MuonH + DSV3 router / LR 扫描**（格子：只改 gate 的 LR：×0.19 / ×0.5 / ×1 / ×2（横轴 = 主体 LR 的倍数，×1 = A） · 0.75× · η* = 0.0123 (1× = 0.0164) · -0.0007 vs 现用 LR（3 seed 平均） · 8 个点（3 个 seed））

| 横轴 | overall_tsqw |
|---|---|
| q0 · 0.19× (η = 0.00312) · seed 3 | +0.0064 vs 现用 LR（3 seed 平均） |
| q0 · 1× (η = 0.01640) · seed 3 | +0.0008 vs 现用 LR（3 seed 平均） |
| q0 · 0.19× (η = 0.00312) · seed 2 | +0.0048 vs 现用 LR（3 seed 平均） |
| q0 · 1× (η = 0.01640) · seed 2 | +0.0015 vs 现用 LR（3 seed 平均） |
| q0 · 0.19× (η = 0.00312) · seed 1 | +0.0040 vs 现用 LR（3 seed 平均） |
| q0 · 0.5× (η = 0.00820) · seed 1 | -0.0015 vs 现用 LR（3 seed 平均） |
| q0 · 1× (η = 0.01640) · seed 1 | -0.0023 vs 现用 LR（3 seed 平均） |
| q0 · 2× (η = 0.03280) · seed 1 | +0.0016 vs 现用 LR（3 seed 平均） |

#### router 健康统计

健康状态取自 Boxiang 的 router 监测（DECISIONS 10-26 19:1x–19:2x）：DSV3 在 q1–q3 上 expert bias 涨到 3–5（预设上限 1）、logit 失控到 60–87，负载均衡全靠 bias 撑——按预设规则不能直接采用；DSV3 + AdamW gate + router LR ×0.19 约 10% 的选中 expert 分数卡在 1 附近、bias 超过 1、logit 持续增大；DSV3 + router LR ×0.19 健康。两种病态原因不同：DSV3 的 gate 在 router LR ×1 下自己学不会均衡，bias 越涨越大，router LR 降到 ×0.19 就健康了；两个 ×0.19 的组 router LR 相同，只有用 AdamW gate 的那组把选中的 expert 分数卡在 1，所以“卡死”来自 AdamW gate 本身（DECISIONS 10-26 17:29）。

来自 Boxiang 的 router 监测 `router_health.py`（每 100 步一次）的汇总行，覆盖所有层。每个 rung 一张图，横轴 = 训练步数，每组一条线；上方按钮切换指标，鼠标悬停看各组数值。MUONH_FIT 的 q2 / q3 没有开监测，所以没有这条线。

**〔视图：MaxVio（层平均）〕**

| 横轴 | softmax（MUONH_FIT） | DSV3 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 | DSV3 去掉 seq aux | DSV3 + MuonH gate |
|---|---|---|---|---|---|---|
| q0 · step ~187 | 1.42 | 2.37 | 2.08 | 1.5 | 4.1 | 1.7 |
| q0 · step ~562 | 1.15 | 1.02 | 1.01 | 0.29 | 1.34 | 0.816 |
| q0 · step ~937 | 1.65 | 0.959 | 0.93 | 0.267 | 1.69 | 1.19 |
| q0 · step ~1,312 | 1.57 | 1.02 | 1.15 | 0.273 | 1.22 | 0.761 |
| q0 · step ~1,687 | 1.59 | 1.04 | 0.975 | 0.277 | 1.29 | 0.85 |
| q0 · step ~2,062 | 1.57 | 1.01 | 0.948 | 0.227 | 1.14 | 0.406 |
| q0 · step ~2,625 | 1.68 | 0.618 | 0.957 | 0.242 | 0.816 | 0.269 |
| q0 · step ~3,000 | 1.57 | 0.516 | 0.7 | 0.193 | 1.07 | 0.334 |
| q0 · step ~3,375 | 1.62 | 0.617 | 0.634 | 0.209 | 0.905 | 0.201 |
| q0 · step ~3,750 | 1.72 | 0.307 | 0.715 | 0.15 | 0.897 | 0.178 |
| q0 · step ~4,125 | 1.55 | 0.262 | 0.642 | 0.185 | 0.742 | 0.194 |
| q0 · step ~4,500 | 1.61 | 0.172 | 0.684 | 0.134 | 0.65 | 0.136 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | softmax（MUONH_FIT） | DSV3 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 |
|---|---|---|---|---|
| q1 · step ~608 | 0.831 | 0.932 | 0.955 | 0.29 |
| q1 · step ~1,825 | 1.27 | 0.851 | 0.86 | 0.236 |
| q1 · step ~3,041 | 1.38 | 0.79 | 0.949 | 0.174 |
| q1 · step ~4,258 | 1.34 | 0.885 | 1.17 | 0.211 |
| q1 · step ~5,475 | 1.46 | 0.67 | 0.92 | 0.231 |
| q1 · step ~6,691 | 1.44 | 0.597 | 0.914 | 0.164 |
| q1 · step ~8,516 | 1.36 | 0.625 | 0.696 | 0.195 |
| q1 · step ~9,733 | 1.23 | 0.565 | 0.751 | 0.202 |
| q1 · step ~10,950 | 1.18 | 0.408 | 0.78 | 0.18 |
| q1 · step ~12,166 | 1.1 | 0.289 | 0.765 | 0.165 |
| q1 · step ~13,383 | 1.07 | 0.204 | 0.631 | 0.136 |
| q1 · step ~14,600 | 1.11 | 0.175 | 0.68 | 0.155 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 |
|---|---|
| q2 · step ~1,412 | 0.632 |
| q2 · step ~4,237 | 0.661 |
| q2 · step ~7,062 | 0.767 |
| q2 · step ~9,887 | 0.686 |
| q2 · step ~12,712 | 0.679 |
| q2 · step ~15,537 | 0.621 |
| q2 · step ~19,775 | 0.46 |
| q2 · step ~22,600 | 0.411 |
| q2 · step ~25,425 | 0.363 |
| q2 · step ~28,250 | 0.303 |
| q2 · step ~31,075 | 0.193 |
| q2 · step ~33,900 | 0.171 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 |
|---|---|
| q3 · step ~2,712 | 0.596 |
| q3 · step ~8,137 | 0.541 |
| q3 · step ~13,562 | 0.61 |
| q3 · step ~18,987 | 0.576 |
| q3 · step ~24,412 | 0.534 |
| q3 · step ~29,837 | 0.493 |
| q3 · step ~37,975 | 0.527 |
| q3 · step ~43,400 | 0.456 |
| q3 · step ~48,825 | 0.424 |
| q3 · step ~54,250 | 0.312 |
| q3 · step ~59,675 | 0.232 |
| q3 · step ~65,100 | 0.153 |

（共 24 个点，均匀取 12 个，含首尾）

每层：一整个训练 step（所有 micro-batch、所有 rank）里分到 token 最多的 expert ÷ 平均值 − 1（DeepSeek-V3 的 MaxVio，0 = 完全均衡）；这里是所有层的平均。

**〔视图：MaxVio（最差层）〕**

| 横轴 | softmax（MUONH_FIT） | DSV3 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 | DSV3 去掉 seq aux | DSV3 + MuonH gate |
|---|---|---|---|---|---|---|
| q0 · step ~187 | 2.37 | 3.75 | 4.74 | 1.97 | 7.61 | 3.21 |
| q0 · step ~562 | 2.43 | 6.09 | 5.61 | 0.592 | 6.88 | 5.41 |
| q0 · step ~937 | 3.67 | 4.39 | 5.17 | 0.684 | 10.7 | 6.22 |
| q0 · step ~1,312 | 4.2 | 5.95 | 5.88 | 0.768 | 7.7 | 5.54 |
| q0 · step ~1,687 | 4.3 | 6.04 | 4.88 | 0.624 | 8.64 | 6.83 |
| q0 · step ~2,062 | 4.36 | 5.47 | 4.83 | 0.744 | 7.18 | 1.96 |
| q0 · step ~2,625 | 4.47 | 3.55 | 3.39 | 0.87 | 4.81 | 0.422 |
| q0 · step ~3,000 | 4.45 | 2.48 | 2.59 | 0.306 | 7.48 | 1.61 |
| q0 · step ~3,375 | 4.26 | 3.45 | 2.27 | 0.417 | 5.98 | 0.311 |
| q0 · step ~3,750 | 4.19 | 0.99 | 2.58 | 0.247 | 5.95 | 0.225 |
| q0 · step ~4,125 | 3.88 | 0.886 | 2.39 | 0.506 | 5.07 | 0.311 |
| q0 · step ~4,500 | 3.94 | 0.687 | 2.68 | 0.481 | 4.19 | 0.338 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | softmax（MUONH_FIT） | DSV3 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 |
|---|---|---|---|---|
| q1 · step ~608 | 1.37 | 6.48 | 5.75 | 0.517 |
| q1 · step ~1,825 | 4.12 | 4.78 | 3.91 | 0.393 |
| q1 · step ~3,041 | 5.08 | 5.6 | 2.8 | 0.271 |
| q1 · step ~4,258 | 5.31 | 6.56 | 3.7 | 0.58 |
| q1 · step ~5,475 | 5.63 | 3.69 | 2.3 | 0.364 |
| q1 · step ~6,691 | 5.27 | 3.42 | 3.99 | 0.477 |
| q1 · step ~8,516 | 4.97 | 4.01 | 2.37 | 0.386 |
| q1 · step ~9,733 | 4.29 | 3.07 | 3.85 | 0.297 |
| q1 · step ~10,950 | 4.14 | 1.98 | 2.54 | 0.272 |
| q1 · step ~12,166 | 3.66 | 0.77 | 3.77 | 0.216 |
| q1 · step ~13,383 | 3.35 | 0.302 | 2.43 | 0.184 |
| q1 · step ~14,600 | 3.3 | 0.42 | 3.57 | 0.269 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 |
|---|---|
| q2 · step ~1,412 | 4.49 |
| q2 · step ~4,237 | 5.55 |
| q2 · step ~7,062 | 4.51 |
| q2 · step ~9,887 | 4.84 |
| q2 · step ~12,712 | 3.66 |
| q2 · step ~15,537 | 3.76 |
| q2 · step ~19,775 | 2.32 |
| q2 · step ~22,600 | 1.46 |
| q2 · step ~25,425 | 1.47 |
| q2 · step ~28,250 | 0.593 |
| q2 · step ~31,075 | 0.376 |
| q2 · step ~33,900 | 0.445 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 |
|---|---|
| q3 · step ~2,712 | 4.72 |
| q3 · step ~8,137 | 4.21 |
| q3 · step ~13,562 | 4.43 |
| q3 · step ~18,987 | 3.75 |
| q3 · step ~24,412 | 2.09 |
| q3 · step ~29,837 | 1.67 |
| q3 · step ~37,975 | 2.41 |
| q3 · step ~43,400 | 1.63 |
| q3 · step ~48,825 | 2.27 |
| q3 · step ~54,250 | 0.879 |
| q3 · step ~59,675 | 0.446 |
| q3 · step ~65,100 | 0.307 |

（共 24 个点，均匀取 12 个，含首尾）

同上，取最不均衡的那一层。

**〔视图：dead expert (%)〕**

| 横轴 | softmax（MUONH_FIT） | DSV3 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 | DSV3 去掉 seq aux | DSV3 + MuonH gate |
|---|---|---|---|---|---|---|
| q0 · step ~187 | 1.04% | 4.3% | 3.39% | 0.26% | 17.4% | 1.82% |
| q0 · step ~562 | 0% | 0.846% | 0.846% | 0% | 1.56% | 0% |
| q0 · step ~937 | 0.0651% | 0.521% | 0.13% | 0% | 2.02% | 0.26% |
| q0 · step ~1,312 | 0.0651% | 0.326% | 0.195% | 0% | 2.41% | 0.13% |
| q0 · step ~1,687 | 0.13% | 0.326% | 0.391% | 0% | 2.86% | 0.26% |
| q0 · step ~2,062 | 0.0651% | 0.26% | 0.0651% | 0% | 1.69% | 0.0651% |
| q0 · step ~2,625 | 0.0651% | 0.195% | 0.13% | 0% | 1.04% | 0.0651% |
| q0 · step ~3,000 | 0.0651% | 0.26% | 0.13% | 0% | 1.76% | 0% |
| q0 · step ~3,375 | 0.0651% | 0% | 0.0651% | 0% | 1.17% | 0% |
| q0 · step ~3,750 | 0.0651% | 0% | 0.195% | 0% | 1.17% | 0% |
| q0 · step ~4,125 | 0.0651% | 0% | 0.13% | 0% | 1.11% | 0% |
| q0 · step ~4,500 | 0.0651% | 0% | 0.13% | 0% | 0.846% | 0% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | softmax（MUONH_FIT） | DSV3 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 |
|---|---|---|---|---|
| q1 · step ~608 | 0% | 0.26% | 0.391% | 0% |
| q1 · step ~1,825 | 0.0434% | 0.174% | 0.13% | 0% |
| q1 · step ~3,041 | 0.0434% | 0.174% | 0% | 0% |
| q1 · step ~4,258 | 0.0868% | 0.13% | 0.0434% | 0% |
| q1 · step ~5,475 | 0.0868% | 0.0868% | 0.217% | 0% |
| q1 · step ~6,691 | 0.0434% | 0.0868% | 0.13% | 0% |
| q1 · step ~8,516 | 0.0434% | 0.0868% | 0.0868% | 0% |
| q1 · step ~9,733 | 0.0434% | 0.0434% | 0.0868% | 0% |
| q1 · step ~10,950 | 0.0434% | 0% | 0.0868% | 0% |
| q1 · step ~12,166 | 0.0434% | 0% | 0.174% | 0% |
| q1 · step ~13,383 | 0.0434% | 0% | 0.13% | 0% |
| q1 · step ~14,600 | 0.0434% | 0% | 0.174% | 0% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 |
|---|---|
| q2 · step ~1,412 | 0.0977% |
| q2 · step ~4,237 | 0.163% |
| q2 · step ~7,062 | 0.13% |
| q2 · step ~9,887 | 0.0326% |
| q2 · step ~12,712 | 0.0651% |
| q2 · step ~15,537 | 0% |
| q2 · step ~19,775 | 0% |
| q2 · step ~22,600 | 0% |
| q2 · step ~25,425 | 0% |
| q2 · step ~28,250 | 0% |
| q2 · step ~31,075 | 0% |
| q2 · step ~33,900 | 0% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 |
|---|---|
| q3 · step ~2,712 | 0.104% |
| q3 · step ~8,137 | 0.13% |
| q3 · step ~13,562 | 0.0521% |
| q3 · step ~18,987 | 0.026% |
| q3 · step ~24,412 | 0% |
| q3 · step ~29,837 | 0.026% |
| q3 · step ~37,975 | 0% |
| q3 · step ~43,400 | 0% |
| q3 · step ~48,825 | 0% |
| q3 · step ~54,250 | 0% |
| q3 · step ~59,675 | 0% |
| q3 · step ~65,100 | 0% |

（共 24 个点，均匀取 12 个，含首尾）

分到的 token 不到平均值 10% 的 expert 数，占（层数 × 128）的百分比。

**〔视图：|bias| 最大〕**

| 横轴 | DSV3 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 | DSV3 去掉 seq aux | DSV3 + MuonH gate |
|---|---|---|---|---|---|
| q0 · step ~187 | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| q0 · step ~562 | 0.214 | 0.2 | 0.222 | 0.26 | 0.332 |
| q0 · step ~937 | 0.296 | 0.254 | 0.218 | 0.292 | 0.288 |
| q0 · step ~1,312 | 0.376 | 0.34 | 0.212 | 0.452 | 0.322 |
| q0 · step ~1,687 | 0.456 | 0.408 | 0.208 | 0.58 | 0.368 |
| q0 · step ~2,062 | 0.568 | 0.52 | 0.218 | 0.764 | 0.432 |
| q0 · step ~2,625 | 0.714 | 0.684 | 0.239 | 0.99 | 0.502 |
| q0 · step ~3,000 | 0.794 | 0.754 | 0.251 | 1.1 | 0.514 |
| q0 · step ~3,375 | 0.872 | 0.832 | 0.27 | 1.25 | 0.54 |
| q0 · step ~3,750 | 0.92 | 0.908 | 0.286 | 1.4 | 0.544 |
| q0 · step ~4,125 | 0.952 | 0.992 | 0.3 | 1.55 | 0.556 |
| q0 · step ~4,500 | 0.99 | 1.1 | 0.312 | 1.7 | 0.568 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 |
|---|---|---|---|
| q1 · step ~608 | 0.222 | 0.24 | 0.214 |
| q1 · step ~1,825 | 0.561 | 0.508 | 0.146 |
| q1 · step ~3,041 | 0.946 | 0.78 | 0.195 |
| q1 · step ~4,258 | 1.32 | 1 | 0.253 |
| q1 · step ~5,475 | 1.69 | 1.18 | 0.299 |
| q1 · step ~6,691 | 2.04 | 1.34 | 0.346 |
| q1 · step ~8,516 | 2.53 | 1.72 | 0.402 |
| q1 · step ~9,733 | 2.8 | 2.04 | 0.436 |
| q1 · step ~10,950 | 3.01 | 2.38 | 0.47 |
| q1 · step ~12,166 | 3.11 | 2.72 | 0.502 |
| q1 · step ~13,383 | 3.16 | 3.1 | 0.528 |
| q1 · step ~14,600 | 3.22 | 3.54 | 0.554 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 |
|---|---|
| q2 · step ~1,412 | 0.432 |
| q2 · step ~4,237 | 1.27 |
| q2 · step ~7,062 | 2.12 |
| q2 · step ~9,887 | 2.92 |
| q2 · step ~12,712 | 3.66 |
| q2 · step ~15,537 | 4.27 |
| q2 · step ~19,775 | 4.97 |
| q2 · step ~22,600 | 5.3 |
| q2 · step ~25,425 | 5.54 |
| q2 · step ~28,250 | 5.67 |
| q2 · step ~31,075 | 5.75 |
| q2 · step ~33,900 | 5.87 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 |
|---|---|
| q3 · step ~2,712 | 0.79 |
| q3 · step ~8,137 | 2.52 |
| q3 · step ~13,562 | 4.08 |
| q3 · step ~18,987 | 5.58 |
| q3 · step ~24,412 | 6.92 |
| q3 · step ~29,837 | 8.06 |
| q3 · step ~37,975 | 9.54 |
| q3 · step ~43,400 | 10.4 |
| q3 · step ~48,825 | 11.1 |
| q3 · step ~54,250 | 11.6 |
| q3 · step ~59,675 | 11.9 |
| q3 · step ~65,100 | 12 |

（共 24 个点，均匀取 12 个，含首尾）

DSV3 每个 expert 的 bias 在所有层里的最大绝对值；红色虚线 = Boxiang 预设的健康上限 1。注意 top-8 选择只看同一层 bias 之间的差，所有 expert 一起抬高并不改变路由（例如 A 的 q3 第 0 层 bias 在 +11.2 到 +11.7 之间）。softmax 没有 bias。

**〔视图：logit 饱和 (%)〕**

| 横轴 | softmax（MUONH_FIT） | DSV3 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 | DSV3 去掉 seq aux | DSV3 + MuonH gate |
|---|---|---|---|---|---|---|
| q0 · step ~187 | 0.19% | 0.12% | 15.8% | 0% | 0.13% | 0.03% |
| q0 · step ~562 | 2.97% | 41.2% | 77.1% | 19.2% | 48.4% | 11% |
| q0 · step ~937 | 11.1% | 40.9% | 79.2% | 33% | 44.8% | 19.9% |
| q0 · step ~1,312 | 17% | 41.9% | 80.3% | 37.7% | 51.3% | 23.6% |
| q0 · step ~1,687 | 23.1% | 45.1% | 80.8% | 44.1% | 54.8% | 28.2% |
| q0 · step ~2,062 | 28.3% | 49.7% | 80.8% | 49.9% | 57.1% | 31.9% |
| q0 · step ~2,625 | 26.3% | 46.8% | 77.9% | 49% | 58.2% | 33.8% |
| q0 · step ~3,000 | 30.7% | 50.2% | 78.3% | 52.6% | 59.1% | 37.2% |
| q0 · step ~3,375 | 29.8% | 47.4% | 76.2% | 50.4% | 60.7% | 36.7% |
| q0 · step ~3,750 | 41.6% | 48.8% | 75% | 51.6% | 59% | 36.3% |
| q0 · step ~4,125 | 34.7% | 51.9% | 75.9% | 54.6% | 61.4% | 40.5% |
| q0 · step ~4,500 | 36.8% | 48.9% | 73.3% | 52.6% | 63.1% | 38.6% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | softmax（MUONH_FIT） | DSV3 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 |
|---|---|---|---|---|
| q1 · step ~608 | 3.08% | 48.2% | 74.8% | 27.6% |
| q1 · step ~1,825 | 42.8% | 57.4% | 83.3% | 56.1% |
| q1 · step ~3,041 | 56.8% | 56.6% | 83.1% | 58.3% |
| q1 · step ~4,258 | 42.8% | 54.2% | 81.6% | 56.1% |
| q1 · step ~5,475 | 41.9% | 55.2% | 81.1% | 58.9% |
| q1 · step ~6,691 | 34.9% | 57.8% | 81.9% | 61% |
| q1 · step ~8,516 | 35.9% | 57.1% | 79.1% | 61.4% |
| q1 · step ~9,733 | 36.3% | 47.8% | 72.3% | 52.8% |
| q1 · step ~10,950 | 36.8% | 49.5% | 71.5% | 55% |
| q1 · step ~12,166 | 39.4% | 53% | 72.2% | 59.2% |
| q1 · step ~13,383 | 37.1% | 52.4% | 69.4% | 59.1% |
| q1 · step ~14,600 | 44.1% | 50.3% | 66.2% | 58.3% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 |
|---|---|
| q2 · step ~1,412 | 61.8% |
| q2 · step ~4,237 | 59.2% |
| q2 · step ~7,062 | 61.4% |
| q2 · step ~9,887 | 62.8% |
| q2 · step ~12,712 | 61.9% |
| q2 · step ~15,537 | 57.4% |
| q2 · step ~19,775 | 54.4% |
| q2 · step ~22,600 | 52% |
| q2 · step ~25,425 | 46.8% |
| q2 · step ~28,250 | 40.7% |
| q2 · step ~31,075 | 42.5% |
| q2 · step ~33,900 | 42.8% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 |
|---|---|
| q3 · step ~2,712 | 65.1% |
| q3 · step ~8,137 | 67.6% |
| q3 · step ~13,562 | 65% |
| q3 · step ~18,987 | 60.2% |
| q3 · step ~24,412 | 54.5% |
| q3 · step ~29,837 | 56.2% |
| q3 · step ~37,975 | 44.2% |
| q3 · step ~43,400 | 51.6% |
| q3 · step ~48,825 | 41.5% |
| q3 · step ~54,250 | 37.6% |
| q3 · step ~59,675 | 32.7% |
| q3 · step ~65,100 | 36.8% |

（共 24 个点，均匀取 12 个，含首尾）

router logit |z| > 4 的比例（各层平均；只看 rank 0 的第一个 micro-batch）。包括 z < −4（明确拒绝某个 expert，正常）。

**〔视图：z > +4 (%)〕**

| 横轴 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 | DSV3 去掉 seq aux | DSV3 + MuonH gate |
|---|---|---|---|---|
| q0 · step ~187 | 13.5% | 0% | 0.02% | 0.03% |
| q0 · step ~562 | 0.37% | 0% | 0.01% | 0.05% |
| q0 · step ~937 | 0.58% | 0.01% | 0.02% | 0.05% |
| q0 · step ~1,312 | 0.6% | 0.01% | 0.03% | 0.07% |
| q0 · step ~1,687 | 0.64% | 0.02% | 0.04% | 0.07% |
| q0 · step ~2,062 | 0.51% | 0.01% | 0.03% | 0.05% |
| q0 · step ~2,625 | 0.66% | 0.03% | 0.07% | 0.09% |
| q0 · step ~3,000 | 0.48% | 0.03% | 0.06% | 0.08% |
| q0 · step ~3,375 | 0.64% | 0.04% | 0.08% | 0.11% |
| q0 · step ~3,750 | 0.5% | 0.03% | 0.06% | 0.08% |
| q0 · step ~4,125 | 0.45% | 0.04% | 0.07% | 0.09% |
| q0 · step ~4,500 | 0.55% | 0.05% | 0.1% | 0.12% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 |
|---|---|---|
| q1 · step ~608 | 0.57% | 0.01% |
| q1 · step ~1,825 | 0.79% | 0.04% |
| q1 · step ~3,041 | 1.02% | 0.08% |
| q1 · step ~4,258 | 1.18% | 0.12% |
| q1 · step ~5,475 | 1.17% | 0.12% |
| q1 · step ~6,691 | 1.14% | 0.13% |
| q1 · step ~8,516 | 1.03% | 0.15% |
| q1 · step ~9,733 | 1.16% | 0.19% |
| q1 · step ~10,950 | 0.97% | 0.16% |
| q1 · step ~12,166 | 0.84% | 0.15% |
| q1 · step ~13,383 | 0.77% | 0.17% |
| q1 · step ~14,600 | 0.68% | 0.14% |

（共 24 个点，均匀取 12 个，含首尾）

z > +4 的比例：分数被钉在 1，选中的 expert 之间失去区分度（坏）；红色虚线 = 预设上限 5%。只有后期日志才有。

**〔视图：top-8 pinned (%)〕**

| 横轴 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 | DSV3 去掉 seq aux | DSV3 + MuonH gate |
|---|---|---|---|---|
| q0 · step ~187 | 59.7% | 0% | 0.38% | 0.46% |
| q0 · step ~562 | 5.92% | 0.03% | 0.08% | 0.78% |
| q0 · step ~937 | 9.27% | 0.15% | 0.35% | 0.78% |
| q0 · step ~1,312 | 9.54% | 0.22% | 0.48% | 1.19% |
| q0 · step ~1,687 | 10.3% | 0.29% | 0.71% | 1.09% |
| q0 · step ~2,062 | 8.22% | 0.21% | 0.51% | 0.87% |
| q0 · step ~2,625 | 10.6% | 0.49% | 1.07% | 1.49% |
| q0 · step ~3,000 | 7.67% | 0.55% | 0.89% | 1.32% |
| q0 · step ~3,375 | 10.2% | 0.58% | 1.32% | 1.73% |
| q0 · step ~3,750 | 7.99% | 0.46% | 0.92% | 1.25% |
| q0 · step ~4,125 | 7.21% | 0.56% | 1.15% | 1.48% |
| q0 · step ~4,500 | 8.87% | 0.77% | 1.53% | 1.91% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 |
|---|---|---|
| q1 · step ~608 | 9.06% | 0.11% |
| q1 · step ~1,825 | 12.6% | 0.62% |
| q1 · step ~3,041 | 16.3% | 1.21% |
| q1 · step ~4,258 | 18.9% | 1.93% |
| q1 · step ~5,475 | 18.7% | 1.85% |
| q1 · step ~6,691 | 18.2% | 2.02% |
| q1 · step ~8,516 | 16.5% | 2.42% |
| q1 · step ~9,733 | 18.5% | 2.99% |
| q1 · step ~10,950 | 15.5% | 2.55% |
| q1 · step ~12,166 | 13.4% | 2.43% |
| q1 · step ~13,383 | 12.3% | 2.73% |
| q1 · step ~14,600 | 10.9% | 2.19% |

（共 24 个点，均匀取 12 个，含首尾）

选中的 8 个 expert 里分数卡在 1 附近的比例；红色虚线 = 预设上限 5%。只有后期日志才有。

**〔视图：最大 |logit|〕**

| 横轴 | softmax（MUONH_FIT） | DSV3 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 | DSV3 去掉 seq aux | DSV3 + MuonH gate |
|---|---|---|---|---|---|---|
| q0 · step ~187 | 7.86 | 6.99 | 16 | 4.47 | 6.25 | 6.21 |
| q0 · step ~562 | 12.8 | 12.4 | 27.9 | 9.55 | 12.3 | 10.5 |
| q0 · step ~937 | 14.3 | 12.7 | 34 | 10.5 | 13.3 | 20 |
| q0 · step ~1,312 | 15.3 | 14.2 | 36.5 | 12.6 | 14.6 | 15.3 |
| q0 · step ~1,687 | 16.6 | 19.9 | 37.7 | 14.3 | 15.6 | 27.9 |
| q0 · step ~2,062 | 15.9 | 20.6 | 40.4 | 14.1 | 15.4 | 14.9 |
| q0 · step ~2,625 | 22.2 | 21 | 48.3 | 13.9 | 19.3 | 14.8 |
| q0 · step ~3,000 | 17.1 | 23.4 | 47.1 | 14.3 | 19 | 21.9 |
| q0 · step ~3,375 | 19 | 25.2 | 47.8 | 14.8 | 22.2 | 17.6 |
| q0 · step ~3,750 | 23.7 | 25.4 | 46.7 | 14.6 | 21.6 | 16.9 |
| q0 · step ~4,125 | 21.4 | 28.8 | 48.4 | 14.9 | 20.6 | 18 |
| q0 · step ~4,500 | 20.2 | 26.3 | 53.6 | 15.2 | 25.9 | 18.6 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | softmax（MUONH_FIT） | DSV3 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 |
|---|---|---|---|---|
| q1 · step ~608 | 13.5 | 16.4 | 30.6 | 10.4 |
| q1 · step ~1,825 | 17.7 | 20.6 | 47.5 | 13.8 |
| q1 · step ~3,041 | 17.9 | 25.5 | 68.8 | 16.9 |
| q1 · step ~4,258 | 19.7 | 32.2 | 96.6 | 19.8 |
| q1 · step ~5,475 | 20.6 | 49 | 119 | 23.6 |
| q1 · step ~6,691 | 26.2 | 56.9 | 127 | 23.1 |
| q1 · step ~8,516 | 26.2 | 73.1 | 131 | 30.1 |
| q1 · step ~9,733 | 30.4 | 89.1 | 139 | 28.9 |
| q1 · step ~10,950 | 31.3 | 89.8 | 146 | 30.5 |
| q1 · step ~12,166 | 36.8 | 92.4 | 152 | 28.8 |
| q1 · step ~13,383 | 40.8 | 100 | 138 | 32.9 |
| q1 · step ~14,600 | 35.3 | 99.9 | 153 | 29 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 |
|---|---|
| q2 · step ~1,412 | 19.1 |
| q2 · step ~4,237 | 28.2 |
| q2 · step ~7,062 | 38.1 |
| q2 · step ~9,887 | 49.5 |
| q2 · step ~12,712 | 64.8 |
| q2 · step ~15,537 | 78.4 |
| q2 · step ~19,775 | 91.3 |
| q2 · step ~22,600 | 103 |
| q2 · step ~25,425 | 120 |
| q2 · step ~28,250 | 123 |
| q2 · step ~31,075 | 132 |
| q2 · step ~33,900 | 146 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 |
|---|---|
| q3 · step ~2,712 | 25.7 |
| q3 · step ~8,137 | 46.3 |
| q3 · step ~13,562 | 59.5 |
| q3 · step ~18,987 | 80.1 |
| q3 · step ~24,412 | 115 |
| q3 · step ~29,837 | 114 |
| q3 · step ~37,975 | 145 |
| q3 · step ~43,400 | 149 |
| q3 · step ~48,825 | 156 |
| q3 · step ~54,250 | 179 |
| q3 · step ~59,675 | 187 |
| q3 · step ~65,100 | 184 |

（共 24 个点，均匀取 12 个，含首尾）

所有层里最大的 |logit|（rank 0 的第一个 micro-batch）。

**〔视图：gate 更新幅度〕**

| 横轴 | softmax（MUONH_FIT） | DSV3 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 | DSV3 去掉 seq aux | DSV3 + MuonH gate |
|---|---|---|---|---|---|---|
| q0 · step ~187 | 0.016 | 0.0161 | 0.0171 | 0.0037 | 0.0159 | 0.0159 |
| q0 · step ~562 | 0.0148 | 0.0145 | 0.0106 | 0.0035 | 0.0145 | 0.0145 |
| q0 · step ~937 | 0.0134 | 0.0132 | 0.0094 | 0.0033 | 0.0131 | 0.0132 |
| q0 · step ~1,312 | 0.012 | 0.0119 | 0.0089 | 0.0031 | 0.0119 | 0.0119 |
| q0 · step ~1,687 | 0.011 | 0.0109 | 0.0082 | 0.0029 | 0.0108 | 0.0109 |
| q0 · step ~2,062 | 0.0096 | 0.0096 | 0.0076 | 0.0026 | 0.0095 | 0.0095 |
| q0 · step ~2,625 | 0.0076 | 0.0075 | 0.0065 | 0.0023 | 0.0075 | 0.0075 |
| q0 · step ~3,000 | 0.0065 | 0.0065 | 0.0059 | 0.0021 | 0.0065 | 0.0065 |
| q0 · step ~3,375 | 0.0052 | 0.0052 | 0.005 | 0.0018 | 0.0052 | 0.0052 |
| q0 · step ~3,750 | 0.004 | 0.004 | 0.004 | 0.0015 | 0.004 | 0.004 |
| q0 · step ~4,125 | 0.0028 | 0.0027 | 0.0028 | 0.0011 | 0.0027 | 0.0027 |
| q0 · step ~4,500 | 0.0013 | 0.0013 | 0.0013 | 0.0006 | 0.0013 | 0.0013 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | softmax（MUONH_FIT） | DSV3 | DSV3 + AdamW gate + router LR ×0.19 | DSV3 + router LR ×0.19 |
|---|---|---|---|---|
| q1 · step ~608 | 0.0111 | 0.011 | 0.0103 | 0.0029 |
| q1 · step ~1,825 | 0.0102 | 0.0102 | 0.0088 | 0.0028 |
| q1 · step ~3,041 | 0.0093 | 0.0093 | 0.0081 | 0.0026 |
| q1 · step ~4,258 | 0.0084 | 0.0084 | 0.0075 | 0.0024 |
| q1 · step ~5,475 | 0.0076 | 0.0075 | 0.007 | 0.0023 |
| q1 · step ~6,691 | 0.0067 | 0.0067 | 0.0064 | 0.0021 |
| q1 · step ~8,516 | 0.0053 | 0.0053 | 0.0055 | 0.0018 |
| q1 · step ~9,733 | 0.0045 | 0.0045 | 0.0049 | 0.0016 |
| q1 · step ~10,950 | 0.0037 | 0.0037 | 0.0041 | 0.0014 |
| q1 · step ~12,166 | 0.003 | 0.0029 | 0.0033 | 0.0012 |
| q1 · step ~13,383 | 0.0021 | 0.0021 | 0.0023 | 0.0009 |
| q1 · step ~14,600 | 0.0008 | 0.0008 | 0.0009 | 0.0004 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 |
|---|---|
| q2 · step ~1,412 | 0.0086 |
| q2 · step ~4,237 | 0.0079 |
| q2 · step ~7,062 | 0.0073 |
| q2 · step ~9,887 | 0.0066 |
| q2 · step ~12,712 | 0.0059 |
| q2 · step ~15,537 | 0.0052 |
| q2 · step ~19,775 | 0.0043 |
| q2 · step ~22,600 | 0.0036 |
| q2 · step ~25,425 | 0.0029 |
| q2 · step ~28,250 | 0.0023 |
| q2 · step ~31,075 | 0.0016 |
| q2 · step ~33,900 | 0.0005 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 |
|---|---|
| q3 · step ~2,712 | 0.0071 |
| q3 · step ~8,137 | 0.0066 |
| q3 · step ~13,562 | 0.006 |
| q3 · step ~18,987 | 0.0055 |
| q3 · step ~24,412 | 0.0049 |
| q3 · step ~29,837 | 0.0044 |
| q3 · step ~37,975 | 0.0036 |
| q3 · step ~43,400 | 0.0031 |
| q3 · step ~48,825 | 0.0025 |
| q3 · step ~54,250 | 0.002 |
| q3 · step ~59,675 | 0.0014 |
| q3 · step ~65,100 | 0.0005 |

（共 24 个点，均匀取 12 个，含首尾）

router 权重这一步的变化量 ÷ 权重大小，各层平均（Boxiang 只作参考，不作判据）。

### Router 负熵均衡 (phi balancing)

负熵均衡损失：每层对 router 概率做 EMA（η 0.65），得到每个 expert 长期分到的平均概率 m_e（在整个 global batch 上统计，每个 step 更新一次）；损失 = α·E·Σ_e p_e·(log m_e + 1)，p_e = 当前这批 token 给 expert e 的平均概率，括号里那项不传梯度，梯度只走 p_e。效果：长期被用得多的 expert（m_e 大）概率被往下压，用得少的被往上拉，把平均分配推向均匀（熵最大）。它替换原来的负载均衡损失（DSV3 底座的 seq aux 1e-4、softmax 底座的 Switch aux 1e-3）；DSV3 底座的 expert bias 默认保留（标了“去掉 bias”的组除外）。和 aux loss 的区别：统计范围是整个 global batch 再做滑动平均，用的是概率而不是实际分到的 token 数，所以和 micro-batch 大小无关。（源自 arXiv 2605.15403）

#### 全注意力底座：各组 − MUONH_FIT（MuonH，softmax router，CE）

0 线 = **MUONH_FIT**：MuonH，新 LR 公式（fit），纯 CE（不用 KD），128 选 8 MoE，全注意力，原版 softmax router + Switch aux loss 1e-3，gate 用 AdamH、LR ×1。

**〔指标：overall_tsqw〕**

**DSV3 底座 + 负熵 − MUONH_FIT（鼠标移到某个 rung 看数值）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3（参照） | 0.0023 | -0.0024 | 0.0018 | 0.0017 |
| DSV3 + 负熵 α 1e-6 | -0.0001 | · | · | · |
| DSV3 + 负熵 α 1e-5 | 0.0001 | -0.0034 | -0.0021 | · |
| DSV3 + 负熵 α 1e-4 | 0.0008 | -0.0032 | · | · |
| DSV3 + 负熵 α 1e-3 | 0.0129 | · | · | · |
| DSV3 + 负熵 α 1e-5，去掉 bias | -0.0064 | · | · | · |
| DSV3 + 负熵 α 1e-4，去掉 bias | 0.0016 | · | · | · |

**softmax 底座 + 负熵 − MUONH_FIT（鼠标移到某个 rung 看数值）**

| 线 | q0 · 48M |
|---|---|
| softmax + 负熵 α 1e-5 | -0.0005 |
| softmax + 负熵 α 1e-4 | 0.0029 |

**〔指标：overall_tw〕**

**DSV3 底座 + 负熵 − MUONH_FIT（鼠标移到某个 rung 看数值）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3（参照） | 0.0021 | -0.0028 | 0.0019 | 0.0021 |
| DSV3 + 负熵 α 1e-6 | -0.0005 | · | · | · |
| DSV3 + 负熵 α 1e-5 | -0.0000 | -0.0038 | -0.0023 | · |
| DSV3 + 负熵 α 1e-4 | 0.0005 | -0.0033 | · | · |
| DSV3 + 负熵 α 1e-3 | 0.0126 | · | · | · |
| DSV3 + 负熵 α 1e-5，去掉 bias | -0.0067 | · | · | · |
| DSV3 + 负熵 α 1e-4，去掉 bias | 0.0013 | · | · | · |

**softmax 底座 + 负熵 − MUONH_FIT（鼠标移到某个 rung 看数值）**

| 线 | q0 · 48M |
|---|---|
| softmax + 负熵 α 1e-5 | -0.0001 |
| softmax + 负熵 α 1e-4 | 0.0024 |

**〔指标：overall_macro〕**

**DSV3 底座 + 负熵 − MUONH_FIT（鼠标移到某个 rung 看数值）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3（参照） | 0.0032 | -0.0024 | 0.0019 | 0.0017 |
| DSV3 + 负熵 α 1e-6 | 0.0011 | · | · | · |
| DSV3 + 负熵 α 1e-5 | 0.0005 | -0.0034 | -0.0015 | · |
| DSV3 + 负熵 α 1e-4 | 0.0017 | -0.0032 | · | · |
| DSV3 + 负熵 α 1e-3 | 0.0136 | · | · | · |
| DSV3 + 负熵 α 1e-5，去掉 bias | -0.0050 | · | · | · |
| DSV3 + 负熵 α 1e-4，去掉 bias | 0.0021 | · | · | · |

**softmax 底座 + 负熵 − MUONH_FIT（鼠标移到某个 rung 看数值）**

| 线 | q0 · 48M |
|---|---|
| softmax + 负熵 α 1e-5 | -0.0005 |
| softmax + 负熵 α 1e-4 | 0.0038 |

**〔指标：val_v2〕**

**DSV3 底座 + 负熵 − MUONH_FIT（鼠标移到某个 rung 看数值）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3（参照） | 0.0011 | -0.0023 | 0.0005 | 0.0003 |
| DSV3 + 负熵 α 1e-6 | -0.0007 | · | · | · |
| DSV3 + 负熵 α 1e-5 | -0.0009 | -0.0027 | -0.0015 | · |
| DSV3 + 负熵 α 1e-4 | -0.0000 | -0.0025 | · | · |
| DSV3 + 负熵 α 1e-3 | 0.0102 | · | · | · |
| DSV3 + 负熵 α 1e-5，去掉 bias | -0.0059 | · | · | · |
| DSV3 + 负熵 α 1e-4，去掉 bias | 0.0015 | · | · | · |

**softmax 底座 + 负熵 − MUONH_FIT（鼠标移到某个 rung 看数值）**

| 线 | q0 · 48M |
|---|---|
| softmax + 负熵 α 1e-5 | -0.0002 |
| softmax + 负熵 α 1e-4 | 0.0026 |

**〔指标：uncheatable〕**

**DSV3 底座 + 负熵 − MUONH_FIT（鼠标移到某个 rung 看数值）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3（参照） | 0.0020 | -0.0040 | 0.0018 | 0.0031 |
| DSV3 + 负熵 α 1e-6 | -0.0004 | · | · | · |
| DSV3 + 负熵 α 1e-5 | -0.0004 | -0.0050 | -0.0026 | · |
| DSV3 + 负熵 α 1e-4 | 0.0004 | -0.0037 | · | · |
| DSV3 + 负熵 α 1e-3 | 0.0127 | · | · | · |
| DSV3 + 负熵 α 1e-5，去掉 bias | -0.0066 | · | · | · |
| DSV3 + 负熵 α 1e-4，去掉 bias | 0.0015 | · | · | · |

**softmax 底座 + 负熵 − MUONH_FIT（鼠标移到某个 rung 看数值）**

| 线 | q0 · 48M |
|---|---|
| softmax + 负熵 α 1e-5 | 0.0022 |
| softmax + 负熵 α 1e-4 | 0.0011 |

**〔指标：codefresh〕**

**DSV3 底座 + 负熵 − MUONH_FIT（鼠标移到某个 rung 看数值）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3（参照） | 0.0061 | 0.0020 | 0.0076 | 0.0057 |
| DSV3 + 负熵 α 1e-6 | 0.0050 | · | · | · |
| DSV3 + 负熵 α 1e-5 | 0.0039 | -0.0027 | -0.0003 | · |
| DSV3 + 负熵 α 1e-4 | 0.0051 | -0.0039 | · | · |
| DSV3 + 负熵 α 1e-3 | 0.0199 | · | · | · |
| DSV3 + 负熵 α 1e-5，去掉 bias | -0.0091 | · | · | · |
| DSV3 + 负熵 α 1e-4，去掉 bias | 0.0036 | · | · | · |

**softmax 底座 + 负熵 − MUONH_FIT（鼠标移到某个 rung 看数值）**

| 线 | q0 · 48M |
|---|---|
| softmax + 负熵 α 1e-5 | -0.0054 |
| softmax + 负熵 α 1e-4 | 0.0040 |

**〔指标：private_internal〕**

**DSV3 底座 + 负熵 − MUONH_FIT（鼠标移到某个 rung 看数值）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3（参照） | 0.0052 | -0.0023 | 0.0019 | 0.0014 |
| DSV3 + 负熵 α 1e-6 | 0.0019 | · | · | · |
| DSV3 + 负熵 α 1e-5 | 0.0020 | -0.0031 | 0.0004 | · |
| DSV3 + 负熵 α 1e-4 | 0.0039 | -0.0033 | · | · |
| DSV3 + 负熵 α 1e-3 | 0.0143 | · | · | · |
| DSV3 + 负熵 α 1e-5，去掉 bias | -0.0056 | · | · | · |
| DSV3 + 负熵 α 1e-4，去掉 bias | 0.0037 | · | · | · |

**softmax 底座 + 负熵 − MUONH_FIT（鼠标移到某个 rung 看数值）**

| 线 | q0 · 48M |
|---|---|
| softmax + 负熵 α 1e-5 | 0.0005 |
| softmax + 负熵 α 1e-4 | 0.0027 |

#### SWS 底座：各组 − DSV3 SWS

0 线 = **DSV3 SWS**（MUONH_FIT_DSV3_SWS）：MuonH，新 LR 公式（fit），纯 CE（不用 KD），128 选 8 MoE，DSV3 router（sigmoid 打分 + expert bias + seq aux 1e-4，gate 用 AdamH、LR ×1），SWA（每 6 层 5 层窗口 2048）+ attention sink。

**SWS 底座 + 负熵 − DSV3 SWS（鼠标移到某个 rung 看数值）**

**〔指标：overall_tsqw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| 1e-5 去掉 bias | -0.0044 | -0.0065 | -0.0070 | -0.0063 | -0.0050 |
| 1e-4 去掉 bias | 0.0149 | · | · | · | · |
| 1e-5 去掉 bias + 按选中频率统计 | -0.0008 | 0.0000 | · | · | · |
| 1e-5 去掉 EMA | 0.0035 | 0.0017 | -0.0030 | · | · |
| 1e-5 + AdamW gate ×0.19 | 0.0050 | 0.0036 | · | · | · |
| 1e-5 + bias γ 1e-4 | 0.0013 | 0.0033 | · | · | · |
| α 4e-4 | 0.0132 | -0.0001 | · | · | · |
| α 1e-3 | 0.0091 | 0.0020 | · | · | · |
| α 1e-6 | 0.0028 | · | · | · | · |
| α 1e-4 | 0.0022 | 0.0028 | -0.0010 | -0.0019 | · |
| α 1e-5 | -0.0004 | -0.0008 | -0.0025 | -0.0054 | · |

**〔指标：overall_tw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| 1e-5 去掉 bias | -0.0039 | -0.0063 | -0.0068 | -0.0066 | -0.0051 |
| 1e-4 去掉 bias | 0.0164 | · | · | · | · |
| 1e-5 去掉 bias + 按选中频率统计 | -0.0003 | 0.0000 | · | · | · |
| 1e-5 去掉 EMA | 0.0036 | 0.0024 | -0.0029 | · | · |
| 1e-5 + AdamW gate ×0.19 | 0.0053 | 0.0037 | · | · | · |
| 1e-5 + bias γ 1e-4 | 0.0017 | 0.0037 | · | · | · |
| α 4e-4 | 0.0140 | 0.0001 | · | · | · |
| α 1e-3 | 0.0095 | 0.0016 | · | · | · |
| α 1e-6 | 0.0030 | · | · | · | · |
| α 1e-4 | 0.0024 | 0.0033 | -0.0005 | -0.0020 | · |
| α 1e-5 | -0.0001 | -0.0006 | -0.0020 | -0.0060 | · |

**〔指标：overall_macro〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| 1e-5 去掉 bias | -0.0049 | -0.0064 | -0.0072 | -0.0060 | -0.0047 |
| 1e-4 去掉 bias | 0.0120 | · | · | · | · |
| 1e-5 去掉 bias + 按选中频率统计 | -0.0018 | -0.0001 | · | · | · |
| 1e-5 去掉 EMA | 0.0034 | 0.0011 | -0.0032 | · | · |
| 1e-5 + AdamW gate ×0.19 | 0.0049 | 0.0036 | · | · | · |
| 1e-5 + bias γ 1e-4 | 0.0010 | 0.0029 | · | · | · |
| α 4e-4 | 0.0115 | 0.0002 | · | · | · |
| α 1e-3 | 0.0081 | 0.0028 | · | · | · |
| α 1e-6 | 0.0032 | · | · | · | · |
| α 1e-4 | 0.0022 | 0.0025 | -0.0016 | -0.0018 | · |
| α 1e-5 | -0.0015 | -0.0007 | -0.0031 | -0.0047 | · |

**〔指标：val_v2〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| 1e-5 去掉 bias | -0.0024 | -0.0042 | -0.0040 | -0.0049 | -0.0038 |
| 1e-4 去掉 bias | 0.0193 | · | · | · | · |
| 1e-5 去掉 bias + 按选中频率统计 | 0.0013 | 0.0007 | · | · | · |
| 1e-5 去掉 EMA | 0.0032 | 0.0010 | -0.0022 | · | · |
| 1e-5 + AdamW gate ×0.19 | 0.0040 | 0.0022 | · | · | · |
| 1e-5 + bias γ 1e-4 | 0.0024 | 0.0039 | · | · | · |
| α 4e-4 | 0.0149 | -0.0003 | · | · | · |
| α 1e-3 | 0.0083 | 0.0019 | · | · | · |
| α 1e-6 | 0.0031 | · | · | · | · |
| α 1e-4 | 0.0014 | 0.0022 | -0.0011 | -0.0011 | · |
| α 1e-5 | 0.0002 | -0.0004 | -0.0019 | -0.0040 | · |

**〔指标：uncheatable〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| 1e-5 去掉 bias | -0.0019 | -0.0058 | -0.0067 | -0.0073 | -0.0051 |
| 1e-4 去掉 bias | 0.0191 | · | · | · | · |
| 1e-5 去掉 bias + 按选中频率统计 | 0.0007 | 0.0001 | · | · | · |
| 1e-5 去掉 EMA | 0.0042 | 0.0045 | -0.0027 | · | · |
| 1e-5 + AdamW gate ×0.19 | 0.0061 | 0.0044 | · | · | · |
| 1e-5 + bias γ 1e-4 | 0.0030 | 0.0052 | · | · | · |
| α 4e-4 | 0.0161 | 0.0003 | · | · | · |
| α 1e-3 | 0.0105 | 0.0013 | · | · | · |
| α 1e-6 | 0.0044 | · | · | · | · |
| α 1e-4 | 0.0036 | 0.0048 | 0.0007 | -0.0024 | · |
| α 1e-5 | 0.0003 | -0.0002 | -0.0009 | -0.0074 | · |

**〔指标：codefresh〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| 1e-5 去掉 bias | -0.0111 | -0.0137 | -0.0148 | -0.0077 | -0.0059 |
| 1e-4 去掉 bias | -0.0047 | · | · | · | · |
| 1e-5 去掉 bias + 按选中频率统计 | -0.0041 | -0.0037 | · | · | · |
| 1e-5 去掉 EMA | 0.0019 | -0.0038 | -0.0053 | · | · |
| 1e-5 + AdamW gate ×0.19 | 0.0067 | 0.0006 | · | · | · |
| 1e-5 + bias γ 1e-4 | -0.0046 | -0.0050 | · | · | · |
| α 1e-3 | 0.0088 | 0.0023 | · | · | · |
| α 4e-4 | 0.0052 | -0.0036 | · | · | · |
| α 1e-6 | 0.0033 | · | · | · | · |
| α 1e-4 | 0.0030 | -0.0018 | -0.0058 | -0.0026 | · |
| α 1e-5 | -0.0031 | -0.0067 | -0.0085 | -0.0040 | · |

**〔指标：private_internal〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| 1e-5 去掉 bias | -0.0065 | -0.0058 | -0.0070 | -0.0061 | -0.0033 |
| 1e-4 去掉 bias | 0.0116 | · | · | · | · |
| 1e-5 去掉 bias + 按选中频率统计 | -0.0028 | 0.0001 | · | · | · |
| 1e-5 去掉 EMA | 0.0023 | 0.0015 | -0.0026 | · | · |
| 1e-5 + AdamW gate ×0.19 | 0.0032 | 0.0045 | · | · | · |
| 1e-5 + bias γ 1e-4 | 0.0021 | 0.0056 | · | · | · |
| α 4e-4 | 0.0097 | 0.0024 | · | · | · |
| α 1e-3 | 0.0057 | 0.0045 | · | · | · |
| α 1e-6 | 0.0021 | · | · | · | · |
| α 1e-4 | 0.0001 | 0.0029 | -0.0016 | -0.0022 | · |
| α 1e-5 | -0.0012 | 0.0004 | -0.0026 | -0.0041 | · |

α 1e-2 没画：q0 +0.036，q1 训崩（+0.256）。

### 第 0 层 dense（第一层不用 MoE）

把第 0 层的 MoE 换成一个普通的 dense FFN（SwiGLU，宽 = 8 × expert），其余层不变；dense 实验都建在 DSV3 SWS 上。
0 线 = **MUONH_FIT**：MuonH，新 LR 公式（fit），纯 CE（不用 KD），128 选 8 MoE，全注意力，原版 softmax router。数值 = 该线 − MUONH_FIT（doc-isolated，按 seed 配对，q0 是 3 个 seed 的平均；负数 = 更好）。
**负熵均衡 φ（phi balancing），系数 α = 1e-5**：每层对 router 概率做滑动平均（EMA），得到每个 expert 长期分到的平均概率 m_e；损失 = α·E·Σ_e p_e·(log m_e + 1)，p_e = 当前这批 token 给 expert e 的平均概率（括号里那项不传梯度）。长期用得多的 expert 被往下压、用得少的被往上拉，把分配推向均匀（熵最大）。两种用法（各组成见下表）：**bias + 负熵** = 保留 DSV3 的 expert bias（γ = 1e-3），只把 seq aux 换成负熵；**只用负熵** = expert bias 和 seq aux 都去掉，只靠这一项均衡。统计范围是整个 global batch，和 micro-batch 大小无关（源自 arXiv 2605.15403；详见前面「Router 负熵均衡 (phi balancing)」一章）。
**结论：不采用。**loss 只比 DSV3 低一点（下图两条负熵实线比蓝色实线低 0.002–0.006），在单 seed 噪声（约 ±0.008）以内；在完整 PROD 配置里几乎没差（PROD1_PHI − PROD1_DPSK：q0 0.000，q1 −0.001，q2 −0.002）；只用负熵的版本 router 不健康（见下面 router 健康图）。Runlong 11-03 定 DSV3 为默认，负熵的 run 全部停掉。

| 线 | 注意力 | 第 0 层 | sigmoid 打分，top-8，权重归一化 | expert bias（γ = 1e-3，每步按负载调） | seq aux（α = 1e-4） | 负熵 φ（α = 1e-5） | Change ID |
|---|---|---|---|---|---|---|---|
| DSV3 · 全 MoE · 全注意力 | 全注意力 | MoE | ✓ | ✓ | ✓ | ✗ | `MUONH_FIT_DSV3` |
| DSV3 · 全 MoE | SWS | MoE | ✓ | ✓ | ✓ | ✗ | `MUONH_FIT_DSV3_SWS` |
| DSV3 · 第 0 层 dense | SWS | dense | ✓ | ✓ | ✓ | ✗ | `MUONH_FIT_DSV3_SWS_DENSE1` |
| DSV3 · 第 0、1 层 dense | SWS | dense（第 1 层也 dense） | ✓ | ✓ | ✓ | ✗ | `MUONH_FIT_DSV3_SWS_DENSE2` |
| DSV3 · 第 0 层 dense（该层全注意力） | SWS；第 0 层全注意力（NoPE） | dense | ✓ | ✓ | ✓ | ✗ | `MUONH_FIT_DSV3_SWS_DENSE1_FA0` |
| bias + 负熵 · 全 MoE | SWS | MoE | ✓ | ✓ | ✗ | ✓ | `ROUTER_NE_1EM5_SWS` |
| bias + 负熵 · 第 0 层 dense | SWS | dense | ✓ | ✓ | ✗ | ✓ | `ROUTER_NE_1EM5_SWS_DENSE1` |
| 只用负熵 · 全 MoE | SWS | MoE | ✓ | ✗ | ✗ | ✓ | `ROUTER_NE_1EM5_NOBIAS_SWS` |
| 只用负熵 · 第 0 层 dense | SWS | dense | ✓ | ✗ | ✗ | ✓ | `ROUTER_NE_1EM5_NOBIAS_SWS_DENSE1` |

router 三种：**DSV3** = expert bias + seq aux；**bias + 负熵** = 把 DSV3 的 seq aux 换成负熵；**只用负熵** = expert bias 和 seq aux 都不要（只剩 sigmoid 打分，已不算 DSV3）。SWS = SWA 窗口 2048，局部:全局 5:1，加 attention sink。所有线相同：MuonH、拟合 LR、CE；128 个 expert 选 8 个（无 shared）；dense 层 = SwiGLU，宽 8 × expert（与 MoE 层激活宽度相同）；seed 1，同一数据顺序。同一种 router 的全 MoE（虚线）和第 0 层 dense（实线）用同一颜色；下面的 router 健康图用同样的颜色和线型。

**各线 − MUONH_FIT；虚线 = 全 MoE 参照（没有 dense 层），实线 = 第 0 层 dense（鼠标移到某个 rung 看所有线的数值）**

**〔指标：overall_tsqw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3 · 全 MoE · 全注意力 | 0.0023 | -0.0024 | 0.0018 | 0.0017 |
| DSV3 · 全 MoE | -0.0134 | -0.0087 | -0.0064 | -0.0078 |
| bias + 负熵 · 全 MoE | -0.0138 | -0.0095 | -0.0089 | -0.0132 |
| 只用负熵 · 全 MoE | -0.0165 | -0.0152 | -0.0134 | -0.0141 |
| DSV3 · 第 0、1 层 dense | · | -0.0126 | -0.0095 | -0.0112 |
| DSV3 · 第 0 层 dense | · | -0.0142 | -0.0124 | · |
| bias + 负熵 · 第 0 层 dense | · | -0.0193 | -0.0148 | · |
| 只用负熵 · 第 0 层 dense | · | -0.0195 | -0.0184 | · |

**〔指标：overall_tw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3 · 全 MoE · 全注意力 | 0.0021 | -0.0028 | 0.0019 | 0.0021 |
| DSV3 · 全 MoE | -0.0135 | -0.0088 | -0.0066 | -0.0077 |
| bias + 负熵 · 全 MoE | -0.0135 | -0.0094 | -0.0086 | -0.0136 |
| 只用负熵 · 全 MoE | -0.0159 | -0.0151 | -0.0134 | -0.0142 |
| DSV3 · 第 0、1 层 dense | · | -0.0123 | -0.0093 | -0.0113 |
| DSV3 · 第 0 层 dense | · | -0.0143 | -0.0123 | · |
| bias + 负熵 · 第 0 层 dense | · | -0.0197 | -0.0146 | · |
| 只用负熵 · 第 0 层 dense | · | -0.0198 | -0.0185 | · |

**〔指标：overall_macro〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3 · 全 MoE · 全注意力 | 0.0032 | -0.0024 | 0.0019 | 0.0017 |
| DSV3 · 全 MoE | -0.0116 | -0.0091 | -0.0061 | -0.0079 |
| bias + 负熵 · 全 MoE | -0.0132 | -0.0098 | -0.0091 | -0.0126 |
| 只用负熵 · 全 MoE | -0.0162 | -0.0154 | -0.0133 | -0.0139 |
| DSV3 · 第 0、1 层 dense | · | -0.0133 | -0.0095 | -0.0111 |
| DSV3 · 第 0 层 dense | · | -0.0145 | -0.0124 | · |
| bias + 负熵 · 第 0 层 dense | · | -0.0191 | -0.0150 | · |
| 只用负熵 · 第 0 层 dense | · | -0.0195 | -0.0182 | · |

**〔指标：val_v2〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3 · 全 MoE · 全注意力 | 0.0011 | -0.0023 | 0.0005 | 0.0003 |
| bias + 负熵 · 全 MoE | -0.0126 | -0.0072 | -0.0069 | -0.0097 |
| DSV3 · 全 MoE | -0.0129 | -0.0068 | -0.0050 | -0.0057 |
| 只用负熵 · 全 MoE | -0.0139 | -0.0110 | -0.0090 | -0.0106 |
| DSV3 · 第 0、1 层 dense | · | -0.0101 | -0.0063 | -0.0079 |
| DSV3 · 第 0 层 dense | · | -0.0116 | -0.0089 | · |
| 只用负熵 · 第 0 层 dense | · | -0.0147 | -0.0130 | · |
| bias + 负熵 · 第 0 层 dense | · | -0.0153 | -0.0098 | · |

**〔指标：uncheatable〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3 · 全 MoE · 全注意力 | 0.0020 | -0.0040 | 0.0018 | 0.0031 |
| bias + 负熵 · 全 MoE | -0.0118 | -0.0090 | -0.0080 | -0.0147 |
| DSV3 · 全 MoE | -0.0121 | -0.0089 | -0.0071 | -0.0074 |
| 只用负熵 · 全 MoE | -0.0125 | -0.0147 | -0.0138 | -0.0146 |
| DSV3 · 第 0、1 层 dense | · | -0.0116 | -0.0091 | -0.0120 |
| DSV3 · 第 0 层 dense | · | -0.0152 | -0.0125 | · |
| 只用负熵 · 第 0 层 dense | · | -0.0212 | -0.0194 | · |
| bias + 负熵 · 第 0 层 dense | · | -0.0214 | -0.0143 | · |

**〔指标：codefresh〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3 · 全 MoE · 全注意力 | 0.0061 | 0.0020 | 0.0076 | 0.0057 |
| DSV3 · 全 MoE | -0.0148 | -0.0059 | -0.0041 | -0.0091 |
| bias + 负熵 · 全 MoE | -0.0179 | -0.0126 | -0.0126 | -0.0131 |
| 只用负熵 · 全 MoE | -0.0224 | -0.0196 | -0.0189 | -0.0168 |
| DSV3 · 第 0、1 层 dense | · | -0.0128 | -0.0115 | -0.0109 |
| DSV3 · 第 0 层 dense | · | -0.0137 | -0.0165 | · |
| bias + 负熵 · 第 0 层 dense | · | -0.0200 | -0.0220 | · |
| 只用负熵 · 第 0 层 dense | · | -0.0240 | -0.0250 | · |

**〔指标：private_internal〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| DSV3 · 全 MoE · 全注意力 | 0.0052 | -0.0023 | 0.0019 | 0.0014 |
| DSV3 · 全 MoE | -0.0094 | -0.0084 | -0.0053 | -0.0067 |
| bias + 负熵 · 全 MoE | -0.0106 | -0.0081 | -0.0078 | -0.0108 |
| 只用负熵 · 全 MoE | -0.0129 | -0.0142 | -0.0123 | -0.0127 |
| DSV3 · 第 0、1 层 dense | · | -0.0116 | -0.0093 | -0.0109 |
| DSV3 · 第 0 层 dense | · | -0.0125 | -0.0109 | · |
| bias + 负熵 · 第 0 层 dense | · | -0.0165 | -0.0140 | · |
| 只用负熵 · 第 0 层 dense | · | -0.0178 | -0.0175 | · |

还没有我们自己的评测、所以没画：DSV3 · 第 0 层 dense（该层全注意力）

#### router 健康：全 MoE（虚线）对比 dense 第 0 层（实线），颜色和线型同上表

与上面「router 健康统计」同一格式：来自训练日志（Boxiang 的 `router_health.py`，每 100 步汇总一次，覆盖所有 MoE 层，seed 1）。每个 rung 一张图，横轴 = 训练步数；上方按钮切换指标，鼠标悬停看数值。q4 的 dense 组还在训练，所以线比较短。

**〔视图：MaxVio（层平均）〕**

| 横轴 | 只用负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | bias + 负熵 · 全 MoE | DSV3 · 全 MoE | bias + 负熵 · 第 0 层 dense | DSV3 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q1 · step ~608 | 1.47 | 1.3 | 0.838 | 0.743 | 0.471 | 0.4 |
| q1 · step ~1,825 | 1.87 | 1.45 | 0.754 | 0.844 | 0.372 | 0.42 |
| q1 · step ~3,041 | 1.8 | 1.5 | 0.601 | 0.823 | 0.382 | 0.48 |
| q1 · step ~4,258 | 1.97 | 1.51 | 0.672 | 0.646 | 0.362 | 0.461 |
| q1 · step ~5,475 | 2 | 1.57 | 0.581 | 0.69 | 0.399 | 0.378 |
| q1 · step ~6,691 | 2 | 1.59 | 0.442 | 0.536 | 0.335 | 0.373 |
| q1 · step ~8,516 | 2.04 | 1.5 | 0.468 | 0.457 | 0.275 | 0.318 |
| q1 · step ~9,733 | 2.02 | 1.42 | 0.369 | 0.529 | 0.268 | 0.247 |
| q1 · step ~10,950 | 2.02 | 1.44 | 0.3 | 0.38 | 0.283 | 0.306 |
| q1 · step ~12,166 | 2 | 1.46 | 0.248 | 0.267 | 0.232 | 0.226 |
| q1 · step ~13,383 | 1.97 | 1.44 | 0.198 | 0.195 | 0.19 | 0.184 |
| q1 · step ~14,600 | 2.08 | 1.52 | 0.176 | 0.174 | 0.167 | 0.154 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | 只用负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | bias + 负熵 · 全 MoE | DSV3 · 全 MoE | bias + 负熵 · 第 0 层 dense | DSV3 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q2 · step ~1,412 | 1.23 | 1.01 | 0.75 | 0.603 | 0.475 | 0.433 |
| q2 · step ~4,237 | 1.39 | 1.27 | 0.514 | 0.653 | 0.465 | 0.459 |
| q2 · step ~7,062 | 1.59 | 1.37 | 0.596 | 0.761 | 0.446 | 0.495 |
| q2 · step ~9,887 | 1.58 | 1.35 | 0.583 | 0.636 | 0.444 | 0.418 |
| q2 · step ~12,712 | 1.59 | 1.38 | 0.489 | 0.548 | 0.367 | 0.433 |
| q2 · step ~15,537 | 1.6 | 1.35 | 0.433 | 0.595 | 0.38 | 0.412 |
| q2 · step ~19,775 | 1.56 | 1.35 | 0.388 | 0.543 | 0.33 | 0.309 |
| q2 · step ~22,600 | 1.45 | 1.26 | 0.43 | 0.483 | 0.397 | 0.374 |
| q2 · step ~25,425 | 1.53 | 1.27 | 0.324 | 0.311 | 0.291 | 0.279 |
| q2 · step ~28,250 | 1.5 | 1.26 | 0.305 | 0.329 | 0.239 | 0.228 |
| q2 · step ~31,075 | 1.47 | 1.23 | 0.218 | 0.195 | 0.187 | 0.174 |
| q2 · step ~33,900 | 1.47 | 1.25 | 0.17 | 0.16 | 0.164 | 0.132 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | 只用负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | DSV3 · 全 MoE | bias + 负熵 · 全 MoE | bias + 负熵 · 第 0 层 dense | DSV3 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q3 · step ~2,712 | 0.923 | 0.833 | 0.553 | 0.543 | 0.515 | 0.419 |
| q3 · step ~8,137 | 1.15 | 1.01 | 0.596 | 0.511 | 0.483 | 0.481 |
| q3 · step ~13,562 | 1.25 | 1.1 | 0.561 | 0.508 | 0.422 | 0.474 |
| q3 · step ~18,987 | 1.31 | 1.2 | 0.518 | 0.447 | 0.385 | 0.465 |
| q3 · step ~24,412 | 1.18 | 1.06 | 0.535 | 0.449 | 0.391 | 0.461 |
| q3 · step ~29,837 | 1.27 | 1.12 | 0.466 | 0.408 | 0.349 | 0.451 |
| q3 · step ~37,975 | 1.23 | 1.14 | 0.447 | 0.36 | 0.296 | 0.397 |
| q3 · step ~43,400 | 1.19 | 1.13 | 0.41 | 0.289 | 0.265 | 0.409 |
| q3 · step ~48,825 | 1.27 | 1.2 | 0.369 | 0.32 | 0.261 | 0.341 |
| q3 · step ~54,250 | 1.19 | 1.12 | 0.295 | 0.21 | 0.212 | 0.256 |
| q3 · step ~59,675 | 1.13 | 1.07 | 0.216 | 0.212 | 0.222 | 0.216 |
| q3 · step ~65,100 | 1.17 | 1.1 | 0.142 | 0.169 | 0.165 | 0.139 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | 只用负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | bias + 负熵 · 第 0 层 dense | bias + 负熵 · 全 MoE | DSV3 · 全 MoE | DSV3 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q4 · step ~4,637 | 0.857 | 0.732 | 0.461 | 0.458 | 0.457 | 0.422 |
| q4 · step ~13,912 | 1.01 | 0.9 | 0.406 | 0.386 | 0.499 | 0.478 |
| q4 · step ~23,187 | 1.09 | 0.916 | 0.41 | 0.37 | 0.499 | 0.442 |
| q4 · step ~32,462 | 1.02 | 0.902 | 0.443 | 0.386 | 0.516 | 0.431 |
| q4 · step ~41,737 | 0.97 | 0.917 | 0.384 | 0.391 | 0.566 | 0.516 |
| q4 · step ~51,012 | 1.03 | 0.946 | 0.366 | 0.354 | 0.558 | 0.473 |
| q4 · step ~64,925 | 0.999 | 0.929 | 0.41 | 0.38 | 0.498 | 0.46 |
| q4 · step ~74,200 | 1.02 | 0.971 | 0.372 | 0.292 | 0.504 | 0.425 |
| q4 · step ~83,475 | 1.01 | 0.959 | 0.403 | 0.248 | 0.412 | 0.357 |
| q4 · step ~92,750 | 1.04 | 1.01 | 0.397 | 0.213 | 0.291 | 0.263 |
| q4 · step ~102,025 | 0.995 | 0.941 | 0.497 | 0.248 | 0.225 | 0.202 |
| q4 · step ~111,300 | 0.978 | 0.926 | 0.454 | 0.234 | 0.144 | 0.136 |

（共 24 个点，均匀取 12 个，含首尾）

每层：一整个训练 step（所有 micro-batch、所有 rank）里分到 token 最多的 expert ÷ 平均值 − 1（DeepSeek-V3 的 MaxVio，0 = 完全均衡）；这里是所有层的平均。

**〔视图：MaxVio（最差层）〕**

| 横轴 | DSV3 · 全 MoE | bias + 负熵 · 全 MoE | 只用负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | bias + 负熵 · 第 0 层 dense | DSV3 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q1 · step ~608 | 5.76 | 5.3 | 4.38 | 2.15 | 0.808 | 0.722 |
| q1 · step ~1,825 | 5.2 | 4.88 | 7.96 | 2.95 | 0.735 | 1.2 |
| q1 · step ~3,041 | 4.63 | 2.83 | 8.48 | 3.52 | 0.802 | 1.26 |
| q1 · step ~4,258 | 5.3 | 4.94 | 8.43 | 3.57 | 0.853 | 1.24 |
| q1 · step ~5,475 | 4.97 | 4.31 | 8.56 | 3.61 | 0.969 | 0.67 |
| q1 · step ~6,691 | 2.87 | 2.54 | 8.67 | 3.65 | 0.728 | 1.08 |
| q1 · step ~8,516 | 2.19 | 3.11 | 8.51 | 3.23 | 0.378 | 0.809 |
| q1 · step ~9,733 | 3.89 | 1.75 | 8.24 | 2.94 | 0.33 | 0.363 |
| q1 · step ~10,950 | 1.69 | 0.719 | 8.65 | 2.81 | 0.359 | 0.608 |
| q1 · step ~12,166 | 0.573 | 0.391 | 8.53 | 3.03 | 0.27 | 0.361 |
| q1 · step ~13,383 | 0.437 | 0.309 | 8.43 | 3.03 | 0.284 | 0.312 |
| q1 · step ~14,600 | 0.28 | 0.271 | 8.46 | 3.4 | 0.242 | 0.275 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | 只用负熵 · 全 MoE | DSV3 · 全 MoE | bias + 负熵 · 全 MoE | bias + 负熵 · 第 0 层 dense | 只用负熵 · 第 0 层 dense | DSV3 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q2 · step ~1,412 | 6.32 | 3.99 | 3.73 | 2.2 | 2.13 | 1.14 |
| q2 · step ~4,237 | 8.93 | 4.79 | 3.04 | 1.9 | 4.19 | 1.78 |
| q2 · step ~7,062 | 9.62 | 4.83 | 3.27 | 1.37 | 4.78 | 1.63 |
| q2 · step ~9,887 | 9.59 | 4.36 | 3.91 | 1.02 | 4.74 | 0.844 |
| q2 · step ~12,712 | 9.64 | 3.97 | 3.02 | 0.668 | 4.31 | 0.864 |
| q2 · step ~15,537 | 9.54 | 3.44 | 2.41 | 1.67 | 4.15 | 1.57 |
| q2 · step ~19,775 | 9.4 | 4.47 | 1.54 | 0.481 | 3.7 | 0.511 |
| q2 · step ~22,600 | 9.41 | 2.83 | 1.22 | 0.639 | 3.38 | 0.978 |
| q2 · step ~25,425 | 9.55 | 0.918 | 0.857 | 0.372 | 3.23 | 0.439 |
| q2 · step ~28,250 | 9.45 | 1.15 | 0.531 | 0.407 | 2.87 | 0.374 |
| q2 · step ~31,075 | 9.26 | 0.392 | 0.487 | 0.272 | 2.67 | 0.238 |
| q2 · step ~33,900 | 9.29 | 0.329 | 0.357 | 0.242 | 2.42 | 0.201 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | 只用负熵 · 全 MoE | DSV3 · 全 MoE | 只用负熵 · 第 0 层 dense | bias + 负熵 · 第 0 层 dense | bias + 负熵 · 全 MoE | DSV3 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q3 · step ~2,712 | 6.25 | 4.65 | 3.54 | 2.93 | 2.72 | 2.6 |
| q3 · step ~8,137 | 8.99 | 5.6 | 5.71 | 2.71 | 3.45 | 2.01 |
| q3 · step ~13,562 | 9.08 | 2.63 | 5.85 | 1.26 | 2.95 | 1.22 |
| q3 · step ~18,987 | 9 | 3.45 | 5.82 | 1.06 | 1.93 | 1.07 |
| q3 · step ~24,412 | 9.04 | 3.35 | 5.92 | 0.669 | 1.95 | 1.12 |
| q3 · step ~29,837 | 8.76 | 1.84 | 5.61 | 0.713 | 2.38 | 1.7 |
| q3 · step ~37,975 | 7.88 | 1.92 | 5.55 | 0.746 | 1.7 | 0.8 |
| q3 · step ~43,400 | 7.5 | 1.74 | 5.55 | 0.664 | 1.26 | 1.13 |
| q3 · step ~48,825 | 7.12 | 1.52 | 5.57 | 0.545 | 1.13 | 0.82 |
| q3 · step ~54,250 | 6.93 | 1.31 | 5.02 | 0.465 | 0.315 | 0.808 |
| q3 · step ~59,675 | 6.72 | 0.434 | 4.35 | 0.37 | 0.304 | 0.472 |
| q3 · step ~65,100 | 6.48 | 0.278 | 4.37 | 0.34 | 0.266 | 0.332 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | 只用负熵 · 全 MoE | bias + 负熵 · 第 0 层 dense | DSV3 · 全 MoE | 只用负熵 · 第 0 层 dense | bias + 负熵 · 全 MoE | DSV3 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q4 · step ~4,637 | 7.63 | 4.79 | 3.23 | 3.15 | 2.61 | 2.4 |
| q4 · step ~13,912 | 9.22 | 2.25 | 2.89 | 4.82 | 1.78 | 2.91 |
| q4 · step ~23,187 | 9.75 | 2.65 | 2.39 | 5.07 | 1.44 | 1.87 |
| q4 · step ~32,462 | 9.38 | 2.12 | 2.39 | 5.13 | 1.34 | 1.19 |
| q4 · step ~41,737 | 9.51 | 1.29 | 3.45 | 5.21 | 1.43 | 1.82 |
| q4 · step ~51,012 | 9.47 | 0.779 | 4.06 | 5.26 | 1.01 | 1.19 |
| q4 · step ~64,925 | 9.71 | 0.674 | 1.55 | 5.05 | 0.629 | 1.57 |
| q4 · step ~74,200 | 9.67 | 1.09 | 2.38 | 4.94 | 0.673 | 1.16 |
| q4 · step ~83,475 | 9.36 | 2.31 | 1.54 | 4.86 | 0.53 | 1.26 |
| q4 · step ~92,750 | 8.99 | 2.8 | 0.63 | 4.69 | 0.376 | 0.655 |
| q4 · step ~102,025 | 8.6 | 4.04 | 0.586 | 4.03 | 0.717 | 0.634 |
| q4 · step ~111,300 | 8.09 | 4.82 | 0.511 | 4 | 1.28 | 0.573 |

（共 24 个点，均匀取 12 个，含首尾）

同上，取最不均衡的那一层。

**〔视图：dead expert (%)〕**

| 横轴 | 只用负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | DSV3 · 全 MoE | bias + 负熵 · 全 MoE | DSV3 · 第 0 层 dense | bias + 负熵 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q1 · step ~608 | 1.65% | 0.23% | 0.217% | 0.0868% | 0% | 0% |
| q1 · step ~1,825 | 2.17% | 0.276% | 0.174% | 0.217% | 0% | 0% |
| q1 · step ~3,041 | 2% | 0.184% | 0.174% | 0.26% | 0% | 0% |
| q1 · step ~4,258 | 2.3% | 0.0919% | 0.174% | 0.13% | 0.046% | 0% |
| q1 · step ~5,475 | 1.91% | 0.23% | 0.0434% | 0.0434% | 0% | 0% |
| q1 · step ~6,691 | 1.91% | 0.184% | 0% | 0.0868% | 0% | 0% |
| q1 · step ~8,516 | 1.87% | 0.0919% | 0% | 0% | 0% | 0% |
| q1 · step ~9,733 | 1.26% | 0.046% | 0% | 0% | 0% | 0% |
| q1 · step ~10,950 | 1.56% | 0% | 0% | 0% | 0% | 0% |
| q1 · step ~12,166 | 0.781% | 0% | 0% | 0% | 0% | 0% |
| q1 · step ~13,383 | 0.608% | 0% | 0% | 0% | 0% | 0% |
| q1 · step ~14,600 | 0.26% | 0% | 0% | 0% | 0% | 0% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | 只用负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | bias + 负熵 · 全 MoE | bias + 负熵 · 第 0 层 dense | DSV3 · 全 MoE | DSV3 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q2 · step ~1,412 | 1.92% | 0.611% | 0.228% | 0.034% | 0.0326% | 0% |
| q2 · step ~4,237 | 2.15% | 0.442% | 0.163% | 0.034% | 0.0326% | 0% |
| q2 · step ~7,062 | 2.12% | 0.34% | 0.0977% | 0% | 0.0326% | 0% |
| q2 · step ~9,887 | 1.89% | 0.238% | 0.163% | 0% | 0.0326% | 0% |
| q2 · step ~12,712 | 2.05% | 0.374% | 0.163% | 0% | 0% | 0% |
| q2 · step ~15,537 | 1.92% | 0.272% | 0.0651% | 0% | 0.0326% | 0% |
| q2 · step ~19,775 | 1.99% | 0.102% | 0.13% | 0% | 0.0326% | 0% |
| q2 · step ~22,600 | 1.56% | 0.034% | 0% | 0% | 0% | 0% |
| q2 · step ~25,425 | 1.2% | 0% | 0% | 0% | 0% | 0% |
| q2 · step ~28,250 | 1.37% | 0% | 0% | 0% | 0% | 0% |
| q2 · step ~31,075 | 1.17% | 0% | 0% | 0% | 0% | 0% |
| q2 · step ~33,900 | 0.684% | 0% | 0% | 0% | 0% | 0% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | 只用负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | bias + 负熵 · 全 MoE | bias + 负熵 · 第 0 层 dense | DSV3 · 全 MoE | DSV3 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q3 · step ~2,712 | 1.41% | 0.835% | 0.104% | 0.0808% | 0.0521% | 0% |
| q3 · step ~8,137 | 1.51% | 0.997% | 0.0521% | 0% | 0.0521% | 0% |
| q3 · step ~13,562 | 1.82% | 0.781% | 0% | 0% | 0% | 0% |
| q3 · step ~18,987 | 1.67% | 0.781% | 0.0521% | 0% | 0.026% | 0.0269% |
| q3 · step ~24,412 | 1.51% | 0.647% | 0.026% | 0% | 0.026% | 0% |
| q3 · step ~29,837 | 1.54% | 0.539% | 0% | 0% | 0% | 0% |
| q3 · step ~37,975 | 1.54% | 0.431% | 0% | 0% | 0.026% | 0% |
| q3 · step ~43,400 | 1.48% | 0.485% | 0% | 0% | 0% | 0% |
| q3 · step ~48,825 | 1.41% | 0.377% | 0% | 0% | 0% | 0% |
| q3 · step ~54,250 | 1.12% | 0.189% | 0% | 0% | 0% | 0% |
| q3 · step ~59,675 | 0.885% | 0.135% | 0% | 0% | 0% | 0% |
| q3 · step ~65,100 | 0.651% | 0% | 0% | 0% | 0% | 0% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | 只用负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | bias + 负熵 · 第 0 层 dense | DSV3 · 全 MoE | bias + 负熵 · 全 MoE | DSV3 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q4 · step ~4,637 | 0.803% | 0.402% | 0.0223% | 0.0217% | 0.0217% | 0% |
| q4 · step ~13,912 | 0.911% | 0.469% | 0% | 0.0217% | 0.0217% | 0.0223% |
| q4 · step ~23,187 | 1.13% | 0.603% | 0% | 0% | 0% | 0% |
| q4 · step ~32,462 | 1.11% | 0.379% | 0% | 0% | 0% | 0% |
| q4 · step ~41,737 | 1.17% | 0.446% | 0% | 0% | 0% | 0% |
| q4 · step ~51,012 | 1.13% | 0.446% | 0% | 0% | 0% | 0% |
| q4 · step ~64,925 | 0.955% | 0.29% | 0% | 0% | 0% | 0% |
| q4 · step ~74,200 | 1.13% | 0.312% | 0.067% | 0% | 0% | 0% |
| q4 · step ~83,475 | 0.911% | 0.223% | 0.692% | 0% | 0% | 0% |
| q4 · step ~92,750 | 0.694% | 0.246% | 0.848% | 0% | 0% | 0% |
| q4 · step ~102,025 | 0.651% | 0.134% | 0.915% | 0% | 0% | 0% |
| q4 · step ~111,300 | 0.434% | 0.0446% | 0.982% | 0% | 0.109% | 0% |

（共 24 个点，均匀取 12 个，含首尾）

分到的 token 不到平均值 10% 的 expert 数，占（层数 × 128）的百分比。

**〔视图：|bias| 最大〕**

| 横轴 | DSV3 · 全 MoE | bias + 负熵 · 全 MoE | DSV3 · 第 0 层 dense | bias + 负熵 · 第 0 层 dense |
|---|---|---|---|---|
| q1 · step ~608 | 0.194 | 0.192 | 0.152 | 0.136 |
| q1 · step ~1,825 | 0.498 | 0.486 | 0.196 | 0.198 |
| q1 · step ~3,041 | 0.848 | 0.77 | 0.258 | 0.276 |
| q1 · step ~4,258 | 1.17 | 1.02 | 0.35 | 0.352 |
| q1 · step ~5,475 | 1.46 | 1.25 | 0.436 | 0.428 |
| q1 · step ~6,691 | 1.72 | 1.45 | 0.512 | 0.488 |
| q1 · step ~8,516 | 2.07 | 1.73 | 0.617 | 0.578 |
| q1 · step ~9,733 | 2.23 | 1.85 | 0.666 | 0.621 |
| q1 · step ~10,950 | 2.34 | 1.94 | 0.707 | 0.652 |
| q1 · step ~12,166 | 2.41 | 1.99 | 0.735 | 0.676 |
| q1 · step ~13,383 | 2.44 | 2.01 | 0.753 | 0.696 |
| q1 · step ~14,600 | 2.48 | 2.04 | 0.771 | 0.714 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | bias + 负熵 · 全 MoE | DSV3 · 全 MoE | bias + 负熵 · 第 0 层 dense | DSV3 · 第 0 层 dense |
|---|---|---|---|---|
| q2 · step ~1,412 | 0.368 | 0.352 | 0.252 | 0.174 |
| q2 · step ~4,237 | 1.08 | 1.12 | 0.63 | 0.428 |
| q2 · step ~7,062 | 1.73 | 1.95 | 0.868 | 0.692 |
| q2 · step ~9,887 | 2.32 | 2.68 | 1.06 | 0.918 |
| q2 · step ~12,712 | 2.87 | 3.39 | 1.22 | 1.11 |
| q2 · step ~15,537 | 3.36 | 3.96 | 1.35 | 1.27 |
| q2 · step ~19,775 | 4 | 4.66 | 1.51 | 1.44 |
| q2 · step ~22,600 | 4.32 | 5.02 | 1.58 | 1.53 |
| q2 · step ~25,425 | 4.53 | 5.29 | 1.63 | 1.59 |
| q2 · step ~28,250 | 4.62 | 5.48 | 1.67 | 1.63 |
| q2 · step ~31,075 | 4.67 | 5.57 | 1.72 | 1.66 |
| q2 · step ~33,900 | 4.73 | 5.66 | 1.76 | 1.69 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | bias + 负熵 · 全 MoE | DSV3 · 全 MoE | bias + 负熵 · 第 0 层 dense | DSV3 · 第 0 层 dense |
|---|---|---|---|---|
| q3 · step ~2,712 | 0.652 | 0.614 | 0.54 | 0.402 |
| q3 · step ~8,137 | 1.78 | 2.05 | 1.33 | 1.3 |
| q3 · step ~13,562 | 2.86 | 3.38 | 2.05 | 2.15 |
| q3 · step ~18,987 | 3.93 | 4.54 | 2.64 | 2.92 |
| q3 · step ~24,412 | 4.92 | 5.64 | 3.13 | 3.63 |
| q3 · step ~29,837 | 5.77 | 6.64 | 3.55 | 4.26 |
| q3 · step ~37,975 | 6.77 | 8.01 | 4.06 | 5.1 |
| q3 · step ~43,400 | 7.25 | 8.84 | 4.35 | 5.57 |
| q3 · step ~48,825 | 7.56 | 9.54 | 4.59 | 5.96 |
| q3 · step ~54,250 | 7.71 | 10.1 | 4.75 | 6.25 |
| q3 · step ~59,675 | 7.8 | 10.3 | 4.87 | 6.38 |
| q3 · step ~65,100 | 7.89 | 10.5 | 4.97 | 6.47 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | bias + 负熵 · 全 MoE | bias + 负熵 · 第 0 层 dense | DSV3 · 全 MoE | DSV3 · 第 0 层 dense |
|---|---|---|---|---|
| q4 · step ~4,637 | 0.966 | 0.954 | 0.882 | 0.702 |
| q4 · step ~13,912 | 2.54 | 2.5 | 3.11 | 2.27 |
| q4 · step ~23,187 | 3.96 | 3.72 | 5.01 | 3.71 |
| q4 · step ~32,462 | 5.22 | 4.69 | 6.72 | 5.02 |
| q4 · step ~41,737 | 6.3 | 5.51 | 8.33 | 6.23 |
| q4 · step ~51,012 | 7.18 | 6.22 | 9.87 | 7.34 |
| q4 · step ~64,925 | 8.23 | 7.1 | 11.9 | 8.85 |
| q4 · step ~74,200 | 8.71 | 7.57 | 13.1 | 9.72 |
| q4 · step ~83,475 | 9.03 | 7.97 | 14.1 | 10.5 |
| q4 · step ~92,750 | 9.24 | 8.27 | 14.8 | 11.1 |
| q4 · step ~102,025 | 9.38 | 8.98 | 15.2 | 11.4 |
| q4 · step ~111,300 | 9.54 | 12.3 | 15.3 | 11.5 |

（共 24 个点，均匀取 12 个，含首尾）

DSV3 每个 expert 的 bias 在所有层里的最大绝对值；红色虚线 = Boxiang 预设的健康上限 1。注意 top-8 选择只看同一层 bias 之间的差，所有 expert 一起抬高并不改变路由（例如 A 的 q3 第 0 层 bias 在 +11.2 到 +11.7 之间）。softmax 没有 bias。

**〔视图：logit 饱和 (%)〕**

| 横轴 | bias + 负熵 · 第 0 层 dense | bias + 负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | 只用负熵 · 全 MoE | DSV3 · 第 0 层 dense | DSV3 · 全 MoE |
|---|---|---|---|---|---|---|
| q1 · step ~608 | 59.2% | 58.4% | 55% | 53.6% | 51.5% | 49.9% |
| q1 · step ~1,825 | 62.8% | 61.5% | 56.4% | 54.9% | 58.7% | 57.8% |
| q1 · step ~3,041 | 61.8% | 61.3% | 55.4% | 53.8% | 54.9% | 55.7% |
| q1 · step ~4,258 | 64.2% | 62.7% | 57.9% | 55% | 52.7% | 53.6% |
| q1 · step ~5,475 | 63.3% | 62.3% | 56.7% | 54.4% | 54.6% | 54.2% |
| q1 · step ~6,691 | 64.1% | 63.5% | 56.7% | 55.3% | 56.5% | 56% |
| q1 · step ~8,516 | 63.5% | 63.5% | 55.7% | 54% | 57% | 56.7% |
| q1 · step ~9,733 | 61.6% | 62.9% | 54.2% | 52.4% | 44.9% | 46.6% |
| q1 · step ~10,950 | 61.1% | 62.5% | 53.8% | 52.1% | 47.3% | 48.9% |
| q1 · step ~12,166 | 63% | 64.4% | 54.4% | 53.1% | 51.1% | 52.1% |
| q1 · step ~13,383 | 63.5% | 65.1% | 54.9% | 53.4% | 50.4% | 51.2% |
| q1 · step ~14,600 | 58.6% | 61% | 48.9% | 48.6% | 48% | 49.2% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | bias + 负熵 · 第 0 层 dense | bias + 负熵 · 全 MoE | DSV3 · 全 MoE | DSV3 · 第 0 层 dense | 只用负熵 · 第 0 层 dense | 只用负熵 · 全 MoE |
|---|---|---|---|---|---|---|
| q2 · step ~1,412 | 67.9% | 66.6% | 63.3% | 62.9% | 62.6% | 61.4% |
| q2 · step ~4,237 | 67.3% | 68.7% | 57.2% | 54.7% | 61.8% | 62.4% |
| q2 · step ~7,062 | 65% | 67.3% | 59% | 55.8% | 59% | 59% |
| q2 · step ~9,887 | 67.6% | 69.8% | 60.3% | 58.1% | 62.1% | 61.8% |
| q2 · step ~12,712 | 68.2% | 70.2% | 59.9% | 57.7% | 61.5% | 61.9% |
| q2 · step ~15,537 | 65.8% | 67.9% | 55% | 52.2% | 59.2% | 59.5% |
| q2 · step ~19,775 | 61.4% | 64.2% | 51.6% | 48.5% | 52.5% | 53.5% |
| q2 · step ~22,600 | 60.3% | 62.9% | 49.6% | 47.1% | 52.3% | 53.3% |
| q2 · step ~25,425 | 62.2% | 64.9% | 45.2% | 42% | 53% | 54.2% |
| q2 · step ~28,250 | 63.4% | 66.3% | 38.4% | 35.3% | 54.8% | 56.5% |
| q2 · step ~31,075 | 65% | 68% | 41% | 37.9% | 56.9% | 58.1% |
| q2 · step ~33,900 | 56.8% | 60.2% | 41% | 37.4% | 46.3% | 47.7% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | bias + 负熵 · 全 MoE | bias + 负熵 · 第 0 层 dense | 只用负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | DSV3 · 全 MoE | DSV3 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q3 · step ~2,712 | 72.2% | 71.4% | 67.4% | 67% | 65.1% | 65% |
| q3 · step ~8,137 | 70.2% | 68% | 64.2% | 62.3% | 66% | 63% |
| q3 · step ~13,562 | 70.9% | 68.1% | 64.4% | 61% | 62.3% | 58.9% |
| q3 · step ~18,987 | 69% | 66% | 62.5% | 59.4% | 56.9% | 53% |
| q3 · step ~24,412 | 71.2% | 68.8% | 64.8% | 61.5% | 50% | 45.9% |
| q3 · step ~29,837 | 70.5% | 67.3% | 63.1% | 60% | 53.4% | 50% |
| q3 · step ~37,975 | 67.1% | 63.5% | 59.2% | 55.2% | 39.9% | 36% |
| q3 · step ~43,400 | 67.1% | 64% | 58.7% | 54.6% | 47.8% | 43.7% |
| q3 · step ~48,825 | 60% | 55.8% | 50.8% | 46.6% | 37.6% | 33.5% |
| q3 · step ~54,250 | 61.2% | 56.8% | 51% | 47.1% | 34.3% | 30.1% |
| q3 · step ~59,675 | 65.1% | 60.6% | 55.2% | 51.4% | 29.7% | 26.1% |
| q3 · step ~65,100 | 58.2% | 52.8% | 46.9% | 42.5% | 34.1% | 29.6% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | bias + 负熵 · 全 MoE | bias + 负熵 · 第 0 层 dense | DSV3 · 全 MoE | 只用负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | DSV3 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q4 · step ~4,637 | 76.1% | 74.5% | 73% | 72.2% | 70.9% | 70.5% |
| q4 · step ~13,912 | 75% | 72.6% | 67.4% | 70.3% | 67.9% | 62.4% |
| q4 · step ~23,187 | 74% | 71.6% | 62% | 68.7% | 66.4% | 56.4% |
| q4 · step ~32,462 | 74.7% | 71.6% | 55.3% | 68.8% | 65.8% | 50.5% |
| q4 · step ~41,737 | 71.7% | 68.3% | 54.8% | 64.9% | 61.1% | 49.3% |
| q4 · step ~51,012 | 69.8% | 65.4% | 45.4% | 61.5% | 57.3% | 40.2% |
| q4 · step ~64,925 | 66.6% | 62.3% | 44.6% | 56.6% | 52.1% | 39.6% |
| q4 · step ~74,200 | 66.2% | 61.7% | 44% | 56.9% | 51.6% | 38% |
| q4 · step ~83,475 | 63.4% | 58.6% | 41.9% | 52.6% | 47.8% | 36.2% |
| q4 · step ~92,750 | 60.3% | 54.7% | 37% | 49% | 43.9% | 31.6% |
| q4 · step ~102,025 | 62.8% | 57.5% | 26.3% | 52.3% | 46.8% | 21.6% |
| q4 · step ~111,300 | 58.6% | 52.9% | 25.1% | 47.5% | 42.3% | 20.2% |

（共 24 个点，均匀取 12 个，含首尾）

router logit |z| > 4 的比例（各层平均；只看 rank 0 的第一个 micro-batch）。包括 z < −4（明确拒绝某个 expert，正常）。

**〔视图：z > +4 (%)〕**

| 横轴 | bias + 负熵 · 第 0 层 dense | DSV3 · 第 0 层 dense | bias + 负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | 只用负熵 · 全 MoE | DSV3 · 全 MoE |
|---|---|---|---|---|---|---|
| q1 · step ~608 | 0.07% | 0.06% | 0.06% | 0.06% | 0.06% | 0.05% |
| q1 · step ~1,825 | 0.09% | 0.08% | 0.09% | 0.09% | 0.1% | 0.09% |
| q1 · step ~3,041 | 0.15% | 0.14% | 0.15% | 0.15% | 0.16% | 0.15% |
| q1 · step ~4,258 | 0.22% | 0.2% | 0.23% | 0.23% | 0.23% | 0.22% |
| q1 · step ~5,475 | 0.21% | 0.21% | 0.22% | 0.22% | 0.23% | 0.22% |
| q1 · step ~6,691 | 0.22% | 0.22% | 0.23% | 0.21% | 0.24% | 0.23% |
| q1 · step ~8,516 | 0.25% | 0.23% | 0.24% | 0.23% | 0.26% | 0.24% |
| q1 · step ~9,733 | 0.31% | 0.29% | 0.33% | 0.32% | 0.34% | 0.3% |
| q1 · step ~10,950 | 0.26% | 0.25% | 0.27% | 0.26% | 0.29% | 0.25% |
| q1 · step ~12,166 | 0.22% | 0.22% | 0.24% | 0.21% | 0.23% | 0.22% |
| q1 · step ~13,383 | 0.24% | 0.24% | 0.25% | 0.21% | 0.27% | 0.23% |
| q1 · step ~14,600 | 0.18% | 0.18% | 0.2% | 0.17% | 0.21% | 0.19% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | 只用负熵 · 全 MoE | DSV3 · 第 0 层 dense | DSV3 · 全 MoE | bias + 负熵 · 第 0 层 dense | bias + 负熵 · 全 MoE | 只用负熵 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q2 · step ~1,412 | 0.15% | 0.14% | 0.14% | 0.14% | 0.14% | 0.14% |
| q2 · step ~4,237 | 0.35% | 0.25% | 0.28% | 0.27% | 0.3% | 0.28% |
| q2 · step ~7,062 | 0.37% | 0.29% | 0.34% | 0.3% | 0.34% | 0.3% |
| q2 · step ~9,887 | 0.45% | 0.33% | 0.39% | 0.37% | 0.43% | 0.35% |
| q2 · step ~12,712 | 0.38% | 0.31% | 0.35% | 0.32% | 0.36% | 0.29% |
| q2 · step ~15,537 | 0.49% | 0.35% | 0.41% | 0.41% | 0.46% | 0.39% |
| q2 · step ~19,775 | 0.39% | 0.3% | 0.34% | 0.32% | 0.38% | 0.29% |
| q2 · step ~22,600 | 0.41% | 0.3% | 0.35% | 0.32% | 0.37% | 0.3% |
| q2 · step ~25,425 | 0.43% | 0.29% | 0.33% | 0.36% | 0.41% | 0.33% |
| q2 · step ~28,250 | 0.53% | 0.3% | 0.34% | 0.45% | 0.53% | 0.45% |
| q2 · step ~31,075 | 0.49% | 0.28% | 0.34% | 0.41% | 0.48% | 0.38% |
| q2 · step ~33,900 | 0.3% | 0.21% | 0.24% | 0.23% | 0.28% | 0.21% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | 只用负熵 · 全 MoE | bias + 负熵 · 全 MoE | DSV3 · 全 MoE | 只用负熵 · 第 0 层 dense | bias + 负熵 · 第 0 层 dense | DSV3 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q3 · step ~2,712 | 0.29% | 0.28% | 0.27% | 0.27% | 0.26% | 0.25% |
| q3 · step ~8,137 | 0.38% | 0.35% | 0.35% | 0.31% | 0.32% | 0.31% |
| q3 · step ~13,562 | 0.51% | 0.49% | 0.47% | 0.43% | 0.44% | 0.41% |
| q3 · step ~18,987 | 0.59% | 0.58% | 0.49% | 0.5% | 0.51% | 0.42% |
| q3 · step ~24,412 | 0.81% | 0.76% | 0.54% | 0.7% | 0.67% | 0.47% |
| q3 · step ~29,837 | 0.62% | 0.58% | 0.43% | 0.52% | 0.49% | 0.36% |
| q3 · step ~37,975 | 0.79% | 0.77% | 0.52% | 0.7% | 0.68% | 0.42% |
| q3 · step ~43,400 | 0.62% | 0.58% | 0.45% | 0.49% | 0.5% | 0.38% |
| q3 · step ~48,825 | 0.57% | 0.53% | 0.43% | 0.47% | 0.45% | 0.35% |
| q3 · step ~54,250 | 0.58% | 0.55% | 0.38% | 0.47% | 0.46% | 0.3% |
| q3 · step ~59,675 | 0.6% | 0.58% | 0.32% | 0.5% | 0.48% | 0.25% |
| q3 · step ~65,100 | 0.42% | 0.4% | 0.32% | 0.33% | 0.31% | 0.26% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | 只用负熵 · 全 MoE | DSV3 · 全 MoE | bias + 负熵 · 全 MoE | DSV3 · 第 0 层 dense | bias + 负熵 · 第 0 层 dense | 只用负熵 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q4 · step ~4,637 | 0.37% | 0.35% | 0.35% | 0.32% | 0.32% | 0.31% |
| q4 · step ~13,912 | 0.52% | 0.47% | 0.49% | 0.41% | 0.42% | 0.43% |
| q4 · step ~23,187 | 0.67% | 0.57% | 0.66% | 0.48% | 0.57% | 0.55% |
| q4 · step ~32,462 | 0.73% | 0.8% | 0.71% | 0.53% | 0.59% | 0.6% |
| q4 · step ~41,737 | 0.81% | 0.73% | 0.75% | 0.52% | 0.68% | 0.65% |
| q4 · step ~51,012 | 1% | 0.91% | 0.96% | 0.63% | 0.9% | 0.84% |
| q4 · step ~64,925 | 0.76% | 0.59% | 0.78% | 0.43% | 0.68% | 0.6% |
| q4 · step ~74,200 | 0.62% | 0.54% | 0.62% | 0.41% | 0.52% | 0.47% |
| q4 · step ~83,475 | 0.54% | 0.46% | 0.56% | 0.35% | 0.49% | 0.39% |
| q4 · step ~92,750 | 0.49% | 0.42% | 0.5% | 0.32% | 0.47% | 0.35% |
| q4 · step ~102,025 | 0.86% | 0.62% | 0.84% | 0.37% | 0.98% | 0.68% |
| q4 · step ~111,300 | 0.69% | 0.43% | 0.7% | 0.3% | 0.85% | 0.54% |

（共 24 个点，均匀取 12 个，含首尾）

z > +4 的比例：分数被钉在 1，选中的 expert 之间失去区分度（坏）；红色虚线 = 预设上限 5%。只有后期日志才有。

**〔视图：top-8 pinned (%)〕**

| 横轴 | bias + 负熵 · 第 0 层 dense | 只用负熵 · 第 0 层 dense | 只用负熵 · 全 MoE | DSV3 · 第 0 层 dense | bias + 负熵 · 全 MoE | DSV3 · 全 MoE |
|---|---|---|---|---|---|---|
| q1 · step ~608 | 1.06% | 1% | 0.92% | 0.89% | 0.88% | 0.82% |
| q1 · step ~1,825 | 1.47% | 1.47% | 1.58% | 1.31% | 1.44% | 1.44% |
| q1 · step ~3,041 | 2.38% | 2.39% | 2.6% | 2.3% | 2.43% | 2.35% |
| q1 · step ~4,258 | 3.53% | 3.63% | 3.68% | 3.27% | 3.61% | 3.46% |
| q1 · step ~5,475 | 3.4% | 3.46% | 3.73% | 3.39% | 3.46% | 3.45% |
| q1 · step ~6,691 | 3.49% | 3.38% | 3.84% | 3.45% | 3.66% | 3.63% |
| q1 · step ~8,516 | 4.04% | 3.74% | 4.23% | 3.64% | 3.89% | 3.81% |
| q1 · step ~9,733 | 4.9% | 5.12% | 5.38% | 4.69% | 5.26% | 4.81% |
| q1 · step ~10,950 | 4.19% | 4.1% | 4.57% | 3.97% | 4.29% | 4.05% |
| q1 · step ~12,166 | 3.54% | 3.39% | 3.76% | 3.53% | 3.77% | 3.55% |
| q1 · step ~13,383 | 3.84% | 3.43% | 4.38% | 3.84% | 4.07% | 3.63% |
| q1 · step ~14,600 | 2.85% | 2.67% | 3.34% | 2.95% | 3.2% | 3.07% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | 只用负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | DSV3 · 全 MoE | DSV3 · 第 0 层 dense | bias + 负熵 · 全 MoE | bias + 负熵 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q2 · step ~1,412 | 2.45% | 2.32% | 2.23% | 2.22% | 2.22% | 2.18% |
| q2 · step ~4,237 | 5.58% | 4.44% | 4.42% | 3.93% | 4.77% | 4.39% |
| q2 · step ~7,062 | 5.86% | 4.75% | 5.37% | 4.67% | 5.45% | 4.83% |
| q2 · step ~9,887 | 7.15% | 5.53% | 6.19% | 5.35% | 6.82% | 5.9% |
| q2 · step ~12,712 | 6.12% | 4.71% | 5.6% | 4.89% | 5.8% | 5.08% |
| q2 · step ~15,537 | 7.85% | 6.22% | 6.54% | 5.68% | 7.32% | 6.52% |
| q2 · step ~19,775 | 6.18% | 4.67% | 5.47% | 4.74% | 6.11% | 5.08% |
| q2 · step ~22,600 | 6.52% | 4.84% | 5.52% | 4.73% | 5.98% | 5.12% |
| q2 · step ~25,425 | 6.95% | 5.27% | 5.25% | 4.61% | 6.52% | 5.77% |
| q2 · step ~28,250 | 8.43% | 7.16% | 5.48% | 4.8% | 8.45% | 7.24% |
| q2 · step ~31,075 | 7.8% | 6.15% | 5.37% | 4.43% | 7.64% | 6.64% |
| q2 · step ~33,900 | 4.84% | 3.31% | 3.87% | 3.41% | 4.41% | 3.75% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | 只用负熵 · 全 MoE | bias + 负熵 · 全 MoE | DSV3 · 全 MoE | 只用负熵 · 第 0 层 dense | bias + 负熵 · 第 0 层 dense | DSV3 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q3 · step ~2,712 | 4.62% | 4.46% | 4.26% | 4.25% | 4.15% | 3.98% |
| q3 · step ~8,137 | 6.14% | 5.64% | 5.65% | 4.95% | 5.05% | 4.99% |
| q3 · step ~13,562 | 8.19% | 7.84% | 7.47% | 6.81% | 7.02% | 6.54% |
| q3 · step ~18,987 | 9.46% | 9.23% | 7.79% | 7.88% | 8.09% | 6.74% |
| q3 · step ~24,412 | 12.9% | 12.1% | 8.46% | 11% | 10.6% | 7.43% |
| q3 · step ~29,837 | 9.9% | 9.21% | 6.72% | 8.27% | 7.77% | 5.72% |
| q3 · step ~37,975 | 12.5% | 12.2% | 8.05% | 10.9% | 10.7% | 6.62% |
| q3 · step ~43,400 | 9.78% | 9.28% | 7.16% | 7.77% | 7.83% | 6.11% |
| q3 · step ~48,825 | 8.95% | 8.48% | 6.8% | 7.32% | 6.98% | 5.52% |
| q3 · step ~54,250 | 9.02% | 8.66% | 5.92% | 7.22% | 7.06% | 4.79% |
| q3 · step ~59,675 | 9.31% | 9.15% | 5.13% | 7.7% | 7.45% | 4.04% |
| q3 · step ~65,100 | 6.63% | 6.35% | 5.08% | 5.24% | 4.98% | 4.2% |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | 只用负熵 · 全 MoE | DSV3 · 全 MoE | bias + 负熵 · 全 MoE | DSV3 · 第 0 层 dense | bias + 负熵 · 第 0 层 dense | 只用负熵 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q4 · step ~4,637 | 5.86% | 5.64% | 5.61% | 5.08% | 5.05% | 5.02% |
| q4 · step ~13,912 | 8.27% | 7.47% | 7.89% | 6.52% | 6.79% | 6.81% |
| q4 · step ~23,187 | 10.7% | 9.11% | 10.6% | 7.66% | 9.19% | 8.71% |
| q4 · step ~32,462 | 11.6% | 9.98% | 11.3% | 7.63% | 9.3% | 9.38% |
| q4 · step ~41,737 | 12.5% | 10.4% | 11.9% | 8.08% | 10.6% | 10.1% |
| q4 · step ~51,012 | 15% | 11.5% | 14.9% | 8.89% | 13.6% | 12.4% |
| q4 · step ~64,925 | 11.7% | 9.2% | 12.2% | 6.8% | 10.5% | 9.22% |
| q4 · step ~74,200 | 9.71% | 8.26% | 9.87% | 6.4% | 8.21% | 7.38% |
| q4 · step ~83,475 | 8.55% | 7.4% | 8.73% | 5.62% | 7.55% | 6.23% |
| q4 · step ~92,750 | 7.78% | 6.75% | 7.91% | 5.12% | 6.95% | 5.57% |
| q4 · step ~102,025 | 12.9% | 7.87% | 12.8% | 5.53% | 12.6% | 10.1% |
| q4 · step ~111,300 | 10.7% | 6.48% | 10.8% | 4.68% | 10.6% | 8.26% |

（共 24 个点，均匀取 12 个，含首尾）

选中的 8 个 expert 里分数卡在 1 附近的比例；红色虚线 = 预设上限 5%。只有后期日志才有。

**〔视图：最大 |logit|〕**

| 横轴 | bias + 负熵 · 全 MoE | DSV3 · 第 0 层 dense | 只用负熵 · 第 0 层 dense | DSV3 · 全 MoE | bias + 负熵 · 第 0 层 dense | 只用负熵 · 全 MoE |
|---|---|---|---|---|---|---|
| q1 · step ~608 | 17 | 14.2 | 14.1 | 13.8 | 13 | 12.6 |
| q1 · step ~1,825 | 33.4 | 17 | 16.6 | 17.7 | 15.7 | 15.3 |
| q1 · step ~3,041 | 37.9 | 20.7 | 23.7 | 21.5 | 19.9 | 23.1 |
| q1 · step ~4,258 | 42.6 | 35.7 | 28.3 | 27.2 | 22.9 | 26.9 |
| q1 · step ~5,475 | 47.9 | 32.4 | 39.9 | 29.9 | 26.3 | 32.9 |
| q1 · step ~6,691 | 48.4 | 39.6 | 45.7 | 39.6 | 31.3 | 38.1 |
| q1 · step ~8,516 | 55.5 | 48.9 | 52.7 | 57.2 | 43.2 | 49.5 |
| q1 · step ~9,733 | 56.8 | 58.2 | 61.9 | 57.1 | 43.6 | 55.7 |
| q1 · step ~10,950 | 60.8 | 64.2 | 64.2 | 59.6 | 45.1 | 59 |
| q1 · step ~12,166 | 64.3 | 66.4 | 60.5 | 69.9 | 54.2 | 69.4 |
| q1 · step ~13,383 | 68.6 | 89.3 | 62.8 | 66.7 | 56.9 | 68 |
| q1 · step ~14,600 | 68.5 | 80.7 | 69.4 | 71.7 | 60.6 | 69 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | 只用负熵 · 全 MoE | DSV3 · 第 0 层 dense | bias + 负熵 · 全 MoE | DSV3 · 全 MoE | bias + 负熵 · 第 0 层 dense | 只用负熵 · 第 0 层 dense |
|---|---|---|---|---|---|---|
| q2 · step ~1,412 | 24.3 | 17.8 | 17.1 | 17 | 16.7 | 16.3 |
| q2 · step ~4,237 | 31.3 | 26.5 | 27.4 | 27.7 | 26.7 | 32.8 |
| q2 · step ~7,062 | 43.9 | 36.1 | 34 | 30.9 | 31.4 | 41.1 |
| q2 · step ~9,887 | 57.8 | 49.2 | 44.4 | 52.7 | 39.1 | 45.4 |
| q2 · step ~12,712 | 70.8 | 56.8 | 51.6 | 52.3 | 45.4 | 51.8 |
| q2 · step ~15,537 | 86.5 | 66.9 | 63.1 | 76.2 | 54.5 | 68.4 |
| q2 · step ~19,775 | 97.9 | 82.6 | 74.4 | 72.7 | 66.4 | 67.6 |
| q2 · step ~22,600 | 98.9 | 86.2 | 77.6 | 93.6 | 73 | 80 |
| q2 · step ~25,425 | 119 | 97.6 | 91.2 | 90.9 | 82.9 | 82.2 |
| q2 · step ~28,250 | 110 | 102 | 97 | 101 | 77.8 | 81.8 |
| q2 · step ~31,075 | 120 | 118 | 96.5 | 118 | 87.6 | 92.8 |
| q2 · step ~33,900 | 127 | 110 | 101 | 93.7 | 97.1 | 83.2 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | 只用负熵 · 第 0 层 dense | DSV3 · 第 0 层 dense | DSV3 · 全 MoE | bias + 负熵 · 第 0 层 dense | 只用负熵 · 全 MoE | bias + 负熵 · 全 MoE |
|---|---|---|---|---|---|---|
| q3 · step ~2,712 | 27.9 | 26.2 | 25 | 22.4 | 21.6 | 21.3 |
| q3 · step ~8,137 | 43.8 | 41.3 | 37.9 | 30.4 | 35.1 | 34.6 |
| q3 · step ~13,562 | 55.2 | 62.7 | 60.5 | 47.2 | 57.1 | 57.8 |
| q3 · step ~18,987 | 69.4 | 76.1 | 79.2 | 54.3 | 62 | 63.8 |
| q3 · step ~24,412 | 71.8 | 95.9 | 100 | 71.5 | 69.2 | 77.6 |
| q3 · step ~29,837 | 76.6 | 102 | 113 | 74.4 | 71.8 | 89.5 |
| q3 · step ~37,975 | 93 | 140 | 129 | 77.1 | 79.9 | 100 |
| q3 · step ~43,400 | 91 | 131 | 161 | 91.3 | 97.7 | 117 |
| q3 · step ~48,825 | 90.3 | 142 | 173 | 96.7 | 95.1 | 120 |
| q3 · step ~54,250 | 96.1 | 145 | 170 | 107 | 120 | 112 |
| q3 · step ~59,675 | 108 | 159 | 173 | 105 | 114 | 157 |
| q3 · step ~65,100 | 106 | 159 | 187 | 118 | 115 | 132 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | bias + 负熵 · 第 0 层 dense | DSV3 · 第 0 层 dense | 只用负熵 · 第 0 层 dense | 只用负熵 · 全 MoE | bias + 负熵 · 全 MoE | DSV3 · 全 MoE |
|---|---|---|---|---|---|---|
| q4 · step ~4,637 | 52.5 | 36.9 | 35.4 | 35.1 | 28.5 | 26.1 |
| q4 · step ~13,912 | 121 | 72.6 | 85.1 | 60 | 45.8 | 53.3 |
| q4 · step ~23,187 | 152 | 95.2 | 91.4 | 62.7 | 58 | 82.7 |
| q4 · step ~32,462 | 168 | 102 | 93 | 82.9 | 64.3 | 103 |
| q4 · step ~41,737 | 175 | 116 | 102 | 81.4 | 72.7 | 102 |
| q4 · step ~51,012 | 190 | 119 | 118 | 86.2 | 84.5 | 119 |
| q4 · step ~64,925 | 224 | 128 | 115 | 99.7 | 97.3 | 143 |
| q4 · step ~74,200 | 225 | 142 | 118 | 106 | 114 | 164 |
| q4 · step ~83,475 | 209 | 152 | 116 | 135 | 120 | 159 |
| q4 · step ~92,750 | 227 | 157 | 122 | 129 | 122 | 177 |
| q4 · step ~102,025 | 236 | 165 | 128 | 150 | 132 | 180 |
| q4 · step ~111,300 | 216 | 166 | 136 | 154 | 133 | 198 |

（共 24 个点，均匀取 12 个，含首尾）

所有层里最大的 |logit|（rank 0 的第一个 micro-batch）。

**〔视图：gate 更新幅度〕**

| 横轴 | DSV3 · 第 0 层 dense | DSV3 · 全 MoE | bias + 负熵 · 第 0 层 dense | bias + 负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | 只用负熵 · 全 MoE |
|---|---|---|---|---|---|---|
| q1 · step ~608 | 0.011 | 0.011 | 0.011 | 0.011 | 0.011 | 0.011 |
| q1 · step ~1,825 | 0.0102 | 0.0102 | 0.0102 | 0.0102 | 0.0102 | 0.0102 |
| q1 · step ~3,041 | 0.0093 | 0.0093 | 0.0093 | 0.0093 | 0.0093 | 0.0093 |
| q1 · step ~4,258 | 0.0084 | 0.0084 | 0.0084 | 0.0084 | 0.0084 | 0.0084 |
| q1 · step ~5,475 | 0.0075 | 0.0075 | 0.0075 | 0.0075 | 0.0075 | 0.0075 |
| q1 · step ~6,691 | 0.0067 | 0.0067 | 0.0067 | 0.0067 | 0.0067 | 0.0067 |
| q1 · step ~8,516 | 0.0053 | 0.0053 | 0.0053 | 0.0053 | 0.0053 | 0.0053 |
| q1 · step ~9,733 | 0.0045 | 0.0045 | 0.0045 | 0.0045 | 0.0045 | 0.0045 |
| q1 · step ~10,950 | 0.0037 | 0.0037 | 0.0037 | 0.0037 | 0.0037 | 0.0037 |
| q1 · step ~12,166 | 0.0029 | 0.0029 | 0.0029 | 0.0029 | 0.0029 | 0.0029 |
| q1 · step ~13,383 | 0.0021 | 0.002 | 0.0021 | 0.0021 | 0.0021 | 0.0021 |
| q1 · step ~14,600 | 0.0008 | 0.0008 | 0.0008 | 0.0008 | 0.0008 | 0.0008 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 · 第 0 层 dense | DSV3 · 全 MoE | bias + 负熵 · 第 0 层 dense | bias + 负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | 只用负熵 · 全 MoE |
|---|---|---|---|---|---|---|
| q2 · step ~1,412 | 0.0086 | 0.0086 | 0.0086 | 0.0086 | 0.0086 | 0.0086 |
| q2 · step ~4,237 | 0.0079 | 0.0079 | 0.0079 | 0.0079 | 0.0079 | 0.0079 |
| q2 · step ~7,062 | 0.0073 | 0.0073 | 0.0073 | 0.0073 | 0.0073 | 0.0073 |
| q2 · step ~9,887 | 0.0066 | 0.0066 | 0.0066 | 0.0066 | 0.0066 | 0.0066 |
| q2 · step ~12,712 | 0.0059 | 0.0059 | 0.0059 | 0.0059 | 0.0059 | 0.0059 |
| q2 · step ~15,537 | 0.0052 | 0.0052 | 0.0052 | 0.0052 | 0.0052 | 0.0052 |
| q2 · step ~19,775 | 0.0043 | 0.0043 | 0.0043 | 0.0043 | 0.0043 | 0.0043 |
| q2 · step ~22,600 | 0.0036 | 0.0036 | 0.0036 | 0.0036 | 0.0036 | 0.0036 |
| q2 · step ~25,425 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 |
| q2 · step ~28,250 | 0.0023 | 0.0023 | 0.0024 | 0.0024 | 0.0024 | 0.0023 |
| q2 · step ~31,075 | 0.0016 | 0.0016 | 0.0017 | 0.0016 | 0.0017 | 0.0016 |
| q2 · step ~33,900 | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 · 第 0 层 dense | DSV3 · 全 MoE | bias + 负熵 · 第 0 层 dense | bias + 负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | 只用负熵 · 全 MoE |
|---|---|---|---|---|---|---|
| q3 · step ~2,712 | 0.0071 | 0.0071 | 0.0071 | 0.0071 | 0.0071 | 0.0071 |
| q3 · step ~8,137 | 0.0066 | 0.0066 | 0.0066 | 0.0066 | 0.0066 | 0.0066 |
| q3 · step ~13,562 | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 | 0.006 |
| q3 · step ~18,987 | 0.0055 | 0.0055 | 0.0055 | 0.0055 | 0.0055 | 0.0055 |
| q3 · step ~24,412 | 0.0049 | 0.0049 | 0.005 | 0.005 | 0.005 | 0.005 |
| q3 · step ~29,837 | 0.0044 | 0.0044 | 0.0044 | 0.0044 | 0.0044 | 0.0044 |
| q3 · step ~37,975 | 0.0036 | 0.0036 | 0.0036 | 0.0036 | 0.0036 | 0.0036 |
| q3 · step ~43,400 | 0.0031 | 0.0031 | 0.0031 | 0.0031 | 0.0031 | 0.0031 |
| q3 · step ~48,825 | 0.0026 | 0.0025 | 0.0026 | 0.0026 | 0.0026 | 0.0026 |
| q3 · step ~54,250 | 0.002 | 0.002 | 0.0021 | 0.0021 | 0.0021 | 0.002 |
| q3 · step ~59,675 | 0.0014 | 0.0014 | 0.0015 | 0.0014 | 0.0015 | 0.0014 |
| q3 · step ~65,100 | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 | 0.0005 |

（共 24 个点，均匀取 12 个，含首尾）

| 横轴 | DSV3 · 第 0 层 dense | DSV3 · 全 MoE | bias + 负熵 · 第 0 层 dense | bias + 负熵 · 全 MoE | 只用负熵 · 第 0 层 dense | 只用负熵 · 全 MoE |
|---|---|---|---|---|---|---|
| q4 · step ~4,637 | 0.0056 | 0.0056 | 0.0056 | 0.0056 | 0.0056 | 0.0056 |
| q4 · step ~13,912 | 0.0052 | 0.0052 | 0.0052 | 0.0052 | 0.0052 | 0.0052 |
| q4 · step ~23,187 | 0.0048 | 0.0048 | 0.0048 | 0.0048 | 0.0048 | 0.0048 |
| q4 · step ~32,462 | 0.0044 | 0.0044 | 0.0044 | 0.0044 | 0.0044 | 0.0044 |
| q4 · step ~41,737 | 0.004 | 0.004 | 0.004 | 0.004 | 0.004 | 0.004 |
| q4 · step ~51,012 | 0.0036 | 0.0036 | 0.0036 | 0.0036 | 0.0036 | 0.0036 |
| q4 · step ~64,925 | 0.003 | 0.0029 | 0.003 | 0.003 | 0.003 | 0.003 |
| q4 · step ~74,200 | 0.0026 | 0.0025 | 0.0026 | 0.0026 | 0.0026 | 0.0026 |
| q4 · step ~83,475 | 0.0021 | 0.0021 | 0.0022 | 0.0022 | 0.0022 | 0.0022 |
| q4 · step ~92,750 | 0.0017 | 0.0017 | 0.0017 | 0.0017 | 0.0017 | 0.0017 |
| q4 · step ~102,025 | 0.0012 | 0.0012 | 0.0012 | 0.0012 | 0.0012 | 0.0012 |
| q4 · step ~111,300 | 0.0004 | 0.0004 | 0.0004 | 0.0004 | 0.0004 | 0.0004 |

（共 24 个点，均匀取 12 个，含首尾）

router 权重这一步的变化量 ÷ 权重大小，各层平均（Boxiang 只作参考，不作判据）。

dense 第 0 层主要救的是只用负熵那组：q1 / q2 / q3 训练结束时 dead expert 从 0.26 / 0.68 / 0.65% 降到 0，MaxVio（最差层）从 8.5 / 9.3 / 6.5 降到 3.4 / 2.4 / 4.4，但仍远高于 DSV3 的约 0.3。不过训练过程中，第 1 层（dense 之后的第一个 MoE 层）仍有 dead expert：q2 前半段约 11–12 个，q3 大部分时间 18–25 个，到训练末尾才降到 0。DSV3 和 bias + 负熵加不加 dense 都健康（dead 0，MaxVio 最差层 ≤ 0.36）。
横轴是训练步数，每个点是该步所有 MoE 层的汇总（层平均 / 最差层），不是按层画的。q4 浅橙色实线（bias + 负熵 · 第 0 层 dense）在第 ~44,200 步的尖峰来自最后两层（L34 / L35，MaxVio 约 8.5，L35 有 4 个 dead expert），约 200 步后恢复；同一步只用负熵的 dense run 也跳了一下（来自第 1 层），两个全 MoE run 没有。

### Router 选择，SWS 架构，更杂的实验

0 线 = **DSV3 SWS**（MUONH_FIT_DSV3_SWS）：MuonH，新 LR 公式（fit），纯 CE（不用 KD），128 选 8 MoE，DSV3 router（sigmoid 打分 + expert bias + seq aux 1e-4，gate 用 AdamH、LR ×1），SWA（每 6 层 5 层窗口 2048）+ attention sink。

| 类别 | 线 | 名称 | 具体改动（相对 DSV3 SWS） | 有数据的 rung |
|---|---|---|---|---|
| 负载均衡 |   | 去掉 seq aux（只靠 bias） | 去掉 seq aux 1e-4，只靠 expert bias 均衡（DeepSeek-V3 的“无 aux loss”做法）。 | q0, q1, q2, q3 |
| 负载均衡 |   | 去掉 bias（只靠 seq aux） | 关掉 expert bias，只靠 seq aux 1e-4 均衡（和上一行互为镜像）。 | q0, q1, q2, q3 |
| 负载均衡 |   | 都去掉 | bias 和 aux 都去掉，完全不做负载均衡；作为下限参照，预计会出现 dead expert。 | q0 |
| 负载均衡 |   | aux 按 micro-batch | 按序列的 seq aux 换成按 micro-batch（32K token）统计的 Switch aux，系数仍是 1e-4；bias 保留。测的是统计范围（文献：micro-batch 最差，序列其次，global batch 最好）。 | q0 |
| 负载均衡 |   | seq aux ×10 | seq aux 系数加大 10 倍（1e-4 → 1e-3），bias 保留。 | q0, q1, q2, q3 |
| bias γ / 打分 |   | γ 1e-4 | expert bias 每步的调整量 γ 从 1e-3 降到 1e-4（调得慢 10 倍）。 | q0 |
| bias γ / 打分 |   | γ 1e-2 | γ 从 1e-3 升到 1e-2（调得快 10 倍；DeepSeek-V3 也扫过 γ）。 | q0 |
| bias γ / 打分 |   | softmax 打分 | 打分从 sigmoid 换成 softmax（选出 top-8 后在这 8 个里重新归一化），bias 和 seq aux 保留（LongCat 式 softmax + bias）。 | 还在训练 / 评测 |
| 缩放 / shared / 正则 |   | routed scaling 2.5 | 被选中 8 个 expert 的权重（加起来 = 1）整体乘 2.5 再相加，即 routed expert 的输出放大 2.5 倍（DeepSeek-V3 的取值）；不加 shared expert。 | q0 |
| 缩放 / shared / 正则 |   | +1 shared expert | 每层加 1 个 shared expert：所有 token 都过它，宽度和一个 routed expert 相同（3h/8，q0 是 192）；权重缩放保持 1。不是等参数，激活参数约多 7–8%。 | q0 |
| 缩放 / shared / 正则 |   | shared + scaling 2.5 | 上面两项一起加：1 个 shared expert + routed scaling 2.5，也就是完整的 DeepSeek-V3 MoE 设置。不是等参数，激活参数约多 7–8%。 | q0, q1, q2, q3 |
| 缩放 / shared / 正则 |   | z-loss 1e-3 | 对 router logit 加 z-loss 1e-3（ST-MoE / OLMoE 的做法）：惩罚 logit 的 log-sum-exp 的平方，防止 logit 变得太大；其余不变。 | q0 |
| gate 优化器 / LR |   | router LR ×0.19 | gate（router 权重）仍用 AdamH，LR 降到主体 LR 的 0.19 倍。 | q0 |
| gate 优化器 / LR |   | router LR ×0.5 | gate 用 AdamH，LR = 主体 LR × 0.5。 | q0, q1, q2, q3 |
| gate 优化器 / LR |   | router LR ×2 | gate 用 AdamH，LR = 主体 LR × 2。 | q0, q1, q2, q3 |
| gate 优化器 / LR |   | MuonH gate | gate 改用 MuonH 更新（在球面上），LR 同主体；GLM-4.5 是唯一公开在 router 上用 Muon 的模型。 | q0, q1, q2, q3 |
| gate 优化器 / LR |   | AdamW gate ×0.19 | gate 改用普通 AdamW（wd 0），LR = 主体 LR × 0.19（AdamW 规则下 router 的 LR）；就是全注意力上 B2 那组的设置搬到 SWS 上。 | q0, q1 |

**各组 − DSV3 SWS（鼠标移到某个 rung 看所有组的数值）**

**〔指标：overall_tsqw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| z-loss 1e-3 | 0.0294 | · | · | · |
| seq aux ×10 | 0.0098 | 0.0127 | 0.0115 | 0.0126 |
| MuonH gate | 0.0090 | -0.0041 | 0.0017 | 0.0026 |
| router LR ×0.19 | 0.0083 | · | · | · |
| router LR ×2 | 0.0053 | 0.0034 | 0.0024 | 0.0035 |
| routed scaling 2.5 | 0.0043 | · | · | · |
| AdamW gate ×0.19 | 0.0042 | 0.0060 | · | · |
| γ 1e-2 | 0.0024 | · | · | · |
| router LR ×0.5 | 0.0023 | -0.0021 | 0.0092 | 0.0011 |
| γ 1e-4 | 0.0018 | · | · | · |
| aux 按 micro-batch | 0.0008 | · | · | · |
| 去掉 seq aux（只靠 bias） | -0.0007 | -0.0010 | 0.0003 | -0.0048 |
| 都去掉 | -0.0032 | · | · | · |
| 去掉 bias（只靠 seq aux） | -0.0040 | -0.0013 | -0.0048 | -0.0013 |
| shared + scaling 2.5 | -0.0185 | -0.0268 | -0.0259 | -0.0249 |
| +1 shared expert | -0.0221 | · | · | · |

**〔指标：overall_tw〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| z-loss 1e-3 | 0.0318 | · | · | · |
| seq aux ×10 | 0.0103 | 0.0136 | 0.0123 | 0.0133 |
| MuonH gate | 0.0103 | -0.0040 | 0.0023 | 0.0027 |
| router LR ×0.19 | 0.0085 | · | · | · |
| router LR ×2 | 0.0054 | 0.0046 | 0.0031 | 0.0036 |
| routed scaling 2.5 | 0.0048 | · | · | · |
| AdamW gate ×0.19 | 0.0042 | 0.0067 | · | · |
| γ 1e-2 | 0.0027 | · | · | · |
| router LR ×0.5 | 0.0024 | -0.0024 | 0.0107 | 0.0009 |
| γ 1e-4 | 0.0020 | · | · | · |
| aux 按 micro-batch | 0.0010 | · | · | · |
| 去掉 seq aux（只靠 bias） | -0.0001 | -0.0008 | 0.0008 | -0.0050 |
| 都去掉 | -0.0029 | · | · | · |
| 去掉 bias（只靠 seq aux） | -0.0034 | -0.0009 | -0.0044 | -0.0015 |
| shared + scaling 2.5 | -0.0181 | -0.0271 | -0.0264 | -0.0259 |
| +1 shared expert | -0.0214 | · | · | · |

**〔指标：overall_macro〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| z-loss 1e-3 | 0.0247 | · | · | · |
| seq aux ×10 | 0.0093 | 0.0120 | 0.0105 | 0.0117 |
| router LR ×0.19 | 0.0085 | · | · | · |
| MuonH gate | 0.0057 | -0.0039 | 0.0011 | 0.0026 |
| router LR ×2 | 0.0051 | 0.0019 | 0.0014 | 0.0034 |
| routed scaling 2.5 | 0.0036 | · | · | · |
| AdamW gate ×0.19 | 0.0034 | 0.0053 | · | · |
| router LR ×0.5 | 0.0025 | -0.0013 | 0.0073 | 0.0015 |
| γ 1e-2 | 0.0019 | · | · | · |
| γ 1e-4 | 0.0014 | · | · | · |
| aux 按 micro-batch | 0.0006 | · | · | · |
| 去掉 seq aux（只靠 bias） | -0.0024 | -0.0011 | -0.0004 | -0.0046 |
| 都去掉 | -0.0033 | · | · | · |
| 去掉 bias（只靠 seq aux） | -0.0048 | -0.0020 | -0.0051 | -0.0010 |
| shared + scaling 2.5 | -0.0193 | -0.0263 | -0.0255 | -0.0239 |
| +1 shared expert | -0.0230 | · | · | · |

**〔指标：val_v2〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| z-loss 1e-3 | 0.0342 | · | · | · |
| MuonH gate | 0.0111 | -0.0036 | 0.0020 | 0.0017 |
| seq aux ×10 | 0.0080 | 0.0100 | 0.0088 | 0.0096 |
| router LR ×0.19 | 0.0074 | · | · | · |
| router LR ×2 | 0.0074 | 0.0035 | 0.0020 | 0.0027 |
| AdamW gate ×0.19 | 0.0053 | 0.0044 | · | · |
| routed scaling 2.5 | 0.0041 | · | · | · |
| γ 1e-2 | 0.0032 | · | · | · |
| γ 1e-4 | 0.0031 | · | · | · |
| router LR ×0.5 | 0.0020 | -0.0017 | 0.0078 | 0.0012 |
| aux 按 micro-batch | 0.0011 | · | · | · |
| 去掉 seq aux（只靠 bias） | 0.0004 | 0.0002 | 0.0014 | -0.0032 |
| 都去掉 | -0.0000 | · | · | · |
| 去掉 bias（只靠 seq aux） | -0.0011 | 0.0001 | -0.0032 | -0.0007 |
| shared + scaling 2.5 | -0.0133 | -0.0198 | -0.0187 | -0.0182 |
| +1 shared expert | -0.0183 | · | · | · |

**〔指标：uncheatable〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| z-loss 1e-3 | 0.0381 | · | · | · |
| MuonH gate | 0.0121 | -0.0044 | 0.0036 | 0.0033 |
| seq aux ×10 | 0.0114 | 0.0161 | 0.0147 | 0.0154 |
| router LR ×0.19 | 0.0097 | · | · | · |
| router LR ×2 | 0.0067 | 0.0076 | 0.0051 | 0.0042 |
| routed scaling 2.5 | 0.0061 | · | · | · |
| AdamW gate ×0.19 | 0.0040 | 0.0088 | · | · |
| router LR ×0.5 | 0.0032 | -0.0028 | 0.0151 | 0.0006 |
| γ 1e-2 | 0.0032 | · | · | · |
| γ 1e-4 | 0.0031 | · | · | · |
| aux 按 micro-batch | 0.0019 | · | · | · |
| 去掉 seq aux（只靠 bias） | 0.0005 | -0.0002 | 0.0022 | -0.0060 |
| 都去掉 | -0.0014 | · | · | · |
| 去掉 bias（只靠 seq aux） | -0.0020 | 0.0002 | -0.0037 | -0.0020 |
| shared + scaling 2.5 | -0.0176 | -0.0288 | -0.0288 | -0.0293 |
| +1 shared expert | -0.0203 | · | · | · |

**〔指标：codefresh〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| seq aux ×10 | 0.0143 | 0.0170 | 0.0136 | 0.0153 |
| router LR ×0.19 | 0.0132 | · | · | · |
| AdamW gate ×0.19 | 0.0058 | 0.0040 | · | · |
| routed scaling 2.5 | 0.0058 | · | · | · |
| router LR ×0.5 | 0.0056 | -0.0032 | 0.0017 | 0.0048 |
| MuonH gate | 0.0052 | -0.0052 | -0.0006 | 0.0049 |
| z-loss 1e-3 | 0.0040 | · | · | · |
| γ 1e-4 | 0.0005 | · | · | · |
| router LR ×2 | -0.0000 | -0.0049 | -0.0028 | 0.0049 |
| γ 1e-2 | -0.0006 | · | · | · |
| aux 按 micro-batch | -0.0009 | · | · | · |
| 去掉 seq aux（只靠 bias） | -0.0013 | -0.0093 | -0.0077 | -0.0060 |
| 去掉 bias（只靠 seq aux） | -0.0116 | -0.0085 | -0.0096 | -0.0011 |
| 都去掉 | -0.0163 | · | · | · |
| shared + scaling 2.5 | -0.0277 | -0.0339 | -0.0296 | -0.0218 |
| +1 shared expert | -0.0317 | · | · | · |

**〔指标：private_internal〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| z-loss 1e-3 | 0.0258 | · | · | · |
| seq aux ×10 | 0.0087 | 0.0112 | 0.0076 | 0.0090 |
| router LR ×0.19 | 0.0084 | · | · | · |
| MuonH gate | 0.0071 | -0.0037 | 0.0014 | 0.0019 |
| routed scaling 2.5 | 0.0030 | · | · | · |
| router LR ×2 | 0.0029 | 0.0021 | 0.0017 | 0.0019 |
| router LR ×0.5 | 0.0024 | 0.0007 | 0.0079 | 0.0017 |
| AdamW gate ×0.19 | 0.0019 | 0.0055 | · | · |
| aux 按 micro-batch | 0.0018 | · | · | · |
| γ 1e-2 | 0.0014 | · | · | · |
| γ 1e-4 | 0.0008 | · | · | · |
| 去掉 seq aux（只靠 bias） | -0.0002 | 0.0019 | 0.0003 | -0.0045 |
| 都去掉 | -0.0052 | · | · | · |
| 去掉 bias（只靠 seq aux） | -0.0071 | -0.0016 | -0.0054 | -0.0017 |
| shared + scaling 2.5 | -0.0167 | -0.0229 | -0.0250 | -0.0231 |
| +1 shared expert | -0.0220 | · | · | · |

### Latent MoE

0 线 = **PROD1_DPSK**（PROD v1）：SWA 512 局部:全局 5:1 + sink，DSV3 router，第 0 层 dense，8 + 1 shared expert，32K 训练 + 文档间 mask，MuonH，纯 CE
latent MoE（Kimi K3 的形式）：MoE 先把 h 维投影到 d = h/2，routed expert 在 d 维上算，RMSNorm 后再投影回 h；router 和 shared expert 仍在 h 维。
数值 = 该线 − PROD1_DPSK（我们的 115 split doc-isolated 评测，seed 1）。这些 run 都是 32K 训练：每个 checkpoint 只 load 一次，同一个 job 出 8K 窗口和 32K 窗口两份分数（图上方 tab 切换）。

|   | 名称 | 改动 | 有数据的 rung |
|---|---|---|---|
|   | latent MoE（等参数） | PROD1_DPSK_LATENT2：d = h/2，256 个 routed expert（宽 3h/8）选 16 个；总参数、routed FLOP、all-to-all 都和 0 线相同（只多两个投影，约每层 8%） | q0 8K, q0 32K, q1 8K, q1 32K, q2 8K, q2 32K, q3 8K, q3 32K |
|   | latent MoE，expert ×2 | PROD1_DPSK_LATENT2_E2X：在上一条基础上把 routed expert 翻倍到 512 个，仍选 16 个。每个 token 的计算量不变，但 routed 参数 ×2（不是等参数） | q0 8K, q0 32K, q1 8K, q1 32K, q2 8K, q2 32K |

**各线 − PROD1_DPSK（越负越好；鼠标移到某个 rung 看数值）**

**〔指标：overall_tsqw〕**

**〔视图：8K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| latent MoE（等参数） | -0.0091 | -0.0140 | -0.0192 | -0.0187 |
| latent MoE，expert ×2 | -0.0458 | -0.0495 | -0.0456 | · |

**〔视图：32K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| latent MoE（等参数） | -0.0095 | -0.0139 | -0.0191 | -0.0185 |
| latent MoE，expert ×2 | -0.0453 | -0.0487 | -0.0448 | · |

**〔指标：overall_tw〕**

**〔视图：8K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| latent MoE（等参数） | -0.0096 | -0.0139 | -0.0199 | -0.0194 |
| latent MoE，expert ×2 | -0.0462 | -0.0503 | -0.0463 | · |

**〔视图：32K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| latent MoE（等参数） | -0.0099 | -0.0138 | -0.0198 | -0.0192 |
| latent MoE，expert ×2 | -0.0457 | -0.0495 | -0.0456 | · |

**〔指标：overall_macro〕**

**〔视图：8K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| latent MoE（等参数） | -0.0094 | -0.0141 | -0.0184 | -0.0181 |
| latent MoE，expert ×2 | -0.0452 | -0.0478 | -0.0440 | · |

**〔视图：32K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| latent MoE（等参数） | -0.0097 | -0.0140 | -0.0183 | -0.0180 |
| latent MoE，expert ×2 | -0.0449 | -0.0472 | -0.0433 | · |

**〔指标：val_v2〕**

**〔视图：8K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| latent MoE（等参数） | -0.0067 | -0.0102 | -0.0131 | -0.0139 |
| latent MoE，expert ×2 | -0.0385 | -0.0383 | -0.0355 | · |

**〔视图：32K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| latent MoE（等参数） | -0.0072 | -0.0097 | -0.0130 | -0.0133 |
| latent MoE，expert ×2 | -0.0376 | -0.0367 | -0.0341 | · |

**〔指标：uncheatable〕**

**〔视图：8K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| latent MoE（等参数） | -0.0115 | -0.0142 | -0.0220 | -0.0217 |
| latent MoE，expert ×2 | -0.0483 | -0.0523 | -0.0483 | · |

**〔视图：32K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| latent MoE（等参数） | -0.0116 | -0.0142 | -0.0219 | -0.0216 |
| latent MoE，expert ×2 | -0.0479 | -0.0519 | -0.0477 | · |

**〔指标：codefresh〕**

**〔视图：8K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| latent MoE（等参数） | -0.0101 | -0.0186 | -0.0209 | -0.0173 |
| latent MoE，expert ×2 | -0.0521 | -0.0573 | -0.0504 | · |

**〔视图：32K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| latent MoE（等参数） | -0.0101 | -0.0186 | -0.0209 | -0.0173 |
| latent MoE，expert ×2 | -0.0521 | -0.0573 | -0.0504 | · |

**〔指标：private_internal〕**

**〔视图：8K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| latent MoE（等参数） | -0.0083 | -0.0118 | -0.0168 | -0.0170 |
| latent MoE，expert ×2 | -0.0377 | -0.0385 | -0.0344 | · |

**〔视图：32K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| latent MoE（等参数） | -0.0083 | -0.0118 | -0.0168 | -0.0170 |
| latent MoE，expert ×2 | -0.0377 | -0.0385 | -0.0343 | · |

### 位置编码：NoPE / RoPE / partial RoPE

0 线 = **PROD1_DPSK**（PROD v1）：SWA 层用完整 RoPE（128 维全部旋转），全局层（每 6 层 1 层）NoPE；其余同上一章的 0 线
partial RoPE：每个 head 的 128 维里只有 32 维（1/4）加 RoPE（NeoX 排法，base 1e4），其余 96 维不带位置。
数值 = 该线 − PROD1_DPSK（我们的 115 split doc-isolated 评测，seed 1），8K / 32K 两个窗口同上一章。

|   | 名称 | 改动 | 有数据的 rung |
|---|---|---|---|
|   | SWA 层 1/4 RoPE | PROD1_DPSK_PROPE25：只改 SWA 层，RoPE 1 → 1/4；全局层仍是 NoPE（单变量） | q0 8K, q0 32K, q1 8K, q1 32K, q2 8K, q2 32K |
|   | 所有层 1/4 RoPE | PROD1_DPSK_PROPE25G：SWA 层 RoPE 1 → 1/4，全局层 NoPE → 1/4 RoPE（改了两处；和上一条之差 = 全局层 NoPE vs 1/4 RoPE） | q0 8K, q0 32K, q1 8K, q1 32K, q2 8K, q2 32K |

**各线 − PROD1_DPSK（越负越好；鼠标移到某个 rung 看数值）**

**〔指标：overall_tsqw〕**

**〔视图：8K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M |
|---|---|---|---|
| 所有层 1/4 RoPE | -0.0002 | -0.0008 | -0.0035 |
| SWA 层 1/4 RoPE | -0.0010 | 0.0006 | -0.0023 |

**〔视图：32K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M |
|---|---|---|---|
| 所有层 1/4 RoPE | 0.0002 | -0.0007 | -0.0034 |
| SWA 层 1/4 RoPE | -0.0005 | 0.0006 | -0.0022 |

**〔指标：overall_tw〕**

**〔视图：8K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M |
|---|---|---|---|
| 所有层 1/4 RoPE | -0.0000 | -0.0009 | -0.0037 |
| SWA 层 1/4 RoPE | -0.0011 | 0.0006 | -0.0021 |

**〔视图：32K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M |
|---|---|---|---|
| 所有层 1/4 RoPE | 0.0004 | -0.0007 | -0.0036 |
| SWA 层 1/4 RoPE | -0.0007 | 0.0006 | -0.0021 |

**〔指标：overall_macro〕**

**〔视图：8K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M |
|---|---|---|---|
| 所有层 1/4 RoPE | -0.0009 | -0.0011 | -0.0033 |
| SWA 层 1/4 RoPE | -0.0012 | 0.0006 | -0.0026 |

**〔视图：32K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M |
|---|---|---|---|
| 所有层 1/4 RoPE | -0.0006 | -0.0010 | -0.0032 |
| SWA 层 1/4 RoPE | -0.0009 | 0.0006 | -0.0025 |

**〔指标：val_v2〕**

**〔视图：8K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M |
|---|---|---|---|
| 所有层 1/4 RoPE | 0.0000 | -0.0009 | -0.0022 |
| SWA 层 1/4 RoPE | -0.0007 | 0.0008 | -0.0018 |

**〔视图：32K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M |
|---|---|---|---|
| 所有层 1/4 RoPE | 0.0004 | -0.0004 | -0.0023 |
| SWA 层 1/4 RoPE | 0.0002 | 0.0010 | -0.0017 |

**〔指标：uncheatable〕**

**〔视图：8K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M |
|---|---|---|---|
| 所有层 1/4 RoPE | 0.0004 | -0.0011 | -0.0046 |
| SWA 层 1/4 RoPE | -0.0013 | 0.0004 | -0.0018 |

**〔视图：32K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M |
|---|---|---|---|
| 所有层 1/4 RoPE | 0.0009 | -0.0010 | -0.0045 |
| SWA 层 1/4 RoPE | -0.0008 | 0.0004 | -0.0018 |

**〔指标：codefresh〕**

**〔视图：8K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M |
|---|---|---|---|
| 所有层 1/4 RoPE | 0.0002 | -0.0031 | -0.0042 |
| SWA 层 1/4 RoPE | -0.0021 | -0.0017 | -0.0041 |

**〔视图：32K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M |
|---|---|---|---|
| 所有层 1/4 RoPE | 0.0002 | -0.0031 | -0.0042 |
| SWA 层 1/4 RoPE | -0.0021 | -0.0017 | -0.0041 |

**〔指标：private_internal〕**

**〔视图：8K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M |
|---|---|---|---|
| 所有层 1/4 RoPE | -0.0002 | -0.0010 | -0.0035 |
| SWA 层 1/4 RoPE | -0.0029 | 0.0007 | -0.0026 |

**〔视图：32K 窗口〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M |
|---|---|---|---|
| 所有层 1/4 RoPE | -0.0003 | -0.0011 | -0.0035 |
| SWA 层 1/4 RoPE | -0.0030 | 0.0007 | -0.0026 |

## 页签：外部参考

### Pretrain doc-isolated 测试

**〔窗口：window = 8K〕**

**〔视图：bpb〕**

|   | model size | vocab size | overall_tsqw | overall_tw | overall_macro | val_v2 | uncheatable | codefresh | v3small | private_internal | pile_test | pile_10k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiMo-V2.5-Pro-Base | 1.02T-A42B | 152,576 | 0.4090 | 0.4076 | 0.4269 | 0.3354 | 0.4128 | 0.1859 | 0.6494 | 0.4881 | 0.4899 | 0.4617 |
| MiMo-V2.6-Pro-RL | 1.02T-A42B | 152,576 | 0.4289 | 0.4254 | 0.4468 | 0.3587 | 0.4238 | 0.2033 | 0.6724 | 0.5037 | 0.5191 | 0.4882 |
| Qwen3-30B-A3B-Base | 30B-A3B | 151,936 | 0.4976 | 0.4926 | 0.5132 | 0.4237 | 0.4838 | 0.2716 | 0.7567 | 0.5413 | 0.6024 | 0.5685 |
| Qwen3-VL-30B-A3B-Instruct | 30B-A3B | 151,936 | 0.5242 | 0.5200 | 0.5406 | 0.4479 | 0.5138 | 0.2853 | 0.7947 | 0.5719 | 0.6299 | 0.5939 |
| Qwen3-VL-30B-A3B-Thinking | 30B-A3B | 151,936 | 0.5180 | 0.5125 | 0.5346 | 0.4447 | 0.5025 | 0.2825 | 0.7890 | 0.5624 | 0.6265 | 0.5909 |
| Qwen3-235B-A22B (2504) | 235B-A22B | 151,936 | 0.4940 | 0.4900 | 0.5135 | 0.4114 | 0.4860 | 0.2677 | 0.7652 | 0.5577 | 0.5928 | 0.5543 |
| Qwen3-235B-A22B-Instruct-2507 | 235B-A22B | 151,936 | 0.4692 | 0.4661 | 0.4898 | 0.3853 | 0.4660 | 0.2498 | 0.7396 | 0.5385 | 0.5597 | 0.5227 |
| Qwen3-235B-A22B-Thinking-2507 | 235B-A22B | 151,936 | 0.4649 | 0.4612 | 0.4843 | 0.3849 | 0.4584 | 0.2481 | 0.7297 | 0.5269 | 0.5580 | 0.5218 |
| Qwen3-VL-235B-A22B-Instruct | 235B-A22B | 151,936 | 0.4807 | 0.4781 | 0.4999 | 0.3981 | 0.4785 | 0.2598 | 0.7456 | 0.5459 | 0.5716 | 0.5361 |
| Qwen3-VL-235B-A22B-Thinking | 235B-A22B | 151,936 | 0.4744 | 0.4707 | 0.4930 | 0.3951 | 0.4678 | 0.2554 | 0.7364 | 0.5348 | 0.5685 | 0.5332 |
| DeepSeek-V4-Flash-Base | 284B-A13B | 129,280 | 0.4227 | 0.4196 | 0.4421 | 0.3552 | 0.4174 | 0.2067 | 0.6680 | 0.4949 | 0.5102 | 0.4708 |
| Kimi-K3 | 2.8T-A104B | 163,840 | 0.3690 | 0.3706 | 0.3915 | 0.3008 | 0.3872 | 0.1549 | 0.6074 | 0.4730 | 0.4258 | 0.3936 |
| Kimi-K3.1-Pro (MXFP4) | 2.8T-A104B | 163,840 | 0.3677 | 0.3696 | 0.3907 | 0.2992 | 0.3867 | 0.1526 | 0.6084 | 0.4733 | 0.4239 | 0.3904 |
| Nemotron-3-Ultra-Base | 550B-A55B | 131,072 | 0.4319 | 0.4264 | 0.4477 | 0.3593 | 0.4171 | 0.2383 | 0.6566 | 0.4896 | 0.5255 | 0.4954 |
| Nemotron-3.5-Super-VL-Base | 120B-A12B | 131,072 | 0.4602 | 0.4537 | 0.4747 | 0.3889 | 0.4396 | 0.2581 | 0.6893 | 0.5077 | 0.5645 | 0.5312 |
| Qwen3.8-Flash-Next (FP8) | 125B-A6B | 248,320 | 0.4837 | 0.4742 | 0.5031 | 0.4227 | 0.4566 | 0.2404 | 0.7640 | 0.5393 | 0.5959 | 0.5681 |
| Qwen3.8-27B | 27B | 248,320 | 0.5030 | 0.4902 | 0.5203 | 0.4539 | 0.4594 | 0.2655 | 0.7784 | 0.5349 | 0.6298 | 0.5969 |
| Qwen3.8-Max (FP8) | 2.4T-A95B | 248,320 | 0.4331 | 0.4264 | 0.4525 | 0.3709 | 0.4185 | 0.2120 | 0.6890 | 0.5000 | 0.5247 | 0.5014 |
| Mistral-Large-4 (FP8) | 1T-A52B | 204,800 | 0.4904 | 0.4787 | 0.5172 | 0.4223 | 0.4614 | 0.2386 | 0.8363 | 0.5433 | 0.6010 | 0.5750 |
| Flint（可能不对） | 40B-A2.5B | 150,144 | 0.7811 | 0.7807 | 0.7826 | 0.6872 | 0.7812 | 0.4637 | 1.0556 | 0.7894 | 0.9188 | 0.9009 |
| Ripple-65B-Base-Preview | 65B | 100,352 | 0.4623 | 0.4578 | 0.4761 | 0.3889 | 0.4510 | 0.2626 | 0.6891 | 0.5093 | 0.5558 | 0.5291 |
| 我们最好的: PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K_W32K · q3 (683M) | · | · | 0.5826 | 0.5793 | 0.5913 | 0.4887 | 0.5649 | 0.3733 | 0.8154 | 0.5975 | 0.7081 | 0.6560 |

**〔视图：val_loss〕**

|   | model size | vocab size | overall_tsqw | overall_tw | overall_macro | val_v2 | uncheatable | codefresh | v3small | private_internal | pile_test | pile_10k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiMo-V2.5-Pro-Base | 1.02T-A42B | 152,576 | 1.0520 | 1.0374 | 1.1438 | 0.8251 | 1.0632 | 0.5086 | 1.8213 | 1.4541 | 1.1903 | 1.1795 |
| MiMo-V2.6-Pro-RL | 1.02T-A42B | 152,576 | 1.1026 | 1.0827 | 1.1962 | 0.8824 | 1.0913 | 0.5562 | 1.8856 | 1.5005 | 1.2613 | 1.2472 |
| Qwen3-30B-A3B-Base | 30B-A3B | 151,936 | 1.2788 | 1.2537 | 1.3716 | 1.0423 | 1.2460 | 0.7429 | 2.1220 | 1.6125 | 1.4637 | 1.4524 |
| Qwen3-VL-30B-A3B-Instruct | 30B-A3B | 151,936 | 1.3471 | 1.3234 | 1.4447 | 1.1019 | 1.3232 | 0.7806 | 2.2286 | 1.7036 | 1.5306 | 1.5172 |
| Qwen3-VL-30B-A3B-Thinking | 30B-A3B | 151,936 | 1.3311 | 1.3043 | 1.4286 | 1.0939 | 1.2942 | 0.7727 | 2.2127 | 1.6756 | 1.5222 | 1.5095 |
| Qwen3-235B-A22B (2504) | 235B-A22B | 151,936 | 1.2702 | 1.2470 | 1.3740 | 1.0119 | 1.2517 | 0.7325 | 2.1460 | 1.6614 | 1.4405 | 1.4161 |
| Qwen3-235B-A22B-Instruct-2507 | 235B-A22B | 151,936 | 1.2069 | 1.1861 | 1.3116 | 0.9477 | 1.2002 | 0.6834 | 2.0741 | 1.6042 | 1.3600 | 1.3353 |
| Qwen3-235B-A22B-Thinking-2507 | 235B-A22B | 151,936 | 1.1957 | 1.1737 | 1.2963 | 0.9468 | 1.1805 | 0.6788 | 2.0464 | 1.5698 | 1.3558 | 1.3331 |
| Qwen3-VL-235B-A22B-Instruct | 235B-A22B | 151,936 | 1.2365 | 1.2167 | 1.3381 | 0.9793 | 1.2324 | 0.7108 | 2.0909 | 1.6262 | 1.3890 | 1.3696 |
| Qwen3-VL-235B-A22B-Thinking | 235B-A22B | 151,936 | 1.2199 | 1.1980 | 1.3192 | 0.9720 | 1.2047 | 0.6988 | 2.0652 | 1.5934 | 1.3813 | 1.3620 |
| DeepSeek-V4-Flash-Base † | 284B-A13B | 129,280 | 1.1280 | 1.1070 | 1.2113 | 0.9272 | 1.1172 | 0.5386 | 1.9526 | 1.4256 | 1.3066 | 1.2866 |
| Kimi-K3 † | 2.8T-A104B | 163,840 | 0.9942 | 0.9839 | 1.0960 | 0.7897 | 1.0317 | 0.4286 | 1.7893 | 1.4398 | 1.0967 | 1.0690 |
| Kimi-K3.1-Pro (MXFP4) † | 2.8T-A104B | 163,840 | 0.9908 | 0.9811 | 1.0939 | 0.7856 | 1.0306 | 0.4221 | 1.7925 | 1.4406 | 1.0920 | 1.0602 |
| Nemotron-3-Ultra-Base † | 550B-A55B | 131,072 | 1.0966 | 1.0751 | 1.1719 | 0.8932 | 1.0743 | 0.6131 | 1.7996 | 1.3954 | 1.2558 | 1.2642 |
| Nemotron-3.5-Super-VL-Base † | 120B-A12B | 131,072 | 1.1679 | 1.1440 | 1.2414 | 0.9667 | 1.1323 | 0.6641 | 1.8893 | 1.4471 | 1.3489 | 1.3554 |
| Qwen3.8-Flash-Next (FP8) † | 125B-A6B | 248,320 | 1.2205 | 1.1864 | 1.3134 | 1.0178 | 1.1661 | 0.6108 | 2.1157 | 1.5392 | 1.4311 | 1.4385 |
| Qwen3.8-27B † | 27B | 248,320 | 1.2678 | 1.2264 | 1.3560 | 1.0928 | 1.1732 | 0.6747 | 2.1556 | 1.5268 | 1.5127 | 1.5114 |
| Qwen3.8-Max (FP8) † | 2.4T-A95B | 248,320 | 1.0936 | 1.0670 | 1.1826 | 0.8930 | 1.0687 | 0.5385 | 1.9082 | 1.4270 | 1.2601 | 1.2696 |
| Mistral-Large-4 (FP8) † | 1T-A52B | 204,800 | 1.3097 | 1.2632 | 1.4257 | 1.1092 | 1.2240 | 0.6417 | 2.4208 | 1.6148 | 1.5435 | 1.5505 |
| Flint（可能不对） † | 40B-A2.5B | 150,144 | 2.1582 | 2.1305 | 2.2271 | 1.8724 | 2.1410 | 1.2905 | 3.1794 | 2.4390 | 2.4404 | 2.5442 |
| Ripple-65B-Base-Preview † | 65B | 100,352 | 1.2177 | 1.1771 | 1.3110 | 0.9994 | 1.1368 | 0.7267 | 2.0246 | 1.5511 | 1.4276 | 1.4258 |
| 我们最好的: PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K_W32K · q3 (683M) | · | · | 1.4969 | 1.4743 | 1.5776 | 1.2022 | 1.4549 | 1.0212 | 2.2868 | 1.7799 | 1.7205 | 1.6758 |

**〔视图：ppl〕**

|   | model size | vocab size | overall_tsqw | overall_tw | overall_macro | val_v2 | uncheatable | codefresh | v3small | private_internal | pile_test | pile_10k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiMo-V2.5-Pro-Base | 1.02T-A42B | 152,576 | 2.863 | 2.822 | 3.139 | 2.282 | 2.896 | 1.663 | 6.180 | 4.281 | 3.288 | 3.253 |
| MiMo-V2.6-Pro-RL | 1.02T-A42B | 152,576 | 3.012 | 2.953 | 3.308 | 2.417 | 2.978 | 1.744 | 6.590 | 4.484 | 3.530 | 3.481 |
| Qwen3-30B-A3B-Base | 30B-A3B | 151,936 | 3.592 | 3.503 | 3.942 | 2.836 | 3.476 | 2.102 | 8.348 | 5.015 | 4.322 | 4.273 |
| Qwen3-VL-30B-A3B-Instruct | 30B-A3B | 151,936 | 3.846 | 3.756 | 4.241 | 3.010 | 3.755 | 2.183 | 9.287 | 5.494 | 4.621 | 4.559 |
| Qwen3-VL-30B-A3B-Thinking | 30B-A3B | 151,936 | 3.785 | 3.685 | 4.173 | 2.986 | 3.648 | 2.166 | 9.140 | 5.342 | 4.582 | 4.524 |
| Qwen3-235B-A22B (2504) | 235B-A22B | 151,936 | 3.562 | 3.480 | 3.951 | 2.751 | 3.496 | 2.080 | 8.551 | 5.267 | 4.223 | 4.121 |
| Qwen3-235B-A22B-Instruct-2507 | 235B-A22B | 151,936 | 3.343 | 3.274 | 3.712 | 2.580 | 3.321 | 1.981 | 7.957 | 4.974 | 3.896 | 3.801 |
| Qwen3-235B-A22B-Thinking-2507 | 235B-A22B | 151,936 | 3.306 | 3.234 | 3.656 | 2.577 | 3.256 | 1.972 | 7.740 | 4.806 | 3.880 | 3.793 |
| Qwen3-VL-235B-A22B-Instruct | 235B-A22B | 151,936 | 3.444 | 3.376 | 3.812 | 2.663 | 3.429 | 2.036 | 8.092 | 5.085 | 4.011 | 3.934 |
| Qwen3-VL-235B-A22B-Thinking | 235B-A22B | 151,936 | 3.387 | 3.313 | 3.741 | 2.643 | 3.336 | 2.011 | 7.887 | 4.920 | 3.980 | 3.904 |
| DeepSeek-V4-Flash-Base † | 284B-A13B | 129,280 | 3.090 | 3.025 | 3.358 | 2.527 | 3.056 | 1.714 | 7.047 | 4.160 | 3.694 | 3.620 |
| Kimi-K3 † | 2.8T-A104B | 163,840 | 2.703 | 2.675 | 2.992 | 2.203 | 2.806 | 1.535 | 5.986 | 4.220 | 2.994 | 2.912 |
| Kimi-K3.1-Pro (MXFP4) † | 2.8T-A104B | 163,840 | 2.693 | 2.667 | 2.986 | 2.194 | 2.803 | 1.525 | 6.004 | 4.223 | 2.980 | 2.887 |
| Nemotron-3-Ultra-Base † | 550B-A55B | 131,072 | 2.994 | 2.930 | 3.228 | 2.443 | 2.928 | 1.846 | 6.047 | 4.037 | 3.511 | 3.540 |
| Nemotron-3.5-Super-VL-Base † | 120B-A12B | 131,072 | 3.215 | 3.139 | 3.460 | 2.629 | 3.103 | 1.943 | 6.615 | 4.251 | 3.853 | 3.878 |
| Qwen3.8-Flash-Next (FP8) † | 125B-A6B | 248,320 | 3.389 | 3.275 | 3.719 | 2.767 | 3.209 | 1.842 | 8.296 | 4.661 | 4.183 | 4.214 |
| Qwen3.8-27B † | 27B | 248,320 | 3.553 | 3.409 | 3.881 | 2.983 | 3.232 | 1.963 | 8.633 | 4.603 | 4.539 | 4.533 |
| Qwen3.8-Max (FP8) † | 2.4T-A95B | 248,320 | 2.985 | 2.907 | 3.263 | 2.442 | 2.911 | 1.714 | 6.741 | 4.166 | 3.526 | 3.559 |
| Mistral-Large-4 (FP8) † | 1T-A52B | 204,800 | 3.705 | 3.537 | 4.161 | 3.032 | 3.401 | 1.900 | 11.255 | 5.027 | 4.681 | 4.714 |
| Flint（可能不对） † | 40B-A2.5B | 150,144 | 8.656 | 8.419 | 9.273 | 6.504 | 8.508 | 3.635 | 24.032 | 11.461 | 11.478 | 12.733 |
| Ripple-65B-Base-Preview † | 65B | 100,352 | 3.379 | 3.245 | 3.710 | 2.717 | 3.117 | 2.068 | 7.573 | 4.717 | 4.169 | 4.161 |
| 我们最好的: PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K_W32K · q3 (683M) | · | · | 4.468 | 4.368 | 4.843 | 3.327 | 4.284 | 2.777 | 9.844 | 5.929 | 5.587 | 5.343 |

**〔窗口：window = 32K〕**

**〔视图：bpb〕**

|   | model size | vocab size | overall_tsqw | overall_tw | overall_macro | val_v2 | uncheatable | codefresh | v3small | private_internal | pile_test | pile_10k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiMo-V2.5-Pro-Base | 1.02T-A42B | 152,576 | 0.4010 | 0.3998 | 0.4212 | 0.3216 | 0.4051 | 0.1859 | 0.6460 | 0.4876 | 0.4809 | 0.4449 |
| MiMo-V2.6-Pro-RL | 1.02T-A42B | 152,576 | 0.4203 | 0.4171 | 0.4407 | 0.3439 | 0.4157 | 0.2033 | 0.6688 | 0.5031 | 0.5096 | 0.4703 |
| Qwen3-30B-A3B-Base | 30B-A3B | 151,936 | 0.4890 | 0.4841 | 0.5071 | 0.4091 | 0.4754 | 0.2716 | 0.7533 | 0.5407 | 0.5923 | 0.5513 |
| Qwen3-VL-30B-A3B-Instruct | 30B-A3B | 151,936 | 0.5153 | 0.5113 | 0.5343 | 0.4324 | 0.5056 | 0.2853 | 0.7914 | 0.5713 | 0.6195 | 0.5762 |
| Qwen3-VL-30B-A3B-Thinking | 30B-A3B | 151,936 | 0.5090 | 0.5037 | 0.5282 | 0.4292 | 0.4941 | 0.2825 | 0.7855 | 0.5619 | 0.6160 | 0.5731 |
| Qwen3-235B-A22B (2504) | 235B-A22B | 151,936 | 0.4855 | 0.4815 | 0.5074 | 0.3973 | 0.4776 | 0.2677 | 0.7618 | 0.5572 | 0.5831 | 0.5378 |
| Qwen3-235B-A22B-Instruct-2507 | 235B-A22B | 151,936 | 0.4609 | 0.4578 | 0.4839 | 0.3716 | 0.4576 | 0.2498 | 0.7361 | 0.5380 | 0.5503 | 0.5067 |
| Qwen3-235B-A22B-Thinking-2507 | 235B-A22B | 151,936 | 0.4567 | 0.4530 | 0.4785 | 0.3714 | 0.4501 | 0.2481 | 0.7263 | 0.5264 | 0.5486 | 0.5057 |
| Qwen3-VL-235B-A22B-Instruct | 235B-A22B | 151,936 | 0.4726 | 0.4701 | 0.4942 | 0.3843 | 0.4709 | 0.2598 | 0.7424 | 0.5454 | 0.5621 | 0.5199 |
| Qwen3-VL-235B-A22B-Thinking | 235B-A22B | 151,936 | 0.4662 | 0.4626 | 0.4872 | 0.3814 | 0.4597 | 0.2554 | 0.7331 | 0.5343 | 0.5590 | 0.5170 |
| DeepSeek-V4-Flash-Base | 284B-A13B | 129,280 | 0.4146 | 0.4119 | 0.4362 | 0.3408 | 0.4105 | 0.2067 | 0.6650 | 0.4934 | 0.5006 | 0.4534 |
| Kimi-K3 | 2.8T-A104B | 163,840 | 0.3618 | 0.3636 | 0.3864 | 0.2882 | 0.3803 | 0.1549 | 0.6044 | 0.4728 | 0.4178 | 0.3789 |
| Kimi-K3.1-Pro (MXFP4) | 2.8T-A104B | 163,840 | 0.3606 | 0.3626 | 0.3856 | 0.2866 | 0.3800 | 0.1526 | 0.6055 | 0.4731 | 0.4159 | 0.3757 |
| Nemotron-3-Ultra-Base | 550B-A55B | 131,072 | 0.4238 | 0.4187 | 0.4419 | 0.3456 | 0.4098 | 0.2383 | 0.6532 | 0.4882 | 0.5163 | 0.4781 |
| Nemotron-3.5-Super-VL-Base | 120B-A12B | 131,072 | 0.4515 | 0.4455 | 0.4684 | 0.3740 | 0.4320 | 0.2581 | 0.6858 | 0.5063 | 0.5544 | 0.5124 |
| Qwen3.8-Flash-Next (FP8) | 125B-A6B | 248,320 | 0.4741 | 0.4656 | 0.4962 | 0.4044 | 0.4503 | 0.2404 | 0.7615 | 0.5373 | 0.5836 | 0.5447 |
| Qwen3.8-27B | 27B | 248,320 | 0.4926 | 0.4805 | 0.5129 | 0.4357 | 0.4510 | 0.2655 | 0.7742 | 0.5334 | 0.6175 | 0.5732 |
| Qwen3.8-Max (FP8) | 2.4T-A95B | 248,320 | 0.4244 | 0.4179 | 0.4462 | 0.3562 | 0.4100 | 0.2120 | 0.6854 | 0.4985 | 0.5150 | 0.4839 |
| Mistral-Large-4 (FP8) | 1T-A52B | 204,800 | 0.4808 | 0.4700 | 0.5104 | 0.4050 | 0.4540 | 0.2386 | 0.8329 | 0.5423 | 0.5896 | 0.5522 |
| Flint（可能不对） | 40B-A2.5B | 150,144 | 0.7648 | 0.7663 | 0.7712 | 0.6538 | 0.7702 | 0.4637 | 1.0507 | 0.7894 | 0.8994 | 0.8615 |
| Ripple-65B-Base-Preview | 65B | 100,352 | 0.4564 | 0.4511 | 0.4721 | 0.3804 | 0.4426 | 0.2626 | 0.6869 | 0.5093 | 0.5506 | 0.5215 |
| 我们最好的: PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K_W32K · q3 (683M) | · | · | - | - | - | - | - | - | - | - | - | - |

**〔视图：val_loss〕**

|   | model size | vocab size | overall_tsqw | overall_tw | overall_macro | val_v2 | uncheatable | codefresh | v3small | private_internal | pile_test | pile_10k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiMo-V2.5-Pro-Base | 1.02T-A42B | 152,576 | 1.0317 | 1.0174 | 1.1293 | 0.7911 | 1.0433 | 0.5085 | 1.8116 | 1.4526 | 1.1686 | 1.1365 |
| MiMo-V2.6-Pro-RL | 1.02T-A42B | 152,576 | 1.0811 | 1.0616 | 1.1809 | 0.8460 | 1.0706 | 0.5562 | 1.8756 | 1.4988 | 1.2382 | 1.2015 |
| Qwen3-30B-A3B-Base | 30B-A3B | 151,936 | 1.2570 | 1.2320 | 1.3561 | 1.0065 | 1.2244 | 0.7429 | 2.1125 | 1.6110 | 1.4392 | 1.4083 |
| Qwen3-VL-30B-A3B-Instruct | 30B-A3B | 151,936 | 1.3249 | 1.3013 | 1.4289 | 1.0637 | 1.3021 | 0.7806 | 2.2195 | 1.7020 | 1.5053 | 1.4719 |
| Qwen3-VL-30B-A3B-Thinking | 30B-A3B | 151,936 | 1.3085 | 1.2818 | 1.4124 | 1.0559 | 1.2724 | 0.7727 | 2.2029 | 1.6740 | 1.4967 | 1.4640 |
| Qwen3-235B-A22B (2504) | 235B-A22B | 151,936 | 1.2489 | 1.2255 | 1.3588 | 0.9774 | 1.2300 | 0.7324 | 2.1363 | 1.6599 | 1.4168 | 1.3740 |
| Qwen3-235B-A22B-Instruct-2507 | 235B-A22B | 151,936 | 1.1860 | 1.1650 | 1.2967 | 0.9142 | 1.1785 | 0.6834 | 2.0645 | 1.6027 | 1.3371 | 1.2943 |
| Qwen3-235B-A22B-Thinking-2507 | 235B-A22B | 151,936 | 1.1750 | 1.1529 | 1.2816 | 0.9136 | 1.1592 | 0.6788 | 2.0368 | 1.5683 | 1.3330 | 1.2919 |
| Qwen3-VL-235B-A22B-Instruct | 235B-A22B | 151,936 | 1.2162 | 1.1965 | 1.3236 | 0.9453 | 1.2127 | 0.7109 | 2.0821 | 1.6247 | 1.3660 | 1.3281 |
| Qwen3-VL-235B-A22B-Thinking | 235B-A22B | 151,936 | 1.1993 | 1.1773 | 1.3045 | 0.9383 | 1.1840 | 0.6988 | 2.0558 | 1.5918 | 1.3584 | 1.3208 |
| DeepSeek-V4-Flash-Base † | 284B-A13B | 129,280 | 1.1065 | 1.0868 | 1.1956 | 0.8897 | 1.0988 | 0.5386 | 1.9439 | 1.4211 | 1.2819 | 1.2392 |
| Kimi-K3 † | 2.8T-A104B | 163,840 | 0.9752 | 0.9653 | 1.0825 | 0.7567 | 1.0135 | 0.4286 | 1.7807 | 1.4393 | 1.0761 | 1.0291 |
| Kimi-K3.1-Pro (MXFP4) † | 2.8T-A104B | 163,840 | 0.9719 | 0.9626 | 1.0804 | 0.7526 | 1.0126 | 0.4221 | 1.7838 | 1.4400 | 1.0714 | 1.0204 |
| Nemotron-3-Ultra-Base † | 550B-A55B | 131,072 | 1.0764 | 1.0557 | 1.1572 | 0.8590 | 1.0557 | 0.6131 | 1.7903 | 1.3916 | 1.2338 | 1.2200 |
| Nemotron-3.5-Super-VL-Base † | 120B-A12B | 131,072 | 1.1462 | 1.1232 | 1.2256 | 0.9296 | 1.1127 | 0.6641 | 1.8795 | 1.4431 | 1.3248 | 1.3075 |
| Qwen3.8-Flash-Next (FP8) † | 125B-A6B | 248,320 | 1.1967 | 1.1651 | 1.2964 | 0.9736 | 1.1498 | 0.6107 | 2.1089 | 1.5336 | 1.4016 | 1.3792 |
| Qwen3.8-27B † | 27B | 248,320 | 1.2421 | 1.2023 | 1.3374 | 1.0489 | 1.1515 | 0.6747 | 2.1441 | 1.5223 | 1.4830 | 1.4514 |
| Qwen3.8-Max (FP8) † | 2.4T-A95B | 248,320 | 1.0720 | 1.0456 | 1.1668 | 0.8577 | 1.0468 | 0.5385 | 1.8981 | 1.4228 | 1.2370 | 1.2253 |
| Mistral-Large-4 (FP8) † | 1T-A52B | 204,800 | 1.2845 | 1.2402 | 1.4078 | 1.0638 | 1.2043 | 0.6417 | 2.4107 | 1.6119 | 1.5143 | 1.4890 |
| Flint（可能不对） † | 40B-A2.5B | 150,144 | 2.1138 | 2.0913 | 2.1959 | 1.7815 | 2.1110 | 1.2905 | 3.1647 | 2.4389 | 2.3889 | 2.4328 |
| Ripple-65B-Base-Preview † | 65B | 100,352 | 1.2028 | 1.1600 | 1.3006 | 0.9775 | 1.1155 | 0.7267 | 2.0183 | 1.5511 | 1.4142 | 1.4055 |
| 我们最好的: PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K_W32K · q3 (683M) | · | · | - | - | - | - | - | - | - | - | - | - |

**〔视图：ppl〕**

|   | model size | vocab size | overall_tsqw | overall_tw | overall_macro | val_v2 | uncheatable | codefresh | v3small | private_internal | pile_test | pile_10k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiMo-V2.5-Pro-Base | 1.02T-A42B | 152,576 | 2.806 | 2.766 | 3.093 | 2.206 | 2.839 | 1.663 | 6.120 | 4.274 | 3.217 | 3.116 |
| MiMo-V2.6-Pro-RL | 1.02T-A42B | 152,576 | 2.948 | 2.891 | 3.257 | 2.330 | 2.917 | 1.744 | 6.525 | 4.477 | 3.449 | 3.325 |
| Qwen3-30B-A3B-Base | 30B-A3B | 151,936 | 3.515 | 3.428 | 3.881 | 2.736 | 3.402 | 2.102 | 8.269 | 5.008 | 4.217 | 4.089 |
| Qwen3-VL-30B-A3B-Instruct | 30B-A3B | 151,936 | 3.762 | 3.674 | 4.174 | 2.897 | 3.677 | 2.183 | 9.203 | 5.485 | 4.506 | 4.358 |
| Qwen3-VL-30B-A3B-Thinking | 30B-A3B | 151,936 | 3.701 | 3.603 | 4.106 | 2.875 | 3.569 | 2.166 | 9.051 | 5.333 | 4.467 | 4.323 |
| Qwen3-235B-A22B (2504) | 235B-A22B | 151,936 | 3.487 | 3.406 | 3.892 | 2.657 | 3.421 | 2.080 | 8.468 | 5.259 | 4.124 | 3.951 |
| Qwen3-235B-A22B-Instruct-2507 | 235B-A22B | 151,936 | 3.274 | 3.206 | 3.657 | 2.495 | 3.250 | 1.981 | 7.881 | 4.966 | 3.808 | 3.648 |
| Qwen3-235B-A22B-Thinking-2507 | 235B-A22B | 151,936 | 3.238 | 3.167 | 3.603 | 2.493 | 3.187 | 1.972 | 7.666 | 4.799 | 3.792 | 3.640 |
| Qwen3-VL-235B-A22B-Instruct | 235B-A22B | 151,936 | 3.374 | 3.309 | 3.757 | 2.574 | 3.363 | 2.036 | 8.021 | 5.077 | 3.919 | 3.774 |
| Qwen3-VL-235B-A22B-Thinking | 235B-A22B | 151,936 | 3.318 | 3.246 | 3.686 | 2.556 | 3.267 | 2.011 | 7.813 | 4.913 | 3.890 | 3.747 |
| DeepSeek-V4-Flash-Base † | 284B-A13B | 129,280 | 3.024 | 2.965 | 3.306 | 2.434 | 3.001 | 1.714 | 6.986 | 4.141 | 3.603 | 3.453 |
| Kimi-K3 † | 2.8T-A104B | 163,840 | 2.652 | 2.626 | 2.952 | 2.131 | 2.755 | 1.535 | 5.934 | 4.218 | 2.933 | 2.798 |
| Kimi-K3.1-Pro (MXFP4) † | 2.8T-A104B | 163,840 | 2.643 | 2.619 | 2.946 | 2.123 | 2.753 | 1.525 | 5.952 | 4.221 | 2.919 | 2.774 |
| Nemotron-3-Ultra-Base † | 550B-A55B | 131,072 | 2.934 | 2.874 | 3.181 | 2.361 | 2.874 | 1.846 | 5.991 | 4.021 | 3.434 | 3.387 |
| Nemotron-3.5-Super-VL-Base † | 120B-A12B | 131,072 | 3.146 | 3.075 | 3.406 | 2.534 | 3.043 | 1.943 | 6.550 | 4.234 | 3.761 | 3.697 |
| Qwen3.8-Flash-Next (FP8) † | 125B-A6B | 248,320 | 3.309 | 3.206 | 3.656 | 2.647 | 3.158 | 1.842 | 8.239 | 4.635 | 4.062 | 3.972 |
| Qwen3.8-27B † | 27B | 248,320 | 3.463 | 3.328 | 3.809 | 2.855 | 3.163 | 1.963 | 8.534 | 4.583 | 4.406 | 4.269 |
| Qwen3.8-Max (FP8) † | 2.4T-A95B | 248,320 | 2.921 | 2.845 | 3.212 | 2.358 | 2.849 | 1.714 | 6.673 | 4.149 | 3.445 | 3.405 |
| Mistral-Large-4 (FP8) † | 1T-A52B | 204,800 | 3.613 | 3.456 | 4.087 | 2.897 | 3.334 | 1.900 | 11.142 | 5.012 | 4.546 | 4.433 |
| Flint（可能不对） † | 40B-A2.5B | 150,144 | 8.279 | 8.095 | 8.988 | 5.939 | 8.256 | 3.635 | 23.681 | 11.461 | 10.901 | 11.391 |
| Ripple-65B-Base-Preview † | 65B | 100,352 | 3.329 | 3.190 | 3.671 | 2.658 | 3.051 | 2.068 | 7.526 | 4.717 | 4.113 | 4.077 |
| 我们最好的: PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K_W32K · q3 (683M) | · | · | - | - | - | - | - | - | - | - | - | - |

**〔窗口：32K − 8K〕**

|   | model size | vocab size | overall_tsqw | overall_tw | overall_macro | val_v2 | uncheatable | codefresh | v3small | private_internal | pile_test | pile_10k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiMo-V2.5-Pro-Base | 1.02T-A42B | 152,576 | -0.0081 | -0.0079 | -0.0057 | -0.0138 | -0.0077 | 0 | -0.0035 | -0.0005 | -0.0089 | -0.0168 |
| MiMo-V2.6-Pro-RL | 1.02T-A42B | 152,576 | -0.0086 | -0.0083 | -0.0061 | -0.0148 | -0.0081 | 0 | -0.0036 | -0.0006 | -0.0095 | -0.0179 |
| Qwen3-30B-A3B-Base | 30B-A3B | 151,936 | -0.0087 | -0.0086 | -0.0062 | -0.0146 | -0.0084 | 0 | -0.0034 | -0.0005 | -0.0101 | -0.0173 |
| Qwen3-VL-30B-A3B-Instruct | 30B-A3B | 151,936 | -0.0089 | -0.0087 | -0.0063 | -0.0155 | -0.0082 | 0 | -0.0032 | -0.0005 | -0.0104 | -0.0177 |
| Qwen3-VL-30B-A3B-Thinking | 30B-A3B | 151,936 | -0.0090 | -0.0088 | -0.0064 | -0.0155 | -0.0085 | 0 | -0.0035 | -0.0005 | -0.0105 | -0.0178 |
| Qwen3-235B-A22B (2504) | 235B-A22B | 151,936 | -0.0085 | -0.0084 | -0.0060 | -0.0140 | -0.0084 | 0 | -0.0034 | -0.0005 | -0.0097 | -0.0165 |
| Qwen3-235B-A22B-Instruct-2507 | 235B-A22B | 151,936 | -0.0083 | -0.0083 | -0.0059 | -0.0136 | -0.0084 | 0 | -0.0034 | -0.0005 | -0.0094 | -0.0160 |
| Qwen3-235B-A22B-Thinking-2507 | 235B-A22B | 151,936 | -0.0082 | -0.0082 | -0.0058 | -0.0135 | -0.0082 | 0 | -0.0034 | -0.0005 | -0.0094 | -0.0161 |
| Qwen3-VL-235B-A22B-Instruct | 235B-A22B | 151,936 | -0.0081 | -0.0079 | -0.0058 | -0.0138 | -0.0076 | 0 | -0.0031 | -0.0005 | -0.0095 | -0.0162 |
| Qwen3-VL-235B-A22B-Thinking | 235B-A22B | 151,936 | -0.0082 | -0.0081 | -0.0058 | -0.0137 | -0.0081 | 0 | -0.0033 | -0.0005 | -0.0094 | -0.0161 |
| DeepSeek-V4-Flash-Base | 284B-A13B | 129,280 | -0.0081 | -0.0077 | -0.0059 | -0.0144 | -0.0069 | 0 | -0.0030 | -0.0016 | -0.0097 | -0.0173 |
| Kimi-K3 | 2.8T-A104B | 163,840 | -0.0072 | -0.0070 | -0.0051 | -0.0126 | -0.0068 | 0 | -0.0029 | -0.0002 | -0.0080 | -0.0147 |
| Kimi-K3.1-Pro (MXFP4) | 2.8T-A104B | 163,840 | -0.0071 | -0.0070 | -0.0051 | -0.0126 | -0.0067 | 0 | -0.0030 | -0.0002 | -0.0080 | -0.0147 |
| Nemotron-3-Ultra-Base | 550B-A55B | 131,072 | -0.0081 | -0.0077 | -0.0058 | -0.0137 | -0.0072 | 0 | -0.0034 | -0.0013 | -0.0092 | -0.0173 |
| Nemotron-3.5-Super-VL-Base | 120B-A12B | 131,072 | -0.0087 | -0.0082 | -0.0063 | -0.0149 | -0.0076 | 0 | -0.0036 | -0.0014 | -0.0101 | -0.0188 |
| Qwen3.8-Flash-Next (FP8) | 125B-A6B | 248,320 | -0.0096 | -0.0085 | -0.0069 | -0.0183 | -0.0064 | 0 | -0.0025 | -0.0020 | -0.0123 | -0.0234 |
| Qwen3.8-27B | 27B | 248,320 | -0.0104 | -0.0096 | -0.0075 | -0.0182 | -0.0085 | 0 | -0.0042 | -0.0016 | -0.0124 | -0.0237 |
| Qwen3.8-Max (FP8) | 2.4T-A95B | 248,320 | -0.0087 | -0.0086 | -0.0063 | -0.0147 | -0.0085 | 0 | -0.0036 | -0.0015 | -0.0096 | -0.0175 |
| Mistral-Large-4 (FP8) | 1T-A52B | 204,800 | -0.0096 | -0.0087 | -0.0068 | -0.0173 | -0.0074 | 0 | -0.0035 | -0.0010 | -0.0114 | -0.0228 |
| Flint（可能不对） | 40B-A2.5B | 150,144 | -0.0163 | -0.0144 | -0.0114 | -0.0334 | -0.0110 | 0 | -0.0049 | 0 | -0.0194 | -0.0394 |
| Ripple-65B-Base-Preview | 65B | 100,352 | -0.0058 | -0.0066 | -0.0040 | -0.0085 | -0.0084 | 0 | -0.0021 | 0 | -0.0052 | -0.0075 |
| 我们最好的: PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K_W32K · q3 (683M) | · | · | - | - | - | - | - | - | - | - | - | - |

vocab size：模型 config 里的 vocab_size，即 embedding 行数（含 padding，所以比 tokenizer 实际的 token 数略大，如 Qwen3 是 151,936 行、151,665 个 token）；我们的 checkpoint 为训练时的 padded vocab。
model size：MoE 写「总参数-A激活参数」，dense 只写总参数，数字取自模型卡；我们的 checkpoint 含 embedding 和输出层（行名里的 1.17B 等是不含 embedding 的激活参数）；Qwen3.8-Flash-Next 另有 51B n-gram embedding 不在 125B 里。
† 用自己的 tokenizer：val_loss / ppl 按它自己的 token 计，和按 Qwen3 token 计的其他行不可比（只看 bpb）。bpb = 总 NLL ÷（原文 UTF-8 字节数 × ln 2），分母与 ladder 完全相同；文本为 ~/data/eval_ppl_decon 的原文，切分规则同 ladder（每篇第一个 token 不计分，BOS 只作上下文）。

### 全量测评范围

下游 benchmark 全量评测，配置是 Boxiang 的 ADLR lm-eval 的 `nemo6_report.yml`（与他评测我们 production 模型时完全相同）。协议：few-shot base prompt，不套 chat 模板；模型用 vLLM 起服务。下面是测哪些 benchmark；各模型的分数另起一章。

先按是否生成分成两大部分（非生成类在前），每部分里再按配置的 task_group 分类；reading 和 aggregate 两个 task_group 两种都有，按任务拆到两边。一个 benchmark 可能有两种跑法（贪心一次 / 每题采样 32 次），各算一个任务。

**nemo6**2 大部分 · 9 类 · 40 任务 · 36 个 benchmark · 75 个指标greedy采样 ×32执行代码LLM 裁判

#### 非生成类 benchmark 11 任务 · 26 指标

多选题：把每个选项接在题目后面，比较模型给各选项的对数概率（acc_norm 按长度归一化），概率最高的选项即答案。不生成文本，所以分数不受回答格式和停止符影响。

**常识推理**reasoning · 8 任务 · 8 指标

英文多选题

- CommonsenseQA 7-shot → acc
- ARC-Challenge 25-shot → acc_norm
- HellaSwag 10-shot → acc_norm
- OpenBookQA → acc_norm
- PIQA → acc_norm
- WinoGrande 5-shot → acc
- Social IQa → acc
- TruthfulQA MC2 0-shot → acc

**阅读理解**reading · 1 任务 · 1 指标

读一篇文章，答多选题

- RACE → acc

**多语言考试**aggregate · 2 任务 · 17 指标

知识考试多选题，8–9 种非英语语言（含中文），每种语言一个分

- Global-MMLU 5-shot → acc × 8 种语言de · es · fr · it · ja · ko · pt · zh
- INCLUDE base-44 5-shot → acc × 9 种语言spanish · german · french · chinese · japanese · russian · italian · portuguese · korean

#### 生成类 benchmark 29 任务 · 49 指标

模型生成回答（贪心一次，或每题采样 32 次），再抽取答案比对、执行代码，或由 LLM 裁判判对错。分数也受回答格式影响（见结果表里的 ‡）。

**代码**code · 6 任务 · 9 指标

生成代码并执行打分

- MBPP sanitized 3-shot执行代码采样 ×32 → pass@1 · pass@32
- HumanEval 0-shot执行代码采样 ×32 → pass@1 · pass@32
- LiveCodeBench release v6
  - greedy · 0-shot执行代码greedy → pass@1
  - 采样 · 0-shot执行代码采样 ×32 → pass@1 · pass@32
- BigCodeBench
  - complete · 0-shot执行代码greedy → pass@1
  - instruct · 0-shot执行代码greedy → pass@1

**综合考试**aggregate · 10 任务 · 19 指标

生成作答（多为 CoT），抽取最终选项 / 答案比对

- MMLU-Pro
  - base · 5-shotgreedy → EM
  - 采样 · 0-shot采样 ×32 → pass@1 · pass@32
- MMLU string 版 5-shotgreedy → EM（总分 / STEM）
- MMLU-Redux string 版 5-shotgreedy → EM
- AGIEval-en CoT greedy → EM
- GPQA-Diamond CoT
  - greedy · 5-shotgreedy → EM
  - 采样 · 0-shot采样 ×32 → pass@1 · pass@32
- SuperGPQA CoT 5-shotgreedy → EM
- HLE 纯文本题 0-shot采样 ×32 → pass@1 · pass@32
- MGSM native CoT 8-shotgreedy → EM × 6 种语言es · de · fr · zh · ja · ru

**数学**math · 6 任务 · 12 指标

生成解答，比对最终答案

- GSM8K CoT 8-shotgreedy → EM
- Minerva MATH 4-shotgreedy → EM · EM (Level 5) · EM (MATH-500 子集)
- MATH-500 4-shot采样 ×32 → pass@1 · pass@32
- AIME 2024 4-shot采样 ×32 → pass@1 · pass@32
- AIME 2025 4-shot采样 ×32 → pass@1 · pass@32
- HMMT Feb 2025 0-shot采样 ×32 → pass@1 · pass@32

**事实问答**reading · 4 任务 · 6 指标

短答案生成，精确匹配

- TriviaQA 5-shotgreedy → EM
- PopQA 5-shotgreedy → EM
- Natural Questions 5-shotgreedy → EM
- SimpleQA 5-shotgreedy → F1 · correct · not_attempted

**知识**knowledge · 2 任务 · 2 指标

LLM 裁判 gpt-oss-120b 判对错

- SimpleQA-Verified 25-shotLLM 裁判greedy → 裁判准确率
- FACTS-parametric 每题判 3 次 25-shotLLM 裁判greedy → 裁判准确率

**多语言**multilingual · 1 任务 · 1 指标

翻译

- WMT24++ 英 → 55 种语言，报一个总分 0-shotgreedy → chrF++

生成类的跑法：greedy = 贪心解码一次；采样 ×32 = 每题采样 32 次，报 pass@1（平均）和 pass@32（至少对一次）；执行代码 = 生成的代码跑测试用例；LLM 裁判 = gpt-oss-120b 判对错。

### 全量测评结果

分数 ×100（准确率 / pass@k 为 %，WMT24++ 为 chrF++）。点表头排序（第一次从高到低）；**粗体** = 该列最高；· = 还没跑出来（job 在跑，跑完后更新）。模型名、大小与上面 Pretrain doc-isolated 表一致。

曲线：每一列在各模型之间做 min–max 归一化（该列最低 = 0，最高 = 1），只看相对位置，纵轴没有绝对意义。0 分、‡ 空回答、以及明显偏低的分（同时低于该列 Q1 − 1.5×IQR 和中位数的一半，至少 4 个模型有分时才判）不参与归一化，画在图下方的“范围外”一栏（空心点，线照样连过去）；只有一个模型有分的列不画。颜色按模型家族，同一家族用同一色系的深浅（Qwen3 和 Qwen3.8 各一个色系，按模型大小由浅到深）。鼠标放在图上显示原始分（与表格相同）。

#### 非生成类 benchmark 多选题，按选项概率打分，不生成 · 11 任务 · 26 指标

##### 常识推理 · 阅读理解 · 多语言考试 reasoning · reading · aggregate

常识推理：英文多选题；阅读理解：读一篇文章，答多选题；多语言考试：知识考试多选题，8–9 种非英语语言（含中文），每种语言一个分

蓝色系：Qwen3.8 — Qwen3.8-27B / Qwen3.8-Flash-Next (FP8) / Qwen3.8-Max (FP8)（由浅到深，按模型大小）

橙色系：Qwen3 — Qwen3-30B-A3B-Base / Qwen3-235B-A22B (2504)（由浅到深，按模型大小）

青绿色系：MiMo — MiMo-V2.5-Pro-Base / MiMo-V2.6-Pro-RL（由浅到深）

黄色：DeepSeek-V4-Flash-Base

品红：Kimi-K3

紫色：Mistral-Large-4 (FP8)

| 横轴 | Qwen3.8-Max (FP8) | Kimi-K3 | MiMo-V2.5-Pro-Base | MiMo-V2.6-Pro-RL | Qwen3-235B-A22B (2504) | Qwen3.8-Flash-Next (FP8) | Qwen3.8-27B | Qwen3-30B-A3B-Base | Mistral-Large-4 (FP8) | DeepSeek-V4-Flash-Base |
|---|---|---|---|---|---|---|---|---|---|---|
| CSQA · acc | 95.7 | 89.6 | 89.2 | 88.6 | 87.8 | 87.7 | 87.7 | 87.4 | 87.1 | 85.0 |
| ARC-C · acc_norm | 99.7 | 97.4 | 97.1 | 96.6 | 96.2 | 97.4 | 96.2 | 94.0 | 96.9 | 96.0 |
| HellaSwag · acc_norm | 90.0 | 91.8 | 89.5 | 89.3 | 86.9 | 88.6 | 75.7 | 82.8 | 87.9 | 89.0 |
| OBQA · acc_norm | 46.8 | 49.4 | 50.6 | 50.4 | 45.8 | 45.6 | 46.6 | 45.4 | 50.6 | 47.6 |
| PIQA · acc_norm | 84.9 | 86.7 | 85.1 | 85.7 | 82.9 | 83.7 | 81.7 | 80.8 | 84.3 | 84.9 |
| WinoG · acc | 67.1 | 82.0 | 85.5 | 83.5 | 80.6 | 75.1 | 79.0 | 78.9 | 80.5 | 83.6 |
| SIQA · acc | 42.9 | 54.7 | 51.4 | 52.7 | 54.9 | 53.4 | 53.0 | 52.4 | 57.8 | 52.2 |
| TruthfulQA · acc | 86.2 | 59.4 | 55.0 | 55.0 | 60.7 | 67.2 | 54.1 | 45.7 | 59.9 | 53.2 |
| RACE · acc | 93.4 | 95.0 | 91.6 | 92.8 | 93.0 | 92.7 | 92.6 | 89.9 | 92.2 | 93.3 |
| G-MMLU · acc 平均 | 88.7 | 90.6 | 87.2 | 86.6 | 85.5 | 86.0 | 82.4 | 76.6 | 84.1 | 88.4 |
| INCLUDE · acc 平均 | 83.5 | 84.1 | 80.6 | 80.3 | 79.3 | 79.8 | 76.3 | 74.9 | 80.9 | 82.3 |

| 模型 | 大小 | CommonsenseQA<br>acc | ARC-Challenge<br>acc_norm | HellaSwag<br>acc_norm | OpenBookQA<br>acc_norm | PIQA<br>acc_norm | WinoGrande<br>acc | Social IQa<br>acc | TruthfulQA<br>acc | RACE<br>acc | Global-MMLU<br>acc · 平均（8 种语言） | INCLUDE<br>acc · 平均（9 种语言） |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiMo-V2.5-Pro-Base | 1.02T-A42B | 89.2 | 97.1 | 89.5 | 50.6 | 85.1 | 85.5 | 51.4 | 55.0 | 91.6 | 87.2 | 80.6 |
| MiMo-V2.6-Pro-RL 后训练 | 1.02T-A42B | 88.6 | 96.6 | 89.3 | 50.4 | 85.7 | 83.5 | 52.7 | 55.0 | 92.8 | 86.6 | 80.3 |
| Qwen3-30B-A3B-Base | 30B-A3B | 87.4 | 94.0 | 82.8 | 45.4 | 80.8 | 78.9 | 52.4 | 45.7 | 89.9 | 76.6 | 74.9 |
| Qwen3-235B-A22B (2504) 后训练 | 235B-A22B | 87.8 | 96.2 | 86.9 | 45.8 | 82.9 | 80.6 | 54.9 | 60.7 | 93.0 | 85.5 | 79.3 |
| DeepSeek-V4-Flash-Base | 284B-A13B | 85.0 | 96.0 | 89.0 | 47.6 | 84.9 | 83.6 | 52.2 | 53.2 | 93.3 | 88.4 | 82.3 |
| Kimi-K3 后训练 | 2.8T-A104B | 89.6 | 97.4 | 91.8 | 49.4 | 86.7 | 82.0 | 54.7 | 59.4 | 95.0 | 90.6 | 84.1 |
| Qwen3.8-Flash-Next (FP8) 后训练 | 125B-A6B | 87.7 | 97.4 | 88.6 | 45.6 | 83.7 | 75.1 | 53.4 | 67.2 | 92.7 | 86.0 | 79.8 |
| Qwen3.8-27B 后训练 | 27B | 87.7 | 96.2 | 75.7 | 46.6 | 81.7 | 79.0 | 53.0 | 54.1 | 92.6 | 82.4 | 76.3 |
| Qwen3.8-Max (FP8) 后训练 | 2.4T-A95B | 95.7 | 99.7 | 90.0 | 46.8 | 84.9 | 67.1 | 42.9 | 86.2 | 93.4 | 88.7 | 83.5 |
| Mistral-Large-4 (FP8) 后训练 | 1T-A52B | 87.1 | 96.9 | 87.9 | 50.6 | 84.3 | 80.5 | 57.8 | 59.9 | 92.2 | 84.1 | 80.9 |

各语言明细（Global-MMLU）

| 模型 | 大小 | de | es | fr | it | ja | ko | pt | zh |
|---|---|---|---|---|---|---|---|---|---|
| MiMo-V2.5-Pro-Base | 1.02T-A42B | 87.8 | 89.0 | 86.2 | 88.8 | 86.5 | 83.8 | 88.8 | 87.2 |
| MiMo-V2.6-Pro-RL 后训练 | 1.02T-A42B | 87.0 | 88.0 | 87.8 | 87.0 | 86.8 | 83.0 | 86.8 | 86.2 |
| Qwen3-30B-A3B-Base | 30B-A3B | 81.0 | 79.2 | 76.5 | 78.0 | 72.2 | 70.5 | 78.8 | 76.8 |
| Qwen3-235B-A22B (2504) 后训练 | 235B-A22B | 85.0 | 87.2 | 86.0 | 88.2 | 83.5 | 78.8 | 88.5 | 86.5 |
| DeepSeek-V4-Flash-Base | 284B-A13B | 87.5 | 90.2 | 88.8 | 89.8 | 88.0 | 85.8 | 89.2 | 87.8 |
| Kimi-K3 后训练 | 2.8T-A104B | 91.2 | 92.0 | 90.2 | 91.5 | 89.5 | 89.0 | 92.0 | 89.0 |
| Qwen3.8-Flash-Next (FP8) 后训练 | 125B-A6B | 88.2 | 88.5 | 85.5 | 89.0 | 83.5 | 83.5 | 87.0 | 82.8 |
| Qwen3.8-27B 后训练 | 27B | 82.8 | 83.5 | 85.0 | 84.0 | 79.5 | 79.8 | 83.5 | 81.2 |
| Qwen3.8-Max (FP8) 后训练 | 2.4T-A95B | 89.8 | 90.0 | 89.0 | 89.0 | 88.0 | 87.0 | 90.0 | 86.8 |
| Mistral-Large-4 (FP8) 后训练 | 1T-A52B | 85.2 | 87.8 | 85.2 | 86.8 | 83.0 | 77.5 | 84.8 | 82.2 |

各语言明细（INCLUDE）

| 模型 | 大小 | spanish | german | french | chinese | japanese | russian | italian | portuguese | korean |
|---|---|---|---|---|---|---|---|---|---|---|
| MiMo-V2.5-Pro-Base | 1.02T-A42B | 82.2 | 62.6 | 83.1 | 89.7 | 92.0 | 77.4 | 88.3 | 76.6 | 73.8 |
| MiMo-V2.6-Pro-RL 后训练 | 1.02T-A42B | 82.5 | 64.0 | 82.8 | 87.9 | 93.0 | 74.1 | 87.0 | 75.5 | 76.0 |
| Qwen3-30B-A3B-Base | 30B-A3B | 78.5 | 58.3 | 70.9 | 85.7 | 86.4 | 73.0 | 82.8 | 71.1 | 67.4 |
| Qwen3-235B-A22B (2504) 后训练 | 235B-A22B | 81.3 | 61.9 | 79.5 | 88.6 | 93.6 | 71.7 | 87.4 | 77.3 | 72.0 |
| DeepSeek-V4-Flash-Base | 284B-A13B | 85.8 | 64.7 | 83.8 | 89.9 | 94.8 | 78.4 | 88.9 | 78.6 | 75.4 |
| Kimi-K3 后训练 | 2.8T-A104B | 84.4 | 64.0 | 86.6 | 91.4 | 97.4 | 81.3 | 91.4 | 81.5 | 78.6 |
| Qwen3.8-Flash-Next (FP8) 后训练 | 125B-A6B | 79.3 | 63.3 | 82.1 | 87.2 | 94.8 | 73.9 | 86.5 | 79.1 | 72.2 |
| Qwen3.8-27B 后训练 | 27B | 77.8 | 63.3 | 74.9 | 80.0 | 87.8 | 72.3 | 85.6 | 75.7 | 69.2 |
| Qwen3.8-Max (FP8) 后训练 | 2.4T-A95B | 85.6 | 67.6 | 86.6 | 86.8 | 96.4 | 79.0 | 91.2 | 80.0 | 77.8 |
| Mistral-Large-4 (FP8) 后训练 | 1T-A52B | 85.3 | 62.6 | 80.9 | 84.4 | 94.8 | 80.4 | 88.3 | 77.5 | 74.0 |

#### 生成类 benchmark 生成回答后打分 · 29 任务 · 49 指标

##### 代码 code

生成代码并执行打分

蓝色系：Qwen3.8 — Qwen3.8-27B / Qwen3.8-Flash-Next (FP8) / Qwen3.8-Max (FP8)（由浅到深，按模型大小）

橙色系：Qwen3 — Qwen3-30B-A3B-Base / Qwen3-235B-A22B (2504)（由浅到深，按模型大小）

青绿色系：MiMo — MiMo-V2.5-Pro-Base / MiMo-V2.6-Pro-RL（由浅到深）

黄色：DeepSeek-V4-Flash-Base

品红：Kimi-K3

紫色：Mistral-Large-4 (FP8)

| 横轴 | Kimi-K3 | Qwen3.8-Max (FP8) | Qwen3-235B-A22B (2504) | Mistral-Large-4 (FP8) | MiMo-V2.5-Pro-Base | MiMo-V2.6-Pro-RL | Qwen3-30B-A3B-Base | Qwen3.8-Flash-Next (FP8) | Qwen3.8-27B | DeepSeek-V4-Flash-Base |
|---|---|---|---|---|---|---|---|---|---|---|
| MBPP · pass@1 | 92.5 | 86.1 | 82.1 | 79.8 | 78.9 | 75.6 | 75.0 | 74.5 | 73.5 | 71.4 |
| MBPP · pass@32 | 98.8 | 97.3 | 93.0 | 91.8 | 93.0 | 93.0 | 93.4 | 94.6 | 91.8 | 92.2 |
| HumanEval · pass@1 | 77.6 | 82.1 | 80.2 | 76.2 | 54.6 | 77.1 | 70.8 | 85.3 | 77.3 | 44.6 |
| HumanEval · pass@32 | 99.4 | 98.2 | 97.0 | 97.0 | 97.6 | 98.2 | 94.5 | 98.8 | 97.0 | 90.9 |
| LCB greedy · pass@1 | 40.0 | 37.7 | 29.0 | 33.1 | 27.5 | 39.8 | 33.1 | 27.7 | 29.9 | 38.6 |
| LCB 采样 · pass@1 | 46.4 | 45.1 | 32.0 | 38.7 | 29.5 | 43.2 | 31.6 | 33.5 | 35.9 | 41.2 |
| LCB 采样 · pass@32 | 70.0 | 70.4 | 58.8 | 69.3 | 63.1 | 68.5 | 53.6 | 64.9 | 65.7 | 63.6 |
| BCB complete · pass@1 | 48.2 | 50.7 | 51.3 | 51.6 | 48.4 | 50.5 | 51.6 | 51.5 | 51.7 | 44.0 |
| BCB instruct · pass@1 | 41.9 | 53.2 | 51.0 | 51.9 | 44.3 | 66.3 | 69.9 | 59.0 | 52.0 | 28.5 |

| 模型 | 大小 | MBPP<br>pass@1 | MBPP<br>pass@32 | HumanEval<br>pass@1 | HumanEval<br>pass@32 | LiveCodeBench greedy<br>pass@1 | LiveCodeBench 采样<br>pass@1 | LiveCodeBench 采样<br>pass@32 | BigCodeBench complete<br>pass@1 | BigCodeBench instruct<br>pass@1 |
|---|---|---|---|---|---|---|---|---|---|---|
| MiMo-V2.5-Pro-Base | 1.02T-A42B | 78.9 | 93.0 | 54.6 | 97.6 | 27.5 | 29.5 | 63.1 | 48.4 | 44.3 |
| MiMo-V2.6-Pro-RL 后训练 | 1.02T-A42B | 75.6 | 93.0 | 77.1 | 98.2 | 39.8 | 43.2 | 68.5 | 50.5 | 66.3 |
| Qwen3-30B-A3B-Base | 30B-A3B | 75.0 | 93.4 | 70.8 | 94.5 | 33.1 | 31.6 | 53.6 | 51.6 | 69.9 |
| Qwen3-235B-A22B (2504) 后训练 | 235B-A22B | 82.1 | 93.0 | 80.2 | 97.0 | 29.0 | 32.0 | 58.8 | 51.3 | 51.0 |
| DeepSeek-V4-Flash-Base | 284B-A13B | 71.4 | 92.2 | 44.6 | 90.9 | 38.6 | 41.2 | 63.6 | 44.0 | 28.5 |
| Kimi-K3 后训练 | 2.8T-A104B | 92.5 | 98.8 | 77.6 | 99.4 | 40.0 | 46.4 | 70.0 | 48.2 | 41.9 |
| Qwen3.8-Flash-Next (FP8) 后训练 | 125B-A6B | 74.5 | 94.6 | 85.3 | 98.8 | 27.7 | 33.5 | 64.9 | 51.5 | 59.0 |
| Qwen3.8-27B 后训练 | 27B | 73.5 | 91.8 | 77.3 | 97.0 | 29.9 | 35.9 | 65.7 | 51.7 | 52.0 |
| Qwen3.8-Max (FP8) 后训练 | 2.4T-A95B | 86.1 | 97.3 | 82.1 | 98.2 | 37.7 | 45.1 | 70.4 | 50.7 | 53.2 |
| Mistral-Large-4 (FP8) 后训练 | 1T-A52B | 79.8 | 91.8 | 76.2 | 97.0 | 33.1 | 38.7 | 69.3 | 51.6 | 51.9 |

##### 综合考试 aggregate

生成作答（多为 CoT），抽取最终选项 / 答案比对

**MMLU 系 · AGIEval**

蓝色系：Qwen3.8 — Qwen3.8-27B / Qwen3.8-Flash-Next (FP8) / Qwen3.8-Max (FP8)（由浅到深，按模型大小）

橙色系：Qwen3 — Qwen3-30B-A3B-Base / Qwen3-235B-A22B (2504)（由浅到深，按模型大小）

青绿色系：MiMo — MiMo-V2.5-Pro-Base / MiMo-V2.6-Pro-RL（由浅到深）

黄色：DeepSeek-V4-Flash-Base

品红：Kimi-K3

紫色：Mistral-Large-4 (FP8)

| 线 | MMLU-Pro base · EM | MMLU-Pro 采样 · pass@1 | MMLU-Pro 采样 · pass@32 | MMLU · EM 总分 | MMLU · EM STEM | MMLU-Redux · EM | AGIEval-en · EM |
|---|---|---|---|---|---|---|---|
| Mistral-Large-4 (FP8) | 78.9 | 66.7 | 92.7 | 87.3 | 86.4 | 87.8 | 75.9 |
| Kimi-K3 | 78.2 | · | · | 91.3 | 94.0 | 91.2 | 74.9 |
| Qwen3.8-Max (FP8) | 77.1 | · | · | 44.9‡ | 51.6‡ | 48.5‡ | 72.9 |
| Qwen3.8-Flash-Next (FP8) | 76.6 | 72.6 | 91.3 | 0.8‡ | 0.4‡ | 0.9‡ | 56.3 |
| Qwen3-235B-A22B (2504) | 72.7 | 69.2 | 93.7 | 86.3 | 88.4 | 87.0 | 76.5 |
| MiMo-V2.6-Pro-RL | 71.3 | 54.0 | 96.0 | 88.7 | 91.2 | 90.3 | 62.3 |
| Qwen3.8-27B | 70.6 | 61.8 | 89.4 | 53.7‡ | 65.6‡ | 59.6‡ | 69.2 |
| DeepSeek-V4-Flash-Base | 66.9 | 52.1 | 97.1 | 88.6 | 90.6 | 89.2 | 67.6 |
| MiMo-V2.5-Pro-Base | 62.9 | 44.7 | 97.1 | 39.8 | 28.6 | 38.2 | 37.9 |
| Qwen3-30B-A3B-Base | 60.5 | 57.0 | 93.5 | 81.3 | 83.7 | 83.1 | 61.0 |

| 模型 | 大小 | MMLU-Pro base<br>EM | MMLU-Pro 采样<br>pass@1 | MMLU-Pro 采样<br>pass@32 | MMLU<br>EM 总分 | MMLU<br>EM STEM | MMLU-Redux<br>EM | AGIEval-en<br>EM |
|---|---|---|---|---|---|---|---|---|
| MiMo-V2.5-Pro-Base | 1.02T-A42B | 62.9 | 44.7 | 97.1 | 39.8 | 28.6 | 38.2 | 37.9 |
| MiMo-V2.6-Pro-RL 后训练 | 1.02T-A42B | 71.3 | 54.0 | 96.0 | 88.7 | 91.2 | 90.3 | 62.3 |
| Qwen3-30B-A3B-Base | 30B-A3B | 60.5 | 57.0 | 93.5 | 81.3 | 83.7 | 83.1 | 61.0 |
| Qwen3-235B-A22B (2504) 后训练 | 235B-A22B | 72.7 | 69.2 | 93.7 | 86.3 | 88.4 | 87.0 | 76.5 |
| DeepSeek-V4-Flash-Base | 284B-A13B | 66.9 | 52.1 | 97.1 | 88.6 | 90.6 | 89.2 | 67.6 |
| Kimi-K3 后训练 | 2.8T-A104B | 78.2 | · | · | 91.3 | 94.0 | 91.2 | 74.9 |
| Qwen3.8-Flash-Next (FP8) 后训练 | 125B-A6B | 76.6 | 72.6 | 91.3 | 0.8‡ | 0.4‡ | 0.9‡ | 56.3 |
| Qwen3.8-27B 后训练 | 27B | 70.6 | 61.8 | 89.4 | 53.7‡ | 65.6‡ | 59.6‡ | 69.2 |
| Qwen3.8-Max (FP8) 后训练 | 2.4T-A95B | 77.1 | · | · | 44.9‡ | 51.6‡ | 48.5‡ | 72.9 |
| Mistral-Large-4 (FP8) 后训练 | 1T-A52B | 78.9 | 66.7 | 92.7 | 87.3 | 86.4 | 87.8 | 75.9 |

‡ 灰色 = 这个模型在这一项有大量空回答（悬停看比例）：它在 few-shot 的 “A:” 后先输出换行，而这一项遇到换行就停止生成，回答被截成空字符串，分数不代表模型能力。只出现在部分后训练模型上（如 Qwen3.8-27B；MiMo-V2.6-Pro-RL 没有这个问题）。

**GPQA · SuperGPQA · HLE · MGSM**

蓝色系：Qwen3.8 — Qwen3.8-27B / Qwen3.8-Flash-Next (FP8) / Qwen3.8-Max (FP8)（由浅到深，按模型大小）

橙色系：Qwen3 — Qwen3-30B-A3B-Base / Qwen3-235B-A22B (2504)（由浅到深，按模型大小）

青绿色系：MiMo — MiMo-V2.5-Pro-Base / MiMo-V2.6-Pro-RL（由浅到深）

黄色：DeepSeek-V4-Flash-Base

品红：Kimi-K3

紫色：Mistral-Large-4 (FP8)

| 线 | GPQA-D greedy · EM | GPQA-D 采样 · pass@1 | GPQA-D 采样 · pass@32 | SuperGPQA · EM | HLE · pass@1 | HLE · pass@32 | MGSM · EM 平均 |
|---|---|---|---|---|---|---|---|
| Kimi-K3 | 69.2 | 58.3 | 98.0 | 53.6 | 0.3 | 5.4 | 90.0 |
| Mistral-Large-4 (FP8) | 65.2 | 54.0 | 93.4 | 51.6 | 0.1 | 1.1 | 68.9 |
| Qwen3.8-27B | 61.6 | 19.3 | 72.7 | 43.0 | 0.3 | 3.8 | 61.4 |
| MiMo-V2.5-Pro-Base | 60.6 | 28.7 | 99.5 | 32.2 | 0.3 | 7.9 | 83.6 |
| Qwen3-235B-A22B (2504) | 60.1 | 54.5 | 98.0 | 45.2 | 0.6 | 9.5 | 88.3 |
| MiMo-V2.6-Pro-RL | 59.6 | 54.4 | 99.0 | 43.1 | 0.6 | 11.1 | 85.5 |
| Qwen3.8-Max (FP8) | 49.5 | 42.3 | 98.5 | 53.6 | 0.6 | 7.5 | 80.9 |
| Qwen3-30B-A3B-Base | 38.4 | 34.1 | 92.4 | 35.5 | 0.2 | 4.4 | 82.0 |
| DeepSeek-V4-Flash-Base | 36.9 | 28.3 | 99.5 | 44.5 | 1.0 | 17.2 | 84.0 |
| Qwen3.8-Flash-Next (FP8) | 11.6 | 29.3 | 87.4 | 40.9 | 0.2 | 2.7 | 49.5 |

| 模型 | 大小 | GPQA-Diamond greedy<br>EM | GPQA-Diamond 采样<br>pass@1 | GPQA-Diamond 采样<br>pass@32 | SuperGPQA<br>EM | HLE<br>pass@1 | HLE<br>pass@32 | MGSM<br>EM · 平均（6 种语言） |
|---|---|---|---|---|---|---|---|---|
| MiMo-V2.5-Pro-Base | 1.02T-A42B | 60.6 | 28.7 | 99.5 | 32.2 | 0.3 | 7.9 | 83.6 |
| MiMo-V2.6-Pro-RL 后训练 | 1.02T-A42B | 59.6 | 54.4 | 99.0 | 43.1 | 0.6 | 11.1 | 85.5 |
| Qwen3-30B-A3B-Base | 30B-A3B | 38.4 | 34.1 | 92.4 | 35.5 | 0.2 | 4.4 | 82.0 |
| Qwen3-235B-A22B (2504) 后训练 | 235B-A22B | 60.1 | 54.5 | 98.0 | 45.2 | 0.6 | 9.5 | 88.3 |
| DeepSeek-V4-Flash-Base | 284B-A13B | 36.9 | 28.3 | 99.5 | 44.5 | 1.0 | 17.2 | 84.0 |
| Kimi-K3 后训练 | 2.8T-A104B | 69.2 | 58.3 | 98.0 | 53.6 | 0.3 | 5.4 | 90.0 |
| Qwen3.8-Flash-Next (FP8) 后训练 | 125B-A6B | 11.6 | 29.3 | 87.4 | 40.9 | 0.2 | 2.7 | 49.5 |
| Qwen3.8-27B 后训练 | 27B | 61.6 | 19.3 | 72.7 | 43.0 | 0.3 | 3.8 | 61.4 |
| Qwen3.8-Max (FP8) 后训练 | 2.4T-A95B | 49.5 | 42.3 | 98.5 | 53.6 | 0.6 | 7.5 | 80.9 |
| Mistral-Large-4 (FP8) 后训练 | 1T-A52B | 65.2 | 54.0 | 93.4 | 51.6 | 0.1 | 1.1 | 68.9 |

各语言明细（MGSM）

| 模型 | 大小 | es | de | fr | zh | ja | ru |
|---|---|---|---|---|---|---|---|
| MiMo-V2.5-Pro-Base | 1.02T-A42B | 84.8 | 82.4 | 84.4 | 84.0 | 77.6 | 88.4 |
| MiMo-V2.6-Pro-RL 后训练 | 1.02T-A42B | 86.8 | 86.8 | 84.0 | 87.6 | 79.2 | 88.8 |
| Qwen3-30B-A3B-Base | 30B-A3B | 85.2 | 82.4 | 83.2 | 82.8 | 73.6 | 84.8 |
| Qwen3-235B-A22B (2504) 后训练 | 235B-A22B | 95.2 | 86.4 | 86.0 | 89.2 | 82.8 | 90.0 |
| DeepSeek-V4-Flash-Base | 284B-A13B | 87.6 | 86.0 | 85.6 | 83.2 | 73.6 | 88.0 |
| Kimi-K3 后训练 | 2.8T-A104B | 94.4 | 88.4 | 92.8 | 92.8 | 80.0 | 91.6 |
| Qwen3.8-Flash-Next (FP8) 后训练 | 125B-A6B | 44.8 | 47.2 | 48.8 | 46.4 | 59.2 | 50.8 |
| Qwen3.8-27B 后训练 | 27B | 66.4 | 68.4 | 68.4 | 56.0 | 48.8 | 60.4 |
| Qwen3.8-Max (FP8) 后训练 | 2.4T-A95B | 89.2 | 80.4 | 88.8 | 61.2 | 73.6 | 92.0 |
| Mistral-Large-4 (FP8) 后训练 | 1T-A52B | 84.4 | 86.8 | 74.8 | 50.4 | 32.0 | 85.2 |

##### 数学 math

生成解答，比对最终答案

蓝色系：Qwen3.8 — Qwen3.8-27B / Qwen3.8-Flash-Next (FP8) / Qwen3.8-Max (FP8)（由浅到深，按模型大小）

橙色系：Qwen3 — Qwen3-30B-A3B-Base / Qwen3-235B-A22B (2504)（由浅到深，按模型大小）

青绿色系：MiMo — MiMo-V2.5-Pro-Base / MiMo-V2.6-Pro-RL（由浅到深）

黄色：DeepSeek-V4-Flash-Base

品红：Kimi-K3

紫色：Mistral-Large-4 (FP8)

| 横轴 | Qwen3.8-Flash-Next (FP8) | Qwen3.8-27B | Kimi-K3 | Qwen3-235B-A22B (2504) | Mistral-Large-4 (FP8) | MiMo-V2.6-Pro-RL | Qwen3-30B-A3B-Base | DeepSeek-V4-Flash-Base | Qwen3.8-Max (FP8) | MiMo-V2.5-Pro-Base |
|---|---|---|---|---|---|---|---|---|---|---|
| GSM8K · EM | 96.5 | 95.8 | 94.1 | 93.7 | 93.2 | 90.9 | 89.1 | 88.2 | 88.2 | 87.6 |
| Minerva · EM | 77.5 | 76.7 | 80.7 | 79.4 | 76.5 | 68.1 | 63.0 | 61.6 | 79.4 | 87.5 |
| Minerva · EM (Level 5) | 56.6 | 59.7 | 65.3 | 62.4 | 58.3 | 46.5 | 38.7 | 35.5 | 61.7 | 79.1 |
| Minerva · EM (MATH-500 子集) | 75.2 | 77.1 | 85.2 | 80.3 | 75.3 | 67.1 | 61.1 | 57.8 | 80.1 | 91.9 |
| MATH-500 · pass@1 | 75.6 | 67.5 | 81.8 | 77.9 | 73.5 | 65.9 | 57.5 | 55.8 | 77.7 | 87.6 |
| MATH-500 · pass@32 | 92.6 | 94.2 | 99.2 | 96.0 | 98.2 | 97.6 | 94.8 | 95.0 | 98.4 | 99.6 |
| AIME 24 · pass@1 | 8.5 | 12.4 | 40.9 | 12.4 | 11.7 | 8.9 | 3.9 | 1.7 | 17.4 | 15.6 |
| AIME 24 · pass@32 | 26.7 | 33.3 | 100.0 | 43.3 | 56.7 | 63.3 | 36.7 | 26.7 | 73.3 | 70.0 |
| AIME 25 · pass@1 | 7.5 | 11.2 | 17.0 | 9.9 | 8.7 | 5.1 | 2.3 | 1.5 | 12.2 | 4.1 |
| AIME 25 · pass@32 | 33.3 | 43.3 | 90.0 | 40.0 | 53.3 | 50.0 | 23.3 | 20.0 | 53.3 | 46.7 |
| HMMT 25 · pass@1 | 10.4 | 8.2 | 16.7 | 8.1 | 19.8 | 10.2 | 2.3 | 4.5 | 20.5 | 5.7 |
| HMMT 25 · pass@32 | 36.7 | 30.0 | 76.7 | 26.7 | 50.0 | 43.3 | 30.0 | 33.3 | 60.0 | 43.3 |

| 模型 | 大小 | GSM8K<br>EM | Minerva MATH<br>EM | Minerva MATH<br>EM (Level 5) | Minerva MATH<br>EM (MATH-500 子集) | MATH-500<br>pass@1 | MATH-500<br>pass@32 | AIME 2024<br>pass@1 | AIME 2024<br>pass@32 | AIME 2025<br>pass@1 | AIME 2025<br>pass@32 | HMMT Feb 2025<br>pass@1 | HMMT Feb 2025<br>pass@32 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MiMo-V2.5-Pro-Base | 1.02T-A42B | 87.6 | 87.5 | 79.1 | 91.9 | 87.6 | 99.6 | 15.6 | 70.0 | 4.1 | 46.7 | 5.7 | 43.3 |
| MiMo-V2.6-Pro-RL 后训练 | 1.02T-A42B | 90.9 | 68.1 | 46.5 | 67.1 | 65.9 | 97.6 | 8.9 | 63.3 | 5.1 | 50.0 | 10.2 | 43.3 |
| Qwen3-30B-A3B-Base | 30B-A3B | 89.1 | 63.0 | 38.7 | 61.1 | 57.5 | 94.8 | 3.9 | 36.7 | 2.3 | 23.3 | 2.3 | 30.0 |
| Qwen3-235B-A22B (2504) 后训练 | 235B-A22B | 93.7 | 79.4 | 62.4 | 80.3 | 77.9 | 96.0 | 12.4 | 43.3 | 9.9 | 40.0 | 8.1 | 26.7 |
| DeepSeek-V4-Flash-Base | 284B-A13B | 88.2 | 61.6 | 35.5 | 57.8 | 55.8 | 95.0 | 1.7 | 26.7 | 1.5 | 20.0 | 4.5 | 33.3 |
| Kimi-K3 后训练 | 2.8T-A104B | 94.1 | 80.7 | 65.3 | 85.2 | 81.8 | 99.2 | 40.9 | 100.0 | 17.0 | 90.0 | 16.7 | 76.7 |
| Qwen3.8-Flash-Next (FP8) 后训练 | 125B-A6B | 96.5 | 77.5 | 56.6 | 75.2 | 75.6 | 92.6 | 8.5 | 26.7 | 7.5 | 33.3 | 10.4 | 36.7 |
| Qwen3.8-27B 后训练 | 27B | 95.8 | 76.7 | 59.7 | 77.1 | 67.5 | 94.2 | 12.4 | 33.3 | 11.2 | 43.3 | 8.2 | 30.0 |
| Qwen3.8-Max (FP8) 后训练 | 2.4T-A95B | 88.2 | 79.4 | 61.7 | 80.1 | 77.7 | 98.4 | 17.4 | 73.3 | 12.2 | 53.3 | 20.5 | 60.0 |
| Mistral-Large-4 (FP8) 后训练 | 1T-A52B | 93.2 | 76.5 | 58.3 | 75.3 | 73.5 | 98.2 | 11.7 | 56.7 | 8.7 | 53.3 | 19.8 | 50.0 |

##### 事实问答 reading

短答案生成，精确匹配

蓝色系：Qwen3.8 — Qwen3.8-27B / Qwen3.8-Flash-Next (FP8) / Qwen3.8-Max (FP8)（由浅到深，按模型大小）

橙色系：Qwen3 — Qwen3-30B-A3B-Base / Qwen3-235B-A22B (2504)（由浅到深，按模型大小）

青绿色系：MiMo — MiMo-V2.5-Pro-Base / MiMo-V2.6-Pro-RL（由浅到深）

黄色：DeepSeek-V4-Flash-Base

品红：Kimi-K3

紫色：Mistral-Large-4 (FP8)

| 线 | TriviaQA · EM | PopQA · EM | NQ · EM | SimpleQA · F1 | SimpleQA · correct |
|---|---|---|---|---|---|
| Kimi-K3 | 85.4 | 56.3 | 45.1 | 41.0 | 40.8 |
| DeepSeek-V4-Flash-Base | 83.9 | 48.5 | 40.5 | 23.2 | 19.0 |
| Qwen3.8-Max (FP8) | 82.1 | 49.8 | 9.9‡ | 0.5‡ | 0.2‡ |
| MiMo-V2.5-Pro-Base | 82.1 | 51.3 | 44.1 | 27.8 | 27.8 |
| MiMo-V2.6-Pro-RL | 80.8 | 48.4 | 42.1 | 21.3 | 20.9 |
| Mistral-Large-4 (FP8) | 79.3 | 34.0 | 36.5 | 14.8 | 14.4 |
| Qwen3-235B-A22B (2504) | 74.8 | 31.9 | 32.4 | 9.0 | 8.9 |
| Qwen3-30B-A3B-Base | 67.2 | 26.7 | 31.5 | 6.3 | 6.3 |
| Qwen3.8-27B | 19.9‡ | 4.0‡ | 19.7‡ | 0.0‡ | 0.0‡ |
| Qwen3.8-Flash-Next (FP8) | 0.8‡ | 0.0‡ | 0.2‡ | 0.0‡ | 0.0‡ |

| 模型 | 大小 | TriviaQA<br>EM | PopQA<br>EM | Natural Questions<br>EM | SimpleQA<br>F1 | SimpleQA<br>correct | SimpleQA<br>not_attempted |
|---|---|---|---|---|---|---|---|
| MiMo-V2.5-Pro-Base | 1.02T-A42B | 82.1 | 51.3 | 44.1 | 27.8 | 27.8 | 0.2 |
| MiMo-V2.6-Pro-RL 后训练 | 1.02T-A42B | 80.8 | 48.4 | 42.1 | 21.3 | 20.9 | 3.5 |
| Qwen3-30B-A3B-Base | 30B-A3B | 67.2 | 26.7 | 31.5 | 6.3 | 6.3 | 0.2 |
| Qwen3-235B-A22B (2504) 后训练 | 235B-A22B | 74.8 | 31.9 | 32.4 | 9.0 | 8.9 | 3.5 |
| DeepSeek-V4-Flash-Base | 284B-A13B | 83.9 | 48.5 | 40.5 | 23.2 | 19.0 | 36.5 |
| Kimi-K3 后训练 | 2.8T-A104B | 85.4 | 56.3 | 45.1 | 41.0 | 40.8 | 0.8 |
| Qwen3.8-Flash-Next (FP8) 后训练 | 125B-A6B | 0.8‡ | 0.0‡ | 0.2‡ | 0.0‡ | 0.0‡ | 100.0‡ |
| Qwen3.8-27B 后训练 | 27B | 19.9‡ | 4.0‡ | 19.7‡ | 0.0‡ | 0.0‡ | 100.0‡ |
| Qwen3.8-Max (FP8) 后训练 | 2.4T-A95B | 82.1 | 49.8 | 9.9‡ | 0.5‡ | 0.2‡ | 99.7‡ |
| Mistral-Large-4 (FP8) 后训练 | 1T-A52B | 79.3 | 34.0 | 36.5 | 14.8 | 14.4 | 5.2 |

‡ 灰色 = 这个模型在这一项有大量空回答（悬停看比例）：它在 few-shot 的 “A:” 后先输出换行，而这一项遇到换行就停止生成，回答被截成空字符串，分数不代表模型能力。只出现在部分后训练模型上（如 Qwen3.8-27B；MiMo-V2.6-Pro-RL 没有这个问题）。

##### 知识 · 多语言 knowledge · multilingual

知识：LLM 裁判 gpt-oss-120b 判对错；多语言：翻译

蓝色系：Qwen3.8 — Qwen3.8-27B / Qwen3.8-Flash-Next (FP8) / Qwen3.8-Max (FP8)（由浅到深，按模型大小）

橙色系：Qwen3 — Qwen3-30B-A3B-Base / Qwen3-235B-A22B (2504)（由浅到深，按模型大小）

青绿色系：MiMo — MiMo-V2.5-Pro-Base / MiMo-V2.6-Pro-RL（由浅到深）

黄色：DeepSeek-V4-Flash-Base

品红：Kimi-K3

紫色：Mistral-Large-4 (FP8)

| 线 | SimpleQA-V · 裁判准确率 | FACTS · 裁判准确率 | WMT24++ · chrF++ |
|---|---|---|---|
| Kimi-K3 | 45.6 | 50.3 | 12.1 |
| MiMo-V2.5-Pro-Base | 28.6 | 36.9 | 22.5 |
| DeepSeek-V4-Flash-Base | 25.6 | 23.2 | 23.4 |
| MiMo-V2.6-Pro-RL | 20.8 | 26.2 | 24.7 |
| Mistral-Large-4 (FP8) | 14.7 | 12.1 | 39.7 |
| Qwen3-235B-A22B (2504) | 10.5 | 5.3 | 37.0 |
| Qwen3.8-Max (FP8) | 10.2‡ | 3.1‡ | 12.7‡ |
| Qwen3-30B-A3B-Base | 6.9 | 3.4 | 35.6 |
| Qwen3.8-27B | 0.5‡ | 0.0‡ | 1.0‡ |
| Qwen3.8-Flash-Next (FP8) | 0.1‡ | 0.0‡ | 2.7‡ |

| 模型 | 大小 | SimpleQA-Verified<br>裁判准确率 | FACTS-parametric<br>裁判准确率 | WMT24++<br>chrF++ |
|---|---|---|---|---|
| MiMo-V2.5-Pro-Base | 1.02T-A42B | 28.6 | 36.9 | 22.5 |
| MiMo-V2.6-Pro-RL 后训练 | 1.02T-A42B | 20.8 | 26.2 | 24.7 |
| Qwen3-30B-A3B-Base | 30B-A3B | 6.9 | 3.4 | 35.6 |
| Qwen3-235B-A22B (2504) 后训练 | 235B-A22B | 10.5 | 5.3 | 37.0 |
| DeepSeek-V4-Flash-Base | 284B-A13B | 25.6 | 23.2 | 23.4 |
| Kimi-K3 后训练 | 2.8T-A104B | 45.6 | 50.3 | 12.1 |
| Qwen3.8-Flash-Next (FP8) 后训练 | 125B-A6B | 0.1‡ | 0.0‡ | 2.7‡ |
| Qwen3.8-27B 后训练 | 27B | 0.5‡ | 0.0‡ | 1.0‡ |
| Qwen3.8-Max (FP8) 后训练 | 2.4T-A95B | 10.2‡ | 3.1‡ | 12.7‡ |
| Mistral-Large-4 (FP8) 后训练 | 1T-A52B | 14.7 | 12.1 | 39.7 |

‡ 灰色 = 这个模型在这一项有大量空回答（悬停看比例）：它在 few-shot 的 “A:” 后先输出换行，而这一项遇到换行就停止生成，回答被截成空字符串，分数不代表模型能力。只出现在部分后训练模型上（如 Qwen3.8-27B；MiMo-V2.6-Pro-RL 没有这个问题）。

## 页签：LR 拟合实验

最优 LR 的扫描与拟合（MuonH 网格、Muon 网格、深度 × 宽度假设检查）。数据同 doc-isolated tab（我们的评测）。

### MuonH 学习率：每个格子找最优 LR（同一 MuonH、CE）

本节固定只看 **overall_tsqw**（token-sqrt-weighted：全部 7 个集合，即 val_v2、uncheatable、codefresh、v3small、private_internal、pile_test、**pile_10k**，各自的 token 加权 loss 再按 √token 数加权平均；权重约 uncheatable 31%、pile_test 23%、val_v2 15%、codefresh 13%、pile_10k 8%、private_internal 5%、v3small 5%，介于 tw 和 macro 之间，避免最长或最短的集合主导；和 overall_tw / macro 不同，pile_10k 也算在内），不随上方的指标切换。 每个格子 = 一个固定的（模型大小，训练 token 数）。格子里按 HyperP 公式算出的 LR 的 0.5 / 0.7 / 1 / 1.2 / 1.4 / 2（/ 2.8 / 4）倍各训一个模型，纵轴 = 相对该格 1× 的 loss 差（越低越好）；格子里的大数字 = 拟合出的最优倍数 **m***（抛物线顶点，可能外推到已测倍数之外；m* > 1 表示公式算低了），下面是对应的绝对 LR **η***（括号里是该格 1× 的 LR，即 HyperP 公式按层数和训练 token 数算出的 LR）和最优点比 1× 好多少；主线 TPP 200 的格子沿对角线排列；**鼠标移到格子上显示该格的 LR 测试曲线**（虚线 = 在 log(倍数) 上拟合的抛物线，橙线 = m*）。同一列看深度、q0 一行看训练量。**不同格子的训练量不同，只在格子内部比较。**空白 = 不在设计里。 **所有倍数都相对 ArchScale 原版公式 η = 0.016 × (L/8)^−0.5 × (T/10.4B)^−0.32，由每个 run 实际的峰值 LR 计算（不看 Change ID 的名字），不随 ladder 改用的新公式变化。**

|   | 9.6B token | ~30B token | 71.1B token | 86.4B token | 136.6B token |
|---|---|---|---|---|---|
| q0<br>12 层 · 48M | 主线 TPP 200 · 1.23× · η* = 0.0165 (1× = 0.0134) · -0.0024 vs 1× · 16 个点（3 个 seed） | 28.8B，TPP 600 · 1.32× · η* = 0.0125 (1× = 0.0094) · -0.0050 vs 1× · 5 个点 |   | 86.4B，TPP 1800 · 1.51× · η* = 0.0100 (1× = 0.0066) · -0.0058 vs 1× · 5 个点 |   |
| q1<br>18 层 · 154M | 9.6B，TPP 62（IT4577） · 1.48× · η* = 0.0162 (1× = 0.0109) · -0.0149 vs 1× · 5 个点 | 主线 TPP 200 · 1.52× · η* = 0.0114 (1× = 0.0075) · -0.0117 vs 1× · 5 个点 |   |   |   |
| q2<br>24 层 · 356M | 9.6B，TPP 27（IT4577） · 1.62× · η* = 0.0153 (1× = 0.0095) · -0.0232 vs 1× · 15 个点（3 个 seed） | 30.8B，TPP 87 · 1.70× · η* = 0.0111 (1× = 0.0065) · -0.0189 vs 1× · 5 个点 | 主线 TPP 200 · 1.73× · η* = 0.0086 (1× = 0.0050) · -0.0169 vs 1× · 5 个点 |   |   |
| q3<br>30 层 · 683M | 9.6B，TPP 14（IT4577） · 1.66× · η* = 0.0141 (1× = 0.0085) · -0.0338 vs 1× · 6 个点 | 30.8B，TPP 45 · 1.76× · η* = 0.0103 (1× = 0.0058) · -0.0278 vs 1× · 6 个点 |   |   | 主线 TPP 200（不在 TODO 3 里） · 待测1× = 0.0036 |
| q4<br>36 层 · 1.17B | 9.6B，TPP 8（IT4577） · 1.76× · η* = 0.0136 (1× = 0.0077) · -0.0391 vs 1× · 5 个点 |   |   |   |   |
| q5<br>42 层 · 1.84B | 9.6B，TPP 5（IT4577） · 1.82× · η* = 0.0131 (1× = 0.0072) · -0.0400 vs 1× · 5 个点 |   |   |   |   |
| q6<br>48 层 · 2.73B | 9.6B，TPP 3.5（IT4577） · 1.80× · η* = 0.0121 (1× = 0.0067) · -0.0425 vs 1× · 5 个点 |   |   |   |   |

**LR 扫描点：q0 / 9.6B token**（格子：主线 TPP 200 · 1.23× · η* = 0.0165 (1× = 0.0134) · -0.0024 vs 1× · 16 个点（3 个 seed））

| 横轴 | overall_tsqw |
|---|---|
| q0 · 0.699967× (η = 0.00938) · seed 3 | +0.0260 vs 1× (按 seed 偏移对齐) |
| q0 · 0.99974× (η = 0.01340) · seed 3 | +0.0015 vs 1× (按 seed 偏移对齐) |
| q0 · 1.19969× (η = 0.01608) · seed 3 | -0.0008 vs 1× (按 seed 偏移对齐) |
| q0 · 1.39964× (η = 0.01876) · seed 3 | +0.0021 vs 1× (按 seed 偏移对齐) |
| q0 · 2.00023× (η = 0.02681) · seed 3 | +0.0137 vs 1× (按 seed 偏移对齐) |
| q0 · 0.699967× (η = 0.00938) · seed 2 | +0.0238 vs 1× (按 seed 偏移对齐) |
| q0 · 0.99974× (η = 0.01340) · seed 2 | +0.0016 vs 1× (按 seed 偏移对齐) |
| q0 · 1.19969× (η = 0.01608) · seed 2 | -0.0005 vs 1× (按 seed 偏移对齐) |
| q0 · 1.39964× (η = 0.01876) · seed 2 | -0.0016 vs 1× (按 seed 偏移对齐) |
| q0 · 2.00023× (η = 0.02681) · seed 2 | +0.0193 vs 1× (按 seed 偏移对齐) |
| q0 · 0.500019× (η = 0.00670) · seed 1 | +0.0705 vs 1× |
| q0 · 0.699967× (η = 0.00938) · seed 1 | +0.0264 vs 1× |
| q0 · 0.99974× (η = 0.01340) · seed 1 | +0.0000 vs 1× |
| q0 · 1.19969× (η = 0.01608) · seed 1 | -0.0038 vs 1× |
| q0 · 1.39964× (η = 0.01876) · seed 1 | -0.0026 vs 1× |
| q0 · 2.00023× (η = 0.02681) · seed 1 | +0.0216 vs 1× |

**LR 扫描点：q0 / ~30B token**（格子：28.8B，TPP 600 · 1.32× · η* = 0.0125 (1× = 0.0094) · -0.0050 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q0 · 0.700046× (η = 0.00660) · seed 1 | +0.0191 vs 1× |
| q0 · 1.00002× (η = 0.00943) · seed 1 | +0.0000 vs 1× |
| q0 · 1.20032× (η = 0.01132) · seed 1 | -0.0046 vs 1× |
| q0 · 1.39967× (η = 0.01320) · seed 1 | -0.0049 vs 1× |
| q0 · 1.99983× (η = 0.01886) · seed 1 | +0.0052 vs 1× |

**LR 扫描点：q0 / 86.4B token**（格子：86.4B，TPP 1800 · 1.51× · η* = 0.0100 (1× = 0.0066) · -0.0058 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q0 · 0.700023× (η = 0.00464) · seed 1 | +0.0136 vs 1× |
| q0 · 0.999925× (η = 0.00664) · seed 1 | +0.0000 vs 1× |
| q0 · 1.20006× (η = 0.00796) · seed 1 | -0.0048 vs 1× |
| q0 · 1.40005× (η = 0.00929) · seed 1 | -0.0050 vs 1× |
| q0 · 1.99985× (η = 0.01327) · seed 1 | -0.0032 vs 1× |

**LR 扫描点：q1 / 9.6B token**（格子：9.6B，TPP 62（IT4577） · 1.48× · η* = 0.0162 (1× = 0.0109) · -0.0149 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q1 · 0.700025× (η = 0.00766) · seed 1 | +0.0388 vs 1× |
| q1 · 0.999644× (η = 0.01094) · seed 1 | +0.0000 vs 1× |
| q1 · 1.19976× (η = 0.01313) · seed 1 | -0.0106 vs 1× |
| q1 · 1.39987× (η = 0.01532) · seed 1 | -0.0148 vs 1× |
| q1 · 2.0002× (η = 0.02189) · seed 1 | -0.0060 vs 1× |

**LR 扫描点：q1 / ~30B token**（格子：主线 TPP 200 · 1.52× · η* = 0.0114 (1× = 0.0075) · -0.0117 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q1 · 0.699946× (η = 0.00528) · seed 1 | +0.0261 vs 1× |
| q1 · 0.99996× (η = 0.00754) · seed 1 | +0.0000 vs 1× |
| q1 · 1.20006× (η = 0.00904) · seed 1 | -0.0076 vs 1× |
| q1 · 1.39989× (η = 0.01055) · seed 1 | -0.0128 vs 1× |
| q1 · 1.99966× (η = 0.01507) · seed 1 | -0.0065 vs 1× |

**LR 扫描点：q2 / 9.6B token**（格子：9.6B，TPP 27（IT4577） · 1.62× · η* = 0.0153 (1× = 0.0095) · -0.0232 vs 1× · 15 个点（3 个 seed））

| 横轴 | overall_tsqw |
|---|---|
| q2 · 0.699959× (η = 0.00663) · seed 3 | +0.0456 vs 1× (按 seed 偏移对齐) |
| q2 · 1.00003× (η = 0.00948) · seed 3 | +0.0014 vs 1× (按 seed 偏移对齐) |
| q2 · 1.19966× (η = 0.01137) · seed 3 | -0.0150 vs 1× (按 seed 偏移对齐) |
| q2 · 1.40013× (η = 0.01327) · seed 3 | -0.0205 vs 1× (按 seed 偏移对齐) |
| q2 · 2.00049× (η = 0.01896) · seed 3 | -0.0191 vs 1× (按 seed 偏移对齐) |
| q2 · 0.699959× (η = 0.00663) · seed 2 | +0.0453 vs 1× (按 seed 偏移对齐) |
| q2 · 1.00003× (η = 0.00948) · seed 2 | +0.0018 vs 1× (按 seed 偏移对齐) |
| q2 · 1.19966× (η = 0.01137) · seed 2 | -0.0150 vs 1× (按 seed 偏移对齐) |
| q2 · 1.40013× (η = 0.01327) · seed 2 | -0.0224 vs 1× (按 seed 偏移对齐) |
| q2 · 2.00049× (η = 0.01896) · seed 2 | -0.0175 vs 1× (按 seed 偏移对齐) |
| q2 · 0.699959× (η = 0.00663) · seed 1 | +0.0477 vs 1× |
| q2 · 1.00003× (η = 0.00948) · seed 1 | +0.0000 vs 1× |
| q2 · 1.19966× (η = 0.01137) · seed 1 | -0.0147 vs 1× |
| q2 · 1.40013× (η = 0.01327) · seed 1 | -0.0221 vs 1× |
| q2 · 2.00049× (η = 0.01896) · seed 1 | -0.0185 vs 1× |

**LR 扫描点：q2 / ~30B token**（格子：30.8B，TPP 87 · 1.70× · η* = 0.0111 (1× = 0.0065) · -0.0189 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q2 · 0.700055× (η = 0.00457) · seed 1 | +0.0352 vs 1× |
| q2 · 1.00006× (η = 0.00653) · seed 1 | +0.0000 vs 1× |
| q2 · 1.20001× (η = 0.00783) · seed 1 | -0.0095 vs 1× |
| q2 · 1.39996× (η = 0.00914) · seed 1 | -0.0169 vs 1× |
| q2 · 1.9995× (η = 0.01305) · seed 1 | -0.0170 vs 1× |

**LR 扫描点：q2 / 71.1B token**（格子：主线 TPP 200 · 1.73× · η* = 0.0086 (1× = 0.0050) · -0.0169 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q2 · 0.699918× (η = 0.00349) · seed 1 | +0.0292 vs 1× |
| q2 · 0.999912× (η = 0.00499) · seed 1 | +0.0000 vs 1× |
| q2 · 1.19997× (η = 0.00599) · seed 1 | -0.0093 vs 1× |
| q2 · 1.40004× (η = 0.00699) · seed 1 | -0.0145 vs 1× |
| q2 · 2.00002× (η = 0.00999) · seed 1 | -0.0156 vs 1× |

**LR 扫描点：q3 / 9.6B token**（格子：9.6B，TPP 14（IT4577） · 1.66× · η* = 0.0141 (1× = 0.0085) · -0.0338 vs 1× · 6 个点）

| 横轴 | overall_tsqw |
|---|---|
| q3 · 0.700003× (η = 0.00593) · seed 1 | +0.0528 vs 1× |
| q3 · 0.999988× (η = 0.00848) · seed 1 | +0.0000 vs 1× |
| q3 · 1.1997× (η = 0.01017) · seed 1 | -0.0213 vs 1× |
| q3 · 1.40024× (η = 0.01187) · seed 1 | -0.0309 vs 1× |
| q3 · 1.9995× (η = 0.01695) · seed 1 | -0.0323 vs 1× |
| q3 · 2.80048× (η = 0.02374) · seed 1 | -0.0007 vs 1× |

**LR 扫描点：q3 / ~30B token**（格子：30.8B，TPP 45 · 1.76× · η* = 0.0103 (1× = 0.0058) · -0.0278 vs 1× · 6 个点）

| 横轴 | overall_tsqw |
|---|---|
| q3 · 0.699946× (η = 0.00409) · seed 1 | +0.0399 vs 1× |
| q3 · 1.00007× (η = 0.00584) · seed 1 | +0.0000 vs 1× |
| q3 · 1.19998× (η = 0.00700) · seed 1 | -0.0148 vs 1× |
| q3 · 1.40006× (η = 0.00817) · seed 1 | -0.0240 vs 1× |
| q3 · 2.00082× (η = 0.01168) · seed 1 | -0.0292 vs 1× |
| q3 · 2.80081× (η = 0.01635) · seed 1 | -0.0089 vs 1× |

**LR 扫描点：q4 / 9.6B token**（格子：9.6B，TPP 8（IT4577） · 1.76× · η* = 0.0136 (1× = 0.0077) · -0.0391 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q4 · 1.00006× (η = 0.00774) · seed 1 | +0.0000 vs 1× |
| q4 · 1.3995× (η = 0.01083) · seed 1 | -0.0334 vs 1× |
| q4 · 2.00039× (η = 0.01548) · seed 1 | -0.0364 vs 1× |
| q4 · 2.80028× (η = 0.02167) · seed 1 | -0.0133 vs 1× |
| q4 · 3.99948× (η = 0.03095) · seed 1 | +0.0426 vs 1× |

**LR 扫描点：q5 / 9.6B token**（格子：9.6B，TPP 5（IT4577） · 1.82× · η* = 0.0131 (1× = 0.0072) · -0.0400 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q5 · 0.999936× (η = 0.00716) · seed 1 | +0.0000 vs 1× |
| q5 · 1.39997× (η = 0.01003) · seed 1 | -0.0341 vs 1× |
| q5 · 2.00015× (η = 0.01433) · seed 1 | -0.0400 vs 1× |
| q5 · 2.79993× (η = 0.02006) · seed 1 | -0.0171 vs 1× |
| q5 · 4.0003× (η = 0.02866) · seed 1 | +0.0257 vs 1× |

**LR 扫描点：q6 / 9.6B token**（格子：9.6B，TPP 3.5（IT4577） · 1.80× · η* = 0.0121 (1× = 0.0067) · -0.0425 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q6 · 1.00004× (η = 0.00670) · seed 1 | +0.0000 vs 1× |
| q6 · 1.39993× (η = 0.00938) · seed 1 | -0.0331 vs 1× |
| q6 · 1.99948× (η = 0.01340) · seed 1 | -0.0384 vs 1× |
| q6 · 2.79927× (η = 0.01876) · seed 1 | -0.0228 vs 1× |
| q6 · 4.00045× (η = 0.02681) · seed 1 | +0.0397 vs 1× |

#### 拟合：最优 LR 随层数和训练量的关系（ArchScale 形式）

|   |   |
|---|---|
| ■ ArchScale 原版（= 每个格子的 1×，ladder 现用的 LR） | η*(L, T) = 0.016 × (L/8)^−0.5 × (T/10.4B)^−0.32 |
| ■ 我们拟合的 | η*(L, T) = 0.0184 × (L/8)^−0.221 × (T/10.4B)^−0.259 |
| ■ Boxiang 的拟合 v2（10-25 冻结，MUONH_FIT 在用；val_v2、16 格） | η*(L, T) = 0.01725 × (L/8)^−0.176 × (T/10.4B)^−0.259 |

**3D 拟合图的数据**（横轴 层数 L，纵轴 训练 token T，高度 η*）

- 灰（参考）：η = 0.016 × (层数 L/8)^-0.5 × (T/10.4B)^-0.32
- 橙（我们拟合的）：η = 0.018364 × (层数 L/8)^-0.2211 × (T/10.4B)^-0.2592
- 青（Boxiang）：η = 0.01725 × (层数 L/8)^-0.1764 × (T/10.4B)^-0.2591

| 格子 | 层数 L | T | 实测最优 η* | 外推 |
|---|---|---|---|---|
| q0 · 12 层 · T 9.6B · TPP 200 | 12 | 9.6B | 0.01651 |   |
| q0 · 12 层 · T 28.8B · TPP 600 | 12 | 28.8B | 0.01249 |   |
| q0 · 12 层 · T 86.4B · TPP 1800 | 12 | 86.4B | 0.00999 |   |
| q1 · 18 层 · T 9.6B · TPP 62 | 18 | 9.6B | 0.01618 |   |
| q1 · 18 层 · T 30.8B · TPP 200 | 18 | 30.8B | 0.01142 |   |
| q2 · 24 层 · T 9.6B · TPP 27 | 24 | 9.6B | 0.01533 |   |
| q2 · 24 层 · T 30.8B · TPP 87 | 24 | 30.8B | 0.01109 |   |
| q2 · 24 层 · T 71.1B · TPP 200 | 24 | 71.1B | 0.00863 |   |
| q3 · 30 层 · T 9.6B · TPP 14 | 30 | 9.6B | 0.01411 |   |
| q4 · 36 层 · T 9.6B · TPP 8 | 36 | 9.6B | 0.01363 |   |
| q5 · 42 层 · T 9.6B · TPP 5 | 42 | 9.6B | 0.01305 |   |
| q6 · 48 层 · T 9.6B · TPP 3.5 | 48 | 9.6B | 0.01207 |   |
| q3 · 30 层 · T 30.8B · TPP 45 | 30 | 30.8B | 0.01028 |   |

### Muon 学习率：同样的格子（同一 Muon、CE）

本节固定只看 **overall_tsqw**（token-sqrt-weighted：全部 7 个集合，即 val_v2、uncheatable、codefresh、v3small、private_internal、pile_test、**pile_10k**，各自的 token 加权 loss 再按 √token 数加权平均；权重约 uncheatable 31%、pile_test 23%、val_v2 15%、codefresh 13%、pile_10k 8%、private_internal 5%、v3small 5%，介于 tw 和 macro 之间，避免最长或最短的集合主导；和 overall_tw / macro 不同，pile_10k 也算在内），不随上方的指标切换。 Boxiang 10-24 决定的 Muon 全因子网格：格子与上面 MuonH 相同，倍数 0.25 / 0.5 / 1 / 2 / 4×（间隔更粗，因为 Muon 的最优点位置未知）。1× = ladder 现用的 Muon LR（AdamW 规则，只随模型大小变，**不随训练 token 变**）。读法同上：大数字 = 抛物线顶点 m*，η* = 绝对 LR，鼠标移到格子上看曲线。目的：Muon 也用各自最优 LR 时，MuonH 是否仍然领先。 **倍数相对各 rung 现用的 Muon LR（固定值），由每个 run 实际的峰值 LR 计算。**

|   | 9.6B token | ~30B token | 71.1B token | 86.4B token | 136.6B token |
|---|---|---|---|---|---|
| q0<br>12 层 · 48M | 主线 TPP 200 · 1.83× · η* = 0.0056 (1× = 0.0031) · -0.0059 vs 1× · 5 个点 | 28.8B，TPP 600 · 1.60× · η* = 0.0049 (1× = 0.0031) · -0.0022 vs 1× · 5 个点 |   | 86.4B，TPP 1800 · 0.99× · η* = 0.0031 (1× = 0.0031) · +0.0002 vs 1× · 5 个点 |   |
| q1<br>18 层 · 154M | 9.6B，TPP 62（IT4577） · 2.05× · η* = 0.0043 (1× = 0.0021) · -0.0110 vs 1× · 5 个点 | 主线 TPP 200 · 1.53× · η* = 0.0032 (1× = 0.0021) · -0.0012 vs 1× · 5 个点 |   |   |   |
| q2<br>24 层 · 356M | 9.6B，TPP 27（IT4577） · 2.37× · η* = 0.0037 (1× = 0.0016) · -0.0157 vs 1× · 5 个点 |   | 主线 TPP 200 · 1.04× · η* = 0.0016 (1× = 0.0016) · +0.0006 vs 1× · 5 个点 |   |   |
| q3<br>30 层 · 683M | 9.6B，TPP 14（IT4577） · 2.46× · η* = 0.0031 (1× = 0.0013) · -0.0154 vs 1× · 5 个点 |   |   |   | 主线 TPP 200 · 0.74× · η* = 0.0009 (1× = 0.0013) · -0.0009 vs 1× · 3 个点 |

**LR 扫描点：q0 / 9.6B token**（格子：主线 TPP 200 · 1.83× · η* = 0.0056 (1× = 0.0031) · -0.0059 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q0 · 0.249968× (η = 0.00077) · seed 1 | +0.0486 vs 1× |
| q0 · 0.5× (η = 0.00154) · seed 1 | +0.0158 vs 1× |
| q0 · 1× (η = 0.00308) · seed 1 | +0.0000 vs 1× |
| q0 · 1.99968× (η = 0.00616) · seed 1 | -0.0059 vs 1× |
| q0 · 4× (η = 0.01232) · seed 1 | +0.0023 vs 1× |

**LR 扫描点：q0 / ~30B token**（格子：28.8B，TPP 600 · 1.60× · η* = 0.0049 (1× = 0.0031) · -0.0022 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q0 · 0.249968× (η = 0.00077) · seed 1 | +0.0331 vs 1× |
| q0 · 0.5× (η = 0.00154) · seed 1 | +0.0086 vs 1× |
| q0 · 1× (η = 0.00308) · seed 1 | +0.0000 vs 1× |
| q0 · 1.99968× (η = 0.00616) · seed 1 | +0.0008 vs 1× |
| q0 · 4× (η = 0.01232) · seed 1 | +0.0048 vs 1× |

**LR 扫描点：q0 / 86.4B token**（格子：86.4B，TPP 1800 · 0.99× · η* = 0.0031 (1× = 0.0031) · +0.0002 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q0 · 0.249968× (η = 0.00077) · seed 1 | +0.0260 vs 1× |
| q0 · 0.5× (η = 0.00154) · seed 1 | +0.0057 vs 1× |
| q0 · 1× (η = 0.00308) · seed 1 | +0.0000 vs 1× |
| q0 · 1.99968× (η = 0.00616) · seed 1 | +0.0080 vs 1× |
| q0 · 4× (η = 0.01232) · seed 1 | +0.0257 vs 1× |

**LR 扫描点：q1 / 9.6B token**（格子：9.6B，TPP 62（IT4577） · 2.05× · η* = 0.0043 (1× = 0.0021) · -0.0110 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q1 · 0.25× (η = 0.00052) · seed 1 | +0.0585 vs 1× |
| q1 · 0.5× (η = 0.00104) · seed 1 | +0.0152 vs 1× |
| q1 · 1× (η = 0.00209) · seed 1 | +0.0000 vs 1× |
| q1 · 2× (η = 0.00418) · seed 1 | -0.0109 vs 1× |
| q1 · 4× (η = 0.00835) · seed 1 | -0.0047 vs 1× |

**LR 扫描点：q1 / ~30B token**（格子：主线 TPP 200 · 1.53× · η* = 0.0032 (1× = 0.0021) · -0.0012 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q1 · 0.25× (η = 0.00052) · seed 1 | +0.0294 vs 1× |
| q1 · 0.5× (η = 0.00104) · seed 1 | +0.0114 vs 1× |
| q1 · 1× (η = 0.00209) · seed 1 | +0.0000 vs 1× |
| q1 · 2× (η = 0.00418) · seed 1 | -0.0008 vs 1× |
| q1 · 4× (η = 0.00835) · seed 1 | +0.0077 vs 1× |

**LR 扫描点：q2 / 9.6B token**（格子：9.6B，TPP 27（IT4577） · 2.37× · η* = 0.0037 (1× = 0.0016) · -0.0157 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q2 · 0.249937× (η = 0.00039) · seed 1 | +0.0656 vs 1× |
| q2 · 0.499937× (η = 0.00079) · seed 1 | +0.0180 vs 1× |
| q2 · 1× (η = 0.00158) · seed 1 | +0.0000 vs 1× |
| q2 · 1.99937× (η = 0.00316) · seed 1 | -0.0161 vs 1× |
| q2 · 3.99937× (η = 0.00632) · seed 1 | -0.0116 vs 1× |

**LR 扫描点：q2 / 71.1B token**（格子：主线 TPP 200 · 1.04× · η* = 0.0016 (1× = 0.0016) · +0.0006 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q2 · 0.249937× (η = 0.00039) · seed 1 | +0.0230 vs 1× |
| q2 · 0.499937× (η = 0.00079) · seed 1 | +0.0074 vs 1× |
| q2 · 1× (η = 0.00158) · seed 1 | +0.0000 vs 1× |
| q2 · 1.99937× (η = 0.00316) · seed 1 | +0.0053 vs 1× |
| q2 · 3.99937× (η = 0.00632) · seed 1 | +0.0207 vs 1× |

**LR 扫描点：q3 / 9.6B token**（格子：9.6B，TPP 14（IT4577） · 2.46× · η* = 0.0031 (1× = 0.0013) · -0.0154 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q3 · 0.249882× (η = 0.00032) · seed 1 | +0.0760 vs 1× |
| q3 · 0.499843× (η = 0.00064) · seed 1 | +0.0248 vs 1× |
| q3 · 1× (η = 0.00127) · seed 1 | +0.0000 vs 1× |
| q3 · 1.99921× (η = 0.00254) · seed 1 | -0.0130 vs 1× |
| q3 · 3.99843× (η = 0.00508) · seed 1 | -0.0124 vs 1× |

**LR 扫描点：q3 / 136.6B token**（格子：主线 TPP 200 · 0.74× · η* = 0.0009 (1× = 0.0013) · -0.0009 vs 1× · 3 个点）

| 横轴 | overall_tsqw |
|---|---|
| q3 · 0.499843× (η = 0.00064) · seed 1 | +0.0007 vs 1× |
| q3 · 1× (η = 0.00127) · seed 1 | +0.0000 vs 1× |
| q3 · 1.99921× (η = 0.00254) · seed 1 | +0.0090 vs 1× |

#### 拟合：Muon 最优 LR 随激活参数量 N 和训练量的关系（与 Muon 现用的 adamw_active 规则同一写法）

|   |   |
|---|---|
| ■ Muon / AdamW 现用 LR（ladder 的 adamw_active 规则，不是 ArchScale；每个格子的 1×） | η = 2.35e-3 × (N_active / 1.08e8)^−1/3（只随激活参数量 N 变，不含 T 项） |
| ■ 我们拟合的 | η* = 0.00501 × (N_active / 1.08e8)^−0.316 × (T/10.4B)^−0.378 |
| ■ Boxiang 的 Muon 拟合 v3（10-26 记录；val_v2、9 格 43 点；未用于实验；有效范围 L 12–30、T 9.6–137B） | η = 4.93e-3 × (N_active / 1.08e8)^−0.339 × (T/10.4B)^−0.361 |

**3D 拟合图的数据**（横轴 激活参数 N，纵轴 训练 token T，高度 η*）

- 灰（参考）：η = 0.00235 × (激活参数 N/1.08e+08)^-0.3333 × (T/10.4B)^0
- 橙（我们拟合的）：η = 0.0050113 × (激活参数 N/1.08e+08)^-0.3157 × (T/10.4B)^-0.3784
- 青（Boxiang）：η = 0.00493 × (激活参数 N/1.08e+08)^-0.339 × (T/10.4B)^-0.361

| 格子 | 激活参数 N | T | 实测最优 η* | 外推 |
|---|---|---|---|---|
| q0 · N 48M · T 9.6B · TPP 200 | 4.8e+07 | 9.6B | 0.00563 |   |
| q0 · N 48M · T 28.8B · TPP 600 | 4.8e+07 | 28.8B | 0.00493 |   |
| q0 · N 48M · T 86.4B · TPP 1800 | 4.8e+07 | 86.4B | 0.00305 |   |
| q1 · N 154M · T 9.6B · TPP 62 | 1.539e+08 | 9.6B | 0.00428 |   |
| q1 · N 154M · T 30.8B · TPP 200 | 1.539e+08 | 30.8B | 0.00319 |   |
| q2 · N 356M · T 9.6B · TPP 27 | 3.555e+08 | 9.6B | 0.00374 |   |
| q2 · N 356M · T 71.1B · TPP 200 | 3.555e+08 | 71.1B | 0.00165 |   |
| q3 · N 683M · T 9.6B · TPP 14 | 6.832e+08 | 9.6B | 0.00313 |   |
| q3 · N 683M · T 136.7B · TPP 200 | 6.832e+08 | 136.7B | 0.00094 |   |

### LR 公式的假设检查：深度 × 宽度，对角线用作 ArchScale 锚点（同一 MuonH、CE、训练量 9.6B）

Runlong 计划 TODO 3 第 27、30、35 步：训练量固定 9.6B（4,577 步，GBS 2M），在不同层数 L 和宽度 h 上各扫一遍 LR（ArchScale 公式的倍数），看最优倍数 m* 是否只随深度变（公式假设宽度无关）。对角线 = ladder 各 rung（q0–q6）；其余格子 = 额外的深度 / 宽度点。指标 overall_tsqw；格子读法同上（悬停看曲线）。注意 9.6B 对大模型是严重欠训练（q6 TPP 3.5）。

| L \ h | h 512 | h 768 | h 1024 | h 1280 | h 1536 | h 1792 | h 2048 | h 4096 |
|---|---|---|---|---|---|---|---|---|
| L 12 | ladder q0 · 1.23× · η* = 0.0165 (1× = 0.0134) · -0.0024 vs 1× · 16 个点（3 个 seed） |   | 额外格子 · 1.11× · η* = 0.0149 (1× = 0.0134) · -0.0029 vs 1× · 5 个点 |   |   |   |   |   |
| L 18 |   | ladder q1 · 1.48× · η* = 0.0162 (1× = 0.0109) · -0.0149 vs 1× · 5 个点 |   |   |   |   |   |   |
| L 24 | 额外格子 · 1.68× · η* = 0.0160 (1× = 0.0095) · -0.0250 vs 1× · 5 个点 |   | ladder q2 · 1.62× · η* = 0.0153 (1× = 0.0095) · -0.0232 vs 1× · 15 个点（3 个 seed） |   | 额外格子 · 1.43× · η* = 0.0135 (1× = 0.0095) · -0.0169 vs 1× · 5 个点 |   | 额外格子 · 0.99× · η* = 0.0093 (1× = 0.0095) · -0.0551 vs 1× · 5 个点 | router 崩溃（负载均衡 loss 9–16），不是 LR 点 · 0.97× · η* = 0.0092 (1× = 0.0095) · -0.0113 vs 1× · 3 个点 |
| L 30 |   |   |   | ladder q3 · 1.66× · η* = 0.0141 (1× = 0.0085) · -0.0338 vs 1× · 6 个点 |   |   |   |   |
| L 36 |   |   |   |   | ladder q4 · 1.76× · η* = 0.0136 (1× = 0.0077) · -0.0391 vs 1× · 5 个点 |   |   |   |
| L 42 |   |   |   |   |   | ladder q5 · 1.82× · η* = 0.0131 (1× = 0.0072) · -0.0400 vs 1× · 5 个点 |   |   |
| L 48 |   |   | 额外格子（无 1×：相对最小倍数） · 2.16× · η* = 0.0145 (1× = 0.0067) · -0.0199 vs 1× · 5 个点 |   |   |   | ladder q6 · 1.80× · η* = 0.0121 (1× = 0.0067) · -0.0425 vs 1× · 5 个点 |   |
| L 60 |   |   | 额外格子 · 2.40× · η* = 0.0144 (1× = 0.0060) · -0.0801 vs 1× · 5 个点 |   |   |   |   |   |
| L 96 |   |   | 额外格子（无 1×：相对最小倍数） · 2.98× · η* = 0.0141 (1× = 0.0047) · -0.0621 vs 1× · 5 个点 |   |   |   |   |   |

**LR 扫描点：L 12 / h 512**（格子：ladder q0 · 1.23× · η* = 0.0165 (1× = 0.0134) · -0.0024 vs 1× · 16 个点（3 个 seed））

| 横轴 | overall_tsqw |
|---|---|
| q0 · 0.699967× (η = 0.00938) · seed 3 | +0.0260 vs 1× (按 seed 偏移对齐) |
| q0 · 0.99974× (η = 0.01340) · seed 3 | +0.0015 vs 1× (按 seed 偏移对齐) |
| q0 · 1.19969× (η = 0.01608) · seed 3 | -0.0008 vs 1× (按 seed 偏移对齐) |
| q0 · 1.39964× (η = 0.01876) · seed 3 | +0.0021 vs 1× (按 seed 偏移对齐) |
| q0 · 2.00023× (η = 0.02681) · seed 3 | +0.0137 vs 1× (按 seed 偏移对齐) |
| q0 · 0.699967× (η = 0.00938) · seed 2 | +0.0238 vs 1× (按 seed 偏移对齐) |
| q0 · 0.99974× (η = 0.01340) · seed 2 | +0.0016 vs 1× (按 seed 偏移对齐) |
| q0 · 1.19969× (η = 0.01608) · seed 2 | -0.0005 vs 1× (按 seed 偏移对齐) |
| q0 · 1.39964× (η = 0.01876) · seed 2 | -0.0016 vs 1× (按 seed 偏移对齐) |
| q0 · 2.00023× (η = 0.02681) · seed 2 | +0.0193 vs 1× (按 seed 偏移对齐) |
| q0 · 0.500019× (η = 0.00670) · seed 1 | +0.0705 vs 1× |
| q0 · 0.699967× (η = 0.00938) · seed 1 | +0.0264 vs 1× |
| q0 · 0.99974× (η = 0.01340) · seed 1 | +0.0000 vs 1× |
| q0 · 1.19969× (η = 0.01608) · seed 1 | -0.0038 vs 1× |
| q0 · 1.39964× (η = 0.01876) · seed 1 | -0.0026 vs 1× |
| q0 · 2.00023× (η = 0.02681) · seed 1 | +0.0216 vs 1× |

**LR 扫描点：L 12 / h 1024**（格子：额外格子 · 1.11× · η* = 0.0149 (1× = 0.0134) · -0.0029 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q2 · 0.699967× (η = 0.00938) · seed 1 | +0.0195 vs 1× |
| q2 · 0.99974× (η = 0.01340) · seed 1 | +0.0000 vs 1× |
| q2 · 1.19969× (η = 0.01608) · seed 1 | -0.0031 vs 1× |
| q2 · 1.39964× (η = 0.01876) · seed 1 | +0.0018 vs 1× |
| q2 · 2.00023× (η = 0.02681) · seed 1 | +0.0339 vs 1× |

**LR 扫描点：L 18 / h 768**（格子：ladder q1 · 1.48× · η* = 0.0162 (1× = 0.0109) · -0.0149 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q1 · 0.700025× (η = 0.00766) · seed 1 | +0.0388 vs 1× |
| q1 · 0.999644× (η = 0.01094) · seed 1 | +0.0000 vs 1× |
| q1 · 1.19976× (η = 0.01313) · seed 1 | -0.0106 vs 1× |
| q1 · 1.39987× (η = 0.01532) · seed 1 | -0.0148 vs 1× |
| q1 · 2.0002× (η = 0.02189) · seed 1 | -0.0060 vs 1× |

**LR 扫描点：L 24 / h 512**（格子：额外格子 · 1.68× · η* = 0.0160 (1× = 0.0095) · -0.0250 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q0 · 0.699959× (η = 0.00663) · seed 1 | +0.0452 vs 1× |
| q0 · 1.00003× (η = 0.00948) · seed 1 | +0.0000 vs 1× |
| q0 · 1.19966× (η = 0.01137) · seed 1 | -0.0140 vs 1× |
| q0 · 1.40013× (η = 0.01327) · seed 1 | -0.0226 vs 1× |
| q0 · 2.00049× (η = 0.01896) · seed 1 | -0.0221 vs 1× |

**LR 扫描点：L 24 / h 1024**（格子：ladder q2 · 1.62× · η* = 0.0153 (1× = 0.0095) · -0.0232 vs 1× · 15 个点（3 个 seed））

| 横轴 | overall_tsqw |
|---|---|
| q2 · 0.699959× (η = 0.00663) · seed 3 | +0.0456 vs 1× (按 seed 偏移对齐) |
| q2 · 1.00003× (η = 0.00948) · seed 3 | +0.0014 vs 1× (按 seed 偏移对齐) |
| q2 · 1.19966× (η = 0.01137) · seed 3 | -0.0150 vs 1× (按 seed 偏移对齐) |
| q2 · 1.40013× (η = 0.01327) · seed 3 | -0.0205 vs 1× (按 seed 偏移对齐) |
| q2 · 2.00049× (η = 0.01896) · seed 3 | -0.0191 vs 1× (按 seed 偏移对齐) |
| q2 · 0.699959× (η = 0.00663) · seed 2 | +0.0453 vs 1× (按 seed 偏移对齐) |
| q2 · 1.00003× (η = 0.00948) · seed 2 | +0.0018 vs 1× (按 seed 偏移对齐) |
| q2 · 1.19966× (η = 0.01137) · seed 2 | -0.0150 vs 1× (按 seed 偏移对齐) |
| q2 · 1.40013× (η = 0.01327) · seed 2 | -0.0224 vs 1× (按 seed 偏移对齐) |
| q2 · 2.00049× (η = 0.01896) · seed 2 | -0.0175 vs 1× (按 seed 偏移对齐) |
| q2 · 0.699959× (η = 0.00663) · seed 1 | +0.0477 vs 1× |
| q2 · 1.00003× (η = 0.00948) · seed 1 | +0.0000 vs 1× |
| q2 · 1.19966× (η = 0.01137) · seed 1 | -0.0147 vs 1× |
| q2 · 1.40013× (η = 0.01327) · seed 1 | -0.0221 vs 1× |
| q2 · 2.00049× (η = 0.01896) · seed 1 | -0.0185 vs 1× |

**LR 扫描点：L 24 / h 1536**（格子：额外格子 · 1.43× · η* = 0.0135 (1× = 0.0095) · -0.0169 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q4 · 0.699959× (η = 0.00663) · seed 1 | +0.0397 vs 1× |
| q4 · 1.00003× (η = 0.00948) · seed 1 | +0.0000 vs 1× |
| q4 · 1.19966× (η = 0.01137) · seed 1 | -0.0143 vs 1× |
| q4 · 1.40013× (η = 0.01327) · seed 1 | -0.0184 vs 1× |
| q4 · 2.00049× (η = 0.01896) · seed 1 | -0.0035 vs 1× |

**LR 扫描点：L 24 / h 2048**（格子：额外格子 · 0.99× · η* = 0.0093 (1× = 0.0095) · -0.0551 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q4 · 0.699959× (η = 0.00663) · seed 1 | +0.0369 vs 1× |
| q4 · 1.00003× (η = 0.00948) · seed 1 | +0.0000 vs 1× |
| q4 · 1.19966× (η = 0.01137) · seed 1 | -0.0074 vs 1× |
| q4 · 1.40013× (η = 0.01327) · seed 1 | -0.0064 vs 1× |
| q4 · 2.00049× (η = 0.01896) · seed 1 | +0.4376 vs 1× |

**LR 扫描点：L 24 / h 4096**（格子：router 崩溃（负载均衡 loss 9–16），不是 LR 点 · 0.97× · η* = 0.0092 (1× = 0.0095) · -0.0113 vs 1× · 3 个点）

| 横轴 | overall_tsqw |
|---|---|
| q6 · 0.699959× (η = 0.00663) · seed 1 | +1.7484 vs 1× |
| q6 · 1.00003× (η = 0.00948) · seed 1 | +0.0000 vs 1× |
| q6 · 1.19966× (η = 0.01137) · seed 1 | +0.6900 vs 1× |

**LR 扫描点：L 30 / h 1280**（格子：ladder q3 · 1.66× · η* = 0.0141 (1× = 0.0085) · -0.0338 vs 1× · 6 个点）

| 横轴 | overall_tsqw |
|---|---|
| q3 · 0.700003× (η = 0.00593) · seed 1 | +0.0528 vs 1× |
| q3 · 0.999988× (η = 0.00848) · seed 1 | +0.0000 vs 1× |
| q3 · 1.1997× (η = 0.01017) · seed 1 | -0.0213 vs 1× |
| q3 · 1.40024× (η = 0.01187) · seed 1 | -0.0309 vs 1× |
| q3 · 1.9995× (η = 0.01695) · seed 1 | -0.0323 vs 1× |
| q3 · 2.80048× (η = 0.02374) · seed 1 | -0.0007 vs 1× |

**LR 扫描点：L 36 / h 1536**（格子：ladder q4 · 1.76× · η* = 0.0136 (1× = 0.0077) · -0.0391 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q4 · 1.00006× (η = 0.00774) · seed 1 | +0.0000 vs 1× |
| q4 · 1.3995× (η = 0.01083) · seed 1 | -0.0334 vs 1× |
| q4 · 2.00039× (η = 0.01548) · seed 1 | -0.0364 vs 1× |
| q4 · 2.80028× (η = 0.02167) · seed 1 | -0.0133 vs 1× |
| q4 · 3.99948× (η = 0.03095) · seed 1 | +0.0426 vs 1× |

**LR 扫描点：L 42 / h 1792**（格子：ladder q5 · 1.82× · η* = 0.0131 (1× = 0.0072) · -0.0400 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q5 · 0.999936× (η = 0.00716) · seed 1 | +0.0000 vs 1× |
| q5 · 1.39997× (η = 0.01003) · seed 1 | -0.0341 vs 1× |
| q5 · 2.00015× (η = 0.01433) · seed 1 | -0.0400 vs 1× |
| q5 · 2.79993× (η = 0.02006) · seed 1 | -0.0171 vs 1× |
| q5 · 4.0003× (η = 0.02866) · seed 1 | +0.0257 vs 1× |

**LR 扫描点：L 48 / h 1024**（格子：额外格子（无 1×：相对最小倍数） · 2.16× · η* = 0.0145 (1× = 0.0067) · -0.0199 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q2 · 1.39993× (η = 0.00938) · seed 1 | +0.0000 vs 1× |
| q2 · 1.99948× (η = 0.01340) · seed 1 | -0.0189 vs 1× |
| q2 · 2.79927× (η = 0.01876) · seed 1 | -0.0126 vs 1× |
| q2 · 4.00045× (η = 0.02681) · seed 1 | +0.0206 vs 1× |
| q2 · 5.60004× (η = 0.03753) · seed 1 | +0.0785 vs 1× |

**LR 扫描点：L 48 / h 2048**（格子：ladder q6 · 1.80× · η* = 0.0121 (1× = 0.0067) · -0.0425 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q6 · 1.00004× (η = 0.00670) · seed 1 | +0.0000 vs 1× |
| q6 · 1.39993× (η = 0.00938) · seed 1 | -0.0331 vs 1× |
| q6 · 1.99948× (η = 0.01340) · seed 1 | -0.0384 vs 1× |
| q6 · 2.79927× (η = 0.01876) · seed 1 | -0.0228 vs 1× |
| q6 · 4.00045× (η = 0.02681) · seed 1 | +0.0397 vs 1× |

**LR 扫描点：L 60 / h 1024**（格子：额外格子 · 2.40× · η* = 0.0144 (1× = 0.0060) · -0.0801 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q2 · 0.999964× (η = 0.00599) · seed 1 | +0.0000 vs 1× |
| q2 · 1.40002× (η = 0.00839) · seed 1 | -0.0494 vs 1× |
| q2 · 2.00026× (η = 0.01199) · seed 1 | -0.0772 vs 1× |
| q2 · 2.79936× (η = 0.01678) · seed 1 | -0.0770 vs 1× |
| q2 · 4.00052× (η = 0.02398) · seed 1 | -0.0532 vs 1× |

**LR 扫描点：L 96 / h 1024**（格子：额外格子（无 1×：相对最小倍数） · 2.98× · η* = 0.0141 (1× = 0.0047) · -0.0621 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q2 · 1.39992× (η = 0.00663) · seed 1 | +0.0000 vs 1× |
| q2 · 2.00007× (η = 0.00948) · seed 1 | -0.0430 vs 1× |
| q2 · 2.80026× (η = 0.01327) · seed 1 | -0.0616 vs 1× |
| q2 · 4.00097× (η = 0.01896) · seed 1 | -0.0543 vs 1× |
| q2 · 5.60052× (η = 0.02654) · seed 1 | -0.0173 vs 1× |

|   |   |
|---|---|
| ■ Runlong 的逐 rung LR | η(rung) = ArchScale(L, T) × m*_9.6B(rung) × (T / 9.6B)^0.03 |

m*_9.6B(rung) = 上表对角线上该 rung 实测的最优倍数（没有拟合）；0.03 是训练量的经验值。q4 已按此用 5.36e-3。

#### 其余检查：batch、LR floor、按参数组分 LR

每格和上面的 LR 格子一样：一个 LR 扫描，大数字 = 抛物线顶点 m*（相对 ArchScale 原版公式，按实际峰值 LR 计算），鼠标移到格子上看曲线。

| 检查 | 对照 | 实验 | 结论 |
|---|---|---|---|
| batch（q1，L 18，9.6B） | GBS 2M · 1.48× · η* = 0.0162 (1× = 0.0109) · -0.0149 vs 1× · 5 个点 | GBS 4M · 1.86× · η* = 0.0204 (1× = 0.0109) · -0.0396 vs 1× · 5 个点 | batch 翻倍、步数减半时最优倍数从 1.48 升到 1.86，最接近 Runlong 预先写下的“LR 按步数算”（1.81）→ 换 batch 时按步数调 LR。 |
| LR floor（q0，L 12） | 衰减到峰值的 1%（ladder） · 1.23× · η* = 0.0165 (1× = 0.0134) · -0.0024 vs 1× · 16 个点（3 个 seed） | 衰减到峰值的 10%（ArchScale） · 1.04× · η* = 0.0139 (1× = 0.0134) · -0.0000 vs 1× · 4 个点 | 衰减到 10% 时 ArchScale 原版几乎正好（m* ≈ 1）；ladder 衰减到 1%，需要高约 20% 的峰值 LR。 |
| LR floor（q3，L 30） | 衰减到峰值的 1%（ladder） · 1.66× · η* = 0.0141 (1× = 0.0085) · -0.0338 vs 1× · 6 个点 | 衰减到峰值的 10%（ArchScale） · 1.47× · η* = 0.0124 (1× = 0.0085) · -0.0235 vs 1× · 3 个点 | q3 上同样：10% floor 时最优点往低移。 |
| 按参数组分 LR（q3，L 30） | 全部参数一起改 LR · 1.66× · η* = 0.0141 (1× = 0.0085) · -0.0338 vs 1× · 6 个点 | 只给一组参数 1.74×（菱形） · 1.66× · η* = 0.0141 (1× = 0.0085) · -0.0338 vs 1× · 6 个点 | 只给 hidden 矩阵加 LR（C）就拿到几乎全部收益；只给 embedding / 1-D 参数（A）没有用。 |

**LR 扫描点：batch（q1，L 18，9.6B） / 对照**（格子：GBS 2M · 1.48× · η* = 0.0162 (1× = 0.0109) · -0.0149 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q1 · 0.700025× (η = 0.00766) · seed 1 | +0.0388 vs 1× |
| q1 · 0.999644× (η = 0.01094) · seed 1 | +0.0000 vs 1× |
| q1 · 1.19976× (η = 0.01313) · seed 1 | -0.0106 vs 1× |
| q1 · 1.39987× (η = 0.01532) · seed 1 | -0.0148 vs 1× |
| q1 · 2.0002× (η = 0.02189) · seed 1 | -0.0060 vs 1× |

**LR 扫描点：batch（q1，L 18，9.6B） / 实验**（格子：GBS 4M · 1.86× · η* = 0.0204 (1× = 0.0109) · -0.0396 vs 1× · 5 个点）

| 横轴 | overall_tsqw |
|---|---|
| q1 · 0.999714× (η = 0.01094) · seed 1 | +0.0000 vs 1× |
| q1 · 1.39997× (η = 0.01532) · seed 1 | -0.0296 vs 1× |
| q1 · 2.00034× (η = 0.02189) · seed 1 | -0.0390 vs 1× |
| q1 · 2.79993× (η = 0.03064) · seed 1 | -0.0237 vs 1× |
| q1 · 3.99977× (η = 0.04377) · seed 1 | +0.0221 vs 1× |

**LR 扫描点：LR floor（q0，L 12） / 对照**（格子：衰减到峰值的 1%（ladder） · 1.23× · η* = 0.0165 (1× = 0.0134) · -0.0024 vs 1× · 16 个点（3 个 seed））

| 横轴 | overall_tsqw |
|---|---|
| q0 · 0.699967× (η = 0.00938) · seed 3 | +0.0260 vs 1× (按 seed 偏移对齐) |
| q0 · 0.99974× (η = 0.01340) · seed 3 | +0.0015 vs 1× (按 seed 偏移对齐) |
| q0 · 1.19969× (η = 0.01608) · seed 3 | -0.0008 vs 1× (按 seed 偏移对齐) |
| q0 · 1.39964× (η = 0.01876) · seed 3 | +0.0021 vs 1× (按 seed 偏移对齐) |
| q0 · 2.00023× (η = 0.02681) · seed 3 | +0.0137 vs 1× (按 seed 偏移对齐) |
| q0 · 0.699967× (η = 0.00938) · seed 2 | +0.0238 vs 1× (按 seed 偏移对齐) |
| q0 · 0.99974× (η = 0.01340) · seed 2 | +0.0016 vs 1× (按 seed 偏移对齐) |
| q0 · 1.19969× (η = 0.01608) · seed 2 | -0.0005 vs 1× (按 seed 偏移对齐) |
| q0 · 1.39964× (η = 0.01876) · seed 2 | -0.0016 vs 1× (按 seed 偏移对齐) |
| q0 · 2.00023× (η = 0.02681) · seed 2 | +0.0193 vs 1× (按 seed 偏移对齐) |
| q0 · 0.500019× (η = 0.00670) · seed 1 | +0.0705 vs 1× |
| q0 · 0.699967× (η = 0.00938) · seed 1 | +0.0264 vs 1× |
| q0 · 0.99974× (η = 0.01340) · seed 1 | +0.0000 vs 1× |
| q0 · 1.19969× (η = 0.01608) · seed 1 | -0.0038 vs 1× |
| q0 · 1.39964× (η = 0.01876) · seed 1 | -0.0026 vs 1× |
| q0 · 2.00023× (η = 0.02681) · seed 1 | +0.0216 vs 1× |

**LR 扫描点：LR floor（q0，L 12） / 实验**（格子：衰减到峰值的 10%（ArchScale） · 1.04× · η* = 0.0139 (1× = 0.0134) · -0.0000 vs 1× · 4 个点）

| 横轴 | overall_tsqw |
|---|---|
| q0 · 0.699967× (η = 0.00938) · seed 1 | +0.0186 vs 1× |
| q0 · 0.99974× (η = 0.01340) · seed 1 | +0.0000 vs 1× |
| q0 · 1.39964× (η = 0.01876) · seed 1 | +0.0106 vs 1× |
| q0 · 2.00023× (η = 0.02681) · seed 1 | +0.0506 vs 1× |

**LR 扫描点：LR floor（q3，L 30） / 对照**（格子：衰减到峰值的 1%（ladder） · 1.66× · η* = 0.0141 (1× = 0.0085) · -0.0338 vs 1× · 6 个点）

| 横轴 | overall_tsqw |
|---|---|
| q3 · 0.700003× (η = 0.00593) · seed 1 | +0.0528 vs 1× |
| q3 · 0.999988× (η = 0.00848) · seed 1 | +0.0000 vs 1× |
| q3 · 1.1997× (η = 0.01017) · seed 1 | -0.0213 vs 1× |
| q3 · 1.40024× (η = 0.01187) · seed 1 | -0.0309 vs 1× |
| q3 · 1.9995× (η = 0.01695) · seed 1 | -0.0323 vs 1× |
| q3 · 2.80048× (η = 0.02374) · seed 1 | -0.0007 vs 1× |

**LR 扫描点：LR floor（q3，L 30） / 实验**（格子：衰减到峰值的 10%（ArchScale） · 1.47× · η* = 0.0124 (1× = 0.0085) · -0.0235 vs 1× · 3 个点）

| 横轴 | overall_tsqw |
|---|---|
| q3 · 0.999988× (η = 0.00848) · seed 1 | +0.0000 vs 1× |
| q3 · 1.40024× (η = 0.01187) · seed 1 | -0.0232 vs 1× |
| q3 · 1.9995× (η = 0.01695) · seed 1 | -0.0083 vs 1× |

**LR 扫描点：按参数组分 LR（q3，L 30） / 对照**（格子：全部参数一起改 LR · 1.66× · η* = 0.0141 (1× = 0.0085) · -0.0338 vs 1× · 6 个点）

| 横轴 | overall_tsqw |
|---|---|
| q3 · 0.700003× (η = 0.00593) · seed 1 | +0.0528 vs 1× |
| q3 · 0.999988× (η = 0.00848) · seed 1 | +0.0000 vs 1× |
| q3 · 1.1997× (η = 0.01017) · seed 1 | -0.0213 vs 1× |
| q3 · 1.40024× (η = 0.01187) · seed 1 | -0.0309 vs 1× |
| q3 · 1.9995× (η = 0.01695) · seed 1 | -0.0323 vs 1× |
| q3 · 2.80048× (η = 0.02374) · seed 1 | -0.0007 vs 1× |

**LR 扫描点：按参数组分 LR（q3，L 30） / 实验**（格子：只给一组参数 1.74×（菱形） · 1.66× · η* = 0.0141 (1× = 0.0085) · -0.0338 vs 1× · 6 个点）

| 线 | q3 · 0.700003× (η = 0.00593) · seed 1 | q3 · 0.999988× (η = 0.00848) · seed 1 | q3 · 1.1997× (η = 0.01017) · seed 1 | q3 · 1.40024× (η = 0.01187) · seed 1 | q3 · 1.9995× (η = 0.01695) · seed 1 | q3 · 2.80048× (η = 0.02374) · seed 1 | x |
|---|---|---|---|---|---|---|---|
| overall_tsqw | +0.0528 vs 1× | +0.0000 vs 1× | -0.0213 vs 1× | -0.0309 vs 1× | -0.0323 vs 1× | -0.0007 vs 1× | · |
| A：只给 AdamW 组（embedding、1-D） | · | · | · | · | · | · | +0.0007 vs 1× |
| B：hidden 矩阵 + 输出头 + router | · | · | · | · | · | · | -0.0355 vs 1× |
| C：只给 hidden 矩阵 | · | · | · | · | · | · | -0.0325 vs 1× |

## 页签：packed（Boxiang 训练时验证）

- 数据来自 Boxiang 的 `results/<Change ID>/scores.json`：训练结束时自带的 **packed** 验证（文档首尾拼接、不加 attention mask），我们不重新测。
- 只有他的 6 个合并验证集（val_v2 不拆 split，没有 private_internal）；“六集平均”= 这 6 个的简单平均（Boxiang 的定义）。
- packed 比 doc-isolated 高约 0.013–0.016，**两个 tab 的绝对值不能互相比较**，只在同一 tab 内比差值。

### 优化器对比

虚线 = Boxiang 按新拟合的 LR 公式跑的同一优化器（MuonH：fit v2 η = 0.01725 · (L/8)^−0.176 · (T/10.4B)^−0.259，目前有 q2 / q3；Muon 没有按新公式跑）。同一数据、同一 token 数，只换优化器（及其 LR 规则 / schedule）。横轴：激活参数（不含 embedding，对数坐标）。

**相对 AdamW 的 Δloss（优化器 − AdamW，越负越好）**

**〔指标：val_v2〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| Muon | -0.0476 | -0.0219 | -0.0168 | -0.0121 | -0.0104 |
| MuonH | -0.0640 | -0.0308 | -0.0225 | -0.0171 | -0.0151 |
| MuonH（新拟合 LR，MUONH_FIT） | -0.0674 | -0.0387 | -0.0336 | -0.0278 | · |

**〔指标：uncheatable〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| Muon | -0.0652 | -0.0320 | -0.0230 | -0.0189 | -0.0150 |
| MuonH | -0.0829 | -0.0417 | -0.0263 | -0.0177 | -0.0107 |
| MuonH（新拟合 LR，MUONH_FIT） | -0.0876 | -0.0532 | -0.0457 | -0.0371 | · |

**〔指标：codefresh〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| Muon | -0.0797 | -0.0312 | -0.0232 | -0.0142 | -0.0133 |
| MuonH | -0.0938 | -0.0403 | -0.0222 | -0.0110 | -0.0084 |
| MuonH（新拟合 LR，MUONH_FIT） | -0.1007 | -0.0556 | -0.0422 | -0.0309 | · |

**〔指标：六集平均〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| Muon | -0.0602 | -0.0283 | -0.0218 | -0.0161 | -0.0127 |
| MuonH | -0.0772 | -0.0387 | -0.0258 | -0.0189 | -0.0137 |
| MuonH（新拟合 LR，MUONH_FIT） | -0.0824 | -0.0497 | -0.0399 | -0.0266 | · |

### KD 对 CE

CE 与 KD 只差 loss（α = 0 vs 0.7 / 0.8 / 0.9），优化器、数据顺序、layout 完全相同。

**KD − CE：各 α 的 Δloss（越负越好）**

**〔指标：val_v2〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| KD α 0.7 | -0.0265 | -0.0304 | -0.0308 | -0.0346 | -0.0367 |
| KD α 0.8 | -0.0277 | -0.0303 | -0.0312 | -0.0348 | -0.0339 |
| KD α 0.9 | -0.0285 | -0.0310 | -0.0305 | -0.0339 | -0.0378 |

**〔指标：uncheatable〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| KD α 0.7 | -0.0455 | -0.0533 | -0.0545 | -0.0604 | -0.0628 |
| KD α 0.8 | -0.0482 | -0.0528 | -0.0577 | -0.0610 | -0.0639 |
| KD α 0.9 | -0.0489 | -0.0564 | -0.0584 | -0.0620 | -0.0650 |

**〔指标：codefresh〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| KD α 0.7 | -0.0431 | -0.0478 | -0.0501 | -0.0546 | -0.0573 |
| KD α 0.8 | -0.0476 | -0.0504 | -0.0529 | -0.0570 | -0.0590 |
| KD α 0.9 | -0.0479 | -0.0519 | -0.0523 | -0.0564 | -0.0611 |

**〔指标：六集平均〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M | q4 · 1.17B |
|---|---|---|---|---|---|
| KD α 0.7 | -0.0370 | -0.0417 | -0.0435 | -0.0474 | -0.0495 |
| KD α 0.8 | -0.0383 | -0.0415 | -0.0444 | -0.0479 | -0.0465 |
| KD α 0.9 | -0.0380 | -0.0422 | -0.0437 | -0.0464 | -0.0455 |

### 尾部估算方式

8 个 run 只差 KD 项里 teacher 分布的截断方式：α = 0.8、MuonH 优化器、seed 1、数据顺序全部相同；基线仍是 **CE**（MuonH，α = 0）；**B64**（top-64 + 剩余 tail 合成一个 bucket）就是 KD α 0.8 本身，作为一条线一起画出来，各变体看是否比 B64 更低。负数 = 比 CE 好。

**命名：字母 = tail 的处理方式 · 数字 = top-k 的大小 · 末尾 Y = 是否把 y 单独精确算**

| 部分 | 含义 |
|---|---|
| B | Bucket：top-k 以外的 tail 合成一个整体，只要求 student 在 tail 上的总概率对 |
| S | Sampled：tail 用 cache 里预存的 4 个采样 token 来估计，结果无偏 |
| DD | Data Draw：不用 teacher 的 tail，改用语料里真实的下一个 token 近似 tail |
| DROP | 丢掉 tail：teacher 只在 top-k 内重新归一化（对照组） |
| 64 / 128 | 精确计算的 top-k 有多少个 token |
| Y | y exact：当真实的下一个 token y 落在 top-k 以外时，把 y 从 tail 里拆出来单独精确算 |

**Δloss（KD 变体 − CE），越低越好**

**〔指标：val_v2〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| B64 · top-64 + bucket（= KD α 0.8） | -0.0277 | -0.0303 | -0.0312 | -0.0348 |
| S64Y · top-64 + 采样 tail + y exact | -0.0311 | -0.0335 | -0.0334 | -0.0368 |
| S64 · top-64 + 采样 tail（4 draws） | -0.0290 | -0.0335 | -0.0342 | -0.0371 |
| S128 · top-128 + 采样 tail | -0.0291 | -0.0336 | -0.0337 | · |
| B64Y · top-64 bucket + y exact | -0.0279 | -0.0312 | -0.0311 | · |
| B128 · top-128 + bucket | -0.0282 | -0.0311 | -0.0317 | · |
| DD64 · top-64 + data draw | -0.0274 | -0.0310 | -0.0303 | · |
| DROP64 · top-64，丢掉 tail（对照） | -0.0287 | · | · | · |
| S128Y · top-128 + 采样 tail + y exact | · | · | · | -0.0369 |

**〔指标：uncheatable〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| B64 · top-64 + bucket（= KD α 0.8） | -0.0482 | -0.0528 | -0.0577 | -0.0610 |
| S64Y · top-64 + 采样 tail + y exact | -0.0534 | -0.0576 | -0.0605 | -0.0652 |
| S64 · top-64 + 采样 tail（4 draws） | -0.0498 | -0.0583 | -0.0604 | -0.0653 |
| S128 · top-128 + 采样 tail | -0.0496 | -0.0568 | -0.0608 | · |
| B64Y · top-64 bucket + y exact | -0.0477 | -0.0549 | -0.0565 | · |
| B128 · top-128 + bucket | -0.0479 | -0.0542 | -0.0570 | · |
| DD64 · top-64 + data draw | -0.0468 | -0.0509 | -0.0540 | · |
| DROP64 · top-64，丢掉 tail（对照） | -0.0491 | · | · | · |
| S128Y · top-128 + 采样 tail + y exact | · | · | · | -0.0649 |

**〔指标：codefresh〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| B64 · top-64 + bucket（= KD α 0.8） | -0.0476 | -0.0504 | -0.0529 | -0.0570 |
| S64Y · top-64 + 采样 tail + y exact | -0.0524 | -0.0545 | -0.0577 | -0.0604 |
| S64 · top-64 + 采样 tail（4 draws） | -0.0498 | -0.0563 | -0.0570 | -0.0592 |
| S128 · top-128 + 采样 tail | -0.0495 | -0.0543 | -0.0554 | · |
| B64Y · top-64 bucket + y exact | -0.0454 | -0.0521 | -0.0533 | · |
| B128 · top-128 + bucket | -0.0478 | -0.0496 | -0.0542 | · |
| DD64 · top-64 + data draw | -0.0464 | -0.0494 | -0.0509 | · |
| DROP64 · top-64，丢掉 tail（对照） | -0.0489 | · | · | · |
| S128Y · top-128 + 采样 tail + y exact | · | · | · | -0.0609 |

**〔指标：六集平均〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| B64 · top-64 + bucket（= KD α 0.8） | -0.0383 | -0.0415 | -0.0444 | -0.0479 |
| S64Y · top-64 + 采样 tail + y exact | -0.0432 | -0.0460 | -0.0479 | -0.0517 |
| S64 · top-64 + 采样 tail（4 draws） | -0.0403 | -0.0463 | -0.0483 | -0.0515 |
| S128 · top-128 + 采样 tail | -0.0398 | -0.0457 | -0.0477 | · |
| B64Y · top-64 bucket + y exact | -0.0375 | -0.0426 | -0.0439 | · |
| B128 · top-128 + bucket | -0.0387 | -0.0434 | -0.0448 | · |
| DD64 · top-64 + data draw | -0.0392 | -0.0415 | -0.0434 | · |
| DROP64 · top-64，丢掉 tail（对照） | -0.0394 | · | · | · |
| S128Y · top-128 + 采样 tail + y exact | · | · | · | -0.0519 |

### 老师 MiMo-2.6 vs. MiMo-2.5

0 线 = **同一条 CE：OPT_MUONH**（MuonH 老 LR，650B 数据；q0 为 3 个 seed 的平均），三个色系都减这一条。每个点 = 该 KD 臂的 seed 平均 − OPT_MUONH；q0 的 3 个 seed 平均成一个点。
绿 vs 橙：LR、45B KD corpus v1（tc_small cache）、seed 和数据顺序完全相同，只有 teacher 不同，两者的距离 = 2.5 − 2.6。蓝 vs 绿：同是 MiMo-2.6，但 LR 和 KD 数据都不同（蓝色系用 650B 数据），所以两者的距离不只是 LR 的影响。蓝色系的 S128Y 只跑了 q3。 这里是 Boxiang 式的数值：训练结束时的 packed 验证（6 个 decon 集合，mean6 = 六个的简单平均），取自训练日志。

**各 KD 臂 − CE（鼠标移到某个 rung 看所有线的数值）**

**蓝色系**：MiMo-2.6 teacher，α 0.8，MuonH **老 LR**（ArchScale 原公式），650B 数据— S64 / S64Y / S128Y（由浅到深；S128Y 只有 q3）

**绿色系**：MiMo-2.6 teacher，α 0.8，MuonH **新 LR 公式**（fit），tc_small 45B — S64 / S64Y / S128Y（由浅到深）

**橙色系**：MiMo-2.5 teacher，α 0.8，MuonH **新 LR 公式**（fit），tc_small 45B — S64 / S64Y / S128Y（由浅到深）

纵轴 = KD − CE：OPT_MUONH（MuonH 老 LR），三个色系都减这同一条；纵轴范围 = 当前数据的最小值到最大值。绿 / 橙色系（45B 语料）的 q3 用的是 TPP 100，和其他点不可比，不画。

**〔指标：六集平均〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MiMo-2.6 老 LR S64 | -0.0427 | -0.0463 | -0.0483 | -0.0515 |
| MiMo-2.6 老 LR S64Y | -0.0456 | -0.0460 | -0.0479 | -0.0517 |
| MiMo-2.6 老 LR S128Y | · | · | · | -0.0519 |
| MiMo-2.6 新 LR S64 | -0.0528 | -0.0610 | -0.0643 | · |
| MiMo-2.6 新 LR S64Y | -0.0525 | -0.0613 | -0.0645 | · |
| MiMo-2.6 新 LR S128Y | -0.0526 | -0.0615 | -0.0637 | · |
| MiMo-2.5 新 LR S64 | -0.0561 | -0.0638 | -0.0661 | · |
| MiMo-2.5 新 LR S64Y | -0.0559 | -0.0635 | -0.0655 | · |
| MiMo-2.5 新 LR S128Y | -0.0554 | -0.0629 | -0.0675 | · |

**〔指标：val_v2〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MiMo-2.6 老 LR S64 | -0.0303 | -0.0335 | -0.0342 | -0.0371 |
| MiMo-2.6 老 LR S64Y | -0.0324 | -0.0335 | -0.0334 | -0.0368 |
| MiMo-2.6 老 LR S128Y | · | · | · | -0.0369 |
| MiMo-2.6 新 LR S64 | -0.0390 | -0.0439 | -0.0442 | · |
| MiMo-2.6 新 LR S64Y | -0.0387 | -0.0440 | -0.0450 | · |
| MiMo-2.6 新 LR S128Y | -0.0392 | -0.0439 | -0.0444 | · |
| MiMo-2.5 新 LR S64 | -0.0420 | -0.0493 | -0.0497 | · |
| MiMo-2.5 新 LR S64Y | -0.0422 | -0.0495 | -0.0493 | · |
| MiMo-2.5 新 LR S128Y | -0.0426 | -0.0488 | -0.0482 | · |

**〔指标：uncheatable〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MiMo-2.6 老 LR S64 | -0.0511 | -0.0583 | -0.0604 | -0.0653 |
| MiMo-2.6 老 LR S64Y | -0.0547 | -0.0576 | -0.0605 | -0.0652 |
| MiMo-2.6 老 LR S128Y | · | · | · | -0.0649 |
| MiMo-2.6 新 LR S64 | -0.0588 | -0.0731 | -0.0767 | · |
| MiMo-2.6 新 LR S64Y | -0.0580 | -0.0717 | -0.0774 | · |
| MiMo-2.6 新 LR S128Y | -0.0586 | -0.0730 | -0.0757 | · |
| MiMo-2.5 新 LR S64 | -0.0590 | -0.0745 | -0.0781 | · |
| MiMo-2.5 新 LR S64Y | -0.0592 | -0.0730 | -0.0780 | · |
| MiMo-2.5 新 LR S128Y | -0.0589 | -0.0736 | -0.0765 | · |

**〔指标：codefresh〕**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| MiMo-2.6 老 LR S64 | -0.0518 | -0.0563 | -0.0570 | -0.0592 |
| MiMo-2.6 老 LR S64Y | -0.0543 | -0.0545 | -0.0577 | -0.0604 |
| MiMo-2.6 老 LR S128Y | · | · | · | -0.0609 |
| MiMo-2.6 新 LR S64 | -0.0608 | -0.0700 | -0.0714 | · |
| MiMo-2.6 新 LR S64Y | -0.0619 | -0.0696 | -0.0710 | · |
| MiMo-2.6 新 LR S128Y | -0.0601 | -0.0701 | -0.0700 | · |
| MiMo-2.5 新 LR S64 | -0.0628 | -0.0713 | -0.0741 | · |
| MiMo-2.5 新 LR S64Y | -0.0633 | -0.0693 | -0.0762 | · |
| MiMo-2.5 新 LR S128Y | -0.0624 | -0.0704 | -0.0695 | · |

### 滑动窗口注意力 SWA

设置：MuonH，新 LR 公式（fit v2），纯 CE（不用 KD），128 选 8 MoE；q0–q3，部分组 q0 有 3 个 seed。
左图：softmax router，0 线 = **MUONH_FIT**（全注意力）；右图：DSV3 router，0 线 = **MUONH_FIT_DSV3**（全注意力）。数值 = 改动 − 0 线（doc-isolated，负数 = 改动更好）；各改动的具体设定见下表。

| 改动 | Change ID | 具体改了什么 | 对照 | LR 公式（按实际峰值 LR 判断） | 已有结果 |
|---|---|---|---|---|---|
| ■ SWA（基础版） | `SWA_FIT` | 每 6 层中 5 层滑动窗口注意力（窗口 2048，RoPE）+ 1 层全注意力（NoPE） | `MUONH_FIT` | 新公式（fit v2） | q0, q1, q2, q3 |
| ■ SWA，全注意力层用 RoPE | `SWA_FIT_ROPE` | 同基础版，但全注意力层也用 RoPE | `MUONH_FIT` | 新公式（fit v2） | q0, q1, q2, q3 |
| ■ SWA，窗口 4096 | `SWA_FIT_W4K` | 同基础版，窗口 4096 | `MUONH_FIT` | 新公式（fit v2） | q0, q1, q2, q3 |
| ■ SWA，窗口 1024 | `SWA_FIT_W1K` | 同基础版，窗口 1024 | `MUONH_FIT` | 新公式（fit v2） | q0, q1, q2, q3 |
| ■ SWA + attention sink | `SWA_FIT_SINK` | 同基础版 + 可学习 attention sink（gpt-oss 式：每层每个 query head 一个可学习的 sink logit） | `MUONH_FIT` | 新公式（fit v2） | q0, q1, q2, q3 |
| ■ SWA + sink + 窗口 1024 | `SWA_FIT_SINK_W1K` | attention sink + 窗口 1024 | `MUONH_FIT` | 新公式（fit v2） | q0, q1, q2, q3 |
| ■ SWA + sink + 窗口 512 | `SWA_FIT_SINK_W512` | attention sink + 窗口 512（窗口这条轴往下扫） | `MUONH_FIT` | 新公式（fit v2） | q0, q1, q2, q3 |
| ■ SWA + sink + 窗口 128 | `SWA_FIT_SINK_W128` | attention sink + 窗口 128（窗口这条轴往下扫） | `MUONH_FIT` | 新公式（fit v2） | q0, q1, q2, q3 |
| ■ SWA + sink + 窗口 512，3:1（靠底，q3 没训） | `SWA_FIT_SINK_W512_R3` | 同窗口 512 + sink，但每 4 层 1 层全注意力（NoPE），局部:全局从 5:1 改成 3:1（比例这条轴；Boxiang 10-30） | `MUONH_FIT` | 新公式（fit v2） | q0, q1 |
| ■ SWA + sink + 窗口 512，3:1（靠顶） | `SWA_FIT_SINK_W512_R3T` | 同 3:1，但全注意力层从顶往下数（最后一层一定是全注意力，和 5:1 一样）；R3 从底往上数，有的深度最顶两层都是 SWA。q0（12 层）和 q2（24 层）层数是 4 的倍数，两种数法排出来一样，两条线用同一个 run（q2 的 argv 与 R3 q2 的 golden 逐项相同）；只有 q1、q3 两者不同，R3 的 q3 没训 | `MUONH_FIT` | 新公式（fit v2） | q1, q2, q3 |

**〔指标：val_v2〕**

**softmax router：各改动 − MUONH_FIT（全注意力，softmax router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA − 无 SWA（MUONH_FIT） | -0.0030 | 0.0012 | -0.0011 | 0.0000 |
| SWA，全注意力层用 RoPE − MUONH_FIT | 0.0216 | -0.0003 | 0.0016 | 0.0011 |
| SWA，窗口 4096 − MUONH_FIT | 0.0026 | 0.0056 | -0.0018 | 0.0003 |
| SWA，窗口 1024 − MUONH_FIT | 0.0023 | -0.0035 | 0.0005 | -0.0012 |
| SWA + 可学习 attention sink（gpt-oss 式）− MUONH_FIT | -0.0031 | -0.0053 | -0.0027 | -0.0020 |
| SWA + sink + 窗口 1024 − MUONH_FIT | -0.0029 | -0.0043 | -0.0043 | -0.0055 |
| SWA + sink + 窗口 512 − MUONH_FIT | -0.0042 | -0.0060 | -0.0081 | -0.0087 |
| SWA + sink + 窗口 128 − MUONH_FIT | 0.0044 | -0.0043 | -0.0056 | -0.0070 |
| SWA + sink + 窗口 512，局部:全局 3:1（全注意力层靠底，q3 没训）− MUONH_FIT | -0.0103 | -0.0131 | · | · |
| SWA + sink + 窗口 512，3:1（全注意力层靠顶）− MUONH_FIT | · | -0.0092 | -0.0086 | -0.0072 |

**DSV3 router：各改动 − MUONH_FIT_DSV3（全注意力，DSV3 router）**

**〔指标：uncheatable〕**

**softmax router：各改动 − MUONH_FIT（全注意力，softmax router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA − 无 SWA（MUONH_FIT） | -0.0025 | 0.0000 | -0.0037 | -0.0040 |
| SWA，全注意力层用 RoPE − MUONH_FIT | 0.0144 | -0.0020 | 0.0015 | 0.0013 |
| SWA，窗口 4096 − MUONH_FIT | 0.0006 | 0.0031 | -0.0034 | -0.0008 |
| SWA，窗口 1024 − MUONH_FIT | 0.0059 | -0.0041 | 0.0011 | -0.0031 |
| SWA + 可学习 attention sink（gpt-oss 式）− MUONH_FIT | -0.0050 | -0.0092 | -0.0048 | -0.0038 |
| SWA + sink + 窗口 1024 − MUONH_FIT | 0.0007 | -0.0065 | -0.0078 | -0.0096 |
| SWA + sink + 窗口 512 − MUONH_FIT | 0.0002 | -0.0107 | -0.0125 | -0.0137 |
| SWA + sink + 窗口 128 − MUONH_FIT | 0.0093 | -0.0079 | -0.0086 | -0.0113 |
| SWA + sink + 窗口 512，局部:全局 3:1（全注意力层靠底，q3 没训）− MUONH_FIT | -0.0071 | -0.0165 | · | · |
| SWA + sink + 窗口 512，3:1（全注意力层靠顶）− MUONH_FIT | · | -0.0149 | -0.0102 | -0.0139 |

**DSV3 router：各改动 − MUONH_FIT_DSV3（全注意力，DSV3 router）**

**〔指标：codefresh〕**

**softmax router：各改动 − MUONH_FIT（全注意力，softmax router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA − 无 SWA（MUONH_FIT） | -0.0054 | -0.0047 | -0.0045 | -0.0030 |
| SWA，全注意力层用 RoPE − MUONH_FIT | -0.0031 | -0.0049 | -0.0030 | -0.0000 |
| SWA，窗口 4096 − MUONH_FIT | 0.0027 | 0.0041 | -0.0032 | 0.0003 |
| SWA，窗口 1024 − MUONH_FIT | 0.0012 | -0.0078 | -0.0037 | -0.0047 |
| SWA + 可学习 attention sink（gpt-oss 式）− MUONH_FIT | -0.0081 | -0.0093 | -0.0069 | -0.0070 |
| SWA + sink + 窗口 1024 − MUONH_FIT | -0.0043 | -0.0133 | -0.0091 | -0.0096 |
| SWA + sink + 窗口 512 − MUONH_FIT | -0.0050 | -0.0124 | -0.0146 | -0.0139 |
| SWA + sink + 窗口 128 − MUONH_FIT | 0.0135 | -0.0060 | -0.0091 | -0.0107 |
| SWA + sink + 窗口 512，局部:全局 3:1（全注意力层靠底，q3 没训）− MUONH_FIT | -0.0042 | -0.0144 | · | · |
| SWA + sink + 窗口 512，3:1（全注意力层靠顶）− MUONH_FIT | · | -0.0143 | -0.0107 | -0.0104 |

**DSV3 router：各改动 − MUONH_FIT_DSV3（全注意力，DSV3 router）**

**〔指标：v3small〕**

**softmax router：各改动 − MUONH_FIT（全注意力，softmax router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA − 无 SWA（MUONH_FIT） | 0.0012 | -0.0019 | -0.0185 | -0.0555 |
| SWA，全注意力层用 RoPE − MUONH_FIT | 0.0114 | -0.0041 | -0.0175 | -0.0529 |
| SWA，窗口 4096 − MUONH_FIT | 0.0029 | -0.0000 | -0.0180 | -0.0511 |
| SWA，窗口 1024 − MUONH_FIT | -0.0001 | -0.0062 | -0.0144 | -0.0534 |
| SWA + 可学习 attention sink（gpt-oss 式）− MUONH_FIT | -0.0010 | -0.0062 | -0.0209 | -0.0567 |
| SWA + sink + 窗口 1024 − MUONH_FIT | -0.0003 | -0.0090 | -0.0209 | -0.0444 |
| SWA + sink + 窗口 512 − MUONH_FIT | -0.0029 | -0.0056 | 0.0292 | -0.0085 |
| SWA + sink + 窗口 128 − MUONH_FIT | 0.0086 | -0.0025 | 0.0012 | -0.0059 |
| SWA + sink + 窗口 512，局部:全局 3:1（全注意力层靠底，q3 没训）− MUONH_FIT | -0.0046 | 0.0018 | · | · |
| SWA + sink + 窗口 512，3:1（全注意力层靠顶）− MUONH_FIT | · | -0.0113 | -0.0060 | -0.0469 |

**DSV3 router：各改动 − MUONH_FIT_DSV3（全注意力，DSV3 router）**

**〔指标：pile_test〕**

**softmax router：各改动 − MUONH_FIT（全注意力，softmax router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA − 无 SWA（MUONH_FIT） | -0.0051 | -0.0010 | -0.0037 | -0.0017 |
| SWA，全注意力层用 RoPE − MUONH_FIT | 0.0131 | 0.0000 | -0.0010 | 0.0010 |
| SWA，窗口 4096 − MUONH_FIT | 0.0007 | 0.0046 | -0.0035 | 0.0014 |
| SWA，窗口 1024 − MUONH_FIT | 0.0004 | -0.0038 | -0.0023 | -0.0029 |
| SWA + 可学习 attention sink（gpt-oss 式）− MUONH_FIT | -0.0066 | -0.0075 | -0.0057 | -0.0033 |
| SWA + sink + 窗口 1024 − MUONH_FIT | -0.0037 | -0.0069 | -0.0090 | -0.0066 |
| SWA + sink + 窗口 512 − MUONH_FIT | -0.0097 | -0.0118 | -0.0114 | -0.0091 |
| SWA + sink + 窗口 128 − MUONH_FIT | -0.0002 | -0.0090 | -0.0100 | -0.0067 |
| SWA + sink + 窗口 512，局部:全局 3:1（全注意力层靠底，q3 没训）− MUONH_FIT | -0.0130 | -0.0134 | · | · |
| SWA + sink + 窗口 512，3:1（全注意力层靠顶）− MUONH_FIT | · | -0.0124 | -0.0080 | -0.0083 |

**DSV3 router：各改动 − MUONH_FIT_DSV3（全注意力，DSV3 router）**

**〔指标：pile_10k〕**

**softmax router：各改动 − MUONH_FIT（全注意力，softmax router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA − 无 SWA（MUONH_FIT） | -0.0058 | 0.0010 | -0.0002 | -0.0022 |
| SWA，全注意力层用 RoPE − MUONH_FIT | 0.0226 | 0.0005 | 0.0052 | 0.0023 |
| SWA，窗口 4096 − MUONH_FIT | 0.0018 | 0.0079 | -0.0009 | 0.0002 |
| SWA，窗口 1024 − MUONH_FIT | 0.0036 | -0.0053 | 0.0014 | -0.0031 |
| SWA + 可学习 attention sink（gpt-oss 式）− MUONH_FIT | -0.0077 | -0.0073 | -0.0020 | -0.0030 |
| SWA + sink + 窗口 1024 − MUONH_FIT | -0.0027 | -0.0065 | -0.0055 | -0.0114 |
| SWA + sink + 窗口 512 − MUONH_FIT | -0.0067 | -0.0151 | -0.0089 | -0.0121 |
| SWA + sink + 窗口 128 − MUONH_FIT | -0.0004 | -0.0113 | -0.0099 | -0.0096 |
| SWA + sink + 窗口 512，局部:全局 3:1（全注意力层靠底，q3 没训）− MUONH_FIT | -0.0163 | -0.0196 | · | · |
| SWA + sink + 窗口 512，3:1（全注意力层靠顶）− MUONH_FIT | · | -0.0136 | -0.0108 | -0.0034 |

**DSV3 router：各改动 − MUONH_FIT_DSV3（全注意力，DSV3 router）**

**〔指标：六集平均〕**

**softmax router：各改动 − MUONH_FIT（全注意力，softmax router）**

| 线 | q0 · 48M | q1 · 154M | q2 · 356M | q3 · 683M |
|---|---|---|---|---|
| SWA − 无 SWA（MUONH_FIT） | -0.0034 | -0.0009 | -0.0053 | -0.0111 |
| SWA，全注意力层用 RoPE − MUONH_FIT | 0.0134 | -0.0018 | -0.0022 | -0.0079 |
| SWA，窗口 4096 − MUONH_FIT | 0.0019 | 0.0042 | -0.0051 | -0.0083 |
| SWA，窗口 1024 − MUONH_FIT | 0.0022 | -0.0051 | -0.0029 | -0.0114 |
| SWA + 可学习 attention sink（gpt-oss 式）− MUONH_FIT | -0.0052 | -0.0075 | -0.0072 | -0.0126 |
| SWA + sink + 窗口 1024 − MUONH_FIT | -0.0022 | -0.0078 | -0.0094 | -0.0145 |
| SWA + sink + 窗口 512 − MUONH_FIT | -0.0047 | -0.0103 | -0.0044 | -0.0110 |
| SWA + sink + 窗口 128 − MUONH_FIT | 0.0059 | -0.0068 | -0.0070 | -0.0085 |
| SWA + sink + 窗口 512，局部:全局 3:1（全注意力层靠底，q3 没训）− MUONH_FIT | -0.0092 | -0.0125 | · | · |
| SWA + sink + 窗口 512，3:1（全注意力层靠顶）− MUONH_FIT | · | -0.0126 | -0.0090 | -0.0150 |

**DSV3 router：各改动 − MUONH_FIT_DSV3（全注意力，DSV3 router）**

每种改动相对各自对照的 Δloss（全部是新 LR 公式；有 3 个 seed 的点 = 3 个 seed 按 seed 配对差的平均，其余是 1 个 seed；鼠标移到某个 rung 看所有改动的数值）

### KD − CE（负数 = KD 更好）

| rung | 优化器 | loss type | α | val_v2 | uncheatable | codefresh | v3small | pile_test | pile_10k | 六集平均 |
|---|---|---|---|---|---|---|---|---|---|---|
| q0 | MuonH | KD | 0.7 | −0.027 | −0.046 | −0.043 | −0.036 | −0.035 | −0.035 | −0.037 |
| q0 | MuonH | KD | 0.8 | −0.028 | −0.048 | −0.048 | −0.036 | −0.035 | −0.035 | −0.038 |
| q0 | MuonH | KD | 0.9 | −0.029 | −0.049 | −0.048 | −0.034 | −0.035 | −0.034 | −0.038 |
| q1 | MuonH | KD | 0.7 | −0.030 | −0.053 | −0.048 | −0.039 | −0.039 | −0.041 | −0.042 |
| q1 | MuonH | KD | 0.8 | −0.030 | −0.053 | −0.050 | −0.036 | −0.038 | −0.041 | −0.041 |
| q1 | MuonH | KD | 0.9 | −0.031 | −0.056 | −0.052 | −0.036 | −0.037 | −0.040 | −0.042 |
| q2 | MuonH | KD | 0.7 | −0.031 | −0.054 | −0.050 | −0.041 | −0.042 | −0.042 | −0.043 |
| q2 | MuonH | KD | 0.8 | −0.031 | −0.058 | −0.053 | −0.041 | −0.041 | −0.042 | −0.044 |
| q2 | MuonH | KD | 0.9 | −0.030 | −0.058 | −0.052 | −0.039 | −0.040 | −0.042 | −0.044 |
| q3 | MuonH | KD | 0.7 | −0.035 | −0.060 | −0.055 | −0.045 | −0.044 | −0.046 | −0.047 |
| q3 | MuonH | KD | 0.8 | −0.035 | −0.061 | −0.057 | −0.043 | −0.045 | −0.046 | −0.048 |
| q3 | MuonH | KD | 0.9 | −0.034 | −0.062 | −0.056 | −0.039 | −0.042 | −0.044 | −0.046 |
| q4 | MuonH | KD | 0.7 | −0.037 | −0.063 | −0.057 | −0.044 | −0.048 | −0.048 | −0.050 |
| q4 | MuonH | KD | 0.8 | −0.034 | −0.064 | −0.059 | −0.015 | −0.048 | −0.059 | −0.046 |
| q4 | MuonH | KD | 0.9 | −0.038 | −0.065 | −0.061 | −0.011 | −0.045 | −0.054 | −0.045 |

## 附录 A：核心 6 组的绝对 loss（doc-isolated，8K 窗口）

Boxiang ladder 的主线：AdamW / Muon / MuonH 三个 CE 优化器，和 MuonH 上 α 0.7 / 0.8 / 0.9 的 KD（teacher MiMo-V2.6-Pro-RL，tail = B64，ArchScale 老 LR，650B 数据，无文档 mask）。KD 的 CE 对照 = OPT_MUONH（同 recipe，α = 0）。`·` = 还没有结果。

| rung | Change ID | 优化器 | loss | α | val_v2 | uncheatable | codefresh | v3small | private_internal | pile_test | pile_10k | overall_tw | overall_macro | overall_tsqw |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q0 | `BASE` | AdamW | CE | — | 1.8226 | 2.2790 | 1.8133 | 3.0675 | 2.4251 | 2.4808 | 2.5078 | 2.2519 | 2.3147 | 2.2608 |
| q0 | `OPT_MUON` | Muon | CE | — | 1.7752 | 2.2147 | 1.7287 | 3.0110 | 2.3683 | 2.4264 | 2.4512 | 2.1907 | 2.2541 | 2.2001 |
| q0 | `OPT_MUONH` | MuonH | CE | — | 1.7580 | 2.1961 | 1.7185 | 2.9948 | 2.3556 | 2.4071 | 2.4307 | 2.1728 | 2.2383 | 2.1829 |
| q0 | `KD_MUONH_A0P7` | MuonH | KD | 0.7 | 1.7318 | 2.1513 | 1.6751 | 2.9596 | 2.3262 | 2.3720 | 2.3956 | 2.1333 | 2.2027 | 2.1453 |
| q0 | `KD_MUONH_A0P8` | MuonH | KD | 0.8 | 1.7307 | 2.1493 | 1.6734 | 2.9605 | 2.3252 | 2.3722 | 2.3960 | 2.1320 | 2.2019 | 2.1444 |
| q0 | `KD_MUONH_A0P9` | MuonH | KD | 0.9 | 1.7297 | 2.1480 | 1.6726 | 2.9620 | 2.3260 | 2.3725 | 2.3963 | 2.1313 | 2.2018 | 2.1439 |
| q1 | `BASE` | AdamW | CE | — | 1.5718 | 1.9481 | 1.4556 | 2.7475 | 2.1421 | 2.1739 | 2.1869 | 1.9353 | 2.0065 | 1.9479 |
| q1 | `OPT_MUON` | Muon | CE | — | 1.5510 | 1.9173 | 1.4205 | 2.7181 | 2.1182 | 2.1481 | 2.1591 | 1.9067 | 1.9789 | 1.9198 |
| q1 | `OPT_MUONH` | MuonH | CE | — | 1.5419 | 1.9065 | 1.4132 | 2.7055 | 2.1127 | 2.1354 | 2.1471 | 1.8960 | 1.9692 | 1.9094 |
| q1 | `KD_MUONH_A0P7` | MuonH | KD | 0.7 | 1.5117 | 1.8544 | 1.3693 | 2.6677 | 2.0809 | 2.0967 | 2.1069 | 1.8512 | 1.9301 | 1.8674 |
| q1 | `KD_MUONH_A0P8` | MuonH | KD | 0.8 | 1.5118 | 1.8550 | 1.3657 | 2.6699 | 2.0822 | 2.0973 | 2.1071 | 1.8514 | 1.9303 | 1.8675 |
| q1 | `KD_MUONH_A0P9` | MuonH | KD | 0.9 | 1.5107 | 1.8514 | 1.3655 | 2.6691 | 2.0792 | 2.0984 | 2.1069 | 1.8497 | 1.9291 | 1.8662 |
| q2 | `BASE` | AdamW | CE | — | 1.4361 | 1.7530 | 1.2677 | 2.5579 | 1.9877 | 2.0021 | 1.9985 | 1.7546 | 1.8341 | 1.7709 |
| q2 | `OPT_MUON` | Muon | CE | — | 1.4197 | 1.7305 | 1.2429 | 2.5357 | 1.9700 | 1.9790 | 1.9761 | 1.7326 | 1.8130 | 1.7491 |
| q2 | `OPT_MUONH` | MuonH | CE | — | 1.4141 | 1.7269 | 1.2432 | 2.5312 | 1.9718 | 1.9730 | 1.9707 | 1.7285 | 1.8100 | 1.7453 |
| q2 | `KD_MUONH_A0P7` | MuonH | KD | 0.7 | 1.3833 | 1.6734 | 1.1944 | 2.4910 | 1.9410 | 1.9313 | 1.9285 | 1.6816 | 1.7690 | 1.7012 |
| q2 | `KD_MUONH_A0P8` | MuonH | KD | 0.8 | 1.3827 | 1.6703 | 1.1930 | 2.4913 | 1.9395 | 1.9314 | 1.9288 | 1.6799 | 1.7680 | 1.7000 |
| q2 | `KD_MUONH_A0P9` | MuonH | KD | 0.9 | 1.3832 | 1.6695 | 1.1914 | 2.4922 | 1.9403 | 1.9326 | 1.9291 | 1.6797 | 1.7682 | 1.7000 |
| q3 | `BASE` | AdamW | CE | — | 1.3442 | 1.6196 | 1.1449 | 2.4307 | 1.8853 | 1.8811 | 1.8703 | 1.6308 | 1.7176 | 1.6502 |
| q3 | `OPT_MUON` | Muon | CE | — | 1.3319 | 1.6014 | 1.1287 | 2.4141 | 1.8735 | 1.8644 | 1.8520 | 1.6139 | 1.7023 | 1.6339 |
| q3 | `OPT_MUONH` | MuonH | CE | — | 1.3266 | 1.6025 | 1.1307 | 2.4079 | 1.8766 | 1.8568 | 1.8467 | 1.6120 | 1.7002 | 1.6314 |
| q3 | `KD_MUONH_A0P7` | MuonH | KD | 0.7 | 1.2924 | 1.5429 | 1.0765 | 2.3654 | 1.8386 | 1.8125 | 1.8010 | 1.5604 | 1.6547 | 1.5829 |
| q3 | `KD_MUONH_A0P8` | MuonH | KD | 0.8 | 1.2919 | 1.5421 | 1.0750 | 2.3661 | 1.8402 | 1.8114 | 1.8004 | 1.5595 | 1.6545 | 1.5822 |
| q3 | `KD_MUONH_A0P9` | MuonH | KD | 0.9 | 1.2926 | 1.5411 | 1.0749 | 2.3688 | 1.8413 | 1.8137 | 1.8021 | 1.5598 | 1.6554 | 1.5829 |
| q4 | `BASE` | AdamW | CE | — | 1.2818 | 1.5284 | 1.0675 | 2.3398 | 1.8144 | 1.7966 | 1.7794 | 1.5462 | 1.6381 | 1.5678 |
| q4 | `OPT_MUON` | Muon | CE | — | 1.2709 | 1.5137 | 1.0527 | 2.3254 | 1.8040 | 1.7820 | 1.7641 | 1.5320 | 1.6248 | 1.5538 |
| q4 | `OPT_MUONH` | MuonH | CE | — | 1.2660 | 1.5182 | 1.0566 | 2.3209 | 1.8120 | 1.7776 | 1.7613 | 1.5329 | 1.6252 | 1.5540 |
| q4 | `KD_MUONH_A0P7` | MuonH | KD | 0.7 | 1.2300 | 1.4559 | 1.0023 | 2.2788 | 1.7741 | 1.7305 | 1.7137 | 1.4790 | 1.5786 | 1.5035 |
| q4 | `KD_MUONH_A0P8` | MuonH | KD | 0.8 | 1.2288 | 1.4550 | 0.9996 | 2.2779 | 1.7729 | 1.7288 | 1.7127 | 1.4777 | 1.5772 | 1.5022 |
| q4 | `KD_MUONH_A0P9` | MuonH | KD | 0.9 | 1.2298 | 1.4541 | 0.9982 | 2.2815 | 1.7750 | 1.7315 | 1.7147 | 1.4781 | 1.5784 | 1.5029 |
| q5 | `BASE` | AdamW | CE | — | 1.2335 | 1.4621 | 1.0061 | 2.2686 | 1.7616 | 1.7306 | 1.7107 | 1.4826 | 1.5771 | 1.5052 |

模型大小（MoE，每层 128 个 expert 选 8 个）：

| rung | 层数 × hidden | 激活参数（不含 embedding） | 总参数（含 embedding） | 主线训练 token（TPP 200） |
|---|---|---:|---:|---:|
| q0 | 12 × 512 | 48M | 0.63B | 9.6B |
| q1 | 18 × 768 | 154M | 1.82B | 30.8B |
| q2 | 24 × 1024 | 356M | 4.06B | 71.1B |
| q3 | 30 × 1280 | 683M | 7.71B | 136.6B |
| q4 | 36 × 1536 | 1.17B | 13.1B | 233.6B |
| q5 | 42 × 1792 | 1.84B | 20.6B | 368.0B |
| q6 | 48 × 2048 | 2.73B | — | — |

## 附录 B：全部 checkpoint 的 overall_tsqw（doc-isolated）

`trichord/dense_pretrain_result` 里的全部 332 个 Change ID。后缀约定：`_S2` / `_S3` = 第 2、3 个 seed（无后缀 = seed 1）；`_W32K` = 同一 checkpoint 按 32K 窗口评（无后缀 = 8K 窗口）；`_PP32K` = 80B pack plan、32K 训练的新数据管线；`_IT4577` 等 = 改了训练步数（LR 格子）。Change ID 的含义以正文各节的说明为准。

| Change ID | q0 | q1 | q2 | q3 | q4 | q5 | q6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `BASE` | 2.2608 | 1.9479 | 1.7709 | 1.6502 | 1.5678 | 1.5052 |  |
| `BASE_DSV3_ONETREE` | 2.2614 | 1.9533 | 1.7731 |  |  |  |  |
| `BASE_DSV3_ONETREE_LR0P5X` | 2.2755 |  |  |  |  |  |  |
| `BASE_DSV3_ONETREE_LR2X` | 2.2630 |  |  |  |  |  |  |
| `BASE_DSV3_ONETREE_S2` | 2.2615 |  |  |  |  |  |  |
| `BASE_DSV3_ONETREE_S3` | 2.2569 |  |  |  |  |  |  |
| `BASE_LR0P5X` |  | 1.9557 | 1.7746 |  |  |  |  |
| `BASE_LR2X` |  | 1.9520 | 1.7780 |  |  |  |  |
| `BASE_S2` | 2.2578 | 1.9557 |  |  |  |  |  |
| `BASE_S3` | 2.2650 | 1.9486 |  |  |  |  |  |
| `KD_MUONH_A0P7` | 2.1453 | 1.8674 | 1.7012 | 1.5829 | 1.5035 |  |  |
| `KD_MUONH_A0P8` | 2.1444 | 1.8675 | 1.7000 | 1.5822 | 1.5022 |  |  |
| `KD_MUONH_A0P8_B128` | 2.1433 | 1.8656 | 1.6997 |  |  |  |  |
| `KD_MUONH_A0P8_B64Y` | 2.1443 | 1.8658 | 1.7003 |  |  |  |  |
| `KD_MUONH_A0P8_DD64` | 2.1439 | 1.8676 | 1.7016 |  |  |  |  |
| `KD_MUONH_A0P8_DROP64` | 2.1421 |  |  |  |  |  |  |
| `KD_MUONH_A0P8_S128` | 2.1418 | 1.8634 | 1.6966 | 1.5785 |  |  |  |
| `KD_MUONH_A0P8_S128Y` |  |  |  | 1.5788 |  |  |  |
| `KD_MUONH_A0P8_S64` | 2.1420 | 1.8622 | 1.6960 | 1.5788 |  |  |  |
| `KD_MUONH_A0P8_S64Y` | 2.1383 | 1.8628 | 1.6969 | 1.5786 |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT` | 2.1335 | 1.8500 | 1.6783 | 1.5641 |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_DOCMASK` | 2.1241 | 1.8365 | 1.6689 | 1.5534 |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_DOCMASK_S2` | 2.1215 |  |  |  |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_DOCMASK_S3` | 2.1220 |  |  |  |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_DOCMASK_W32K` | 2.3985 | 2.0606 | 2.0691 | 1.9835 |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_DOCMASK_W32K_S2` | 2.3046 |  |  |  |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_DOCMASK_W32K_S3` | 2.3450 |  |  |  |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_MIMO25_PP32K` | 2.1274 | 1.8428 | 1.6760 | 1.5590 |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_MIMO25_PP32K_S2` | 2.1294 |  |  |  |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_MIMO25_PP32K_S3` | 2.1320 |  |  |  |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_MIMO25_PP32K_W32K` | 2.1040 | 1.8174 | 1.6507 | 1.5341 |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_MIMO25_PP32K_W32K_S2` | 2.1058 |  |  |  |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_MIMO25_PP32K_W32K_S3` | 2.1093 |  |  |  |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_PP32K` | 2.1302 | 1.8456 | 1.6763 | 1.5612 |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_PP32K_S2` | 2.1326 |  |  |  |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_PP32K_S3` | 2.1326 |  |  |  |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_PP32K_W32K` | 2.1068 | 1.8203 | 1.6508 | 1.5361 |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_PP32K_W32K_S2` | 2.1085 |  |  |  |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_PP32K_W32K_S3` | 2.1097 |  |  |  |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_S2` | 2.1372 |  |  |  |  |  |  |
| `KD_MUONH_A0P8_S64Y_FIT_S3` | 2.1396 |  |  |  |  |  |  |
| `KD_MUONH_A0P9` | 2.1439 | 1.8662 | 1.7000 | 1.5829 | 1.5029 |  |  |
| `KD_MUONH_FIT_A0P8_S128Y_TCS_MIMO25` | 2.1306 | 1.8452 | 1.6794 |  |  |  |  |
| `KD_MUONH_FIT_A0P8_S128Y_TCS_MIMO25_S2` | 2.1262 |  |  |  |  |  |  |
| `KD_MUONH_FIT_A0P8_S128Y_TCS_MIMO25_S3` | 2.1325 |  |  |  |  |  |  |
| `KD_MUONH_FIT_A0P8_S128Y_TCS_MIMO26` | 2.1323 | 1.8479 | 1.6811 |  |  |  |  |
| `KD_MUONH_FIT_A0P8_S128Y_TCS_MIMO26_S2` | 2.1288 |  |  |  |  |  |  |
| `KD_MUONH_FIT_A0P8_S128Y_TCS_MIMO26_S3` | 2.1359 |  |  |  |  |  |  |
| `KD_MUONH_FIT_A0P8_S64Y_TCS_MIMO25` | 2.1309 | 1.8455 | 1.6782 | 1.6132 |  |  |  |
| `KD_MUONH_FIT_A0P8_S64Y_TCS_MIMO25_S2` | 2.1263 |  |  |  |  |  |  |
| `KD_MUONH_FIT_A0P8_S64Y_TCS_MIMO25_S3` | 2.1312 |  |  |  |  |  |  |
| `KD_MUONH_FIT_A0P8_S64Y_TCS_MIMO26` | 2.1338 | 1.8483 | 1.6802 | 1.6127 |  |  |  |
| `KD_MUONH_FIT_A0P8_S64Y_TCS_MIMO26_S2` | 2.1289 |  |  |  |  |  |  |
| `KD_MUONH_FIT_A0P8_S64Y_TCS_MIMO26_S3` | 2.1342 |  |  |  |  |  |  |
| `KD_MUONH_FIT_A0P8_S64_TCS_MIMO25` | 2.1324 | 1.8443 | 1.6775 | 1.6120 |  |  |  |
| `KD_MUONH_FIT_A0P8_S64_TCS_MIMO25_S2` | 2.1256 |  |  |  |  |  |  |
| `KD_MUONH_FIT_A0P8_S64_TCS_MIMO25_S3` | 2.1321 |  |  |  |  |  |  |
| `KD_MUONH_FIT_A0P8_S64_TCS_MIMO26` | 2.1328 | 1.8488 | 1.6805 | 1.6111 |  |  |  |
| `KD_MUONH_FIT_A0P8_S64_TCS_MIMO26_S2` | 2.1297 |  |  |  |  |  |  |
| `KD_MUONH_FIT_A0P8_S64_TCS_MIMO26_S3` | 2.1325 |  |  |  |  |  |  |
| `MUONH_FIT` | 2.1779 | 1.8984 | 1.7280 | 1.6151 |  |  |  |
| `MUONH_FIT_DSV3` | 2.1801 | 1.8960 | 1.7298 | 1.6168 |  |  |  |
| `MUONH_FIT_DSV3_BIASONLY` | 2.1775 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_LR0P7X` | 2.1936 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_LR1P4X` | 2.1880 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_NE_1EM4` | 2.1787 | 1.8952 |  |  |  |  |  |
| `MUONH_FIT_DSV3_NE_1EM5` | 2.1784 | 1.8950 | 1.7260 |  |  |  |  |
| `MUONH_FIT_DSV3_NE_1EM5_S2` | 2.1832 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_NE_1EM5_S3` | 2.1789 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_RT0P19` | 2.1864 | 1.9034 |  |  |  |  |  |
| `MUONH_FIT_DSV3_RT0P19_S2` | 2.1872 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_RT0P19_S3` | 2.1888 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_RT0P5` | 2.1809 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_RT2` | 2.1840 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_RTADAMW_RT0P19` | 2.1832 | 1.9004 |  |  |  |  |  |
| `MUONH_FIT_DSV3_RTADAMW_RT0P19_S2` | 2.1861 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_RTADAMW_RT0P19_S3` | 2.1862 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_RTMUONH` | 2.1807 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_S2` | 2.1839 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_S3` | 2.1832 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWA` | 2.1735 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWA_S2` | 2.1718 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWA_S3` | 2.1757 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS` | 2.1658 | 1.8897 | 1.7216 | 1.6073 | 1.5263 |  |  |
| `MUONH_FIT_DSV3_SWS_AUX1EM3` | 2.1756 | 1.9024 | 1.7331 | 1.6199 |  |  |  |
| `MUONH_FIT_DSV3_SWS_AUXBATCH` | 2.1665 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_BIASONLY` | 2.1682 | 1.8887 | 1.7219 | 1.6025 |  |  |  |
| `MUONH_FIT_DSV3_SWS_BIASONLY_S2` | 2.1657 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_BIASONLY_S3` | 2.1639 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_DENSE1` |  | 1.8842 | 1.7156 |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_DENSE2` |  | 1.8858 | 1.7185 | 1.6039 |  |  |  |
| `MUONH_FIT_DSV3_SWS_GAMMA1EM2` | 2.1682 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_GAMMA1EM4` | 2.1676 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_NE_1EM3` |  | 1.8917 |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_NE_1EM4` |  |  |  | 1.6054 |  |  |  |
| `MUONH_FIT_DSV3_SWS_NE_1EM5_BATCH` | 2.1693 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_NE_1EM5_DENSE1` |  | 1.8791 |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_NE_1EM5_DENSE2` |  | 1.8871 | 1.7148 |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_NE_1EM5_GAMMA1EM4` | 2.1670 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_NE_1EM5_NOBIAS` |  |  | 1.7147 | 1.6010 |  |  |  |
| `MUONH_FIT_DSV3_SWS_NE_1EM5_NOBIAS_DENSE2` |  | 1.8805 | 1.7121 |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_NE_1EM5_NOBIAS_FREQ` | 2.1650 | 1.8897 |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_NE_1EM5_S2` | 2.1677 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_NE_1EM5_S3` | 2.1635 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_NE_4EM4` |  | 1.8896 |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_NOBAL` | 2.1626 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_NOBIAS` | 2.1618 | 1.8884 | 1.7168 | 1.6060 |  |  |  |
| `MUONH_FIT_DSV3_SWS_RS2P5` | 2.1700 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_RT0P19` | 2.1741 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_RT0P5` | 2.1681 | 1.8876 | 1.7308 | 1.6084 |  |  |  |
| `MUONH_FIT_DSV3_SWS_RT2` | 2.1711 | 1.8931 | 1.7240 | 1.6108 |  |  |  |
| `MUONH_FIT_DSV3_SWS_RTADAMW_RT0P19` | 2.1699 | 1.8957 |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_RTMUONH` | 2.1908 | 1.8856 | 1.7233 | 1.6099 |  |  |  |
| `MUONH_FIT_DSV3_SWS_RTMUONH_S2` | 2.1703 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_RTMUONH_S3` | 2.1659 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_S2` | 2.1720 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_S3` | 2.1622 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_SHARED` | 2.1436 |  |  |  |  |  |  |
| `MUONH_FIT_DSV3_SWS_SHARED_RS2P5` | 2.1473 | 1.8629 | 1.6957 | 1.5824 |  |  |  |
| `MUONH_FIT_DSV3_SWS_ZLOSS1EM3` | 2.1951 |  |  |  |  |  |  |
| `MUONH_FIT_PP32K` | 2.1771 | 1.8951 | 1.7285 | 1.6178 |  |  |  |
| `MUONH_FIT_PP32K_S2` | 2.1751 |  |  |  |  |  |  |
| `MUONH_FIT_PP32K_S3` | 2.1757 |  |  |  |  |  |  |
| `MUONH_FIT_PP32K_W32K` | 2.1532 | 1.8694 | 1.7027 | 1.5922 |  |  |  |
| `MUONH_FIT_PP32K_W32K_S2` | 2.1508 |  |  |  |  |  |  |
| `MUONH_FIT_PP32K_W32K_S3` | 2.1521 |  |  |  |  |  |  |
| `MUONH_FIT_S16K` | 2.1783 | 1.8841 |  |  |  |  |  |
| `MUONH_FIT_S2` | 2.1825 | 1.8980 |  |  |  |  |  |
| `MUONH_FIT_S3` | 2.1799 | 1.8950 |  |  |  |  |  |
| `MUONH_FIT_SWA_ROPE` | 2.1955 | 1.8968 |  |  |  |  |  |
| `MUONH_FIT_SWA_S2` | 2.1778 |  |  |  |  |  |  |
| `MUONH_FIT_SWA_S3` | 2.1737 |  |  |  |  |  |  |
| `MUONH_FIT_SWA_SINK` |  |  |  | 1.6097 |  |  |  |
| `MUONH_FIT_SWA_SINK_S2` | 2.1662 |  |  |  |  |  |  |
| `MUONH_FIT_SWA_SINK_S3` | 2.1663 |  |  |  |  |  |  |
| `MUONH_FIT_SWA_SINK_SMOKE60_IT60` | 6.9350 |  |  |  |  |  |  |
| `MUONH_FIT_SWA_SINK_W128` |  |  |  | 1.6034 |  |  |  |
| `MUONH_FIT_SWA_SINK_W128_SMOKE60_IT60` | 6.8516 |  |  |  |  |  |  |
| `MUONH_FIT_SWA_SINK_W1K` |  | 1.8886 | 1.7185 |  |  |  |  |
| `MUONH_FIT_SWA_SINK_W512` |  |  |  | 1.6009 |  |  |  |
| `MUONH_FIT_SWA_SINK_W512_L32K` |  | 1.8590 |  |  |  |  |  |
| `MUONH_FIT_SWA_SINK_W512_R3` |  | 1.8809 |  |  |  |  |  |
| `MUONH_FIT_SWA_SINK_W512_S16K` | 2.1649 | 1.8714 |  |  |  |  |  |
| `MUONH_FIT_SWA_W1K_S2` | 2.1778 |  |  |  |  |  |  |
| `MUONH_FIT_SWA_W1K_S3` | 2.1815 |  |  |  |  |  |  |
| `MUONH_FIT_SWA_W4K` |  |  |  | 1.6150 |  |  |  |
| `MUONH_LR0P5X` | 2.2534 |  |  |  |  |  |  |
| `MUONH_LR0P7X` | 2.2093 | 1.9355 | 1.7745 |  |  |  |  |
| `MUONH_LR0P7X_FLOOR10` | 2.2147 |  |  |  |  |  |  |
| `MUONH_LR0P7X_IT14685` |  |  | 1.8598 | 1.8146 |  |  |  |
| `MUONH_LR0P7X_IT4577` |  | 2.0872 | 2.0262 | 1.9878 |  |  |  |
| `MUONH_LR0P7X_IT4577_S2` |  |  | 2.0207 |  |  |  |  |
| `MUONH_LR0P7X_IT4577_S3` |  |  | 2.0209 |  |  |  |  |
| `MUONH_LR0P7X_L12H1024_IT4577` |  |  | 2.0532 |  |  |  |  |
| `MUONH_LR0P7X_L24H1536_IT4577` |  |  |  |  | 1.9608 |  |  |
| `MUONH_LR0P7X_L24H2048_IT4577` |  |  |  |  | 1.9252 |  |  |
| `MUONH_LR0P7X_L24H4096_IT4577` |  |  |  |  |  |  | 4.1875 |
| `MUONH_LR0P7X_L24H512_IT4577` | 2.1580 |  |  |  |  |  |  |
| `MUONH_LR0P7X_S2` | 2.2068 |  |  |  |  |  |  |
| `MUONH_LR0P7X_S3` | 2.2092 |  |  |  |  |  |  |
| `MUONH_LR0P7X_TPP1800` | 2.0157 |  |  |  |  |  |  |
| `MUONH_LR0P7X_TPP600` | 2.0921 |  |  |  |  |  |  |
| `MUONH_LR1P2X` | 2.1791 | 1.9018 | 1.7360 |  |  |  |  |
| `MUONH_LR1P2X_IT14685` |  |  | 1.8150 | 1.7599 |  |  |  |
| `MUONH_LR1P2X_IT4577` |  | 2.0379 | 1.9637 | 1.9138 |  |  |  |
| `MUONH_LR1P2X_IT4577_S2` |  |  | 1.9605 |  |  |  |  |
| `MUONH_LR1P2X_IT4577_S3` |  |  | 1.9603 |  |  |  |  |
| `MUONH_LR1P2X_L12H1024_IT4577` |  |  | 2.0306 |  |  |  |  |
| `MUONH_LR1P2X_L24H1536_IT4577` |  |  |  |  | 1.9068 |  |  |
| `MUONH_LR1P2X_L24H2048_IT4577` |  |  |  |  | 1.8809 |  |  |
| `MUONH_LR1P2X_L24H4096_IT4577` |  |  |  |  |  |  | 3.1291 |
| `MUONH_LR1P2X_L24H512_IT4577` | 2.0989 |  |  |  |  |  |  |
| `MUONH_LR1P2X_S2` | 2.1825 |  |  |  |  |  |  |
| `MUONH_LR1P2X_S3` | 2.1825 |  |  |  |  |  |  |
| `MUONH_LR1P2X_TPP1800` | 1.9973 |  |  |  |  |  |  |
| `MUONH_LR1P2X_TPP600` | 2.0683 |  |  |  |  |  |  |
| `MUONH_LR1P4X` | 2.1803 | 1.8966 | 1.7308 |  |  |  |  |
| `MUONH_LR1P4X_B4M_IT2289` |  | 2.0634 |  |  |  |  |  |
| `MUONH_LR1P4X_FLOOR10` | 2.2067 |  |  |  |  |  |  |
| `MUONH_LR1P4X_FLOOR10_IT4577` |  |  |  | 1.9142 |  |  |  |
| `MUONH_LR1P4X_IT14685` |  |  | 1.8077 | 1.7507 |  |  |  |
| `MUONH_LR1P4X_IT4577` |  | 2.0336 | 1.9563 | 1.9041 | 1.8763 | 1.8556 | 1.8399 |
| `MUONH_LR1P4X_IT4577_FLOOR10` |  |  |  | 1.9142 |  |  |  |
| `MUONH_LR1P4X_IT4577_S2` |  |  | 1.9531 |  |  |  |  |
| `MUONH_LR1P4X_IT4577_S3` |  |  | 1.9549 |  |  |  |  |
| `MUONH_LR1P4X_L12H1024_IT4577` |  |  | 2.0355 |  |  |  |  |
| `MUONH_LR1P4X_L24H1536_IT4577` |  |  |  |  | 1.9026 |  |  |
| `MUONH_LR1P4X_L24H2048_IT4577` |  |  |  |  | 1.8819 |  |  |
| `MUONH_LR1P4X_L24H512_IT4577` | 2.0902 |  |  |  |  |  |  |
| `MUONH_LR1P4X_L48H1024_IT4577` |  |  | 1.9138 |  |  |  |  |
| `MUONH_LR1P4X_L60H1024_IT4577` |  |  | 1.9120 |  |  |  |  |
| `MUONH_LR1P4X_L96H1024_IT4577` |  |  | 1.9125 |  |  |  |  |
| `MUONH_LR1P4X_S2` | 2.1814 |  |  |  |  |  |  |
| `MUONH_LR1P4X_S3` | 2.1853 |  |  |  |  |  |  |
| `MUONH_LR1P4X_TPP1800` | 1.9971 |  |  |  |  |  |  |
| `MUONH_LR1P4X_TPP600` | 2.0680 |  |  |  |  |  |  |
| `MUONH_LR1X_B4M_IT2289` |  | 2.0931 |  |  |  |  |  |
| `MUONH_LR1X_FLOOR10` | 2.1961 |  |  |  |  |  |  |
| `MUONH_LR1X_FLOOR10_IT4577` |  |  |  | 1.9374 |  |  |  |
| `MUONH_LR1X_IT14685` |  |  | 1.8246 | 1.7747 |  |  |  |
| `MUONH_LR1X_IT4577` |  | 2.0485 | 1.9785 | 1.9350 | 1.9097 | 1.8897 | 1.8730 |
| `MUONH_LR1X_IT4577_FLOOR10` |  |  |  | 1.9374 |  |  |  |
| `MUONH_LR1X_IT4577_S2` |  |  | 1.9773 |  |  |  |  |
| `MUONH_LR1X_IT4577_S3` |  |  | 1.9767 |  |  |  |  |
| `MUONH_LR1X_L12H1024_IT4577` |  |  | 2.0337 |  |  |  |  |
| `MUONH_LR1X_L24H1536_IT4577` |  |  |  |  | 1.9211 |  |  |
| `MUONH_LR1X_L24H2048_IT4577` |  |  |  |  | 1.8883 |  |  |
| `MUONH_LR1X_L24H4096_IT4577` |  |  |  |  |  |  | 2.4391 |
| `MUONH_LR1X_L24H512_IT4577` | 2.1129 |  |  |  |  |  |  |
| `MUONH_LR1X_L60H1024_IT4577` |  |  | 1.9614 |  |  |  |  |
| `MUONH_LR1X_TPP1800` | 2.0021 |  |  |  |  |  |  |
| `MUONH_LR1X_TPP600` | 2.0729 |  |  |  |  |  |  |
| `MUONH_LR2P8X_B4M_IT2289` |  | 2.0694 |  |  |  |  |  |
| `MUONH_LR2P8X_IT14685` |  |  |  | 1.7658 |  |  |  |
| `MUONH_LR2P8X_IT4577` |  |  |  | 1.9344 | 1.8964 | 1.8726 | 1.8502 |
| `MUONH_LR2P8X_L48H1024_IT4577` |  |  | 1.9012 |  |  |  |  |
| `MUONH_LR2P8X_L60H1024_IT4577` |  |  | 1.8843 |  |  |  |  |
| `MUONH_LR2P8X_L96H1024_IT4577` |  |  | 1.8509 |  |  |  |  |
| `MUONH_LR2X` | 2.2045 | 1.9029 | 1.7296 |  |  |  |  |
| `MUONH_LR2X_B4M_IT2289` |  | 2.0541 |  |  |  |  |  |
| `MUONH_LR2X_FLOOR10` | 2.2467 |  |  |  |  |  |  |
| `MUONH_LR2X_FLOOR10_IT4577` |  |  |  | 1.9291 |  |  |  |
| `MUONH_LR2X_IT14685` |  |  | 1.8076 | 1.7455 |  |  |  |
| `MUONH_LR2X_IT4577` |  | 2.0424 | 1.9599 | 1.9028 | 1.8733 | 1.8496 | 1.8346 |
| `MUONH_LR2X_IT4577_FLOOR10` |  |  |  | 1.9291 |  |  |  |
| `MUONH_LR2X_IT4577_S2` |  |  | 1.9579 |  |  |  |  |
| `MUONH_LR2X_IT4577_S3` |  |  | 1.9562 |  |  |  |  |
| `MUONH_LR2X_L12H1024_IT4577` |  |  | 2.0676 |  |  |  |  |
| `MUONH_LR2X_L24H1536_IT4577` |  |  |  |  | 1.9175 |  |  |
| `MUONH_LR2X_L24H2048_IT4577` |  |  |  |  | 2.3259 |  |  |
| `MUONH_LR2X_L24H512_IT4577` | 2.0907 |  |  |  |  |  |  |
| `MUONH_LR2X_L48H1024_IT4577` |  |  | 1.8949 |  |  |  |  |
| `MUONH_LR2X_L60H1024_IT4577` |  |  | 1.8842 |  |  |  |  |
| `MUONH_LR2X_L96H1024_IT4577` |  |  | 1.8695 |  |  |  |  |
| `MUONH_LR2X_S2` | 2.2022 |  |  |  |  |  |  |
| `MUONH_LR2X_S3` | 2.1969 |  |  |  |  |  |  |
| `MUONH_LR2X_TPP1800` | 1.9989 |  |  |  |  |  |  |
| `MUONH_LR2X_TPP600` | 2.0781 |  |  |  |  |  |  |
| `MUONH_LR4X_B4M_IT2289` |  | 2.1152 |  |  |  |  |  |
| `MUONH_LR4X_IT4577` |  |  |  |  | 1.9523 | 1.9154 | 1.9127 |
| `MUONH_LR4X_L48H1024_IT4577` |  |  | 1.9344 |  |  |  |  |
| `MUONH_LR4X_L60H1024_IT4577` |  |  | 1.9081 |  |  |  |  |
| `MUONH_LR4X_L96H1024_IT4577` |  |  | 1.8582 |  |  |  |  |
| `MUONH_LR5P6X_L48H1024_IT4577` |  |  | 1.9923 |  |  |  |  |
| `MUONH_LR5P6X_L96H1024_IT4577` |  |  | 1.8952 |  |  |  |  |
| `MUONH_LRSPLIT_A_IT4577` |  |  |  | 1.9357 |  |  |  |
| `MUONH_LRSPLIT_B_IT4577` |  |  |  | 1.8995 |  |  |  |
| `MUONH_LRSPLIT_C_IT4577` |  |  |  | 1.9025 |  |  |  |
| `MUONH_SPLITFC1` | 2.1821 |  |  |  |  |  |  |
| `MUONH_SWA` | 2.1796 |  |  |  |  |  |  |
| `MUON_LR0P25X` |  | 1.9492 |  |  |  |  |  |
| `MUON_LR0P5X` |  |  |  | 1.6346 |  |  |  |
| `MUON_LR0P5X_TPP1800` | 2.0134 |  |  |  |  |  |  |
| `MUON_LR1X_TPP1800` | 2.0077 |  |  |  |  |  |  |
| `MUON_LR2X` |  |  |  | 1.6429 |  |  |  |
| `OPT_MUON` | 2.2001 | 1.9198 | 1.7491 | 1.6339 | 1.5538 |  |  |
| `OPT_MUONH` | 2.1829 | 1.9094 | 1.7453 | 1.6314 | 1.5540 |  |  |
| `OPT_MUONH_S2` | 2.1846 |  |  |  |  |  |  |
| `OPT_MUONH_S3` | 2.1848 |  |  |  |  |  |  |
| `OPT_MUON_LR0P25X` | 2.2487 |  | 1.7721 |  |  |  |  |
| `OPT_MUON_LR0P25X_IT4577` |  | 2.1199 | 2.0539 | 2.0153 |  |  |  |
| `OPT_MUON_LR0P25X_TPP1800` | 2.0337 |  |  |  |  |  |  |
| `OPT_MUON_LR0P25X_TPP600` | 2.1211 |  |  |  |  |  |  |
| `OPT_MUON_LR0P5X` | 2.2159 | 1.9312 | 1.7565 |  |  |  |  |
| `OPT_MUON_LR0P5X_IT4577` |  | 2.0766 | 2.0063 | 1.9642 |  |  |  |
| `OPT_MUON_LR0P5X_TPP600` | 2.0966 |  |  |  |  |  |  |
| `OPT_MUON_LR1X_IT4577` |  | 2.0614 | 1.9883 | 1.9394 |  |  |  |
| `OPT_MUON_LR1X_TPP600` | 2.0880 |  |  |  |  |  |  |
| `OPT_MUON_LR2X` | 2.1942 | 1.9190 | 1.7544 |  |  |  |  |
| `OPT_MUON_LR2X_IT4577` |  | 2.0505 | 1.9722 | 1.9263 |  |  |  |
| `OPT_MUON_LR2X_TPP1800` | 2.0157 |  |  |  |  |  |  |
| `OPT_MUON_LR2X_TPP600` | 2.0888 |  |  |  |  |  |  |
| `OPT_MUON_LR4X` | 2.2024 | 1.9275 | 1.7698 |  |  |  |  |
| `OPT_MUON_LR4X_IT4577` |  | 2.0567 | 1.9767 | 1.9270 |  |  |  |
| `OPT_MUON_LR4X_TPP1800` | 2.0334 |  |  |  |  |  |  |
| `OPT_MUON_LR4X_TPP600` | 2.0928 |  |  |  |  |  |  |
| `OPT_MUON_S2` | 2.1986 | 1.9210 |  |  |  |  |  |
| `OPT_MUON_S3` | 2.2019 | 1.9192 |  |  |  |  |  |
| `PROD1_DPSK` | 2.1454 | 1.8567 | 1.6886 | 1.5754 |  |  |  |
| `PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K` | 2.1002 | 1.8114 | 1.6398 | 1.5254 |  |  |  |
| `PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K_S2` | 2.1035 |  |  |  |  |  |  |
| `PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K_S3` | 2.1040 |  |  |  |  |  |  |
| `PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K_W32K` | 2.0669 | 1.7805 | 1.6101 | 1.4969 |  |  |  |
| `PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K_W32K_S2` | 2.0706 |  |  |  |  |  |  |
| `PROD1_DPSK_KD_A0P8_S64Y_MIMO25_PP32K_W32K_S3` | 2.0712 |  |  |  |  |  |  |
| `PROD1_DPSK_LATENT2` | 2.1363 | 1.8427 | 1.6693 | 1.5567 |  |  |  |
| `PROD1_DPSK_LATENT2_E2X` | 2.0996 | 1.8072 | 1.6429 |  |  |  |  |
| `PROD1_DPSK_LATENT2_E2X_W32K` | 2.0671 | 1.7768 | 1.6138 |  |  |  |  |
| `PROD1_DPSK_LATENT2_W32K` | 2.1029 | 1.8116 | 1.6394 | 1.5280 |  |  |  |
| `PROD1_DPSK_PROPE25` | 2.1444 | 1.8573 | 1.6863 |  |  |  |  |
| `PROD1_DPSK_PROPE25G` | 2.1452 | 1.8559 | 1.6851 |  |  |  |  |
| `PROD1_DPSK_PROPE25G_W32K` | 2.1126 | 1.8248 | 1.6552 |  |  |  |  |
| `PROD1_DPSK_PROPE25_W32K` | 2.1118 | 1.8261 | 1.6563 |  |  |  |  |
| `PROD1_DPSK_W32K` | 2.1124 | 1.8255 | 1.6586 | 1.5465 |  |  |  |
| `ROUTER_NE_1EM2_SWS` | 2.2021 | 2.1458 |  |  |  |  |  |
| `ROUTER_NE_1EM3` | 2.1907 |  |  |  |  |  |  |
| `ROUTER_NE_1EM3_SWS` | 2.1748 | 1.8917 |  |  |  |  |  |
| `ROUTER_NE_1EM4` | 2.1787 |  |  |  |  |  |  |
| `ROUTER_NE_1EM4_NOBIAS` | 2.1795 |  |  |  |  |  |  |
| `ROUTER_NE_1EM4_NOBIAS_SWS` | 2.1807 |  |  |  |  |  |  |
| `ROUTER_NE_1EM4_SOFTMAX` | 2.1808 |  |  |  |  |  |  |
| `ROUTER_NE_1EM4_SWS` | 2.1679 | 1.8925 | 1.7206 | 1.6054 |  |  |  |
| `ROUTER_NE_1EM5` | 2.1784 |  |  |  |  |  |  |
| `ROUTER_NE_1EM5_BATCH_SWS` | 2.1693 | 1.8914 | 1.7186 |  |  |  |  |
| `ROUTER_NE_1EM5_GAMMA1EM4_SWS` | 2.1670 | 1.8930 |  |  |  |  |  |
| `ROUTER_NE_1EM5_NOBIAS` | 2.1715 |  |  |  |  |  |  |
| `ROUTER_NE_1EM5_NOBIAS_FREQ_SWS` | 2.1650 | 1.8897 |  |  |  |  |  |
| `ROUTER_NE_1EM5_NOBIAS_SWS` | 2.1613 | 1.8832 | 1.7147 | 1.6010 | 1.5213 |  |  |
| `ROUTER_NE_1EM5_NOBIAS_SWS_DENSE1` |  | 1.8789 | 1.7096 |  |  |  |  |
| `ROUTER_NE_1EM5_NOBIAS_SWS_DENSE2` |  | 1.8805 | 1.7121 |  |  |  |  |
| `ROUTER_NE_1EM5_RTADAMW_SWS` | 2.1707 | 1.8933 |  |  |  |  |  |
| `ROUTER_NE_1EM5_SOFTMAX` | 2.1774 |  |  |  |  |  |  |
| `ROUTER_NE_1EM5_SWS` | 2.1675 | 1.8889 | 1.7191 | 1.6019 |  |  |  |
| `ROUTER_NE_1EM5_SWS_DENSE1` |  | 1.8791 | 1.7132 |  |  |  |  |
| `ROUTER_NE_1EM5_SWS_DENSE2` |  | 1.8871 | 1.7148 |  |  |  |  |
| `ROUTER_NE_1EM6` | 2.1778 |  |  |  |  |  |  |
| `ROUTER_NE_1EM6_SWS` | 2.1686 |  |  |  |  |  |  |
| `ROUTER_NE_4EM4_SWS` | 2.1790 | 1.8896 |  |  |  |  |  |
| `SWA` | 2.1796 |  |  |  |  |  |  |
| `SWA_FIT` | 2.1746 | 1.8988 | 1.7252 | 1.6123 |  |  |  |
| `SWA_FIT_ROPE` |  | 1.8968 | 1.7285 | 1.6154 |  |  |  |
| `SWA_FIT_SINK` | 2.1691 | 1.8877 | 1.7214 |  |  |  |  |
| `SWA_FIT_SINK_W128` | 2.1807 | 1.8889 | 1.7185 |  |  |  |  |
| `SWA_FIT_SINK_W1K` | 2.1726 |  |  | 1.6045 |  |  |  |
| `SWA_FIT_SINK_W512` | 2.1707 | 1.8853 | 1.7143 |  |  |  |  |
| `SWA_FIT_SINK_W512_L32K` | 2.1430 |  |  |  |  |  |  |
| `SWA_FIT_SINK_W512_R3` | 2.1650 |  |  |  |  |  |  |
| `SWA_FIT_SINK_W512_R3T` |  | 1.8825 | 1.7164 | 1.6021 |  |  |  |
| `SWA_FIT_SINK_W512_S32K` | 2.1668 | 1.8755 |  |  |  |  |  |
| `SWA_FIT_SINK_W512_W32K` | 2.1410 | 1.8562 | 1.6866 | 1.5747 |  |  |  |
| `SWA_FIT_W1K` | 2.1813 | 1.8936 | 1.7276 | 1.6118 |  |  |  |
| `SWA_FIT_W4K` | 2.1798 | 1.9040 | 1.7250 |  |  |  |  |
