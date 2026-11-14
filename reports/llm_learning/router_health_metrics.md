# MoE router 健康指标怎么读：9 个监测指标

> 2026-11-07 整理。定义依据 Boxiang 的 `cosmos_llm_scaling_ladder/code/scaling_ladder/router_health.py`（`LADDER_ROUTER_HEALTH=100`，每 100 步一次）和评测报告 `qwen3_scaling_ladder_eval.html` 里的指标说明；数字来自报告「router 健康统计」一节（训练结束时的值，seed 1）。
> router 本身的结构、aux loss、expert bias 的更新规则见同目录的 [moe_router.md](moe_router.md)，本文不重复。

**一句话**：9 个指标回答两个问题：expert 之间**分活均不均**（MaxVio、dead），以及 router **有没有失控**（bias、logit 饱和、pinned、最大 |logit|、gate 更新幅度）。

---

## 1. router 一步在做什么（DSV3 sigmoid router）

```
token x ──W_g──► 128 个 logit z_e ──sigmoid──► 分数 s_e ∈ (0,1)
                                                │
               选人：按 s_e + b_e 取 top-8 ◄─────┤   b_e = 每个 expert 的 bias，只管"选谁"
               权重：g_e = c · s_e / Σ(选中 8 个的 s)  ◄┘   bias 不进权重；c = 缩放系数（PROD1 是 2.5）
输出：y = shared(x) + Σ(选中的 8 个) g_e · Expert_e(x)
bias 更新（不走梯度）：每步对每个 expert，分到的 token 少于平均就 +γ，多于平均就 −γ（γ = 1e-3）
```

- 分数 s_e 有两个用途：**决定选谁**（加上 bias 后排序），以及**决定权重**（选中后归一化）。
- 选谁是取 top-8，离散操作，**没有梯度**。router 唯一的训练信号来自权重 g_e，也就是"这个 expert 的输出该多算一点还是少算一点"。
- softmax router（Qwen3 默认，主线 BASE / OPT_MUON / OPT_MUONH / KD / MUONH_FIT）对 128 个 logit 做 softmax 取 top-8，**没有 bias**，均衡只靠 aux loss（1e-3）。

---

## 2. 速查表

| 指标 | 量什么 | 统计范围 | 健康标准 | softmax 能看吗 |
|---|---|---|---|---|
| MaxVio（层平均） | 最忙的 expert ÷ 平均 − 1，所有层取平均 | 一整个 step，所有 micro-batch、所有 rank | 逐渐下降 | 能 |
| MaxVio（最差层） | 同上，取最不均衡的一层 | 同上 | 逐渐下降 | 能 |
| dead expert (%) | 分到不足平均 10% 的 expert 占（层数 × 128）的比例 | 同上 | 0 | 能 |
| \|bias\| 最大 | 所有层、所有 expert 的 bias 最大绝对值 | bias buffer | < 1 | 不适用（没有 bias） |
| logit 饱和 (%) | \|z\| > 4 的（token, expert）对的比例，含 z < −4 | rank 0 的第一个 micro-batch | 本身不是判据 | 只作数量级参考 |
| z > +4 (%) | 只算正向饱和 | 同上 | < 5% | 无意义 |
| top-8 pinned (%) | 选中的 8 个里 z > 4（分数钉在 1）的比例 | 同上 | < 5% | 无意义 |
| 最大 \|logit\| | 所有层里最大的 \|z\|，正负都算 | 同上 | 看趋势，不应一路上涨 | 能（看趋势） |
| gate 更新幅度 | 这一步 router 权重变化量 ÷ 权重大小，各层平均 | 一整个 step | 只作参考，不作判据 | 能 |

---

## 3. 负载：MaxVio 与 dead expert

一个 step 有 2^21 个 token，每个 token 选 8 个 expert，摊到 128 个 expert 上，**平均每个 131,072 个 token**。

- **MaxVio**（DeepSeek-V3 的定义）= 最忙的 expert 分到的 token ÷ 平均值 − 1。
  - 0 = 完全均衡；0.17 = 最忙的分到约 15.3 万；3.9 = 最忙的分到 4.9 倍、约 64 万。
  - 为什么重要：expert 并行时，大家都要等最忙的那张卡，它同时决定速度和显存。
  - **最差层**要单独看：层平均不错时，问题往往藏在某一层里（例如第 0 层）。
- **dead expert**：一个 step 里分到不足平均 10%（约 1.3 万 token）的 expert。它拿不到梯度，就不会变好，也就更不会被选中，是自我强化的浪费。
  - 把百分比换成个数更直观：q0 有 12 层 × 128 = 1536 个 expert，0.0651% = 1 个，0.846% = 13 个；q1 有 18 × 128 = 2304 个，0.0434% = 1 个，0.174% = 4 个。

---

## 4. bias：只有 DSV3 / DPSK 才有

只有开了 `--moe-router-enable-expert-bias` 的 run 才有 bias。在我们 ladder 里，这个开关总是和 sigmoid 打分一起出现：报告里所有 "DSV3…" 组，以及 PROD1_DPSK（sigmoid + expert bias 1e-3 + seq_aux_loss 1e-4）。Megatron 里这两个开关是独立的，理论上 softmax 也可以加 bias，只是我们没这么配过。

**|bias| 上限为什么是 1**：分数 s 在 0 到 1 之间。两个 expert 的 bias 差一旦超过 1，bias 低的那个对**任何** token 都赢不了，先后顺序已经和内容无关，由 bias 决定。

> 例：A 的分数 0.9、bias 0；B 的分数 0.1、bias +0.85。B 的选人分 0.95 > 0.9，选了 B，尽管 router 本身更想要 A。

两点注意：
- **真正起作用的是同一层内 bias 的差**（"bias 极差"，报告逐层热力图里有）。所有 expert 一起抬高不改变路由，例如 DSV3 q3 第 0 层的 bias 都在 +11.2 到 +11.7 之间，|b| 很大，但极差只有 0.5。
- **整体漂移的来源**：bias 按符号更新，不是零和的。多数 expert 略低于平均、少数远高于平均时，加的总量大于减的总量，整层一起往上漂。

**softmax 靠什么维持均衡**：靠 aux loss 通过梯度推 logit，没有一个单独的量能读出"有多少是硬撑的"，只能看 MaxVio 和 dead 的结果。

---

## 5. logit：饱和、z > +4、pinned、最大 |logit|

### 5.1 为什么饱和 = 梯度传不过去

梯度要沿这条链传回 router：**loss → y → g_e → s_e → z_e → router 权重 w_e**。其中 s_e → z_e 这一环乘的是 sigmoid 的导数 σ'(z) = s·(1 − s)：

| z | 分数 s | σ'(z) | z 增加 1 时 s 变多少 |
|---:|---:|---:|---:|
| 0 | 0.5 | 0.25 | +0.23 |
| 2 | 0.88 | 0.10 | +0.07 |
| 4 | 0.982 | 0.018 | +0.012 |
| 6 | 0.9975 | 0.0025 | +0.002 |

> 例：某个 token 选中了 A，loss 的梯度说"A 很有用，权重应该再大一点"。z_A = 0 时，router 把 z_A 调大一点，权重就明显上升，信号传得过去；z_A = 6 时，分数已经是 0.9975，再怎么调 z_A 也几乎不变，"A 应该更重要"这个信号到这里就断了。

Adam 类优化器会把整体步长归一化，router 不会完全停住；但饱和的（token, expert）对，对更新方向几乎没有贡献。

### 5.2 只切断 router 这一路

对一个被选中的 expert：

| 路径 | 要不要乘 σ'(z) | 饱和时 |
|---|---|---|
| ① loss → Expert_e 的参数（乘 g_e ≈ c/8） | 不要 | **照常学习**，权重钉在 c/8 反而稳定 |
| ② loss → g_e → s_e → z_e → router 权重 W_g | 要 | **被切断**：router 学不动 |
| ③ loss → g_e → s_e → z_e → 隐状态 x | 要 | 被切断，但这一支很细；x 还能从 expert、残差、attention 拿到梯度 |

- 没被选中的 expert 本来就不参与输出，收不到这个 token 的梯度，原因是没被选中，和饱和无关。
- **aux loss 也走 ②**：seq_aux_loss 用 router 分数，推均衡的梯度同样要乘 σ'。主 loss 和 aux loss 的信号同时变弱，只剩不走梯度的 bias 还能调负载。所以在 DSV3 里，logit 失控和"均衡靠 bias 撑"会一起出现。
- 危害是**间接的**：模型主体照常学，但"哪个 token 交给哪个 expert"的分工冻结在早期状态。伤不伤 loss、伤多少，要看报告里 router 各组的 loss 对比，监测指标给不了答案。

### 5.3 z > +4 是坏信号，z < −4 无所谓

- **logit 饱和 (%)** 包含 z < −4（"明确不要这个 expert"）。128 个只要 8 个，大部分本来就该是明确的"不要"，而且这些 expert 没被选中，饱和不损失任何东西。所以这个比例在 37–73% **本身不算坏**。
- **z > +4 (%)**：被选中的 expert 分数钉在 1。一个 token 选中的 8 个都约等于 1 时，归一化后每个都是 c/8，router 无法表达"A 比 B 更有用"，而且改不动。上限 5%。
- **top-8 pinned (%)**：同一件事，分母从 128 换成 8，所以 **pinned ≈ 16 × (z > +4)**。实测对得上：0.55% × 16 ≈ 8.9%，0.05% × 16 ≈ 0.8%。上限 5%。
- 对 softmax router：给所有 logit 加同一个常数，结果不变，起作用的是 logit 之间的差，所以 |z| 是否大于 4 这个绝对阈值说明不了什么。

### 5.4 最大 |logit| 与"硬分类器"

- **软**：router 给出有梯度的分数（A 0.7、B 0.55），能被梯度慢慢调整。**硬**：分数只剩 0 和 1，像盖章，被选中的几个之间也分不出高低。
- "硬分类器"指 router 对每个 token 都**极其确定**，不是分不了类。坏处有三个：
  1. **确定不等于正确**：决定大多在训练早期形成，之后梯度传不过去，早期的分配被锁死；
  2. **权重失去表达力**：选中的 8 个权重一样；
  3. **DSV3 里选人变成由 bias 决定**：很多 expert 的分数都约等于 1 时，按 s + b 排序，s 之间没有差别，谁被选中就由只看负载、不看内容的 bias 决定。
- **为什么只能看趋势**：最大 |logit| 只是最极端的一个值，正负都算，z = −184 只是非常坚决地不选，本身无害。它说明的是 router 权重（或输入）的尺度在变大，而尺度变大会把所有 logit 往两端推。被选中的是否已经变硬，要看 z > +4 和 pinned。两者在数据里是一起动的：

| 组 | 最大 \|logit\| | top-8 pinned |
|---|---:|---:|
| DSV3 + router LR ×0.19（B3） | 15–29 | 0.8–2.2%（健康） |
| DSV3 + AdamW gate + LR ×0.19（B2） | 54–153 | 8.9–10.9%（超过 5%） |

---

## 6. gate 更新幅度

‖W_after − W_before‖_F ÷ ‖W_before‖_F，各层平均，是 router 权重这一步的相对变化。训练初期，在 MuonH LR 律给出的 LR 下约 0.8，router LR ×0.19 时约 0.16，BASE 自己正常工作的 router 约 0.15。Boxiang 明确它**只作参考，不作判据**，读趋势。

它是在 bf16 的权重副本上量的；训练末期每步的变化比 bf16 的精度还小，读数会被量化，所以末期的值不能按 LR 成比例来读（这一条是推断，未单独验证）。

---

## 7. 预设判据与当前读数

Boxiang 预设的判据（`dsv3/DESIGN.md` 第 3 节）：正向饱和（z > +4、pinned）< 5%；dead = 0；|b| < 1 且大致按每步 γ 增长；MaxVio 逐渐下降；gate 更新幅度只作参考。

训练结束时的读数（2026-11-07，报告「router 健康统计」；组名 A / B2 / B3 见 [moe_router.md](moe_router.md) 第 7 节）：

| 组 | MaxVio 层平均 / 最差层 | dead | \|b\| 最大 | pinned | 最大 \|logit\| | 判断 |
|---|---|---|---|---|---|---|
| DSV3 + router LR ×0.19（B3），q0 / q1 | 0.13–0.16 / 0.27–0.48 | 0 | 0.31–0.55 | 0.8–2.2% | 15–29 | 全部达标 |
| DSV3 默认 LR（A），q0 → q3 | 0.15–0.17 / 0.31–0.69 | 0 | 1.0 → 3.2 → 5.9 → 12 | 无数据 | 26 → 100 → 146 → 184 | 均衡很好，但 \|b\| 和 logit 随 rung 上涨，按预设规则不能直接采用 |
| DSV3 + AdamW gate + LR ×0.19（B2），q0 / q1 | 0.68 / 2.7–3.6 | 2–4 个 | 1.1–3.5 | **8.9–10.9%** | 54–153 | pinned 超限 |
| softmax（主线 MUONH_FIT），q0 / q1 | **1.1–1.6 / 3.3–3.9** | 1 个 | 无 bias | 无数据 | 20–35 | 负载明显不均，aux loss 没把均衡压紧 |
| DSV3 去掉 seq aux，q0 | 0.65 / **4.2** | **13 个** | 1.7 | 1.5% | 26 | seq aux 有必要 |
| DSV3 + MuonH gate，q0 | 0.14 / 0.34 | 0 | 0.57 | 1.9% | 19 | 达标 |

- A 组 |b| 包含整体漂移，要配合逐层热力图里的"bias 极差"看。
- "无数据"：z > +4 和 pinned 是后期才加的监测，那几组跑完时还没有。
- 主线 softmax 这一条，就是 11-05 MoE 手册对照筛查里说的"主线负载均衡没人监控"：aux loss 读数在 1 附近看起来正常，但最差层 MaxVio 已经到 3–4（见 `reports/dense_pretrain/2026-11-05_moe_handbook_audit.md`）。

---

## 8. 来源

- 指标定义：`cosmos_llm_scaling_ladder/code/scaling_ladder/router_health.py`（文件头注释、`load_stats`、`logit_stats`）。
- 报告文字与数字：`cosmos_llm_scaling_ladder/reports_xx/gen_eval_html.py`（`rt_st_def`、`rt_hm_note`），`qwen3_scaling_ladder_eval.md`「router 健康统计」。
- PROD1_DPSK 的 router 参数：`runs/scaling_ladder/prod_v1/qwen3_tpp200_prod1_dpsk-q0-tpp200-s1/argv.txt`。
