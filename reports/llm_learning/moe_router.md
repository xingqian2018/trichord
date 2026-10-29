# MoE router：softmax 与 DeepSeek-V3 式 router、负载均衡、gate 的优化器（以 Qwen3 ladder 为例）

> 依据：Boxiang 的 router 消融（`cosmos_llm_scaling_ladder/reports/dsv3_router_experiment_design_2026-10-26.md`、`DECISIONS.md` 10-26），
> Runlong.md TODO 9（负熵均衡损失），以及 ladder 使用的 Megatron 代码 `code/megatron-lm-80781ab0ab/megatron/core/`（下文代码行号均指这份代码）。
> 结果数字来自我们的 doc-isolated 评测（`reports_xx/qwen3_scaling_ladder_eval.html`，指标 overall_tsqw，负数 = 更好）。2026-10-29 整理。

---

## 1. router 在做什么

ladder 的每个 MoE 层有 **128 个 expert，每个 token 选 8 个**（没有 shared expert）。router 就是决定"选哪 8 个、各占多少权重"的那一小块网络：

- **gate 权重 W_g**：形状 128 × 宽度。每个 token 的隐状态 x 乘上 W_g，得到 128 个 logit z。
- **打分**：把 logit 变成分数（softmax 或 sigmoid），选出 8 个。
- **加权**：被选中的 8 个 expert 的输出按权重相加。
- **负载均衡**：防止大部分 token 挤到少数几个 expert 上。

---

## 2. 两种 router

### 2.1 softmax router（原来的底座，Qwen3 原版）
- 对 128 个 logit 做 **softmax**，取分数最高的 8 个，权重就是这 8 个的 softmax 概率（再归一化）。
- 负载均衡只靠一项 **aux loss**（Switch Transformer 式，系数 **1e-3**），见第 3 节。

### 2.2 DeepSeek-V3 式 router（DSV3）
- 每个 expert 的分数**单独过 sigmoid**，各自在 0–1 之间，互相不竞争。
- **选谁**：按"sigmoid 分数 + bias"排序取前 8。
- **权重**：只用这 8 个的**原始 sigmoid 分数**（不含 bias）在 8 个之间归一化。
- 负载均衡**主要靠 bias**（第 4 节），另外保留一个很小的 **seq aux loss（1e-4）** 兜底（第 3 节）。
- 设计思路：尽量不用 aux loss，避免它干扰语言模型的主 loss。

代码（`transformer/moe/moe_utils.py` L953–959）：

```python
scores_for_routing = scores + expert_bias      # bias 只加在"用来排序"的分数上
_, top_indices = topk(scores_for_routing, 8)   # 按"分数 + bias"选 8 个
scores = gather(scores, top_indices)           # 取回这 8 个的原始 sigmoid 分数（不含 bias）
probs = scores / scores.sum(...)               # 在 8 个里归一化，作为输出权重
```

**结论：bias 只影响选谁，不影响被选中 expert 的输出权重。**

---

## 3. 两种 aux loss：aux loss 1e-3 与 seq aux 1e-4

两者公式形式相同：

  **L_aux = α · Σ_e f_e · P_e**

- f_e：某个 expert 被选中的比例（乘 128/8 归一化，完全均衡时为 1）。它是**数出来的**，不可导。
- P_e：所有 token 给这个 expert 的平均分数。它**可导**，梯度只经过这一项。

某个 expert 被选得多、分数又高，这一项就大，梯度会压低大家给它的分数。

| | **aux loss**（softmax 底座） | **seq aux**（DSV3 底座） |
|---|---|---|
| Megatron 参数 | `--moe-router-load-balancing-type aux_loss` | `seq_aux_loss` |
| 系数 α | 1e-3 | **1e-4** |
| 统计范围 | 整个 micro-batch 一起算 | **每条序列单独算**，再对序列取平均 |
| P_e 用的分数 | softmax 概率 | sigmoid 分数在 128 个 expert 上归一化后的值 |
| f_e 用的 top-8 | 实际路由的 top-8 | **不加 bias 的 top-8**（只看 sigmoid 分数）；实际路由用的是加了 bias 的 top-8，两者分开 |
| 在负载均衡里的地位 | 唯一手段 | 辅助，主要靠 bias |

### 3.1 "每条序列单独算"具体指什么
ladder 每张卡的 micro-batch 是 **4 条序列 × 8192 token = 32k token**（q2：TP 1、CP 1、EP 4）。

- **seq aux**：4 条 8k 序列**各算一次** f_e、P_e，得到 4 个 loss 再平均。代码把 batch 维度并进 expert 维度（`reshape(seq_length, -1)`），一次调用同时算出各条序列（`router.py` `_apply_seq_aux_loss`）。
- **aux loss**：32k token **放在一起算一次**。
- 如果一条序列被切到多张卡上（TP / CP > 1），seq aux 会先在这几张卡之间汇总，保证用的是整条序列的统计（`_get_aux_loss_groups` 的 reduce group = TP × CP）。**不会跨 data-parallel 的 rank 汇总**：不同 rank 上是不同的序列。
- 所以 seq aux 管得更细：即使 32k 总体是均衡的，只要某一条 8k 序列内部集中在少数几个 expert，它也会惩罚。

### 3.2 aux loss 会更新什么
- **gate 权重 W_g：会更新。** aux loss 是普通的可导损失，加在总 loss 里，梯度经 P_e 反传到 W_g，再由 gate 的优化器更新。
- **expert bias：不会更新**（见第 4 节）。

### 3.3 统计范围为什么重要：micro-batch、序列、global batch

Megatron 有三种 aux loss，公式一样，区别只在 **f_e 是在多少 token 上数出来的**：

| `--moe-router-load-balancing-type` | f_e 的统计范围 | 跨哪些卡汇总 f_e | 谁在用 |
|---|---|---|---|
| `aux_loss` | 一个 micro-batch 的全部 token | TP × CP（同一批序列被切开的几张卡） | ladder 的 softmax 底座 |
| `seq_aux_loss` | 每条序列单独 | TP × CP | ladder 的 DSV3 底座、DeepSeek-V3 |
| `global_aux_loss` | 整个 global batch | TP × **DP** × CP，再对一步内的各 micro-batch 累计平均（每步清零） | Qwen3 官方（"global-batch load balancing loss"） |

三种都一样：**P_e 只用本卡自己的 token**（可导），各卡的部分加起来才是完整的 loss（Megatron `switch_load_balancing_loss_func` 的注释）。

**数值例子：两条序列，一条全是代码，一条全是中文。** 假设代码序列只用 expert 1–64，中文序列只用 expert 65–128，各自内部均匀：
- 合在一起数（`aux_loss`）：128 个 expert 被选的次数一样多，loss = 1（完全均衡），**没有惩罚**。
- 每条序列单独数（`seq_aux_loss`）：每条序列只用了一半 expert，loss ≈ 2，**有惩罚**，推着每条序列内部也要用满 128 个 expert。
- 所以统计范围越小，越压制"不同领域用不同 expert"（专业化）；范围越大，越允许专业化。Qwen3 用 global batch 就是为了鼓励专业化（Qiu et al. 2025，arXiv 2501.11873）。
- ladder 设计时（`reports/qwen3_ladder_design_v1_2026-10-19.md`）发现，当时 fork 里 `global_aux_loss` 的实际系数是 1e-3 / DP，DP 不同的 rung 之间差 16 倍，所以 ladder 改用 `aux_loss`。

**"32k 一起算"在 ladder q2 上是什么意思**：每张卡一个 micro-batch = 4 条序列 × 8192 = 32,768 token（TP 1、CP 1）。`aux_loss` 把这 32,768 个 token 的 8 次选择（共 262,144 次）放在一起，数出每个 expert 的 f_e，再用这 32,768 个 token 的分数求 P_e，得到 **1 个** loss。`seq_aux_loss` 则对 4 条序列各算一次，得到 4 个 loss 再平均。不跨卡、也不跨 micro-batch：一步 2M token（256 条序列）在 aux loss 里被切成 64 个互不相干的 32k 小块。

这 32k 是**一次** forward：4 条序列组成一个 [8192, 4, 宽度] 的张量一起算，router 把它摊平成 32,768 行 logit，所以 `aux_loss` 直接对这 32,768 行求和；`seq_aux_loss` 要先把 batch 维拆回来才能按序列算。每步每张卡做几次 forward 取决于卡数：GBS 256 条 = micro-batch 4 条 × DP × 累积次数。

| rung | 节点（每节点 4 卡） | DP | 每张卡每步的 micro-batch 数（梯度累积） | 64 个 32k 小块分布在 |
|---|---|---|---|---|
| q0 | 4 | 16 | 4 | 16 张卡 × 4 次 |
| q1 | 8 | 32 | 2 | 32 张卡 × 2 次 |
| q2、q3 | 16 | 64 | 1（没有梯度累积） | 64 张卡 × 1 次 |

注意："序列"是一个 8192 token 的训练样本，里面可能拼了好几篇文档。attention 上：
- **同一个 micro-batch 里的 4 条序列互相看不到**：它们是 batch 维上的 4 个独立样本，attention 只在每条序列内部算。4 条序列只在 router 的 `aux_loss` 统计里被放到一起。
- **同一条序列里拼进去的几篇文档能互相看到**：ladder 训练不加文档 mask（没开 `--reset-attention-mask` / `--reset-position-ids`，DECISIONS.md "packed 8192-token samples without attention mask"），只有 causal mask，后面的文档能 attend 前面的文档，位置编号也是连续的。
- 我们的 doc-isolated 评测则是一篇文档一个样本，所以不会有跨文档的 attention。

### 3.4 复杂一点的例子：64 张 GB300（16 节点 × 4）训练 Qwen3-235B-A22B

模型：94 层，宽度 4096，每层 128 选 8，无 shared expert。**下面的切分是为了举例假设的**（235B 的实际 layout 和 GBS 还没定）：

- TP 2（开 sequence parallel）、PP 4、CP 1 → DP = 64 / (2 × 4) = 8。
- 每个 PP stage 有 16 张卡、约 23–24 层。expert 部分把这 16 张卡重新分组成 EP 16（每卡存每层 128 / 16 = 8 个 expert）。
- 序列长 8192，micro-batch 2 条序列，global batch 1024 条序列（约 8.4M token）。每个 DP 副本一步处理 1024 / 8 = 128 条序列 = 64 个 micro-batch。

对某一层（比如 stage 0 上的第 5 层），一步之内：

1. **一个 micro-batch**：2 条序列 × 8192 = 16,384 token。sequence parallel 把序列维度切成两半，TP 的两张卡各拿 2 × 4096 = 8,192 个 token，router 各算各的。
2. **`aux_loss`**：每张卡数自己 8,192 个 token 的选择，在 TP 的两张卡之间求和 → f_e 覆盖整个 micro-batch 的 16,384 token（两条序列混在一起）。P_e 各卡只用自己的 8,192 个 token。结果：**每个 micro-batch 一个 loss**。不跨 DP，不跨 micro-batch：一步里有 8（DP）× 64（micro-batch）= 512 个各自独立的 16k 小块。
3. **`seq_aux_loss`**：同一个 micro-batch，两条序列各自数 f_e（也在 TP 两张卡间求和，凑齐每条序列完整的 8192 token）→ 2 个 loss 再平均。一步里共 1024 个按序列算的 loss。
4. **`global_aux_loss`**：f_e 在本 stage 全部 16 张卡（TP 2 × DP 8）之间求和 → 一次覆盖 8 个 micro-batch × 16,384 = 131k token；并且在一步的 64 个 micro-batch 之间累计平均（第 k 个 micro-batch 用前 k 个的平均）。到最后一个 micro-batch，f_e 覆盖整个 global batch 的 8.4M token。
5. **DSV3 的 expert bias**：每张卡把 64 个 micro-batch 的选择次数累加起来；一步结束时（`finalize_model_grads.py`）在本 stage 的 16 张卡（TP × DP × CP）之间求和 → 整个 global batch 8.4M token 的计数 → 每个 expert 的 bias ±1e-3，每层一次。

另外几点：
- **PP**：每层的 router 只在自己的 stage 上，所有汇总都只在本 stage 的 16 张卡里做，不同 stage 的层互不相干。
- **EP 不参与这些统计**：router 不按 expert 切开（每张卡都有完整的 W_g 和 bias），EP 只决定选好之后 token 被 all-to-all 发到哪张卡上去算 expert。
- **TP 两张卡上的 bias 永远一样**：它们求和后看到的计数相同，按同样的规则更新。
- 规模变大后，bias 看的范围（整个 global batch）不变；`aux_loss` 和 `seq_aux_loss` 看的范围由 micro-batch 大小和序列长度决定，与 GPU 数无关；`global_aux_loss` 看的范围也是整个 global batch。

---

## 4. expert bias（DSV3）

- **不是参数，而是 buffer**（`router.py` L238–245 `register_buffer('expert_bias', zeros(128, float32))`），没有梯度，**任何 loss 都改不了它**。
- **更新规则**（`moe_utils.py` L1245 `get_updated_expert_bias`，由 `distributed/finalize_model_grads.py` L377 `_update_router_expert_bias` 在每步调用）：
  1. 每个 micro-batch 累计各 expert 实际分到的 token 数（`local_tokens_per_expert`，int64）。
  2. 每步在**所有 TP × DP × CP 的 rank 之间求和**，也就是整个 global batch（2M token）。
  3. 对每个 expert：分到的 token **高于平均就减 γ，低于平均就加 γ**（`sign(平均 − 实际)`），γ = `--moe-router-bias-update-rate` = **1e-3**。
  4. 每一层的 bias 各自独立更新。这一步和 loss、优化器都无关。
- 只有相对大小有用：所有 expert 的 bias 一起加同一个数，不改变 top-8。

### 4.1 bias 太大会怎样（病态）
预设的健康上限是 |bias| < 1。bias 远超 1 时，选择几乎完全由 bias 决定：有些 expert 的 sigmoid 分数接近 0，却因为 bias 大而被选中；选中后权重又接近 0，对输出几乎没有贡献，**等于浪费了 8 个名额里的一部分**。负载看起来是均衡的（MaxVio 很低），实际上是 bias 在硬撑，gate 自己没学会。

---

## 5. gate 用什么优化器、多大 LR

- **AdamH**：MuonH 默认的做法，用 Adam 更新，但把 W_g 约束在固定半径的球面上（hyperball），权重范数不会变大。
- **AdamW**：普通 AdamW，权重范数可以自由变化。
- **router LR ×1**：gate 和模型主体用同一个 LR（新 LR 公式下 q0 是 0.0164）。
- **router LR ×0.19**：gate 的 LR 降到约 3.1e-3，大约是 AdamW 基线自己的 LR 水平。探针 B1（AdamW gate、×1）的分数饱和比例在 300 步内从 97% 只降到 67–69%，远超 5% 的门槛，所以才降到 0.19。

---

## 6. 健康指标（Boxiang 的 router 监测，`router_health.py`，每 100 步一次）

| 指标 | 含义 | 预设上限 |
|---|---|---|
| MaxVio | 最忙的 expert 比平均多分到多少 token（0 = 完全均衡） | — |
| dead expert | 几乎没有 token 分到的 expert 占比 | — |
| \|bias\| 最大值 | 负载是不是靠 bias 硬撑 | 1 |
| z > +4 / top-8 pinned | 被选中的 expert 分数卡在 1 附近的比例，gate 失去区分度 | 5% |
| \|z\| > 4（不分正负） | 包括 z < −4（明确拒绝某个 expert，是正常的） | — |
| 最大 \|logit\| | logit 有没有失控变大 | — |

---

## 7. ladder 上的实验和结论（新 LR 公式，128 选 8）

数值 = 实验组 − 对照（overall_tsqw，负数 = 更好）；q0 是 3 个 seed 的平均，其余 rung 1 个 seed。健康数字来自 Boxiang 的 router 监测（DECISIONS.md 10-26）。

**MuonH 主体（对照 ①：MUONH_FIT，softmax router + aux 1e-3，AdamH gate，router LR ×1）**

| 组 | router | gate | router LR | 相对 ① | 健康 |
|---|---|---|---|---|---|
| A MUONH_FIT_DSV3 | DSV3 | AdamH | ×1 | q0 +0.0023、q1 −0.0024、q2 +0.0018、q3 +0.0017 | **病态**：bias 一直涨，q1–q3 到 3–5，logit 60–87 |
| B2 | DSV3 | AdamW | ×0.19 | q0 +0.0051、q1 +0.0020 | **病态**：约 10% 被选中 expert 的分数卡在 1，bias > 1 |
| B3 | DSV3 | AdamH | ×0.19 | q0 +0.0074 | **最健康**：q0 第 3000 步 \|bias\| 0.26、最大 logit 14.8 |

**AdamW 主体（对照 ②：BASE，softmax router + aux 1e-3，AdamW gate，router LR ×1）**

| 组 | router | gate | router LR | 相对 ② | 健康 |
|---|---|---|---|---|---|
| C BASE_DSV3_ONETREE | DSV3 | AdamW | ×1 | q0 −0.0012、q1 +0.0054、q2 +0.0022 | 正常（q0 末 \|bias\| 0.40、最大 logit 18） |

结论：
1. **loss 上，DSV3 router 和 softmax 没有实质差别**：MuonH 上 q0–q3 都在 ±0.003 以内；AdamW 上也没有一致的方向。
2. **两种病态，原因不同**：
   - A（AdamH gate、router LR ×1）：gate 自己学不会负载均衡，全靠 bias 补，bias 越训越大。router LR 降到 ×0.19（B3）就健康了，所以这个问题出在 **router LR**。
   - B2（AdamW gate、×0.19）：和 B3 的 router LR 一样，只有 B2 把选中的分数卡在 1，所以"卡死"出在 **AdamW gate 本身**（DECISIONS 10-26 17:29）。
3. 但 **健康的 B3 在 q0 loss 更差**（比 softmax 差约 0.007），"健康"和"loss 不变差"目前还没有同时做到。
4. 换不换 router，**MuonH 相对 AdamW 的领先只变了约 0.004**（Boxiang 的 q0 三 seed 读数），优化器的结论不依赖 router。
5. 换 router 后，**新 LR 公式仍然接近最优**（DSV3 下最优倍数约 1.05）。

### 7.1 宽模型上 router 会崩（h ≥ 3072）
LR 公式检查里的 L24 h4096 格子（235B 的宽度）：MuonH + softmax router 在 h ≥ 3072 时 router 崩掉，负载均衡 loss 第 50 步就到 15（1 = 均衡，上限 E/k = 16 = 所有 token 都选同样 8 个 expert）。探针结果（DECISIONS.md 10-26）：
- gate 改用 AdamW：没用；
- **router LR 降到 ×0.25：修好了**。

结论和第 7 节一致：**router 的 LR 要比主体低，而且越宽越要低**。Runlong 的候选规则是 `router LR 倍数 = min(1, 2048 / h)`，要在 h4096 的 DSV3 底座上验证（TODO 8 A.4）。

---

## 8. 下一步：负熵均衡损失（Runlong TODO 9，arXiv 2605.15403）

把负载均衡损失（DSV3 底座的 seq aux 1e-4，或 softmax 底座的 aux 1e-3）换成论文的**负熵均衡损失**：
- 每层维护一个 router 概率（top-8 之前的分数）在 batch 上的指数滑动平均 m（η = 0.65）；
- 损失 = α · E · Σ_e p_e · (log m_e + 1)，即对 m 的负熵求梯度。
- 和 Switch 式 aux loss 的区别：Switch 用的是当前 batch 实际的选中频率 f；负熵损失用的是概率的历史平均，在接近均衡时"刚度"是 Switch 的 E 倍（E = 128），并对快饿死的 expert 有 log 屏障。
- 第一阶段 q0（单 seed，负数更好）：DSV3 底座保留 bias，α = 1e-5 / 1e-4，overall_tsqw 分别是 −0.0017 / −0.0015；softmax 底座按论文原样复现，α = 1e-5 / 1e-4，分别是 −0.0005 / +0.0029。都还在噪声范围内。
- Runlong 的采用规则：在 q2 读数时，q0 三个 seed + q1 + q2 方向一致、差距 ≥ 0.003、≥ 3 个标准误、健康不变差，才进入底座。
