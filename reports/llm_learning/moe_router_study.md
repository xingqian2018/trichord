# MoE router 学习笔记（moe_router_study）

> **这是给 agent 读的版本**：事实、公式、数字、出处都在这里，叙述从简。给人读的是同目录的 `moe_router_study.html`，内容相同，多了图和排版。
> - 最后更新：2026-11-10。
> - 合并自 `moe_router.md`（10-29，router 结构、aux loss、bias、gate 优化器、10 月消融）和 `router_health_metrics.md`（11-07，9 个健康指标）。
> - 新增：Quantile Balancing 三种实现 QB / QBX / QBK3（11-05 起在 E2X 上的实验），router 的梯度，门控权重的归一化，一批常见误解。
> - 约定：Δ = 实验组 − 对照。指标 overall_tsqw（loss，负数 = 更好），来自我们的 doc-isolated 评测（`cosmos_llm_scaling_ladder/reports_xx/qwen3_scaling_ladder_eval.md`）。单 seed 噪声约 ±0.002–0.003。
> - 可信度标记：**【核对】** = 看过代码、argv 或日志；**【推断】** = 没有单独验证；**【模拟】** = 我们自己的玩具模拟，不是训练结果。

---

## 0. 速览：最容易答错的 12 件事

1. router 的分数有两个用途：**选谁**（top-k，离散，没有梯度）和**权重**（有梯度）。bias / β / 门槛只影响选谁，不进权重。【核对】
2. sigmoid router 的权重：g_e = c · σ(z_e) / Σ_{j∈选中} σ(z_j)。只在选中的 k 个里归一化，c = 2.5（PROD1 / E2X），所以 Σg = 2.5，不是 1。【核对】
3. router 的梯度只经过权重 g：∂L/∂z_i = g_i (1 − p_i)(a_i − ā)。没选中的 expert 拿不到这个 token 的梯度。QB / QBK3 没有 aux loss，所以 router 只剩这一条梯度来源。【推导】
4. QB 的 β：**每层 × 每个 expert 一个数**，是 buffer，存进 checkpoint，所有 token 共用。α：**每层 × 每个 token 一个数**，每次 forward 现算，用完就扔。没有"整层一个"的门槛。【核对】
5. QB 取分位数的对象是 margin = z − α，不是原始 z。"新门槛下正好 C 个 token 选它"只在 α 不变时成立，实际只是近似。【核对 + 模拟】
6. 三种 QB 实现的区别（详见 §7.4）：
   - QB：在 logit 空间，每个 micro-batch 各取分位数，再平均。
   - QBX：在 logit 空间，整个 global batch 合并成直方图再取分位数。
   - QBK3：在 sigmoid 分数空间（即 DSV3 的 bias），整个 global batch 合并成直方图再取分位数。
   - **三者都要跨卡同步**，区别在同步的是什么（E 个分位数，还是直方图）。【核对】
7. QBX **不是** K3 的做法：它只借了 K3 的全局合并。QBK3 才是按 K3 原文实现的。【核对：`share/megatron_qbx_tests/K3_QB_REFERENCE.md`】
8. 生产配置（截至 11-09）是 E2X + QBK3：59B-A4B 的 de-risk 默认 `MOE_BALANCING=qbk3`，PROD_LR_FIT 的 41 个 run 全是 `_qbk3`。461B 的脚本仍默认 dsv3。【核对：lb-inventory 记录】
9. E2X 上的 loss：BIASONLY、QB、QBX、QBK3 在 q2 都约为 −0.003，q3 都在 ±0.0005 以内。**loss 上分不出谁更好**；差别在 router health 和工程性质上（§10.1）。【核对】
10. pinned 和 z > +4 由 router 的 hyperball（AdamH）挡住，和 aux loss 无关。去掉 aux loss 后变多的只是 z < −4，这是无害的。【核对：数据；机制部分为推断】
11. "logit 饱和 (%)" 包含 z < −4，不能当作坏信号。|bias| 要看同一层里的极差，不看绝对值。"top-8 pinned" 在 E2X（选 16 个）上**仍然只看最大的 8 个原始 logit**。【核对：`router_health.py` L69 默认 topk=8，L213 调用时没传参】
12. PROD1 系列的 micro-batch 永远是 1 条 32K 序列（q0–q5 都是），所以 seq aux 和"按 micro-batch 统计"在这里统计的是同一批 token；QB 的"每个 micro-batch 取分位数"等于"每条序列取分位数"。【核对：argv】

---

## 1. 配置速查

| 配置 | 时间 | 训练长度 / micro-batch | routed expert / 选几个 / shared | expert 形状 | 打分 | 负载均衡 | router 优化器 |
|---|---|---|---|---|---|---|---|
| 老 ladder softmax 底座（BASE、MUONH_FIT、OPT_*、KD 主线） | 10 月 | 8K；micro-batch q0–q3 是 4 条、q4 是 2 条、q5–q6 是 1 条 | 128 / 8 / 0 | h → 3h/8 → h，每层都是 MoE | softmax | aux_loss 1e-3 | MuonH 主体用 AdamH，AdamW 主体用 AdamW |
| 老 ladder DSV3 组（MUONH_FIT_DSV3、B2、B3 等） | 10 月 | 同上 | 128 / 8 / 0 | 同上 | sigmoid | bias（1e-3）+ seq aux 1e-4 | 见 §10.2 |
| PROD1_DPSK | 11 月 | 32K；micro-batch 1 条；GBS 64 条（2^21 token） | 128 / 8 / 1 | h → 3h/8 → h；第 0 层是 dense | sigmoid，× 2.5 | bias 1e-3 + seq aux 1e-4 | AdamH |
| PROD1_DPSK_LATENT2_E2X（简称 E2X） | 11 月 | 同上 | 512 / 16 / 1 | d → 3h/8 → d，d = h/2；前后各一个 h↔d 投影 | sigmoid，× 2.5 | bias 1e-3 + seq aux 1e-4 | AdamH（`--hyperp-router adam_hyperball`） |
| E2X + QB / QBX / QBK3 | 11-05 起 | 同 E2X | 同 E2X | 同 E2X | 同 E2X | 见 §7 | AdamH |

E2X 在 q1 的 router 相关参数（来自 `runs/scaling_ladder/prod_v1/qwen3_tpp200_prod1_dpsk_latent2_e2x-q1-iters15824-s1/argv.txt`）：

```
--num-experts 512 --moe-router-topk 16 --moe-ffn-hidden-size 288 --moe-latent-size 384
--moe-shared-expert-intermediate-size 288 --moe-router-score-function sigmoid
--moe-router-enable-expert-bias --moe-router-bias-update-rate 1e-3
--moe-router-load-balancing-type seq_aux_loss --moe-aux-loss-coeff 1e-4
--moe-router-topk-scaling-factor 2.5 --moe-router-dtype fp32 --hyperp-router adam_hyperball
```

各 arm 和 E2X 的差别：

| arm | argv 的差别 | 环境变量 / Megatron 代码 |
|---|---|---|
| BIASONLY | 去掉 seq aux | 原版 Megatron |
| QB | `--moe-router-load-balancing-type quantile_balancing --moe-aux-loss-coeff 0`，不开 expert bias | 原版 Megatron |
| QBX | 和 QB 完全相同 | `MOE_QB_EXACT=1`，`MOE_QB_EXACT_WINDOW=2.0`，`MOE_QB_EXACT_BINS=255`；代码用 `share/megatron-lm-80781ab0ab-qbx2` |
| QBK3 | `--moe-router-load-balancing-type none`，保留 expert bias；argv 里还写着 aux coeff 1e-4，但类型是 none，不生效 | `MOE_QBK3=1`，BINS 1000，EMA 0（写死在 exp 里）；代码用 `share/megatron-lm-80781ab0ab-qbk3`，每个 job 启动时核对补丁文件的 sha256 |

---

## 2. router 一步在做什么

```
x（token 隐状态，维度 h）──W_r──► z ∈ R^E（logit，fp32）
   打分：s = σ(z)（sigmoid router）或 softmax(z)
   选谁：S = top-k(选人分)，选人分因方法而异：
         DSV3 / QBK3：σ(z) + b；QB / QBX：z − β；softmax：softmax(z)
   权重：g_e = c · s_e / Σ_{j∈S} s_j，e ∈ S（sigmoid；c = 2.5）
   输出：y = shared(x) + Σ_{e∈S} g_e · Expert_e(x)
```

代码（`-qbx2` 树的 `moe_utils.py` L1053–1072，原版 80781ab0ab 在 L953–959 附近）：

```python
scores = torch.sigmoid(logits.float())                 # 全部 E 个 expert
# 选下标：expert bias 走 topk(scores + expert_bias)；QB 走 precomputed_indices = topk(logits - qb_beta)
scores = torch.gather(scores, 1, top_indices)          # 只取选中的 k 个（不含 bias）
probs  = scores / (scores.sum(-1, keepdim=True) + 1e-20) if topk > 1 else scores
probs  = probs * scaling_factor                        # 2.5
```

- k = 1 时不归一化，否则权重恒为 1，router 拿不到梯度。
- 例子：选中的 3 个 expert z = [2, 0, −1] → σ = [0.881, 0.500, 0.269] → 归一化 [0.534, 0.303, 0.163] → × 2.5 = [1.335, 0.758, 0.408]。如果在选中的里做 softmax，会是 [0.844, 0.114, 0.042]，比 sigmoid 尖锐得多。直接归一化 z 不可行：z 有正有负，和可能是 0。
- softmax router（老底座）：对 E 个 logit 做 softmax，取前 k 个，权重是这 k 个的 softmax 值（再归一化）。没有 bias，均衡只靠 aux loss。
- 历史沿革【记忆，未核对原文】：
  - DeepSeek-V2：softmax 打分，选出前 k 个后不重新归一化。
  - DeepSeek-V3：sigmoid，按 s + b 选，权重用不带 b 的 s 归一化，`routed_scaling_factor 2.5`，还有分组限制路由（n_group 8、topk_group 4），我们没开分组。
  - Kimi K3（据 `K3_QB_REFERENCE.md`）：896 个 expert 选 16 个，2 个 shared，`routed_scaling_factor 1.0`。

---

## 3. router 的梯度

选谁（top-k）和 bias / β 都没有梯度。对选中的 expert i，记 p = σ(z)，w_e = p_e / Σ_{j∈S} p_j，a_e = ⟨∂L/∂y, Expert_e(x)⟩（越负说明这个 expert 越有用），ā = Σ_{e∈S} w_e a_e：

**∂L/∂z_i = g_i · (1 − p_i) · (a_i − ā)**，没选中的 expert：∂L/∂z = 0。

推导：∂g_e/∂z_i = (δ_ei − w_e) · g_i (1 − p_i)，然后对 e 求和。

由此可以读出：

1. **只在选中的 k 个之间挪份量**：Σ_i g_i (a_i − ā) = 0。比加权平均更有用的 expert（a_i < ā），logit 被推高；更没用的被压低。
2. **排名靠间接变化**：expert 进出前 k 名，只能靠它被别的 token 选中时 W_r 里它那一行的更新，以及 x 本身的变化。冷门 expert 选不上，就拿不到梯度，所以需要不走梯度的均衡手段（bias、β）。
3. **饱和**：g_i (1 − p_i) ∝ σ'(z_i) = p(1 − p)。

   | z | σ(z) | σ'(z) | z 加 1 时 σ 增加 |
   |---:|---:|---:|---:|
   | 0 | 0.5 | 0.25 | 0.23 |
   | 2 | 0.88 | 0.10 | 0.07 |
   | 4 | 0.982 | 0.018 | 0.012 |
   | 6 | 0.9975 | 0.0025 | 0.002 |

4. **三条梯度路径**（看一个被选中的 expert）：
   - ① loss → expert 自己的参数：不乘 σ'，饱和时照常学；
   - ② loss → g → s → z → W_r：乘 σ'，饱和时被切断；
   - ③ loss → g → s → z → x：乘 σ'，饱和时被切断，但这条很细，x 还能从别处拿到梯度。

   aux loss 也走 ②，饱和时同样变弱。所以在 DSV3 里，"logit 失控"和"均衡全靠 bias 撑"会一起出现。
5. Adam 类优化器会把整体步长归一化，router 不会完全停下；但饱和的 (token, expert) 对几乎不影响更新方向。

---

## 4. 负载均衡方法总览

| 方法 | Megatron 开关 / 代码 | 机制 | 走梯度吗 | 统计范围 | 加在哪个空间 | 超参数 | 谁在用 |
|---|---|---|---|---|---|---|---|
| aux loss（Switch 式） | `aux_loss` | L = α·E·Σ f_e P_e | 是（经 P_e） | 一个 micro-batch（跨 TP×CP 汇总） | 梯度 → logit | α | 老 ladder softmax 底座（1e-3） |
| seq aux | `seq_aux_loss` | 同上，每条序列单独算再平均 | 是 | 每条序列（PROD1 里等于一个 micro-batch） | 同上 | α | DSV3、PROD1、E2X（1e-4） |
| global aux | `global_aux_loss` | 同上，f_e 跨 TP×DP×CP、并在一步的 micro-batch 间累计 | 是 | 整个 global batch | 同上 | α | Qwen3 官方 |
| DSV3 expert bias | `--moe-router-enable-expert-bias` | b_e ± γ·sign(平均 − 负载) | 否（buffer） | 整个 global batch 的计数 | sigmoid 分数之后 | γ（1e-3） | DSV3、PROD1、E2X |
| QB | `quantile_balancing` | β = 各 micro-batch 的 margin 分位数取平均，居中 | 否 | 每个 micro-batch 取分位数，再平均 | logit（sigmoid 之前） | 可选 EMA（默认 0） | E2X arm |
| QBX | QB + `MOE_QB_EXACT=1`（-qbx2） | β = 全局 margin 直方图的分位数 | 否 | 整个 global batch | logit | 窗口 ±2，255 格 | E2X arm |
| QBK3 | `none` + expert bias + `MOE_QBK3=1`（-qbk3） | b = 全局直方图的分位数，居中 | 否 | 整个 global batch | sigmoid 分数之后（和 DSV3 同一个 b） | 1000 格（K3 的值） | E2X arm；生产 |
| 负熵均衡损失 | Runlong 的补丁（TODO 9） | α·E·Σ p_e (log m_e + 1)，m = 概率的 EMA | 是 | EMA（η = 0.65） | 梯度 → logit | α、η | 实验中（§10.3） |

**统计范围越小，越压制专业化。** 例子：两条序列，一条全是代码、只用 expert 1–64，一条全是中文、只用 expert 65–128。合在一起数，128 个 expert 完全均衡，loss = 1，没有惩罚；每条序列单独数，每条只用了一半 expert，loss ≈ 2，有惩罚。所以 Qwen3 用 global batch 统计（Qiu et al. 2025，arXiv 2501.11873）。

---

## 5. aux loss 系列

- 公式：L_aux = α · Σ_e f_e · P_e。
  - f_e 是选中比例，乘 E/k 归一化，完全均衡时为 1。它是数出来的，**不可导**。
  - P_e 是平均分数，**可导**，梯度只经过它。
  - f_e 大的 expert，P_e 被往下压。
- 各卡只用自己的 token 算 P_e，所有卡的部分加起来才是完整的 loss（`switch_load_balancing_loss_func` 的注释）。
- seq aux 用来算 f_e 的 top-k 是**不加 bias** 的 top-k，和实际路由用的（加了 bias）不是同一个。P_e 用的是 sigmoid 分数在全部 E 个 expert 上归一化后的值，所以梯度会流到**所有** expert 的分数上，包括没选中的。
- aux loss 只更新 W_r（经 P_e），**不会**更新 expert bias。
- 统计范围随配置变化：

  | 配置 | 每次统计多少 token |
  |---|---|
  | 老 ladder `aux_loss`（8K） | q0–q3：4 条 × 8192 = 32,768；q4：16,384；q5、q6：8,192。也就是随 rung 从 32K 降到 8K，11-05 对照 MoE 手册筛查时已列为问题 |
  | 老 ladder `seq_aux` | 每条 8,192 |
  | PROD1 / E2X `seq_aux`（32K，micro-batch 1） | 每条 32,768，等于一个 micro-batch |
  | `aux_loss` 与 GBS 的关系 | 不跨卡、不跨 micro-batch。一步 2M token 被切成 64 个互不相关的 32K 小块 |

- 老 ladder 用 `aux_loss` 而不用 `global_aux_loss` 的原因：当时 fork 里 global aux 的实际系数是 1e-3 / DP，DP 不同的 rung 之间差 16 倍（`reports/qwen3_ladder_design_v1_2026-10-19.md`）。
- 文档 mask：老 ladder（8K）不加文档 mask，同一条序列里拼进去的几篇文档能互相看到；PROD1 加了（`--dataloader-inter-document-masking`）。同一个 micro-batch 里的几条序列在 attention 上互相看不到，只在 `aux_loss` 的统计里被放在一起。

### 5.1 举例：64 张 GB300 训练 Qwen3-235B-A22B（layout 是假设的）

- 配置：TP 2（开 SP）、PP 4、CP 1 → DP 8。每个 stage 16 张卡，EP 16（每卡每层 8 个 expert）。序列 8192，micro-batch 2 条，GBS 1024 条，一个 DP 副本一步要跑 64 个 micro-batch。
- 各方法在一步里的统计范围：
  - `aux_loss`：f_e 在 TP 的两张卡之间汇总，覆盖一个 micro-batch 的 16,384 个 token，每个 micro-batch 一个 loss。一步里有 8 × 64 = 512 个互不相关的小块。
  - `seq_aux`：每条序列 8,192 个 token，一步里 1024 个 loss。
  - `global_aux`：f_e 在本 stage 的 16 张卡（TP × DP）之间汇总，并在一步的 64 个 micro-batch 之间累计平均。到最后一个 micro-batch 时，覆盖整个 global batch 的 8.4M token。
  - DSV3 bias：每张卡把 64 个 micro-batch 的计数累加起来，步末在本 stage 的 TP × DP × CP 之间求和，覆盖 8.4M token。
- 另外几点：
  - PP 下所有汇总只在本 stage 内做；
  - EP 不参与这些统计（router 不按 expert 切开，每张卡都有完整的 W_r 和 bias）；
  - TP 的两张卡上 bias 永远一样。

---

## 6. DSV3 expert bias

- 是 buffer，不是参数：`router.py` L239–245 `register_buffer('expert_bias', zeros(E, float32))`。任何 loss 都改不了它。
- 更新规则：
  1. 每个 micro-batch 累加各 expert 分到的 token 数（`local_tokens_per_expert`）；
  2. 步末在 TP × DP × CP 之间求和（`finalize_model_grads._update_router_expert_bias` L377 → `moe_utils.get_updated_expert_bias` L1245）；
  3. b_e += γ · sign(平均 − 负载)，γ = 1e-3；
  4. 每层各自更新，和优化器无关。
- 只有相对大小有用：所有 b 一起加同一个常数，不改变路由。
- **|b| 上限为什么是 1**：s ∈ (0, 1)。两个 expert 的 b 差一旦超过 1，b 低的那个对任何 token 都赢不了，排名和内容无关了。例：A 的 s = 0.9、b = 0；B 的 s = 0.1、b = +0.85。B 的选人分 0.95 > 0.9，选了 B，虽然 router 更想要 A。
- **要看极差，不看绝对值**：例如 DSV3 q3 第 0 层的 b 都在 +11.2 到 +11.7 之间，|b| 很大，但极差只有 0.5。
- **整体漂移**：按符号更新不是零和的。多数 expert 略低于平均、少数远高于平均时，加的总量大于减的总量，整层一起往上漂。
- **病态**：b 远大于 1 时，选人几乎由 b 决定，分数接近 0 的 expert 也会被选上，选上后权重又接近 0，等于浪费名额。这时 MaxVio 看起来很好，其实是 bias 硬撑。
- **固定步长的局限**：E2X 底座的 |b| 最大值 q1 1.73 → q2 4.19 → q3 7.03（含整体漂移）。11-05 手册筛查 2.4 节也指出，固定步长平衡不了第 0 层。QBK3 解决的就是这个问题（§7.3）。

---

## 7. Quantile Balancing（QB / QBX / QBK3）

### 7.1 QB（Megatron 自带）

记号（每层各自做）：z_te 是 router logit；k 是每个 token 选几个；E 是 expert 数；T 是一个 micro-batch 的 token 数；C = T·k/E 是每个 expert 的目标 token 数。

1. **路由**：每个 token 选 z_te − β_e 最大的 k 个，β 用上一个 step 算好的值。
2. **α_t**：同一排序里第 k+1 名的值，也就是落选的选项里最好的那个。
3. **β_local**：每个 expert 在这个 micro-batch 的 T 个 token 上取 margin m_te = z_te − α_t 的第 C+1 大值。
4. **步末**：先对 micro-batch 取平均，再在 DP rank 之间取平均，然后减去 E 个的均值（居中），写回 β。可选 EMA，默认系数 0，我们没用。评测时 β 固定。

代码（`-qbx2` 树）：`moe_utils.qb_dual_update` L220–255（`alpha = topk(scores - beta, k+1).values[:, -1:]`，`beta_local = (scores - alpha).topk(C+1, dim=0).values[-1]`）；`router.py` L280–286 注册 `qb_beta`（形状 [E]）；`finalize_model_grads._update_router_qb_beta` L450（all-reduce 平均 → 可选 EMA → 居中）。原版 Megatron 里的 QB 和这里一样。

粒度：

| | β_e | α_t |
|---|---|---|
| 有几个 | 每层 × 每个 expert（E2X 每层 512 个） | 每层 × 每个 token |
| 什么时候算 | 每个 global batch（= 每个 step）一次 | 每次 forward |
| 存不存 | 存（buffer，进 checkpoint） | 不存 |
| 用途 | 路由 | 只用来算新 β |
| 评测时 | 固定 | 不用 |

**为什么取的是 margin 而不是原始 z**：margin = "这个 token 想要 e，比它落选的头号选项高出多少"。例：3 个 expert、选 1 个。token 1 的 z = (A 0.9, B 1.5, C 1.2)，α = 1.2，A 的 margin = −0.3；token 2 的 z = (0.6, 0.0, −0.2)，α = 0.0，A 的 margin = +0.6。按原始 z，token 1 对 A 更积极；按 margin，token 2 才是 A 的真正候选，token 1 有更好的选项，根本不会来抢 A。

**单个 expert 的门槛（编的数，C = 2，8 个 token）**：margin 为 0.40, 0.31, 0.12, 0.08, 0.02, −0.05, −0.11, −0.20。
- 旧 β = 0 时，5 个 token 选它，过载；
- 新 β = 第 3 大 = 0.12，只有 2 个 token 过线（等于门槛的不算）。
- 热门 expert 的 β 上涨，冷门 expert 的 β 下降。

**"正好 C 个"只是近似**：
- 对选中的 expert，判据（margin > β）在其他 β 不变时是精确的；对没选中的 expert，要挤掉的是第 k 名，比第 k+1 名的 α 更高，判据偏宽。
- 512 个 β 一起变，α 也跟着变。
- 新 β 用在下一批 token 上。
- 这是对偶问题上的块坐标下降：α 是 token 侧的对偶变量，β 是 expert 侧的。同一批数据上，一步不一定收敛，在 8×4 的玩具例子里会卡住，欠载的 expert 一直只收到 1 个 token【模拟】。连续分布、每步换一批新 token 的模拟（64 个 expert 选 4 个，高斯 logit）：MaxVio 第一步从 8.8 降到 0.44，之后稳定在 0.15–0.2【模拟】。
- 附带现象：每个 token 的第 k+1 名 expert，margin 正好等于它自己的 β，所以每个 expert 都有一部分 token 堆在门槛上（约 m/E）。这是 K3 的 review 里说的 tie spike，会让 MaxVio 留下 0.03–0.18 的底【核对：REVIEW_REPORT.md】。

**我们的配置**：E2X 是 E = 512、k = 16，micro-batch 是 1 条 32K 序列，所以 T = 32,768、C = 1,024，也就是每条序列取第 1,025 大的 margin。一个 global batch 64 条，β 每个 step 更新一次。

**限制**：必须关掉 aux loss；不支持 router fusion 和分组路由；T·k 必须能被 E 整除。TP > 1 时，先在 TP 组内 all-gather logit 再算（`router.py` 里的 `gather_group`）；CP 的情况未核对。

### 7.2 QBX（全局精确分位数，我们的补丁）

- 区别只在第 4 步：QB 用"各 micro-batch 分位数的平均"，QBX 用"合并后的分位数"。每张卡把每个 micro-batch 的 margin 按 expert 记进直方图，跨 micro-batch 累加，再跨 DP × CP 求和，取全局第 C+1 大（这时 C = 全局 token 数 × k / E），然后居中，下一步生效。
- **为什么平均不等于全局**（C = 1/micro-batch，全局 C = 2）：
  - micro-batch 1 的 margin 是 0.9, 0.8, 0.7, 0.6，本地第 2 大是 0.8；micro-batch 2 是 0.2, 0.1, 0.0, −0.1，本地第 2 大是 0.1。
  - QB：β = 0.45，8 个 token 里有 4 个过线，是目标的两倍。
  - QBX：8 个合起来取第 3 大，β = 0.7，正好 2 个过线。
  - 序列之间对同一个 expert 的偏好差得越多，QB 偏得越厉害。我们一个 micro-batch 就是一条序列，代码、中文、数学之间差别很大。
- 设置：窗口是中心 ± 2.0 logit，255 格（每格约 0.016）。格数取奇数，因为每个 token 都有一个 margin 正好落在中心上，偶数格时它落在格子边界，读数会系统性偏高约 0.3 格；256 格的 run 已在 11-06 删掉重跑。落在窗口外的 expert 退回用平均值，并截到窗口边界；q2 结束时每步有 4–16 个 (router, expert) 组合落在窗口外（共 11,776 个）。
- 要求 TP = CP = 1；开 full recompute 时 router 会算两次，计数翻倍，但分位数基本不变。
- 代码：`share/megatron-lm-80781ab0ab-qbx2`（改了 3 个文件），`share/megatron_qbx_tests/`（README、`qbx2.patch`、测试）。

### 7.3 QBK3（完全按 Kimi K3）

来源：K3 技术报告（arXiv 2607.24653v2）2.3.3 节公式 13–14、附录 C / D，以及 HF 上 `moonshotai/Kimi-K3` 的 `KimiMoEGate`（只有推理代码，没有公开训练代码）。核对记录见 `share/megatron_qbx_tests/K3_QB_REFERENCE.md`。

- **路由不变**：仍是 DSV3 的形式，s = σ(z)，S = top-k(s + b)，权重 c·s/Σs 不含 b。这就是 Megatron 原本的 expert-bias 路径。
- **只换 b 的更新**（每个 step 一次，覆盖整个 global batch 的全部 token）：
  - α_i = (s_i + b) 里第 k+1 大的值；
  - r_ij = α_i − s_ij，表示"expert j 要挤进 token i 的前 k 名还差多少 bias"：b_j > r_ij 时 j 就会被选中；
  - b̂_j = 第 ⌈q⌉ 小的 r，q = m·k/E，m 是这一步的全部 token 数；
  - b = b̂ − mean(b̂)，下一步生效。没有步长，默认也不用 EMA。
- **直方图范围天然有界**：s ∈ (0, 1)，所以 r 一定落在 [b_min − 1, b_max + 1] 内。每步按当前的 b 重新算范围，分 1000 格（K3 用的值，误差只有千分之几），格内线性插值。不存在落在范围外、需要退回的情况。
- **跨卡**：每张卡只记自己的 token，每层每步做一次 int32 的 E × 1000 求和 all-reduce（TP × DP × CP）。不管 token 怎么切分，合出来的直方图都逐位相同，所以 TP、CP 都能开；生产代码里在 CPU 上验证过 CP 为 1、2、4 时逐位一致。
- **代价**（review）：约 1% 的 step 时间；常驻显存多 33 MiB（q1）/ 45 MiB（q2）；峰值显存不变。
- **代码**（`-qbk3` 树）：
  - `moe_utils.py`：`qbk3_range` L1834，`qbk3_accumulate_histogram` L1844，`qbk3_bias_from_histogram` L1882，`qbk3_observe_microbatch` L1906；
  - `finalize_model_grads._update_router_expert_bias_qbk3` L449；
  - 补丁 `share/megatron_qbk3_tests/qbk3.patch`，22 个 agent 的 review 见 `REVIEW_REPORT.md`，结论是没有严重或重要的缺陷。
- **启动瞬态**：没有 LR warmup 时，开头几十步 MaxVio 约 11–15。review 认为这是 router 在漂移，不是估计方法的问题。所以另开了 QBK3_WU（1% warmup，只跑 q1–q2）；它和 QBK3 的 loss 差里混有 warmup 本身的效果。
- **合成测试**（router 正弦漂移，40 步，从 b = 0 开始）：第 8 步以后的 MaxVio，QBK3 是 0.20，按符号步长 1e-3 更新的是 5.6。因为固定步长从 0 开始本来就追不上，这个数只能说明 QBK3 反应快，不代表长时间训练里的差距【核对：test_qbk3_unit.py】。
- **和 K3 原版的不同**：我们只搬了均衡方法，结构还是 E2X（512 选 16，1 个 shared，权重 × 2.5）。

### 7.4 三者对比

| | QB | QBX | QBK3 |
|---|---|---|---|
| 选人分 | z − β（logit） | z − β（logit） | σ(z) + b（sigmoid 分数） |
| 直方图 / 分位数取在 | z − α | z − α | α − σ(z) |
| 合并方式 | 每个 micro-batch 各取分位数 → 平均 | 全局直方图 → 一个分位数 | 全局直方图 → 一个分位数 |
| 跨卡同步什么 | E 个分位数（all-reduce 平均） | 直方图 E × 257 | 直方图 E × 1000 |
| 直方图范围 | 不用直方图 | 中心 ± 2 logit，超出要退回 | [b_min − 1, b_max + 1]，精确覆盖 |
| TP / CP | 支持 TP（先 all-gather logit），CP 未核对 | 必须都是 1 | 不限 |
| 推理 | 要自己写 z − β 的路由 | 同左 | 就是 DSV3 的 `e_score_correction_bias`，vLLM / SGLang 已经支持 |
| 和 K3 的关系 | 不是 | 只借了合并方式 | 是 |

**logit 空间和 sigmoid 空间挤走的 token 不同**（只看 A 和 B，每个 token 选 1 个，要压 A）：

| | z_A | z_B | σ_A | σ_B | QBX：β_A = 2.5 | QBK3：b_A = −0.2 |
|---|---|---|---|---|---|---|
| token 1 | 6 | 3 | 0.998 | 0.953 | 3.5 > 3 → 留在 A | 0.798 < 0.953 → 改去 B |
| token 2 | 0 | −2 | 0.500 | 0.119 | −2.5 < −2 → 改去 B | 0.300 > 0.119 → 留在 A |

- logit 空间按 logit 差来挤，先挤走差距小的 token 2；sigmoid 空间按分数差来挤，先挤走两个分数都接近 1 的 token 1。
- 【推断】后者顺带的好处：两个分数都接近 1 时，归一化后的权重本来就差不多，换 expert 对输出影响小。没有实验验证过哪种更好。

### 7.5 现状（截至 11-09 / 11-10）

- 生产：E2X + QBK3（生产代码 `share/megatron-lm-ultra-prod-qbk3-*`，分支 ultra-prod）。59B-A4B 默认 `MOE_BALANCING=qbk3`（要求 latent MoE），461B 和 mock 脚本仍默认 dsv3。
- PROD_LR_FIT：41 个 run，全部是 E2X + QBK3、CE，在生产配置上重新标定 LR 公式；用的是 Boxiang 的评测，结果不在我们的报告里。
- ladder：QBK3 跑 q1–q4，QBK3_WU 跑 q1–q2，评测由 tmux `rt_autoeval` 自动提交。

---

## 8. router 的优化器与 LR

- 两种选择：
  - AdamH：Adam 更新，但把整个 W_r 投回固定半径的球面（hyperball），范数不会变大。这是 MuonH 主体的默认。
  - AdamW：普通 AdamW，范数可以自由变化。
- **hyperball 挡住了 logit 变大**：z = W_r · x，x 前面有 norm。在 E2X 的 arm 里：
  - 所有 AdamH 的 arm（不管用哪种均衡），最大 |logit| 在 q2 都是 16–18，pinned 不超过 0.2%；
  - 换成 AdamW（RTADAMW，LR × 0.19）后，最大 |logit| 在 q1 是 37.5、q2 是 65.2，pinned 3.85% / 4.26%；
  - 每一行单独做 hyperball（RTPIN）约束更紧，最大 |logit| 最低（q1 11.3，q2 16.1）。
- 老 ladder 的探针：B1（AdamW gate、LR × 1）的分数饱和比例在 300 步内只从 97% 降到 67–69%，所以降到 × 0.19。
- **宽模型上 router 会崩**：L24 h4096 的格子上，MuonH + softmax router 在 h ≥ 3072 时崩掉，第 50 步 LBL 就到 15（上限 E/k = 16）。把 gate 换成 AdamW 没用；router LR 降到 × 0.25 修好了。Runlong 的候选规则是 router LR 倍数 = min(1, 2048 / h)。PROD_LR_FIT 里有 router LR × 0.25 / 0.5 / 1 在 L12H2048 和 L12H4096 上的对比。
- 11-05 手册筛查 2.3 节：MuonH 的 router LR 不随宽度缩放。
- 老 ladder 10 月消融的结论（§10.2）：DSV3 + AdamH + LR × 1 时 bias 一路涨（病态 A）；LR × 0.19 时健康（B3）；AdamW gate 时分数卡在 1（B2）。

---

## 9. 健康指标（Boxiang 的 `router_health.py`，每 100 步一次，`LADDER_ROUTER_HEALTH=100`）

| 指标 | 量什么 | 统计范围 | 健康标准 | softmax 能看吗 |
|---|---|---|---|---|
| MaxVio（层平均） | 最忙的 expert ÷ 平均 − 1，所有层取平均 | 一整个 step（所有 micro-batch、所有 rank） | 逐渐下降 | 能 |
| MaxVio（最差层） | 同上，取最差的一层 | 同上 | 逐渐下降 | 能 |
| dead expert (%) | 负载低于平均 10%（`DEAD_FRAC = 0.10`）的 expert 占（层数 × E）的比例 | 同上 | 0 | 能 |
| \|bias\| 最大 | 所有层、所有 expert 的 \|b\| 最大值（QB / QBX 的 β 不统计） | buffer | < 1，看同层极差 | 不适用 |
| logit 饱和 (%) | \|z\| > 4 的比例，包括 z < −4 | rank 0 的第一个 micro-batch | 不是判据 | 只作数量级参考 |
| z > +4 (%) | 只算正向 | 同上 | < 5% | 无意义 |
| top-8 pinned (%) | 每个 token **最大的 8 个原始 logit** 里 z > 4 的比例（E2X 选 16 个，也只看 8 个；不加 bias） | 同上 | < 5% | 无意义 |
| 最大 \|logit\| | 所有层里最大的 \|z\| | 同上 | 看趋势 | 能（看趋势） |
| gate 更新幅度 | ‖ΔW_r‖_F / ‖W_r‖_F，各层平均 | 一个 step | 只作参考 | 能 |

要点：
- **MaxVio**：一个 step 2^21 个 token。128 选 8 时平均每个 expert 131,072 个，512 选 16 时 65,536 个。0.17 表示最忙的 expert 分到约 1.17 倍。EP 时所有卡都要等最忙的那张，它同时决定速度和显存。最差层要单独看，问题常常藏在第 0 层。
- **dead**：负载低于平均的 10%。拿不到梯度就不会变好，会越来越冷。换算成个数更直观：老 q0 是 12 层 × 128 = 1,536 个，0.0651% = 1 个；E2X q2 按 24 × 512 = 12,288 算，0.17% ≈ 21 个。
- **pinned ≈ 16 × (z > +4)**（128 选 8 时，分母 8 对 128）：实测 0.55% × 16 ≈ 8.9% 对得上。被钉住的几个 σ 都约等于 1，归一化后权重也几乎一样，router 既分不出高低，梯度也传不进去。
- **"硬分类器"**：指 router 对每个 token 都极其确定，不是分不了类。坏处有三个：早期的分配被锁死；选中的几个权重失去区分度；DSV3 里选谁变成由 bias 决定。
- **最大 |logit|**：只是一个极值，正负都算，只能看趋势。被选中的是否变硬，要看 z > +4 和 pinned。
- **gate 更新幅度**：在 bf16 副本上量的，训练末期每步的变化比 bf16 精度还小，读数会被量化【推断】。初期在 MuonH 的 LR 下约 0.8，LR × 0.19 时约 0.16。
- **Boxiang 预设的判据**（`dsv3/DESIGN.md` 第 3 节）：z > +4 和 pinned < 5%；dead = 0；|b| < 1，且大致按每步 γ 增长；MaxVio 逐渐下降；gate 更新幅度只作参考。

---

## 10. 实验结果

### 10.1 E2X 上的负载均衡（11 月，512 选 16，seed 1）

overall_tsqw，8K 窗口，Δ 相对 E2X（32K 窗口的数字几乎一样）：

| arm | q1 | q2 | q3 |
|---|---|---|---|
| BIASONLY（去掉 seq aux） | −0.0012 | −0.0035 | 未出 |
| QBK3 | −0.0008 | −0.0035 | −0.0004 |
| QB | +0.0002 | −0.0035 | +0.0003 |
| QBK3_WU | +0.0002 | — | — |
| QBX | +0.0007 | −0.0029 | −0.0002 |
| bias 更新率 × 0.5 | +0.0007 | −0.0006 | — |
| RTPIN | +0.0013 | −0.0011 | 已停（Runlong 11-07） |
| bias 更新率 × 0.25 | +0.0019 | −0.0007 | — |
| RTADAMW | +0.0055 | +0.0060 | 已停 |

训练最后一步的 router health（顺序是 q1 / q2 / q3）：

| arm | MaxVio 层平均 | MaxVio 最差层 | dead | \|b\| 最大 | logit 饱和 | pinned | 最大 \|logit\| |
|---|---|---|---|---|---|---|---|
| E2X | 0.60 / 0.44 / 0.63 | 0.88 / 0.81 / 1.16 | 0 / 0 / 0 | 1.73 / 4.19 / 7.03 | 58 / 51 / 47% | 0.05 / 0.17 / 0.54% | 13.9 / 17.1 / 25.5 |
| BIASONLY | 0.65 / 0.57 / — | 1.31 / 1.22 / — | 0 / 0 / — | 2.79 / 3.41 / — | 75 / 70 / —% | 0.03 / 0.19 / —% | 13.0 / 17.6 / — |
| QB | 1.46 / 1.43 / 1.39 | 3.10 / 2.53 / 3.20 | 0 / 0 / 0 | 不统计 | 74 / 69 / 72% | 0.04 / 0.20 / 0.83% | 13.7 / 16.3 / 19.1 |
| QBX | 0.73 / 0.85 / 0.73 | 1.14 / 2.35 / 1.26 | 0 / 0.17 / 0.09% | 不统计 | 74 / 70 / 73% | 0.03 / 0.19 / 0.78% | 13.1 / 15.8 / 28.0 |
| QBK3 | 0.77 / 0.72 / 0.70 | 1.47 / 1.08 / 1.22 | 0 / 0.17 / 0.08% | **0.073 / 0.096 / 0.102** | 74 / 70 / 72% | 0.04 / 0.20 / 0.77% | 12.9 / 16.8 / 21.6 |
| RTADAMW | 2.08 / 2.49 / — | 4.64 / 8.37 / — | 0 / 0.10 / — | 2.21 / 8.24 / — | 83 / 79 / —% | **3.85 / 4.26** / —% | **37.5 / 65.2** / — |

怎么读：
1. **loss 上分不出高低**：BIASONLY、QB、QBX、QBK3 在 q2 都是 −0.003 左右，到 q3 都在 ±0.0005 以内。q2 那 −0.003 是"没有 seq aux"带来的，四者共有。
2. **QB 的 MaxVio 后期反而更差**：前期 QB 和 QBX 明显更均衡（q1 step 3,291：E2X 2.45、QB 1.26、QBX 0.82），后期 E2X 的 bias 越累积越准，降到 0.4–0.6；QB 每步重新算、不累积，又用的是平均的分位数，停在 1.4 左右；QBX 和 QBK3 在 0.7 左右。
3. **QBK3 的 |b| 一直很小**（≤ 0.10），E2X 底座的 |b| 却涨到 7.0，说明固定步长的 bias 会一路漂。
4. **QBX 和 QBK3 在 q2 / q3 有约 12–21 个 dead expert**（按 层数 × 512 换算），E2X 和 QB 是 0。原因还没查【未查】。
5. **去掉 seq aux（BIASONLY、QB、QBX、QBK3）后，logit 饱和从约 50% 涨到约 70%**，而 z > +4 仍然几乎为 0，所以涨的全是 z < −4，无害。【推断】seq aux 的 P_e 在全部 E 个 expert 上归一化，梯度会流到没选中的 expert，把很负的 logit 往回拉一点；去掉以后没有东西拦着它们继续变负。
6. **pinned 只有 RTADAMW 超标**，和用哪种均衡方法无关，见 §8 的 hyperball。

### 10.2 老 ladder 的 router 消融（10 月，128 选 8，8K）

对照 ① MUONH_FIT（softmax + aux 1e-3，AdamH gate，router LR × 1）；对照 ② BASE（softmax + aux 1e-3，AdamW gate，LR × 1）。q0 是 3 个 seed 的平均。

| 组 | router | gate | router LR | Δ | 健康 |
|---|---|---|---|---|---|
| A MUONH_FIT_DSV3 | DSV3 | AdamH | × 1 | 相对 ①：q0 +0.0023，q1 −0.0024，q2 +0.0018，q3 +0.0017 | 病态：|b| 从 q0 到 q3 是 1.0 → 3.2 → 5.9 → 12，最大 logit 26 → 184 |
| B2 | DSV3 | AdamW | × 0.19 | 相对 ①：q0 +0.0051，q1 +0.0020 | 病态：pinned 8.9–10.9%，|b| 1.1–3.5，最大 logit 54–153 |
| B3 | DSV3 | AdamH | × 0.19 | 相对 ①：q0 +0.0074 | 最健康：MaxVio 0.13–0.16，pinned 0.8–2.2%，|b| 0.31–0.55 |
| C BASE_DSV3_ONETREE | DSV3 | AdamW | × 1 | 相对 ②：q0 −0.0012，q1 +0.0054，q2 +0.0022 | 正常 |

其他读数（训练结束时）：
- softmax 主线 MUONH_FIT：MaxVio 1.1–1.6，最差层 3.3–3.9，1 个 dead。aux loss 没把均衡压紧，这就是 11-05 手册筛查说的"主线负载均衡没人监控"。
- DSV3 去掉 seq aux（q0）：最差层 4.2，13 个 dead。
- DSV3 + MuonH gate（q0）：全部达标。

结论：
- DSV3 和 softmax 在 loss 上没有实质差别，都在 ±0.003 以内。
- 两种病态的原因不同：A 是 router LR 太大，B2 是 AdamW gate 本身的问题。
- 健康的 B3 在 q0 的 loss 反而差约 0.007。
- MuonH 相对 AdamW 的领先，换不换 router 只变了约 0.004。
- 换 router 后，新 LR 公式仍然接近最优（DSV3 下最优倍数约 1.05）。

### 10.3 负熵均衡损失（Runlong TODO 9，arXiv 2605.15403）

- 公式：每层维护 router 概率的 EMA m（η = 0.65），损失 = α · E · Σ_e p_e (log m_e + 1)。
- 和 Switch 式 aux loss 比：接近均衡时"刚度"是它的 E 倍，并且对快要饿死的 expert 有一个 log 屏障。
- q0 第一阶段（单 seed）：DSV3 底座保留 bias，α = 1e-5 / 1e-4 时分别是 −0.0017 / −0.0015；softmax 底座按论文原样，α = 1e-5 / 1e-4 时分别是 −0.0005 / +0.0029。都在噪声范围内。后续在 SWS 底座上有一组 α 扫描（见报告的 SWS 一节），这里没有整理。

---

## 11. 常见误解（我们讨论中出现过的）

| 说法 | 对不对 | 正确说法 |
|---|---|---|
| "QB 的门槛是整层一个" / "每个 token 一个门槛" | 都不对 | β 每个 expert 一个（每层 E 个），α 每个 token 一个，只是中间量 |
| "QB 不需要全局同步" | 不对 | β 必须在所有卡上相同，QB 也做 all-reduce；区别在同步的是分位数还是直方图 |
| "QBX 就是 K3" | 不对 | QBX 只借了 K3 的全局合并，打分空间仍是 Megatron QB 的 logit 空间 |
| "QBX 和 QBK3 一样" | 一半对 | 合并方式一样；选人分的空间不同，挤走的 token 也不同（§7.4 的例子） |
| "权重是把 z 归一化" | 不对 | 先过 sigmoid，再在选中的 k 个里归一化，再乘 2.5 |
| "权重归一化成 Σσ(z) = 1，对全部 expert" | 不对 | 只对选中的 k 个；乘 2.5 后总和是 2.5 |
| "QB 没有 aux loss 却让 pinned 变好" | 前提不成立 | QB 的 pinned 和 E2X 一样；pinned 由 hyperball 挡住 |
| "QB 的负载均衡一直比底座好" | 只有前期对 | 后期 E2X 0.4–0.6，QB 约 1.4 |
| "z < −4 的饱和是坏事" | 不对 | 只是明确不选；坏的是 z > +4 / pinned |
| "|b| 大就是病态" | 不一定 | 看同层极差；整层一起漂不影响路由 |
| "我们的 micro-batch 会变" | PROD1 系列不会 | q0–q5 都是 1 条 32K 序列；大 rung 的卡数正好把 GBS 64 占满 |

---

## 12. 来源与代码位置

路径相对 `/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/`：

| 内容 | 位置 |
|---|---|
| 健康指标定义 | `Project/cosmos_llm_scaling_ladder/code/scaling_ladder/router_health.py`（`DEAD_FRAC` L55，`logit_stats` L69，调用处 L213） |
| 原版 Megatron（expert bias、aux loss） | `Project/cosmos_llm_scaling_ladder/code/megatron-lm-80781ab0ab/megatron/core/`（`moe_utils.py` L953–959 选人，L1245 `get_updated_expert_bias`；`finalize_model_grads.py` L377；`router.py` L239–245） |
| QB / QBX 代码 | `share/megatron-lm-80781ab0ab-qbx2/`（`moe_utils.qb_dual_update` L220–255，`router.py` L280–286，`finalize_model_grads._update_router_qb_beta` L450）；`share/megatron_qbx_tests/`（README、`qbx2.patch`、测试） |
| K3 原文核对 | `share/megatron_qbx_tests/K3_QB_REFERENCE.md` |
| QBK3 代码 | `share/megatron-lm-80781ab0ab-qbk3/`（`moe_utils.py` L1834–1906，`finalize_model_grads.py` L449）；`share/megatron_qbk3_tests/`（`qbk3.patch`、`REVIEW_REPORT.md`、`test_qbk3_unit.py`、`test_qbk3_dist.py`） |
| 生产代码 | `share/megatron-lm-ultra-prod-qbk3-7d7b376f`、`-c7f5e964` |
| Runlong 的实验日志 | `cosmos_llm_scaling_ladder` 仓库分支 `origin/runlong/prod-xx`：`prod_xx/NOTES.md`（11-07 "QBK3" 一节） |
| 实验定义 | `share/cosmos_llm_scaling_ladder_wt_prod/code/scaling_ladder/prod/exp/qwen3_tpp200_prod1_dpsk_latent2_e2x*.sh` |
| run 目录与 argv | `runs/scaling_ladder/prod_v1/qwen3_tpp200_prod1_dpsk_latent2_e2x{,_qb,_qbx,_qbk3,_qbk3_wu,…}-q{1..4}-iters*-s1/argv.txt` |
| 评测与 router health 数字 | `Project/cosmos_llm_scaling_ladder/reports_xx/qwen3_scaling_ladder_eval.md`："E2X 上的 router / 负载均衡" 一节（含 QB 算法说明）和它的 "训练中的负载均衡" 子节 |
| 10 月消融 | `cosmos_llm_scaling_ladder/reports/dsv3_router_experiment_design_2026-10-26.md`，`DECISIONS.md` 10-26 |
| 手册对照筛查 | `Project/trichord/reports/dense_pretrain/2026-11-05_moe_handbook_audit.md` |
| QB 示意图（独立网页） | https://claude.ai/artifact/QdAcmHVaUceC5NhJo5pdm2 |
