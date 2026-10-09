# Muon / MuonH 学习笔记

> 本文讲 Muon 和 MuonH 两个优化器本身：更新规则、"更新有多大"、两者的 lr 有什么不同。
> 建立在 MuonH 之上的超参迁移律（HyperP：深度、token 数、MoE 粒度怎么缩放 lr）见 [HyperP.md](HyperP.md)。
> 文中不加下标的 ‖·‖ 一律指 **Frobenius 范数**（见 §0）。

---

## 核心结论：Muon 和 MuonH 里的 lr 是两个东西

| | Muon | MuonH |
|---|---|---|
| lr 控制的是 | **绝对步长** ‖ΔW‖ | **相对改变比例** ‖ΔW‖/‖W‖ |
| 每步相对改变 | lr · √min(m, n) / ‖W‖，随层形状、训练进度变化 | **恒等于 lr** |
| 各层能否共用同一个 lr | 不能直接共用，需要形状缩放项 / μP 逐层倍率 | 能 |

所以同一个数值（比如 0.016）在 Muon 里和在 MuonH 里含义不同，不能直接搬用。下面 §1–§5 解释原因。

---

## 0. 范数约定

‖W‖、‖U‖、‖ΔW‖ 都是 **Frobenius 范数**：‖W‖_F = √(所有元素平方和)，即把矩阵摊平成一个长向量后的长度。
- MuonH 用的是 **Frobenius 球面**：约束 ‖W‖_F = 常数（初始化时的值）。
- 相对更新也按 F 范数衡量：‖ΔW‖_F / ‖W‖_F。
- 代码（`dpl/megatron/_hyperball.py`）里的 `‖X‖`、`‖U‖` 都是对整个矩阵求 F 范数。
- 另有工作（如 Bernstein 的 modular norm）主张用谱范数，但 HyperP 论文明确选 Frobenius。

---

## 1. Muon 的更新规则

对一个隐藏层矩阵 W（形状 m×n）：

```
G = 动量梯度
U = NS(G)            # Newton-Schulz 迭代，把 G 的奇异值都压成 ≈1
W = W - lr * U       # W 的长度不受约束，可以自由变化
```

**NS 不负责球面。** NS 只是把更新方向"整形"，球面约束是 MuonH 额外加的两步（见 §5）。

---

## 2. NS(G) 有多大？

NS 把 G 的所有奇异值都压成 ≈1，只保留方向。因此：
- **和梯度大小无关**：NS(G) ≈ NS(100·G)。
- **和矩阵形状有关**：奇异值个数 = min(m, n)，每个都 ≈1。

NS 输出 U（m×n）的"大小"，换不同范数衡量，结论不同：

| 衡量方式 | 值 | 矩阵变大时 |
|---|---|---|
| 谱范数（最大奇异值） | ≈ 1 | 不变 |
| Frobenius 范数 | ≈ √min(m, n) | 变大 |
| 每个元素的平均大小（RMS） | ≈ 1/√max(m, n) | 变小 |

例：2×2 的 NS 输出 ‖U‖_F ≈ √2；1024×1024 的 ‖U‖_F ≈ √1024 = 32。

---

## 3. Muon 的 lr：管住了"走多远"，没管住"相对改变多少"

绝对步长 ‖ΔW‖ = lr · ‖U‖ ≈ lr · √min(m, n)，确实由 lr（加上形状）决定。

但模型行为取决于的是**相对自身改变了多少**，即 ‖ΔW‖ / ‖W‖。类比：同样走 1 米，对 10 米长的东西是 10%，对 100 米长的东西只有 1%。

```
相对改变 = ‖ΔW‖ / ‖W‖ = lr · √min(m, n) / ‖W‖
```

这个比值不固定，有两层原因。

### 原因 1：初始化时不同形状的层就已经不同

W 为 d_out × d_in，每个元素 std = 1/√d_in 时，‖W‖ ≈ √(d_out · d_in / d_in) = **√d_out**。所以：

```
初始化时的相对改变 ≈ lr · √min(d_out, d_in) / √d_out
```

以 h = 1024 为例：

| 层 | 形状 (d_out × d_in) | √min | √d_out | 相对改变 |
|---|---|---|---|---|
| attention proj | 1024 × 1024 | 32 | 32 | **lr** |
| MLP fc1（升维） | 4096 × 1024 | 32 | 64 | **lr / 2** |
| MLP fc2（降维） | 1024 × 4096 | 32 | 32 | **lr** |

同一个 lr 下，fc1 每步的相对改变只有其他层的一半。方阵恰好等于 lr，非方阵就会偏。

### 原因 2：训练中 ‖W‖ 会漂移

W 变长，分母变大，同一个 lr 的实际效果变小；W 变短则相反（weight decay 也会影响 ‖W‖）。普通 Muon 不约束 ‖W‖，所以这个比值在整个训练过程中都在变。

---

## 4. 实际 Muon 实现里的形状缩放项

实际实现通常会给 U 再乘一个按形状算的系数，例如 √max(1, m/n) 或 0.2·√max(m, n)（后者是为了让更新的 RMS 和 AdamW 对齐）。

这些系数修正的是**原因 1**（形状带来的差异），修正不了**原因 2**（训练中 ‖W‖ 的漂移）。

μP 的逐层 LR 倍率（如 √(d_out/d_in)）也是在处理同一类"不同形状的层，同一个 lr 效果不同"的问题。

---

## 5. MuonH：把 lr 变成"相对改变比例"

MuonH 是 Wen 等人 [WDL+25] 提出的优化器（不是 HyperP 论文的贡献）。

### 5.1 更新规则（HyperP 论文 Eq. 2）

对一个隐藏层矩阵 W，半径 c_W = ‖W0‖（初始化时的范数）：

```
U = NS(G)
U = U / ‖U‖ * ‖W‖     # A：更新归一化，更新长度规定为 ‖W‖ 的 1 倍
W = W - lr * U
W = W / ‖W‖ * c_W     # B：权重归一化，缩回原长度（这才是"绑在球面上"）
```

- A 消掉原因 1：U 的大小不再由形状决定，§4 的形状缩放项也被除以 ‖U‖ 抵消，所以 recipe 里的 `scale_mode`、`extra_scale_factor` 在 MuonH 下无效（审计 F5）。
- B 消掉原因 2：‖W‖ 永远固定为初始化时的值。
- 结果：每步 **‖ΔW‖ / ‖W‖ ≈ lr**，与层的形状、训练进度都无关。所以 MuonH 下所有层可以共用同一个 lr，不需要 μP 的逐层倍率。

dpl 里的实现（`dpl/megatron/_hyperball.py::_hyperball_update`）几乎就是这两行：

```
X ← X − lr · (‖X‖/‖U‖) · U        # U 是 NS 正交化后的方向
X ← X · ‖X_pre‖ / ‖X‖             # 拉回原半径
```

AdamH 是同样的 A、B 两步，只是方向 U 换成 AdamW 的 m/√v（不做 NS）。所以 AdamH 的 lr 含义和 MuonH 一样，两者可以共用一个 lr。

### 5.2 MuonH 没解决的：深度和训练时长

球面约束只抹平了层与层之间、宽度之间的差异。模型深度和训练 token 数仍然影响最优 lr，MuonH 原作者"天然可跨深度迁移"的说法也被 HyperP 论文推翻了。怎么按深度和 token 数缩放 lr，见 [HyperP.md](HyperP.md)。

---

## 6. 实际怎么实现：Muon / MuonH 只管隐藏层的线性映射

**Muon 和 MuonH 都不作用在全部参数上**，只作用在 Transformer block 内部的 2-D 线性映射权重上。其余参数交给 AdamW（或 AdamH）。所以实现上一定是"一个模型、多个参数组、每组一种优化器"。

### 6.1 为什么只管隐藏层线性映射

- **1-D 参数**（norm 增益、bias）：没有矩阵结构，NS（正交化）无从谈起。
- **Embedding**：本质是查表，每步只有出现过的 token 那几行有梯度，把整张表当成一个"线性映射"做正交化没有意义。
- **LM head**：输出维度是词表大小 V，形状极不对称，且直接决定 logit 尺度，惯例上用 AdamW（MuonH 体系里用 AdamH）。
- **MoE router**（E × h）：是一张"路由打分表"，正交化后的步长没有谱意义。dpl 的 hyperball 实验里把 router 放进 AdamW 会塌（见 [HyperP.md](HyperP.md) §5），最后改成 AdamH。

### 6.2 dpl 的参数分组（Megatron 参数名）

分组规则在 `dpl/recipes/_family.py`：用正则匹配参数名，分成 `embedding` / `head` / `hidden` 三组，参数归入第一个匹配的组。

| 参数（Megatron 名） | 形状 | 组 | 普通 Muon recipe | MuonH recipe（HyperP） |
|---|---|---|---|---|
| `self_attention.linear_qkv.weight` | (q+2kv, h) | hidden | **Muon** | **MuonH** |
| `self_attention.linear_proj.weight` | (h, q) | hidden | **Muon** | **MuonH** |
| `mlp.linear_fc1.weight` / `linear_fc2.weight` | (2·ffn, h) / (h, ffn) | hidden | **Muon** | **MuonH** |
| `mlp.experts.linear_fc{1,2}.weight<e>`（每个专家一个） | 同上 | hidden | **Muon** | **MuonH**，每个专家矩阵一个独立球面 |
| `mlp.router.weight` | (E, h) | hidden → 视 recipe 而定 | Muon（Megatron 默认；`--moe-router-skip-muon` 可排除） | **AdamH**（v1；v0 用 AdamW 塌了） |
| `embedding.word_embeddings.weight` | (V, h) | embedding | AdamW | AdamW（`exclude_embedding: true`） |
| `output_layer.weight` | (V, h) | head | AdamW | **AdamH**（需要不绑定 embedding、init std > 0） |
| 各种 layernorm 增益、bias | 1-D | embedding | AdamW，wd 0 | AdamW，wd 0 |

两套 recipe 的入口：
- 普通 Muon：`dpl/recipes/muon_wsd.py`，走 Megatron 原生 `--optimizer muon`。规则和 Megatron 一致：**2-D 且不是 embedding/output 的参数才进 Muon**，其余交给 `--muon-scalar-optimizer`（adam）。
- MuonH：`dpl/recipes/hyperp_muonh.py` + `dpl/megatron/_hyperball.py::HyperballOptimizer`。hidden 组走 `muon_hyperball`，head（和 router）走 `adam_hyperball`，其余走 Megatron 自带的 Adam。`HyperballOptimizer.step()` 遇到 1-D 参数会直接报错，wd ≠ 0 也报错。

### 6.3 几个实现细节

- **fused 矩阵要不要拆开做 NS**：普通 Muon recipe 打开了 `split_qkv` 和 `split_swiglu`，把 qkv 拆成 Q/K/V、把 SwiGLU 的 fc1 拆成 gate/up 两半，各自做 NS（推断：它们本来就是不同的映射，拼在一起正交化会把它们的奇异值混在一起统一压平）。dpl 的 MuonH（`_muonh_step`）对整个 fused 矩阵做一次 NS，球面也是整个矩阵一个。
  - **SwiGLU fc1：和参考一致。** ArchScale 的 `mlp.w1` 本身就是 fused 的 (8192, 1024)，作为一个 MuonH 矩阵、一个球面（审计 §2 参数表）。
  - **QKV：和参考不一致，且没有记录为偏差。** ArchScale 模型是 `separate_qkv`：`q_proj`、`k_proj`、`v_proj`、`g_proj`（gate）是 4 个独立矩阵，各自做 NS、各自一个球面（半径 26.1 / 13.1 / 13.1 / 2.31）。dpl 跑在 Megatron 上，`linear_qkv` 是 fused 的 q+2kv(+gate)，MuonH 把它当一个矩阵：一次 NS、一个球面。`hyperp_muonh.py::reference_deviations` 和审计里都没有列这一条（只在 `docs/dev/change_requests/G2.md` 提到 fused qkv 的 dim0 % tp 问题）。
  - 影响（推断，未实验验证）：① NS 会把 Q、K、V、gate 的奇异值放在一起统一压平；② 一个共享球面只约束四者的**总**范数，Q/K/V/gate 之间的相对大小可以自由此消彼长（例如 V 变大、gate 变小），参考实现里这些都是各自钉死的。打开了 QK-norm 时 Q、K 的尺度影响较小，V 和 gate 的尺度不受此保护。
- **每组 lr**：普通 Muon recipe 下，Muon 组和 Adam 组的 lr 含义不同（一个是绝对步长 + 形状缩放，一个是 Adam 步长）。MuonH recipe 下，MuonH 和 AdamH 的 lr 都是"相对改变比例"，所以可以共用；但 embedding / norm 仍是普通 AdamW，和它们共用全局 lr 是 ArchScale 代码的选择，不是理论推出来的（见 [HyperP.md](HyperP.md) §4"论文 vs 代码"）。
- **MuonH 的工程限制**：TP 必须为 1，不能用 distributed optimizer，Newton-Schulz 对每个本地专家矩阵逐个做，m3 步时开销 +70–90%（见 [HyperP.md](HyperP.md) §5）。

---

## 7. Muon 实现细节（伪代码）

代码依据：dpl `dpl/megatron/_hyperball.py`（`newton_schulz`、`_muonh_step`、`_hyperball_update`）、`dpl/megatron/optim.py`（`NS_COEFFICIENTS`），以及 ArchScale 自带的 emerging_optimizers `orthogonalized_optimizers/muon.py`（`get_muon_scale_factor`）。

### 7.1 一步 Muon 更新

```python
# 对每个 hidden 组的 2-D 参数 W（形状 m×n），G = W.grad
M = mu * M + G                    # 动量（dpl 用 sum 形式，mu = 0.95）
U = mu * M + G if nesterov else M # Nesterov：用"再往前看一步"的动量
U = newton_schulz(U, steps=5)     # 正交化：奇异值 → ≈1，见 7.2

# ---- 从这里开始，Muon 和 MuonH 分叉 ----

# 普通 Muon：
U = U * shape_scale(m, n) * extra_scale_factor   # 形状缩放，见 7.4
W = W - lr * wd * W                                # 解耦 weight decay（像 AdamW）
W = W - lr * U

# MuonH：
U = U * (‖W‖ / ‖U‖)               # 形状缩放被这一步抵消，所以 MuonH 不需要它
W = W - lr * U
W = W * (c_W / ‖W‖)               # 拉回球面；wd 必须为 0
```

### 7.2 Newton-Schulz：到底在算什么

**目标**：给定 G = U_svd · S · V_svdᵀ（SVD），想要 U_svd · V_svdᵀ，即**保留方向（左右奇异向量），把所有奇异值换成 1**。直接做 SVD 太贵、在 GPU 上不友好，NS 只用矩阵乘法来近似它。

```python
def newton_schulz(G, steps=5, coeffs=QUINTIC, eps=1e-8):
    X = G.to(bfloat16)                 # 全程 bf16
    tall = X.rows > X.cols
    if tall:
        X = X.T                        # 转成"宽"矩阵，让 X @ X.T 是小的那个 Gram 矩阵 (min×min)
    X = X / (‖X‖_F + eps)              # 先归一化：保证所有奇异值 ≤ 1（σ_max ≤ ‖X‖_F）
    for i in range(steps):
        a, b, c = coeffs[i % len(coeffs)]
        A = X @ X.T                    # min×min
        B = b * A + c * (A @ A)
        X = a * X + B @ X              # = a·X + b·(XXᵀ)X + c·(XXᵀ)²X
    if tall:
        X = X.T
    return X
```

**为什么这样能把奇异值推向 1**：设 X = U·S·Vᵀ，则 XXᵀ = U·S²·Uᵀ，代入：

```
a·X + b·(XXᵀ)X + c·(XXᵀ)²X = U · (a·S + b·S³ + c·S⁵) · Vᵀ
```

奇异向量 U、V 完全不变，只是**每个奇异值 σ 被同一个奇次多项式 p(σ) = aσ + bσ³ + cσ⁵ 映射一次**。系数 (a, b, c) 被调成：小的 σ 被放大（a ≈ 3–4，近似每步 ×a），接近 1 的 σ 被拉回 1 附近。迭代 5 次，所有奇异值都被挤到 1 附近。

几个要点：
- **先归一化是必须的**：多项式只在 [0, 1] 区间上设计过，σ > 1 时会发散。除以 F 范数保证 σ_max ≤ 1。
- **转置 tall 矩阵**：只是为了让 A = XXᵀ 是 min(m,n)×min(m,n)，省算力；结果再转回来。
- **每步 3 次矩阵乘**（XXᵀ、A·A、B·X），5 步共 15 次，全部 bf16。这就是 MoE 上 NS 对每个专家矩阵逐个做时步时开销大的原因。
- **NS 输出不是精确正交**：奇异值只是落在 1 附近，不等于 1（见 7.3）。

### 7.3 两套系数，以及奇异值实际怎么变

dpl `NS_COEFFICIENTS`：

| 名字 | 系数 | 谁在用 |
|---|---|---|
| `simple` | 每步都是 (3.4445, −4.7750, 2.0315) | Keller Jordan 原版 Muon；Boxiang 的生产 Muon recipe |
| `quintic` | 每步不同：(4.0848, −6.8946, 2.9270), (3.9505, −6.3029, 2.6377), (3.7418, −5.5913, 2.3037), (2.8769, −3.1427, 1.2046), (2.8366, −3.0525, 1.2012) | Dion 的系数；ArchScale / dpl MuonH |

把单个奇异值 σ 代入 p(σ) 迭代 5 步（我用 python 按上面系数算的）：

| 初始 σ | simple：第 1→5 步 | quintic：第 1→5 步 |
|---|---|---|
| 0.001 | 0.003, 0.012, 0.041, 0.14, **0.47** | 0.004, 0.016, 0.06, 0.17, **0.48** |
| 0.01 | 0.03, 0.12, 0.40, 1.09, **0.70** | 0.04, 0.16, 0.58, 1.13, **1.02** |
| 0.1 | 0.34, 0.99, 0.71, 1.11, **0.71** | 0.40, 1.21, 0.58, 1.14, **1.02** |
| 0.5 | 1.19, 0.90, 0.82, 0.94, **0.77** | 1.27, 0.84, 0.80, 1.09, **0.99** |
| 1.0 | 0.70, 1.11, 0.72, 1.09, **0.70** | 0.12, 0.45, 1.22, 1.06, **0.98** |

可以看出：
- `simple` 不收敛到 1，而是在 **≈0.7–1.1 之间来回跳**；它追求的是"小奇异值快速放大"，不追求精确。
- `quintic`（每步换系数）最后几步收得更紧，终点 **≈0.98–1.02**。
- **很小的奇异值（≤0.001）5 步内到不了 1**，只到 ≈0.5。归一化后 1024 维矩阵的典型 σ 约 1/32 ≈ 0.03，在可收敛的范围内。

所以 §2 里"NS 输出奇异值 ≈1、‖U‖_F ≈ √min(m,n)"是近似；用 `simple` 时 ‖U‖_F 会明显小于 √min(m,n)。对 MuonH 无所谓（会除以 ‖U‖），对普通 Muon 会影响实际步长。

### 7.4 普通 Muon 的形状缩放（`scale_mode`）

ArchScale 自带的 emerging_optimizers 版本里，`get_muon_scale_factor(size_out, size_in, mode)`：

| mode | 乘的系数 | 出处 / 目的 |
|---|---|---|
| `shape_scaling` | √max(1, out/in) | Keller Jordan 原版 |
| `align_adamw_rms` | 0.2 · √max(out, in) | Kimi Moonlight：让更新的 RMS 和 AdamW 对齐，从而能沿用 AdamW 的 lr |
| `spectral_mup` | √(out/in) | Scion / Bernstein：谱范数意义下的 μP |

最后再乘 `extra_scale_factor`。

**乘在哪里**：乘在**更新**上，不是乘在权重上（`muon.py:104-105`：`return orth_grad * scale_factor * extra_scale_factor`）。顺序是 U = NS(动量) → U × scale → W ← W − lr·U，所以效果等价于给每层一个按形状算的 lr 倍率。

**为什么需要它**：NS 输出 U 的大小只由形状决定（谱范数 ≈1，元素 RMS ≈1/√max(m,n)），但"谱范数 = 1"对不同形状的层实际效果不同，所以要按形状重新定一个更新大小。三种 mode 选的"标准"不同（W 为 d_out × d_in）：

| mode | 乘完后 U 满足 | 目的 |
|---|---|---|
| `shape_scaling` | 元素 RMS = **1/√d_in** | 让更新和权重同一尺度 |
| `align_adamw_rms` | 元素 RMS = **0.2** | 冒充 AdamW，直接沿用 AdamW 的 lr / wd |
| `spectral_mup` | 谱范数 = **√(d_out/d_in)** | 谱 μP：宽度变化时激活变化 Θ(1)，lr 跨宽度迁移 |

- **shape_scaling 和 MuonH 最接近**：权重初始化 std = 1/√d_in，更新元素 RMS 也是 1/√d_in，于是 ‖ΔW‖_F = lr·√d_out，而初始化时 ‖W‖_F = √d_out，**初始化时每层相对改变都恰好 = lr**——和 MuonH 一样，但只在初始化那一刻成立，‖W‖ 漂移后就不成立了（§3 原因 2）。即：shape_scaling 用固定公式近似"相对改变 = lr"，MuonH 每一步用当前 ‖W‖ 保证它。
- **align_adamw_rms 与相对改变无关**：只是让更新 RMS 和 AdamW 的典型值（≈0.2）对齐，好处是 AdamW 调好的 lr 和 wd 可以直接用（Kimi Moonlight）。
- **spectral_mup 是理论派**：对应论文 Table 1 里 μP 行的 √(d_out/d_in)。

一句话：三种 scale 都是"按形状给更新定大小"的固定公式；MuonH 改为"按当前 ‖W‖ 的比例定"，所以这些公式在 MuonH 下全部被抵消。

Boxiang 的生产 recipe 用的是 `--muon-scale-mode spectral --muon-extra-scale-factor 0.2`，Megatron 的默认值也是 `spectral`。但 ArchScale 自带版本里**没有 `spectral` 这个分支**，Megatron 实际用的 emerging_optimizers 版本本机没装，所以 **`spectral` 的具体公式我没有核实**。ArchScale 的普通 Muon 分支用的是 `adjust_lr="spectral_norm"`，即 lr × √(fan_out/fan_in)（`docs/refs/hyperball_notes.md`），推断 Megatron 的 `spectral` 是同一类东西。

在 MuonH 下这些缩放全部失效：缩放后的 U 会被除以 ‖U‖，审计 F5 实测 `adjust_lr="spectral_norm"` 和 `None` 的 MuonH 结果完全一致（bit-identical）。

### 7.5 其他细节

- **fused 矩阵拆分**：普通 Muon recipe 打开 `split_qkv` / `split_swiglu`，NS 对 Q、K、V 和 gate、up 分别做（见 §6.3）。dpl 的 MuonH 对整个 fused 矩阵只做一次：SwiGLU 和参考一致，QKV 是一处未记录的偏差（见 §6.3）。
- **精度**：动量 M 用 fp32 存；NS 用 bf16 算；MuonH 的范数和拉回用 fp32。
- **零矩阵**：NS 对全零输入输出全零（eps 防止 NaN）。MuonH 下零范数矩阵的球面半径为 0，会被永远冻住，所以 dpl 直接报错（审计 F4）。

### 7.6 Weight decay：`W = W - lr * wd * W` 是什么

等价于 `W = W * (1 - lr*wd)`：每步把 W 整体按比例缩小一点、往 0 拉，再走正常的梯度更新。例：lr = 0.02、wd = 0.1 → 每步缩小 0.2%。

**来源是 L2 正则**：loss 里加 (wd/2)·‖W‖²，其梯度为 wd·W。普通 SGD 下 `W - lr*(G + wd*W) = W - lr*G - lr*wd*W`，即"loss 里加 L2"和"每步缩小 lr·wd 倍"完全等价。

**为什么要解耦**（单独一行作用在 W 上，而不是加进梯度）：
- Adam：wd·W 若加进梯度，会被一起除以 √v，梯度大的参数 decay 被削弱、梯度小的被放大。AdamW（Loshchilov & Hutter）把 decay 拿到优化器外面直接作用在 W 上。
- Muon 更严重：wd·W 若加进 G 再做 NS，会被一起正交化、丢掉大小信息，wd 失效或变意思。所以 Megatron 的 Muon 默认 `use_decoupled_weight_decay=True`。

**为什么乘 lr**：惯例（PyTorch AdamW 也是 `p.mul_(1 - lr*wd)`），让 decay 跟随 lr 调度、lr 降到 0 时 decay 也停。代价是 lr 和 wd 耦合，这也是"最优 wd 依赖 lr 和训练时长"的原因之一。

**和 MuonH 的联系**：普通 Muon/AdamW 里，wd 是**间接控制 ‖W‖ 的软旋钮**——梯度更新倾向于让 W 变长，wd 每步缩短一点，两者平衡出一个 ‖W‖。这就是 §3"原因 2"（‖W‖ 漂移）的另一面。MuonH 用硬约束直接钉死 ‖W‖，wd 无事可做：wd 的方向沿 W 本身（径向），而球面更新只保留切向分量，径向被投影掉，所以一阶无效（论文 Theorem 1）。dpl 强制 wd = 0；wd > 0 时球面半径会按 (1 − lr·wd)^t 慢慢缩小（审计 F2）。
