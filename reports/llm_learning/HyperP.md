# HyperP 学习笔记

> Muon / MuonH 优化器本身（更新规则、两者 lr 的区别、NS 输出大小、参数分组）见 [Muon.md](Muon.md)。

> 论文：*Rethinking Language Model Scaling under Transferable Hypersphere Optimization*，arXiv 2603.28743（Microsoft，Ren / Liu / Shen / Chen），代码 ArchScale。
> 本文依据：Dense-Pretraining-LMs 仓库里的 `docs/refs/hyperball_notes.md`（论文逐页摘录）、`docs/audits/MUONH_AUDIT.md`（代码审计）、`dpl/` 里的实现，以及 `docs/reports/2026-10-02_kd_scaling_v1.md` 第 7 节的实验。
> 原文 PDF 和提取文本在同目录：`HyperP_2603.28743.pdf` / `.txt`。已对照原文核对了摘要、Table 1、Theorem 3、Eq. 8、batch 律、训练设置；Theorem 1/2 的证明、SqrtGate 推导、附录未逐字读。标注"推断"的是我自己推的。

---

## 0. 一句话版本

**MuonH**：把每个隐藏层权重矩阵钉在 Frobenius 范数球面上（半径 = 初始化时的范数），每步只"转方向"，不"改长度"。
**HyperP**：在这个约束之上的一套**超参迁移律**——只在最小模型上调一个基准学习率 η0，就能按公式外推到不同深度、不同训练 token 数、不同 MoE 粒度，不用每个规模重调。

两个名字别混：MuonH 是已有的优化器（Wen 等人），HyperP 是这篇论文的贡献（迁移律 + 参数化规则）。

---

## 1. 先对比 μP：它们解决的是同一类问题

目标都是"小模型上调好的超参，能直接用到大模型"。

- **μP**：权重范数不受限制，靠**给每类参数配不同的 LR 倍率 / 初始化 / 输出倍率**，让各层的更新幅度在宽度变化时保持稳定。宽度项（√(d_out/d_in)、1/w 等）显式出现在公式里。
- **HyperP**：**权重范数被球面约束固定**，更新的相对大小 ‖ΔW‖_F/‖W‖_F 恒等于 lr，与矩阵形状无关。宽度的影响被约束"吸收"了，LR 公式里**不再有宽度项**。

论文 Table 1（隐藏层权重一行）：

| 方案 | LR 倍率 | 初始化 std | 残差倍率 | 权重 WD |
|---|---|---|---|---|
| μP | ∝ √(d_out/d_in) | ∝ 1/√d_in | 1 | ∝ 1/w |
| μP++ | ∝ √(d_out/(d_in·d)) | ∝ 1/√d_in | 1/√(2d) | ∝ 1/w |
| **HyperP** | **∝ 1/(T^0.32 · √d)** | ∝ 1/√d_in | 1/√(2d) | **0** |

（w = 宽度，d = 深度，T = 训练 token 数。）

原文 Table 1 还有另外两行（上表只列了隐藏层）：

| 参数 | 方案 | LR 倍率 | 权重 WD |
|---|---|---|---|
| Embedding / 向量 | HyperP | ∝ 1/√d | 0 |
| Unembedding（head） | HyperP | ∝ 1/√d | 0 |

即**论文里 embedding/head 的 LR 倍率只有 1/√d，没有 T^-0.32**；T^-0.32 只写在隐藏层那一行。这和 ArchScale 代码（所有 group 共用同一全局 LR）确实不一致，见 §4 末尾"论文 vs 代码"。优化器分配：隐藏矩阵 MuonH，unembedding AdamH，向量参数和 embedding 用原版 AdamW。

所以你说"μP 更多不是这样控制 learning rate"——对：μP 的 LR 主要随**宽度/形状**变；HyperP 的全局 LR 随**深度和训练 token 数**变，宽度被球面约束吃掉了。

---

## 2. MuonH 怎么更新

已移到 [Muon.md](Muon.md) §5。要点：MuonH 每步 ‖ΔW‖_F/‖W‖_F ≈ lr，lr 的含义是"相对改变比例"，和普通 Muon 的 lr（绝对步长）是两个东西。

---

## 3. 论文的几个理论结论

### 3.1 Weight decay 在球面上一阶无效（Theorem 1）
球面更新后，W⁺ − W = Π_T(Δ) + O(‖Δ‖²)，其中 Π_T 是切空间投影。weight decay 项 −ηλW 正好沿着径向（W 的方向），被投影掉了。
结论：**WD 不再是超参**，搜索空间从 (η, λ) 二维降到 η 一维。
> 实现提醒（审计 F2）：论文说的是"一阶无效"。但 ArchScale 代码里如果 wd>0，半径会按 (1 − lr·wd)^t 衰减，相当于半径调度，不再是无效。所以代码强制 wd=0，dpl 里 recipe 校验也要求 wd=0。

### 3.2 宽度迁移（Theorem 2）
若 ‖W‖_F = C·√d_out（C 与宽度无关），且 W 近似各向同性，则输出 RMS ≈ C·输入 RMS。因此**不需要 μP 那种显式的 1/w 学习率缩放**。
各向同性的依据：Muon 正交化后的更新倾向于让奇异值谱比较平。

### 3.3 深度迁移（Theorem 3）：Depth-μP 仍然需要
残差网络 x_{l+1} = x_l + α_L·f_l(x_l; W_l)。如果同时归一化权重和更新（即 MuonH），则要求 η_l = O(1/(L·α_L))；取 α_L = L^{-1/2} 时 η_l = O(L^{-1/2})。
这就是公式里 **√(d0/d)** 的来源。论文特别指出：MuonH 原作者声称它"天然可跨深度迁移"，这个说法不对——残差连接累加会产生角度漂移，所以深度项仍然要显式写。

### 3.4 Token 数迁移（经验律，不是推导）
固定深度 d=8（208M 参数），训练 token 从 10.4B 扫到 166.4B，对每个 token 数在 LR 网格上扫描，对 log(η) 做二次拟合，得到：

```
η* = 24.27 · T^(-0.320)        （论文 Eq. 8）
```

留一法交叉验证对最优 LR 的平均绝对误差是 1.5%。指数 0.32 和 Bjorck 等人在 AdamW 上发现的"magic exponent"一致。
这一项是**拟合出来的经验规律**，没有理论推导。

### 3.5 有界 logit
球面约束 ‖W‖_F = C，所以 ‖Wx‖₂ ≤ C·‖x‖₂，attention logit 和 MoE router logit 天然有界，不需要 z-loss。

### 3.6 SqrtGate（MoE 粒度）
常规 gating 下，k 个激活专家输出的 RMS ≈ r/√k，随粒度 k 变化。把 gate 值换成 √g_i，输出 RMS ≈ r，与 k 无关。有共享专家时再乘 1/√2。
（dpl 的 gpt-oss MoE 实验没有用 SqrtGate，这里只作了解。）

---

## 4. 最终的 LR 律怎么拼出来的

```
lr = η0 · sqrt(B/B0) · sqrt(d0/d) · (T/t0)^(-0.32)
```

| 项 | 来源 | 在我们的 ladder 里 |
|---|---|---|
| η0 = 0.016 | 在 d=8、T=10.4B 处调出的基准 LR | 固定 |
| sqrt(d0/d) | 3.3 的深度迁移 | d0=8，d 为层数 |
| (T/t0)^-0.32 | 3.4 的 token 数律 | t0=10.4e9 |
| sqrt(B/B0) | batch 项（论文实测 batch 指数 0.558，代码用的是 √） | batch 固定 2^21 token = B0，所以这项 = 1 |

我们的 ladder 参数在 `configs/recipes/hyperp_muonh_moe_v1.yaml`：`lr: {rule: hyperp, eta0: 0.016, d0: 8, t0: 1.04e+10, exponent: -0.32, depth_exponent: -0.5}`，规则实现在 `dpl/recipes/rules.py` 的 `_lr_hyperp`。

**数值检查**（我自己算的）：
- 24.27 × (10.4e9)^-0.32 ≈ 0.0151，和 η0=0.016 对得上（0.016 是 LR 网格上的最优点之一）。
- m0 在 tpp20 时 T ≈ 1.52B，(T/10.4e9)^-0.32 ≈ 1.85，所以 lr ≈ 0.016 × 1.85 ≈ 0.0296，正好和报告里的 0.0296 吻合。由此**推断** m0 的层数 L = 8（√(8/L)=1）。
- tpp200、tpp500 的 T 更大，LR 更小。

### 论文 vs 代码的一处出入（审计 F11）
论文 Table 1 里 T^-0.32 只写在隐藏权重一行（embedding/head 行只有 1/√d，已对照原文确认）；但发布的 ArchScale 代码把整套全局 LR 同时用于 embedding / 向量 / head（`weight_lr_mult = 1.0`）。dpl 选择**跟代码走**（所有 group 共用同一 LR、lr_mult=1），而不是按论文字面。

### 其它配套规则（也是 HyperP 的一部分）
- 无 warmup（代码强制 warmup=0）。
- 线性衰减到峰值的 0.1。
- wd = 0，grad clip 1.0，动量 0.95 Nesterov，NS 5 步（Dion 的 quintic 系数）。
- 初始化：隐藏层 kaiming-uniform（std = 1/√(3·fan_in)，半径 √(fan_out/3)）；embedding N(0, 1e-4)；head N(0, 0.02)；残差输出 ×1/√(2L)。
- 球面半径 = 初始化范数，因此**范数为 0 的矩阵会被永远冻住**（审计 F4）——所以 head 必须有 >0 的初始化 std。

---

## 5. 在 dpl 里怎么落地（MoE 版）

代码位置：
- 配置：`configs/recipes/hyperp_muonh_moe_v1.yaml`、`configs/ladders/oss_moe_qwen38_muonh_v1.yaml`（CE）和 `..._kd_muonh_v1.yaml`（KD）。
- 校验与分组：`dpl/recipes/hyperp_muonh.py`。
- 优化器：`dpl/megatron/_hyperball.py::HyperballOptimizer`，接入 Megatron 的逻辑在 `dpl/megatron/optim.py::install_megatron_registry`。

MoE 上的参数分配：

| 参数 | 优化器 |
|---|---|
| attention qkv / proj | MuonH |
| 每个 expert 的 fc1 `[2·ffn, h]`、fc2 `[h, ffn]` | MuonH，**每个矩阵一个独立球面** |
| router（E×h） | **AdamH**（范数钉在 0.02·√(E·h)） |
| LM head | AdamH |
| embedding | AdamW（用户决定不上球面，对应参考实现里的 `exclude_from_hyperball(wte)`） |
| norm / bias / attention sink | AdamW |

### v0 的失败：router 放 AdamW 会塌
第一版把 router 放进 AdamW 组，吃到全局 HyperP LR 0.0296（比 AdamW ladder 的 router LR 2.6e-3 大约 11 倍，还没有 warmup）。30 步内 load-balancing loss 从 1.4 升到 7.8 = num_experts/topk，意思是所有 token 都进了同样的 4 个专家，之后一直没恢复。
v1 把 router 改成 AdamH（范数钉死），路由 logit 的尺度不会失控，方向仍按 HyperP 速率学习，load-balancing loss 的走势和 AdamW 基线接近。
**教训**：HyperP 的"所有 group 共用同一个 LR"这个设计，对这种 AdamW 小矩阵（router）并不安全，在 MoE 上要想清楚每个参数该归哪个 group。

### 工程限制
- TP 必须 = 1（hyperball group 不支持张量并行，审计 F26）。
- 不能用 distributed optimizer（`use_distributed_optimizer: false`）；fp32 主权重和状态按本地参数全量复制，m3 每卡约 100GB。
- 有一个 sphere monitor，每 100 步打印 `max |‖X‖/‖X_init‖ − 1|`，应该只有 fp32 舍入误差（报告中读数 0–5e-6）。偏离说明有参数被错误地放到球面之外或被 wd 污染。
- 步时开销：m0/m1 +12%，m2 +26%，m3 +70–90%（Newton-Schulz 对本地每个专家矩阵逐个做）。

---

## 6. 实验结果（我们的 ladder）

- 26 个 MuonH run 全部跑完（CE 13 + KD 13）。
- KD 相对 CE 的增益在 MuonH 下依然成立，且每个 cell 都略大：MuonH −0.059/−0.037/−0.031/−0.056/−0.035/−0.057/−0.062，AdamW −0.024/−0.029/−0.029/−0.048/−0.034/−0.051/−0.053。
- MuonH 相对 AdamW 的优势在短训练、小模型上很大（m0 tpp20 约 0.32 nats），随 token 数和模型规模增大快速缩小。
- 注意：LR 律**没有**为 MoE、也没有为 KD 重新调过，直接沿用 dense 实验的 η0。这既是复现论文的设定，也是一个潜在的未调优点。

---

## 7. 我还没搞清楚 / 值得继续看的

1. LR 律在 MoE 上的最优 η0 是否真的和 dense 一样？m0 用的是 L=8 的小模型，正好落在基准点上，m1–m3 的层数更深，依赖 √(d0/d) 外推，没有在我们的模型上扫过 LR 验证。
2. 论文 Table 1 明确 embedding / head 不乘 T^-0.32（只乘 1/√d），但代码对所有 group 用同一全局 LR；我们跟随代码，没有验证过这个选择对结果的影响。
3. 论文笔记里 Section "Ambiguities"（`docs/refs/hyperball_notes.md` 后半）列了论文没写清的地方，值得再读一遍。
4. 想自己验证的话：在 m0 上对 η0 做一个小范围扫描，看最优点是否在 0.016 附近。

---

## 附：关键文件速查

| 想看什么 | 去哪 |
|---|---|
| 论文逐页摘录 | `docs/refs/hyperball_notes.md` |
| 代码审计（F1–F26） | `docs/audits/MUONH_AUDIT.md` |
| LR 律实现 | `dpl/recipes/rules.py`（`_lr_hyperp`） |
| 参数分组与校验 | `dpl/recipes/hyperp_muonh.py` |
| 优化器本体 | `dpl/megatron/_hyperball.py` |
| ArchScale 原始参考 | `reference/archscale/pretrain.py`、`reference/archscale/lit_gpt/optim/muon.py` |
| 实验报告 | `docs/reports/2026-10-02_kd_scaling_v1.md` 第 7 节 |

---

## QA Session

> 关于 Muon / MuonH 本身的问答（范数是什么、lr 含义、NS 输出多大、普通 Muon 的相对改变）已移到 [Muon.md](Muon.md)。

### Q1：MuonH 和 HyperP 是一个东西吗？
不是。**MuonH** 是已有的优化器（Wen 等人 [WDL+25]），作用是把权重钉在 Frobenius 球面上。**HyperP** 是这篇论文的贡献，是建立在 MuonH 之上的超参迁移律（宽度、深度、token 数、MoE 粒度）。原文摘要和引言都明确区分了这两者。

### Q2：那为什么还要调 global LR？
球面约束只抹平了层与层之间、宽度之间的差异，下面两个因素仍然影响最优 lr：
- **深度**：残差连接累加会产生角度漂移，模型越深同样的 lr 总效果越大，所以要乘 √(d0/d)（论文 Theorem 3，并指出 MuonH 作者"天然可跨深度迁移"的说法不成立）。
- **训练 token 数**：训练越久最优 lr 越小，经验律 T^-0.32（论文 Eq. 8，没有理论推导）。

HyperP 的作用：在小模型上调好一个基准 lr（η0），换成更深或训练更久的模型时，按上面两项缩放，不用重调。

### 一句话总结
球面约束让"每层步长一致"，所以不需要 μP 的逐层倍率；但深度和训练时长仍然影响最优 lr，所以 HyperP 提供一个全局缩放公式。
