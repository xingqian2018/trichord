# MuonH 作用在哪些网络层上（Qwen3-MoE 梯度 OPT_MUONH / KD 各臂）

日期：2026-10-24 · 适用代码：`code/scaling_ladder/hyperp/`（入口 `pretrain_hyperp.py`，Megatron 树 `megatron-lm-80781ab0ab`）

## 0. 一句话结论

`OPTIMIZER=muon_hyperball` 并不是"所有参数都用 MuonH"，而是一个三路优化器（HyperP 配方 `hyperp_muonh_moe_v1`）：

| 算法 | 作用的参数 | 是否在球面上（范数恒定） |
|---|---|---|
| **MuonH**（`muon_hyperball`） | 每层的 4 类隐藏矩阵：attention `linear_qkv`、`linear_proj`；MoE 每个 expert 各自的 `linear_fc1`、`linear_fc2` | 是：每个矩阵各自一个球，半径 = 初始化时的 Frobenius 范数 |
| **AdamH**（`adam_hyperball`） | LM head `output_layer.weight`；每层的 MoE router `mlp.router.weight`（默认） | 是 |
| **AdamW**（Megatron 自带 Adam，wd = 0） | 词嵌入 `embedding.word_embeddings.weight`；所有 1-D 参数（各种 RMSNorm 权重，包括 QK-norm） | 否 |

一个 q0 上实际跑出来的日志（job 1174569，`qwen3_tpp200_muonh_kd_a0p7-q0`）：

```
[hyperp] parameter algorithms (pre-wrap): {'adamw': 50, 'muon_hyperball': 3096, 'adam_hyperball': 13}
```

下文说明这三个数字怎么来的、按名字怎么分、以及几处容易误解的细节。

---

## 1. 分配逻辑在哪里

分配是**按 Megatron 参数名的正则匹配**做的，唯一真相是 [hyperp/policy.py](../code/scaling_ladder/hyperp/policy.py)：

```python
HEAD      = r"^output_layer\.weight$"
ROUTER    = r"\.mlp\.router\.weight$"
EMBEDDING = r"^embedding\.(word|position)_embeddings\.weight$"
HIDDEN    = r"\.(self_attention\.linear_qkv|self_attention\.linear_proj|mlp\.linear_fc1|mlp\.linear_fc2"
            r"|mlp\.experts\.linear_fc[12]|mlp\.shared_experts\.linear_fc[12])\.weight\d*$"

def algorithm_for(name, ndim, router="adam_hyperball"):
    if HEAD:       return "adam_hyperball"
    if ROUTER:     return router            # --hyperp-router，默认 adam_hyperball
    if EMBEDDING:  return "adamw"
    if ndim <= 1:  return "adamw"
    if HIDDEN:     return "muon_hyperball"
    raise PolicyError(...)                  # 其它 >=2-D 参数：直接报错，绝不静默地用错规则
```

判断顺序很重要：**head → router → embedding → 1-D → hidden → 报错**。

调用链：

1. `pretrain_hyperp.py` 在模型构建后、`.cuda()` / Float16Module / DDP 包装之前，挂一个 `pre_wrap_hook`（`tagging_hook`），调用 `policy.tag_parameters`，给每个可训练参数打上属性 `param.hyperp_algorithm`。
2. `install.register_param_attributes` 把这个属性加进 Megatron 的 TP 属性表，于是 Megatron 创建 fp32 main param 时会把标签一起复制过去。
3. `install.install_registry` 向 Megatron 的 emerging-optimizer 注册表注册 `muon_hyperball`，并通过 `default_param_overrides` 把所有 `algorithm_for(...) == "adamw"` 的参数路由到 **Megatron 自己的 Adam**（weight decay 被强制为 0）；其余参数进入 `HyperballOptimizer`。
4. `HyperballOptimizer`（生产用 [hyperball_batched.py](../code/scaling_ladder/hyperp/hyperball_batched.py)，参考实现 [hyperball.py](../code/scaling_ladder/hyperp/hyperball.py)）对每个参数读 `hyperp_algorithm`，`muon_hyperball` 走 `_muonh_step`，`adam_hyperball` 走 `_adamh_step`。

---

## 2. Qwen3-MoE 每一层的参数怎么分

架构见 [sh/arch/qwen3moe.sh](../code/scaling_ladder/sh/arch/qwen3moe.sh)：GQA（head dim 128，4 个 KV head）、QK-norm、SwiGLU、RMSNorm、**不共享**输入/输出嵌入、**无 bias**、每层都是 MoE（128 个 routed expert，top-8，宽度 3h/8，**无 shared expert**），`--moe-grouped-gemm`。

### 2.1 一个 decoder 层（以 q0：h = 512，8 个 q head 为例）

| 参数名（去掉 `module.` 前缀） | 形状 (q0) | 算法 | 初始化（`hyperp/init.py`） |
|---|---|---|---|
| `decoder.layers.i.self_attention.linear_qkv.weight` | 2048 × 512 | **MuonH** | kaiming-uniform U(±1/√fan_in) |
| `decoder.layers.i.self_attention.linear_proj.weight` | 512 × 1024 | **MuonH** | kaiming-uniform × 1/√(2L)（残差输出） |
| `decoder.layers.i.mlp.experts.linear_fc1.weight0 … weight127` | 每个 384 × 512 | **MuonH**（128 个独立矩阵） | kaiming-uniform |
| `decoder.layers.i.mlp.experts.linear_fc2.weight0 … weight127` | 每个 512 × 192 | **MuonH**（128 个独立矩阵） | kaiming-uniform × 1/√(2L)（残差输出） |
| `decoder.layers.i.mlp.router.weight` | 128 × 512 | **AdamH**（默认；可改，见 §4） | N(0, 0.02) |
| `decoder.layers.i.self_attention.linear_qkv.layer_norm_weight`（attention 前的 RMSNorm，TE 融合进 qkv） | 512 | AdamW | 1 |
| `decoder.layers.i.self_attention.q_layernorm.weight` | 128 | AdamW | 1 |
| `decoder.layers.i.self_attention.k_layernorm.weight` | 128 | AdamW | 1 |
| `decoder.layers.i.pre_mlp_layernorm.weight`（MoE 前的 RMSNorm） | 512 | AdamW | 1 |

每层：**258 个 MuonH 矩阵**（2 个 attention + 128 × 2 个 expert），1 个 AdamH（router），4 个 AdamW（norm 向量）。

### 2.2 层以外的参数

| 参数名 | 形状 (q0) | 算法 | 初始化 |
|---|---|---|---|
| `embedding.word_embeddings.weight` | 151936 × 512 | AdamW（2026-10-04 的决定：embedding 不放在球面上） | N(0, 1e-4) |
| `output_layer.weight`（LM head，untied） | 151936 × 512 | **AdamH** | N(0, 0.02)，球半径 = 0.02·√(V·h) ≈ 176.4 |
| `decoder.final_layernorm.weight` | 512 | AdamW | 1 |

### 2.3 计数核对（q0，L = 12）

- MuonH：12 × 258 = **3096** ✓
- AdamH：12 个 router + 1 个 head = **13** ✓
- AdamW：12 × 4 个 norm + final norm + embedding = **50** ✓

与日志完全一致。

---

## 3. 各个梯级的规模

矩阵个数 = MuonH L × 258，AdamH L + 1，AdamW 4L + 2。参数量按形状计算（q0 的合计 628.3M 与日志里的 `number of parameters … 628243968` 吻合）。

| 梯级 | L × h | `linear_qkv` | `linear_proj` | expert `fc1` | expert `fc2` | #MuonH 矩阵 | MuonH 参数（占比） | AdamH 参数（占比） | AdamW 参数（占比） |
|---|---|---|---|---|---|---|---|---|---|
| q0 | 12 × 512 | 2048×512 | 512×1024 | 384×512 | 512×192 | 3096 | 471.9M（75.1%） | 78.6M（12.5%） | 77.8M（12.4%） |
| q1 | 18 × 768 | 2560×768 | 768×1536 | 576×768 | 768×288 | 4644 | 1585.4M（87.1%） | 118.5M（6.5%） | 116.7M（6.4%） |
| q2 | 24 × 1024 | 3072×1024 | 1024×2048 | 768×1024 | 1024×384 | 6192 | 3749.7M（92.3%） | 158.7M（3.9%） | 155.7M（3.8%） |
| q3 | 30 × 1280 | 3584×1280 | 1280×2560 | 960×1280 | 1280×480 | 7740 | 7313.8M（94.9%） | 199.4M（2.6%） | 194.6M（2.5%） |
| q4 | 36 × 1536 | 4096×1536 | 1536×3072 | 1152×1536 | 1536×576 | 9288 | 12627.0M（96.4%） | 240.5M（1.8%） | 233.6M（1.8%） |

（按**总**参数量统计；AdamH 里几乎全是 LM head，router 只占很小一部分。在**激活**参数里，expert 只有 8/128 参与，MuonH 的占比会低一些，head 与 embedding 的相对份量更大。）

可以看到：MuonH 覆盖了绝大部分参数，而且随规模增大，覆盖比例从 75% 涨到 96%；剩下的主要是词表大小的 head（AdamH）和 embedding（AdamW）。

---

## 4. 几个容易误解的细节

1. **`linear_qkv` 是整体一个矩阵做 Newton-Schulz**，不拆成 Q / K / V，也不按 head 拆。q0 的 2048 行 = 8 个 q head × 128 + 4 个 k head × 128 + 4 个 v head × 128。也是因为这个原因，`base.sh` 明确拒绝 `muon_hyperball` 与 attention output gate 同时使用（"hyperp/ treats linear_qkv as one matrix"）。
2. **expert 的 `fc1` 是 gate | up 拼在一起的一个矩阵**（384 = 2 × 192），整体正交化，不拆成 gate 和 up 两个矩阵（这个 Megatron 版本没有 `--muon-split-swiglu`）。
3. **每个 expert 各自是独立的矩阵、各自一个球。** `--moe-grouped-gemm` 下 TE 的 `TEGroupedLinear` 把 128 个 expert 存成 `weight0 … weight127` 这 128 个独立参数（正则末尾的 `\d*` 就是匹配这个），所以每个 expert 单独做 NS、单独保持范数，而不是把 128 个 expert 拼成一个大张量来做。batched 实现只是把**同形状**的矩阵每 16 个（`--hyperp-ns-batch-size`）一组堆成 [B, m, n] 一起做 NS，数学上仍然是逐矩阵的。
4. **Router 默认是 AdamH，不是 MuonH。** `--hyperp-router` / `HYPERP_ROUTER` 可选 `adam_hyperball`（默认，范数钉在 0.02·√(E·h)）、`adamw`（MUONH_RTADAMW 臂：`qwen3_tpp200_muonh_rtadamw.sh`）、`muon_hyperball`。注释里写明在 HyperP 的学习率下用 AdamW 会在 30 步内让路由崩掉（gpt-oss 上的经验），所以 v1 选了 AdamH。
5. **LM head 是 AdamH，不是 MuonH**（151936 × h 的矩阵不做正交化）；**embedding 根本不在球上**，用 AdamW。
6. **QK-norm、各种 RMSNorm 这些 1-D 参数全部走 AdamW**，由 `ndim <= 1` 那一条规则兜底。gpt-oss 梯级里的 attention sink（`softmax_offset`）和 bias 同样走 AdamW（Qwen3 梯级没有这两类参数）。
7. **Weight decay 必须是 0**（`base.sh`、`check_args`、优化器构造函数、每一步里都检查一次）：否则球半径会按 (1 − lr·wd)^t 衰减。所以 AdamW 那一组在这个臂里的 wd 也是 0。
8. **任何没被匹配到的 ≥2-D 参数都会直接 `PolicyError`**，不会被悄悄地交给错误的优化器。例如以后加 shared expert（`mlp.shared_experts.linear_fc[12]`）已经在正则里；但加 attention output gate、MLA 之类新的 2-D 权重时，要先改 `policy.py`。

---

## 5. MuonH 对这些矩阵到底做了什么

对每个 MuonH 矩阵 X（梯度 G，fp32 状态）：

```
M ← μ·M + G                                   # sum-form momentum，μ = 0.95
U ← NS5_bf16(μ·M + G)                         # Nesterov；5 步 quintic（Dion 系数）Newton-Schulz，bf16
X ← X − lr · (‖X‖_F / ‖U‖_F) · U              # 相对步长：‖ΔX‖_F / ‖X‖_F = lr
X ← X · ‖X_pre‖_F / ‖X‖_F                     # 重新投影回球面，范数恒等于初始化范数
```

AdamH（head、router）用同样的"按范数比例走一步 + 投影回球面"，只是把 U 换成 Adam 的方向 m̂ / (√v̂ + ε)。

因此 **init 的范数就是整个训练过程中的球半径**，这也是为什么 `hyperp/init.py` 的初始化（kaiming-uniform，残差输出再乘 1/√(2L)，head 0.02，embedding 1e-4）和 policy 是成对出现的。运行时每 100 步打印一次 `max |norm/ref_norm - 1|`（`--hyperp-norm-monitor`），这个值应停留在 fp32 舍入误差（~1e-6）量级；变大就说明有参数被错误地踩出了球面（路由错误、wd 泄漏或 dtype 问题）。

---

## 6. 对照：普通 Muon（BASE / OPT_MUON）用在哪

为了对比，普通 Muon 臂（`OPTIMIZER=muon`，Megatron 自带实现）：隐藏矩阵用 Muon（spectral scaling，extra factor 0.2，`simple` NS 系数），**embedding、head、norm、sink、router 全部走 AdamW**（`MUON_ROUTER_SKIP_MUON=1`），wd 0.1，分布式优化器。也就是说，两者用正交化更新的层是同一批（qkv、proj、expert fc1/fc2），区别在于：

| | Muon | MuonH |
|---|---|---|
| 隐藏矩阵 | Muon，有 wd，范数自由变化 | MuonH，wd = 0，范数固定在球面上 |
| LM head | AdamW | AdamH（球面） |
| Router | AdamW | AdamH（球面，默认） |
| Embedding / norm | AdamW | AdamW（wd = 0） |
| 优化器状态 | 分布式 | 复制在每个 DP rank 上（因此需要 `resync.py` 定期同步） |

---

## 7. 相关文件

- 分配规则：[hyperp/policy.py](../code/scaling_ladder/hyperp/policy.py)
- 优化器数学：[hyperp/hyperball.py](../code/scaling_ladder/hyperp/hyperball.py)（参考）、[hyperp/hyperball_batched.py](../code/scaling_ladder/hyperp/hyperball_batched.py)（生产）
- 初始化（= 球半径）：[hyperp/init.py](../code/scaling_ladder/hyperp/init.py)
- 接入 Megatron 与路由：[hyperp/install.py](../code/scaling_ladder/hyperp/install.py)、[pretrain_hyperp.py](../code/scaling_ladder/pretrain_hyperp.py)
- 实验配置：[sh/exp/qwen3_tpp200_muonh.sh](../code/scaling_ladder/sh/exp/qwen3_tpp200_muonh.sh)、[sh/exp/qwen3_tpp200_muonh_rtadamw.sh](../code/scaling_ladder/sh/exp/qwen3_tpp200_muonh_rtadamw.sh)、[sh/base.sh](../code/scaling_ladder/sh/base.sh)
- 架构：[sh/arch/qwen3moe.sh](../code/scaling_ladder/sh/arch/qwen3moe.sh)
