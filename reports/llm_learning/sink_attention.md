# Sink attention 学习笔记（2026-10-20）

> 本文讲 attention sink（gpt-oss 的 learnable sink）是什么、在 Megatron 里怎么实现、和滑动窗口注意力（SWA）的关系。
> 结合我们的实验：dpl 的 m 系列（`oss_moe_m*`）和 Boxiang 的 gpt-oss 梯子（g0–g6）都用 learnable sink；Qwen3 梯子（q0–q6）不用。

---

## 核心结论

- sink = 每个 head 在 softmax 里多出的**一个虚拟位置**：**有分数 b_h，没有 key、query，也没有 value**。
- b_h 是**可学的标量参数**，每层每个 head 一个，**不是 Q×K 算出来的**，对所有 query、所有位置都一样。
- sink **永远不被 mask**，每个 query 都能"看到"它；分给它的注意力乘的是 0，不往输出里加任何东西。
- 作用：一个 head 不想关注任何 token 时，可以把注意力"放空"给 sink，输出接近 0。
- **短窗口的 SWA 需要 sink**（gpt-oss：窗口 128 + sink + 每两层一层 full attention）；长窗口（如 4096）对 sink 的需求小得多。

---

## 1. 公式

普通 attention，query 对第 j 个 token 的分数：

```
s_j = (q · k_j) / sqrt(d)          # d = head 维度（m 系列是 64）
p_j = exp(s_j) / Σ_j exp(s_j)       # 只对能看到的 token 求和
out = Σ_j p_j · v_j
```

加 sink 后，分母多一项 exp(b_h)：

```
p_j = exp(s_j) / ( exp(b_h) + Σ_j exp(s_j) )
out = Σ_j p_j · v_j                 # sink 没有 value，等价于 value = 0
```

| | key / query | 分数 | value |
|---|---|---|---|
| 普通 token | 有 | q · k_j，随 query 和 key 变化 | 有，加权求和进输出 |
| sink | 没有 | 固定的 b_h（可学） | **没有**（= 0） |

- **h 是 head 的编号**，不是位置。m1 每层 24 个 head，一共 12 层 × 24 = 288 个 b。
- **每个 head 只有一个 sink**，学的是它的分数，不是数量。b_h 越大，这一个 sink 的"分量"越重：
  exp(b_h) = 1（b_h = 0）相当于 1 个分数为 0 的 token；exp(b_h) = 3 相当于 3 个；b_h 很负时几乎不起作用。
- sink **不是第 0 个 token**：它在序列之外，没有位置、不加 RoPE。StreamingLLM 那种 "attention sink" 是拿真实的第一个
  token（分数 q·k_0，有 value）当 sink，和这里不同。

### Megatron 的三种 `--softmax-type`

| 值 | sink 分数 | 谁在用 |
|---|---|---|
| `vanilla` | 没有 sink | Qwen3 梯子（q0–q6） |
| `off-by-one` | 固定 b_h = 0，即分母"加 1"：exp(s_j) / (1 + Σ exp(s_j)) | 目前没人用 |
| `learnable` | 每层每个 head 一个可学的 b_h | **gpt-oss**：dpl 的 m 系列、Boxiang 的 g 系列 |

---

## 2. 画出来：有 sink 和没有 sink 的 SWA

窗口 3、6 个 token 的例子。每一行是一个 query，每一列是一个 key；`■` = 能看到（分数 q·k），`·` = 被 mask。

**没有 sink**

```
              key:  t0  t1  t2  t3  t4  t5
query t0           [ ■   ·   ·   ·   ·   · ]
query t1           [ ■   ■   ·   ·   ·   · ]
query t2           [ ■   ■   ■   ·   ·   · ]
query t3           [ ·   ■   ■   ■   ·   · ]
query t4           [ ·   ·   ■   ■   ■   · ]
query t5           [ ·   ·   ·   ■   ■   ■ ]

每一行：能看到的 token 的权重加起来 = 1
```

**有 sink**：每行最右边多一格，填固定数 b（每个 head 一个），每一行都能看到。

```
              key:  t0  t1  t2  t3  t4  t5 | sink
query t0           [ ■   ·   ·   ·   ·   · |  b  ]
query t1           [ ■   ■   ·   ·   ·   · |  b  ]
query t2           [ ■   ■   ■   ·   ·   · |  b  ]
query t3           [ ·   ■   ■   ■   ·   · |  b  ]
query t4           [ ·   ·   ■   ■   ■   · |  b  ]
query t5           [ ·   ·   ·   ■   ■   ■ |  b  ]

每一行：token 的权重 + sink 的权重 = 1；sink 没有 value，分给它的部分不产生输出
```

**算一行（query t5）**：对 t3、t4、t5 的分数是 2.0、1.0、0.5，b = 1.5。

| | t3 | t4 | t5 | sink | token 权重之和 |
|---|---:|---:|---:|---:|---:|
| 没有 sink | 0.63 | 0.23 | 0.14 | — | 1.00 |
| 有 sink | 0.46 | 0.17 | 0.10 | 0.28 | 0.72 |

有 sink 时这个 query 只把 72% 给了真实 token，剩下 28% 给 sink、什么也没取回来，输出整体变小；token 之间的相对比例不变。
sink 实际分到多少，取决于 b_h 和当前 query 对其他 token 的分数的相对大小：s_j 都低时 sink 分得多，某个 s_j 很高时 sink 分得少。

---

## 3. Megatron 里的实现

`megatron/core/transformer/dot_product_attention.py`：

```python
if config.softmax_type == "learnable":
    self.softmax_offset = nn.Parameter(torch.empty(num_attention_heads_per_partition))  # [n_heads]，即 b_h
    self.softmax_offset = config.init_method(self.softmax_offset)                       # 和其他权重一样初始化
elif config.softmax_type == "off-by-one":
    self.softmax_offset = torch.zeros(num_attention_heads_per_partition)                # 固定为 0，不是参数
```

`megatron/core/fusions/fused_softmax.py` 的 `SoftmaxOne`（PyTorch 参考实现）：**先**加 causal / 滑动窗口 mask，**再**把 sink
拼成最后一列，在 sk + 1 列上做 softmax，然后丢掉 sink 列：

```python
mask_output = mask_func(scores, mask)                 # [b, h, sq, sk]，被 mask 的位置 = -inf
sink = b.reshape(1, -1, 1, 1).expand(b_sz, -1, sq, 1)  # [b, h, sq, 1]，每个 query 同一个 b_h
qk = torch.cat([mask_output, sink], dim=-1)            # [b, h, sq, sk+1]
probs = torch.softmax(qk, dim=-1)[..., :-1]            # 丢掉 sink 列
```

- sink 是在 mask **之后**拼上的，所以永远可见。
- b_h 只通过 softmax 的分母得到梯度，和其他参数一起训练。
- 我们的训练走 Transformer Engine 的 fused attention，把 `softmax_offset` 传进 kernel，数学相同；
  因为要支持这个 offset，TE 会**关掉 FlashAttention**（训练日志里有这条提示）。

---

## 4. sink 和 SWA 窗口长度

两者解决不同的问题：

| | 解决什么 | 不能解决什么 |
|---|---|---|
| sink | 窗口里没有值得看的 token 时，注意力有地方放空；避免模型把第一个 token 当"垃圾桶"（SWA 下第一个 token 很快滑出窗口） | 看到窗口以外的内容 |
| 长窗口 | 每层直接看到更远的上下文 | — |

- **短窗口需要 sink**：窗口里常常没有可看的 token，没有 sink 只能硬塞给窗口里某个 token。
- gpt-oss 能用 128 的窗口，靠两样一起：sink 让短窗口稳定，**每两层一层 full attention** 负责远距离信息。
- full attention 层很少时（如 Qwen3 梯子的 SWA 对照：5 层 SWA 4096 : 1 层 full NoPE），远距离信息主要靠窗口本身，窗口不能太短。
- 可以验证的实验：在 Qwen3 梯子的 SWA 对照里加一个"短窗口（128 / 512）+ learnable sink"，和现在的"4096、无 sink"比较。

---

## 5. 我们的实验里用到的情况（2026-10-20 核对）

| 系列 | `softmax_type` | SWA | 依据 |
|---|---|---|---|
| dpl m 系列（`oss_moe_m0`–`m4`，MiMo / Qwen3.8 版） | `learnable` | 窗口 128，SWA 与 full 1:1 交替 | 模型 yaml；运行中的 run 的 `argv.txt` 有 `--softmax-type learnable`；训练日志 `softmax_type .... learnable` |
| Boxiang gpt-oss 梯子（g0–g6） | `learnable` | 窗口 128，1:1 交替 | `configs/models/gptoss_g*.yaml`；启动层可用 `SOFTMAX_TYPE=off` 关掉做对照 |
| Boxiang Qwen3 梯子（q0–q6） | `vanilla`（无 sink） | baseline 无 SWA（全 full attention）；SWA 对照为窗口 4096、5:1、full 层 NoPE | 设计文档 v1 / v1.1；`qwen3_moe_q*.yaml` |

开销：每层的 sink 参数数 = head 数（m1、g1 每层 24 个）。
