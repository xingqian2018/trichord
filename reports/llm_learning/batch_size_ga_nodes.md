# Batch size、梯度累积、节点数怎么选（2026-10-20）

> 本文讲 scaling ladder 里每个实验的 GBS、micro batch（mbs）、节点数、梯度累积（GA）、EP 是怎么定的，以及它们哪些影响结果、哪些只影响速度。
> 例子来自 dpl 的 MiMo KD 实验、Qwen3-MoE CE 梯子（q0 / q1），以及 Boxiang 的 gpt-oss 梯子（g0–g6）。

---

## 核心结论

- **GBS 是实验设计的一部分**，所有档固定 2^21 token = 256 条 × 8192；改了就是另一个实验。
- **mbs、节点数、GA、EP 只影响速度和显存，不影响 loss**（同一个 GBS 下数学上是同一个更新）。
- 选法：**先定 mbs（显存放得下尽量大）→ 再定节点数（想多快、排得到多少）→ GA 是算出来的**。
- 节点数有上限：GA 最小是 1，所以 DP ≤ 256 / mbs。

---

## 1. 关系式

```
GBS（每步序列数）= mbs × DP × GA
DP（数据并行份数）  = 节点数 × 4 GPU / (TP × PP × CP)      # 我们 TP = PP = CP = 1，所以 DP = GPU 数
GA                = GBS / (mbs × DP)                       # 必须是整数
```

| 量 | 谁决定 | 影响结果吗 |
|---|---|---|
| **GBS** | 实验设计：每档都是 2^21 token = 256 × 8192 | **影响**（Boxiang 专门做过 GBS 2^20 的对照） |
| **mbs**（每张 GPU 每次前向的序列数） | 显存和吞吐 | 不影响 |
| **节点数** | 想多快跑完、能排到多少节点 | 不影响 |
| **GA**（梯度累积步数） | 由上面三个算出来，不单独选 | 不影响 |
| **EP**（expert 并行） | expert 太多单卡放不下时才开 | 不影响 |

---

## 2. 选的顺序

1. **mbs**：显存放得下就尽量大，一般 4。模型大了降到 2 或 1，主要是因为 15 万词表的 logits 缓冲：mbs 4 时 fp32 logits 约 20 GB。
2. **节点数**：节点越多 GA 越小、越快，直到 GA = 1。上限：

   ```
   最大 DP = 256 / mbs     →  mbs 4：最多 64 张 GPU = 16 个节点
                              mbs 2：最多 128 张 GPU = 32 个节点
                              mbs 1：最多 256 张 GPU = 64 个节点
   ```

   想用更多节点只能降 mbs。Boxiang 的 g5 / g6 用 mbs 1、64 个节点，就是 GBS 256 允许的最宽布局。
3. **GA**：GA = 256 / (mbs × DP)，必须是整数。
4. **EP**：必须同时整除 expert 数和 DP。

小模型通常不开到节点上限：本来就跑得快，节点多了难排队，通信占比也变高。

---

## 3. 实际例子

| 实验 | 节点 | DP | mbs | mbs × DP | GA | EP |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-MoE q0（CE，`qwen3moe_q0_ce_adamw_wsd_tpp200`） | 4 | 16 | 4 | 64 | 4 | 1 |
| Qwen3-MoE q1（CE，`qwen3moe_q1_ce_adamw_wsd_tpp200`） | 8 | 32 | 4 | 128 | 2 | 1 |
| MiMo m1 tpp200（dp64 / dp128） | 8 | 32 | 4 | 128 | 2 | 1 |
| Boxiang g4 | 32 | 128 | 2 | 256 | 1 | 4 |
| Boxiang g5 / g6 | 64 | 256 | 1 | 256 | 1 | 8 |

---

## 4. 两个额外的约束

**评测的 DP 下限**

- packed 评测（Megatron `--full-validation`）把第 i 个样本分给第 i mod DP 个 rank：每个验证集（dpl 里按 suite 合并后）的样本数必须 ≥ DP，否则崩溃。DP 越大越容易碰到。
- 文档隔离评测会把样本数补齐到 DP 的整数倍，没有这个限制。

**dpl 的 run hash**

| 配置位置 | 参与 run hash | 改了之后 |
|---|---|---|
| `resources`（节点数、walltime、QOS） | 否 | 可以直接续跑同一个 run |
| `budget`（GBS、mbs） | 是 | 变成新 run（即使 mbs 不影响结果） |
| `parallel`（TP、PP、EP…） | 是 | 变成新 run |

所以 Boxiang 在 dpl 里为不同 mbs 定义了不同的 budget（如 `tpp200_mb2`、`tpp200_mb1`）。
