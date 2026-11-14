# QImageBench 打分汇总（2026-11-08）

汇总 QImageBench 上 6 个公开模型和 2 个 Cosmos3 checkpoint 的打分结果。
同一批公开模型在 UGB 上的结果见 [ugb_baseline_summary_20261023.md](../ugb_baseline/ugb_baseline_summary_20261023.md)，模型规模和 upsampler 见 [ugb_public_t2i_models_20261022.md](../ugb_public_t2i_models_20261022.md)。

- Prompt：arena1000（1000 条）+ cosmos500（500 条），共 1500 条，每条生成 1 张图。arena1000 另有 OCR 标签：350 条要求在图里写字（`arena1000_ocr`），650 条不要求（`arena1000_noocr`）。manifest 在 `s3://nv-00-10206-vfm/debug/xingqianx/evaluation/qimagebench/`（`arena1000.json`、`cosmos500.json`）。
- 结果目录（每组的结果文件是目录下的 `qimagebench_result.json`）：
  - 公开模型：`s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/qimagebench_baseline/default/<run>/`
  - Cosmos3：`s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/qimagebench/c3json_opus4p7/<run>/`
- 分数范围 0–100。

---

## 0. 读表前必须知道的四点

1. **生成 prompt 不同，打分 prompt 相同。** 公开模型吃原版 prompt，其中 FLUX.3、Fibo、Ideogram、Qwen-Image 还按各家官方流程改写 prompt；Cosmos3 吃的是 Opus 4.7 按 Cosmos3-Super-Text2Image 官方模板改写的 JSON prompt（`c3json_opus4p7`）。打分时一律用原版 prompt。所以这是"改写 + 生成"的系统级对比，分辨率也不统一（见第 5 节）。
2. **overall 是两个子集的等权平均。** overall = (arena1000 + cosmos500) / 2，不是 1500 张图直接平均；L1、L2 各列同理。
3. **Judge 与计分。** Judge 是 Q-Judger（`Qwen/Qwen-Image-Bench`，基于 Qwen3.6-27B），用 vLLM 部署在 Lepton，评测协议从官方 repo 原样移植。
   - judge 给每个 L3 检查项打 0 / 1 / 2，映射成 0 / 60 / 100；L3 取平均得 L2，L2 取平均得 L1，L1 取平均得这张图的分数。
   - Quality、Aesthetics、Alignment 每条都考。Real-world Fidelity（RWF）和 Creative Generation（CG）只考一部分 prompt，由 opus-5.5、kimi-k3、gpt-6-astra 三个模型标注：RWF 任一模型选中就考，CG 按多数票。arena1000 中考 RWF、CG 的分别有 513、613 条，cosmos500 中分别有 373、189 条。标注方法和与官方标签的一致性见 [DIMS_REPORT.md](../../scripts/t2i_eval/blend/DIMS_REPORT.md)。
4. **所有组都是 1500/1500 张打了分。** 有 4 张图的 Quality 维度 judge 输出没能解析（status=failed），这几张只有 Quality 不计入，其余维度照常计分：Nano Banana 2.1 1 张（`cosmos500_0433`），Cosmos3 sft0 UHQ 2 张（`arena1000_0691`、`cosmos500_0213`），Cosmos3 exp009c 1 张（`arena1000_0591`）。

---

## 1. 总分

| 模型 | overall | arena1000 | cosmos500 | 分辨率 | run |
|---|---|---|---|---|---|
| GPT-image-2.5 sunburst，quality low | **62.53** | 63.73 | 61.34 | 1024² | `gpt_image_2p5_sunburst_quality_low` |
| Nano Banana 2.1 | 61.04 | 62.55 | 59.52 | 1024² | `nano_banana_2p1` |
| Qwen-Image 2.1，pe | 59.33 | 60.99 | 57.66 | 2048² | `qwen_image_2p1_pe` |
| Ideogram 4 nf4（9.3B×2，48 步） | 58.51 | 60.27 | 56.75 | 2048² | `ideogram4_nf4` |
| FLUX.3 klein-base（5.6B，50 步，cfg 3.5） | 58.24 | 60.05 | 56.42 | 1024² | `flux3_klein_base` |
| Fibo 2.0 dev（8.1B，50 步，cfg 5.0） | 57.63 | 59.41 | 55.84 | 1024² | `fibo2_dev` |
| **Cosmos3 64bm32b dense，exp009 sft0 UHQ，iter31000** | **55.08** | 57.44 | 52.72 | 1024² | `exp009_sft0_uhq_iter31k` |
| Cosmos3 64bm32b dense，exp009c union5，iter100 | 54.10 | 56.15 | 52.06 | 1024² | `exp009c_x9c_iter100` |

观察：
- **API 模型领先。** GPT-image-2.5 sunburst 最高（62.53），Nano Banana 2.1 第二（61.04）。开源模型里 Qwen-Image 2.1 pe 最好（59.33），其后 Ideogram 4 nf4（58.51）、FLUX.3 klein-base（58.24）、Fibo 2.0 dev（57.63）三个相差不到 0.9。
- **Cosmos3 sft0 UHQ 低于所有公开模型：比 Fibo 2.0 dev 低 2.5，比 FLUX.3 klein-base 低 3.2，比 Qwen-Image 2.1 低 4.2，比 GPT-image-2.5 低 7.5。**
- **exp009c iter100 比 sft0 UHQ 低 1.0**，主要低在 Creative Generation（−3.4）。
- 所有模型在 cosmos500 上都比 arena1000 低（2.4–4.7 分），Cosmos3 sft0 UHQ 掉得最多（−4.7）。

---

## 2. L1 维度（overall）

| 模型 | overall | Quality | Aesthetics | Alignment | Real-world Fidelity | Creative Generation |
|---|---|---|---|---|---|---|
| GPT-image-2.5 sunburst，low | **62.53** | **58.16** | **67.68** | **62.33** | **56.57** | **70.96** |
| Nano Banana 2.1 | 61.04 | 57.72 | 65.71 | 60.69 | 54.12 | 68.53 |
| Qwen-Image 2.1，pe | 59.33 | 57.56 | 64.83 | 56.76 | 53.80 | 64.36 |
| Ideogram 4 nf4 | 58.51 | 57.36 | 62.46 | 57.13 | 53.40 | 63.14 |
| FLUX.3 klein-base | 58.24 | 56.32 | 63.07 | 57.37 | 51.65 | 63.35 |
| Fibo 2.0 dev | 57.63 | 56.19 | 62.71 | 56.79 | 50.87 | 59.26 |
| Cosmos3 sft0 UHQ iter31000 | 55.08 | 53.19 | 61.16 | 55.30 | 45.97 | 55.59 |
| Cosmos3 exp009c iter100 | 54.10 | 52.58 | 60.78 | 54.56 | 44.82 | 52.18 |
| *Cosmos3 sft0 UHQ − FLUX.3* | *−3.16* | *−3.13* | *−1.91* | *−2.07* | *−5.69* | *−7.76* |

观察：
- 和 FLUX.3 klein-base 比，**Cosmos3 sft0 UHQ 差得最多的是 Creative Generation（−7.8）和 Real-world Fidelity（−5.7）**；Aesthetics、Alignment 只差约 2。
- GPT-image-2.5 在 5 个 L1 上全部第一。
- Ideogram 4 nf4 的 5 个 L1 和 FLUX.3 相差都不到 1.8。Fibo 2.0 dev 只有 Creative Generation 明显偏低（比 FLUX.3 低 4.1，公开模型里最低），其余 4 个 L1 和 FLUX.3 相差不到 0.8。

---

## 3. L2 维度（overall）

每张表最后一行是 Cosmos3 sft0 UHQ 减 FLUX.3 klein-base。

**Quality**

| 模型 | Detail | Realism | Resolution |
|---|---|---|---|
| GPT-image-2.5 sunburst，low | 61.25 | 49.68 | 62.64 |
| Nano Banana 2.1 | 61.25 | 49.53 | 61.72 |
| Qwen-Image 2.1，pe | 60.80 | 49.84 | 61.64 |
| Ideogram 4 nf4 | 61.03 | 48.53 | 61.84 |
| FLUX.3 klein-base | 59.58 | 47.27 | 61.31 |
| Fibo 2.0 dev | 59.27 | 47.46 | 61.30 |
| Cosmos3 sft0 UHQ iter31000 | 55.71 | 41.82 | 61.10 |
| Cosmos3 exp009c iter100 | 54.84 | 41.14 | 60.82 |
| *sft0 UHQ − FLUX.3* | *−3.86* | *−5.45* | *−0.21* |

**Aesthetics**

| 模型 | Anatomical Portraiture | Color Harmony | Composition | Emotional Expression | Lighting | Style Control |
|---|---|---|---|---|---|---|
| GPT-image-2.5 sunburst，low | 25.07 | 74.01 | 68.38 | 64.32 | 74.02 | 75.40 |
| Nano Banana 2.1 | 21.88 | 71.89 | 66.13 | 63.71 | 71.96 | 73.29 |
| Qwen-Image 2.1，pe | 21.90 | 71.16 | 65.05 | 62.96 | 73.33 | 67.62 |
| Ideogram 4 nf4 | 18.37 | 69.26 | 63.01 | 60.57 | 66.87 | 69.81 |
| FLUX.3 klein-base | 13.44 | 70.22 | 63.80 | 61.19 | 69.28 | 68.91 |
| Fibo 2.0 dev | 12.21 | 70.06 | 62.38 | 61.56 | 70.62 | 65.36 |
| Cosmos3 sft0 UHQ iter31000 | 11.09 | 69.76 | 61.75 | 60.83 | 68.31 | 59.78 |
| Cosmos3 exp009c iter100 | 10.72 | 69.61 | 62.13 | 61.01 | 68.78 | 54.84 |
| *sft0 UHQ − FLUX.3* | *−2.35* | *−0.46* | *−2.05* | *−0.36* | *−0.97* | *−9.12* |

**Alignment**

| 模型 | Actions | Attributes | Layout | Relations | Scene |
|---|---|---|---|---|---|
| GPT-image-2.5 sunburst，low | 52.75 | 62.41 | 61.33 | 61.14 | 71.44 |
| Nano Banana 2.1 | 49.95 | 61.13 | 59.10 | 57.87 | 72.06 |
| Qwen-Image 2.1，pe | 45.92 | 57.60 | 55.51 | 53.44 | 69.19 |
| Ideogram 4 nf4 | 46.06 | 58.03 | 56.89 | 54.07 | 68.10 |
| FLUX.3 klein-base | 46.66 | 59.73 | 57.45 | 53.43 | 67.14 |
| Fibo 2.0 dev | 46.34 | 58.91 | 56.64 | 51.26 | 69.13 |
| Cosmos3 sft0 UHQ iter31000 | 45.05 | 57.76 | 56.35 | 49.90 | 66.16 |
| Cosmos3 exp009c iter100 | 44.57 | 57.24 | 55.70 | 48.15 | 66.51 |
| *sft0 UHQ − FLUX.3* | *−1.61* | *−1.97* | *−1.09* | *−3.53* | *−0.98* |

**Real-world Fidelity**

| 模型 | Fairness | Safety & Compliance | World Knowledge |
|---|---|---|---|
| GPT-image-2.5 sunburst，low | 60.04 | 59.88 | 51.67 |
| Nano Banana 2.1 | 60.07 | 59.94 | 46.20 |
| Qwen-Image 2.1，pe | 59.88 | 59.88 | 45.59 |
| Ideogram 4 nf4 | 60.00 | 59.71 | 45.30 |
| FLUX.3 klein-base | 60.08 | 59.94 | 40.73 |
| Fibo 2.0 dev | 60.00 | 59.94 | 38.83 |
| Cosmos3 sft0 UHQ iter31000 | 60.08 | 59.94 | 28.18 |
| Cosmos3 exp009c iter100 | 60.00 | 59.82 | 25.43 |
| *sft0 UHQ − FLUX.3* | *0.00* | *0.00* | *−12.55* |

**Creative Generation**

| 模型 | Design Applications | Feature Matching | Imagination | Logical Resolution | Text Rendering | Visual Storytelling |
|---|---|---|---|---|---|---|
| GPT-image-2.5 sunburst，low | 78.97 | 56.18 | 70.48 | 43.16 | 86.13 | 72.28 |
| Nano Banana 2.1 | 76.69 | 52.86 | 68.50 | 43.56 | 79.05 | 71.53 |
| Qwen-Image 2.1，pe | 73.80 | 49.37 | 65.05 | 31.74 | 82.53 | 65.86 |
| Ideogram 4 nf4 | 72.16 | 50.94 | 63.75 | 31.40 | 77.03 | 65.05 |
| FLUX.3 klein-base | 71.22 | 46.43 | 62.44 | 32.19 | 80.46 | 69.33 |
| Fibo 2.0 dev | 67.43 | 46.60 | 61.70 | 26.42 | 71.07 | 66.41 |
| Cosmos3 sft0 UHQ iter31000 | 65.19 | 46.20 | 64.81 | 24.01 | 58.87 | 67.51 |
| Cosmos3 exp009c iter100 | 62.91 | 40.00 | 63.70 | 18.96 | 55.07 | 65.85 |
| *sft0 UHQ − FLUX.3* | *−6.03* | *−0.23* | *+2.38* | *−8.17* | *−21.58* | *−1.82* |

观察：
- **差距集中在少数几个 L2。** 和 FLUX.3 比，Cosmos3 sft0 UHQ 在 23 个 L2 中有 15 个差距在 2.4 分以内，其中 Imagination 还高 2.4。差距大的是 Text Rendering（−21.6）、World Knowledge（−12.5）、Style Control（−9.1）、Logical Resolution（−8.2）、Design Applications（−6.0）、Realism（−5.4）。
- 和 GPT-image-2.5 比，Text Rendering 差 27.3（58.87 vs 86.13），World Knowledge 差 23.5（28.18 vs 51.67）。
- **Real-world Fidelity 的差距全部来自 World Knowledge。** Fairness 和 Safety & Compliance 所有模型都在 60 左右，区分不出模型。
- Anatomical Portraiture 所有模型都很低（10.7–25.1）。
- exp009c iter100 比 sft0 UHQ 低得最多的是 Feature Matching（−6.2）、Logical Resolution（−5.1）、Style Control（−4.9）、Text Rendering（−3.8）。
- 新加的两个开源模型：Fibo 2.0 dev 和 FLUX.3 差距最大的两项是 Text Rendering（−9.4）和 Logical Resolution（−5.8）；Ideogram 4 nf4 的 World Knowledge（45.30）比 FLUX.3 高 4.6，Anatomical Portraiture 高 4.9。

---

## 4. 子集：要求写字的题与 cosmos500

| 模型 | arena1000_ocr（350） | arena1000_noocr（650） | cosmos500（500） |
|---|---|---|---|
| GPT-image-2.5 sunburst，low | **70.07** | **60.31** | **61.34** |
| Nano Banana 2.1 | 68.26 | 59.48 | 59.52 |
| Qwen-Image 2.1，pe | 66.95 | 57.78 | 57.66 |
| Ideogram 4 nf4 | 65.61 | 57.39 | 56.75 |
| FLUX.3 klein-base | 66.10 | 56.80 | 56.42 |
| Fibo 2.0 dev | 63.68 | 57.11 | 55.84 |
| Cosmos3 sft0 UHQ iter31000 | 60.02 | 56.04 | 52.72 |
| Cosmos3 exp009c iter100 | 58.72 | 54.76 | 52.06 |
| *Cosmos3 sft0 UHQ − FLUX.3* | *−6.08* | *−0.75* | *−3.70* |

观察：
- **不要求写字的 arena 题上，Cosmos3 sft0 UHQ 和 FLUX.3 基本持平**（56.04 vs 56.80），Aesthetics 和 Creative Generation 还略高（+0.2、+0.8）；和 Fibo 2.0 dev、Ideogram 4 nf4 也只差 1.1、1.3。**要求写字的题上差 6.1。** arena1000 上的差距主要来自写字的题。
- 公开模型在写字题上的分数比非写字题高 6.6–9.8（最少的是 Fibo 2.0 dev），Cosmos3 只高 4.0。
- **cosmos500 上差 3.7**，主要差在 Creative Generation（−10.9）和 Real-world Fidelity（−6.7）。

---

## 5. 生成设置

| 模型 | 分辨率 | 生成 prompt / upsampler | 其他 |
|---|---|---|---|
| GPT-image-2.5 sunburst | 1024² | 原版 prompt，API 内部处理 | `quality=low` |
| Nano Banana 2.1 | 1024² | 原版 prompt，API 内部处理 | gateway `gcp/google/gemini-nano-banana-2.1` |
| Qwen-Image 2.1，pe | 2048² | 原版 prompt + 官方 prompt enhancement（pe） | — |
| FLUX.3 klein-base | 1024² | Claude Opus 4.8 + BFL 官方 system prompt | 50 步，cfg 3.5 |
| Fibo 2.0 dev | 1024² | bria-vgl + gemini-3.5-flash-lite | 50 步，cfg 5.0，官方 JSON negative prompt 和字体 |
| Ideogram 4 nf4 | 2048² | Claude Opus 4.8 + 官方 `v1.txt` system prompt | pipeline 默认 48 步；1 条改写结果超过 2048 token 上限，截断到 2040 |
| Cosmos3 sft0 UHQ / exp009c | 1024² | `c3json_opus4p7`：Opus 4.7 + Cosmos3-Super-Text2Image 官方模板（768、1:1） | cfg 4，50 步，cosmos3 negative prompt |

Cosmos3 checkpoint（都在 `s3://nv-00-10206-checkpoint/cosmos3_vfm/cosmos3_ga_t2ionly/` 下）：
- sft0 UHQ：`cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5/checkpoints/iter_000031000`，用 EMA 权重。
- exp009c：`cosmos3_ga_64bm32b_t2ionly_exp009c_union5_from_frozen_midtrain_lr5em5/checkpoints/iter_000000100`。

改写模型的替补（gpt-6-astra）：
- `cosmos500_0046`（茶园里的喷洒无人机）：Anthropic 模型对这条 prompt 一直返回空回复，推测是被安全分类器拦了。所以 FLUX.3、Ideogram 和 `c3json_opus4p7` 的这一条都改用 gpt-6-astra 改写。
- Fibo 2.0 dev 有 5 张是 gpt-6-astra 改写的版本：`arena1000_0088`、`arena1000_0256`、`arena1000_0308`、`cosmos500_0110`、`cosmos500_0238`；其余 1495 张用 gemini。要不要换回 gemini 重新生成，还没定。
