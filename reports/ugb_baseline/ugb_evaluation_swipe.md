# UGB 推理设置扫描：shift / CFG 区间 / seed（2026-10-23）

目的：确定 Cosmos3.5（60bm30b Super 系列）在 UGB 上的分数变化里，哪些来自推理设置、哪些来自模型本身，给后续评测（包括 Ultra 470bm235b）定一个默认推理配置。

数据来源：
- 分数：S3 上各目录的 `unigenbench_result.json`
- 生成设置：各目录的 `config.json`
- 逐步消融的原始记录：[score_report_20261020_dataset_ablation/ugb.md](../score_report_20261020_dataset_ablation/ugb.md)

全部汇总见 [ugb_baseline_summary_20261023.md](ugb_baseline_summary_20261023.md)。

---

## 0. 结论

1. **全程 CFG 是唯一稳定的提升。** 把 CFG 区间从 `[400, 1000]` 改成每一步都做，三个 checkpoint 都涨分（+0.39 到 +0.82），六组对比全部为正，平均 +0.48。
2. **shift 3 → 6 没有作用。** 三个 checkpoint 上是 −0.33 到 +0.11，平均约 0，都在 judge 噪声（约 0.35）以内。
3. **seed 0 三次都为正（+0.18 到 +0.52），但不应当作模型改进。** seed 只决定采样起点，单次增益也没明显超出噪声。
4. **推理设置能带来的总提升有限。** 原设置到最好的设置大约 +0.6 到 +1.1；同一采样设置下，SFT 数据配方（getty 对 pretrain）的差距是 +1.0 到 +1.55，而且在每种采样设置下都成立。
5. **建议的默认评测配置**：shift 3 或 6 都可以，**全程 CFG**，行号 seed，guidance 4，noneg。对外报"完整 recipe"（shift 6 + 全程 CFG + seed 0）时，要单独标明用了 seed 0。

---

## 1. 变量与对应脚本

所有脚本都在 `projects/cosmos3/cosmos3/evaluation/text_to_image/` 下。原版 `inference_ugb_i4.py` 不改，变体通过子类覆盖 `shift` 和 `guidance_interval`。

| 设置 | 脚本 | shift | CFG 区间 | seed |
|---|---|---|---|---|
| 原设置 | `inference_ugb_i4.py` | 3 | [400, 1000] | 行号 |
| 只改 CFG | `inference_ugb_i4_20261020_shift3_fullcfg.py` | 3 | 全程 | 行号 |
| 只改 shift | `inference_ugb_i4_20261020.py` | 6 | [400, 1000] | 行号 |
| shift + CFG | `inference_ugb_i4_20261020_fullcfg.py` | 6 | 全程 | 行号 |
| 完整 recipe | `inference_ugb_i4_20261020_fullcfg_seed0.py` | 6 | 全程 | 恒为 0 |

guidance 4、不带 negative prompt 都是默认设置，本报告不扫这两个参数。

分辨率约定：SFT 前的 checkpoint（`60bm30b_v2p5_pretrain/iter_000741260`）用 960×960 + `cosmos3p5_ga_60bm30b_base`；SFT checkpoint（exp003_*）用 1024×1024 + `base_768`。

所有生成都用 hamid benchmark（GPT-5.5 改写的 prompt），打分用原版 `v2_1170L` 的 testpoint，judge 是 gemini-3.1-pro，50 步，EMA。

---

## 2. shift / CFG 区间 / seed 逐步消融

### 2.1 SFT 前 Super（`iter_000741260`，960，noneg，guidance 4）

| 设置 | all | orig | phi |
|---|---|---|---|
| shift 3，[400, 1000]，行号 seed（原设置） | 86.10 | 88.01 | 84.34 |
| shift 3，全程 CFG，行号 seed | 86.72 | 88.24 | 85.33 |
| shift 6，[400, 1000]，行号 seed | 86.21 | 87.88 | 84.67 |
| shift 6，全程 CFG，行号 seed | 87.03 | 88.83 | 85.37 |
| shift 6，全程 CFG，seed 0（完整 recipe） | **87.21** | **89.18** | **85.40** |

### 2.2 exp003_000 pretrain（`iter_000025000`，1024，noneg，guidance 4）

| 设置 | all | orig | phi |
|---|---|---|---|
| shift 3，[400, 1000]，行号 seed（原设置） | 86.29 | 88.83 | 83.97 |
| shift 3，全程 CFG，行号 seed | 86.37 | 88.72 | 84.20 |
| shift 6，[400, 1000]，行号 seed | 86.00 | 88.88 | 83.36 |
| shift 6，全程 CFG，行号 seed | 86.61 | 89.18 | 84.25 |
| shift 6，全程 CFG，seed 0（完整 recipe） | **86.87** | 88.83 | **85.07** |

### 2.3 exp003_002 getty（`iter_000025000`，1024，noneg，guidance 4）

| 设置 | all | orig | phi |
|---|---|---|---|
| shift 3，[400, 1000]，行号 seed（原设置） | 87.56 | 89.80 | 85.51 |
| shift 3，全程 CFG，行号 seed | 87.92 | 90.18 | 85.84 |
| shift 6，[400, 1000]，行号 seed | 87.23 | 89.57 | 85.09 |
| shift 6，全程 CFG，行号 seed | 87.62 | 89.69 | 85.72 |
| shift 6，全程 CFG，seed 0（完整 recipe） | **88.14** | **90.26** | **86.19** |

### 2.4 每个变量单独的贡献（all）

| 改动（其他不变） | SFT 前（960） | exp003_000（1024） | exp003_002（1024） | 判断 |
|---|---|---|---|---|
| shift 3 → 6（[400, 1000]，行号 seed） | +0.11 | −0.29 | −0.33 | 无作用 |
| CFG [400, 1000] → 全程（shift 6，行号 seed） | **+0.82** | **+0.61** | **+0.39** | 稳定提升 |
| 行号 seed → seed 0（shift 6，全程 CFG） | +0.18 | +0.26 | +0.52 | 偏正，但不算模型能力 |
| 原设置 → 完整 recipe | +1.11 | +0.58 | +0.58 | |

### 2.5 shift × CFG 区间 2×2（行号 seed，all）

| checkpoint | shift 3 + [400, 1000] | shift 3 + 全程 | shift 6 + [400, 1000] | shift 6 + 全程 |
|---|---|---|---|---|
| SFT 前（960） | 86.10 | 86.72 | 86.21 | 87.03 |
| exp003_000 pretrain（1024） | 86.29 | 86.37 | 86.00 | 86.61 |
| exp003_002 getty（1024） | 87.56 | 87.92 | 87.23 | 87.62 |
| getty − pretrain | +1.27 | +1.55 | +1.23 | +1.01 |

- 全程 CFG 对比 [400, 1000]：shift 3 下 +0.62 / +0.08 / +0.36，shift 6 下 +0.82 / +0.61 / +0.39。六组全部为正，平均 +0.48。
- shift 3 对比 6：有正有负，平均约 0。
- getty 对比 pretrain：每个格子里都是 +1.0 到 +1.55。SFT 数据带来的提升比推理设置更大，而且不依赖采样设置。

---

## 3. 生产配置复现（ruofan）

用的是生产实验配置（`cosmos3p5_ga_60bm30b_v2_pretrain` 模块，`configs/base/config.py`），960×960，shift 6，带 default negative prompt，同一份 gpt55_a2 prompt。

| run | guidance | all（我们打分） | all（他自己打分） |
|---|---|---|---|
| `c3p5_super_v2p5_iter741260_ruofan_g4_shift6_negdefault` | 4 | 86.79 | 86.90 |
| `c3p5_super_v2p5_iter741260_ruofan_g6_shift6_negdefault` | 6 | **87.22** | 87.23 |

- 生产配置 g6 得 87.22，和我们的完整 recipe（g4、noneg、`inference_helper` 推理路径）的 87.21 接近。但生产配置在 g4 下只有 86.79，比 recipe 低 0.42。两边的差别有三处：带不带 neg、代码路径不同、CFG 区间可能也不一样（生产配置的 `config.json` 没记录 CFG 区间）。所以这两组只能说明分数在同一个水平，不能证明推理路径完全等价。
- 同一批图打两次分只差 0.01 到 0.11。

---

## 4. 噪声参考

| 来源 | 差距 |
|---|---|
| 同一批图、同一个 judge 再打一次（原始消融报告） | 约 0.35 |
| 同一批图，`gemini-3.1-pro` 和 `gemini-3.1-pro@nvidia` 两种写法各打一次（ruofan 两组） | 0.01 到 0.11 |
| judge 换成 GPT-5.5（同一批图） | **+2.6**（系统性偏高），不能和 Gemini 的分数混着比 |

判断标准：单个设置的差距在 0.35 以内的，不应解读为有效。

---

## 5. 对 Ultra（470bm235b）评测的建议

按上面的结论，Ultra v3p4 `iter_000360000` 最有信息量的组合是：

| 优先级 | 设置 | 目的 |
|---|---|---|
| 1 | shift 3，[400, 1000]，行号 seed，noneg，cfg 4（原设置） | 和 Super 原设置（86.10）对齐 |
| 2 | shift 3 或 6，**全程 CFG**，行号 seed，noneg，cfg 4 | 验证全程 CFG 在 Ultra 上是否同样有效，对应 Super 的 86.72 / 87.03 |
| 3 | shift 6，全程 CFG，seed 0（完整 recipe） | 和 Super 的最好成绩（87.21）、生产配置（87.22）对齐 |

单独比 shift 3 和 shift 6 的价值不大，上面已经证明 shift 不起作用。

Ultra 推理需要用 `cosmos3p5_ga_470bm235b_base_cp1_for_eval`，即 CP=1。原来的 base 设了 `context_parallel_shard_degree=8`，在"每个 rank 跑不同 prompt"的推理脚本里会卡在 all-to-all 上。