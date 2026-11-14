# UGB 打分汇总（2026-10-23）

汇总 `unigenbench_for_cosmos3p5/` 下所有已打分的 UGB 结果，共 47 组。
公开模型的规模、语言塔、upsampler 等信息见 [ugb_public_t2i_models_20261022.md](../ugb_public_t2i_models_20261022.md)。

- S3 根目录：`s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/<benchmark>/<run>/`
- 每组的结果文件是该目录下的 `unigenbench_result.json`；少数目录另有带 signature 的结果，见第 5 节。
- 表中数字均为准确率（%）。`orig` 是原始 prompt 子集，`phi` 是另一半子集，`overall` 合并两者。

---

## 0. 读表前必须知道的三点

1. **生成 prompt 不同，打分 testpoint 相同。** 三个 benchmark 目录的区别只在**生成时**喂给模型的 prompt；**打分时**一律用原版 `v2_1170L` 的 testpoint。所以跨目录的分数差，混合了"prompt 改写带来的增益"。
   | 目录 | 生成用的 prompt |
   |---|---|
   | `v2_1170L` | 原版 UGB prompt（短） |
   | `v2_1170L_opus` | Opus 改写后的 prompt |
   | `v2_1170L_gpt5p5_cosmos3p5_dense_hamid` | GPT-5.5 改写的 dense prompt（与官方 `gpt55_a2/unigenbench-t2i.json` 逐字相同） |
2. **Judge。** 默认是 Gemini 3.1 Pro。`gemini-3.1-pro@nvidia` 和早期结果里写的 `gemini-3.1-pro` 是同一个模型，只是写法不同；同一批图分别用两种写法打分，差在 0.1 分以内（见第 5 节）。用 GPT-5.5 当 judge 的结果分数系统性偏高约 2.6 分，**不能和 Gemini judge 的结果混比**。
3. **缺图不计入分母。** API 和开源模型因为安全过滤或 upsampler 失败，少了 1–5 张图（`success` < 1170）。准确率只按实际打了分的样本算。

---

## 0b. Baseline 测试原则：每个模型用它最擅长的分辨率（2026-10-27 起）

公开模型的 baseline 一律按**各家官方推荐的 1:1 分辨率**生成，不再统一 1024²。宽高比固定 1:1；upsampler 仍按各家官方流程（见 [ugb_public_t2i_models_20261022.md](../ugb_public_t2i_models_20261022.md) 第 3 节）。

| 模型 | 分辨率 | 依据 |
|---|---|---|
| Krea 2 Turbo | **2048²** | 官方 codebase 推理示例 `--width 2048 --height 2048 --mu 1.15`；turbo checkpoint 固定 `mu=1.15`（= `max_shift`，对应 ≥ 6400 图像 token 的高分辨率）。diffusers 默认 1024 不是它的最佳分辨率 |
| Ideogram 4 nf4 | **2048²** | README：*"For the highest-quality images, set `--height 2048 --width 2048`"*；diffusers pipeline 默认 2048、48 步 |
| Qwen-Image 2.1（pe / nope） | **2048²** | README 的 1:1 尺寸即 2048×2048；pipeline 默认按此出图 |
| FLUX.3 klein / klein-base | 1024² | modular pipeline 默认 1024；README 没有更高的推荐分辨率 |
| Fibo 2.0 dev | 1024² | README：*"about one megapixel, the area Fibo 2.0 was trained at … `1024×1024` (the default)"* |
| GPT-image-2 / 2.5 sunburst | 1024² | API 标准尺寸 |
| Gemini 3.1 flash image | 1024²（`1k`） | `NvidiaGatewayImageGenerator` 目前只支持 `1k`；2K 是否原生生成未确认 |

**对本报告的影响：** 第 1 节的 UGB 结果是按旧做法生成的（除 Qwen-Image 2.1 外都是 1024²），所以 **Krea 2 Turbo 和 Ideogram 4 的分数可能被低估**。UGB 暂未按新原则重跑；新原则先用于 CVTG baseline。

---

## 1. 公开模型对比（`v2_1170L`，原版 prompt）

| 模型 | overall | orig | phi | success | judge | run |
|---|---|---|---|---|---|---|
| GPT-image-2.5 sunburst, quality high | **94.45** | 96.45 | 92.60 | 1165/1170 | gemini-3.1-pro@nvidia | `gpt_image_2p5_sunburst_quality_high` |
| GPT-image-2.5 sunburst, low（默认） | 94.22 | 95.89 | 92.69 | 1167/1170 | gemini-3.1-pro@nvidia | `gpt_image_2p5_sunburst` |
| GPT-image-2, low（默认） | 93.09 | 95.43 | 90.93 | 1168/1170 | gemini-3.1-pro@nvidia | `gpt_image_2` |
| GPT-image-2, quality high | 92.97 | 95.48 | 90.64 | 1166/1170 | gemini-3.1-pro@nvidia | `gpt_image_2_quality_high` |
| Gemini 3.1 flash image | 92.17 | 94.20 | 90.31 | 1169/1170 | gemini-3.1-pro@nvidia | `gemini_3_1_flash_image` |
| **cosmos3-super-text2image** | **91.63** | 93.85 | 89.59 | 1170/1170 | gemini-3.1-pro@nvidia | `c3_dense_exp009_sft0_iter31000` |
| FLUX.3 klein-base（5.6B，50 步，cfg 3.5） | 90.63 | 91.84 | 89.51 | 1167/1170 | gemini-3.1-pro@nvidia | `flux3_klein_base` |
| FLUX.3 klein（5.6B，蒸馏 16 步） | 89.72 | 91.71 | 87.89 | 1167/1170 | gemini-3.1-pro@nvidia | `flux3_klein` |
| **Cosmos3.5 Ultra v3p4 iter360000** 预训练 | **88.45** | 90.26 | 86.80 | 1170/1170 | gemini-3.1-pro@nvidia | `c3p5_ultra_v3p4_iter360000_shift3_fullcfg_seedrow_noneg` |
| Fibo 2.0 dev（8.1B） | 87.43 | 89.20 | 85.81 | 1167/1170 | gemini-3.1-pro@nvidia | `fibo2_dev` |
| Qwen-Image 2.1, nope | 86.99 | 90.71 | 83.57 | 1170/1170 | gemini-3.1-pro | `qwen_image_2p1_nope` |
| Ideogram 4（nf4，9.3B×2） | 86.19 | 86.89 | 85.54 | 1166/1170 | gemini-3.1-pro@nvidia | `ideogram4_nf4` |
| Qwen-Image 2.1, pe | 85.80 | 88.34 | 83.48 | 1170/1170 | gemini-3.1-pro | `qwen_image_2p1_pe` |
| Krea 2 Turbo（12.8B，8 步） | 85.60 | 87.19 | 84.13 | 1170/1170 | gemini-3.1-pro@nvidia | `krea2_turbo` |

观察：
- **闭源 API 模型整体领先。** GPT-image-2.5 sunburst 最高（94.2–94.5）。最弱的 API（Gemini 92.17）比最强的开源（FLUX.3 klein-base 90.63）高 1.5 分，最强的 API 比最弱的开源（Krea 85.60）高 8.9 分。
- **GPT-image 的 `quality=high` 在 UGB 上基本没有收益**：2.5 sunburst 高 0.2 分，2 反而低 0.1 分，都在噪声范围内。UGB 主要考 prompt 遵循，画质提升体现不出来。
- **开源模型里 FLUX.3 klein-base 最好**（90.6），比蒸馏版 klein 高 0.9。FLUX.3、Ideogram、Fibo 都走了 LLM upsampler（Claude Opus 4.8 / Gemini 3.5 flash-lite），分数里含 LLM 改写的贡献。
- Krea 2 Turbo 虽然 DiT 最大（12.8B），但只跑 8 步、没有 upsampler，排在最后。

---

## 2. Cosmos 与公开模型的同条件对比（`v2_1170L_opus`）

这个目录里 Cosmos 模型和 API 模型用的是**同一份 opus 改写 prompt**，是目前唯一能直接横比的地方。

| 模型 | run | overall | orig | phi | success | judge |
|---|---|---|---|---|---|---|
| GPT-image-2.5 sunburst（low） | `gpt_image_2p5_sunburst` | **94.69** | 96.03 | 93.45 | 1165/1170 | gemini-3.1-pro@nvidia |
| GPT-image-2（low） | `gpt_image_2` | 94.25 | 96.08 | 92.56 | 1165/1170 | gemini-3.1-pro@nvidia |
| Gemini 3.1 flash image | `gemini_3_1_flash_image` | 91.91 | 93.79 | 90.19 | 1169/1170 | gemini-3.1-pro@nvidia |
| Cosmos3 64bm32b dense，exp009 sft0，iter31000 | `c3_dense_exp009_sft0_iter31000` | 91.63 | 93.85 | 89.59 | 1170/1170 | gemini-3.1-pro@nvidia |
| Cosmos3 64bm32b dense，exp009 union5（UHQ SFT 前），iter25000 | `cosmos3_super_text2image_before_uhg` | 91.09 | 93.06 | 89.28 | 1170/1170 | gemini-3.1-pro@nvidia |
| Cosmos3 runlong RL，iter100 | `c3_runlong_rl_iter100` | 90.75 | 93.65 | 88.09 | 1170/1170 | gemini-3.1-pro@nvidia |
| Cosmos3 runlong RL，iter80 | `c3_runlong_rl_iter80` | 90.70 | 93.37 | 88.25 | 1170/1170 | gemini-3.1-pro@nvidia |
| Cosmos3.5 Super v2p5 iter741260（960²，cfg4，neg） | `c3p5_super_v2p5_iter741260_ts1000` | 87.79 | 89.52 | 86.22 | 1170/1170 | gemini-3.1-pro@nvidia |

观察：
- 同一份 prompt 下，**Cosmos3 dense SFT（91.6）已和 Gemini 3.1 flash image（91.9）持平**，比 GPT-image 低约 2.6–3 分。
- **Cosmos3.5 Super 预训练 checkpoint 比 Cosmos3 dense SFT 低 3.8 分。** 前者是未经 SFT 的 pretrain checkpoint；预期 SFT 之后会缩小差距（参考第 4 节 exp003 系列）。
- 对比第 1 节，opus 改写对 GPT-image-2 有 +1.2 的增益（93.09 → 94.25），对 2.5 sunburst 有 +0.5，对 Gemini 基本没有（92.17 → 91.91）。

---

## 3. Cosmos3.5 Super v2p5 iter741260 的推理设置扫描（hamid prompt）

所有行都是同一个 checkpoint，只改推理设置。run 名后缀的含义：
- `cfgN`：guidance N。没写就是 4。
- `noneg`：不用 cosmos3 negative prompt。
- `shiftN`：sampler shift，默认是 3。
- `fullcfg`：每一步都做 CFG，默认只在 [400, 1000] 区间做。
- `seed0` / `seedrow`：所有 prompt 同用 seed 0 / 每个 prompt 用自己的行号当 seed。
- `recipe` / `ruofan`：cosmos-benchmarks 官方 recipe 的复现。

| run | overall | orig | phi | success | judge |
|---|---|---|---|---|---|
| `c3p5_super_v2p5_iter741260_ruofan_g6_shift6_negdefault` | **87.22** | 89.16 | 85.44 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_super_v2p5_iter741260_recipe_shift6_fullcfg_seed0` | 87.21 | 89.18 | 85.40 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_super_v2p5_iter741260_shift6_fullcfg_seedrow_noneg` | 87.03 | 88.83 | 85.37 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_super_v2p5_iter741260_ruofan_g4_shift6_negdefault` | 86.79 | 88.83 | 84.93 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_super_v2p5_iter741260_shift3_fullcfg_seedrow_noneg` | 86.72 | 88.24 | 85.33 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_super_v2p5_iter741260_cfg8` | 86.70 | 88.34 | 85.19 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_super_v2p5_iter741260_cfg6_noneg` | 86.40 | 87.63 | 85.28 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_super_v2p5_iter741260_cfg8_noneg` | 86.37 | 87.63 | 85.21 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_super_v2p5_iter741260_shift6_cfg400to1000_seedrow_noneg` | 86.21 | 87.88 | 84.67 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_super_v2p5_iter741260`（cfg4，neg，shift3，即 baseline） | 86.13 | 88.32 | 84.13 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_super_v2p5_iter741260_noneg` | 86.10 | 88.01 | 84.34 | 1170/1170 | gemini-3.1-pro |

观察：
- **所有设置的差距只有约 1.1 分（86.1–87.2）。** 推理参数不是瓶颈。
- 最有效的是 **shift 6 加每步 CFG**（+0.9，87.0–87.2），与官方 recipe 一致。
- cfg 从 4 提到 8：+0.6（86.13 → 86.70）。去掉 negative prompt 基本没影响（±0.3）。
- cfg16 只生成了 160 张就取消了，没有打分。

---

## 4. Cosmos3.5 exp003 SFT 系列（hamid prompt，1024²）

| run | overall | orig | phi | success | judge |
|---|---|---|---|---|---|
| `c3p5_exp003_002_pretrain_getty_iter25000_1024_shift6_fullcfg_seed0_noneg` | **88.14** | 90.26 | 86.19 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_exp003_002_pretrain_getty_iter25000_1024_shift3_fullcfg_seedrow_noneg` | 87.92 | 90.18 | 85.84 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_exp003_003_pretrain_midtrain_lr2em5_iter25000_1024_shift3_fullcfg_seedrow_noneg` | 87.88 | 90.13 | 85.82 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_exp003_002_pretrain_getty_iter25000_recipe_shift6_fullcfg_seed0_noneg` | 87.88 | 90.13 | 85.82 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_exp003_002_pretrain_getty_iter25000_recipe_shift6_fullcfg_seedrow_noneg` | 87.67 | 89.92 | 85.61 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_exp003_002_pretrain_getty_iter25000_1024_shift6_fullcfg_seedrow_noneg` | 87.62 | 89.69 | 85.72 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_exp003_002_pretrain_getty_iter25000_noneg` | 87.56 | 89.80 | 85.51 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_exp003_002_pretrain_getty_iter25000_1024_shift6_cfg400to1000_seedrow_noneg` | 87.23 | 89.57 | 85.09 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_exp003_000_pretrain_iter25000_1024_shift6_fullcfg_seed0_noneg` | 86.87 | 88.83 | 85.07 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_exp003_000_pretrain_iter25000_1024_shift6_fullcfg_seedrow_noneg` | 86.61 | 89.18 | 84.25 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_exp003_001_pretrain_midtrain_iter25000_noneg` | 86.44 | 88.88 | 84.20 | 1170/1170 | gemini-3.1-pro |
| `c3p5_exp003_000_pretrain_iter25000_1024_shift3_fullcfg_seedrow_noneg` | 86.37 | 88.72 | 84.20 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_exp003_000_pretrain_iter25000_noneg` | 86.29 | 88.83 | 83.97 | 1170/1170 | gemini-3.1-pro |
| `c3p5_exp003_001_pretrain_midtrain_iter25000` | 86.05 | 87.96 | 84.30 | 1170/1170 | gemini-3.1-pro |
| `c3p5_exp003_000_pretrain_iter25000_1024_shift6_cfg400to1000_seedrow_noneg` | 86.00 | 88.88 | 83.36 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_exp003_000_pretrain_iter25000` | 85.80 | 87.81 | 83.97 | 1170/1170 | gemini-3.1-pro |

观察：
- **`002_pretrain_getty` 是三个 SFT 数据配方里最好的**：最佳 88.14，五组相同设置下比 `000_pretrain` 高 1.0–1.6 分，`noneg` 设置下比 `001_midtrain` 高 1.1 分。
- 推理设置的规律和第 3 节一致：shift 6 加每步 CFG 最好；只在 [400, 1000] 区间做 CFG 最差。
- exp003 最佳（88.14）比 Super base 最佳（87.22）高 0.9 分。
- **lr 2e-5 的两组（002 getty、003 midtrain）**：在 cfg 4、shift 3、全程 CFG、行号 seed、noneg 下分别是 87.92 和 87.88，差 −0.04，在噪声内，看不出区别。000 / 001 的 lr 是 1e-4，不和它们比。

---

## 4b. Cosmos3.5 Ultra（470bm235b，hamid prompt）

checkpoint：`s3://nv-00-10206-checkpoint/cosmos3_generator/3p5_production_runs/470bm235b_v3p4_pretrain/checkpoints/iter_000360000/`（预训练，未 SFT）。experiment：`cosmos3p5_ga_470bm235b_base_cp1_for_eval`（CP=1）。960×960，cfg 4，shift 3，全程 CFG，行号 seed，noneg，EMA。生成 job 2629685，4 节点，3 小时 13 分。

| run | overall | orig | phi | success | judge |
|---|---|---|---|---|---|
| `c3p5_ultra_v3p4_iter360000_shift3_fullcfg_seedrow_noneg` | **88.45** | 90.26 | 86.80 | 1170/1170 | gemini-3.1-pro@nvidia |

同设置（cfg 4、shift 3、全程 CFG、行号 seed、noneg）对比：

| 模型 | 阶段 | 分辨率 | overall |
|---|---|---|---|
| **Ultra v3p4 iter360000** | 预训练 | 960 | **88.45** |
| exp003_002 getty | Super + SFT | 1024 | 87.92 |
| exp003_003 midtrain lr2e-5 | Super + SFT | 1024 | 87.88 |
| Super v2p5 iter741260 | 预训练 | 960 | 86.72 |

- 同为预训练、同分辨率，**Ultra 比 Super 高 +1.73**，远超约 0.35 的 judge 噪声。
- Ultra 预训练已经比 Super 做完 SFT 后的最好结果高约 0.5。但分辨率和训练阶段都不同，这一项只作参考。
- 测完时是 Cosmos3.5 在 hamid prompt 上的最高分，之后被 4c 节 ruofan 的 iter30k（90.15）超过。

---

## 4c. Cosmos3.5 Ultra 图像混合训练（ruofan，hamid prompt）

checkpoint：`s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/image_mix_ultra/real70hq30_c3sdg15_text15_res10_30_60_seed42/checkpoints/iter_000030000/`（ruofan 2026-10-27 给的 best）。仓库里没有这个实验的 config，用 `cosmos3p5_ga_470bm235b_base_cp1_for_eval` 加载。1024×1024，cfg 4，shift 3，全程 CFG，行号 seed，noneg，EMA。生成 job 2710684，16 节点，1 小时。打分时 `v2_1170L` 已改为读 `v2_1170L.json`，与原 CSV 的打分输入逐行一致。

| run | overall | orig | phi | success | judge |
|---|---|---|---|---|---|
| `cosmos3p5_ultra_ruofan_20261027_iter30k_best` | **90.15** | 92.17 | 88.30 | 1170/1170 | gemini-3.1-pro@nvidia |

同设置（cfg 4、shift 3、全程 CFG、行号 seed、noneg）对比：

| 模型 | 阶段 | 分辨率 | overall | orig | phi |
|---|---|---|---|---|---|
| **Ultra ruofan iter30k** | Ultra + 图像混合训练 | 1024 | **90.15** | 92.17 | 88.30 |
| Ultra v3p4 iter360000 | 预训练 | 960 | 88.45 | 90.26 | 86.80 |
| exp003_002 getty | Super + SFT | 1024 | 87.92 | 90.18 | 85.84 |
| Super v2p5 iter741260 | 预训练 | 960 | 86.72 | 88.24 | 85.33 |

- 比 Ultra v3p4 预训练高 **+1.70**，约为 judge 噪声（0.35）的 5 倍。但分辨率不同（1024 对 960），分辨率贡献了多少还分不开。
- 这是目前 Cosmos3.5 在 hamid prompt 上的最高分。
- 分项最弱的是 Text Generation（92/208，44.23%），其次是 Logical Reasoning（125/155，80.65%）；Style（98.71%）和 World Knowledge（95.41%）最强。

### 4c-a4. 同一 checkpoint 换 a4 prompt

生成用新 benchmark `v2_1170L_gpt5p5_img_a4`（gpt5p5 改写的 prompt），其余设置同上，8 节点。打分仍用 `v2_1170L` 的原始 prompt 和 testpoint。

| run | 生成 prompt | overall | orig | phi | success | judge |
|---|---|---|---|---|---|---|
| `cosmos3p5_ultra_ruofan_20261027_iter30k_best` | a4 | **91.25** | 92.50 | 90.10 | 1170/1170 | gemini-3.1-pro@nvidia |
| `cosmos3p5_ultra_ruofan_20261027_iter30k_best` | hamid | 90.15 | 92.17 | 88.30 | 1170/1170 | gemini-3.1-pro@nvidia |

- a4 比 hamid 高 **+1.10**，提升主要来自 phi（+1.80），orig 只高 0.33。
- Text Generation 从 44.23% 升到 51.44%（107/208），仍是最弱项；Logical Reasoning 80.00%（124/155），与 hamid 基本持平。
- 最弱的动作类：Action - Hand 61.08%、Action - Full-body 69.50%。

## 4d. Cosmos3 Super UHQ SFT 前（exp009 union5 iter25000）

checkpoint：`s3://nv-00-10206-checkpoint/cosmos3_vfm/cosmos3_ga_t2ionly/cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5/checkpoints/iter_000025000/model`，即 exp009 sft0（UHQ SFT）的起点。experiment：`cosmos3_ga_64bm32b_t2ionly_base_mini`。1024×1024，cfg 4，50 步，EMA，Cosmos3 默认 negative prompt。4 节点，约 45 分钟。打分统一用 `v2_1170L`。

| run | 生成 prompt | overall | orig | phi | success | judge |
|---|---|---|---|---|---|---|
| `cosmos3_super_text2image_before_uhg` | opus | **91.09** | 93.06 | 89.28 | 1170/1170 | gemini-3.1-pro@nvidia |
| `cosmos3_super_text2image_before_uhg` | a4 | 88.09 | 90.38 | 85.98 | 1170/1170 | gemini-3.1-pro@nvidia |

- 用 opus prompt 时，比 UHQ SFT 之后的 exp009 sft0 iter31000（91.63）低 0.54，UHQ SFT 的增益不大。
- a4 prompt 对 Cosmos3 不利：比 opus 低 **3.00**。Text Generation 从 66.83% 掉到 47.60%，Logical Reasoning 从 76.13% 掉到 70.97%，Action - Hand 从 72.97% 掉到 64.86%。Cosmos3.5 Ultra（4c-a4）正相反，a4 比 hamid 高 1.10。
- 所以 Cosmos3 按惯例用 opus prompt 评测，a4 分数不宜和 Cosmos3.5 的 a4 分数直接横比。

## 4e. Cosmos3.5 Ultra exp005_001：v3p4 在 multiaspect-4k 全量上 SFT（a4 prompt）

checkpoint：`s3://nv-00-10206-checkpoint-experiments/cosmos3_generator/cosmos3plus_t2ionly/cosmos3p5_ga_470bm235b_t2ionly_moe_exp005_001_ultra_sanity_4kfull_n16/checkpoints/iter_000005000/`。从 Ultra v3p4 iter360000 出发，multiaspect-4k 全量数据，lr 2e-5，训练 5000 步。experiment：`cosmos3p5_ga_470bm235b_base_cp1_for_eval`。生成用 `v2_1170L_gpt5p5_img_a4`，1024×1024，cfg 4，shift 3，全程 CFG，行号 seed，noneg，EMA。生成 job 2757283，8 节点（中途被抢占一次，断点续跑）。

| run | overall | orig | phi | success | judge |
|---|---|---|---|---|---|
| `cosmos3p5_ultra_exp005_001_sanity_4kfull_iter5000` | **89.27** | 90.51 | 88.13 | 1170/1170 | gemini-3.1-pro@nvidia |
| `cosmos3p5_ultra_ruofan_20261027_iter30k_best`（对照，4c-a4） | 91.25 | 92.50 | 90.10 | 1170/1170 | gemini-3.1-pro@nvidia |

- 比 ruofan 图像混合训练低 **1.98**。
- 文字和手部动作明显变差：Text Generation 28.37%（59/208，ruofan 51.44%），Action - Hand 43.78%（81/185，ruofan 61.08%）。Logical Reasoning 80.00%，与 ruofan 持平；Style 99.35%、World Knowledge 94.80% 仍然很强。
- exp005_000（16 张图过拟合，iter2000）生成在 768/1170 张时被取消，尚无分数。

---

## 5. 附加结果（带 signature 的文件）

| run | result file | judge | overall | orig | phi |
|---|---|---|---|---|---|
| `c3p5_super_v2p5_iter741260_cfg6_noneg` | `unigenbench_result_judge_gpt5p5.json` | gpt-5.5 | 89.04 | 89.57 | 88.56 |
| `c3p5_super_v2p5_iter741260_noneg` | `unigenbench_result_judge_gpt5p5.json` | gpt-5.5 | 88.69 | 89.46 | 87.97 |
| `c3p5_super_v2p5_iter741260_ruofan_g6_shift6_negdefault` | `unigenbench_result_ruofan.json` | gemini-3.1-pro | 87.23 | 89.16 | 85.47 |
| `c3p5_super_v2p5_iter741260_ruofan_g4_shift6_negdefault` | `unigenbench_result_ruofan.json` | gemini-3.1-pro | 86.90 | 89.03 | 84.95 |

- **GPT-5.5 judge 比 Gemini judge 高约 2.6 分**：同一批图，noneg 是 88.69 对 86.10，cfg6_noneg 是 89.04 对 86.40。
- `ruofan` 那两行是同一批图用 `gemini-3.1-pro` 再打一次分，和 `@nvidia` 版本相差 0.01–0.11，可以作为 judge 自身的噪声参考。

---

## 6. 未打分 / 未完成

| run | 状态 | 原因 |
|---|---|---|
| `v2_1170L_gpt5p5_cosmos3p5_dense_hamid/c3p5_ultra_v3p5_iter384000` | 0 张图 | 用原 Ultra base（`context_parallel_shard_degree=8`）跑时卡在 CP all-to-all。之后改用 `cosmos3p5_ga_470bm235b_base_cp1_for_eval` 测了 v3p4 `iter_000360000`（见 4b 节）；v3p5 这个 checkpoint 没有重跑 |
| `.../c3p5_super_v2p5_iter741260_cfg16` | 160 张 | 按要求中途取消，没有打分 |
| `v2_1170L/fibo2_dev` | 1167 张 | 已经用换 seed 重试补回了 103 张，剩下 3 张 upsampler 始终没能产出合法 JSON |

---

## 7. HPSv3（人类偏好分，2026-10-28 起补算）

HPSv3 衡量的是图像整体好不好看、是否贴合 prompt（人类偏好），与 UGB 的逐 testpoint 准确率是两种指标，不能直接混比。

- 打分模型：HPSv3 reward service（Lepton gateway `scorer-hpsv3`），内部是官方 `HPSv3RewardInferencer`，取输出的 `mu`。
- 打分 prompt：一律用原版 `v2_1170L` 的**普通 prompt**（不是生成时的 opus / hamid / a4 改写 prompt），与 UGB 打分保持一致。
- 每组分数先对每个 prompt 求平均，再对 1170 个 prompt 求平均；`std` 是 prompt 之间的标准差。缺图不计入。
- 结果文件：各目录下的 `result_hpsv3_v2_1170L.json`。

viewer（`ugb_s3_viewer.py`）里的 21 组已全部算完。缺图的 6 组只少了本来就缺的那几张，其余都是 1170/1170。UGB overall 取自同目录的 `unigenbench_result.json`。

### 7.1 公开模型与 Cosmos3 super（按 HPSv3 排序）

| run | 生成 prompt | HPSv3 | std | success | UGB overall |
|---|---|---|---|---|---|
| `v2_1170L/gpt_image_2p5_sunburst` | 原版 | **12.253** | 1.574 | 1167/1170 | 94.22 |
| `v2_1170L/gpt_image_2p5_sunburst_quality_high` | 原版 | 12.245 | 1.571 | 1165/1170 | 94.45 |
| `v2_1170L/qwen_image_2p1_pe` | 原版 | 11.788 | 1.613 | 1170/1170 | 85.80 |
| `v2_1170L_opus/c3_dense_exp009_sft0_iter31000`（cosmos3-super-text2image） | opus | 11.598 | 1.613 | 1170/1170 | 91.63 |
| `v2_1170L/flux3_klein` | 原版 | 11.575 | 1.589 | 1167/1170 | 89.72 |
| `v2_1170L/krea2_turbo` | 原版 | 11.463 | 1.690 | 1170/1170 | 85.60 |
| `v2_1170L/fibo2_dev` | 原版 | 11.391 | 1.650 | 1167/1170 | 87.43 |
| `v2_1170L/flux3_klein_base` | 原版 | 11.123 | 1.593 | 1167/1170 | 90.63 |
| `v2_1170L/ideogram4_nf4` | 原版 | 10.767 | 2.114 | 1166/1170 | 86.19 |
| `v2_1170L/qwen_image_2p1_nope` | 原版 | 10.694 | 1.700 | 1170/1170 | 86.99 |

### 7.2 Cosmos3.5（按 HPSv3 排序）

| run | 生成 prompt | HPSv3 | std | success | UGB overall |
|---|---|---|---|---|---|
| `c3p5_ultra_v3p4_iter360000_shift3_fullcfg_seedrow_noneg` | hamid | **10.978** | 1.600 | 1170/1170 | 88.45 |
| `c3p5_exp003_003_pretrain_midtrain_lr2em5_iter25000_1024_shift3_fullcfg_seedrow_noneg` | hamid | 10.800 | 1.644 | 1170/1170 | 87.88 |
| `c3p5_super_v2p5_iter741260_shift3_fullcfg_seedrow_noneg` | hamid | 10.747 | 1.628 | 1170/1170 | 86.72 |
| `cosmos3p5_ultra_ruofan_20261027_iter30k_best` | hamid | 10.727 | 1.621 | 1170/1170 | 90.15 |
| `c3p5_exp004_003_pretrain_mha_dense_constlr_lr2em5_iter25000_1024_shift3_fullcfg_seedrow_noneg` | hamid | 10.702 | 1.662 | 1170/1170 | 84.36 |
| `c3p5_exp004_002_pretrain_mha_dense_lr2em5_iter25000_1024_shift3_fullcfg_seedrow_noneg` | hamid | 10.700 | 1.618 | 1170/1170 | 84.30 |
| `c3p5_exp004_000_pretrain_lr2em5_iter25000_1024_shift3_fullcfg_seedrow_noneg` | hamid | 10.667 | 1.633 | 1170/1170 | 88.12 |
| `c3p5_exp004_001_pretrain_mha_lr2em5_iter25000_1024_shift3_fullcfg_seedrow_noneg` | hamid | 10.639 | 1.629 | 1170/1170 | 87.73 |
| `c3p5_super_v2p5_iter741260_noneg` | hamid | 10.609 | 1.648 | 1170/1170 | 86.10 |
| `c3p5_exp003_002_pretrain_getty_iter25000_1024_shift3_fullcfg_seedrow_noneg` | hamid | 10.594 | 1.632 | 1170/1170 | 87.92 |
| `cosmos3p5_ultra_ruofan_20261027_iter30k_best`（`v2_1170L_gpt5p5_img_a4`） | a4 | 10.565 | 1.594 | 1170/1170 | 91.25 |

除最后一行在 `v2_1170L_gpt5p5_img_a4` 下，其余都在 `v2_1170L_gpt5p5_cosmos3p5_dense_hamid` 下。

观察：
- **HPSv3 与 UGB 基本不相关。** 21 组的 Spearman ρ = 0.29；9 个公开模型内部 0.42；11 组 Cosmos3.5 内部 −0.06。两个指标要分开看，不能用一个代替另一个。
- **Cosmos3.5 全部落在 HPSv3 最低的一档（10.57–10.98）**，低于除 Ideogram 4 和 Qwen nope 以外的所有公开模型，也比 Cosmos3 super（11.60）低 0.6–1.0。但 Cosmos3.5 是用 hamid dense prompt 生成、用原版短 prompt 打分，公开模型是原版 prompt 生成，这里分不开是模型差异还是生成 prompt 的差异。
- **Cosmos3.5 内部 HPSv3 几乎拉不开**：11 组只差 0.41，而 UGB 差 6.95（84.30–91.25）。exp004_002 / 003（mha_dense）的 UGB 比 exp004_000 低约 3.8，HPSv3 却基本一样。
- **UGB 上的提升没有带来 HPSv3 提升**：ruofan iter30k 换 a4 prompt，UGB +1.10，HPSv3 −0.16；ruofan 是 UGB 最高的 Cosmos3.5（hamid 90.15），HPSv3 却低于 Ultra v3p4 预训练（10.73 对 10.98）。
- **公开模型里，UGB 和 HPSv3 的取舍方向相反的有两对**：FLUX.3 klein 蒸馏版 HPSv3 比 klein-base 高 0.45，UGB 低 0.91；Qwen-Image 2.1 的 prompt enhancer 让 HPSv3 从 10.69 升到 11.79，UGB 却降了 1.19。
- GPT-image-2.5 的 `quality=high` 在 HPSv3 上也没有收益（12.245 对 12.253）。

---

## 附：打分协议

```bash
PYTHONPATH=. python projects/cosmos3/cosmos3/evaluation/text_to_image/compute_unigenbench_metric.py \
  --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/<benchmark>/<run> \
  --s3_cred credentials/gcs.secret \
  --benchmark_name v2_1170L \
  --judge_model gemini-3.1-pro@nvidia \
  --num_concurrency 128 --batch_size 1170 --extension png
```

本次新打的 18 组都是在本地跑的：打分只用 CPU，调用 gateway，每组约 6 分钟，3 组并行。

HPSv3（第 7 节）：

```bash
PYTHONPATH=. python projects/cosmos3/cosmos3/evaluation/text_to_image/compute_hpsv3_metric.py \
  --io_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/<benchmark>/<run> \
  --io_cred credentials/gcs.secret \
  --benchmark_name v2_1170L \
  --image_extension png
```

同样在本地跑，只调 HPSv3 gateway；单组顺序跑约 3–9 分钟。
