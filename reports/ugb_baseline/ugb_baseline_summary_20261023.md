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
- 这是目前 Cosmos3.5 在 hamid prompt 上的最高分。

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
