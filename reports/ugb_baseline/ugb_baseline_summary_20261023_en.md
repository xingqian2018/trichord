# UGB Score Summary (2026-10-23)

A summary of all scored UGB results under `unigenbench_for_cosmos3p5/`, 47 runs in total.
For the public models' size, text tower, upsampler, etc., see [ugb_public_t2i_models_20261022.md](../ugb_public_t2i_models_20261022.md).

- S3 root: `s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/<benchmark>/<run>/`
- Each run's result file is `unigenbench_result.json` in that directory; a few directories also have results with a signature suffix, see Section 5.
- All numbers in the tables are accuracy (%). `orig` is the original-prompt subset, `phi` is the other half, and `overall` combines both.

---

## 0. Three things to know before reading the tables

1. **Generation prompts differ, scoring testpoints are the same.** The three benchmark directories differ only in the prompt fed to the model **at generation time**; **at scoring time** all of them use the original `v2_1170L` testpoints. So score differences across directories include the gain from prompt rewriting.
   | Directory | Prompt used for generation |
   |---|---|
   | `v2_1170L` | Original UGB prompt (short) |
   | `v2_1170L_opus` | Prompt rewritten by Opus |
   | `v2_1170L_gpt5p5_cosmos3p5_dense_hamid` | Dense prompt rewritten by GPT-5.5 (verbatim identical to the official `gpt55_a2/unigenbench-t2i.json`) |
2. **Judge.** The default is Gemini 3.1 Pro. `gemini-3.1-pro@nvidia` and `gemini-3.1-pro` (used in earlier results) are the same model under different names; scoring the same images with both gives differences within 0.1 (see Section 5). Results with GPT-5.5 as the judge are systematically about 2.6 points higher and **must not be compared with Gemini-judged results**.
3. **Missing images are excluded from the denominator.** API and open-source models are missing 1–5 images (`success` < 1170) due to safety filters or upsampler failures. Accuracy is computed only over the samples that were actually scored.

---

## 0b. Baseline testing principle: each model at its best resolution (from 2026-10-27)

Public baseline models are generated at **each vendor's officially recommended 1:1 resolution**, no longer a uniform 1024². The aspect ratio stays 1:1; upsamplers still follow each vendor's official pipeline (see Section 3 of [ugb_public_t2i_models_20261022.md](../ugb_public_t2i_models_20261022.md)).

| Model | Resolution | Basis |
|---|---|---|
| Krea 2 Turbo | **2048²** | The official codebase example uses `--width 2048 --height 2048 --mu 1.15`; the turbo checkpoint fixes `mu=1.15` (= `max_shift`, which corresponds to ≥ 6400 image tokens, i.e. high resolution). The diffusers default of 1024 is not its best resolution |
| Ideogram 4 nf4 | **2048²** | README: *"For the highest-quality images, set `--height 2048 --width 2048`"*; the diffusers pipeline defaults to 2048 and 48 steps |
| Qwen-Image 2.1 (pe / nope) | **2048²** | The README's 1:1 size is 2048×2048; the pipeline renders at that size by default |
| FLUX.3 klein / klein-base | 1024² | The modular pipeline defaults to 1024; the README gives no higher recommended resolution |
| Fibo 2.0 dev | 1024² | README: *"about one megapixel, the area Fibo 2.0 was trained at … `1024×1024` (the default)"* |
| GPT-image-2 / 2.5 sunburst | 1024² | Standard API size |
| Gemini 3.1 flash image | 1024² (`1k`) | `NvidiaGatewayImageGenerator` currently supports only `1k`; whether 2K is generated natively is unconfirmed |

**Impact on this report:** the UGB results in Section 1 were generated under the old approach (1024² for everything except Qwen-Image 2.1), so **Krea 2 Turbo and Ideogram 4 may be underscored**. UGB has not yet been rerun under the new principle; it is applied first to the CVTG baselines.

---

## 1. Public model comparison (`v2_1170L`, original prompt)

| Model | overall | orig | phi | success | judge | run |
|---|---|---|---|---|---|---|
| GPT-image-2.5 sunburst, quality high | **94.45** | 96.45 | 92.60 | 1165/1170 | gemini-3.1-pro@nvidia | `gpt_image_2p5_sunburst_quality_high` |
| GPT-image-2.5 sunburst, low (default) | 94.22 | 95.89 | 92.69 | 1167/1170 | gemini-3.1-pro@nvidia | `gpt_image_2p5_sunburst` |
| GPT-image-2, low (default) | 93.09 | 95.43 | 90.93 | 1168/1170 | gemini-3.1-pro@nvidia | `gpt_image_2` |
| GPT-image-2, quality high | 92.97 | 95.48 | 90.64 | 1166/1170 | gemini-3.1-pro@nvidia | `gpt_image_2_quality_high` |
| Gemini 3.1 flash image | 92.17 | 94.20 | 90.31 | 1169/1170 | gemini-3.1-pro@nvidia | `gemini_3_1_flash_image` |
| **cosmos3-super-text2image** | **91.63** | 93.85 | 89.59 | 1170/1170 | gemini-3.1-pro@nvidia | `c3_dense_exp009_sft0_iter31000` |
| FLUX.3 klein-base (5.6B, 50 steps, cfg 3.5) | 90.63 | 91.84 | 89.51 | 1167/1170 | gemini-3.1-pro@nvidia | `flux3_klein_base` |
| FLUX.3 klein (5.6B, distilled, 16 steps) | 89.72 | 91.71 | 87.89 | 1167/1170 | gemini-3.1-pro@nvidia | `flux3_klein` |
| **Cosmos3.5 Ultra v3p4 iter360000** pretrain | **88.45** | 90.26 | 86.80 | 1170/1170 | gemini-3.1-pro@nvidia | `c3p5_ultra_v3p4_iter360000_shift3_fullcfg_seedrow_noneg` |
| Fibo 2.0 dev (8.1B) | 87.43 | 89.20 | 85.81 | 1167/1170 | gemini-3.1-pro@nvidia | `fibo2_dev` |
| Qwen-Image 2.1, nope | 86.99 | 90.71 | 83.57 | 1170/1170 | gemini-3.1-pro | `qwen_image_2p1_nope` |
| Ideogram 4 (nf4, 9.3B×2) | 86.19 | 86.89 | 85.54 | 1166/1170 | gemini-3.1-pro@nvidia | `ideogram4_nf4` |
| Qwen-Image 2.1, pe | 85.80 | 88.34 | 83.48 | 1170/1170 | gemini-3.1-pro | `qwen_image_2p1_pe` |
| Krea 2 Turbo (12.8B, 8 steps) | 85.60 | 87.19 | 84.13 | 1170/1170 | gemini-3.1-pro@nvidia | `krea2_turbo` |

Observations:
- **Closed-source API models lead overall.** GPT-image-2.5 sunburst is the highest (94.2–94.5). The weakest API model (Gemini, 92.17) is 1.5 points above the strongest open-source model (FLUX.3 klein-base, 90.63), and the strongest API model is 8.9 points above the weakest open-source model (Krea, 85.60).
- **GPT-image's `quality=high` gives essentially no gain on UGB**: +0.2 for 2.5 sunburst and −0.1 for 2, both within noise. UGB mainly tests prompt following, so image quality improvements do not show up.
- **FLUX.3 klein-base is the best open-source model** (90.6), 0.9 above the distilled klein. FLUX.3, Ideogram and Fibo all use an LLM upsampler (Claude Opus 4.8 / Gemini 3.5 flash-lite), so their scores include the contribution of LLM rewriting.
- Krea 2 Turbo has the largest DiT (12.8B) but runs only 8 steps with no upsampler, and ranks last.

---

## 2. Cosmos vs. public models under the same conditions (`v2_1170L_opus`)

In this directory, the Cosmos models and the API models use **the same Opus-rewritten prompts**. This is currently the only place where they can be compared directly.

| Model | run | overall | orig | phi | success | judge |
|---|---|---|---|---|---|---|
| GPT-image-2.5 sunburst (low) | `gpt_image_2p5_sunburst` | **94.69** | 96.03 | 93.45 | 1165/1170 | gemini-3.1-pro@nvidia |
| GPT-image-2 (low) | `gpt_image_2` | 94.25 | 96.08 | 92.56 | 1165/1170 | gemini-3.1-pro@nvidia |
| Gemini 3.1 flash image | `gemini_3_1_flash_image` | 91.91 | 93.79 | 90.19 | 1169/1170 | gemini-3.1-pro@nvidia |
| Cosmos3 64bm32b dense, exp009 sft0, iter31000 | `c3_dense_exp009_sft0_iter31000` | 91.63 | 93.85 | 89.59 | 1170/1170 | gemini-3.1-pro@nvidia |
| Cosmos3 64bm32b dense, exp009 union5 (before UHQ SFT), iter25000 | `cosmos3_super_text2image_before_uhg` | 91.09 | 93.06 | 89.28 | 1170/1170 | gemini-3.1-pro@nvidia |
| Cosmos3 runlong RL, iter100 | `c3_runlong_rl_iter100` | 90.75 | 93.65 | 88.09 | 1170/1170 | gemini-3.1-pro@nvidia |
| Cosmos3 runlong RL, iter80 | `c3_runlong_rl_iter80` | 90.70 | 93.37 | 88.25 | 1170/1170 | gemini-3.1-pro@nvidia |
| Cosmos3.5 Super v2p5 iter741260 (960², cfg4, neg) | `c3p5_super_v2p5_iter741260_ts1000` | 87.79 | 89.52 | 86.22 | 1170/1170 | gemini-3.1-pro@nvidia |

Observations:
- With the same prompts, **Cosmos3 dense SFT (91.6) is on par with Gemini 3.1 flash image (91.9)**, and about 2.6–3 points below GPT-image.
- **The Cosmos3.5 Super pretrain checkpoint is 3.8 points below Cosmos3 dense SFT.** The former is a pretrain checkpoint without SFT; the gap is expected to narrow after SFT (see the exp003 series in Section 4).
- Compared with Section 1, Opus rewriting gives GPT-image-2 a +1.2 gain (93.09 → 94.25), 2.5 sunburst +0.5, and Gemini essentially nothing (92.17 → 91.91).

---

## 3. Inference setting sweep for Cosmos3.5 Super v2p5 iter741260 (hamid prompt)

All rows are the same checkpoint with only the inference settings changed. Meaning of the run name suffixes:
- `cfgN`: guidance N. 4 if not specified.
- `noneg`: no cosmos3 negative prompt.
- `shiftN`: sampler shift, default 3.
- `fullcfg`: CFG at every step; by default CFG is only applied in the [400, 1000] range.
- `seed0` / `seedrow`: all prompts use seed 0 / each prompt uses its own row index as the seed.
- `recipe` / `ruofan`: reproductions of the official cosmos-benchmarks recipe.

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
| `c3p5_super_v2p5_iter741260` (cfg4, neg, shift3, i.e. baseline) | 86.13 | 88.32 | 84.13 | 1170/1170 | gemini-3.1-pro@nvidia |
| `c3p5_super_v2p5_iter741260_noneg` | 86.10 | 88.01 | 84.34 | 1170/1170 | gemini-3.1-pro |

Observations:
- **All settings fall within about 1.1 points (86.1–87.2).** Inference parameters are not the bottleneck.
- The most effective setting is **shift 6 with CFG at every step** (+0.9, 87.0–87.2), consistent with the official recipe.
- Raising cfg from 4 to 8: +0.6 (86.13 → 86.70). Removing the negative prompt has essentially no effect (±0.3).
- cfg16 was cancelled after generating only 160 images and was not scored.

---

## 4. Cosmos3.5 exp003 SFT series (hamid prompt, 1024²)

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

Observations:
- **`002_pretrain_getty` is the best of the three SFT data recipes**: best score 88.14; under five matched settings it is 1.0–1.6 points above `000_pretrain`, and under the `noneg` setting it is 1.1 points above `001_midtrain`.
- The inference-setting pattern matches Section 3: shift 6 with CFG at every step is best; applying CFG only in the [400, 1000] range is worst.
- The best exp003 result (88.14) is 0.9 points above the best Super base result (87.22).
- **The two lr 2e-5 runs (002 getty, 003 midtrain)**: under cfg 4, shift 3, full CFG, row-index seed, noneg, they score 87.92 and 87.88 respectively, a −0.04 difference that is within noise, so no difference can be seen. 000 / 001 use lr 1e-4 and are not compared with them.

---

## 4b. Cosmos3.5 Ultra (470bm235b, hamid prompt)

Checkpoint: `s3://nv-00-10206-checkpoint/cosmos3_generator/3p5_production_runs/470bm235b_v3p4_pretrain/checkpoints/iter_000360000/` (pretrain, no SFT). Experiment: `cosmos3p5_ga_470bm235b_base_cp1_for_eval` (CP=1). 960×960, cfg 4, shift 3, full CFG, row-index seed, noneg, EMA. Generation job 2629685, 4 nodes, 3 h 13 min.

| run | overall | orig | phi | success | judge |
|---|---|---|---|---|---|
| `c3p5_ultra_v3p4_iter360000_shift3_fullcfg_seedrow_noneg` | **88.45** | 90.26 | 86.80 | 1170/1170 | gemini-3.1-pro@nvidia |

Comparison under the same settings (cfg 4, shift 3, full CFG, row-index seed, noneg):

| Model | Stage | Resolution | overall |
|---|---|---|---|
| **Ultra v3p4 iter360000** | Pretrain | 960 | **88.45** |
| exp003_002 getty | Super + SFT | 1024 | 87.92 |
| exp003_003 midtrain lr2e-5 | Super + SFT | 1024 | 87.88 |
| Super v2p5 iter741260 | Pretrain | 960 | 86.72 |

- Both pretrain and at the same resolution, **Ultra is +1.73 above Super**, well beyond the ~0.35 judge noise.
- Ultra pretrain is already about 0.5 above Super's best post-SFT result. However, the resolution and training stage both differ, so this is for reference only.
- At the time it was scored, this was the highest Cosmos3.5 score on the hamid prompts; it was later surpassed by ruofan's iter30k in Section 4c (90.15).

---

## 4c. Cosmos3.5 Ultra image-mix training (ruofan, hamid prompt)

Checkpoint: `s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/image_mix_ultra/real70hq30_c3sdg15_text15_res10_30_60_seed42/checkpoints/iter_000030000/` (the best one ruofan gave on 2026-10-27). The repo has no config for this experiment, so it is loaded with `cosmos3p5_ga_470bm235b_base_cp1_for_eval`. 1024×1024, cfg 4, shift 3, full CFG, row-index seed, noneg, EMA. Generation job 2710684, 16 nodes, 1 h. At scoring time `v2_1170L` had been switched to read `v2_1170L.json`, whose scoring inputs match the original CSV row by row.

| run | overall | orig | phi | success | judge |
|---|---|---|---|---|---|
| `cosmos3p5_ultra_ruofan_20261027_iter30k_best` | **90.15** | 92.17 | 88.30 | 1170/1170 | gemini-3.1-pro@nvidia |

Comparison under the same settings (cfg 4, shift 3, full CFG, row-index seed, noneg):

| Model | Stage | Resolution | overall | orig | phi |
|---|---|---|---|---|---|
| **Ultra ruofan iter30k** | Ultra + image-mix training | 1024 | **90.15** | 92.17 | 88.30 |
| Ultra v3p4 iter360000 | Pretrain | 960 | 88.45 | 90.26 | 86.80 |
| exp003_002 getty | Super + SFT | 1024 | 87.92 | 90.18 | 85.84 |
| Super v2p5 iter741260 | Pretrain | 960 | 86.72 | 88.24 | 85.33 |

- **+1.70** above Ultra v3p4 pretrain, about 5× the judge noise (0.35). But the resolution differs (1024 vs. 960), so how much of this comes from resolution cannot yet be separated.
- This is currently the highest Cosmos3.5 score on the hamid prompts.
- The weakest category is Text Generation (92/208, 44.23%), followed by Logical Reasoning (125/155, 80.65%); Style (98.71%) and World Knowledge (95.41%) are the strongest.

### 4c-a4. Same checkpoint with the a4 prompts

Generated with the new benchmark `v2_1170L_gpt5p5_img_a4` (prompts rewritten by gpt5p5); all other settings as above, 8 nodes. Scoring still uses the original `v2_1170L` prompts and testpoints.

| run | Generation prompt | overall | orig | phi | success | judge |
|---|---|---|---|---|---|---|
| `cosmos3p5_ultra_ruofan_20261027_iter30k_best` | a4 | **91.25** | 92.50 | 90.10 | 1170/1170 | gemini-3.1-pro@nvidia |
| `cosmos3p5_ultra_ruofan_20261027_iter30k_best` | hamid | 90.15 | 92.17 | 88.30 | 1170/1170 | gemini-3.1-pro@nvidia |

- a4 is **+1.10** above hamid, mostly from phi (+1.80); orig is only 0.33 higher.
- Text Generation rises from 44.23% to 51.44% (107/208) but is still the weakest category; Logical Reasoning is 80.00% (124/155), essentially unchanged from hamid.
- The weakest action categories: Action - Hand 61.08%, Action - Full-body 69.50%.

## 4d. Cosmos3 Super before UHQ SFT (exp009 union5 iter25000)

Checkpoint: `s3://nv-00-10206-checkpoint/cosmos3_vfm/cosmos3_ga_t2ionly/cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5/checkpoints/iter_000025000/model`, the starting point of exp009 sft0 (UHQ SFT). Experiment: `cosmos3_ga_64bm32b_t2ionly_base_mini`. 1024×1024, cfg 4, 50 steps, EMA, Cosmos3 default negative prompt. 4 nodes, about 45 minutes. All scoring uses `v2_1170L`.

| run | Generation prompt | overall | orig | phi | success | judge |
|---|---|---|---|---|---|---|
| `cosmos3_super_text2image_before_uhg` | opus | **91.09** | 93.06 | 89.28 | 1170/1170 | gemini-3.1-pro@nvidia |
| `cosmos3_super_text2image_before_uhg` | a4 | 88.09 | 90.38 | 85.98 | 1170/1170 | gemini-3.1-pro@nvidia |

- With opus prompts it is 0.54 below exp009 sft0 iter31000 after UHQ SFT (91.63), so UHQ SFT adds little.
- The a4 prompts hurt Cosmos3: **3.00** below opus. Text Generation drops from 66.83% to 47.60%, Logical Reasoning from 76.13% to 70.97%, Action - Hand from 72.97% to 64.86%. Cosmos3.5 Ultra (4c-a4) goes the other way, with a4 1.10 above hamid.
- So Cosmos3 should be evaluated with opus prompts as usual, and its a4 score should not be compared directly with Cosmos3.5 a4 scores.

---

## 5. Additional results (files with a signature suffix)

| run | result file | judge | overall | orig | phi |
|---|---|---|---|---|---|
| `c3p5_super_v2p5_iter741260_cfg6_noneg` | `unigenbench_result_judge_gpt5p5.json` | gpt-5.5 | 89.04 | 89.57 | 88.56 |
| `c3p5_super_v2p5_iter741260_noneg` | `unigenbench_result_judge_gpt5p5.json` | gpt-5.5 | 88.69 | 89.46 | 87.97 |
| `c3p5_super_v2p5_iter741260_ruofan_g6_shift6_negdefault` | `unigenbench_result_ruofan.json` | gemini-3.1-pro | 87.23 | 89.16 | 85.47 |
| `c3p5_super_v2p5_iter741260_ruofan_g4_shift6_negdefault` | `unigenbench_result_ruofan.json` | gemini-3.1-pro | 86.90 | 89.03 | 84.95 |

- **The GPT-5.5 judge scores about 2.6 points higher than the Gemini judge**: on the same images, noneg is 88.69 vs. 86.10, and cfg6_noneg is 89.04 vs. 86.40.
- The two `ruofan` rows are the same images re-scored with `gemini-3.1-pro`; they differ from the `@nvidia` version by 0.01–0.11 and serve as a reference for the judge's own noise.

---

## 6. Unscored / incomplete

| run | Status | Reason |
|---|---|---|
| `v2_1170L_gpt5p5_cosmos3p5_dense_hamid/c3p5_ultra_v3p5_iter384000` | 0 images | Running with the original Ultra base (`context_parallel_shard_degree=8`) hung in the CP all-to-all. We then switched to `cosmos3p5_ga_470bm235b_base_cp1_for_eval` and tested v3p4 `iter_000360000` (see Section 4b); this v3p5 checkpoint was not re-run |
| `.../c3p5_super_v2p5_iter741260_cfg16` | 160 images | Cancelled midway as requested, not scored |
| `v2_1170L/fibo2_dev` | 1167 images | 103 images were recovered by retrying with different seeds; for the remaining 3, the upsampler never produced valid JSON |

---

## 7. HPSv3 (human preference score, backfilled from 2026-10-28)

HPSv3 measures how good the image looks overall and how well it fits the prompt (human preference). It is a different metric from UGB's per-testpoint accuracy, and the two must not be compared directly.

- Scorer: the HPSv3 reward service (Lepton gateway `scorer-hpsv3`), which runs the official `HPSv3RewardInferencer` and returns its `mu` output.
- Scoring prompt: always the **plain prompt** from the original `v2_1170L` (not the opus / hamid / a4 rewritten prompt used at generation time), consistent with UGB scoring.
- Each run's score is first averaged per prompt, then averaged over the 1170 prompts; `std` is the standard deviation across prompts. Missing images are excluded.
- Result file: `result_hpsv3_v2_1170L.json` in each run directory.

All 21 runs in the viewer (`ugb_s3_viewer.py`) are scored. The 6 runs with missing images lose only those already-missing images; all others are 1170/1170. UGB overall is read from `unigenbench_result.json` in the same directory.

### 7.1 Public models and Cosmos3 super (sorted by HPSv3)

| run | Generation prompt | HPSv3 | std | success | UGB overall |
|---|---|---|---|---|---|
| `v2_1170L/gpt_image_2p5_sunburst` | Original | **12.253** | 1.574 | 1167/1170 | 94.22 |
| `v2_1170L/gpt_image_2p5_sunburst_quality_high` | Original | 12.245 | 1.571 | 1165/1170 | 94.45 |
| `v2_1170L/qwen_image_2p1_pe` | Original | 11.788 | 1.613 | 1170/1170 | 85.80 |
| `v2_1170L_opus/c3_dense_exp009_sft0_iter31000` (cosmos3-super-text2image) | opus | 11.598 | 1.613 | 1170/1170 | 91.63 |
| `v2_1170L/flux3_klein` | Original | 11.575 | 1.589 | 1167/1170 | 89.72 |
| `v2_1170L/krea2_turbo` | Original | 11.463 | 1.690 | 1170/1170 | 85.60 |
| `v2_1170L/fibo2_dev` | Original | 11.391 | 1.650 | 1167/1170 | 87.43 |
| `v2_1170L/flux3_klein_base` | Original | 11.123 | 1.593 | 1167/1170 | 90.63 |
| `v2_1170L/ideogram4_nf4` | Original | 10.767 | 2.114 | 1166/1170 | 86.19 |
| `v2_1170L/qwen_image_2p1_nope` | Original | 10.694 | 1.700 | 1170/1170 | 86.99 |

### 7.2 Cosmos3.5 (sorted by HPSv3)

| run | Generation prompt | HPSv3 | std | success | UGB overall |
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
| `cosmos3p5_ultra_ruofan_20261027_iter30k_best` (`v2_1170L_gpt5p5_img_a4`) | a4 | 10.565 | 1.594 | 1170/1170 | 91.25 |

All rows except the last are under `v2_1170L_gpt5p5_cosmos3p5_dense_hamid`; the last is under `v2_1170L_gpt5p5_img_a4`.

Observations:
- **HPSv3 and UGB are essentially uncorrelated.** Spearman ρ = 0.29 over all 21 runs, 0.42 within the 9 public models, and −0.06 within the 11 Cosmos3.5 runs. The two metrics have to be read separately; neither stands in for the other.
- **Every Cosmos3.5 run falls in the lowest HPSv3 band (10.57–10.98)**, below every public model except Ideogram 4 and Qwen nope, and 0.6–1.0 below Cosmos3 super (11.60). However, Cosmos3.5 is generated from hamid dense prompts and scored on the original short prompts, while the public models are generated from the original prompts, so model differences cannot be separated from generation-prompt differences here.
- **HPSv3 barely separates the Cosmos3.5 runs**: the 11 runs span only 0.41, versus 6.95 on UGB (84.30–91.25). exp004_002 / 003 (mha_dense) are about 3.8 below exp004_000 on UGB but essentially equal on HPSv3.
- **UGB gains do not carry over to HPSv3**: switching ruofan iter30k to the a4 prompts gives UGB +1.10 but HPSv3 −0.16; ruofan has the best Cosmos3.5 UGB (90.15 on hamid) yet a lower HPSv3 than Ultra v3p4 pretrain (10.73 vs. 10.98).
- **Two public-model pairs trade off in opposite directions**: FLUX.3 klein (distilled) is 0.45 above klein-base on HPSv3 but 0.91 below it on UGB; Qwen-Image 2.1's prompt enhancer lifts HPSv3 from 10.69 to 11.79 while UGB drops 1.19.
- GPT-image-2.5's `quality=high` gives no HPSv3 gain either (12.245 vs. 12.253).

---

## Appendix: scoring protocol

```bash
PYTHONPATH=. python projects/cosmos3/cosmos3/evaluation/text_to_image/compute_unigenbench_metric.py \
  --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/<benchmark>/<run> \
  --s3_cred credentials/gcs.secret \
  --benchmark_name v2_1170L \
  --judge_model gemini-3.1-pro@nvidia \
  --num_concurrency 128 --batch_size 1170 --extension png
```

The 18 newly scored runs in this round were all run locally: scoring is CPU-only and calls the gateway, about 6 minutes per run, 3 runs in parallel.

HPSv3 (Section 7):

```bash
PYTHONPATH=. python projects/cosmos3/cosmos3/evaluation/text_to_image/compute_hpsv3_metric.py \
  --io_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/<benchmark>/<run> \
  --io_cred credentials/gcs.secret \
  --benchmark_name v2_1170L \
  --image_extension png
```

Also run locally, calling only the HPSv3 gateway; about 3–9 minutes per run when run one at a time.
