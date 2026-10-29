# UGB Score Report — Dataset Ablation (exp003)

Experiments: `cosmos3p5_ga_60bm30b_t2ionly_moe_exp003_*`, 25k steps each from
`s3://nv-00-10206-checkpoint/cosmos3_generator/3p5_production_runs/60bm30b_v2p5_pretrain/checkpoints/iter_000741260/`.

| Experiment | Data mix (ratio) | lr |
|---|---|---|
| exp003_000_pretrain | pretrain_real 8 : pretrain_sdg 1 : pretrain_text 1 | 1e-4 |
| exp003_001_pretrain_midtrain | pretrain_real 8 : sdg 1 : text 1 : midtrain 2.5 (midtrain 20%) | 1e-4 |
| exp003_002_pretrain_getty | pretrain_real 7.2 : sdg 0.9 : text 0.9 : getty 1.0 (getty 10%, `getty_image_350k_test.lance`) | 2e-5 |
| exp003_003_pretrain_midtrain_lr2em5 | same as exp003_001 (midtrain 20%) | 2e-5 |
| exp004_000_pretrain | same as exp003_000 (only lr changed) | 2e-5 |
| exp004_001_pretrain_mha | same as exp004_000, gen pathway GQA → MHA (`gen_num_key_value_heads=32`) | 2e-5 |
| exp004_002_pretrain_mha_dense | same as exp004_001, gen MoE MLP → dense (`gen_dense_mlp_intermediate_size=6912`, 8 × 768 routed + 768 shared) | 2e-5 |

Scoring for every row: `compute_unigenbench_metric.py --benchmark_name v2_1170L` (original, non-upsampled UGB prompts and testpoints), judge `gemini-3.1-pro`.

## Cosmos3.5 — generated with v2_1170L_gpt5p5_cosmos3p5_dense_hamid

Generation: `inference_ugb_i4.py`, `--experiment_name cosmos3p5_ga_60bm30b_base_768`, 1024×1024, guidance 4.0, 50 steps, `--use_ema`. `neg` = `--use_cosmos3_negative_prompt`, `noneg` = without it.

| Run                                                              | Images   | all       | orig      | phi       | success      |
|------------------------------------------------------------------|----------|-----------|-----------|-----------|--------------|
| c3p5_super_v2p5_iter741260_noneg (pre-SFT, 960×960, `base`)      | 1170 png | 86.10     | 88.01     | 84.34     | 1170/1170    |
| ---------------------------------------------------------------- | -------- | --------- | --------- | --------- | ------------ |
| c3p5_exp003_000_pretrain_iter25000 (neg)                         | 1170 png | 85.80     | 87.81     | 83.97     | 1170/1170    |
| c3p5_exp003_001_pretrain_midtrain_iter25000 (neg)                | 1170 png | 86.05     | 87.96     | 84.30     | 1170/1170    |
| ---------------------------------------------------------------- | -------- | --------- | --------- | --------- | ------------ |
| c3p5_exp003_000_pretrain_iter25000_noneg                         | 1170 png | 86.29     | 88.83     | 83.97     | 1170/1170    |
| c3p5_exp003_001_pretrain_midtrain_iter25000_noneg                | 1170 png | 86.44     | 88.88     | 84.20     | 1170/1170    |
| c3p5_exp003_002_pretrain_getty_iter25000_noneg                   | 1170 png | **87.56** | **89.80** | **85.51** | 1170/1170    |
| c3p5_exp003_003_pretrain_midtrain_lr2em5_iter25000_1024_shift3_fullcfg_seedrow_noneg ¹ | 1170 png | 87.88 | 90.13 | 85.82 | 1170/1170 |
| c3p5_exp004_000_pretrain_lr2em5_iter25000_1024_shift3_fullcfg_seedrow_noneg ¹ | 1170 png | 88.12 | 90.36 | 86.08 | 1170/1170 |
| c3p5_exp004_001_pretrain_mha_lr2em5_iter25000_1024_shift3_fullcfg_seedrow_noneg ¹ ² | 1170 png | 87.73 | 90.18 | 85.49 | 1170/1170 |
| c3p5_exp004_002_pretrain_mha_dense_lr2em5_iter25000_1024_shift3_fullcfg_seedrow_noneg ¹ ³ | 1170 png | 84.30 | 86.22 | 82.54 | 1170/1170 |

¹ exp003_003 was generated with shift 3, **CFG on every step**, per-prompt row seed (the new `Cosmos3p5Generator` default, `cd2022e3408`), not the CFG [400, 1000] used by the other rows of this table, so it is not directly comparable to them. It is compared only with exp003_002, which shares lr 2e-5; exp003_000 / 001 use lr 1e-4 and are not compared. Under the same setting (1024, shift 3, full CFG, row seed, noneg): exp003_002 getty 87.92, exp003_003 midtrain **87.88**, a difference of −0.04, inside the ~0.35 judge noise, so the two are indistinguishable. exp004_000 uses the same setting and scores **88.12**: +0.20 over exp003_002 and +0.24 over exp003_003, both inside the noise, so all three are tied. exp004_000 differs from exp003_000 only in lr. Under the same setting exp003_000 (lr 1e-4) scores 86.37, and exp004_000 (lr 2e-5) is **+1.75** higher, well beyond the noise. So the earlier gain of exp003_002 / 003 over exp003_000 / 001 comes mainly from lowering lr to 2e-5, not from the getty or midtrain data.

² exp004_001 was generated with `--experiment_name cosmos3p5_ga_60bm30b_base_768_gen_mha32` (the MHA eval config); every other setting matches exp004_000. It differs from exp004_000 only in the gen pathway's GQA → MHA and scores **87.73**, 0.39 lower, about the size of the judge noise, so GQA → MHA shows no gain on UGB.

³ exp004_002 was generated with `--experiment_name cosmos3p5_ga_60bm30b_base_768_gen_mha32_dense`; every other setting matches exp004_001. It differs from exp004_001 only in replacing the gen MoE MLP with a dense MLP of the same activated width, and scores **84.30**, **3.43** lower, well beyond the noise and also below the pre-SFT 86.10. Text Generation drops from 16.35% to 4.81% (10/208). After the MoE → dense conversion, 25k steps did not recover the MoE level.

## Sampling progression on exp003_002 getty (1024×1024)

Every SFT checkpoint is evaluated at 1024×1024 with `base_768` (only the pre-SFT checkpoint uses 960×960 with `base`). Same checkpoint `exp003_002_pretrain_getty/iter_000025000`, noneg, guidance 4, 50 steps, EMA; adjacent rows differ by one variable.

| Setting | Generation script | all | orig | phi | success |
|---|---|---|---|---|---|
| shift 3, CFG [400, 1000], row seed (original) | `inference_ugb_i4.py` | 87.56 | 89.80 | 85.51 | 1170/1170 |
| shift 3, CFG every step, row seed | `inference_ugb_i4_20261020_shift3_fullcfg.py` | 87.92 | 90.18 | 85.84 | 1170/1170 |
| shift 6, CFG [400, 1000], row seed | `inference_ugb_i4_20261020.py` | 87.23 | 89.57 | 85.09 | 1170/1170 |
| shift 6, CFG every step, row seed | `inference_ugb_i4_20261020_fullcfg.py` | 87.62 | 89.69 | 85.72 | 1170/1170 |
| shift 6, CFG every step, seed 0 | `inference_ugb_i4_20261020_fullcfg_seed0.py` | **88.14** | **90.26** | **86.19** | 1170/1170 |

Step by step: shift 3 → 6 is −0.33, CFG [400, 1000] → every step is +0.39, row seed → seed 0 is +0.52; original to the full recipe is +0.58 in total. The first two steps sit within the ~0.35-point judge re-run noise, so shift 6 by itself brings no gain at 1024. The seed-0 step is the largest and points the same way at 960 (87.67 → 87.88), but the seed only sets the sampling noise, not model capability, so it should not be read as a model improvement.

Output folders: `.../v2_1170L_gpt5p5_cosmos3p5_dense_hamid/c3p5_exp003_002_pretrain_getty_iter25000_1024_shift6_{cfg400to1000_seedrow,fullcfg_seedrow,fullcfg_seed0}_noneg`. Re-running the judge alone moves a score by about 0.35 points.

The same progression on exp003_000 pretrain (`exp003_000_pretrain/iter_000025000`), 1024×1024, `base_768`, noneg:

| Setting | Generation script | all | orig | phi | success |
|---|---|---|---|---|---|
| shift 3, CFG [400, 1000], row seed (original) | `inference_ugb_i4.py` | 86.29 | 88.83 | 83.97 | 1170/1170 |
| shift 3, CFG every step, row seed | `inference_ugb_i4_20261020_shift3_fullcfg.py` | 86.37 | 88.72 | 84.20 | 1170/1170 |
| shift 6, CFG [400, 1000], row seed | `inference_ugb_i4_20261020.py` | 86.00 | 88.88 | 83.36 | 1170/1170 |
| shift 6, CFG every step, row seed | `inference_ugb_i4_20261020_fullcfg.py` | 86.61 | 89.18 | 84.25 | 1170/1170 |
| shift 6, CFG every step, seed 0 | `inference_ugb_i4_20261020_fullcfg_seed0.py` | 86.87 | 88.83 | 85.07 | 1170/1170 |

**Step-by-step change across the three checkpoints (all):**

| Step | pre-SFT (960) | exp003_000 pretrain (1024) | exp003_002 getty (1024) |
|---|---|---|---|
| shift 3 → 6 | +0.11 | −0.29 | −0.33 |
| CFG [400, 1000] → every step | +0.82 | +0.61 | +0.39 |
| row seed → seed 0 | +0.18 | +0.26 | +0.52 |
| original → full recipe | +1.11 | +0.58 | +0.58 |

**Getty over pretrain under each sampling setting (1024):**

| Setting | pretrain | getty | Δ |
|---|---|---|---|
| shift 3, CFG [400, 1000], row seed | 86.29 | 87.56 | +1.27 |
| shift 6, CFG [400, 1000], row seed | 86.00 | 87.23 | +1.23 |
| shift 6, CFG every step, row seed | 86.61 | 87.62 | +1.01 |
| shift 6, CFG every step, seed 0 | 86.87 | 88.14 | +1.27 |

**shift × CFG interval 2×2 (row seed, all):**

| Checkpoint | shift 3 + [400, 1000] | shift 3 + every step | shift 6 + [400, 1000] | shift 6 + every step |
|---|---|---|---|---|
| pre-SFT (960) | 86.10 | 86.72 | 86.21 | 87.03 |
| exp003_000 pretrain (1024) | 86.29 | 86.37 | 86.00 | 86.61 |
| exp003_002 getty (1024) | 87.56 | 87.92 | 87.23 | 87.62 |
| getty − pretrain | +1.27 | +1.55 | +1.23 | +1.01 |

CFG on every step beats [400, 1000] in all six comparisons (shift 3: +0.62, +0.08, +0.36; shift 6: +0.82, +0.61, +0.39; mean ≈ +0.48); shift 3 → 6 goes both ways (mean ≈ 0). Getty over pretrain stays between +1.0 and +1.55 under all five sampling settings.

Conclusions:
- shift 6 alone does nothing on any of the three checkpoints (−0.33 to +0.11, all inside the ~0.35-point judge noise).
- CFG on every step is the only consistent sampling gain: positive on all three checkpoints (+0.39 to +0.82).
- seed 0 is positive all three times (+0.18 to +0.52) but never far above noise; it should not be read as a model improvement.
- Getty's gain over pretrain stays between +1.0 and +1.3 under all four sampling settings, so the result does not depend on the sampler.

The same progression on the pre-SFT checkpoint (exp003's common start `iter_000741260`), 960×960, `base`, noneg:

| Setting | Generation script | all | orig | phi | success |
|---|---|---|---|---|---|
| shift 3, CFG [400, 1000], row seed (original) | `inference_ugb_i4.py` | 86.10 | 88.01 | 84.34 | 1170/1170 |
| shift 3, CFG every step, row seed | `inference_ugb_i4_20261020_shift3_fullcfg.py` | 86.72 | 88.24 | 85.33 | 1170/1170 |
| shift 6, CFG [400, 1000], row seed | `inference_ugb_i4_20261020.py` | 86.21 | 87.88 | 84.67 | 1170/1170 |
| shift 6, CFG every step, row seed | `inference_ugb_i4_20261020_fullcfg.py` | 87.03 | 88.83 | 85.37 | 1170/1170 |
| shift 6, CFG every step, seed 0 | `inference_ugb_i4_20261020.py` (earlier version) | 87.21 | 89.18 | 85.40 | 1170/1170 |

The two getty recipe rows in the "Base checkpoint" section below (87.88, 87.67) were rendered at 960×960 with `base`, which breaks the resolution rule above; this section's 1024 results supersede them.

## Base checkpoint — guidance and judge

Checkpoint `60bm30b_v2p5_pretrain/checkpoints/iter_000741260` (pre-SFT). Generation: `--experiment_name cosmos3p5_ga_60bm30b_base`, 960×960, 50 steps, `--use_ema`, noneg, v2_1170L_gpt5p5_cosmos3p5_dense_hamid prompts. Scoring prompts `v2_1170L` for every cell. GPT-5.5 results are stored as `unigenbench_result_judge_gpt5p5.json` (`--signature judge_gpt5p5`) next to the gemini `unigenbench_result.json`.

| Run                                         | guidance | judge                 | all       | orig      | phi       | success      |
|---------------------------------------------|----------|-----------------------|-----------|-----------|-----------|--------------|
| c3p5_super_v2p5_iter741260_noneg            | 4.0      | gemini-3.1-pro        | 86.10     | 88.01     | 84.34     | 1170/1170    |
| c3p5_super_v2p5_iter741260_noneg            | 4.0      | gpt-5.5               | 88.69     | 89.46     | 87.97     | 1170/1170    |
| c3p5_super_v2p5_iter741260_cfg6_noneg       | 6.0      | gemini-3.1-pro@nvidia | 86.40     | 87.63     | 85.28     | 1170/1170    |
| c3p5_super_v2p5_iter741260_cfg6_noneg       | 6.0      | gpt-5.5               | 89.04     | 89.57     | 88.56     | 1170/1170    |
| c3p5_super_v2p5_iter741260_recipe_shift6_fullcfg_seed0 (noneg) | 4.0 | gemini-3.1-pro@nvidia | **87.21** | **89.18** | 85.40 | 1170/1170 |
| c3p5_super_v2p5_iter741260_ruofan_g4_shift6_negdefault       | 4.0      | gemini-3.1-pro@nvidia | 86.79     | 88.83     | 84.93     | 1170/1170    |
| c3p5_super_v2p5_iter741260_ruofan_g6_shift6_negdefault       | 6.0      | gemini-3.1-pro@nvidia | 87.22     | 89.16     | 85.44     | 1170/1170    |

The two ruofan rows are images generated by ruofanl with the cosmos-benchmarks system, copied from gcpcode lustre into our S3 folder and re-scored with our scorer:
- g4: `/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/ruofanl/local_eval/ugb_img_a4_xckpt_s0_720p_20261014/results/prod_g4_gpt55_a2/ruofanl_60bm30b_v2p5_pretrain_iter_000741260_720p_g4_nocache_gpt55_a2`
- g6: `/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/ruofanl/local_eval/getty_base_741260_ugb_20260928/results/base/ruofanl_60bm30b_v2p5_pretrain_iter_000741260_720p_g6_nocache`

His generation settings (from each folder's `config.json`): same checkpoint `iter_000741260`, the production experiment config (`cosmos3p5_ga_60bm30b_v2_pretrain` module, `configs/base/config.py`), 960×960, shift 6, `negative_prompt_version: default` (**with** a negative prompt), the same gpt55_a2 prompts. His own scores (`unigenbench_result_ruofan.json`, also gemini-3.1-pro + `v2_1170L_G3F.csv`): g4 86.90 / 89.03 / 84.95, g6 87.23 / 89.16 / 85.47.

Our re-scores are within 0.0–0.1 points of his, so **the scoring pipelines agree and the gap comes from generation**. At guidance 4 and 960×960, our images score 86.10 and his 86.79, a 0.69-point gap. Known generation differences: shift (ours hardcoded 3.0, his 6), CFG interval (ours hardcoded [400, 1000], his every step), seed, negative prompt (ours none, his default), and experiment config (ours text_rendering `base`, his the production config). The recipe row (job 2576804) aligns the first three to measure how much of the gap they explain.

**Result:** changing only shift (3.0 → 6), the CFG interval ([400, 1000] → every step) and the seed lifts our own pipeline at guidance 4 from 86.10 to 87.21 — above ruofan's g4 (86.79) and level with his g6 (87.22), without a negative prompt. The ~0.7-point gap is therefore essentially the `shift=3.0` and `guidance_interval=[400, 1000]` hardcoded in `Cosmos3Generator.generate_image` (`inference_helper.py`); the negative prompt and the experiment config are not the main cause. Raising guidance from 4 to 6 under the old sampling only adds 0.30 (86.10 → 86.40).

**getty under the recipe sampling:** same `inference_ugb_i4_20261020.py` (shift 6, CFG on every step, seed 0), `cosmos3p5_ga_60bm30b_base`, 960×960, guidance 4, noneg, on exp003_002 `iter_000025000`:

| Run | all | orig | phi | success |
|---|---|---|---|---|
| c3p5_super_v2p5_iter741260_recipe_shift6_fullcfg_seed0 (pre-SFT) | 87.21 | 89.18 | 85.40 | 1170/1170 |
| c3p5_exp003_002_pretrain_getty_iter25000_recipe_shift6_fullcfg_seed0_noneg | **87.88** | **90.13** | **85.82** | 1170/1170 |
| c3p5_exp003_002_pretrain_getty_iter25000_recipe_shift6_fullcfg_seedrow_noneg (seed back to prompt row index) | 87.67 | 89.92 | 85.61 | 1170/1170 |

Switching the seed from 0 back to the prompt row index moves getty from 87.88 to 87.67 (−0.21). The first scoring of the row-seed images gave 87.32 (1169/1170: one judge call still failed after retries and was dropped from the stats); a `--force_rescore` pass on the same images gave 87.67. **Re-running the judge alone on identical images moved the score by 0.35 points**, more than the seed change did. Differences within roughly 0.3–0.5 points should be treated as noise.

Under the same sampling, the getty run is 0.67 points above pre-SFT (orig +0.95, phi +0.42). Under the old sampling (1024, `base_768`, shift 3) it was 1.46 above pre-SFT (87.56 vs 86.10), but those two rows also differ in resolution and config; this pair is the single-variable comparison, and the gain shrinks but holds.

**Impact:** every Cosmos3.5 score generated with `inference_ugb_i4.py` so far, including all exp003 rows, used shift 3 and CFG interval [400, 1000]. They remain comparable with each other but sit roughly 0.7–1 point below scores from the cosmos-benchmarks system.

The last row is generated with `inference_ugb_i4_20261020.py`, whose sampling matches the cosmos-benchmarks `cosmos3_eval` unigenbench-t2i recipe: shift 6 (the original script hardcodes 3.0), CFG on every step (the original hardcodes `guidance_interval=[400, 1000]`), seed 0 for every prompt (the original uses the prompt row index); no negative prompt. Everything else matches `c3p5_super_v2p5_iter741260_noneg`; the run exists to locate the ~0.8-point gap to a colleague's result.

### Judge comparison — gemini-3.1-pro vs gpt-5.5 on the same images

Images: `c3p5_super_v2p5_iter741260_noneg` (pre-SFT, 960×960, guidance 4.0). Both judges scored all 1170 images and the same 8193 testpoints (orig 3920, phi 4273); every image has as many verdicts as testpoints. gpt-5.5 needed one retry pass for 9 images whose first answer was not valid YAML; all 9 were scored on retry. Correct testpoints: gemini 7054 (86.10%), gpt-5.5 7266 (88.69%).

| Verdict pair          | Testpoints | Share  |
|-----------------------|------------|--------|
| both pass             | 6779       | 82.7%  |
| both fail             | 652        | 8.0%   |
| only gpt-5.5 passes   | 487        | 5.9%   |
| only gemini passes    | 275        | 3.4%   |

The judges agree on 90.7% of testpoints. gpt-5.5 passes 212 more testpoints than gemini among the disagreements, which is the whole +2.6-point gap.

Disagreements concentrate on actions and spatial relations:

| Primary dimension | gpt-5.5 only | gemini only | Testpoints |
|-------------------|--------------|-------------|------------|
| Action            | 191          | 117         | 1525       |
| Relationship      | 84           | 23          | 1011       |
| Entity Layout     | 64           | 14          | 1110       |
| Attribute         | 61           | 53          | 2595       |
| Compound          | 28           | 11          | 414        |
| Logical Reasoning | 15           | 20          | 155        |
| World Knowledge   | 14           | 21          | 327        |
| Text Generation   | 13           | 6           | 208        |
| Style             | 10           | 0           | 620        |
| Grammar           | 7            | 10          | 228        |

The phi subset (physical-AI scenes) carries most of these testpoints, so phi moves most (+3.6 vs +1.5 on orig). Style and Text Generation barely differ.

Sampled disagreements (8 read by hand, not a systematic annotation):
- **gpt-5.5 passes, gemini fails** — mostly gpt-5.5 missing detail: `phi138` asks for a view from behind the mower, the image shows its front; gemini flags it, gpt-5.5 calls it "from slightly behind". `phi51` asks for the bottle just below the surface, its upper half is above the water, and gpt-5.5 passes it. Some are gemini being over-strict, e.g. `phi554` failed on a "strict definition of inclusion".
- **gemini passes, gpt-5.5 fails** — mostly gpt-5.5 being literal: `phi552` "full-body action" failed because the chef is framed from the waist up; `phi122` failed because "no motion blur shows the robot is moving".

Conclusion: gpt-5.5 is more lenient on spatial and action details and gemini catches more real errors, so gemini-3.1-pro stays the primary judge. gpt-5.5 scores are reference only and are not compared against gemini scores.

## Reference — Qwen-Image-2.1, generated with original v2_1170L prompts

Generation: ad hoc client over the self-hosted vLLM-Omni service (`Qwen/Qwen-Image-2.1`), 2048×2048, 40 steps, `true_cfg_scale=1.0`, seed = prompt row index. `pe` = prompts rewritten by `Qwen/Qwen-Image-2.1-PE-T2I` first (aspect ratio fixed to 1:1).

| Run                                                              | Images   | all       | orig      | phi       | success      |
|------------------------------------------------------------------|----------|-----------|-----------|-----------|--------------|
| qwen_image_2p1_nope                                              | 1170 png | **86.99** | **90.71** | **83.57** | 1170/1170    |
| qwen_image_2p1_pe                                                | 1170 png | 85.80     | 88.34     | 83.48     | 1170/1170    |

## Primary dimensions (all subset)

| Metric            | pre-SFT noneg | 000 neg | 001 neg | 000 noneg | 001 noneg | 002 noneg | Qwen-2.1 no PE | Qwen-2.1 PE |
|-------------------|---------------|---------|---------|-----------|-----------|-----------|----------------|-------------|
| Action            | 73.11         | 72.39   | 71.93   | 73.18     | 72.39     | 74.43     | 73.25          | 73.38       |
| Attribute         | 95.07         | 93.87   | 94.61   | 94.07     | 94.80     | 95.72     | 94.72          | 92.06       |
| Compound          | 84.30         | 84.30   | 83.09   | 84.54     | 85.27     | 89.37     | 82.85          | 81.40       |
| Entity Layout     | 92.25         | 91.98   | 92.70   | 92.61     | 91.80     | 92.07     | 89.01          | 86.04       |
| Grammar           | 88.60         | 90.35   | 88.60   | 89.47     | 90.35     | 88.60     | 85.96          | 82.89       |
| Logical Reasoning | 70.32         | 73.55   | 71.61   | 76.77     | 74.84     | 80.00     | 76.13          | 70.32       |
| Relationship      | 85.16         | 87.04   | 87.04   | 86.94     | 87.44     | 88.72     | 85.95          | 82.89       |
| Style             | 98.06         | 96.45   | 96.61   | 98.71     | 97.74     | 98.39     | 95.97          | 94.84       |
| Text Generation   | 12.50         | 13.46   | 16.83   | 12.50     | 15.87     | 15.38     | 63.46          | 93.75       |
| World Knowledge   | 89.60         | 89.91   | 91.13   | 90.21     | 92.05     | 91.13     | 95.11          | 95.11       |

Regenerate this table with `python reports/score_report_20261020_dataset_ablation/gather_ugb_scores.py` (reads each run's `unigenbench_result.json`).

## Findings

- exp003_002 (getty 10%, lr 2e-5) is the best Cosmos3.5 run: all 87.56, +1.12 over exp003_001 noneg and +1.27 over exp003_000 noneg, the largest gap in this ablation. The gain is spread across Compound (+4.1 vs 001 noneg), Logical Reasoning (+5.2), Action, Attribute and Relationship; Text Generation does not move (15.38). It still changes two things at once — data and lr (1e-4 → 2e-5) — so the gain cannot yet be attributed to getty alone; a lr-2e-5 run without getty is the missing control.
- exp003_002 was judged with `--judge_model gemini-3.1-pro@nvidia` (the bare name now fails with `KeyError`); the other rows were judged earlier with the bare `gemini-3.1-pro`. Same model, possibly a different gateway.
- exp003_000 and exp003_001 sit within ~0.7 points of each other and of the pre-SFT checkpoint. With one image per prompt this is within noise; neither the extra 25k steps nor midtrain shows a clear gain.
- Training loss stayed flat over the 25k steps (~1.66–1.72 for exp003_000, ~1.69–1.72 for exp003_001): pretrain_real/sdg/text are fragments of `image_meta_table_full`, the same table the v2p5 checkpoint was pretrained on, so the runs see little new signal.
- Dropping the negative prompt adds ~0.4 points to both runs.
- Midtrain moves Text Generation in the expected direction (+3.4 points over exp003_000 under both neg and noneg), but the dimension has only 208 testpoints.
- Qwen-Image-2.1 is not comparable row-for-row: it renders the short original prompts at 2048, while Cosmos3.5 renders GPT-5.5-upsampled prompts at 1024. Its lead is concentrated in Text Generation (63.46 vs ≤16.83). Its own PE lifts Text Generation to 93.75 but costs 3–6 points on Entity Layout, Grammar, Relationship and Logical Reasoning, so the total drops to 85.80.
- Pre-SFT row is 960×960 with `base` (shift 3); exp003 rows are 1024×1024 with `base_768` (shift 5).
