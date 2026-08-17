# CVTG Score Report — T2I Distill Evaluation

Complex Visual Text Generation. Stage 1: gen images on GPU. Stage 2: OCR + score text fidelity (GNED / PNED) with `gemini-3.1-pro`.

Metrics: **GNED** (Global Normalized Edit Distance, Hungarian-matched; 1.0 = perfect), **PNED** (Paired Normalized Edit Distance; 1.0 = perfect).

## CVTG 500L (English)

Benchmark `cvtg500L` (prompt_upsampled).

### Baseline (non-distilled)

| Run                      | Steps   | Neg prompt   | Images    | GNED   | PNED   | success   |
|--------------------------|---------|--------------|-----------|--------|--------|-----------|
| ga_super_t2i             | 50      | yes          | FromPaper | 80.88  | 89.08  | -         |
| ga_super_t2i_4step_noneg | 4       | no           | -         | -      | -      | -         |

### Distilled (base_distill_32b_xx)

| Run                                                                | Steps   | Neg prompt   | Images    | GNED   | PNED   | success   |
|--------------------------------------------------------------------|---------|--------------|-----------|--------|--------|-----------|
| base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter2k                  | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter4k                  | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter6k                  | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter8k                  | 4       | no           | 500 webp  | 80.33  | 88.39  | 497/500   |
| base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter2k | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter4k | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter6k | 4       | no           | 500 webp  | 80.13  | 87.40  | 498/500   |
| base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter8k | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp002_00_sfreq8_iter2k                        | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp002_00_sfreq8_iter3k                        | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp003_01_guidance4_betas0p1and0p999_iter2k    | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp003_01_guidance4_betas0p1and0p999_iter4k    | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp003_00_guidance4_iter2k                     | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp003_00_guidance4_iter3k                     | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp003_00_guidance4_iter4k                     | 4       | no           | -         | -      | -      | -         |

### Distilled (720p768p — dmd2 ablationV2_g3)

| Run                                                        | Steps   | Neg prompt   | Images    | GNED   | PNED   | success   |
|------------------------------------------------------------|---------|--------------|-----------|--------|--------|-----------|
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_iter2k          | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_iter3k          | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_iter4k          | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_iter6k          | 4       | no           | 500 webp  | 78.78  | 88.66  | 499/500   |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_iter8k          | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter2k       | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter3k       | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter4k       | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter6k       | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter8k       | 4       | no           | 500 webp  | 75.85  | 87.32  | 499/500   |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter2k | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter3k | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter4k | 4       | no           | 500 webp  | 78.60  | 87.68  | 500/500   |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter6k | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter8k | 4       | no           | -         | -      | -      | -         |

## CVTG 102CH (Chinese)

Benchmark `cvtg102ch`.

### Baseline (non-distilled)

| Run                      | Steps   | Neg prompt   | Images   | GNED   | PNED   | success   |
|--------------------------|---------|--------------|----------|--------|--------|-----------|
| ga_super_t2i             | 50      | yes          | -        | -      | -      | -         |
| ga_super_t2i_4step_noneg | 4       | no           | -        | -      | -      | -         |

### Distilled (base_distill_32b_xx)

| Run                                                                | Steps   | Neg prompt   | Images    | GNED   | PNED   | success   |
|--------------------------------------------------------------------|---------|--------------|-----------|--------|--------|-----------|
| base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter2k                  | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter4k                  | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter6k                  | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter8k                  | 4       | no           | 102 webp  | 27.22  | 36.08  | 101/102   |
| base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter2k | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter4k | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter6k | 4       | no           | 102 webp  | 29.28  | 38.76  | 102/102   |
| base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter8k | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp002_00_sfreq8_iter2k                        | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp002_00_sfreq8_iter3k                        | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp003_01_guidance4_betas0p1and0p999_iter2k    | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp003_01_guidance4_betas0p1and0p999_iter4k    | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp003_00_guidance4_iter2k                     | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp003_00_guidance4_iter3k                     | 4       | no           | -         | -      | -      | -         |
| base_distill_32b_xx_exp003_00_guidance4_iter4k                     | 4       | no           | -         | -      | -      | -         |

### Distilled (720p768p — dmd2 ablationV2_g3)

| Run                                                        | Steps   | Neg prompt   | Images    | GNED   | PNED   | success   |
|------------------------------------------------------------|---------|--------------|-----------|--------|--------|-----------|
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_iter2k          | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_iter3k          | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_iter4k          | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_iter6k          | 4       | no           | 102 webp  | 29.22  | 39.43  | 102/102   |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_iter8k          | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter2k       | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter3k       | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter4k       | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter6k       | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter8k       | 4       | no           | 102 webp  | 26.77  | 35.86  | 102/102   |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter2k | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter3k | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter4k | 4       | no           | 102 webp  | 29.30  | 37.49  | 102/102   |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter6k | 4       | no           | -         | -      | -      | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter8k | 4       | no           | -         | -      | -      | -         |
