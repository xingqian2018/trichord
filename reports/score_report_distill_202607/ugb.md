# UGB Score Report — T2I Distill Evaluation

Benchmark: `v2_1170L_opus4p7_ga` (gen) / scored on `v2_1170L` with `gemini-3.1-pro`.

## Baseline (non-distilled)

| Run                      | Steps   | Neg prompt   | Images   | all       | orig      | phi       | success   |
|--------------------------|---------|--------------|----------|-----------|-----------|-----------|-----------|
| ga_super_t2i             | 50      | yes          | 1170 png | **91.02** | **93.62** | **88.63** | 1170/1170 |
| ga_super_t2i_4step_noneg | 4       | no           | 1170 png | 77.98     | 79.87     | 76.25     | 1170/1170 |

## Distilled (base_distill_32b_xx)

| Run                                                                | Steps   | Neg prompt   | Images   | all   | orig   | phi   | success   |
|--------------------------------------------------------------------|---------|--------------|----------|-------|--------|-------|-----------|
| base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter2k                  | 4       | no           | 1170 png | 90.33 | 92.93  | 87.95 | 1170/1170 |
| base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter4k                  | 4       | no           | 1170 png | 90.8  | 92.96  | 88.81 | 1170/1170 |
| base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter6k                  | 4       | no           | 1170 png | 91.09 | 93.21  | 89.14 | 1170/1170 |
| base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter8k                  | 4       | no           | 1170 png | 91.18 | 93.37  | 89.16 | 1170/1170 |
| base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter2k | 4       | no           | 1170 png | 90.3  | 92.65  | 88.13 | 1170/1170 |
| base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter4k | 4       | no           | 1170 png | 91.08 | 93.39  | 88.95 | 1170/1170 |
| base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter6k | 4       | no           | 1170 png | **91.35** | 93.52  | 89.35 | 1170/1170 |
| base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter8k | 4       | no           | 1170 png | 91.16 | 93.11  | 89.38 | 1170/1170 |
| base_distill_32b_xx_exp002_00_sfreq8_iter2k                        | 4       | no           | 1170 png | 89.91 | 92.24  | 87.76 | 1170/1170 |
| base_distill_32b_xx_exp002_00_sfreq8_iter3k                        | 4       | no           | 1170 png | 90.3  | 92.47  | 88.3  | 1170/1170 |
| base_distill_32b_xx_exp003_01_guidance4_betas0p1and0p999_iter2k    | 4       | no           | 1170 png | 90.35 | 92.63  | 88.25 | 1170/1170 |
| base_distill_32b_xx_exp003_01_guidance4_betas0p1and0p999_iter4k    | 4       | no           | 1170 png | 90.53 | 93.01  | 88.25 | 1170/1170 |
| base_distill_32b_xx_exp003_00_guidance4_iter2k                     | 4       | no           | 1170 png | 90.76 | 92.91  | 88.79 | 1170/1170 |
| base_distill_32b_xx_exp003_00_guidance4_iter3k                     | 4       | no           | 1170 png | 90.78 | 92.7   | 89.02 | 1170/1170 |
| base_distill_32b_xx_exp003_00_guidance4_iter4k                     | 4       | no           | 1170 png | 90.59 | 92.63  | 88.72 | 1170/1170 |

## Distilled (720p768p — dmd2 ablationV2_g3)

Gen at 1024×1024, cfg=1, 4-step, no-neg (standard experiment_name).

| Run                                                          | Steps   | Neg prompt   | Images   | all   | orig   | phi   | success   |
|--------------------------------------------------------------|---------|--------------|----------|-------|--------|-------|-----------|
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_iter2k            | 4       | no           | 1170 png | 89.77 | 91.76  | 87.95 | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_iter3k            | 4       | no           | 1170 png | 90.35 | 92.83  | 88.06 | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_iter4k            | 4       | no           | 1170 png | 90.99 | 93.06  | 89.09 | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_iter6k            | 4       | no           | 1170 png | 91.25 | 93.29  | 89.38 | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_iter8k            | 4       | no           | 1170 png | 91.09 | 93.11  | 89.23 | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter2k         | 4       | no           | 1170 png | 89.42 | 91.99  | 87.06 | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter3k         | 4       | no           | 1170 png | 89.91 | 92.19  | 87.81 | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter4k         | 4       | no           | 1170 png | 90.37 | 92.68  | 88.25 | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter6k         | 4       | no           | 1170 png | 90.81 | 93.42  | 88.42 | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter8k         | 4       | no           | 1170 png | 90.96 | 93.47  | 88.65 | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter2k   | 4       | no           | 1170 png | 90.85 | 93.01  | 88.86 | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter3k   | 4       | no           | 1170 png | 91.26 | 93.16  | 89.52 | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter4k   | 4       | no           | 1170 png | **91.37** | 93.42  | 89.49 | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter6k   | 4       | no           | 1170 png | 91.11 | 93.01  | 89.38 | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter8k   | 4       | no           | 1170 png | 90.93 | 93.01  | 89.02 | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_neg_prompt_iter2k | 4       | no           | 1170 png | 89.36 | 91.84  | 87.08 | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_neg_prompt_iter4k | 4       | no           | 1170 png | 90.43 | 92.78  | 88.28 | 1170/1170 |

## Distilled (super_t2i_dmd)

Gen at 1024×1024, cfg=1, 4-step, no-neg (t2idistill_base_nosound).

| Run                                          | Steps   | Neg prompt   | Images   | all       | orig      | phi       | success   |
|----------------------------------------------|---------|--------------|----------|-----------|-----------|-----------|-----------|
| cosmos3_super_t2i_dmd_dCM_3k_init_iter3k     | 4       | no           | 1170 png | 91.15     | 93.09     | 89.38     | 1170/1170 |
| cosmos3_super_t2i_dmd_dCM_3k_init_iter5k     | 4       | no           | 1170 png | 91.42     | 93.37     | **89.63** | 1170/1170 |
| cosmos3_super_t2i_dmd_dCM_3k_init_iter8k     | 4       | no           | 1170 png | 90.66     | 93.29     | 88.25     | 1170/1170 |
| cosmos3_super_t2i_dmd_dCM_3k_init_iter10k    | 4       | no           | 1170 png | 91.33     | 93.39     | 89.45     | 1170/1170 |
| cosmos3_super_t2i_dmd_default_iter3k         | 4       | no           | 1170 png | 90.41     | 92.55     | 88.44     | 1170/1170 |
| cosmos3_super_t2i_dmd_default_iter5k         | 4       | no           | 1170 png | 91.03     | 93.09     | 89.14     | 1170/1170 |
| cosmos3_super_t2i_dmd_default_iter8k         | 4       | no           | 1170 png | **91.52** | **93.57** | **89.63** | 1170/1170 |
| cosmos3_super_t2i_dmd_default_iter10k        | 4       | no           | 1170 png | 91.22     | 93.47     | 89.16     | 1170/1170 |
| cosmos3_super_t2i_dmd_optim_beta0pt1_iter3k  | 4       | no           | 1170 png | 90.20     | 92.65     | 87.95     | 1170/1170 |
| cosmos3_super_t2i_dmd_optim_beta0pt1_iter5k  | 4       | no           | 1170 png | 90.94     | 92.98     | 89.07     | 1170/1170 |
| cosmos3_super_t2i_dmd_optim_beta0pt1_iter8k  | 4       | no           | 1170 png | 91.12     | 93.34     | 89.09     | 1170/1170 |
| cosmos3_super_t2i_dmd_optim_beta0pt1_iter10k | 4       | no           | 1170 png | 91.11     | 93.37     | 89.05     | 1170/1170 |
