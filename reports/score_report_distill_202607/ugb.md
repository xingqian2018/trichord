# UGB Score Report — T2I Distill Evaluation

Benchmark: `v2_1170L_opus4p7_ga` (gen) / scored on `v2_1170L` with `gemini-3.1-pro`.

## Baseline (non-distilled)

| Run                      | Steps   | Neg prompt   | Images   | all       | orig      | phi       | success   |
|--------------------------|---------|--------------|----------|-----------|-----------|-----------|-----------|
| ga_super_t2i             | 50      | yes          | 1170 png | **91.02** | **93.62** | **88.63** | 1170/1170 |
| ga_super_t2i_4step_noneg | 4       | no           | 1170 png | 77.98     | 79.87     | 76.25     | 1170/1170 |

## Distilled (base_distill_32b_xx)

| Run                                                                | Steps   | Neg prompt   | Images   | all       | orig      | phi       | success   |
|--------------------------------------------------------------------|---------|--------------|----------|-----------|-----------|-----------|-----------|
| base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter3k                  | 4       | no           | 1170 png | 90.38     | 92.88     | 88.09     | 1170/1170 |
| base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter4k                  | 4       | no           | 1170 png | 90.41     | 92.78     | 88.23     | 1170/1170 |
| base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter5k                  | 4       | no           | 1170 png | 90.85     | **93.11** | 88.77     | 1170/1170 |
| base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter6k                  | 4       | no           | 1170 png | 90.58     | 92.86     | 88.49     | 1170/1170 |
| base_distill_32b_xx_exp000_00_lr4em6lr8em6_iter7k                  | 4       | no           | 1170 png | 90.66     | 92.73     | 88.77     | 1170/1170 |
| base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter3k | 4       | no           | 1170 png | 89.98     | 92.63     | 87.55     | 1170/1170 |
| base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter4k | 4       | no           | 1170 png | 90.25     | 92.53     | 88.16     | 1170/1170 |
| base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter5k | 4       | no           | 1170 png | 90.77     | **93.11** | 88.63     | 1170/1170 |
| base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter6k | 4       | no           | 1170 png | **90.88** | 92.65     | **89.26** | 1170/1170 |
| base_distill_32b_xx_exp001_00_lr4em6lr8em6_betas0p1and0p999_iter7k | 4       | no           | 1170 png | 90.74     | 92.81     | 88.84     | 1170/1170 |
| base_distill_32b_xx_exp002_00_sfreq8_iter2k                        | 4       | no           | 1170 png | 89.97     | 91.99     | 88.11     | 1170/1170 |
| base_distill_32b_xx_exp002_00_sfreq8_iter3k                        | 4       | no           | 1170 png | 89.92     | 92.27     | 87.76     | 1170/1170 |
| base_distill_32b_xx_exp003_01_guidance4_betas0p1and0p999_iter2k    | 4       | no           | 1170 png | 90.43     | 92.83     | 88.23     | 1170/1170 |
| base_distill_32b_xx_exp003_01_guidance4_betas0p1and0p999_iter3k    | 4       | no           | 1170 png | 90.67     | 92.86     | 88.67     | 1170/1170 |
| base_distill_32b_xx_exp003_01_guidance4_betas0p1and0p999_iter4k    | 4       | no           | 1170 png | 90.66     | 92.83     | 88.67     | 1170/1170 |
| base_distill_32b_xx_exp003_00_guidance4_iter2k                     | 4       | no           | 1170 png | 90.81     | 92.86     | 88.93     | 1170/1170 |
| base_distill_32b_xx_exp003_00_guidance4_iter3k                     | 4       | no           | 1170 png | 90.87     | 92.81     | 89.09     | 1170/1170 |
| base_distill_32b_xx_exp003_00_guidance4_iter4k                     | 4       | no           | 1170 png | 90.44     | 92.68     | 88.39     | 1170/1170 |

## Distilled (720p768p — dmd2 ablationV2_g3)

Gen at 1024×1024, cfg=1, 4-step, no-neg (standard experiment_name).

| Run                                                                 | Steps   | Neg prompt   | Images   | all       | orig      | phi       | success   |
|---------------------------------------------------------------------|---------|--------------|----------|-----------|-----------|-----------|-----------|
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter2k                | 4       | no           | 1170 png | -         | -         | -         | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter3k                | 4       | no           | 1170 png | 89.98     | 92.14     | 87.99     | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter4k                | 4       | no           | 1170 png | 90.26     | 92.27     | 88.42     | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter5k                | 4       | no           | 1170 png | 90.24     | 92.42     | 88.23     | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter6k                | 4       | no           | 1170 png | 90.27     | 92.50     | 88.23     | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_g3_iter7k                | 4       | no           | 1170 png | 90.60     | 92.63     | 88.74     | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_iter8k                   | 4       | no           | 1170 png | **90.96** | **93.09** | **89.00** | 1170/1170 |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter3000_iter3k | 4       | no           | -        | -         | -         | -         | -         |
| base_distill_dmd2_ga_pt_32b_t2i_ablationV2_dcm_init_iter3000_iter8k | 4       | no           | 1170 png | 90.85     | 92.96     | 88.91     | 1170/1170 |
