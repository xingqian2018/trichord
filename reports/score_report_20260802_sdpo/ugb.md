# UGB Score Report — SDPO Evaluation

Scored on `v2_1170L` with `gemini-3.1-pro@nvidia`.

## v2_1170L_opus

| Run                                                                              | Guidance   | Images       | all   | orig   | phi   | success   |
|----------------------------------------------------------------------------------|------------|--------------|-------|--------|-------|-----------|
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_000_sgdtext_vanilla_sft_iter5k_guidance2 | 2          | 1170 png     | 86.42 | 88.47  | 84.53 | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_000_sgdtext_vanilla_sft_iter5k           | 4          | 1170 png     | 87.42 | 89.87  | 85.16 | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_000_sgdtext_vanilla_sft_iter9k           | 4          | 640/1170 png | —     | —      | —     | —         |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_001_sgdtext_shift10_iter5k               | 4          | 640/1170 png | —     | —      | —     | —         |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_001_sgdtext_shift10_iter10k              | 4          | 512/1170 png | —     | —      | —     | —         |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_002_sgdtext_shift5_iter5k                | 4          | 256/1170 png | —     | —      | —     | —         |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_002_sgdtext_shift5_iter10k               | 4          | 0/1170 png   | —     | —      | —     | —         |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_003_sgdtext_shift3_iter5k                | 4          | 0/1170 png   | —     | —      | —     | —         |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_003_sgdtext_shift3_iter10k               | 4          | 0/1170 png   | —     | —      | —     | —         |

## v2_1170L_opus4p7_gc

| Run                                                                              | Guidance   | Images       | all   | orig   | phi   | success   |
|----------------------------------------------------------------------------------|------------|--------------|-------|--------|-------|-----------|
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_000_sgdtext_vanilla_sft_iter5k_guidance2 | 2          | 1170 png     | 85.24 | 88.93  | 81.86 | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_000_sgdtext_vanilla_sft_iter5k           | 4          | 1170 png     | 86.10 | 89.29  | 83.17 | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_000_sgdtext_vanilla_sft_iter9k           | 4          | 1170 png     | 85.02 | 88.67  | 81.68 | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_001_sgdtext_shift10_iter5k               | 4          | 1170 png     | 85.76 | 88.83  | 82.94 | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_001_sgdtext_shift10_iter10k              | 4          | 1170 png     | 85.07 | 88.75  | 81.70 | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_002_sgdtext_shift5_iter5k                | 4          | 1170 png     | 86.18 | 89.41  | 83.22 | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_002_sgdtext_shift5_iter10k               | 4          | 1170 png     | —     | —      | —     | —         |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_003_sgdtext_shift3_iter5k                | 4          | 1170 png     | 86.11 | 89.72  | 82.80 | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_003_sgdtext_shift3_iter10k               | 4          | 1170 png     | 85.37 | 88.90  | 82.12 | 1170/1170 |
