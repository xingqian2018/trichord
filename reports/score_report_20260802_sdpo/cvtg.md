# CVTG Score Report — SDPO Evaluation

Gen: 50 steps, neg prompt on, guidance 4.0 (unless noted). Scored with `gemini-3.1-pro@nvidia`.

## cvtg500L_opus

| Run                                                                              | Guidance   | Images   | gned   | pned   | success   |
|----------------------------------------------------------------------------------|------------|----------|--------|--------|-----------|
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_000_sgdtext_vanilla_sft_iter5k_guidance2 | 2          | 500 webp | —      | —      | —         |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_000_sgdtext_vanilla_sft_iter5k           | 4          | 500 webp | 69.89  | 76.95  | 500/500   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_004_sgdtext_vanilla_sft_shift5_iter10k   | 4          | 500 webp | 69.92  | 79.61  | 499/500   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_005_sgdtext_shift5_betazero_iter10k      | 4          | 500 webp | 56.49  | 72.10  | 498/500   |

## cvtg500L_gc

| Run                                                                                         | Guidance   | Images         | gned      | pned      | success     |
|---------------------------------------------------------------------------------------------|------------|----------------|-----------|-----------|-------------|
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_000_sgdtext_vanilla_sft_iter5k_guidance2            | 2          | 500 webp       | 74.49     | 79.56     | 500/500     |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_000_sgdtext_vanilla_sft_iter5k                      | 4          | 500 webp       | 79.09     | 83.17     | 500/500     |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_000_sgdtext_vanilla_sft_iter9k                      | 4          | 500 webp       | 80.19     | 86.01     | 500/500     |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_001_sgdtext_shift10_iter5k                          | 4          | 500 webp       | 77.88     | 82.93     | 499/500     |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_001_sgdtext_shift10_iter10k                         | 4          | 500 webp       | 79.36     | 83.68     | 500/500     |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_002_sgdtext_shift5_iter5k                           | 4          | 500 webp       | 80.32     | 84.96     | 500/500     |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_002_sgdtext_shift5_iter10k                          | 4          | 500 webp       | 80.52     | 85.59     | 500/500     |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_003_sgdtext_shift3_iter5k                           | 4          | 500 webp       | 77.38     | 81.93     | 500/500     |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_003_sgdtext_shift3_iter10k                          | 4          | 500 webp       | 80.45     | 85.13     | 500/500     |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_004_sgdtext_vanilla_sft_shift5_iter10k              | 4          | 500 webp       | 82.14     | 88.55     | 500/500     |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_005_sgdtext_shift5_betazero_iter10k                 | 4          | 500 webp       | 72.24     | 79.74     | 495/500     |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_006_sgdtext_shift5_beta1to2_iter10k                 | 4          | 500 webp       | 82.05     | 88.47     | 500/500     |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_007_sgdtext_shift5_beta0to2_loss2piplus_iter10k     | 4          | 500 webp       | 79.66     | 86.15     | 500/500     |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_008_sgdtext_shift5_beta0to1_loss2piplus_iter10k     | 4          | 500 webp       | 79.19     | 84.65     | 500/500     |

`Selected results`

| Run                                                                                     | Images       | gned   | pned   | success   |
|-----------------------------------------------------------------------------------------|--------------|--------|--------|-----------|
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_002_sgdtext_shift5_iter10k                      | 500 webp     | 80.52  | 85.59  | 500/500   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_004_sgdtext_vanilla_sft_shift5_iter10k          | 500 webp     | 82.14  | 88.55  | 500/500   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_006_sgdtext_shift5_beta1to2_iter10k             | 500 webp     | 82.05  | 88.47  | 500/500   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_007_sgdtext_shift5_beta0to2_loss2piplus_iter10k | 500 webp     | 79.66  | 86.15  | 500/500   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_008_sgdtext_shift5_beta0to1_loss2piplus_iter10k | 500 webp     | 79.19  | 84.65  | 500/500   |


## cvtg102ch

| Run                                                                              | Guidance   | Images   | gned   | pned   | success   |
|----------------------------------------------------------------------------------|------------|----------|--------|--------|-----------|
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_000_sgdtext_vanilla_sft_iter5k_guidance2 | 2          | 102 webp | 32.28  | 44.7   | 102/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_000_sgdtext_vanilla_sft_iter5k           | 4          | 102 webp | 39.3   | 49.15  | 102/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_004_sgdtext_vanilla_sft_shift5_iter10k   | 4          | 102 webp | 43.11  | 57.28  | 100/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_005_sgdtext_shift5_betazero_iter10k      | 4          | 102 webp | 26.78  | 39.78  | 102/102   |

## cvtg102ch_gc

| Run                                                                                     | Guidance   | Images   | gned   | pned   | success   |
|-----------------------------------------------------------------------------------------|------------|----------|--------|--------|-----------|
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_000_sgdtext_vanilla_sft_iter5k_guidance2        | 2          | 102 webp | —      | —      | —         |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_000_sgdtext_vanilla_sft_iter5k                  | 4          | 102 webp | 48.09  | 57.59  | 102/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_000_sgdtext_vanilla_sft_iter10k                 | 4          | 102 webp | 51.02  | 59.75  | 101/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_001_sgdtext_shift10_iter5k                      | 4          | 102 webp | 44.01  | 54.32  | 102/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_001_sgdtext_shift10_iter10k                     | 4          | 102 webp | 46.74  | 58.00  | 102/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_002_sgdtext_shift5_iter5k                       | 4          | 102 webp | 47.17  | 56.26  | 102/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_002_sgdtext_shift5_iter10k                      | 4          | 102 webp | 50.88  | 61.32  | 102/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_003_sgdtext_shift3_iter5k                       | 4          | 102 webp | 44.06  | 56.98  | 102/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_003_sgdtext_shift3_iter10k                      | 4          | 102 webp | 50.51  | 60.58  | 102/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_004_sgdtext_vanilla_sft_shift5_iter10k          | 4          | 102 webp | 56.14  | 68.02  | 102/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_005_sgdtext_shift5_betazero_iter10k             | 4          | 102 webp | 39.92  | 51.59  | 101/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_006_sgdtext_shift5_beta1to2_iter10k             | 4          | 102 webp | 53.78  | 66.63  | 101/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_007_sgdtext_shift5_beta0to2_loss2piplus_iter10k | 4          | 102 webp | 53.33  | 64.06  | 102/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_008_sgdtext_shift5_beta0to1_loss2piplus_iter10k | 4          | 102 webp | 51.12  | 63.94  | 102/102   |

`Selected results`

| Run                                                                                     | Images   | gned   | pned   | success   |
|-----------------------------------------------------------------------------------------|----------|--------|--------|-----------|
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_002_sgdtext_shift5_iter10k                      | 102 webp | 50.88  | 61.32  | 102/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_004_sgdtext_vanilla_sft_shift5_iter10k          | 102 webp | 56.14  | 68.02  | 102/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_006_sgdtext_shift5_beta1to2_iter10k             | 102 webp | 53.78  | 66.63  | 101/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_007_sgdtext_shift5_beta0to2_loss2piplus_iter10k | 102 webp | 53.33  | 64.06  | 102/102   |
| cosmos3plus_64bm32b_t2ionly_sdpo_exp000_008_sgdtext_shift5_beta0to1_loss2piplus_iter10k | 102 webp | 51.12  | 63.94  | 102/102   |

<!-- Result files: result_cvtg_<benchmark>.json -->
