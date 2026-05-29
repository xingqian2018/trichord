# Evaluation Result Folders on GCS

Base path: `gcs:nv-00-10206-vfm/debug/xingqianx/evaluation_results/`

---

## UGB (UniGenBench)

Path: `gcs:nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench/`

### v2_1170L_G3F/
| Run                                                                          | Images   | Result JSON                                                                            |
|------------------------------------------------------------------------------|----------|----------------------------------------------------------------------------------------|
| flux_1_kontext_dev                                                           | 0        | unigenbench_result.json, unigenbench_result_gemini-3p1-pro.json                        |
| flux_2_klein_9b                                                              | 0        | unigenbench_result_gemini-3p1-pro-earlier.json, unigenbench_result_gemini-3p1-pro.json |
| glm_image                                                                    | 0        | unigenbench_result_gemini-3p1-pro.json                                                 |
| nano_banana_pro                                                              | 0        | unigenbench_result_gemini-3p1-pro.json, unigenbench_result_rerun_20260505.json         |
| qwen_image                                                                   | 0        | unigenbench_result.json, unigenbench_result_gemini-3p1-pro.json                        |
| qwen_image_2512                                                              | 0        | unigenbench_result.json, unigenbench_result_gemini-3p1-pro.json, ... <4 more>          |
| qwen_image_2512_rerun                                                        | 1170     |                                                                                        |
| sd_v3p5_large                                                                | 0        | unigenbench_result.json, unigenbench_result_gemini-3p1-pro.json                        |
| t2w_mot_exp302_000_qwen3_vl_8b_multires_recipe_v7_iter000030000              | 1170     | unigenbench_result.json                                                                |
| t2w_mot_exp302_000_qwen3_vl_8b_multires_recipe_v7_iter000030000_withnp       | 1170     | unigenbench_result.json                                                                |
| t2w_mot_exp302_003_qwen3_vl_8b_multires_modality_offset_iter000006750        | 1170     | unigenbench_result.json                                                                |
| t2w_mot_exp302_003_qwen3_vl_8b_multires_modality_offset_iter000006750_withnp | 1170     | unigenbench_result.json                                                                |
| z_image_turbo                                                                | 0        | unigenbench_result_gemini-3p1-pro.json                                                 |

### v2_1170L_opus/
| Run                                                                                    | Images     | Result JSON                                                                                  |
|----------------------------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------|
| qwen_image_2512                                                                        | 1170       | unigenbench_result_gemini-3p1-pro.json, unigenbench_result_using_eval_prompt_v2_1170L.json   |
| cosmos3_ga_16bm8b_v2_pretrain_iter85k                                                  | 1170       | unigenbench_result_gemini-3p1-pro.json                                                       |
| cosmos3_ga_64bm32b_from_v2_image_only_v1p2_iter81p5k                                   | 1170       | unigenbench_result_gemini-3p1-pro_v2_1170l_g3f.json                                          |
| cosmos3_ga_64bm32b_from_v2_image_only_v1p4_iter93p25k                                  | 1170       | unigenbench_result_gemini-3p1-pro_v2_1170l_g3f.json                                          |
| cosmos3_ga_16bm8b_image_only_data_mixture_000_iter20k                                  | 1170       | unigenbench_result_gemini-3p1-pro.json                                                       |
| cosmos3_ga_64bm32b_image_only_data_mixture_000_iter10k                                 | 1170       | unigenbench_result_gemini-3p1-pro_v2_1170l_g3f.json                                          |
| cosmos3_ga_64bm32b_image_only_data_mixture_000_iter20k                                 | 1170       | unigenbench_result_gemini-3p1-pro_v2_1170L_G3F.json                                          |

| cosmos3_ga_64bm32b_image_only_data_mixture_000_iter5k                                  | 1170       | unigenbench_result_gemini-3p1-pro_v2_1170l_g3f.json                                          |
| cosmos3_ga_64bm32b_image_only_data_mixture_001_iter5k                                  | 1170       | unigenbench_result_gemini-3p1-pro_v2_1170l_g3f.json                                          |
| cosmos3_ga_64bm32b_v3p1_midtrain_iter000000900                                         | 1170       | unigenbench_result.json                                                                      |
| -------------------------------------------------------------                          | ---------- | -------------------------------------------------------------------------------------------- |
| cosmos3_ga_64bm32b_t2ionly_base_v1_v1p5_iter108k                                       | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_16bm8b_v2_midtrain_iter6k                                                   | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_v3_midtrain_iter1800                                                | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_base_v2_v1_iter1k                                           | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_base_iter_000033000_v1p2                                    | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_base_iter_000040000_v1p4                                    | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp000_text_only_sft_iter1p5k                               | 1170       | unigenbench_result_gemini-3p1-pro_v2_1170L_G3F.json                                          |
| cosmos3_ga_64bm32b_t2ionly_exp000_text_only_sft_iter4k                                 | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp001_text_mixture_sft_iter1000                            | 1170       | unigenbench_result_gemini-3p1-pro_v2_1170L_G3F.json                                          |
| cosmos3_ga_64bm32b_t2ionly_exp001_text_mixture_sft_iter1500                            | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp002_text_only_sft_from_frozen_midtrain_iter1000          | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp003_text_mixture2_sft_iter1k                             | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp003_text_mixture2_sft_iter2k                             | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp004_text_mixture2_sft_from_frozen_midtrain_iter2k        | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp005_text_mixture3_sft_iter1k                             | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp006_text_mixture3_sft_from_frozen_midtrain_iter500       | 1170       |                                                                                              |
| cosmos3_ga_64bm32b_t2ionly_exp007_text_mixture4_sft_lr1em5_iter4k                      | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp008_text_mixture4_sft_from_frozen_midtrain_lr1em5_iter4k | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter4k            | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter10k           | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter15k           | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter20k           | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter25k           | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter26k              | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter28k              | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter30k              | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter35k              | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter26k             | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter28k             | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter30k             | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter35k             | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter40k             | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_merged_000                                                  | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_merged_003                                                  | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_merged_006                                                  | 1170       | unigenbench_result.json                                                                      |
| cosmos3_ga_64bm32b_t2ionly_merged_007                                                  | 1170       | unigenbench_result.json                                                                      |

### aa_opus/
| Run                                                                                    | Images   | Result JSON              |
|----------------------------------------------------------------------------------------|----------|--------------------------|
| cosmos3_ga_64bm32b_t2ionly_base_v1_v1p5_iter108k                                       | 1567     | rubric_result_g3p1p.json |
| cosmos3_ga_64bm32b_v3_midtrain_iter1800                                                | 1567     | rubric_result_g3p1p.json |
| cosmos3_ga_64bm32b_image_only_v3p1_iter18k                                             | 1567     | rubric_result_g3p1p.json |
| cosmos3_ga_64bm32b_image_only_v3p3_iter22k                                             | 1567     | rubric_result_g3p1p.json |

| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter4k            | 0        |                          |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter10k           | 1567     | rubric_result_g3p1p.json |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter15k           | 1567     | rubric_result_g3p1p.json |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter20k           | 1567     | rubric_result_g3p1p.json |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter25k           | 1567     | rubric_result_g3p1p.json |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter30k              | 1569     |                          |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter35k              |          |                          |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter30k             | 1569     |                          |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter35k             |          |                          |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter40k             |          |                          |
| cosmos3_ga_64bm32b_t2ionly_merged_000                                                  | 1569     |                          |
| cosmos3_ga_64bm32b_t2ionly_merged_000_2                                                | 1569     |                          |
| cosmos3_ga_64bm32b_t2ionly_merged_001                                                  | 1569     |                          |
| cosmos3_ga_64bm32b_t2ionly_merged_002                                                  | 1569     |                          |
| cosmos3_ga_64bm32b_t2ionly_merged_003                                                  | 1569     |                          |
| cosmos3_ga_64bm32b_t2ionly_merged_004                                                  | 1569     |                          |
| cosmos3_ga_64bm32b_t2ionly_merged_006                                                  | 1569     |                          |
| cosmos3_ga_64bm32b_t2ionly_merged_007                                                  | 1569     |                          |

---

## CVTG

Path: `gcs:nv-00-10206-vfm/debug/xingqianx/evaluation_results/cvtg/`

### cvtg102ch_ascii/
| Run                | Images   | Result JSON                                                                         |
|--------------------|----------|-------------------------------------------------------------------------------------|
| flux_1_kontext_dev | 102      | result_cvtg_cvtg102ch_g3p1p.json                                                    |
| flux_2_dev         | 102      | result_cvtg_cvtg102ch_g3p1p.json                                                    |
| flux_2_klein_9b    | 102      | result_cvtg_cvtg102ch_g3p1p.json                                                    |
| nano_banana_pro    | 102      | result_cvtg_cvtg102ch_g3p1p.json, result_cvtg_cvtg102ch_g3p1p_capital_agnostic.json |
| qwen_image         | 102      | result_cvtg_cvtg102ch_g3p1p.json                                                    |
| qwen_image_2512    | 102      | result_cvtg_cvtg102ch_g3p1p.json                                                    |
| sd_v3p5_large      | 102      | result_cvtg_cvtg102ch_g3p1p.json                                                    |
| z_image_turbo      | 102      | result_cvtg_cvtg102ch_g3p1p.json                                                    |

### cvtg102ch_opus/
| Run                                                                             | Images   | Result JSON                           |
|---------------------------------------------------------------------------------|----------|---------------------------------------|
| cosmos3_ga_64bm32b_t2ionly_exp000_text_only_sft_iter1p5k                        | 102      | result_cvtg_cvtg102ch_opus_g3p1p.json |
| cosmos3_ga_64bm32b_t2ionly_exp000_text_only_sft_iter4k                          | 102      | result_cvtg_cvtg102ch_opus_g3p1p.json |
| cosmos3_ga_64bm32b_t2ionly_exp001_text_mixture_sft_iter1500                     | 102      | result_cvtg_cvtg102ch_opus_g3p1p.json |
| cosmos3_ga_64bm32b_t2ionly_exp002_text_only_sft_from_frozen_midtrain_iter1000   | 102      | result_cvtg_cvtg102ch_opus_g3p1p.json |
| cosmos3_ga_64bm32b_t2ionly_exp003_text_mixture2_sft_iter1k                      | 102      | result_cvtg_cvtg102ch_opus_g3p1p.json |
| cosmos3_ga_64bm32b_t2ionly_exp003_text_mixture2_sft_iter2k                      | 102      | result_cvtg_cvtg102ch_opus_g3p1p.json |
| cosmos3_ga_64bm32b_t2ionly_exp004_text_mixture2_sft_from_frozen_midtrain_iter2k | 102      | result_cvtg_cvtg102ch_opus_g3p1p.json |

### cvtg102ch_opus_ascii/
| Run                                                                                    | Images     | Result JSON                                   |
|----------------------------------------------------------------------------------------|------------|-----------------------------------------------|
| hunyuan_image_3                                                                        | 102        | result_cvtg_cvtg102ch_opus_g3p1p.json         |
| cosmos3_ga_64bm32b_image_only_data_mixture_000_iter20k                                 | 102        | result_cvtg_cvtg102ch_opus_g3p1p.json         |
| cosmos3_ga_64bm32b_t2ionly_base_v1_v1p4_iter100k                                       | 102        | result_cvtg_cvtg102ch_opus_g3p1p.json         |
| cosmos3_ga_64bm32b_t2ionly_base_iter_000032000_1024                                    | 102        |                                               |
| cosmos3_ga_64bm32b_t2ionly_base_iter_000036000_v1p3                                    | 102        | result_cvtg_cvtg102ch_opus_g3p1p.json         |
| cosmos3_ga_64bm32b_t2ionly_base_iter_000036000_v1p3_1024                               | 102        | result_cvtg_cvtg102ch_opus_g3p1p.json         |
| cosmos3_ga_64bm32b_t2ionly_base_iter_000036000_v1p3_1024_ensure_ascii_False            | 102        | result_cvtg_cvtg102ch_opus_g3p1p.json         |
| cosmos3_ga_64bm32b_t2ionly_base_iter_000036000_v1p3_1024_ensure_ascii_True             | 102        |                                               |
| cosmos3_ga_64bm32b_v3p1_midtrain_iter000000900                                         | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| -----------------------------------------------------------------------------          | ---------- | --------------------------------------------- |
| cosmos3_ga_64bm32b_t2ionly_base_v1_v1p5_iter108k                                       | 102        | result_cvtg_cvtg102ch_opus_g3p1p.json         |
| cosmos3_ga_16bm8b_v2_midtrain_iter6k                                                   | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_v3_midtrain_iter1800                                                | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_base_v2_v1_iter1k                                           | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_base_iter_000040000_v1p4                                    | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_base_iter_000033000_v1p2                                    | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_image_only_v3p3_iter22k                                             | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |

| cosmos3_ga_64bm32b_t2ionly_exp005_text_mixture3_sft_iter1k                             | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp006_text_mixture3_sft_from_frozen_midtrain_iter500       | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp007_text_mixture4_sft_lr1em5_iter4k                      | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp008_text_mixture4_sft_from_frozen_midtrain_lr1em5_iter4k | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter4k            | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter10k           | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter15k           | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter20k           | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter25k           | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter26k              | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter28k              | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter30k              | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter35k              | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter26k             | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter28k             | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter30k             | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter35k             | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter40k             | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_merged_000                                                  | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_merged_003                                                  | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_merged_006                                                  | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_merged_007                                                  | 102        | result_cvtg_cvtg102ch_opus_ascii_g3p1p.json   |

### cvtg500L/
| Run                | Images   | Result JSON                                                                       |
|--------------------|----------|-----------------------------------------------------------------------------------|
| flux_1_kontext_dev | 500      | result_cvtg_cvtg500L_g3p1p.json                                                   |
| flux_2_dev         | 500      | result_cvtg_cvtg500L_g3p1p.json                                                   |
| flux_2_klein_9b    | 500      | result_cvtg_cvtg500L_g3p1p.json                                                   |
| hunyuan_image_3    | 500      | result_cvtg_cvtg500L_g3p1p.json                                                   |
| nano_banana_pro    | 500      | result_cvtg_cvtg500L_g3p1p.json, result_cvtg_cvtg500L_g3p1p_capital_agnostic.json |
| qwen_image         | 500      | result_cvtg_cvtg500L_g3p1p.json                                                   |
| qwen_image_2512    | 500      | result_cvtg_cvtg500L_g3p1p.json                                                   |
| sd_v3p5_large      | 500      | result_cvtg_cvtg500L_g3p1p.json                                                   |
| z_image_turbo      | 500      | result_cvtg_cvtg500L_g3p1p.json                                                   |

### cvtg500L_opus/
| Run                                                                                    | Images     | Result JSON                            |
|----------------------------------------------------------------------------------------|------------|----------------------------------------|
| cosmos3_ga_64bm32b_image_only_data_mixture_000_iter20k                                 | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_base_v1_v1p4_iter100k                                       | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_base_iter_000032000_1024                                    | 500        |                                        |
| cosmos3_ga_64bm32b_t2ionly_base_iter_000036000_v1p3                                    | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_base_iter_000036000_v1p3_1024                               | 500        |                                        |
| cosmos3_ga_64bm32b_t2ionly_base_iter_000040000_v1p4                                    | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_v3p1_midtrain_iter000000900                                         | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| -------------------------------------------------------------------------------        | ---------- | -------------------------------------- |
| cosmos3_ga_64bm32b_t2ionly_base_v1_v1p5_iter108k                                       | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_16bm8b_v2_midtrain_iter6k                                                   | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_v3_midtrain_iter1800                                                | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_base_iter_000033000_v1p2                                    | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_base_v2_v1_iter1k                                           | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_image_only_v3p3_iter22k                                             | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |

| cosmos3_ga_64bm32b_t2ionly_exp000_text_only_sft_iter1p5k                               | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp000_text_only_sft_iter4k                                 | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp001_text_mixture_sft_iter1500                            | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp002_text_only_sft_from_frozen_midtrain_iter1000          | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp003_text_mixture2_sft_iter1k                             | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp003_text_mixture2_sft_iter2k                             | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp004_text_mixture2_sft_from_frozen_midtrain_iter2k        | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp005_text_mixture3_sft_iter1k                             | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp006_text_mixture3_sft_from_frozen_midtrain_iter500       | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp007_text_mixture4_sft_lr1em5_iter4k                      | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp008_text_mixture4_sft_from_frozen_midtrain_lr1em5_iter4k | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter4k            | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter10k           | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter15k           | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter20k           | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter25k           | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter26k              | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter28k              | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter30k              | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter35k              |            |                                        |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter26k             | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter28k             | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter30k             | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter35k             | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_exp009_sft1_text_from_exp009_25k_lr1em5_iter40k             | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_merged_000                                                  | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_merged_003                                                  | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_merged_006                                                  | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
| cosmos3_ga_64bm32b_t2ionly_merged_007                                                  | 500        | result_cvtg_cvtg500L_opus_g3p1p.json   |
