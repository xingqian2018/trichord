# UniGenBench Score Report

## v2_1170L — Baselines

| Model              | benchmark    | All (1170L)   | Orig (600L)   | Phi (570Phi)   | Success   |
|--------------------|--------------|---------------|---------------|----------------|-----------|
| nano_banana_pro    | v2_1170L_G3F | 90.85         | 92.91         | 88.95          | 1170/1170 |
| flux_2_klein_9b    | v2_1170L_G3F | 85.22         | 88.01         | 82.66          | 1170/1170 |
| qwen_image_2512    | v2_1170L_G3F | 84.36         | 87.53         | 81.47          | 1170/1170 |
| qwen_image         | v2_1170L_G3F | 83            | 86.48         | 79.8           | 1170/1170 |
| z_image_turbo      | v2_1170L_G3F | 77.57         | 81.12         | 74.3           | 1170/1170 |
| flux_1_kontext_dev | v2_1170L_G3F | 67.95         | 72.4          | 63.87          | 1170/1170 |
| sd_v3p5_large      | v2_1170L_G3F | 63.69         | 68.78         | 59.02          | 1170/1170 |



## v2_1170L_G3F — Cosmos3 T2I-Only SFT

| Model                                                      | benchmark     | All (1170L)   | Orig (600L)   | Phi (570Phi)   | Success   |
|------------------------------------------------------------|---------------|---------------|---------------|----------------|-----------|
| cosmos3_t2i_exp009_union5_from_frozen_iter25k              | v2_1170L_opus | 91.14         | 93.29         | 89.16          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L_opus | 91.36         | 93.34         | 89.54          | 1170/1170 |

## Multi-Judge — cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k

| Model                                                      | benchmark               | All (1170L)   | Orig (600L)   | Phi (570Phi)   | Success   |
|------------------------------------------------------------|-------------------------|---------------|---------------|----------------|-----------|
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L                | 88.09         | 91.10         | 85.33          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L_nemotron3ultra | 88.58         | 90.61         | 86.71          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L_qwen3p5_397b   | 89.47         | 92.12         | 87.03          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L_gpt5p5         | 90.96         | 92.88         | 89.19          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L_opus           | 91.36         | 93.34         | 89.54          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L_gemini3p1pro   | **91.93**     | **93.62**     | **90.38**      | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L_opus4p7        | 90.21         | 92.53         | 88.09          | 1170/1170 |

## Prompt Format Ablation — cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k (judge: v2_1170L_gpt5p5)

| Model                                                                    | benchmark       | All (1170L)   | Orig (600L)   | Phi (570Phi)   | Success   |
|--------------------------------------------------------------------------|-----------------|---------------|---------------|----------------|-----------|
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k (json)        | v2_1170L_gpt5p5 | 90.96         | 92.88         | 89.19          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_html          | v2_1170L_gpt5p5 | 90.98         | 93.01         | 89.12          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_xml           | v2_1170L_gpt5p5 | 91.52         | 93.01         | 90.15          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_yaml          | v2_1170L_gpt5p5 | 91.40         | 93.09         | 89.84          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_markdown      | v2_1170L_gpt5p5 | 91.48         | 93.11         | 89.98          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_keyvalue      | v2_1170L_gpt5p5 | 90.99         | 92.83         | 89.3           | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_ast           | v2_1170L_gpt5p5 | 91.11         | 92.83         | 89.54          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_ascii_tree    | v2_1170L_gpt5p5 | 91.03         | 92.78         | 89.42          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_div_nullified | v2_1170L_gpt5p5 | 90.53         | 92.4          | 88.81          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_flattext      | v2_1170L_gpt5p5 | 91.14         | 93.06         | 89.38          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_bag_of_words  | v2_1170L_gpt5p5 | 46.53         | 47.96         | 45.21          | 1170/1170 |

## GPT5.5 Benchmark Version Ablation — cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k

| Model                                                      | benchmark            | All (1170L)   | Orig (600L)   | Phi (570Phi)   | Success   |
|------------------------------------------------------------|----------------------|---------------|---------------|----------------|-----------|
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L_gpt5p5      | 90.96         | 92.88         | 89.19          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L_gpt5p5_v6   | 89.48         | 92.27         | 86.92          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L_gpt5p5_v6p1 | 88.77         | 91.81         | 85.98          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L_gpt5p5_v6p2 | 89.11         | 92.22         | 86.26          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L_gpt5p5_v6p3 | 88.88         | 91.91         | 86.10          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L_gpt5p5_v6p4 | 89.60         | 92.27         | 87.15          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L_gpt5p5_v6p5 | 89.56         | 92.73         | 86.66          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L_gpt5p5_v6p6 | 89.75         | 92.40         | 87.32          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L_gpt5p5_v6p7 | 89.54         | 92.76         | 86.59          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k | v2_1170L_gpt5p5_v6p8 | 89.81         | 92.24         | 87.57          | 1170/1170 |

## GPT5.5 Benchmark Version Ablation (markdown) — cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k

| Model                                                               | benchmark            | All (1170L)   | Orig (600L)   | Phi (570Phi)   | Success   |
|---------------------------------------------------------------------|----------------------|---------------|---------------|----------------|-----------|
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_markdown | v2_1170L_gpt5p5      | 91.48         | 93.11         | 89.98          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_markdown | v2_1170L_gpt5p5_v6   | 88.89         | 92.17         | 85.89          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_markdown | v2_1170L_gpt5p5_v6p1 | 88.71         | 91.68         | 85.98          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_markdown | v2_1170L_gpt5p5_v6p2 | 89.81         | 92.47         | 87.36          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_markdown | v2_1170L_gpt5p5_v6p3 | 89.32         | 92.68         | 86.24          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_markdown | v2_1170L_gpt5p5_v6p4 | 90.14         | 93.19         | 87.34          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_markdown | v2_1170L_gpt5p5_v6p5 | 89.78         | 92.58         | 87.22          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_markdown | v2_1170L_gpt5p5_v6p6 | 89.26         | 92.32         | 86.45          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_markdown | v2_1170L_gpt5p5_v6p7 | 89.80         | 93.06         | 86.80          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_markdown | v2_1170L_gpt5p5_v6p8 | 89.42         | 92.19         | 86.87          | 1170/1170 |

## GPT5.5 Benchmark Version Ablation (xml) — cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k

| Model                                                            | benchmark            | All (1170L)   | Orig (600L)   | Phi (570Phi)   | Success   |
|------------------------------------------------------------------|----------------------|---------------|---------------|----------------|-----------|
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_xml  | v2_1170L_gpt5p5      | 91.52         | 93.01         | 90.15          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_xml  | v2_1170L_gpt5p5_v6   | 89.34         | 92.35         | 86.59          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_xml  | v2_1170L_gpt5p5_v6p1 | 88.75         | 91.71         | 86.03          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_xml  | v2_1170L_gpt5p5_v6p2 | 89.22         | 91.81         | 86.85          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_xml  | v2_1170L_gpt5p5_v6p3 | 89.21         | 92.45         | 86.24          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_xml  | v2_1170L_gpt5p5_v6p4 | 89.98         | 92.76         | 87.43          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_xml  | v2_1170L_gpt5p5_v6p5 | 89.83         | 92.50         | 87.39          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_xml  | v2_1170L_gpt5p5_v6p6 | 89.25         | 92.07         | 86.66          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_xml  | v2_1170L_gpt5p5_v6p7 | 90.04         | 92.42         | 87.85          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k_xml  | v2_1170L_gpt5p5_v6p8 | 89.65         | 92.60         | 86.94          | 1170/1170 |
