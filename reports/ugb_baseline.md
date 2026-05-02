# UniGenBench (UGB) Baseline — v2_1170L_G3F

- Benchmark: `v2_1170L_G3F` (1170 prompts)
- Judge model: `gemini-3.1-pro` (signature `gemini-3p1-pro`)
- Stage 2 input root: `s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/unigenbench/v2_1170L_G3F/`
- Note: `glm_image` excluded (smaller image set, not comparable)

## Overall Score Table

| Model              | All (1170L) | Orig (600L) | Phi (570Phi) | Success |
|--------------------|------------:|------------:|-------------:|:-------:|
| nano_banana_pro    | **90.85%**  | **92.91%**  | **88.95%**   | 1170/1170 |
| flux_2_klein_9b    | 85.22%      | 88.01%      | 82.66%       | 1170/1170 |
| qwen_image_2512    | 84.36%      | 87.53%      | 81.47%       | 1170/1170 |
| qwen_image         | 83.00%      | 86.48%      | 79.80%       | 1170/1170 |
| z_image_turbo      | 77.57%      | 81.12%      | 74.30%       | 1170/1170 |
| flux_1_kontext_dev | 67.95%      | 72.40%      | 63.87%       | 1170/1170 |
| sd_v3p5_large      | 63.69%      | 68.78%      | 59.02%       | 1170/1170 |

Sorted by All (1170L) accuracy, descending. Bold = top score in column.

## Primary Dimensions (All / 1170L split)

| Primary Dimension | nano_banana_pro | flux_2_klein_9b | qwen_image_2512 | qwen_image | z_image_turbo | flux_1_kontext_dev | sd_v3p5_large |
|---|---:|---:|---:|---:|---:|---:|---:|
| Action            | 78.49% | 69.51% | 71.28% | 68.66% | 60.26% | 47.41% | 42.75% |
| Attribute         | 95.76% | 93.60% | 93.14% | 92.64% | 88.25% | 79.46% | 77.34% |
| Compound          | 87.68% | 80.92% | 75.85% | 74.88% | 66.18% | 62.32% | 52.42% |
| Entity Layout     | 93.15% | 88.92% | 86.49% | 84.77% | 80.54% | 71.98% | 64.14% |
| Grammar           | 92.98% | 83.33% | 75.88% | 70.18% | 74.56% | 71.93% | 70.18% |
| Logical Reasoning | 82.58% | 67.74% | 67.74% | 58.06% | 54.19% | 41.29% | 34.84% |
| Relationship      | 90.11% | 85.16% | 81.90% | 81.50% | 74.48% | 62.02% | 56.38% |
| Style             | 99.35% | 98.06% | 95.65% | 95.65% | 94.84% | 90.16% | 89.68% |
| Text Generation   | 89.90% | 53.37% | 66.83% | 62.02% | 47.12% | 35.10% | 19.71% |
| World Knowledge   | 94.80% | 90.52% | 90.52% | 92.35% | 87.16% | 72.78% | 76.15% |

## Correct / Total Counts (All / 1170L)

| Primary Dimension | Total | nano_banana_pro | flux_2_klein_9b | qwen_image_2512 | qwen_image | z_image_turbo | flux_1_kontext_dev | sd_v3p5_large |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Action            | 1525 | 1197 | 1060 | 1087 | 1047 |  919 |  723 |  652 |
| Attribute         | 2595 | 2485 | 2429 | 2417 | 2404 | 2290 | 2062 | 2007 |
| Compound          |  414 |  363 |  335 |  314 |  310 |  274 |  258 |  217 |
| Entity Layout     | 1110 | 1034 |  987 |  960 |  941 |  894 |  799 |  712 |
| Grammar           |  228 |  212 |  190 |  173 |  160 |  170 |  164 |  160 |
| Logical Reasoning |  155 |  128 |  105 |  105 |   90 |   84 |   64 |   54 |
| Relationship      | 1011 |  911 |  861 |  828 |  824 |  753 |  627 |  570 |
| Style             |  620 |  616 |  608 |  593 |  593 |  588 |  559 |  556 |
| Text Generation   |  208 |  187 |  111 |  139 |  129 |   98 |   73 |   41 |
| World Knowledge   |  327 |  310 |  296 |  296 |  302 |  285 |  238 |  249 |
| **Overall**       | **8193** | **7443** | **6982** | **6912** | **6800** | **6355** | **5567** | **5218** |

