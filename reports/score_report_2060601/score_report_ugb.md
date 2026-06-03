# UniGenBench (UGB) Baseline — v2_1170L_G3F

- Benchmark: `v2_1170L_G3F` (1170 prompts)
- Judge model: `gemini-3.1-pro` (signature `gemini-3p1-pro`)
- Stage 2 input root: `s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/unigenbench/v2_1170L_G3F/`
- Note: `glm_image` excluded (smaller image set, not comparable)

## Overall Score Table

| Model              | All (1170L)   | Orig (600L)   | Phi (570Phi)   | Success   |
|--------------------|---------------|---------------|----------------|-----------|
| nano_banana_pro    | **90.85%**    | **92.91%**    | **88.95%**     | 1170/1170 |
| flux_2_klein_9b    | 85.22%        | 88.01%        | 82.66%         | 1170/1170 |
| qwen_image_2512    | 84.36%        | 87.53%        | 81.47%         | 1170/1170 |
| qwen_image         | 83.00%        | 86.48%        | 79.80%         | 1170/1170 |
| z_image_turbo      | 77.57%        | 81.12%        | 74.30%         | 1170/1170 |
| flux_1_kontext_dev | 67.95%        | 72.40%        | 63.87%         | 1170/1170 |
| sd_v3p5_large      | 63.69%        | 68.78%        | 59.02%         | 1170/1170 |

Sorted by All (1170L) accuracy, descending. Bold = top score in column.

## Primary Dimensions (All / 1170L split)

| Primary Dimension   | nano_banana_pro   | flux_2_klein_9b   | qwen_image_2512   | qwen_image   | z_image_turbo   | flux_1_kontext_dev   | sd_v3p5_large   |
|---------------------|-------------------|-------------------|-------------------|--------------|-----------------|----------------------|-----------------|
| Action              | 78.49%            | 69.51%            | 71.28%            | 68.66%       | 60.26%          | 47.41%               | 42.75%          |
| Attribute           | 95.76%            | 93.60%            | 93.14%            | 92.64%       | 88.25%          | 79.46%               | 77.34%          |
| Compound            | 87.68%            | 80.92%            | 75.85%            | 74.88%       | 66.18%          | 62.32%               | 52.42%          |
| Entity Layout       | 93.15%            | 88.92%            | 86.49%            | 84.77%       | 80.54%          | 71.98%               | 64.14%          |
| Grammar             | 92.98%            | 83.33%            | 75.88%            | 70.18%       | 74.56%          | 71.93%               | 70.18%          |
| Logical Reasoning   | 82.58%            | 67.74%            | 67.74%            | 58.06%       | 54.19%          | 41.29%               | 34.84%          |
| Relationship        | 90.11%            | 85.16%            | 81.90%            | 81.50%       | 74.48%          | 62.02%               | 56.38%          |
| Style               | 99.35%            | 98.06%            | 95.65%            | 95.65%       | 94.84%          | 90.16%               | 89.68%          |
| Text Generation     | 89.90%            | 53.37%            | 66.83%            | 62.02%       | 47.12%          | 35.10%               | 19.71%          |
| World Knowledge     | 94.80%            | 90.52%            | 90.52%            | 92.35%       | 87.16%          | 72.78%               | 76.15%          |

## Correct / Total Counts (All / 1170L)

| Primary Dimension   | Total    | nano_banana_pro   | flux_2_klein_9b   | qwen_image_2512   | qwen_image   | z_image_turbo   | flux_1_kontext_dev   | sd_v3p5_large   |
|---------------------|----------|-------------------|-------------------|-------------------|--------------|-----------------|----------------------|-----------------|
| Action              | 1525     | 1197              | 1060              | 1087              | 1047         | 919             | 723                  | 652             |
| Attribute           | 2595     | 2485              | 2429              | 2417              | 2404         | 2290            | 2062                 | 2007            |
| Compound            | 414      | 363               | 335               | 314               | 310          | 274             | 258                  | 217             |
| Entity Layout       | 1110     | 1034              | 987               | 960               | 941          | 894             | 799                  | 712             |
| Grammar             | 228      | 212               | 190               | 173               | 160          | 170             | 164                  | 160             |
| Logical Reasoning   | 155      | 128               | 105               | 105               | 90           | 84              | 64                   | 54              |
| Relationship        | 1011     | 911               | 861               | 828               | 824          | 753             | 627                  | 570             |
| Style               | 620      | 616               | 608               | 593               | 593          | 588             | 559                  | 556             |
| Text Generation     | 208      | 187               | 111               | 139               | 129          | 98              | 73                   | 41              |
| World Knowledge     | 327      | 310               | 296               | 296               | 302          | 285             | 238                  | 249             |
| **Overall**         | **8193** | **7443**          | **6982**          | **6912**          | **6800**     | **6355**        | **5567**             | **5218**        |

## Reasoner-Upsampler v1 (apr27) — qwen_image_2512

- Model: `qwen_image_2512` (guidance 4.0, 50 steps, 1328x1328, auto-applied Chinese negative prompt)
- Judge model: `gemini-3.1-pro` (signature `gemini-3p1-pro`)
- Stage 2 input root: `s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/unigenbench/<benchmark>/qwen_image_2512/`
- Same 1170 prompts as `v2_1170L_G3F`, but rewritten by different upsamplers (Opus / Qwen3VL8B / pre_exp015_372_ft8b)

| Gen-Benchmark          | Eval-Benchmark         | All (1170L)   | Orig (600L)   | Phi (570Phi)   | Success   |
|------------------------|------------------------|---------------|---------------|----------------|-----------|
| v2_1170L_G3F           | v2_1170L_G3F           | **84.36%**    | **87.53%**    | **81.47%**     | 1170/1170 |
| v2_1170L_qwen3vl8b     | v2_1170L_qwen3vl8b     | 76.96%        | 79.90%        | 74.26%         | 1170/1170 |
| v2_1170L_qwen3vl8b     | v2_1170L_G3F           | 76.52%        | 79.82%        | 73.48%         | 1170/1170 |
| v2_1170L_opus          | v2_1170L_opus          | 74.11%        | 79.80%        | 68.90%         | 1170/1170 |
| v2_1170L_opus          | v2_1170L_G3F           | 73.77%        | 79.87%        | 68.17%         | 1170/1170 |
| v2_1170L_preexp015ft8b | v2_1170L_preexp015ft8b | 70.73%        | 75.20%        | 66.63%         | 1170/1170 |
| v2_1170L_preexp015ft8b | v2_1170L_G3F           | 70.85%        | 75.82%        | 66.30%         | 1170/1170 |

Grouped by Gen-Benchmark, with the `gen = eval` row first and the `eval = v2_1170L_G3F` row second within each group. Bold = top score in column. *Gen-Benchmark* = prompt set fed to the image generator; *Eval-Benchmark* = prompt set the judge model sees alongside the image.

For each upsampler, the two eval variants land within ~0.5pt of each other — re-judging the same image against the original short prompt instead of the upsampled long prompt barely shifts the score. So the drop relative to the all-G3F baseline (84.36% → ~74%) is *not* mostly a stricter-judge effect; it's the generator producing weaker images when conditioned on a longer/more specific prompt.

Result files (one per row, alongside the gen images):
- `unigenbench_result_gemini-3p1-pro.json` — `gen = eval` rows
- `unigenbench_result_using_eval_prompt_v2_1170L.json` — `eval = v2_1170L_G3F` rows
- The original Stage 1 config snapshot now lives at `config_gen.json` (renamed from `config.json`) so the Stage 2 script falls through to the `--benchmark_name` flag.

---

# UGB Newer Round — G3.1-Pro Judge

Date: 2026-05-14
Judge: gemini-3.1-pro (signature: gemini-3p1-pro_G3F)
Benchmark: v2_1170L_G3F
Stage 2 input root: `s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/unigenbench/v2_1170L_G3F/`

---

## v2_1170L_G3F — Baselines

| Model                       | benchmark     | All (1170L)   | Orig (600L)   | Phi (570Phi)   | Success   |
|-----------------------------|---------------|---------------|---------------|----------------|-----------|
| cosmos3_image_v1p5_iter108k | v2_1170L_opus | **91.26**     | **93.34**     | **89.35**      | 1170/1170 |
| nano_banana_pro             | v2_1170L_G3F  | 90.85         | 92.91         | 88.95          | 1170/1170 |
| flux_2_klein_9b             | v2_1170L_G3F  | 85.22         | 88.01         | 82.66          | 1170/1170 |
| qwen_image_2512             | v2_1170L_G3F  | 84.36         | 87.53         | 81.47          | 1170/1170 |
| qwen_image                  | v2_1170L_G3F  | 83            | 86.48         | 79.8           | 1170/1170 |
| z_image_turbo               | v2_1170L_G3F  | 77.57         | 81.12         | 74.3           | 1170/1170 |
| flux_1_kontext_dev          | v2_1170L_G3F  | 67.95         | 72.4          | 63.87          | 1170/1170 |
| sd_v3p5_large               | v2_1170L_G3F  | 63.69         | 68.78         | 59.02          | 1170/1170 |

## v2_1170L_G3F — Cosmos3

| Model                              | benchmark     | All (1170L)   | Orig (600L)   | Phi (570Phi)   | Success   |
|------------------------------------|---------------|---------------|---------------|----------------|-----------|
| cosmos3-nano                       | v2_1170L_opus | 84.22         | 86.17         | 82.42          | 1170/1170 |
| cosmos3-frozen-midtrain-v3p1       | v2_1170L_opus | 86.23         | 88.16         | 84.46          | 1170/1170 |
| cosmos3-frozen-midtrain-v3         | v2_1170L_opus | 86.61         | 88.24         | 85.12          | 1170/1170 |
| cosmos3_image_v2_v1_iter1k         | v2_1170L_opus | 88.92         | 90.46         | 87.50          | 1170/1170 |
| cosmos3_image_text_focused_iter20k | v2_1170L_opus | 88.33         | 90.31         | 86.52          | 1170/1170 |
| cosmos3_image_v1_v1p4_iter100k     | v2_1170L_opus | 89.38         | 91.3          | 87.62          | 1170/1170 |
| cosmos3_image_v1_v1p5_iter108k     | v2_1170L_opus | 90.81         | 93.34         | 88.49          | 1170/1170 |
| cosmos3_image_v2_v1p2_iter33k      | v2_1170L_opus | 91.26         | 93.70         | 89.02          | 1170/1170 |
| cosmos3-image_v2_v1p4_iter40k      | v2_1170L_opus | **91.32**     | **93.37**     | **89.45**      | 1170/1170 |


## v2_1170L_G3F — Cosmos3 T2I-Only SFT

| Model                                                       | benchmark     | All (1170L)   | Orig (600L)   | Phi (570Phi)   | Success   |
|-------------------------------------------------------------|---------------|---------------|---------------|----------------|-----------|
| cosmos3_t2i_exp000_text_only_iter4k                         | v2_1170L_opus | 85.87         | 90.23         | 81.86          | 1170/1170 |
| cosmos3_t2i_exp000_text_only_iter1p5k                       | v2_1170L_opus | 88.55         | 91.30         | 86.03          | 1170/1170 |
| cosmos3_t2i_exp001_text_mix_iter1k                          | v2_1170L_opus | 90.35         | 92.24         | 88.60          | 1170/1170 |
| cosmos3_t2i_exp001_text_mix_iter1p5k                        | v2_1170L_opus | 89.70         | 91.71         | 87.85          | 1170/1170 |
| cosmos3_t2i_exp002_text_only_from_frozen_iter1k             | v2_1170L_opus | 84.62         | 87.37         | 82.10          | 1170/1170 |
| cosmos3_t2i_exp003_text_mix2_iter1k                         | v2_1170L_opus | 87.71         | 90.03         | 85.58          | 1170/1170 |
| cosmos3_t2i_exp003_text_mix2_iter2k                         | v2_1170L_opus | 87.57         | 89.85         | 85.49          | 1170/1170 |
| cosmos3_t2i_exp004_text_mix2_from_frozen_iter2k             | v2_1170L_opus | 83.23         | 85.97         | 80.72          | 1170/1170 |
| cosmos3_t2i_exp005_text_mix3_iter1k                         | v2_1170L_opus | 87.46         | 90.64         | 84.55          | 1170/1170 |
| cosmos3_t2i_exp006_text_mix3_from_frozen_iter500            | v2_1170L_opus |               |               |                |           |
| cosmos3_t2i_exp007_text_mix4_iter4k                         | v2_1170L_opus | 88.26         | 90.43         | 86.26          | 1170/1170 |
| cosmos3_t2i_exp008_text_mix4_from_frozen_iter4k             | v2_1170L_opus | 87.20         | 89.36         | 85.21          | 1170/1170 |
| cosmos3_t2i_exp009_union5_from_frozen_iter4k                | v2_1170L_opus | 89.69         | 91.28         | 88.23          | 1170/1170 |
| cosmos3_t2i_exp009_union5_from_frozen_iter10k               | v2_1170L_opus | 90.57         | 92.19         | 89.07          | 1170/1170 |
| cosmos3_t2i_exp009_union5_from_frozen_iter15k               | v2_1170L_opus | 90.97         | 93.14         | 88.98          | 1170/1170 |
| cosmos3_t2i_exp009_union5_from_frozen_iter20k               | v2_1170L_opus | 91.10         | 93.01         | 89.35          | 1170/1170 |
| cosmos3_t2i_exp009_union5_from_frozen_iter25k               | v2_1170L_opus | 91.14         | 93.29         | 89.16          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter26k  | v2_1170L_opus | 91.50         | 93.75         | 89.45          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter28k  | v2_1170L_opus | 91.41         | **94.06**     | 88.98          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter30k  | v2_1170L_opus | 91.25         | 93.44         | 89.23          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k  | v2_1170L_opus | 91.36         | 93.34         | 89.54          | 1170/1170 |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter35k  | v2_1170L_opus | 89.83         | 92.14         | 87.71          | 1170/1170 |
| cosmos3_t2i_exp009_sft1_text_from_exp009_25k_lr1em5_iter26k | v2_1170L_opus | 91.24         | 93.21         | 89.42          | 1170/1170 |
| cosmos3_t2i_exp009_sft1_text_from_exp009_25k_lr1em5_iter28k | v2_1170L_opus | 91.33         | 93.24         | 89.59          | 1170/1170 |
| cosmos3_t2i_exp009_sft1_text_from_exp009_25k_lr1em5_iter30k | v2_1170L_opus | 90.92         | 93.01         | 89.00          | 1170/1170 |
| cosmos3_t2i_exp009_sft1_text_from_exp009_25k_lr1em5_iter35k | v2_1170L_opus | 90.11         | 92.24         | 88.16          | 1170/1170 |
| cosmos3_t2i_exp009_sft1_text_from_exp009_25k_lr1em5_iter40k | v2_1170L_opus | 90.26         | 92.55         | 88.16          | 1170/1170 |
| cosmos3_t2i_merged_000                                      | v2_1170L_opus | 91.20         | 93.37         | 89.21          | 1170/1170 |
| cosmos3_t2i_merged_003                                      | v2_1170L_opus | **91.98**     | 94.03         | 90.10          | 1170/1170 |
| cosmos3_t2i_merged_006                                      | v2_1170L_opus | 91.63         | 93.95         | 89.49          | 1170/1170 |
| cosmos3_t2i_merged_007                                      | v2_1170L_opus | **91.98**     | 94.01         | **90.12**      | 1170/1170 |
| cosmos3_t2i_exp010_sft0_union6_from_merge007_iter9k         | v2_1170L_opus |               |               |                |           |

