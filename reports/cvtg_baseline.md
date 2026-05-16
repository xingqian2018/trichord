# CVTG Baseline Scoring Report

Date: 2026-05-12

---

## cvtg102ch_score

| Model            | gned   | pned   |
|------------------|--------|--------|
| flux_1           | 0.0301 | 0.0831 |
| flux_2           | 0.1602 | 0.4693 |
| glmimage         | 0.5540 | 0.8799 |
| npb              | 0.4272 | 0.8359 |
| qwen_image       | 0.4521 | 0.8160 |
| qwen_image_2512  | 0.4245 | 0.8417 |
| sd3p5            | 0.0341 | 0.0980 |
| zimage           | 0.5473 | 0.8527 |

---

## cvtg500L_score

| Model            | gned   | pned   |
|------------------|--------|--------|
| flux_1           | 0.2860 | 0.4894 |
| flux_2           | 0.5371 | 0.8564 |
| glmimage         | 0.6705 | 0.8562 |
| npb              | 0.5541 | 0.7942 |
| qwen_image       | 0.6600 | 0.9242 |
| qwen_image_2512  | 0.6448 | 0.9539 |
| sd3p5            | 0.2763 | 0.5235 |
| zimage           | 0.5891 | 0.9073 |

---

Source logs (GCP):
- `/home/xingqianx/log/slurm/cvtg102ch_score/`
- `/home/xingqianx/log/slurm/cvtg500L_score/`

---

# CVTG Newer Round — G3.1-Pro Judge

Date: 2026-05-14
Judge: gemini-3.1-pro (signature: g3p1p)
Images: s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/cvtg/

---

## cvtg500L — Baselines

| Model                       | benchmark     | gned       | pned       | success   |
|-----------------------------|---------------|------------|------------|-----------|
| cosmos3_image_v1p5_iter108k | cvtg500L_opus | **0.8480** | 0.8826     | 498/500   |
| nano_banana_pro             | cvtg500L      | 0.5924     | 0.7179     | 500/500   |
| flux_2_klein_9b             | cvtg500L      | 0.6618     | 0.8138     | 499/500   |
| qwen_image_2512             | cvtg500L      | 0.7968     | **0.9086** | 499/500   |
| qwen_image                  | cvtg500L      | 0.7348     | 0.8672     | 500/500   |
| z_image_turbo               | cvtg500L      | 0.7520     | 0.8695     | 499/500   |
| flux_2_dev                  | cvtg500L      | 0.7471     | 0.8498     | 499/500   |
| flux_1_kontext_dev          | cvtg500L      | 0.3572     | 0.4648     | 499/500   |
| sd_v3p5_large               | cvtg500L      | 0.3323     | 0.5242     | 498/500   |

## cvtg500L — Cosmos3

| Model                              | benchmark     | gned       | pned       | success   |
|------------------------------------|---------------|------------|------------|-----------|
| cosmos3_image_v1p5_iter108k        | cvtg500L_opus | 0.8480     | 0.8826     | 498/500   |
| cosmos3_image_v1p4_iter100k        | cvtg500L_opus | 0.8496     | 0.8884     | 500/500   |
| cosmos3_image_text_focused_iter20k | cvtg500L_opus | **0.8586** | **0.8912** | 500/500   |

---

## cvtg102ch — Baselines

| Model                       | benchmark      | gned       | pned       | success   |
|-----------------------------|----------------|------------|------------|-----------|
| cosmos3_image_v1p5_iter108k | cvtg102ch_opus | 0.4770     | 0.6264     | 100/102   |
| nano_banana_pro             | cvtg102ch      | 0.4600     | **0.7640** | 101/102   |
| flux_2_klein_9b             | cvtg102ch      | 0.1888     | 0.3445     | 98/102    |
| qwen_image_2512             | cvtg102ch      | 0.4633     | 0.7126     | 101/102   |
| qwen_image                  | cvtg102ch      | 0.4883     | 0.6846     | 102/102   |
| z_image_turbo               | cvtg102ch      | **0.4918** | 0.7332     | 101/102   |
| flux_2_dev                  | cvtg102ch      | 0.4433     | 0.6874     | 102/102   |
| flux_1_kontext_dev          | cvtg102ch      | 0.0432     | 0.0721     | 102/102   |
| sd_v3p5_large               | cvtg102ch      | 0.0598     | 0.1403     | 101/102   |

## cvtg102ch — Cosmos3

| Model                              | benchmark      | gned       | pned       | success   |
|------------------------------------|----------------|------------|------------|-----------|
| cosmos3_image_v1p5_iter108k        | cvtg102ch_opus | 0.4770     | 0.6264     | 100/102   |
| cosmos3_image_v1p4_iter100k        | cvtg102ch_opus | 0.5009     | 0.6290     | 102/102   |
| cosmos3_image_text_focused_iter20k | cvtg102ch_opus | **0.5067** | **0.6524** | 102/102   |

---

Note: sd_v3p5_large/cvtg500L and nano_banana_pro/cvtg500L Stage 2 scoring in progress as of 2026-05-14.
