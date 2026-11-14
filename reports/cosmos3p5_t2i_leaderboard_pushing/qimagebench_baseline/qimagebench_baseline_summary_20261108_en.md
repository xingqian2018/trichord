# QImageBench Score Summary (2026-11-08)

Summary of QImageBench scores for 6 public models and 2 Cosmos3 checkpoints.
For UGB results on the same public models, see [ugb_baseline_summary_20261023_en.md](../ugb_baseline/ugb_baseline_summary_20261023_en.md); for model sizes and upsamplers, see [ugb_public_t2i_models_20261022.md](../ugb_public_t2i_models_20261022.md).

- Prompts: arena1000 (1000) + cosmos500 (500), 1500 in total, one image per prompt. arena1000 also carries an OCR label: 350 prompts ask for text in the image (`arena1000_ocr`), 650 do not (`arena1000_noocr`). Manifests are in `s3://nv-00-10206-vfm/debug/xingqianx/evaluation/qimagebench/` (`arena1000.json`, `cosmos500.json`).
- Result folders (each holds its result file `qimagebench_result.json`):
  - Public models: `s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/qimagebench_baseline/default/<run>/`
  - Cosmos3: `s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/qimagebench/c3json_opus4p7/<run>/`
- Scores range from 0 to 100.

---

## 0. Four things to know before reading the tables

1. **Generation prompts differ; scoring prompts are the same.** Public models get the original prompt, and FLUX.3, Fibo, Ideogram and Qwen-Image also rewrite it with each vendor's official flow; Cosmos3 gets JSON prompts rewritten by Opus 4.7 with the official Cosmos3-Super-Text2Image template (`c3json_opus4p7`). Scoring always uses the original prompt. This is therefore a system-level "rewrite + generation" comparison, and resolutions also differ (Section 5).
2. **overall is the unweighted mean of the two subsets.** overall = (arena1000 + cosmos500) / 2, not the mean over all 1500 images; the same holds for every L1 and L2 column.
3. **Judge and scoring.** The judge is Q-Judger (`Qwen/Qwen-Image-Bench`, built on Qwen3.6-27B), served with vLLM on Lepton; the evaluation protocol is ported verbatim from the official repo.
   - The judge scores each L3 checklist item 0 / 1 / 2, mapped to 0 / 60 / 100; L3 items are averaged into L2, L2 into L1, and L1 into the image score.
   - Quality, Aesthetics and Alignment are tested on every prompt. Real-world Fidelity (RWF) and Creative Generation (CG) are tested on a subset, labeled by opus-5.5, kimi-k3 and gpt-6-astra: RWF is tested if any model selects it, CG by majority vote. In arena1000, 513 prompts test RWF and 613 test CG; in cosmos500, 373 and 189. The labeling method and its agreement with the official labels are in [DIMS_REPORT.md](../../scripts/t2i_eval/blend/DIMS_REPORT.md).
4. **Every run scored 1500/1500 images.** For 4 images the judge output for Quality could not be parsed (status=failed); only their Quality score is dropped, the other dimensions still count: Nano Banana 2.1 has 1 (`cosmos500_0433`), Cosmos3 sft0 UHQ has 2 (`arena1000_0691`, `cosmos500_0213`), Cosmos3 exp009c has 1 (`arena1000_0591`).

---

## 1. Overall scores

| Model | overall | arena1000 | cosmos500 | Resolution | run |
|---|---|---|---|---|---|
| GPT-image-2.5 sunburst, quality low | **62.53** | 63.73 | 61.34 | 1024² | `gpt_image_2p5_sunburst_quality_low` |
| Nano Banana 2.1 | 61.04 | 62.55 | 59.52 | 1024² | `nano_banana_2p1` |
| Qwen-Image 2.1, pe | 59.33 | 60.99 | 57.66 | 2048² | `qwen_image_2p1_pe` |
| Ideogram 4 nf4 (9.3B×2, 48 steps) | 58.51 | 60.27 | 56.75 | 2048² | `ideogram4_nf4` |
| FLUX.3 klein-base (5.6B, 50 steps, cfg 3.5) | 58.24 | 60.05 | 56.42 | 1024² | `flux3_klein_base` |
| Fibo 2.0 dev (8.1B, 50 steps, cfg 5.0) | 57.63 | 59.41 | 55.84 | 1024² | `fibo2_dev` |
| **Cosmos3 64bm32b dense, exp009 sft0 UHQ, iter31000** | **55.08** | 57.44 | 52.72 | 1024² | `exp009_sft0_uhq_iter31k` |
| Cosmos3 64bm32b dense, exp009c union5, iter100 | 54.10 | 56.15 | 52.06 | 1024² | `exp009c_x9c_iter100` |

Observations:
- **API models lead.** GPT-image-2.5 sunburst is highest (62.53) and Nano Banana 2.1 second (61.04). Among open models, Qwen-Image 2.1 pe is best (59.33); the next three, Ideogram 4 nf4 (58.51), FLUX.3 klein-base (58.24) and Fibo 2.0 dev (57.63), are within 0.9 of each other.
- **Cosmos3 sft0 UHQ is below every public model: 2.5 below Fibo 2.0 dev, 3.2 below FLUX.3 klein-base, 4.2 below Qwen-Image 2.1 and 7.5 below GPT-image-2.5.**
- **exp009c iter100 is 1.0 below sft0 UHQ**, mostly in Creative Generation (−3.4).
- Every model scores lower on cosmos500 than on arena1000 (by 2.4–4.7); Cosmos3 sft0 UHQ drops the most (−4.7).

---

## 2. L1 dimensions (overall)

| Model | overall | Quality | Aesthetics | Alignment | Real-world Fidelity | Creative Generation |
|---|---|---|---|---|---|---|
| GPT-image-2.5 sunburst, low | **62.53** | **58.16** | **67.68** | **62.33** | **56.57** | **70.96** |
| Nano Banana 2.1 | 61.04 | 57.72 | 65.71 | 60.69 | 54.12 | 68.53 |
| Qwen-Image 2.1, pe | 59.33 | 57.56 | 64.83 | 56.76 | 53.80 | 64.36 |
| Ideogram 4 nf4 | 58.51 | 57.36 | 62.46 | 57.13 | 53.40 | 63.14 |
| FLUX.3 klein-base | 58.24 | 56.32 | 63.07 | 57.37 | 51.65 | 63.35 |
| Fibo 2.0 dev | 57.63 | 56.19 | 62.71 | 56.79 | 50.87 | 59.26 |
| Cosmos3 sft0 UHQ iter31000 | 55.08 | 53.19 | 61.16 | 55.30 | 45.97 | 55.59 |
| Cosmos3 exp009c iter100 | 54.10 | 52.58 | 60.78 | 54.56 | 44.82 | 52.18 |
| *Cosmos3 sft0 UHQ − FLUX.3* | *−3.16* | *−3.13* | *−1.91* | *−2.07* | *−5.69* | *−7.76* |

Observations:
- Against FLUX.3 klein-base, **Cosmos3 sft0 UHQ loses most on Creative Generation (−7.8) and Real-world Fidelity (−5.7)**; Aesthetics and Alignment are only about 2 lower.
- GPT-image-2.5 ranks first on all 5 L1 dimensions.
- Ideogram 4 nf4 is within 1.8 of FLUX.3 on all 5 L1 dimensions. Fibo 2.0 dev is clearly lower only on Creative Generation (4.1 below FLUX.3, the lowest among public models) and within 0.8 of FLUX.3 on the other 4.

---

## 3. L2 dimensions (overall)

The last row of each table is Cosmos3 sft0 UHQ minus FLUX.3 klein-base.

**Quality**

| Model | Detail | Realism | Resolution |
|---|---|---|---|
| GPT-image-2.5 sunburst, low | 61.25 | 49.68 | 62.64 |
| Nano Banana 2.1 | 61.25 | 49.53 | 61.72 |
| Qwen-Image 2.1, pe | 60.80 | 49.84 | 61.64 |
| Ideogram 4 nf4 | 61.03 | 48.53 | 61.84 |
| FLUX.3 klein-base | 59.58 | 47.27 | 61.31 |
| Fibo 2.0 dev | 59.27 | 47.46 | 61.30 |
| Cosmos3 sft0 UHQ iter31000 | 55.71 | 41.82 | 61.10 |
| Cosmos3 exp009c iter100 | 54.84 | 41.14 | 60.82 |
| *sft0 UHQ − FLUX.3* | *−3.86* | *−5.45* | *−0.21* |

**Aesthetics**

| Model | Anatomical Portraiture | Color Harmony | Composition | Emotional Expression | Lighting | Style Control |
|---|---|---|---|---|---|---|
| GPT-image-2.5 sunburst, low | 25.07 | 74.01 | 68.38 | 64.32 | 74.02 | 75.40 |
| Nano Banana 2.1 | 21.88 | 71.89 | 66.13 | 63.71 | 71.96 | 73.29 |
| Qwen-Image 2.1, pe | 21.90 | 71.16 | 65.05 | 62.96 | 73.33 | 67.62 |
| Ideogram 4 nf4 | 18.37 | 69.26 | 63.01 | 60.57 | 66.87 | 69.81 |
| FLUX.3 klein-base | 13.44 | 70.22 | 63.80 | 61.19 | 69.28 | 68.91 |
| Fibo 2.0 dev | 12.21 | 70.06 | 62.38 | 61.56 | 70.62 | 65.36 |
| Cosmos3 sft0 UHQ iter31000 | 11.09 | 69.76 | 61.75 | 60.83 | 68.31 | 59.78 |
| Cosmos3 exp009c iter100 | 10.72 | 69.61 | 62.13 | 61.01 | 68.78 | 54.84 |
| *sft0 UHQ − FLUX.3* | *−2.35* | *−0.46* | *−2.05* | *−0.36* | *−0.97* | *−9.12* |

**Alignment**

| Model | Actions | Attributes | Layout | Relations | Scene |
|---|---|---|---|---|---|
| GPT-image-2.5 sunburst, low | 52.75 | 62.41 | 61.33 | 61.14 | 71.44 |
| Nano Banana 2.1 | 49.95 | 61.13 | 59.10 | 57.87 | 72.06 |
| Qwen-Image 2.1, pe | 45.92 | 57.60 | 55.51 | 53.44 | 69.19 |
| Ideogram 4 nf4 | 46.06 | 58.03 | 56.89 | 54.07 | 68.10 |
| FLUX.3 klein-base | 46.66 | 59.73 | 57.45 | 53.43 | 67.14 |
| Fibo 2.0 dev | 46.34 | 58.91 | 56.64 | 51.26 | 69.13 |
| Cosmos3 sft0 UHQ iter31000 | 45.05 | 57.76 | 56.35 | 49.90 | 66.16 |
| Cosmos3 exp009c iter100 | 44.57 | 57.24 | 55.70 | 48.15 | 66.51 |
| *sft0 UHQ − FLUX.3* | *−1.61* | *−1.97* | *−1.09* | *−3.53* | *−0.98* |

**Real-world Fidelity**

| Model | Fairness | Safety & Compliance | World Knowledge |
|---|---|---|---|
| GPT-image-2.5 sunburst, low | 60.04 | 59.88 | 51.67 |
| Nano Banana 2.1 | 60.07 | 59.94 | 46.20 |
| Qwen-Image 2.1, pe | 59.88 | 59.88 | 45.59 |
| Ideogram 4 nf4 | 60.00 | 59.71 | 45.30 |
| FLUX.3 klein-base | 60.08 | 59.94 | 40.73 |
| Fibo 2.0 dev | 60.00 | 59.94 | 38.83 |
| Cosmos3 sft0 UHQ iter31000 | 60.08 | 59.94 | 28.18 |
| Cosmos3 exp009c iter100 | 60.00 | 59.82 | 25.43 |
| *sft0 UHQ − FLUX.3* | *0.00* | *0.00* | *−12.55* |

**Creative Generation**

| Model | Design Applications | Feature Matching | Imagination | Logical Resolution | Text Rendering | Visual Storytelling |
|---|---|---|---|---|---|---|
| GPT-image-2.5 sunburst, low | 78.97 | 56.18 | 70.48 | 43.16 | 86.13 | 72.28 |
| Nano Banana 2.1 | 76.69 | 52.86 | 68.50 | 43.56 | 79.05 | 71.53 |
| Qwen-Image 2.1, pe | 73.80 | 49.37 | 65.05 | 31.74 | 82.53 | 65.86 |
| Ideogram 4 nf4 | 72.16 | 50.94 | 63.75 | 31.40 | 77.03 | 65.05 |
| FLUX.3 klein-base | 71.22 | 46.43 | 62.44 | 32.19 | 80.46 | 69.33 |
| Fibo 2.0 dev | 67.43 | 46.60 | 61.70 | 26.42 | 71.07 | 66.41 |
| Cosmos3 sft0 UHQ iter31000 | 65.19 | 46.20 | 64.81 | 24.01 | 58.87 | 67.51 |
| Cosmos3 exp009c iter100 | 62.91 | 40.00 | 63.70 | 18.96 | 55.07 | 65.85 |
| *sft0 UHQ − FLUX.3* | *−6.03* | *−0.23* | *+2.38* | *−8.17* | *−21.58* | *−1.82* |

Observations:
- **The gap is concentrated in a few L2 dimensions.** Against FLUX.3, Cosmos3 sft0 UHQ is within 2.4 points on 15 of the 23 L2 dimensions, and is 2.4 higher on Imagination. The large gaps are Text Rendering (−21.6), World Knowledge (−12.5), Style Control (−9.1), Logical Resolution (−8.2), Design Applications (−6.0) and Realism (−5.4).
- Against GPT-image-2.5, Text Rendering is 27.3 lower (58.87 vs 86.13) and World Knowledge 23.5 lower (28.18 vs 51.67).
- **The Real-world Fidelity gap comes entirely from World Knowledge.** Fairness and Safety & Compliance sit around 60 for every model and do not separate models.
- Anatomical Portraiture is low for every model (10.7–25.1).
- exp009c iter100 falls furthest behind sft0 UHQ on Feature Matching (−6.2), Logical Resolution (−5.1), Style Control (−4.9) and Text Rendering (−3.8).
- Of the two newly added open models, Fibo 2.0 dev trails FLUX.3 most on Text Rendering (−9.4) and Logical Resolution (−5.8); Ideogram 4 nf4 is 4.6 above FLUX.3 on World Knowledge (45.30) and 4.9 above on Anatomical Portraiture.

---

## 4. Subsets: text-rendering prompts and cosmos500

| Model | arena1000_ocr (350) | arena1000_noocr (650) | cosmos500 (500) |
|---|---|---|---|
| GPT-image-2.5 sunburst, low | **70.07** | **60.31** | **61.34** |
| Nano Banana 2.1 | 68.26 | 59.48 | 59.52 |
| Qwen-Image 2.1, pe | 66.95 | 57.78 | 57.66 |
| Ideogram 4 nf4 | 65.61 | 57.39 | 56.75 |
| FLUX.3 klein-base | 66.10 | 56.80 | 56.42 |
| Fibo 2.0 dev | 63.68 | 57.11 | 55.84 |
| Cosmos3 sft0 UHQ iter31000 | 60.02 | 56.04 | 52.72 |
| Cosmos3 exp009c iter100 | 58.72 | 54.76 | 52.06 |
| *Cosmos3 sft0 UHQ − FLUX.3* | *−6.08* | *−0.75* | *−3.70* |

Observations:
- **On arena prompts without text, Cosmos3 sft0 UHQ is roughly on par with FLUX.3** (56.04 vs 56.80) and even slightly higher on Aesthetics and Creative Generation (+0.2, +0.8); it is also only 1.1 and 1.3 below Fibo 2.0 dev and Ideogram 4 nf4. **On prompts that ask for text, it is 6.1 lower.** The arena1000 gap comes mainly from the text prompts.
- Public models score 6.6–9.8 higher on text prompts than on non-text prompts (Fibo 2.0 dev has the smallest gain); Cosmos3 only 4.0 higher.
- **On cosmos500 the gap is 3.7**, mostly in Creative Generation (−10.9) and Real-world Fidelity (−6.7).

---

## 5. Generation settings

| Model | Resolution | Generation prompt / upsampler | Other |
|---|---|---|---|
| GPT-image-2.5 sunburst | 1024² | Original prompt, handled inside the API | `quality=low` |
| Nano Banana 2.1 | 1024² | Original prompt, handled inside the API | gateway `gcp/google/gemini-nano-banana-2.1` |
| Qwen-Image 2.1, pe | 2048² | Original prompt + official prompt enhancement (pe) | — |
| FLUX.3 klein-base | 1024² | Claude Opus 4.8 + official BFL system prompt | 50 steps, cfg 3.5 |
| Fibo 2.0 dev | 1024² | bria-vgl + gemini-3.5-flash-lite | 50 steps, cfg 5.0, official JSON negative prompt and fonts |
| Ideogram 4 nf4 | 2048² | Claude Opus 4.8 + official `v1.txt` system prompt | pipeline default 48 steps; 1 rewritten caption exceeded the 2048-token limit and was truncated to 2040 |
| Cosmos3 sft0 UHQ / exp009c | 1024² | `c3json_opus4p7`: Opus 4.7 + official Cosmos3-Super-Text2Image template (768, 1:1) | cfg 4, 50 steps, cosmos3 negative prompt |

Cosmos3 checkpoints (both under `s3://nv-00-10206-checkpoint/cosmos3_vfm/cosmos3_ga_t2ionly/`):
- sft0 UHQ: `cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5/checkpoints/iter_000031000`, EMA weights.
- exp009c: `cosmos3_ga_64bm32b_t2ionly_exp009c_union5_from_frozen_midtrain_lr5em5/checkpoints/iter_000000100`.

Rewrite-model fallback (gpt-6-astra):
- `cosmos500_0046` (spray drone over a tea plantation): Anthropic models always return an empty response for this prompt, presumably blocked by a safety classifier. FLUX.3, Ideogram and `c3json_opus4p7` therefore rewrite this one prompt with gpt-6-astra.
- 5 Fibo 2.0 dev images use gpt-6-astra rewrites: `arena1000_0088`, `arena1000_0256`, `arena1000_0308`, `cosmos500_0110`, `cosmos500_0238`; the other 1495 use gemini. Whether to regenerate them with gemini is still undecided.
