# UGB qwen_image_2512 — Score Analysis over Upsampled Prompts

Investigation: same generator (`qwen_image_2512`) is run on `v2_1170L_G3F` (original prompts) and on three reasoner-upsampler v1 (apr27) variants (Opus / Qwen3VL8B / preexp015ft8b). All four are judged on the same eval prompts (`v2_1170L_G3F`). Why does the upsampled-gen score drop so much vs. the original?

Judge: `gemini-3.1-pro`. Generator config: guidance 4.0, 50 steps, 1328x1328, auto-applied Chinese negative prompt.

## Headline Numbers

Three sliced views of the same dataset of runs.

### Table 1 — Gen and Eval prompts the same (diagonal)

| Gen-Benchmark            | Eval-Benchmark           | All (1170L) | Orig (600L) | Phi (570Phi) | Success |
|--------------------------|--------------------------|------------:|------------:|-------------:|:-------:|
| v2_1170L_G3F             | v2_1170L_G3F             | **84.36%**  | **87.53%**  | **81.47%**   | 1170/1170 |
| v2_1170L_qwen3vl8b       | v2_1170L_qwen3vl8b       | 76.96%      | 79.90%      | 74.26%       | 1170/1170 |
| v2_1170L_opus            | v2_1170L_opus            | 74.11%      | 79.80%      | 68.90%       | 1170/1170 |
| v2_1170L_preexp015ft8b   | v2_1170L_preexp015ft8b   | 70.73%      | 75.20%      | 66.63%       | 1170/1170 |

### Table 2 — Gen always v2_1170L_G3F (varying eval)

Same images (the original `xingqianx_qwen_image_2512` G3F-generated set), judged against four different prompt sets. Isolates the *judge* (eval-prompt) effect.

| Gen-Benchmark   | Eval-Benchmark           | All (1170L) | Orig (600L) | Phi (570Phi) | Success    |
|-----------------|--------------------------|------------:|------------:|-------------:|:----------:|
| v2_1170L_G3F    | v2_1170L_G3F             | **84.36%**  | **87.53%**  | 81.47%       | 1170/1170  |
| v2_1170L_G3F    | v2_1170L_qwen3vl8b       | 84.22%      | 87.04%      | **81.63%**   | 1170/1170  |
| v2_1170L_G3F    | v2_1170L_preexp015ft8b   | 83.00%      | 85.89%      | 80.34%       | 1170/1170  |
| v2_1170L_G3F    | v2_1170L_opus            | 82.68%      | 86.63%      | 79.05%       | 1169/1170  |

Spread: *1.68pt* across all four eval prompt sets on the same images. (For reference, a fresh re-judge of the same `gen=G3F, eval=G3F` setup on a separate signature `gemini-3p1-pro_rerun` came in at 84.27% — judge-side stochastic variance is ~0.1pt.)

### Table 3 — Eval always v2_1170L_G3F (varying gen)

Same eval prompts (the canonical `v2_1170L_G3F` set), but generated from four different prompt sets. Isolates the *generator* (gen-prompt) effect.

| Gen-Benchmark            | Eval-Benchmark   | All (1170L) | Orig (600L) | Phi (570Phi) | Success    |
|--------------------------|------------------|------------:|------------:|-------------:|:----------:|
| v2_1170L_G3F             | v2_1170L_G3F     | **84.36%**  | **87.53%**  | **81.47%**   | 1170/1170  |
| v2_1170L_qwen3vl8b       | v2_1170L_G3F     | 76.52%      | 79.82%      | 73.48%       | 1170/1170  |
| v2_1170L_opus            | v2_1170L_G3F     | 73.77%      | 79.87%      | 68.17%       | 1170/1170  |
| v2_1170L_preexp015ft8b   | v2_1170L_G3F     | 70.85%      | 75.82%      | 66.30%       | 1170/1170  |

Spread: *13.51pt* — nearly *8x* the eval-side spread.

### Takeaway

Comparing Tables 2 and 3:

- *Eval-side* (which prompt set the judge sees) shifts the score by ~1.7pt total.
- *Gen-side* (which prompt set the generator sees) shifts the score by ~13.5pt total.
- The dominant variable driving these scores is what prompt the generator was given. Re-judging the same image against an upsampled prompt barely shifts the verdict; conditioning the generator on the upsampled prompt drops it by 8–14pt.

The `gen = eval` rows in Table 1 land ~0.5pt away from the corresponding `eval = G3F` rows in Table 3, consistent with the same conclusion.

## A vs D — Original G3F gen vs preexp015ft8b gen (both eval=G3F)

A = `xingqianx_qwen_image_2512` on `v2_1170L_G3F` (84.36%, 6912 / 8193 correct)
D = `qwen_image_2512` on `v2_1170L_preexp015ft8b`, eval-G3F (70.85%, 5805 / 8193 correct)
Delta: *−13.51pt, −1107 correct*.

### Big-class breakdown (sorted by accuracy drop)

| Big Class         | A_acc   | D_acc   | Δ pt    | A_corr | D_corr | Lost  | Total |
|-------------------|--------:|--------:|--------:|-------:|-------:|------:|------:|
| Text Generation   | 66.83%  | 19.23%  | **−47.60** |    139 |     40 |    99 |   208 |
| Entity Layout     | 86.49%  | 65.41%  | −21.08  |    960 |    726 |   234 |  1110 |
| Style             | 95.65%  | 79.03%  | −16.61  |    593 |    490 |   103 |   620 |
| Relationship      | 81.90%  | 65.58%  | −16.32  |    828 |    663 |   165 |  1011 |
| Logical Reasoning | 67.74%  | 54.84%  | −12.90  |    105 |     85 |    20 |   155 |
| World Knowledge   | 90.52%  | 77.98%  | −12.54  |    296 |    255 |    41 |   327 |
| Attribute         | 93.14%  | 82.58%  | −10.56  |   2417 |   2143 |   274 |  2595 |
| Action            | 71.28%  | 61.31%  |  −9.97  |   1087 |    935 |   152 |  1525 |
| Compound          | 75.85%  | 70.29%  |  −5.56  |    314 |    291 |    23 |   414 |
| Grammar           | 75.88%  | 77.63%  | **+1.75** |    173 |    177 |    −4 |   228 |

### Top sub-classes by raw count lost

| Sub Class                              | A_acc  | D_acc  | Lost | Total |
|----------------------------------------|-------:|-------:|-----:|------:|
| Attribute - Color                      | 98.22% | 84.17% |  142 |  1011 |
| Entity Layout - Two-Dimensional Space  | 86.26% | 63.65% |  130 |   575 |
| Relationship - Composition             | 82.72% | 64.57% |  104 |   573 |
| Entity Layout - Three-Dimensional Space| 86.73% | 67.29% |  104 |   535 |
| Style                                  | 95.65% | 79.03% |  103 |   620 |
| Text Generation                        | 66.83% | 19.23% |   99 |   208 |
| Action - State                         | 82.08% | 67.55% |   77 |   530 |

### Reading

- **Text Generation collapses** (−47.6pt). When the upsampler rewords a text-rendering prompt, the literal string the model is asked to render is likely being paraphrased, reformatted, or buried in extra description — the model can't fish out the exact string anymore.
- **Spatial layout breaks** (Entity Layout 2D/3D, Relationship - Composition all drop 18–22pt). The upsampler likely adds redundant or contradictory spatial language.
- **Color / Style / Material** — high-baseline bins (>94% in A) all drop 10–16pt. Upsampled prompts probably overspecify or contradict the canonical color/style cue.
- **Grammar is flat** (+1.75pt) — the only category that did not degrade. Those prompts may already be short and the upsampler's expansion adds little.
- **Action, Logical Reasoning, World Knowledge** drop 10–13pt each.

## C vs D — Opus-upsampled gen vs preexp015ft8b-upsampled gen (both eval=G3F)

C = `qwen_image_2512` on `v2_1170L_opus`, eval-G3F (73.77%, 6044 / 8193 correct)
D = `qwen_image_2512` on `v2_1170L_preexp015ft8b`, eval-G3F (70.85%, 5805 / 8193 correct)
Delta: *−2.92pt, −239 correct*. Both are upsamplers — this isolates which one writes better prompts for `qwen_image_2512`.

### Big-class breakdown

| Big Class         | C_acc  | D_acc  | Δ pt   | Lost |
|-------------------|-------:|-------:|-------:|-----:|
| Compound          | 75.12% | 70.29% | −4.83  |   20 |
| Entity Layout     | 70.09% | 65.41% | −4.68  |   52 |
| Action            | 65.77% | 61.31% | −4.46  |   68 |
| World Knowledge   | 81.65% | 77.98% | −3.67  |   12 |
| Style             | 82.26% | 79.03% | −3.23  |   20 |
| Logical Reasoning | 57.42% | 54.84% | −2.58  |    4 |
| Text Generation   | 21.15% | 19.23% | −1.92  |    4 |
| Relationship      | 67.46% | 65.58% | −1.88  |   19 |
| Attribute         | 84.05% | 82.58% | −1.46  |   38 |
| Grammar           | 78.51% | 77.63% | −0.88  |    2 |

D loses every single big class — there is no category where preexp015ft8b beats Opus.

### Standout sub-class drops (D worse than C)

| Sub Class                                          | C_acc  | D_acc  | Δ pt    | Lost | Total |
|----------------------------------------------------|-------:|-------:|--------:|-----:|------:|
| **Action - Animal**                                | 78.65% | 59.55% | **−19.10** |   17 |    89 |
| Action - Full-body (Character/Anthropomorphic)     | 55.35% | 50.00% | −5.35   |   17 |   318 |
| Action - State                                     | 72.45% | 67.55% | −4.91   |   26 |   530 |
| Entity Layout - Two-Dimensional Space              | 68.35% | 63.65% | −4.70   |   27 |   575 |
| Entity Layout - Three-Dimensional Space            | 71.96% | 67.29% | −4.67   |   25 |   535 |
| Style                                              | 82.26% | 79.03% | −3.23   |   20 |   620 |
| Attribute - Material                               | 87.86% | 85.10% | −2.76   |   20 |   725 |
| Attribute - Expression                             | 85.71% | 78.10% | −7.62   |   16 |   210 |

### Where D matched or beat C (small wins, all <2pt)

- Attribute - Size: 81.07% → 82.72% (+1.65pt, +4 correct)
- Grammar - Negation: 71.72% → 73.74% (+2.02pt, +2 correct)
- Grammar - Pronoun Reference: 92.06% → 93.65% (+1.59pt, +1 correct)
- Attribute - Shape: 81.08% → 81.47% (+0.39pt, +1 correct)
- Relationship - Similarity: 67.44% → 68.60% (+1.16pt, +1 correct)

### Reading

- **Action — esp. Animal** is the standout (−19.1pt in a 89-prompt bin). preexp015ft8b's prompts particularly hurt animal-action rendering. Worth eyeballing prompt diffs there.
- **Entity Layout (2D + 3D)** drops ~5pt each — preexp015ft8b's spatial wording is consistently a bit harder to obey than Opus's.
- **Text Generation already collapsed in both** (~20%). Both upsamplers ruin literal-string rendering equally; preexp015ft8b is barely worse here.
- **Grammar is essentially tied** — short prompts don't get much expansion from either upsampler.

So preexp015ft8b is uniformly slightly weaker than Opus for this generator, with `Action - Animal` as the one outlier worth root-causing.

## Source result files

- A: `gcs:nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/unigenbench/v2_1170L_G3F/xingqianx_qwen_image_2512/unigenbench_result_gemini-3p1-pro.json`
- C: `gcs:nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/unigenbench/v2_1170L_opus/qwen_image_2512/unigenbench_result_using_eval_prompt_v2_1170L.json`
- D: `gcs:nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/unigenbench/v2_1170L_preexp015ft8b/qwen_image_2512/unigenbench_result_using_eval_prompt_v2_1170L.json`
