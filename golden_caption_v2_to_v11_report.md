# Captioning Quality Stress Testing via Iterative Refinement and Judging System

**Author:** Xingqian (xingqianx@nvidia.com)
**Date:** 2026-04-24
**Scope:** `golden_caption_v2` → `golden_caption_v11` on CosCapBenchImage V1 (~299 images)

---

## Motivation

We want to know two things:

1. **What is the best "legal" caption we can squeeze out of today's VLM/LLMs?**
   "Legal" here means: no hand-written prompts that are overfitted to the test set, no per-image supervision — only a generator VLM plus a judge VLM in a refinement loop.
2. **Is a frontier VLM/LLM actually smart enough to improve its own captions?**
   If yes, this opens the door to a larger agentic caption-refinement pipeline (propose → critique → fix → accept) as a drop-in replacement for one-shot captioners. If no, the failure mode is itself informative.

The v2→v11 sweep is a stress test along two axes: (a) the generator/judge model mix, and (b) the structural design of the pipeline (full 4-stage vs. stage-1-frozen / stage-2-onwards-only).

---

## Gains and Conclusions

- **Literature review.** Full review is in a companion doc: *(link TBD)*.
- **Infrastructure sanity.** Before drawing any conclusions about the *algorithm*, we wanted to rule out "it's just a bad pipeline". The stage1→4 harness is now reliable:
  - All eleven versions complete end-to-end on the same 299-image benchmark.
  - All three downstream caption variants (stage-3 dense, stage-3 structured, stage-4 structured-with-camera/style) are produced for every version.
  - Precision and recall are evaluated by the same `gemini-3.1-pro` judge across all versions, so cross-version numbers are directly comparable.
  - Run-settings are snapshotted as JSON next to each stage's output (see the `stage{N}_run_setting_*.json` files under each version folder on S3). Reproducibility is not the blocker.
- **Interesting details observed along the way.**
  - **Generator choice dominates precision.** Swapping in `gemini-3.1-pro` as generator (v4) jumps precision from ~0.87 (qwen-235b, v6) to ~0.97 — a ~10 pp absolute gap that has nothing to do with the judging loop.
  - **But generator choice trades precision for recall.** The same v4 that wins precision is the *worst* on recall. The pipeline can be tuned to minimize hallucination or maximize coverage, but not both with the same model.
  - **Judge-side recursion doesn't unlock much.** Allowing up to 5 battle rounds of judge-critique does move numbers a little, but most gain comes from picking the right generator, not from iterating.
  - **Stage 4 is consistently the best variant.** Across all eleven versions, the `stage4_structured_caption` F1 beats both stage-3 dense and stage-3 structured by 1–4 pp. The camera + style supplement adds real recall.
  - **Frozen stage-1 experiments (v7–v11) don't rescue weak generators.** Re-running stages 2–4 with a stronger model on top of a weaker stage-1 entity list caps the achievable quality; the entity list set in stage 1 bounds what everything downstream can say.
- **[Negative] Iterative refinement doesn't clearly beat Seungjun's one-shot 9-grid captioner.**
  This is the headline surprise. After eleven versions of iterative propose→judge→refine (up to 5 rounds, with the frontier `gemini-3.1-pro` as judge), the best F1 we observed was **0.7973 (v3, stage 4 structured)**. That is within noise of — and in some setups *worse than* — Seungjun's one-shot 9-grid-search captioner trained on the same benchmark. If the iteration loop were exploiting real self-improvement signal, we would expect a clear gap in its favor; we do not see one. Two possible reads:
  1. Current frontier VLMs are already very close to the ceiling a single forward pass can reach on this benchmark; the judge does not see mistakes the generator can't already see.
  2. The loop is over-correcting in one direction (precision) at the cost of the other (recall), and F1 stays roughly constant.

---

## Methods

### Pipeline diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             INPUT: raw image                                │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1 — Entity Search        (iterative, up to 5 battle rounds)           │
│                                                                             │
│   Gen VLM      ──── propose entities [(identifier, location, reason), ...]  │
│      │                                                                      │
│      ▼                                                                      │
│   Judge VLM    ──── per-entity verdict (identifier OK? location OK?)        │
│                ──── flag missing entities                                   │
│      │                                                                      │
│      ▼                                                                      │
│   Gen VLM      ──── REFINE: fix flagged, drop hallucinations, add missing   │
│                                                                             │
│   stop when (all-correct ∧ nothing-missing)  OR  5 rounds exhausted         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │  entity list (ID, rough location)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2 — Entity Structured Grounding   (iterative, up to 5 battle rounds)  │
│                                                                             │
│   For each entity, fill 8 structured sub-fields:                            │
│     description · location · relation · pose · materials ·                  │
│     clothing · facial_and_expression · text_and_signage                     │
│                                                                             │
│   Gen VLM      ──── fill all 8 fields                                       │
│   Judge VLM    ──── per-field booleans:  correct?  complete?                │
│   Gen VLM      ──── re-ground ONLY flagged entities                         │
│                                                                             │
│   stop when (every field correct ∧ complete)  OR  5 rounds                  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │  per-entity structured JSON
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3 — Dense Grounding Synthesis   (single-pass, no judge)               │
│                                                                             │
│   LLM          ──── read 8 structured sub-fields                            │
│                ──── emit (a) grounding_dense   — full paragraph             │
│                         (b) grounding_dense_downsampled   — 15–100 words    │
│                                                                             │
│   retry up to 3× on parse failure                                           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │  dense + structured captions
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4 — Camera & Style Analysis    (single-pass, no judge)                │
│                                                                             │
│   Gen VLM      ──── read image + stage-3 JSON                               │
│                ──── add overall_style and camera_details                    │
│                                                                             │
│   retry up to 3× on parse failure                                           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
                            FINAL:  stage3_dense_caption
                                    stage3_structured_caption
                                    stage4_structured_caption  ← strongest
```

### What changes between versions (summary)

| Version | Judge (all stages) | Gen VLM (s1, s2, s4) | Gen LLM (s3) | Notes |
|---|---|---|---|---|
| v2 | mixed | mixed | same as VLM | First iterative loop |
| v3 | gemini-3.1-pro | mixed | same | |
| v4 | gemini-3.1-pro | gemini-3.1-pro | same | Strongest model on both sides |
| v5 | gemini-3.1-pro | gemini-3.1-flash | same | |
| v6 | gemini-3.1-pro | qwen3-vl-235b | same | |
| v7 | gemini-3.1-pro | gemini-3-flash | same | Stage 2-to-4 only; frozen s1 from v6 (qwen) |
| v8 | gemini-3.1-pro | gemini-3-flash | same | Stage 2-to-4 only; frozen s1 from v5 (g3f) |
| v9 | gemini-3.1-pro | gemini-3-flash | same | Stage 2-to-4 only; frozen s1 from v5 (g3f) |
| v10p1 | gemini-3.1-pro | gemini-3.1-pro | same | Stage 2-to-4 only; frozen s1 from v6 (qwen) |
| v11 | gemini-3.1-pro | qwen3-vl-235b-instruct | qwen3-235b-a22b-instruct | Stage 2-to-4 only; frozen s1 from v5 (g3f) |

*(v10 was a partial stage-2-only run and is omitted from results.)*

---

## Results

Numbers are measured with `gemini-3.1-pro` as judge on CosCapBenchImage V1. Each row is the three downstream caption variants of one version. **Precision = fraction of model claims judged correct** (uncertain claims counted as incorrect). **Recall = fraction of benchmark assertions the caption satisfies.** F1 is their harmonic mean.

| Version | Variant | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| v2  | stage-3 dense        | 0.9331 | 0.6377 | 0.7576 |
| v2  | stage-3 structured   | 0.9353 | 0.6402 | 0.7601 |
| v2  | **stage-4 structured** | 0.9393 | 0.6706 | **0.7825** |
| v3  | stage-3 dense        | 0.9350 | 0.6609 | 0.7744 |
| v3  | stage-3 structured   | 0.9380 | 0.6583 | 0.7737 |
| v3  | **stage-4 structured** | 0.9390 | 0.6927 | **0.7973** ← best overall |
| v4  | stage-3 dense        | 0.9762 | 0.6008 | 0.7439 |
| v4  | stage-3 structured   | 0.9752 | 0.5925 | 0.7371 |
| v4  | stage-4 structured   | 0.9742 | 0.6531 | 0.7820 |
| v5  | stage-3 dense        | 0.9708 | 0.6281 | 0.7627 |
| v5  | stage-3 structured   | 0.9706 | 0.6211 | 0.7575 |
| v5  | stage-4 structured   | 0.9689 | 0.6688 | 0.7914 |
| v6  | stage-3 dense        | 0.8656 | 0.6629 | 0.7508 |
| v6  | stage-3 structured   | 0.8598 | 0.6485 | 0.7393 |
| v6  | stage-4 structured   | 0.8719 | 0.6875 | 0.7688 |
| v7  | stage-3 dense        | 0.8753 | 0.6593 | 0.7521 |
| v7  | stage-3 structured   | 0.8670 | 0.6574 | 0.7478 |
| v7  | stage-4 structured   | 0.8758 | 0.6877 | 0.7705 |
| v8  | stage-3 dense        | 0.9055 | 0.6165 | 0.7335 |
| v8  | stage-3 structured   | 0.8948 | 0.6045 | 0.7215 |
| v8  | stage-4 structured   | 0.9003 | 0.6514 | 0.7559 |
| v9  | stage-3 dense        | 0.9709 | 0.6226 | 0.7587 |
| v9  | stage-3 structured   | 0.9733 | 0.6130 | 0.7522 |
| v9  | stage-4 structured   | 0.9724 | 0.6500 | 0.7792 |
| v10p1 | stage-3 dense      | 0.9690 | 0.6249 | 0.7598 |
| v10p1 | stage-3 structured | 0.9696 | 0.6180 | 0.7549 |
| v10p1 | stage-4 structured | 0.9709 | 0.6563 | 0.7832 |
| v11 | stage-3 dense        | 0.9459 | 0.6431 | 0.7657 |
| v11 | stage-3 structured   | 0.9377 | 0.6329 | 0.7557 |
| v11 | stage-4 structured   | 0.9349 | 0.6680 | 0.7792 |

Best F1 per version (stage-4 structured is always the winning variant):

| Version | Best F1 | Precision | Recall |
|---|---:|---:|---:|
| v2    | 0.7825 | 0.9393 | 0.6706 |
| **v3** | **0.7973** | 0.9390 | 0.6927 |
| v4    | 0.7820 | 0.9742 | 0.6531 |
| v5    | 0.7914 | 0.9689 | 0.6688 |
| v6    | 0.7688 | 0.8719 | 0.6875 |
| v7    | 0.7705 | 0.8758 | 0.6877 |
| v8    | 0.7559 | 0.9003 | 0.6514 |
| v9    | 0.7792 | 0.9724 | 0.6500 |
| v10p1 | 0.7832 | 0.9709 | 0.6563 |
| v11   | 0.7792 | 0.9349 | 0.6680 |

---

## Analysis

**1. The spread in F1 across 10 versions is only ~4 pp (0.7559 – 0.7973).**
That is a very tight band for a configuration sweep that changes judge, generator, and pipeline shape. The first thing this says: most of our levers only move the P/R trade-off, not the frontier.

**2. Generators cluster into a "high-precision / lower-recall" band and a "lower-precision / higher-recall" band.**

- **Precision leaders (≈0.97):** v4, v5, v9, v10p1 — all use gemini-3.1-pro or gemini-3.1-flash or gemini-3-flash as generator. Their recall drops to 0.65–0.67.
- **Recall leaders (≈0.68–0.69):** v3 and v6/v7 (qwen-235b family). Their precision falls to 0.86–0.94.
- v3 sits in the sweet spot: precision 0.939, recall 0.693 — which is why it wins F1. It is *not* the most extreme on either axis.

This pattern is a strong hint that **the judge critique mostly prunes** — it removes claims the judge isn't sure about, which inflates precision and erodes recall. We don't see a version where the loop both adds new correct information *and* keeps precision high.

**3. Stage 4 is uniformly better than stage 3.**
Across every single version, `stage4_structured_caption` F1 beats both stage-3 variants by 1–4 pp. The delta is driven by recall (stage 4 recall is 2–5 pp higher than stage-3 recall in every version) while precision stays roughly flat. The camera/style supplement is cheap (single pass, no judge) and has the best return-on-tokens of anything in the pipeline.

**4. Freezing stage 1 caps everything downstream.**
v7 (frozen stage 1 = qwen) and v9 (frozen stage 1 = g3f), both re-using `gemini-3-flash` from stage 2 on, produce very different F1 (0.7705 vs 0.7792). v8 (same generator, frozen s1 = g3f) is actually the lowest at 0.7559 — we suspect a run-specific artifact. The bigger point: **the entity list from stage 1 is the ceiling**; refining later stages on top of a weak list can only do so much.

**5. The iterative loop is not "for free".**
Each battle round is ~2× the gen-side cost and adds judge cost on top. If v3's 0.7973 is really the ceiling we can reach with 5 rounds, and Seungjun's one-shot 9-grid captioner lands in the same neighborhood, then the loop is paying a multiplicative token bill for a margin that is within noise. That does not mean the loop is worthless — it gives us *detailed judge traces* and *per-entity grounding*, which a one-shot captioner does not — but on the narrow metric of (P, R) on CosCapBenchImage V1, we do not see the improvement we hoped for.

**6. What we would try next.**

- **Break the P/R trade-off.** The judge prompt currently encourages "flag anything you're not sure of." Re-tuning the judge to *prefer keeping a claim unless clearly wrong* should recover recall without trashing precision.
- **Swap stage 1 for a structure-first approach.** The v12+ series (already in flight) is experimenting with a YAML-I/O stage 1/2 redesign — expected to address the "stage-1 ceiling" we see here.
- **Ablate the loop length.** Run v3's configuration with max_battle_rounds ∈ {0, 1, 2, 5} to measure how much of the 0.7973 comes from iteration vs. from the model pair itself. If 0 rounds gets 0.79, iteration is adding nothing. This is the cleanest test of "can the VLM improve itself".
- **Compare apples-to-apples to Seungjun.** Run Seungjun's one-shot 9-grid captioner through the *same* `gemini-3.1-pro` evaluator on the *same* 299 images, and record P/R side-by-side with v3. Right now the "no gain" claim rests on numbers measured through slightly different evaluation setups.
