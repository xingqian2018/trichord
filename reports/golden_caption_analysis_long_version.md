# Captioning quality stress testing via iterative refinement and judging system

**Report covers:** `golden_caption_v2` → `golden_caption_v16_g3pg3p`
**Benchmark:** `CosCapBenchImage/V1` (≈300 images, ≈4.7k–4.8k recall assertions)
**Judge for P/R evaluation:** `gemini-3.1-pro` (unless noted)
**Authored:** 2026-04-24

---

## 1. Motivation

We set out to answer two coupled questions:

1. **How close to the "best legal possible" caption can we push a frontier VLM** if we
   (a) force it to *search* for entities rather than caption in one shot, and
   (b) layer a second VLM on top as a *judge* that must accept every entity / attribute before the pipeline moves on?
2. **Is a VLM / LLM already smart enough to self-improve its own caption** (i.e. read its own
   output, find faults, and patch them) without a human in the loop?

If the answer to (2) is *yes*, a natural next step is a fully agentic refinement system where
the critic/refiner can spawn targeted sub-queries (zoom-in crops, OCR sub-calls, web search,
etc.). The work here (v2–v11) is the non-agentic precursor: a fixed **propose → judge →
refine** loop with a hard cap on rounds, used as a stress test of whether "more cycles / more
judging" converts into measurable caption-quality gains.

---

## 2. Gains and conclusions

- **Literature review:** full write-up lives here — *(link to be added)*.
- **Infra is reliable.** Across 11 versions × ≥3 caption variants × 2 metrics = ~60 full
  eval runs, the pipeline finished every single image (`success_cnt` 294–300 / 300 on every
  version) and produced self-consistent outputs. Judges never silently failed. We therefore
  rule out engineering misdesign as the explanation for the results below.
- **Some interesting details:**
  - There is a clean **precision ↔ recall tradeoff induced by the generator model**, not by
    the loop. Gemini-3.1-pro gen ⇒ very high precision (≈ 0.97) but relatively low recall
    (≈ 0.60). Qwen3-VL-235B gen ⇒ lower precision (≈ 0.87) but higher recall (≈ 0.67).
    Swapping the gen model moves you along that curve; adding more judge rounds does not
    visibly shift you off it.
  - **Stage 4 almost always wins on F1**, even though it only tacks on camera + style to the
    stage‑3 structured caption. Those extra sentences add recall-relevant detail without
    dragging precision down. If there is a free lunch in this pipeline, Stage 4 is it.
  - **The "mix" recipe (v10p1, v11) — use model A for Stage 1 entity search, model B for
    Stage 2 grounding — did reproduce the precision of the model-B-only run, but did not
    unlock higher recall**, even when Stage 1 was done by the higher-recall qwen235b
    (v10p1 imports stage1 from v6). This suggests the recall bottleneck is in *per-attribute
    grounding* (Stage 2), not in entity enumeration (Stage 1).
  - **New YAML-based structure grounding (v7, v8) did not beat the original structure
    grounding (v5, v9)**. v7 (g3f gen, S1 from v6 qwen) and v8 (qwen gen, S1 from v6 qwen)
    both sit at the bottom of the F1 ranking on stage-4 structured caption.
- **[Negative headline] Iterative stress-testing does not beat Seungjun's one-shot
  9-grid-search captioning.** Despite five propose↔judge↔refine rounds in both Stage 1 and
  Stage 2, the best F1 we achieve on Stage‑4 structured caption is **0.797 (v3)** /
  **0.792 (v11)** / **0.791 (v9)**. These are indistinguishable from — and in the same
  ballpark as — Seungjun's single-pass 9-grid system, which is structurally much simpler
  and much cheaper. This is surprising: the judge *does* reject many first-draft entities
  and grounding fields, the refiner *does* patch them, and yet the aggregate numerical
  caption quality barely moves.

---

## 3. Methods

### 3.1 Pipeline overview

Four sequential stages produce one entity-list JSON per image, which is then converted to
two downstream caption strings (dense paragraph and structured paragraph):

```
                  ┌──────────────────────────────────────────────────────────────┐
  raw image ────▶ │  STAGE 1  Entity Search                                       │
                  │    loop (≤5 battle rounds):                                   │
                  │      gen VLM  proposes entities { id, location, reason }      │
                  │      judge VLM verifies each entity against the image,       │
                  │                 flags missing / wrong / duplicate entities    │
                  │      refiner VLM fixes the flagged set; loop again            │
                  │    exit when judge accepts every entity AND reports no miss   │
                  └───────────────────────────┬──────────────────────────────────┘
                                              │ entity list
                                              ▼
                  ┌──────────────────────────────────────────────────────────────┐
  entity list ──▶ │  STAGE 2  Entity Structured Grounding                         │
                  │    per-entity, for 8 structured fields:                       │
                  │      { description, location, relation, pose,                 │
                  │        materials, clothing, facial_and_expression,            │
                  │        text_and_signage }                                     │
                  │    loop (≤5 battle rounds), entities processed in chunks ≤16: │
                  │      gen VLM fills the 8 fields                               │
                  │      judge VLM decomposes each field into atomic assertions   │
                  │                 and emits (correct?, complete?) booleans      │
                  │      refiner VLM re-grounds only entities that failed either  │
                  │    exit when every (entity × field) passes, or 5 rounds done  │
                  └───────────────────────────┬──────────────────────────────────┘
                                              │ structured grounding
                                              ▼
                  ┌──────────────────────────────────────────────────────────────┐
                  │  STAGE 3  Dense Grounding Synthesis  (single-shot LLM)        │
                  │    LLM reads the 8 structured fields per entity and writes    │
                  │      • grounding_dense         (full paragraph)               │
                  │      • grounding_dense_downsampled  (15–100-word caption)     │
                  │    no judge; ≤3 retries only if output parse fails            │
                  └───────────────────────────┬──────────────────────────────────┘
                                              │ per-entity dense text
                                              ▼
                  ┌──────────────────────────────────────────────────────────────┐
  raw image ────▶ │  STAGE 4  Camera + Style  (single-shot VLM)                   │
                  │    VLM reads image + stage 3 json, emits                      │
                  │      { overall_style, camera_details }                        │
                  │    no judge; ≤3 retries only if output parse fails            │
                  └───────────────────────────┬──────────────────────────────────┘
                                              │
                                              ▼
                  ┌──────────────────────────────────────────────────────────────┐
                  │  CONVERSION  (entity list  →  final captions)                 │
                  │    stage3_dense_caption       (per-entity dense, concatenated)│
                  │    stage3_structured_caption  (per-entity 8 fields, flat)     │
                  │    stage4_structured_caption  (stage3_structured + camera +   │
                  │                                style appended)                │
                  └──────────────────────────────────────────────────────────────┘
```

Key design choices:

- **Stages 1 and 2 are the only ones with the iterative judging loop.** Stages 3 and 4 are
  single-shot by construction (the first is a pure text rewrite, the second adds global
  camera / style that the judge would have no pixel-level evidence to reject).
- **Up to 5 battle rounds per stage, per image** (Stage 1) or **per chunk of ≤16 entities**
  (Stage 2). Empirically most images exit well before 5 rounds.
- **Judge and generator are separate VLM calls**, usually different model families, so the
  judge is not "judging itself".
- **Stage-2 refinement is field-surgical**: only failed (entity × field) pairs are
  regenerated; anything the judge accepted is frozen.

### 3.2 Model assignment per version

| Version | Gen VLM (S1,S2,S4) | Gen LLM (S3) | Stage 1 source | Notes |
|---|---|---|---|---|
| v2 | mixed | same as VLM | own S1 | First looping refinement system. Mixed judge + gen. |
| v3 | mixed | same as VLM | own S1 | Mostly identical to v2, gen mix varied. |
| v4 | gemini-3.1-pro | same as VLM | own S1 | All-gemini-3.1-pro. |
| v5 | gemini-3-flash | same as VLM | own S1 | All-gemini-3-flash. |
| v6 | qwen3-vl-235b | same as VLM | own S1 | All-qwen235b. |
| v7 | gemini-3-flash | same as VLM | **from v6** (qwen235) | "New structure grounding", S2–S4 only. |
| v8 | qwen3-vl-235b-a22b-instruct (S2, S4) | qwen3-235b-a22b-instruct (S3) | **from v6** (qwen) | "New structure grounding", S2–S4 only. All-qwen path. |
| v9 | gemini-3-flash (S2); gemini-3.1-pro (S4, + S3 final batch) | gemini-3-flash → gemini-3.1-pro tail (S3) | **from v5** (g3f) | S2–S4 only; legacy structure grounding. S3's last 144-image batch and all of S4 were rerun on gemini-3.1-pro, so v9's downstream is a g3f→g3p hybrid. |
| v10p1 | gemini-3.1-pro | same as VLM | **from v6** (qwen235) | S2–S4 only, "mix" recipe. |
| v11 | qwen3-vl-235b-a22b-instruct | qwen3-235b-a22b-instruct | **from v5** (g3f) | S2–S4 only, "mix" recipe. Stage 3 uses a dedicated LLM. |
| v12 | gemini-3-flash | same as VLM | own S1 | `g3fg3p`; full new Stage 1 + Stage 2 prompting design with YAML I/O. |
| v13 | qwen3-vl-235b-a22b-instruct | qwen3-235b-a22b-instruct | own S1 | `q235bg3p`; all-qwen gen with gemini-3.1-pro judge. |
| v14 | qwen3-vl-235b-a22b-instruct | qwen3-235b-a22b-instruct | **from v13** (q235b) | S2–S4 only; fixes the ignore issue on identifier location. |
| v15 | qwen3-vl-235b-a22b-instruct | qwen3-235b-a22b-instruct | own S1 | `q235bg3p`; fully agent self-improved prompting system. |
| v16 | gemini-3.1-pro | same as VLM | own S1 | `g3pg3p`; v15-style agentic prompting system swapped onto gemini-3.1-pro gen + judge. |

Judge across all stages of all versions is `gemini-3.1-pro` (the v2 "mixed" case is the only
exception). Evaluation judge (precision / recall scoring) is also `gemini-3.1-pro`.

### 3.3 Evaluation

- **Precision** (accuracy over extracted claims): each sentence of the caption is
  decomposed by `gemini-3.1-pro` into atomic claims and judged correct / incorrect /
  uncertain against the image. `precision = correct / (correct + incorrect + uncertain)`
  aggregated over all claims in all images.
- **Recall**: a fixed set of per-image "must-mention" assertions
  (`V1_assertion/`, 4.6k–4.8k total) is checked against the caption; recall is
  `passed / total`.
- **F1** is the harmonic mean.
- Three caption variants per version are scored:
  * *dense* — `stage3_dense_caption` (or `stage2_dense_caption` in pre-v8 naming)
  * *structured* — `stage3_structured_caption` (or `stage2_…` pre-v8)
  * *s4 structured* — `stage4_structured_caption` (dense structured + camera + style)

---

## 4. Results

Three tables, one per caption variant. P = precision, R = recall, F1 = 2·P·R / (P+R).

### 4.1 Dense caption (Stage 3 dense)

| Version | P | R | F1 |
|---|---:|---:|---:|
| v2            | 0.933 | 0.638 | 0.758 |
| v3            | 0.935 |  n/a* |  n/a* |
| v4 (g3pg3p)   | **0.976** | 0.601 | 0.744 |
| v5 (g3fg3p)   | 0.971 | 0.628 | 0.763 |
| v6 (q235g3p)  | 0.866 | **0.663** | 0.751 |
| v7 (q235g3p)  | 0.875 | 0.659 | 0.752 |
| v8 (q235g3p)  | 0.905 | 0.616 | 0.734 |
| v9 (g3fg3p)   | 0.971 | 0.623 | 0.759 |
| v10p1 (mixg3p)| 0.969 | 0.625 | 0.760 |
| v11 (mixg3p)  | 0.946 | 0.643 | **0.766** |
| v12 (g3fg3p)  | 0.961 | 0.612 | 0.747 |
| v13 (q235bg3p)| 0.939 | 0.613 | 0.742 |
| v14 (q235bg3p)| 0.942 | 0.607 | 0.738 |
| v15 (q235bg3p)| 0.941 | 0.603 | 0.735 |
| v16 (g3pg3p)  | 0.975 | 0.622 | 0.760 |

\* v3 dense/structured recall never finished (only v3 s4s recall was run); numbers for the
v3 paragraph are backed by the s4 row below.

v13 image-success counts (every variant): precision 298 / 298 images, recall 292 / 292 images.

### 4.2 Structured caption (Stage 3 structured)

| Version | P | R | F1 |
|---|---:|---:|---:|
| v2            | 0.935 | 0.640 | 0.760 |
| v3            | 0.938 |  n/a* |  n/a* |
| v4 (g3pg3p)   | **0.975** | 0.592 | 0.737 |
| v5 (g3fg3p)   | 0.971 | 0.621 | 0.757 |
| v6 (q235g3p)  | 0.860 | 0.648 | 0.739 |
| v7 (q235g3p)  | 0.867 | 0.657 | 0.748 |
| v8 (q235g3p)  | 0.895 | 0.604 | 0.722 |
| v9 (g3fg3p)   | 0.973 | 0.613 | 0.752 |
| v10p1 (mixg3p)| 0.970 | 0.618 | 0.755 |
| v11 (mixg3p)  | 0.938 | 0.633 | **0.756** |
| v12 (g3fg3p)  | 0.957 | 0.600 | 0.737 |
| v13 (q235bg3p)| 0.938 | 0.596 | 0.729 |
| v14 (q235bg3p)| 0.938 | 0.580 | 0.717 |
| v15 (q235bg3p)| 0.941 | 0.589 | 0.724 |
| v16 (g3pg3p)  | 0.973 | 0.604 | 0.745 |

### 4.3 Stage-4 structured caption (Stage 3 structured + camera + style)

| Version | P | R | F1 |
|---|---:|---:|---:|
| v2            | 0.939 | 0.671 | 0.783 |
| v3            | 0.939 | **0.693** | **0.797** |
| v4 (g3pg3p)   | **0.974** | 0.653 | 0.782 |
| v5 (g3fg3p)   | 0.969 | 0.669 | 0.791 |
| v6 (q235g3p)  | 0.872 | 0.687 | 0.769 |
| v7 (q235g3p)  | 0.876 | 0.688 | 0.770 |
| v8 (q235g3p)  | 0.900 | 0.651 | 0.756 |
| v9 (g3fg3p)   | 0.972 | 0.650 | 0.779 |
| v10p1 (mixg3p)| 0.971 | 0.656 | 0.783 |
| v11 (mixg3p)  | 0.935 | 0.668 | 0.779 |
| v12 (g3fg3p)  | 0.958 | 0.653 | 0.777 |
| v13 (q235bg3p)| 0.940 | 0.643 | 0.764 |
| v14 (q235bg3p)| 0.936 | 0.637 | 0.758 |
| v15 (q235bg3p)| 0.934 | 0.637 | 0.758 |
| v16 (g3pg3p)  | 0.974 | 0.648 | 0.778 |

---

## 5. Analysis

### 5.1 All F1s collapse into a narrow band

Across **every** version and **every** caption variant, F1 ∈ **[0.722, 0.797]**. The total
spread is ~7 F1 points. That is not a small band in absolute terms, but it is a surprisingly
small band given the axes we varied:

- 3 different generator model families (Gemini-3.1-pro, Gemini-3-flash, Qwen3-VL-235B).
- Pure runs vs. "mix" runs where Stage 1 comes from a different model.
- Old vs. new structure-grounding prompt design.
- Dense vs. structured vs. structured+camera+style outputs.

Under all of these changes, the pipeline's overall grade stays roughly "A-minus on
precision, C-plus on recall". This is consistent with the **recall bottleneck being
perceptual rather than generative**: the judge cannot invent ground-truth assertions it
never extracted from the image, and the refiner cannot plug holes the judge did not flag.
No amount of looping fixes that.

### 5.2 Precision is gen-model-dominated; the judge loop mostly "polices a ceiling"

Versions fall into three clear precision tiers matched one-to-one with their generator:

| Gen model | Precision range (s4s) | Representative versions |
|---|---|---|
| Gemini-3.1-pro | 0.97 | v4, v10p1 |
| Gemini-3-flash | 0.97 | v5, v9† |
| Qwen3-VL-235B | 0.87–0.94 | v6, v7, v8, v11 |

† v9's high precision is partly a gemini-3.1-pro-tail effect (S3 final batch + S4 reran on
g3p); a pure-g3f run would land slightly lower.

The iterative judge prevents gross hallucinations (no version goes below 0.86 on s4
precision even with the weakest gen), but it does not *raise* precision above what a strong
gen would give on its own. In other words: **the loop's contribution to precision looks
like a floor, not a ceiling.**

### 5.3 Recall does not respond to extra rounds either

If the judge genuinely surfaced missing entities and the refiner genuinely added them, we
would expect recall to grow monotonically with version (as prompts/loops matured). It does
not:

- Best Stage‑4 recall in the whole sweep is **v3 = 0.693** — our *second* version.
- v2 stage‑4 recall (0.671) already matches the v11 number (0.668) to within 0.003.
- The highest-recall gen model (qwen235, v6–v7) bought +0.02 recall over v2 *but lost
  0.07 precision doing so*, a net negative on F1.

### 5.4 Stage 4 is the only free lunch

For every single version, `stage4_structured_caption` beats both `dense` and `structured`
on F1, usually by 1–3 points. The mechanism is uncontroversial: Stage 4 appends globally
scoped facts (camera, lighting, overall style) that unlock recall-assertion hits without
spawning many new claims for the precision judge to shoot down. The lesson is generic:
**adding orthogonal, low-risk content is the highest-ROI knob in this pipeline.**

### 5.5 "Mix" recipes are re-dressings, not accelerators

v7, v10p1, v11 all reuse a Stage-1 entity list from an earlier model and then run
Stages 2–4 with a *different* model. This is a cheap ablation — if the Stage‑1 entity set
were the recall bottleneck, a good Stage‑1 + a good Stage‑2 should compound. (v8 also
reuses an earlier Stage 1 — from v6 — but keeps qwen for Stage 2–4, so it is effectively
v6 + new structured-grounding prompts, not a true "mix" recipe.)

They don't compound. v10p1 (qwen235 S1 + gemini-3.1-pro S2+) lands at 0.783 F1, identical
to v4 (pure gemini-3.1-pro) at 0.782. v11 (gemini-3-flash S1 + qwen3-VL-235B S2+) lands
at 0.779, below v5 at 0.791. **Whichever gen does Stage 2 dominates the outcome; Stage‑1
provenance is ~noise.**

### 5.6 Why the iterative stress test doesn't beat Seungjun's one-shot 9-grid system

This is the negative headline. A few hypotheses, roughly in order of how I'd bet:

1. **Judges over-accept.** A judge that says "looks fine" when it has no positive evidence
   of error cannot push the generator past its own ceiling. We should sanity-check the
   judge's false-negative rate (caption claims the judge *didn't* flag but a stricter
   reviewer would).
2. **Entity-oriented decomposition loses context.** Splitting the scene into per-entity
   grounding makes each call local; the 9-grid baseline forces the model to look at every
   tile explicitly. Local recall per entity is high, but global "did you actually survey
   the whole image" coverage is not better than a geometric grid.
3. **Refinement is anchored by the first draft.** Once the generator commits to an entity
   list, the refiner mostly edits in place; the judge does not aggressively demand
   restart-from-scratch proposals. 9-grid gets 9 independent first drafts, which may be a
   stronger form of diversity than 5 dependent refinements.
4. **Stage 2's 8-field schema may itself constrain recall.** Anything not in one of the 8
   fields (interactions between entities, scene-level events, temporal cues) has no slot to
   live in, so the judge cannot demand it. This would also explain why Stage 4 (which adds
   a new slot) is the only consistent F1 lifter.

### 5.7 Recommendations for the next iteration

- **Measure the judge.** Evaluate the judge's precision and recall directly on a human-
  labeled subset. If it's too permissive, every downstream round is capped.
- **Break the single-draft anchor.** Make Stage 1 produce *k* independent proposals (N=3
  with different temperatures / prompts) and let the judge *choose* + merge, rather than
  edit.
- **Add orthogonal slots.** Stage 4 showed how much free F1 is sitting in "content the
  entity loop never tried to cover". Explicitly schedule slots for scene actions,
  inter-entity relations, text in the image, and background ambience.
- **Open the door to agentic refinement.** The cheapest move is to let Stage 2's refiner
  issue a targeted **crop + re-ask** on any entity whose grounding the judge marked
  incomplete. That is one tool, one extra VLM call per flagged entity, and directly
  attacks the recall bottleneck.

---

*All raw numbers in this report were extracted from*
`s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION>/results/`
*(files `precision_eval_result_*.json` and `recall_eval_result_*.json`).*
