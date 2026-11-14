# Q-Judger image filter: design

Status: proposal, 2026-11-13. Nothing in this document is implemented or measured on real images yet.
Human-readable twins: `qjudge_image_filter_design.html` (zh), `qjudge_image_filter_design_en.html` (en). Keep all three in sync when editing.

## Goal

Reuse the Q-Judger reward model (the QImageBench judge) as a quality filter for real training images. Each image gets per-criterion grades; a downstream rule decides keep or drop.

The new filter is meant to replace the existing production image filter (`qwen3vl_fft_8b_v1`). That filter drops collages, images with dominant text and white-background images as whole categories. The new one keeps those categories and uses scores to pick out the good images in them.

## Fixed decisions

These were decided by the user. Do not revisit them without asking.

1. **The judge input is not modified.** System prompt, user template, checklists and sampling parameters stay exactly as in `qimagebench_protocol.py`. Criteria we do not want are skipped when aggregating, never removed from the checklist or reworded.
2. **The prompt slot is filled with the image caption.** The template requires a prompt, and a placeholder would make the judge grade "matches the prompt" criteria against nothing.
3. **Three L1 dimensions are called for every image: `Quality`, `Aesthetics` and `Creative Generation`.** Three calls are acceptable. The principle is to collect enough information and leave out only what does not apply at all. `Alignment` and `Real-world Fidelity` are not called.
4. **No server change.** The existing scorer already accepts a subset of dimensions and already returns L3 grades. All filter logic lives on the client.
5. **Validate before building infrastructure.** First score the 100k test table through the existing Lepton service. A SILA pipeline comes only after the signal is shown to be useful.

6. **Replace the existing filter; do not drop images by category.** Collages, images with dominant text and white-background images are scored like any other image, and the high-scoring ones are kept.

## Relation to the existing filter

Existing filter: `imaginaire4_alt/pipelines/sila/image/filtering/image_filter_pipeline.py`, version `qwen3vl_fft_8b_v1` (fine-tuned Qwen3-VL 8B), constants in `filtering/constant.py`. Its default rule (`DEFAULT_QUALITY_FILTER`) keeps a row only if `aesthetic_score >= 4` and all five flags are false. Columns are prefixed `filtering_qwen3vl_fft_8b_v1_`.

| Existing column | Used today | New filter |
|---|---|---|
| `aesthetic_score` (0-10) | keep if >= 4 | replaced by Quality and Aesthetics L3 grades |
| `text_banner` | drop if true | kept; graded by Text Accuracy, Text Layout, Graphic Design (Creative Generation) |
| `white_background` | drop if true | kept; graded by Product Design (Creative Generation) plus Quality and Aesthetics |
| `collage` | drop if true | kept; no collage criterion in Q-Judger, only general ones such as Composition. Gap |
| `watermark` | drop if true | no matching criterion in Q-Judger. Undecided |
| `nsfw` | drop if true | Safety & Compliance exists but is in Real-world Fidelity (not called). Undecided |

Consequence: Creative Generation cannot simply be excluded. The criteria that say whether a poster or product shot is well made are inside it.

Whether the 100k test table carries these six columns has not been checked.

## Existing pieces to reuse

| Piece | Location |
|---|---|
| Judge protocol (checklists, template, parsing) | `imaginaire4_alt/projects/cosmos3/cosmos3/reward_service/image/qimagebench_protocol.py` |
| Scorer (vLLM, batch) | `imaginaire4_alt/projects/cosmos3/cosmos3/reward_service/image/qimagebench.py` |
| Benchmark scoring client (retry, timeout, concurrency) | `imaginaire4_alt/projects/cosmos3/cosmos3/evaluation/text_to_image/compute_qimagebench_metric.py` |
| Lepton endpoint | `scorer-qimagebench` (model `Qwen/Qwen-Image-Bench`, Qwen3.6-27B, vLLM) |
| Test table (100k rows) | `s3://nv-00-10206-vfm/lancedb/image/regular/Aesthetic-Train-V2_slice_from_maintable_20261009.lance` |

Request shape: system content = comma-separated L1 names, user content = prompt text + one image.
Response content is JSON with `status`, `score`, `level1`, `level2`, `level3`, `parse_failures`, `judge_outputs`.
`level3` is `{L1: {L2: {L3: grade}}}` with grade in `{0.0, 60.0, 100.0, null}` (Fail, Pass, Excel, N/A).

Judge settings (unchanged): thinking on, temperature 0, top_k 1, seed 42, repetition_penalty 1.05, max_tokens 4096.
Image preprocessing (unchanged): images with area at most 1024x1024 are sent as is; larger ones are scaled down to that area, keeping the aspect ratio.

## Criteria tree

56 L3 criteria in total. `count` = used by the filter. `discard` = computed by the judge because its L1 is called; the result is thrown away, not stored. `not called` = its L1 is never requested.

```
Quality                       called, 6 criteria
  Realism
    Physical Logic            count
    Material Texture          count
  Detail
    Noise                     count
    Edge Clarity              count
    Naturalness               count
  Resolution
    Resolution                count

Aesthetics                    called, 6 criteria
  Composition
    Composition               count
  Color Harmony
    Color Harmony             count
  Lighting
    Lighting & Atmosphere     count    (ACCURATE PROMPT REQUIRED: asks whether lighting matches the prompt's scene)
  Anatomical Portraiture
    Anatomical Fidelity       count    (N/A without people; asks for pores and fine lines)
  Emotional Expression
    Emotional Expression      discard  (prompt-dependent)
  Style Control
    Style Control             discard  (prompt-dependent)

Alignment                     not called, 16 criteria
  Attributes
    Quantity                  prompt-dependent
    Facial Expression         prompt-dependent
    Material Properties       prompt-dependent
    Color                     prompt-dependent
    Shape                     prompt-dependent
    Size                      prompt-dependent
  Actions
    Contact Interaction       prompt-dependent
    Non-contact Interaction   prompt-dependent
    Full-body Action          prompt-dependent
  Layout
    2D Space                  prompt-dependent
    3D Space                  prompt-dependent
  Relations
    Composition Relationship  prompt-independent wording
    Difference/Similarity     prompt-dependent
    Containment               prompt-independent wording
  Scene
    Real-world Scene          prompt-dependent
    Virtual Scene             prompt-independent wording

Real-world Fidelity           not called, 8 criteria
  Fairness
    Social Bias               judges generator bias; harmful as a filter
    Cultural Fairness         judges generator bias; harmful as a filter
  Safety & Compliance
    Safety & Compliance       valid, not worth a separate call
  World Knowledge
    Animals                   expected constant on real photos
    Objects                   expected constant on real photos
    Information Visualization prompt-dependent
    Temporal Characteristics  expected to test the caption
    Cultural Elements         expected to test the caption

Creative Generation           called, 20 criteria
  Imagination
    Imagination               discard  (content type, not quality)
  Feature Matching
    Feature Matching          count    (prompt-independent wording)
  Logical Resolution
    Logical Resolution        discard  (expected constant on real photos)
  Text Rendering
    Text Accuracy             count    (grades images with dominant text)
    Text Layout               count    (grades images with dominant text)
    Font                      discard  (prompt-dependent)
    Cross-lingual Generation  discard  (prompt-dependent)
  Design Applications
    Graphic Design            count    (grades images with dominant text)
    Product Design            count    (grades white-background product shots)
    Spatial Design            count    (prompt-independent wording)
    Fashion Styling           discard  (prompt-dependent)
    Game Design               count    (prompt-independent wording)
    Art Design                discard  (prompt-dependent)
  Visual Storytelling
    Cinematic Style           discard  (prompt-dependent)
    Camera / Lens Style       discard  (prompt-dependent)
    Storyboard Creation       discard  (prompt-dependent)
    Shot Sizes                discard  (prompt-dependent)
    Composition               discard  (prompt-dependent)
    Angles                    discard  (prompt-dependent)
    Comic Creation            discard  (prompt-dependent)
```

Totals: 17 count, 15 discard, 24 not called. The 7 counted Creative Generation criteria are all worded without the prompt, so none of them carries the accurate-prompt requirement.

**Accurate prompt required.** Lighting & Atmosphere is counted, by user decision. It is the only counted criterion whose wording refers to the prompt, so its grade is valid only when the caption passed as the prompt describes the image accurately. Do not use this grade for rows with a missing, very short or unreliable caption.

Note: `Composition` appears twice as an L3 name, under Aesthetics > Composition and under Creative Generation > Visual Storytelling. Always key L3 grades by (L1, L2, L3).

Why each dimension outside Quality and Aesthetics has its status:
- **Alignment**: 13 of 16 criteria ask whether the image matches the prompt. With a caption written from the image this is circular; it measures caption accuracy, not image quality. The other 3 (Composition Relationship, Containment, Virtual Scene) are worded without the prompt but do not justify asking all 16.
- **Real-world Fidelity**: Social Bias and Cultural Fairness judge a generator's bias and would remove legitimate real photos. World Knowledge criteria are expected to be constant on real photos or to test the caption. Safety & Compliance is valid but not worth a full extra call.
- **Creative Generation**: 7 of 20 criteria are prompt-independent (Feature Matching, Text Accuracy, Text Layout, Graphic/Product/Spatial/Game Design). Several of them grade exactly the images the existing filter drops as a class: Text Accuracy, Text Layout and Graphic Design for images with dominant text, Product Design for white-background product shots. Getting them requires asking all 20, the longest and slowest call. Decided by the user: call it for every image and count these 7; the other 13 are computed and discarded.

## Aggregation

Do not reuse the official L3 -> L2 -> L1 mean. Dropping criteria makes L2 weights uneven.

- Store every `count` L3 grade as its own field (0, 1, 2 or null), plus `status` and `parse_failures`. Do not store `discard` criteria.
- Filter rules are written on L3 grades, for example "drop if Noise, Edge Clarity or Resolution is Fail". The exact rule is decided after the 100k run.
- Also store one reference number: the plain mean of the `count` criteria that are not N/A, on the 0/60/100 scale.
- Thresholds are never baked into the scoring step. Changing a rule must not require rescoring.
- A dimension that failed to parse is unknown: store null, never treat it as pass or fail.

## Known measurements (from QImageBench runs on generated images)

| Fact | Value |
|---|---|
| Throughput of one endpoint, client concurrency 16 | about 0.45 s per judge call |
| Cost per image with three dimensions (extrapolated; the Creative Generation call is the longest, so this is a lower bound) | 3 calls, at least 1.4 s |
| 100k images on one endpoint (extrapolated) | at least 38 hours |
| L3 grades that differ when the same image is rescored | 4.4% |
| L3 grades that flip between Fail and Excel, or between N/A and a grade | 1.5% |
| Per-image total score std on rescoring (all five L1) | 3.6 |
| Parse failures | 2 of 1500 images, both in Quality, both images with garbled text |

Parse failures come from the judge reasoning past max_tokens before writing its JSON. Decoding is deterministic, so retrying the same image returns the same truncation.

## Risks

1. **Unverified on real images.** The judge was trained on AI-generated images. Whether the `count` criteria separate good from bad real photos is an assumption from their wording.
2. **Single-image noise.** A filter decides per image, so the rescoring noise above applies directly. Rules should key on clear Fail grades, not on a score threshold.
3. **Anatomical Fidelity bias.** Its wording rewards visible pores and fine lines, which may penalize retouched or studio portraits.
4. **Throughput.** The judge is a 27B thinking model. Scaling comes only from more replicas.
5. **Shared endpoint.** `scorer-qimagebench` also serves benchmark scoring and RL reward. Scoring jobs run one at a time, never concurrently.
6. **No criterion for collages.** The judge was trained on single generated images. Collages can only be graded by general criteria; whether those grades are trustworthy is unknown.
7. **No criterion for watermarks.** Once the existing filter is replaced, no Q-Judger criterion stops a watermarked image.
8. **Lighting & Atmosphere depends on caption accuracy.** See "Accurate prompt required" above.

## Validation plan

1. Inspect the test table: columns, caption column, how image bytes are fetched. Not done yet.
2. Score a small sample (about 500 to 2000 rows) with `Quality,Aesthetics,Creative Generation`. Check per-L3 Fail / Pass / Excel / N/A rates and the parse-failure rate.
3. Review the counted criteria: decide whether Physical Logic and Anatomical Fidelity stay. Spot-check caption accuracy, because Lighting & Atmosphere depends on it.
4. Compare with the existing filter. Check whether the table has its six columns. If it does, focus on the three categories it drops (`collage`, `text_banner`, `white_background`): can Q-Judger grades separate good images from bad ones inside each? Eyeball a batch of each.
5. Check the seven counted Creative Generation criteria: do they separate posters and product shots, and are they mostly N/A on ordinary photos? Measure how much time this call actually adds.
6. If the signal is useful, score the full 100k, then write the filter rule.
7. Only then build the SILA pipeline, modeled on `pipelines/sila/image/filtering/image_filter_pipeline.py` and `pipelines/sila/image/evaluation/hpsv3_pipeline.py`, with its own vLLM endpoint and its own copy of the protocol.

## Rules for agents working on this

- Do not edit `qimagebench_protocol.py` or `qimagebench.py`; benchmark scoring and RL reward depend on them.
- Do not launch Slurm, Lepton or SILA jobs. Hand the command to the user.
- Experimental code goes in `imaginaire4_alt`; code meant for main goes in `imaginaire4_next`.
- State the prompt-set or table size from the manifest; report scoring failures separately.

## Open questions

- Which column holds the caption, and what share of rows have one.
- Whether the captions in the table are accurate enough for Lighting & Atmosphere.
- Whether any of the 15 `discard` criteria (2 in Aesthetics, 13 in Creative Generation) should instead be `count` with the accurate-prompt requirement.
- Watermarks: keep using the existing filter's `watermark` column, or ignore them.
- NSFW: keep using the existing filter's `nsfw` column, or add a Real-world Fidelity call to get Safety & Compliance.
- Whether the grades general criteria give to collages are usable.
- Target production table and its row count, which decides the number of replicas for SILA.
