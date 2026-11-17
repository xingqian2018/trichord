# Q-Judger real-image filter: design

Status (2026-11-16): this is v0, the first version of the filter (output column `filter_real_image_qjudge_v0`). Prompt final, calibrated on 1,000 real photos; the unscored `upright_orientation` item was added to the Quality call on 2026-11-16. The first draft (2026-11-13) was a proposal; its decisions "judge input unchanged", "caption fills the prompt slot" and "no server change" are reversed (see "Changes since the 11-13 draft").

Human-readable twins: `qjudge_image_filter_design.html` (zh) and `qjudge_image_filter_design_en.html` (en). Keep all three in sync when editing.

## Goal

Use Q-Judger (`Qwen/Qwen-Image-Bench`, Qwen3.6-27B, the QImageBench judge) to score real training images. The filter replaces the existing one (`qwen3vl_fft_8b_v1`).
- Target: pick about 1M top images out of 40M. Some false rejections are acceptable.
- Collages, images with dominant text and white-background images are scored like any other image, not dropped as categories.

## Fixed decisions

These were decided by the user. Do not revisit them without asking.

1. **The judge sees only the image.** No caption is passed, even though the caller's request carries a prompt.
   - System prompt: `You are an expert evaluator of the quality of real photographs. Given a real photograph taken by a camera, you evaluate the image on specific quality criteria using a structured checklist.`
   - The user message has three parts, in order:
     - `# Real Photograph\n`;
     - the image;
     - the official `USER_PROMPT_TEMPLATE` text after the image placeholder, formatted with `prompt=""` and the displayed dimension name, with `this image/prompt` replaced by `this image`.
2. **3 judge calls per image, 17 criteria.** Each checklist contains exactly the counted criteria; no other official criterion is asked.
   - One extra, unscored item is asked at the end of the Quality checklist: `upright_orientation` (true = the image is upright, false = stored rotated sideways or upside down). It is a flag, not a criterion: it has no grade and is not part of the 17.
   - Wording, appended under `## Orientation`: `upright_orientation: Is the photograph upright, i.e. not rotated sideways or upside down (e.g., the horizon is horizontal, people and buildings stand upright)? This item is not scored: instead of a score, answer {"value": true} if the image is upright, otherwise {"value": false}.`
3. **The judge and the caller see different names for the third dimension.** The judge sees `Text and Design`; the returned key is `Creative Generation`.
4. **Wording rules.**
   - Use the official wording wherever possible.
   - Changes only remove prompt dependence, remove the "AI-generated" framing, or add "Answer N/A if/otherwise".
   - Noise keeps the official wording.
   - Text Accuracy keeps "misspellings, or typographical errors".
   - Feature Matching stays counted.
   - Never drop a counted criterion without asking the user.
5. **Sampling and preprocessing match the official protocol.**
   - Sampling: temperature 0, top_p 1, top_k 1, repetition_penalty 1.05, max_tokens 4096, seed 42, `enable_thinking=True`.
   - Preprocessing: iso-area resize to at most 1024x1024 with LANCZOS.
6. **The server does everything.** The caller sends one request per image (prompt text plus image) and gets one `{L1: {L2: {L3: score}}}` dict back.
   - The server makes the 3 judge calls, parses, and retries.
   - The caller cannot set the system prompt or choose dimensions; the request's system content is ignored. Do not add caller-side knobs.
7. **Every implementation uses the same judge input.** The reference is `/home/xingqianx/tmp/qjudge_final_prompts.json`; generated messages must match it word for word. Shared pieces are inherited from or imported from `qimagebench_protocol`, never copied.

## Changes since the 11-13 draft

| Draft | Now | Why |
|---|---|---|
| Judge input unchanged | Real-photo system prompt; checklists hold only the 17 counted criteria, 11 reworded | The official prompt frames images as generated, and the judge invents AI defects on real photos. On about 7,000 images of the 100k test table, Physical Logic was Fail on 46%. A random 16 of those were all normal photos: occlusion and shallow depth of field were read as "fusion deformities". |
| Caption fills the prompt slot | No caption | On 60 images, dropping the caption removed the false Fails as well and was more sensitive to real defects. Composition and Lighting & Atmosphere were reworded without the prompt, so no accurate caption is needed. |
| No server change, filter logic on client | All scoring on server | User decision on the caller contract |
| 15 `discard` criteria still asked | Not asked | Shorter checklists |
| No orientation check | Unscored `upright_orientation` flag in the Quality call | Images stored rotated 90 degrees are a known data error; no counted criterion catches them reliably (see Calibration) |

## Relation to the existing filter

The existing filter is `qwen3vl_fft_8b_v1`, a fine-tuned Qwen3-VL 8B. Its default rule keeps a row only if `aesthetic_score >= 4` and all five flags are false.

| Existing column | Used today | New filter |
|---|---|---|
| `aesthetic_score` (0-10) | keep if >= 4 | replaced by Quality and Aesthetics L3 scores |
| `text_banner` | drop if true | kept; graded by Text Accuracy, Text Layout, Graphic Design |
| `white_background` | drop if true | kept; graded by Product Design plus Quality and Aesthetics |
| `collage` | drop if true | kept; no collage criterion. Reworded Feature Matching targets cut-out/pasted seams, not multi-photo layouts. Gap |
| `watermark` | drop if true | Text Accuracy lists watermarks as an example but grades legibility, not presence. Gap, undecided |
| `nsfw` | drop if true | Safety & Compliance is in Real-world Fidelity, not asked. Undecided |

## Criteria tree

56 official L3 criteria; 17 are asked. `official` = wording unchanged; `reworded` = wording changed. Fail rates are on 1,000 real photos, official prompt -> final.

```
Quality                      call 1 (judge sees "Quality")
  Realism
    Physical Logic           official   Fail 48.1% -> 1.4%; remaining ~10/14 are renders, composites, digital paintings
    Material Texture         official
  Detail
    Noise                    official   Fail 0.9% -> ~3.5% (night/film grain); rewording tried, no effect; accepted
    Edge Clarity             official
    Naturalness              reworded   "AI-generated plastic look" -> "heavy retouching, over-processing, over-smoothed"; Fail 14.1% -> 3.9%
  Resolution
    Resolution               official
  Orientation
    upright_orientation      new, unscored true/false flag; not one of the 17

Aesthetics                   call 2 (judge sees "Aesthetics")
  Composition
    Composition              reworded   was "composition rules required by the prompt"; now "balanced, visually guided, aesthetically pleasing"
  Color Harmony
    Color Harmony            official
  Lighting
    Lighting & Atmosphere    reworded   was "matches the scene setting of the prompt"; now "well handled and suited to the scene shown"
  Anatomical Portraiture
    Anatomical Fidelity      reworded   only if people are visible, else N/A; Fail 12.8% -> 0.3% (animals were judged by human anatomy)

Creative Generation          call 3 (judge sees "Text and Design")
  Feature Matching
    Feature Matching         reworded   seams where subjects/regions meet, cut-out edges, halos, pasted-in elements; Fail 1.8%; unstable
  Text Rendering
    Text Accuracy            reworded   real-world text examples incl. watermarks; N/A without text; misspelling clause kept; Fail 4.5% -> 0.4%
    Text Layout              reworded   N/A without text
  Design Applications
    Graphic Design           reworded   only if the image is/prominently shows a poster, ad, cover; else N/A
    Product Design           reworded   only if a manufactured product is the main subject; else N/A
    Spatial Design           reworded   only if an interior/architectural space is the main subject; else N/A
    Game Design              reworded   only if game props or game UI are shown; else N/A
```

The scoring rules' N/A line changed from "does not apply to this image/prompt" to "does not apply to this image".

Not asked (39):
- Alignment (16): 13 compare against the prompt; the other 3 are not worth an extra call.
- Real-world Fidelity (8):
  - Social Bias and Cultural Fairness judge a generator's bias and would remove legitimate photos.
  - World Knowledge criteria are constant on real photos or prompt-dependent.
  - Safety & Compliance is valid but not worth a separate call.
- Aesthetics: Emotional Expression and Style Control (prompt-dependent).
- Creative Generation (13):
  - Imagination and Logical Resolution: not applicable to real photos.
  - Font, Cross-lingual Generation, Fashion Styling, Art Design: prompt-dependent.
  - The 7 Visual Storytelling criteria: prompt-dependent.

Why `Text and Design`: calibration always used it, and showing `Creative Generation` truncated more answers (4.8% vs 1.8%).

`Composition` exists twice as an L3 name in the official tree. Only the Aesthetics one is asked, but always key L3 scores by (L1, L2, L3).

## Output and parsing

- **Return value.** `{L1: {L2: {L3: score}}}` with L1 keys `Quality`, `Aesthetics`, `Creative Generation`, 17 L3 entries in total. Each score is in `{0.0, 60.0, 100.0, None}`, meaning Fail, Pass, Excel, N/A.
- **Orientation flag.** `Quality.Orientation.upright_orientation` is `true`, `false` or `None`. `None` when the judge answers anything other than true/false (it answered `"N/A"` on 2 of 150 test calls) or the Quality call does not parse. The parser reads `{"value": ...}` (also `{"score": ...}`, strings "true"/"false", and the flag placed directly under `Orientation`); a missing flag never discards the Quality scores.
- **Parsing.**
  - First `extract_json_from_response`, then a recursive walk that collects every object with a `score` key, at any depth, and places it by name.
  - An L2 name with a single L3 maps to that L3 (e.g. `Lighting`).
  - `Feature Mapping` is renamed to `Feature Matching`.
  - This covers flat L3, flat L2, L1-wrapped and mixed layouts.
  - Regression: on 11,406 past official-prompt outputs, the result is identical to the old parser.
- **Retries.** Up to 2 attempts per dimension; only the failed call is rerun. A dimension still unparsed has all its criteria `None`: unknown, never pass or fail.
- **No thresholds in the scoring step.** Changing a rule must not require rescoring.

## Calibration (1,000 real photos)

The sample is a random 1,000 of the ~7,000 images already scored with the official prompt on `s3://nv-00-10206-vfm/lancedb/image/regular/Aesthetic-Train-V2_slice_from_maintable_20261009.lance`. Conclusions were checked by looking at images and judge reasoning.

| Criterion | Official prompt Fail | Final Fail | Notes |
|---|---|---|---|
| Physical Logic | 48.1% | 1.4% | remaining are mostly renders/composites |
| Naturalness | 14.1% | 3.9% | slightly strict |
| Anatomical Fidelity | 12.8% | 0.3% | animals no longer judged by human anatomy |
| Text Accuracy | 4.5% | 0.4% | real signage no longer flagged |
| Noise | 0.9% | ~3.5% | night/film grain; accepted |
| Feature Matching | - | 1.8% | includes confirmed composites; unstable |
| Four Design criteria | - | 1 image | almost always N/A |
| Truncation (no JSON within 4096 tokens) | - | 1.3% | 14% on text-heavy generated images |

Controls:
- **Degraded images.** Blur, JPEG and pixelation (10 each) are almost always Fail on Material Texture, Edge Clarity and Resolution.
- **Added Gaussian noise.** Noise is Fail on 19/20 at sigma 40 and 14/20 at sigma 20.
- **Garbled text on generated images** (QImageBench outputs of fibo2, flux3_klein, ideogram4).
  - The final prompt catches 61% of the official Text Accuracy = 0 images.
  - 5 of about 43 official 100s get Fail; about 3 of those have genuinely garbled background text.
- **Misspelling clause cost.** About 0.3% of real photos get Fail for text that is misspelled in reality.

Orientation (50 random calibration photos, each judged as original, rotated 90 degrees left and 90 degrees right):

| | Original | Rotated left | Rotated right |
|---|---|---|---|
| `upright_orientation` = false | 0 / 50 | 34 / 50 | 32 / 50 |
| At least one counted Fail, without the flag | 6 / 50 | 32 / 50 | 25 / 50 |

- No original photo is flagged.
- Without the flag, rotation shows up mainly as Composition Fail (2 -> 23 of 50); every image caught that way is also caught by the flag.
- Misses fall in two groups. About half have no clear up direction (aerial, top-down food, textures), which is acceptable. The rest are clearly rotated but the judge either skipped reasoning and answered true, or explained the view away (a sideways face read as a person lying down, a bird on a "diagonal branch").
- An earlier wording, `Rotation Failure` (true = rotated), caught 29 and 28; the positive `upright_orientation` wording was kept.
- Adding the item did not move the other Quality scores beyond rerun noise: 84% identical on originals and 91% on rotated images, against 83-94% for an unchanged rerun.

Stability:
- Criteria with identical wording give identical scores on a rerun only 83-94% of the time, mostly 60 and 100 swapping.
- The sets of images graded Excel (100) overlap only 40-65% between runs.
- Feature Matching Fails changed mostly when only the dimension name changed.

## Using it for 40M -> 1M

- Gating on Fails alone removes about 9%: 90.7% of 975 real photos have no Fail on any criterion. This was computed on 16 criteria, before Feature Matching was restored.
- Keeping the top 2.5% needs ranking by number of Excel grades. The cut is about 7 of 16 at Excel, with ties.
- Excel instability makes that ranking noisy.
- Recommended role: a gate that removes Fails.
- Ranking needs a finer signal, e.g. an expected score from the judge's per-grade probabilities. Not built.

## Risks

1. Single-image noise; Excel overlaps 40-65% between runs.
2. Noise fails about 3.5% of real photos for visible grain (accepted).
3. The misspelling clause fails about 0.3% of real photos with real-world misspellings (accepted).
4. Feature Matching is unstable; do not use it alone as a hard rule.
5. Truncation: 1.3% on real photos, 14% on text-heavy images; that dimension becomes null.
6. Anatomical Fidelity rewards visible pores and fine lines; it may penalize retouched or studio portraits.
7. There is no criterion for collages or watermarks.
8. Calibrated only on the Aesthetic-Train-V2 slice.
9. `upright_orientation` catches about two thirds of images rotated 90 degrees and misses some clearly rotated ones; it has no false alarms on upright photos in the test. It was tested on 50 images only.

## Rules for agents working on this

- Do not edit `qimagebench_protocol.py` or `qimagebench.py`; benchmark scoring and RL reward depend on them.
- Keep the judge messages identical to `/home/xingqianx/tmp/qjudge_final_prompts.json`. Any prompt change needs user approval and recalibration.
- Do not drop or add counted criteria without asking. `upright_orientation` is a flag, not a counted criterion; keep it out of any score aggregate.
- Keep this document to design only: no run logs, job ids, throughput or deployment details.

## Open questions

- The filter rule: how to define the gate, and how to rank the top 2.5%.
- Whether to build a probability-based expected score.
- Watermarks: keep the existing filter's `watermark` column, or ignore them.
- NSFW: keep the existing filter's `nsfw` column, or add a call for Safety & Compliance.
- Collages.
