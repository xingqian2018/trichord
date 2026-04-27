# Why v15's "self-improved prompts" did not beat v14

**Subject:** `golden_caption_v15_q235bg3p` vs `golden_caption_v14_q235bg3p`
**Benchmark:** `CosCapBenchImage/V1` (≈300 images, ≈4.7k–4.8k recall assertions)
**Judge for P/R evaluation:** `gemini-3.1-pro`
**Authored:** 2026-04-27

---

## 1. TL;DR

v15 was billed as a "fully agent self-improved prompting system" on top of v14. On every caption variant the eval moves by **<1 percentage point** in either direction — i.e. statistical noise. The change was a **single commit, prose-only**, that pushed producer and judge in opposite directions on the same set of attributes. Inside the F1 metric the two pressures cancel.

| Variant | v14 P → v15 P | Δ P     | v14 R → v15 R | Δ R     | v14 F1 → v15 F1 |
|---------|---------------|---------|---------------|---------|------------------|
| s3d     | 0.9419 → 0.9409 | −0.10pp | 0.6071 → 0.6034 | −0.37pp | 0.7384 → 0.7353 |
| s3s     | 0.9384 → 0.9410 | +0.26pp | 0.5801 → 0.5889 | +0.88pp | 0.7170 → 0.7244 |
| s4s     | 0.9361 → 0.9341 | −0.20pp | 0.6374 → 0.6370 | −0.04pp | 0.7584 → 0.7575 |

(Image counts are close but not identical: v14 evaluated 299 images, v15 evaluated 298. Sub-1pp deltas should be treated as noise.)

---

## 2. The change

Single author, single commit in the v14→v15 window:

| SHA          | Date (PDT)     | Author    | Message                       | Files                                                                                                          |
|--------------|----------------|-----------|-------------------------------|----------------------------------------------------------------------------------------------------------------|
| `ef83010d4d` | 2026-04-26 15:21 | xingqianx | self improved gp prompts      | `prompts/stage1_entity_{search,refinement,judge}_system.md`, `stage2_entity_grounding.py`, `stage4_camera_and_style.py` |

Diff size: **+176 / −16** lines. No collaborator commits in window. No `CHANGELOG.md` / version-notes touched.

What did **not** change:
- No code-flow change (loop structure, max iterations, retry policy untouched)
- No JSON schema change (no new fields, no per-claim `confidence`, no atomic-claim format)
- No model swap, no temperature change
- No new few-shot examples added

Everything was prose added to system prompts.

---

## 3. What the prose added (categorized)

| Bucket                      | Share of diff | Examples                                                                                                                  |
|-----------------------------|---------------|---------------------------------------------------------------------------------------------------------------------------|
| (a) Clarity / wording tweaks  | ~10%          | Adding "materials" to assertions list; minor restatement                                                                  |
| (b) **New constraints**     | **~75%**      | Highest-risk-category checklists; gating conditions for DOF / camera angle / palette / composition; "if judge flagged X do Y" repair tables |
| (c) New few-shot examples   | 0%            | None                                                                                                                      |
| (d) Restructured loop control | 0%            | None                                                                                                                      |
| (e) Cosmetic / formatting   | ~15%          | Final pre-output check reminders that restate earlier rules                                                               |

Representative additions:

**Stage 1 (search system prompt) — "highest-risk attributes" checklist:**
> - Color (most common error class): only include a color word when the color is unambiguous on a clean, well-lit portion …
> - Counts and quantities: only embed a count … when you can literally count each instance …
> - Material / finish: only include "matte", "glossy", "polished" … when the finish is unmistakable.
> - Left vs right (CAMERA perspective, not the depicted person's body frame) …

**Stage 2 (grounding system prompt) — stakes preamble + 11-item checklist:**
> EVERY assertion you write becomes an independent claim that downstream judges will verify …
> Past evaluation runs show the dominant failure modes are, in order:
> L/R from the wrong frame …, picture-plane vs imagined-3D confusion …

**Stage 2 (judge prompt) — old 5-bullet "be vigilant about" list expanded into the SAME 11 failure modes** the producer was just warned about (L/R camera frame, above/below picture-plane, color shade, counts, material, hallucinated sub-detail, OCR, orientation, occlusion, compound spatial, light/shadow consistency).

**Stage 4 (camera/style) — disambiguation rules block:**
> - `low angle` means the camera is positioned BELOW the subject's eye-line …
> - `rule of thirds` requires the main subject (or horizon) to lie roughly on a third-line, NOT centered …
> - `muted` / `desaturated` / `washed-out` requires actually low-saturation colors throughout.

---

## 4. Why this likely produced ~0 movement

1. **Symmetric pressure on producer and judge cancels inside F1.** Producer prompts (search, grounding, stage 4) were taught to *under-claim* on color shade, counts, finish, sub-details, compound L/R. The judge prompt was simultaneously taught to *more aggressively flag* those same categories. So whatever recall the producer gives up by retreating to family-level color or "a row of" instead of "three", the judge no longer penalizes anyway. The two changes net out: P and R each drift, F1 stays put. The ±1pp envelope across **all three** variants (s3d / s3s / s4s) is the textbook signature of this kind of symmetric cancellation.

2. **Exhortations, not constraints.** Nothing in the diff *enforces* the new rules: no JSON field for per-claim `confidence`, no schema requiring decomposition into atomic claims, no pass/fail validator. Qwen3-VL-235B already has high prior on these failure modes; +160 lines of "remember to check L/R from the camera frame" mostly restates capabilities the model already has rather than unlocking new behavior. Changes that *do* move benchmarks tend to be schema additions, decoder constraints, retrieval, ensembling, or model swaps — not more imperative bullets.

3. **Uniform prior, not a targeted fix.** The 11-item checklist treats every failure mode as equally important. There is no v14 error-class breakdown showing that L/R-camera-frame / picture-plane / matte-vs-glossy is the actual bulk of the ≈24% of stage-4 claims that fail. Without that breakdown, the rewrite is a uniform prior over an already-calibrated model. Uniform priors over calibrated models produce uniform null effects, which is what we observe.

A secondary observation: the stage 1 refinement and stage 2 BATTLE_NEW_GROUNDING blocks got more verbose, but the loop control around them was untouched. If v14 already converged in roughly the same number of iterations as v15, more elaborate per-iteration instructions don't expand the search space.

---

## 5. Recommendations for the next iteration

To actually move the benchmark, the next round likely needs at least one of:

- **Targeted, not uniform.** Slice v14 (or v15) failed claims by category — color shade, counts, L/R camera frame, picture-plane, occlusion, OCR — using the existing precision-judge output JSON. Pick the **largest failing bucket(s)** and rewrite only those. A targeted prompt change against a measured bottleneck is the cheapest "real" intervention.

- **Structural change, not prose.** Add a per-claim `confidence` field; require atomic-claim output; insert a separate cheap verifier model that votes against borderline claims; or constrain the decoder so unsupported tokens are unreachable. Any of these change the *output distribution* in a way prose cannot.

- **Break the symmetric-cancellation pattern.** Either pressure producer and judge in the **same** direction (both more permissive, both stricter), or decouple them entirely (judge only sees a cropped region, judge sees a different model's caption to compare, etc.). Right now they are taught to be cautious about identical attributes, which guarantees no F1 movement.

- **Open the door to agentic refinement (already in the v2–v13 long-version recommendations).** Let stage 2's refiner issue a targeted **crop + re-ask** on any entity whose grounding the judge marked incomplete. One tool, one extra VLM call per flagged entity, and it directly attacks the recall bottleneck rather than retreading the precision/recall trade-off.

---

## 6. Pointers

- Commit: `git show ef83010d4d` in `~/Project/imaginaire4` (branch `xingqianx/cosmos3_aid`)
- Touched prompt files (all under `imaginaire4/projects/cosmos3/vfm/evaluation/captioning/golden_caption/`):
  - `prompts/stage1_entity_search_system.md`
  - `prompts/stage1_entity_refinement_system.md`
  - `prompts/stage1_entity_judge_system.md`
  - `stage2_entity_grounding.py`
  - `stage4_camera_and_style.py`
- v14 / v15 result JSONs: `s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_v1{4,5}_q235bg3p/results/{precision,recall}_eval_result_v1{4,5}s{3d,3s,4s}.json`
- Companion reports:
  - `~/Project/trichord/reports/golden_caption_analysis.md` (summary, v2 → v15)
  - `~/Project/trichord/reports/golden_caption_analysis_long_version.md` (long, v2 → v15)
