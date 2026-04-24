# Captioning quality stress testing via iterative refinement and judging system

_Stress test of the `golden_caption` pipeline, v2 → v11, on `CosCapBenchImage/V1`._

---

## Motivation

Get the best legally-usable caption we can, and find out whether today's VLM/LLM stack is smart enough to **self-improve on captioning**. If it is, the natural next step is an agentic refinement loop that keeps editing/re-grounding until a judge is satisfied.

---

## Gains and conclusions

- **Full literature review:** see this doc: _\<i will add a link later\>_.
- **Infra stress test passed.** 10 end-to-end runs (v2 … v11), 4 gen models, 2 prompt-design eras, mixed stage-1 inheritance — all produced complete 3-variant read-outs in the expected P/R regime, so differences are attributable to model/prompt, not bugs.
- **Interesting details.**
  - Stage-4 structured caption is the best variant in **every** version (+1–3 F1 pts over stage 3).
  - `gemini-3.1-pro` gen (v4) has the highest precision (0.974) but the lowest recall / fewest claims — it writes short, safe captions.
  - Swapping stage-1 source between runs moves F1 by <1–2 pts.
  - Recall appears to saturate at ≈0.69; no version broke it.
- **[Negative] Iteration did NOT beat Seungjun's one-shot 9-grid captioner.** Despite 5-round judge-refine loops, 4 frontier VLMs swapped in/out, and two prompt rewrites, **no v2–v11 run exceeds F1 ≈ 0.80**. The best (v3 s4 = 0.7973) uses the simplest setup. Surprising.

---

## Methods

Stages 1–2 run a **propose → judge → refine** battle loop (up to 5 rounds). Stages 3–4 are single-pass with parse-retry only.

```
image ──► STAGE 1: entity search
             gen   : propose [(id, location, reason), …]
             judge : per-entity (id ok? loc ok?) + missing-entity sweep
             refine: fix flagged, drop hallucinations, add missing
             loop  : ≤5 rounds, stop early on all-pass
             out   : accepted entity list
       ──► STAGE 2: grounding then entity
             gen / judge (correct + complete booleans) / refine flagged only
             loop  : ≤5 rounds, chunked 16 entities
             out   : per-entity structured grounding JSON
       ──► STAGE 3: dense / mid-length grounding synthesis  (LLM only, no judge)
             gen   : dense paragraph + downsampled (15–100w)
             retry : parse-retry ≤3×
             out   : stage3_dense_caption/ , stage3_structured_caption/
       ──► STAGE 4: camera + style  (VLM, single-pass, no judge)
             gen   : image + stage-3 JSON → overall_style, camera_details
             out   : stage4_structured_caption/
       ──► EVAL (gemini-3.1-pro judge):
             precision = correct claims / total claims
             recall    = passed V1_assertions / total assertions
             F1        = 2PR / (P+R)
```

**Version / model sweep.** Judge VLM is held at `gemini-3.1-pro` from v3 onwards (only v2 mixes judges); generator side changes per row.

| Ver   | Judge VLM        | Gen VLM (S1/S2/S4)        | Gen LLM (S3) | Notes                                |
|-------|------------------|---------------------------|--------------|--------------------------------------|
| v2    | mixed            | mixed                     | same         | First looping-refinement version     |
| v3    | gemini-3.1-pro   | mixed                     | same         | —                                    |
| v4    | gemini-3.1-pro   | gemini-3.1-pro            | same         | Pure "pro"                           |
| v5    | gemini-3.1-pro   | gemini-3.1-flash          | same         | Pure "flash"                         |
| v6    | gemini-3.1-pro   | qwen3-vl-235b             | same         | Pure qwen                            |
| v7    | gemini-3.1-pro   | gemini-3-flash (S2–S4)    | same         | S1 from v6; new structured prompts   |
| v8    | gemini-3.1-pro   | gemini-3-flash (S2–S4)    | same         | S1 from v5; new structured prompts   |
| v9    | gemini-3.1-pro   | gemini-3-flash (S2–S4)    | same         | S1 from v5                           |
| v10p1 | gemini-3.1-pro   | gemini-3.1-pro (S2–S4)    | same         | S1 from v6                           |
| v11   | gemini-3.1-pro   | qwen3-vl-235b (S2–S4)     | qwen3-235b   | S1 from v5                           |

---

## Results (precision / recall / F1)

Three caption variants: **dense** (stage-3 paragraph), **struct** (stage-3 structured, 8 fields), **s4** (struct + camera + style).

| Ver   | Variant | Prec      | Rec    | F1         | #claims | #assert |
|-------|---------|-----------|--------|------------|---------|---------|
| v2    | dense   | 0.9331    | 0.6377 | 0.7576     | 51,191  | 4,797   |
| v2    | struct  | 0.9353    | 0.6402 | 0.7601     | 55,686  | 4,797   |
| v2    | s4      | 0.9393    | 0.6706 | 0.7825     | 54,032  | 4,797   |
| v3    | dense   | 0.9350    | 0.6609 | 0.7744     | 55,466  | 4,777   |
| v3    | struct  | 0.9380    | 0.6583 | 0.7737     | 60,082  | 4,797   |
| v3    | s4      | 0.9390    | 0.6927 | **0.7973** | 60,834  | 4,797   |
| v4    | dense   | **0.9762**| 0.6008 | 0.7439     | 33,698  | 4,765   |
| v4    | struct  | 0.9752    | 0.5925 | 0.7371     | 31,588  | 4,797   |
| v4    | s4      | 0.9742    | 0.6531 | 0.7820     | 33,774  | 4,797   |
| v5    | dense   | 0.9708    | 0.6281 | 0.7627     | 33,541  | 4,765   |
| v5    | struct  | 0.9706    | 0.6211 | 0.7575     | 31,865  | 4,769   |
| v5    | s4      | 0.9689    | 0.6688 | 0.7914     | 33,439  | 4,780   |
| v6    | dense   | 0.8656    | 0.6629 | 0.7508     | 56,153  | 4,728   |
| v6    | struct  | 0.8598    | 0.6485 | 0.7393     | 57,118  | 4,745   |
| v6    | s4      | 0.8719    | **0.6875** | 0.7688 | 56,996  | 4,745   |
| v7    | dense   | 0.8753    | 0.6593 | 0.7521     | 54,387  | 4,738   |
| v7    | struct  | 0.8670    | 0.6574 | 0.7478     | 59,335  | 4,749   |
| v7    | s4      | 0.8758    | 0.6877 | 0.7705     | 58,226  | 4,749   |
| v8    | dense   | 0.9055    | 0.6165 | 0.7335     | 18,636  | 4,649   |
| v8    | struct  | 0.8948    | 0.6045 | 0.7215     | 27,158  | 4,690   |
| v8    | s4      | 0.9003    | 0.6514 | 0.7559     | 30,346  | 4,690   |
| v9    | dense   | 0.9709    | 0.6226 | 0.7587     | 20,345  | 4,780   |
| v9    | struct  | 0.9733    | 0.6130 | 0.7522     | 28,853  | 4,780   |
| v9    | s4      | 0.9724    | 0.6500 | 0.7792     | 30,468  | 4,780   |
| v10p1 | dense   | 0.9690    | 0.6249 | 0.7598     | 19,399  | 4,772   |
| v10p1 | struct  | 0.9696    | 0.6180 | 0.7549     | 25,582  | 4,772   |
| v10p1 | s4      | 0.9709    | 0.6563 | 0.7832     | 27,524  | 4,772   |
| v11   | dense   | 0.9459    | 0.6431 | 0.7657     | 17,718  | 4,733   |
| v11   | struct  | 0.9377    | 0.6329 | 0.7557     | 27,621  | 4,756   |
| v11   | s4      | 0.9349    | 0.6680 | 0.7792     | 32,488  | 4,756   |

Per-version best F1 (stage-4 variant, sorted):

| Ver   | F1 (s4)    | Prec   | Rec    |
|-------|------------|--------|--------|
| v3    | **0.7973** | 0.9390 | 0.6927 |
| v5    | 0.7914     | 0.9689 | 0.6688 |
| v10p1 | 0.7832     | 0.9709 | 0.6563 |
| v2    | 0.7825     | 0.9393 | 0.6706 |
| v4    | 0.7820     | 0.9742 | 0.6531 |
| v9    | 0.7792     | 0.9724 | 0.6500 |
| v11   | 0.7792     | 0.9349 | 0.6680 |
| v7    | 0.7705     | 0.8758 | 0.6877 |
| v6    | 0.7688     | 0.8719 | 0.6875 |
| v8    | 0.7559     | 0.9003 | 0.6514 |

---

## Analysis

1. **Stage-4 is free quality.** Every version's F1 peaks on s4 (+1–3 pts over stage 3). Camera/style tags add recall-able claims that the judge rarely flags as wrong.

2. **Gen model picks the P/R operating point; F1 barely moves.** Across all 10 versions × 3 variants, F1 spans only 0.72 → 0.80. Precision swings 10 pts (0.87 → 0.97) and recall swings 10 pts (0.59 → 0.69), but they trade off ~1-for-1 — F1 is nearly invariant.

3. **More claims ≠ more recall.** v6 (57k claims, R=0.687) ≈ v3 (60k, 0.693) ≈ v10p1 (27k, 0.656). Doubling caption length mostly adds already-covered detail.

4. **Iterating harder didn't unlock anything.** v7/v8 (new structured prompts), v10p1 (pro on S2–S4), v11 (full qwen) were all designed to push past 0.80 F1. None did; the best run (v3 s4) uses the simplest setup.

5. **Why?** Two hypotheses:
   - Judge and refiner are the **same model family** that wrote the caption, so the loop converges to the model's own "complete" rather than ground truth.
   - V1 assertions saturate around 0.69 recall because many need fine-grained spatial/counting claims that frontier VLMs can't reliably see — more rounds won't fix that.

Next step: test **agentic** refinement (judge requests targeted crop re-views, not full-image re-views) to try to break the 0.80 F1 plateau.

---

### Appendix

- Pipeline outputs: `gcs://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION_LONG>/`
- Eval results: `…/results/{precision,recall}_eval_result_<SIGNATURE>.json`
- Benchmark: `…/CosCapBenchImage/{V1, V1_assertion}/`
- Stage code: `imaginaire4/projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage{1..4}_*.py`
