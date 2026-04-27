# Captioning quality stress testing via iterative refinement and judging system

_Stress test of the `golden_caption` pipeline, v2 → v15, on `CosCapBenchImage/V1`._

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
| v5    | gemini-3.1-pro   | gemini-3-flash            | same         | Pure "flash"                         |
| v6    | gemini-3.1-pro   | qwen3-vl-235b             | same         | Pure qwen                            |
| v7    | gemini-3.1-pro   | gemini-3-flash (S2–S4)    | same         | S1 from v6; new structured prompts   |
| v8    | gemini-3.1-pro   | qwen3-vl-235b (S2–S4)     | qwen3-235b   | S1 from v6; new structured prompts   |
| v9    | gemini-3.1-pro   | gemini-3-flash S2; mixed S3 / g3p S4 | g3f → g3p tail | S1 from v5; S3 last 144-img batch + S4 reran on gemini-3.1-pro |
| v10p1 | gemini-3.1-pro   | gemini-3.1-pro (S2–S4)    | same         | S1 from v6                           |
| v11   | gemini-3.1-pro   | qwen3-vl-235b (S2–S4)     | qwen3-235b   | S1 from v5                           |
| v12   | gemini-3.1-pro   | gemini-3-flash            | same         | New S1+S2 prompting design (YAML I/O)|
| v13   | gemini-3.1-pro   | qwen3-vl-235b (S2–S4)     | qwen3-235b   | `q235bg3p`; qwen gen, pro judge      |
| v14   | gemini-3.1-pro   | qwen3-vl-235b (S2–S4)     | qwen3-235b   | S1 from v13; identifier-location fix |
| v15   | gemini-3.1-pro   | qwen3-vl-235b             | qwen3-235b   | Fully agent self-improved prompting system |

---

## Results (precision / recall / F1)

Three caption variants: **dense** (stage-3 paragraph), **struct** (stage-3 structured, 8 fields), **s4** (struct + camera + style).

Model abbreviations: **g3.1p** = gemini-3.1-pro, **g3f** = gemini-3-flash, **q235b** = qwen3-vl-235b, **mix** = mixed.

`#img_P` / `#img_R` are the number of benchmark images successfully evaluated by the precision / recall judges respectively. Bench size is ≈300 images.

| Ver   | S1    | S2–S4  | Variant | Prec      | Rec    | F1         | #claims | #img_P | #img_R |
|-------|-------|--------|---------|-----------|--------|------------|---------|--------|--------|
| v2    | mix   | mix    | dense   | 0.9331    | 0.6377 | 0.7576     | 51,191  | 299    | 294    |
| v2    | mix   | mix    | struct  | 0.9353    | 0.6402 | 0.7601     | 55,686  | 300    | 294    |
| v2    | mix   | mix    | s4      | 0.9393    | 0.6706 | 0.7825     | 54,032  | 300    | 294    |
| v3    | mix   | mix    | dense   | 0.9350    | 0.6609 | 0.7744     | 55,466  | 299    | 293    |
| v3    | mix   | mix    | struct  | 0.9380    | 0.6583 | 0.7737     | 60,082  | 300    | 294    |
| v3    | mix   | mix    | s4      | 0.9390    | 0.6927 | **0.7973** | 60,834  | 300    | 294    |
| v4    | g3.1p | g3.1p  | dense   | **0.9762**| 0.6008 | 0.7439     | 33,698  | 300    | 293    |
| v4    | g3.1p | g3.1p  | struct  | 0.9752    | 0.5925 | 0.7371     | 31,588  | 300    | 294    |
| v4    | g3.1p | g3.1p  | s4      | 0.9742    | 0.6531 | 0.7820     | 33,774  | 300    | 294    |
| v5    | g3f   | g3f    | dense   | 0.9708    | 0.6281 | 0.7627     | 33,541  | 299    | 292    |
| v5    | g3f   | g3f    | struct  | 0.9706    | 0.6211 | 0.7575     | 31,865  | 299    | 292    |
| v5    | g3f   | g3f    | s4      | 0.9689    | 0.6688 | 0.7914     | 33,439  | 299    | 293    |
| v6    | q235b | q235b  | dense   | 0.8656    | 0.6629 | 0.7508     | 56,153  | 297    | 290    |
| v6    | q235b | q235b  | struct  | 0.8598    | 0.6485 | 0.7393     | 57,118  | 297    | 291    |
| v6    | q235b | q235b  | s4      | 0.8719    | **0.6875** | 0.7688 | 56,996  | 297    | 291    |
| v7    | q235b | g3f    | dense   | 0.8753    | 0.6593 | 0.7521     | 54,387  | 297    | 290    |
| v7    | q235b | g3f    | struct  | 0.8670    | 0.6574 | 0.7478     | 59,335  | 297    | 291    |
| v7    | q235b | g3f    | s4      | 0.8758    | 0.6877 | 0.7705     | 58,226  | 297    | 291    |
| v8    | q235b | q235b  | dense   | 0.9055    | 0.6165 | 0.7335     | 18,636  | 291    | 286    |
| v8    | q235b | q235b  | struct  | 0.8948    | 0.6045 | 0.7215     | 27,158  | 291    | 288    |
| v8    | q235b | q235b  | s4      | 0.9003    | 0.6514 | 0.7559     | 30,346  | 294    | 288    |
| v9    | g3f   | g3f†   | dense   | 0.9709    | 0.6226 | 0.7587     | 20,345  | 298    | 293    |
| v9    | g3f   | g3f†   | struct  | 0.9733    | 0.6130 | 0.7522     | 28,853  | 299    | 293    |
| v9    | g3f   | g3f†   | s4      | 0.9724    | 0.6500 | 0.7792     | 30,468  | 297    | 293    |
| v10p1 | q235b | g3.1p  | dense   | 0.9690    | 0.6249 | 0.7598     | 19,399  | 299    | 293    |
| v10p1 | q235b | g3.1p  | struct  | 0.9696    | 0.6180 | 0.7549     | 25,582  | 299    | 293    |
| v10p1 | q235b | g3.1p  | s4      | 0.9709    | 0.6563 | 0.7832     | 27,524  | 299    | 293    |
| v11   | g3f   | q235b  | dense   | 0.9459    | 0.6431 | 0.7657     | 17,718  | 288    | 290    |
| v11   | g3f   | q235b  | struct  | 0.9377    | 0.6329 | 0.7557     | 27,621  | 291    | 291    |
| v11   | g3f   | q235b  | s4      | 0.9349    | 0.6680 | 0.7792     | 32,488  | 292    | 291    |
| v12   | g3f   | g3f    | dense   | 0.9608    | 0.6116 | 0.7474     | 19,133  | 297    | 292    |
| v12   | g3f   | g3f    | struct  | 0.9565    | 0.5996 | 0.7371     | 28,244  | 298    | 292    |
| v12   | g3f   | g3f    | s4      | 0.9581    | 0.6532 | 0.7768     | 29,797  | 298    | 292    |
| v13   | q235b | q235b  | dense   | 0.9388    | 0.6130 | 0.7417     | 23,085  | 298    | 292    |
| v13   | q235b | q235b  | struct  | 0.9380    | 0.5962 | 0.7290     | 26,937  | 298    | 292    |
| v13   | q235b | q235b  | s4      | 0.9400    | 0.6434 | 0.7639     | 31,574  | 298    | 292    |
| v14   | q235b | q235b  | dense   | 0.9419    | 0.6071 | 0.7384     | 21,801  | 296    | 293    |
| v14   | q235b | q235b  | struct  | 0.9384    | 0.5801 | 0.7170     | 24,499  | 299    | 293    |
| v14   | q235b | q235b  | s4      | 0.9361    | 0.6374 | 0.7584     | 31,412  | 299    | 293    |
| v15   | q235b | q235b  | dense   | 0.9409    | 0.6034 | 0.7353     | 22,137  | 298    | 292    |
| v15   | q235b | q235b  | struct  | 0.9410    | 0.5889 | 0.7244     | 23,646  | 296    | 292    |
| v15   | q235b | q235b  | s4      | 0.9341    | 0.6370 | 0.7575     | 32,049  | 298    | 292    |

† v9's "S2–S4" is heterogeneous: S2 grounding ran on gemini-3-flash, but the final S3 dense-rewriter batch (144 images) was redone with gemini-3.1-pro, and S4 (camera+style) ran end-to-end on gemini-3.1-pro. Treat v9's high precision (≈0.97) as a g3p-tail effect, not as pure-flash performance.

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
| v12   | 0.7768     | 0.9581 | 0.6532 |
| v7    | 0.7705     | 0.8758 | 0.6877 |
| v6    | 0.7688     | 0.8719 | 0.6875 |
| v13   | 0.7639     | 0.9400 | 0.6434 |
| v14   | 0.7584     | 0.9361 | 0.6374 |
| v15   | 0.7575     | 0.9341 | 0.6370 |
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
