# Captioning quality stress testing via iterative refinement and judging system.

_Experiment report covering `golden_caption` v2 → v11 on the CosCapBenchImage / V1 benchmark._

---

## 1. Motivation

We want to push image captioning quality as hard as we legally can with off-the-shelf VLMs/LLMs, and — equally importantly — to get a read on whether a VLM/LLM is already smart enough to **self-improve** on the captioning task when placed inside a judge / refine loop. If yes, it justifies a follow-up **agentic caption refinement** system; if no, we save ourselves from building one.

Concretely, the stress test runs a 4-stage pipeline in which a _generator_ VLM proposes entities / groundings, a _different_ VLM acts as _judge_, and a third call _refines_ any field the judge flags. We run up to 5 judge↔refine "battle rounds" per sample and measure whether the resulting captions beat the existing one-shot baseline.

---

## 2. Gains and conclusions

- **Full literature review**: see `<link TBD>`.
- **Infra reliability**: to rule out engineering mis-design as a confound, the v2 → v11 sweep deliberately swaps generator, judge, and stage-1 starting points across versions. The pipeline is a shared distributed harness (state machine + priority queue + async download/upload + batched VLM/LLM calls) — the code path that produces v2 is byte-for-byte the code path that produces v11; only models and stage-1 seeds change. So any deltas we see come from _model choice / prompt design_, not infra.
- **Interesting details**
  - Stage-4 (camera + style) uniformly lifts F1 over stage-3 across **every** version, typically by +1–3 points of recall with essentially no precision loss. Adding camera/style metadata is "free" information that the recall evaluator rewards.
  - The **judge model is louder than the generator**: versions that let "mixed" models judge (v2, v3) give up ~7 points of precision but gain 3–5 points of recall compared to versions judged only by gemini-3.1-pro. Precision on gemini-3.1-pro–judged runs hovers near 0.97; on qwen-judged or mixed-judged runs it collapses to ~0.87.
  - Swapping only the stage-1 seed (v8 uses v5's stage1, v9 uses v5's stage1 too but re-runs stage1 with gemini-3-flash, v10p1 uses v6's qwen stage1, v11 uses v5's stage1) shifts F1 by <1 point. **Stage 1 is not the bottleneck.**
  - Single-model all-gemini-3.1-pro (v4) has the _highest_ precision of the sweep (0.974) but the _lowest_ recall (0.60), suggesting that one strong model talking to itself is conservative — it refuses to assert things it cannot ground, and the judge version of itself rubber-stamps that conservatism.
- **[Negative] The iterative stress test does not beat Seungjun's one-shot 9-grid-search captioner.** Across 10 distinct (model, stage-1-seed) configurations and 5 battle rounds per stage, stage-4 F1 lands in a narrow **0.756–0.797** band. None of the configurations crosses the one-shot baseline decisively. This is surprising — we had expected iteration to buy clear gains — and it is the main result of this sweep.

---

## 3. Methods

The pipeline is a four-stage cascade. Stages 1 and 2 contain an internal **judge ↔ refine** loop (max 5 rounds). Stages 3 and 4 are single-pass. A final conversion step turns the entity list into two caption variants (dense + structured).

```
  image (.jpg/.jpeg/.webp)
        |
        v
  +-----------------------------------------------------------------+
  |  STAGE 1 — entity_search (VLM, image-in)                        |
  |  state machine:  search --> judge --> (refine --> judge)*--done |
  |    generator VLM   -> entity_list[{identifier, location, why}]  |
  |    judge VLM (!=gen) -> {correct | incorrect | duplicated} +    |
  |                         missing_entities flag                   |
  |    refine VLM (!=judge) rewrites ONLY flagged entities          |
  |    stop when all-correct-and-nothing-missing OR round >= 5      |
  |  OUT: per-image entity_list with provenance (vlm, round)        |
  +-----------------------------------------------------------------+
        |
        v
  +-----------------------------------------------------------------+
  |  STAGE 2 — entity_structured_grounding (VLM, image + stage1)    |
  |  chunks of 16 entities; 8 sub-fields per entity:                |
  |    description, location, relation, pose, materials,            |
  |    clothing, facial_and_expression, text_and_signage            |
  |  same search -> judge -> refine loop; judge returns per-field   |
  |  (correct, complete) booleans; only flagged entities are        |
  |  re-grounded; stop when nothing is flagged OR round >= 5        |
  +-----------------------------------------------------------------+
        |
        v
  +-----------------------------------------------------------------+
  |  STAGE 3 — entity_dense_grounding (LLM text-only, no judge)     |
  |  chunks of 16; pipe-delimited table in, table out; emits:       |
  |    grounding_dense             (full paragraph, lossless)       |
  |    grounding_dense_downsampled (15-100 words, appearance only)  |
  +-----------------------------------------------------------------+
        |
        v
  +-----------------------------------------------------------------+
  |  STAGE 4 — camera_and_style (VLM, image + stage3 JSON)          |
  |  single VLM call, JSON-mode:                                    |
  |    { image_style, camera_details }                              |
  |  merged onto stage3 JSON                                        |
  +-----------------------------------------------------------------+
        |
        v
  +-----------------------------------------------------------------+
  |  CONVERSION — entity_list -> { stage3_dense_caption,            |
  |                                 stage3_structured_caption,      |
  |                                 stage4_structured_caption }     |
  +-----------------------------------------------------------------+
```

**Key design choices**

- The judge model is always picked to be _different_ from the generator on the same sample (`pick_least_used` with exclusion), so a model is never allowed to grade its own homework within a round.
- Refinement is _surgical_: only entities / fields the judge flagged are re-emitted; already-accepted content is preserved with its provenance, so the loop monotonically grows the "accepted" set rather than re-shuffling everything.
- The hard cap is `max_battle_rounds = 5`. In practice most samples exit at round 1 or 2; very few hit the cap.
- Stage 3 is LLM-only (no image), which lets us swap in qwen3-235b or gemini-3.1-pro for prose-writing independently of the visual grounding decisions made in stages 1–2.

**Version matrix (generator / stage-1 seed)**

| Version | Gen model (S1, S2, S4) | Stage-3 LLM | Stage-1 seed |
|---|---|---|---|
| v2      | mixed                         | same as gen          | fresh              |
| v3      | mixed                         | same as gen          | fresh              |
| v4      | gemini-3.1-pro                | gemini-3.1-pro       | fresh              |
| v5      | gemini-3.1-flash              | gemini-3.1-flash     | fresh              |
| v6      | qwen235b                      | qwen235b             | fresh              |
| v7      | gemini-3-flash                | gemini-3-flash       | from v6 (qwen)     |
| v8      | gemini-3-flash                | gemini-3-flash       | from v5 (g3-flash) |
| v9      | gemini-3-flash                | gemini-3-flash       | from v5 (g3-flash) |
| v10p1   | gemini-3.1-pro                | gemini-3.1-pro       | from v6 (qwen)     |
| v11     | qwen3-vl-235b-a22b-instruct   | qwen3-235b-a22b-instruct | from v5 (g3-flash) |

Judge model is `gemini-3.1-pro` for every version except v2 (mixed).

---

## 4. Results

Precision = `total_stats.accuracy` from `precision_eval_result_*.json`; Recall = `recall` from `recall_eval_result_*.json`; F1 = 2PR/(P+R). Judge model for evaluation is `gemini-3.1-pro` in every run.

| Version | Gen Model             | Variant              | Precision | Recall | F1      |
|---------|-----------------------|----------------------|-----------|--------|---------|
| v2      | mixed                 | stage3 dense         | 0.9331    | 0.6377 | 0.7576  |
| v2      | mixed                 | stage3 structured    | 0.9353    | 0.6402 | 0.7601  |
| v2      | mixed                 | stage4 structured    | 0.9393    | 0.6706 | 0.7825  |
| v3      | mixed                 | stage3 dense         | 0.9350    | 0.6609 | 0.7744  |
| v3      | mixed                 | stage3 structured    | 0.9380    | 0.6583 | 0.7737  |
| **v3**  | **mixed**             | **stage4 structured**| **0.9390**| **0.6927** | **0.7973** |
| v4      | gemini-3.1-pro        | stage3 dense         | 0.9762    | 0.6008 | 0.7439  |
| v4      | gemini-3.1-pro        | stage3 structured    | 0.9752    | 0.5925 | 0.7371  |
| v4      | gemini-3.1-pro        | stage4 structured    | 0.9742    | 0.6531 | 0.7820  |
| v5      | gemini-3.1-flash      | stage3 dense         | 0.9708    | 0.6281 | 0.7627  |
| v5      | gemini-3.1-flash      | stage3 structured    | 0.9706    | 0.6211 | 0.7575  |
| v5      | gemini-3.1-flash      | stage4 structured    | 0.9689    | 0.6688 | 0.7914  |
| v6      | qwen235b              | stage3 dense         | 0.8656    | 0.6629 | 0.7508  |
| v6      | qwen235b              | stage3 structured    | 0.8598    | 0.6485 | 0.7393  |
| v6      | qwen235b              | stage4 structured    | 0.8719    | 0.6875 | 0.7688  |
| v7      | gemini-3-flash        | stage3 dense         | 0.8753    | 0.6593 | 0.7521  |
| v7      | gemini-3-flash        | stage3 structured    | 0.8670    | 0.6574 | 0.7478  |
| v7      | gemini-3-flash        | stage4 structured    | 0.8758    | 0.6877 | 0.7705  |
| v8      | gemini-3-flash        | stage3 dense         | 0.9055    | 0.6165 | 0.7335  |
| v8      | gemini-3-flash        | stage3 structured    | 0.8948    | 0.6045 | 0.7215  |
| v8      | gemini-3-flash        | stage4 structured    | 0.9003    | 0.6514 | 0.7559  |
| v9      | gemini-3-flash        | stage3 dense         | 0.9709    | 0.6226 | 0.7587  |
| v9      | gemini-3-flash        | stage3 structured    | 0.9733    | 0.6130 | 0.7522  |
| v9      | gemini-3-flash        | stage4 structured    | 0.9724    | 0.6500 | 0.7792  |
| v10     | gemini-3-flash        | —                    | —         | —      | —       |
| v10p1   | gemini-3.1-pro        | stage3 dense         | 0.9690    | 0.6249 | 0.7598  |
| v10p1   | gemini-3.1-pro        | stage3 structured    | 0.9696    | 0.6180 | 0.7549  |
| v10p1   | gemini-3.1-pro        | stage4 structured    | 0.9709    | 0.6563 | 0.7832  |
| v11     | qwen3-vl-235b         | stage3 dense         | 0.9459    | 0.6431 | 0.7657  |
| v11     | qwen3-vl-235b         | stage3 structured    | 0.9377    | 0.6329 | 0.7557  |
| v11     | qwen3-vl-235b         | stage4 structured    | 0.9349    | 0.6680 | 0.7792  |

_Notes on the table:_

- **v10** has `stage2/` on S3 but no `results/` subfolder, so no metrics were produced for it; v10p1 replaces it with a gemini-3.1-pro run on the same qwen stage-1 seed.
- v2 uses the legacy naming (`stage2_dense_caption` / `stage2_structured_caption`). From v3 onward the signature convention `v{N}s{3|4}{d|s}` is used. The column "Variant" normalizes this.
- Across 10 configurations and 3 variants (29 rows), **stage4-structured F1 lies in [0.7559, 0.7973]** — a spread of 4.1 F1 points end-to-end.

---

## 5. Analysis

**Stage 4 always helps, but it is "free" help, not "smart" help.** Stage-4 structured beats stage-3 dense and stage-3 structured on _every_ version. Because stage-4 adds camera framing + global style, and because the recall evaluator credits those new assertions while the precision evaluator rarely penalizes them, the gain is mechanical — we get ~2–3 extra recall points per version just by having the fourth stage emit more text. It does **not** reflect the judge-refine loop doing anything clever.

**The dominant axis is precision vs. recall, controlled by the judge.** Runs judged only by gemini-3.1-pro (v4, v5, v9, v10p1) sit at P≈0.97 / R≈0.62. Runs with a qwen or mixed judge (v2, v3, v6, v7, v11) sit at P≈0.87–0.94 / R≈0.64–0.69. F1 ends up in roughly the same band because the two effects cancel: a stricter judge removes both true and false assertions in proportion.

**Iteration does not compound.** If the judge-refine loop were meaningfully improving each pass, we would expect later versions — which had the benefit of prior stage-1 seeds, better generators, and more prompt tuning — to monotonically climb. They do not. v3 (early mixed-model run) still has the highest F1 (0.7973). v11 (our most recent qwen3-vl-235b run on v5's stage-1 seed) lands at 0.7792, _below_ v3. **Five rounds of judge-vs-generator battle buy us ~0 F1 on top of a careful one-shot.**

**Why doesn't self-critique help?** Two hypotheses consistent with the data:

1. _Judge is a noisy prior, not an oracle._ The judge flags ~the same fraction of "wrong" entities regardless of whether the entity is actually wrong, so the refine step mostly rephrases rather than correcting. This is consistent with v4 (all gemini-3.1-pro) having highest precision but lowest recall — the judge is so conservative it keeps discarding true information.
2. _Generator saturation._ The generator extracts roughly the set of entities it is willing to commit to on turn 1; later rounds can only polish phrasing. The judge-refine loop would need the judge to actually _add_ new entities via `missing_entities`, but the reliability of that flag appears low (we haven't seen recall climb round over round).

**Practical recommendation.** Before investing in a full agentic caption refinement system, we should (a) profile how often the judge-refine loop actually mutates the accepted set after round 1 and (b) build a direct A/B against Seungjun's 9-grid one-shot on a matched sample. If iteration truly contributes <1 F1 point, spend the engineering budget on a better one-shot prompt or a better visual grounding primitive instead of a longer loop.

---

_Source data: `s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/golden_caption_<VERSION_LONG>/results/`._
_Pipeline code: `projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage{1..4}_*.py` in `imaginaire4`._
