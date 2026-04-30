# rFID — execution progress

_Companion to [`rfid_design.md`](./rfid_design.md). This file tracks the rollout of the rFID protocol on a real prompt set: anchor generation, baseline sanity check, and Cosmos read-out._

---

## Plan at a glance

| Stage | What                                                                   | Status |
|-------|------------------------------------------------------------------------|--------|
| (a)   | Generate SetA / SetB from GPT-Image-2 on the 1170-line UnigenBranch prompt | ☐ todo |
| (b)   | Score existing baselines (NBP-Pro, Qwen, Flux, …) → baseline table     | ☐ todo |
| (c)   | Score Cosmos3 (pretrained, structure-caption-FT) → eyebrow check       | ☐ todo |

Encoder: **DINOv2** (default per design doc).
Score: `rFID-gen = FID(model, SetB) / FID(SetA, SetB)`.

---

## (a) Anchor generation — GPT-Image-2 on UnigenBranch prompts

**Prompt source.** UnigenBranch eval list, **n = 1170 prompts**. Used as-is, no filtering, in fixed order.

**Anchor build.** GPT-Image-2 (snapshot date logged in manifest). Two independent draws:

| Set  | Model         | Seed     | n     | Resolution | Notes                          |
|------|---------------|----------|-------|------------|--------------------------------|
| SetA | GPT-Image-2   | seed_A   | 1170  | TBD        | one-shot, no best-of-k         |
| SetB | GPT-Image-2   | seed_B   | 1170  | TBD        | same prompts, same order       |

**Sanity gates before we compute anything downstream:**
- [ ] All 1170 prompts produce a valid image in both sets (no API failures, no NSFW filter drops).
- [ ] Resolution / aspect ratio identical across SetA, SetB, and every model we will compare.
- [ ] DINOv2 features extracted and cached, keyed on `(set_id, prompt_idx, model_version, anchor_build)`.
- [ ] `FID_lb = FID(SetA, SetB)` computed and **logged with a bootstrap CI**. n=1170 is on the small side per the design doc — record the CI width so we know what gap is detectable above noise.

**Open / decisions needed before kickoff:**
- Resolution? (Default proposal: 1024×1024.)
- One seed pair (k=1) for now, or k=3 for a tighter floor at 3× the API cost?
- Cache anchor outputs to a versioned bucket so (b) and (c) reuse them — *do not regenerate per evaluation*.

---

## (b) Baseline table — are the scores reasonable?

For each baseline, generate 1170 images on the **same** prompts, **same** resolution, then compute rFID-gen against the cached SetB.

### Models in scope

- NBP-Pro
- Qwen-Image (latest)
- FLUX (variant TBD — `.1-dev`? `.1-pro`? whatever we currently host)
- _(extend list as needed)_

### Table to fill

| Model            | Build / version | n    | FID(model, SetB) | rFID-gen | CI (bootstrap) | Eyebrow?      |
|------------------|-----------------|------|------------------|----------|----------------|---------------|
| GPT-Image-2 (B') | seed_B'         | 1170 | (= FID_lb)       | 1.00     | ±…             | reference     |
| NBP-Pro          | …               | 1170 |                  |          |                |               |
| Qwen-Image       | …               | 1170 |                  |          |                |               |
| FLUX             | …               | 1170 |                  |          |                |               |
| …                | …               | 1170 |                  |          |                |               |

### What "reasonable" looks like

The point of this stage is to check that rFID-gen agrees with **prior knowledge of model quality**. Ordering we expect (rough, public-knowledge prior):

```
GPT-Image-2  <  FLUX-pro  ≲  Qwen-Image  ≲  NBP-Pro  ≲  older baselines
   (rFID≈1)        (low)        (mid)         (mid–high)    (high)
```

Pass criteria:
- [ ] Strict ordering matches the prior on **at least the headline pairs** (GPT-Image-2 < FLUX, FLUX < older SD-class models).
- [ ] No baseline lands *below* 1.0 (would imply it beats GPT-Image-2 against itself — almost certainly a bug).
- [ ] CI on rFID is narrower than the gap between adjacent rows; otherwise n=1170 is too small for this protocol and we need to scale up before trusting (c).

If ordering is wrong: probable causes to check, in order — resolution mismatch, prompt-order desync, encoder confound (try Inception-V3 as a tie-break), API filter dropping different prompts in different sets.

---

## (c) Cosmos3 — does the score match our gut feel?

Two checkpoints, same protocol:

| Run                              | Checkpoint                                  | n    | FID(model, SetB) | rFID-gen | CI | Notes |
|----------------------------------|---------------------------------------------|------|------------------|----------|----|-------|
| Cosmos3 — pretrained             | `<ckpt path>`                               | 1170 |                  |          |    |       |
| Cosmos3 — structure-caption FT   | `<ckpt path>` (golden-caption-stage4 FT)    | 1170 |                  |          |    |       |

### Eyebrow check — what we expect

- **Pretrained** should land somewhere between SoTA baselines and the older models. If it's worse than older SD-class models on rFID-gen, that contradicts qualitative impressions and is a signal to debug, not a signal to stop trusting rFID.
- **Structure-caption FT** should improve over pretrained on the *style* dimension (rFID-gen). If it makes rFID-gen worse but qualitatively looks better, that is the interesting finding — likely means we're trading reference-style for photorealism, which is exactly what rFID-real (next phase) is designed to disentangle.

Things to write down per run:
- [ ] rFID-gen with CI.
- [ ] Side-by-side qualitative grid (24 prompts, fixed indices) — does the rank order match human eyeball ranking?
- [ ] Per-slice breakdown if UnigenBranch has slice tags (photoreal / illustration / text-in-image / etc.). A single global rFID is less informative than 4–5 per-slice numbers.

---

## Risks / things that could derail this read

- **n = 1170 is small** for FID. The lower bound `FID_lb` may itself have a CI ≳ 10% of its value, which inflates rFID error bars. If the (b) ordering is noisy, scale to k=3 seed pairs or extend the prompt list before continuing to (c).
- **Anchor drift.** OpenAI may silently update GPT-Image-2 between (a) and (c). Lock the anchor: generate SetA + SetB once, freeze them on disk + a versioned bucket, and reuse across (b) and (c). Re-anchoring mid-experiment invalidates everything above.
- **Prompt-order desync.** All sets must share the prompt order from the UnigenBranch list verbatim. Trivial bug, ruinous result.
- **Resolution mismatch.** Cosmos3 native res may differ from GPT-Image-2 native res. Fix the resize/crop policy *before* generation, not after.
- **License on cached outputs.** GPT-Image-2 outputs cached for reuse — confirm internal-use is fine before committing them to a shared bucket.

---

## Next actions (ordered)

1. Lock resolution + seed-pair count (k=1 vs. k=3) → unblock (a).
2. Run (a), commit cached features + `FID_lb` with CI to a versioned manifest.
3. Run (b) baselines in parallel (one job per model).
4. Inspect (b) table; only proceed to (c) if pass criteria above are met.
5. Run (c) on both Cosmos3 checkpoints, write up the eyebrow read in the table here, and decide whether rFID earns a permanent dashboard slot.
