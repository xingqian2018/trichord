# Relative FID (rFID): an anchor-normalized score for Cosmos image quality

**Authors:** [Xingqian Xu (US)](mailto:xingqianx@nvidia.com)
**Created:** Feb 26, 2026
**Status:** Work In Progress

| Reviewer | Last Updated | Status       |
|----------|--------------|--------------|
| Person   | Date         | Not Reviewed |

---

## Motivation

Raw FID numbers are already backlogged in our design due to being hard to read and judge. But tentatively, we can give a shot on the popular rFID, making a relative comparison between our model and SoTA model, on both heavy visual quality / aesthetic style images and real images.

---

## Design

Two anchored scores, same protocol, different anchors.

### rFID-gen — anchor on the best generator

Apply the best SoTA image model today: **GPT-Image-2**.

```
prompts P (shared)
   │
   ├──► GPT-Image-2, seed_A ──► SetA   (n images)
   ├──► GPT-Image-2, seed_B ──► SetB   (n images)
   └──► Cosmos,     seed_C ──► SetC   (n images)

FID_lb   = FID( SetA , SetB )     # intra-model floor: same model, different seed
FID_ours = FID( SetC , SetB )     # Cosmos vs. reference
rFID-gen = FID_ours / FID_lb
```

**Interpretation.** How close Cosmos is to being statistically indistinguishable from GPT-Image-2 under a given encoder.

- **rFID-gen → 1**: Cosmos is statistically indistinguishable from GPT-Image-2 — our ultimate goal.
- **rFID-gen > 1**: headroom for improvement.

### rFID-real — anchor on real data

Replace SetA & SetB with real photographs of similar semantics, i.e. maintain a similar or identical distribution. Default recipe:

1. Curate a real-photo set R (photoreal-skewed, license-clean).
2. Extract semantics (caption / tags) from each photo in R via the golden-caption pipeline.
3. Regenerate **SetB** from those captions with GPT-Image-2.
4. Generate **SetC** from the same captions with Cosmos.

```
real photos R (curated, photorealism-skewed)
   │
   ├──► R itself ─────────────────────────► SetA  (real)
   ├──► caption(R) ► GPT-Image-2, seed_B ─► SetB  (synthetic of same semantics)
   └──► caption(R) ► Cosmos,     seed_C ─► SetC

FID_lbd   = FID( SetA , SetB )    # real-vs-best-model floor for these semantics
FID_ours  = FID( SetC , SetB )
rFID-real = FID_ours / FID_lbd
```

---

## rFID improves over regular FID

| Property                              | Raw FID | rFID                       |
|---------------------------------------|---------|----------------------------|
| Has a meaningful "perfect" value      | no      | yes (= 1.0)                |
| Stable across prompt-set sizes        | no      | yes (once floor converges) |
| Comparable across encoder swaps       | no      | partially                  |
| Captures "how close to GPT-Image-2"   | no      | yes (rFID-gen)             |
| Separates style gap vs. realism gap   | no      | yes (2 variants)           |

---

## Encoder

Default to **DINOv3** (or **SigLIP2** when stable). Reasons:

- The underlying assumption of a good encoder is a strong semantic extractor with strong spatial detail awareness.
- Old Inception-V3 features are tied to ImageNet-1k classes — known to be a poor proxy for modern generative quality, especially for non-object scenes.
- DINOv2/v3 is self-supervised, semantically rich, and the de-facto choice in recent generative-eval papers.
- SigLIP2 serves a similar purpose.

Keep in mind that different choices of encoder may yield different scores, but the ranking is expected to be stable (a good model should be good across all aspects).
