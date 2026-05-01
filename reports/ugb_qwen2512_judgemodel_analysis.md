# UGB Judge-Model Comparison: gemini-3-flash vs gemini-3.1-pro

**Subject:** `xingqianx_qwen_image_2512` on UniGenBench v2 (1170 prompts, 8193 testpoints)
**Source files** (`gcs:nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/unigenbench/v2_1170L_G3F/xingqianx_qwen_image_2512/`):
- `unigenbench_result.json` — judge = `gemini-3-flash`
- `unigenbench_result_gemini-3p1-pro.json` — judge = `gemini-3.1-pro`

## Headline numbers

| Split | flash acc | pro acc | Δ (flash − pro) |
|---|---|---|---|
| orig (3920 pts) | 0.9418 | 0.8753 | +0.0665 |
| phi  (4273 pts) | 0.9179 | 0.8147 | +0.1032 |
| **all (8193 pts)** | **0.9293** | **0.8436** | **+0.0857** |

The same image set, same testpoints, same generations — only the judge changed.

## Where the gap comes from

A pairwise testpoint comparison (flash vs pro on the *same* 8193 testpoints):

| Outcome | Count | Share |
|---|---|---|
| Both judges PASS | 6828 | 83.3% |
| Both judges FAIL |  495 |  6.0% |
| flash PASS, pro FAIL (pro stricter) | **786** | **9.6%** |
| pro PASS, flash FAIL (flash stricter) | 84 | 1.0% |
| Judge agreement | — | 89.4% |

Pro flips ~9.6× as many verdicts to FAIL as flash does. Pro is strictly lower than flash in **all 27/27** small categories — this is a uniform stringency shift, not a regression in any one class.

## Per-category breakdown (where pro is harshest)

Net = (#flash-pass-pro-fail) − (#pro-pass-flash-fail), i.e. flips lost when switching to pro.

| Category          |    n | flash-only | pro-only |  net |  net/n |
|-------------------|-----:|-----------:|---------:|-----:|-------:|
| Action            | 1525 |        278 |       33 | +245 | +16.1% |
| Compound          |  414 |         59 |        2 |  +57 | +13.8% |
| Text Generation   |  208 |         32 |        5 |  +27 | +13.0% |
| Relationship      | 1011 |        127 |        4 | +123 | +12.2% |
| Logical Reasoning |  155 |         17 |        2 |  +15 |  +9.7% |
| Entity Layout     | 1110 |        108 |        8 | +100 |  +9.0% |
| Grammar           |  228 |         22 |        4 |  +18 |  +7.9% |
| World Knowledge   |  327 |         22 |        3 |  +19 |  +5.8% |
| Attribute         | 2595 |        100 |       18 |  +82 |  +3.2% |
| Style             |  620 |         21 |        5 |  +16 |  +2.6% |

The hardest-hit small classes:
- `Action - Full-body (Character/Anthropomorphic)`: 0.858 → 0.563 (−29.6 pp)
- `Action - Hand (Character/Anthropomorphic)`: 0.876 → 0.670 (−20.5 pp)
- `Relationship - Similarity`: 0.895 → 0.721 (−17.4 pp)
- `Action - Contact Interaction`: 0.825 → 0.667 (−15.8 pp)
- `Grammar - Consistency`: 0.879 → 0.727 (−15.2 pp)

## Why pro is stricter (qualitative pattern)

Same image, same testpoint — pro reads the prompt more literally and refuses softer matches. Concrete examples from the breakdowns:

1. **`orig126` · Action - Contact Interaction** ("eagle gripping rocky peak")
   - flash: *"talons positioned firmly on the jagged rock, claws appearing to grip the bedrock surface."* → PASS
   - pro:   *"talons visible resting on and gripping the surface, but not deeply embedded or sunken into the bedrock."* → FAIL

2. **`orig143` · Action - Contact Interaction** ("neon tube wrapped around zipper")
   - flash: *"tube crosses over the jacket and appears to pass through or wrap around the central zipper."* → PASS
   - pro:   *"tube forms an 'X' across the chest passing through the zipper pull, but is not tightly wrapped around the length of the zipper."* → FAIL

3. **`orig146` · Action - State** ("flickering light tubes")
   - flash: *"uneven illumination and dimmer/partially dark tubes suggest a state of unstable flickering."* → PASS
   - pro:   *"Flickering is a dynamic, temporal state that cannot be observed in a single static image."* → FAIL

The pattern across the 786 flash-pass / pro-fail cases:
- Pro insists on the **exact** physical/spatial configuration in the prompt (deeply embedded vs resting, tightly wrapped along length vs crossing over).
- Pro rejects **temporal/dynamic** properties (flickering, motion blur, "is moving") on the principle that they can't be verified from a still.
- Pro is harsher on **anthropomorphic full-body / hand poses** — the largest single bucket of disagreement, where flash credits "approximately the right pose" and pro requires every joint/contact to match.
- Pro is harsher on **compositional/relationship** prompts that ask for several entities to interact; flash credits the gist, pro flags any sub-clause that's off.

## Takeaway

The 8.6 pp gap is **almost entirely a judge-stringency artifact**, not a real model-quality signal:
- 89.4% of testpoint verdicts agree.
- Of the 10.6% that disagree, 9.6% are pro-stricter and only 1.0% flash-stricter — a 9.4× asymmetry.
- The disagreement concentrates in classes where the prompt-to-image grounding is fuzzy by nature (action poses, compositional relationships, dynamic states, text rendering).

Practical implications:
- For **absolute scores**, gemini-3.1-pro will report systematically lower numbers (~5–10 pp on UGB v2 1170L). Don't compare a flash-judged number with a pro-judged number — they are on different scales.
- For **model A vs model B comparisons**, pick one judge and use it consistently. Pro's higher stringency may give better resolving power on Action / Compound / Relationship classes where flash saturates near ceiling.
- If using pro going forward, expect Action and Compound classes to dominate the headline accuracy movement, since they have both the largest n and the largest stringency delta.

---
*Generated 2026-04-30. Raw files cached at `/tmp/ugb_compare/{flash,pro}.json`.*
