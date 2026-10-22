## Getty data ablation

**Question.** Does adding Getty images to the text-to-image training mix improve prompt following on UniGenBench?

**Setup.**
- Start: Cosmos3.5 60B/30B-A3B pretrain checkpoint `60bm30b_v2p5_pretrain/iter_000741260`.
- Training: 25k steps on the pretrain image mix with Getty added at 10% of samples. Getty = `getty_image_350k_test.lance`, 350k images at ~2K resolution with dense captions.
- Eval: UniGenBench v2, 1170 prompts. Images rendered from the official gpt55-img-a2 upsampled prompts, scored against the original prompts and testpoints with gemini-3.1-pro. 960×960, guidance 4, 50 steps, shift 6, no negative prompt.

**Results** (UniGenBench accuracy, %):

| Checkpoint | all | orig | phi |
|---|---|---|---|
| Pretrain | 87.21 | 89.18 | 85.40 |
| + Getty | **87.88** | **90.13** | **85.82** |
| Δ | +0.67 | +0.95 | +0.42 |

By primary dimension:

| Dimension | Pretrain | + Getty | Δ |
|---|---|---|---|
| Logical Reasoning | 76.77 | 79.35 | +2.58 |
| Style | 97.90 | 99.52 | +1.62 |
| World Knowledge | 91.13 | 92.35 | +1.22 |
| Relationship | 87.93 | 89.12 | +1.19 |
| Action | 73.70 | 74.62 | +0.92 |
| Attribute | 95.14 | 95.80 | +0.66 |
| Compound | 87.20 | 87.44 | +0.24 |
| Entity Layout | 93.51 | 93.24 | −0.27 |
| Grammar | 91.23 | 90.35 | −0.88 |
| Text Generation | 15.38 | 14.42 | −0.96 |

**Takeaways.**
- Adding Getty raises UniGenBench from 87.21 to 87.88 (+0.67), improving both the original prompts (+0.95) and the physical-AI prompts (+0.42).
- The gains concentrate in Logical Reasoning (+2.58), Style (+1.62), World Knowledge (+1.22) and Relationship (+1.19).
- Text Generation does not improve (−0.96).
