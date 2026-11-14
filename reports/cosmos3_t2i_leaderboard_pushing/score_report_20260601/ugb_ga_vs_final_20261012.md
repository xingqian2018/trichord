# UGB Summary: Cosmos3 GA vs. Pre-Cooldown vs. Final

Benchmark `v2_1170L_opus`, all 1170/1170 prompts scored successfully. Source: `score_report_ugb_20260601.md`.

| Model                          | Checkpoint                                                                         | All (1170L) | Orig (600L) | Phi (570Phi) |
|--------------------------------|------------------------------------------------------------------------------------|-------------|-------------|--------------|
| Cosmos3 GA                     | `cosmos3_ga_64bm32b_v3p1_midtrain` iter 900                                        | 86.23       | 88.16       | 84.46        |
| Cosmos3 (before SFT cooldown)  | `cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5` iter 25k    | 91.14       | 93.29       | 89.16        |
| Cosmos3 final                  | `cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5` iter 31k                      | 91.36       | 93.34       | 89.54        |
| Δ (final vs. GA)               |                                                                                    | +5.13       | +5.18       | +5.08        |

## Notes

- The pre-cooldown checkpoint is trained from the GA checkpoint on the union5 data mixture for 25k iterations (`lr=5e-5`), config `cosmos3_ga_64bm32b_t2ionly_exp000.py`.
- Final is the SFT (UHQ) cooldown from the pre-cooldown 25k checkpoint, `lr=1e-5`, iter 31k.
