# CVTG Score Report — NFT (64bm32b)

LoRA checkpoints use `cosmos3_ga_64bm32b_t2ionly_base_mini_lora_rank32` with no `--use_ema`; the baseline uses `cosmos3_ga_64bm32b_t2ionly_base_mini` with `--use_ema`.

## cvtg500L_opus (canonical, matches score_report_20260601 recipe)

Gen: `--benchmark_name cvtg500L_opus`, 1024x1024, guidance 4.0, 50 steps, neg prompt on, shift 3.0 (hard-coded). Score: `--benchmark_name cvtg500L_opus --force_resize 960x960 --max_retry 5 --image_extension webp`, judge `gemini-3.1-pro@nvidia|@nvidiak`. Output under `s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/cvtg_for_nft/cvtg500L_opus/<run>`.

| Run                                                                            | Images   | gned       | pned       | success   |
|--------------------------------------------------------------------------------|----------|------------|------------|-----------|
| June-01 ref: cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k        | 500      | 0.8088     | 0.873      | 500/500   |
| June-01 ref: cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter30k        | 500      | 0.8378     | 0.8908     | 500/500   |
| baseline (exp009_sft0 iter31k, base_mini, --use_ema, regenerated 2026-09-16)   | 500      | 0.8005     | 0.8743     | 500/500   |
| cosmos3plus_64bm32b_t2ionly_nft_exp007_000_paddleocr_n24_iter50                | 500      | 0.8626     | 0.9010     | 500/500   |
| cosmos3plus_64bm32b_t2ionly_nft_exp007_000_paddleocr_n24_iter80                | 500      | 0.8546     | 0.8935     | 500/500   |
| cosmos3plus_64bm32b_t2ionly_nft_exp007_000_paddleocr_n24_iter90                | 500      | 0.8496     | 0.8902     | 500/500   |
| cosmos3plus_64bm32b_t2ionly_nft_exp007_002_paddleocr_oldmixdefault_n24_iter50  | 500      | 0.8373     | 0.8940     | 500/500   |
| cosmos3plus_64bm32b_t2ionly_nft_exp007_002_paddleocr_oldmixdefault_n24_iter120 | 500      | 0.8421     | 0.8948     | 500/500   |
| cosmos3plus_64bm32b_t2ionly_nft_exp007_003_paddleocr_pnedgned_n24_iter50       | 500      | **0.8812** | 0.9352     | 500/500   |
| cosmos3plus_64bm32b_t2ionly_nft_exp007_003_paddleocr_pnedgned_n24_iter90       | 500      | 0.8587     | **0.9524** | 499/500   |
| cosmos3plus_64bm32b_t2ionly_nft_exp007_003_paddleocr_pnedgned_n24_iter110      | 500      | 0.8523     | 0.9354     | 500/500   |

Notes (2026-09-16, cvtg500L_opus, scored at 960x960):
- Regenerated baseline (0.8005 / 0.8743) reproduces the June-01 iter31k reference (0.8088 / 0.8730) within 0.01 — recipe validated. Earlier gc-set numbers were on the wrong prompt set and are deleted (GCS + local).
- exp007_000 (PaddleOCRv5 reward, LoRA rank32): iter50 **0.8626 / 0.9010** (+0.062 / +0.027 vs baseline), iter80 0.8546 / 0.8935, iter90 0.8496 / 0.8902. Gain is largest at iter50 and tapers; zero samples with unreadable text at any of the three. UGB at the same checkpoints stays at 89–90 (baseline 91.35), so the text gain costs ~1–2 UGB points.
- iter100 and iter150 are collapsed (see ugb.md) and intentionally not scored here. Best usable checkpoint of this run: iter50.
- exp007_002 (same as exp007_000 but net_old_mixing_type=default) iter120: gned 0.8421 / pned 0.8948 (+0.042 / +0.021 vs baseline), 1/500 empty-text sample. Healthy 20+ steps past where exp007_000 collapsed; UGB 89.82 (see ugb.md). Text gain slightly below exp007_000 iter50 (0.8626) but the run is stable.
- exp007_002 iter50: gned 0.8373 / pned 0.8940 (+0.037 / +0.020 vs baseline), 0/500 empty-text. Default old-policy mixing gains text accuracy more slowly than exp007_000 (0.8626 at iter50) but keeps improving to iter120 (0.8421) instead of collapsing.
- Backup (2026-09-18): exp007_000 iter50 (best text gain, UGB 90.28) copied to `gcs:nv-00-10206-checkpoint/cosmos3_generator/cosmos3plus_t2ionly/<run>/checkpoints/iter_000000050/` (full DCP: model + optim + scheduler + trainer).
- exp007_003 (pned+gned reward, synthetic scene-text prompts): iter50 **0.8812 / 0.9352** (+0.081 / +0.061 vs baseline, best gned in table), iter90 0.8587 / **0.9524** (best pned), iter110 0.8523 / 0.9354. Beats every exp007_000/exp007_002 checkpoint on both metrics at every iteration; gned peaks at 50 and eases ~0.03 by 110 while pned stays ≥0.935. UGB cost: −1.4 (50), −2.5 (90), −4.3 (110) vs baseline. Recommended: iter50 for max exact-match text with least adherence cost; iter90 if partial-match matters more.
