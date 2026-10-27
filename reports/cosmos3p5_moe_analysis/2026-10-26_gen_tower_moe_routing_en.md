# Cosmos3.5 60B/30B-A3B — Generation-Tower MoE Routing on UGB

Routing statistics of every generation-tower (`mlp_moe_gen`) MoE layer while the pre-SFT Cosmos3.5 base checkpoint generates all 1170 UniGenBench prompts. The goal is to understand how much the generation-tower router actually routes, as input to the exp004 ablations (generation tower GQA → MHA, MoE → dense with the same activated width).

## Summary

1. **Load concentration is a property of the model, not of the prompt sample.** Per-layer effective expert counts from 10 prompts and from all 1170 prompts agree with correlation 0.996 across layers. Layers 0–6 and 25–47 stay close to balanced (104–123 effective experts of 128); layers 7–24 are concentrated (38–96). Layer 7 is extreme: 6 experts carry half of the load and 50 experts get less than 10% of a fair share.
2. **The router scores barely discriminate between experts.** The mean per-token router entropy is 4.74–4.85 nats in every layer against a uniform maximum of ln 128 = 4.85, and the router probability mass per expert is uncorrelated with the dispatched load (−0.29 to +0.13). The experts a token is dispatched to are therefore decided mostly by the aux-loss-free selection bias (`expert_bias`) plus small score differences, not by a confident router.
3. **Layers 0–24 route the same way for every image; layers 26–47 route by content.** In layers 6–19 each image's expert-load distribution is almost identical to the 1170-image aggregate (median JS divergence 0.002–0.008, 7–8 of the image's 8 heaviest experts are the aggregate's 8 heaviest). In layers 26–47 images differ (JS 0.017–0.049, only 3–4 of 8 heaviest experts shared, per-image effective experts 8–21 below the aggregate).
4. **For exp004_002 (MoE → dense):** replacing the MoE with a fixed dense FFN loses little where routing is input-independent (layers 0–24) and mostly removes the content-dependent routing of layers 26–47.

## Setup

| | |
|---|---|
| Checkpoint | `s3://nv-00-10206-checkpoint/cosmos3_generator/cosmos3plus_t2ionly/3p5_production_runs/60bm30b_v2p5_pretrain/checkpoints/iter_000741260_gen_mha32/` — `60bm30b_v2p5_pretrain/iter_000741260` with the generation-tower K/V projections expanded from 4 to 32 heads. The expansion is exact (UGB 88.21 vs 88.24 on the GQA original), and the MoE weights, router and `expert_bias` are untouched, so the routing is that of the GQA checkpoint. |
| Experiment config | `cosmos3p5_ga_60bm30b_base_gen_mha32` (= `cosmos3p5_ga_60bm30b_base` + `gen_num_key_value_heads=32`) |
| Generation tower MoE | 48 layers, 128 routed experts (width 768), top-8, plus one always-on shared expert (width 768); sigmoid router with aux-loss-free load balancing (selection on score + `expert_bias`, combine weights from the normalized scores) |
| Prompts | All 1170 prompts of `v2_1170L_gpt5p5_cosmos3p5_dense_hamid`, in file order, seed = prompt index |
| Sampling | Same generator as `inference_ugb_i4.py` (`Cosmos3p5Generator`, compiled): 960×960, guidance 4.0, 50 steps, EMA weights, no negative prompt, CFG on every step |
| Tokens counted | 900 image tokens × 50 steps × 2 CFG branches per image = 105.3 M tokens per layer (padding excluded) |
| Job | gcpcode Slurm 2668794, 8 nodes × 4 GB200 (prompts dealt round-robin across 32 ranks) |

The statistics are the ones every `Qwen3VLMoeTextSparseMoeBlock` already accumulates in `_update_moe_callback_stats`; no model code was changed. They are read and zeroed after each prompt.

- **Load** of an expert: tokens dispatched to it, i.e. the actual (score + `expert_bias`) top-8 selection.
- **Router probability**: the normalized router scores before the bias; **token entropy** is the entropy of that 128-way distribution per token.

Raw statistics: `gcpcode:~/tmp/gen_moe_calib/gen_moe_routing_stats_iter741260_mha32_ugb1170.pt` (1170 prompts, per-layer totals plus per-prompt load) and `..._ugb10.pt` (first 10 prompts).

## Metrics

| Column | Definition |
|---|---|
| Eff. experts | exp(entropy) of the layer's load distribution over 128 experts; 128 = perfectly balanced |
| Top-8 load share | share of tokens dispatched to the 8 most loaded experts (balanced: 0.0625) |
| Experts for 50% load | fewest experts that together receive half of the dispatched tokens (balanced: 64) |
| Experts < 0.1x / > 2x mean | experts whose load is below 10% / above 200% of the per-expert mean |
| Mean token entropy | mean over tokens of the router-probability entropy (nats); ln 128 = 4.85 is uniform |
| Prob-load corr | Pearson correlation over experts between summed router probability and dispatched load |
| Per-image JS | median over the 1170 images of the JS divergence (nats, max ln 2 = 0.69) between the image's load distribution and the 1170-image aggregate |
| Per-image top-8 overlap | median number of the image's 8 most loaded experts that are among the aggregate's 8 most loaded |
| Per-image eff. experts | median over images of the effective expert count within a single image |

## Results

| Layer | Eff. experts | Eff. experts (10 prompts) | Top-8 load share | Experts for 50% load | Experts < 0.1x mean | Experts > 2x mean | Mean token entropy | Prob-load corr | Per-image JS | Per-image top-8 overlap | Per-image eff. experts |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 121.9 | 121.8 | 0.112 | 49 | 0 | 2 | 4.85 | +0.01 | 0.002 | 6 | 120.8 |
| 1 | 119.2 | 119.2 | 0.119 | 46 | 0 | 2 | 4.85 | +0.10 | 0.002 | 6 | 118.3 |
| 2 | 122.6 | 122.6 | 0.113 | 51 | 0 | 2 | 4.85 | +0.13 | 0.002 | 6 | 121.7 |
| 3 | 121.9 | 121.7 | 0.101 | 50 | 0 | 1 | 4.85 | +0.03 | 0.002 | 4 | 120.9 |
| 4 | 116.2 | 116.2 | 0.122 | 46 | 2 | 1 | 4.85 | -0.08 | 0.002 | 5 | 115.2 |
| 5 | 116.7 | 116.5 | 0.124 | 44 | 0 | 3 | 4.85 | +0.13 | 0.002 | 6 | 115.8 |
| 6 | 106.9 | 106.6 | 0.186 | 38 | 2 | 8 | 4.84 | +0.00 | 0.002 | 8 | 105.8 |
| **7** | **37.7** | 37.9 | **0.546** | **6** | **50** | 9 | 4.84 | +0.01 | 0.003 | 8 | 37.3 |
| 8 | 87.2 | 86.2 | 0.276 | 28 | 4 | 9 | 4.84 | -0.02 | 0.004 | 8 | 85.7 |
| 9 | 74.7 | 75.1 | 0.307 | 22 | 22 | 12 | 4.84 | -0.01 | 0.004 | 7 | 73.5 |
| 10 | 73.1 | 72.5 | 0.332 | 22 | 14 | 10 | 4.83 | -0.01 | 0.005 | 7 | 71.5 |
| 11 | 76.4 | 75.7 | 0.326 | 23 | 14 | 10 | 4.82 | +0.03 | 0.006 | 7 | 74.5 |
| 12 | 74.9 | 74.0 | 0.282 | 22 | 22 | 13 | 4.83 | +0.02 | 0.005 | 7 | 73.4 |
| 13 | 67.9 | 67.2 | 0.362 | 19 | 22 | 10 | 4.82 | +0.02 | 0.006 | 8 | 66.1 |
| 14 | 82.3 | 82.4 | 0.277 | 25 | 13 | 11 | 4.83 | +0.02 | 0.006 | 7 | 80.2 |
| 15 | 77.4 | 76.5 | 0.290 | 21 | 12 | 14 | 4.83 | +0.04 | 0.006 | 7 | 75.3 |
| 16 | 80.5 | 80.7 | 0.291 | 24 | 14 | 10 | 4.81 | +0.01 | 0.008 | 8 | 77.8 |
| 17 | 80.1 | 80.2 | 0.285 | 23 | 12 | 14 | 4.82 | +0.00 | 0.008 | 7 | 77.7 |
| **18** | **55.9** | 55.8 | **0.412** | **14** | **32** | 9 | 4.82 | +0.01 | 0.008 | 7 | 54.0 |
| 19 | 81.2 | 80.2 | 0.257 | 25 | 23 | 14 | 4.82 | +0.02 | 0.008 | 7 | 78.2 |
| 20 | 87.5 | 86.0 | 0.232 | 28 | 16 | 13 | 4.81 | -0.11 | 0.012 | 6 | 83.1 |
| 21 | 84.3 | 83.0 | 0.245 | 28 | 12 | 10 | 4.80 | -0.07 | 0.010 | 6 | 80.7 |
| 22 | 75.9 | 75.6 | 0.304 | 24 | 20 | 10 | 4.79 | -0.03 | 0.010 | 6 | 72.9 |
| 23 | 93.8 | 92.0 | 0.231 | 32 | 7 | 7 | 4.78 | -0.02 | 0.013 | 6 | 88.9 |
| 24 | 95.9 | 94.8 | 0.214 | 32 | 9 | 9 | 4.77 | +0.00 | 0.014 | 6 | 90.6 |
| 25 | 104.5 | 102.4 | 0.171 | 37 | 9 | 10 | 4.77 | -0.06 | 0.016 | 6 | 97.8 |
| 26 | 112.7 | 109.6 | 0.142 | 43 | 2 | 3 | 4.76 | -0.09 | 0.019 | 4 | 104.3 |
| 27 | 119.5 | 117.1 | 0.118 | 47 | 0 | 2 | 4.77 | -0.19 | 0.017 | 3 | 111.1 |
| 28 | 119.0 | 114.5 | 0.124 | 46 | 0 | 3 | 4.77 | -0.14 | 0.029 | 3 | 105.6 |
| 29 | 117.8 | 109.1 | 0.117 | 45 | 0 | 1 | 4.76 | -0.10 | 0.026 | 3 | 106.2 |
| 30 | 116.4 | 107.8 | 0.133 | 44 | 0 | 4 | 4.76 | -0.10 | **0.049** | 3 | 95.5 |
| 31 | 117.8 | 112.0 | 0.126 | 45 | 2 | 5 | 4.75 | -0.14 | 0.039 | 3 | 100.1 |
| 32 | 117.6 | 115.3 | 0.124 | 46 | 0 | 3 | 4.75 | -0.20 | 0.030 | 3 | 103.9 |
| 33 | 115.9 | 113.3 | 0.123 | 45 | 0 | 2 | 4.76 | -0.19 | 0.022 | 3 | 105.7 |
| 34 | 117.8 | 115.3 | 0.119 | 46 | 2 | 4 | 4.74 | -0.20 | 0.021 | 3 | 108.0 |
| 35 | 117.0 | 113.8 | 0.122 | 46 | 1 | 1 | 4.74 | -0.19 | 0.020 | 3 | 107.9 |
| 36 | 117.8 | 115.0 | 0.125 | 46 | 0 | 3 | 4.74 | -0.18 | 0.019 | 4 | 109.1 |
| 37 | 118.4 | 115.0 | 0.126 | 46 | 0 | 3 | 4.75 | -0.18 | 0.018 | 4 | 109.7 |
| 38 | 113.3 | 109.7 | 0.133 | 43 | 2 | 5 | 4.77 | -0.08 | 0.020 | 4 | 104.3 |
| 39 | 104.3 | 101.5 | 0.153 | 38 | 3 | 8 | 4.79 | +0.10 | 0.019 | 4 | 96.4 |
| 40 | 109.9 | 108.2 | 0.133 | 41 | 1 | 4 | 4.79 | -0.08 | 0.019 | 4 | 101.7 |
| 41 | 104.9 | 101.8 | 0.142 | 36 | 4 | 10 | 4.79 | -0.07 | 0.019 | 3 | 96.6 |
| 42 | 109.2 | 107.5 | 0.152 | 39 | 0 | 6 | 4.80 | -0.09 | 0.020 | 4 | 100.3 |
| 43 | 111.8 | 108.7 | 0.142 | 40 | 1 | 4 | 4.80 | -0.29 | 0.025 | 3 | 100.7 |
| 44 | 117.3 | 114.0 | 0.119 | 45 | 1 | 3 | 4.80 | -0.20 | 0.021 | 3 | 107.6 |
| 45 | 111.7 | 108.8 | 0.146 | 41 | 1 | 6 | 4.80 | -0.18 | 0.025 | 4 | 100.9 |
| 46 | 103.9 | 101.6 | 0.158 | 35 | 3 | 8 | 4.81 | -0.25 | 0.028 | 3 | 92.5 |
| 47 | 111.8 | 111.2 | 0.132 | 40 | 1 | 6 | 4.82 | -0.03 | 0.022 | 3 | 101.8 |

Across layers: corr(eff. experts, 1170 vs 10 prompts) = 0.996; mean token entropy 4.80 nats (min 4.74, layer 35).

### By depth

| Layers | Eff. experts | Top-8 load share | Experts < 0.1x mean | Per-image JS | Per-image top-8 overlap | Reading |
|---|---|---|---|---|---|---|
| 0–5 | 116–123 | 0.10–0.12 | 0–2 | 0.002 | 4–6 ¹ | balanced, image-independent |
| 6–19 | 38–107 | 0.19–0.55 | 2–50 | 0.002–0.008 | 7–8 | concentrated on a fixed expert subset, image-independent |
| 20–25 | 84–105 | 0.17–0.30 | 7–20 | 0.010–0.016 | 6 | moderately concentrated, mostly image-independent |
| 26–47 | 104–120 | 0.12–0.16 | 0–4 | 0.017–0.049 | 3–4 | balanced in aggregate, routing varies with image content |

¹ When the load is near-uniform the 8 heaviest experts are close to ties, so the overlap is not informative in layers 0–5; the JS divergence is.

## Findings

### 1. Concentration in layers 7–24 is stable across prompts

Ten prompts already reproduce the 1170-prompt picture (layer 7: 37.9 vs 37.7 effective experts; layer 18: 55.8 vs 55.9), and in layers 6–19 a single image has nearly the same load distribution as the whole benchmark. Within this text-to-image workload the concentration is a fixed property of those layers.

### 2. Router probabilities are close to uniform for every token

A per-token entropy of 4.74–4.85 nats against a maximum of 4.85 means each token's normalized router scores are nearly flat over 128 experts — not only their sum over tokens. The router probability mass of an expert does not predict how many tokens it receives (correlation −0.29 to +0.13; mostly slightly negative in layers 26–46). The dispatched top-8 is chosen on score + `expert_bias`; with nearly flat scores, the bias and small score differences decide it.

This is an interpretation from the statistics; the `expert_bias` magnitudes themselves were not inspected (see Next steps).

### 3. Content-dependent routing exists only in the later layers

Layers 26–47 are balanced in aggregate but not per image: a single image uses fewer experts than the benchmark as a whole (e.g. layer 30: 95.5 vs 116.4) and different images favor different experts (only 3 of the image's 8 heaviest experts are the aggregate's). This is the only depth range where the router behaves like a content router.

## Implications for exp004

| Arm | Generation tower | Relevance of these findings |
|---|---|---|
| exp004_000 | GQA + MoE (control) | — |
| exp004_001 | MHA + MoE | routing unchanged at initialization (MHA conversion is exact) |
| exp004_002 | MHA + dense, width 6912 = 8 × 768 routed + 768 shared, experts merged by contiguous groups of 16 | layers 0–24 route nearly independently of the input, so a fixed dense FFN loses little there; layers 26–47 lose content-dependent routing |

- A calibrated merge (group experts by co-activation, weight members and groups by dispatched load) is most meaningful in layers 7–24, where the load is concentrated on a stable subset. In the balanced layers any grouping amounts to averaging many experts. On the 10-prompt statistics, co-activation grouping raises the within-group share of pairwise co-activation affinity from 0.123 (contiguous groups, ≈ random 0.118) to 0.201, a modest gain because experts co-activate fairly independently.
- With load-proportional group scales, some groups in the concentrated layers start almost switched off (layer 18: scales 0.002–0.53 on the 10-prompt statistics). Whether to floor the group scales is open.

## Caveats

- One workload: text-to-image, 960×960, UGB prompts. Video, other resolutions and the training data distribution were not measured.
- All 50 denoising steps and both CFG branches are summed. The image-independent routing of layers 0–24 may still depend on the timestep; that cannot be separated here.
- EMA weights of the pre-SFT checkpoint only. SFT checkpoints (exp003 / exp004) may route differently; `expert_bias` is frozen during inference and keeps its checkpoint value.
- The causal attribution of finding 2 to `expert_bias` is inferred, not measured.

## Next steps

1. Read the per-layer `expert_bias` from the checkpoint and compare its spread with the router-logit spread, to confirm that the bias decides the dispatch.
2. Split the statistics by denoising step to test whether the routing of layers 0–24 depends on the timestep.
3. Re-run the same measurement on exp004_000 / exp004_001 checkpoints after training, to see whether SFT changes the routing.
4. Build the calibrated dense checkpoint (`iter_000741260_gen_mha32_dense8_calib`) from the 1170-prompt statistics and compare it with exp004_002's contiguous merge.

## Reproduce

- Collection: `projects/cosmos3/cosmos3/scripts/calibrate_gen_moe_routing.py` (imaginaire4_alt, branch `xingqianx/cosmos3_scenetext_run`), launched with `--num_prompts 1170 --height 960 --width 960 --guidance 4.0 --num_inference_steps 50 --use_ema` on 8 nodes.
- Tables: `python routing_tables.py gen_moe_routing_stats_iter741260_mha32_ugb1170.pt gen_moe_routing_stats_iter741260_mha32_ugb10.pt` (this folder).
