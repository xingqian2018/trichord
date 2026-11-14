# UGB Score Report — SDG Ablation

Gen: `v2_1170L_opus`, 50 steps, neg prompt on, 640×640. Scored on `v2_1170L` with `gemini-3.1-pro`.


<!-- ## cosmos3plus_64bm32b_t2ionly_exp002 (iter 5k)

| Run                                                    | Benchmark     | Images   | all   | orig   | phi   | success   |
|--------------------------------------------------------|---------------|----------|-------|--------|-------|-----------|
| cosmos3plus_64bm32b_t2ionly_exp002_000_pretrain_iter5k | v2_1170L_opus | — png    | —     | —      | —     | —/1170    |
| cosmos3plus_64bm32b_t2ionly_exp002_001_pretrain_sdg_iter5k | v2_1170L_opus | — png    | —     | —     | —     | —/1170  | -->


## cosmos3p5_ga_60bm30b_t2ionly_moe_exp000 (iter 97k / 100k, scored with gemini-3.1-pro@nvidia)

| Run                                                                          | Benchmark     | Images   | all       | orig      | phi       | success   |
|------------------------------------------------------------------------------|---------------|----------|-----------|-----------|-----------|-----------|
| cosmos3p5_ga_60bm30b_t2ionly_moe_exp000_000_pretrain_iter100k                | v2_1170L_opus | 1170 png | 71.0      | 73.0      | 68.0      | 1170/1170 |
| cosmos3p5_ga_60bm30b_t2ionly_moe_exp000_001_pretrain_sdg_iter97k             | v2_1170L_opus | 1170 png | **75.0**  | **77.0**  | **73.0**  | 1170/1170 |
| cosmos3p5_ga_60bm30b_t2ionly_moe_exp000_001_pretrain_sdg_iter100k            | v2_1170L_opus | 1170 png | 74.26     | 76.02     | 72.64     | 1170/1170 |
| cosmos3p5_ga_60bm30b_t2ionly_moe_exp000_000_pretrain_sftcooldown_iter10k     | v2_1170L_opus | 1170 png | 77.41     | 80.46     | 74.61     | 1170/1170 |
| cosmos3p5_ga_60bm30b_t2ionly_moe_exp000_001_pretrain_sdg_sftcooldown_iter10k | v2_1170L_opus | 1170 png | 78.74     | 82.35     | 75.43     | 1170/1170 |
| cosmos3p5_ga_60bm30b_t2ionly_moe_exp000_002_pretrain_iter100k                | v2_1170L_opus | 1170 png | 80.72     | 83.70     | 77.98     | 1170/1170 |
| cosmos3p5_ga_60bm30b_t2ionly_moe_exp000_003_pretrain_sdg_iter100k            | v2_1170L_opus | 1170 png | **82.01** | **83.88** | **80.29** | 1170/1170 |
