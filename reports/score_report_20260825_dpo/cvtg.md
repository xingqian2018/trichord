# CVTG Score Report — SDPO Evaluation

Gen: 50 steps, neg prompt on, guidance 4.0 (unless noted). Scored with `gemini-3.1-pro@nvidia`.

## cvtg500L_gc

| Run                                                                                  | iterNk     | Images     | gned     | pned     | success     |
|--------------------------------------------------------------------------------------|------------|------------|----------|----------|-------------|
| BASELINE                                                                             | —          | 500        | 0.8088   | 0.8908   | 500/500     |
| ------------------------------------------------------------------------------------ | ---------- | ---------- | -------- | -------- | ----------- |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter100    | 100        | 500        | 0.7417   | 0.8405   | 500/500     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter200    | 200        | 500        | 0.6653   | 0.7984   | 500/500     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter300    | 300        | 500        | 0.6951   | 0.8079   | 500/500     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter400    | 400        | 500        | 0.7339   | 0.8191   | 500/500     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter500    | 500        | 500        | 0.7524   | 0.8098   | 500/500     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter600    | 600        | 500        | 0.6863   | 0.7354   | 500/500     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter700    | 700        | 500        | 0.6706   | 0.7249   | 500/500     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter800    | 800        | 500        | 0.6670   | 0.7126   | 500/500     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter900    | 900        | 500        | —        | —        | —           |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter1000   | 1000       | 500        | —        | —        | —           |
| ------------------------------------------------------------------------------------ | ---------- | ---------- | -------- | -------- | ----------- |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter100     | 100        | —          | —        | —        | —           |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter200     | 200        | —          | —        | —        | —           |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter300     | 300        | —          | —        | —        | —           |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter400     | 400        | —          | —        | —        | —           |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter500     | 500        | —          | —        | —        | —           |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter600     | 600        | —          | —        | —        | —           |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter700     | 700        | —          | —        | —        | —           |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter800     | 800        | —          | —        | —        | —           |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter900     | 900        | —          | —        | —        | —           |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter1000    | 1000       | —          | —        | —        | —           |

## cvtg102ch_gc

| Run                                                                                        | iterNk     | Images     | gned     | pned     | success     |
|--------------------------------------------------------------------------------------------|------------|------------|----------|----------|-------------|
| BASELINE                                                                                   | —          | 102        | 0.3202   | 0.4122   | 102/102     |
| ------------------------------------------------------------------------------------       | ---------- | ---------- | -------- | -------- | ----------- |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter100          | 100        | 102        | 0.2616   | 0.3546   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter200          | 200        | 102        | 0.2054   | 0.2810   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter300          | 300        | 102        | 0.2294   | 0.3205   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter400          | 400        | 102        | 0.2089   | 0.2978   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter500          | 500        | 102        | 0.1982   | 0.2869   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter600          | 600        | 102        | 0.1803   | 0.2469   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter700          | 700        | 102        | 0.1430   | 0.2435   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter800          | 800        | 102        | 0.1587   | 0.2360   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter900          | 900        | 102        | 0.1410   | 0.2101   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter1000         | 1000       | 102        | 0.1432   | 0.2283   | 102/102     |
| ------------------------------------------------------------------------------------       | ---------- | ---------- | -------- | -------- | ----------- |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter100           | 100        | 102        | 0.3311   | 0.4292   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter200           | 200        | 102        | 0.3082   | 0.4063   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter300           | 300        | 102        | 0.2736   | 0.3619   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter400           | 400        | 102        | 0.2900   | 0.3901   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter500           | 500        | 102        | 0.2403   | 0.3234   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter600           | 600        | 102        | 0.2275   | 0.3111   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter700           | 700        | 102        | 0.1955   | 0.3104   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter800           | 800        | 102        | 0.1962   | 0.2810   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter900           | 900        | 102        | 0.1997   | 0.2812   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_002_sgdtext_shift5_beta500_alpha0_iter1000          | 1000       | 102        | 0.1285   | 0.2037   | 102/102     |
| ------------------------------------------------------------------------------------       | ---------- | ---------- | -------- | -------- | ----------- |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_001_sgdtext_shift5_beta2000_alpha0_lr5em6_iter100   | 100        | 102        | 0.3581   | 0.4715   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_001_sgdtext_shift5_beta2000_alpha0_lr5em6_iter200   | 200        | 102        | 0.3302   | 0.4348   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_001_sgdtext_shift5_beta2000_alpha0_lr5em6_iter300   | 300        | 102        | 0.3490   | 0.4414   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_001_sgdtext_shift5_beta2000_alpha0_lr5em6_iter400   | 400        | 102        | 0.3476   | 0.4626   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_001_sgdtext_shift5_beta2000_alpha0_lr5em6_iter500   | 500        | 102        | 0.3397   | 0.4668   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_001_sgdtext_shift5_beta2000_alpha0_lr5em6_iter600   | 600        | 102        | 0.3385   | 0.4415   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_001_sgdtext_shift5_beta2000_alpha0_lr5em6_iter700   | 700        | 102        | 0.3304   | 0.4211   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_001_sgdtext_shift5_beta2000_alpha0_lr5em6_iter800   | 800        | 102        | 0.2738   | 0.3979   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_001_sgdtext_shift5_beta2000_alpha0_lr5em6_iter900   | 900        | 102        | 0.2639   | 0.3754   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_001_sgdtext_shift5_beta2000_alpha0_lr5em6_iter1000  | 1000       | 102        | 0.2440   | 0.3656   | 102/102     |
| ------------------------------------------------------------------------------------       | ---------- | ---------- | -------- | -------- | ----------- |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_002_sgdtext_shift5_beta500_alpha0_lr5em6_iter100    | 100        | 102        | 0.3050   | 0.4234   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_002_sgdtext_shift5_beta500_alpha0_lr5em6_iter200    | 200        | 102        | 0.3128   | 0.4250   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_002_sgdtext_shift5_beta500_alpha0_lr5em6_iter300    | 300        | 102        | 0.2944   | 0.4028   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_002_sgdtext_shift5_beta500_alpha0_lr5em6_iter400    | 400        | 102        | 0.2952   | 0.3880   | 101/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_002_sgdtext_shift5_beta500_alpha0_lr5em6_iter500    | 500        | 102        | 0.2973   | 0.4083   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_002_sgdtext_shift5_beta500_alpha0_lr5em6_iter600    | 600        | 102        | 0.2836   | 0.3870   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_002_sgdtext_shift5_beta500_alpha0_lr5em6_iter700    | 700        | 102        | 0.2764   | 0.3798   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_002_sgdtext_shift5_beta500_alpha0_lr5em6_iter800    | 800        | 102        | 0.2895   | 0.3892   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_002_sgdtext_shift5_beta500_alpha0_lr5em6_iter900    | 900        | 102        | 0.2876   | 0.3760   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp001_002_sgdtext_shift5_beta500_alpha0_lr5em6_iter1000   | 1000       | 102        | 0.2893   | 0.3893   | 102/102     |
| ------------------------------------------------------------------------------------       | ---------- | ---------- | -------- | -------- | ----------- |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_001_sgdtext_shift5_beta500_alpha0p1_lr5em6_iter100  | 100        | 102        | 0.3588   | 0.4565   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_001_sgdtext_shift5_beta500_alpha0p1_lr5em6_iter200  | 200        | 102        | 0.3878   | 0.4966   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_001_sgdtext_shift5_beta500_alpha0p1_lr5em6_iter300  | 300        | 102        | 0.3813   | 0.4974   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_001_sgdtext_shift5_beta500_alpha0p1_lr5em6_iter400  | 400        | 102        | 0.3631   | 0.4828   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_001_sgdtext_shift5_beta500_alpha0p1_lr5em6_iter500  | 500        | 102        | 0.3821   | 0.4943   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_001_sgdtext_shift5_beta500_alpha0p1_lr5em6_iter600  | 600        | 102        | 0.3373   | 0.4464   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_001_sgdtext_shift5_beta500_alpha0p1_lr5em6_iter700  | 700        | 102        | 0.3635   | 0.4570   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_001_sgdtext_shift5_beta500_alpha0p1_lr5em6_iter800  | 800        | 102        | 0.3845   | 0.4791   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_001_sgdtext_shift5_beta500_alpha0p1_lr5em6_iter900  | 900        | 102        | 0.4047   | 0.4899   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_001_sgdtext_shift5_beta500_alpha0p1_lr5em6_iter1000 | 1000       | 102        | 0.3975   | 0.4955   | 102/102     |
| ------------------------------------------------------------------------------------       | ---------- | ---------- | -------- | -------- | ----------- |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_002_sgdtext_shift5_beta500_alpha1_lr5em6_iter100    | 100        | 102        | 0.3781   | 0.4825   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_002_sgdtext_shift5_beta500_alpha1_lr5em6_iter200    | 200        | 102        | 0.4065   | 0.5063   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_002_sgdtext_shift5_beta500_alpha1_lr5em6_iter300    | 300        | 102        | 0.4162   | 0.5181   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_002_sgdtext_shift5_beta500_alpha1_lr5em6_iter400    | 400        | 102        | 0.4290   | 0.5399   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_002_sgdtext_shift5_beta500_alpha1_lr5em6_iter500    | 500        | 102        | 0.4500   | 0.5537   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_002_sgdtext_shift5_beta500_alpha1_lr5em6_iter600    | 600        | 102        | 0.4387   | 0.5372   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_002_sgdtext_shift5_beta500_alpha1_lr5em6_iter700    | 700        | 102        | 0.4417   | 0.5426   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_002_sgdtext_shift5_beta500_alpha1_lr5em6_iter800    | 800        | 102        | 0.4410   | 0.5413   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_002_sgdtext_shift5_beta500_alpha1_lr5em6_iter900    | 900        | 102        | 0.4289   | 0.5441   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_002_sgdtext_shift5_beta500_alpha1_lr5em6_iter1000   | 1000       | 102        | 0.4442   | 0.5493   | 102/102     |
| ------------------------------------------------------------------------------------       | ---------- | ---------- | -------- | -------- | ----------- |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_003_sgdtext_shift5_beta500_alpha10_lr5em6_iter100   | 100        | 102        | 0.4134   | 0.5041   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_003_sgdtext_shift5_beta500_alpha10_lr5em6_iter200   | 200        | 102        | 0.4228   | 0.5296   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_003_sgdtext_shift5_beta500_alpha10_lr5em6_iter300   | 300        | 102        | 0.4396   | 0.5378   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_003_sgdtext_shift5_beta500_alpha10_lr5em6_iter400   | 400        | 102        | 0.4283   | 0.5417   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_003_sgdtext_shift5_beta500_alpha10_lr5em6_iter500   | 500        | 102        | 0.4377   | 0.5384   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_003_sgdtext_shift5_beta500_alpha10_lr5em6_iter600   | 600        | 102        | 0.4546   | 0.5567   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_003_sgdtext_shift5_beta500_alpha10_lr5em6_iter700   | 700        | 102        | 0.4362   | 0.5364   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_003_sgdtext_shift5_beta500_alpha10_lr5em6_iter800   | 800        | 102        | 0.4623   | 0.5644   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_003_sgdtext_shift5_beta500_alpha10_lr5em6_iter900   | 900        | 102        | 0.4572   | 0.5529   | 102/102     |
| cosmos3plus_64bm32b_t2ionly_dpo_exp002_003_sgdtext_shift5_beta500_alpha10_lr5em6_iter1000  | 1000       | 102        | 0.4646   | 0.5687   | 101/102     |

## cvtg102ch_gc (no-EMA)

| Run                                                                               | iterNk   | Images   | gned   | pned   | success   |
|-----------------------------------------------------------------------------------|----------|----------|--------|--------|-----------|
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter100 | 100      | 102      | 0.2598 | 0.3523 | 102/102   |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter200 | 200      | 102      | 0.2192 | 0.3062 | 102/102   |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter300 | 300      | 102      | 0.2339 | 0.3018 | 102/102   |
| cosmos3plus_64bm32b_t2ionly_dpo_exp000_001_sgdtext_shift5_beta2000_alpha0_iter500 | 500      | 102      | 0.1792 | 0.25   | 102/102   |
