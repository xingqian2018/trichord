# CVTG Baseline Scoring Report

Date: 2026-05-12

---

## cvtg102ch_score

| Model           | gned   | pned   |
|-----------------|--------|--------|
| flux_1          | 0.0301 | 0.0831 |
| flux_2          | 0.1602 | 0.4693 |
| glmimage        | 0.554  | 0.8799 |
| npb             | 0.4272 | 0.8359 |
| qwen_image      | 0.4521 | 0.816  |
| qwen_image_2512 | 0.4245 | 0.8417 |
| sd3p5           | 0.0341 | 0.098  |
| zimage          | 0.5473 | 0.8527 |

---

## cvtg500L_score

| Model           | gned   | pned   |
|-----------------|--------|--------|
| flux_1          | 0.286  | 0.4894 |
| flux_2          | 0.5371 | 0.8564 |
| glmimage        | 0.6705 | 0.8562 |
| npb             | 0.5541 | 0.7942 |
| qwen_image      | 0.66   | 0.9242 |
| qwen_image_2512 | 0.6448 | 0.9539 |
| sd3p5           | 0.2763 | 0.5235 |
| zimage          | 0.5891 | 0.9073 |

---

Source logs (GCP):
- `/home/xingqianx/log/slurm/cvtg102ch_score/`
- `/home/xingqianx/log/slurm/cvtg500L_score/`

---

# CVTG Newer Round — G3.1-Pro Judge

Date: 2026-05-14
Judge: gemini-3.1-pro (signature: g3p1p)
Images: s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/cvtg/

---

## cvtg500L — Baselines

| Model                              | benchmark     | gned       | pned       | success   |
|------------------------------------|---------------|------------|------------|-----------|
| cosmos3_image_v1p5_iter108k        | cvtg500L_opus | **0.8480** | 0.8826     | 498/500   |
| nano_banana_pro (capital agnostic) | cvtg500L      | 0.7597     | **0.9145** | 499/500   |
| nano_banana_pro                    | cvtg500L      | 0.5924     | 0.7179     | 500/500   |
| flux_2_klein_9b                    | cvtg500L      | 0.6618     | 0.8138     | 499/500   |
| qwen_image_2512                    | cvtg500L      | 0.7968     | 0.9086     | 499/500   |
| qwen_image                         | cvtg500L      | 0.7348     | 0.8672     | 500/500   |
| z_image_turbo                      | cvtg500L      | 0.7520     | 0.8695     | 499/500   |
| flux_2_dev                         | cvtg500L      | 0.7471     | 0.8498     | 499/500   |
| hunyuan_image_3                    | cvtg500L      | 0.7140     | 0.8768     | 500/500   |
| flux_1_kontext_dev                 | cvtg500L      | 0.3572     | 0.4648     | 499/500   |
| sd_v3p5_large                      | cvtg500L      | 0.3323     | 0.5242     | 498/500   |

## cvtg500L — Cosmos3

| Model                                                       | benchmark       | gned         | pned         | success     |
|-------------------------------------------------------------|-----------------|--------------|--------------|-------------|
| cosmos3_image_text_focused_iter20k                          | cvtg500L_opus   | 0.8586       | 0.8912       | 500/500     |
| cosmos3_image_v1_v1p4_iter100k                              | cvtg500L_opus   | 0.8496       | 0.8884       | 500/500     |
| cosmos3_image_v2_v1_iter1k                                  | cvtg500L_opus   | 0.7228       | 0.7516       | 497/500     |
| cosmos3_image_v2_v1p3_iter36k_960                           | cvtg500L_opus   | 0.8531       | 0.8920       | 500/500     |
| cosmos3-frozen-midtrain-v3p1                                | cvtg500L_opus   | 0.6630       | 0.6937       | 497/500     |
| ------------------------------------                        | --------------- | ------------ | ------------ | ----------- |
| cosmos3-nano                                                | cvtg500L_opus   | 0.2423       | 0.2653       | 499/500     |
| cosmos3-frozen-midtrain-v3                                  | cvtg500L_opus   | 0.6677       | 0.7097       | 497/500     |
| cosmos3_image_v1_v1p5_iter108k                              | cvtg500L_opus   | 0.8480       | 0.8826       | 498/500     |
| cosmos3_image_v2_v1p2_iter33k                               | cvtg500L_opus   | 0.8565       | 0.8936       | 497/500     |
| cosmos3_image_v2_v1p4_iter40k                               | cvtg500L_opus   | 0.8508       | 0.8869       | 494/500     |
| cosmos3_image_only_v3p3_iter22k                             | cvtg500L_opus   | 0.8251       | 0.8939       | 498/500     |
| cosmos3_image_only_v3_merged_t04g06_iter0                   | cvtg500L_opus   | 0.8618       | 0.9054       | 499/500     |
| cosmos3_image_only_v4_merged_t05g05_iter0                   | cvtg500L_opus   | 0.8663       | 0.9134       | 500/500     |

| cosmos3_t2i_exp000_text_only_iter1p5k                       | cvtg500L_opus   | 0.8034       | 0.8877       | 500/500     |
| cosmos3_t2i_exp000_text_only_iter4k                         | cvtg500L_opus   | 0.7658       | 0.8726       | 499/500     |
| cosmos3_t2i_exp001_text_mix_iter1p5k                        | cvtg500L_opus   | 0.8221       | 0.8911       | 498/500     |
| cosmos3_t2i_exp002_text_only_from_frozen_iter1k             | cvtg500L_opus   | 0.6590       | 0.7334       | 497/500     |
| cosmos3_t2i_exp003_text_mix2_iter1k                         | cvtg500L_opus   | 0.7506       | 0.8419       | 500/500     |
| cosmos3_t2i_exp003_text_mix2_iter2k                         | cvtg500L_opus   | 0.7376       | 0.8273       | 498/500     |
| cosmos3_t2i_exp004_text_mix2_from_frozen_iter2k             | cvtg500L_opus   | 0.5934       | 0.6835       | 495/500     |
| cosmos3_t2i_exp005_text_mix3_iter1k                         | cvtg500L_opus   | 0.7216       | 0.8163       | 497/500     |
| cosmos3_t2i_exp006_text_mix3_from_frozen_iter500            | cvtg500L_opus   | 0.6767       | 0.7377       | 497/500     |
| cosmos3_t2i_exp007_text_mix4_iter4k                         | cvtg500L_opus   | 0.7798       | 0.8548       | 499/500     |
| cosmos3_t2i_exp008_text_mix4_from_frozen_iter4k             | cvtg500L_opus   | 0.7216       | 0.7663       | 497/500     |
| cosmos3_t2i_exp009_union5_from_frozen_iter4k                | cvtg500L_opus   | 0.7960       | 0.8337       | 499/500     |
| cosmos3_t2i_exp009_union5_from_frozen_iter10k               | cvtg500L_opus   | 0.8294       | 0.8706       | 497/500     |
| cosmos3_t2i_exp009_union5_from_frozen_iter15k               | cvtg500L_opus   | 0.8388       | 0.8857       | 500/500     |
| cosmos3_t2i_exp009_union5_from_frozen_iter20k               | cvtg500L_opus   | 0.8514       | 0.8918       | 500/500     |
| cosmos3_t2i_exp009_union5_from_frozen_iter25k               | cvtg500L_opus   | 0.8577       | 0.8958       | 499/500     |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter26k  | cvtg500L_opus   | 0.8561       | 0.9002       | 500/500     |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter28k  | cvtg500L_opus   | 0.8566       | 0.8984       | 500/500     |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter30k  | cvtg500L_opus   | 0.8378       | 0.8908       | 500/500     |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k  | cvtg500L_opus   | 0.8088       | 0.8730       | 500/500     |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter35k  | cvtg500L_opus   |              |              |             |
| cosmos3_t2i_exp009_sft1_text_from_exp009_25k_lr1em5_iter26k | cvtg500L_opus   | 0.8608       | 0.9002       | 499/500     |
| cosmos3_t2i_exp009_sft1_text_from_exp009_25k_lr1em5_iter28k | cvtg500L_opus   | 0.8699       | 0.9065       | 496/500     |
| cosmos3_t2i_exp009_sft1_text_from_exp009_25k_lr1em5_iter30k | cvtg500L_opus   | 0.8632       | 0.9070       | 499/500     |
| cosmos3_t2i_exp009_sft1_text_from_exp009_25k_lr1em5_iter35k | cvtg500L_opus   | **0.8707**   | 0.9098       | 500/500     |
| cosmos3_t2i_exp009_sft1_text_from_exp009_25k_lr1em5_iter40k | cvtg500L_opus   | 0.8701       | **0.9173**   | 499/500     |
| cosmos3_t2i_merged_000                                      | cvtg500L_opus   | 0.8669       | 0.9061       | 498/500     |
| cosmos3_t2i_merged_003                                      | cvtg500L_opus   | 0.8579       | 0.8991       | 496/500     |
| cosmos3_t2i_merged_006                                      | cvtg500L_opus   | 0.8662       | 0.9097       | 497/500     |
| cosmos3_t2i_merged_007                                      | cvtg500L_opus   | 0.8683       | 0.9104       | 500/500     |
| cosmos3_t2i_exp010_sft0_union6_from_merge007_iter9k         | cvtg500L_opus   | 0.7285       | 0.8403       | 497/500     |

---

## cvtg102ch — Baselines

| Model                       | benchmark      | gned       | pned       | success   |
|-----------------------------|----------------|------------|------------|-----------|
| cosmos3_image_v1p5_iter108k | cvtg102ch_opus | 0.4770     | 0.6264     | 100/102   |
| nano_banana_pro             | cvtg102ch      | 0.4600     | **0.7640** | 101/102   |
| flux_2_klein_9b             | cvtg102ch      | 0.1888     | 0.3445     | 98/102    |
| qwen_image_2512             | cvtg102ch      | 0.4633     | 0.7126     | 101/102   |
| qwen_image                  | cvtg102ch      | 0.4883     | 0.6846     | 102/102   |
| z_image_turbo               | cvtg102ch      | **0.4918** | 0.7332     | 101/102   |
| flux_2_dev                  | cvtg102ch      | 0.4433     | 0.6874     | 102/102   |
| hunyuan_image_3             | cvtg102ch_opus | 0.4905     | 0.7131     | 102/102   |
| flux_1_kontext_dev          | cvtg102ch      | 0.0432     | 0.0721     | 102/102   |
| sd_v3p5_large               | cvtg102ch      | 0.0598     | 0.1403     | 101/102   |

## cvtg102ch — Cosmos3

| Model                                                       | benchmark              | gned         | pned         | success     |
|-------------------------------------------------------------|------------------------|--------------|--------------|-------------|
| cosmos3_image_v2_v1_iter1k                                  | cvtg102ch_opus_ascii   | 0.1272       | 0.2000       | 102/102     |
| cosmos3_image_text_focused_iter20k                          | cvtg102ch_opus         | **0.5067**   | **0.6524**   | 102/102     |
| cosmos3_image_v1_v1p4_iter100k                              | cvtg102ch_opus         | 0.5009       | 0.6290       | 102/102     |
| cosmos3_image_v2_v1p3_iter36k_960                           | cvtg102ch_opus         | 0.5064       | 0.6429       | 102/102     |
| cosmos3_image_v2_v1p3_iter36k_1024                          | cvtg102ch_opus         | 0.4851       | 0.6410       | 102/102     |
| cosmos3-frozen-midtrain-v3p1                                | cvtg102ch_opus_ascii   | 0.0838       | 0.1513       | 102/102     |
| ------------------------------------                        | ---------------------- | ------------ | ------------ | ----------- |
| cosmos3-nano                                                | cvtg102ch_opus_ascii   | 0.0463       | 0.0970       | 102/102     |
| cosmos3-frozen-midtrain-v3                                  | cvtg102ch_opus_ascii   | 0.0848       | 0.1631       | 101/102     |
| cosmos3_image_v1_v1p5_iter108k                              | cvtg102ch_opus         | 0.4770       | 0.6264       | 100/102     |
| cosmos3_image_v2_v1p2_iter33k                               | cvtg102ch_opus_ascii   | 0.4618       | 0.5988       | 100/102     |
| cosmos3_image_v2_v1p4_iter40k                               | cvtg102ch_opus_ascii   | 0.4645       | 0.6249       | 101/102     |
| cosmos3_image_only_v3p3_iter22k                             | cvtg102ch_opus_ascii   | 0.4675       | 0.5983       | 100/102     |
| cosmos3_image_only_v3_merged_t04g06_iter0                   | cvtg102ch_opus_ascii   | 0.4396       | 0.5567       | 101/102     |
| cosmos3_image_only_v4_merged_t05g05_iter0                   | cvtg102ch_opus_ascii   | 0.4507       | 0.5890       | 102/102     |

| cosmos3_t2i_exp000_text_only_iter1p5k                       | cvtg102ch_opus         | 0.5103       | 0.6889       | 102/102     |
| cosmos3_t2i_exp000_text_only_iter4k                         | cvtg102ch_opus         | 0.5063       | 0.6887       | 101/102     |
| cosmos3_t2i_exp001_text_mix_iter1p5k                        | cvtg102ch_opus         | 0.5190       | 0.6842       | 99/102      |
| cosmos3_t2i_exp002_text_only_from_frozen_iter1k             | cvtg102ch_opus         | 0.2081       | 0.2834       | 102/102     |
| cosmos3_t2i_exp003_text_mix2_iter1k                         | cvtg102ch_opus         | 0.5130       | 0.6748       | 102/102     |
| cosmos3_t2i_exp003_text_mix2_iter2k                         | cvtg102ch_opus         | 0.4923       | 0.6379       | 102/102     |
| cosmos3_t2i_exp004_text_mix2_from_frozen_iter2k             | cvtg102ch_opus         | 0.1991       | 0.2675       | 102/102     |
| cosmos3_t2i_exp005_text_mix3_iter1k                         | cvtg102ch_opus_ascii   | 0.4519       | 0.5639       | 101/102     |
| cosmos3_t2i_exp006_text_mix3_from_frozen_iter500            | cvtg102ch_opus_ascii   | 0.1053       | 0.1803       | 102/102     |
| cosmos3_t2i_exp007_text_mix4_iter4k                         | cvtg102ch_opus_ascii   | 0.4681       | 0.5939       | 102/102     |
| cosmos3_t2i_exp008_text_mix4_from_frozen_iter4k             | cvtg102ch_opus_ascii   | 0.1261       | 0.1961       | 102/102     |
| cosmos3_t2i_exp009_union5_from_frozen_iter4k                | cvtg102ch_opus_ascii   | 0.1871       | 0.2662       | 101/102     |
| cosmos3_t2i_exp009_union5_from_frozen_iter10k               | cvtg102ch_opus_ascii   | 0.2843       | 0.3585       | 102/102     |
| cosmos3_t2i_exp009_union5_from_frozen_iter15k               | cvtg102ch_opus_ascii   | 0.3018       | 0.3958       | 101/102     |
| cosmos3_t2i_exp009_union5_from_frozen_iter20k               | cvtg102ch_opus_ascii   | 0.3358       | 0.4342       | 102/102     |
| cosmos3_t2i_exp009_union5_from_frozen_iter25k               | cvtg102ch_opus_ascii   | 0.3779       | 0.4677       | 102/102     |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter26k  | cvtg102ch_opus_ascii   | 0.3750       | 0.4760       | 101/102     |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter28k  | cvtg102ch_opus_ascii   | 0.3522       | 0.4656       | 101/102     |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter30k  | cvtg102ch_opus_ascii   | 0.3236       | 0.4361       | 102/102     |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter31k  | cvtg102ch_opus_ascii   | 0.3202       | 0.4122       | 101/102     |
| cosmos3_t2i_exp009_sft0_uhq_from_exp009_25k_lr1em5_iter35k  | cvtg102ch_opus_ascii   | 0.2539       | 0.3594       | 99/102      |
| cosmos3_t2i_exp009_sft1_text_from_exp009_25k_lr1em5_iter26k | cvtg102ch_opus_ascii   | 0.3860       | 0.4823       | 102/102     |
| cosmos3_t2i_exp009_sft1_text_from_exp009_25k_lr1em5_iter28k | cvtg102ch_opus_ascii   | 0.4173       | 0.5255       | 100/102     |
| cosmos3_t2i_exp009_sft1_text_from_exp009_25k_lr1em5_iter30k | cvtg102ch_opus_ascii   | 0.4259       | 0.5589       | 101/102     |
| cosmos3_t2i_exp009_sft1_text_from_exp009_25k_lr1em5_iter35k | cvtg102ch_opus_ascii   | 0.4655       | 0.6033       | 102/102     |
| cosmos3_t2i_exp009_sft1_text_from_exp009_25k_lr1em5_iter40k | cvtg102ch_opus_ascii   | **0.4953**   | **0.6108**   | 101/102     |
| cosmos3_t2i_merged_000                                      | cvtg102ch_opus_ascii   | 0.3998       | 0.5210       | 102/102     |
| cosmos3_t2i_merged_003                                      | cvtg102ch_opus_ascii   | 0.3936       | 0.5038       | 102/102     |
| cosmos3_t2i_merged_006                                      | cvtg102ch_opus_ascii   | 0.4335       | 0.5692       | 102/102     |
| cosmos3_t2i_merged_007                                      | cvtg102ch_opus_ascii   | 0.4489       | 0.5836       | 100/102     |
| cosmos3_t2i_exp010_sft0_union6_from_merge007_iter9k         | cvtg102ch_opus_ascii   | 0.4448       | 0.6003       | 101/102     |

---

## ensure_ascii Experiment — cosmos3_image_v2_v1p3_iter36k_1024

Date: 2026-05-21
Benchmark: cvtg102ch_opus_ascii

| Run                                    | gned   | pned   | success   |
|----------------------------------------|--------|--------|-----------|
| ensure_ascii=True (default, v1p3_1024) | 0.4851 | 0.641  | 102/102   |
| ensure_ascii=False                     | 0.486  | 0.6318 | 101/102   |

Note: `ensure_ascii=True` result is from the `v1p3_1024` (no-suffix) run which uses default JSON encoding. The explicit `_ensure_ascii_True` run has images but scoring not yet done.

---

Note: sd_v3p5_large/cvtg500L scoring in progress as of 2026-05-14.
