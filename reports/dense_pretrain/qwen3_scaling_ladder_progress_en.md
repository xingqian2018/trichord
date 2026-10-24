# Qwen3-MoE scaling ladder — experiment progress tracker (Boxiang's runs)

_Sources: Boxiang's repo `cosmos_base_training/cosmos_llm_scaling_ladder` (origin/main @ 908f38f) + slurm / run logs / eval directories.
Doc-isolated results come from `results/<Change ID>/ckpts/<rung>/eval/RESULTS.md` (the new layout since 10-22 13:41), packed results from `results/<Change ID>/scores.json`.
Results Boxiang has not filed yet are read directly from run logs or eval directories and marked †. Last sync: 2026-10-22 18:05._

- **Primary metric: val_v2_decon, doc-isolated protocol** (each document evaluated on its own, no packing; protocol in `reports/val_protocol_doc_isolated_2026-10-20.md`).
  nats/token, lower is better; dCE = arm − reference, **negative is better**.
- Doc-isolated losses are ~0.013–0.016 below the packed numbers from training (BASE q0: 1.8226 vs 1.8372; q4: 1.2818 vs 1.2945), so the two protocols **must not be compared with each other**; within one protocol, however, the dCEs are nearly identical (differ by ≤ 0.002).
- One seed. Seed noise ≈ **0.008 nats** (range of the 3 BASE seeds at q0/q1, packed).
- Status: ✅ doc-isolated available · 🟡 training done, doc eval not out yet · ⏳ training / queued · ⏸ paused (checkpoint kept, resumable) · ⚠️ crashed, needs action · — not planned
- Scope: every arm runs q0–q5 first; q6 is deferred (BASE q6 paused at step 42000 / 260421). **All q4 runs are currently paused; BASE is the only q5 still running.**

---

## 0. Rung specs

| rung | L × h | N_active (non-emb) | N_total (non-emb) | tokens @TPP200 | iters | est. cost |
|---|---|---:|---:|---:|---:|---|
| q0 | 12 × 512  | 48.0M  | 0.47B  | 9.6B   | 4,575   | ~25 GPU-h |
| q1 | 18 × 768  | 153.9M | 1.59B  | 30.8B  | 14,682  | ~170 GPU-h |
| q2 | 24 × 1024 | 355.5M | 3.75B  | 71.1B  | 33,900  | ~720 GPU-h |
| q3 | 30 × 1280 | 683.2M | 7.32B  | 136.6B | 65,157  | ~2,300 GPU-h |
| q4 | 36 × 1536 | 1.17B  | 12.63B | 233.6B | 111,375 | ~6,000 GPU-h |
| q5 | 42 × 1792 | 1.84B  | 20.05B | 368.0B | 175,482 | ~13,700 GPU-h |
| q6 | 48 × 2048 | 2.73B  | 29.91B | 546.1B | 260,400 | ~28k GPU-h (deferred) |

---

## 1. Main table: val_v2_decon, doc-isolated (dCE vs BASE in parentheses)

| arm | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| BASE (AdamW + WSD) | ✅ 1.8226 | ✅ 1.5718 | ✅ 1.4361 | ✅ 1.3442 | ✅ 1.2818 | ⏳ |
| OPT_MUON | ✅ 1.7752 (−0.047) | ✅ 1.5510 (−0.021) | ✅ 1.4197 (−0.016) | ✅ 1.3319 (−0.012) | ⏸ | ⏸ |
| OPT_MUONH | ✅ 1.7580 (−0.065) | ✅ 1.5419 (−0.030) | ✅ 1.4141 (−0.022) | ✅ 1.3266 (−0.018) | ⏸ | ⏸ |
| KD_MUONH_A0P7 | ✅ 1.7318 (−0.091) | ✅ 1.5117 (−0.060) | 🟡 | ⏳ | ⏸ | ⏸ |
| KD_MUONH_A0P8 | ✅ 1.7307 (−0.092) | ✅ 1.5118 (−0.060) | 🟡 | ⏳ | ⏸ | ⏸ |
| KD_MUONH_A0P9 | ✅ 1.7297 (−0.093) | ✅ 1.5107 (−0.061) | 🟡 | ⏳ | ⏸ | ⏸ |

🟡: KD 0.7 / 0.8 / 0.9 q2 finished training at 12:47 / 13:02 / 16:50; not filed yet, and no doc eval submitted yet.

---

## 1b. Main table: val_v2_decon, packed protocol (final validation at the end of training; dCE vs BASE in parentheses)

Packed is the final validation built into training: it is available as soon as a rung finishes, so it covers more rungs than doc-isolated; it is also what Boxiang's README / `scores.json` use. **Do not compare these numbers with section 1** — compare dCEs only within the same protocol.

| arm | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| BASE (AdamW + WSD) | 1.8372 | 1.5874 | 1.4517 | 1.3592 | 1.2945 | ⏳ |
| OPT_MUON | 1.7896 (−0.048) | 1.5655 (−0.022) | 1.4349 (−0.017) | 1.3471 (−0.012) | ⏸ | ⏸ |
| OPT_MUONH | 1.7732 (−0.064) | 1.5566 (−0.031) | 1.4292 (−0.023) | 1.3421 (−0.017) | ⏸ | ⏸ |
| KD_MUONH_A0P7 | 1.7467 (−0.091) | 1.5262 (−0.061) | 1.3984 (−0.053)† | ⏳ | ⏸ | ⏸ |
| KD_MUONH_A0P8 | 1.7455 (−0.092) | 1.5263 (−0.061) | 1.3980 (−0.054)† | ⏳ | ⏸ | ⏸ |
| KD_MUONH_A0P9 | 1.7447 (−0.093) | 1.5256 (−0.062) | 1.3987 (−0.053)† | ⏳ | ⏸ | ⏸ |

Mean of six sets (packed): BASE 2.3442 / 2.0312 / 1.8501 / 1.7293 / 1.6491 (q0–q4); Muon 2.2839 / 2.0029 / 1.8283 / 1.7132 (q0–q3); MuonH 2.2670 / 1.9925 / 1.8243 / 1.7104 (q0–q3);
KD α0.7 / 0.8 / 0.9: q0 2.2300 / 2.2287 / 2.2290, q1 1.9508 / 1.9510 / 1.9503, q2 1.7809† / 1.7800† / 1.7807†.

BASE seed noise (packed): q0 s1–s3 = 1.8372 / 1.8377 / 1.8401, q1 = 1.5874 / 1.5951 / 1.5887. LR bracket (q1): 0.5× 1.5940, 1× 1.5874, 2× 1.5908.

---

## 2. Optimizer arms: dCE vs BASE (doc-isolated)

| rung | Muon: val_v2 / six-set mean | MuonH: val_v2 / six-set mean | MuonH − Muon: val_v2 / six-set mean |
|---|---|---|---|
| q0 | −0.047 / −0.061 | −0.065 / −0.078 | −0.017 / −0.017 |
| q1 | −0.021 / −0.028 | −0.030 / −0.039 | −0.009 / −0.011 |
| q2 | −0.016 / −0.022 | −0.022 / −0.026 | −0.006 / −0.004 |
| q3 | −0.012 / −0.016 | −0.018 / −0.020 | −0.005 / −0.004 |
| q4 | _TBD_ (⏸ at 54,600) | _TBD_ (⏸ at 45,600) | |
| q5 | _TBD_ (⏸) | _TBD_ (⏸) | |

- Ordering MuonH < Muon < AdamW holds on val_v2 at q0–q3.
- **Both optimizers' lead over BASE shrinks with scale** (Muon −0.047 → −0.012, MuonH −0.065 → −0.018).
- **MuonH's lead over Muon shrinks faster**: six-set mean from −0.017 to −0.004; at q3 MuonH is already behind Muon on uncheatable (+0.001) and codefresh (+0.002). On gpt-oss this lead was flat (~−0.009) over g1–g3.
  Boxiang's analysis: `reports/muonh_vs_muon_qwen3_2026-10-22.md`; the MuonH recipe test launched for it has been abandoned (see section 7).

---

## 3. KD arm (TODO 1): alpha sweep on OPT_MUONH, top-64 + tail bucket

Loss = (1 − α)·CE + α·KL(teacher top-64 + tail ‖ student). Teacher: MiMo-V2.6-Pro-RL, cache `dpl.teacher_cache/2.1`. Step time ≈ 1.15× OPT_MUONH.

**dCE vs OPT_MUONH (val_v2_decon / six-set mean)**

| rung | α 0.7 | α 0.8 | α 0.9 | protocol / notes |
|---|---|---|---|---|
| q0 | −0.026 / −0.037 | −0.027 / −0.037 | −0.028 / −0.037 | doc; better on all six sets |
| q1 | −0.030 / −0.040 | −0.030 / −0.040 | −0.031 / −0.041 | doc; better on all six sets |
| q2 | −0.031 / −0.043† | −0.031 / −0.044† | −0.031 / −0.044† | **packed**, doc eval not submitted |
| q3 | ⏳ (93 %) | ⏳ (91 %) | ⏳ (84 %) | finals expected tonight / tomorrow morning |
| q4 | ⏸ at 50,400 | ⏸ at 35,400 | ⏸ at 25,200 | all q4 paused at 14:3x |
| q5 | ⏸ at 19,200 | ⏸ at 23,200 | ⏸ at 14,400 | stopped by Boxiang at 11:30 |

- **KD's gain grows slightly with scale** (val_v2: q0 −0.027 → q1 −0.030 → q2 −0.031; six-set mean −0.037 → −0.040 → −0.044), the opposite of the optimizer arms' trend.
- Stacked with MuonH, KD at q2 is 0.053–0.054 below BASE (packed), of which MuonH alone contributes −0.023.
- **Pending decision**: the three alphas are within 0.002 of each other at q0–q2 (all three at −0.031 at q2); the choice waits for q3. TODO 2 keeps using 0.8 for now.
- Reference from gpt-oss: KD lowered val_v2 by 0.024–0.062 at m0–m4, with the held-out gain 1.1–1.8× the val_v2 gain and growing with N.

---

## 4. Tail-treatment screen (TODO 2): α 0.8, paired against KD_MUONH_A0P8 (B64 bucket)

Promotion rule: a variant goes on to q3–q5 only if it beats B64 by more than ~0.008 at **every** rung q0, q1, q2. Jobs 1176336–1176366. TODO 2 has **packed results only** so far; no doc eval yet.

**dCE vs B64 (packed; q0 B64 = 1.7455, q1 B64 = 1.5263, q2 B64 = 1.3980†)**

| variant | description | q0: val_v2 / six-set mean | q1: val_v2 | q2 | promoted? |
|---|---|---|---|---|---|
| S64Y   | sampled + y exact | −0.0034 / −0.0049 | ⏳ (95 %) | ⏳ (84 %) | _TBD_ |
| S128   | top-128 sampled | −0.0014 / −0.0015 | ⏳ (81 %) | ⏳ (75 %) | _TBD_ |
| S64    | top-64 + sampled tail (4 draws) | −0.0013 / −0.0020 | −0.0032† | ⏳ (83 %) | _TBD_ |
| DROP64 | drop tail (control) | −0.0010 / −0.0011 | — | — | — |
| B128   | top-128 bucket | −0.0005 / −0.0004 | ⚠️ crashed at 85 % | ⏳ (75 %) | _TBD_ |
| B64Y   | top-64 bucket + y exact | −0.0002 / +0.0007 | −0.0009† | ⏳ (84 %) | _TBD_ |
| DD64   | data draw (design 27) | +0.0003 / −0.0009 | ⏳ (87 %) | ⏳ (77 %) | _TBD_ |

⚠️ B128 q1 (job 1176348) died at 17:59 at step 12,520 / 14,685 on an NCCL collective timeout; q1 has a single segment, so **no resume segment is queued** — it needs a resubmit from the latest checkpoint.

- At q0 every variant is inside the noise (paired-difference SD 0.003–0.008), but the ordering matches the dense report's expectation: the unbiased estimators (S64Y / S128 / S64) are slightly better; S64Y is below B64 at every in-loop point from step 3000 on and on all six sets.
- At q1, S64 is −0.0032, larger than its −0.0013 at q0 but still inside the noise; B64Y is still ≈ 0.
- **Top-64 vs top-128** (pairs that differ only in k): at q0, B128 − B64 = −0.0005 and S128 − S64 = −0.0001, no measurable difference; top-128 is slower (B128 +2 %, S128 +7 % s/iter).
- DROP64 is not worse at q0: the "over-confident student" penalty the report expects does not show in packed CE; Boxiang suggests looking at calibration or the doc-isolated results.
- Prior: Xingqian's m1 study found the sampled tail 0.004–0.008 better than the bucket (one seed, inside the noise); this round confirms or refutes it with paired rungs.

---

## 5. Per-set results, all six validation sets (doc-isolated and packed)

<!-- settables:begin -->
Each cell is the loss with the dCE vs BASE in parentheses (negative is better); · = no result yet; † = not filed by Boxiang yet, read from the run log. This section is generated by `reports_xx/gen_tables.py`; do not edit by hand.

**val_v2_decon — doc-isolated**

| arm | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| BASE | 1.8226 | 1.5718 | 1.4361 | 1.3442 | 1.2818 | · |
| OPT_MUON | 1.7752 (−0.047) | 1.5510 (−0.021) | 1.4197 (−0.016) | 1.3319 (−0.012) | · | · |
| OPT_MUONH | 1.7580 (−0.065) | 1.5419 (−0.030) | 1.4141 (−0.022) | 1.3266 (−0.018) | · | · |
| KD_MUONH_A0P7 | 1.7318 (−0.091) | 1.5117 (−0.060) | · | · | · | · |
| KD_MUONH_A0P8 | 1.7307 (−0.092) | 1.5118 (−0.060) | · | · | · | · |
| KD_MUONH_A0P9 | 1.7297 (−0.093) | 1.5107 (−0.061) | · | · | · | · |

**uncheatable_decon — doc-isolated**

| arm | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| BASE | 2.2790 | 1.9481 | 1.7530 | 1.6196 | 1.5284 | · |
| OPT_MUON | 2.2147 (−0.064) | 1.9173 (−0.031) | 1.7305 (−0.022) | 1.6014 (−0.018) | · | · |
| OPT_MUONH | 2.1961 (−0.083) | 1.9065 (−0.042) | 1.7269 (−0.026) | 1.6025 (−0.017) | · | · |
| KD_MUONH_A0P7 | 2.1513 (−0.128) | 1.8544 (−0.094) | · | · | · | · |
| KD_MUONH_A0P8 | 2.1493 (−0.130) | 1.8550 (−0.093) | · | · | · | · |
| KD_MUONH_A0P9 | 2.1480 (−0.131) | 1.8514 (−0.097) | · | · | · | · |

**codefresh_decon — doc-isolated**

| arm | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| BASE | 1.8133 | 1.4556 | 1.2677 | 1.1449 | 1.0675 | · |
| OPT_MUON | 1.7287 (−0.085) | 1.4205 (−0.035) | 1.2429 (−0.025) | 1.1287 (−0.016) | · | · |
| OPT_MUONH | 1.7185 (−0.095) | 1.4132 (−0.042) | 1.2432 (−0.024) | 1.1307 (−0.014) | · | · |
| KD_MUONH_A0P7 | 1.6751 (−0.138) | 1.3693 (−0.086) | · | · | · | · |
| KD_MUONH_A0P8 | 1.6734 (−0.140) | 1.3657 (−0.090) | · | · | · | · |
| KD_MUONH_A0P9 | 1.6726 (−0.141) | 1.3655 (−0.090) | · | · | · | · |

**v3small_decon — doc-isolated**

| arm | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| BASE | 3.0675 | 2.7475 | 2.5579 | 2.4307 | 2.3398 | · |
| OPT_MUON | 3.0110 (−0.056) | 2.7181 (−0.029) | 2.5357 (−0.022) | 2.4141 (−0.017) | · | · |
| OPT_MUONH | 2.9948 (−0.073) | 2.7055 (−0.042) | 2.5312 (−0.027) | 2.4079 (−0.023) | · | · |
| KD_MUONH_A0P7 | 2.9596 (−0.108) | 2.6677 (−0.080) | · | · | · | · |
| KD_MUONH_A0P8 | 2.9605 (−0.107) | 2.6699 (−0.078) | · | · | · | · |
| KD_MUONH_A0P9 | 2.9620 (−0.105) | 2.6691 (−0.078) | · | · | · | · |

**pile_test_decon — doc-isolated**

| arm | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| BASE | 2.4808 | 2.1739 | 2.0021 | 1.8811 | 1.7966 | · |
| OPT_MUON | 2.4264 (−0.054) | 2.1481 (−0.026) | 1.9790 (−0.023) | 1.8644 (−0.017) | · | · |
| OPT_MUONH | 2.4070 (−0.074) | 2.1354 (−0.038) | 1.9730 (−0.029) | 1.8568 (−0.024) | · | · |
| KD_MUONH_A0P7 | 2.3720 (−0.109) | 2.0967 (−0.077) | · | · | · | · |
| KD_MUONH_A0P8 | 2.3722 (−0.109) | 2.0973 (−0.077) | · | · | · | · |
| KD_MUONH_A0P9 | 2.3725 (−0.108) | 2.0984 (−0.076) | · | · | · | · |

**pile_10k_decon — doc-isolated**

| arm | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| BASE | 2.5078 | 2.1869 | 1.9985 | 1.8703 | 1.7794 | · |
| OPT_MUON | 2.4512 (−0.057) | 2.1591 (−0.028) | 1.9761 (−0.022) | 1.8520 (−0.018) | · | · |
| OPT_MUONH | 2.4307 (−0.077) | 2.1471 (−0.040) | 1.9707 (−0.028) | 1.8467 (−0.024) | · | · |
| KD_MUONH_A0P7 | 2.3956 (−0.112) | 2.1069 (−0.080) | · | · | · | · |
| KD_MUONH_A0P8 | 2.3960 (−0.112) | 2.1071 (−0.080) | · | · | · | · |
| KD_MUONH_A0P9 | 2.3963 (−0.111) | 2.1069 (−0.080) | · | · | · | · |

**six-set mean — doc-isolated**

| arm | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| BASE | 2.3285 | 2.0140 | 1.8359 | 1.7151 | 1.6322 | · |
| OPT_MUON | 2.2679 (−0.061) | 1.9857 (−0.028) | 1.8140 (−0.022) | 1.6988 (−0.016) | · | · |
| OPT_MUONH | 2.2509 (−0.078) | 1.9749 (−0.039) | 1.8098 (−0.026) | 1.6952 (−0.020) | · | · |
| KD_MUONH_A0P7 | 2.2142 (−0.114) | 1.9345 (−0.080) | · | · | · | · |
| KD_MUONH_A0P8 | 2.2137 (−0.115) | 1.9345 (−0.080) | · | · | · | · |
| KD_MUONH_A0P9 | 2.2135 (−0.115) | 1.9337 (−0.080) | · | · | · | · |

**val_v2_decon — packed**

| arm | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| BASE | 1.8372 | 1.5874 | 1.4517 | 1.3592 | 1.2945 | · |
| OPT_MUON | 1.7896 (−0.048) | 1.5655 (−0.022) | 1.4349 (−0.017) | 1.3471 (−0.012) | · | · |
| OPT_MUONH | 1.7732 (−0.064) | 1.5566 (−0.031) | 1.4292 (−0.022) | 1.3421 (−0.017) | · | · |
| KD_MUONH_A0P7 | 1.7467 (−0.091) | 1.5262 (−0.061) | 1.3984 (−0.053)† | · | · | · |
| KD_MUONH_A0P8 | 1.7455 (−0.092) | 1.5263 (−0.061) | 1.3980 (−0.054)† | · | · | · |
| KD_MUONH_A0P9 | 1.7447 (−0.093) | 1.5256 (−0.062) | 1.3987 (−0.053)† | · | · | · |

**uncheatable_decon — packed**

| arm | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| BASE | 2.3089 | 1.9783 | 1.7835 | 1.6502 | 1.5582 | · |
| OPT_MUON | 2.2437 (−0.065) | 1.9463 (−0.032) | 1.7605 (−0.023) | 1.6313 (−0.019) | · | · |
| OPT_MUONH | 2.2260 (−0.083) | 1.9366 (−0.042) | 1.7572 (−0.026) | 1.6325 (−0.018) | · | · |
| KD_MUONH_A0P7 | 2.1805 (−0.128) | 1.8833 (−0.095) | 1.7027 (−0.081)† | · | · | · |
| KD_MUONH_A0P8 | 2.1778 (−0.131) | 1.8838 (−0.095) | 1.6995 (−0.084)† | · | · | · |
| KD_MUONH_A0P9 | 2.1771 (−0.132) | 1.8802 (−0.098) | 1.6988 (−0.085)† | · | · | · |

**codefresh_decon — packed**

| arm | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| BASE | 1.8260 | 1.4728 | 1.2859 | 1.1625 | 1.0860 | · |
| OPT_MUON | 1.7463 (−0.080) | 1.4416 (−0.031) | 1.2627 (−0.023) | 1.1483 (−0.014) | · | · |
| OPT_MUONH | 1.7322 (−0.094) | 1.4325 (−0.040) | 1.2637 (−0.022) | 1.1515 (−0.011) | · | · |
| KD_MUONH_A0P7 | 1.6891 (−0.137) | 1.3847 (−0.088) | 1.2136 (−0.072)† | · | · | · |
| KD_MUONH_A0P8 | 1.6846 (−0.141) | 1.3821 (−0.091) | 1.2108 (−0.075)† | · | · | · |
| KD_MUONH_A0P9 | 1.6843 (−0.142) | 1.3806 (−0.092) | 1.2114 (−0.074)† | · | · | · |

**v3small_decon — packed**

| arm | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| BASE | 3.0720 | 2.7598 | 2.5475 | 2.4216 | 2.3544 | · |
| OPT_MUON | 3.0153 (−0.057) | 2.7261 (−0.034) | 2.5256 (−0.022) | 2.4054 (−0.016) | · | · |
| OPT_MUONH | 3.0008 (−0.071) | 2.7148 (−0.045) | 2.5213 (−0.026) | 2.4008 (−0.021) | · | · |
| KD_MUONH_A0P7 | 2.9643 (−0.108) | 2.6762 (−0.084) | 2.4799 (−0.068)† | · | · | · |
| KD_MUONH_A0P8 | 2.9646 (−0.107) | 2.6784 (−0.081) | 2.4804 (−0.067)† | · | · | · |
| KD_MUONH_A0P9 | 2.9669 (−0.105) | 2.6786 (−0.081) | 2.4821 (−0.065)† | · | · | · |

**pile_test_decon — packed**

| arm | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| BASE | 2.4974 | 2.1927 | 2.0196 | 1.8984 | 1.8142 | · |
| OPT_MUON | 2.4420 (−0.055) | 2.1661 (−0.027) | 1.9962 (−0.023) | 1.8814 (−0.017) | · | · |
| OPT_MUONH | 2.4237 (−0.074) | 2.1542 (−0.038) | 1.9906 (−0.029) | 1.8747 (−0.024) | · | · |
| KD_MUONH_A0P7 | 2.3885 (−0.109) | 2.1150 (−0.078) | 1.9487 (−0.071)† | · | · | · |
| KD_MUONH_A0P8 | 2.3884 (−0.109) | 2.1158 (−0.077) | 1.9491 (−0.070)† | · | · | · |
| KD_MUONH_A0P9 | 2.3889 (−0.108) | 2.1172 (−0.075) | 1.9504 (−0.069)† | · | · | · |

**pile_10k_decon — packed**

| arm | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| BASE | 2.5236 | 2.1964 | 2.0124 | 1.8838 | 1.7875 | · |
| OPT_MUON | 2.4668 (−0.057) | 2.1718 (−0.025) | 1.9898 (−0.023) | 1.8657 (−0.018) | · | · |
| OPT_MUONH | 2.4462 (−0.077) | 2.1603 (−0.036) | 1.9841 (−0.028) | 1.8607 (−0.023) | · | · |
| KD_MUONH_A0P7 | 2.4110 (−0.113) | 2.1195 (−0.077) | 1.9420 (−0.070)† | · | · | · |
| KD_MUONH_A0P8 | 2.4115 (−0.112) | 2.1197 (−0.077) | 1.9421 (−0.070)† | · | · | · |
| KD_MUONH_A0P9 | 2.4120 (−0.112) | 2.1198 (−0.077) | 1.9425 (−0.070)† | · | · | · |

**six-set mean — packed**

| arm | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| BASE | 2.3442 | 2.0312 | 1.8501 | 1.7293 | 1.6491 | · |
| OPT_MUON | 2.2839 (−0.060) | 2.0029 (−0.028) | 1.8283 (−0.022) | 1.7132 (−0.016) | · | · |
| OPT_MUONH | 2.2670 (−0.077) | 1.9925 (−0.039) | 1.8243 (−0.026) | 1.7104 (−0.019) | · | · |
| KD_MUONH_A0P7 | 2.2300 (−0.114) | 1.9508 (−0.080) | 1.7809 (−0.069)† | · | · | · |
| KD_MUONH_A0P8 | 2.2287 (−0.115) | 1.9510 (−0.080) | 1.7800 (−0.070)† | · | · | · |
| KD_MUONH_A0P9 | 2.2290 (−0.115) | 1.9503 (−0.081) | 1.7806 (−0.069)† | · | · | · |
<!-- settables:end -->

- KD's gain (α0.8, q1, vs MuonH) is largest on uncheatable / codefresh (−0.052 / −0.048) and smallest on v3small (−0.036).
- At q3, MuonH leads Muon only on val_v2, v3small and the two pile sets; it is already behind on uncheatable and codefresh.

---

## 7. Known issues / operations

- **None of Boxiang's runs use the evaluation settings we merged** (checked 10-22 18:30 on the argv of all 148 runs): none sets `VAL_PER_SPLIT=1` (per-split validation on the 9 decon suites incl. private_internal, with dpl's tw / macro / overall), and none uses `EVAL_AT_ITERS` / `EVAL_TPP_STEP`.
  He uses his own protocol: 6 pooled validation sets (val_v2 not split into cc / python / arxiv, no private_internal), in-loop eval every 3500 steps, doc-isolated eval run afterwards. So the numbers here are not directly comparable with dpl's metrics.
- **All q4 runs paused** (14:3x, user: "stop them all once the checkpoints are saved"): q4 is too slow (MuonH family at 1.34–1.40 s/it, ~41 h per run, the five runs held 160 nodes); the nodes go to the q0–q3 work first.
  Every run stopped right after a checkpoint save; resuming = resubmitting the chain. So the optimizer and KD arms have only q0–q3 for the fits for now.
- **q5 paused**: Muon / MuonH q5 (00:40, user), KD q5 ×3 (11:30, Boxiang: the hash check restarting every 20k steps wasted too much of 64 nodes). BASE q5 is the only one still running.
- **MuonH DP-replica divergence at EP 8: workaround in place**: `pretrain_hyperp.py` now re-syncs parameters and optimizer state every 1000 steps. With it, KD 0.9 q4 passed the 20000-step check with zero drift.
  Telemetry: drift appears only in the 4 EP groups whose two nodes share a rack (a multi-node NVLink domain); the 12 groups spanning two racks never drifted; 1–2 bf16 ulp per 1000 steps; unrelated to checkpoint saves.
  **MNNVL hypothesis refuted** (17:1x): fp32 all-reduce bit tests at 16 and 128 ranks with the training's NCCL environment, MNNVL on and off, found no rank ever differing from rank 0.
  New leading suspect: a timing race under `--overlap-grad-reduce` — a rank reads main-grad before its bucket's all-reduce has landed, and same-rack EP groups hit a different timing, so they are more exposed.
  Planned test: q4 layout, overlap on vs off, ~2000 steps each (re-sync every 100 steps). **MuonH q0–q3 (EP 4) are unaffected.**
- **MuonH recipe test abandoned** (launched 13:00, all cancelled and checkpoints deleted by 15:35): to explain MuonH's shrinking lead, three arms × q0–q3 tested LR 2×, LR 0.5× and the router on AdamW.
  Only mid-run readings exist: at q0, 2× is 0.11 worse and 0.5× is 0.02 worse, so the HyperP LR law looks near-optimal at q0; RTADAMW was misconfigured (router at the HyperP LR, routing collapsed). **No verdict**; q1–q3 can only be restarted from scratch.
  Follow-up (TODO 3 prep, **not launched**): experiment files and a report skeleton for a narrower LR bracket (×0.7 / ×1.4) are ready.
- **KD q4 stays paused with the other q4 chains** (15:54, Boxiang); they resume together when the pause is lifted.
- **New reports from Boxiang** in `reports/`: why MuonH is slow and how to make it fast (timers, micro-benchmarks, A/B/C plan; f3f8185), and a design review of an SWA hybrid-attention arm (research only, not launched; 85374fe).

---

## 8. Changelog

| date | update |
|---|---|
| 2026-10-22 07:40 | Table created; primary metric set to doc-isolated val_v2_decon. Doc results available: BASE / Muon / MuonH q0–q2. Added MuonH q3, KD q1 (packed), TODO 2 q0 (inside the noise), q5 pause, EP 8 divergence issue |
| 2026-10-22 10:15 | Synced to 8b56385. New doc-isolated: KD q0 ×3, KD q1 α0.7 / α0.9. BASE q4 finished (packed 1.2945). TODO 2 q1 B64Y / S64 finished. MuonH q4 on hold; KD 0.9 q4 and 0.7 q5 chains exhausted |
| 2026-10-22 15:45 | Synced to 75c268f (new layout: `results/<Change ID>/`). **New doc-isolated: BASE q3 / q4, Muon q3, MuonH q3, KD α0.8 q1**: at q3 Muon −0.012, MuonH −0.018 vs BASE; MuonH's lead over Muon down to −0.004 (six-set mean). **KD q2 finished (α0.7 / 0.8, packed −0.031 vs MuonH)**; KD's gain grows slightly with scale. All q4 paused, KD q5 stopped; re-sync workaround for the EP 8 divergence; MuonH recipe test abandoned |
| 2026-10-22 18:05 | Synced to 908f38f. No new doc-isolated results. **KD 0.9 q2 finished (packed 1.3987, −0.031 vs MuonH)**; all three alphas at −0.031 at q2. KD q3 at 84–93 %. TODO 2 B128 q1 died at 85 % on an NCCL timeout, no resume segment. The MNNVL hypothesis for the EP 8 divergence was refuted by a bit test; new suspect is a race in the overlapped grad-reduce |
| 2026-10-22 18:30 | Section 5 now has per-set results for all six validation sets (7 tables each for doc-isolated and packed), generated by `reports_xx/gen_tables.py`. Section 7: none of Boxiang's runs use our per-split / TPP evaluation settings |
