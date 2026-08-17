# Golden Caption Battle Number Ablation

**GCS Path:** `gcs:nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/exp_20260612/debug/`

## Output Count by Battle Number and Stage

| Battle Num   | Subfolder                               | Stage 1   | Stage 2   | Stage 3   | Stage 4   | Stage 5   | Tier 0   | Tier 1   | Tier 2   | Tier 3   | Tier 4   |
|--------------|-----------------------------------------|-----------|-----------|-----------|-----------|-----------|----------|----------|----------|----------|----------|
| BN=5         | stage{N}                                | 300       | 300       | 300       | 300       | 300       | 300      | 300      | 300      | 300      | 300      |
| BN=4         | stage{N}_ablation_battle_4              | 300       | 300       | 300       | -         | 300       | 300      | 300      | 300      | 300      | 300      |
| BN=3         | stage{N}_ablation_battle_3              | 300       | 300       | 300       | -         | 300       | 300      | 300      | 300      | 300      | 300      |
| BN=2         | stage{N}_ablation_battle_2              | 300       | 300       | 300       | -         | 300       | 300      | 300      | 300      | 300      | 300      |
| BN=1         | stage{N}_ablation_battle_1              | 300       | 300       | 300       | -         | 300       | 300      | 300      | 300      | 300      | 300      |
| BN=0         | stage{N}_ablation_battle_0              | 300       | 300       | 300       | -         | 300       | 300      | 300      | 300      | 300      | 300      |
| MX Tier0     | exp_20260707/debug/mx_tier0_captioning/ | 300       | -         | -         | -         | -         | 300      | -        | -        | -        | -        |
| MX Tier1     | exp_20260707/debug/mx_tier1_captioning/ | 300       | -         | -         | -         | -         | 300      | -        | -        | -        | -        |

> **Note:** Stage 4 subfolder is always `stage4` across all battle number variants — it is shared and not ablated.

## Precision by Battle Number and Tier

| Battle Num   | Settings   | Tier 0                         | Tier 1                         | Tier 2                         | Tier 3                           | Tier 4                           |
|--------------|------------|--------------------------------|--------------------------------|--------------------------------|----------------------------------|----------------------------------|
| BN=5         |            | 0.9765 / 492/492 / 54504/54692 | 0.9604 / 774/775 / 90942/91225 | 0.9625 / 735/735 / 85058/86173 | 0.9646 / 733/735 / 107879/110911 | 0.9629 / 735/735 / 106392/106862 |
| BN=4         |            |                                |                                |                                |                                  |                                  |
| BN=3         |            |                                |                                |                                |                                  |                                  |
| BN=2         |            | 0.9793 / 435/436 / 44415/44543 | 0.9675 / 650/651 / 75120/76446 | 0.9694 / 623/623 / 71513/73029 | 0.9706 / 621/623 / 92163/92922   | 0.9704 / 623/623 / 88336/89017   |
| BN=1         |            | 0.9823 / 413/413 / 40302/40302 | 0.9698 / 612/613 / 68718/69623 | 0.9688 / 585/585 / 64999/66510 | 0.9700 / 585/585 / 83905/85101   | 0.9709 / 585/585 / 80323/81927   |
| BN=0         |            | 0.9817 / 381/381 / 36962/36962 | 0.9705 / 569/570 / 62385/62618 | 0.9691 / 556/556 / 59265/59951 | 0.9720 / 555/556 / 75718/76259   | 0.9719 / 556/556 / 72722/73235   |

## Precision by Caption Type and Setting

| Battle Num                   | Settings                    | Scores / Success Cnt           |
|------------------------------|-----------------------------|--------------------------------|
| BN 1 Tier 1                  | gemini                      | 0.9698 / 612/613 / 68718/69623 |
| MX Tier0                     | gemini                      | 0.9872 / 300/300 / 16875/16932 |
| MX Tier1                     | gemini                      | 0.9836 / 300/300 / 26624/26730 |
| MX Tier1s21p                 | gemini                      | 0.9433 / 319/319 / 41948/42206 |
| MX Tier1s21p                 | gpt-5.5                     | 0.9473 / 319/319 / 32723/32723 |
| BN 1 Tier 1                  | opus decomp + gemini judge  | 0.9741 / 613/613 / 50582/50718 |
| MX Tier1                     | opus decomp + gemini judge  | 0.9812 / 300/300 / 16006/16022 |
| MX Tier1s21p                 | opus decomp + gemini judge  | 0.9413 / 319/319 / 25062/25177 |
| MX Tier1 qwen35-v0           | opus decomp + gemini judge  | 0.9486 / 307/307 / 15725/15815 |
| MX Tier1 qwen35-v1           | opus decomp + gemini judge  | 0.9456 / 306/306 / 14602/14638 |
| old_json                     | opus decomp + gemini judge  | 0.9527 / 300/300 / 25391/25435 |
| BN 1 Tier 1                  | opus decomp + gpt-5.5 judge | 0.9338 / 613/613 / 50530/50530 |
| MX Tier1                     | opus decomp + gpt-5.5 judge | 0.9554 / 300/300 / 15913/15913 |
| MX Tier1s21p                 | opus decomp + gpt-5.5 judge | 0.9502 / 319/319 / 25194/25194 |
| MX Tier1 qwen35-v0           | opus decomp + gpt-5.5 judge | 0.9563 / 307/307 / 15770/15857 |
| MX Tier1 qwen35-v1           | opus decomp + gpt-5.5 judge | 0.9599 / 306/306 / 14581/14674 |
| old_json                     | opus decomp + gpt-5.5 judge | 0.9260 / 300/300 / 25347/25347 |
| hamid_snah (gemini)          | opus decomp + gemini judge  | 0.9819 / 300/300 / 15377/15379 |
| hamid_snah (gemini)          | opus decomp + gpt-5.5 judge | 0.9568 / 300/300 / 15382/15382 |
| hamid_snah (gpt-5.5)         | opus decomp + gemini judge  | 0.9550 / 300/300 / 24465/24474 |
| hamid_snah (gpt-5.5)         | opus decomp + gpt-5.5 judge | 0.9891 / 300/300 / 24353/24476 |
| hamid_snah gpt55 r2048       | opus decomp + gemini judge  | 0.9754 / 300/300 / 24323/24341 |
| hamid_snah gpt55 r2048       | opus decomp + gpt-5.5 judge | 0.9913 / 300/300 / 24237/24341 |
| hamid_snah gemini r2048      | opus decomp + gemini judge  | 0.9794 / 300/300 / 15490/15492 |
| hamid_snah gemini r2048      | opus decomp + gpt-5.5 judge | 0.9684 / 300/300 / 15492/15492 |
| hamid_snah nocot gpt55 r2048 | opus decomp + gemini judge  | 0.9681 / 300/300 / 18671/18777 |
| hamid_snah nocot gpt55 r2048 | opus decomp + gpt-5.5 judge | 0.9887 / 300/300 / 18660/18777 |
| hamid_snah qwen v2 r2048     | opus decomp + gemini judge  | 0.9750 / 300/300 / 21564/21619 |
| hamid_snah qwen v2 r2048     | opus decomp + gpt-5.5 judge | 0.9845 / 300/300 / 21619/21619 |
| hamid_snah qwen v3 r2048     | opus decomp + gemini judge  | 0.9772 / 300/300 / 22952/22992 |
| hamid_snah qwen v3 r2048     | opus decomp + gpt-5.5 judge | 0.9866 / 300/300 / 22992/22992 |
| hamid_snah nocot v2 r2048    | opus decomp + gemini judge  | 0.9732 / 300/300 / 21514/21577 |
| hamid_snah nocot v2 r2048    | opus decomp + gpt-5.5 judge | 0.9842 / 300/300 / 21577/21577 |
| hamid_snah nocot v3 r2048    | opus decomp + gemini judge  | 0.9760 / 300/300 / 22962/23018 |
| hamid_snah nocot v3 r2048    | opus decomp + gpt-5.5 judge | 0.9846 / 300/300 / 23018/23018 |
| MX Tier1s21p                 | opus decomp + s21p judge    | 0.9740 / 319/319 / 23453/25204 |




| Setting             | Settings                    | Scores / Claim Cnt |
|---------------------|-----------------------------|--------------------|
| hamid_snah v2       | opus decomp + gemini judge  | 0.9750 / 21619     |
| hamid_snah v3       | opus decomp + gemini judge  | 0.9772 / 22992     |
| hamid_snah nocot v2 | opus decomp + gemini judge  | 0.9732 / 21577     |
| hamid_snah nocot v3 | opus decomp + gemini judge  | 0.9760 / 23018     |
|---------------------|-----------------------------|--------------------|
| hamid_snah v2       | opus decomp + gpt-5.5 judge | 0.9845 / 21619     |
| hamid_snah v3       | opus decomp + gpt-5.5 judge | 0.9866 / 22992     |
| hamid_snah nocot v2 | opus decomp + gpt-5.5 judge | 0.9842 / 21577     |
| hamid_snah nocot v3 | opus decomp + gpt-5.5 judge | 0.9846 / 23018     |



> Format: `precision / success_decompose / success_claim_judged-total_claims`

## Recall by Battle Number and Tier

| Battle Num         | Tier 0                       | Tier 1                       | Tier 2                       | Tier 3                       | Tier 4                       |
|--------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| BN=5               |                              |                              |                              |                              |                              |
| BN=4               |                              |                              |                              |                              |                              |
| BN=3               |                              |                              |                              |                              |                              |
| BN=2               |                              |                              |                              |                              |                              |
| BN=1               | 0.4063 / 286/294 / 1869/4600 | 0.5513 / 285/294 / 2505/4544 | 0.5498 / 283/294 / 2483/4516 | 0.5555 / 292/294 / 2627/4729 | 0.5536 / 291/294 / 2607/4709 |
| BN=0               |                              |                              |                              |                              |                              |
| MX Tier0           | 0.4157 / 291/294 / 1964/4725 |                              |                              |                              |                              |
| MX Tier1           |                              | 0.5433 / 292/294 / 2576/4741 |                              |                              |                              |
| MX Tier1 seed21pro |                              | 0.5537 / 286/294 / 2526/4562 |                              |                              |                              |

## Recall Incorrect / Missing Ratio by Battle Number and Tier

> Format: `incorrect_ratio / missing_ratio`

| Battle Num         | Tier 0          | Tier 1          | Tier 2          | Tier 3          | Tier 4          |
|--------------------|-----------------|-----------------|-----------------|-----------------|-----------------|
| BN=5               |                 |                 |                 |                 |                 |
| BN=4               |                 |                 |                 |                 |                 |
| BN=3               |                 |                 |                 |                 |                 |
| BN=2               |                 |                 |                 |                 |                 |
| BN=1               | 0.1896 / 0.4041 | 0.2716 / 0.1772 | 0.2755 / 0.1747 | 0.2787 / 0.1658 | 0.2850 / 0.1614 |
| BN=0               |                 |                 |                 |                 |                 |
| MX Tier0           | 0.1733 / 0.4110 |                 |                 |                 |                 |
| MX Tier1           |                 | 0.2584 / 0.1983 |                 |                 |                 |
| MX Tier1 seed21pro |                 | 0.2854 / 0.1609 |                 |                 |                 |

## Entity Count by Battle Number

| Battle Num   | Images   | Total Entities   | Avg Entities/Img   | Min   | Max   |
|--------------|----------|------------------|--------------------|-------|-------|
| BN=5         | 300      | 8125             | 27.08              | 2     | 75    |
| BN=4         | 300      | 7531             | 25.1               | 2     | 71    |
| BN=3         | 300      | 6922             | 23.07              | 2     | 57    |
| BN=2         | 300      | 6321             | 21.07              | 2     | 56    |
| BN=1         | 300      | 5729             | 19.1               | 2     | 53    |
| BN=0         | 300      | 5110             | 17.03              | 2     | 49    |

> Counted from `entity_list` in stage5 tier4 JSONs. Higher BN → more battle refinement → more entities surfaced.

---

## Thinking Model Ablation (exp_20260707, BN=1, gemini-3.1-pro@nvidia)

**GCS Path:** `gcs:nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/exp_20260707/debug/`

### Output Count

| Variant        | Stage 1   | Stage 2   | Stage 3   | Stage 4   | Stage 5   |
|----------------|-----------|-----------|-----------|-----------|-----------|
| thinking_low   | 300       | 300       | 300       | 300       | 300       |
| thinking_high  | 300       | 300       | 300       | 300       | 300       |
| gemini35_flash | 300       | 300       | 300       | 300       | 300       |
| seed21pro      | 300       | 240       |           | 300       |           |

### Abalation for Precision API Call setting / VLM choice

> Format: `precision / success_decompose / success_claim_judged`

| Variant             | Downsampled Dense              | Dense                          | Structured Dense               | Structured                     | Structured + Dense             |
|---------------------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|
| BN=1                | 0.9823 / 413/413 / 40302/40302 | 0.9698 / 612/613 / 68718/69623 | 0.9688 / 585/585 / 64999/66510 | 0.9700 / 585/585 / 83905/85101 | 0.9709 / 585/585 / 80323/81927 |
| BN=1 thinking_low   | 0.9805 / 405/405 / 41677/41677 | 0.9690 / 639/639 / 72060/72298 | 0.9692 / 595/595 / 66686/66980 | 0.9717 / 595/595 / 86394/86522 | 0.9697 / 595/595 / 82288/82416 |
| BN=1 thinking_high  | 0.9782 / 405/405 / 40906/40908 | 0.9657 / 628/628 / 70852/70980 | 0.9675 / 587/587 / 66559/66591 | 0.9683 / 587/587 / 86778/86778 | 0.9663 / 587/587 / 83335/83335 |
| BN=1 gemini35_flash | 0.9757 / 303/303 / 24138/24138 | 0.9559 / 469/469 / 48885/48885 | 0.9549 / 379/379 / 43416/43416 | 0.9586 / 379/379 / 55327/55327 | 0.9549 / 379/379 / 53971/53971 |
| BN=1 seed21pro      |                                |                                |                                |                                |                                |

## Result file path verbose

### hamid_snah_gpt55_r2048

**Caption folder / Eval results**
`gcs:nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/exp_20260810/hamid_snah_capbalance_gpt55_resize2048/`
`gcs:nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/exp_20260810_result/hamid_snah_capbalance_gpt55_resize2048/`

| File                                                                      | Description                                                    |
|---------------------------------------------------------------------------|----------------------------------------------------------------|
| `precision_eval_decomposed_hamid_snah_gpt55_r2048_opus_gemini_judge.json` | Decomposed claims (opus-4.7 decomp) — reused across judge runs |
| `precision_eval_result_hamid_snah_gpt55_r2048_opus_gemini_judge.json`     | Precision result — gemini-3.1-pro judge                        |
| `precision_eval_claims_hamid_snah_gpt55_r2048_opus_gemini_judge.json`     | Per-image claims — gemini-3.1-pro judge                        |
| `precision_eval_result_hamid_snah_gpt55_r2048_opus_gpt55judge.json`       | Precision result — gpt-5.5 judge                               |
| `precision_eval_claims_hamid_snah_gpt55_r2048_opus_gpt55judge.json`       | Per-image claims — gpt-5.5 judge                               |

## Worksheet

### Stage 2

```bash
PYTHONPATH=/home/xingqianx/Project/imaginaire4 \
python -m torch.distributed.run --nproc_per_node=1 --master_port=24823 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage2_structured_captioning.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/V1/ \
    --input_credential credentials/gcs.secret \
    --input_entity_list_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/exp_20260612/debug/stage1_ablation_battle_2/ \
    --output_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/exp_20260612/debug/stage2_ablation_battle_2/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 800 \
    --timeout 200 \
    --max_retry 3 \
    --max_battle_rounds 2 \
    --force_judge_model gemini-3.1-pro \
    --force_gen_model gemini-3.1-pro
```

### Stage 3

```bash
PYTHONPATH=/home/xingqianx/Project/imaginaire4 \
python -m torch.distributed.run --nproc_per_node=1 --master_port=24823 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/stage3_dense_captioning.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/exp_20260612/debug/stage2/ \
    --input_credential credentials/gcs.secret \
    --output_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/exp_20260612/debug/stage3/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 1000 \
    --timeout 200 \
    --max_retry 3 \
    --force_gen_model gemini-3.1-pro@nvidiak
```

### MXTier1

```bash
cd ~/Project/imaginaire4
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4 \
bash $HOME/Project/bashrc/sbatch_launch/main.sh cpu 1x1 golden_caption_mx_tier1_0707 \
    projects/cosmos3/vfm/evaluation/captioning/golden_caption/mx_tier1_captioning.py \
    --input_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/V1/ \
    --input_credential credentials/gcs.secret \
    --output_folder s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/exp_20260707/debug/mx_tier1_captioning/ \
    --output_credential credentials/gcs.secret \
    --num_concurrency 128 \
    --batch_size 300 \
    --timeout 1000 \
    --max_retry 3 \
    --force_model gemini-3.1-pro@nvidiak
```

### Precision Eval

```bash
PYTHONPATH=/home/xingqianx/Project/imaginaire4 \
python -m torch.distributed.run --nproc_per_node=1 --master_port=25730 \
    projects/cosmos3/vfm/evaluation/captioning/evaluate_image_caption_precision.py \
    --image_dir s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/V1/ \
    --image_cred credentials/gcs.secret \
    --caption_dir s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/exp_20260612/debug/stage5_ablation_battle_0_tier0/ \
    --caption_cred credentials/gcs.secret \
    --output_dir s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/exp_20260612/debug/eval_results/ \
    --output_cred credentials/gcs.secret \
    --judge_model gemini-3.1-pro \
    --batch_size 2000 \
    --num_concurrency 128 \
    --signature abl_bn0_t0
```

### Recall Eval

```bash
PYTHONPATH=/home/xingqianx/Project/imaginaire4 \
python -m torch.distributed.run --nproc_per_node=1 --master_port=25731 \
    projects/cosmos3/vfm/evaluation/captioning/evaluate_image_caption_recall.py \
    --assertion_dir s3://nv-00-10206-vfm/debug/xingqianx/evaluation/CosCapBenchImage/V1_assertion/ \
    --assertion_cred credentials/gcs.secret \
    --caption_dir s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/exp_20260612/debug/stage5_ablation_battle_2_tier1/ \
    --caption_cred credentials/gcs.secret \
    --output_dir s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/exp_20260612/debug/eval_results/ \
    --output_cred credentials/gcs.secret \
    --judge_model gemini-3.1-pro \
    --batch_size 1000 \
    --num_concurrency 128 \
    --signature abl_bn2_t1
```
