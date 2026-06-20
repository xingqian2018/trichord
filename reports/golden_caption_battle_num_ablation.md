# Golden Caption Battle Number Ablation

**GCS Path:** `gcs:nv-00-10206-vfm/debug/xingqianx/evaluation_results/golden_caption/exp_20260612/debug/`

## Output Count by Battle Number and Stage

| Battle Num   | Subfolder                  | Stage 1   | Stage 2   | Stage 3   | Stage 4   | Stage 5   | Tier 0   | Tier 1   | Tier 2   | Tier 3   | Tier 4   |
|--------------|----------------------------|-----------|-----------|-----------|-----------|-----------|----------|----------|----------|----------|----------|
| BN=5         | stage{N}                   | 300       | 300       | 300       | 300       | 300       | 300      | 300      | 300      | 300      | 300      |
| BN=4         | stage{N}_ablation_battle_4 | 300       | 300       | 300       | -         | 300       | 300      | 300      | 300      | 300      | 300      |
| BN=3         | stage{N}_ablation_battle_3 | 300       | 300       | 300       | -         | 300       | 300      | 300      | 300      | 300      | 300      |
| BN=2         | stage{N}_ablation_battle_2 | 300       | 300       | 300       | -         | 300       | 300      | 300      | 300      | 300      | 300      |
| BN=1         | stage{N}_ablation_battle_1 | 300       | 300       | 300       | -         | 300       | 300      | 300      | 300      | 300      | 300      |
| BN=0         | stage{N}_ablation_battle_0 | 300       | 300       | 300       | -         | 300       | 300      | 300      | 300      | 300      | 300      |

> **Note:** Stage 4 subfolder is always `stage4` across all battle number variants — it is shared and not ablated.

## Precision by Battle Number and Tier

| Battle Num   | Tier 0                         | Tier 1                         | Tier 2                         | Tier 3                         | Tier 4                         |
|--------------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|--------------------------------|
| BN=5         | 0.9765 / 492/492 / 54504/54692 | 0.9604 / 774/775 / 90942/91225 | 0.9625 / 735/735 / 85058/86173 | 0.9646 / 733/735 / 107879/110911 | 0.9629 / 735/735 / 106392/106862 |
| BN=4         |                                |                                |                                |                                |                                |
| BN=3         |                                |                                |                                |                                |                                |
| BN=2         | 0.9793 / 435/436 / 44415/44543 | 0.9675 / 650/651 / 75120/76446 | 0.9694 / 623/623 / 71513/73029 | 0.9706 / 621/623 / 92163/92922 | 0.9704 / 623/623 / 88336/89017 |
| BN=1         | 0.9823 / 413/413 / 40302/40302 | 0.9698 / 612/613 / 68718/69623 | 0.9688 / 585/585 / 64999/66510 | 0.9700 / 585/585 / 83905/85101 | 0.9709 / 585/585 / 80323/81927 |
| BN=0         | 0.9817 / 381/381 / 36962/36962 | 0.9705 / 569/570 / 62385/62618 | 0.9691 / 556/556 / 59265/59951 | 0.9720 / 555/556 / 75718/76259 | 0.9719 / 556/556 / 72722/73235 |

> Format: `precision / success_decompose / success_claim_judged-total_claims`

## Recall by Battle Number and Tier

| Battle Num   | Tier 0   | Tier 1   | Tier 2   | Tier 3   | Tier 4   |
|--------------|----------|----------|----------|----------|----------|
| BN=5         |          |          |          |          |          |
| BN=4         |          |          |          |          |          |
| BN=3         |          |          |          |          |          |
| BN=2         |          |          |          |          |          |
| BN=1         |          |          |          |          |          |
| BN=0         |          |          |          |          |          |

## Entity Count by Battle Number

| Battle Num | Images | Total Entities | Avg Entities/Img | Min | Max |
|------------|--------|----------------|------------------|-----|-----|
| BN=5       | 300    | 8125           | 27.08            | 2   | 75  |
| BN=4       | 300    | 7531           | 25.10            | 2   | 71  |
| BN=3       | 300    | 6922           | 23.07            | 2   | 57  |
| BN=2       | 300    | 6321           | 21.07            | 2   | 56  |
| BN=1       | 300    | 5729           | 19.10            | 2   | 53  |
| BN=0       | 300    | 5110           | 17.03            | 2   | 49  |

> Counted from `entity_list` in stage5 tier4 JSONs. Higher BN → more battle refinement → more entities surfaced.

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
    --force_gen_model gemini-3.1-pro
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
    --timeout 200 \
    --signature abl_bn2_t1
```
