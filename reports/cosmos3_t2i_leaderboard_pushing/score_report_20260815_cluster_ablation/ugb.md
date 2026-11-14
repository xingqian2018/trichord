# UGB Score Report — Cluster Balanced Ablation

## Launch UGBGen

```BASH
CONTAINER_WORKDIR=/home/xingqianx/Project/imaginaire4_alt \
bash $HOME/Project/bashrc/sbatch_launch/main.sh small 8 ugb_gen_cluster_balanced_iter50k \
    projects/cosmos3/cosmos3/evaluation/text_to_image/inference_ugb_i4.py \
    --experiment_name cosmos3p5_ga_60bm30b_base_480 \
    --checkpoint_path s3://nv-00-10206-checkpoint-experiments/cosmos3_generator/cosmos3plus_t2ionly/cosmos3p5_ga_60bm30b_t2ionly_moe_exp001_002_cluster_balanced_n32xgpu4xbs4/checkpoints/iter_000050000/model \
    --credential_path credentials/gcs.secret \
    --benchmark_name v2_1170L_opus \
    --benchmark_credential_path credentials/gcs.secret \
    --num_batch_size 4 --guidance 4.0 --num_inference_steps 50 \
    --height 640 --width 640 \
    --use_ema --use_cosmos3_negative_prompt \
    --output_path s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cluster_balancing/v2_1170L_opus/cosmos3p5_ga_60bm30b_t2ionly_moe_exp001_002_cluster_balanced_n32xgpu4xbs4_iter50k \
    --output_credential_path credentials/gcs.secret
```

## v2_1170L_opus

| Run                                                                                      | Guidance | Images   | all   | orig  | phi   | success   |
|-------------------------------------------------------------------------------------------|----------|----------|-------|-------|-------|-----------|
| cosmos3p5_ga_60bm30b_t2ionly_moe_exp000_002_pretrain_n32xgpu4xbs4_iter20k                 | 4        | 1170 png | 68.64 | 71.25 | 66.25 | 1170/1170 |
| cosmos3p5_ga_60bm30b_t2ionly_moe_exp001_002_cluster_balanced_n32xgpu4xbs4_iter20k         | 4        | 1170 png | 63.55 | 65.71 | 61.57 | 1170/1170 |
| cosmos3p5_ga_60bm30b_t2ionly_moe_exp000_002_pretrain_n32xgpu4xbs4_iter50k                 | 4        | 1170 png | 76.59 | 79.18 | 74.21 | 1170/1170 |
| cosmos3p5_ga_60bm30b_t2ionly_moe_exp001_002_cluster_balanced_n32xgpu4xbs4_iter50k         | 4        | 1170 png | 74.22 | 77.07 | 71.61 | 1170/1170 |

