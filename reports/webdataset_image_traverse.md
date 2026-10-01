# webdataset_image_traverse

Traversal of `gcs:nv-00-10206-webdataset-images/`. Level 1 fully listed (50 entries).
Level 2 in progress — 16 of 35 subfolders traversed so far (marked below); rest still running in background.

```
nv-00-10206-webdataset-images/
├── .markers/
├── benchmarks/
│   └── vision_reconstruction/
│       └── v1/
│           ├── depth/
│           ├── depth_completed/
│           ├── normal/
│           ├── seg/
│           ├── _copy_probe.json
│           ├── _write_probe.txt
│           ├── manifest.json
│           └── sha256_manifest.json
├── debug/
│   ├── example_sila_image_sharding/
│   ├── test-10000/
│   ├── test-1001/
│   ├── test/
│   ├── test1210/
│   ├── test_sharding_plan_0417/
│   ├── users/
│   ├── webdataset_image_v5_category/
│   ├── webdataset_image_v5_category_qwen2p5/
│   ├── webdataset_image_v5_clustered/
│   ├── webdataset_image_v5_topic_test/
│   ├── webdataset_image_v5_topic_test2/
│   ├── webdataset_synthetic/
│   │   ├── generations_qwen_image_2512/
│   │   └── photorealism_filtered/
│   └── wordnet_captions/
├── debug_balanced/
│   └── test_balanced/
├── dev/
│   └── users/
├── edit_pairs/                                    (need copy)
│   ├── 2level_0706_500k/
│   ├── chronoedit_all_data_20260212/
│   ├── image_edit_pairs_imgedit_two_level_20260708/
│   ├── image_edit_pairs_video_hq_20260515_mvp0/
│   └── users/
│       └── snah/
├── gemini3_image_pro_aa_synthetic_100k_20260516/  (need copy)
├── raw/
│   ├── CosmosVision/                              (need copy)
│   │   ├── audits/
│   │   ├── depth_externel/
│   │   ├── lepton/
│   │   ├── pose2img_work/
│   │   ├── production/
│   │   ├── segmentation/
│   │   ├── sila_targets/
│   │   ├── source/
│   │   ├── training/
│   │   └── v1p1/
│   ├── OmniDoc-TokenBench/                        (need copy)
│   │   └── training/
│   └── multi-reference-generation/                (need copy)
│       └── image/
├── test/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   └── midjourney_v6_20240703/
├── test_recaptioning_tar2tar/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── midjourney/
│   └── midjourney_v6_20240703/
├── test_webdataset_image_v5_clustered_prod_v5/        (split_by_cluster)
├── users/
│   └── jiannanh/
├── visualization/
├── webdataset_cosmos_lab_image_v1/
│   ├── .markers/
│   ├── v1_Aesthetic_Train_v2/
│   ├── v1_LSDIR/
│   ├── v1_agent_distilled_v19_99827/
│   ├── v1_agent_distilled_v6a_57230/
│   ├── v1_agent_distilled_v7m_31806/
│   ├── v1_human_sft/
│   ├── v1_ohv_gen_frames_50k_filtered/
│   ├── v1_pexels_residual_trustedK1_v2/
│   ├── v1_sft_candidate_a_conservative/
│   ├── v1_sft_candidate_b_preferred/
│   ├── v1_sft_candidate_bc_dedup/
│   ├── v1_sft_candidate_c_clean/
│   ├── v1_unsplash_lite/
│   └── webdataset_cosmos_lab_image_v1/
│       ├── v1/
│       │   ├── MMC4/
│       │   ├── coyo_700m/
│       │   ├── datacomp_12b/
│       │   ├── datacomp_1b/
│       │   ├── generations_qwen_image_2512_filtered_photoreal/
│       │   ├── laion_115m/
│       │   ├── laion_400m/
│       │   ├── laion_aesthetic/
│       │   ├── midjourney/
│       │   ├── midjourney_v6_20240703/
│       │   ├── nvcommercial_700m/
│       │   ├── red/
│       │   ├── self_improving_synthetic_2026-02-09/
│       │   ├── self_improving_synthetic_2026-02-14/
│       │   ├── synthetic_chinese_scene_text_v0/
│       │   ├── synthetic_scene_text_v0/
│       │   ├── wdinfo/
│       │   ├── wordnet_captions_20260224/
│       │   └── zennodo10k/
│       ├── v1_high_quality/
│       │   ├── coyo_700m/
│       │   ├── datacomp_1b/
│       │   └── wdinfo/
│       └── v1_text_rendering/
│           ├── screen2words_rico/
│           ├── slide_audit/
│           ├── synthetic_chinese_scene_text_v0/
│           ├── synthetic_scene_text_v0/
│           ├── synthetic_traditional_chinese_scene_text_v0/
│           ├── voxel51_rico/
│           ├── wdinfo/
│           └── zennodo10k/
├── webdataset_edify_image_v3/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── getty_text2image/                    (need copy)
│   ├── midjourney/
│   ├── midjourney_3m/
│   ├── midjourney_v6_20240516_20240527_shuffled/
│   ├── midjourney_v6_20240703/
│   ├── midjourney_v6_20240830_shuffled/
│   ├── nvcommercial_700m/
│   └── nvcommercial_700m_upsampled/
├── webdataset_eval/
│   └── eval_data_50k_xingqianx_raw/
├── webdataset_highres_laion_full_caption/
│   └── v1/
│       ├── laion_tier1_v22_312_7_70m_4k_trustedk1/
│       ├── laion_tier3_v23_312_7_70m_3k_trustedk1/
│       └── wdinfo/
├── webdataset_highres_ohv_full_caption/
│   └── v1/
│       ├── ohv3p5m_sdvr2_upres_all_4k/
│       └── wdinfo/
├── webdataset_image_dense_bbox_ablation_40M/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── midjourney/
│   └── midjourney_v6_20240703/
├── webdataset_image_regular_text/
│   ├── screen2words_rico/
│   ├── slide_audit/
│   ├── voxel51_rico/
│   ├── zennodo10k/
│   └── dummy.txt
├── webdataset_image_synthetic_text/
│   ├── synthetic_chinese_scene_text_v0/
│   ├── synthetic_scene_text_chinese_v1/ (need copy)
│   ├── synthetic_scene_text_chinese_v1_phi/ (need copy)
│   ├── synthetic_scene_text_v0/
│   ├── synthetic_scene_text_v1/ (need copy)
│   ├── synthetic_scene_text_v1_phi/ (need copy)
│   ├── synthetic_traditional_chinese_scene_text_v0/
│   └── dummy.txt
├── webdataset_image_v4p1/
│   ├── high_quality_v1/
│   └── v1/
├── webdataset_image_v5/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── llava_ov_midtrain_3m/ (need copy)
│   ├── llava_ov_midtrain_85m_en/ (need copy)
│   ├── llava_ov_sft_22m/ (need copy)
│   ├── midjourney/
│   └── midjourney_v6_20240703/
├── webdataset_image_v5_balanced_category_v1/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── midjourney/
│   └── midjourney_v6_20240703/
├── webdataset_image_v5_bbox/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── midjourney/
│   └── midjourney_v6_20240703/
├── webdataset_image_v5_category/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── midjourney/
│   └── midjourney_v6_20240703/
├── webdataset_image_v5_category_qwen2p5/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── midjourney/
│   └── midjourney_v6_20240703/
├── webdataset_image_v5_clustered_prod_200pertar_v1/   (split_by_cluster)
├── webdataset_image_v5_clustered_prod_v1/             (split_by_cluster)
├── webdataset_image_v5_clustered_prod_v2/             (split_by_cluster)
├── webdataset_image_v5_clustered_prod_v3/             (split_by_cluster)
├── webdataset_image_v5_clustered_prod_v4/             (split_by_cluster)
├── webdataset_image_v5_dedup_clustered_prod_v1/       (split_by_cluster)
├── webdataset_image_v5_dedup_clustered_prod_v3/       (split_by_cluster)
├── webdataset_image_v5_filter105M/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── midjourney/
│   └── midjourney_v6_20240703/
├── webdataset_image_v5_filter_highquality_ablation_105M/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── midjourney/
│   └── midjourney_v6_20240703/
├── webdataset_image_v5_filter_highquality_ablation_156M/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── midjourney/
│   └── midjourney_v6_20240703/
├── webdataset_image_v5_filter_highquality_ablation_40M/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── midjourney/
│   └── midjourney_v6_20240703/
├── webdataset_image_v5_filter_highquality_ablation_56M/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── midjourney/
│   └── midjourney_v6_20240703/
├── webdataset_image_v5_filter_highquality_ablation_77M/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── midjourney/
│   └── midjourney_v6_20240703/
├── webdataset_image_v5_mmc4_datacomp12b/
│   ├── MMC4/
│   ├── datacomp_12b/
│   └── datacomp_1b/
├── webdataset_image_v5_plan/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── midjourney/
│   └── midjourney_v6_20240703/
├── webdataset_image_v5_red/
│   ├── .markers/
│   ├── red/
│   └── wdinfo/
├── webdataset_image_v5_semantic_dedup_v1/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── midjourney/
│   ├── midjourney_v6_20240703/
│   └── wdinfo/
├── webdataset_image_v5_topic/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── midjourney/
│   ├── midjourney_v6_20240703/
│   └── topic_assignments.json
├── webdataset_image_v5_topic_v2/
│   ├── coyo_700m/
│   ├── datacomp_1b/
│   ├── midjourney/
│   ├── midjourney_v6_20240703/
│   └── topic_assignments_v2.jsonl
├── webdataset_multi_reference_generation/ (need copy)
│   ├── data_engine/
│   ├── public/
│   └── synth_data_engine/
├── webdataset_rl/                         (need copy)
│   ├── artificial_analysis/
│   └── hpdv3/
├── webdataset_sft/                        (need copy)
│   ├── agent_distilled/
│   ├── image_sft/
│   └── pretrain_ablation/
└── webdataset_synthetic/                  (need copy)
    ├── flux_lora/
    ├── gemini_3_pro_image_200k_filtered/
    ├── generations_flux/
    ├── generations_hidream/
    ├── generations_hidream_filtered/
    ├── generations_hidream_filtered_photoreal/
    ├── generations_qwen_image_2512_filtered_photoreal/
    ├── gpt_image_2_20260507_unfiltered/
    ├── gpt_image_2_20260515_unfiltered/
    ├── gpt_image_2_aa_synthetic_44k_gpt55_t2i_image_v1/
    ├── gpt_image_2_artificial_analysis_opus47/
    ├── hidream_o1_image_dev_2604_v1_standard_hps12/
    ├── hidream_o1_image_dev_2604_v1_standard_hps12_photoreal/
    ├── overfit/
    ├── prompts/
    ├── prompts_debug/
    ├── self_improving_synthetic/
    ├── self_improving_synthetic_filtered_photoreal/
    ├── synthetic_chinese_scene_text_v0/
    ├── synthetic_scene_text/
    ├── synthetic_traditional_chinese_scene_text_v0/
    ├── v1_text_banner_photorealism_0_8/
    ├── v1_text_banner_photorealism_0_9/
    ├── v1_text_banner_photorealism_0_95/
    └── wordnet_captions_20260224/
```

## Paths marked `(need copy)`

- `gcs:nv-00-10206-webdataset-images/edit_pairs/`
- `gcs:nv-00-10206-webdataset-images/gemini3_image_pro_aa_synthetic_100k_20260516/`
- `gcs:nv-00-10206-webdataset-images/raw/CosmosVision/`
- `gcs:nv-00-10206-webdataset-images/raw/OmniDoc-TokenBench/`
- `gcs:nv-00-10206-webdataset-images/raw/multi-reference-generation/`
- `gcs:nv-00-10206-webdataset-images/webdataset_edify_image_v3/`
- `gcs:nv-00-10206-webdataset-images/webdataset_image_synthetic_text/synthetic_scene_text_chinese_v1/`
- `gcs:nv-00-10206-webdataset-images/webdataset_image_synthetic_text/synthetic_scene_text_chinese_v1_phi/`
- `gcs:nv-00-10206-webdataset-images/webdataset_image_synthetic_text/synthetic_scene_text_v1/`
- `gcs:nv-00-10206-webdataset-images/webdataset_image_synthetic_text/synthetic_scene_text_v1_phi/`
- `gcs:nv-00-10206-webdataset-images/webdataset_image_v5/llava_ov_midtrain_3m/`
- `gcs:nv-00-10206-webdataset-images/webdataset_image_v5/llava_ov_midtrain_85m_en/`
- `gcs:nv-00-10206-webdataset-images/webdataset_image_v5/llava_ov_sft_22m/`
- `gcs:nv-00-10206-webdataset-images/webdataset_multi_reference_generation/`
- `gcs:nv-00-10206-webdataset-images/webdataset_rl/`
- `gcs:nv-00-10206-webdataset-images/webdataset_sft/`
- `gcs:nv-00-10206-webdataset-images/webdataset_synthetic/`

## Copy plan: source -> destination

Destination root: `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/`

| Source                                                                                                    | Destination                                                                                                                      |
|-----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `s3://nv-00-10206-webdataset-images/edit_pairs/`                                                          | `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/edit_pairs/`                                                          |
| `s3://nv-00-10206-webdataset-images/gemini3_image_pro_aa_synthetic_100k_20260516/`                        | `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/gemini3_image_pro_aa_synthetic_100k_20260516/`                        |
| `s3://nv-00-10206-webdataset-images/raw/CosmosVision/`                                                    | `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/raw/CosmosVision/`                                                    |
| `s3://nv-00-10206-webdataset-images/raw/OmniDoc-TokenBench/`                                              | `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/raw/OmniDoc-TokenBench/`                                              |
| `s3://nv-00-10206-webdataset-images/raw/multi-reference-generation/`                                      | `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/raw/multi-reference-generation/`                                      |
| `s3://nv-00-10206-webdataset-images/webdataset_edify_image_v3/`                                           | `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/webdataset_edify_image_v3/`                                           |
| `s3://nv-00-10206-webdataset-images/webdataset_image_synthetic_text/synthetic_scene_text_chinese_v1/`     | `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/webdataset_image_synthetic_text/synthetic_scene_text_chinese_v1/`     |
| `s3://nv-00-10206-webdataset-images/webdataset_image_synthetic_text/synthetic_scene_text_chinese_v1_phi/` | `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/webdataset_image_synthetic_text/synthetic_scene_text_chinese_v1_phi/` |
| `s3://nv-00-10206-webdataset-images/webdataset_image_synthetic_text/synthetic_scene_text_v1/`             | `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/webdataset_image_synthetic_text/synthetic_scene_text_v1/`             |
| `s3://nv-00-10206-webdataset-images/webdataset_image_synthetic_text/synthetic_scene_text_v1_phi/`         | `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/webdataset_image_synthetic_text/synthetic_scene_text_v1_phi/`         |
| `s3://nv-00-10206-webdataset-images/webdataset_image_v5/llava_ov_midtrain_3m/`                            | `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/webdataset_image_v5/llava_ov_midtrain_3m/`                            |
| `s3://nv-00-10206-webdataset-images/webdataset_image_v5/llava_ov_midtrain_85m_en/`                        | `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/webdataset_image_v5/llava_ov_midtrain_85m_en/`                        |
| `s3://nv-00-10206-webdataset-images/webdataset_image_v5/llava_ov_sft_22m/`                                | `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/webdataset_image_v5/llava_ov_sft_22m/`                                |
| `s3://nv-00-10206-webdataset-images/webdataset_multi_reference_generation/`                               | `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/webdataset_multi_reference_generation/`                               |
| `s3://nv-00-10206-webdataset-images/webdataset_rl/`                                                       | `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/webdataset_rl/`                                                       |
| `s3://nv-00-10206-webdataset-images/webdataset_sft/`                                                      | `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/webdataset_sft/`                                                      |
| `s3://nv-00-10206-webdataset-images/webdataset_synthetic/`                                                | `s3://nv-10206-images/nv-00-10206-webdataset-images-mirror/webdataset_synthetic/`                                                |
