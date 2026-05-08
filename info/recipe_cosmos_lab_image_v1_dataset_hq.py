# Three levels of organization:
# 1. sources
# 2. collections
# 3. recipes

# How to run:
# .venv/bin/python -m projects.cosmos3.vfm.datasets.cli_ingest_from_recipe_script --storage-type gcp --script ~/info/recipe_cosmos_lab_image_v1_dataset_hq.py --no-dry-run

sources = {
    "image": {
        "nonsensitive": {
            "cosmos_lab_image_v1_screen2words_rico" : "webdataset_image_regular_text/screen2words_rico/wdinfo",
            "cosmos_lab_image_v1_slide_audit" : "webdataset_image_regular_text/slide_audit/wdinfo",
            "cosmos_lab_image_v1_voxel51_rico" : "webdataset_image_regular_text/voxel51_rico/wdinfo",
            "cosmos_lab_image_v1_zennodo10k" : "webdataset_image_regular_text/zennodo10k/wdinfo",

            "cosmos_lab_image_v1_synthetic_scene_text_v0" : "webdataset_image_synthetic_text/synthetic_scene_text_v0/wdinfo",
            "cosmos_lab_image_v1_synthetic_chinese_scene_text_v0" : "webdataset_image_synthetic_text/synthetic_chinese_scene_text_v0/wdinfo",
            "cosmos_lab_image_v1_synthetic_traditional_chinese_scene_text_v0" : "webdataset_image_synthetic_text/synthetic_traditional_chinese_scene_text_v0/wdinfo",

            "cosmos_lab_image_v1_coyo_700m": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/coyo_700m",
            "cosmos_lab_image_v1_midjourney": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/midjourney",
            "cosmos_lab_image_v1_midjourney_v6_20240703": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/midjourney_v6_20240703",

            "cosmos_lab_image_v1_red": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/red",

            "cosmos_lab_image_v1_regular_hq": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1_high_quality/wdinfo",

            "cosmos_lab_image_v1_self_improving_synthetic_2026_02_09": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/self_improving_synthetic_2026-02-09",
            "cosmos_lab_image_v1_self_improving_synthetic_2026_02_14": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/self_improving_synthetic_2026-02-14",
            "cosmos_lab_image_v1_wordnet_captions_20260224": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/wordnet_captions_20260224",

            "cosmos_lab_image_v1_human_sft": "webdataset_cosmos_lab_image_v1/v1_human_sft/wdinfo",

            "cosmos_lab_image_v1_koi_sft_a": "webdataset_cosmos_lab_image_v1/v1_agent_distilled_v7m_31806/wdinfo",
            "cosmos_lab_image_v1_koi_sft_b": "webdataset_cosmos_lab_image_v1/v1_agent_distilled_v6a_57230/wdinfo",
            "cosmos_lab_image_v1_koi_sft_c": "webdataset_cosmos_lab_image_v1/v1_agent_distilled_v19_99827/wdinfo",

            "cosmos_lab_image_v1_adhoc_filtered_sft_a": "webdataset_cosmos_lab_image_v1/v1_sft_candidate_a_conservative/wdinfo",
            "cosmos_lab_image_v1_adhoc_filtered_sft_b": "webdataset_cosmos_lab_image_v1/v1_sft_candidate_c_clean/wdinfo",
        },
    }
}


collections = {
    "image": {
        "cosmos_lab_image_v1_reg_text": {
            "cosmos_lab_image_v1_screen2words_rico": 1.0,
            "cosmos_lab_image_v1_slide_audit": 1.0,
            "cosmos_lab_image_v1_voxel51_rico": 1.0,
            "cosmos_lab_image_v1_zennodo10k": 1.0,
        },
        "cosmos_lab_image_v1_sgd_text": {
            "cosmos_lab_image_v1_synthetic_scene_text_v0": 1.0,
            "cosmos_lab_image_v1_synthetic_chinese_scene_text_v0": 1.0,
            "cosmos_lab_image_v1_synthetic_traditional_chinese_scene_text_v0": 1.0,
        },
        "cosmos_lab_image_v1_reg_subset_v0": {
            "cosmos_lab_image_v1_coyo_700m": 1.0,
            "cosmos_lab_image_v1_midjourney": 1.0,
            "cosmos_lab_image_v1_midjourney_v6_20240703": 1.0,
        },
        "cosmos_lab_image_v1_red": {
            "cosmos_lab_image_v1_red": 1.0
        },
        "cosmos_lab_image_v1_reg_subset_hq_v0": {
            "cosmos_lab_image_v1_regular_hq": 1.0
        },
        "cosmos_lab_image_v1_sgd_subset_v0": {
            "cosmos_lab_image_v1_self_improving_synthetic_2026_02_09": 1.0,
            "cosmos_lab_image_v1_self_improving_synthetic_2026_02_14": 1.0,
            "cosmos_lab_image_v1_wordnet_captions_20260224": 1.0,
        },
        "cosmos_lab_image_v1_human_sft": {
            "cosmos_lab_image_v1_human_sft": 1.0
        },
        "cosmos_lab_image_v1_koi_sft_v0": {
            "cosmos_lab_image_v1_koi_sft_a": 1.0,
            "cosmos_lab_image_v1_koi_sft_b": 1.0,
            "cosmos_lab_image_v1_koi_sft_c": 1.0,
        },
        "cosmos_lab_image_v1_adhoc_filtered_sft_v0": {
            "cosmos_lab_image_v1_adhoc_filtered_sft_a": 1.0,
            "cosmos_lab_image_v1_adhoc_filtered_sft_b": 1.0,
        },
    },
}


recipes = {
    "XX_COSMOS_LAB_IMAGE_V1_HQV0": {
        "cosmos_lab_image_v1_sgd_text"             : {"type": "image", "ratio": 5.0},
        "cosmos_lab_image_v1_reg_text"             : {"type": "image", "ratio": 0.1},
        "cosmos_lab_image_v1_reg_subset_v0"        : {"type": "image", "ratio": 1.0},
        "cosmos_lab_image_v1_red"                  : {"type": "image", "ratio": 1.0},
        "cosmos_lab_image_v1_reg_subset_hq_v0"     : {"type": "image", "ratio": 1.0},
        "cosmos_lab_image_v1_sgd_subset_v0"        : {"type": "image", "ratio": 3.0},
        "cosmos_lab_image_v1_human_sft"            : {"type": "image", "ratio": 1.0},
        "cosmos_lab_image_v1_koi_sft_v0"           : {"type": "image", "ratio": 2.0},
        "cosmos_lab_image_v1_adhoc_filtered_sft_v0": {"type": "image", "ratio": 0.5},
    },
}

