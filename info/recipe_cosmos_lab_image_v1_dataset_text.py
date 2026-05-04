# Three levels of organization:
# 1. sources
# 2. collections
# 3. recipes

# How to run:
# .venv/bin/python -m projects.cosmos3.vfm.datasets.cli_ingest_from_recipe_script --storage-type gcp --script ~/Project/trichord/info/ --no-dry-run

sources = {
    "image": {
        "nonsensitive": {
            "cosmos_lab_image_v1_screen2words_rico" : "webdataset_image_regular_text/screen2words_rico/wdinfo/",
            "cosmos_lab_image_v1_slide_audit" : "webdataset_image_regular_text/slide_audit/wdinfo/",
            "cosmos_lab_image_v1_voxel51_rico" : "webdataset_image_regular_text/voxel51_rico/wdinfo/",
            "cosmos_lab_image_v1_zennodo10k" : "webdataset_image_regular_text/zennodo10k/wdinfo/",
            "cosmos_lab_image_v1_synthetic_scene_text_v0" : "webdataset_image_synthetic_text/synthetic_scene_text_v0/wdinfo/",
            "cosmos_lab_image_v1_synthetic_chinese_scene_text_v0" : "webdataset_image_synthetic_text/synthetic_chinese_scene_text_v0/wdinfo/",
            "cosmos_lab_image_v1_synthetic_traditional_chinese_scene_text_v0" : "webdataset_image_synthetic_text/synthetic_traditional_chinese_scene_text_v0/wdinfo/",
        },
    }
}


collections = {
    "image": {
        "cosmos_lab_image_v1_reg": {
            "cosmos_lab_image_v1_screen2words_rico": 1.0,
            "cosmos_lab_image_v1_slide_audit": 1.0,
            "cosmos_lab_image_v1_voxel51_rico": 1.0,
            "cosmos_lab_image_v1_zennodo10k": 1.0,
        },
        "cosmos_lab_image_v1_sgd": {
            "cosmos_lab_image_v1_synthetic_scene_text_v0": 1.0,
            "cosmos_lab_image_v1_synthetic_chinese_scene_text_v0": 1.0,
            "cosmos_lab_image_v1_synthetic_traditional_chinese_scene_text_v0": 1.0,
        },
    },
}


recipes = {
    "XX_COSMOS_LAB_IMAGE_V1_TEXT": {
        "cosmos_lab_image_v1_reg_text": {"type": "image", "ratio": 1.0},
        "cosmos_lab_image_v1_sdg_text": {"type": "image", "ratio": 0.05},
    },
}

