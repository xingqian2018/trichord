# Three levels of organization:
# 1. sources
# 2. collections
# 3. recipes

# How to run:
# .venv/bin/python -m projects.cosmos3.vfm.datasets.cli_ingest_from_recipe_script --storage-type gcp --script ~/Project/trichord/info/ --no-dry-run

sources = {
    "image": {
        "nonsensitive": {
            "cosmos_lab_image_v1_coyo_700m": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/coyo_700m",
            "cosmos_lab_image_v1_datacomp_12b": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/datacomp_12b",
            "cosmos_lab_image_v1_datacomp_1b": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/datacomp_1b",
            "cosmos_lab_image_v1_laion_115m": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/laion_115m",
            "cosmos_lab_image_v1_laion_400m": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/laion_400m",
            "cosmos_lab_image_v1_midjourney": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/midjourney",
            "cosmos_lab_image_v1_midjourney_v6_20240703": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/midjourney_v6_20240703",
            "cosmos_lab_image_v1_nvcommercial_700m": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/nvcommercial_700m",
            "cosmos_lab_image_v1_red": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/red",

            "cosmos_lab_image_v1_self_improving_synthetic_2026_02_09": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/self_improving_synthetic_2026-02-09",
            "cosmos_lab_image_v1_self_improving_synthetic_2026_02_14": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/self_improving_synthetic_2026-02-14",
            "cosmos_lab_image_v1_wordnet_captions_20260224": "webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/wordnet_captions_20260224",
        },
    }
}


collections = {
    "image": {
        "cosmos_lab_image_v1_reg": {
            "cosmos_lab_image_v1_coyo_700m": 1.0,
            "cosmos_lab_image_v1_datacomp_12b": 1.0,
            "cosmos_lab_image_v1_datacomp_1b": 1.0,
            "cosmos_lab_image_v1_laion_115m": 1.0,
            "cosmos_lab_image_v1_laion_400m": 1.0,
            "cosmos_lab_image_v1_midjourney": 1.0,
            "cosmos_lab_image_v1_midjourney_v6_20240703": 1.0,
            "cosmos_lab_image_v1_nvcommercial_700m": 1.0,
            "cosmos_lab_image_v1_red": 1.0,
        },
        "cosmos_lab_image_v1_sgd": {
            "cosmos_lab_image_v1_self_improving_synthetic_2026_02_09": 1.0,
            "cosmos_lab_image_v1_self_improving_synthetic_2026_02_14": 1.0,
            "cosmos_lab_image_v1_wordnet_captions_20260224": 1.0,
        },
    },
}


recipes = {
    "XX_COSMOS_LAB_IMAGE_V1_REGULAR": {
        "cosmos_lab_image_v1_reg": {"type": "image", "ratio": 1.0},
    },
    "XX_COSMOS_LAB_IMAGE_V1_SGD": {
        "cosmos_lab_image_v1_sgd": {"type": "image", "ratio": 1.0},
    },
}
