# Common Root Location Alias on s3

| Alias                           | Path                                                                 |
|---------------------------------|----------------------------------------------------------------------|
| `<webds_image_reg_text>`        | `s3://nv-00-10206-webdataset-images/webdataset_image_regular_text`   |
| `<webds_image_sdg_text>`        | `s3://nv-00-10206-webdataset-images/webdataset_image_synthetic_text` |
| `<webds_image_reg>`             | `s3://nv-00-10206-vfm/webdataset_image_regular`                      |
| `<webds_image_sdg>`             | `s3://nv-00-10206-vfm/webdataset_image_synthetic`                    |
| `<logged_image_root>`           | `s3://nv-00-10206-images/logged_images`                              |
| `<logged_meta_root>`            | `s3://nv-00-10206-images/logged_metas`                               |
| `<lancedb_image_root>`          | `gs://nv-00-10206-vfm/lancedb/image/`                                |
| `<lancedb_image_root_reg>`      | `gs://nv-00-10206-vfm/lancedb/image/regular/`                        |
| `<lancedb_image_root_sdg>`      | `gs://nv-00-10206-vfm/lancedb/image/synthetic/`                      |
| `<lancedb_image_root_reg_text>` | `gs://nv-00-10206-vfm/lancedb/image/regular_text/`                   |
| `<lancedb_image_root_sdg_text>` | `gs://nv-00-10206-vfm/lancedb/image/synthetic_text/`                 |


# WebDataset Locations on s3

| DatasetName                                   | WebDS Path                                                            |
|-----------------------------------------------|-----------------------------------------------------------------------|
| `screen2words_rico`                           | `<webds_image_reg_text>/screen2words_rico/`                           |
| `slide_audit`                                 | `<webds_image_reg_text>/slide_audit/`                                 |
| `voxel51_rico`                                | `<webds_image_reg_text>/voxel51_rico/`                                |
| `zennodo10k`                                  | `<webds_image_reg_text>/zennodo10k/`                                  |
| `synthetic_scene_text_v0`                     | `<webds_image_sdg_text>/synthetic_scene_text_v0/`                     |
| `synthetic_chinese_scene_text_v0`             | `<webds_image_sdg_text>/synthetic_chinese_scene_text_v0/`             |
| `synthetic_traditional_chinese_scene_text_v0` | `<webds_image_sdg_text>/synthetic_traditional_chinese_scene_text_v0/` |
|-----------------------------------------------|-----------------------------------------------------------------------|
| `synthetic_scene_text_v1`                         | `<webds_image_sdg_text>/synthetic_scene_text_v1/`                        |
| `synthetic_scene_text_v1_phi`                     | `<webds_image_sdg_text>/synthetic_scene_text_v1_phi/`                    |
| `synthetic_scene_text_chinese_v1`                 | `<webds_image_sdg_text>/synthetic_scene_text_chinese_v1/`                |
| `synthetic_scene_text_chinese_v1_phi`             | `<webds_image_sdg_text>/synthetic_scene_text_chinese_v1_phi/`            |
| `synthetic_scene_text_traditional_chinese_v1`     | `<webds_image_sdg_text>/synthetic_scene_text_traditional_chinese_v1/`    |
| `synthetic_scene_text_traditional_chinese_v1_phi` | `<webds_image_sdg_text>/synthetic_scene_text_traditional_chinese_v1_phi/`|


# LanceDB Locations on s3

`<table_postfix>` = `slice_from_maintable_YYYYmmdd` (e.g. `slice_from_maintable_20260506`)

| DatasetName                                       | LanceDB Path                                                                                                     |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `screen2words_rico`                               | `<lancedb_image_root_reg_text>/screen2words_rico_<table_postfix>.lance/`                                         |
| `slide_audit`                                     | `<lancedb_image_root_reg_text>/slide_audit_<table_postfix>.lance/`                                               |
| `voxel51_rico`                                    | `<lancedb_image_root_reg_text>/voxel51_rico_<table_postfix>.lance/`                                              |
| `zennodo10k`                                      | `<lancedb_image_root_reg_text>/zennodo10k_<table_postfix>.lance/`                                                |
| `synthetic_scene_text_v0`                         | `<lancedb_image_root_sdg_text>/synthetic_scene_text_v0_<table_postfix>.lance/`                                   |
| `synthetic_chinese_scene_text_v0`                 | `<lancedb_image_root_sdg_text>/synthetic_chinese_scene_text_v0_<table_postfix>.lance/`                           |
| `synthetic_traditional_chinese_scene_text_v0`     | `<lancedb_image_root_sdg_text>/synthetic_traditional_chinese_scene_text_v0_<table_postfix>.lance/`               |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `red`                                             | `<lancedb_image_root_reg>/red_<table_postfix>.lance/`                                                            |
| `coyo_700m`                                       | `<lancedb_image_root_reg>/coyo_700m_<table_postfix>.lance/`                                                      |
| `pexels_residual_trustedK1_v2`                    | `<lancedb_image_root_reg>/pexels_residual_trustedK1_v2_<table_postfix>.lance/`                                   |
|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `MMC4`                                            | `<lancedb_image_root_sdg>/mmc4_<table_postfix>.lance/`                                                           |
| `generations_qwen_image_2512_filtered_photoreal`  | `<lancedb_image_root_sdg>/generations_qwen_image_2512_filtered_photoreal_<table_postfix>.lance/`                 |
| `wordnet_captions_20260224`                       | `<lancedb_image_root_sdg>/wordnet_captions_20260224_<table_postfix>.lance/`                                      |
| `datacomp_1b`                                     | `<lancedb_image_root_sdg>/datacomp_1b_<table_postfix>.lance/`                                                    |
| `midjourney`                                      | `<lancedb_image_root_sdg>/midjourney_<table_postfix>.lance/`                                                     |
| `midjourney_v6_20240703`                          | `<lancedb_image_root_sdg>/midjourney_v6_20240703_<table_postfix>.lance/`                                         |
