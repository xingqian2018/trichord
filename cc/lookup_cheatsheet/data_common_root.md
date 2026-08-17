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

| DatasetName                                        | LanceDB Path                                                                                                      |
|----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `nvcommercial_700m`                                | `<lancedb_image_root_reg>/nvcommercial_700m_<table_postfix>.lance/`                                               |
| `coyo_700m`                                        | `<lancedb_image_root_reg>/coyo_700m_<table_postfix>.lance/`                                                       |
| `MMC4`                                             | `<lancedb_image_root_reg>/mmc4_<table_postfix>.lance/`                                                            |
| `datacomp_1b`                                      | `<lancedb_image_root_reg>/datacomp_1b_<table_postfix>.lance/`                                                     |
| `red`                                              | `<lancedb_image_root_reg>/red_<table_postfix>.lance/`                                                             |
| `human_sft`                                        | `<lancedb_image_root_reg>/human_sft_<table_postfix>.lance/`                                                       |
| `pexels_residual_trustedK1_v2`                     | `<lancedb_image_root_reg>/pexels_residual_trustedK1_v2_<table_postfix>.lance/`                                    |
| `laion_aesthetic`                                  | `<lancedb_image_root_reg>/laion_aesthetic_<table_postfix>.lance/`                                                 |
|----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `generations_qwen_image_2512_filtered_photoreal`   | `<lancedb_image_root_sdg>/generations_qwen_image_2512_filtered_photoreal_<table_postfix>.lance/`                  |
| `wordnet_captions_20260224`                        | `<lancedb_image_root_sdg>/wordnet_captions_20260224_<table_postfix>.lance/`                                       |
| `self_improving_synthetic_2026-02-09`              | `<lancedb_image_root_sdg>/self_improving_synthetic_2026-02-09_<table_postfix>.lance/`                             |
| `self_improving_synthetic_2026-02-14`              | `<lancedb_image_root_sdg>/self_improving_synthetic_2026-02-14_<table_postfix>.lance/`                             |
| `gemini3_image_pro_aa_synthetic_100k_20260516`     | `<lancedb_image_root_sdg>/gemini3_image_pro_aa_synthetic_100k_20260516_<table_postfix>.lance/`                    |
| `gemini_3_pro_image_200k`                          | `<lancedb_image_root_sdg>/gemini_3_pro_image_200k_<table_postfix>.lance/`                                         |
| `gpt_image_2_20260507`                             | `<lancedb_image_root_sdg>/gpt_image_2_20260507_<table_postfix>.lance/`                                            |
| `gpt_image_2_20260515`                             | `<lancedb_image_root_sdg>/gpt_image_2_20260515_<table_postfix>.lance/`                                            |
| `gpt_image_2_artificial_analysis_opus47`           | `<lancedb_image_root_sdg>/gpt_image_2_artificial_analysis_opus47_<table_postfix>.lance/`                          |
| `gpt_image_2_aa_synthetic_44k_gpt55_t2i_image_v1`  | `<lancedb_image_root_sdg>/gpt_image_2_aa_synthetic_44k_gpt55_t2i_image_v1_<table_postfix>.lance/`                 |
| `midjourney`                                       | `<lancedb_image_root_sdg>/midjourney_<table_postfix>.lance/`                                                      |
| `midjourney_v6_20240703`                           | `<lancedb_image_root_sdg>/midjourney_v6_20240703_<table_postfix>.lance/`                                          |
| `v1_agent_distilled_v19_99827`                     | `<lancedb_image_root_sdg>/v1_agent_distilled_v19_99827_<table_postfix>.lance/`                                    |
| `v1_agent_distilled_v6a_57230`                     | `<lancedb_image_root_sdg>/v1_agent_distilled_v6a_57230_<table_postfix>.lance/`                                    |
| `v1_agent_distilled_v7m_31806`                     | `<lancedb_image_root_sdg>/v1_agent_distilled_v7m_31806_<table_postfix>.lance/`                                    |
|----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `screen2words_rico`                                | `<lancedb_image_root_reg_text>/screen2words_rico_<table_postfix>.lance/`                                          |
| `slide_audit`                                      | `<lancedb_image_root_reg_text>/slide_audit_<table_postfix>.lance/`                                                |
| `voxel51_rico`                                     | `<lancedb_image_root_reg_text>/voxel51_rico_<table_postfix>.lance/`                                               |
| `zennodo10k`                                       | `<lancedb_image_root_reg_text>/zennodo10k_<table_postfix>.lance/`                                                 |
|----------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `synthetic_scene_text_v0`                          | `<lancedb_image_root_sdg_text>/synthetic_scene_text_v0_<table_postfix>.lance/`                                    |
| `synthetic_chinese_scene_text_v0`                  | `<lancedb_image_root_sdg_text>/synthetic_chinese_scene_text_v0_<table_postfix>.lance/`                            |
| `synthetic_traditional_chinese_scene_text_v0`      | `<lancedb_image_root_sdg_text>/synthetic_traditional_chinese_scene_text_v0_<table_postfix>.lance/`                |


# Data Count

| DatasetName                                        | Sample Count  |
|----------------------------------------------------|---------------|
| `nvcommercial_700m`                                | 644,577,000   |
| `coyo_700m`                                        | 540,879,510   |
| `MMC4`                                             | 2,715,072     |
| `datacomp_1b`                                      | 598,141,002   |
| `red`                                              | 1,249,912     |
| `human_sft`                                        | 70,159        |
| `pexels_residual_trustedK1_v2`                     | 38,804        |
| `laion_aesthetic`                                  |               |
|----------------------------------------------------|---------------|
| `generations_qwen_image_2512_filtered_photoreal`   | 10,947,502    |
| `wordnet_captions_20260224`                        | 8,211,750     |
| `self_improving_synthetic_2026-02-09`              | 7,800,000     |
| `self_improving_synthetic_2026-02-14`              | 3,900,000     |
| `gemini3_image_pro_aa_synthetic_100k_20260516`     |               |
| `gemini_3_pro_image_200k`                          | 200,000       |
| `gpt_image_2_20260507`                             | 21,546        |
| `gpt_image_2_20260515`                             | 9,206         |
| `gpt_image_2_artificial_analysis_opus47`           |               |
| `gpt_image_2_aa_synthetic_44k_gpt55_t2i_image_v1`  |               |
| `v1_agent_distilled_v19_99827`                     |               |
| `v1_agent_distilled_v6a_57230`                     |               |
| `v1_agent_distilled_v7m_31806`                     |               |
|----------------------------------------------------|---------------|
| `voxel51_rico`                                     | 66,261        |
| `screen2words_rico`                                | 22,417        |
| `slide_audit`                                      | 2,400         |
| `zennodo10k`                                       | 254,141       |
|----------------------------------------------------|---------------|
| `synthetic_scene_text_v0`                          | 3,000,000     |
| `synthetic_chinese_scene_text_v0`                  | 3,000,000     |
| `synthetic_traditional_chinese_scene_text_v0`      | 500,000       |
