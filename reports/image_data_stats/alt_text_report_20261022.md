# LanceDB `image_meta_table_full` — alt_text Report (20261022)

Source: `metadata_stats_20261022.json` (full table scan, latest_fragment_id=40988, 52 datasets). Rule: `alt_text IS NOT NULL`.

## alt_text by source dataset

| dataset                                        | total             | alt_text       | alt_text %   |
|------------------------------------------------|-------------------|----------------|--------------|
| **Pretraining**                                |                   |                |              |
| datacomp_12b                                   | 5,897,063,208     | 0              | 0.00%        |
| nvcommercial_700m                              | 644,577,000       | 0              | 0.00%        |
| coyo_700m                                      | 540,879,510       | 0              | 0.00%        |
| laion_aesthetic                                | 39,006,946        | 0              | 0.00%        |
| generations_qwen_image_2512_filtered_photoreal | 10,947,502        | 0              | 0.00%        |
| wordnet_captions_20260224                      | 8,211,750         | 0              | 0.00%        |
| self_improving_synthetic_2026-02-09            | 7,800,000         | 0              | 0.00%        |
| self_improving_synthetic_2026-02-14            | 3,900,000         | 0              | 0.00%        |
| synthetic_chinese_scene_text_v0                | 3,000,000         | 0              | 0.00%        |
| synthetic_scene_text_v0                        | 3,000,000         | 0              | 0.00%        |
| MMC4                                           | 2,715,072         | 0              | 0.00%        |
| synthetic_traditional_chinese_scene_text_v0    | 500,000           | 0              | 0.00%        |
| zennodo10k                                     | 254,141           | 0              | 0.00%        |
| voxel51_rico                                   | 66,261            | 0              | 0.00%        |
| screen2words_rico                              | 22,417            | 0              | 0.00%        |
| slide_audit                                    | 2,400             | 0              | 0.00%        |
| *subtotal*                                     | *7,161,946,207*   | *0*            | *0.00%*      |
| **Midtrain**                                   |                   |                |              |
| commoncatalog-cc-by                            | 14,581,672        | 0              | 0.00%        |
| megalith-10m-original                          | 8,716,221         | 0              | 0.00%        |
| megascenes                                     | 8,715,454         | 0              | 0.00%        |
| commoncatalog-cc-by-sa                         | 7,891,493         | 0              | 0.00%        |
| multiaspect-4k-1m                              | 1,007,138         | 0              | 0.00%        |
| photo-concept-bucket                           | 567,698           | 0              | 0.00%        |
| Fine-T2I-curated                               | 168,424           | 0              | 0.00%        |
| 4KLSDB                                         | 129,484           | 0              | 0.00%        |
| Aesthetic-Train-V2                             | 105,288           | 0              | 0.00%        |
| LSDIR                                          | 84,991            | 0              | 0.00%        |
| human_sft                                      | 70,159            | 0              | 0.00%        |
| pexels_residual_trustedK1_v2                   | 38,804            | 0              | 0.00%        |
| hq-50k                                         | 29,189            | 0              | 0.00%        |
| unsplash_lite                                  | 25,000            | 0              | 0.00%        |
| uhd-iqa                                        | 4,269             | 0              | 0.00%        |
| flickr2k                                       | 2,650             | 0              | 0.00%        |
| *subtotal*                                     | *42,137,934*      | *0*            | *0.00%*      |
| **Dataset we don't care**                      |                   |                |              |
| datacomp_1b                                    | 598,141,002       | 96,063,219     | 16.06%       |
| laion_400m                                     | 224,459,219       | 0              | 0.00%        |
| laion_115m                                     | 85,067,400        | 0              | 0.00%        |
| midjourney_v6_20240703                         | 81,394,500        | 0              | 0.00%        |
| megalith-10m                                   | 9,279,122         | 0              | 0.00%        |
| midjourney                                     | 8,064,300         | 0              | 0.00%        |
| laion_tier3_v23_312_7_70m_3k_trustedk1         | 128,749           | 0              | 0.00%        |
| laion_tier1_v22_312_7_70m_4k_trustedk1         | 54,847            | 0              | 0.00%        |
| reddit                                         | 7,903             | 0              | 0.00%        |
| datacomp_12b_test                              | 40                | 0              | 0.00%        |
| *subtotal*                                     | *1,006,597,082*   | *96,063,219*   | *9.54%*      |
| **Dataset we don't have legal approved**       |                   |                |              |
| red                                            | 1,249,912         | 0              | 0.00%        |
| ohv3p5m_sdvr2_upres_all_4k                     | 503,149           | 0              | 0.00%        |
| gemini_3_pro_image_200k                        | 200,000           | 0              | 0.00%        |
| agent_distilled_20260414_v19_99827_v2          | 99,827            | 0              | 0.00%        |
| agent_distilled_20260420_v6a_57230             | 57,230            | 0              | 0.00%        |
| ohv_gen_frames_50k_filtered                    | 39,990            | 0              | 0.00%        |
| agent_distilled_20260420_v7m_31806             | 35,254            | 0              | 0.00%        |
| gpt_image_2_20260507                           | 21,546            | 0              | 0.00%        |
| gpt_image_2_20260515                           | 9,206             | 0              | 0.00%        |
| *subtotal*                                     | *2,216,114*       | *0*            | *0.00%*      |
| **Dataset new (uncategorized)**                |                   |                |              |
| sdg_cosmos3_t2i_v0                             | 261,000           | 0              | 0.00%        |
| *subtotal*                                     | *261,000*         | *0*            | *0.00%*      |
| **TOTAL**                                      | **8,213,158,337** | **96,063,219** | **1.17%**    |

## datacomp_1b: alt_text by aspect-ratio shard (fragment 3200)

| aspect-ratio shard in `original_data_path`   | rows          | with alt_text   | %         |
|----------------------------------------------|---------------|-----------------|-----------|
| aspect_ratio_16_9                            | 159,216       | 96,851          | 60.8%     |
| aspect_ratio_1_1                             | 312,633       | 63,626          | 20.4%     |
| aspect_ratio_4_3                             | 300,320       | 0               | 0.0%      |
| aspect_ratio_3_4                             | 185,377       | 0               | 0.0%      |
| aspect_ratio_9_16                            | 42,454        | 0               | 0.0%      |
| **all**                                      | **1,000,000** | **160,477**     | **16.0%** |

## datacomp_1b: alt_text per fragment (every 20th fragment, 2932-3512)

| fragments sampled   | min %   | max %   |
|---------------------|---------|---------|
| 30                  | 16.01%  | 16.14%  |

## datacomp_1b: alt_text columns (9,000 rows with alt_text, fragments 2932 / 3200 / 3500)

| column                        | non-null      |
|-------------------------------|---------------|
| alt_text                      | 9,000 / 9,000 |
| alt_text_source_id            | 9,000 / 9,000 |
| alt_text_source_loc           | 9,000 / 9,000 |
| alt_clip_l14_similarity_score | 9,000 / 9,000 |
| alt_text_title                | 0 / 9,000     |
| alt_text_language             | 0 / 9,000     |

| stat                          | p10   | p50   | p90   |
|-------------------------------|-------|-------|-------|
| alt_text length (chars)       | 21    | 48    | 105   |
| alt_clip_l14_similarity_score | 0.25  | 0.279 | 0.332 |

Fragment 3200: alt_clip_l14_similarity_score min 0.243, max 0.494 over 160,477 rows.

## datacomp_1b: alt_text examples

| alt_text                                                                                                                                               | clip L/14 sim   |
|--------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------|
| Apartment Calle Los Cipresses 3                                                                                                                        | 0.25            |
| Forest Hills Golf Club                                                                                                                                 | 0.245           |
| History of Montserrat Font                                                                                                                             | 0.37            |
| Weirdmailus  Pleasing Make An Editableprintable Html Invoice  Csstricks With Foxy This Of Course Cant Do Nearly All The Awesome Stuff Freshbooks Can D | 0.263           |
| maxresdefault android 6 0 1998 2006 bmw m3 3 series e46 radio gps sat nav dvd E46 Sunroof Wiring-Diagram at soozxer.org                                | 0.276           |
| 2014-10-19-Noah 02 Clean christening cake for a baby boy.                                                                                              | 0.271           |
| Marjet van WIjk Medium Audience Influencer Badge                                                                                                       | 0.37            |
| File:Male Arm Guards 49 (TKD).png                                                                                                                      | 0.262           |
| Large Plastic Syringe, Transparent, Reusable, Sterile Measuring Injection,                                                                             | 0.292           |
| Brown Labradordoodle and a goldendoodle puppies                                                                                                        | 0.283           |
| Sunset Tuscany pan                                                                                                                                     | 0.253           |
| Fun-ctional Training                                                                                                                                   | 0.255           |
