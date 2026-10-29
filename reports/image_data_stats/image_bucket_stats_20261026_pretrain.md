# Pretrain Image Bucket Stats (20261026r)

Source: `bucket_stats_20261026r.json` + `_progress.json`, table version 254020–254020, 40,658 fragments, 16 pretrain datasets, failed fragments: 0.

| rule         | definition                                                                                               |
|--------------|----------------------------------------------------------------------------------------------------------|
| bucket key   | `source_dataset\|taxonomy\|resolution\|aspect_ratio\|filter\|aesthetic`                                  |
| taxonomy     | first item of `taxonomy_tag_v0_subcategory` (JSON list, most likely first), one of 41 labels             |
| resolution   | tier by pixel area w×h                                                                                   |
| aspect ratio | nearest of 7 target ratios by log-midpoint range; `<=1:2` / `>=2:1` open-ended                           |
| filter       | qwen3vl flags all false = `filter_yes`; any true = `filter_no`                                           |
| aesthetic    | qwen3vl aesthetic score 0–9                                                                              |
| excluded     | `tobe_remove` (datacomp_12b face duplicates) and rows missing taxonomy / size / filter flags / aesthetic |

## Rows per dataset

| dataset                                        | total             | tobe_remove     | missing taxonomy   | missing filter   | missing aesthetic   | bucketed        | bucketed %   |
|------------------------------------------------|-------------------|-----------------|--------------------|------------------|---------------------|-----------------|--------------|
| datacomp_12b                                   | 5,897,063,208     | 338,705,595     | 4,908,232,746      | 14,519,937       | 14,517,021          | 650,124,867     | 11.0%        |
| nvcommercial_700m                              | 644,577,000       | 0               | 514,559,764        | 496              | 184                 | 130,017,236     | 20.2%        |
| coyo_700m                                      | 540,879,510       | 0               | 403,751,564        | 702,420          | 702,126             | 137,127,946     | 25.4%        |
| laion_aesthetic                                | 39,006,946        | 0               | 12,592,497         | 1                | 0                   | 26,414,449      | 67.7%        |
| generations_qwen_image_2512_filtered_photoreal | 10,947,502        | 0               | 1,698,865          | 0                | 0                   | 9,248,637       | 84.5%        |
| wordnet_captions_20260224                      | 8,211,750         | 0               | 1,416,913          | 0                | 0                   | 6,794,837       | 82.7%        |
| self_improving_synthetic_2026-02-09            | 7,800,000         | 0               | 531,023            | 0                | 0                   | 7,268,977       | 93.2%        |
| self_improving_synthetic_2026-02-14            | 3,900,000         | 0               | 347,094            | 0                | 0                   | 3,552,906       | 91.1%        |
| synthetic_chinese_scene_text_v0                | 3,000,000         | 0               | 1,091,894          | 0                | 0                   | 1,908,106       | 63.6%        |
| synthetic_scene_text_v0                        | 3,000,000         | 0               | 305,212            | 0                | 0                   | 2,694,788       | 89.8%        |
| MMC4                                           | 2,715,072         | 0               | 2,316,214          | 6                | 0                   | 398,858         | 14.7%        |
| synthetic_traditional_chinese_scene_text_v0    | 500,000           | 0               | 256,595            | 0                | 0                   | 243,405         | 48.7%        |
| zennodo10k                                     | 254,141           | 0               | 253,989            | 0                | 0                   | 152             | 0.1%         |
| voxel51_rico                                   | 66,261            | 0               | 66,261             | 0                | 0                   | 0               | 0.0%         |
| screen2words_rico                              | 22,417            | 0               | 22,417             | 0                | 0                   | 0               | 0.0%         |
| slide_audit                                    | 2,400             | 0               | 2,400              | 0                | 0                   | 0               | 0.0%         |
| **TOTAL**                                      | **7,161,946,207** | **338,705,595** | **5,847,445,448**  | **15,222,860**   | **15,219,331**      | **975,795,164** | **13.6%**    |

Filter dimension: `filter_yes` 975,795,164.

## Taxonomy

| taxonomy                                            | rows            | share      |
|-----------------------------------------------------|-----------------|------------|
| General Image of Architecture and Interiors         | 263,042,697     | 27.0%      |
| General Image of People and Portraits               | 191,706,047     | 19.6%      |
| General Image of Food and Drink                     | 96,259,917      | 9.9%       |
| General Image of Other Objects                      | 85,920,757      | 8.8%       |
| General Image of Nature and Landscapes              | 84,560,846      | 8.7%       |
| Art and Music                                       | 81,211,659      | 8.3%       |
| General Image of Cars and Vehicles                  | 47,336,749      | 4.9%       |
| General Image of Clothing and Accessories           | 44,188,030      | 4.5%       |
| General Image of Animals and Wildlife               | 32,298,839      | 3.3%       |
| General Image of Plants and Flora                   | 19,583,622      | 2.0%       |
| Technology and Engineering                          | 7,873,939       | 0.8%       |
| Posters and Design                                  | 7,746,400       | 0.8%       |
| Medical and Health Sciences                         | 5,259,188       | 0.5%       |
| Physical Sciences                                   | 2,071,250       | 0.2%       |
| Screenshot of Other Digital Interfaces              | 1,835,694       | 0.2%       |
| Life Sciences                                       | 1,086,038       | 0.1%       |
| Maps                                                | 786,134         | 0.1%       |
| Screenshot of Mobile Apps                           | 743,042         | 0.1%       |
| Plots and Charts                                    | 611,022         | 0.1%       |
| Handwritings                                        | 413,342         | 0.0%       |
| Geometric Shapes                                    | 276,010         | 0.0%       |
| Documents                                           | 272,968         | 0.0%       |
| Screenshot of Websites                              | 137,602         | 0.0%       |
| Screenshot of Desktop Apps                          | 135,022         | 0.0%       |
| Industrial Robotics and Factory Scenes              | 129,806         | 0.0%       |
| Structuralized Diagrams                             | 79,090          | 0.0%       |
| Fixed-Base Robotic Manipulators                     | 43,808          | 0.0%       |
| Tables                                              | 34,435          | 0.0%       |
| Mathematical Notations                              | 34,268          | 0.0%       |
| Legged and Humanoid Robots                          | 31,330          | 0.0%       |
| Infographics                                        | 25,272          | 0.0%       |
| Expressive and Social Robots                        | 21,003          | 0.0%       |
| Mobile Robots and Bases                             | 19,958          | 0.0%       |
| Aerial and Aquatic Robots                           | 8,223           | 0.0%       |
| Human-Robot Collaboration                           | 7,833           | 0.0%       |
| Fluid Dynamics and Thermodynamics                   | 1,578           | 0.0%       |
| Autonomous Vehicle Normal Driving Scenes            | 753             | 0.0%       |
| Teleoperation and VR Control                        | 495             | 0.0%       |
| Autonomous Vehicle Adverse and Hazardous Conditions | 355             | 0.0%       |
| Soft Body and Deformation                           | 93              | 0.0%       |
| Rigid Body Mechanics                                | 50              | 0.0%       |
| **TOTAL**                                           | **975,795,164** | **100.0%** |

Taxonomy labels with 0 rows: none.

## Resolution

| resolution   | rows            | share      |
|--------------|-----------------|------------|
| lt_256p      | 101,493,993     | 10.4%      |
| 256p         | 482,856,934     | 49.5%      |
| 480p         | 240,078,505     | 24.6%      |
| 720p         | 25,726,353      | 2.6%       |
| 768p         | 80,955,872      | 8.3%       |
| 1080p        | 23,229,789      | 2.4%       |
| 1440p        | 3,009,028       | 0.3%       |
| 1536p        | 10,669,777      | 1.1%       |
| 2160p        | 7,774,913       | 0.8%       |
| **TOTAL**    | **975,795,164** | **100.0%** |

## Aspect ratio

| aspect_ratio   | rows            | share      |
|----------------|-----------------|------------|
| <=1:2          | 1,909,909       | 0.2%       |
| 9:16           | 12,318,961      | 1.3%       |
| 2:3            | 71,282,132      | 7.3%       |
| 3:4            | 74,962,468      | 7.7%       |
| 1:1            | 159,880,224     | 16.4%      |
| 4:3            | 165,118,630     | 16.9%      |
| 3:2            | 316,956,499     | 32.5%      |
| 16:9           | 135,915,143     | 13.9%      |
| >=2:1          | 37,451,198      | 3.8%       |
| **TOTAL**      | **975,795,164** | **100.0%** |

## Aesthetic

| aesthetic   | rows            | share      |
|-------------|-----------------|------------|
| aes_4       | 744,231,056     | 76.3%      |
| aes_5       | 166,015,598     | 17.0%      |
| aes_6       | 64,451,021      | 6.6%       |
| aes_7       | 1,091,732       | 0.1%       |
| aes_8       | 5,757           | 0.0%       |
| **TOTAL**   | **975,795,164** | **100.0%** |

## Resolution by dataset (% of bucketed rows)

| dataset                                        | bucketed        | lt_256p   | 256p      | 480p      | 720p     | 768p     | 1080p    | 1440p    | 1536p    | 2160p    |
|------------------------------------------------|-----------------|-----------|-----------|-----------|----------|----------|----------|----------|----------|----------|
| datacomp_12b                                   | 650,124,867     | 15.6%     | 49.2%     | 21.3%     | 2.6%     | 5.8%     | 2.8%     | 0.4%     | 1.4%     | 1.0%     |
| nvcommercial_700m                              | 130,017,236     | 0.0%      | 51.7%     | 45.6%     | 2.5%     | 0.1%     | 0.0%     | 0.0%     | 0.0%     | 0.0%     |
| coyo_700m                                      | 137,127,946     | 0.0%      | 57.0%     | 26.4%     | 3.5%     | 6.6%     | 3.6%     | 0.5%     | 1.3%     | 1.0%     |
| laion_aesthetic                                | 26,414,449      | 0.0%      | 64.5%     | 22.9%     | 2.5%     | 9.9%     | 0.1%     | 0.0%     | 0.0%     | 0.0%     |
| generations_qwen_image_2512_filtered_photoreal | 9,248,637       | 0.0%      | 0.0%      | 0.0%      | 0.0%     | 100.0%   | 0.0%     | 0.0%     | 0.0%     | 0.0%     |
| wordnet_captions_20260224                      | 6,794,837       | 0.0%      | 0.0%      | 0.0%      | 0.0%     | 100.0%   | 0.0%     | 0.0%     | 0.0%     | 0.0%     |
| self_improving_synthetic_2026-02-09            | 7,268,977       | 0.0%      | 0.0%      | 0.0%      | 0.0%     | 100.0%   | 0.0%     | 0.0%     | 0.0%     | 0.0%     |
| self_improving_synthetic_2026-02-14            | 3,552,906       | 0.0%      | 0.0%      | 0.0%      | 0.0%     | 100.0%   | 0.0%     | 0.0%     | 0.0%     | 0.0%     |
| synthetic_chinese_scene_text_v0                | 1,908,106       | 0.0%      | 0.0%      | 0.0%      | 0.0%     | 100.0%   | 0.0%     | 0.0%     | 0.0%     | 0.0%     |
| synthetic_scene_text_v0                        | 2,694,788       | 0.0%      | 0.0%      | 0.0%      | 0.0%     | 100.0%   | 0.0%     | 0.0%     | 0.0%     | 0.0%     |
| MMC4                                           | 398,858         | 7.1%      | 56.9%     | 36.0%     | 0.0%     | 0.0%     | 0.0%     | 0.0%     | 0.0%     | 0.0%     |
| synthetic_traditional_chinese_scene_text_v0    | 243,405         | 0.0%      | 0.0%      | 0.0%      | 0.0%     | 100.0%   | 0.0%     | 0.0%     | 0.0%     | 0.0%     |
| zennodo10k                                     | 152             | 0.0%      | 0.0%      | 0.0%      | 0.0%     | 0.0%     | 100.0%   | 0.0%     | 0.0%     | 0.0%     |
| **TOTAL**                                      | **975,795,164** | **10.4%** | **49.5%** | **24.6%** | **2.6%** | **8.3%** | **2.4%** | **0.3%** | **1.1%** | **0.8%** |

## Aesthetic by dataset (% of bucketed rows)

| dataset                                        | bucketed        | aes_4     | aes_5     | aes_6    | aes_7    | aes_8    |
|------------------------------------------------|-----------------|-----------|-----------|----------|----------|----------|
| datacomp_12b                                   | 650,124,867     | 80.5%     | 15.4%     | 4.0%     | 0.0%     | 0.0%     |
| nvcommercial_700m                              | 130,017,236     | 76.0%     | 18.5%     | 5.4%     | 0.1%     | 0.0%     |
| coyo_700m                                      | 137,127,946     | 74.6%     | 19.7%     | 5.6%     | 0.1%     | 0.0%     |
| laion_aesthetic                                | 26,414,449      | 56.4%     | 29.6%     | 13.6%    | 0.4%     | 0.0%     |
| generations_qwen_image_2512_filtered_photoreal | 9,248,637       | 11.6%     | 28.1%     | 59.9%    | 0.4%     | 0.0%     |
| wordnet_captions_20260224                      | 6,794,837       | 12.2%     | 20.7%     | 65.0%    | 2.1%     | 0.0%     |
| self_improving_synthetic_2026-02-09            | 7,268,977       | 6.1%      | 13.9%     | 75.9%    | 4.1%     | 0.0%     |
| self_improving_synthetic_2026-02-14            | 3,552,906       | 12.6%     | 22.6%     | 63.4%    | 1.4%     | 0.0%     |
| synthetic_chinese_scene_text_v0                | 1,908,106       | 19.2%     | 26.3%     | 54.2%    | 0.3%     | 0.0%     |
| synthetic_scene_text_v0                        | 2,694,788       | 38.9%     | 24.8%     | 36.2%    | 0.0%     | 0.0%     |
| MMC4                                           | 398,858         | 80.8%     | 16.6%     | 2.6%     | 0.0%     | 0.0%     |
| synthetic_traditional_chinese_scene_text_v0    | 243,405         | 14.8%     | 20.8%     | 63.5%    | 0.9%     | 0.0%     |
| zennodo10k                                     | 152             | 78.9%     | 14.5%     | 5.9%     | 0.7%     | 0.0%     |
| **TOTAL**                                      | **975,795,164** | **76.3%** | **17.0%** | **6.6%** | **0.1%** | **0.0%** |
