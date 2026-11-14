# WebDS Cluster Distribution Report

**Source:** `gcs:nv-00-10206-vfm/debug/xingqianx/evaluation_results/webds_cluster_dist/stats_20260818.json`
**Datasets scanned:** 6 (`nvcommercial_700m`, `MMC4`, `coyo_700m`, `red`, `v1_high_quality`, `pexels_residual_trustedK1_v2`)
**Workers:** 32
**Grand total images:** 91,321,713

---

## Dataset Totals

| Dataset                      | Input wdinfo path                                                                                                                           | Images         | % of grand total   |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|----------------|--------------------|
| nvcommercial_700m            | `s3://nv-00-10206-webdataset-images/webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/nvcommercial_700m` | 32,938,121     | 36.07%             |
| MMC4                         | `s3://nv-00-10206-webdataset-images/webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/MMC4`              | 181,721        | 0.20%              |
| coyo_700m                    | `s3://nv-00-10206-webdataset-images/webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/coyo_700m`         | 56,363,516     | 61.72%             |
| red                          | `s3://nv-00-10206-webdataset-images/webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1/wdinfo/dual_caption/red`               | 561,038        | 0.61%              |
| v1_high_quality              | `s3://nv-00-10206-webdataset-images/webdataset_cosmos_lab_image_v1/webdataset_cosmos_lab_image_v1/v1_high_quality/wdinfo`                   | 1,239,517      | 1.36%              |
| pexels_residual_trustedK1_v2 | `s3://nv-00-10206-webdataset-images/webdataset_cosmos_lab_image_v1/v1_pexels_residual_trustedK1_v2/wdinfo/pexels_residual_trustedK1_v2`     | 37,800         | 0.04%              |
| **TOTAL**                    |                                                                                                                                             | **91,321,713** | **100.00%**        |

`coyo_700m` and `nvcommercial_700m` together account for **97.79%** of all images; the remaining 4 datasets contribute only ~2.2%.

---

## Category Distribution (per dataset, sorted alphabetically by category)

31 cluster categories observed across all datasets.

| Category                          | nvcommercial_700m   | MMC4        | coyo_700m      | red         | v1_high_quality   | pexels_residual_trustedK1_v2   | Total          | % of grand   | 1/√(frac)   |
|-----------------------------------|---------------------|-------------|----------------|-------------|-------------------|--------------------------------|----------------|--------------|-------------|
| a1_faces_portraits                | —                   | 557         | 22,386,604     | 111,337     | 306,234           | 27,524                         | 22,832,256     | 25.00%       | 1.9999      |
| a2_human_activities_sports        | —                   | 2,211       | —              | 71,845      | 100,752           | 4,883                          | 179,691        | 0.20%        | 22.5436     |
| a3_fashion_apparel-on-body        | —                   | 1,125       | 4,768,693      | 40,790      | 118,701           | 487                            | 4,929,796      | 5.40%        | 4.3040      |
| b1_packshots_e-commerce           | 3,455,011           | 14,244      | 4,671,370      | 27,247      | 59,790            | 12                             | 8,227,674      | 9.01%        | 3.3316      |
| b2_household_tools_gadgets        | 4,381,199           | 20,571      | 7,961          | 11,489      | 46,203            | 7                              | 4,467,430      | 4.89%        | 4.5212      |
| b3_food_beverages                 | 8,069,212           | 25,988      | —              | 17,677      | 161,915           | 112                            | 8,274,904      | 9.06%        | 3.3220      |
| c1_animals                        | 2,451,567           | 6,147       | —              | 49,631      | 66,885            | 71                             | 2,574,301      | 2.82%        | 5.9560      |
| c2_plants_close-ups               | 1,409,827           | 3,893       | 1,154          | 4,643       | 43,376            | 8                              | 1,462,901      | 1.60%        | 7.9010      |
| c3_landscapes_weather             | 16                  | 9,832       | 3,718,860      | 24,874      | 76,611            | —                              | 3,830,193      | 4.19%        | 4.8829      |
| d1_indoor_scenes                  | 29                  | 31,527      | 11,197,245     | —           | 23,145            | 2,946                          | 11,254,892     | 12.32%       | 2.8485      |
| d2_architecture_urban_outdoors    | 310                 | 28,162      | 344            | 128,476     | 108,093           | 439                            | 265,824        | 0.29%        | 18.5349     |
| d3_vehicles_transport             | 4,662,255           | 16,931      | 3,276,905      | 39,112      | 69,933            | 120                            | 8,065,256      | 8.83%        | 3.3649      |
| e1_scanned_documents              | 25,638              | 78          | —              | 86          | 114               | —                              | 25,916         | 0.03%        | 59.3613     |
| e2_photos_with_signage            | 1,524,595           | 638         | 270            | 8,317       | 8,185             | 411                            | 1,542,416      | 1.69%        | 7.6946      |
| e3_handwritten_notes_whiteboards  | 107,348             | 61          | 169,666        | 303         | 487               | 57                             | 277,922        | 0.30%        | 18.1270     |
| f1_charts_plots                   | —                   | —           | 108,698        | —           | 197               | 3                              | 108,898        | 0.12%        | 28.9586     |
| f2_tables_spreadsheets            | 9,739               | 18          | 31,392         | —           | 48                | —                              | 41,197         | 0.05%        | 47.0819     |
| f3_diagrams_flowcharts_schematics | 49,283              | 134         | 108,399        | 32          | 120               | 3                              | 157,971        | 0.17%        | 24.0435     |
| f4_ui_screenshots_code_snippets   | 61                  | 105         | 121,505        | 17          | 650               | 2                              | 122,340        | 0.13%        | 27.3214     |
| g1_illustrations_comics_manga     | —                   | 771         | 964,985        | 4,948       | 985               | —                              | 971,689        | 1.06%        | 9.6945      |
| g2_paintings_drawings_fine_art    | —                   | 3,447       | —              | 12,878      | 4,712             | 84                             | 21,121         | 0.02%        | 65.7552     |
| g3_memes_typography_posters       | —                   | 241         | 209,545        | 255         | 203               | —                              | 210,244        | 0.23%        | 20.8413     |
| h1_3d_renders_cad_cgi             | 2,264,587           | 6,216       | 1,827,750      | —           | 7,831             | —                              | 4,106,384      | 4.50%        | 4.7158      |
| h2_game_imagery                   | 1,197,798           | 2,161       | —              | 1,140       | 6,210             | 11                             | 1,207,320      | 1.32%        | 8.6971      |
| h2_household_tools_gadgets        | 3,115               | —           | 1,338          | 2           | 4                 | —                              | 4,459          | 0.00%        | 143.1094    |
| h3_synthetic_photos_ai-generated  | 1,307,512           | 913         | —              | —           | 9,567             | 65                             | 1,318,057      | 1.44%        | 8.3238      |
| i1_medical_x-ray_ct_pathology     | 635,940             | —           | 853,760        | 660         | 3,861             | 36                             | 1,494,257      | 1.64%        | 7.8176      |
| i2_microscopy_lab_setups          | 449,923             | 17          | 742,853        | 16          | 4,219             | 512                            | 1,197,540      | 1.31%        | 8.7326      |
| i3_remote_sensing_maps            | 34,988              | 8           | 37,756         | 64          | 308               | —                              | 73,124         | 0.08%        | 35.3392     |
| i4_industrial_engineering         | 235,384             | 695         | 320,249        | 597         | 1,502             | 7                              | 558,434        | 0.61%        | 12.7880     |
| other                             | 662,784             | 5,030       | 836,214        | 4,602       | 8,676             | —                              | 1,517,306      | 1.66%        | 7.7580      |
| **TOTAL**                         | **32,938,121**      | **181,721** | **56,363,516** | **561,038** | **1,239,517**     | **37,800**                     | **91,321,713** | **100.00%**  | —           |

*`—` denotes 0 / category not present in that dataset. `1/√(frac)` = inverse-square-root resampling weight, where `frac` is the category's fraction of the grand total (not the percentage number) — higher weight for rarer clusters.*

---

## Rebalanced Sampling Weights

Each category's `1/√(frac)` weight rescaled against `other` (factor = 1.0), giving the relative oversampling factor for every other category. `Capped factor` clips anything above 5.0 down to 5.0. `Final factor = capped_factor / global_factor` (global_factor = 0.543166, see below) — normalized so `Σ(final_factor × total) = grand_total`.

| Category                          | Total          | % of grand   | 1/√(frac)    | Rebalance factor (vs other)   | Capped factor (max 5)   | Final factor   |
|-----------------------------------|----------------|--------------|--------------|-------------------------------|-------------------------|----------------|
| a1_faces_portraits                | 22,832,256     | 25.00%       | 1.9999       | 0.2578                        | 0.2578                  | 0.4746         |
| a2_human_activities_sports        | 179,691        | 0.20%        | 22.5436      | 2.9059                        | 2.9059                  | 5.3498         |
| a3_fashion_apparel-on-body        | 4,929,796      | 5.40%        | 4.3040       | 0.5548                        | 0.5548                  | 1.0214         |
| b1_packshots_e-commerce           | 8,227,674      | 9.01%        | 3.3316       | 0.4294                        | 0.4294                  | 0.7906         |
| b2_household_tools_gadgets        | 4,467,430      | 4.89%        | 4.5212       | 0.5828                        | 0.5828                  | 1.0729         |
| b3_food_beverages                 | 8,274,904      | 9.06%        | 3.3220       | 0.4282                        | 0.4282                  | 0.7884         |
| c1_animals                        | 2,574,301      | 2.82%        | 5.9560       | 0.7677                        | 0.7677                  | 1.4134         |
| c2_plants_close-ups               | 1,462,901      | 1.60%        | 7.9010       | 1.0184                        | 1.0184                  | 1.8750         |
| c3_landscapes_weather             | 3,830,193      | 4.19%        | 4.8829       | 0.6294                        | 0.6294                  | 1.1588         |
| d1_indoor_scenes                  | 11,254,892     | 12.32%       | 2.8485       | 0.3672                        | 0.3672                  | 0.6760         |
| d2_architecture_urban_outdoors    | 265,824        | 0.29%        | 18.5349      | 2.3891                        | 2.3891                  | 4.3985         |
| d3_vehicles_transport             | 8,065,256      | 8.83%        | 3.3649       | 0.4337                        | 0.4337                  | 0.7985         |
| e1_scanned_documents              | 25,916         | 0.03%        | 59.3613      | 7.6516                        | 5.0000                  | 9.2053         |
| e2_photos_with_signage            | 1,542,416      | 1.69%        | 7.6946       | 0.9918                        | 0.9918                  | 1.8260         |
| e3_handwritten_notes_whiteboards  | 277,922        | 0.30%        | 18.1270      | 2.3366                        | 2.3366                  | 4.3017         |
| f1_charts_plots                   | 108,898        | 0.12%        | 28.9586      | 3.7327                        | 3.7327                  | 6.8722         |
| f2_tables_spreadsheets            | 41,197         | 0.05%        | 47.0819      | 6.0688                        | 5.0000                  | 9.2053         |
| f3_diagrams_flowcharts_schematics | 157,971        | 0.17%        | 24.0435      | 3.0992                        | 3.0992                  | 5.7058         |
| f4_ui_screenshots_code_snippets   | 122,340        | 0.13%        | 27.3214      | 3.5217                        | 3.5217                  | 6.4837         |
| g1_illustrations_comics_manga     | 971,689        | 1.06%        | 9.6945       | 1.2496                        | 1.2496                  | 2.3006         |
| g2_paintings_drawings_fine_art    | 21,121         | 0.02%        | 65.7552      | 8.4758                        | 5.0000                  | 9.2053         |
| g3_memes_typography_posters       | 210,244        | 0.23%        | 20.8413      | 2.6864                        | 2.6864                  | 4.9459         |
| h1_3d_renders_cad_cgi             | 4,106,384      | 4.50%        | 4.7158       | 0.6079                        | 0.6079                  | 1.1191         |
| h2_game_imagery                   | 1,207,320      | 1.32%        | 8.6971       | 1.1211                        | 1.1211                  | 2.0639         |
| h2_household_tools_gadgets        | 4,459          | 0.00%        | 143.1094     | 18.4467                       | 5.0000                  | 9.2053         |
| h3_synthetic_photos_ai-generated  | 1,318,057      | 1.44%        | 8.3238       | 1.0729                        | 1.0729                  | 1.9753         |
| i1_medical_x-ray_ct_pathology     | 1,494,257      | 1.64%        | 7.8176       | 1.0077                        | 1.0077                  | 1.8552         |
| i2_microscopy_lab_setups          | 1,197,540      | 1.31%        | 8.7326       | 1.1256                        | 1.1256                  | 2.0723         |
| i3_remote_sensing_maps            | 73,124         | 0.08%        | 35.3392      | 4.5552                        | 4.5552                  | 8.3864         |
| i4_industrial_engineering         | 558,434        | 0.61%        | 12.7880      | 1.6484                        | 1.6484                  | 3.0347         |
| other                             | 1,517,306      | 1.66%        | 7.7580       | 1.0000                        | 1.0000                  | 1.8411         |
| **TOTAL**                         | **91,321,713** | **100.00%**  | **629.6713** | —                             | —                       | —              |

**Verification:** `Σ(final_factor × total) = 91,321,713 = grand_total`. ✓

### Global Factor

`global_factor = Σ(capped_factor_i × total_i) / grand_total` — term-by-term breakdown for verification:

| Category                                | capped_factor   | total_i        | capped_factor x total_i   |
|-----------------------------------------|-----------------|----------------|---------------------------|
| a1_faces_portraits                      | 0.2578          | 22,832,256     | 5,885,874.53              |
| a2_human_activities_sports              | 2.9059          | 179,691        | 522,155.37                |
| a3_fashion_apparel-on-body              | 0.5548          | 4,929,796      | 2,734,960.52              |
| b1_packshots_e-commerce                 | 0.4294          | 8,227,674      | 3,533,256.16              |
| b2_household_tools_gadgets              | 0.5828          | 4,467,430      | 2,603,547.26              |
| b3_food_beverages                       | 0.4282          | 8,274,904      | 3,543,382.77              |
| c1_animals                              | 0.7677          | 2,574,301      | 1,976,360.89              |
| c2_plants_close-ups                     | 1.0184          | 1,462,901      | 1,489,855.18              |
| c3_landscapes_weather                   | 0.6294          | 3,830,193      | 2,410,720.81              |
| d1_indoor_scenes                        | 0.3672          | 11,254,892     | 4,132,446.63              |
| d2_architecture_urban_outdoors          | 2.3891          | 265,824        | 635,087.67                |
| d3_vehicles_transport                   | 0.4337          | 8,065,256      | 3,498,208.30              |
| e1_scanned_documents                    | 5               | 25,916         | 129,580.00                |
| e2_photos_with_signage                  | 0.9918          | 1,542,416      | 1,529,809.48              |
| e3_handwritten_notes_whiteboards        | 2.3366          | 277,922        | 649,378.72                |
| f1_charts_plots                         | 3.7327          | 108,898        | 406,486.89                |
| f2_tables_spreadsheets                  | 5               | 41,197         | 205,985.00                |
| f3_diagrams_flowcharts_schematics       | 3.0992          | 157,971        | 489,581.81                |
| f4_ui_screenshots_code_snippets         | 3.5217          | 122,340        | 430,844.77                |
| g1_illustrations_comics_manga           | 1.2496          | 971,689        | 1,214,227.96              |
| g2_paintings_drawings_fine_art          | 5               | 21,121         | 105,605.00                |
| g3_memes_typography_posters             | 2.6864          | 210,244        | 564,804.82                |
| h1_3d_renders_cad_cgi                   | 0.6079          | 4,106,384      | 2,496,125.21              |
| h2_game_imagery                         | 1.1211          | 1,207,320      | 1,353,467.35              |
| h2_household_tools_gadgets              | 5               | 4,459          | 22,295.00                 |
| h3_synthetic_photos_ai-generated        | 1.0729          | 1,318,057      | 1,414,176.72              |
| i1_medical_x-ray_ct_pathology           | 1.0077          | 1,494,257      | 1,505,737.40              |
| i2_microscopy_lab_setups                | 1.1256          | 1,197,540      | 1,347,974.27              |
| i3_remote_sensing_maps                  | 4.5552          | 73,124         | 333,093.81                |
| i4_industrial_engineering               | 1.6484          | 558,434        | 920,497.29                |
| other                                   | 1               | 1,517,306      | 1,517,306.00              |
| **SUM (= Σ capped_factor_i × total_i)** |                 | **91,321,713** | **49,602,833.60**         |

`global_factor = SUM / grand_total = 49,602,833.60 / 91,321,713` = **0.543166**

---

## Notes

- Top 3 categories (`a1_faces_portraits`, `d1_indoor_scenes`, `b3_food_beverages`) make up **46.38%** of the entire cluster distribution.
- `coyo_700m` is heavily concentrated in `a1_faces_portraits` (22.4M of its 56.4M images, ~39.7%) and `d1_indoor_scenes` (11.2M, ~19.9%).
- `nvcommercial_700m` skews toward product/commercial categories: `b3_food_beverages`, `b2_household_tools_gadgets`, `d3_vehicles_transport`, `b1_packshots_e-commerce`.
- Long-tail categories (`h2_household_tools_gadgets`, `g2_paintings_drawings_fine_art`, `e1_scanned_documents`, `f2_tables_spreadsheets`, `i3_remote_sensing_maps`) each contribute < 0.1% of the grand total.
