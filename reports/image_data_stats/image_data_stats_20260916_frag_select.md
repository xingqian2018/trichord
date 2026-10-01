# LanceDB `image_meta_table_full` — Fragment-Selected Stats Report

Run: `stats_20260916_frag_select.json` (`latest_fragment_id=40578`)

Scoped to `--fragment_id 37008-37044,37045,37159,37168,37169-37170,37171,40561-40566,40567-40577,40578` — the recover-captioning pipeline's target fragments.

| dataset                      | total      | filtered   | dedup   | filtered_and_dedup   | captioned   | filtered_and_captioned   | filtered_and_dedup_and_captioned   | captioning_visited   | recover_captioned   | recover_captioned_visited   | tobe_remove   |
|------------------------------|------------|------------|---------|----------------------|-------------|--------------------------|------------------------------------|----------------------|---------------------|-----------------------------|---------------|
| **REAL**                     |            |            |         |                      |             |                          |                                    |                      |                     |                             |               |
| human_sft                    | 70,159     | 67,442     | 0       | 0                    | 67,442      | 67,442                   | 0                                  | 67,442               | 0                   | 0                           | 0             |
| pexels_residual_trustedK1_v2 | 38,804     | 37,844     | 0       | 0                    | 37,844      | 37,844                   | 0                                  | 37,844               | 0                   | 0                           | 0             |
| **OTHER**                    |            |            |         |                      |             |                          |                                    |                      |                     |                             |               |
| megalith-10m                 | 9,279,122  | 5,086,678  | 0       | 0                    | 0           | 0                        | 0                                  | 0                    | 0                   | 0                           | 0             |
| multiaspect-4k-1m            | 1,007,138  | 0          | 0       | 0                    | 0           | 0                        | 0                                  | 0                    | 0                   | 0                           | 0             |
| photo-concept-bucket         | 567,698    | 0          | 0       | 0                    | 0           | 0                        | 0                                  | 0                    | 0                   | 0                           | 0             |
| Aesthetic-Train-V2           | 105,288    | 0          | 0       | 0                    | 0           | 0                        | 0                                  | 0                    | 0                   | 0                           | 0             |
| LSDIR                        | 84,991     | 0          | 0       | 0                    | 0           | 0                        | 0                                  | 0                    | 0                   | 0                           | 0             |
| unsplash_lite                | 25,000     | 0          | 0       | 0                    | 0           | 0                        | 0                                  | 0                    | 0                   | 0                           | 0             |
| flickr2k                     | 2,650      | 0          | 0       | 0                    | 0           | 0                        | 0                                  | 0                    | 0                   | 0                           | 0             |
| **TOTAL**                    | 11,180,850 | 5,191,964  | 0       | 0                    | 105,286     | 105,286                  | 0                                  | 105,286              | 0                   | 0                           | 0             |
