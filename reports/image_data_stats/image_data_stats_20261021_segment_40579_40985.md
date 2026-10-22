# LanceDB `image_meta_table_full` — Segment Report: fragments 40579-40985

Scope: fragments 40579-40985, i.e. everything added to the main table after stats_20260915.

## Counts

| dataset                      | total          | filter<br>visited   | filtered   | dedup<br>visited   | dedup   | caption<br>visited   | captioned   | recover<br>visited   | recover<br>captioned   | short side<br>≥512   | short side<br>≥1024   | short side<br>≥2048   |
|------------------------------|----------------|---------------------|------------|--------------------|---------|----------------------|-------------|----------------------|------------------------|----------------------|-----------------------|-----------------------|
| **NEW BY 20261014 (sliced)** |                |                     |            |                    |         |                      |             |                      |                        |                      |                       |                       |
| commoncatalog-cc-by          | 14,581,672     | 0                   | 0          | 0                  | 0       | 0                    | 0           | 14,581,672           | 14,581,671             | 11,856,004           | 9,434,920             | 5,557,946             |
| megascenes                   | 8,715,454      | 0                   | 0          | 0                  | 0       | 0                    | 0           | 8,715,454            | 8,711,565              | 8,132,234            | 6,783,023             | 13,627                |
| commoncatalog-cc-by-sa       | 7,891,493      | 0                   | 0          | 0                  | 0       | 0                    | 0           | 7,891,493            | 7,891,493              | 7,538,563            | 6,121,380             | 3,533,792             |
| Fine-T2I-curated             | 168,424        | 0                   | 0          | 0                  | 0       | 0                    | 0           | 168,424              | 168,424                | 168,424              | 138,411               | 129,628               |
| 4KLSDB                       | 129,484        | 0                   | 0          | 0                  | 0       | 0                    | 0           | 129,484              | 129,484                | 129,484              | 129,484               | 129,484               |
| hq-50k                       | 29,189         | 0                   | 0          | 0                  | 0       | 0                    | 0           | 29,189               | 29,189                 | 27,998               | 26,448                | 3,604                 |
| uhd-iqa                      | 4,269          | 0                   | 0          | 0                  | 0       | 0                    | 0           | 4,269                | 4,269                  | 4,269                | 4,268                 | 4,087                 |
| *subtotal*                   | *31,519,985*   | *0*                 | *0*        | *0*                | *0*     | *0*                  | *0*         | *31,519,985*         | *31,516,095*           | *27,856,976*         | *22,637,934*          | *9,372,168*           |
| **NEW SINCE 20261014**       |                |                     |            |                    |         |                      |             |                      |                        |                      |                       |                       |
| megalith-10m-original        | 8,716,221      | 0                   | 0          | 0                  | 0       | 0                    | 0           | 8,716,221            | 8,715,145              | 8,541,249            | 7,667,007             | 5,804,499             |
| *subtotal*                   | *8,716,221*    | *0*                 | *0*        | *0*                | *0*     | *0*                  | *0*         | *8,716,221*          | *8,715,145*            | *8,541,249*          | *7,667,007*           | *5,804,499*           |
| **TOTAL**                    | **40,236,206** | **0**               | **0**      | **0**              | **0**   | **0**                | **0**       | **40,236,206**       | **40,231,240**         | **36,398,225**       | **30,304,941**        | **15,176,667**        |

## Coverage and resolution ratios

| dataset                      | filter coverage   | filter pass rate   | dedup coverage   | caption coverage   | recover coverage   | short side ≥512   | short side ≥1024   | short side ≥2048   |
|------------------------------|-------------------|--------------------|------------------|--------------------|--------------------|-------------------|--------------------|--------------------|
| **NEW BY 20261014 (sliced)** |                   |                    |                  |                    |                    |                   |                    |                    |
| commoncatalog-cc-by          | 0.0%              | –                  | 0.0%             | 0.0%               | 100.0%             | 81.3%             | 64.7%              | 38.1%              |
| megascenes                   | 0.0%              | –                  | 0.0%             | 0.0%               | 100.0%             | 93.3%             | 77.8%              | 0.2%               |
| commoncatalog-cc-by-sa       | 0.0%              | –                  | 0.0%             | 0.0%               | 100.0%             | 95.5%             | 77.6%              | 44.8%              |
| Fine-T2I-curated             | 0.0%              | –                  | 0.0%             | 0.0%               | 100.0%             | 100.0%            | 82.2%              | 77.0%              |
| 4KLSDB                       | 0.0%              | –                  | 0.0%             | 0.0%               | 100.0%             | 100.0%            | 100.0%             | 100.0%             |
| hq-50k                       | 0.0%              | –                  | 0.0%             | 0.0%               | 100.0%             | 95.9%             | 90.6%              | 12.3%              |
| uhd-iqa                      | 0.0%              | –                  | 0.0%             | 0.0%               | 100.0%             | 100.0%            | 100.0%             | 95.7%              |
| **NEW SINCE 20261014**       |                   |                    |                  |                    |                    |                   |                    |                    |
| megalith-10m-original        | 0.0%              | –                  | 0.0%             | 0.0%               | 100.0%             | 98.0%             | 88.0%              | 66.6%              |
| **TOTAL**                    | **0.0%**          | **–**              | **0.0%**         | **0.0%**           | **100.0%**         | **90.5%**         | **75.3%**          | **37.7%**          |

## Sliced tables

| dataset                | path                                                                                                    |
|------------------------|---------------------------------------------------------------------------------------------------------|
| commoncatalog-cc-by    | `s3://nv-00-10206-vfm/lancedb/image/regular/commoncatalog-cc-by_slice_from_maintable_20261014.lance`    |
| megascenes             | `s3://nv-00-10206-vfm/lancedb/image/regular/megascenes_slice_from_maintable_20261014.lance`             |
| commoncatalog-cc-by-sa | `s3://nv-00-10206-vfm/lancedb/image/regular/commoncatalog-cc-by-sa_slice_from_maintable_20261014.lance` |
| Fine-T2I-curated       | `s3://nv-00-10206-vfm/lancedb/image/regular/Fine-T2I-curated_slice_from_maintable_20261014.lance`       |
| 4KLSDB                 | `s3://nv-00-10206-vfm/lancedb/image/regular/4KLSDB_slice_from_maintable_20261014.lance`                 |
| hq-50k                 | `s3://nv-00-10206-vfm/lancedb/image/regular/hq-50k_slice_from_maintable_20261014.lance`                 |
| uhd-iqa                | `s3://nv-00-10206-vfm/lancedb/image/regular/uhd-iqa_slice_from_maintable_20261014.lance`                |
| megalith-10m-original  | not sliced                                                                                              |
