# Rubric Scores — aa_opus Benchmark

- Benchmark: `aa_opus` (1567 prompts — `artificial_analysis_1567_opus47_t2i_1024x1024.csv`)
- Judge model: `gemini-3.1-pro` (signature `g3p1p`)
- Script: `projects/cosmos3/vfm/evaluation/text_to_image/compute_rubric.py`
- Result root: `gcs:nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench/aa_opus/`
- Result file: `rubric_result_g3p1p.json`
- Scores are 0–10; higher is better.

---

## Overall Score Table

| Model                                         | prompt_adherence   | visual_quality   | aesthetics   | physical_plausibility   | category   | overall   | Success   |
|-----------------------------------------------|--------------------|------------------|--------------|-------------------------|------------|-----------|-----------|
| cosmos3_image_v1_v1p5_iter108k                | 6.86               | 8.11             | 8.62         | 7.2                     | 7.55       | **6.68**  | 1567/1567 |
| cosmos3_frozen_midtrain_v3_iter1800           | 6.05               | 7.41             | 8.03         | 6.55                    | 6.64       | 5.84      | 1567/1567 |
| cosmos3_image_only_v3p1_iter18k               | 5.28               | 6.54             | 7.25         | 5.96                    | 6.06       | 5.25      | 1567/1567 |
| cosmos3_t2i_exp009_union5_from_frozen_iter10k | 6.45               | 7.91             | 8.41         | 6.92                    | 7.2        | 6.32      | 1567/1567 |
| cosmos3_t2i_exp009_union5_from_frozen_iter15k | 6.55               | 7.98             | 8.47         | 6.99                    | 7.37       | 6.45      | 1566/1567 |
| cosmos3_t2i_exp009_union5_from_frozen_iter20k | 6.52               | 7.96             | 8.45         | 6.95                    | 7.25       | 6.41      | 1567/1567 |
| cosmos3_t2i_exp009_union5_from_frozen_iter25k | 6.52               | 7.96             | 8.47         | 6.94                    | 7.28       | 6.40      | 1567/1567 |
| cosmos3_image_only_v3p3_iter22k               | 5.9                | 7.18             | 7.89         | 6.47                    | 6.72       | 5.88      | 1567/1567 |

---

## Big-Class Breakdown — overall_score (mean)

| Model                                         | Text Gen   | Entity Layout   | Action   | Style   | Logical Reasoning   | Relationship   | Attribute   | Compound   | World Knowledge   |
|-----------------------------------------------|------------|-----------------|----------|---------|---------------------|----------------|-------------|------------|-------------------|
| cosmos3_image_v1_v1p5_iter108k                | 6.1        | 6.46            | 6.6      | 6.96    | 6.67                | 6.86           | 6.64        | 6.39       | 6.34              |
| cosmos3_frozen_midtrain_v3_iter1800           | 4.42       | 5.7             | 5.79     | 6.01    | 5.92                | 6.09           | 5.84        | 5.83       | 5.61              |
| cosmos3_image_only_v3p1_iter18k               | 5.08       | 4.96            | 5.06     | 5.32    | 5.1                 | 5.52           | 5.26        | 5.45       | 4.56              |
| cosmos3_image_only_v3p3_iter22k               | 5.95       | 5.65            | 5.7      | 6.01    | 5.72                | 6.1            | 5.89        | 6          | 5.28              |
| cosmos3_t2i_exp009_union5_from_frozen_iter10k | 5.28       | 6.18            | 6.25     | 6.54    | 6.32                | 6.54           | 6.31        | 6.17       | 5.94              |
| cosmos3_t2i_exp009_union5_from_frozen_iter15k | 5.61       | 6.3             | 6.38     | 6.66    | 6.46                | 6.69           | 6.37        | 6.1        | 6.11              |
| cosmos3_t2i_exp009_union5_from_frozen_iter20k | 5.66       | 6.28            | 6.35     | 6.67    | 6.45                | 6.63           | 6.31        | 6.09       | 6.05              |
| cosmos3_t2i_exp009_union5_from_frozen_iter25k | 5.51       | 6.22            | 6.28     | 6.69    | 6.48                | 6.6            | 6.35        | 6.16       | 5.94              |
