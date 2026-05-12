Can you help look into the folder:

/home/xingqianx/log/slurm_20260317_ugb_baselline

The folder contains a group of <score_type>_score_<model_type>.<pid>.e files.

At the end of each file: you will fine a table for scores.

A summary will looks like

============================================================
Summary:
all: 78.30% orig: 83.06% phi: 73.93%
============================================================

And three session for primary dimensions breakdown will looks like

Primary Dimensions:
| Primary Dimension   |   Correct |   Total | Accuracy   |
|:--------------------|----------:|--------:|:-----------|
| Action              |       457 |     622 | 73.47%     |
| Attribute           |      1169 |    1311 | 89.17%     |
| Compound            |       287 |     343 | 83.67%     |
| Entity Layout       |       274 |     315 | 86.98%     |
| Grammar             |       156 |     197 | 79.19%     |
| Logical Reasoning   |        82 |     102 | 80.39%     |
| Relationship        |       373 |     464 | 80.39%     |
| Style               |       285 |     301 | 94.68%     |
| Text Generation     |        22 |      92 | 23.91%     |
| World Knowledge     |       151 |     173 | 87.28%     |

I need you to give me a table that fomated like this:

	Nano-Banana-Pro	Qwen-Image-2512	Qwen-Image	Flux_2_klein_9b	Flux_1_kontext_dev	Glm-Image	Z-Image-Turbo	SD-3.5-Large
Overall	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx
Action	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx
Attribute	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx
Compound	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx
Entity Layout	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx
Grammar	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx
Logical Reasoning	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx
Relationship	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx
Style	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx
Text Generation	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx
World Knowledge	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx	xx.xx

Help collect one type of score, a score type (i.e ugb1170L all, or ugb1170uL phi, etc..), the fill all xx.xx in the table. If a score cannot be found, skip it by keeping the xx.xx

You may find tool: /home/xingqianx/log/cc_tools/gather_ugb_score.sh perfectly fitting this task. (Don't edit this file)

Output only the table format as the template above.
