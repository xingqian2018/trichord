---
name: cmd_ugb
description: Give me the commend evaluating a ugb
user_invocable: true
---

# Summary

In this skill, based on users input, we need to provide the shell commend execute the evaluation.

The evaluation is specifically contains two stage, involving two script.
- ```projects/cosmos3/vfm/evaluation/text_to_image/inference_unigenbench_distributed.py``` 
- ```projects/cosmos3/vfm/evaluation/text_to_image/compute_unigenbench_metric.py```

First stage generate the image and second stage evaluate the image.

# Stage 1 - Inference

* A sample command I want to see on console for first stage is below:

```
slaunch small 1 <ugb1170L or ugb1170uL>_<some_tags> \
    projects/cosmos3/vfm/evaluation/text_to_image/inference_unigenbench_distributed.py \
    --experiment_name t2i_mot_expDAITR004_002_8b_480p_1sst3M_lr5em5 \
    --checkpoint_path <checkpoint_path> \
    --credential_path credentials/gcp_checkpoint.secret \
    --benchmark_name <benchmark_name> --num_batch_size 32 \
    --guidance 4.0 --num_inference_steps 50 --height 640 --width 640 --use_ema \
    --output_path s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/t2i_mot_expDAITR/t2i_mot_expDAITR004_002_8b_480p_1sst3M_lr5em5/EveryNEval/unigenbench-on-<benchmark_name>/iter_000050000/
```

### ```--experiment_name```
- You shouldn't change the ```--experiment_name```, unless specified

### ```--checkpoint_path```
- The user should specific the <checkpoint_path> as one of the conversational input.
- ```--checkpoint_path``` usually ends with a /model/
- If the user provide a path not end with /model/, help the user to append it.

### ```--width --height```
-  If the user is trying to restrict the output with certain resolution and aspect ratio. 
- You should follow this table and map the correct width and height from widthxheight to the argument --width --height

Tier    | 1:1         | 4:3         | 3:4         | 16:9        | 9:16
--------|-------------|-------------|-------------|-------------|-------------
256     | 256x256     | 320x256     | 256x320     | 320x192     | 192x320
480     | 640x640     | 736x544     | 544x736     | 832x480     | 480x832
720     | 960x960     | 1104x832    | 832x1104    | 1280x720    | 720x1280
1080    | 1440x1440   | 1664x1248   | 1248x1664   | 1920x1080   | 1080x1920
1280    | 1712x1712   | 1968x1472   | 1472x1968   | 2272x1280   | 1280x2272
2048    | 2728x2728   | 3160x2368   | 2368x3160   | 3640x2048   | 2048x3640
gt_2048 | 5464x5464   | 6304x4728   | 4728x6304   | 7280x4096   | 4096x7280


### ```--output_path```
- Regarding output, if the user ask to generate image at the exact model checkpoint location.
- That usually means that ```--output_path``` argument should looks like

```
--checkpoint_path <some_common_prefix>/checkpoints/iter_000050000/model/
--output_path     <some_common_prefix>/EveryNEval/unigenbench-on-<benchmark_name>/iter_000050000/
```

* If the user ask to generate image output to a default location, or the user doesn't say anything about this.
* That means:

```
--output_path s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/evaluation/text_to_image/unigenbench/<benchmark_name>/xingqianx_<somename>
```

### ```--benchmark_name```
- ```--benchmark_name``` can be ```v2_1170L_G3F``` (default) or ```v2_1170uL_G3F```
- Use v2_1170L_G3F as default if the user doesn't specify.
- Remeber that this should also be map to other input's <benchmark_name> placeholder


# Stage 2 - Score

* A sample command I want to see on console for this stage (i.e. score stage) is below:

```
slaunch small 1 expDAITR004_pre_ugb1170L_score \
    projects/cosmos3/vfm/evaluation/text_to_image/compute_unigenbench_metric.py \
    --input_folder s3://nv-00-10206-checkpoint-experiments/cosmos3_vfm/t2i_mot_expDAITR/t2i_mot_expDAITR004_000_8b_480p_1sst1M/EveryNEval/unigenbench-on-v2_1170L_G3F/iter_000000000/ \
    --judge_model gemini-3-flash \
    --benchmark_name v2_1170L_G3F \
    --extension webp \
    --num_concurrency 32
```


EOF