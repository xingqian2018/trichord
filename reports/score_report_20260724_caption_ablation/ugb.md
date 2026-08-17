# UGB Score Report — Caption Ablation Evaluation

Gen: `v2_1170L_opus4p7_mx_tier1`, 50 steps, neg prompt on. Scored on `v2_1170L` with `gemini-3.1-pro`.


## baseline

baseline = cosmos3_ga_64bm32b_t2ionly_exp009_union5_from_frozen_midtrain_lr5em5_iter29k

| Run      | Benchmark                   | Images   | all       | orig      | phi       | success   |
|----------|-----------------------------|----------|-----------|-----------|-----------|-----------|
| baseline | v2_1170L_opus               | 1170 png | **90.94** | **92.76** | **89.28** | 1170/1170 |
| baseline | v2_1170L_opus4p7_mx_tier0   | 1170 png | 88.88     | 90.84     | 87.08     | 1170/1170 |
| baseline | v2_1170L_opus4p7_mx_tier1   | 1170 png | 88.97     | 90.99     | 87.11     | 1170/1170 |
| baseline | v2_1170L_opus4p7_gc_tier4   | 1170 png | 87.78     | 91.33     | 84.53     | 1170/1170 |
| baseline | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 89.11     | 91.02     | 87.36     | 1170/1170 |
| baseline | v2_1170L_opus4p7_gc         | 1170 png | 88.33     | **91.99** | 84.98     | 1170/1170 |


## cosmos3plus_64bm32b_t2ionly_exp001 (iter 4k)

| Run                                                     | Benchmark                 | Images   | all       | orig      | phi       | success   |
|---------------------------------------------------------|---------------------------|----------|-----------|-----------|-----------|-----------|
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter4k   | v2_1170L_opus             | 1170 png | 90.22     | 91.99     | 88.60     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter4k   | v2_1170L_opus             | 1170 png | 90.11     | 92.27     | 88.13     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter4k   | v2_1170L_opus             | 1170 png | 90.69     | 92.50     | 89.02     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter4k | v2_1170L_opus             | 1170 png | 90.46     | 92.30     | 88.77     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter4k        | v2_1170L_opus             | 1170 png | **90.91** | **93.04** | **88.95** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter4k   | v2_1170L_opus4p7_mx_tier0 | 1170 png | 89.61     | 91.17     | 88.18     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter4k   | v2_1170L_opus4p7_mx_tier0 | 1170 png | **89.82** | **91.40** | **88.37** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter4k   | v2_1170L_opus4p7_mx_tier0 | 1170 png | 88.93     | 90.89     | 87.13     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter4k | v2_1170L_opus4p7_mx_tier0 | 1170 png | 88.48     | 89.90     | 87.18     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter4k        | v2_1170L_opus4p7_mx_tier0 | 1170 png | 88.65     | 90.69     | 86.78     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter4k   | v2_1170L_opus4p7_mx_tier1 | 1170 png | 89.22     | 90.94     | 87.64     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter4k   | v2_1170L_opus4p7_mx_tier1 | 1170 png | **89.74** | **92.04** | 87.62     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter4k   | v2_1170L_opus4p7_mx_tier1 | 1170 png | 89.32     | 91.71     | 87.13     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter4k | v2_1170L_opus4p7_mx_tier1 | 1170 png | 88.95     | 90.99     | 87.08     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter4k        | v2_1170L_opus4p7_mx_tier1 | 1170 png | 88.86     | 90.66     | **87.20** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter4k   | v2_1170L_opus4p7_gc_tier4 | 1170 png | 89.15     | 91.86     | 86.66     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter4k   | v2_1170L_opus4p7_gc_tier4 | 1170 png | 89.36     | 91.28     | 87.60     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter4k   | v2_1170L_opus4p7_gc_tier4 | 1170 png | **89.66** | **91.71** | **87.78** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter4k | v2_1170L_opus4p7_gc_tier4 | 1170 png | 89.28     | 91.51     | 87.25     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter4k        | v2_1170L_opus4p7_gc_tier4 | 1170 png | 89.05     | 91.33     | 86.96     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter4k   | v2_1170L_opus4p7_gc_tier4 | 1170 png | 89.15     | 91.86     | 86.66     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter4k   | v2_1170L_opus4p7_gc_tier4 | 1170 png | 89.36     | 91.28     | 87.60     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter4k   | v2_1170L_opus4p7_gc_tier4 | 1170 png | **89.66** | **91.71** | **87.78** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter4k | v2_1170L_opus4p7_gc_tier4 | 1170 png | 89.28     | 91.51     | 87.25     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter4k        | v2_1170L_opus4p7_gc_tier4 | 1170 png | 89.05     | 91.33     | 86.96     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter4k   | v2_1170L_opus4p7_gc       | 1170 png | 88.78     | 91.96     | 85.86     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter4k   | v2_1170L_opus4p7_gc       | 1170 png | 88.89     | 91.73     | 86.29     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter4k   | v2_1170L_opus4p7_gc       | 1170 png | 89.66     | 92.24     | 87.29     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter4k | v2_1170L_opus4p7_gc       | 1170 png | **90.21** | **92.14** | **88.44** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter4k        | v2_1170L_opus4p7_gc       | 1170 png | 90.02     | 92.19     | 88.02     | 1170/1170 |


## cosmos3plus_64bm32b_t2ionly_exp001 (iter 5k)

| Run                                                     | Benchmark                   | Images   | all       | orig      | phi       | success   |
|---------------------------------------------------------|-----------------------------|----------|-----------|-----------|-----------|-----------|
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter5k   | v2_1170L_opus               | 1170 png | 90.28     | 91.99     | 88.72     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter5k   | v2_1170L_opus               | 1170 png | 89.94     | 91.86     | 88.18     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter5k   | v2_1170L_opus               | 1170 png | 90.54     | 92.17     | 89.05     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter5k | v2_1170L_opus               | 1170 png | 90.35     | 92.42     | 88.44     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter5k        | v2_1170L_opus               | 1170 png | 90.27     | 92.24     | 88.46     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter5k   | v2_1170L_opus4p7_mx_tier0   | 1170 png | **89.12** | **91.20** | 87.22     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter5k   | v2_1170L_opus4p7_mx_tier0   | 1170 png | 88.97     | 90.56     | **87.50** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter5k   | v2_1170L_opus4p7_mx_tier0   | 1170 png | 88.47     | 90.66     | 86.45     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter5k | v2_1170L_opus4p7_mx_tier0   | 1170 png | 88.26     | 90.21     | 86.47     | 1169/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter5k        | v2_1170L_opus4p7_mx_tier0   | 1170 png | 88.21     | 90.36     | 86.24     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter5k   | v2_1170L_opus4p7_mx_tier1   | 1170 png | 88.98     | 90.82     | 87.29     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter5k   | v2_1170L_opus4p7_mx_tier1   | 1170 png | **89.22** | **91.53** | 87.11     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter5k   | v2_1170L_opus4p7_mx_tier1   | 1170 png | 89.03     | 91.07     | 87.15     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter5k | v2_1170L_opus4p7_mx_tier1   | 1170 png | 88.65     | 91.25     | 86.26     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter5k        | v2_1170L_opus4p7_mx_tier1   | 1170 png | 88.95     | 91.35     | **87.29** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter5k   | v2_1170L_opus4p7_gc_tier4   | 1170 png | **89.16** | 90.94     | **87.53** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter5k   | v2_1170L_opus4p7_gc_tier4   | 1170 png | 89.14     | 90.94     | 87.48     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter5k   | v2_1170L_opus4p7_gc_tier4   | 1170 png | 88.70     | **91.58** | 86.05     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter5k | v2_1170L_opus4p7_gc_tier4   | 1170 png | 88.81     | 91.20     | 86.61     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter5k        | v2_1170L_opus4p7_gc_tier4   | 1170 png | 88.77     | 91.20     | 86.54     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter5k   | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 89.26     | 91.07     | 87.60     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter5k   | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | **89.27** | **91.07** | **87.62** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter5k   | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 88.67     | 90.84     | 86.68     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter5k | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 87.95     | 89.87     | 86.19     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter5k        | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 88.48     | 90.82     | 86.33     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter5k   | v2_1170L_opus4p7_gc         | 1170 png | 88.60     | 91.48     | 85.96     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter5k   | v2_1170L_opus4p7_gc         | 1170 png | 88.66     | 91.35     | 86.19     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter5k   | v2_1170L_opus4p7_gc         | 1170 png | 89.42     | 92.12     | 86.94     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter5k | v2_1170L_opus4p7_gc         | 1170 png | **89.59** | 91.94     | **87.43** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter5k        | v2_1170L_opus4p7_gc         | 1170 png | 89.41     | 91.63     | 87.36     | 1170/1170 |


## cosmos3plus_64bm32b_t2ionly_exp001 (iter 6k)

| Run                                                     | Benchmark                   | Images   | all       | orig      | phi       | success   |
|---------------------------------------------------------|-----------------------------|----------|-----------|-----------|-----------|-----------|
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter6k   | v2_1170L_opus               | 1170 png | 89.91     | 91.43     | 88.51     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter6k   | v2_1170L_opus               | 1170 png | 90.17     | 92.04     | 88.46     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter6k   | v2_1170L_opus               | 1170 png | **90.67** | **92.35** | **89.14** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter6k | v2_1170L_opus               | 1170 png | 90.39     | 92.14     | 88.79     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter6k        | v2_1170L_opus               | 1170 png | 90.63     | 92.22     | 89.16     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter6k   | v2_1170L_opus4p7_mx_tier0   | 1170 png | **88.60** | 90.33     | **87.01** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter6k   | v2_1170L_opus4p7_mx_tier0   | 1170 png | 88.22     | 90.23     | 86.38     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter6k   | v2_1170L_opus4p7_mx_tier0   | 1170 png | 88.61     | **91.22** | 86.22     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter6k | v2_1170L_opus4p7_mx_tier0   | 1170 png | 88.38     | 90.48     | 86.45     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter6k        | v2_1170L_opus4p7_mx_tier0   | 1170 png | 88.36     | 90.41     | 86.47     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter6k   | v2_1170L_opus4p7_mx_tier1   | 1170 png | 88.69     | 90.51     | 87.01     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter6k   | v2_1170L_opus4p7_mx_tier1   | 1170 png | 89.11     | 91.10     | 87.29     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter6k   | v2_1170L_opus4p7_mx_tier1   | 1170 png | 88.94     | 91.17     | 86.89     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter6k | v2_1170L_opus4p7_mx_tier1   | 1170 png | 88.98     | 91.15     | 86.99     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter6k        | v2_1170L_opus4p7_mx_tier1   | 1170 png | **89.03** | **91.23** | **87.36** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter6k   | v2_1170L_opus4p7_gc_tier4   | 1170 png | 88.20     | 90.31     | 86.26     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter6k   | v2_1170L_opus4p7_gc_tier4   | 1170 png | 88.16     | 90.69     | 85.84     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter6k   | v2_1170L_opus4p7_gc_tier4   | 1170 png | **89.14** | **91.56** | **86.92** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter6k | v2_1170L_opus4p7_gc_tier4   | 1170 png | 88.72     | 91.20     | 86.45     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter6k        | v2_1170L_opus4p7_gc_tier4   | 1170 png | 88.94     | 90.82     | 87.22     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter6k   | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | **88.86** | 90.33     | **87.50** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter6k   | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 88.20     | 90.28     | 86.29     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter6k   | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 88.66     | **90.89** | 86.61     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter6k | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 88.32     | 90.59     | 86.24     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter6k        | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 88.27     | 90.10     | 86.59     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter6k   | v2_1170L_opus4p7_gc         | 1170 png | 88.29     | 91.20     | 85.63     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter6k   | v2_1170L_opus4p7_gc         | 1170 png | 88.03     | 91.02     | 85.28     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter6k   | v2_1170L_opus4p7_gc         | 1170 png | 89.30     | 91.94     | 86.87     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter6k | v2_1170L_opus4p7_gc         | 1170 png | 89.37     | **92.19** | 86.78     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter6k        | v2_1170L_opus4p7_gc         | 1170 png | **89.61** | 91.96     | **87.46** | 1170/1170 |


## cosmos3plus_64bm32b_t2ionly_exp001 (iter 8k)

| Run                                                     | Benchmark                   | Images   | all       | orig      | phi       | success   |
|---------------------------------------------------------|-----------------------------|----------|-----------|-----------|-----------|-----------|
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter8k   | v2_1170L_opus               | 1170 png | 87.12     | 89.39     | 85.05     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter8k   | v2_1170L_opus               | 1170 png | 86.63     | 88.83     | 84.62     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter8k   | v2_1170L_opus               | 1170 png | **90.60** | **92.83** | **88.56** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter8k | v2_1170L_opus               | 1170 png | 89.94     | 91.73     | 88.30     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter8k        | v2_1170L_opus               | 1170 png | 90.00     | 91.33     | 88.79     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter8k   | v2_1170L_opus4p7_mx_tier0   | 1170 png | 83.85     | 86.79     | 81.16     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter8k   | v2_1170L_opus4p7_mx_tier0   | 1170 png | 82.80     | 85.41     | 80.41     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter8k   | v2_1170L_opus4p7_mx_tier0   | 1170 png | **88.54** | **90.79** | **86.47** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter8k | v2_1170L_opus4p7_mx_tier0   | 1170 png | 88.10     | 89.82     | 86.52     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter8k        | v2_1170L_opus4p7_mx_tier0   | 1170 png | 88.16     | 90.31     | 86.19     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter8k   | v2_1170L_opus4p7_mx_tier1   | 1170 png | 84.76     | 87.24     | 82.47     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter8k   | v2_1170L_opus4p7_mx_tier1   | 1170 png | 85.67     | 88.52     | 83.06     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter8k   | v2_1170L_opus4p7_mx_tier1   | 1170 png | **89.03** | **91.35** | 86.89     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter8k | v2_1170L_opus4p7_mx_tier1   | 1170 png | 89.14     | 91.17     | **87.27** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter8k        | v2_1170L_opus4p7_mx_tier1   | 1170 png | 88.77     | 90.92     | 86.80     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter8k   | v2_1170L_opus4p7_gc_tier4   | 1170 png | 85.61     | 88.65     | 82.82     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter8k   | v2_1170L_opus4p7_gc_tier4   | 1170 png | 85.63     | 88.42     | 83.08     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter8k   | v2_1170L_opus4p7_gc_tier4   | 1170 png | **89.17** | **91.73** | **86.82** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter8k | v2_1170L_opus4p7_gc_tier4   | 1170 png | 88.26     | 90.97     | 85.77     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter8k        | v2_1170L_opus4p7_gc_tier4   | 1170 png | 88.45     | 90.71     | 86.38     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter8k   | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 84.06     | 86.43     | 81.89     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter8k   | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 83.23     | 85.71     | 80.95     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter8k   | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | **88.73** | **90.94** | **86.71** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter8k | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 88.36     | 90.41     | 86.47     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter8k        | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 88.26     | 90.46     | 86.24     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter8k   | v2_1170L_opus4p7_gc         | 1170 png | 84.91     | 88.29     | 81.82     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter8k   | v2_1170L_opus4p7_gc         | 1170 png | 84.69     | 87.81     | 81.84     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter8k   | v2_1170L_opus4p7_gc         | 1170 png | **89.52** | **91.86** | **87.36** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter8k | v2_1170L_opus4p7_gc         | 1170 png | 89.04     | 92.02     | 86.31     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter8k        | v2_1170L_opus4p7_gc         | 1170 png | 89.36     | 91.66     | 87.25     | 1170/1170 |


## cosmos3plus_64bm32b_t2ionly_exp001 (iter 10k)

| Run                                                      | Benchmark                   | Images   | all       | orig      | phi       | success   |
|----------------------------------------------------------|-----------------------------|----------|-----------|-----------|-----------|-----------|
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter10k   | v2_1170L_opus               | 1170 png | 80.79     | 84.80     | 77.11     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter10k   | v2_1170L_opus               | 1170 png | 81.30     | 85.00     | 77.91     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter10k   | v2_1170L_opus               | 1170 png | **90.26** | **92.04** | **88.63** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter10k | v2_1170L_opus               | 1170 png | 89.98     | 91.79     | 88.32     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter10k        | v2_1170L_opus               | 1170 png | 89.92     | 91.58     | 88.39     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter10k   | v2_1170L_opus4p7_mx_tier0   | 1170 png | 75.80     | 80.74     | 71.26     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter10k   | v2_1170L_opus4p7_mx_tier0   | 1170 png | 76.09     | 81.10     | 71.50     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter10k   | v2_1170L_opus4p7_mx_tier0   | 1170 png | **88.21** | **90.31** | **86.29** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter10k | v2_1170L_opus4p7_mx_tier0   | 1170 png | 87.62     | 89.54     | 85.86     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter10k        | v2_1170L_opus4p7_mx_tier0   | 1170 png | 87.68     | 89.82     | 85.72     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter10k   | v2_1170L_opus4p7_mx_tier1   | 1170 png | 77.92     | 82.37     | 73.84     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter10k   | v2_1170L_opus4p7_mx_tier1   | 1170 png | 78.76     | 83.21     | 74.68     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter10k   | v2_1170L_opus4p7_mx_tier1   | 1170 png | **88.76** | **90.87** | **86.82** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter10k | v2_1170L_opus4p7_mx_tier1   | 1170 png | 88.50     | 90.51     | 86.66     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter10k        | v2_1170L_opus4p7_mx_tier1   | 1170 png | 87.95     | 89.80     | 86.26     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter10k   | v2_1170L_opus4p7_gc_tier4   | 1170 png | 79.79     | 83.67     | 76.22     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter10k   | v2_1170L_opus4p7_gc_tier4   | 1170 png | 80.21     | 84.16     | 76.60     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter10k   | v2_1170L_opus4p7_gc_tier4   | 1170 png | **88.98** | **91.51** | **86.66** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter10k | v2_1170L_opus4p7_gc_tier4   | 1170 png | 88.38     | 90.31     | 86.61     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter10k        | v2_1170L_opus4p7_gc_tier4   | 1170 png | 88.76     | 91.12     | 86.59     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter10k   | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 75.77     | 80.97     | 71.00     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter10k   | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 76.28     | 81.17     | 71.80     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter10k   | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | **88.12** | **90.38** | **86.05** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter10k | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 87.28     | 89.52     | 85.23     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter10k        | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 87.46     | 89.59     | 85.51     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter10k   | v2_1170L_opus4p7_gc         | 1170 png | 78.32     | 83.21     | 73.84     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter10k   | v2_1170L_opus4p7_gc         | 1170 png | 79.25     | 83.14     | 75.68     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter10k   | v2_1170L_opus4p7_gc         | 1170 png | **89.30** | **91.99** | 86.82     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter10k | v2_1170L_opus4p7_gc         | 1170 png | 88.99     | 91.68     | 86.52     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter10k        | v2_1170L_opus4p7_gc         | 1170 png | 88.93     | 91.05     | **86.99** | 1170/1170 |


## cosmos3plus_64bm32b_t2ionly_exp001 (iter 15k, gc-series only)

| Run                                                      | Benchmark                   | Images   | all       | orig      | phi       | success   |
|----------------------------------------------------------|-----------------------------|----------|-----------|-----------|-----------|-----------|
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter15k   | v2_1170L_opus               | 1170 png | 89.81     | 91.35     | 88.39     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter15k | v2_1170L_opus               | 1170 png | 89.52     | 91.05     | 88.11     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter15k        | v2_1170L_opus               | 1170 png | **89.89** | **91.43** | **88.49** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter15k   | v2_1170L_opus4p7_mx_tier0   | 1170 png | 87.26     | 90.05     | 84.69     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter15k | v2_1170L_opus4p7_mx_tier0   | 1170 png | **87.45** | 89.08     | **85.96** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter15k        | v2_1170L_opus4p7_mx_tier0   | 1170 png | 86.95     | **90.05** | 84.76     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter15k   | v2_1170L_opus4p7_mx_tier1   | 1170 png | 87.61     | 90.03     | 85.40     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter15k | v2_1170L_opus4p7_mx_tier1   | 1170 png | **87.81** | **90.10** | **85.70** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter15k        | v2_1170L_opus4p7_mx_tier1   | 1170 png | 87.62     | 89.85     | 85.58     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter15k   | v2_1170L_opus4p7_gc_tier4   | 1170 png | **88.51** | **91.25** | **86.01** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter15k | v2_1170L_opus4p7_gc_tier4   | 1170 png | 87.65     | 90.13     | 85.37     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter15k        | v2_1170L_opus4p7_gc_tier4   | 1170 png | 88.06     | 90.74     | 85.61     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter15k   | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | **87.14** | **89.95** | 84.55     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter15k | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 86.99     | 88.83     | **85.30** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter15k        | v2_1170L_opus4p7_gc_tier4p5 | 1170 png | 86.77     | 89.26     | 84.48     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter15k   | v2_1170L_opus4p7_gc         | 1170 png | **88.76** | **91.45** | **86.29** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter15k | v2_1170L_opus4p7_gc         | 1170 png | 88.14     | 90.97     | 85.54     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter15k        | v2_1170L_opus4p7_gc         | 1170 png | 88.40     | 91.05     | 85.98     | 1170/1170 |


---

## UGB All Score Summary (pivot)

Scored on `v2_1170L` with `gemini-3.1-pro@nvidia/k/ms`. Rows = model, columns = gen benchmark.

### iter 5k

![iter 5k](benchmark_score_by_exp_at_iter5k.png)

| Run           | opus      | mx_tier0   | mx_tier1   | gc_tier4   | gc_tier4p5   | gc        |
|---------------|-----------|------------|------------|------------|--------------|-----------|
| baseline      | **90.94** | 88.88      | 88.97      | 87.78      | 89.11        | 88.33     |
| 000_mxtier0   | 90.28     | **89.12**  | 88.98      | **89.16**  | 89.26        | 88.60     |
| 001_mxtier1   | 89.94     | 88.97      | **89.22**  | 89.14      | **89.27**    | 88.66     |
| 004_gctier4   | 90.54     | 88.47      | 89.03      | 88.70      | 88.67        | 89.42     |
| 005_gctier4p5 | 90.35     | 88.26      | 88.65      | 88.81      | 87.95        | **89.59** |
| 006_gc        | 90.27     | 88.21      | 88.95      | 88.77      | 88.48        | 89.41     |

### iter 8k

![iter 8k](benchmark_score_by_exp_at_iter8k.png)

| Run           | opus      | mx_tier0   | mx_tier1   | gc_tier4   | gc_tier4p5   | gc        |
|---------------|-----------|------------|------------|------------|--------------|-----------|
| 000_mxtier0   | 87.12     | 83.85      | 84.76      | 85.61      | 84.06        | 84.91     |
| 001_mxtier1   | 86.63     | 82.80      | 85.67      | 85.63      | 83.23        | 84.69     |
| 004_gctier4   | **90.60** | **88.54**  | 89.03      | **89.17**  | **88.73**    | **89.52** |
| 005_gctier4p5 | 89.94     | 88.10      | **89.14**  | 88.26      | 88.36        | 89.04     |
| 006_gc        | 90.00     | 88.16      | 88.77      | 88.45      | 88.26        | 89.36     |

### iter 10k

![iter 10k](benchmark_score_by_exp_at_iter10k.png)

| Run           | opus      | mx_tier0   | mx_tier1   | gc_tier4   | gc_tier4p5   | gc        |
|---------------|-----------|------------|------------|------------|--------------|-----------|
| 000_mxtier0   | 80.79     | 75.80      | 77.92      | 79.79      | 75.77        | 78.32     |
| 001_mxtier1   | 81.30     | 76.09      | 78.76      | 80.21      | 76.28        | 79.25     |
| 004_gctier4   | **90.26** | **88.21**  | **88.76**  | **88.98**  | **88.12**    | **89.30** |
| 005_gctier4p5 | 89.98     | 87.62      | 88.50      | 88.38      | 87.28        | 88.99     |
| 006_gc        | 89.92     | 87.68      | 87.95      | 88.76      | 87.46        | 88.93     |


### All Time High Mix

Per-experiment peak UGB all score across all iterations (5k/6k/8k/10k). Each cell shows the best score for that experiment on that benchmark.

| Run           | opus           | mx_tier0       | mx_tier1       | gc_tier4       | gc_tier4p5     | gc             |
|---------------|----------------|----------------|----------------|----------------|----------------|----------------|
| 000_mxtier0   | 90.28 (5k)     | **89.12 (5k)** | 88.98 (5k)     | 89.16 (5k)     | 89.26 (5k)     | 88.60 (5k)     |
| 001_mxtier1   | 90.17 (6k)     | 88.97 (5k)     | **89.22 (5k)** | 89.14 (5k)     | **89.27 (5k)** | 88.66 (5k)     |
| 004_gctier4   | **90.67 (6k)** | 88.61 (6k)     | 89.03 (5k)     | **89.17 (8k)** | 88.73 (8k)     | 89.52 (8k)     |
| 005_gctier4p5 | 90.39 (6k)     | 88.38 (6k)     | 89.14 (8k)     | 88.81 (5k)     | 88.36 (8k)     | 89.59 (5k)     |
| 006_gc        | 90.63 (6k)     | 88.36 (6k)     | 89.03 (6k)     | 88.94 (6k)     | 88.48 (5k)     | **89.61 (6k)** |

---

## UGB All Score — Iter Progression

Rows = experiment × benchmark. Columns = checkpoint iteration. Scored on `v2_1170L` with `gemini-3.1-pro`.

| Benchmark   | Experiment    | 3k        | 4k        | 5k        | 6k    | 8k        | 10k   | 15k   |
|-------------|---------------|-----------|-----------|-----------|-------|-----------|-------|-------|
| opus        | 000_mxtier0   | **90.8**  | 90.22     | 90.28     | 89.91 | 87.12     | 80.79 | —     |
| opus        | 001_mxtier1   | **90.88** | 90.11     | 89.94     | 90.17 | 86.63     | 81.3  | —     |
| opus        | 004_gctier4   | **91.18** | 90.69     | 90.54     | 90.67 | 90.6      | 90.26 | 89.81 |
| opus        | 005_gctier4p5 | **90.52** | 90.46     | 90.35     | 90.39 | 89.94     | 89.98 | 89.52 |
| opus        | 006_gc        | **90.94** | 90.91     | 90.27     | 90.63 | 90.0      | 89.92 | 89.89 |
| mx_tier0    | 000_mxtier0   | **89.82** | 89.61     | 89.12     | 88.6  | 83.85     | 75.8  | —     |
| mx_tier0    | 001_mxtier1   | 89.34     | **89.82** | 88.97     | 88.22 | 82.8      | 76.09 | —     |
| mx_tier0    | 004_gctier4   | 88.84     | **88.93** | 88.47     | 88.61 | 88.54     | 88.21 | 87.26 |
| mx_tier0    | 005_gctier4p5 | **88.59** | 88.48     | 88.26     | 88.38 | 88.1      | 87.62 | 87.45 |
| mx_tier0    | 006_gc        | 88.55     | **88.65** | 88.21     | 88.36 | 88.16     | 87.68 | 86.95 |
| mx_tier1    | 000_mxtier0   | **89.8**  | 89.22     | 88.98     | 88.69 | 84.76     | 77.92 | —     |
| mx_tier1    | 001_mxtier1   | **90.06** | 89.74     | 89.22     | 89.11 | 85.67     | 78.76 | —     |
| mx_tier1    | 004_gctier4   | **89.56** | 89.32     | 89.03     | 88.94 | 89.03     | 88.76 | 87.61 |
| mx_tier1    | 005_gctier4p5 | 89.12     | 88.95     | 88.65     | 88.98 | **89.14** | 88.5  | 87.81 |
| mx_tier1    | 006_gc        | **89.41** | 88.86     | 88.95     | 89.03 | 88.77     | 87.95 | 87.62 |
| gc_tier4    | 000_mxtier0   | 89.08     | 89.15     | **89.16** | 88.2  | 85.61     | 79.79 | —     |
| gc_tier4    | 001_mxtier1   | **89.37** | 89.36     | 89.14     | 88.16 | 85.63     | 80.21 | —     |
| gc_tier4    | 004_gctier4   | 89.32     | **89.66** | 88.7      | 89.14 | 89.17     | 88.98 | 88.51 |
| gc_tier4    | 005_gctier4p5 | **89.43** | 89.28     | 88.81     | 88.72 | 88.26     | 88.38 | 87.65 |
| gc_tier4    | 006_gc        | 88.89     | **89.05** | 88.77     | 88.94 | 88.45     | 88.76 | 88.06 |
| gc_tier4p5  | 000_mxtier0   | **89.89** | 89.48     | 89.26     | 88.86 | 84.06     | 75.77 | —     |
| gc_tier4p5  | 001_mxtier1   | **89.93** | 89.52     | 89.27     | 88.2  | 83.23     | 76.28 | —     |
| gc_tier4p5  | 004_gctier4   | **88.98** | 88.67     | 88.67     | 88.66 | 88.73     | 88.12 | 87.14 |
| gc_tier4p5  | 005_gctier4p5 | **88.92** | 88.53     | 87.95     | 88.32 | 88.36     | 87.28 | 86.99 |
| gc_tier4p5  | 006_gc        | 88.7      | **88.83** | 88.48     | 88.27 | 88.26     | 87.46 | 86.77 |
| gc          | 000_mxtier0   | **89.42** | 88.78     | 88.6      | 88.29 | 84.91     | 78.32 | —     |
| gc          | 001_mxtier1   | 88.71     | **88.89** | 88.66     | 88.03 | 84.69     | 79.25 | —     |
| gc          | 004_gctier4   | **90.32** | 89.66     | 89.42     | 89.3  | 89.52     | 89.3  | 88.76 |
| gc          | 005_gctier4p5 | 89.36     | **90.21** | 89.59     | 89.37 | 89.04     | 88.99 | 88.14 |
| gc          | 006_gc        | **90.67** | 90.02     | 89.41     | 89.61 | 89.36     | 88.93 | 88.4  |

### Baseline (iter 29k, cosmos3_ga_64bm32b_t2ionly_exp009_union5)

| Benchmark   | Experiment   | Score     |
|-------------|--------------|-----------|
| opus        | baseline     | **90.94** |
| mx_tier0    | baseline     | 88.88     |
| mx_tier1    | baseline     | 88.97     |
| gc_tier4    | baseline     | 87.78     |
| gc_tier4p5  | baseline     | 89.11     |
| gc          | baseline     | 88.33     |

## cosmos3plus_64bm32b_t2ionly_exp001 (iter 3k)

| Run                                                     | Benchmark                 | Images   | all       | orig      | phi       | success   |
|---------------------------------------------------------|---------------------------|----------|-----------|-----------|-----------|-----------|
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter3k   | v2_1170L_opus             | 1170 png | 90.80     | —         | —         | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter3k   | v2_1170L_opus             | 1170 png | **90.88** | —         | —         | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter3k   | v2_1170L_opus             | 1170 png | **91.18** | —         | —         | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter3k | v2_1170L_opus             | 1170 png | 90.52     | —         | —         | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter3k        | v2_1170L_opus             | 1170 png | 90.94     | —         | —         | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter3k   | v2_1170L_opus4p7_mx_tier0 | 1170 png | **89.82** | —         | —         | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter3k   | v2_1170L_opus4p7_mx_tier0 | 1170 png | 89.34     | —         | —         | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter3k   | v2_1170L_opus4p7_mx_tier0 | 1170 png | 88.84     | —         | —         | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter3k | v2_1170L_opus4p7_mx_tier0 | 1170 png | 88.59     | —         | —         | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter3k        | v2_1170L_opus4p7_mx_tier0 | 1170 png | 88.55     | —         | —         | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter3k   | v2_1170L_opus4p7_mx_tier1 | 1170 png | 89.80     | —         | —         | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter3k   | v2_1170L_opus4p7_mx_tier1 | 1170 png | **90.06** | —         | —         | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter3k   | v2_1170L_opus4p7_mx_tier1 | 1170 png | 89.56     | —         | —         | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter3k | v2_1170L_opus4p7_mx_tier1 | 1170 png | 89.12     | —         | —         | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter3k        | v2_1170L_opus4p7_mx_tier1 | 1170 png | 89.41     | —         | —         | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter3k   | v2_1170L_opus4p7_gc_tier4 | 1170 png | 89.08     | 91.66     | 86.71     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter3k   | v2_1170L_opus4p7_gc_tier4 | 1170 png | 89.37     | 91.45     | 87.46     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter3k   | v2_1170L_opus4p7_gc_tier4 | 1170 png | 89.32     | 91.71     | 87.13     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter3k | v2_1170L_opus4p7_gc_tier4 | 1170 png | **89.43** | **91.53** | **87.50** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter3k        | v2_1170L_opus4p7_gc_tier4 | 1170 png | 88.89     | 91.20     | 86.78     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter3k   | v2_1170L_opus4p7_gc_tier4 | 1170 png | 89.08     | 91.66     | 86.71     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter3k   | v2_1170L_opus4p7_gc_tier4 | 1170 png | 89.37     | 91.45     | 87.46     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter3k   | v2_1170L_opus4p7_gc_tier4 | 1170 png | 89.32     | 91.71     | 87.13     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter3k | v2_1170L_opus4p7_gc_tier4 | 1170 png | **89.43** | **91.53** | **87.50** | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter3k        | v2_1170L_opus4p7_gc_tier4 | 1170 png | 88.89     | 91.20     | 86.78     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_000_mxtier0_iter3k   | v2_1170L_opus4p7_gc       | 1170 png | 89.42     | 92.24     | 86.82     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_001_mxtier1_iter3k   | v2_1170L_opus4p7_gc       | 1170 png | 88.71     | 91.86     | 85.82     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_004_gctier4_iter3k   | v2_1170L_opus4p7_gc       | 1170 png | 90.32     | **92.88** | 87.97     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_005_gctier4p5_iter3k | v2_1170L_opus4p7_gc       | 1170 png | 89.36     | 91.91     | 87.01     | 1170/1170 |
| cosmos3plus_64bm32b_t2ionly_exp001_006_gc_iter3k        | v2_1170L_opus4p7_gc       | 1170 png | **90.67** | 92.45     | **88.13** | 1170/1170 |
