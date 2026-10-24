# UGB 分数报告 — 数据消融(exp003)

英文版:`ugb_en.md`。

实验:`cosmos3p5_ga_60bm30b_t2ionly_moe_exp003_*`,都从
`s3://nv-00-10206-checkpoint/cosmos3_generator/3p5_production_runs/60bm30b_v2p5_pretrain/checkpoints/iter_000741260/` 开始训练 25k 步。

| 实验 | 数据混合(比例) | lr |
|---|---|---|
| exp003_000_pretrain | pretrain_real 8 : pretrain_sdg 1 : pretrain_text 1 | 1e-4 |
| exp003_001_pretrain_midtrain | pretrain_real 8 : sdg 1 : text 1 : midtrain 2.5(midtrain 占 20%) | 1e-4 |
| exp003_002_pretrain_getty | pretrain_real 7.2 : sdg 0.9 : text 0.9 : getty 1.0(getty 占 10%,`getty_image_350k_test.lance`) | 2e-5 |
| exp003_003_pretrain_midtrain_lr2em5 | 同 exp003_001(midtrain 占 20%) | 2e-5 |

所有行的打分方式:`compute_unigenbench_metric.py --benchmark_name v2_1170L`(UGB 原版、未 upsample 的 prompt 和测试点),judge 为 `gemini-3.1-pro`。

## Cosmos3.5 — 用 v2_1170L_gpt5p5_cosmos3p5_dense_hamid 生成

生成:`inference_ugb_i4.py`,`--experiment_name cosmos3p5_ga_60bm30b_base_768`,1024×1024,guidance 4.0,50 步,`--use_ema`。`neg` = 加 `--use_cosmos3_negative_prompt`,`noneg` = 不加。

| Run                                                              | 图片     | all       | orig      | phi       | success      |
|------------------------------------------------------------------|----------|-----------|-----------|-----------|--------------|
| c3p5_super_v2p5_iter741260_noneg(SFT 前,960×960,`base`)       | 1170 png | 86.10     | 88.01     | 84.34     | 1170/1170    |
| ---------------------------------------------------------------- | -------- | --------- | --------- | --------- | ------------ |
| c3p5_exp003_000_pretrain_iter25000(neg)                         | 1170 png | 85.80     | 87.81     | 83.97     | 1170/1170    |
| c3p5_exp003_001_pretrain_midtrain_iter25000(neg)                | 1170 png | 86.05     | 87.96     | 84.30     | 1170/1170    |
| ---------------------------------------------------------------- | -------- | --------- | --------- | --------- | ------------ |
| c3p5_exp003_000_pretrain_iter25000_noneg                         | 1170 png | 86.29     | 88.83     | 83.97     | 1170/1170    |
| c3p5_exp003_001_pretrain_midtrain_iter25000_noneg                | 1170 png | 86.44     | 88.88     | 84.20     | 1170/1170    |
| c3p5_exp003_002_pretrain_getty_iter25000_noneg                   | 1170 png | **87.56** | **89.80** | **85.51** | 1170/1170    |
| c3p5_exp003_003_pretrain_midtrain_lr2em5_iter25000_1024_shift3_fullcfg_seedrow_noneg ¹ | 1170 png | 87.88 | 90.13 | 85.82 | 1170/1170 |

¹ exp003_003 用的是 shift 3、**全程 CFG**、行号 seed（`Cosmos3p5Generator` 的新默认，`cd2022e3408`），不是本表其他行的 CFG [400, 1000]，不能和本表直接比。只和 lr 相同（2e-5）的 exp003_002 比；exp003_000 / 001 是 lr 1e-4，不做对比。同设置（1024、shift 3、全程 CFG、行号 seed、noneg）下：exp003_002 getty 87.92，exp003_003 midtrain **87.88**，相差 −0.04，在 judge 约 0.35 的噪声内，看不出差别。

## 采样设置逐步改变:exp003_002 getty(1024×1024)

所有 SFT checkpoint 一律用 1024×1024、`base_768` 测(只有 SFT 前 checkpoint 用 960×960、`base`)。同一个 checkpoint `exp003_002_pretrain_getty/iter_000025000`,noneg,guidance 4,50 步,EMA;相邻两行只差一个变量。

| 设置 | 生成脚本 | all | orig | phi | success |
|---|---|---|---|---|---|
| shift 3,CFG [400, 1000],行号 seed(原设置) | `inference_ugb_i4.py` | 87.56 | 89.80 | 85.51 | 1170/1170 |
| shift 3,全程 CFG,行号 seed | `inference_ugb_i4_20261020_shift3_fullcfg.py` | 87.92 | 90.18 | 85.84 | 1170/1170 |
| shift 6,CFG [400, 1000],行号 seed | `inference_ugb_i4_20261020.py` | 87.23 | 89.57 | 85.09 | 1170/1170 |
| shift 6,全程 CFG,行号 seed | `inference_ugb_i4_20261020_fullcfg.py` | 87.62 | 89.69 | 85.72 | 1170/1170 |
| shift 6,全程 CFG,seed 恒为 0 | `inference_ugb_i4_20261020_fullcfg_seed0.py` | **88.14** | **90.26** | **86.19** | 1170/1170 |

逐步变化:shift 3 → 6 为 −0.33,CFG [400, 1000] → 全程为 +0.39,行号 seed → seed 0 为 +0.52;原设置到完整 recipe 总共 +0.58。前两步都在 judge 重跑的约 0.35 噪声范围内,shift 6 本身在 1024 下没有带来提升。seed 0 这一步最大,而且在 960 下也是同方向(87.67 → 87.88),但 seed 只决定采样起点,不改变模型能力,不应作为模型改进来解读。

输出目录:`.../v2_1170L_gpt5p5_cosmos3p5_dense_hamid/c3p5_exp003_002_pretrain_getty_iter25000_1024_shift6_{cfg400to1000_seedrow,fullcfg_seedrow,fullcfg_seed0}_noneg`。judge 重跑一次的噪声约 0.35 个点。

同一组设置在 exp003_000 pretrain(`exp003_000_pretrain/iter_000025000`)上验证,1024×1024、`base_768`、noneg:

| 设置 | 生成脚本 | all | orig | phi | success |
|---|---|---|---|---|---|
| shift 3,CFG [400, 1000],行号 seed(原设置) | `inference_ugb_i4.py` | 86.29 | 88.83 | 83.97 | 1170/1170 |
| shift 3,全程 CFG,行号 seed | `inference_ugb_i4_20261020_shift3_fullcfg.py` | 86.37 | 88.72 | 84.20 | 1170/1170 |
| shift 6,CFG [400, 1000],行号 seed | `inference_ugb_i4_20261020.py` | 86.00 | 88.88 | 83.36 | 1170/1170 |
| shift 6,全程 CFG,行号 seed | `inference_ugb_i4_20261020_fullcfg.py` | 86.61 | 89.18 | 84.25 | 1170/1170 |
| shift 6,全程 CFG,seed 恒为 0 | `inference_ugb_i4_20261020_fullcfg_seed0.py` | 86.87 | 88.83 | 85.07 | 1170/1170 |

**三个 checkpoint 的逐步变化(all):**

| 步骤 | SFT 前(960) | exp003_000 pretrain(1024) | exp003_002 getty(1024) |
|---|---|---|---|
| shift 3 → 6 | +0.11 | −0.29 | −0.33 |
| CFG [400, 1000] → 全程 | +0.82 | +0.61 | +0.39 |
| 行号 seed → seed 0 | +0.18 | +0.26 | +0.52 |
| 原设置 → 完整 recipe | +1.11 | +0.58 | +0.58 |

**getty 相对 pretrain 的提升在每种采样设置下(1024):**

| 设置 | pretrain | getty | Δ |
|---|---|---|---|
| shift 3,CFG [400, 1000],行号 seed | 86.29 | 87.56 | +1.27 |
| shift 6,CFG [400, 1000],行号 seed | 86.00 | 87.23 | +1.23 |
| shift 6,全程 CFG,行号 seed | 86.61 | 87.62 | +1.01 |
| shift 6,全程 CFG,seed 恒为 0 | 86.87 | 88.14 | +1.27 |

**shift × CFG 区间 2×2(行号 seed,all):**

| checkpoint | shift 3 + [400, 1000] | shift 3 + 全程 | shift 6 + [400, 1000] | shift 6 + 全程 |
|---|---|---|---|---|
| SFT 前(960) | 86.10 | 86.72 | 86.21 | 87.03 |
| exp003_000 pretrain(1024) | 86.29 | 86.37 | 86.00 | 86.61 |
| exp003_002 getty(1024) | 87.56 | 87.92 | 87.23 | 87.62 |
| getty − pretrain | +1.27 | +1.55 | +1.23 | +1.01 |

全程 CFG 相对 [400, 1000] 在六组对比中都为正(shift 3 下 +0.62、+0.08、+0.36;shift 6 下 +0.82、+0.61、+0.39,平均约 +0.48);shift 3 → 6 有正有负(平均约 0)。getty 相对 pretrain 在五种采样设置下都在 +1.0 到 +1.55 之间。

结论:
- shift 6 本身在三个 checkpoint 上都没有作用(−0.33 到 +0.11,都在约 0.35 的 judge 噪声内)。
- 全程 CFG 是采样设置里唯一稳定的提升,三个 checkpoint 都是正的(+0.39 到 +0.82)。
- seed 0 三次都是正的(+0.18 到 +0.52),但单次都不超过噪声太多,不应作为模型改进解读。
- getty 相对 pretrain 的提升在四种采样设置下都在 +1.0 到 +1.3 之间,不依赖采样设置,结论稳健。

同一组设置在 SFT 前 checkpoint(exp003 的共同起点 `iter_000741260`)上验证,960×960、`base`、noneg:

| 设置 | 生成脚本 | all | orig | phi | success |
|---|---|---|---|---|---|
| shift 3,CFG [400, 1000],行号 seed(原设置) | `inference_ugb_i4.py` | 86.10 | 88.01 | 84.34 | 1170/1170 |
| shift 3,全程 CFG,行号 seed | `inference_ugb_i4_20261020_shift3_fullcfg.py` | 86.72 | 88.24 | 85.33 | 1170/1170 |
| shift 6,CFG [400, 1000],行号 seed | `inference_ugb_i4_20261020.py` | 86.21 | 87.88 | 84.67 | 1170/1170 |
| shift 6,全程 CFG,行号 seed | `inference_ugb_i4_20261020_fullcfg.py` | 87.03 | 88.83 | 85.37 | 1170/1170 |
| shift 6,全程 CFG,seed 恒为 0 | `inference_ugb_i4_20261020.py`(当时版本) | 87.21 | 89.18 | 85.40 | 1170/1170 |

下面「Base checkpoint」一节里 getty 的两行 recipe 结果(87.88、87.67)是 960×960、`base` 测的,不符合上面的分辨率约定,以本节 1024 的结果为准。

## Base checkpoint — guidance 和 judge

Checkpoint `60bm30b_v2p5_pretrain/checkpoints/iter_000741260`(SFT 前)。生成:`--experiment_name cosmos3p5_ga_60bm30b_base`,960×960,50 步,`--use_ema`,noneg,用 v2_1170L_gpt5p5_cosmos3p5_dense_hamid 的 prompt。所有格子的打分 prompt 都是 `v2_1170L`。GPT-5.5 的结果存为 `unigenbench_result_judge_gpt5p5.json`(`--signature judge_gpt5p5`),和 gemini 的 `unigenbench_result.json` 放在同一目录。

| Run                                         | guidance | judge                 | all       | orig      | phi       | success      |
|---------------------------------------------|----------|-----------------------|-----------|-----------|-----------|--------------|
| c3p5_super_v2p5_iter741260_noneg            | 4.0      | gemini-3.1-pro        | 86.10     | 88.01     | 84.34     | 1170/1170    |
| c3p5_super_v2p5_iter741260_noneg            | 4.0      | gpt-5.5               | 88.69     | 89.46     | 87.97     | 1170/1170    |
| c3p5_super_v2p5_iter741260_cfg6_noneg       | 6.0      | gemini-3.1-pro@nvidia | 86.40     | 87.63     | 85.28     | 1170/1170    |
| c3p5_super_v2p5_iter741260_cfg6_noneg       | 6.0      | gpt-5.5               | 89.04     | 89.57     | 88.56     | 1170/1170    |
| c3p5_super_v2p5_iter741260_recipe_shift6_fullcfg_seed0(noneg) | 4.0 | gemini-3.1-pro@nvidia | **87.21** | **89.18** | 85.40 | 1170/1170 |
| c3p5_super_v2p5_iter741260_ruofan_g4_shift6_negdefault       | 4.0      | gemini-3.1-pro@nvidia | 86.79     | 88.83     | 84.93     | 1170/1170    |
| c3p5_super_v2p5_iter741260_ruofan_g6_shift6_negdefault       | 6.0      | gemini-3.1-pro@nvidia | 87.22     | 89.16     | 85.44     | 1170/1170    |

ruofan 两行是同事 ruofanl 用 cosmos-benchmarks 系统生成的图,从 gcpcode lustre 拷到我们的 S3 目录后用我们的打分脚本重新打分:
- g4:`/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/ruofanl/local_eval/ugb_img_a4_xckpt_s0_720p_20261014/results/prod_g4_gpt55_a2/ruofanl_60bm30b_v2p5_pretrain_iter_000741260_720p_g4_nocache_gpt55_a2`
- g6:`/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/ruofanl/local_eval/getty_base_741260_ugb_20260928/results/base/ruofanl_60bm30b_v2p5_pretrain_iter_000741260_720p_g6_nocache`

他的生成设置(来自各目录的 `config.json`):同一个 checkpoint `iter_000741260`,生产实验配置(`cosmos3p5_ga_60bm30b_v2_pretrain` 模块,`configs/base/config.py`),960×960,shift 6,`negative_prompt_version: default`(**带** negative prompt),同一份 gpt55_a2 prompt。他自己的打分(`unigenbench_result_ruofan.json`,同样是 gemini-3.1-pro + `v2_1170L_G3F.csv`):g4 86.90 / 89.03 / 84.95,g6 87.23 / 89.16 / 85.47。

我们重打的分和他自己打的分只差 0.0–0.1 个点,所以**打分流程两边一致,差距来自生成**。同样 cfg 4、同样 960×960,我们的图 86.10,他的图 86.79,差 0.69 个点。已知的生成差异:shift(我们写死 3.0,他 6)、CFG 区间(我们写死 [400, 1000],他全程)、seed、negative prompt(我们不加,他用 default)、实验配置(我们 text_rendering 的 `base`,他用生产配置)。recipe 那一行(作业 2576804)对齐了前三项,用来看能补上多少。

**结果:** 只改 shift(3.0 → 6)、CFG 区间([400, 1000] → 全程)和 seed,我们自己的 pipeline 在 cfg 4 下就从 86.10 涨到 87.21,超过了 ruofan 的 g4(86.79),和他的 g6(87.22)持平,而且我们没加 negative prompt。所以约 0.7 个点的差距基本都来自 `inference_helper.py` 里 `Cosmos3Generator.generate_image` 写死的 `shift=3.0` 和 `guidance_interval=[400, 1000]`,negative prompt 和实验配置不是主因。另外,在旧的采样设置下把 cfg 从 4 提到 6 只涨了 0.30(86.10 → 86.40),远不如改采样设置。

**getty 在 recipe 采样下:** 用同样的 `inference_ugb_i4_20261020.py`(shift 6、全程 CFG、seed 0)、`cosmos3p5_ga_60bm30b_base`、960×960、cfg 4、noneg,生成 exp003_002 的 `iter_000025000`:

| Run | all | orig | phi | success |
|---|---|---|---|---|
| c3p5_super_v2p5_iter741260_recipe_shift6_fullcfg_seed0(SFT 前) | 87.21 | 89.18 | 85.40 | 1170/1170 |
| c3p5_exp003_002_pretrain_getty_iter25000_recipe_shift6_fullcfg_seed0_noneg | **87.88** | **90.13** | **85.82** | 1170/1170 |
| c3p5_exp003_002_pretrain_getty_iter25000_recipe_shift6_fullcfg_seedrow_noneg(seed 改回 prompt 行号) | 87.67 | 89.92 | 85.61 | 1170/1170 |

seed 从 0 改回 prompt 行号后,getty 从 87.88 变成 87.67(−0.21)。同一批行号 seed 的图第一次打分是 87.32(1169/1170,有 1 张 judge 调用重试后仍失败,被排除在统计外),`--force_rescore` 重打一遍后是 87.67:**同一批图只因 judge 重跑就差了 0.35 个点**,比换 seed 带来的变化还大。所以 0.3–0.5 个点以内的差异都要当作噪声看。

同一套采样设置下,getty 实验比 SFT 前高 0.67 个点(orig +0.95,phi +0.42)。旧采样设置下(1024、`base_768`、shift 3)它比 SFT 前高 1.46 个点(87.56 vs 86.10),但那两行的分辨率和配置也不同,不是严格对比;这一组才是单变量对比,提升缩小但仍在。

**影响:** 之前所有用 `inference_ugb_i4.py` 生成的 Cosmos3.5 分数(包括 exp003 各行)都是在 shift 3、CFG 区间 [400, 1000] 下测的,彼此之间仍可比,但会比 cosmos-benchmarks 系统的分数系统性偏低约 0.7–1 个点。

最后一行用 `inference_ugb_i4_20261020.py` 生成,采样设置对齐 cosmos-benchmarks `cosmos3_eval` 的 unigenbench-t2i recipe:shift 6(原脚本写死 3.0)、全程 CFG(原脚本写死 `guidance_interval=[400, 1000]`)、每个 prompt 的 seed 都是 0(原脚本用 prompt 行号);不加 negative prompt。其余设置与 `c3p5_super_v2p5_iter741260_noneg` 相同,用来定位和同事结果相差约 0.8 个点的原因。

### Judge 对比:同一批图上 gemini-3.1-pro vs gpt-5.5

图:`c3p5_super_v2p5_iter741260_noneg`(SFT 前,960×960,guidance 4.0)。两个 judge 都打完了全部 1170 张图、同样的 8193 个测试点(orig 3920,phi 4273),每张图的判定条数都和测试点条数相等。gpt-5.5 有 9 张第一次返回的不是合法 YAML,脚本对这 9 张重跑一遍后都拿到了完整判定。判对的测试点:gemini 7054 个(86.10%),gpt-5.5 7266 个(88.69%)。

| 判定组合          | 测试点数 | 占比   |
|-------------------|----------|--------|
| 两个都判对        | 6779     | 82.7%  |
| 两个都判错        | 652      | 8.0%   |
| 只有 gpt-5.5 判对 | 487      | 5.9%   |
| 只有 gemini 判对  | 275      | 3.4%   |

两个 judge 在 90.7% 的测试点上意见一致。分歧里 gpt-5.5 比 gemini 多放过 212 个测试点,正好就是总分高出的约 2.6 个点。

分歧集中在动作和空间关系上:

| 主维度            | 只有 gpt-5.5 判对 | 只有 gemini 判对 | 测试点数 |
|-------------------|-------------------|------------------|----------|
| Action            | 191               | 117              | 1525     |
| Relationship      | 84                | 23               | 1011     |
| Entity Layout     | 64                | 14               | 1110     |
| Attribute         | 61                | 53               | 2595     |
| Compound          | 28                | 11               | 414      |
| Logical Reasoning | 15                | 20               | 155      |
| World Knowledge   | 14                | 21               | 327      |
| Text Generation   | 13                | 6                | 208      |
| Style             | 10                | 0                | 620      |
| Grammar           | 7                 | 10               | 228      |

phi 子集(物理 AI 场景)里这类测试点最多,所以 phi 涨得最多(+3.6),orig 只涨 +1.5。Style、Text Generation 等维度两边差不多。

抽样看了 8 条分歧(人工阅读,不是系统性标注):
- **gpt-5.5 判对、gemini 判错:** 多数是 gpt-5.5 看得不够细。`phi138` 要求从割草机后方拍,图是从前方拍的,gemini 指出了,gpt-5.5 却说"从稍后方";`phi51` 要求瓶子刚好没入水面以下,实际上半截露在水面外,gpt-5.5 判对。也有 gemini 过严的,比如 `phi554` 按"包含关系的严格定义"判错。
- **gemini 判对、gpt-5.5 判错:** 多是 gpt-5.5 抠字眼。`phi552` 要求"全身动作",因为厨师只拍到腰部以上被判错;`phi122` 因为"没有运动模糊、看不出机器人在移动"被判错。

结论:gpt-5.5 在空间和动作细节上更宽松,gemini 抓错更准。主要分数继续用 gemini-3.1-pro,gpt-5.5 的分数只作参考,不和 gemini 的分数放在一起比。

## 参考 — Qwen-Image-2.1,用 v2_1170L 原版 prompt 生成

生成:临时写的客户端,调用自建的 vLLM-Omni 服务(`Qwen/Qwen-Image-2.1`),2048×2048,40 步,`true_cfg_scale=1.0`,seed 为 prompt 行号。`pe` = 先用 `Qwen/Qwen-Image-2.1-PE-T2I` 改写 prompt 再生成(宽高比固定为 1:1)。

| Run                                                              | 图片     | all       | orig      | phi       | success      |
|------------------------------------------------------------------|----------|-----------|-----------|-----------|--------------|
| qwen_image_2p1_nope                                              | 1170 png | **86.99** | **90.71** | **83.57** | 1170/1170    |
| qwen_image_2p1_pe                                                | 1170 png | 85.80     | 88.34     | 83.48     | 1170/1170    |

## 主维度分数(all 子集)

| 指标              | SFT 前 noneg | 000 neg | 001 neg | 000 noneg | 001 noneg | 002 noneg | Qwen-2.1 无 PE | Qwen-2.1 有 PE |
|-------------------|--------------|---------|---------|-----------|-----------|-----------|----------------|----------------|
| Action            | 73.11        | 72.39   | 71.93   | 73.18     | 72.39     | 74.43     | 73.25          | 73.38          |
| Attribute         | 95.07        | 93.87   | 94.61   | 94.07     | 94.80     | 95.72     | 94.72          | 92.06          |
| Compound          | 84.30        | 84.30   | 83.09   | 84.54     | 85.27     | 89.37     | 82.85          | 81.40          |
| Entity Layout     | 92.25        | 91.98   | 92.70   | 92.61     | 91.80     | 92.07     | 89.01          | 86.04          |
| Grammar           | 88.60        | 90.35   | 88.60   | 89.47     | 90.35     | 88.60     | 85.96          | 82.89          |
| Logical Reasoning | 70.32        | 73.55   | 71.61   | 76.77     | 74.84     | 80.00     | 76.13          | 70.32          |
| Relationship      | 85.16        | 87.04   | 87.04   | 86.94     | 87.44     | 88.72     | 85.95          | 82.89          |
| Style             | 98.06        | 96.45   | 96.61   | 98.71     | 97.74     | 98.39     | 95.97          | 94.84          |
| Text Generation   | 12.50        | 13.46   | 16.83   | 12.50     | 15.87     | 15.38     | 63.46          | 93.75          |
| World Knowledge   | 89.60        | 89.91   | 91.13   | 90.21     | 92.05     | 91.13     | 95.11          | 95.11          |

用 `python reports/score_report_20261020_dataset_ablation/gather_ugb_scores.py` 重新生成这张表(读取各目录的 `unigenbench_result.json`)。

## 结论

- exp003_002(getty 10%,lr 2e-5)是目前最好的 Cosmos3.5 结果:all 87.56,比 exp003_001 noneg 高 1.12,比 exp003_000 noneg 高 1.27,是这组消融里最大的差距。涨分分布在 Compound(比 001 noneg +4.1)、Logical Reasoning(+5.2)、Action、Attribute 和 Relationship 上;Text Generation 没有变化(15.38)。但 002 同时改了数据和 lr(1e-4 → 2e-5),所以还不能把提升归功于 getty;缺一个"lr 2e-5、不加 getty"的对照实验。
- exp003_002 打分用的是 `--judge_model gemini-3.1-pro@nvidia`(现在不带后缀的名字会报 `KeyError`);其他行是更早用不带后缀的 `gemini-3.1-pro` 打的。模型相同,网关可能不同。
- exp003_000 和 exp003_001 彼此之间、以及和 SFT 前 checkpoint 之间的差距都在约 0.7 个点以内。每个 prompt 只生成一张图,这个差距在噪声范围内;多训 25k 步和加 midtrain 都没有看到明确收益。
- 25k 步里训练 loss 基本不降(exp003_000 约 1.66–1.72,exp003_001 约 1.69–1.72):pretrain_real/sdg/text 都是 `image_meta_table_full` 的 fragment,也就是 v2p5 预训练用过的同一张表,所以这些实验见到的新信号很少。
- 去掉 negative prompt 后,两个实验都高了约 0.4 个点。
- midtrain 让 Text Generation 往预期方向走(neg 和 noneg 下都比 exp003_000 高 3.4 个点),但这个维度只有 208 个测试点。
- Qwen-Image-2.1 不能逐行直接比:它用的是原版短 prompt、2048 分辨率,Cosmos3.5 用的是 GPT-5.5 upsample 后的 prompt、1024 分辨率。它的领先集中在 Text Generation(63.46 vs 最多 16.83)。它自己的 PE 把 Text Generation 提到 93.75,但 Entity Layout、Grammar、Relationship、Logical Reasoning 掉了 3–6 个点,总分反而降到 85.80。
- SFT 前那一行是 960×960、`base`(shift 3);exp003 各行是 1024×1024、`base_768`(shift 5)。
