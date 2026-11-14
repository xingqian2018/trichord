# Downstream 测评细节

本文档记录我们现在怎么测 downstream 任务：每个任务用什么参数（温度、top-p、top-k、截断上限、每题份数、停止符），以及和官方或公开做法的对照。

- 只做记录，不改变任何测评设置。决定（2026-11-13）：保持现在的测法和参数不变，已有的 baseline 不重测。
- 任务清单是 `cosmos_ultra_ladder`：36 个任务，66 个分数。所有模型用同一份清单。
- 参数取自测评存下来的记录，也就是实际发给模型的值，以 Qwen3.8-Max 那次测评为例。
- 更新日期：2026-11-13。

## 1. 所有任务共用的设置

| 项目 | 我们的做法 |
|---|---|
| top-k | 所有任务都不用，请求里不带这个参数 |
| top-p | 只有三个任务带：HumanEval 0.95、LiveCodeBench 0.95、MBPP 1.0（等于不截）。其余任务不带，用完整分布 |
| 模型自带的采样默认值 | 不采用。参考模型的服务以"只用请求里给的采样参数"的方式启动；我们自己的 checkpoint 在测评前检查过不带参数时用的是完整分布 |
| 服务端长度上限 | 16,384 token（提示加生成） |
| 示例题的随机种子 | 1234，记在每个分数文件里 |
| 采样任务的种子 | 第 r 份回答用种子 r，所以采样结果可以复现 |

两类任务对"截断"的处理不同：

- **选择题**（13 个任务）：模型不写字。提示加选项的长度上限是 16,383 token。实测 MMLU 最长的提示是 3,097 token，没有任何题被截。旧的测评流程用的是默认上限 2,048，那时 MMLU 有 373 道题的开头被截掉。
- **生成类**（23 个任务）：提示原文整段发出，不截。模型写的内容有上限（见 2.1 的"生成上限"），写到上限就被切断；另外在停止符处也会切断。

## 2. 每个任务的参数

### 2.1 生成类任务（23 个）

"温度 0"表示不采样，每题只写最有把握的一份。

| 任务 | 题数 | 示例题 | 每题份数 | 温度 | top-p | 生成上限 | 停止符 |
|---|---:|---:|---:|---:|---|---:|---|
| mbpp_sanitized_3shot_sampled | 257 | 3 | 32 | 0.5 | 1.0 | 768 | `[DONE]`、`[DONE`、`\nYou are an expert Python programmer`、`</s>` |
| humaneval_sampled | 164 | 0 | 32 | 0.2 | 0.95 | 512 | 12 个，见表后 |
| lcb_codegen | 1,055 | 1 | 1 | 0 | 0.95 | 1,024 | `###` |
| lcb_codegen_sampled | 1,055 | 1 | 32 | 0.2 | 0.95 | 1,024 | `###` |
| mmlu_pro_5_shot_base | 12,032 | 5 | 1 | 0 | 不带 | 2,048 | `Question:` |
| agieval_en_cot | 2,546 | 2 到 5 | 1 | 0 | 不带 | 2,000 | `<END>` |
| gpqa_diamond_cot_n_shot | 196 | 5 | 1 | 0 | 不带 | 2,048 | `</s>`、`Question:` |
| gpqa_diamond_cot_n_shot_sampled | 196 | 5 | 32 | 0.7 | 不带 | 2,048 | `</s>`、`Question:`、模型结束标记 |
| supergpqa_main_cot_n_shot | 26,529 | 5 | 1 | 0 | 不带 | 2,048 | `</s>`、`Question:` |
| mgsm_native_cot_zh | 250 | 8 | 1 | 0 | 不带 | 4,096 | `问题:`、`问题：`、`</s>`、连续三个换行、模型结束标记 |
| math_500_sampled | 500 | 4 | 32 | 0.7 | 不带 | 4,096 | `Problem:`、`Question:`、`Q:`、`</s>`、模型结束标记 |
| hmmt_sampled | 30 | 4 | 32 | 0.7 | 不带 | 4,096 | `Problem:`、`Question:`、`</s>`、模型结束标记 |
| aime24 | 30 | 4 | 32 | 0.7 | 不带 | 4,096 | `Problem:`、`Question:`、`Q:`、`</s>`、模型结束标记 |
| aime25 | 30 | 4 | 32 | 0.7 | 不带 | 4,096 | 同 aime24 |
| minerva_math_nemo | 5,000 | 4 | 1 | 0 | 不带 | 4,096 | 同 aime24 |
| gsm8k_cot | 1,319 | 8 | 1 | 0 | 不带 | 1,024 | `Q:`、`</s>`、模型结束标记 |
| gsm8k_cot_nemo | 1,319 | 8 | 1 | 0 | 不带 | 1,024 | `Problem:`、`Question:`、`Q:`、`</s>`、模型结束标记 |
| trivia_qa | 17,944 | 5 | 1 | 0 | 不带 | 256 | 换行（答案只取第一行） |
| popqa_15shot | 14,267 | 15 | 1 | 0 | 不带 | 15 | 换行 |
| naturalqs_5shot | 3,610 | 5 | 1 | 0 | 不带 | 256 | 换行 |
| simpleqa_5shot | 4,326 | 5 | 1 | 0 | 不带 | 64 | 换行、`</s>`、模型结束标记 |
| simpleqa_verified_25shot | 1,000 | 25 | 1 | 0 | 不带 | 64 | 换行、`</s>`、模型结束标记 |
| facts_parametric_25shot | 1,052 | 25 | 1 | 0 | 不带 | 64 | 换行、`</s>`、模型结束标记 |

说明：

- HumanEval 的 12 个停止符：`</s>`、代码围栏、`\ndef `、`\nimport `、`\nfrom `、`\nclass `、`\nprint(`、`\nassert `、`\nif __name__`、`\n@`，以及两种三引号开头的行。
- "模型结束标记"指各家模型自己的结束符号（im_end、endoftext、eot_id 等）。
- LiveCodeBench 的"示例题 1"是提示里固定带的一道示例题，见第 3 节。
- 五个数学任务（math_500、hmmt、aime24、aime25、minerva）的值取自数学任务的重跑（`MATHFIX_*`）。

### 2.2 选择题任务（13 个）

不生成文字，所以没有温度、top-p、生成上限和停止符。做法是把每个选项分别接在提示后面，比较模型认为哪个最可能。

| 任务 | 题数 | 子科目 | 示例题 | 每题选项数 |
|---|---:|---:|---:|---|
| mmlu | 14,042 | 57 | 5 | 4 |
| mmlu_redux | 5,330 | 57 | 5 | 4 |
| global_mmlu_zh | 400 | 6 | 5 | 4 |
| include_base_44_chinese | 545 | 8 | 5 | 4 |
| commonsense_qa | 1,221 | 1 | 7 | 5 |
| arc_challenge_llama | 1,172 | 1 | 25 | 4（少数题 3 或 5） |
| hellaswag | 10,042 | 1 | 10 | 4 |
| openbookqa | 500 | 1 | 0 | 4 |
| piqa | 1,838 | 1 | 0 | 2 |
| social_iqa | 1,954 | 1 | 0 | 3 |
| truthfulqa_mc2 | 817 | 1 | 0 | 2 到 20 |
| winogrande | 1,267 | 1 | 5 | 2 |
| race_high_all | 3,498 | 1 | 0 | 4 |

## 3. LiveCodeBench 的具体测法

这里的 LCB 指 LiveCodeBench（竞赛编程题）。另有一个 LongCodeBench（把整个代码仓库放进上下文，最长 100 万 token），缩写也是 LCB，我们没有测过。

### 3.1 题集

| 项目 | 内容 |
|---|---|
| 数据集 | `sam-paech/livecodebench-code_generation_lite`，是官方 `livecodebench/code_generation_lite` 的镜像 |
| 版本 | `release_v6`，1,055 道题，和官方 release_v6 的题数一致 |
| 题目时间 | 2023 年 5 月到 2025 年 4 月 |
| 来源 | AtCoder 602、LeetCode 444、Codeforces 9 |
| 难度 | 简单 322、中等 383、困难 350 |
| 题型 | 读标准输入的 611 道（AtCoder、Codeforces）；补全函数的 444 道（LeetCode） |

### 3.2 两个任务

| 任务 | 每题写几份 | 随机性 | 报的分数 |
|---|---:|---|---|
| `lcb_codegen` | 1 | 无（温度 0） | 通过率 |
| `lcb_codegen_sampled` | 32 | 温度 0.2，top-p 0.95 | 单份通过率；32 份里至少一份通过的比例 |

### 3.3 提示怎么拼

用的是官方给基础模型（非对话模型）的格式：一道固定的示例题带参考程序，后面接待测题，停在 `### Answer` 之后。

```
### Question
（示例题的题面）

### Answer

（示例题的参考程序）

### Question
（待测题的题面）

### Answer

```

补全函数的题在题面后多一段 `### Starter Code`，给出函数签名。两种题型各用一道固定的示例题。

提示长度（按 Qwen3 的分词器）：中位数 1,412 token，最长 2,935 token，远低于服务端上限 16,384，题目不会被截。

### 3.4 生成参数

| 项目 | `lcb_codegen` | `lcb_codegen_sampled` |
|---|---|---|
| 每题份数 | 1 | 32 |
| 温度 | 0 | 0.2 |
| top-p | 0.95（温度为 0 时不起作用） | 0.95 |
| top-k | 不用 | 不用 |
| 生成上限 | 1,024 token | 1,024 token |
| 停止符 | `###` | `###` |

### 3.5 怎么判对错

1. 模型的回答去掉首尾空白后，直接当作程序运行。不去掉 markdown 代码围栏，这是官方对基础模型的做法。
2. 测试包括公开样例和隐藏测试，全部通过才算这份程序通过。每题的测试数从 1 个到 103 个，中位数 13 个。
3. 读标准输入的题：把输入喂给程序，逐行比对输出（去掉首尾空白；不相等时再按数值比一次）。
4. 补全函数的题：调用模型写的函数，返回值和期望值相等才算对。
5. 每个测试限时 6 秒。结果分五种：通过、答案错、超时、运行出错、测试程序出错。
6. 16 个进程并行判分，每个测试进程最多新增 16 GB 内存。

分数的算法：

- **通过率（写 1 份）**：通过的题数除以 1,055。
- **单份通过率（写 32 份）**：每题 32 份里通过的比例，再对所有题取平均。
- **32 份里至少一份通过**：至少有一份通过的题数除以 1,055。

分数文件里还按难度（简单、中等、困难）分别给出通过率，不计入 66 个分数。

### 3.6 和官方做法的对照

官方值来自 LiveCodeBench 仓库的参数定义文件和说明文档。

| 项目 | 我们 | 官方 | 一致 |
|---|---|---|---|
| 温度 | 0.2 | 0.2 | 是 |
| top-p | 0.95 | 0.95 | 是 |
| top-k | 不用 | 没有这个参数 | 是 |
| 停止符 | `###` | `###` | 是 |
| 生成上限 | 1,024 | 2,000 | 否 |
| 每题份数 | 32 | 10 | 否 |
| 提示格式、回答的处理 | 官方的基础模型格式 | | 是 |

官方说明里的原话是"We use n=10 and temperature=0.2 for generation"。`lcb_codegen`（不采样、写 1 份）不是官方设置，是我们自己加的。

### 3.7 两处差别的影响

**生成上限 1,024 对 2,000。** 写到上限还没停的回答占比（`lcb_codegen` / `lcb_codegen_sampled`，按每次测评的前 400 道题统计）：

| 模型 | 写 1 份 | 写 32 份 |
|---|---:|---:|
| Qwen3.8-Max | 0.5% | 0.4% |
| MiMo-V2.5-Pro-Base | 1.0% | 0.7% |
| Qwen3-235B | 1.2% | 1.4% |
| MiMo-V2.6-Pro-RL | 2.2% | 1.3% |
| PROD1_DPSK_LATENT2_E2X_QBK3 q3 | 2.2% | 2.3% |
| 59B iter18000 | 3.0% | 2.8% |
| Qwen3-30B-Base | 4.2% | 3.9% |
| Qwen3.8-27B | 5.5% | 6.0% |
| Qwen3.8-Flash-Next | 6.5% | 5.5% |
| PROD1_DPSK_LATENT2_E2X q3 | 6.8% | 6.1% |

- 正常的程序很短，中位数 110 到 130 token。
- 撞上限的原因分两种。我们自己的 E2X q3 有 83% 是在重复绕圈，放宽上限也过不了；Qwen3.8-27B 和 Flash-Next 分别只有 7% 和 1% 是绕圈，其余是真的写得长。
- 任务清单的注释里记有一个估算：在已有回答上，上限 1,024 和 2,000 的差别是 −0.04 到 +0.01 分。

**每题 32 份对 10 份。** 单份通过率不受份数影响，可以和官方数字比。"N 份里至少一份通过"取决于 N，我们的 32 份和官方的 10 份不能直接比。

### 3.8 两个任务的关系

10 次测评里，两个任务给出的结论几乎相同：

| 模型 | 写 1 份的通过率 | 写 32 份的单份通过率 | 差 | 32 份至少一份通过 | 用时（分钟） |
|---|---:|---:|---:|---:|---|
| Qwen3.8-Max | 52.6 | 51.7 | 0.9 | 68.8 | 10 / 157 |
| MiMo-V2.6-Pro-RL | 49.7 | 50.0 | −0.3 | 69.3 | 7 / 93 |
| MiMo-V2.5-Pro-Base | 45.8 | 45.9 | −0.1 | 65.2 | 7 / 96 |
| Qwen3.8-27B | 40.7 | 41.1 | −0.5 | 62.0 | 5 / 92 |
| Qwen3.8-Flash-Next | 40.4 | 37.8 | 2.6 | 65.6 | 8 / 155 |
| Qwen3-235B | 38.3 | 36.5 | 1.8 | 56.1 | 5 / 97 |
| Qwen3-30B-Base | 36.3 | 35.8 | 0.5 | 51.5 | 4 / 42 |
| 59B iter18000 | 23.3 | 22.9 | 0.4 | 35.8 | 13 / 255 |
| PROD1_DPSK_LATENT2_E2X_QBK3 q3 | 16.3 | 15.9 | 0.4 | 26.2 | 8 / 151 |
| PROD1_DPSK_LATENT2_E2X q3 | 14.6 | 14.2 | 0.4 | 24.8 | 9 / 179 |

- 两个通过率的相关系数是 0.9975，10 个模型的排名完全相同；平均差 0.8 分，最大差 2.6 分。
- 每个数的误差范围是 ±1.5 和 ±1.35，几乎一样：误差主要来自只有 1,055 道题，多写 31 份并没有让结果更准多少。
- 写 32 份独有的信息是"32 份里至少一份通过"。
- 用时是作业记录的墙钟时间，期间服务器同时在跑别的任务，只能看量级。
- 决定（2026-11-13）：两个任务都继续测。

## 4. HumanEval 和 MBPP：和公开做法的对照

这两个 benchmark 除原论文外没有统一规定，下面把常被引用的几家并排列出。

- **原论文**：出题人自己的测法。HumanEval 是 OpenAI 2021 年的 Codex 论文，MBPP 是 Google 2021 年的论文。
- **上游 lm-eval**：我们这套测评框架的公开原版，里面自带的任务。
- **EvalPlus**：公开的代码测评工具和榜单。
- **bigcode**：BigCode 榜单用的测评工具。

### 4.1 HumanEval

| 项目 | 我们 | 原论文 | 上游 lm-eval | EvalPlus | bigcode |
|---|---|---|---|---|---|
| 温度 | 0.2 | 单份通过率用 0.2 | 不采样 | 不采样 | 0.2 |
| top-p | 0.95 | 0.95 | 无 | 未设 | 0.95 |
| top-k | 不用 | 没提 | 无 | 未设 | 关 |
| 生成上限 | 512 | 没有规定 | 1,024 | 768 | 512 |
| 每题份数 | 32 | 200 | 1 | 1 | 200 |

- 原论文每题写 200 份，是因为要报"100 份里至少一份通过"；只看单份通过率不需要这么多。
- 生成上限：原论文没有规定，我们的 512 是这几家里最紧的。bigcode 的 512 是否把提示也算在内，它的文档没有说清。

### 4.2 MBPP

| 项目 | 我们 | 原论文 | 上游 lm-eval | EvalPlus | bigcode |
|---|---|---|---|---|---|
| 温度 | 0.5 | 0.5 | 不采样 | 不采样 | 0.1 |
| top-p、top-k | 都不用 | 都没提 | 无 | 未设 | top-p 0.95 |
| 生成上限 | 768 | 没有规定 | 未设 | 768 | 512 |
| 每题份数 | 32 | 80 | 1 | 1 | 15 |
| 示例题 | 第 2、3、4 题 | 数据集说明里写的是第 2、3、4 题 | 第 2、3、4 题 | | 另一种提示 |

- 原论文的主指标是"80 份里只要有一份通过就算解出"，没有报单份通过率。和它对应的是我们的"32 份里至少一份通过"，但份数不同，不能直接比。
- 题集不同：原论文用 500 道测试题，我们用的是人工校对过的子集，257 道。

## 5. 截断的实际影响：撞到生成上限的比例

写到上限还没停的回答占比，按每个任务的前 400 道题统计。

| 任务 | 上限 | Qwen3.8-Max | Qwen3.8-27B | Qwen3-30B-Base | E2X q3 | 59B iter18000 |
|---|---:|---:|---:|---:|---:|---:|
| popqa_15shot | 15 | 0% | 0% | 0% | 0% | 0% |
| simpleqa_5shot | 64 | 0% | 0% | 1.0% | 0.2% | 0.5% |
| simpleqa_verified_25shot | 64 | 0% | 0% | 0.5% | 0.2% | 0.2% |
| facts_parametric_25shot | 64 | 0% | 0% | 2.2% | 5.0% | 11.2% |
| trivia_qa | 256 | 0% | 0% | 0.2% | 0% | 0% |
| naturalqs_5shot | 256 | 0% | 0% | 0.2% | 0.5% | 1.5% |
| humaneval_sampled | 512 | 1.7% | 0.7% | 1.1% | 8.9% | 15.1% |
| mbpp_sanitized_3shot_sampled | 768 | 0% | 5.7% | 0% | 0.2% | 0.2% |
| lcb_codegen | 1,024 | 0.5% | 5.5% | 4.2% | 6.8% | 3.0% |
| lcb_codegen_sampled | 1,024 | 0.4% | 6.0% | 3.9% | 6.1% | 2.8% |
| gsm8k_cot | 1,024 | 0.2% | 0.2% | 0% | 1.2% | 1.0% |
| gsm8k_cot_nemo | 1,024 | 0% | 0% | 0% | 1.8% | 0.5% |
| agieval_en_cot | 2,000 | 0.9% | 9.5% | 3.0% | 12.7% | 9.4% |
| mmlu_pro_5_shot_base | 2,048 | 3.3% | 6.6% | 0.4% | 4.2% | 1.9% |
| supergpqa_main_cot_n_shot | 2,048 | 7.2% | 24.2% | 8.2% | 20.0% | 12.0% |
| gpqa_diamond_cot_n_shot | 2,048 | 5.6% | 55.1% | 6.1% | 33.2% | 13.3% |
| gpqa_diamond_cot_n_shot_sampled | 2,048 | 0.9% | 51.3% | 1.3% | 9.1% | 4.5% |
| math_500_sampled | 4,096 | 1.5% | 5.4% | 1.1% | 8.8% | 3.4% |
| minerva_math_nemo | 4,096 | 2.4% | 7.1% | 4.2% | 16.3% | 11.4% |
| aime24 | 4,096 | 16.2% | 41.0% | 4.4% | 24.5% | 13.3% |
| aime25 | 4,096 | 12.9% | 49.3% | 4.4% | 22.2% | 15.4% |
| hmmt_sampled | 4,096 | 23.8% | 66.6% | 4.4% | 19.1% | 12.5% |
| mgsm_native_cot_zh | 4,096 | 0% | 0% | 0% | 0.4% | 0.8% |

- Qwen3.8-27B 在 HMMT、GPQA、AIME、SuperGPQA 上撞上限的比例很高（24% 到 67%）：它的推理过程写得长，这几个任务上的分数受上限影响，和其他模型比较时要注意。
- 撞上限不等于吃亏。回答如果是在重复绕圈，放宽上限也不会变对。

## 6. 来源

- LiveCodeBench 官方参数：<https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/main/lcb_runner/runner/parser.py>
- LiveCodeBench 官方说明：<https://raw.githubusercontent.com/LiveCodeBench/LiveCodeBench/main/README.md>
- HumanEval 原论文（Codex，arXiv 2107.03374）：<https://ar5iv.labs.arxiv.org/html/2107.03374>
- MBPP 原论文（arXiv 2108.07732）：<https://ar5iv.labs.arxiv.org/html/2108.07732>
- MBPP 数据集说明：<https://raw.githubusercontent.com/google-research/google-research/master/mbpp/README.md>
- 上游 lm-eval 的 HumanEval 任务：<https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/humaneval/humaneval.yaml>
- 上游 lm-eval 的 MBPP 任务：<https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/lm_eval/tasks/mbpp/mbpp.yaml>
- EvalPlus：<https://raw.githubusercontent.com/evalplus/evalplus/master/evalplus/codegen.py>
- bigcode-evaluation-harness：<https://raw.githubusercontent.com/bigcode-project/bigcode-evaluation-harness/main/docs/README.md>
- LongCodeBench 论文（arXiv 2505.07897）：<https://arxiv.org/abs/2505.07897v3>

我们自己的数字来自 `runs/cosmos_eval_downstream/` 下各次测评的分数文件、逐题记录和健康检查文件。官方和公开做法的内容是 2026-11-13 从上面的页面读取的。
