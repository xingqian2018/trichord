# 基于 VLM 的 AI 生成图像 Artifact 检查标准：Open T2I Pilot 分析报告

**日期：** 2026-10-22
**数据来源：** Open T2I Pilot Artifact（https://claude.ai/artifact/P7myb4kGd3e4RqtkrMYGRv），4 个 T2I 模型 × 56 个 prompt = 224 张图
**状态：** 标准 v1 已完成；标签为 silver label（模型标注，未经人工复核）；VLM 效果测评待启动

---

## 1. 目标

1. 逐张检查 Open T2I Pilot 的 224 张生成图，归纳 AI 生成图像中实际出现的 artifact 类型。
2. 据此制定一套 VLM 可以逐条执行的检查标准，用于检查来源未知的输入图片。
3. 以这 224 张图作为带标签的测试集，测评 VLM 按该标准检查的效果（precision / recall）。

本报告完成了第 1、2 步，第 3 步的方案见第 7 节。

## 2. 数据

| 模型 | 设置 |
|---|---|
| krea2（Krea 2 Turbo） | 8 steps，长边 2048 |
| ideogram4（Ideogram 4） | 20 steps 加速，约 4 MP |
| flux3（FLUX 3 Klein） | 16 steps，长边 2048 |
| qwen21（Qwen-Image 2.1） | 40 steps，约 4 MP |

- 56 个 prompt 分 7 个大类，每类 8 个：Design & text、People & performance、Stylized & imaginative、Objects/products & food、Natural world、Built environment、Life/culture & narrative。
- prompt 语言：英语 51 个，西班牙语 2 个，葡萄牙语、韩语、阿拉伯语各 1 个。
- 每张图都附有一条重写的 caption（Haiku），计划作为 T2I 训练对使用。

## 3. 方法

1. **逐图审查**：4 个 agent 并行，每个模型一个，共用同一份评分标准（[review_rubric.md](review_rubric.md)）。每张图都实际打开查看，细小文字会裁剪放大。每张图输出以下内容：
   - 6 个维度的评分：渲染失败、prompt 遵循、文字准确、artifact、美学、caption 一致性；
   - 一个处理建议：keep / fix_caption / reject；
   - 逐条的 artifact 记录，每条包含类型、位置、严重程度和描述。
2. **补查**：ideogram4 前 32 张、qwen21 前 36 张的 artifact 记录最初是凭第一轮笔记补填的。逐张重新查看后，qwen21 这部分的记录从 47 条增加到 74 条，ideogram4 补查的 28 张从 60 条增加到 111 条，可见凭笔记补填大约漏掉三到四成。
3. **人工抽查**：我抽查了 5 张图（krea2 9020fa4d、flux3 4ee0b90e、qwen21 c055516a、ideogram4 c055516a、ideogram4 两张占位图），agent 的判断全部属实。
4. **归纳 taxonomy**：共 487 条 artifact 记录，先自下而上归纳出 v0 分类，再由一个 agent 逐条归类。33 条（7%）归类不合适，据此补出 v1：新增 A18、A19、B07，放宽 A04、A06、A14、C01 的定义，并写明 9 组易混类别的边界规则。v1 之后不再有归类不合适的记录。

## 4. 主要发现

### 4.1 总体

- **69 / 224 张（31%）** 至少有一处 moderate 及以上的图像 artifact；46 张完全没有发现 A 层 artifact。
- 按大类统计 moderate 及以上 artifact 的比例：

| 大类 | 有问题的图 / 总数 |
|---|---|
| Design & text | 16 / 32 |
| People & performance | 16 / 32 |
| Life/culture & narrative | 11 / 32 |
| Stylized & imaginative | 9 / 32 |
| Built environment | 7 / 32 |
| Natural world | 6 / 32 |
| Objects/products & food | 4 / 32 |

  文字和人物是两大难点，物体和自然场景的问题最少。

### 4.2 各模型对比

| 模型 | keep | fix_caption | reject | 无 A 类 artifact | 有 severe artifact |
|---|---|---|---|---|---|
| krea2 | 34 | 20 | 2 | 5 | 2 |
| ideogram4 | 33 | 15 | 8 | 9 | 8 |
| flux3 | 42 | 11 | 3 | 17 | 2 |
| qwen21 | 45 | 11 | 0 | 15 | 0 |

- **qwen21 最稳**：没有 reject，也没有 severe artifact。
- **ideogram4 问题最多**，有 8 张 reject，具体是：
  - 2 张 "Image blocked by safety filter" 占位图；
  - 一只狗的身后多出第二个狗头；
  - 文字缺字母，如 "BIRTHDA" 和 "AKES THE THRONE"；
  - 签名、印章和假透明棋盘格直接画在像素里。
- **注意**：原 Pilot 页面把这 2 张占位图算作生成成功，错误数和吞吐统计需要修正。
- **注意**：4 个模型由 4 个不同的 agent 审查，标注细致程度不一，记录条数从 99 到 156 不等。因此模型间的横向差异有一部分来自审查者，仅供参考。

### 4.3 Caption 问题（对训练数据影响最大）

224 张中有 **57 张被判为 fix_caption**：图本身可以用，但 caption 描述的内容图里并没有。主要原因是重写 caption 的模型照抄了 prompt，没有根据图来描述。典型例子：

- 图里写的是 "Mon Tue Wed Fru Sun"，caption 却写完整的周一到周日（krea2 538c255f）。
- 图里写的是 "HAPPY TALS"，caption 悄悄改成了 "TAILS"（qwen21 c055516a）。
- 图里有 7 个或 8 个学生，caption 照 prompt 写 6 个（8df9b2b0，三个模型都是这样）。
- prompt 要求的切面图、抓脚踝等动作在图里没有出现，caption 仍然照写。
- 两张 safety filter 占位图也配了完整的 caption，描述的是并不存在的画面。

这类图文不一致的数据直接拿去训练，会教模型"文字写错也算对"。建议在 7M 全量生产前，把 caption 一致性检查（或改为对图做 grounded recaption）加进流水线。

### 4.4 最常见的 artifact

| 代码 | 类型 | 涉及图片数（任意严重程度 / moderate+） |
|---|---|---|
| A03 | 背景伪文字 | 38 / 7 |
| A02 | 主要文字错误 | 27 / 17 |
| A10 | 不可能的结构 | 26 / 8 |
| A12 | 物理 / 光照错误 | 26 / 8 |
| A09 | 悬浮 / 无支撑 | 24 / 6 |
| A08 | 物体融合 | 22 / 3 |
| A05 | 身体结构 | 21 / 8 |
| A13 | 重复 / 复制粘贴 | 19 / 4 |
| A04 | 手部结构 | 16 / 2 |

A02 的 moderate+ 比例最高（17/27）：文字一旦出错，通常都很显眼。

## 5. 检查标准 v1 概要

完整定义、触发条件、检查问题、严重程度锚点、实例和输出 schema 见 [criteria_v1.md](criteria_v1.md)。该文档用英文写，可以直接作为 VLM prompt 使用。

### 5.1 分三层

- **A 层：只看图（19 条）。** 用于检查来源未知的图片，是这套标准的主体。
- **B 层：需要 prompt（7 条）。** 数量错误、缺失或错误的物体、动作或姿态、文字内容不符、属性、风格或视角不符、空间布局错误、出现了 prompt 明确排除的元素。
- **C 层：风险标记（1 条）。** 真实品牌、真实人物、真实地标。本批出现过真实俱乐部队徽加球员名，以及 KODAK logo。

### 5.2 A 层 19 条

| 代码 | 名称 | 代码 | 名称 |
|---|---|---|---|
| A01 | 渲染失败 / 占位图 | A11 | 透视 / 深度错误 |
| A02 | 主要文字错误 | A12 | 物理 / 光照 / 反射错误 |
| A03 | 背景伪文字 | A13 | 重复 / 复制粘贴 |
| A04 | 手部结构 | A14 | 纹理异常（融化、塑料感、污渍） |
| A05 | 身体结构 / 肢体归属 | A15 | 合成痕迹（贴图感、风格或光照割裂） |
| A06 | 面部结构 | A16 | 非内容叠加（水印、签名、棋盘格、留白画布） |
| A07 | 幻影重复（多出的头、人、部件） | A17 | 构图裁切不当 |
| A08 | 物体融合 | A18 | 版式对齐（设计稿、UI） |
| A09 | 悬浮 / 无支撑 | A19 | 场景不协调 |
| A10 | 不可能的结构 | | |

### 5.3 设计要点

- **触发条件（gate）**：例如画面里没有手，A04 就输出 `n/a` 而不是 `absent`。这样测评时可以区分"没看出来"和"不适用"。
- **证据要求**：判为 present 时必须给出位置和实际看到的内容，文字类要逐字转录。这样可以抽查，也能减少 VLM 凭 prompt 脑补。
- **边界规则**：9 组易混类别都有明确规则，例如：
  - A07 幻影重复 vs B01 数量错误：多出来的实例本身异常（没有身体、出现在不可能的位置）算 A07；多出一个完整、合理的实例算 B01。
  - A05 vs A08：轮廓融在一起算 A08；轮廓清楚但看不出肢体属于谁算 A05。

## 6. 局限

1. **标签是 silver label**：由 Claude agent 标注，只人工抽查了 5 张。如果被测 VLM 也是 Claude 或同类模型，两者的共同盲区不会显示为漏检，测评结果会偏乐观。
2. **样本量小**：A06、A16、A18、A19 等类别只涉及 5–8 张图，per-criterion 指标的置信区间会很宽。
3. **审查者差异**：见 4.2 的说明。
4. **未覆盖的类型**：这批图里没有明显的 NSFW、暴力或极端低分辨率样本，v1 暂未覆盖这些类型。

## 7. 下一步

1. **人工复核标签**：做一个标注页面（Artifact 加共享数据库），逐条对 agent 标注打勾、打叉或修改严重程度，并补充漏标。预计 1–2 人时，完成后得到 gold label。
2. **确定被测 VLM**：候选包括 Qwen-VL、SILA captioning 模型、Gemini、Claude，待定。
3. **测评方案**：
   - 被测 VLM 按 criteria_v1 对 224 张图逐条输出 present / absent / n/a、严重程度和证据；
   - 指标包括：每条标准的 precision / recall / F1；图像级的误杀率与漏检率（以 moderate+ 为阈值）；按 T2I 模型和大类拆分；
   - 对比两种调用方式：一次性检查全部标准，和逐条单独提问；
   - 抽查证据字段的真实性，看 VLM 是否在没有看清的情况下编造内容。
4. **7M 全量前的改进**：
   - 过滤 safety 占位图；
   - 加入 caption 一致性检查，或改为 grounded recaption；
   - 对 Design & text 类图片加强 A02 检查。

## 附：文件

| 文件 | 内容 |
|---|---|
| [criteria_v1.md](criteria_v1.md) | 检查标准 v1（英文，可直接用作 VLM prompt） |
| [review_rubric.md](review_rubric.md) | 逐图审查使用的评分标准 |
| [data/results_*.json](data/) | 4 个模型的逐图审查结果（评分、verdict、issues、artifacts_detail） |
| [data/artifacts_coded_v1.json](data/artifacts_coded_v1.json) | 487 条 artifact 记录及其 v1 分类代码 |
| [data/image_labels_v1.json](data/image_labels_v1.json) | 224 张图的 silver label（代码 → 最高严重程度：1 minor / 2 moderate / 3 severe） |
| [data/pilot_entries.json](data/pilot_entries.json) | Pilot 原始元数据（prompt、caption、图片路径、耗时） |
