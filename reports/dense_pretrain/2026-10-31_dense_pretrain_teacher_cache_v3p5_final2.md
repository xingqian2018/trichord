# 3p5 数据最终计划 2：按窗口块打分的 teacher cache，训练时 min padding

2026-10-31 · 状态：第 1–5 章（数据清单、special token 处理、tokenize、teacher cache、训练端）；后续章节待写

## 1. 数据清单与特殊情况

### 1.1 数据类型

| 类型 | 内容 | prefix | 源 token（N4） | 占源数据 | phase-1 权重 | 处理 |
|---|---|---:|---:|---:|---:|---|
| A 英文网页原文 | Common Crawl 2013–2025 的页面原文（CC99、CC8、OpenWebText2 等），以及翻译成英文的网页 | 881 | 3.79T | 26.8% | 18.47% | 普通文本 |
| B 英文网页改写 | 同一批网页被 LLM 改写成 5 种形式：diverse_qa（原文加问答）、wrap_medium（换文风）、extract_knowledge、knowledge_list、distill | 688 | 4.35T | 30.7% | 32.15% | 普通文本 |
| C 代码 | nemotron-codedata 的源码文件（单文件，按语言分）、合成的转译代码，以及被 LLM 重写成 Markdown 的代码网页（nemotron-cc-code-2.0） | 158 | 2.05T | 14.4% | 16.18% | 普通文本（见 1.3） |
| D PDF / 书 / 论文 | finepdfs（网上的 PDF）、eclair PDF、书和论文（其中有不少是法院判决书） | 52 | 1.47T | 10.4% | 6.48% | 普通文本 |
| E 数学网页 / 维基 | 数学网页、英文维基、维基改写，少量题库 | 12 | 0.25T | 1.8% | 6.53% | 普通文本 |
| F 非英语 | FineWeb2、2025 年的非英语网页、翻译数据（英文翻成多语的问答）、非英语维基 | 57 | 1.57T | 11.1% | 5.12% | 普通文本 |
| G 合成教材 / STEM | nv_domain 合成教材和事实问答、ndd_stem 研究级 STEM 问答 | 10 | 0.20T | 1.4% | 2.21% | 普通文本 |
| **H SFT / 推理 / 题库** | `sft-filtered-2025-*`（74 个）、`code-sft`、`math-sft`、`general-sft`、`tasks-data/*/mmlu-sft\|math-sft`：题目加解答，部分带 R1 思考过程 | 117 | 0.49T | 3.5% | 12.86% | **见 1.2** |
| **合计** | | **1,975** | **14.17T** | **100%** | **100%** | |

*“普通文本”的处理：用 Qwen3 `tokenizer.json`（sha256 `aeb13307…`）切词，自带 NFC，不加 BOS；26 个 added token 的字符串出现在正文里时一律按普通文字切；每篇末尾由代码追加 EOD 151643。这是语料里 151643 唯一的来源。H 类权重远高于它的数据量占比（12.86% 对 3.5%），所以训练时这部分会被重复多次。*

### 1.2 SFT / 推理 / 题库：结构情况

| # | 情况 | 原文例子 | prefix | phase-1 权重 | 处理 |
|---|---|---|---|---:|---|
| 1 | JSON 外壳 | N4 binidx 里存的是整条 JSON：`{"expected_answer":"4160","majority_res":"5/8","text":"Consider the set $\\{1, 2, 3, \\ldots, 8\\}$. …` | 74 个，即全部 `sft-filtered-2025-10-17/*` 和 `sft-filtered-2025-11-05/*` | 9.96% | **从 binidx 里存的整条 JSON 解析出 `text` 字段来切，不读原始 JSONL（N4 切词前的预处理删过部分记录，两边行号对不上）** |
| 2 | 完整思考块 | `…for a given positive integer k.⏎⏎<think>⏎I need to find all positive integers n such that … So final answer.⏎</think>⏎The problem requires finding all positive integers n …` | 23 个：R1 类（`*_r1_*`、`math-sft_r1-0528`、`igor-r1`、`som-r1`、`hackerrank_r1_0528` 等）、staging 的 `code-sft` 3 个、`math-sft` 1 个 | 5.17% | **`<think>` 和 `</think>` 保留单个 id**（151667 / 151668） |
| 3 | 空思考块 | `Give me an altered plot of the Tangled movie where Gothel … ⏎⏎<think></think>⏎⏎**Title: Tangled …`；`<think>⏎⏎</think>⏎⏎Which one of the following most closely parallels …` | 7 个，主要是 `lsat_qwen3_*_aug01` ×3（48–94% 的篇）、`its-hs3-*-sys1`（100%）、`widentity_…system1_system2`（50%）。这是 Qwen3 关闭思考模式时的输出残留 | 0.41% | 保留：`<think>` `</think>` 仍是单个 id（和情况 2 一致） |
| 4 | `<think>` 没闭合 | 思考一直写到全文结尾，没有 `</think>`，后面也没有正式的答案部分：`…it's \boxed{5} when t is small and \boxed{3} otherwise. However, the problem might have a different intended solution.⏎`（这篇有 49,595 字符） | 9 个推理类 prefix（`igor-r1`、`math_r1_sdg_*`、`math-sft_r1-0528`、`gradient_entropy_3.3M` 等），每个只占 ≤2% 的篇 | — | 原样保留 |
| 5 | 思考标签被去掉 | `…J: The expansion of European exploration and colonization⏎⏎⏎I need to determine which factor was most significant …` | 9 个 R1-0528 solutions 类（`openstem` 2.09%、`HLE_mcq` 0.32%、`hle_openq`、`syn_gpqa`、`limo` 等）：R1 的思考过程还在，但标签被上游去掉了 | 2.73% | 切词不受影响；只需知道同类 R1 数据有的带标签、有的不带 |
| 6 | 纯文字版式 | `input: In a convex quadrilateral (ABCD), the diagonals intersect at …`；`Question: … Answer: …`；`\boxed{…}` | 其余大多数（`genetic_instruct_*`、`aops`、`longmath`、FLAN、`tasks-data` 等） | — | 普通文本，无需处理 |
| 7 | 对话模板 / 工具 / FIM | `<\|im_start\|>` `<\|im_end\|>` `<tool_call>` `<\|fim_prefix\|>` `[INST]` `<extra_id_1>` … | **0**：所有抽样里都没出现 | — | 以防万一，H 类里对话和工具 token 保持单个 id，FIM 拆开（第 2 章） |

*⏎ 表示换行。一个 prefix 可以同时属于多种情况：例如 74 个 JSON 外壳 prefix 里，也有情况 2 到 6 的各种版式。*

### 1.3 普通文本里的“假控制 token”

正文里偶尔会出现特殊 token 的字符串，它们只是普通文字。如果按 HF/vLLM 的默认方式切，就会变成真的控制 id，所以必须拆开。

| 字符串 | 原文例子 | 频率（抽样） | 按 HF 默认切会变成 |
|---|---|---|---|
| `<\|endoftext\|>` | 一段韩文 Python 微调脚本：`prompt = f"질문: {…}\n답변: {…}\n<\|endoftext\|>"` | 3 / 580,442 篇（都在代码里）；650B 未覆盖的 SFT 里 1 / 35,100 篇 | EOD 151643：文档被假切断，docmask 也会在这里断开 |
| `<\|repo_name\|>` `<\|file_sep\|>` | `…we will do our best to help.<\|repo_name\|>mklarner/tao<\|file_sep\|>/tao.egg-info/PKG-INFO …` | `nemotron-cc-code-2.0` 的 300 篇样本里有 1 篇 | FIM / repo 控制 id 151663 / 151664 |

### 1.4 切词规则汇总

- **全部 1,975 个 prefix**：Qwen3 `tokenizer.json`（sha256 `aeb13307…`），每篇末尾追加 EOD 151643，不加 BOS。
- **special token 按数据类型处理**（第 2 章）：A–G 类的 26 个 added token 全部按普通文字切；H 类（117 个 SFT prefix）保留 8 个结构 token 的单个 id（`<|im_start|>` `<|im_end|>` `<think>` `</think>` `<tool_call>` `</tool_call>` `<tool_response>` `</tool_response>`），其余 18 个也拆开。
- **74 个 `sft-filtered-*`**：解码 N4 binidx 得到整条 JSON，取其中的 `text` 字段，不读原始 JSONL。
- 情况 3（空思考块）和情况 4（`<think>` 未闭合）都原样保留。

### 1.5 抽样方法与局限

| 样本 | 范围 | 篇数 |
|---|---|---:|
| 650B 已覆盖的部分 | 1,975 个 prefix，每个取前 ≤300 篇（≤100 万 token） | 580,442 |
| 650B 未覆盖的 SFT | 117 个 SFT prefix，每个从 650B 截断点之后随机抽 300 篇 | 35,100 |
| SFT 版式归类 | 117 个 SFT prefix，每个在整个 prefix 里随机抽 200 篇 | 23,400 |

规则按数据类型执行（第 2 章），不依赖抽样名单，也不需要逐篇扫完 15T。抽样只能保证“经常出现”的情况都已找到：每个 SFT prefix 约看了 800 篇，出现率 ≥1% 的情况漏掉的概率约 0.03%，出现率 0.1% 的情况漏掉的概率约 45%。普通文本只抽了每个 prefix 的开头。正式切词时，在构建过程中顺便统计每个 prefix 各个 special token 的计数（2.4），复核这一章的结论。

## 2. special token 处理

Qwen3 词表里有 26 个 added token（id 151643–151668）。同一个字符串，**出现在正文里时是数据**，按普通文字切；**被数据格式用来表示结构时**（对话模板、思考块、工具调用）才合并成单个 id。所以处理方式取决于数据类型：A–G 类是普通文本，正文里出现的都是字面量；H 类（SFT / 推理 / 题库）的版式可能用到结构 token。

### 2.1 处理规则

| id | token | 原 tokenizer 标记 | A–G 普通文本 | H SFT / 推理 / 题库 | 原因 |
|---|---|---|---|---|---|
| 151643 | 篇尾 EOD | — | 单个 id（代码追加） | 单个 id（代码追加） | 文档边界只由代码决定 |
| 151643 | 正文里的 `<\|endoftext\|>` | special | 拆开 | 拆开 | 如果篇中出现真的 EOD，一篇会被切成两篇，docmask 也会在这里断开 |
| 151644–151645 | `<\|im_start\|>` `<\|im_end\|>` | special | 拆开 | **单个 id** | 在 SFT 里只可能是对话模板（抽样中 0 次，以防万一） |
| 151646–151656 | 11 个多模态 token：`<\|object_ref_start\|>` `<\|object_ref_end\|>` `<\|box_start\|>` `<\|box_end\|>` `<\|quad_start\|>` `<\|quad_end\|>` `<\|vision_start\|>` `<\|vision_end\|>` `<\|vision_pad\|>` `<\|image_pad\|>` `<\|video_pad\|>` | special | 拆开 | 拆开 | 纯文本数据里没有图像，它们不可能是结构 |
| 151657–151658 | `<tool_call>` `</tool_call>` | 非 special | 拆开 | **单个 id** | 工具调用格式（抽样中 0 次，以防万一） |
| 151659–151664 | `<\|fim_prefix\|>` `<\|fim_middle\|>` `<\|fim_suffix\|>` `<\|fim_pad\|>` `<\|repo_name\|>` `<\|file_sep\|>` | 非 special | 拆开 | 拆开 | 数据里没有 FIM / 多文件格式的代码；唯一见到的一次是代码网页里的字面量（1.3） |
| 151665–151666 | `<tool_response>` `</tool_response>` | 非 special | 拆开 | **单个 id** | 工具返回格式（抽样中 0 次，以防万一） |
| 151667–151668 | `<think>` `</think>` | 非 special | 拆开 | **单个 id** | 思考块（1.2 情况 2、3） |

- **“拆开”**指按普通文字切，例如 `<|im_start|>` 切成 `<` `|` `im` `_start` `|` `>`，`<think>` 切成 `<th` `ink` `>`。
- **H 类**就是第 1 章里路径含 `sft` 的 117 个 prefix，规则对整类生效，不依赖抽样名单。H 类里本来不含这些 token 的 prefix，这条规则不改变任何结果。
- 按目前的抽样，H 类里实际会合并的只有 `<think>` 和 `</think>`；对话和工具 token 的规则是以防万一。
- 1.2 的情况 3、4 按原样切：空思考块是 151667、换行、151668；未闭合的思考块只有 151667。两种都原样保留。

### 2.2 和其他切法的对比

| | HF / vLLM 默认 | 650B / 45B 现有语料（`plain`） | 计划 2：A–G | 计划 2：H |
|---|---|---|---|---|
| 14 个 special（`<\|endoftext\|>`、`<\|im_*\|>`、11 个多模态） | 单个 id | 拆开 | 拆开 | `<\|im_start\|>` `<\|im_end\|>` 单个 id，其余 12 个拆开 |
| 12 个非 special（tool ×4、fim/repo ×6、think ×2） | 单个 id | 单个 id | 拆开 | tool、think 单个 id，fim/repo 拆开 |

在第 1 章的抽样数据上，计划 2 和 650B 语料只差 1 篇：`nemotron-cc-code-2.0` 里带 `<|repo_name|>` / `<|file_sep|>` 字面量的那篇代码网页，在计划 2 里被拆开。其余差别只会出现在抽样没见到的文档上。

### 2.3 实现

- **两份切词器配置，来自同一份** Qwen3 `tokenizer.json`（sha256 `aeb13307…`）：重写每个 added token 的 `special` 标记，再打开 `encode_special_tokens=True`（tokenizers 0.23）。标记为 special 的 token 按普通文字切，没标记的匹配成单个 id。
  - `general`（A–G）：26 个全部标记为 special，全部拆开。
  - `sft`（H）：上表 8 个不标记，其余 18 个标记。
- **已实测**：一段含全部 26 个 token 的文本，用 `general` 切后没有 ≥151643 的 id，用 `sft` 切后只有这 8 个 id；一段含对话模板、思考块、工具调用的样例，用 `sft` 切后能原样解码还原；不含这些字符串的普通文本，两份配置和原 tokenizer 切出的 id 完全相同。
- **每篇**：`ids = encode(text) + [151643]`。不加 BOS，不用 post-processor 模板。
- **按 prefix 类型选配置**：源路径含 `sft` 用 `sft`，否则用 `general`。manifest 记录 tokenizer 的 sha256、配置名和保留的 token 列表。
- **teacher 侧**：producer 直接把 id 交给 vLLM，不经过 vLLM 的文本切词，所以 teacher 看到的就是这些 id。MiMo-2.5 的 tokenizer 和 Qwen3 相同；MiMo-2.6 多出的 6 个 id（151669–151674）不会出现在我们的 id 里。词表上限参数 151669 仍然保留，作为保险。

### 2.4 验收（构建时逐篇检查）

- 每篇恰好一个 151643，并且在最后一位。
- 所有 id < 151669。
- A–G：除了篇尾 EOD，没有 ≥151643 的 id。
- H：≥151643 的 id 只能是 8 个保留 token 加篇尾 EOD。每个 prefix 记录这 8 个 id 各自出现的篇数和次数，以及 `<think>` 比 `</think>` 多的篇数（即 1.2 情况 4）。
- 抽样解码：去掉篇尾 EOD 后 `decode(ids)` 必须等于 NFC(原文)。
- 每个 prefix 顺便统计 26 个字符串在原文里出现的篇数，用来复核第 1 章，例如 A–G 里 `<think>` 的实际出现次数。

## 3. 标准版 tokenize

### 3.1 切词步骤

每篇文档直接切词，不打乱、不装箱、不在这一步切块。

1. **范围**：Ultra-v3 phase-1 的全部 1,975 个 prefix，每个 prefix 的每一篇文档，约 14.7T Qwen3 token。
2. **按存储顺序逐篇切词**：用第 2 章的规则（A–G 用 `general` 配置，H 用 `sft` 配置），篇尾追加 EOD 151643，不加 BOS。
3. **取文本**：全部从 N4 binidx 解码；74 个 `sft-filtered-*` 解码出来是整条 JSON，再取其中的 `text` 字段。
4. **编号不变**：输出的第 i 篇就是源 prefix 的第 i 篇；空文档写成只有一个 EOD，保持一一对应。（3.4、3.5 删掉的篇除外：删掉之后，这些 prefix 里排在后面的文档编号前移）

### 3.2 目录结构

```
pretrain/qwen3_tokenized/ultra_v3_phase1_full/    全部 1,975 个 prefix，每篇文档都切词
  summary.json  blend.json                        合计；Megatron 的 blend
  <prefix_key>/
    data.bin  data.idx                            Megatron indexed dataset，int32，一篇一条
    manifest.json                                 来源、类型、切词配置、计数、校验
    [removed_docs_*.json]                         3.4 / 3.5 删掉的文档（删除前的编号）
```

### 3.3 文件定义

和现有的 650B 语料（`ultra_v3_phase1_650e9`）格式相同，由 `dpl data build-kd-corpus` 生成。

| 文件 | 内容 |
|---|---|
| `data.bin` / `data.idx` | Megatron indexed dataset：int32，一篇文档一条序列，`doc_idx` 为 `[0..N]` |
| `manifest.json` | 源 prefix 和 `.idx` 的 sha256、类型（general / sft）、取文本的方式（N4 解码 / N4 解码后取 JSON 的 `text`）、切词配置、tokenizer sha256、文档数、token 数、UTF-8 失败数、26 个 added token 各自出现的篇数（用于第 2 章 2.4 的验收） |
| `summary.json` | 全部 prefix 的合计 |
| `blend.json` | Megatron 的 per-split blend：1,975 个 (权重, prefix) |

### 3.4 gradient_entropy_3.3M：`<think>` `</think>` 失控推理的处置

`math-sft/2025-12-05/gradient_entropy_3.3M`（H 类，配比 0.36%）里有一部分推理生成失控：一直写到长度上限，后半段是乱码，有的夹着成百上千个 `<think>` `</think>`。按第 2 章的 `sft` 规则，这些标签都会切成真正的控制 id。80B 验证语料逐 prefix 统计，其余 H 类 prefix 没有 think 标签超量的情况。

逐篇数 `<think>` 和 `</think>` 的个数，按下表分档。统计来自 80B 验证语料里这个 prefix 的前 67,894 篇（全 prefix 3,299,477 篇的 2%）；各档互斥，合计 67,894 篇。

| 档 | 判据 | 篇数 | 占篇数 | 占 token | 样子 | 处置 |
|---|---|---:|---:|---:|---|---|
| 正常 | 1 个 `<think>` + 1 个 `</think>`（温和 B 除外） | 66,535 | 98.0% | 93.7% | 题目 → 推理 → `\boxed{}` 答案 | 保留 |
| 温和 A | 2 个 `<think>` + 1 个 `</think>` | 352 | 0.52% | 0.49% | 开头的 `<think>` 重复一次（`<think>⏎<think>Alright…`），正文完整 | 保留 |
| 温和 B | 1 个 `<think>` + 1 个 `</think>`，结尾 2,048 个 token 里不同的 4-gram 占比 < 10%（复读） | 7 | 0.01% | 0.04% | 推理正常，结尾反复同一句话直到长度上限 | 保留 |
| 温和 C | 1 个 `<think>` + 2 个 `</think>` | 9 | 0.01% | 0.04% | 思考闭合后又"重新检查"一轮，再闭合一次，正文可读 | 保留 |
| **严重 A** | `<think>` 和 `</think>` 合计 ≥ 4 | 824 | 1.21% | 4.75% | 后半段是乱码，夹着成百上千个 think 标签（最多 11,162 个） | **丢弃** |
| **严重 B** | 有 `<think>`、没有 `</think>`（合计 ≤ 3） | 167 | 0.25% | 0.97% | 全部写到长度上限（≥ 8,113 token），结尾是 "Wait, but wait, which is…" 一类的乱码 | **丢弃** |

> 丢弃条件（满足任一）：① `<think>` 和 `</think>` 合计 ≥ 4；② 有 `<think>`、没有 `</think>`。其余全部保留。

- 严重 B 是 1.2 情况 4（未闭合的思考块原样保留）在这个 prefix 的例外：这里的未闭合不是格式问题，而是内容已经失控。
- 丢弃约占这个 prefix 1.5% 的篇数、5.7% 的 token，约为总语料的 0.02%。
- 实现：tokenize 代码不改。tokenize 之后把这些篇从这个 prefix 的 `data.bin` / `data.idx` 里直接删掉，就当它们不存在，和 N4 删掉的记录一样。teacher cache 和训练直接读删完的数据。
- 80B 验证语料先做（删 991 篇）；全量语料等这个 prefix 切完后再做。

### 3.5 rare token 过滤：只删 rare token 标出的低质量文档

2026-11-01 加的一步，分两段。先由 Boxiang 用 rare-token 规则找出候选文档（3.5.1）；Runlong 决定不全删，只删候选里的低质量文档（3.5.2）。删除直接作用在 `ultra_v3_phase1_full` 上，只改 `data.idx`，`data.bin` 一个字节都没动（3.5.3）。teacher cache 的 plan 是删完之后才写的，所以被删的文档根本不会进入 teacher 打分（3.5.4）。

| 步骤 | 谁 | 结果 | teacher cache / 训练是否使用 |
|---|---|---|---|
| rare-token 规则，阈值 500 | Boxiang | 标出 110,046 篇 / 1.89B token；另建了一份把它们全删掉的语料 `ultra_v3_phase1_full_rtf500` | 只当候选；`rtf500` 不用 |
| 候选里只删低质量 | Runlong | 24,349 篇 / 131.6M token，涉及 529 个 prefix | **使用**：已直接删在 `ultra_v3_phase1_full` 里 |

#### 3.5.1 Boxiang：Qwen3 rare-token 规则（找候选）

规则沿用 N3-Ultra / N4 的做法（Brandon Norick），换成 Qwen3 词表重新统计：

1. 对 Qwen3 词表（151,936）里的每个 token，统计全语料 18,252,000,700 篇里有多少篇含它（文档频率 df）。
2. df ≤ 阈值的 token 记为 banned。两类永远不算：256 个单字节 token，以及 id ≥ 151643 的特殊 token。
3. 含任意一个 banned token 的文档，整篇标出。

各阈值的结果（`sweep_final.tsv`）：

| 阈值 | banned id | 其中语料里出现的 | 标出篇数 | 标出 token | 占 token |
|---:|---:|---:|---:|---:|---:|
| 300 | 2,411 | 490 | 46,775 | 1.11B | 0.0076% |
| **500** | **2,668** | **747** | **110,046** | **1.89B** | **0.0129%** |
| 750 | 2,939 | 1,018 | 218,283 | 3.05B | 0.0208% |
| 1000 | 3,135 | 1,214 | 340,259 | 4.19B | 0.0286% |
| 2000 | 3,893 | 1,972 | 1,103,924 | 10.89B | 0.0743% |

- 阈值 500 的 2,668 个 banned id 里，有 1,921 个 df = 0：Qwen3 自己的 tokenizer 根本切不出它们（pre-tokenizer 会切开泰文的元音和声调，NFC 会改写 CJK 兼容字），不影响任何文档。
- 出现过的 banned token 大多是 Qwen3 词表本身的特点，不是垃圾：常见词在 BPE 中途合并出的片段（越南语、波兰语、俄语、土耳其语）、罕见的中文词组、罕见的韩文音节。只有一部分是真正的垃圾信号，例如自动生成代码里的怪标识符、emoji 串。被标出的文档约 89% 只含 1 个 banned token。全删会伤到好数据：中文维基会损失 3.2% 的 token（阈值 1000 时 8.8%）。
- Boxiang 按阈值 500 生成了 `qwen3_tokenized/ultra_v3_phase1_full_rtf500`：只重写 idx 的语料视图，`data.bin` 链接回原语料，110,046 篇全删。**teacher cache 和训练都不用它。**
- 代码：`cosmos_llm_scaling_ladder/code/scaling_ladder/tools/rare_token_filter/rare_filter.py`。统计和报告：`qwen3_tokenized/rare_token_filter/ultra_v3_phase1/`（`report_final_t500`、`report_final_t1000`、`sweep_final.tsv`）。

#### 3.5.2 Runlong：只删候选里的低质量文档（实际执行）

决定（Runlong，2026-11-01）："coding 这种好的 data 放过，高质量的留下，低质量的删去"。同时满足下面 3 条的文档才删：

1. 被阈值 500 标出（阈值 1000 新增的不算）。
2. 不属于 code / wiki / SFT，来源也不是 stackexchange。
3. 命中至少一条质量条件（`scripts/rtf_common.py` 的 `v2b_clause` / `v3_extra`）。

另外，盲审判为 KEEP 的 34 篇不删。各质量条件删掉的量：

| 条件 | 篇数 | token | 盲审精度 |
|---|---:|---:|---|
| 韩文招妓 spam 关键词 ≥ 3 | 11,316 | 21.2M | 150/150 该删 |
| 来源 FineWeb2 AR | 2,349 | 13.4M | 65/70 该删 |
| 英文网页 / 合成改写里 U+FFFD ≥ 3（PDF-SEO 拼接页，及在其上生成的 QA） | 2,220 | 10.0M | 约 91% |
| 来源 persona math（答案尾部退化成乱码） | 1,769 | 2.5M | 6 删 / 3 存疑 / 0 留 |
| 字母占比 < 0.3（Unicode 码表、数字表格） | 295 | 45.4M | 约 78% |
| 阿拉伯 presentation forms、中文赌博、英文下载 spam | 770 | 3.2M | 77–93% |
| v3 扩展：韩文 spam 词、CC8 退化的 LLM 文本、英文 spam 词、U+ 码表页、在码表上生成的 QA、赌博词、乱码等 | 5,630 | 36.0M | 小样本 |

- 结果：**24,349 篇 / 131,594,332 token（约占语料的 0.0009%），涉及 529 个 prefix**。在 1,134 篇盲审样本上估计：精度约 89%，误删约 5%（v3 条件的阈值是在同一份样本上调的，数字偏乐观）。
- 没有删的：阈值 1000 新增的候选（过 v3 规则后 45,532 篇 / 178M token）；候选以外的垃圾（规则估计约 4,200 万篇 / 约 700 亿 token，但规则在候选以外的精度只有 50–68%，不做 classifier）。
- 名单和复核过程：`qwen3_tokenized/rare_token_quality_filter/ultra_v3_phase1/`（`drop_list.json`、`removed/<dataset_key>.npy`、`docs.tsv`、`datasets.tsv`、`REVIEW.md`、`scripts/`）。
- 语料合计：18,252,000,700 篇 / 14,661,224,033,933 token → **18,251,976,351 篇 / 14,661,092,439,601 token**。

#### 3.5.3 怎么删：`data.bin` 不动，只改 `data.idx`

2026-11-01 21:51–22:12 原地执行（Runlong 决定"直接删（移动）"）。

Megatron 的 `data.idx` 是 `data.bin` 的目录：每篇文档一条记录，记着它的长度和它在 `data.bin` 里的起始字节，第 i 条记录就是第 i 篇文档。训练和 teacher cache 都只通过 `data.idx` 找文档。

删除只做一件事：把被删文档的记录从 `data.idx` 里去掉。其余文档的长度和起始字节都不变；被删文档的 token 还在 `data.bin` 原来的位置，只是没有任何记录再指向它们（合计约 0.5 GB）。

```
data.bin   | doc 0 | doc 1 | doc 2（删）| doc 3 | doc 4 | ...    一个字节都不改
旧 idx       0       1       2            3       4
新 idx       0       1       ·            2       3               去掉 doc 2 的记录，后面的编号前移 1
```

实例（`CC-MAIN-2019-22.jsonl-6392e2e3`，删 3 篇）：

| | 删除前 | 删除后 |
|---|---:|---:|
| 文档数 | 2,938,830 | 2,938,827 |
| `data.bin` 字节数 | 9,275,424,456 | 9,275,424,456（不变） |
| idx 引用的字节 | 9,275,424,456 | 9,275,273,992 |
| 不再被引用的字节 | 0 | 150,464（= 37,616 token × 4） |
| 被删文档（旧编号） | — | 389135、463188、2023714 |
| 旧第 389136 篇 | 起始字节 1,219,653,656 | 变成新第 389135 篇，起始字节仍是 1,219,653,656 |

去掉这 3 条以后，其余每篇的长度和起始字节都和删除前逐条相同。

- 每个改动的 prefix：`data.idx` 重写；`manifest.json` 更新计数，删除前的值记在 `quality_filter` 字段；新增 `removed_docs_quality_t500.json`，记被删文档的旧编号。根目录 `summary.json` 同步更新。`blend.json` 没改，所以 teacher cache 的 blend_hash 不变。
- 备份：删除前的 `data.idx` 和 `manifest.json` 在 `rare_token_quality_filter/ultra_v3_phase1/original_idx/`；被删文档的 token 另存了一份在 `dropped/<prefix>/`（Megatron 格式，附旧编号）。恢复就是把 `original_idx/` 里的文件拷回去。
- 校验（`scripts/rtq_verify.py`）：529 个 prefix 全部通过。新 idx 等于原 idx 去掉名单内的文档；被删文档逐 token 和原 `data.bin` 一致；Megatron `IndexedDataset` 能读全部 529 个 prefix。
- 和 3.4 的区别：3.4 把留下的文档重新写成新的 `data.bin` 和 `data.idx`，被删的 token 物理上已经不在了；3.5 只改 `data.idx`。对读数据的一方效果相同：被删文档不存在，后面的编号前移。
- **编号前移**：这 529 个 prefix 里，排在被删文档之后的文档编号都变了。按旧编号建的东西（teacher cache 的 plan 和 unit、旧的下标名单）都不能再用。

#### 3.5.4 teacher cache 怎么跳过这些文档

producer 里没有任何"跳过"的逻辑，也不需要：被删的文档已经不在 `data.idx` 里，teacher cache 根本看不到它们。

1. **plan 按删完的 idx 写**：`pretrain/tc_full/plan_full_block32768.json` 在 11-01 23:04 写成，晚于删除（21:51–22:12）。它按每个 prefix 当前 `data.idx` 的文档数，把文档按约 1e8 token 一段切成 unit，每个 unit 是一段新编号的区间 [doc_lo, doc_hi)。
2. **按 idx 读文档**：producer 处理一个 unit 时，按 `data.idx` 逐篇读第 doc_lo 到第 doc_hi − 1 篇。被删文档没有记录，不会被读到，不会发请求，也不会有行；`data.bin` 里不再被引用的那些字节永远不会被读。
3. **编号和训练一致**：cache 的 `docs.idx` 记的是新编号，训练读的也是同一份 `data.idx`，所以训练里的第 d 篇就是 cache 里的第 d 篇，5.3 的取行规则不变。每篇的采样随机数也按新编号算（`sha256(blend_hash | prefix_key | doc_idx | seed)`）。
4. **删除前的那次试跑作废**：11-01 17:14–21:50 按删除前的编号跑过一次（86.4B 行），已改名为 `pretrain/tc_full_20261101_stopped`，不续跑，也不和现在的 cache 混用。
5. **核对**：
   - plan 覆盖全部 1,975 个 prefix，区间连续，每个 prefix 的 doc_hi 等于当前 `data.idx` 的文档数；合计 147,131 个 unit、14,661,092,439,601 token，等于删除后的 `summary.json`。
   - 11-02 复核：已完成的 487 个 unit（25.0B 行）和当前语料逐篇对齐，0 个问题。
6. **之后语料不能再改**：生产用的 producer（9b2dd41）不记录语料指纹。`data.idx` 再改一次，编号又会前移，已经写好的 cache 却发现不了。所以语料目录保持只读（文件 444、目录 555），plan 也是只读的（sha256 `0bb609b9…`）。如果还要删文档，必须重写 plan、重做 cache。
7. 80B 验证语料（`ultra_v3_phase1_80e9`）和 tc_80b 没有做 3.5，只做了 3.4。

## 4. teacher cache：按文档存，按窗口块打分

格式就是 3p5 最终计划（[2026-10-29_dense_pretrain_teacher_cache_v3p5_final](2026-10-29_dense_pretrain_teacher_cache_v3p5_final.html)）的 v2.2：按文档存，每篇文档只算一次。唯一的变化是窗口：按 4.1 的规则切块，每块单独打分，没有 overlap，teacher 不多看任何上下文。

### 4.1 窗口切块规则

窗口 W = 32768。切块只在 teacher 打分（4.4）和训练装箱（第 5 章）时按这条规则做，语料本身不切。

```
doc tokens          | 0 ... 32767           | 32768 ... 65535       | 65536 ... L-1 (EOD)
chunk               | k = 0                 | k = 1                 | k = 2 (last)
teacher request     | tokens 0 .. 32768     | tokens 32768 .. 65536 | tokens 65536 .. L-1
rows kept           | 0 .. 32767            | 32768 .. 65535        | 65536 .. L-2
context of row t    | 0 .. t                | 32768 .. t            | 65536 .. t
```

- 长度 L ≤ W 的文档就是一块。
- L > W 的文档从第 0 个 token 开始，每 W 个 token 切一块：[0, W)、[W, 2W)、…，最后一块是余数。只有最后一块带 EOD。
- **块与块完全隔离**：每一块都从 position 0 开始，看不到前面的块。teacher 和学生都这样看，切点也完全相同。
- 训练时每块整块放进一个样本，不会再被切开（第 5 章）。

### 4.2 目录结构

```
pretrain/tc/<teacher>/<blend_hash>/<prefix_key>/  生产（15T）；45B 试跑用 pretrain/tc_small/
  [parts/<lo>-<hi>/]                              大 prefix 按文档区间分单元
    meta.json  docs.idx  writer.lock              单元元数据、每篇的行位置、写锁
    top64_id.bin                                  uint8   [n, 64, 3]
    top64_logp.bin                                f16     [n, 64]
    next_token_id_and_rank.bin                    uint32  [n]
    next_token_logp.bin                           f16     [n]
    off64_tail_logp.bin                           f16     [n]
    off64_tail_stats.bin                          f16     [n, 3]
    off64_sample_id.bin                           uint8   [n, 4, 3]
    off64_sample_logp.bin                         f16     [n, 4]
```

### 4.3 文件定义

n 是单元内所有文档的行数之和。一篇 L 个 token（含 EOD）的文档有 L−1 行，第 t 行是对第 t+1 个 token 的预测。

| 文件 | 形状 | dtype | 内容 |
|---|---|---|---|
| `top64_id.bin` | [n, 64, 3] | uint8 | teacher 概率最高的 64 个 id（uint24），降序 |
| `top64_logp.bin` | [n, 64] | float16 | 对应的 log 概率（T=1） |
| `next_token_id_and_rank.bin` | [n] | uint32 | `id<<8 \| rank`：id 是实际的下一个 token，rank 是它在 top-64 中的名次 0–63，不在 top-64 时为 255 |
| `next_token_logp.bin` | [n] | float16 | 实际下一个 token 的 log 概率 |
| `off64_tail_logp.bin` | [n] | float16 | top-64 以外的总概率取 log |
| `off64_tail_stats.bin` | [n, 3] | float16 | top-64 以外：log(−Σ p log p)、log Σ p²、log Σ √p |
| `off64_sample_id.bin` | [n, 4, 3] | uint8 | 从 top-64 以外的分布有放回采样的 4 个 id（uint24） |
| `off64_sample_logp.bin` | [n, 4] | float16 | 这 4 个 id 的 log 概率 |
| `docs.idx` | [文档数, 3] | int64 | 每篇 `(doc_idx, row_start, n_rows)`，第 t 行在 `row_start + t` |
| `meta.json` | — | JSON | 格式版本、teacher、k、m、随机数规则、vocab_limit = 151669、**window = W、window_context = 0**、行数、每个文件的字节数和校验和 |

各字段的计算方法、uint24 和 `id<<8 | rank` 的拼法、采样规则，都和 3p5 最终计划相同。

### 4.4 每块怎么打分

- 第 k 块的请求是这一块的全部 token，再加下一块的第 0 个 token（只当 label 用）。这样块的最后一个 token 也有一行预测，每篇文档正好 L−1 行，行的排布和 v2.2 完全一样。
- 第 t 行的上下文是从本块开头到第 t 个 token，不含前面的块。这正是学生在第 5 章的样本里看到的上下文。
- 最后一块的请求就是这一块，以 EOD 结尾；EOD 没有对应的行。
- 每个请求最长 W + 1 = 8,193 个 token，`--max-model-len` 设为 8,193 以上。

### 4.5 各字段的计算精度

| 存储文件 | 用什么精度算 | 用什么精度存 |
|---|---|---|
| `top64_id` / `top64_logp` | float32（vLLM 的 log_softmax） | uint24 / float16 |
| `off64_tail_logp`：tail 总质量 | **float32** | float16 |
| `off64_tail_stats`：三个统计量 | **float32** | float16 |
| `off64_sample_id`：抽哪 4 个 token | 每组的质量用 float32 算；只有“沿着累加值找落点”那两小步用 float64 | uint24 |

- **float32 足够**：相对误差约 1e-7，而存成 float16 本身就有约 1e-3 的误差。原来对整个词表（约 15 万个数）做的 float64 运算已经全部换成 float32：GB300 的 float64 很弱，每 512 行从 9.0 ms 降到 3.9 ms，抽样结果和改之前一致（提交 6006c71、95000fd）。
- **“沿着累加值找落点”是什么（逆 CDF 采样）**：把 top-64 以外的 token 排成一把尺子，每个 token 占的长度等于它的概率；生成一个随机数，在尺子上扎一个点，从起点一段一段把长度加上去，加到超过这个点时，点所在的那一段就是被抽中的 token。为了快，分两级：每 1,024 个 token 一组、共 149 组，先在 149 个组里累加找到点落在哪一组，再在这一组的 1,024 个 token 里累加找到具体的 token。
- **这两步为什么用 float64**：累加的总和越来越大，float32 只有约 7 位有效数字，概率比总和小一千万倍以上的 token 会被舍入成 0、永远抽不到；float64 能保住它们。每次只加 149 个和 1,024 个数，用 float64 也不慢。采样用的随机数同样是 float64（53 位）。

## 5. 训练端：min padding 装箱

> **学生训练必须开 docmask：样本里每篇文档只看得到自己、position 从 0 开始；teacher 就是这样逐块打分的，不开 docmask，学生和 teacher 看到的上下文就对不上。**

训练端在另一个仓库实现，这里只写要求。

### 5.1 怎么组样本

1. **单位是块**：4.1 切出的每一块是一个单位，长度不超过 W。
2. **每个 epoch 重新抽**：按 blend 权重从各 prefix 抽块；需要重复的 prefix，每一遍都重新抽、重新装箱，组合每遍都不同。cache 按文档存，每遍都复用。
3. **min padding 装箱**：
   1. 维持一个 buffer，里面至少有 2 × W 个 token 的待装块，越大越好。
   2. 取 buffer 里最早放进来的块，放在样本的第 0 位。
   3. 在 buffer 里找放得进剩余空间的最长的一块，接在后面；一样长时取先放进来的。重复，直到放不下为止。
   4. 剩下的空位补 pad。块绝不切开：buffer 里只剩放不下的长块时，宁可补 pad。
4. **开 docmask**：每块只看自己，position 从 0 开始；pad 不算 loss。

### 5.2 每个位置的 label 和 loss

| 位置 | label | loss |
|---|---|---|
| 块内（不是块的最后一个 token） | 块内的下一个 token | CE + KD（cache 第 t 行） |
| 文档最后一块的 EOD | 样本里下一块的第 0 个 token | 只算 CE（没有 teacher 预测） |
| 中间块的最后一个 token | 真正的下一个 token 在下一块里，不在这个样本 | 不算 loss |
| pad | — | 不算 loss |

### 5.3 从 cache 取 KD 的行

文档 d 的第 k 块覆盖第 kW 到 min((k+1)W, L) − 1 个 token。块内第 j 位就是文档的第 t = kW + j 个 token，它的 teacher 行在 `row_start(d) + t`（`docs.idx`）。

### 5.4 待确认

| # | 问题 | 建议 |
|---|---|---|
| 1 | buffer 大小 | 至少 2 × W 个 token，越大 padding 越少 |
| 2 | 文档最后一块的 EOD 要不要算 CE | 算（和 Megatron 一样） |
