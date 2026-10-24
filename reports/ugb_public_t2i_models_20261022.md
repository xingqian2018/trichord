# UGB 对比所用公开 T2I 模型调研（2026-10-22）

本报告汇总在 UniGenBench（UGB）上跑过的对比模型：模型规模、语言塔（text encoder）、视觉塔（diffusion transformer + VAE）、是否依赖 prompt upsampler、prompt 格式，以及我们实际的推理设置和出图状态。

- 开源模型的参数量是从 HF 上 safetensors 文件头逐 tensor 求和得到的精确值，不是按文件大小估算；架构字段取自各 repo 的 `config.json` / `model_index.json`。
- 闭源 API 模型（GPT-image、Gemini image）不公开架构，表中标"未公开"。

---

## 1. 总览

| 模型 | 来源 | DiT 参数 | 语言塔 | VAE | 总参数 | 官方要求 upsampler | Prompt 格式 | License |
|---|---|---|---|---|---|---|---|---|
| **Krea 2 Turbo** | `krea/Krea-2-Turbo` | 12.82B | Qwen3-VL-4B（4.44B） | `AutoencoderKLQwenImage`，16ch（0.13B） | ~17.4B | 否 | 自然语言 | krea-2-community-license |
| **FLUX.3 klein** | `black-forest-labs/flux-3-image-klein` | 5.55B | Qwen3-VL-4B（4.44B） | `AutoencoderKLFlux2`，32ch（0.08B） | ~10.1B | **是**（必须） | 完整场景描述（自然语言，可附 bbox JSON） | flux-kommunity-license（非商用） |
| **FLUX.3 klein-base** | `black-forest-labs/flux-3-image-klein-base` | 5.55B | Qwen3-VL-4B（4.44B） | `AutoencoderKLFlux2`，32ch | ~10.1B | **是**（必须） | 同 klein | flux-kommunity-license（非商用） |
| **Ideogram 4** | `ideogram-ai/ideogram-4-nf4-diffusers` | 9.28B × 2（条件 + 无条件两个 DiT） | Qwen3-VL-8B（8.15B） | `AutoencoderKLFlux2`，32ch（0.08B） | ~26.9B | **是**（必须） | 结构化 JSON caption | ideogram-4-non-commercial |
| **Fibo 2.0 dev** | `briaai/Fibo-2.0-dev` | 8.13B | Qwen3-VL-4B（4.44B） | `AutoencoderKLFlux2`，32ch（0.08B） | ~12.6B | **是**（必须） | 结构化 VGL JSON | bria-fibo |
| GPT-image-2 | NVIDIA gateway `openai/openai/gpt-image-2` | 未公开 | 未公开 | 未公开 | 未公开 | 不适用（API 内部处理） | 自然语言 | 闭源 API |
| GPT-image-2.5 sunburst | NVIDIA gateway `openai/openai/gpt-image-2.5-sunburst` | 未公开 | 未公开 | 未公开 | 未公开 | 不适用 | 自然语言 | 闭源 API |
| Gemini 3.1 flash image | NVIDIA gateway `gcp/google/gemini-3.1-flash-image` | 未公开 | 未公开 | 未公开 | 未公开 | 不适用 | 自然语言（chat） | 闭源 API |

参考（我们自己的模型）：

| 模型 | 生成部分 | 语言 / 理解部分 | VAE | 结构 |
|---|---|---|---|---|
| Cosmos3.5 Super（`60bm30b`） | 30B-A3B MoE 生成专家 | Cosmos3.5-Super-Reasoner（Qwen3-VL-30B-A3B 系） | Wan2.2，4×16×16，48ch | MoT：生成专家与 reasoner 共享 attention，没有单独的 text encoder |
| Cosmos3.5 Ultra（`470bm235b`） | 235B MoE 生成专家 | Cosmos3.5-Ultra-Reasoner | Wan2.2，48ch | 同上 |

几个共性：

- 四个开源模型的语言塔全部是 **Qwen3-VL**（Krea / FLUX.3 / Fibo 用 4B，Ideogram 用 8B）。没有一个用 T5 或 CLIP。
- 三个模型（FLUX.3 / Ideogram / Fibo）用的是 Flux2 的 32 通道 VAE；Krea 用 QwenImage 的 16 通道 VAE。
- 都从 Qwen3-VL 的**多个中间层**取特征再拼接，而不是只取最后一层（见第 2 节）。

---

## 2. 架构细节

### Krea 2 Turbo
- **DiT**：`Krea2Transformer2DModel`，28 层，48 头 × 128 维（hidden 6144），GQA（12 个 KV 头），FFN 16384。
- **文本条件**：Qwen3-VL-4B（`Qwen3VLModel`，36 层，hidden 2560），从 12 层（2, 5, 8, …, 35）取 hidden states；DiT 内另有 2 个 text refiner block 和 2 个 layerwise text block。
- **蒸馏版**：官方推理 8 步，cfg 0（`is_distilled: true`）。

### FLUX.3 klein / klein-base
- **DiT**：`Flux3Transformer2DModel`，5 个 double-stream 层 + 28 个 single-stream 层，24 头 × 128 维。
- **文本条件**：Qwen3-VL-4B（`Qwen3VLForConditionalGeneration`），`joint_attention_dim` = 20480 = 8 × 2560，推测拼接了 8 层 hidden states（未在官方文档中确认）。
- **klein**：蒸馏版，16 步、无 guidance，是 BFL API 背后的 checkpoint。
- **klein-base**：未蒸馏版，50 步、cfg 3.5，官方定位是用于 fine-tune / 研究。
- 两者 DiT 规模相同（5.55B），是这批开源模型里最小的。FLUX.3 没有放出更大的开源权重；目前最大的开源 FLUX 是 **FLUX.2 dev（32B）**。

### Ideogram 4
- **DiT**：`Ideogram4Transformer2DModel`，34 层**全 single-stream**（文本与图像 token 拼成一个序列），18 头 × 256 维。
- **双分支 CFG**：条件和无条件各有一个完整的 9.28B DiT，这是总参数接近 27B 的原因。官方 README 把模型规模写作 9.3B。
- **文本条件**：Qwen3-VL-8B（hidden 4096），取 13 个中间层拼接（`llm_features_dim` = 53248 = 13 × 4096）。
- **权重格式**：fp8 版官方标注"不支持 diffusers"，所以我们用的是 **nf4（bitsandbytes 4-bit）diffusers 版**，质量可能略低于 fp8 / bf16。

### Fibo 2.0 dev
- **DiT**：`BriaFibo2Transformer2DModel`，44 层，dim 3584，28 头，patch 2。
- **文本条件**：Qwen3-VL-4B，`cap_feat_dim` 7680（= 3 × 2560）；文本特征在第 6/14/22/30/38 层注入；用 perceiver（256–512 个 gist token）压缩长 caption（支持到 2048 token）。
- 官方定位是"完整未蒸馏的 base 模型"，推荐 50 步、cfg 5，并配一个结构化 JSON 形式的 negative prompt。

---

## 3. Prompt upsampler 与 prompt 格式

| 模型 | 官方 upsampler | 我们实际用的 | 与官方的差异 | 输出格式 |
|---|---|---|---|---|
| Krea 2 Turbo | 无 | 无，原版 prompt 直接喂 | 无 | 原版自然语言 |
| FLUX.3 klein / klein-base | `flux-3-image-klein-pup`（Qwen3.5，18.8GB，本地跑） | **Claude Opus 4.8**（NVIDIA gateway）+ BFL 官方 `system_prompts["t2i"]`，temperature 0，max_tokens 3584 | LLM 不同。BFL 明确允许"任何 LLM + 官方 system prompt"替代 pup | 约 1.3–1.8k 字符的完整场景描述 |
| Ideogram 4 | Ideogram 托管的 magic-prompt API（需 `IDEOGRAM_API_KEY`），或开源 system prompt + Claude Opus 4.8（OpenRouter） | **Claude Opus 4.8**（NVIDIA gateway）+ 官方开源 `v1.txt` system prompt，temperature 1.0，`strip_aspect_ratio_and_bboxes` 后处理 | 走的是官方测试过的"开源 prompt + Opus 4.8"路径，不是托管 API；官方注明二者不完全相同 | JSON：`high_level_description`, `compositional_deconstruction` |
| Fibo 2.0 dev | `bria-vgl` 包，默认 `gemini/gemini-3.5-flash-lite` | **bria-vgl** + gateway 上的 `gcp/google/gemini-3.5-flash-lite`（经 litellm OpenAI 兼容接口） | 模型相同、调用路径不同。gateway 不生效 `responseSchema`（结构化输出约束），导致约 9% 的回复 JSON 不合法（见第 5 节） | VGL JSON：`short_description`, `objects`, `background_setting`, `lighting`, `aesthetics`, `photographic_characteristics`, `style_medium`, `context`, `artistic_style` |
| GPT-image / Gemini image | API 内部处理 | 原版 prompt 直接喂 | — | 自然语言 |

要点：

- **三个模型对 upsampler 有强依赖**，官方原话分别是：FLUX.3 *"Do not use the model without a prompt upsampler"*；Ideogram 只用结构化 JSON 训练，plain text *"won't perform as well"*；Fibo 只读结构化 VGL caption。
- 因此这批对比**不是严格的"同一 prompt 比生成能力"**。FLUX.3 / Ideogram / Fibo 实际上是"upsampler LLM + 生成模型"的系统级对比，结果中混入了 Claude Opus 4.8 / Gemini 3.5 flash-lite 的改写能力。
- Krea 和 API 模型吃的是原版短 prompt。

---

## 4. 推理设置（v2_1170L，原版 UGB prompt）

| 模型 | 分辨率 | 步数 | guidance | 其他 | 资源 | 每张耗时（冒烟测试） | 显存峰值 |
|---|---|---|---|---|---|---|---|
| Krea 2 Turbo | 1024² | 8 | 0 | — | 1 节点（4×GB200） | ~4s | 36.6 GB |
| FLUX.3 klein | 1024² | 16（默认） | 无 | — | 2 节点 | ~27s（含 upsample） | 38.8 GB |
| FLUX.3 klein-base | 1024² | 50（默认） | 3.5（默认） | — | 4 节点 | ~6× klein | — |
| Ideogram 4 nf4 | 1024² | 48（pipeline 默认） | 双分支 CFG（默认） | — | 2 节点 | ~40s（含 upsample） | 19.2 GB |
| Fibo 2.0 dev | 1024² | 50 | 5.0 | 官方 JSON negative prompt、官方字体 | 2 节点 | ~22s（含 upsample） | 26.3 GB |
| GPT-image-2 / 2.5 sunburst | 1024² | — | — | `quality` 见下 | API | 14s（low）/ 45s（high） | — |
| Gemini 3.1 flash image | 1024²（`1k`, `1:1`） | — | — | — | API | ~9s | — |

- **GPT-image 的 quality 坑**：不传 `quality` 时 gateway 默认 `low`（196 image tokens）；`quality="high"` 是 1756 tokens。原始的 `gpt_image_2/` 和 `gpt_image_2p5_sunburst/` 两组是 low，`*_quality_high/` 两组是 high。
- **`thinking_level`**：`images/generations` 接口不接受 `thinking_level` / `reasoning_effort` 等参数（400 Unknown parameter），所以 GPT-image 只有 `quality` 可调。
- **环境**：四个开源模型需要 diffusers 0.42 dev 和 transformers 5.x，与 i4 主环境（transformers 4.57）冲突，所以在 gcpcode 上单独建了 venv：`~/Project/ugb_newmodels/.venv`，`--system-site-packages`，继承容器的 torch。

---

## 5. 出图状态（截至 2026-10-22 晚）

所有路径的前缀都是 `s3://nv-00-10206-vfm/debug/xingqianx/evaluation_results/unigenbench_for_cosmos3p5/v2_1170L/`

| 模型 | 子目录 | 已出图 / 1170 | 状态 | 缺图原因 |
|---|---|---|---|---|
| Krea 2 Turbo | `krea2_turbo/` | 1170 | 完成 | — |
| FLUX.3 klein | `flux3_klein/` | 1167 | 基本完成 | `phi128`, `phi192`, `phi32`：Claude 连续 5 次返回空内容（疑似拒绝改写） |
| FLUX.3 klein-base | `flux3_klein_base/` | 128 | 运行中（job 2616209） | — |
| Ideogram 4 nf4 | `ideogram4_nf4/` | 1166 | 运行中（job 2615553），基本完成 | — |
| Fibo 2.0 dev | `fibo2_dev/` | 1064 | 完成，**缺 106 张** | Gemini 回复的 JSON 不合法；seed 固定，重试三轮都复现同样的错误。修复方案（换 seed 重试）待确认 |
| GPT-image-2，low | `gpt_image_2/` | 1168 | 完成 | 安全过滤 |
| GPT-image-2，high | `gpt_image_2_quality_high/` | 1166 | 完成 | 安全过滤 |
| GPT-image-2.5 sunburst，low | `gpt_image_2p5_sunburst/` | 1167 | 完成 | 安全过滤 |
| GPT-image-2.5 sunburst，high | `gpt_image_2p5_sunburst_quality_high/` | 1165 | 完成 | 安全过滤 |
| Gemini 3.1 flash image | `gemini_3_1_flash_image/` | 1169 | 完成 | `orig564`：安全过滤 |

已知问题：`inference_ugb_i4.py` 的重试逻辑在"部分 rank 有失败 prompt"时会在 `dist.barrier()` / `all_gather_object` 之间死锁，job 卡 30 分钟后被 NCCL watchdog 杀掉，显示 FAILED。已生成的图不受影响，但 job 状态和 `completed.txt` 不可靠，以 S3 上的实际图数为准。

---

## 6. 观察与建议

1. **规模不对等**：开源模型的 DiT 从 5.6B（FLUX.3 klein）到 12.8B（Krea 2），Ideogram 算上双 DiT 为 18.6B；Cosmos3.5 Super 的生成专家是 30B-A3B MoE。比较时应注明规模差异。
2. **upsampler 是混杂变量**：FLUX.3 / Ideogram / Fibo 的分数含 LLM 改写的贡献。如果想单独比较生成模型，可以把同一份 dense prompt（例如 `v2_1170L_gpt5p5_cosmos3p5_dense_hamid`）直接喂给 FLUX.3 和 Krea；Ideogram 和 Fibo 必须用 JSON，没法这样做。
3. **FLUX.3 想更贴近官方**：venv 已经升级到 transformers 5.x，可以换回官方 pup（Qwen3.5）重跑。如果想测更大的 FLUX，可以跑 FLUX.2 dev（32B）。
4. **Ideogram 质量上限**：目前是 nf4 量化。fp8 或 bf16 需要官方 `ideogram4` 推理代码（不走 diffusers）。
5. **Fibo 缺图**：建议在 upsampler 解析失败时换 seed 重试，补跑 106 张后再评分。
