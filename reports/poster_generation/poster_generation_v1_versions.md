# Agentic 海报数据生成：v1 / v1p1 / v1p2 / v1p3 / v1p4 / v1p5 版本说明

更新时间：2026-10-21（产量汇总）
代码位置：`scripts/model_playground_v1/`
输出位置：`gcs:nv-00-10206-vfm/debug/xingqianx/agentic_data/`

## 一句话区别

| 版本 | 相对上一版的变化 |
|---|---|
| v1 | 在原版 agent 基础上，把生图工具换成自建的 Cosmos3 文生图服务 |
| v1p1 | 代码与 v1 相同，修正运行环境：补齐凭据、加长 Cosmos3 超时、删除提示词里的示例文件名、每个任务独立 `/tmp` |
| v1p2 | 主 agent 和 critic 改用 Slurm 上自建的 Kimi K3 服务池（`kimi-k3@k3slurm`），不再走 NVIDIA gateway；同时关闭 Cosmos3 安全过滤 |
| v1p3 | 在 v1p2 上加：上下文超长时自动切回 NVIDIA gateway、critic 超时放宽、工具参数错误交还模型重试、渲染改为 1024 档 |
| v1p4 | 代码与 v1p3 相同，Kimi K3 服务换成我们自己部署的副本（上下文上限 1048576），输出到新文件夹 |
| v1p5 | 生图从 Lepton 换成我们自己部署的 Cosmos3 副本（同一个模型），不再调用 Lepton；没有可用副本时等待而不是让样本失败 |
| v2 | 生图前先由独立的 Kimi K3 upsampler 把简短需求扩写成 Cosmos3 官方结构化 JSON（`upsample_prompt` 工具），再用 exp009c（x9c）权重出图 |

## 共同部分

四个版本都是同一个 agentic 流程：Kimi K3 阅读题目，写 HTML 海报，调用生图工具生成素材，用 Chrome 渲染成 PNG，再请 critic（也是 Kimi K3）评审，按意见修改后输出。

- **生图模型：** 都是 `LeptonTR/Cosmos3-Super-Text2Image`，部署在 Lepton。对模型暴露的工具名仍是 `generate_image`，对话里不出现 Cosmos3 字样。
- **Cosmos3 安全过滤：** v1、v1p1 开启，从 v1p2 起关闭，详见下一节。
- **生图分辨率：** Cosmos3 用 1024 档（1:1 为 1024×1024，3:4 为 880×1184 等）。在 2048 档下它会把同一张海报重复平铺、文字乱码，所以只在工具层改成 1024 档。
- **taxonomy：** 旧版 `poster_topics`（500 个题目）。
- **入口：** `script/run_poster_generation_<版本>.py`，都复用 `script/run_poster_generation.py` 的主循环，只换 agent 类。
- **存储格式：** 每个样本一个 `<run_id>/` 文件夹，含 `setting.json`、`chat.json`（原始请求格式）、`derived/`（最终 HTML 与海报）、`tool/<工具名>/call/<tool_call_id>/`（每次工具调用的参数、结果和文件）。失败的样本不保存。

## Cosmos3 安全过滤时间线

| 项目 | 内容 |
|---|---|
| 代码改动 | `tool/generate_image_cosmos3.py` 新增 `COSMOS3_EXTRA_PARAMS = {"guardrails": False}`，随每个生图请求以 `extra_params` 发送 |
| 提交 | commit `0257d42` "c3 guardrail removal"，2026-10-09 15:46 |
| 最后一次被过滤拒绝 | 2026-10-09 14:42，v1p1 的任务 905867 |
| 第一个关闭过滤的任务 | v1p2 的 911772，2026-10-09 15:50 开始，提交后 4 分钟 |

各任务日志里被安全过滤拒绝的次数（只有 rank 0 会打印警告，所以是部分统计）：

| 任务 | 版本 | 开始时间 | 被过滤拒绝 | 生图失败 |
|---|---|---|---|---|
| 895867 | v1 | 10-08 23:13 | 4 | 6 |
| 896448 | v1p1 | 10-08 23:41 | 128 | 183 |
| 905867 | v1p1 | 10-09 10:39 | 54 | 81 |
| 911772 | v1p2 | 10-09 15:55 | 0 | 0 |
| 934616 | v1p3 | 10-10 19:58 | 0 | 0 |

影响：
- **数据：** v1、v1p1 的样本是在过滤开启时生成的，v1p2、v1p3 是关闭时生成的。
- **失败率：** v1p1 约 35% 的失败几乎全部来自过滤拒绝，关闭后 v1p2 的失败中没有一次是过滤造成的。
- **复现：** `generate_image_cosmos3.py` 为各版本共用，现在重跑 v1 或 v1p1 也会是关闭状态，与当时生成的数据条件不同。

## 各版本详情

### v1

- **agent：** `tool/poster_generation_v1.py`，在原版 `poster_generation` 上只把生图工具换成 `generate_image_cosmos3`。
- **模型：** 主 agent 与 critic 都是 `kimi-k3@nvidiak`（NVIDIA gateway）。
- **最终海报：** 渲染为 2048 档。
- **输出：** `poster_generation_v1_kimi_cosmos3_kimi/`
- **运行情况：**
  - 895274：gb300 的凭据文件缺少 Cosmos3 的 key，每个样本在第一次生图时报 `KeyError`，已取消。
  - 895867：补上 key 后跑了约 14 分钟即取消。期间发现 32 个任务共用一个 `/tmp`，生成的文件会互相撞名甚至被覆盖。
- **结论：** 只用于打通流程，S3 上有 76 个样本。

### v1p1

- **代码：** 与 v1 相同（同一个入口和 agent），改动都在运行环境和公共代码：
  - gb300 的 `credentials/gateway.json` 补上 `LEPTON_COSMOS3_SUPER_T2I`。
  - Cosmos3 生图超时从 300 秒加到 1800 秒（多任务同时请求时会排队）。
  - 删除提示词和工具说明里的示例文件名 `poster_v1.png`、`hero_scene.png`，避免所有 agent 选同一个名字。
  - 新增 `script/isolated_tmp.sh`：同一节点的任务共用一个容器，这个脚本给每个任务挂一个私有的内存 `/tmp`。
  - QOS 从 `cpu-long`（每人 1 个节点）改为 `cpu-normal`（每人 2 个节点，最长 1 天）。
- **输出：** `poster_generation_v1p1_kimi_cosmos3_kimi/`
- **运行情况：**

| 任务 | 规模 | 时长 | 结束原因 | 成功 | 失败 |
|---|---|---|---|---|---|
| 896448 | 1 节点 × 32 任务 | 9 小时 20 分 | 内存溢出（89 次 OOM） | 3,697 | 1,963 |
| 905867 | 2 节点 × 16 任务 | 4 小时 9 分 | 手动取消 | 1,707 | 913 |

- **主要问题：**
  - S3 上原有 5,565 个样本（两次运行合计）；其中 1,088 个因人脸被安全过滤打码已永久删除，现存 4,477 个（见"安全过滤打码样本清理"）。
  - 失败率约 35%，几乎都是 Cosmos3 安全过滤拒绝（"Input was blocked by Cosmos3 guardrails"）。生图失败会让整个样本作废，前面的 Kimi 调用也白费。
  - 失败样本不保存，它产生的文件留在 `/tmp`；而隔离后 `/tmp` 在内存里，持续累积。这是内存增长的原因之一，但估算只有约 20 GB，不足以解释 350 GB 的溢出，其余原因未查明。

### v1p2

- **代码：** `tool/poster_generation_v1p2.py` + `gateway_kimik3.py` + `k3_endpoints.py`。
  - Cosmos3 安全过滤关闭（见上文"共同部分"）。
  - 主 agent 与 critic 都改用 `kimi-k3@k3slurm`：请求发到 Slurm 上 `sdharur` 的 `k3-scale2048` Kimi K3 服务池，不再消耗 NVIDIA gateway 额度。
  - 批量任务所在的主机每 5 分钟用 `squeue` 扫描一次服务列表，写入 `$K3_ENDPOINTS_FILE`；容器内没有 `squeue`，由 gateway 读取这个文件。
  - 选服务的方式是打乱后轮流使用；某个服务连不上时冷却 120 秒，换下一个。
- **输出：** S3 上实际写入 `poster_generation_v1p2_kimi_cosmos3_kimi/`（脚本头部示例命令里写的是 `..._k3slurm_cosmos3_k3slurm`，两者不一致）。
- **运行情况：** 911772，2 节点 × 16 任务，跑满 1 天时限。S3 上 4,755 个样本；日志统计失败 1,146（约 20%），没有一次是安全过滤造成的。
- **主要问题：**
  - **慢：** 平均每个样本约 7.5 分钟，是 v1p1 的约 2.4 倍。服务池同时处理约 3.5 万个请求，每个服务 140 到 256 个，单个请求生成很慢。轮流选服务又不看负载（约 30 个空闲服务用不上），并且同一个 agent 的每一步换一个服务，服务端的前缀缓存用不上。
  - **失败：** 主要是对话超过服务的最大上下文长度（400 错误）；另有工具参数缺失（`KeyError: 'html'`）、critic 拿到不存在的图片路径（`FileNotFoundError`）等本可由模型自行修正的错误，却让整个样本失败。
  - 服务列表本身是准确的：255 个地址全部可用，与实际运行的服务一致，8 小时内只有 3 个服务停止。

### v1p3

- **代码：** `tool/poster_generation_v1p3.py` + `gateway_kimik3_fallback.py`，在 v1p2 上增加：
  1. **上下文超长回退：** `k3slurm` 返回 "maximum context length" 时，这个 agent 之后的请求都改走 `kimi-k3@nvidiak`。回退次数记在 `setting.json` 的 `fallback_requests`。
  2. **critic 放宽：** 超时 600 秒、2 次尝试、输出上限 16384 tokens（原来是 180 秒、1 次、4096）。
  3. **工具错误可恢复：** 缺少必填参数、critic 的图片路径不存在时，作为 `ToolCallError` 交还模型重试，不再让样本失败。
  4. **渲染改为 1024 档：** `render_poster_v1p3` 使用 Cosmos3 的尺寸表，最终海报与素材分辨率一致（v1 到 v1p2 的最终海报是 2048 档）。
  - 规模加大到每节点 32 个任务，每个任务上限 625 个样本。
- **输出：** `poster_generation_v1p3_kimik3_cosmos3_kimik3/`
- **运行情况：**

| 任务 | 规模 | 时长 | 结束原因 | 成功 | 失败 |
|---|---|---|---|---|---|
| 934616 | 2 节点 × 32 任务 | 24 小时 | 到 1 天时限 | 7,404（S3 实数） | 约 99（日志统计） |
| 966996 | 2 节点 × 32 任务 | 11 分 46 秒 | 所有样本失败后退出（FAILED） | 0 | 全部失败 |

- **效果：** 失败率从 v1p2 的约 20% 降到约 1.3%，速度约每小时 310 个。
- **当前问题（2026-10-12）：** `sdharur` 的 `k3-scale2048` 服务池已全部下线（10-11 起约 253 个被取消、约 116 个失败），服务列表扫描结果为 0。回退逻辑只覆盖"上下文超长"，不覆盖"没有可用服务"，所以 966996 的每个样本都立即报 "no Kimi-K3 endpoint found" 并失败，约 12 分钟后用完配额退出，没有产出数据。由于没有服务可调用，也没有消耗 Kimi 额度。

### v1p4

- **代码：** 与 v1p3 完全相同（同一个入口 `script/run_poster_generation_v1p3.py`、同一个 agent `poster_generation_v1p3`），所以 `setting.json` 里记录的类名仍是 `poster_generation_v1p3`。区分 v1p4 靠输出文件夹。
- **改动只在启动命令：**
  - 服务扫描从 `sdharur` 的 `k3-scale2048` 改为我们自己的任务：两处 `k3_endpoints.py` 都加上 `--user xingqianx --job-prefix k3-xqx`。
  - 输出文件夹改为 `poster_generation_v1p4_kimik3_cosmos3_kimik3/`。
- **Kimi K3 服务：** `scripts/serving/serve_kimi_k3.sbatch`，在 gb300 上每个副本占 2 节点 × 4 GB300（TP8 / EP8 / DCP8）。
  - 模型权重、镜像和 vLLM 参数都沿用 `sdharur` 的配置，只把 `--max-model-len` 从 32768 提到模型上限 1048576。
  - 端口为 `20000 + jobid % 20000`，就绪后在 `run/k3_serve/job-<id>/endpoint.env` 写出地址。
- **上下文超长：** v1p2 的主要失败原因是服务只开了 32768。v1p4 的服务开到 1M，一个海报 agent 的对话远到不了这个长度，回退到 `kimi-k3@nvidiak` 的情况应该基本为 0，几乎不消耗 gateway 额度。可以用各样本 `setting.json` 的 `fallback_requests` 核对。
- **其他错误仍不走 gateway：** 连接失败、超时、500 会换下一个自建服务，出错的那个冷却 120 秒；所有服务都不可用时样本失败。所以必须在服务就绪后再启动。
- **Kimi 服务运行情况：**
  - 968459（batch，4 小时）用于验证：服务正常启动，`/v1/models` 显示 `max_model_len` 为 1048576，测试请求正常返回。
  - 971679、971680（batch_long，48 小时）是正式副本。10-12 19:27 两者同时被 Slurm 重新排队（requeue，原因未查明），之后在新节点上自动重启，到 10-13 已重启 3 次；每次换节点地址都会变，要等重新加载完才可用。
- **生成任务：**

| 任务 | 规模 | 时长 | 结束原因 | 成功 |
|---|---|---|---|---|
| 971940 | 2 节点 × 32 任务 | 11 小时 48 分（10-12 15:52 – 10-13 03:40） | Kimi 副本重启期间样本快速失败、配额耗尽（FAILED） | 2,100（S3 实数） |
| 986675 | 2 节点 × 32 任务 | 10-13 11:10 – 约 17:00 | 手动取消：服务全部换成带 key 的新版本，改跑 v1p5 | 953（S3 实数） |

- **971940 的经过：**
  - 15:52 – 19:27 约 100 个/小时，19:27 之后升到约 230 个/小时（期间在 Lepton 端加大了生图吞吐，说明瓶颈在生图）。两个 Kimi 副本始终很空：64 个 agent 同时只有约 10 个 Kimi 请求，没有排队。
  - 19:27 Kimi 副本被重新排队后，服务列表扫描为 0，旧地址全部失效。没有可用服务时样本几秒就失败（仅 rank 0 一个任务在 19 点就失败 617 次），多数任务很快用完 625 个样本的配额而退出；03:39 又一次扫描为 0，剩下的任务也随之结束。
  - 除此之外只有 1 个样本因 Cosmos3 返回 504 失败；没有上下文超长，也没有回退到 gateway。
- **遗留问题：** Kimi 这边"没有可用服务时样本立即失败"仍未修复（v1p5 只对生图做了等待处理），副本被重新排队时仍会损失配额。

### v1p5

- **为什么要新版本：** v1p4 的样本里生图很慢。两个 Kimi 副本始终很空（64 个 agent 同时只有约 10 个 Kimi 请求，没有排队），而在 Lepton 端加大生图吞吐后，v1p4 的产量从约 100 个/小时涨到约 230 个/小时，说明瓶颈在 Lepton 上的 Cosmos3。所以和 Kimi 一样，把 Cosmos3 也部署到我们自己的 GPU 节点上。
- **相对 v1p4 只变了生图：**

| 项目 | v1p4 | v1p5 |
|---|---|---|
| 生图工具类 | `generate_image_cosmos3_v1p3` | `generate_image_cosmos3_slurm_v1p5` |
| 生图服务 | Lepton（`lepton_cosmos3_t2i`） | 自建副本 `c3-xqx*`，完全不调用 Lepton |
| 模型 | `Cosmos3-Super-Text2Image` | 同一个模型，HF `nvidia/Cosmos3-Super-Text2Image`（版本 `daf3d374`） |
| 服务端 | Lepton 上的 vLLM-Omni | vLLM-Omni 镜像 `vllm/vllm-omni:cosmos3-arm64`，参数与 Lepton 相同 |
| 对模型暴露的工具名 | `generate_image` | `generate_image`（不变，对话中看不出服务来源） |
| 分辨率、guardrail | 1024 档，关闭 | 不变 |
| 没有可用生图服务时 | 报错，样本失败 | 每 30 秒重读服务列表等待，最长到工具超时（1800 秒 × 2 次） |
| Kimi K3、渲染、critic | 自建 Kimi、1024 档渲染、critic 600 秒 × 2 | 不变 |

- **代码：**
  - `tool/generate_image_cosmos3_slurm.py`：新生图工具。继承 `generate_image_cosmos3`，只换掉发请求的那一层：在 `C3_ENDPOINTS_FILE` 列出的副本之间随机选择，出错的副本冷却 120 秒后换下一个。
  - `tool/poster_generation_v1p5.py`：继承 `poster_generation_v1p3`，只把工具列表里的生图工具换成上面这个（保留 v1p3 的必填参数检查）。
  - `script/run_poster_generation_v1p5.py`：入口，文件头是完整的启动命令。
  - `k3_endpoints.py`：新增 `--model`（Cosmos3 副本按 `LeptonTR/Cosmos3-Super-Text2Image` 探活）和 `--api-key-name`（从 `credentials/gateway.json` 读取 key）两个参数，默认值不变。另外修了一个问题：只扫一次时扫到 0 个服务会返回 1，启动命令里的 `|| exit 1` 从此生效（v1p2 到 v1p4 的同一检查此前从未生效）。
- **服务：** `scripts/serving/serve_cosmos3_super_t2i.sbatch`，每个副本 1 个节点 × 4 GB300（CFG 并行 2 × Ulysses 2，HSDP 分片 4）。
  - 132.5 GB 权重放在团队共享的 HF 缓存里，第一次运行时自动下载。
  - 加了 API key：所有请求都必须带 key（`credentials/gateway.json` 里的 `SLURM_COSMOS3_API_KEY`），别人拿到地址也调不了。
  - 统一用 `bash scripts/serving/serve.sh <服务名> <副本数> [batch|batch_long]` 启动，用 `bash scripts/serving/serve.sh status` 查看。
- **启动命令相对 v1p4 的变化：** 多了一组 Cosmos3 扫描（启动时扫一次，之后每 5 分钟刷新），`C3_ENDPOINTS_FILE` 通过 `--container-env` 传进容器；任务名 `poster_gen_v1p5`。
- **输出：** `poster_generation_v1p5_kimik3_cosmos3_kimik3/`
- **运行情况（2026-10-13）：** 尚未启动。需要先起 Cosmos3 副本并确认能正常出图；按今天下午的估计，batch 4 小时最早 10-14 00:49 开始。

## 产量汇总

统计时间：2026-10-21（v2 已于 10-21 01:36 停止）。包括本报告之前的 v0 和 lite v0。

| 版本 | 海报数 | 生成日期 | 生图 | S3 文件夹（`agentic_data/` 下） |
|---|---|---|---|---|
| v0 | 31,416 | 10-01 ~ 10-04 | nano-banana-2 | `poster_generation_kimi_nb2_kimi_v0/` |
| lite v0 | 19,930 | 10-08 ~ 10-09 | gpt-image-2（sunburst），整张海报一次生成，没有 agent 流程 | `poster_generation_lite_v0_gpt2p5sunburst/` |
| v1 | 68（原 76，移除 8） | 10-08 | Cosmos3（Lepton，安全过滤开） | `poster_generation_v1_kimi_cosmos3_kimi/` |
| v1p1 | 4,477（原 5,565，移除 1,088） | 10-08 ~ 10-09 | Cosmos3（Lepton，安全过滤开） | `poster_generation_v1p1_kimi_cosmos3_kimi/` |
| v1p2 | 4,755 | 10-09 ~ 10-10 | Cosmos3（Lepton） | `poster_generation_v1p2_kimi_cosmos3_kimi/` |
| v1p3 | 7,404 | 10-10 ~ 10-11 | Cosmos3（Lepton） | `poster_generation_v1p3_kimik3_cosmos3_kimik3/` |
| v1p4 | 10,896 | 10-12 ~ 10-15 | Cosmos3（Lepton） | `poster_generation_v1p4_kimik3_cosmos3_kimik3/` |
| v1p5 | 19,799 | 10-13 ~ 10-16 | Cosmos3（自建 Slurm 副本） | `poster_generation_v1p5_kimik3_cosmos3_kimik3/` |
| v2 | 10,810 | 10-20 ~ 10-21 | Cosmos3 exp009c（x9c）权重 + Kimi K3 prompt upsampling | `poster_generation_v2_kimik3_cosmos3x9cupsampled_kimik3/` |
| **合计** | **109,555** | | | |

其中 agent 流程生成的（v0、v1 ~ v2）共 89,625 个，lite 一次生成的共 19,930 个。

数量为 S3 上的样本文件夹数：每个文件夹是一个成功保存的样本（失败的样本不保存）。统计时只列一层目录，不展开文件；按文件夹名里的时间分小时并行列，几十秒即可完成。v0 的文件夹名带 `poster_generation_agent_` 前缀，其余版本以时间开头。"生成日期"按文件夹名里的开始时间。v1、v1p1 为 2026-10-14 删除打码样本之后的数量。

## 安全过滤打码样本清理（2026-10-14）

v1、v1p1 生成时 Cosmos3 的安全过滤是开启的，它会把检测到的人脸替换成马赛克。这些图片会进入最终海报，不适合作训练数据。

- **检查方法：** 把 v1、v1p1 中 Cosmos3 生成的全部原始图片（`tool/generate_image/call/*/*.png`，共 8,325 张）逐张交给我们自建的 Kimi K3，问"是否有人脸被像素马赛克遮挡"，只回答 PIXELATED 或 CLEAN。
- **结果：** 1,708 张被判为 PIXELATED（约 20.5%），涉及 1,096 个样本（v1 8 个、v1p1 1,088 个）。没有无法判定的图片。随机抽查 6 张被标记的图，全部确实有人脸马赛克；漏判率（CLEAN 里是否还有马赛克）未检查。
- **处理：** 1,096 个样本的完整轨迹（整个文件夹，共 32,085 个对象）先移入隔离目录、逐个核对后删除原位置，随后隔离目录也已永久删除（2026-10-14），S3 上不再保留。
- **记录：** 被标记图片清单在 gb300 的 `~/tmp/guardrail_scan/pixelated_v1_v1p1.txt`，已移动的样本文件夹清单在同目录 `quarantine_moved.txt`。
- v1p2 起安全过滤已关闭，不受影响。

## 待解决

1. **Kimi 服务重启时样本失败：** `sdharur` 的服务池已下线，v1p4 起改用我们自己的副本。副本被重新排队时，Kimi 这边没有可用服务会让样本立即失败、很快耗尽配额（971940 就是这样结束的）。应该像 v1p5 的生图一样改成等待重试。
2. **服务选择（如继续用自建服务池）：** 同一个 agent 固定使用一个服务，开始时选负载最低的服务，以利用前缀缓存、用上空闲服务。
3. **Cosmos3 生图失败：** 安全过滤关闭后已不再是主要失败来源；如今后重新开启过滤，被拒绝时应作为可恢复错误交还模型改写提示词，而不是让样本失败。
4. **内存：** 每个样本结束后清空私有 `/tmp/poster_agent`，并在进度日志里记录内存占用，以便查清 v1p1 内存溢出的其余原因。
