# Lance 数据读取代码的设计逻辑（我们自研部分）

目的：把我们自己设计、而不是直接复用 imaginaire / webdataset 现成机制的那部分数据读取逻辑写清楚，方便之后复现、迁移或排查。范围只覆盖 `projects/cosmos3/cosmos3/datasets/` 下 Lance 相关的新代码，以及 `configs/base/experiment/text_rendering/data_utils_lance.py` 里的装配方式。复用自 imaginaire 的部分（`bolk_distribute_ratio_to_shard`、`format_batch_so_model_can_handle` / `recollate_fn`、`TokenizeText` 等）只标注来源，不展开。

## 0. 总览

两条读取链路，共用同一个 `LanceScanner`：

```
训练图像链路（text-to-image 预训练 / 数据消融）
  LanceImageDataLoader(torch DataLoader, forkserver workers)
    └─ LanceImageUnionDataset            # 多表按 ratio 分配 worker 槽位
         └─ LanceImageAdvancedDataset ×N # 每表：整 batch 输出
              ├─ LanceScanner            # chunk 分发 + 过滤 + 行级打乱
              ├─ filter (LanceResolutionFilter)
              ├─ media_loader (S3RangeImageMediaLoader)   # 按字节范围读图
              └─ formatter (LanceImageFormatter)          # resize/pad/normalize + caption + tokenize

RL prompt 链路（NFT / SIFR 的 rollout 提示词）
  RLPromptDataloader（文本文件）  /  RLPromptLanceDataloader（Lance 表）
    └─ rank0 取样 → group 重复 → broadcast → 按 rank 切片
         └─ LanceCaptionFormatter（无图像，只出 caption + token）
```

| 文件 | 角色 |
|---|---|
| `datasets/lance/lance_scanner.py` | 按 chunk 读 Lance 表；chunk 由 TCPStore 上的全局计数器认领 |
| `datasets/lance/lance_image_dataset_v2.py` | 过滤器、S3 字节范围读图、两种 formatter、单表/联合数据集 |
| `datasets/lance/lance_image_dataloader_v2.py` | `LanceImageDataLoader`：把数据集包进 DataLoader |
| `datasets/lance/ordered_prefetch.py` | 有界、保序的线程预取 |
| `datasets/rl_prompt_dataloader.py` | RL 用的 prompt 加载器（文件版与 Lance 版） |
| `configs/base/experiment/text_rendering/data_utils_lance.py` | 各数据集（`ds_*`）、联合配比（`ds_union_*`）、`get_dataloader` |

## 1. LanceScanner：全作业共享的 chunk 分发

### 1.1 要解决的问题

多机多 rank、每 rank 多个 DataLoader worker 同时读同一张大表，要求：
- 不同 worker 读到的行不重叠，且不需要预先给每个 worker 划分固定分片（节点数、worker 数可变，续训时也可变）；
- 打乱顺序可复现（给定 seed），但不给定 seed 时每次运行不同；
- 带 SQL 过滤条件时，过滤后行数不均匀也不能卡死；
- 表在训练过程中被写入新版本时，所有 worker 仍读同一个版本。

### 1.2 做法

1. **固定大小的 chunk。** `LANCE_DISPATCH_CHUNK_ROWS = 1000`。`ChunkLayout` 把（可选的）选中 fragment 按行数排序后依次切成 1000 行一块，`cumulative_chunks` 做前缀和，`locate(chunk_index)` 用 `searchsorted` 把全局 chunk 序号映射为 (fragment, start, stop)。
2. **用 TCPStore 做全局认领计数器。** 每个 worker 通过 `MASTER_ADDR/MASTER_PORT` 连到训练作业自己的 TCPStore（`connect_store`），`_next_chunk` 对键 `lance_scanner/<identity hash>` 做原子 `add(key, 1)`，返回值减 1 就是本次认领的序号 `claim`。
   - `epoch, position = divmod(claim, total_chunks)`：计数器超过总 chunk 数后自动进入下一个 epoch，不需要任何 rank 间的 epoch 同步。
   - `shuffle=True` 时，每个 epoch 用 `derive_seed(*shuffle_seed_global, f"epoch_{epoch}")` 生成一个 chunk 排列（`np.random.default_rng(seed).permutation`），`position` 在排列里取实际 chunk；`shuffle=False` 时按 `position` 顺序读。
3. **版本与 seed 在作业内冻结。** `build()` 先看 store 里有没有 `run_state`：没有就打开最新版本，然后用 `compare_set` 把 `{version, shuffle_seed_global}` 写进去（谁先写谁赢）；已有就按 store 里的版本重新打开。这样所有 worker 读同一个表版本、同一个全局 seed。
4. **行级打乱。** chunk 内按 batch 读取（`lance_batchsize=256`），每个 batch 用 `derive_seed(全局 seed, 本 worker 的局部标识, 递增计数)` 生成新的 seed 后 `random.Random(seed).shuffle`，不同 worker 的打乱互相独立但可复现。worker 局部标识是 `worker_{rank*num_workers+worker_id}`，其中 rank 由 `get_worker_rank_world_size()` 取（spawned worker 里没有 torch 进程组，所以回退读环境变量 `RANK/WORLD_SIZE`）。
5. **过滤在 chunk 读出之后做。** 每个 batch 先读出来，如果有 `lance_filter`，就写成 `memory://` 的临时 Lance 数据集再用 `scanner(filter=...)` 过滤（让过滤表达式走 Lance 的 SQL 语法，而不是自己在 Python 里实现）。所以 chunk 的边界是按“未过滤行数”划分的，过滤后一个 chunk 可能产出很少甚至零行。
6. **空 chunk 保护。** `__call__` 在当前 chunk 耗尽后继续认领下一个；连续 100 个 chunk（`_MAX_CONSECUTIVE_EMPTY_CHUNKS`）都没有通过过滤的行就抛错，而不是死循环。
7. **作业身份。** `identity = f"{lance_uri}|{lance_filter}|{fragment_ids}"`，计数器的键是它的 blake2b 摘要。同一个作业里两个数据集只要 uri / 过滤 / fragment 不全相同，就用不同的计数器，互不干扰；完全相同的两个 scanner 则共享计数器（它们会瓜分同一个 chunk 序列）。
8. **fragment 选择。** `fragment_ids="0-543,3842-3881"` 这种字符串由 `parse_fragment_ids` 解析（范围两端都包含），只对单表有效；多表（`lance_uri` 为列表）时禁止同时设置。这就是把一张全量主表切成 `pretrain_real / pretrain_sdg / pretrain_text` 三个逻辑数据集的方式（同一个 uri，不同 fragment 范围）。

### 1.3 复现要点

- 必须有 TCPStore：scanner 依赖 `MASTER_ADDR` / `MASTER_PORT`（torchrun 和 Slurm 启动器会导出）。单进程本地调试也需要一个起着的 store，否则 `connect_store` 直接报错。
- 想要整次实验可复现，要给 `LanceImageDataLoader(seed=...)` 或 `RLPromptLanceDataloader(seed=...)` 传 seed；`None` 表示每次运行随机抽 `SystemRandom`。
- 续训时 store 是新的，计数器从 0 开始：epoch 内的位置不会被恢复（这一点和 checkpoint 里是否保存数据位置无关，当前设计里没有保存）。

## 2. 单表数据集：`LanceImageDataset` / `LanceImageAdvancedDataset`

`__iter__` 的流水线：

1. 依次调用 scanner、media_loader、filter、formatter 的 `build()`（有则调）——这些对象在每个 DataLoader worker 里各自构建，比如 S3 客户端、SSL 证书预热。
2. `_iter_rows`：无限循环地从 scanner 取行，`filter(row)` 返回 **True 表示丢弃**（命名上是“是否过滤掉”，不是“是否保留”）。
3. `_build_sample`：补上 `data_collection_name = 数据集名`，依次过 `media_loader`（下载+解码）和 `formatter`；任一步返回 `None` 都丢掉该样本。
4. 用 `iter_ordered_prefetch(rows, _build_sample, max_concurrent=prefetch_concurrency)` 在线程池里并发做第 3 步，**按输入顺序**产出结果。

`LanceImageAdvancedDataset` 再往上包一层：从样本流里每凑够 `batch_size` 个就调用 `collate_lance_samples`（`recollate_fn` + `format_batch_so_model_can_handle`）整体输出一个 batch；凑不满就结束。也就是说 **batch 是在 Dataset 里组好的**，DataLoader 的 `batch_size=None`。

### 2.1 `iter_ordered_prefetch`

- 有界：同时在飞的任务不超过 `max_concurrent`；保序：结果按提交顺序 yield，慢任务会阻塞后面的结果，但不会乱序，保证给定 seed 时样本顺序确定。
- `max_concurrent == 1` 时不建线程，行为与同步调用一致。
- 提前关闭时取消还没开始的任务，已经进入阻塞 I/O 的任务不等待（守护线程自行结束），避免 DataLoader 关闭时被对象存储重试卡住。

### 2.2 `S3RangeImageMediaLoader`

Lance 表里存的是 `image_s3_range` 这种 `s3r:{profile}//bucket/path:start-length[@crc]` 的字符串，而不是图像字节；加载器按（offset, size）做 **字节范围读取**，从大 shard 文件里直接取单张图。
- `_parse_s3r_uri` 解析 URI；`_normalize_image_s3r_path` 把旧的 SwiftStack bucket 名（`logged_images`、`annotated_data`、`webdataset`、`debug`）映射到迁移后的 GCS bucket / 前缀。
- 通过 multistorageclient 建 GCS S3-compat 客户端，凭据复用访问 Lance 表的同一个 JSON 文件；`connect_timeout 10s`、`read_timeout 30s`、不重试（`total_max_attempts=1`）。
- 每个 worker 的第一次请求用锁串行化（`download_s3_range`），避免多线程同时初始化凭据/连接池时的竞态。
- 读失败、解码失败只记 warning 并返回 `None`，样本被丢弃，不会让训练崩。

### 2.3 `LanceResolutionFilter`

按目标分辨率档（480 / 720 / 768 / 1080）和图像最近的宽高比，查 `IMAGE_RES_SIZE_INFO` 得到该档的最小尺寸；当 **宽、高同时小于** 该尺寸时返回 True（即丢弃），宽或高缺失也返回 True。

### 2.4 `LanceImageFormatter`（图像 + caption）

- 图像：`ResizeLargestSideAspectPreserving`（lanczos，目标尺寸按宽高比档位取）→ `ReflectionPadding` → `Normalize(mean=0.5, std=0.5)`；同时在 `__metadata__` 里记最近的宽高比。
- caption：按 `caption_type` 选取，选不到就返回 `None` 丢样本：
  - `cosmos3_structured_json` / `cosmos3_structured_json_comprehensive_t2i_dense`：在两列 `captioning_cosmos_captioner_image_v1_filtered_caption_cosmos_captioner_image` 和 `..._full_caption_...` 里挑非空的，随机选一个（用 `time.time_ns()` 做种子，不可复现，有意如此）。之后由 `GetGoldenCaption` 按 `gc_reformat_p` 的权重把结构化 caption 随机重排成不同文本格式。
  - `cosmos3.5_hamid_snah_capbalance_nocot_dense`：优先取 `..._dense_recover_caption`，没有再取 `..._dense_caption`，都空则丢样本。
- `gc_reformat_p` 默认权重（相对权重，不要求和为 1）：`json 1, xml 3, html 1, yaml 5, markdown 5, keyvalue 1, prose 1, ast 1, ascii_tree 1, div_nullified 0.2, flattext 0.5`。
- tokenize：`TokenizeText`，`use_system_prompt=False`，`cfg_dropout_rate` 默认 0.05（训练时用于 CFG 的无条件分支）。
- 输出字段只保留 `data_collection_name / images / image_size / ai_caption / text_token_ids`。

### 2.5 `LanceCaptionFormatter`（只要 caption，给 RL 用）

与上面同样的 caption 选取逻辑，但不做图像变换，输出 `data_collection_name / ai_caption / text_token_ids`；与 `media_loader=None` 配合。额外开关 `cosmos3_structure_json_text_special_handling=True` 时，先过 `GetCosmos3StructureJsonTextAndSignage` 从结构化 caption 里抽出画面文字（`text_and_signage`），并把它输出给 OCR 奖励当真值；该开关要求 `caption_type` 是 cosmos3 结构化 JSON 类型。

## 3. 多表混合：`LanceImageUnionDataset`

目的：按固定配比混合多个数据集，并且 **配比在 worker 槽位层面实现，而不是在 batch 内抽样**。

1. 总槽位数 `total_slots = world_size × num_workers`，本 worker 的槽位 `slot = rank × num_workers + worker_id`。
2. 用 imaginaire 现有的 `bolk_distribute_ratio_to_shard(ratios, total_slots, large_bolk_first=True)` 把 [0,1) 数轴按 ratio 切给各槽位：每个槽位拿到“对每个数据集的份额”，大数据集优先连续分配，因此一个槽位通常只负责 1 个数据集，只有边界槽位才同时负责两个。
3. 本 worker 只对份额大于 `1e-9` 的数据集建迭代器；每次产出时在这些数据集里按份额做加权随机选择，然后 `next()` 一个 **整 batch**。
4. 所以一个训练 step 拿到的 batch 来自单一数据集；数据集之间的比例由“多少个 worker 槽位在读它”决定。`slot == 0` 的 worker 会打印每个数据集占用的槽位数，便于核对配比。
5. 各数据集的 `batch_size` 在联合配置里单独指定（`dict(dataset=..., ratio=..., batch_size=16)`），`LanceImageUnionDataset.__init__` 调 `override_batch_size` 写入。
6. 随机选择的 RNG：有 seed 时用 `derive_seed(seed, "union", slot)`，否则 `SystemRandom`。

联合配比（`data_utils_lance.py`）：
- `ds_union_pretrain`：`pretrain_real 8 : pretrain_sdg 1 : pretrain_text 1`
- `ds_union_pretrain_midtrain`：上面三个再加 `midtrain 2.5`
- `ds_union_pretrain_getty`：`pretrain_real 7.2 : pretrain_sdg 0.9 : pretrain_text 0.9 : getty 1.0`（即把 getty 占 10%，其余按 8:1:1 缩小 0.9 倍）

## 4. DataLoader 封装：`LanceImageDataLoader`

- 单个数据集会被自动包成只有一项的 `LanceImageUnionDataset`（ratio 1.0），这样单表与多表走同一条路径。
- 给了 `batch_size` 就覆盖数据集/联合项里的 batch size；给了 `seed` 就调用 `dataset.set_seed`，该 seed 向下传到每个 scanner 的 chunk 排列和每个 worker 的局部 seed。
- `torch.utils.data.DataLoader(dataset, batch_size=None, ...)`：batch 已在 Dataset 里组好；`multiprocessing_context="forkserver"`（有 worker 时）；`persistent_workers=True`；默认 `num_workers=4`、`prefetch_factor=2`、`pin_memory=False`。
- 装配入口 `get_dataloader(ds_cfg, batch_size=None, seed=None, num_workers=4)` 返回 Hydra 配置字典；实验配置里写 `dataloader_train=data_utils_lance.get_dataloader(data_utils_lance.ds_union_pretrain, batch_size=16)`。

## 5. 数据集定义（`data_utils_lance.py`）

共同点：`lance_column` 只读必要列（`uuid, image_s3_range, source_dataset, height, width` + 两个 hamid dense caption 列），`media_loader` 都是 `S3RangeImageMediaLoader`（`prefetch_concurrency=4`），`formatter` 都是 768 分辨率 + `cosmos3.5_hamid_snah_capbalance_nocot_dense` caption，凭据 `credentials/gcp_training.secret`，默认 `LanceResolutionFilter("768")`。

| 名称 | Lance 表 | 选择方式 | 额外过滤 |
|---|---|---|---|
| `coyo_700m` | `lancedb/image/regular/coyo_700m_slice_from_maintable_20260828.lance` | 整表 | caption 非空 |
| `midtrain` | `lancedb/image/unified/midtrain_made_on_20260929.lance` | 整表 | caption 非空 |
| `pretrain_real` | `nv-00-10206-lancedb/prod/image/image_meta_table_full.lance` | `fragment_ids="0-543,3842-3881,3973-4617"` | caption 非空、aesthetic ≥ 4、非 collage / text_banner / nsfw / watermark / white_background、`dedup_v3_is_duplicate IS NOT TRUE`（额外读取这些过滤列） |
| `pretrain_sdg` | 同上主表 | `fragment_ids="4735-4943,5116-5232"` | caption 非空；`filter=None`（不做分辨率过滤） |
| `pretrain_text` | 同上主表 | `fragment_ids="5073-5102,36952-36992"` | caption 非空；`filter=None` |
| `getty` | `lancedb/image/regular/getty_image_350k_test.lance` | 整表 | 只读 recover caption 列；无 SQL 过滤，分辨率过滤保留 |

“caption 非空”指 `dense_caption IS NOT NULL OR dense_recover_caption IS NOT NULL`。

## 6. RL prompt 加载器

两个类接口一致：`__next__` 返回 `{key: [每个本地样本的值]}`，并把 `text_token_ids` 再包一层列表（`batch["text_token_ids"] = [batch["text_token_ids"]]`），与训练 batch 的格式对齐。

### 6.1 共同的分组逻辑（NFT / SIFR 的核心约束）

- 每个 rank 每次拿 `batch_size`（通常 1）条 prompt；`global_batchsize = batch_size × world_size`。
- `group_size`：同一条 prompt 在连续的 `group_size` 个位置重复，于是相邻的 `group_size` 个 rank 拿到 **同一个 prompt**；要求 `world_size % group_size == 0`。奖励侧再把这些 rank 的 rollout 汇聚成一个 GRPO 组（`true_num_rollouts_per_sample = group_size × num_rollouts_per_sample`，并由 `validate_group_tags` 校验同组 caption 一致）。
- 只有 rank 0 读数据：取出 `global_batchsize` 条后用 `distributed.broadcast_object` 广播给所有 rank，各 rank 取自己的切片 `[rank×bs, (rank+1)×bs)`。这保证所有 rank 在同一步看到的 prompt 完全一致且分组对齐，不依赖各 rank 独立随机。

### 6.2 文件版 `RLPromptDataloader`

- 读 `prompt_file`（每行一条，丢掉空行），`shuffle=True` 时用不带种子的 `random.Random()` 整体打乱，再把每条重复 `group_size` 次。
- 缓冲区不够一个全局 batch 时重新读文件并（再次打乱）追加，实现无限循环；`debug_limit_n` 只取前 N 条。
- `text_and_signage_from_prompt`：用第一对双引号里的内容作为 OCR 奖励的文字真值（没有成对引号或引号内为空则为 `None`）。
- 输出样本带 `ai_caption / text_and_signage / data_collection_name="rl_prompt_file"`，再过 formatter（`TokenizeText`，`use_system_prompt=False`）。

### 6.3 Lance 版 `RLPromptLanceDataloader`

- 只有 rank 0 调用 `lance_scanner.build()`，因此也只有 rank 0 去 TCPStore 上认领 chunk。
- `_pull_samples(n)`：循环 `scanner()` 取行、设置 `data_collection_name="rl_prompt_lance"`、过 formatter；formatter 返回 `None` 的行（caption 缺失）直接跳过，直到凑够 n 条。
- 每轮拉取 `global_batchsize // group_size` 条 **不同** prompt，scanner 开着 shuffle 时再整体打乱一次，然后每条重复 `group_size` 次。
- `debug_limit_n` 模式：只第一次拉取 N 条并缓存到 `_debug_pool`，之后每轮都复用这 N 条（用于固定小集合过拟合调试）。
- 与文件版不同：行本身是字典（含 `text_and_signage` 等 formatter 产出的字段），不需要再从 prompt 里切引号。

### 6.4 NFT 实验里的实际配置

| 配置名 | 来源 | caption 类型 | 备注 |
|---|---|---|---|
| `dl_rl_prompt_only_pickscore` | `pickscore_train.txt`（25,432 条） | 原文 | `group_size=4`，`shuffle=True` |
| `dl_rl_prompt_only_ocr` | `ocr_train.txt` | 原文 | 复制自 pickscore 配置，换文件 |
| `dl_rl_prompt_only_lance_coyo700m` | coyo700m slice，过滤要求 hamid caption 与 cosmos captioner caption 都存在 | `cosmos3_structured_json`，`gc_reformat_p={"markdown": 1}`，`cfg_dropout_rate=0` | `group_size=4` |
| `dl_rl_prompt_only_lance_synthetic_scene_text_v0` | `synthetic_text/synthetic_scene_text_v0_slice_from_maintable_20260904.lance` | `cosmos3_structured_json_comprehensive_t2i_dense`，`cosmos3_structure_json_text_special_handling=True` | 其余同 coyo700m 版 |

实验配置里再用 `dataloader_train["group_size"] = 2` 等覆盖组大小；RL 加载器里 `gc_reformat_p={"markdown": 1}` 表示只用 markdown 重排，目的是让 rollout 的 prompt 格式固定。

## 7. 设计取舍与复现清单

设计取舍（为什么这么做）：
- chunk 计数器放在 TCPStore，而不是预分片：节点数、worker 数、续训后的拓扑都可以变，不需要重新切数据；代价是续训时不能恢复 epoch 内位置。
- 配比用“worker 槽位份额”实现：避免在 batch 内混合不同数据集带来的尺寸/packing 不均，每个 batch 单一来源；代价是配比的粒度受 `world_size × num_workers` 限制（槽位总数太少时，小比例数据集可能分不到槽位，需看 `slot==0` 打印的槽位分配）。
- batch 在 Dataset 里组、DataLoader `batch_size=None`：让组 batch 的 CPU 工作留在 worker 进程里。
- 过滤交给 Lance SQL：改过滤条件只需改配置字符串，不改代码。
- RL 提示词只由 rank 0 采样并广播：保证 GRPO 分组严格对齐。

复现清单：
1. 作业环境里有可访问的 `MASTER_ADDR/MASTER_PORT`（TCPStore）与 `RANK/WORLD_SIZE`；凭据文件 `credentials/gcp_training.secret` 存在且能同时读 Lance 表与 S3 兼容的图像 bucket。
2. 想要可复现的数据顺序：`get_dataloader(..., seed=<int>)`（图像）或 `RLPromptLanceDataloader(seed=<int>)`（RL）；注意 caption 选择里的 `time.time_ns()` 随机和 `gc_reformat_p` 重排不受该 seed 控制。
3. 核对联合配比：看启动日志里 `[LanceImageUnionDataset] worker slots per dataset` 一行，确认槽位数与预期比例大致相符。
4. 核对 `lance_filter` 的实际通过率：如果连续 100 个 chunk 没有通过的行会直接报错；通过率极低的过滤条件会让 chunk 认领速度远快于样本产出。
5. 新增数据集时只需要一个 `ds_*` 字典：`lance_uri` +（可选）`fragment_ids` + `lance_column`（覆盖 formatter 用到的所有列）+ `lance_filter` + `media_loader` + `filter` + `formatter`，然后挂进某个 `ds_union_*` 的 `dataset_list`。
6. 测过的/未测过的：本文是对代码的阅读整理，没有在本地跑这条数据链路（本地 `.venv` 不是训练环境）；表中数字（ratio、fragment 范围、阈值）取自当前工作区代码，复现前请以对应 commit 为准。
