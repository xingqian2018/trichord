# 3p5 数据最终计划：dense pretrain teacher 缓存格式（top-64）

top-64 + next token 名次 + off-top-64 采样 · 2026-10-29 · 状态：格式确定，代码已写好（CPU 测试通过，尚未在 GPU 上跑过）

本格式适用于 dense pretrain 的任意 teacher，目标数据为约 15T token 的预训练数据。

**与 v2.1 的区别**（[2026-10-17_dense_pretrain_teacher_cache_v2](2026-10-17_dense_pretrain_teacher_cache_v2.html)）：① 只存 top-64（v2.1 为 top-128）；② tail 采样只留 off64 一份，去掉 off128；③ off64 的 tail 总质量和三个 tail 统计量直接存（v2.1 由 off128 的值加回第 65–128 名算出）。分块、uint24、next token 名次与 logp、词表限定、逐行随机数都不变。每行 694 → 354 字节。

## 1. 概览

- **topK 主数据**：sorted top-64 的 id（uint24）和 log 概率（float16）；实际下一个 token 的 id 及其名次（`next_token_id_and_rank`）和 log 概率（`next_token_logp`，每行都存）。
- **offtopK 数据**：top-64 之外的 tail 总质量和三个 tail 统计量；以及一份 tail 采样 off64：从 top-64 之外重新归一化的分布中有放回采样 4 个，存 id 和 log 概率。
- tail 采样的用途：训练时 `exact_k` = 64，top-64 以内精确计算，tail 用这 4 个样本做无偏估计。
- tail 统计量的用途：精确算出 teacher 的熵、‖p‖²，以及温度 T = 0.5 和 T = 2 时的 tail 质量。它们要对完整词表求和，只能在 teacher 打分时算，事后只能用 4 个样本粗略估计。
- 每行 354 字节，15T token 预训练数据总量约 5.3 PB。

## 2. 分块与目录

分块沿用现有机制：每个 prefix 一个目录，较大的 prefix 由 producer 的 `plan_units`（现有实现在 `dpl/distill/producers/vllm_topk.py`）按文档边界切成 `parts/<lo>-<hi>/`。

**共享存放目录**：`/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_llm_datasets/pretrain/tc/<model>/`，`<model>` 为 teacher 名（如 `mimo-v2.6-pro-rl`）。

```
/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_llm_datasets/pretrain/tc/<model>/<blend_hash>/<prefix_key>/      # 主目录或 parts/<lo>-<hi>/
    meta.json
    docs.idx
    top64_id.bin              top64_logp.bin
    next_token_id_and_rank.bin    next_token_logp.bin
    off64_tail_logp.bin           off64_tail_stats.bin
    off64_sample_id.bin       off64_sample_logp.bin
```

- **单元（unit）**：主目录或一个 `parts/<lo>-<hi>/` 目录。每个单元是完整、独立的缓存，有自己的 `meta.json` 和 `docs.idx`，`row_start` 是单元内的局部行号。
- **切块边界**：严格按文档边界，一篇文档不跨单元，重算单个单元不会牵连相邻单元。
- **单元大小**：由 `--part-tokens` 控制。
- **读取**：读取端可以直接按单元读取，不需要合并 parts。
- **并发**：同一目录只允许一个 writer（`writer.lock`），任务按单元切分到 Slurm array。

## 3. 文件定义

记 n 为该单元所有已缓存文档的行数之和。所有 `.bin` 都是无文件头、小端序、按行展开的定长数组，可以用 `np.memmap` 读取。

| 文件 | 形状 | dtype | 内容 |
|---|---|---|---|
| `top64_id.bin` | [n,64,3] | uint8 | teacher 对下一个 token 概率最高的 64 个 token id（uint24，小端），按概率降序 |
| `top64_logp.bin` | [n,64] | float16 | 对应的 log 概率（自然对数，T=1），降序 |
| `next_token_id_and_rank.bin` | [n] | uint32 | `id<<8 \| rank`：id 为实际下一个 token（uint24），rank 为它在 top-64 中的名次 0..63，不在 top-64 时为 255 |
| `next_token_logp.bin` | [n] | float16 | 实际下一个 token 的 teacher log 概率（T=1），每行都存；rank < 64 时等于 `top64_logp[rank]` |
| `off64_tail_logp.bin` | [n] | float16 | log Σ_{i∉top-64} p_i：在 float32 中对 top-64 以外的 log 概率做 logsumexp，再转 float16。不要用 log(1 − Σ top-64 概率)：tail 很小时相减会损失精度 |
| `off64_tail_stats.bin` | [n,3] | float16 | top-64 以外 token 的三个求和的 log：[:,0] = log(−Σ p log p)，[:,1] = log Σ p²，[:,2] = log Σ √p（定义和用法见下） |
| `off64_sample_id.bin` | [n,4,3] | uint8 | 从 top-64 之外的 tail 分布（重新归一化）有放回采样的 4 个 token id（uint24） |
| `off64_sample_logp.bin` | [n,4] | float16 | 这 4 个 token 的 teacher log 概率（T=1，不是条件概率） |
| `docs.idx` | [文档数,3] | int64 | 每行 `(doc_idx, row_start, n_rows)`，按 `doc_idx` 排序 |
| `meta.json` | — | JSON | 格式版本（`dpl.teacher_cache/2.2`）、teacher、k=64、采样的 m=4、temperature、随机数规则、三个 tail 统计量的定义、vocab_limit、blend_hash、行数，以及每个文件的字节数和校验和 |

行的含义：文档有 L 个 token（含结尾 EOD）时有 L−1 行，第 t 行是 teacher 看到前 t+1 个 token 后对第 t+1 个 token 的预测，在各文件中的行号为 `row_start + t`。

**词表限定：teacher 的分布只在 Qwen3 的 151,669 个 token（id 0–151668）上计算。id ≥ 151669 的 logit（MiMo-V2 多出的 6 个特殊 token 151669–151674，以及输出层补齐用的 pad 行，到 152575）先置为 −∞，再归一化：lp_i = z_i − log Σ_{j<151669} exp(z_j)。因此所有文件中的 id 都 < 151669，与 Qwen3 词表兼容；`meta.json` 记录 `vocab_limit = 151669`。**

**teacher 词表核对**（2026-10-29，直接读 tokenizer.json 和 config.json）：3p5 的 teacher 为 MiMo-2.5-Pro-Base 或 MiMo-2.6-Pro-RL，两者都按上面的规则截到 151,669 个 id。

| 模型 | 真实 id | 前 151,669 个 id 与 Qwen3 一致 | id ≥ 151669 的 token | 输出维度（含 pad） | EOD |
|---|---|---|---|---|---|
| Qwen3-32B | 0–151668 | — | 无 | 151,936 | 151643 |
| MiMo-2.6-Pro-RL | 0–151674 | 是 | 6 个 audio/video 特殊 token（151669–151674） | 152,576 | 151643 |
| MiMo-2.5-Pro-Base | 0–151668 | 是 | 无 | 152,576 | 151643 |

限定到 151669 后，MiMo-2.6 去掉 6 个特殊 token 和 pad 行（151675–152575），MiMo-2.5 只去掉 pad 行（151669–152575）。EOD（`<|endoftext|>`）在三者中都是 151643。

**uint24 的读写**：id 小于 2^24（要求 teacher 词表 ≤ 16,777,216，现有 teacher 均满足）。每个 id 占最后一维的 3 个 uint8 字节 `a[..., 0..2]`，小端序（低字节在前）：

```
byte:   a[...,0]          a[...,1]              a[...,2]
        bit 7..0          bit 15..8             bit 23..16
写入:   id & 0xFF         (id >> 8) & 0xFF      (id >> 16) & 0xFF
读取:   id = a[...,0] | (a[...,1] << 8) | (a[...,2] << 16)      # 先转 int32 再移位
```

- 即 `a[...,k] = (id >> 8k) & 0xFF`（k = 0, 1, 2），i.e. **`id = a[0] + a[1]·2^8 + a[2]·2^16`**（`a[0]` 是最低字节）。
- 例：id = 1000 = 0x0003E8 → `[0xE8, 0x03, 0x00]` = `[232, 3, 0]`；还原 232 + 3×256 + 0×65536 = 1000。

**`next_token_id_and_rank` 的读写**：一个 uint32 里，高 24 位是 id，低 8 位是 rank。

```
bit:   31 ............... 8 | 7 ...... 0
       id(24 位)            | rank(8 位)
```

- 打包：`v = (id << 8) | rank`
- 解包：`id = v >> 8`，`rank = v & 0xFF`（取末 8 位）
- 例：id = 1000、rank = 3 → v = 1000×256 + 3 = 256003。按小端序存储时，第 0 个字节是 rank，第 1 到 3 个字节是 id。

**`off64_tail_logp` 的计算**：p_i 为 teacher 对 token i 的概率（T = 1），lp_i = log p_i。

```
off64_tail_logp = log Σ_{i∉top-64} p_i = logsumexp_{i∉top-64} lp_i
```

即 top-64 以外所有 token 的概率之和取 log。用 logsumexp 直接对 tail 求和，不用 log(1 − Σ_{top-64} p_i)：tail 很小时相减会损失精度。

**`off64_tail_stats` 的定义**：求和范围都是 top-64 以外的 token，p 为 T=1 的完整分布。

```
[:,0] = log( −Σ p_i log p_i )    tail 对 teacher 熵的贡献
[:,1] = log( Σ p_i² )            tail 对 ‖p‖² 的贡献，也是 T = 0.5 时的 Σ p_i^(1/T)
[:,2] = log( Σ √p_i )            T = 2 时的 Σ p_i^(1/T)
```

- **用途**：teacher 熵 H = −Σ_{top-64} p log p + exp([:,0])，用于按熵加权或跳过低熵位置。‖p‖² = Σ_{top-64} p² + exp([:,1])；tail 形状 p/P_tail64 的 ‖·‖² = exp([:,1]) / P_tail64²，决定 tail 采样的方差，即用多少样本才够。温度 T 下 teacher 分布 ∝ p^(1/T)，归一化常数是 Σ_{top-64} p^(1/T) + Σ_{tail} p^(1/T)：T = 0.5 时第二项是 exp([:,1])，T = 2 时是 exp([:,2])。T 在 0.8–1.33 之间时，第二项可以用 4 个样本估计（P_tail64 · 样本上 p^(1/T−1) 的平均）。
- **计算**：与 tail 总质量相同，在 float32 中对 top-64 以外的 log 概率 lp 做 logsumexp：[:,0] = logsumexp(lp + log(−lp))，[:,1] = logsumexp(2·lp)，[:,2] = logsumexp(lp / 2)。p = 0（lp = −inf）的 token 先去掉，否则 [:,0] 会得到 NaN。tail 质量为 0 的行，三列都为 −inf。

**不存 `token_ids.bin`**：`next_token_id_and_rank` 里的 id 就是语料中第 t+1 个 token，可以对照语料逐行校验，不需要单独的输入 token 文件。

## 4. next token logp 与一致性校验

next token 的 log 概率每行都存（`next_token_logp.bin`，稠密 float16），不再区分它是否在 top-64 中。稠密数组和其他 `.bin` 一样按行对齐，不存在按 `rank==255` 的出现顺序隐式对应的错位问题，也可以直接 `np.memmap`，不需要整体载入。

- **大小**：每行 2 字节，15T 行约 30 TB，约占总量的 0.6%。
- **读取时校验**：rank < 64 的行，`next_token_logp` 必须等于 `top64_logp[rank]`（两者来自同一个 float32 值，转 float16 后应逐位相同）；rank == 255 的行，`next_token_logp` 必须 ≤ `top64_logp[63]` 且 ≤ `off64_tail_logp`。不一致即报错。
- **提交顺序**：沿用"先写数据文件，再原子地写 `docs.idx` 和 `meta.json`"；所有 `.bin` 都在 `docs.idx` 提交前写好（tmp + `os.replace`）。
- **损坏的影响范围**：`meta.json` 中记录每个文件的校验和，损坏时只重算一个单元。
- **窗口与上下文**（2026-10-30 补记）：文档 body 超过 `--window`（默认 8191）个 token 时分窗口打分：第一个窗口是 body[0:window]，之后每个窗口在它负责的第一行之前多带 `--window-context`（默认 2048）个 token 作为上下文，只保留非重叠部分的行，最后一个窗口带 EOD。所以位置 ≥ window 的行是在 context..window−1 个前文 token 的条件下预测的，不是全文前缀；两个参数写入 `meta.json`（`window`、`window_context`、`layouts.window`），也写入 plan 文件，producer 会拒绝与 plan 不一致的设置。15T 生产前要明确选择这两个值（候选：32767 / 8192，配 `--max-model-len 32768`，使每一行至少有 8K 上下文；MiMo 的原生预训练长度是 32K）。
- **空文档**（2026-10-30 补记）：只有 EOD 或完全为空的文档有 0 行（没有任何 token 能预测它的第一个 token），照样写入 `docs.idx`（`n_rows = 0`），这样单元才能完整；reader 要接受 `n_rows = 0` 的条目。
- **统计量边界的容差**（2026-10-30 补记）：`--validate` 检查 `log Σp² ≤ 2·log P`、`log Σ√p ≥ ½·log P`、`log(−Σp log p) ≥ log P + log(−log P)` 时，容差按每个 float16 存储值自身的 ulp 逐行缩放（|x| ≥ 16 时 fp16 间距已经大于固定的 0.01），否则真实数据会被误判。
- **packed 模式**（2026-10-31 补记）：生产语料会预先 pack 成 8K / 32K 的训练序列，producer 加 `--packed`：每个“文档”就是一个 pack，整条原样作为 prompt 打分（不追加 EOD、不裁剪、不分窗口；pack 内跨文档是普通 causal attention，即不做 document mask 的 student 看到的东西）；L 个 token 的 pack 得 L−1 行，第 j 行是对第 j+1 个 token 的预测；每个 pack 必须 ≤ `--max-model-len`（超长直接报错，不截断）。`meta.json` 和 plan 文件记录 `packed`，两种模式的单元互不混用；`--validate` 按 pack 的规则核对 next-token id。如果 student 训练时做 document mask，就不要用 packed 模式，沿用文档模式（每篇文档单独打分等价于 teacher 侧的 document mask）。

## 5. 采样规则

记第 t 行 teacher 的完整分布为 p。

- **off64**：q64(i) = p_i / P_tail64（i 不在 top-64 中），有放回采样 4 个。
- 采样必须在 GPU 上用 teacher 的完整分布进行，不能用 top-64 之外的近似分布。
- 采样出的 token 可以与 next token 重复，也可以互相重复，这是有放回采样的正常结果。
- **可复现性**：随机数逐行确定。每篇文档一个 key，取 sha256(blend_hash | prefix_key | doc_idx | seed) 的 63 位；第 t 行第 j 个样本的 uniform 由 hash(key, t, j) 生成（53 位），再在 tail 的累积分布上做逆变换采样。batching、窗口切分、中断续跑或单独重算某篇文档，抽到的样本都一样。规则写入 `meta.json`。
- **精度**：log 概率、tail 总质量和三个 tail 统计量都在 float32 中用 logsumexp 计算，最后才转 float16；逆变换采样用的 tail 累积分布在 float64 中计算，概率极小的 token 也能抽到。
- **融合 tail 计算与 chunk 依赖**（2026-10-31 补记）：插件把 tail 统计与采样合并成一次 fp32 exp（按行减最大值）+ 普通 sum 归约 + 两级 float64 逆 CDF（块质量 → 块内），与 numpy float64 逐行参考比较：trailer 绝对误差 ≤ 2e−6，3072 个样本全部一致；GB300 上每 512 行 9.0 ms → 3.9 ms。已知的非确定性：CUDA 上 fp32 归约（`restrict_vocab_` 的 logsumexp，以及 vLLM 自己的 log_softmax）的求和顺序随 chunk 行数变化，最后一位可能不同——换一种 chunk 划分时约 0.005% 的 top-64 log 概率在 float16 上差 1 ulp、约 0.01–0.05% 的样本落到相邻 id；同形状重复运行逐位相同。这比 MoE teacher 自身的 run-to-run 抖动（KL 均值 0.006 nat）小得多，不处理。
- **可行性**：本格式的 producer（Dense-Pretraining-LMs 仓库的 `tools/distill/dense_teacher_cache_producer_3p5.py`）和插件（`tools/distill/vllm_offtopk_plugin_3p5/`）是 v2.1 代码的单份采样副本，v2.1 的代码不变；v2.1 已经用同一套 vLLM 钩子在 MiMo-2.6 的 45B 和 650B 语料上生产。换其他 teacher 时需要在对应模型上重新验证。

## 6. 大小估算

按 15T token 的预训练数据估算，行数 N ≈ 15T（每篇文档少一行，可忽略）。

| 文件 | 每行字节 | 总大小 |
|---|---|---|
| `top64_id.bin` | 192 | 2,880 TB |
| `top64_logp.bin` | 128 | 1,920 TB |
| `next_token_id_and_rank.bin` | 4 | 60 TB |
| `next_token_logp.bin` | 2 | 30 TB |
| `off64_tail_logp.bin` | 2 | 30 TB |
| `off64_tail_stats.bin` | 6 | 90 TB |
| `off64_sample_id.bin` + `off64_sample_logp.bin` | 12 + 8 | 300 TB |
| **合计** | **354** | **约 5,310 TB（5.3 PB，约 4.7 PiB）** |

`docs.idx` 按现有语料的平均文档长度（约 750 token）估算约 200 亿篇、约 480 GB，约占总量的 0.01%，可以忽略。
