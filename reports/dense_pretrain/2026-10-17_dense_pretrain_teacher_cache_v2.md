# Dense pretrain teacher 缓存格式 v2（top-128 + next token 名次 + 双份 off-top-k 采样）

状态：格式已讨论确定（2026-10-17），代码尚未实现。本格式适用于 dense pretrain 的任意 teacher（当前第一个是 `mimo-v2.6-pro-rl`），目标数据为约 600B token 的预训练数据。

修订（2026-10-17，Runlong）：① 新增 `off128_tail_stats.bin`，三个 tail 统计量（第 3 节）；② next token 的 log 概率改为每行都存的稠密数组 `next_token_logp.bin`，取代稀疏 pickle，单元 ≤ 100M 行的限制随之取消（第 4 节）；③ tail 总质量写明用 logsumexp 计算（第 3 节）；④ 补充 `exact_k` = 64 时两份采样合用、逐行随机种子的实现要求（第 5 节）。每行 686 → 694 字节。

## 1. 概览

- **topK 主数据**：sorted top-128 的 id（uint24）和 log 概率（float16）；实际下一个 token 的 id 及其名次（`next_token_id_and_rank`）和 log 概率（`next_token_logp`，每行都存）。
- **offtopK 数据**：top-128 之外的 tail 总质量和三个 tail 统计量；以及两份 tail 采样，各 4 个 id 和 log 概率：off64（假定只有 top-64 完整数据，从 top-64 之外重新归一化的分布采样）和 off128（从 top-128 之外重新归一化的分布采样）。
- 两份采样的用途：训练时 `exact_k` 取 64 还是 128，都有对应的无偏 tail 估计，不需要为 top-64 实验另跑一遍 teacher。`exact_k` = 64 时，off64 的 4 个样本和由 off128 拼出的 4 个样本合起来是 8 个独立样本（第 5 节）。
- tail 统计量的用途：精确算出 teacher 的熵、‖p‖²，以及温度 T = 0.5 和 T = 2 时的 tail 质量。它们要对完整词表求和，只能在 teacher 打分时算，事后只能用 4 个样本粗略估计。
- 每行 694 字节，600B token 预训练数据总量约 416 TB。

## 2. 分块与目录

分块沿用现有机制：每个 prefix 一个目录，较大的 prefix 由 producer（现有实现为 `tools/distill/mimo_offtopk_producer.py`）的 `plan_units` 按文档边界切成 `parts/<lo>-<hi>/`。

**共享存放目录**：`/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_llm_datasets/pretrain/tc/<model>/`，`<model>` 为 teacher 名（如 `mimo-v2.6-pro-rl`）。

```
/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_llm_datasets/pretrain/tc/<model>/<blend_hash>/<prefix_key>/      # 主目录或 parts/<lo>-<hi>/
    meta.json
    docs.idx
    top128_id.bin             top128_logp.bin
    next_token_id_and_rank.bin    next_token_logp.bin
    off128_tail_logp.bin          off128_tail_stats.bin
    off64_sample_id.bin       off64_sample_logp.bin
    off128_sample_id.bin      off128_sample_logp.bin
```

- **单元（unit）**：主目录或一个 `parts/<lo>-<hi>/` 目录。每个单元是完整、独立的缓存，有自己的 `meta.json` 和 `docs.idx`，`row_start` 是单元内的局部行号。
- **切块边界**：严格按文档边界，一篇文档不跨单元。这样重算单个单元不会牵连相邻单元。
- **单元大小**：由 `--part-tokens` 控制。
- **读取**：`ShardedOfftopkReader` 直接按单元读取，不需要合并 parts。
- **并发**：同一目录只允许一个 writer（`writer.lock`），任务按单元切分到 Slurm array。

## 3. 文件定义

记 n 为该单元所有已缓存文档的行数之和。所有 `.bin` 都是无文件头、小端序、按行展开的定长数组，可以用 `np.memmap` 读取。

| 文件 | 形状 | dtype | 内容 |
|---|---|---|---|
| `top128_id.bin` | [n,128,3] | uint8 | teacher 对下一个 token 概率最高的 128 个 token id（uint24，小端），按概率降序 |
| `top128_logp.bin` | [n,128] | float16 | 对应的 log 概率（自然对数，T=1），降序 |
| `next_token_id_and_rank.bin` | [n] | uint32 | `id<<8 \| rank`：id 为实际下一个 token（uint24），rank 为它在 top-128 中的名次 0..127，不在 top-128 时为 255 |
| `next_token_logp.bin` | [n] | float16 | 实际下一个 token 的 teacher log 概率（T=1），每行都存；rank < 128 时等于 `top128_logp[rank]` |
| `off128_tail_logp.bin` | [n] | float16 | log Σ_{i∉top-128} p_i：在 float32 中对 top-128 以外的 log 概率做 logsumexp，再转 float16（现有 `dpl_vllm_offtopk` 钩子就是这样算的）。不要用 log(1 − Σ top-128 概率)：tail 很小时相减会损失精度 |
| `off128_tail_stats.bin` | [n,3] | float16 | top-128 以外 token 的三个求和的 log：[:,0] = log(−Σ p log p)，[:,1] = log Σ p²，[:,2] = log Σ √p（定义和用法见下） |
| `off64_sample_id.bin` | [n,4,3] | uint8 | 从 top-64 之外的 tail 分布（重新归一化）有放回采样的 4 个 token id（uint24） |
| `off64_sample_logp.bin` | [n,4] | float16 | 这 4 个 token 的 teacher log 概率（T=1，不是条件概率） |
| `off128_sample_id.bin` | [n,4,3] | uint8 | 从 top-128 之外的 tail 分布（重新归一化）有放回采样的 4 个 token id（uint24） |
| `off128_sample_logp.bin` | [n,4] | float16 | 同上，对应 off128 |
| `docs.idx` | [文档数,3] | int64 | 每行 `(doc_idx, row_start, n_rows)`，按 `doc_idx` 排序 |
| `meta.json` | — | JSON | 格式版本、teacher、k=128、两份采样的 m=4、temperature、随机种子规则、三个 tail 统计量的定义、blend_hash、行数，以及每个文件的字节数和校验和 |

行的含义：文档有 L 个 token（含结尾 EOD）时有 L−1 行，第 t 行是 teacher 看到前 t+1 个 token 后对第 t+1 个 token 的预测，在各文件中的行号为 `row_start + t`。

**词表限定：teacher 的分布只在 Qwen3 的 151,669 个 token（id 0–151668）上计算。id ≥ 151669 的 logit（MiMo-V2 多出的 6 个特殊 token 151669–151674，以及输出层补齐用的 pad 行，到 152575）先置为 −∞，再归一化：lp_i = z_i − log Σ_{j<151669} exp(z_j)。因此所有文件中的 id 都 < 151669，与 Qwen3 词表兼容；`meta.json` 记录 `vocab_limit = 151669`。**

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

**`off128_tail_logp` 的计算**：p_i 为 teacher 对 token i 的概率（T = 1），lp_i = log p_i。

```
off128_tail_logp = log Σ_{i∉top-128} p_i = logsumexp_{i∉top-128} lp_i
```

即 top-128 以外所有 token 的概率之和取 log。用 logsumexp 直接对 tail 求和，不用 log(1 − Σ_{top-128} p_i)：tail 很小时相减会损失精度。

**`off64_tail_logp` 不单独存**：`off64_tail_logp = log(exp(off128_tail_logp) + Σ_{k=65..128} exp(top128_logp[k]))`，由已有数据算出。

**`off128_tail_stats` 的定义**：求和范围都是 top-128 以外的 token，p 为 T=1 的完整分布。

```
[:,0] = log( −Σ p_i log p_i )    tail 对 teacher 熵的贡献
[:,1] = log( Σ p_i² )            tail 对 ‖p‖² 的贡献，也是 T = 0.5 时的 Σ p_i^(1/T)
[:,2] = log( Σ √p_i )            T = 2 时的 Σ p_i^(1/T)
```

- **用途**：teacher 熵 H = −Σ_{top-128} p log p + exp([:,0])，用于按熵加权或跳过低熵位置。‖p‖² = Σ_{top-128} p² + exp([:,1])；tail 形状 p/P_tail128 的 ‖·‖² = exp([:,1]) / P_tail128²，决定 tail 采样的方差，即用多少样本才够。温度 T 下 teacher 分布 ∝ p^(1/T)，归一化常数是 Σ_{top-128} p^(1/T) + Σ_{tail} p^(1/T)：T = 0.5 时第二项是 exp([:,1])，T = 2 时是 exp([:,2])。T 在 0.8–1.33 之间时，第二项可以用 4 个样本估计（P_tail128 · 样本上 p^(1/T−1) 的平均）。
- **计算**：与 tail 总质量相同，在 float32 中对 top-128 以外的 log 概率 lp 做 logsumexp：[:,0] = logsumexp(lp + log(−lp))，[:,1] = logsumexp(2·lp)，[:,2] = logsumexp(lp / 2)。p = 0（lp = −inf）的 token 先去掉，否则 [:,0] 会得到 NaN。tail 质量为 0 的行，三列都为 −inf。
- **`exact_k` = 64 时**：与 `off64_tail_logp` 一样不单独存，把第 65–128 名的对应项加回去，例如 Σ_{i∉top-64} p² = exp([:,1]) + Σ_{k=65..128} exp(2·top128_logp[k])。

**不存 `token_ids.bin`**：`next_token_id_and_rank` 里的 id 就是语料中第 t+1 个 token，可以对照语料逐行校验，不需要单独的输入 token 文件。

## 4. next token logp 与一致性校验

next token 的 log 概率每行都存（`next_token_logp.bin`，稠密 float16），不再区分它是否在 top-128 中。稠密数组和其他 `.bin` 一样按行对齐，不存在"按 `rank==255` 的出现顺序隐式对应"的错位问题，也可以直接 `np.memmap`，不需要整体载入。

- **大小**：每行 2 字节，600B 行约 1.2 TB，约占总量的 0.3%。对比：稀疏 pickle 在磁盘上更小，但每个 1 亿行的单元载入内存约 120 MB，每个 dataloader worker 各一份，且不能 memmap。
- **读取时校验**：rank < 128 的行，`next_token_logp` 必须等于 `top128_logp[rank]`（两者来自同一个 float32 值，转 float16 后应逐位相同）；rank == 255 的行，`next_token_logp` 必须 ≤ `top128_logp[127]` 且 ≤ `off128_tail_logp`。不一致即报错。
- **提交顺序**：沿用现有的"先写数据文件，再原子地写 `docs.idx` 和 `meta.json`"；`next_token_logp.bin` 和 `off128_tail_stats.bin` 作为数据文件，在 `docs.idx` 提交前写好（tmp + `os.replace`）。
- **损坏的影响范围**：`meta.json` 中记录每个文件的校验和。损坏时只重算一个单元。

## 5. 采样规则

记第 t 行 teacher 的完整分布为 p。

- **off128**：q128(i) = p_i / P_tail128（i 不在 top-128 中），有放回采样 4 个。
- **off64**：q64(i) = p_i / P_tail64（i 不在 top-64 中，即包含第 65 到 128 名），有放回采样 4 个。
- **`exact_k` = 64 时合用两份采样**：off128 的样本加上第 65–128 名，可以拼出与 off64 同分布的样本。每个槽以 P_mid / P_tail64 的概率按 p 从第 65–128 名中抽一个（P_mid = Σ_{k=65..128} p_k），否则用该槽的 off128 样本；`dpl/megatron/dp.py` 的 `offtopk_forward_kl` 已经这样做。拼出的 4 个与 off64 的 4 个相互独立，合起来是 8 个从 q64 抽的独立样本，tail 估计的方差减半。`exact_k` = 128 时只用 off128。
- 采样必须在 GPU 上用 teacher 的完整分布进行，不能用 top-128 之外的近似分布。
- 采样出的 token 可以与 next token 重复，也可以互相重复，这是有放回采样的正常结果。
- 两份采样相互独立，使用不同的随机流。
- **可复现性**：随机种子由 `(blend_hash, prefix_key, doc_idx, 行号, 采样份名)` 确定，规则写入 `meta.json`，中断续跑或单独重算某篇文档，结果一致。现有 `dpl_vllm_offtopk` 钩子每个进程用一个 `torch.Generator`，只在相同的 batching 和调用顺序下可复现，满足不了这一条，需要改成按行确定的随机数，例如由 (种子, 行, 槽) 的哈希生成 uniform，再在 tail 的累积分布上做逆变换采样。
- **精度**：log 概率、tail 总质量和三个 tail 统计量都在 float32 以上精度下用 logsumexp 计算，最后才转 float16。
- **可行性**：现有 MiMo top-128 缓存已经用 `dpl_vllm_offtopk` 钩子（vLLM 0.30.0 + MiMo tp=4）生成，tail 采样钩子在该环境下可行；换其他 teacher 时需要在对应模型上重新验证。

## 6. 大小估算

按 600B token 的预训练数据估算，行数 N ≈ 600B（每篇文档少一行，可忽略）。

| 文件 | 每行字节 | 总大小 |
|---|---|---|
| `top128_id.bin` | 384 | 230.4 TB |
| `top128_logp.bin` | 256 | 153.6 TB |
| `next_token_id_and_rank.bin` | 4 | 2.4 TB |
| `next_token_logp.bin` | 2 | 1.2 TB |
| `off128_tail_logp.bin` | 2 | 1.2 TB |
| `off128_tail_stats.bin` | 6 | 3.6 TB |
| `off64_sample_id.bin` + `off64_sample_logp.bin` | 12 + 8 | 12.0 TB |
| `off128_sample_id.bin` + `off128_sample_logp.bin` | 12 + 8 | 12.0 TB |
| **合计** | **694** | **约 416.4 TB** |

`docs.idx` 按现有语料的平均文档长度估算约 8 亿篇、约 20 GB，可以忽略。
