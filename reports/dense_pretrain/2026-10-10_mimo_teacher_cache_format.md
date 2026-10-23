# MiMo teacher 缓存格式（top-128 + off-top-k 采样）

状态：格式已确定（2026-10-10），代码尚未实现。teacher 为 `mimo-v2.6-pro-rl`，语料为 `kd_corpus/mimo_v2_v1/`（与 `qwen3_v1/` 逐字节相同）。

## 1. 与 Runlong 现有格式（`dpl.teacher_cache/1`）的区别

| | 现有格式（Qwen3.8 / DeepSeek 缓存） | 新格式 |
|---|---|---|
| top-k | k=64（DeepSeek 为 32） | k=128 |
| 实际下一个 token 的概率 | 不存（vLLM 返回了，但只用于统计） | 不在 top-128 时存入 `offtopk_*` 第 0 列 |
| tail 内部信息 | 无，只有总质量 | 每个位置 7 到 8 个按 teacher tail 分布的随机采样 |
| `offsets.bin` | 有（int64，跨 tokenizer 对齐用） | 去掉（identity 对齐下用不到） |
| `token_ids.bin` | 有 | 保留（逐 token 校验对齐） |

## 2. 目录结构

与现有格式相同，按 prefix 存放：

```
<teacher_cache_root>/mimo-v2.6-pro-rl/<blend_hash>/<prefix_key>/
    meta.json
    docs.idx
    token_ids.bin
    topk_ids.bin
    topk_logp.bin
    tail_logp.bin
    offtopk_ids.bin
    offtopk_logp.bin
```

`<prefix_key>` 与语料目录名相同；`<blend_hash>` 来自 `finalize-kd-corpus` 生成的 blend JSON。

## 3. 文件定义

记 N 为该 prefix 所有已缓存文档的行数之和。所有 `.bin` 都是无文件头、小端序、按行展开的定长数组，可以直接用 `np.memmap` 读取。

| 文件 | 形状 | dtype | 内容 |
|---|---|---|---|
| `token_ids.bin` | [N] | int32 | 第 t 行的输入 token（等于语料中该位置的 token，用于校验） |
| `topk_ids.bin` | [N, 128] | int32 | teacher 对下一个 token 概率最高的 128 个 token id，按概率降序 |
| `topk_logp.bin` | [N, 128] | float16 | 对应的 log 概率（自然对数，T=1），降序 |
| `tail_logp.bin` | [N] | float16 | log(1 − Σ top-128 概率)，在 float32 以上精度下计算后再转 float16 |
| `offtopk_ids.bin` | [N, 8] | int32 | top-128 之外的 8 个 token，规则见第 4 节 |
| `offtopk_logp.bin` | [N, 8] | float16 | 这 8 个 token 的 teacher log 概率（自然对数，T=1，不是条件概率） |
| `docs.idx` | [文档数, 3] | int64 | 每行 `(doc_idx, row_start, n_rows)`，按 `doc_idx` 排序 |
| `meta.json` | — | JSON | 格式版本、teacher、k=128、m=8、temperature、采样随机种子规则、blend_hash、行数等 |

行的含义：文档 d 有 L 个 token（含结尾 EOD）时有 L−1 行，第 t 行（t = 0..L−2）是 teacher 看到该文档前 t+1 个 token 后，对第 t+1 个 token 的预测。文档 d 第 t 行在各 `.bin` 中的行号为 `row_start(d) + t`。

`int32` 用于 id：MiMo 词表 152,576 超过 int16 / uint16 的范围。

## 4. `offtopk_*` 的规则

记第 t 行的实际下一个 token 为 y（即语料中第 t+1 个 token），teacher 的 tail 条件分布为 q(i) = p_i / P_tail（i 不在 top-128 中）。

- **y 不在 top-128 中**：第 0 列存 y 和 log p_y；第 1 到 7 列是从 q 中**有放回**随机采样的 7 个 token。
- **y 在 top-128 中**：8 列全部是从 q 中有放回随机采样的 token。

是否属于第一种情况不需要额外标记：检查 y 是否出现在 `topk_ids[t]` 中即可判断。随机采样的 token 可能与 y 重复，也可能互相重复，这是有放回采样的正常结果。

采样必须在 GPU 上用 teacher 的完整分布按 q 进行，不能用 top-128 之外的近似分布。

## 5. 训练时的用法（KD loss）

学生在位置 t 的完整 log 概率为 s（对整个词表 log_softmax）。

top-128 部分精确计算：

```
KL_top = Σ_{k=1..128} p_k · (log p_k − s[topk_ids[t,k]])
```

tail 部分用随机采样做无偏估计（只用随机采样的列，不用第 0 列的 y）：

```
记 R 为随机采样列的集合（y 不在 top-128 时为第 1..7 列，否则为第 0..7 列），M = |R|
KL_tail ≈ P_tail · (1/M) · Σ_{m∈R} ( offtopk_logp[t,m] − s[offtopk_ids[t,m]] )
```

总的 KL = KL_top + KL_tail，loss = (1 − α)·CE + α·T²·KL（现有设置 α=0.75，T=1）。因为 q 可以由存储的 log p_i 和 P_tail 直接算出，所以不需要另存采样的提议概率。

第 0 列的 y（不在 top-128 时）不是从 q 采样得到的，不能放进上面的平均，否则估计有偏。它的用途是诊断 teacher 对实际 token 的拟合程度，以及将来可能的其他估计方法。

## 6. 大小估算

MiMo 语料共 46,658,149,267 个 token、63,035,592 篇文档，行数 N = 46,595,113,675。

| 文件 | 每行字节 | 总大小 |
|---|---|---|
| `token_ids.bin` | 4 | 0.19 TB |
| `topk_ids.bin` | 512 | 23.86 TB |
| `topk_logp.bin` | 256 | 11.93 TB |
| `tail_logp.bin` | 2 | 0.09 TB |
| `offtopk_ids.bin` | 32 | 1.49 TB |
| `offtopk_logp.bin` | 16 | 0.75 TB |
| **合计** | **822** | **约 38.3 TB** |

`docs.idx` 约 1.5 GB，可以忽略。约 62% 的空间是 `topk_ids.bin`；如果以后需要压缩，id 按 18 位打包（152,576 < 2^18）是收益最大的一项，能把总量降到约 27 TB，但需要改读取代码。

## 7. 与现有训练缓存的关系

- 训练用的 top-64 缓存可以从本格式转出：取前 64 列，`tail_64 = log(exp(tail_128) + Σ_{k=65..128} p_k)`。第 65 到 128 列的概率较小，float16 舍入对 tail_64 的影响有限。这样 MiMo 只需要跑一遍。
- 本格式比现有格式多了 `offtopk_*` 两个文件、少了 `offsets.bin`，Runlong 的 `TeacherCacheReader` 不能直接读，需要扩展读取代码或单独写读取器。

## 8. 生成侧的要求和未解决的问题

- vLLM：`prompt_logprobs=128`、`flat_logprobs=True`，第 0 列自带实际 token 的 log 概率，所以 y 的概率不需要额外计算。
- **tail 采样**：vLLM 的 `prompt_logprobs` 不提供完整分布，也不支持按 tail 条件分布采样。需要在计算 prompt logits 的地方加一个 GPU 采样钩子，只把 8 个采样结果传回。**这一步在 vLLM 0.30.0 + MiMo（tp=4，自定义模型）上是否可行，尚未验证**，是整个方案的前提。
- **可复现性**：建议随机种子由 `(blend_hash, prefix_key, doc_idx, 行号)` 确定，写入 `meta.json`，这样中断后续跑或单独重算某篇文档，结果都一致。
- **精度**：log 概率和 tail 都在 float32 以上精度下计算，最后才转 float16。
- **引擎**：MiMo 引擎目前在 GPU kernel 缓存这一步还没跑通（FlashInfer 的 kernel 多进程同时下载，段错误），要先解决。

## 9. 待确认

1. y 在 top-128 中时，8 列全部用随机采样，是否符合预期？
2. 采样随机种子的规则是否按第 8 节的建议？
3. 是否需要按 18 位打包 id，还是先用 int32 保持简单？
4. 38 TB 放在哪个目录、是否在配额以内？
