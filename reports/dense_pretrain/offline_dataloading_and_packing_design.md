# 离线 dataloading 与装箱设计：32K 窗口块的 plan

2026-11-03 · 状态：已实现（`dpl/data/pack_plan_32k.py`），80e9 r500 view 的 q4 tpp200 plan 已生成并验证 · 前置文档：[3p5 数据最终计划 2](2026-10-31_dense_pretrain_teacher_cache_v3p5_final2.html)（4.1 切块规则、第 5 章训练端要求）

> **一句话**：训练要用的每一个样本，由哪几块组成、每块放在第几位、哪里补 pad，全部离线算好，写成一份 plan 文件。每块记录里直接写着 token 和 teacher 行的地址，训练端拿到行号就能读，不做任何抽样、装箱或索引查找。

## 1. 目标与范围

**为什么要离线算**

- teacher cache 按 `| 0 : 32768 | 32768 : 65536 | ...` 切块打分，块和块之间互相看不到。学生必须看到完全相同的块：整块放进样本，绝不切开，开 docmask（3p5 计划 4.1、第 5 章）。
- Megatron 的 `GPTDataset` 是把文档首尾相接、每 seq_len 切一刀，会把块切开，所以不能直接用。
- 装箱结果事先算好有三个好处：
  - 训练前就能检查 padding 比例、blend 比例和每块的完整性；
  - 训练端代码简单，读第几个样本就是查 plan 的第几行；
  - 断点续训、换 GBS、换 DP 规模，结果都不变。

**范围**

- 只管训练集（`blend.json` 的 `train`）。valid / test 沿用现有的 doc-isolated 评测，不在本设计内。
- 本文定义 plan 怎么算、存成什么样、训练端怎么读。训练端读 plan 的代码另做。

## 2. 输入

| 输入 | 值 / 来源 |
|---|---|
| 语料 | 一个 Megatron blend（`blend.json` 的 `train`：权重，prefix），每个 prefix 一份 `data.bin` / `data.idx`，一篇一条 |
| 每篇的长度 | `data.idx` 里的 `sequence_lengths`（int32，含篇尾 EOD）。生成 plan 只读 idx，不读 `data.bin` |
| teacher cache | 可选。cache 所对应语料的根目录（`--cache-corpus`）和 cache 根目录（`--teacher-root`，可给多个，必须同一布局） |
| 窗口 W | 32768 = 训练 seq_len = teacher cache 的块大小 |
| 行数 | train_iters × GBS |
| 种子 | 一个全局 seed |
| buffer 大小 B | 单位 token，见 5.3 |
| 装箱算法 | first_fit 或 best_fit，见 5.4 |

**举例：full 语料按 `blend.json` 的权重训 14.66T token，每个 prefix 要过几遍**

| 统计 | 值 |
|---|---|
| 每个 prefix 要过的遍数 | 最少 0.012，p10 0.29，中位数 1.48，p90 1.57，最多 4.99（`Python.ndjson-223bfc70`） |
| 超过 1 遍的 prefix | 1,063 / 1,975 |
| 来自重复遍的 token | 32.6% |
| 用到的 unique token | 67.4% |

所以必须支持"一个 prefix 过好几遍"和"只过一部分"，这正是第 4 节块流要处理的。

## 3. 切块：和 teacher cache 完全一致

和 3p5 计划 4.1 相同：长度 L（含 EOD）的文档，从第 0 个 token 起每 W 个切一块。

```
doc tokens   | 0 ... 32767     | 32768 ... 65535   | 65536 ... L-1 (EOD)
block        | k = 0           | k = 1             | k = 2（最后一块）
len          | 32768           | 32768             | L - 65536
```

- 块数 K = ceil(L / W)；第 k 块覆盖文档的第 kW 到 min((k+1)W, L) − 1 个 token。
- **满块（len = W）一块就占满一个样本**，不需要装箱。真正参与装箱的只有短文档（L < W）和长文档的最后一块。
- 边界情况：L = nW + 1 的文档，最后一块只有一个 EOD；空文档本身就是一个只有 EOD 的块（L = 1）。照常处理。

> **抽样和装箱的单位都是切好的块，不是文档。** 每次抽一块：抽到长文档时，只拿它的某一节（某个 32K 块），而不是把整篇文档的所有块一起拿进来。长文档的各块在这一遍里随机分散出现；块与块本来就互相看不到，和 teacher 打分时一致。

## 4. 块流：每个 prefix 的顺序和 blend 交错

块流是一条无限长的序列，一块一块交给第 5 节的装箱。

### 4.1 每个 prefix 内部：每遍把全部块重新打乱

- prefix i 的第 p 遍，用种子 `pass_seed(seed, prefix_key, p) = sha256("dpl.pack_plan_32k|seed|prefix_key|p")` 的前 8 字节，生成这个 prefix **全部块**的一个随机排列（Fisher–Yates，xoshiro256** 随机数）。
- prefix i 的流就是第 0 遍排列、第 1 遍排列……首尾相接。每一遍里每块恰好出现一次；要重复的 prefix 每遍顺序都不同；只过一部分的 prefix，就只走到最后一遍排列的前一段。
- **不放回抽样**：一块被抽到以后，要等这个 prefix 的这一遍全部取完、下一遍重新打乱，才可能再被抽到。verify 逐块核对：每个 prefix 内所有块被抽到的次数最多差 1（第 8 节第 4 条）。
  - "遍"（epoch）是每个 prefix 各自的：各 prefix 的遍数不同（e9r500 q4 实例里是 0.25 到 3.65 遍），各自在不同时刻进入下一遍，没有全局统一的 epoch 边界。
  - 换遍的那一刻：一块如果排在第 p 遍的末尾，又恰好排在第 p+1 遍的开头，两次出现会隔得很近。这是每遍重新打乱的固有现象（Megatron 也一样），只发生在一个 prefix 换遍时；需要的话可以加"两次出现之间的最小间隔"约束。
- **为什么按块而不按篇**：按篇抽的话，一篇 686 万 token 的文档（`part00011-of-00020`）会一次带来 210 个满块，连续占 210 行，在 q4（每步 64 行）上等于连续 3 个多 global batch 几乎全来自这一篇。按块抽以后，这 210 块分散在整遍里。
- 种子用 `prefix_key` 而不是 prefix 在 blend 里的序号，增删别的 prefix 不会改变它的顺序。
- 文档编号就是当前 `data.idx` 的编号。

### 4.2 blend 交错：按 token 配平

**注意**：blend 的权重是 token 比例，不能按 w_i 的概率每次抽一个 prefix（平均块越长的 prefix 拿到的 token 越多）。

用确定性的配平（和 Megatron `BlendedDataset` 同一个思路，只是从"按样本"改成"按 token"）：

- 每个 prefix 记一个"已出 token 数" u_i。
- 每次取 (u_i + 它下一块的长度) / w_i 最小的 prefix（相同时取序号小的），出下一块，u_i 加上这块的长度。用一个最小堆实现。
- 保证：设 v 为最近一次出块时的 key（只增不减），对每个 prefix 都有 u_i / w_i ≤ v ≤ (u_i + 下一块长度) / w_i。所以任何时刻每个 prefix 实际出的 token 和 w_i × 总数之间，最多差一块（≤ W 个 token）。verify 直接检查这条不变式。

### 4.3 训练量用行数来定

块流不设终点，一直供到装出规定的行数为止。各 prefix 过几遍由配平自动决定。plan 是逐行顺序生成的，多生成一些行不会改变前面的行。

## 5. 装箱

### 5.1 算法（first_fit，按先进先出顺序）

```
buffer：等待装箱的块，先进先出；只放 len < W 的块
满块：从块流出来就直接成为一行，不进 buffer

循环：
  1. 有等待输出的满块：输出一行
  2. 否则 buffer 不足 B 个 token：从块流取一块
  3. 否则装一行：剩余空间 = W，按先进先出顺序扫一遍 buffer，放得下就放、从 buffer 移走，放不下就跳过；剩余补 pad
```

- 剩余空间只会越来越小，前面跳过的块后面也不会放得下，所以扫一遍就够了。
- buffer 的第一块（最早进来的）一定被放进当前行，所以每块最多等"排在它前面的块数"那么多个装箱行。

**例子**（为了好看，W = 10，B = 2W = 20；G 是一篇 23 个 token 的长文档，切成 G.k0 = 10、G.k1 = 10、G.k2 = 3，三块在块流里分别出现）：

```
块流：A4 B3 C5 D2 E7 F1 G.k1(10) H6 G.k2(3) I3 ... G.k0(10) ...（G.k0 在这一遍后面某处）

第 0 行  buffer：A4 B3 C5 D2 E7（共 21）
         放 A4（剩 6）→ 放 B3（剩 3）→ C5 放不下 → 放 D2（剩 1）→ E7 放不下 → 补 1 个 pad
         | A A A A | B B B | D D | pad |
         buffer 剩 C5 E7（12 < 20）→ 取 F1 → 取 G.k1，是满块，直接成为第 1 行
第 1 行  | G.k1 × 10 |
         buffer 13 < 20 → 取 H6 → 取 G.k2（3），和短文档一样进 buffer（共 22）
第 2 行  放 C5（剩 5）→ E7 放不下 → 放 F1（剩 4）→ H6 放不下 → 放 G.k2（剩 1）→ 补 1 个 pad
         | C C C C C | F | G.k2 G.k2 G.k2 | pad |
```

### 5.2 满块为什么不进 buffer

满块在 buffer 里只能等到排第一才放得下，期间白占 buffer 的容量。满块直接成行，结果一样（满块本来就独占一行），buffer 里留的都是能装箱的块。

### 5.3 buffer 大小

- buffer 越大，填缝的选择越多，padding 越少；代价是块被提前或推后的幅度越大（上限是 buffer 里的块数）。这只改变块出现的先后，不改变 blend 比例。
- 选定 **B = 8W = 262,144 个 token**（第 5.4 节的试算）。

### 5.4 first_fit 还是 best_fit：试算结果

在 80e9 r500 view 上按 q4 的 7,712,000 行试算（按篇抽的版本；按块抽之后正式 plan 的 padding 见第 9 节）：

| 算法 | B | padding | 平均等待（行） | 最长等待（行） |
|---|---|---:|---:|---:|
| first_fit | 2W | 0.166% | 1.0 | 442 |
| **first_fit** | **8W** | **0.045%** | **7.2** | **453** |
| first_fit | 32W | 0.035% | 32 | 477 |
| best_fit | 2W | 0.017% | 12.7 | 880 |
| best_fit | 8W | 0.017% | 87 | 3,428 |
| best_fit | 32W | 0.017% | 386 | 13,940 |

- best_fit = 先放最早的一块，再反复放"放得下的最长一块"。
- 选 **first_fit + 8W**：padding 已经只有 0.045%，块被打乱的幅度小。best_fit 的 padding 更低，但块被提前或推后得多得多。

### 5.5 收尾和不打乱

- plan 停在规定行数时，buffer 和满块队列里可能还有已抽出、没装出去的块。它们原样记在 manifest 的 `end_state` 里，verify 用它们还原完整的抽样计数。
- 装完后不再打乱行的顺序：块流本来就是随机的，连续的行已经相互独立。

## 6. plan 文件

### 6.1 目录和文件

```
pretrain/pack_plan/<plan 名>/
  manifest.json                      输入、参数、统计、每份的范围和 sha256、收尾状态
  prefixes.json                      prefix_id → key、path（token 文件 = path + ".bin"）、cache 语料路径、权重、文档数、块数、过的遍数
  teacher_units.json                 tc_unit → teacher unit 目录（相对 teacher 根目录），以及核对过的 teacher 根目录
  rows_0000000-0999999.blocks.npy    每块一条，32 字节（见下表）
  rows_0000000-0999999.nblocks.npy   uint16，每行几块
  rows_1000000-1999999.*             每 1,000,000 行一份
  ...
```

**每块的记录（32 字节）**

| 字段 | 类型 | 内容 |
|---|---|---|
| `doc` | u4 | 这篇文档在 blend 语料（`data.idx`）里的编号 |
| `cache_doc` | u4 | 这篇文档在 teacher cache 所用语料里的编号。只有那份语料的 `data.idx` 不同时才和 `doc` 不同（按 `data.bin` 起始字节对应） |
| `prefix` | u2 | 第几个 prefix，查 `prefixes.json` |
| `k` | u2 | 文档的第几块 |
| `len` | u2 | 这块有几个 token |
| `flags` | u2 | 位 0：文档最后一块（以 EOD 结尾）；位 1：len = W |
| `tok` | i8 | **token 地址**：这块在 `prefixes[prefix].path + ".bin"` 里从第几个 int32 token 开始 |
| `tc_unit` | u4 | **teacher 地址**：unit 编号，查 `teacher_units.json` |
| `tc_row` | u4 | **teacher 地址**：这块第 0 位在那个 unit 的行文件里是第几行 |

- 没有 teacher cache 时，`tc_unit` 和 `tc_row` 都是 0xFFFFFFFF。
- 每行几块存成 `nblocks`：第 0 行是最前面 `nblocks[0]` 块，第 1 行是接下来的 `nblocks[1]` 块……打开时累加一次就能定位任意一行。

### 6.2 第几步用哪些行

- plan 按行排，和 GBS、DP 规模无关；第 i 行是训练消费的第 i 个样本（Megatron 的 `consumed_samples` 顺序）。
- prod q4（16 节点 × 4 GPU，DP 64，micro-batch 1）：第 n 步第 r 个 rank 读第 n × 64 + r 行（从 0 数）。
- 第 i 行在第 i // 1,000,000 份里，份内第 i % 1,000,000 行。断点续训只需要 `consumed_samples`。

### 6.3 可读版本

`python -m dpl.data.pack_plan_32k dump <plan> --rows 0:10` 每行打印一条，例如：

```
row 0 | [0:1200) p0012 KO.jsonl-797c267b doc 3381 k0 EOD tok@1234567 tc data-1a2b3c4d@45678 | ... | [32700:32768) pad |
```

## 7. 训练端怎么读一行

对第 i 行，按顺序处理它的每块 `b`，**不需要查任何索引**：

1. **token**：`np.memmap(prefixes[b.prefix].path + ".bin", int32)[b.tok : b.tok + b.len]`。
2. **position**：每块从 0 开始。
3. **docmask**：`cu_seqlens` 就是各块的边界；pad 单独成一段（或用 `cu_seqlens_padded`），不进任何真实块的 attention。
4. **teacher 行**：`<teacher 根目录>/teacher_units[b.tc_unit]/` 下每个行文件的第 `b.tc_row` 到 `b.tc_row + b.len − (b.flags & 1)` 行。第 tc_row + j 行是 teacher 对第 tok + j + 1 个 token 的预测，上下文就是这一块。文档的 EOD 没有行。
5. **核对**：开训时核对 `prefixes.json` 里每个 prefix 的文档数和当前 `data.idx` 一致，对不上就拒绝启动。

`PackPlan(root)` 提供 `row(i)`、`segments(i)`（每块在行里的起止和 pad）、`tokens(b, label=True)`、`teacher_rows(b)`、`describe(i)`。

> **学生训练必须开 docmask：每块只看得到自己，position 从 0 开始。**

**每个位置的 label 和 loss**

| 位置（块内第 j 位） | label | loss |
|---|---|---|
| 不是块的最后一个 token | `bin[tok + j + 1]` | CE + KD（teacher 第 tc_row + j 行） |
| 中间块（flags 位 0 = 0）的最后一个 token | `bin[tok + len]`，即下一块的第 0 个 token（不在本行里） | CE + KD（teacher 有这一行，见下） |
| 文档最后一块的 EOD | 行里下一块的第 0 个 token；后面是 pad 或行尾时没有 label | 只算 CE（teacher 没有这一行） |
| pad | — | 不算 |

**和 3p5 计划的一处出入**：3p5 计划 5.2 写中间块的最后一个 token"不算 loss"。但按 4.4，teacher 打分时请求里带了下一块的第 0 个 token，这个位置在 cache 里有行（每篇正好 L − 1 行），上下文也和学生一致。所以它可以正常算 CE + KD，label 从 `data.bin` 取。见第 10 节第 3 条。

## 8. 验证（`verify`）

plan 生成后、交给训练前必须全部通过：

1. 每个文件的 sha256 和 manifest 一致；各份的行范围首尾相接。
2. 每行 Σlen ≤ W；每块的 len、flags 和 `data.idx` 里的文档长度一致。
3. **地址**：每块的 `tok` = `data.idx` 起始字节 / 4 + kW；`tc_unit` / `tc_row` 等于从 teacher cache 的 `docs.idx` 独立重建出来的值；`cache_doc` 等于按 `data.bin` 起始字节重算的编号。对全部块逐块核对。
4. **每遍完整**：把收尾时没装出去的块也算上，每个 prefix 内所有块被抽到的次数最多差 1。
5. **blend**：4.2 的配平不变式成立。
6. **teacher 覆盖**：plan 里用到的每篇文档都在 teacher cache 里，行数 = L − 1；多个 teacher 根目录的布局（所有 `docs.idx`）逐字节相同。
7. **端到端**：抽 64 行，**只用 plan 里的地址**读 token 和 teacher 行，teacher 记录的下一个 token（`next_token_id_and_rank >> 8`）必须等于 `data.bin` 里的下一个 token。
8. **可复现**：用同样的输入重跑前 1,000,000 行，必须逐块相同。

## 9. 实现和第一个实例

**代码**（都是新文件）

- `dpl/data/pack_plan_32k.py`：`plan` / `verify` / `dump` 三个子命令和读取类 `PackPlan`。
- `dpl/data/pack_plan_32k_core.cpp`：串行的内层循环（块流、配平、装箱），用 g++ 编译成 .so，通过 ctypes 调用。同一个算法在 Python 里另有一份参考实现，测试要求两者逐块相同。
- `tests/test_pack_plan_32k.py`：15 个测试，包括 5.1 的例子、C++ 和参考实现逐块对比、长文档的块被打散、小语料上 plan → verify 全流程（含过滤后的 view 和两份假 teacher cache），以及故意改坏 teacher 行或 plan 后 verify 必须报错。

**第一个实例：80e9 r500 view，q4 tpp200**

`pretrain/pack_plan/e9r500_q4tpp200_W32768_first_fit_B8W_s1/`（只读）

| 项 | 值 |
|---|---|
| 语料 | `qwen3_tokenized/ultra_v3_phase1_80e9_r500_filtered_view`（1,975 个 prefix，104,490,127 篇，79,997,809,740 token） |
| teacher cache | tc_80b 的 `mimo-v2.5-pro-base` 和 `mimo-v2.6-pro-rl`（布局相同，2,281 个 unit）；cache 语料 = `ultra_v3_phase1_80e9`，25 个 prefix 的编号和 view 不同 |
| 参数 | 7,712,000 行（120,500 × 64），first_fit，B = 8W，seed 1 |
| 结果 | 330,222,004 块，252,592,282,736 个 token；**padding 0.0453%**；每行平均 42.8 块（中位数 46，p99 69，最多 127）；306,842 行是单个满块；块平均等 7.2 行、最长 14 行；各 prefix 过 0.25–3.65 遍，每块最多被抽 4 次 |
| blend | 各 prefix 实际 token 和 w × 总数最多差 70,002 个 token（权重 ≥ 1e-4 的 prefix 相对偏差 ≤ 0.10%）；按篇抽时是 6,375,784 |
| 验证 | 第 8 节全部通过，0 个问题：两份 teacher 各端到端核对 2,093,106 个位置，0 个不一致；重跑前 1,000,000 行逐块相同 |
| 大小 | 9.9 GB：8 份，每份 `blocks.npy` 1.37 GB + `nblocks.npy` 2 MB |
| 耗时 | CPU 节点上生成 15 分钟（按块抽每次要二分查找块表，比按篇抽慢约 4 倍），验证 5 分钟 |

**语料不能再改**：plan 依赖当前 `data.idx` 的编号和 teacher cache 的布局。任何一个变了，plan 必须重算；verify 会查出来。

## 10. 决定和待确认

| # | 问题 | 结论 |
|---|---|---|
| 1 | 抽样单位 | **块**（不是文档），长文档的各块分散在整遍里 |
| 2 | 算法和 B | first_fit，B = 8W（5.4） |
| 3 | 中间块的最后一个 token 算不算 loss | 建议算 CE + KD，label 从 `data.bin` 取（第 7 节）；训练端定 |
| 4 | 文档最后一块的 EOD 算不算 CE | 建议算（3p5 计划 5.4，和 Megatron 一样）；训练端定 |
| 5 | blend 交错 | 按 token 配平，确定性（4.2） |
| 6 | 装完后是否打乱行 | 不打乱（5.5） |
| 7 | 训练量 | 按 ladder 的定义固定 iteration 数，pad 也算在 TPP 里；padding 约 0.05%，影响可以忽略 |
| 8 | full 语料 | 同一套代码；块表（每篇 8 字节）在 18.25B 篇时约 146 GB，届时要换更省内存的块表 |
