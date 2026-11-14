# Doc-isolated（iso-doc）测评手册：测什么、结果格式、存在哪、怎么跑

## 1. 这个测评测什么

**定义**：每篇验证文档单独成为一个样本，不带任何别的文档的上下文，对文档里的每个 token 算预测它的负对数似然（NLL），再取平均。训练时的验证是把文档首尾相接装进窗口（packed），文档之间能互相看到；doc-isolated 没有这个问题，所以不同装箱方式、不同训练长度的模型可以放在一起比。

**打分规则**（代码在 harness repo 的 `scripts/doc_isolated/`，见 4.3）：

- 每篇文档单独计分，看不到任何别的文档（怎么做到的见下面的「两种实现」）。
- 文档共 N 个 token（含结尾的 EOD）时，输入是第 1 到第 N−1 个，标签是第 2 到第 N 个。第一个 token 不被预测，EOD 不作为目标，所以每篇文档有 N−2 个被预测的 token。
- 被预测的 token 数超过窗口的文档，切成连续的几段，每段是独立样本，**段的开头看不到上一段**。
- 一个 split 的 loss = 该 split 所有被预测 token 的 NLL 平均值（按 token 加权）。

**两种实现，计分完全相同**：

- **packed（新方法，2026-11-13 起默认）**：每行固定 32,768 个 token，一行放多篇文档，用模型自己的段隔离（`cu_seqlens`）让它们互相看不到。一个 checkpoint 一个作业、一次加载，依次跑下表三遍。8K 结果 = S + L8，32K 结果 = S + L32。
- **padded（老方法）**：一篇文档一行，后面补 pad 到窗口长度。现在只用来复现老结果。

| 遍 | 放哪些文档 | 段长 | 得到 |
|---|---|---|---|
| S | 被预测 token ≤ 8,192 的文档，装箱 | 一篇一段 | 两个窗口共用的分数 |
| L32 | 其余的长文档，切成 32K 的段后装箱 | 32,768 | 它们的 32K 分数 |
| L8 | 和 L32 同样的行 | 8,192 | 它们的 8K 分数 |

- 两种实现逐个 split 对比过：59B 的 step 10,000 / 20,000 / 24,000 两个窗口共 690 个 split；ladder 的 OPT_MUONH q1（按 8K 训练、训练时没开文档隔离）和 PROD1_DPSK q1 共 345 个 split。token 数全部相同，loss 最大差 1.2e-7。
- 新方法和 checkpoint 怎么训练的无关：评测时一律打开段隔离（`--dataloader-inter-document-masking`，不改任何权重），行宽一律 32K。
- 成本：59B 一个 checkpoint 用 1 个节点约 1 小时（老方法约 13.6 节点小时）；ladder q1 约 15 分钟。

**两个窗口**：

| 窗口 | 长度 | 用途 |
|---|---|---|
| 8K | 8,192 | 和所有历史结果可比，是默认的对比口径 |
| 32K | 32,768 | 和按 32K 训练的模型的训练长度一致 |

- 两个窗口计分的 token 完全相同，差别只在长文档切几段。短于 8K 的文档两个窗口结果相同；长文档在 32K 下上下文更完整，loss 更低。
- 按 8K 训练的 checkpoint 不要看它的 32K 结果：超过 8,192 的位置它没见过，loss 会明显变差（实测：8K 训练的 `KD_MUONH_A0P8_S64Y_FIT_DOCMASK` q0–q2，32K 窗口的 overall_tsqw 比 8K 高 0.18–0.40）。新方法对按短于 32K 训练的 checkpoint 只出 8K 结果。
- 两个窗口必须出自**同一次模型加载**（用户 2026-11-04 的要求）：新方法在一个作业里跑完 S、L32、L8 三遍。

**数据**：9 个 suite，115 个 split，130,957 篇文档，234,291,595 个计分 token。suite 归并成 7 个集合（collection）。

| suite | 集合 | split 数 | 文档数 | 计分 token |
|---|---|---:|---:|---:|
| `cosmos_val_v2_cc_decon_qwen3` | val_v2 | 1 | 9,784 | 12,938,323 |
| `cosmos_val_v2_python_decon_qwen3` | val_v2 | 1 | 3,200 | 5,671,655 |
| `cosmos_val_v2_arxiv_decon_qwen3` | val_v2 | 1 | 205 | 7,976,849 |
| `uncheatable_decon_qwen3` | uncheatable | 55 | 26,689 | 114,928,353 |
| `codefresh_decon_qwen3` | codefresh | 30 | 26,720 | 19,029,400 |
| `v3small_decon_qwen3` | v3small | 8 | 4,572 | 2,765,345 |
| `private_internal_decon_qwen3` | private_internal | 2 | 1,968 | 3,204,144 |
| `pile_test_decon_qwen3` | pile_test | 16 | 54,977 | 60,395,136 |
| `pile_10k_decon_qwen3` | pile_10k | 1 | 2,842 | 7,382,390 |

- token 文件和 suite 的 yaml：`/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/cosmos3_llm_datasets/pretrain/qwen3_tokenized/val_set/eval_ppl_decon/`，yaml 在其中的 `suites/`，每个 split 记着 token 数、文档数、字节数。
- 原文（每行一篇文档的 JSONL，和 token 文件逐篇对应）：`~/data/eval_ppl_decon/`，给别的 tokenizer 重新分词用。
- `private_internal` 是非公开数据，文件权限 600；结果文件里只有它的 loss 数字。

## 2. 指标

每个指标都有三个数：`loss`（nats / token，越低越好）、`ppl` = exp(loss)、`bpb`（bits per byte）= loss × token 数 ÷ (字节数 × ln 2)。

| 指标 key | 显示名 | 定义 |
|---|---|---|
| `00_overall_tsqw` | overall_tsqw | **主指标**。7 个集合各自的 token 加权 loss，再按 √token 数加权平均 |
| `00_overall_tw` | overall_tw | 6 个集合（不含 pile_10k）的全部 token 合在一起取平均 |
| `00_overall_macro` | overall_macro | 上面 6 个集合的 loss 直接平均 |
| `01_val_v2_overall_tw` | val_v2 | cc + python + arxiv 三个 suite 合并，token 加权 |
| `01_val_v2_cc` / `_python` / `_arxiv` | val_v2 的三个子项 | 各自的 loss |
| `02_uncheatable_tw` | uncheatable | 55 个 split，token 加权 |
| `03_codefresh_tw` | codefresh | 30 个 split，token 加权 |
| `04_v3small_tw` | v3small | 8 个 split，token 加权 |
| `04_private_internal_tw` | private_internal | 2 个 split，token 加权 |
| `05_pile_test_tw` | pile_test | 16 个 split，token 加权 |
| `05_pile_10k_tw` | pile_10k | 1 个 split |
| `02_uncheatable_macro`、`04_v3small_macro`、`05_pile_test_macro` | （macro） | 同一集合的各 split loss 直接平均，不按 token 加权 |

各集合在两个 overall 指标里的权重：

| 集合 | 计分 token | overall_tsqw 权重 | overall_tw 权重 |
|---|---:|---:|---:|
| uncheatable | 114,928,353 | 31.4% | 50.6% |
| pile_test | 60,395,136 | 22.7% | 26.6% |
| val_v2 | 26,586,827 | 15.1% | 11.7% |
| codefresh | 19,029,400 | 12.8% | 8.4% |
| pile_10k | 7,382,390 | 7.9% | 不计入 |
| private_internal | 3,204,144 | 5.2% | 1.4% |
| v3small | 2,765,345 | 4.9% | 1.2% |

**比较规则**：

- tokenizer 相同的模型之间比 `loss`。我们的模型、Qwen3 系列和 MiMo 的 token id 相同。
- tokenizer 不同的模型之间只能比 `bpb`。overall_tsqw 的权重对所有模型都用 Qwen3 的 token 数。
- 新旧结果对比用 8K 窗口。

## 3. 结果文件格式

### 3.1 完整结果：一个 checkpoint、一个窗口、一个 JSON

文件名 `<Change ID>_<rung>.json`，约 14 KB。新老方法的结果文件字段完全相同。

- **Change ID**：实验的名字，例如 `PROD1_DPSK_LATENT2_E2X_QBK3`。
- **rung**：`q0`–`q6` 是 ladder 的模型大小档位；`ref` 表示外部模型；`it<N>` 表示生产 run 的第 N 步 checkpoint。
- **窗口**：32K 的结果在 Change ID 里多一个 `_W32K`，例如 `PROD1_DPSK_LATENT2_E2X_QBK3_W32K_q3.json`。文件里没有单独的 window 字段，评测窗口记在 `train.seq_length`（这个字段记的是评测时的 `--seq-length`，不是训练长度）。

| 字段 | 内容 |
|---|---|
| `change`、`rung` | Change ID 和 rung，和文件名一致 |
| `iteration` | checkpoint 的训练步数（外部模型为 0） |
| `protocol` | 固定为 `doc_isolated` |
| `checkpoint` | checkpoint 的完整路径（外部模型是 HF 模型目录） |
| `eval_job`、`eval_dir`、`evaluated_at` | 评测的 Slurm 作业号、原始评测目录、完成时间。新方法的 `eval_dir` 是该窗口长文档那一遍的目录（`<ID>__PKL8-…` 或 `<ID>__PKL32-…`）；老方法分三部分跑的，`eval_dir` 是第 1 部分的目录，`eval_job` 是最大的作业号 |
| `train` | 评测参数里的 `lr`、`min_lr`、`train_iters`、`num_layers`、`hidden_size`、`global_batch_size`、`seq_length`；外部模型为 null |
| `metrics` | 16 个指标（第 2 节），每个是 `{loss, ppl, bpb}` |
| `suites` | 9 个 suite，每个有 `loss_tw`（token 加权 loss）、`tokens`、`splits`（每个 split 的 `loss` 和 `tokens`） |

外部模型如果用的是自己的 tokenizer，`suites` 里每个 suite 和 split 还多 `bytes` 和 `bpb`。

真实例子（有删节）：

```json
{
 "change": "PROD1_DPSK_LATENT2_E2X_QBK3", "rung": "q3", "iteration": 70438,
 "protocol": "doc_isolated",
 "checkpoint": ".../results/PROD1_DPSK_LATENT2_E2X_QBK3/ckpts/q3/iter_0070438",
 "eval_job": 56590, "eval_dir": ".../runs/ladder_doc_eval/PROD1_DPSK_LATENT2_E2X_QBK3__P1of3-q3-56588",
 "evaluated_at": "2026-11-10T01:56:48",
 "train": {"lr": 0.00687, "num_layers": 30, "hidden_size": 1280, "seq_length": 8192},
 "metrics": {
  "00_overall_tsqw":      {"loss": 1.5320, "ppl": 4.6275, "bpb": 0.5966},
  "01_val_v2_overall_tw": {"loss": 1.2549, "ppl": 3.5074, "bpb": 0.5101}
 },
 "suites": {
  "v3small_decon_qwen3": {"loss_tw": 2.3000, "tokens": 2765345,
    "splits": {"c4_en": {"loss": 2.4880, "tokens": 463467},
               "dolma_common_crawl": {"loss": 2.8063, "tokens": 195508}}}
 }
}
```

### 3.2 合并集合结果（Boxiang 的评测，报告页面不显示）

生产 run（59B，`ULTRA59B_MUONH_KD_QBK3`）在 2026-11-13 之前只有 5 个这种结果（step 4,000 / 10,000 / 18,000 / 20,000 / 24,000，8K），来自 Boxiang 给 downstream 评测做 loss 对齐用的评测。现在 59B 已有 3.1 的完整结果；这些合并集合结果只留在文件里，报告页面上不显示。它把一个集合的全部文档合成一个验证集，所以一个集合只有一个 loss，没有 split 明细。打分代码和 3.1 是同一份，文档数和 token 数也和我们的同名集合相同，所以同一集合的数值是同一个量。

文件在 ladder repo 下：`results_xx/prod_isodoc/<RUN>[_W32K]_iter<7 位步数>.json`：

| 字段 | 内容 |
|---|---|
| `run`、`iteration`、`window` | run 名、步数、评测窗口（8192 或 32768） |
| `pooled_sets` | 固定为 true |
| `checkpoint`、`train` | checkpoint 路径；训练配置（总步数、训练长度、层数、KD teacher 等） |
| `metrics` | 只有测过的集合，key 和 3.1 相同（目前是 `01_val_v2_overall_tw`、`04_v3small_tw`），另带 `tokens`、`docs`、`set` |
| `not_scored` | 没测的集合的 key |
| `source` | 评测目录、作业号、时间、谁跑的、用的 Megatron 和 ladder 代码 |
| `other_evals` | 同一 checkpoint、同一窗口的其它评测（只用最新的一次） |

### 3.3 新文件夹里每个模型的文件

`~/runs/cosmos_eval_doc_isolate/<分类>/<模型>/`：

| 文件 | 内容 |
|---|---|
| `summary.md` | 给人看：所有指标一张表，每个窗口一组 loss / bpb 列 |
| `summary.json` | 同样的数字：`windows.<窗口>.metrics.<指标 key>.{loss, ppl, bpb}`，以及每个窗口的 `eval_job`、`evaluated_at`、`what`（完整结果还是合并集合） |
| `identity.json` | 测的是什么：`checkpoint`、`iteration`、`train`、`protocol`；`evals.<窗口>` 记着结果文件名、作业号、原始评测目录、结果文件从哪里复制来 |
| `breakdowns/<窗口>.json` | 该窗口的完整结果文件，即 3.1 或 3.2 的 JSON 原样复制。窗口名是 `8k`、`16k`、`32k` |
| `runs/<目录名>` | 指向原始评测目录的链接 |

窗口名取自结果文件的 `train.seq_length`，所以按 16K 评测的模型显示为 `16k`。

### 3.4 原始评测目录里的文件

2026-11-13 起在 `~/runs/cosmos_eval_doc_isolate/_runs/`，之前的在 `~/runs/ladder_doc_eval/`。目录名：

- 新方法：一个作业三个目录，`<ID>__PKS-<rung>-<作业号>`、`<ID>__PKL32-…`、`<ID>__PKL8-…`（按短于 32K 训练的没有 `__PKL32`）。
- 老方法：`<ID>[_W32K][__P<i>of3]-<rung>-<作业号>`。


| 文件 | 内容 |
|---|---|
| `argv_doc.txt` | 这次评测传给 Megatron 的全部参数，每行一个 |
| `logs/doc.log` | 评测日志 |
| `doc_eval_stats.json` | 每个 split 的文档数、被切段的文档数、样本数、被预测 token 数、填充率 |
| `eval/val_splits_iter<N>_doc_isolated.json` | 原始结果：`iteration`、`time`、`protocol`、`suites.<suite>.{labels, loss, tokens}`（三个等长的列表，一项对应一个 split） |
| `RESULTS.md` | 评测结束时生成的汇总表 |
| `data_cache/`、`tensorboard/` | 这次评测用的 per-split JSON 和索引缓存；tensorboard 输出 |
| `PACKED.json` | 新方法的每个目录都有：这是哪一遍（S / L32 / L8）、段长、这个作业跑了哪几遍。日志和 `RESULTS.md` 只在 `__PKS` 目录里 |
| `DUALWIN.md` | 老方法：只在第二个窗口（8K）的目录里，说明它和哪个 32K 目录出自同一次加载。日志和 `RESULTS.md` 只在 32K 的目录里 |

## 4. 存储位置

### 4.1 结果的家

`~/runs/cosmos_eval_doc_isolate/`（`~/runs/cosmos_eval_doc_isolated` 是指向它的链接）。旁边是 downstream 测评的 `~/runs/cosmos_eval_downstream/`。

```
baseline/<模型>/                   外部参考模型，21 个
ours_ladder/<Change ID>_<rung>/    ladder 的 checkpoint，599 个
ours_prod/<run>_iter<步数>/        生产 run 的 checkpoint
    summary.md  summary.json  identity.json  breakdowns/  runs/
<分类>/index.md                    每个模型一行：overall_tsqw 和 val_v2
_runs/                             2026-11-13 起所有评测作业的原始目录
_slurm/                            这些作业的 Slurm 日志
README.md
```

- 模型目录名和 downstream 那边一致，例如 `QWEN3_30B_A3B_BASE`、`PROD1_DPSK_LATENT2_E2X_QBK3_q3`、`ULTRA59B_MUONH_KD_QBK3_iter0024000`。
- 模型目录是从结果文件重建出来的，删掉不丢东西。
- `ours_ladder/` 里是全部已有结果，包括老实验。

### 4.2 其它位置

| 内容 | 路径 | 说明 |
|---|---|---|
| 老的原始评测目录和辅助文件 | `~/runs/ladder_doc_eval/` | 2026-11-13 之前的约 1,170 个原始目录，暂不搬动。还有 `stage/`（摆放的生产 run checkpoint）、`code_snapshots/`（每批作业运行时用的代码拷贝）、`packed_trees/` 和 `dualwin_trees/`（代码链接树）、`data_cache/corpus_v2/`（共享的数据索引缓存）、`sbatch/`（老作业日志、`finish_*.log`）、`lustre_stuck_nodes.txt`（提交时自动避开的节点） |
| 测试输出 | `~/runs/ladder_doc_eval_speedtest/` | 验证和测速的作业写这里。收集脚本不读，不会进结果的家 |
| 结果 JSON，原件 | `~/Project/trichord/dense_pretrain_result/` | 收集脚本先写这里。它在 trichord 的 git 里，不要移动、删除或换成链接 |
| 结果 JSON，组内可读的拷贝 | `/scratch/fsw/portfolios/cosmos/projects/cosmos_base_training/users/xingqianx/dense_pretrain_result/` | 给别人读 |
| 结果 JSON，报告用的拷贝 | `~/Project/cosmos_llm_scaling_ladder/results_xx/` | 报告生成器读这里；`results_xx/prod_isodoc/` 是生产 run 的合并集合结果 |
| 生产 run 合并集合评测的原始目录 | `~/runs/cosmos_eval_downstream/_doc_eval59/` 和 `/scratch/fsw/portfolios/cosmos/projects/cosmos_base_data/users/boxiangw/cosmos_ultra_pre/runs/eval59/doc_eval/` | 前者是从我们账号提交的，后者是 Boxiang 的 |
| 报告 | `~/Project/cosmos_llm_scaling_ladder/reports_xx/qwen3_scaling_ladder_eval.html`、`_en.html`、`.md` | 页签依次是 doc-isolated、prod iso-doc、prod downstream、外部参考、老实验、LR 拟合实验、packed（2026-11-14 起的名字和顺序）。地址 `http://oci-jhb-slurm-2-vscode-02:8765/qwen3_scaling_ladder_eval.html` |

权限：`~/Project/` 和 trichord 只有本人可读，不要放开；要给别人看的东西复制到别处。

### 4.3 代码

代码的家是 lm-evaluation-harness repo 的 `scripts/doc_isolated/`（`~/Project/lm-evaluation-harness`）。2026-11-13 从 ladder repo 复制过来，原处的文件保留不动、不再使用。已提交到 `xingqianx/eval` 分支（2026-11-14，提交 9edbbea7，尚未推送）。下表路径相对于这个目录：

| 文件 | 作用 |
|---|---|
| `core/doc_eval.py` | 打分核心：怎么切文档、哪些 token 计分（老方法的数据集也在这里） |
| `core/packed/` | **新方法**：`doc_eval_packed.py`（装箱、行的格式）、`pretrain_doceval.py`（一次加载跑 S、L32、L8 三遍）、`test_doc_eval_packed_cpu.py`（和老方法逐段比对的 CPU 测试） |
| `core/pretrain_doceval.py` | 评测入口，套在 Megatron 的 `pretrain_gpt.py` 外面 |
| `core/val_metrics.py` | per-split 验证和指标定义 |
| `core/sh/eval_ckpt.sh`、`core/tools/eval_argv.py` | 评测作业的主体；把训练参数改成评测参数 |
| `core/dualwin/` | 老方法：一次加载跑两个窗口 |
| `run_doc_eval.sh` | **提交评测** |
| `stage_prod_ckpt.py` | 把生产 run 的某一步 checkpoint 摆成 ladder 的目录格式 |
| `mimo_doc_eval.py`、`mimo_doc_eval.sbatch` | 外部模型的评测（vLLM） |
| `collect_doc_eval.py` | **收集**成结果 JSON |
| `collect_prod_isodoc.py` | 收集生产 run 的合并集合结果 |
| `organize_doc_isolated.py` | **整理**成 `~/runs/cosmos_eval_doc_isolate/` |
| `compare_packed_padded.py` | 同一个 checkpoint 的新老方法逐 split 对比（只读） |
| `finish_when_done.sh` | 等作业结束，自动收集、整理、发布 prod iso-doc 页签（在 tmux 里跑） |
| `README.md` | 英文说明 |

- 报告代码仍在 ladder repo 的 `reports_xx/`：`gen_eval_html.py`、`prod_isodoc_section.py`（prod iso-doc 页签）、`render_doc.py`（本手册的 HTML）。`reports_xx/` 里同名的评测脚本是旧拷贝，不要再用。
- `core/doc_eval.py` 和 `core/val_metrics.py` 与 ladder repo 的 `code/scaling_ladder/`、Boxiang 给 59B 用的快照 `cosmos_ultra_pre/ladder-prod-a38190d8` 逐字节相同。

## 5. 怎么跑

### 5.0 通用

- 集群 oci-jhb-slurm-2，账号 `cosmos_base_training`。每个评测作业 1 个节点、4 张 GPU。
- **从冻结的代码拷贝提交**，作业运行期间改 repo 里的代码不会影响它：

```bash
S=~/runs/ladder_doc_eval/code_snapshots/doc_isolated_$(date +%Y%m%d_%H%M)
rsync -a --exclude __pycache__ ~/Project/lm-evaluation-harness/scripts/doc_isolated/ $S/ && cd $S
```

- **先加 `--dry` 看一遍要提交什么**，再去掉它正式提交。
- 默认就是新方法：一个 checkpoint 一个作业，一次出该出的窗口。
- `EVAL_QOS=normal`：在 slurm-2 上会被改写成 `normal_quota`。脚本默认 wckey p0（用户 2026-11-13 的要求），会抢占同一 QOS 里的 p1 作业，例如 tc_full 教师缓存任务（它们会自动重新排队）。这些是 2026-11-13 的情况，集群规则经常变，提交前先核对。

### 5.1 ladder 的 checkpoint

```bash
EVAL_QOS=normal EVAL_EXTRA_ARGS= bash run_doc_eval.sh --dry <ID>/q3=run:<run 目录的绝对路径>
```

- Change ID 不带 `_W32K`。按 32K（或更长）训练的出 8K 和 32K 两个结果，32K 的写到 `<ID>_W32K`；按更短长度训练的只出 8K。
- checkpoint 的写法：`=run:<run 目录>`（取该 run 最后一个 checkpoint）；`=p:<ID>`（已 promote 的 Change ID）；不写等号则取同名的 promote 目录。
- 作业名 `xxdoc-<ID>__PK-<rung>`。
- 滑动窗口注意力的 checkpoint（`SWA_FIT_SINK_*`）还没用新方法验证过，测之前先拿一个和老方法对比（`compare_packed_padded.py`）。

可用的环境变量：

| 变量 | 作用 |
|---|---|
| `EVAL_QOS=normal` | 用哪个 QOS。不设则是 hero-res，slurm-2 上不可用 |
| `SBATCH_WCKEY` | 优先级，默认 p0 |
| `EVAL_LADDER_CODE=<目录>` | 用哪棵 ladder 代码树，要和训练时一致（PROD1 的 overlay 树） |
| `EVAL_EXTRA_ARGS="--flag 值"` | 改写评测参数，例如 `--distributed-timeout-minutes 240` |
| `EVAL_TIME=HH:MM:SS` | 直接指定时限。不设则按 rung 取下表；生产 run 的 `it<N>` 默认 3 小时 |
| `EVAL_CACHE_PER_JOB=1` | 每个作业用自己的数据索引缓存，不用共享缓存 |
| `EVAL_RUN_ROOT=<目录>` | 把评测目录写到别处（测试用，写到 `~/runs/ladder_doc_eval_speedtest/` 下；收集脚本不会读） |
| `EVAL_PACKED=0` | 改用老方法。老方法的 `EVAL_DUALWIN_SEQ=8192`（同一次加载再跑 8K，Change ID 要带 `_W32K`）、`EVAL_SPLIT3=1`（拆成 3 个并行作业）、`EVAL_ONLY_PARTS=2` 设了任何一个就自动用老方法 |

不设 `EVAL_TIME` 时的时限（按老方法定的，新方法远用不完）：q0 2:30；q1、q2 3:30；q3 4:30；q4 5:30；q5 8:00；q6 12:00。新方法实测：ladder q1 13–17 分钟，59B 1:01。

### 5.2 生产 run 的 checkpoint

生产 run 没有 ladder 那样的 checkpoint 目录，先摆一个，再用同一个提交脚本：

```bash
BX=/scratch/fsw/portfolios/cosmos/projects/cosmos_base_data/users/boxiangw/cosmos_ultra_pre   # Boxiang 的生产目录，只读

# 1. 摆放：在我们自己的目录下生成 argv.txt、run.json、指向 checkpoint 的链接，打印目录名
python3 stage_prod_ckpt.py --view $BX/runs/eval59/view/ultra59b_muonh_kd_qbk3 --iter 30000 --change ULTRA59B_MUONH_KD_QBK3

# 2. 提交：rung 写成 it<步数>
EVAL_QOS=normal EVAL_EXTRA_ARGS= EVAL_CACHE_PER_JOB=1 \
bash run_doc_eval.sh --dry ULTRA59B_MUONH_KD_QBK3/it30000=ckpt:$HOME/runs/ladder_doc_eval/stage/ULTRA59B_MUONH_KD_QBK3/ckpts/it30000
```

- 一个 checkpoint 调用一次脚本。一次传多个 checkpoint 时，它们会排成一串，后一个等前一个结束。
- 摆放脚本去掉了训练专用的参数（pack plan、wandb 等），并且**不带训练数据集**，评测不会去建训练集的索引。
- 评测参数和 Boxiang 那套 59B 评测相比，模型部分完全一致（2026-11-13 逐项比对过）。
- 实测：step 10,000 / 20,000 / 24,000 每个 1 个节点、1:00–1:01，8K 和 32K 两个结果都和老方法逐 split 一致。

### 5.3 外部模型

用 vLLM 离线打分，一篇文档一个序列，规则和上面相同，每个 split 的 token 数会和 ladder 的记录核对。新方法只用于 Megatron 的 checkpoint。

```bash
sbatch -N 1 -t 08:00:00 --export=ALL,NAME=<名字>,MODEL=<HF 模型目录>,PP=1 mimo_doc_eval.sbatch
```

- 32K 窗口：加 `SEQ=32760`，名字写成 `<名字>_W32K`。
- 自带 tokenizer 的模型：加 `HFO='{}' EXTRA="--text-dir ~/data/eval_ppl_decon"`，需要 BOS 的再加 `--bos`。这类模型的 loss 是按它自己的 token 算的，只能比 bpb。
- 大模型用 2 个节点（`-N 2`，`PP=2`）。各模型用过的具体参数写在 `mimo_doc_eval.sbatch` 开头的注释里。
- 输出目录 `~/runs/cosmos_eval_doc_isolate/_runs/<名字>-ref-<作业号>/`。

### 5.4 Boxiang 的合并集合评测（不属于这条流水线）

Boxiang 的 `cosmos_ultra_pre/launch/eval59/doc_eval_59b.sbatch`：一个 checkpoint、1 个节点、约 1 小时，只测他配置的合并集合（目前 val_v2 和 v3small），窗口默认 8K。它是 downstream 评测启动前做 loss 对齐检查的参照，由 downstream 那边的会话在用。它的结果由 `collect_prod_isodoc.py` 收进来，报告页面不显示。

### 5.5 收集、整理、发布

最省事的是提交后在 tmux 里挂一个等待脚本，作业结束或出现新结果就自动收集、整理、发布 prod iso-doc 页签：

```bash
tmux new -d -s iso_finish "bash finish_when_done.sh <作业号> [<作业号> ...]"   # 日志 ~/runs/ladder_doc_eval/sbatch/finish_<第一个作业号>.log
```

手动执行（在 `scripts/doc_isolated/` 或它的冻结拷贝下）：

```bash
python3 collect_doc_eval.py        # 原始目录 -> 结果 JSON（先写 trichord，再拷到另外两处）
python3 collect_prod_isodoc.py     # 生产 run 的合并集合结果 -> results_xx/prod_isodoc/
python3 organize_doc_isolated.py   # 结果 JSON -> ~/runs/cosmos_eval_doc_isolate/
cd ~/Project/cosmos_llm_scaling_ladder/reports_xx
python3 gen_eval_html.py --only prod                                       # 只重新发布报告的 prod iso-doc 页签
rsync -a --exclude __pycache__ ./ ../../trichord/reports/dense_pretrain/   # 报告镜像
```

- `collect_doc_eval.py` 对每个（Change ID，rung，窗口）取作业号最新的那次评测：新方法在 S 和该窗口那一遍（L32 或 L8）都有结果时就收集这个窗口，所以 32K 比 8K 早约 17 分钟出现在报告里；老方法要 3 个部分都完成、步数一致。每个 split 的计分 token 数会和 suite 的 yaml 核对。
- 新老方法对比：`python3 compare_packed_padded.py <ID> <rung> [<rung> ...]`，退出码 0 表示全部一致。
- 报告由几个会话共同生成，**只用 `--only <页签>` 发布自己的页签**，不要整份重建。

### 5.6 出问题时先查什么

| 现象 | 先查 |
|---|---|
| 作业一开始就退出，日志里有 `LUSTRE CHECK FAILED` | 节点的 Lustre 卡住了。把节点名加进 `~/runs/ladder_doc_eval/lustre_stuck_nodes.txt` 再重新提交，提交时会自动避开 |
| 作业在第一次读文件时卡住不动 | 同上（2026-11-13 的 nvl72d007-t01，OST0157/0158） |
| 作业长时间没有输出，最后 NCCL 超时 | 数据索引缓存里有没有 0 字节文件：`find ~/runs/ladder_doc_eval/data_cache/corpus_v2 -name '*.npy' -size -128c` |
| 作业一直 PENDING，原因是 requeued held | `scontrol release <作业号>` |
| 同一批里有的作业被抢占重排 | 这批作业的 wckey 不一致，p0 会抢占同一 QOS 里的非 p0 作业 |
| `sacct` 的状态看不出是不是取消 | 用 `sacct -o State%30`，默认宽度会把 `CANCELLED by <uid>` 截断 |
| 作业显示 PREEMPTED | slurm-2 会把它重新排队，之后还会跑完，不算结束 |
| 评测一上来就建几千个索引文件 | 这个 run 带着训练数据集。用预装箱数据训练的 run 会这样；生产 run 用 5.2 的摆放方式可以避开 |
| 某个 checkpoint 没有结果 JSON | 新方法：`__PKS`、`__PKL8`（和 `__PKL32`）是否都有 `eval/val_splits_*.json`；老方法：3 个部分是否都有、步数是否一致 |
| 新方法作业报 `--dataloader-inter-document-masking` 缺失 | 没有用 `run_doc_eval.sh` 提交；它会给每个 checkpoint 加上这个开关 |

## 6. 当前状态（2026-11-13 23:30）

- **新方法是默认**（23:26 起）：所有新的 iso-doc 评测都用新方法，结果文件的结构和内容与老结果相同。已有的老结果不需要重测。
- **59B**（`ULTRA59B_MUONH_KD_QBK3`）：step 10,000 / 20,000 / 24,000 的 8K 和 32K 都有完整结果（新老方法各测了一次，逐 split 一致，结果文件取新方法那次）；step 30,000 也已测完（新方法）。step 4,000 / 18,000 只有 Boxiang 的合并集合结果，报告页面不显示。
- **step 30,000** 已测完（作业 65045，2026-11-14 00:07 上报告页面）。
- **还没定**：`~/runs/ladder_doc_eval/` 里 2026-11-13 之前的原始目录要不要搬进 `_runs/`。
