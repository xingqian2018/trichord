# MoE 训练手册对照筛查：我们的 MoE 有没有手册里说的问题

2026-11-05 · 对照：Hugging Face《The Ultimate MoE Training Handbook》（2026-11-05 发布，基于 Megatron-LM）huggingface.co/spaces/huggingface/moe-blogpost · 方法：10 个只读 agent 按手册章节分工，对照两棵 Megatron 树、所有 qwen3 golden argv 和约 780 个 run 的日志（第 7 节）· 没有改代码，没有提交作业 · English: 2026-11-05_moe_handbook_audit_en.html

**一句话**：MoE 机制本身没有 bug，router 精度、dropless、aux 缩放、expert bias，以及我们自己的 KD / MuonH / eval 封装都是对的。但手册点名的陷阱里有 5 个确实中了，其中两个会直接影响结论：**aux loss 的作用范围随 rung 变化**（干扰梯子外推），**MuonH 的 router 学习率不随宽度缩放**（影响 235B，h=4096）。

最便宜的下一步：对现有 KD q2/q3 checkpoint 跑一次 eval-only 的 MaxVio，就能知道主线到底均不均衡。

## 1. 总览

状态：受影响 有证据说明问题存在；部分 风险条件存在但目前没有造成后果；未知 没有数据，无法判断；不受影响；不适用 我们不用这个功能。

| # | 问题 | 手册章节 | 状态 | 影响面 | 下一步 |
|---|---|---|---|---|---|
| 1 | 主线负载均衡看不见，LBL≈1 被当成"已均衡" | 3.0, A.1, 3.3 | 受影响 | 所有主线 run（MuonH+KD、BASE） | 开 router health 和逐层日志；对现有 KD checkpoint 跑 eval MaxVio |
| 2 | aux loss 的作用范围随 rung 从 32K 降到 8K token | 3.3 Scope | 受影响 | 梯子 q4–q6 的拟合点 | q0/q1 上 mbs 1 和 mbs 4 对照 |
| 3 | MuonH 的 router 学习率不随宽度缩放 | 3.6 Optimizer | 受影响（h ≥ 3072） | 235B（h=4096）的外推 | h=4096 用 router ×0.25 跑一次全长验证 |
| 4 | 固定步长的 expert bias 平衡不了第一个 MoE 层 | 3.3 Loss-free bias | 受影响（全 MoE 的 DSV3 分支） | DSV3 / router_ne 研究分支；PROD 基本不受影响 | quantile balancing 对照 |
| 5 | 没有节点健康和吞吐闸门，也从没做过 profile | 2.1, 3.7 | 受影响 | 已浪费约 3k GPU-h | 预检、step-time 看门狗、q3/q4 各抓一次 trace |
| 6 | 所有 run 都关了 rerun | 3.6 Stability | 部分 | 出 NaN 时无法区分坏 GPU 和可复现的 NaN | 非 KD 分支改回 `validate_results` |
| 7 | BASE q5/q6 warmup 期大尖峰，全仓库没有专家输出范数监控 | 3.4, 3.6 | 部分 | BASE AdamW 大 rung | 前兆报警；专家输出范数 hook |
| 8 | softmax router logits 整体上漂，没开 z-loss | 3.1 Precision | 部分 | 主线 | 记录 logsumexp；q0/q1 z-loss 对照 |
| 9 | gate 锐度从没测过 | 3.1 Sharpness | 未知 | 主线 | 从 checkpoint 离线算 top-1 权重和 Max1/Max2 |
| 10 | MuonH 下冷门专家的步长偏大 | 3.6 Optimizer | 未知 | 主线 | 逐专家 token 数和更新范数监控 |

## 2. 确认存在的问题

### 2.1 主线负载均衡看不见，LBL≈1 被当成"已均衡"

手册怎么说

- Megatron 的 aux loss 是 E·Σ f_i·P_i。router 概率平坦时它恒等于 1，跟实际负载无关，所以要看 MaxVio（最忙专家的负载比均值高多少），不要看 aux loss（3.3；A.2 "aux loss near 1 but one GPU slow"）。
- 在你做均衡的范围上看 MaxVio：per micro-batch 和 per global batch 都要看（A.1）。

我们的情况

- 主线（KD s64y、s64y_fit、docmask、packplan）以及 BASE v1、Muon、MuonH 都没开 `LADDER_ROUTER_HEALTH`，也没有一个 golden 带 `--moe-per-layer-logging`。日志里只有一个各层平均后的 `load_balancing_loss`。只有 dsv3 / swa / router_ne 的 exp 文件和一个 prod packplan exp 开了 router health。
- 主线的 LBL 全都接近 1（训练最后 20% 的均值）：

| 分支 | q0 | q1 | q2 | q3 | q4 | q5 |
|---|---|---|---|---|---|---|
| BASE v1 | 1.054 | 1.031 | 1.022 | 1.018 | 1.019 | 1.019 |
| KD s64y_fit | 1.097 | 1.042 | 1.024 | 1.016 |   |   |

- 但用同款 softmax router（aux 1e-3）、开了 router health 的 run 显示，LBL≈1 的同时负载明显不均：

| run | 窗口 | LBL | MaxVio 层均值 | 最差层 MaxVio | 死专家 |
|---|---|---|---|---|---|
| MUONH_FIT q0 s1 | step 4500 | 1.110 | 1.61 | 3.94（L00） | 1 |
| MUONH_FIT q0（3 个 seed） | 训练结束 | — | 1.3–1.5（层中位数） | 4.03 / 5.36 / 5.21（L00） | — |
| MUONH_FIT q1 | 训练结束 | 1.05 | — | 3.30（L00） | step 1000 后一直 1–3 个 |
| MUONH_FIT_SWA q3 | 最后 20% | 1.024 | 0.66 | 2.0 | — |
| MUONH_FIT q0，eval（val_v2_decon） | 最终 checkpoint | — | 2.3–3.8 | 4–5.7 | codefresh 上约 140/1536 |

- q0 上第 0 层的不均衡在训练中越来越大：前 10% 的报告里是 1.9–3.2，结束时 3.5–5.4。
- 我们按 micro-batch 做均衡，但 `router_health.py` 只测 global batch 的 MaxVio。per micro-batch 和每个 EP rank 的负载都没测过，而决定 EP step time 的正是后者。
- 文档里把 LBL 当成均衡指标的地方：`DECISIONS.md:1696` 写的是 "load_balancing_loss (1.0 = balanced, 128 = one expert)"，其实 top-8 的上界是 E/k = 16，所以 W3 记到的 14.5–15.7 已经接近完全塌到 8 个专家上。另外还有 `reports/muon_vs_adamw_2026-10-17.md:334`、`dsv3_router_assessment_2026-10-26.md:18` 和 `gptoss_tpp200_aux10x.sh` 的头注释。eval 报告的 router 部分用的是 MaxVio，不受影响。

建议

- 新的主线 run 默认开 `LADDER_ROUTER_HEALTH=100` 和 `--moe-per-layer-logging`。router health 在 q3 上每次约 1.07 s，每 100 步一次，开销不到 1%。
- 对现有 KD q2/q3 checkpoint 跑 eval-only 的 `LADDER_ROUTER_MAXVIO_VAL=1`，不用重训就能知道主线的情况。
- `router_health.py` 增加 per-micro-batch 和每个 EP rank 的 MaxVio。均衡判据改成 MaxVio（层均值加最差层），不再引用 LBL。

### 2.2 aux loss 的作用范围随 rung 从 32K 降到 8K token

手册怎么说

- 3.3 Scope：如果一个 micro-batch 里全是同一领域的 token，在这个范围内强行均衡，会阻止专家形成领域分工（Qiu et al., "Demons in the Detail"）。global batch 范围对质量更好，代价约 +5.8% step time。A.2 "experts that do not specialize by domain"。

我们的情况

- `aux_loss` 在单卡内统计（TP=CP=1），每个均衡组 = mbs × 8192 token。mbs 按 rung 设定（`sh/arch/qwen3moe.sh` 的 `arch_rung_defaults`）：

|   | q0 | q1 | q2 | q3 | q4 | q5 | q6 |
|---|---|---|---|---|---|---|---|
| DP | 16 | 32 | 64 | 64 | 128 | 256 | 256 |
| mbs | 4 | 4 | 4 | 4 | 2 | 1 | 1 |
| 每步的 micro-batch 数 | 4 | 2 | 1 | 1 | 1 | 1 | 1 |
| 每个均衡组的 token 数 | 32,768 | 32,768 | 32,768 | 32,768 | 16,384 | 8,192 | 8,192 |
| 每个专家期望分到的 token 数 | 2048 | 2048 | 2048 | 2048 | 1024 | 512 | 512 |

- 每条 8K 样本只来自 1975 个 prefix 中的一个（BlendedDataset；`kd/data.py` 的 docstring 对 KD 主线也是这么写的）。所以 q5/q6 的 aux loss 实际上是作用在单一领域序列上的 sequence-wise Switch loss，系数还是 DeepSeek-V3 sequence-wise 系数（1e-4）的 10 倍。
- 侧证：DSV3_SWS_AUX1EM3 分支把 sequence-wise 系数从 1e-4 提到 1e-3，q0/q1/q2 上分别差了 +0.0078 / +0.0098 / +0.0066 nats，说明范围变窄、系数变大确实会损失 LM loss。
- README 当初否掉 `global_aux_loss` 的理由是它的实际强度是 1e-3/DP，跨 rung 不一致。这个说法核实正确（两棵树代码相同），但现在的 per-micro-batch 方案同样跨 rung 不一致。GBS 固定在 2^21，打了 DP 补丁（`code/scaling_ladder/tools/patches/global_aux_loss_dp_scale.patch`）的 global aux 反而对 rung 不变。
- 还有一个小差异：KD 的软标签给 router 的 LM 梯度更小，同样的 1e-3 在 KD 下相对更强。q1 iter 14680 上 LBL：CE（OPT_MUONH）1.055，KD α0.8 1.041。aux 的梯度缩放本身在 CE 和 KD 下完全一样，不是 bug。

建议

- 先量化：q0/q1 在相同 GBS 下比较 mbs 1 和 mbs 4。
- 如果有影响：在固定大小（例如 32K token）的 DP 子组上 all-reduce `tokens_per_expert`，或者改用打了 DP 补丁的 global aux。后者还要把计数器改成 int64 / fp32（见第 4 节 bf16 计数器一行）。

### 2.3 MuonH 的 router 学习率不随宽度缩放

手册怎么说

- 3.6 Optimizer：Muon 可能让 router 不稳定，Qwen3.8-Next 就把 router 换回了 AdamW；Megatron 提供 `--moe-router-skip-muon`。A.2 "experts collapse layer by layer under Muon"。

我们的情况

- Muon 分支：`--moe-router-skip-muon`，router 用 AdamW，wd 0.1。符合
- MuonH 主线：router 用 AdamH（`--hyperp-router adam_hyperball`），范数固定在 0.02·√(E·h)，wd 0，学习率是 hidden 的 1 倍。主线没有设 `HYPERP_GROUP_LR_MULT`，任何 rung 都没有 router 学习率倍率。
- h=4096（W3 cell）50 步内就塌了：LBL 14.5–15.9，grad norm 最高 511,442，loss 到 10.4。h ≤ 2048 是干净的：50 步时 ≤ 3.8，300 步时约 1.1（`DECISIONS.md` 约 1689–1733 行）。
- 100 步探针的结果：
  - router ×0.25：step 100 时 LBL 2.15，修好了。
  - router 换成 AdamW：没修好，LBL 13–16。
  - aux 提到 1e-2：只修好一部分。
  - RTADAMW 分支（用 HyperP 学习率的 AdamW router）也塌了：LBL 3.8–8.2，分别有 24 和 44 次 loss 尖峰。
- 全长验证没做：RT0P25 开跑前被取消，DSV3 的 h=4096 cell 还是 Planned。
- 相关：router 的输入是共享的 pre-MLP RMSNorm 输出，没有单独的 router norm，也没有 1/√d。初始 logits 的标准差是 0.02·√h，从 q0 到 q6 翻了一倍。

建议

- 235B 之前先定 router 的宽度规则：×0.25，或者学习率 ∝ 1/h，或者换成 DSV3 router。在 h=4096 上开着 router health 跑一次全长验证，不能只靠 100 步探针。

### 2.4 固定步长的 expert bias 平衡不了第一个 MoE 层

手册怎么说

- 3.3 Loss-free bias：固定步长的符号更新会跟不上 router 的变化，"loss-free bias run still unbalanced after many steps"。Li et al.：128 个专家、γ 1e-3 时最忙的专家约是均值的 20 倍。手册给的解法是 quantile balancing。

我们的情况

- 全 MoE 的 MUONH_FIT_DSV3，第一个 MoE 层（L0）在训练中段（10–70% 步数）的 global-batch MaxVio：

|   | q0 | q1 | q2 | q3 |
|---|---|---|---|---|
| L0 | 5.45 | 5.16 | 3.60 | 3.30 |
| 其他层 | 0.46–0.52 | 0.46–0.52 | 0.46–0.52 | 0.46–0.52 |

- q3 的 390 次报告里，L0 有 381 次是最不均衡的层；step 500–40k 之间 92% 的报告里 L0 都大于 2。直到 LR 衰减后才下来（结束时 0.24–0.69）。汇报里写的 "MaxVio 0.15–0.18" 只是结束时的值。
- 调步长修不好（SWS，q0，L0 中段 MaxVio）：γ 1e-4 → 6.93；1e-3 → 3.85；1e-2 → 中段 4.45，结束时 9.21（振荡）。
- 同一个 router 用 AdamW（BASE_DSV3_ONETREE）时 L0 只有 0.46–0.88，说明问题出在 MuonH 下 gate 变化太快。
- 能修好 L0 的办法：
  - seq-aux 1e-3：L0 降到 0.80，但损失 +0.008 至 +0.010 nats。
  - dense 第 0 层：PROD（dense L0 + 共享专家）中段第一个 MoE 层 0.70–0.84，最差层 1.05–1.14（q4/q5）。
- BIASONLY（不带 seq-aux）q3 的 L0 是 11.97，约为均值的 13 倍，接近手册说的 20 倍。
- eval 集上 bias 不迁移：DSV3 q1 对照训练中 MaxVio 0.17，val_v2 上 1.98，codefresh 上 5.05，有 93 个死专家。
- quantile balancing 两棵 Megatron 树都支持（`--moe-router-load-balancing-type quantile_balancing`），但被 `sh/arch/qwen3moe.sh` 的 `MOE_ROUTER_LOAD_BALANCING` 白名单挡掉了（约 59 行），`router_ne_xx/NOTES.md:123` 也记了跳过。Megatron 要求它和所有 aux loss 互斥，所以 seq-aux 1e-4 也要去掉。
- batch-size 研究有混淆：GBS 跨了 128 倍（4 到 512 条 32K 序列），γ 却固定在每步 1e-3，相当于每个 token 的 bias 适应速度差了 128 倍。

建议

- QB 对照：q0 三个 seed 加 q1，在全 MoE 的 SWS base 和 PROD dense-L0 base 上各跑一组，aux 系数 0，不用 expert bias，EMA 取 0 和 0.9。看 L0 和最差层的 MaxVio 曲线、EP ≥ 4 时的 step time，以及 R1 val。白名单里加一个词就能跑。
- 健康判据改成"中段最差层 MaxVio"，不要只看结束时的均值。

### 2.5 没有节点健康和吞吐闸门，也从没做过 profile

手册怎么说

- 2.1：每次分配都跑 DCGM `diag -r 1` 和 nccl-tests 的 all-to-all 基线，或者用 Megatron 的 `--gpu-sniff-test-interval`。3.7：每个新配置都在实际节点数上 trace 一个 step。

我们的情况

- `base.sh` 和 `submit.sh` 里没有 DCGM、nccl-tests 或 sniff test（两棵树里都有 `gpu_sniff_test.py`，但没用上）。唯一的检查是一次性的 2 节点 all_reduce 探针（job 34988：IB，NCCL 2.30.7，290 GB/s）。坏节点只在出错之后才 exclude。
- 日志里找到的整段都慢的分配：

| run | job | step 中位数 | 参考 | 慢多少 | 备注 |
|---|---|---|---|---|---|
| BASE v1 q5 | 1158001 | 1762 ms | 1040 ms | +69% | 27.2k iter，约 1.4k GPU-h；唯一被人工发现的（DECISIONS:1590） |
| DSV3_SWS_NE_1EM5 q4 | 1460541 | 4085 ms | 1289 ms | +217% | 约 0.6k GPU-h |
| OPT_MUON q4 | 1173380 | 1357 ms | 953 ms | +42% | 可能部分来自代码差异 |
| DSV3_SWS_DENSE1 q4 | 1495712 | — | — | +30% |   |

- 原因不是跨机架：slurm-2 上 32 台机器分布在 32 个机架的 q4 段反而最快（job 35003 / 36135 / 36137：1045–1108 ms），更像是坏节点或坏链路。
- 从来没有在生产布局上抓过 nsys 或 torch profiler trace，唯一的 profile 是 10-20 的 q0 单节点 perf。EP8（q4–q6）下 all-to-all 和负载不均各占多少时间未知。
- EP 通信 overlap 在结构上用不了：q2–q6 每步只有 1 个 micro-batch，KD 路径又禁用了 `--overlap-moe-expert-parallel-comm`（`kd/forward_step.py:31,52`）。
- 训练中 step time 漂慢 6–8%，同期 LBL 从 1.08 降到 1.01：v1 q2 689 → 734 ms，v1 q3 901 → 959 ms，KD q3 1300 → 1407 ms。

建议

- `base.sh` 加预检：每个节点跑 `dcgmi diag -r 1` 加一个短的 all-to-all 带宽测试，或者打开 `--gpu-sniff-test-interval`。
- step-time 看门狗：约 200 iter 后，segment 中位数比该 rung 的参考值慢 15% 以上就报警，并带 `--exclude` 重排。
- q3（16 节点 EP4）和 q4（32 节点 EP8）各抓一次 3 步 trace，再用 `--moe-router-force-load-balancing` 抓一次对比，就能知道负载不均占多少时间。

## 3. 次要问题

### 3.1 所有 run 都关了 rerun

- 全部约 787 份 argv 都是 `--rerun-mode disabled`（`sh/base.sh` 的公共 argv），而两棵树的默认值是 `validate_results`。KD 必须关（replay 会丢掉 KD side channel，`kd/forward_step.py:17-26`），其他分支只是跟着继承了，没找到记录的理由。
- NaN 检查还开着。在 disabled 模式下，NaN 会直接抛 `RuntimeError`，进程以 exit 1 退出（而不是 16），不会换 GPU 重跑，也没法归因。
- 目前没造成后果：779 个 run 里 NaN/Inf、跳过的 iteration 和 rerun 消息都是 0。见到的 exit code 有 1、2、134、135、139、143，全是基础设施问题。到 235B 就有风险：续训可能又落到同一块坏 GPU 上反复崩。

### 3.2 BASE q5/q6 warmup 期大尖峰，全仓库没有专家输出范数监控

| run | iter | 前兆 | 尖峰 | 恢复 |
|---|---|---|---|---|
| BASE q5 | 1922–1923 | 1920 时 grad norm 9.9（约正常的 20 倍），1921 时 LBL 1.21 | grad norm 185 / 283，lm loss 19.5（高于 ln V = 11.9，比均匀分布还差），LBL 峰值 3.9 | 约 50 步 |
| BASE q6 | 3179–3180 | 3176 时 grad norm 9.0，3175–3177 时 LBL 1.14 → 1.24 | grad norm 254 / 482，lm loss 13.4 | 约 15 步 |
| BASE q4 | 5090 |   | grad norm 45（中位数的 577 倍），loss +0.31 |   |
| lr0p5x q1 | 740 | warmup 结束时 | loss 3.0 → 7.2，LBL 2.86 |   |

- 手册说的前兆确实出现了：grad norm 和 LBL 在尖峰前 1–3 步上升；router 塌缩的 case（宽 cell、rtadamw）里 LBL 提前几百步就在升。但都是事后才发现的，没有任何报警。
- MuonH+KD 主线（q0–q5）及其 packplan 变体：0 次 loss 尖峰（比前 20 个记录点的中位数高 0.1 以上算一次）。
- 全仓库没有 `--save-activations-interval`，也没有专家输出的 hook。`router_health.py` 只看负载、MaxVio 和 logits，MuonH 的 sphere monitor 只看权重范数。MuonH 下权重范数是固定的，rogue expert 只能从激活上看出来。
- 主线上的 grad-norm 跳变（最高 KD s64y q3 在 55070 步 17.9，为中位数 40 倍；packplan q2 在 28740 步 56.3，为 111 倍）都被 clip 1.0 吸收，没有引起 loss 跳变。57% 的跳变落在不同 run 的同一个 iteration 上（55070、10690、35940、56820 各 13–19 个 run），来源是数据顺序，不是硬件或数值精度。

### 3.3 softmax router logits 整体上漂，没开 z-loss

- z-loss 是为了对齐 Qwen3 有意关掉的（`qwen3moe.sh:47,146`），gpt-oss 梯子当时用的是 1e-3。
- max|z| 一直在涨：MUONH_FIT q1 是 2.3（step 1）→ 17.8（1k）→ 30.5（10k）→ 35–39（结束）；MUONH_FIT_SWA q3 是 4.2 → 44.3，|z| > +4 的占比从 0 涨到 81%。带符号的监控（job 1377911）显示全部是正向的整体偏移，正是手册说该加 z-loss 的情形。sigmoid router 更高：DSV3 q3 约 175，PROD1/RS1 为 63–159。
- router 是 fp32，没有精度风险；主线 q3–q5 也没有尖峰。但主线 q3 以上没开 router health，logsumexp 也从不记录（Megatron 只在系数非 0 时才记）。

### 3.4 gate 锐度从没测过

- softmax router 的 top-1 权重、Max1/Max2 和熵都没记录过（`router_health.py` 的 "top-8 pinned" 只对 sigmoid 有意义）。唯一的参考是 phi-loss q0 分支上的全 softmax 熵：3.4–4.2 nats（ln 128 = 4.85），看起来偏平。手册的判据：Max1/Max2 → 1，就说明这一层在对 top-k 取平均（3.1 Sharpness）。
- router 初始化在每个 rung 都是 N(0, 0.02)（`hyperp/init.py:11`，和 Qwen3 一样），没有 jitter，也没有 logit 归一化。

### 3.5 MuonH 下冷门专家的步长偏大

- MuonH 的更新是 X −= lr·‖X‖_F/‖U‖_F·U，不管梯度秩多少，每步的 Frobenius 步长都固定。所以手册说的"per-expert 的步长会随 token 数变少而变小"在我们这里不成立。
- CPU 上验算（q3 fc1 960×1280，lr 7.01e-3，quintic NS）：rank-1 梯度的谱步长 / 权重谱范数为 0.116，满秩时为 0.0044，差 26 倍。
- 零 token 的专家会靠残留动量继续以满 lr 走约 300 步，总位移约为半径的 2 倍；AdamW 下约 50 步就停了。
- 会不会真出问题未知：主线没有逐专家监控，看不到。

### 3.6 其他

- **bf16 Newton-Schulz**：两条路径都在 bf16 下算 NS（MuonH `ns_dtype=torch.bfloat16`，`hyperball_batched.py:276`；Muon `--muon-fp32-matmul-prec` 默认 medium）。没观察到能归因到它的尖峰。
- **PROD v1 的设计对照**：
  - 共享专家是在 top-8 之上额外加的，专家算力多 12.5%，没有做消融。手册建议公平比较时用它替换一个 routed slot。
  - latent2 是在相同 token 数下比较的：q3 上 N_active +7.3%，每 token FLOPs +3.8%，q3 的 −0.011 里约 0.003 来自多出的算力。latent 下 routed scale 2.5 不起作用（RMSNorm 在 `fc2_latent_proj` 之前）。
  - e2x 和 latent2 的吞吐低 13–39%，应该在相同墙钟时间下也比一次。
  - dense L0 被算成了 MoE 层，N_total 多算了 3–8%。
- **显存只记 DP rank 0**（`report_memory`）：W3/W4 的显存门槛是按 rank 0 的峰值定的。目前余量充足（KD q3 约 150 GB / 约 284 GB），对更宽的 cell 和 235B 要注意。
- **q0 吞吐只有 47 TFLOP/s/GPU**：TE 2.19 的 CUTLASS grouped GEMM 只支持 SM90，GB300 上退回成逐专家循环调用 cuBLAS。这是吞吐问题，不是质量问题。专家粒度本身就在手册的收益平台点上。
- **没有 dense 参照，也没开 deterministic mode**：影响低。现在靠多 seed 统计验收、CPU parity 测试和 DP 权重 hash 检查来兜底。

## 4. 确认没问题的

| 项目 | 手册的要求或警告 | 我们的情况 | 证据 |
|---|---|---|---|
| router 精度 | 必须 `--moe-router-dtype fp32`，bf16 会翻转接近的 logits | 不受影响 所有 qwen3 golden 都带 fp32；bf16 输入、fp32 输出 logits，top-k 和 softmax 都在 fp32 上做 | `qwen3moe.sh:183`；`router.py:142-147`；`moe_utils.py:925-966, 1441-1446` |
| top-k 归一化 | k > 1 时要重归一化 | 不受影响 softmax-after-top-k 等价于 Qwen3 的 `norm_topk_prob` | 没有 `--moe-router-pre-softmax`，没有 scaling factor |
| 丢 token | 设了 capacity factor 会静默丢 token，且不记日志 | 不受影响 dropless | 两棵树 `moe_expert_capacity_factor=None`，argv 里没有 capacity 相关 flag |
| aux 梯度缩放 | — | 不受影响 CE 和 KD 走同一条路径：1/num_microbatches，再 DDP 平均 | `schedules.py:337-341, 363-367`；`kd/loss.py:386,413` |
| 我们自己的封装 | — | 不受影响 KD / MuonH / docmask / packplan / eval 都没有跳过 aux loss、bias 更新、MoE 日志或 eval gating | `pretrain_hyperp.py` 调用的是 stock `pg.pretrain`，patch 都是包一层再调用；eval 时 aux 由 `training` 和 `is_grad_enabled` 控制 |
| 没有 loss 的 token | — | 不受影响 合计不到 0.2% | loss mask 约 0.12%，packplan padding 0.045% |
| expert bias 机制 | 旧版用 bf16 计数，可能让 bias 更新方向翻转（d6316eb45 修复） | 不受影响 int64 计数，fp32 bias，每步用全局计数更新，存进 checkpoint，不进 optimizer，各 rank 一致（36,141/36,141 次报告） | `router.py:226-245`；`finalize_model_grads.py:167-177` |
| DSV3 seq-aux | α = 1e-4 | 不受影响 所有 DSV3 / prod argv 都是 1e-4 | argv |
| 正交化粒度 | 按专家分别正交化 | 不受影响 Muon 和 MuonH 都是逐专家 | `moe_single_grouped_weight=False`；MuonH q3 日志 1980 个矩阵 = 30 × (qkv + proj + 32 × 2) |
| Muon 更新尺度 | 从 `--muon-extra-scale-factor 0.2` 起步 | 不受影响 spectral 模式，0.2·√max(m,n)，router 用 skip-muon | q3 argv |
| `global_aux_loss` | 超过 256 个 micro-batch 时 bf16 计数器失效 | 不受影响 bug 在两棵树里都存在，但我们最多 4 个 micro-batch，也不用它。README 的 1/DP 说法正确 | `router.py:251-265`；`reports/global_aux_loss_scaling_2026-10-20.md` |
| 路由方式 | expert choice 会泄露未来 token | 不受影响 token choice，没有 hash 层，eval 路由和训练一致 | dual-window 8K 与单独 8K eval bit-identical（11-04） |
| dense 首层 | "Do not count on them to fix early-layer imbalance" | 和手册一致 已消融：DENSE1 q1–q4 Δ −0.0045 / −0.0040 / +0.0001 / −0.0009，DENSE2 q1–q3 −0.0029 / −0.0016 / −0.0007；不均衡转移到 L1（SWS_DENSE1 q4 的 L1 bias +11.3，其他层 +3.4 至 +5.8） | live README 288-290 行 |
| 专家粒度 | 128 个 d/4 宽的专家、top-8 之后再细分没有收益 | 不受影响 128、top-8、宽度 3h/8，每个 rung 比例相同 | `qwen3moe.sh:100` |
| 激活 clamp | 7 / 10，质量基本不变 | 不受影响 主线不 clamp；PROD clamp7 / clamp20 对照在噪声内 | `prod_v1/RESULTS.md` |
| NCCL 传输 | 走 Socket 会慢约 25 倍 | 不受影响（推断） 2 节点探针走的是 IB；32–64 节点每步 1.0–1.4 s，Socket 做不到。训练日志里不记录 | job 34988 |
| upcycling / MTP | — | 不适用 都没用 | argv |

## 5. 工具与文档的小错

- `dsv3/tools/bias_spread_trend.py`：router health 的逐层日志超过约 1 KB 会折行（续行以 " 0: " 开头），工具只读到 q1/q2/q3 的 18/24/30 层中的前 14 层。`DECISIONS.md` 约 1997 行 "every arm PASSES the hard criterion (largest spread 0.59)" 依赖它。重新解析 DSV3 A q1–q3 后，最大 spread 0.34 / 0.31 / 0.32 都在前 14 层，A 的结论不变；其他分支还没复查。
- `DECISIONS.md:1696`：LBL 的上界不是 128，top-8 时是 16。
- README 的 "init 0.02" 不完整：AdamW 下 fc2 用的是 0.02/√(2L)（q0 为 0.0041，q6 为 0.0020）；Qwen3 HF 全部用 0.02。
- README 的吞吐（q2 264、q3 338 TFLOP/s）是训练早期的数字，全程中位数低 3–7%（v1 q2 254，q3 314）。
- README 说 "no compensation outside `--calculate-per-token-loss`"，其实 per-token 分支也是 1/DP。
- `router_maxvio_doc_isolated.json` 把 pad token 也算进去了（`router_health.py` 里有说明）。
- `boxiangw/envs/dpl/lib` 里还有一份 conda-forge 的 `libnccl.so.2.32.3`，有潜在的覆盖风险（torch 目前通过 RPATH 加载 pip 的 2.30.7）。

## 6. 建议

### 6.1 不占 GPU

- 新的主线 launch 默认开 `LADDER_ROUTER_HEALTH=100` 和 `--moe-per-layer-logging`。`base.sh` 在作业开跑时被实时 source，要在两次提交之间改。
- `router_health.py` 增加：per-micro-batch 和每个 EP rank 的 MaxVio、RMS logsumexp、top-1 权重和 Max1/Max2，以及专家 fc2 输出范数的 max/median。
- `base.sh` 加节点预检和 step-time 看门狗；rank 0 上开 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET`；显存改成 all-reduce MAX 后再记录。
- 修正第 5 节的文档和工具小错。
- 非 KD 分支把 `--rerun-mode` 改回 `validate_results`；KD 保留关闭，并写明原因。
- 把 55070 / 56820 / 28740 步的样本 dump 出来，看有没有坏文档。

### 6.2 需要 GPU（都还没提交）

| 优先级 | 实验 | 回答什么 | 规模 |
|---|---|---|---|
| P0 | 现有 KD q2/q3 checkpoint 跑 eval-only `LADDER_ROUTER_MAXVIO_VAL=1` | 主线到底均不均衡（2.1） | 小，只做 eval |
| P0 | q0（3 个 seed）和 q1，相同 GBS 下比较 mbs 1 和 mbs 4 | aux 作用范围对 loss 的影响（2.2） | 小 |
| P1 | h=4096 用 router ×0.25 跑全长，开 router health | 235B 的 router 规则（2.3） | 大 |
| P1 | q3（16 节点 EP4）和 q4（32 节点 EP8）各抓一次 3 步 trace，再用 force-balance 抓一次对比 | all-to-all 和负载不均占多少时间（2.5） | 很小 |
| P2 | quantile balancing 对照，q0 三个 seed 加 q1 | 第一层的 bias 滞后（2.4） | 中 |
| P2 | q0/q1 上：z-loss 1e-3；NS 用 fp32；aux 系数扫 5e-4 / 1e-3 / 3e-3 / 1e-2 | logits 漂移、bf16 NS、系数是否合适（3.3、3.6） | 中 |
| P3 | PROD：不带共享专家、top-7 加共享专家，q0–q2；latent / e2x 在相同墙钟时间下比较 | PROD 设计的对照（3.6） | 中 |
| P3 | q0/q1 在相同 active 参数下的 dense 参照 | 区分训练栈的问题和 MoE 特有的问题 | 小 |

## 7. 方法

- 手册全文（6 章，HTML 去标签后约 33 万字符）下载后转成纯文本，按章节分给 10 个 agent：

| # | 方向 | 手册章节 |
|---|---|---|
| 1 | 监控 | 3.0, A.1 |
| 2 | router 打分：精度、z-loss、归一化、锐度 | 3.1 |
| 3 | 路由策略与 block 布局 | 3.2, 3.5 |
| 4 | aux 系数与作用范围 | 3.3 前半 |
| 5 | 无 aux 的 bias 与 quantile balancing | 3.3 后半 |
| 6 | 专家数量、宽度、共享专家、激活 | 3.4 |
| 7 | Muon / MuonH 优化器 | 3.6 Optimizer |
| 8 | 稳定性、日志尖峰扫描、2.5 检查 | 3.6, 2.5 |
| 9 | 系统与吞吐 | 3.7, 2.1 |
| 10 | 我们自己的封装代码 | 全文机制、A.2 |

- 每个 agent 都只读，查的范围：
  - 两棵 Megatron 树：fork @60547a3f4（AdamW BASE、Muon），ADLR @80781ab0ab（MuonH、KD）；
  - launch layer（我们的 clone 和 jobs 实际 source 的共享副本，两者只在 QOS 行有差别）；
  - print-mode 渲染出的 argv 和 golden；
  - boxiangw 和 xingqianx 两边约 780 个 run 的 argv、日志和 tensorboard；
  - `DECISIONS.md` 和 reports。
- 交叉印证：aux 作用范围随 rung 变化由 agent 4 和 10 各自独立查出；L0 不均衡由 1/3/4/5/8 五个 agent 从不同 run 得到；router 学习率问题由 2/7/8 三个 agent 得到。我另外抽查了 `qwen3moe.sh` 的 mbs 表、负载均衡白名单、`--moe-router-dtype fp32`（183 行）、`DECISIONS.md:1696` 和 `base.sh` 的 `--rerun-mode disabled`，都和 agent 的报告一致。
- 局限：
  - 多数数值来自 agent 解析日志，没有逐条独立复核。
  - 共享副本的 `DECISIONS.md` 没有读权限，读的是 clone 里的版本。
  - 主线 q2 及以上没有任何逐层数据，第 2.1 节对主线的判断是用同款 router 的 MUONH_FIT / SWA run 推断出来的，P0 的 eval-only MaxVio 可以直接验证。
