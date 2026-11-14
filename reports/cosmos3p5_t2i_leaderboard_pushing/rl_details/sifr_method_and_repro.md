# SIFR（Signed-RF `constraint` 算法）方法、实现细节与复现手册

SIFR（原名 FlowCTRL）是 Signed-RF-Playground 的 `constraint` 算法（Flow_RL notes 式 31/32）在 imaginaire4 里的实现，构建在 NFT 的 rollout / 奖励 / 旧策略机制之上。本文写到可以照着复现：网络与权重体系（net / net_old / net_ref / net_ema / 评测权重）、奖励与优势怎么算、损失与梯度、对偶价格控制器的每一步、参数总表、实验清单、已记录的现象、复现步骤。与 NFT 共用的部分（奖励服务、三种奖励的计算、噪声采样、LoRA 注入、EMA 公式）只在这里写差异，细节见同目录 `nft_method_and_repro.md`。

代码位置（均在 `projects/cosmos3/cosmos3/` 下）：

| 内容 | 文件 |
|---|---|
| 算法、控制器、rollout/round 逻辑 | `models/omni_mot_model_v2_sifr.py`（`OmniMoTModel_SIFR`、`DualPriceController`） |
| 训练循环（一个 round 内 S 次 optimizer step）、控制器状态存取 | `trainer_sifr.py`（`ImaginaireTrainer_SIFR`、`SIFRDistributedCheckpointer`） |
| 实验配置 | `configs/base/experiment/text_rendering/cosmos3plus_64bm32b_t2ionly_sifr_exp000.py` |
| wandb | `callbacks/wandb_log_sifr.py`（`train_sifr/*`） |
| PickScore 训练中评测 | `callbacks/every_n_pickscore_eval.py`（`EveryNPickScoreEval`） |
| 继承的 NFT 部分 | `models/omni_mot_model_v2_rl.py`（`OmniMoTModel_NFT`、`OmniMoTModel_RL`） |

## 0. 约定

- t 的方向、`x_t`、`v*` 同 NFT：干净样本在 t=0，`x_t = ε·t + x0·(1−t)`，`Y = dot_x_t = ε − x0`。
- 记号：`θ`=`net`，`θ_old`=`net_old`，`θ_ref`=`net_ref`，`θ_ema`=`net_ema`；`A`=带符号、**未截断**的组归一化优势；`Y`=目标速度；`λ_ref, λ_policy`=两个锚点的价格。
- **一次迭代 = 一个 round = Playground 的一个 epoch**：1 次 rollout，拆成 `optim_steps_per_round`（S=3）份，每份一次 optimizer step，然后一次对偶价格更新、一次 `net_old` 折叠、一次评测 EMA 更新。迭代数、`max_iter`、`save_iter`、`every_n`、wandb 步数全部以 round 计，wandb 每个点是该 round 的平均。
- 与 NFT 的对比：相同迭代数下 rollout 量相同，SIFR 的 optimizer step 是 NFT 的 3 倍；NFT 把优势压成 [0,1] 的分支权重，SIFR 直接用带符号的优势乘以对目标的误差，用对旧策略（和可选的参考策略）的二次项约束漂移。

## 1. 网络与权重体系

### 1.1 各网络是什么

| 网络 | 创建 | 梯度 | 用途 | SIFR 里的特殊点 |
|---|---|---|---|---|
| `net` | `OmniMoTModel.set_up_model` | 有（仅 `lora_` 参数） | 在线策略 θ | 同 NFT；LoRA rank32/α64（继承 `nft_base_lora_rank32`） |
| `net_old` | `OmniMoTModel_NFT.set_up_model`（`on_policy=False` 恒成立，所以一定创建） | 无 | **rollout 的采样网络**，并提供 `v_old` 锚点 | 每个 round 折叠一次；其 LoRA 张量以 `net_old.*` 键写进 checkpoint（冻结底座与 `net` 相同，启动时重新拷贝） |
| `net_ref` | 仅当 `ref_price is not None`（`use_kl`） | 无 | 参考锚点 `v_ref`（初始策略） | 默认 `ref_price=None`，**所有已有实验都不建 `net_ref`**；若启用，启动时 = `net` 且 `lora_B` 清零，即初始策略 |
| `net_ema` | `config.ema.enabled=True`（SIFR 基础配置打开）；fp32 | 无 | **评测权重** | `ema_beta` 被覆盖成常数 `eval_ema_beta=0.9`（不是 NFT/基类的 power EMA），每个 round 更新一次（Playground 的 eval adapter） |

### 1.2 初始化与加载顺序

1. DCP 加载同 NFT：`load_path` 为 exp009_sft0 iter31k，`load_ema_to_reg=True`（读 EMA 权重填入常规 `net.*` 键），`load_training_state=False`；LoRA 键跳过。SIFR 额外把 `"net_old."` 加进 `keys_to_skip_loading`（`["lora_", "net_old."]`），因为 warm start 的 checkpoint 里没有 `net_old.*`。
2. `OmniMoTModel_SIFR.load_pretrained_model_if_needed` **绕过** NFT 的版本（调用 `super(OmniMoTModel_NFT, self)`）：
   - 若是同作业续训（`has_resumable_checkpoint`），先快照 `net_old` 里已由 checkpoint 恢复的 LoRA 张量；
   - `copy_to(net → net_old)`，然后把快照的 LoRA 写回 `net_old`（续训时 `net_old` 保持 checkpoint 里的值，而不是被覆盖成当前 `net`）；
   - 若有 `net_ref`：`copy_to(net → net_ref)` 并把所有 `lora_B` 清零（= 初始策略，续训后也不会漂）。
3. LoRA A 初始化由 `sifr_config.lora_a_init` 控制：
   - `"kaiming"`（默认）：沿用 i4，`lora_A ~ kaiming_uniform_(a=√5)`；
   - `"gaussian"`：对齐 Playground 的 PEFT `init_lora_weights="gaussian"`，`lora_A ~ N(0, (1/lora_rank)²)`（rank 32 时 std=1/32；约比 kaiming 大 4–5 倍，见实验记录），并把 dtype 转回底座 dtype。
   `lora_B` 两种情况都是 0，所以初始策略等于基础模型。该选项对启动时构建的所有网络（`net`、`net_old`、`net_ref`）生效；续训时 LoRA 从 checkpoint 读，不再重新初始化。
4. checkpoint 内容：`net.*`、`net_ema.*`、`net_old.*`（仅 LoRA 键）、优化器 / scheduler / trainer，以及与每个 checkpoint 并列的 `sifr_lambda_iter_<iteration:09d>.json`（见 3.4）。

### 1.3 `net_old` 的折叠与评测 EMA 的更新时机

每个 round 的最后一次 optimizer step 之后，trainer 依次：`callbacks.on_before_zero_grad` → `model.sifr_end_round`（探测 + 价格更新）→ `model.on_before_zero_grad`（NFT 版：折叠 `net_old`；父类版：更新 EMA）→ `zero_grad`。因此：
- **探测漂移时 `net_old` 还是折叠之前的值**（即本 round 的 rollout 采样网络和锚点），之后才折叠；
- `net_old ← d·net_old + (1−d)·net`，`d` 来自 `net_old_mixing_type="constant_<d>"`。SIFR 要求必须是 `constant_` 形式（断言），`on_policy` 恒为 False：
  - `constant_0.5`（基础配置）：每个 round 折叠一半；
  - `constant_0`（exp003_004 起）：每个 round 把 `net_old` 直接同步成当前 `net`；
- 评测 EMA：`net_ema ← 0.9·net_ema + 0.1·net`，每个 round 一次，每个 round 结尾的第 S 步之后。

### 1.4 评测权重

| 场景 | 权重 | 说明 |
|---|---|---|
| 训练中 `EveryNEval`（UGB） | `net_ema` | 基础配置里 `eval_ugb.use_ema=True`；预览图也用评测 EMA 生成 |
| 训练中 `EveryNPickScoreEval`（exp004_000、004_004+） | `net_ema` | guidance 2.0，35 步，shift 5.0，768，宽高比 `1,1`，`pickscore_test.txt` 全量（`num_prompts=None`），`batch_size_per_rank=4`，每 10 round（exp004_004+ 为每 50 round，不在起点评测） |
| 离线评测 | `net_ema` 或 `net` | SIFR checkpoint 里同时有两者；用 `--use_ema` 取 EMA，不加取在线 LoRA。按 NFT 的约定 LoRA 用 `..._base_mini_lora_rank32` 做 `--experiment_name`（评测 EMA 需要该实验配置开启 EMA，使用前先确认） |

## 2. 一个 round 的完整流程

```
trainer.training_step(data, iteration)                       # ImaginaireTrainer_SIFR
  parts, round_output = model.sifr_start_round(data, iteration)
      ├─ rollout(data, net=net_old)                          # 同 NFT：guidance/步数/shift/宽高比
      ├─ one_cycle_off=False：训练这次的 rollout
      │  one_cycle_off=True ：训练上个 round 暂存的 rollout，暂存这次的；
      │                       启动/续训后第一个 round 没有暂存，会 rollout 两次（训练第一次、暂存第二次）
      ├─ collect_reward
      ├─ A = reward_fn.compute_group_normalized_advantage(reward, group_num, validation_tag=ai_caption)
      │   （未截断、不做 0.5 平移；PickScore 内部先 /26；NaN → 0）
      ├─ 本 rank 的 num_images 张图随机打乱，均分成 S 份（要求 num_images % S == 0，断言）
      └─ round_stats: reward_mean, reward_success_ratio, advantage_mean, advantage_abs_max(, reward_mean_div26_for_pickscore)
  for 每一份 part（共 S 份）：
      model.training_step(part)      # 对该份每个 (loss_compute_batch_size 切片 × num_timestep_samples) 做 forward+backward
      optimizer 步骤：
         前 S−1 份：只把优化器钩子派发给 SkipNaNStep / GradClip / LowPrecisionCallback（梯度更新类 callback）
         最后一份：派发给完整 callback 组，然后：
              round_metrics = model.sifr_end_round(iteration)   # 探测 + 对偶价格更新
              model.on_before_zero_grad(...)                    # net_old 折叠 + 评测 EMA 更新
  output = S 份输出的均值 + round_stats + round_metrics
```

要点：
- **优势在整个 rollout 上一次性算好**（全部 rank、全部图像），再切成 S 份；同一张图的全部 `num_timestep_samples_per_rollout`（10）个时间步样本留在同一份里。
- **每份的 loss 里用的价格是上个 round 末更新后的价格**（一个 round 内价格不变）。
- 每份重新采噪声与打包序列（同 NFT 第 3 节）；SIFR 的 `_sifr_velocities` 用 `_get_train_noise_level_vision` 的训练 RF 采样，**不支持** NFT 的 `rollout_sigma_sampling_type`。
- 不支持 context parallel；单视觉 item / 样本。
- 每份的 optimizer step 前 `model.on_after_backward()`；`ImaginaireTrainer_SIFR` 断言 `grad_accum_iter in (1, None)`。
- 梯度裁剪：默认 callback 组里的 `GradClip(clip_norm=1.0)` 每个 optimizer step 都生效（属于“梯度更新类 callback”）。

### 2.1 每 rank / 每步的数据量（24 节点，每节点 4 GPU → 96 rank，未用日志核实）

| 配置 | num_rollouts_per_sample（每 rank） | group_size | 每组 rollout | 每 round 图像 | 每份（每 rank） | 每 optimizer step 全局图像 |
|---|---|---|---|---|---|---|
| 基础 / exp000–exp002、exp003_000、003_002–004、exp004_* | 6 | 2 | 12 | 576 | 2 | 192 |
| exp003_001 | 12 | 2 | 24 | 1152 | 4 | 384 |

Playground 的设置：每步 384 张图（24 条/prompt）、`beta2 0.99`、PEFT gaussian LoRA A。

## 3. 损失、价格与控制器

### 3.1 两种损失形式（`sifr_config.loss_form`）

对每张图 i（取一个噪声水平，`v=v_θ`），`reward_term_i = A_i · mean((v − Y)²)`，锚点集合 `{ref: v_ref(若启用), policy: v_old}`：

- **`signed`（式 32，所有实验使用）：**
  `L_i = A_i·‖v−Y‖² + Σ_j λ_j·‖v − v_anchor_j‖²`（`‖·‖²` 均为逐元素均值）。
- **`detached`（式 31）：**
  `Λ = Σλ_j`，`v̄ = Σ λ_j·v_anchor_j / Λ`，`target = v̄ + (A_i/Λ)(Y − v.detach())`，`L_i = Λ·mean((v − target.detach())²)`。

总损失 `loss = mean_i L_i`，记录 `loss_policy = mean_i reward_term_i`（只是奖励项），`loss_drift_{ref,policy} = mean (v.detach() − v_anchor)²`，`lambda_{ref,policy}`。各份、各时间步样本的损失按 `(end−start)/N/num_timestep_samples` 加权累加。

### 3.2 `signed` 形式的极小点与不稳定性（从公式推出，非实验证明）

对 `v` 逐元素求导：`2A(v−Y) + 2λ(v−v_old)=0` → `v* = (A·Y + λ·v_old)/(A+λ)`，仅当 `A+λ>0` 时是极小点：
- `A>0`：在 `Y` 与 `v_old` 之间，`A≫λ` 时≈`Y`；
- `A<0` 且 `|A|<λ`：`v*` 在 `v_old` 外侧（被推离 `Y`），仍有极小点；
- **`A<0` 且 `|A|>λ`：二次项系数 `A+λ<0`，损失对 `v` 无下界**，梯度会把 `v` 一直推离 `Y`，没有内置的恢复力。

基础配置 `λ_policy` 初值 0.01，上限 1.0（exp000 起改为 0.3）；实测 `|A|max` 在 std `by_group` 时约 1.1，`by_epoch` 时约 0.3–0.45，都远大于 0.01。也就是说一开始所有负优势行都处在“无下界”区间，漂移被拉住全靠价格 `λ` 在之后涨到 > |A|（或奖励项的正优势行抵消）。这是 exp002_000 / exp003_000 发散的候选解释之一（实验记录见第 6 节）。

### 3.3 对偶价格控制器（`DualPriceController`，Playground BudgetController 标量模式）

参数：`init, bounds=[min,max], budget_value κ, halflife, double_rounds, deadband, hmax`。基础值：`init 1e-2, bounds [1e-4, 1.0], κ 1e-3, halflife 10, double_rounds 50, deadband 0.075, hmax 1.0`；exp000 起所有实验把上界改为 0.3。

每个 round 的 `sifr_end_round`：
1. **探测**：对本 round 的 S 份数据各调用一次 `_sifr_velocities`（**重采噪声与时间步**，无梯度，`net`/`net_old`/`net_ref` 用完后 `reshard`），对每张图、每个 dual 锚点累加
   - `drift = mean (v_θ − v_anchor)²`
   - `calib = mean (Y − v_anchor)²`
   与计数；
2. 各 rank 的 `[drift_sum, calib_sum, count]` 做 `all_reduce(SUM)`，得到全局均值 `drift`、`calib`；
3. `update`：
   - 若非有限 → 价格不变，记 `poisoned=1`；
   - 首个 round：`budget = κ · calib`（断言 `calib>0`）；
   - `drift_ema = drift`（首个）或 `decay·drift_ema + (1−decay)·drift`，`decay = 2^(−1/halflife)`（halflife 10 → 0.933）；
   - `h = clip(drift_ema/budget − 1, −hmax, hmax)`，`|h| < deadband` 则 `h=0`；
   - `log(price) += (ln2/double_rounds) · h`，再夹到 `[price_min, price_max]`。
4. 记录 `ctrl_policy_{lambda_next, dbar(=drift_ema), delta_sq(=budget), h, drift_round, poisoned}`。

数量级：`double_rounds=50` 时每个 round 价格最多乘 `2^(1/50)=1.0140`；从 0.01 涨到 0.0111（约 +11%）需要约 8 个 round，与 exp002_000 的记录（0.0101 → 0.0111）一致——**控制器的最大速度远低于漂移失控的速度**（该实验 3 个 round 内漂移涨了 30 倍）。价格是固定值 float 时 `FixedPrice` 不更新；填 `None` 去掉该锚点；至少要有一个锚点。

### 3.4 控制器状态的保存与续训

`SIFRDistributedCheckpointer` 在每次保存 checkpoint 时由 rank 0 写 `sifr_lambda_iter_<iteration:09d>.json`（`optim_steps_per_round` + 每个 dual 锚点的 `price, drift_ema, budget, rounds, kappa`），位于该次运行的 checkpoint 目录旁。续训：
- 缺 json → 直接抛 `FileNotFoundError`，不会悄悄把价格重置为 `init`；
- 保存的 `optim_steps_per_round` 与配置不同、或保存的 `kappa` 与配置不同 → 抛 `ValueError`（预算在不同定义下标定的，要求重开）；
- warm start（有 `load_path` 无 `latest_checkpoint.txt`）→ 价格从 `init` 开始。

### 3.5 评测 EMA

`ema_beta` 恒为 `eval_ema_beta`（0.9），不是 power EMA；每个 round 在 `on_before_zero_grad` 里更新一次。

## 4. 奖励与优势（与 NFT 的差异）

奖励服务、三种奖励的计算、NaN 语义、`reward_call_id` 机制全部同 NFT（见 NFT 报告第 4.1–4.2 节）。SIFR 目前只跑过 PickScore。

优势：调用的是 `compute_group_normalized_advantage`，即 NFT 的第 1–6 步（all-gather → 校验同组 caption → `[N_group, group_num]` → 组均值/（组 或 epoch）std → `A = deviation/(std+1e-4)`；PickScore 先 `/26`），**之后不截断到 ±5、不做 `(A/5)/2+0.5` 的平移**，原始带符号的 `A` 直接乘到奖励项上。
- `advantage_mean` / `advantage_std` 各自 `by_group`（默认，基础配置显式设置）或 `by_epoch`；
- NaN 奖励的 `A=0`，不影响其他样本统计；
- 每个 round 记录 `advantage_mean`、`advantage_abs_max`；

`|A|` 的量级决定奖励项与锚点项的相对强度，这是 SIFR 对 std 模式敏感的原因：by_group 下每组被拉到单位方差，`|A|max≈1.1`；by_epoch 下全局 std 更大，`|A|max≈0.3–0.45`（日志实测）。

## 5. 参数总表

### 5.1 基础配置（`cosmos3plus_64bm32b_t2ionly_sifr_base_lora_rank32`）

继承 `cosmos3plus_64bm32b_t2ionly_nft_base_lora_rank32`（LoRA rank32/α64，`lr 3e-4`，`betas [0.9,0.999]`，`weight_decay 1e-4`，rollout guidance 1.0 / 35 步 / shift 5.0 / 768 / `mix`，`rollout_batch_size 6`，`num_rollouts_per_sample 6`，`num_timestep_samples_per_rollout 10`，`loss_compute_batch_size 4`，`max_iter 500`，`save_iter 10`，训练 RF shift `{"768": 5}`，`data_parallel_shard_degree 32`，初始化 checkpoint 同 NFT）。SIFR 的改动：

| 类别 | 参数 | 取值 |
|---|---|---|
| 类型 | `model_type` / `trainer_type` | `OmniMoTModel_SIFR` / `ImaginaireTrainer_SIFR` |
| | `ema` | `enabled=True` |
| | callback | 去掉 `wandb_nft`，加 `wandb_sifr`（`SIFRWandbCallback`）；`eval_ugb.use_ema=True` |
| NFT 部分 | `net_old_mixing_type` | `constant_0.5` |
| | `nft_rollout_one_cycle_off` | False |
| | `kl_loss_weight` | 0 |
| SIFR | `loss_form` | `signed` |
| | `optim_steps_per_round` | 3 |
| | `ref_price` | None |
| | `policy_price` | `init 1e-2, bounds [1e-4,1.0], budget_value 1e-3, halflife 10, double_rounds 50, deadband 0.075, hmax 1.0` |
| | `eval_ema_beta` | 0.9 |
| | `lora_a_init` | `kaiming`（默认） |
| 奖励 | `timeout` / `collect_timeout` | 120 / 20 |
| | `advantage_mean` / `advantage_std` | `by_group` / `by_group` |
| 数据 | `true_num_rollouts_per_sample` / `group_size` | 12 / 2 |
| checkpoint | `keys_to_skip_loading` | `["lora_", "net_old."]` |

### 5.2 为什么要 `lora_a_init` 与 `betas`

这两项是为了向 Playground 对齐而加的：gaussian LoRA A 把有效学习率放大约 4–5 倍；`beta2 0.99` 让二阶矩更快适应梯度尺度突变（SIFR 每个 round 内的 3 次 step 梯度尺度变化大）。

## 6. 实验清单

每行继承父配置，只改所列项；奖励均为 PickScore。

| 实验 | 父配置 | 改动 |
|---|---|---|
| exp000_000 | base | `policy_price.bounds=[1e-4, 0.3]`；宽高比 `mix` |
| exp001_000 | exp000_000 | `rollout_aspect_ratio="1,1"`（sq） |
| exp001_001 | exp001_000 | 显式 `nft_rollout_one_cycle_off=False`（当前配置下与父配置等价） |
| exp002_000 | base | sq，`bounds=[1e-4,0.3]`（当前配置等价于 exp001_000） |
| exp003_000 | exp002_000 | `advantage_std="by_epoch"`，`collect_timeout 120` |
| exp003_001 | exp003_000 | `betas [0.9,0.99]`，`num_rollouts_per_sample 12`，r24（每步 384 张） |
| exp003_002 | exp003_001 | `num_rollouts_per_sample 6`，r12（每步 192 张） |
| exp003_003 | exp003_002 | + `nft_rollout_one_cycle_off=True` |
| exp003_004 | exp003_002 | `net_old_mixing_type="constant_0"` |
| exp004_000 | exp003_004 | 新增 `EveryNPickScoreEval`（每 10 round，guidance 2.0，35 步，shift 5，768，`1,1`，`use_ema`，起点评测） |
| exp004_001 | exp004_000 | `lora_a_init="gaussian"`，去掉 `eval_pickscore` |
| exp004_002 | exp004_001 | rollout `guidance=2.0` |
| exp004_003 | exp004_002 | `advantage_std="by_group"` |
| exp004_004 | exp004_003 | `lora_a_init="kaiming"`；`eval_pickscore` 每 50 round，不在起点评测 |
| exp004_005 | exp004_004 | rollout `guidance=4.0` |
| exp004_006 | exp004_004 | `net_old_mixing_type="constant_0.5"` |

**配置与日志不一致的地方**：当前基础配置显式把 `advantage_mean/std` 设为 `by_group`，所以 exp001_001 现在继承的是 `by_group`；而 2026-10-07/08 的训练记录里它是“同样的 SIFR 配置但 std by_epoch”。说明基础配置的 std 默认值在那之后改过。解读下面的现象表时以训练当时的 commit 为准。

## 7. 已记录的现象（训练日志 / wandb，单 seed，2026-10-07 ~ 10-08；本目录没有 SIFR 的 PickScore / UGB / CVTG 打分）

| 实验 | 设置要点 | 现象 |
|---|---|---|
| exp001_001 | std `by_epoch`，`|A|max` 约 0.3–0.45 | 稳定 |
| exp002_000 | mean+std 均 `by_group`，每次 rollout 分 3 步，`|A|max` 约 1.1 | 约 6 个 round 后发散：漂移 `‖v−v_old‖²` 3 个 round 内 0.012 → 0.37；带符号奖励项 `A‖v−Y‖²` 降到 −0.55（负优势行失控）；奖励 22.4 → 20.3；价格只从 0.0101 爬到 0.0111（每 round 最多 ×1.014） |
| exp003_000（job 2398570） | 改回 std `by_epoch` | 仍发散：round 10（= 30 个 optimizer step）首次 loss 尖峰，漂移 0.3–3，梯度范数 0.3–3（正常约 1e-3），奖励 ~21.8 → 17–19，已取消 |
| exp003_001 | beta2 0.99 + r24（每步 384 张） | 不稳定：漂移比 exp003_002 高 6–30 倍，有尖峰 |
| exp003_002 | beta2 0.99，r12 | 30 个 round 内稳定（漂移 ~1e-3，round 25 奖励 23.7）；按 optimizer step 计比 NFT 慢约 0.3–0.6 |
| exp003_003 | exp003_002 + one-cycle-off | round 10 发散（漂移 0.6 → 1.3） |
| exp003_004、exp004_00x | | 本记录里没有结果 |

对照 NFT 同一设置的基线 exp005_006 自身也会在后期崩溃（~iter425–450 奖励 ~24 → ~16）。

### 7.1 初步结论与待验证项

- **std 模式不是 SIFR 发散的根因**：exp003_000 改成 `by_epoch` 仍发散。exp002/exp003 与稳定的 exp001 的共同差异是“每次 rollout 拆 3 步、每步只用 1/3（192 张）”。
- **one-cycle-off 与 SIFR 不兼容**：采样器（`net_old_{k−1}`）与锚点（`net_old_k`）不一致，而 Playground 要求 rollout 从中心 adapter 采样；exp003_003 发散与此一致。
- **价格上升太慢**：`double_rounds=50` 时每 round 最多 ×1.014，压不住漂移；并且 3.2 节指出 `|A|>λ` 的负优势行损失无下界。
- **对齐 Playground 的组合**：`beta2 0.99` + 12 条/prompt 稳定（exp003_002）；再加到 24 条/prompt 反而更不稳定（exp003_001），与“样本量越大越稳”的直觉相反，原因未明。
- exp004 系列（gaussian LoRA A、guidance 2.0/4.0、`constant_0` vs `constant_0.5`、PickScore 评测 EMA）是对这些假设的后续检验，尚无记录。

## 8. 复现步骤

1. 提交（gb300，用户个人启动器 `slaunch`，本文不代为提交）。模板与 NFT 相同，换实验名：

```bash
slaunch train 24 sifr_exp003_002_beta2_0p99_n24 \
  -m scripts.train \
  --config=projects/cosmos3/cosmos3/configs/base/experiment/text_rendering/config.py \
  -- experiment=cosmos3plus_64bm32b_t2ionly_sifr_exp003_002_pickscore_beta2_0p99 \
     job.group=cosmos3plus_t2ionly \
     job.name=cosmos3plus_64bm32b_t2ionly_sifr_exp003_002_pickscore_beta2_0p99_n24
```

2. **关键断言**（违反会直接报错，不要绕过）：`net_old_mixing_type` 必须是 `constant_<d>`；每 rank 图像数必须能被 `optim_steps_per_round` 整除；`ref_price` 与 `policy_price` 至少一个非 None；`loss_form ∈ {signed, detached}`；`eval_ema_beta ∈ [0,1]`；不支持 CP；`grad_accum_iter in (1, None)`。
3. **续训**：保证 `sifr_lambda_iter_<iter>.json` 与 checkpoint 一起存在；不要改 `optim_steps_per_round`、`budget_value`。
4. **监控 `train_sifr/*`**：`reward_mean`、`reward_success_ratio`、`advantage_abs_max`、`loss_policy`（奖励项，可以为负）、`loss_drift_policy`（漂移，正常 ~1e-3；> 0.1 已是发散前兆）、`lambda_policy`、`ctrl_policy_{dbar, delta_sq, h, drift_round, poisoned}`；梯度范数（正常 ~1e-3，0.3 以上是异常）。
5. **评测**：训练中 UGB / PickScore 用 EMA；离线评测按 NFT 报告的命令模板（`score_report_20260910_nft/how_to_report.md`），自己决定用 EMA 还是在线权重，并确保 `--experiment_name` 与权重结构一致。

## 9. 已知问题与坑

- 发散根因未定（见 7.1）；目前能稳定跑的是 exp001_001（by_epoch，记录当时 commit）与 exp003_002。
- `|A|>λ` 的负优势行无下界（3.2），一旦漂移起来没有及时的恢复力；价格控制器的最大速度远低于发散速度。
- **配置与日志对不上**：基础配置 std 默认值曾变更（第 6 节）。
- 本文是读代码整理，没有在本地运行（`.venv` 不是训练环境）；所有数值现象来自训练日志记录，未重新核实。
