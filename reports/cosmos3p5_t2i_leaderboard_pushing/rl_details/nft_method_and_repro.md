# DiffusionNFT（Cosmos3plus 64bm32b）方法、实现细节与复现手册

本文把 NFT 在 imaginaire4 里的实现拆到可以照着复现的粒度：网络与权重体系（net / net_old / net_ref / net_ema / 评测权重）、奖励怎么算、优势怎么算、损失与梯度、噪声与时间步、参数总表、实验清单、结果、已知坑。

代码位置（均在 `projects/cosmos3/cosmos3/` 下，除非另有说明）：

| 内容 | 文件 |
|---|---|
| NFT 算法与 rollout | `models/omni_mot_model_v2_rl.py`（`OmniMoTModel_RL`、`OmniMoTModel_NFT`） |
| 基类：net / EMA / LoRA / 噪声 / 采样 | `models/omni_mot_model.py` |
| 训练循环（每步只 step 一次） | `imaginaire/trainer2_rl.py`（`ImaginaireTrainer_RL`） |
| 奖励客户端 | `reward_service/image/{base_score_fn,pickscore_reward_fn,paddleocrv5_reward_fn,vlmocr_reward_fn}.py` |
| 网关配置 | `reward_service/utils.py`（`GATEWAY_CONFIG`） |
| 实验配置 | `configs/base/experiment/text_rendering/cosmos3plus_64bm32b_t2ionly_nft_exp000.py` |
| 基础训练配置 | `configs/base/experiment/text_rendering/cosmos3_ga_64bm32b_t2ionly_base.py`（`..._base_mini`） |
| 数据 | `datasets/rl_prompt_dataloader.py`，见 `dataloader_design/lance_dataloader_design_logic.md` |
| LoRA | `utils/lora.py` |
| 训练时 wandb | `callbacks/wandb_log_nft.py`（`train_nft/*`） |
| 提交 | `configs/base/experiment/text_rendering/_submit_gb300.py` |

## 0. 约定

- **t 的方向：** 干净样本在 t=0，噪声在 t=1。`x_t = ε·t + x0·(1−t)`，目标速度 `v* = dot_x_t = ε − x0`（`RectifiedFlow.get_interpolation`）。注意这与 Signed-RF-Playground 相反（Playground 里干净样本在 t=1），所以移植时 x0 端点公式要翻转（`x0_pred = x_t − t·v`，见 `_adaptive_x1_residual_and_mae` 的注释）。
- 优势符号：优势越大 = 奖励比组内均值越高 = 越应该靠近目标 `v*`。
- “一次迭代” = 1 次 rollout + 1 次 optimizer step（SIFR 则是 1 次 rollout + 3 次 optimizer step，见 SIFR 报告）。
- 记号：`θ` = 在线策略 `net`；`θ_old` = `net_old`；`θ_ref` = `net_ref`；`θ_ema` = `net_ema`。

## 1. 网络与权重体系

### 1.1 四个网络各自是什么

| 网络 | 谁创建 | 梯度 | dtype / 分片 | 用途 |
|---|---|---|---|---|
| `net` | `OmniMoTModel.set_up_model` → `build_net(dtype=master_dtype, mp_policy)` | 有（只训练 `optimizer.keys_to_select` 里的参数） | master dtype（FSDP2 混合精度：fp32 master，计算时 cast 到 `self.precision`），FSDP 分片 `data_parallel_shard_degree=32` | 在线策略 θ；训练的对象 |
| `net_old` | `OmniMoTModel_NFT.set_up_model` 里 `build_net(dtype=self.precision)`，`requires_grad_(False)`，`eval()` | 无 | `self.precision` | 旧策略：rollout 的采样网络，且在损失里提供 `v_old`（锚点）。`net_old_mixing_type=constant_0` 时**不创建** |
| `net_ref` | 同上 | 无 | `self.precision` | KL 参考策略；仅当 `kl_loss_weight != 0` 才创建 |
| `net_ema` | `OmniMoTModel.set_up_model`，仅 `config.ema.enabled` 时创建，`build_net(dtype=torch.float32)` | 无 | fp32，FSDP | 评测用的指数滑动平均权重；**所有 LoRA 配置都把 EMA 关掉**（`ema=dict(enabled=False)`） |

`net_old`、`net_ref` 的构造在 `super().set_up_model()` 之前（所以 `build_net` 里按 `config.lora_enabled` 注入 LoRA 的逻辑对它们同样生效，三个网络结构一致，参数名一一对应；`on_before_zero_grad` 里会 assert 名字顺序一致）。

### 1.2 初始化与权重加载顺序

1. **DCP 加载：** `checkpoint.load_path` 指向 exp009_sft0 iter31k（bucket `nv-00-10206-checkpoint`），`load_from_object_store`。基础配置里 `load_ema_to_reg=True`、`load_training_state=False`：从 checkpoint 里读 **EMA 权重** 填进常规 `net.*` 键（`_replace_keys_with_ema_keys`），不加载优化器 / scheduler / trainer 状态。这等价于“用 sft0 的 EMA 权重作为起点”。
2. **LoRA 键跳过：** LoRA 配置设 `checkpoint.keys_to_skip_loading=["lora_"]`，checkpoint 里没有 LoRA 参数，LoRA 不从 checkpoint 读。
3. **LoRA 注入与初始化：** 在 FSDP 包装之前、meta 设备上注入（`inject_lora_pre_fsdp`），只替换 `q_proj_moe_gen / k_proj_moe_gen / v_proj_moe_gen / o_proj_moe_gen`（生成专家的注意力投影）；前向 `y = base(x) + (alpha/r)·lora_B(lora_A(x))`，缩放系数 `alpha/r = 2`（rank16/α32 与 rank32/α64 都是 2）；`init_weights` 之后 `lora_A ~ kaiming_uniform_(a=√5)`，`lora_B = 0`，所以初始策略严格等于基础模型。只有 `lora_` 参数进入优化器（`optimizer.keys_to_select=["lora_"]`）。
4. **旧/参考网络拷贝：** `OmniMoTModel_NFT.load_pretrained_model_if_needed` 在父类逻辑（DCP 之后补 HF 权重等）完成后，取 `ref_source = net_ema（如果存在）否则 net`，用 `DTensorFastEmaModelUpdater().copy_to` 把 `ref_source` 拷到 `net_ref`（若有）和 `net_old`（若有）。LoRA 配置里 EMA 关闭 → `ref_source = net`，此时 LoRA 刚初始化（B=0），所以 `net_old` 和 `net_ref` 的 LoRA 也是 B=0，行为等于基础模型。全量微调配置（exp000_000，没有关 EMA）则是从 `net_ema` 拷。
5. **checkpoint 里存什么：** `OmniMoTModel_NFT` 没有覆盖 `state_dict`，所以 checkpoint 只含 `net.*`（以及开启 EMA 时的 `net_ema.*`）和优化器等，**不含 `net_old` / `net_ref`**。从 NFT checkpoint 续训时，第 4 步会再把续训后的 `net`（或 `net_ema`）拷给 `net_old` / `net_ref`——即 `net_ref` 变成“续训点的策略”而不是初始策略。需要严格复现 KL 锚点时不要中途续训，或改成像 SIFR 那样单独保存（SIFR 保存 `net_old.*` 并把 `net_ref` 重建为 lora_B 清零的 net）。

### 1.3 `net_old` 的更新（旧策略怎么跟随在线策略）

`on_before_zero_grad`（`ImaginaireTrainer_RL` 里在 `optimizer.step()` 之后调用）里逐参数：

```
p_old ← mix · p_old + (1 − mix) · p_new          # 对 net 与 net_old 的所有 named_parameters
```

DTensor 的处理：两边都是 DTensor 则对本地分片 `dt2lt` 就地更新；`net` 是 DTensor 而 `net_old` 不是则取 `p_new.full_tensor()`；反过来的情况直接 assert 失败。`mix` 由 `net_old_mixing_type` 决定：

| 取值 | mix | 含义 |
|---|---|---|
| `constant_0` | 0 | `on_policy=True`，不建 `net_old`；rollout 用在线 `net` 本身（临时 `net.eval()`，结束后恢复 train 模式）；损失里 `v_old = v_pol.detach()` |
| `default` | `min(0.001·iteration, 0.5)` | 随迭代逐步增大的滑动平均，上限 0.5，所以 `net_old` 最多落后在线策略约一步的一半 |
| `constant_<d>` | `d` | 例如 `constant_0.5`（SIFR 用） |

`constant_0` 与 `default` 的差别是实验里最敏感的开关之一：exp007_000（`constant_0`）在 iter90→100 崩溃，exp007_002（`default`）同配置跑到 iter120 稳定。

### 1.4 评测用权重（net_eval）怎么取

| 场景 | 取哪个权重 | 怎么做 |
|---|---|---|
| NFT LoRA 训练中的 `EveryNEval`（UGB） | 在线 `net`（含 LoRA），不用 EMA | 配置里 `eval_ugb.use_ema=False`；`use_ema=True` 且 EMA 关闭时该 callback 会直接跳过评测 |
| 训练后离线评测 LoRA checkpoint | 在线 `net` | 推理脚本 **不要加 `--use_ema`**，`--experiment_name` 用 `cosmos3_ga_64bm32b_t2ionly_base_mini_lora`（rank16/α32）或 `..._base_mini_lora_rank32`（rank32/α64，exp005 起） |
| 基线（sft0 iter31k） | EMA | 加 `--use_ema`，`--experiment_name cosmos3_ga_64bm32b_t2ionly_base_mini` |
| 开启 EMA 的配置（exp000 全量微调、SIFR） | `net_ema` | `ema_scope`：`reshard` → 缓存 `net` 参数 → 把 `net_ema` 拷进 `net` → 评测 → 恢复缓存 |

EMA 本身（开启时）：功率 EMA，`rate=0.1`，系数 `c = max Re roots([1, 7, 16−rate⁻², 12−rate⁻²]) ≈ 6.937`，每步 `beta(i) = (1 − 1/(i+1))^(c+1)`，`net_ema ← beta·net_ema + (1−beta)·net`，在 `on_before_zero_grad` 里更新。典型值：i=10 → 0.47，50 → 0.85，100 → 0.92，500 → 0.984，1000 → 0.992。NFT 的 LoRA 实验没有用 EMA；SIFR 把 `ema_beta` 覆盖成常数 0.9（每 round 一次）。

## 2. 一次迭代的完整流程

```
training_step(data_batch, iteration)
 1. 每个 rank 取 1 条 prompt（RLPromptDataloader，group_size 个相邻 rank 共享同一条）
 2. rollout：
      - _set_rollout_canvas：宽高比 "mix" 时每个 prompt 在 768 档的 IMAGE_RES_SIZE_INFO 里随机选一个；"1,1" 则固定方图
      - 每条 prompt 复制 num_rollouts_per_sample 份，每份一个 seed = torch.randint(0, 2**31-1)
      - 每 rollout_batch_size(6) 张为一个 chunk，generate_samples_from_batch(net=net_old 或 live net, guidance=1.0, num_steps=35, shift=5.0)
      - 保存解码前的 VAE latent（rollout_x0）和解码后像素 ((1+decode)/2).clamp(0,1)（rollout_x0_decoded）
      - 每个 chunk 解码完立即 reward_fn.submit(...)（异步，不等结果）
 3. one_cycle_off=True：交换到上一个迭代的 batch 来训练；首迭代无前序 batch → 返回 loss=0、不做 backward
 4. collect_reward：阻塞取回奖励，失败为 NaN
 5. compute_group_advantage → r ∈ [0,1]（见第 4 节）
 6. 对每个 loss_compute_batch_size(4) 张图的切片 × num_timestep_samples(10) 次：
      compute_nft_loss → (loss × 权重).backward()
 7. trainer：optimizer.step → net_old 折叠 / EMA 更新 → zero_grad
```

要点：
- 梯度是在 `training_step` 里手工累加的，不走 Trainer 的 grad accumulation；`ImaginaireTrainer_RL` 断言 `grad_accum_iter in (1, None)`，每次调用无条件 step 一次。
- 每个 (切片, 时间步样本) 的损失权重 `(end−start)/N/num_timestep_samples`，累加起来等价于对本 rank 所有图像和所有时间步样本取平均；跨 rank 的平均由 FSDP 的梯度归约完成。
- 时间步样本的循环里每次都会重新调用 `compute_nft_loss`，因此每次重新采噪声水平，重新打包序列（`_get_training_inputs`、`_pack_input_sequence`）。
- 训练输入里的图像：`data_batch["images"]` 由 `rollout_x0_decoded*2−1` 重建，只用来走数据管线（产生序列计划、位置编码等）；真正参与损失的 x0 是 rollout 时存下的 latent（`gen_data_clean.x0_tokens_vision` 被替换为 `rollout_x0`），避免“解码再编码”的误差。
- 只支持单个视觉 item / 样本、不支持 context parallel（`cp_size>1` 直接抛错）。

### 2.1 `nft_rollout_one_cycle_off`

打开后，第 *k* 次迭代 rollout 出的 batch 存入 `pending_train_batch`，本次训练使用第 *k−1* 次存下的 batch，这样第 *k* 次的奖励网络等待与第 *k* 次的训练重叠。后果：
- 样本来自 θ_{k−1}（采样时的 `net_old` 或在线策略），而锚点 `v_old` 在训练时是 θ_k（当前 `net_old` / detach 的在线策略），采样策略与锚点不一致；
- `reward_config.collect_timeout` 因此可以设得更紧（配置里 20 s）；
- 第一次迭代 loss=0。

## 3. 噪声与时间步（训练时）

`_get_train_noise_level_vision` 对每张图采一个 σ（`[B,1]`）：

1. 取该样本分辨率档的 shift：`rectified_flow_training_config.shift={"768": 5}`（NFT 配置覆盖了 base_mini 的 `{"768": 10}`）。
2. 原始分布：图像用 `train_time_image_distribution`，base 默认为 `logitnormal`，即 `t = sigmoid(N(0,1))`（base_mini 只把视频分布改成了 `waver`）。
3. 应用 shift：`σ = s·t / (1 + (s−1)·t)`，s=5。
4. `timesteps = σ · num_train_timesteps`。

`rollout_sigma_sampling_type` 可以替换这套采样：
- `"uniformly_sample"`：在 `linspace(0,1,N+2)[1:-1]`（N=`num_timestep_samples_per_rollout`=10）的格点上均匀随机选；
- `"sample_from_rollout_sigma"`：在 35 步 shift-5 的 UniPC 调度（`FlowUniPCMultistepScheduler.set_timesteps(35, shift=5.0)`）的 sigma 格点（去掉最后一个 0）上均匀随机选，要求 rollout 采样器是 `UniPCSampler`。

`adaptive_x1_weight=True` 时把 v 空间 MSE 换成 x0 空间 MSE 再除以其 detach 的 MAE，并乘 `adv_clip_max`（Playground 的做法）；所有 NFT 实验都关着它。

## 4. 奖励与优势怎么算

### 4.1 异步奖励服务（`BaseScoreRewardFunction`）

- 每个 rank 一个后台 asyncio 事件循环线程；`submit` 把每个样本放进队列，`dispatcher` 用信号量限制同时在飞的请求为 `num_concurrency`（NFT 配置 16）。
- 每个样本最多尝试 `num_max_retry`（2）次，每次 `asyncio.wait_for(score_one, timeout)`（120 s，VLMOCR 180 s）；全部失败则该样本结果为 `None`。
- `collect` 对每个样本 `future.result(timeout=collect_timeout)`；`collect_timeout` 默认等于 `timeout`，one-cycle-off 配置设 20 s（VLMOCR 30 s），超时也记 `None`。
- `collect_reward` 把 `None` 变成 NaN；打分函数自身解析失败也返回 NaN。NaN 的语义：**不进任何统计量，且该样本优势为 0（`r=0.5`）**。
- 每个样本通过 `reward_call_id`（`r{rank}_{uuid}_{i}`）关联结果，同一 id 重复提交会抛 `KeyError`；结果缓冲 6000 s 后过期。
- 图像在送网关前被编码：PickScore 用 PNG（`(x*255).astype(uint8)`，截断），PaddleOCR 用 WebP（`(x*255).round().clamp`，quality 85、method 5），VLMOCR 用 PNG（`round` 量化）。量化方式不同是有意对齐各自参考实现的。

### 4.2 三种奖励

**PickScore**（`PickScoreRewardFunction`）
- 网关 `pickscore`（Lepton 上的 scorer，凭据 `credentials/lepton_reward.secret`），输入（图像, `ai_caption`），返回一个浮点分数（典型 17–30）。
- 在统计优势之前先 **除以 26**（`gathered_reward / 26`），让量级落到 DiffusionNFT / flow_grpo / Playground 约定的小数值，因为 `eps=1e-4` 是加在这个量级的 std 上的。

**PaddleOCRv5**（`PaddleOCRv5RewardFunction`）
- 网关 `paddleocrv5`（Lepton scorer）。输入图像 + 真值文字；`model_name` 三选一：`paddleocrv5`、`paddleocrv5_strict`、`paddleocrv5_pned_gned`。
- 真值：`text_and_signage`。文件 prompt 版由 `RLPromptDataloader.text_and_signage_from_prompt` 取 prompt 中第一对双引号里的内容（无成对引号或为空则 `None`）；Lance 版由 `LanceCaptionFormatter(cosmos3_structure_json_text_special_handling=True)` 从结构化 caption 里抽成字符串列表。
- `paddleocrv5`：服务端返回一个标量字符串，客户端 `float()`；失败 → NaN。
- `paddleocrv5_pned_gned`：真值是字符串列表（可为空），JSON 编码后发给服务端；服务端返回 JSON `{ned_sum, min_count_pred_target, max_count_pred_target}`，客户端算
  `pned = ned_sum / min_count`，`gned = ned_sum / max_count`，`reward = 2·pned·gned/(pned+gned)`（二者之和为 0 时为 0；分母为 0 的项记 0）；任一字段缺失 / 为 None → NaN。
- 奖励本身在 [0,1]，不做 /26。

**VLMOCR**（`VLMOCRRewardFunction`）
- 通过统一网关调用 VLM（本实验 `model_name="gemini-3.8-flash@nvidiak"`，类默认 `gemini-3.1-pro`），`language="eng"`；system prompt 读自 `evaluation/text_to_image/prompts/eval_cvtg_system_prompt_eng.md`，user prompt 固定为 `Extract all visible text regions from the image, following the system instructions exactly.`，`temperature=0.1`，`max_tokens=16384`。
- 回复解析：取 YAML 代码块（优先 `yaml/yml` 标签、取最长），得到列表，每项含 `ocr` 与 `reasons_and_comments`；`ocr` 缺失、非 YAML 列表、解析失败 → NaN；空串项被丢弃。
- 打分：识别列表与真值列表做配对 NED —— 小写后的 Levenshtein 归一化 `NED_cost = ED / max(len)`，用 Hungarian（`scipy.optimize.linear_sum_assignment`）求最小总代价，`ned_sum = min(|pred|,|gt|) − min_cost`，再同样取 `pned / gned` 的调和平均；两边都为空 → 1.0。与 `evaluation/text_to_image/compute_cvtg_metric.py` 的 PNED/GNED 逐字一致（复制实现）。

### 4.3 优势（`compute_group_normalized_advantage` + `compute_group_advantage`）

输入：每个 rank 的本地 `reward`（1D，长度 `num_rollouts_per_sample`）。步骤：

1. `all_gather_tensor` 把所有 rank 的奖励按 rank 顺序拼成一个长向量（rank r 的 6 个奖励连续排在一起）。
2. 先 `validate_group_tags`：把 `ai_caption` 同样 all-gather，检查每个连续 `group_num` 块里的 caption 完全相同，否则抛 `RuntimeError`（保证“相邻 rank 共享同一 prompt”这一假设成立）。
3. `view(-1, group_num)` 得到 `[N_group, group_num]`，其中 `group_num = true_num_rollouts_per_sample = group_size × num_rollouts_per_sample`（r12：2×6；r24：4×6；r8：2×4）。
4. `valid = ~isnan`。`mean`：
   - `advantage_mean="by_group"`（默认）：每组有效值的均值；
   - `"by_epoch"`：整个迭代所有有效奖励的均值（不是 Playground 的模式，仅消融用）。
   `deviation = where(valid, reward − mean, 0)`（NaN 的偏差置 0）。
5. `std`：
   - `advantage_std="by_group"`（默认，标准 GRPO / Playground `scale_rewards: group`）：每组的总体标准差 `sqrt(Σ deviation² / count)`（只在有效项上）；
   - `"by_epoch"`（Playground `scale_rewards: epoch`）：整个迭代所有有效奖励的一个总体标准差 `std(correction=0)`。
6. `A = deviation / (std + 1e-4)`。
7. `compute_group_advantage` 再做：`A ← clamp(A, −5, 5)`，`r = (A / 5)/2 + 0.5`，得到 [0,1] 的权重 `r`；然后每个 rank 取回属于自己的那一段（按各 rank 的奖励长度累加偏移）。

数值例子（算术示意，不是实测）：某组 4 条 PickScore = [21.5, 22.5, 23.5, 24.5] → /26 = [0.8269, 0.8654, 0.9038, 0.9423]，均值 0.8846，偏差 [−0.0577, −0.0192, 0.0192, 0.0577]。
- by_group std = 0.0430 → A = [−1.34, −0.45, 0.45, 1.34] → r = [0.366, 0.455, 0.545, 0.634]；
- by_epoch，假设整个迭代 std 为 0.08 → A = [−0.72, −0.24, 0.24, 0.72] → r = [0.428, 0.476, 0.524, 0.572]。
by_epoch 下组内离散度小于全局离散度的组，优势被整体压小（r 更接近 0.5）；by_group 则把每个组拉到单位方差，组内差异再小也推得一样用力。

### 4.4 日志里怎么看

`train_nft/reward_mean`（每个 rank 先对每个 prompt 的 `num_rollouts_per_sample` 个奖励做 nanmean，再 nanmean）、`reward_mean_div26_for_pickscore`、`reward_success_ratio`（非 NaN 比例，应接近 1）、`advantage_mean`（记录的是 `r`，by_group 下围绕 0.5；失败打分的样本固定为 0.5）、`loss_policy`、`loss_kl`。wandb 去向：GCP 上进 OneLogger 的 run（见各次提交的日志）。

## 5. 损失与梯度

对每张 rollout 图像 i（取该图一个噪声水平的 `x_t`、`v*`），在线策略预测 `v_θ`，旧策略预测 `v_old`（无梯度）：

```
v_pos = β·v_θ + (1−β)·v_old
v_neg = (1+β)·v_old − β·v_θ
pos_err = mean((v_pos − v*)²)      neg_err = mean((v_neg − v*)²)
L_i = r_i·pos_err + (1−r_i)·neg_err
loss = policy_loss_weight · mean_i(L_i) + kl_loss_weight · mean_i(mean((v_θ − v_ref)²))
```

β=1（所有实验）时 `L = r·(v_θ−v*)² + (1−r)·(2v_old − v_θ − v*)²`。逐元素对 `v_θ` 求导并令其为 0，得到该损失的极小点：

```
v_opt = v_old + (2r − 1)·(v* − v_old)
```

其中 `2r−1 = clip(A,−5,5)/5 ∈ [−1,1]`。直观理解：
- `r=1`（强正优势）：目标是 `v_opt = v*`，把预测拉向真实速度；
- `r=0.5`（优势 0，包括失败打分的样本）：`v_opt = v_old`，只起到“把策略拉回旧策略”的近端项作用；
- `r=0`（强负优势）：`v_opt = 2·v_old − v*`，把预测推到远离 `v*` 的另一侧。
所以 NFT 的步长天然被旧策略限制，强度由 `|A|` 控制；`kl_loss_weight` 则额外把策略拉回 `net_ref`。

## 6. 参数总表

### 6.1 共用基础（`cosmos3plus_64bm32b_t2ionly_nft_base`）

| 类别 | 参数 | 取值 |
|---|---|---|
| 模型 | 初始化 checkpoint | `cosmos3_vfm/cosmos3_ga_t2ionly/cosmos3_ga_64bm32b_t2ionly_exp009_sft0_uhq_from_exp009_25k_lr1em5/checkpoints/iter_000031000/`（bucket `nv-00-10206-checkpoint`） |
| | 父配置 | `cosmos3_ga_64bm32b_t2ionly_base_mini`（分辨率 768，`max_num_tokens_after_packing=65536`，`joint_attn_implementation="two_way"`，`patch_spatial=2`，激活检查点 `selective`），scheduler 覆盖为 `constant` |
| | 训练 RF shift | `{"768": 5}`（base_mini 为 10） |
| | 并行 | `data_parallel_shard_degree=32`，无 CP，`compile` 关闭 |
| Rollout | guidance / 步数 / shift | 1.0 / 35 / 5.0 |
| | 分辨率 / 宽高比 | `"768"` / `"mix"` |
| | `rollout_batch_size` | 6 |
| | `num_rollouts_per_sample`（每 rank） | 6 |
| | `true_num_rollouts_per_sample`（每组） | 24（base） |
| NFT | `beta` | 1.0 |
| | `policy_loss_weight` | 1.0 |
| | `kl_loss_weight` | 0（base） |
| | `num_timestep_samples_per_rollout` | 10 |
| | `loss_compute_batch_size` | 4 |
| | `adaptive_x1_weight` | False |
| | `net_old_mixing_type` | `constant_0`（base） |
| | `rollout_sigma_sampling_type` | `None` |
| | `nft_rollout_one_cycle_off` | False（base） |
| 奖励客户端 | `num_concurrency` / `num_max_retry` / `timeout` | 16 / 2 / 120 s |
| | `advantage_mean` / `advantage_std` | `by_group` / `by_group` |
| | `adv_clip_max` / `eps` | 5.0 / 1e-4 |
| 优化器 | AdamW | `lr 3e-4`，`betas [0.9, 0.999]`，`weight_decay 1e-4`（base_mini 的 `1e-4 / [0.9,0.99] / 0.05` 被覆盖） |
| | 梯度裁剪 | 默认 callback 组里 `GradClip(clip_norm=1.0)`；exp007_001 覆盖为 0.05 |
| 训练器 | `max_iter` / `save_iter` / `logging_iter` | 500 / 10 / 1 |
| | 训练中评测 | `EveryNEval` UGB `v2_1170L_opus`，每 10 迭代，guidance 4.0，50 步，`limit_one_per_rank_step_25`，`run_at_start=True`，`wandb_display_num=4` |
| | callback | `datasample_count every_n=10`，`wandb_nft = NFTWandbCallback` |
| 数据 | 训练 prompt | `dl_rl_prompt_only_pickscore`：`pickscore_train.txt`（25,432 条），`shuffle=True`，每 rank `batch_size=1`，`group_size=4` |

**LoRA 变体**（`nft_base_lora`，exp004 起全部使用）：`lora_enabled=True`，rank 16 / α 32，目标层 `q/k/v/o_proj_moe_gen`，`ema.enabled=False`，`optimizer.keys_to_select=["lora_"]`，`lr 3e-4`，`checkpoint.keys_to_skip_loading=["lora_"]`，`load_ema_to_reg=True`，`eval_ugb.use_ema=False`。`nft_base_lora_rank32`：rank 32 / α 64。

### 6.2 规模推导（24 节点，按每节点 4 GPU → 96 个 rank，未用日志核实）

| 设置 | group_size | 每组 rollout | 每迭代不同 prompt 数 | 每迭代图像数 |
|---|---|---|---|---|
| exp000 / exp004_000 / 004_001 | 4 | 24 | 24 | 576 |
| r12 系列（exp004_002+、exp005_000–004、006、007） | 2 | 12 | 48 | 576 |
| exp005_005（r24） | 4 | 24 | 24 | 576 |
| exp004_004（r8） | 2 | 8 | 48 | 384（`num_rollouts_per_sample=4`） |

每张图 10 个时间步样本，576 张图 → 每迭代约 5,760 次前向 + 反向。

## 7. 实验清单

每行继承“父配置”，只改所列项。“1c-off” = `nft_rollout_one_cycle_off=True`（同时 `reward_config.timeout 120`、`collect_timeout 20`）；“r” = `true_num_rollouts_per_sample`；“gs” = `dataloader_train.group_size`。

| 实验 | 父配置 | 改动 | 奖励 | 训练 prompt |
|---|---|---|---|---|
| exp000_000 | base | 全参数微调（无 LoRA，EMA 保持默认），r24 | PickScore | pickscore_train |
| exp004_000 | base_lora（r16/α32） | LoRA，r24，gs4 | PickScore | pickscore_train |
| exp004_001 | base_lora | + 1c-off | PickScore | pickscore_train |
| exp004_002 | base_lora | r12，gs2 | PickScore | pickscore_train |
| exp004_003 | 004_002 | + 1c-off | PickScore | pickscore_train |
| exp004_004 | 004_003 | `rollout_batch_size 4`、`num_rollouts_per_sample 4`、r8、gs2 | PickScore | pickscore_train |
| exp004_005 | 004_003 | `kl_loss_weight=0.1` | PickScore | pickscore_train |
| exp004_006 | 004_003 | `kl_loss_weight=1e-4` | PickScore | pickscore_train |
| exp004_007 | 004_003 | `rollout_sigma_sampling_type="uniformly_sample"` | PickScore | pickscore_train |
| exp004_008 | 004_003 | `rollout_sigma_sampling_type="sample_from_rollout_sigma"` | PickScore | pickscore_train |
| exp005_000 | 004_003 | LoRA rank 32 / α 64，`kl 1e-4` | PickScore | pickscore_train |
| exp005_001 | 005_000 | `lr 1e-4` | PickScore | pickscore_train |
| exp005_002 | 005_000 | rollout `guidance=4.0` | PickScore | pickscore_train |
| exp005_003 | 005_000 | `advantage_std="by_epoch"` | PickScore | pickscore_train |
| exp005_004 | 005_003 | `lr 1e-4` | PickScore | pickscore_train |
| exp005_005 | 005_003 | r24，gs4 | PickScore | pickscore_train |
| **exp005_006** | 005_003 | `rollout_aspect_ratio="1,1"` | PickScore | pickscore_train |
| exp006_000 | 004_000 | rank 32/α 64，r12，宽高比 `1,1`，1c-off，`kl 1e-4`，std `by_epoch`，gs2 | PickScore | Lance coyo700m，`cosmos3_structured_json`，`gc_reformat_p={"markdown":1}`，`cfg_dropout_rate=0` |
| exp007_000 | base_lora_rank32 | PaddleOCRv5，1c-off，`kl 1e-4`，std `by_epoch`，r12，gs2 | PaddleOCRv5 | `ocr_train.txt` |
| exp007_001 | 007_000 | `grad_clip.clip_norm=0.05` | PaddleOCRv5 | `ocr_train.txt` |
| exp007_002 | 007_000 | `net_old_mixing_type="default"` | PaddleOCRv5 | `ocr_train.txt` |
| exp007_003 | 007_000 | `model_name="paddleocrv5_pned_gned"`，数据 → Lance `synthetic_scene_text_v0`（`cosmos3_structured_json_comprehensive_t2i_dense`，`cosmos3_structure_json_text_special_handling=True`） | PaddleOCRv5 pned+gned | synthetic scene text v0 |
| exp007_004 | 007_003 | `caption_type="cosmos3_structured_json"` | PaddleOCRv5 pned+gned | synthetic scene text v0 |
| exp007_005 | 007_003 | `type="VLMOCR"`，`gemini-3.8-flash@nvidiak`，timeout 180 / collect 30 | VLMOCR | synthetic scene text v0 |
| exp007_006 | 007_005 | `caption_type="cosmos3_structured_json"` | VLMOCR | synthetic scene text v0 |

配置层面的坑：exp004_00x 与 exp005_00x 通过 `copy.deepcopy` 从 `base_lora` 派生，rank 32 是在 exp005_000 里显式覆盖并沿 exp005 链传下去的，exp004_00x 仍是 rank16/α32。exp006_000 从 exp004_000（r24 基线）派生后覆盖了一堆项，exp007 从 `base_lora_rank32` 派生，二者与 exp005 链**不是同一条继承链**，对比时要逐项核对。

## 8. 结果

基线 = exp009_sft0 iter31k（PickScore 21.91，UGB all 91.35 / Text Gen 67.3，CVTG gned 0.8005 / pned 0.8743）。数字取自 `score_report_20260910_nft/{pickscore,ugb,cvtg}.md`，`—` 表示该处无记录。PickScore：2048 条 prompt，1024²，guidance 4.0，50 步。UGB：`v2_1170L_opus`，1024²，judge `gemini-3.1-pro@nvidia`。CVTG：`cvtg500L_opus`。

### 8.1 PickScore 奖励

| 实验 | iter | PickScore | UGB all | 备注 |
|---|---|---|---|---|
| exp001_000（更早系列，LoRA n48） | 60 | 22.83 | — | |
| exp004_000 | 30 | **23.02** | — | exp004 在 iter30 最佳 |
| exp004_001（1c-off） | 30 | 22.96 | — | |
| exp004_002（r12） | 30 | 22.95 | — | |
| exp004_003（r12+1c-off） | 30 | 22.91 | — | |
| exp004_004（r8+1c-off） | 30 | 22.74 | — | 每组 rollout 更少更差 |
| exp005_006 | 30/100/150/210/300/360/400 | 22.86/23.44/23.60/23.72/23.64/23.73/22.36 | 88.98 @150 | iter10→60 快速上升；iter70–190 平台；iter200–290 峰值带；~iter300 后振荡（相邻 checkpoint 相差约 1 分）；更稳妥的选择是 iter210 |
| exp006_000（coyo700m prompt） | 50/100/150/200 | 22.38/23.07/22.78/23.11 | 57.17 @200 | UGB 崩 ~34 分，Text Gen 16.8 |

exp004_005–008 与 exp005_000–005 无打分记录。

### 8.2 OCR 奖励

| 实验 | iter | UGB all / TG | CVTG gned / pned | 备注 |
|---|---|---|---|---|
| exp007_000（PaddleOCR，`constant_0`） | 50 | 90.28 / — | 0.8626 / 0.9010 | 可用 checkpoint 中最佳 |
| | 80 / 90 | 89.74 / 89.19 | 0.8546 / 0.8935；0.8496 / 0.8902 | 正常 |
| | 100 / 150 | 45.66 / 44.28 | 未打分 | iter90→100 间崩溃（竖条纹 → 绿色噪点） |
| exp007_002（`net_old` default mix） | 50 | 90.46 / TG 70.7 | 0.8373 / 0.8940 | |
| | 120 | 89.82 / TG 70.7 | 0.8421 / 0.8948 | 越过 007_000 崩溃窗口，仍稳定 |
| exp007_003（pned+gned，synthetic scene text） | 50 / 90 / 110 | 89.92 / 88.81 / 87.01（TG 61.1/69.7/65.9） | 0.8812/0.9352；0.8587/**0.9524**；0.8523/0.9354 | 相对基线 UGB −1.4 / −2.5 / −4.3 |
| exp007_004（007_003+结构化 caption） | 50 / 80 | 89.03 / 86.40（TG 61.5/64.9） | 0.8448/0.9058；**0.9143/0.9613** | CVTG 全表最佳，iter80 UGB −4.9 |
| exp007_005（VLMOCR） | 50 | 89.42 / TG 66.3 | 0.8081 / 0.9480 | pned 高，精确匹配低 |
| exp007_006（VLMOCR+结构化 caption） | 50/80/90 | 89.94/89.05/86.17（TG 72.6/70.2/70.7） | 0.8244/0.9457；0.8931/0.9574；0.8985/0.9594 | Text Gen 全表最佳；iter80→90 断崖 |

exp007_001（grad clip 0.05）无打分记录。

## 9. 复现步骤

1. **提交训练**（gb300，用户的个人启动器 `slaunch`；本文不代为提交）：

```bash
slaunch train 24 nft_exp005_006_pickscore_lora_r12_n24 \
  -m scripts.train \
  --config=projects/cosmos3/cosmos3/configs/base/experiment/text_rendering/config.py \
  -- experiment=cosmos3plus_64bm32b_t2ionly_nft_exp005_006_pickscore_lora_r12_onecycleoff_rank32_epochstd_sq \
     job.group=cosmos3plus_t2ionly \
     job.name=cosmos3plus_64bm32b_t2ionly_nft_exp005_006_pickscore_lora_r12_onecycleoff_rank32_epochstd_sq_n24
```

   `job.name` 末尾的 `_n24` 是节点数。凭据：`credentials/lepton_reward.secret`（奖励网关）、`credentials/gcp_checkpoint.secret`（checkpoint 与 debug 图像）、数据凭据见数据读取报告。OneLogger 规则、OCI gb300 的 `NCCL_IB_HCA` / preflight 注意事项见 `.claude/projects/.../memory` 的集群备忘。
2. **选对实验名**：按第 7 节找到实验，核对它的继承链（尤其 rank、`kl_loss_weight`、`advantage_std`、`one_cycle_off`、`rollout_aspect_ratio`）。PickScore 实验必须用方图 rollout（`rollout_aspect_ratio="1,1"`），不要用默认的 `"mix"`。
3. **监控**：`train_nft/reward_mean`、`reward_success_ratio`、`loss_policy`；PickScore 的每 10 迭代 UGB 在 wandb 的 `eval_ugb` 下。reward_mean 持续下滑或 UGB 分数跳崖说明已进入崩溃区，参照第 8 节选 checkpoint。
4. **评测**：命令模板见 `score_report_20260910_nft/how_to_report.md` 与 `pickscore.md`（LoRA 评测一律不加 `--use_ema`，`--experiment_name` 用 `..._base_mini_lora` / `..._base_mini_lora_rank32`）。

## 10. 已知问题与坑

- **崩溃根因未定。** 组内 std 放大崩溃，`by_epoch` 只推迟；NFT 的 PickScore 基线 exp005_006 在 ~iter300 后振荡，iter425–450 附近奖励从 ~24 掉到 ~16。尚存的与 Playground 非崩溃基线的差异：one-cycle-off 的陈旧性、`constant_0` 与默认 mix、r12 vs r24、训练 sigma 调度（exp004_007/008 在测）、lr 3e-4 vs 1e-4（exp005_001/004）、原始策略评测 vs EMA-0.9 评测 adapter。
- **`net_old_mixing_type` 是最敏感开关。** exp007_000 与 exp007_002 唯一差别就是它。
- **PickScore 过拟合：** 数据分布窄（coyo700m 短 caption）时 UGB 崩 34 分（exp006_000）；奖励是审美偏好，没有 prompt 遵循信号，KL 1e-4 + one-cycle-off 不足以保住遵循。
- **续训不会恢复 `net_old` / `net_ref`**（见 1.2 第 5 点），也不会恢复数据位置。
- **失败打分必须保持零优势**：偏差用 `where(valid, r−mean, 0)`，零填充只在求和内部安全。
- **本文是读代码整理，没有在本地运行**；`.venv` 不是训练环境，行为以集群上的运行为准。
