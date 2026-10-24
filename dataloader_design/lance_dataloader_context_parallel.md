# Lance v2 dataloader 在 context parallel（CP）下的数据流

背景：测试 470B（`cosmos3p5_ga_470bm235b_*`，dp_shard=128 × cp=8）时，担心 Lance v2 dataloader 不支持 CP。理由是开 CP 后同一个 CP 组的 rank 应该算同一个 batch，而 v2 让每个 rank 读到的数据各不相同。本文说明为什么这不是问题。Lance v2 本身的设计见同目录 `lance_dataloader_design_logic.md`。

## 1. 结论

CP 确实要求同一个 CP 组的所有 rank 在同一步算同一个 batch，各自算序列的 1/C。但这 **不需要** dataloader 在组内读同一份数据。

- Lance v2 loader 不感知 CP，每个 rank 照常读互不重复的独立 batch。
- "组内同一个 batch" 由 trainer 和 model 在 dataloader 之后完成。

所以 Lance v2 不需要为 CP 做任何改动。

## 2. 机制：CP data window 轮转

以 C 个 step（C 为 CP 组大小）为一个窗口。470B 实验里 C=8，下图用 C=4 举例。

1. **窗口开头，组内所有 rank 同时 `next()` 一次。** 各自读到不同的 batch（A/B/C/D，互不重复由 `LanceScanner` 的 TCPStore 全局 chunk 计数器保证），各自做一次 tokenize + VAE 编码，把编码后的 payload 缓存在本地。
2. **之后 C−1 步谁都不 `next()`。** trainer 把窗口开头取到的 batch 原样返回，model 忽略它。
3. **第 s 步由 CP rank s 把自己的缓存广播给全组。** 全组用这一份 payload 计算，打包后的序列再按 CP 切分。
4. 最后一个 slot 之后清空缓存，进入下一个窗口。

```
              rank0    rank1    rank2    rank3
step 0        next→A   next→B   next→C   next→D    ← 同时各读 1 个 batch，各编码 1 次
              rank0 ──broadcast A──▶ 全组：[A¼] [A¼] [A¼] [A¼]
step 1        -        -        -        -         ← 不读，用缓存
              rank1 ──broadcast B──▶ 全组：[B¼] [B¼] [B¼] [B¼]
step 2        -        -        -        -
              rank2 ──broadcast C──▶ 全组：[C¼] [C¼] [C¼] [C¼]
step 3        -        -        -        -
              rank3 ──broadcast D──▶ 全组：[D¼] [D¼] [D¼] [D¼]
              ─────────── 窗口结束，清缓存 ───────────
step 4        next→E   next→F   next→G   next→H    ← 下一个窗口
```

容易混淆的一点："每步只有一个 rank 出数据" 发生在 **广播层**，不是 dataloader 层。dataloader 层是窗口开头全组同时读一次，之后 C−1 步都不读。

## 3. 为什么这样设计

| 同样算 C 个 step | 组内读同一个 batch | 轮转（当前实现） |
|---|---|---|
| dataloader 读取次数 | C × C | C |
| tokenize + VAE 编码次数 | C × C（同一份重复算） | C |
| 每步组内算的数据 | 同一个 batch | 同一个 batch |

两种方式每步算的都是同一个 batch，但轮转方式没有重复的下载、解码和编码。每个 rank 对 dataloader 的吞吐需求也降到原来的 1/C。

## 4. 代码位置（470B 实际走的路径）

| 环节 | 位置 | 说明 |
|---|---|---|
| Trainer 类 | `projects/cosmos3/cosmos3/configs/base/config.py` | `c.trainer.type = ImaginaireTrainer`，cosmos3 没有子类覆盖 |
| 取数据 | `imaginaire/trainer.py` 的 `ContextParallelDataWindow`、`_fetch_data_batch` | 只有窗口未激活（窗口开头）时才 `next(dataloader_iter)`；在 CP 组内 all-reduce stop 信号 |
| Model 类 | `configs/base/defaults/model.py`：`mot_fsdp2` → `omni_mot_model_v2.OmniMoTModel` | 继承 `omni_mot_model.OmniMoTModel`，只覆盖 optimizer 初始化和采样，没有覆盖训练路径 |
| 编码 + 广播 | `models/omni_mot_model.py` 的 `_get_training_inputs` | slot 0 调 `_prepare_training_data`（tokenize + VAE）后用 `_pack_training_payload` 缓存；每个 slot 用 `broadcast_context_parallel_object(owner_rank=slot)` 广播；`_cp_window_slot` 按 C 取模推进 |
| 一致性检查 | `ContextParallelDataWindow.assert_synced_with_model` | trainer 的 offset 与 model 的 `_cp_window_slot` 不一致时直接报错 |
| 续训 | `ImaginaireTrainer._resume_dataloader_fetch_count` | 丢弃未完成的窗口，从下一个原始 batch 开始 |

引入该机制的 commit：`feb199e6150`（"Optimization to reduce the tokenizer overhead for CP > 1"，2026-07-30）。

## 5. 对 Lance v2 的影响

- **数据不重复：** 各 rank 读到的数据由 `LanceScanner` 的全局计数器保证互不重复，这正是轮转机制要求 dataloader 提供的唯一性质。
- **不会多读：** DataLoader worker 预取 `prefetch_factor`（默认 2）个 batch 后就阻塞等待，不会因为消费变成 1/C 而多读。
- **配比：** `LanceImageUnionDataset` 的槽位按 world rank 分配，同一 CP 组内的 rank 可能读不同数据集。每步只用一个 owner 的 batch，所以配比在期望上不变；每步仍是单一数据集来源。
- **`DataSampleCount` 回调：** CP 下统计的是本 rank 窗口开头取到的 batch，每个 batch 会被计 C 次，而且计的是本地 batch，不是当步广播的 owner batch。它上报的是比例，所以配比曲线不受影响，但绝对计数不能直接解读为样本数。

## 6. 验证状态与核对方法

本文是对代码的阅读整理，未在集群上实测。核对方法：

1. 在 `_get_training_inputs` 广播之后，临时打印 `(iteration, cp_rank, cp_window_slot)` 和 payload 中第一个样本 caption 的前若干字符。预期：
   - 同一 iteration 内，CP 组所有 rank 打印的内容一致；
   - 相邻 C 个 iteration 之间互不相同。
2. 看 wandb 的 `timer/dataloader_train`：只在每个窗口的第一步有明显耗时，其余 C−1 步接近 0。
