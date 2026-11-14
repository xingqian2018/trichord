# REPA：原论文 vs meshy-research 里的实现

2026-11-07

本文说明 REPA（Representation Alignment）的做法，并把原论文（Yu et al., ICLR 2025，SiT-XL/2）和 meshy-research repo 里的实现逐项对比。

- 代码位置：meshy-research 的 `origin/latv54-dino` 分支（`origin/latv54` 上也有）
- 相关 commit：`61dd69b add REPA`、`e63d412 fix local attn and repa`
- 不包含：DINO 体素特征的离线生成过程（多视角 DINO 特征如何反投到体素）

## 1. REPA 是什么

REPA 只做一件事：在扩散 loss 之外再加一个辅助 loss，让 denoiser 某几层的 hidden state 去对齐一个冻结的预训练 encoder（DINO）的特征。推理时投影头直接丢掉，没有额外开销。

```
  干净数据 x0 ─────────┬──────────────────────────────────────────┐
  (latent tokens)     │                                          │
                      │ 加噪 (t ~ U[0,1])                         │ 冻结的 DINO（离线算好也行）
                      ▼                                          ▼
                 x_t = (1-t)·x0 + t·ε                      y = DINO(x0)   [L, 1024]
                      │                                    (和 token 一一对应)
                      ▼                                          │
        ┌───────────────────────────┐                            │
        │   Denoiser (DiT)          │                            │
        │   block 0                 │                            │
        │   ...                     │                            │
        │   block 3 ──────── h3 ────┼──► MLP_3 ──► ĥ3 ──┐        │
        │   ...                     │                   │        │
        │   block 7 ──────── h7 ────┼──► MLP_7 ──► ĥ7 ──┤        │
        │   ...                     │   (投影头，         │        │
        │   block 11 ─────── h11 ───┼──► 训练完就丢掉)──┤        │
        │   ...                     │                   ▼        ▼
        │   block N                 │          ┌──────────────────────────┐
        │      │                    │          │ 逐 token 算 cosine 相似度  │
        └──────┼────────────────────┘          │ L_align = Σ_k λ_k ·       │
               ▼                               │   mean(1 - cos(ĥk, y))    │
          v_pred                               └────────────┬─────────────┘
               │                                            │
     L_diff = ‖v_pred - (ε - x0)‖²                          │
               │                                            │
               └──────────────► L = L_diff + L_align ◄──────┘
                                       │
                                    backward
                         (梯度流过 MLP_k 和 denoiser 的前 k 层；DINO 不更新)
```

### 逐 token 对齐

```
  denoiser 中间层 (只取 latent token，丢掉 text/image 条件 token)
  h_k:   | tok0 | tok1 | tok2 | tok3 | ... | tokL | pad | pad |
            │      │      │      │            │
          MLP    MLP    MLP    MLP          MLP        ← 同一个 MLP_k，逐 token 作用
           L2     L2     L2     L2           L2        ← 归一化
           cos    cos    cos    cos          cos       ← 和下面对应位置算点积
           L2     L2     L2     L2           L2
  y:     | dino0| dino1|  0   | dino3| ... | dinoL|  0  |  0  |
                         ↑                           ↑
                     mask=False                  mask=False
                   (这个位置没有 DINO 特征)        (padding)
```

### 实现只需要 4 步

| 步骤 | 做什么 | repo 里对应的代码 |
|---|---|---|
| ① 准备目标 | 每个 token 一个 DINO 特征 y_i，加一个有效 mask | `get_sparse_feats()` 按体素坐标 gather；`align_mask = abs().sum() > 1e-3` |
| ② 取中间层 | denoiser.forward 多返回几层的 hidden state，只取要对齐的 token | `par_diffusion.py` 的 `intermediate_feat_names` |
| ③ 投影 + loss | 每层一个小 MLP 投到目标维度，loss = λ·mean(1 − cos) | `AlignmentHead` / `AlignmentMLP` |
| ④ 加到总 loss | `L = L_diff + L_align`，一起 backward；推理时删掉 MLP | `train_dm_repa.py`：`loss = diff_loss.mean() + align_loss` |

## 2. 对齐哪一层：靠近输入还是靠近输出

**应该对齐靠近输入的前中段层，不要对齐靠近输出的层。**

- 原论文在 SiT-XL/2（28 层）上做过消融：对齐第 8 层左右（前 1/3 内）效果最好，对齐的层越深 FID 越差。
- 原因：前中段负责从带噪输入里认出语义和结构，这正是 DINO 擅长的；后段要把语义转成具体的 velocity 预测，需要高频、跟 t 相关的细节，DINO 特征里没有这些。在后段强行对齐会和扩散 loss 抢容量。

```
  x_t ──► [ 前中段：理解"这是什么" ] ──► [ 后段：生成"该怎么去噪" ] ──► v_pred
              ↑                              ↑
         语义表征，适合和 DINO 对齐        高频细节和速度场，跟 DINO 无关
         (REPA 加在这里)                  (要留给它自由学)
```

repo 的 denoiser 是 `multi_stream_layers: 4` + `fused_layers: 26`，一共 30 层，对齐的层换算成全局层号是：

```
 层号:   1  2  3  [4] 5  6  7  [8] 9 10 11 [12] 13 14 15 [16] 17 18 19 [20] 21 ... 30
         └─ msblock ─┘└──────────────────── fusedblock ─────────────────────────────┘
 对齐:            ✓              ✓              ✓              ✓              ✓     (后 10 层不对齐)
```

## 3. 原论文 vs repo 逐项对比

| 项目 | 原论文 | repo（`origin/latv54-dino`） |
|---|---|---|
| 任务 | 2D 图像，ImageNet 256 | 3D 稀疏体素 latent（SDF） |
| denoiser | SiT-XL/2，28 层 | 30 层：4 层 multi-stream + 26 层 fused，带 local/global window attention |
| 对齐目标 | 训练时在线跑冻结的 DINOv2，得到 patch 特征 | 离线预存的 3D 体素 DINO 特征，按体素坐标对应到 latent token |
| 目标维度 | 768（DINOv2-B） | 1024 |
| token 对应 | patch ↔ patch，天然对齐 | 先 scatter 到稠密网格，再按 latent 坐标 gather |
| 无效 token | 没有这个问题 | 用 mask 去掉：没有 DINO 特征的体素和 padding |
| 对齐层数 | 1 层 | 5 层 |
| 对齐位置 | 第 8 层，约 28% | 第 4/8/12/16/20 层，覆盖 13%~67% |
| 只取哪些 token | 所有 image token | 只取 LATENTS token，丢掉 text/image 条件 token |
| 投影头 | 3 层 MLP + SiLU，最后一层是线性的 | 每层一个独立 MLP，`n_layers=2`，最后一层后面也有 SiLU |
| loss | 负 cosine 相似度，逐 patch 取平均 | `1 - cos`（用 `rms_norm·D^-0.5` 做 L2 归一化），逐 token 取平均 |
| 权重 λ | 0.5 | 每层 0.1，5 层加起来 0.5 |
| 样本权重 | 没有 | 和扩散 loss 共用每个样本的 `loss_weight` |
| 时间步 | 所有 t 都加，不做 t 加权 | 一样，所有 t 都加 |
| 并行 | 普通 DDP | denoiser 用 FSDP2 + SP/HP；中间特征先 `all_gather` 拼成全长；投影头单独走 DDP |
| 推理 | 丢掉投影头 | 丢掉投影头 |

一致的部分：loss 形式、总权重 0.5、所有时间步都加、推理时丢掉投影头。

## 4. 主要差异和风险点

1. **对齐层数多、位置偏深。** 最深的对齐层是第 20 层（67%），而原论文的消融显示对齐越深越差。建议做对照实验：只对齐 `fusedblock_03`（第 8 层），和 5 层全对齐比较。
2. **投影头最后一层带 SiLU。** 输出被限制在 ≥ −0.28 的范围，而 DINO 特征有正有负，cosine 可能对不满。改成最后一层是线性的，和原论文一致，更稳妥。
3. **离线目标。** 省掉了训练时跑 DINO 的计算，但目标特征的质量完全取决于离线生成那一步。这是 3D 场景下合理的改动。

## 5. 并行实现细节

**`get_full_seq`：把 SP 切开的序列拼回来。** denoiser 开了 sequence parallel，中间层每张卡只有 1/sp_size 段 token。但 `token_types` mask 和 DINO 目标都是全长的，所以取中间特征时先用可反传的 `torch.distributed.nn.all_gather` 拼回全长，再挑出 latent token。代价是 SP group 内每张卡都冗余地算一遍对齐 loss，并且多存 5 份全长中间特征。模型最终输出前也是同样 gather 的，所以两个 loss 的梯度路径一致。

**投影头单独用 DDP（`force_ddp=True`）。** 投影头很小，用 FSDP2 分片省不了显存，反而多了通信，所以每张卡存一份完整参数，梯度用 all-reduce。SP group 内各卡算的 loss 相同，取平均后梯度不变。另外，`force_ddp` 的模型会跳过 EMA；投影头推理时不用，所以没有影响。

## 6. 配置（`configs/super_dm/training/repa.yaml`）

```yaml
training:
  alignment:
    n_layers: 2
    feat_names: ["msblock_03", "fusedblock_03", "fusedblock_07", "fusedblock_11", "fusedblock_15"]
    loss_weights: [0.1, 0.1, 0.1, 0.1, 0.1]
    target_dim: 1024
datasets:
  train:
    params:
      load_dino_voxels: true
```

## 7. 新接 REPA 的建议起点

1. 先只对齐一层，放在总深度的 1/4 到 1/3 处，λ = 0.5。
2. 想对齐多层的话，也都放在前半段，后 1/3 不要碰。
3. 多层时权重分摊，总和保持在 0.5 左右。
4. 投影头用原论文的结构：3 层 MLP + SiLU，最后一层线性。
