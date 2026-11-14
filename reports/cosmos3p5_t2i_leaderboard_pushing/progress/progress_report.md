# Cosmos3.5 T2I Leaderboard 冲刺进展

更新于 2026-11-12

## 整体 Action 计划

### Evaluation

- [Done] 接入 Qwen-Image-Bench（EMNLP 2026, arXiv 2605.28091）作为额外 benchmark。
  - 5 个维度：visual quality、aesthetics、prompt alignment、real-world fidelity、creative generation
- [Done] 准备适合 Cosmos 需求的 baseline prompts。
- [Done] 重新计算 baseline scores。
  - 报告：[Design Doc: QImageBench](https://docs.google.com/document/d/14rYXYUI6Jpq4WHGXrDpOEZWXdPR8-oGPVnucS2ntxTw/edit?tab=t.0)
- [Untracked] 登陆团队 official benchmark system。

### Data

- [In Progress] 追踪并整理现有 Cosmos 图像数据。凡是数据相关的都算 UQ 数据候选，按子项目分别跟踪。
  - [Done] 针对所有 Image 数据的追踪统计
    - 报告：[Report: Cosmos3.5 Image Training Data Stats](https://docs.google.com/document/d/1o1P3OlvYmV4ibPRFlXARaENRbbL8FXD63oU2geswZEc/edit?tab=t.0#heading=h.e3rm0z80wmcg)
  - [Not Start] Super resolution 数据研究
  - [Not Start] Editing 数据研究
  - [Not Start] Multi-reference 数据研究
- [In Progress] 追踪 Getty / SSTK 新数据获取并做测试。
  - 涉及外部协议，可能赶不上计划时间线。
  - [Done] Getty 数据研究
    - 报告：[Report: Cosmos3.5 Getty Image Data Investigation](https://docs.google.com/document/d/1gldvwfqfYIIwVo6WBIVLYtHtQH-_smwxkVZjx3iD2nk/edit?tab=t.0#heading=h.e3rm0z80wmcg)
  - [Done] SSTK 数据研究
    - 报告：[Report: Cosmos3.5 SSTK Image Data Investigation](https://docs.google.com/document/d/1bSdqa7wkP0YUAS-MVuo8C17zcUTHJQMPQ2q-SoRPokM/edit?tab=t.0)
  - [Blocked] 正式数据获取
- [In Progress] OCR：持续生成新的 poster / UI / chart 数据。
  - [In Progress] 高质量图片数据生成
  - 高清分辨率图片生成，作为 SR 数据
  - [Not Start] 高质量 Caption 生成
- [In Progress] 设计新的 filter，可以将 40M 数据筛选到 1M 大小。
- [In Progress] SDG 数据研究
  - [In Progress] 针对已经生成的 7M Cosmos3 Proxy 数据进行可能的优化和 filter
  - [Not Start] 根据已有的 Super Resolution 模型，把图片放大再缩小，看数据质量是否有提升。

### Modeling & Training

- [Done] 对 midtrain 数据 recipe 的研究，以及数据使用报告的最终指引。
  - 报告：[Report: Cosmos3.5 Mid-train Image Recipe](https://docs.google.com/document/d/1cC03VVSlRiEv8Z_S8ULZAGeH-wSr0lNUXLZ6hjwyy2M/edit?tab=t.0)
- [In Progress] 对 SDG 数据混合比例做 ablation 研究。
- [Not Start] 针对 real 数据 filter 等的 ablation 研究。
- [Not Start] 对 LoRA 设置做 ablation，找到更适合 text-to-image 的配置。
- [Not Start] 针对模型进行 RL 的研究（可作为最后的提升手段）。

### Prompt Upsampling

- [In Progress] 找到最佳 upsampler 方法。
  - 报告：[Design Doc: Cosmos 3.5 Image Prompt Upsampler Capability](https://docs.google.com/document/d/1WL13lMwbGdVlZRwTgH6oxGwqcqqoNUIHNCTrS4mZHmg/edit?tab=t.nt1wa0vr8cte)
  - [Done] K19 Schema 最优：长而密集的 prompt，加轻量的结构排序。
  - [Done] 采用新的 QImageBench，看 K19 是否依旧有效。
  - [Deprioritized] 进一步研究更前沿的 Prompt Upsampling Schema。
  - [In Progress] Finetune Reasoner 作为前沿的 Image Prompt Upsampler，研究提升效果是否能追赶 KIMI / Astra 等前沿模型。
- [Not Start] 设计并执行最新的 Captioner，让他与 Prompt Upsampler 的基本原则对齐。
  - [Not Start] Captioner V3 基础设施的建设
  - [Not Start] Captioner V3 针对高质量数据的 Captioning

### Leaderboard Tracking

- [Not Start] 搭建 Arena serving 的 endpoint。
- [Not Start] 早期版本 leaderboard 提交，再根据实际测试覆盖的内容调优模型。

### Other Research

- [Deprioritized] 尝试 hot-swap VAE，看能否提升 Cosmos3.5 图像质量，尤其是小字。
- [Not Start] 按需 ablate 其他模型参数。

## 周汇报流水

周汇报文档：[Design Doc: Cosmos3.5 Text2Image Lora UQ-Midtrain SoTA Plan](https://docs.google.com/document/d/1L6LFDJkRzTjT2koI2FjSvaqi8by18Fb-7uSGFiWAONI/edit?tab=t.0#heading=h.e3rm0z80wmcg)

### 2026-11-12

#### 本周结果

1. 新的 QImageBench（Xingqian Xu 开发）已正式采用，并合并进 data 和 research 两个环节。
   - 设计文档：[Design Doc: QImageBench](https://docs.google.com/document/d/14rYXYUI6Jpq4WHGXrDpOEZWXdPR8-oGPVnucS2ntxTw/edit?tab=t.0)
2. **关键结论：**Ruofan Liang 和 Jiaojiao Fan 在新 QImageBench 上验证，K19 prompt upsampling 依然有效。他们正基于 QImageBench 更细粒度的打分，在 open loop 和 close loop 两个方向上继续提升。
3. Ruofan 用 QImageBench 打分器对 SDG 数据和真实数据做了初步对比研究，没有发现 SDG 数据有显著问题。至少在这个结果上，基于 Cosmos 的 SDG 数据可以安全地用于训练。
4. Jiaojiao 正在 finetune Cosmos Reason，验证 prompt upsampling 的 data trace 是否有用。
5. Ultra Training 的 infrastructure 已跑通，为更大规模的实验做好准备。

#### 近期计划

1. 对当前所有 midtrain 训练数据做新一轮 filtering，筛选出最佳训练数据。@Xingqian
2. 继续跟踪 Getty、SSTK 等新的高质量 stock photo 数据源。@Xingqian
   - 报告：[Report: Cosmos3.5 SSTK Image Data Investigation](https://docs.google.com/document/d/1bSdqa7wkP0YUAS-MVuo8C17zcUTHJQMPQ2q-SoRPokM/edit?tab=t.0)
3. 继续通过 open loop 和 close loop 提升 prompt upsampling 效果。内容包括新的测试、新的 reason ablation，以及 reason 与 generator 如何协作。@Jiaojiao
4. 对 SDG 数据做进一步的质量提升，通过 ablation 进行验证。@Ruofan
5. 对 OCR 数据做 caption 处理。@Xingqian
