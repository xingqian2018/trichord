# Cosmos3.5 Text2Image LoRA UQ-Midtrain 计划

日期：2026-11-04　状态：草稿（需求清单，持续补充）

## 一、需求清单

### 1. 测试端

- [ ] 接入（ingest）新的 Qwen-Image-Bench（EMNLP 2026，arXiv 2605.28091），作为我们额外的 benchmark
  - 5 个维度：visual quality、aesthetics、prompt alignment、real-world fidelity、creative generation
  - 准备符合 cosmos 需求的提示词
  - 把所有的 baseline 重新跑一遍

### 2. 数据端

- [ ] 跟踪整理 cosmos 已有的 image 数据
  - super resolution、editing、multi-ref 等，只要和数据相关的，都是 UQ 数据候选
- [ ] 追踪新的 Getty / SSTK 数据获取，并做测试
  - 涉及外部协议，未必能赶上计划
- [ ] OCR 端：继续生成新的 poster / UI / chart 数据
- [ ] 设计新的 filter，更好地为 UQ 数据服务

### 3. 模型端

- [ ] LoRA 实际参数量的选择，要和 Video、Action 同步进行

### 4. 训练端

- [ ] SFT（数据 mixing）
- [ ] RL（最后提升）

### 5. Prompt Upsampling 端

- [ ] 找到最好的 upsampler 算法
- [ ] 长 prompt 做一定的类结构化，与我现在的做法一致
- [ ] 新的 captioner 要和 prompt upsampler 对齐

### 6. 榜单跟踪

- [ ] 设定提供 Arena 服务的端口
  - 可能提前 submit 一个版本，再根据实际测试内容调整模型

### 7. 研究端

- [ ] 尝试 hot 更换 VAE，看能否改善 3.5 的生图质量，尤其是小文字
- [ ] ablate 其他可能的参数
