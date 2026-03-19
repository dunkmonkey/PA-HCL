# SinCNeXt-BM 首轮训练超参数推荐表

**文档日期**: 2026-01-25  
**版本**: SinCNeXt-BM v1.0  
**配置系统**: OmegaConf (YAML)

---

## 拓扑速览

| 训练线 | 目标 | 建议周期 | 学习率 | 预热 | Batch Size |
|-------|------|--------|-------|------|-----------|
| **预训练** | 自监督特征学习 | 100-150 | 2e-3 | 10-15 | 128-256 |
| **微调** (线性评估) | 冻结编码器 + 训练分类头 | 50-100 | 1e-4 (enc) / 1e-3 (cls) | 5-10 | 64-128 |
| **微调** (全参数) | 端到端微调 | 100-150 | 1e-4-5e-4 (enc) / 5e-4-1e-3 (cls) | 10-15 | 32-64 |
| **监督基线** | 从零开始的监督学习 | 50-100 | 1e-3-5e-4 | 5-10 | 64-128 |

---

## 1️⃣ 预训练阶段（PA-HCL 自监督）

### 1.1 基础配置

```yaml
experiment:
  name: "pretrain_sincnet_v1"
  seed: 42

data:
  sample_rate: 5000
  target_length: 4000
  num_substructures: 4
  augmentation_strength: "medium"  # light, medium, strong

model:
  encoder_type: "sincnet_eca_mamba"
  use_moco: true
  moco_queue_size: 65536
  moco_momentum: 0.999
  temperature: 0.2
  
  # SinCNeXt-BM 核心参数（固定推荐）
  sinc_out_channels: 64
  sinc_kernel_size: 251
  sinc_stride: 1
  sinc_min_low_hz: 20.0
  sinc_min_band_hz: 20.0
  sinc_max_high_hz: 500.0
  
  local_dim: 128
  convnext_kernel_size: 7
  convnext_expansion: 4
  mamba_d_model: 128
  mamba_n_layers: 6
  mamba_d_state: 16
  mamba_d_conv: 4
  mamba_expansion_factor: 2
  
  use_bidirectional: true
  bidirectional_fusion: "add"
  drop_path_rate: 0.1
  
  use_groupnorm: true
  num_groups: 8
  cycle_output_dim: 256
  pool_type: "asp"
  
  # 投影头
  cycle_proj_hidden: 256
  cycle_proj_output: 128
  cycle_proj_layers: 2
  sub_proj_hidden: 128
  sub_proj_output: 64
  sub_proj_layers: 2

training:
  num_epochs: 100  # [100, 150] 推荐范围
  batch_size: 256  # GPU 显存充足时推荐 256，否则 128
  learning_rate: 2e-3
  weight_decay: 1e-4
  
  warmup_epochs: 10
  warmup_strategy: "linear"
  min_lr: 1e-6
  scheduler: "cosine"
  
  optimizer: "adamw"
  betas: [0.9, 0.999]
  eps: 1e-8
  
  grad_clip_norm: 1.0
  use_amp: true  # 混合精度加速
  
  early_stopping_patience: 30
  save_interval: 5
  log_interval: 50
  
  num_workers: 8
  pin_memory: true
  prefetch_factor: 2

loss:
  cycle_loss_weight: 1.0
  sub_loss_weight: 0.5
  use_moco_loss: true

tracking:
  use_wandb: true
  wandb_project: "PA-HCL-PretrainSinCNeXt"
  save_model_summary: true
```

### 1.2 超参数调优指南

| 参数 | 初始值 | 调优范围 | 影响 | 说明 |
|-----|-------|--------|------|------|
| `num_epochs` | 100 | 80-200 | 收敛速度 | 取决于数据量；PhysioNet 2016 (1000 sig) 100 周期通常足够 |
| `batch_size` | 256 | 128-512 | 收敛稳定性 | GPU 显存 >= 32GB 推荐 256；否则 128 |
| `learning_rate` | 2e-3 | 1e-3 - 5e-3 | 收敛速度 | 较大 LR(5e-3) 训练费但不稳定；较小 LR(1e-3) 更稳定但慢 |
| `weight_decay` | 1e-4 | 1e-5 - 1e-3 | 正则化强度 | 避免过拟合；通常 1e-4 是最佳选择 |
| `drop_path_rate` | 0.1 | 0.05 - 0.3 | 正则化 | 更高的 drop_path 防止过拟合但可能影响可学习性 |
| `warmup_epochs` | 10 | 5-20 | LR 增长 | PhysioNet 规模 10 周期足够 |
| `cycle_loss_weight` | 1.0 | 0.5-2.0 | 周期 vs 子结构损失 | 默认 1:0.5 比例对心音数据友好 |
| `sub_loss_weight` | 0.5 | 0.3-1.0 | 子结构学习 | 增加会强化子结构判别性 |
| `temperature` | 0.2 | 0.1-0.5 | 对比损失锐度 | 较小 T(0.1) 更锐；较大 T(0.5) 更软 |

### 1.3 预期性能

- **周期损失 (Cycle Loss)**: 从 ~10 快速降低到 2-3，然后缓慢收敛到 1-1.5
- **子结构损失 (Sub Loss)**: 从 ~8 降低到 1-2（变化较平缓）
- **总损失**: 最终稳定在 2.5-3.5
- **训练时间**: V100/A100 单卡 ~6-12 小时（100 周期）
- **模型大小**: ~2.1M 参数

### 1.4 常见问题

**Q: 损失不下降？**  
→ 检查学习率是否过小或 warmup 周期过长；验证数据正确加载

**Q: 显存溢出？**  
→ 降低 batch_size (256→128) 或 cycle_proj_output (128→64)

**Q: 过拟合明显？**  
→ 增加 drop_path_rate (0.1→0.2)，增加 weight_decay (1e-4→5e-4)，或启用更强增强

---

## 2️⃣ 微调阶段 - 线性评估（Linear Probe）

*冻结预训练编码器，仅训练分类头。用于评估预训练质量。*

### 2.1 配置模板

```yaml
experiment:
  name: "finetune_linear_physionet2016"
  task: "physionet2016"
  finetune_mode: "linear_probe"  # 仅训练分类头

data:
  sample_rate: 5000
  target_length: 4000
  num_substructures: 4

model:
  encoder_type: "sincnet_eca_mamba"
  pretrained_checkpoint: "path/to/pretraining/ckpt.pt"
  freeze_encoder: true  # 关键：冻结编码器
  
  # 使用预训练的所有参数
  sinc_out_channels: 64
  local_dim: 128
  mamba_d_model: 128
  mamba_n_layers: 6
  cycle_output_dim: 256
  
  # 分类头
  classifier:
    hidden_dim: 128
    num_layers: 1
    dropout: 0.3
    use_bn: true

training:
  num_epochs: 50  # [30-100] 线性评估通常快速收敛
  batch_size: 128
  learning_rate_encoder: 0.0  # 固定（不更新）
  learning_rate_classifier: 1e-3
  weight_decay: 1e-4
  
  warmup_epochs: 5
  scheduler: "cosine"
  min_lr: 1e-6
  
  optimizer: "adamw"
  grad_clip_norm: 1.0
  use_amp: true
  
  early_stopping_patience: 15
  save_best_only: true

evaluation:
  primary_metric: "auroc"
  save_best_checkpoint: true
```

### 2.2 期望结果

- **AUROC**: 0.75-0.85 (PhysioNet 2016)
- **收敛周期**: 30-50 周期快速收敛
- **训练时间**: ~1-2 小时

---

## 3️⃣ 微调阶段 - 全参数微调（Full FineTune）

*端到端微调编码器和分类头。通常获得最佳性能。*

### 3.1 配置模板

```yaml
experiment:
  name: "finetune_full_physionet2016"
  task: "physionet2016"
  finetune_mode: "full"

data:
  sample_rate: 5000
  target_length: 4000

model:
  encoder_type: "sincnet_eca_mamba"
  pretrained_checkpoint: "path/to/pretraining/ckpt.pt"
  freeze_encoder: false
  
  # 编码器参数同上
  sinc_out_channels: 64
  local_dim: 128
  mamba_d_model: 128
  mamba_n_layers: 6
  cycle_output_dim: 256
  
  # 分类头
  classifier:
    hidden_dim: 128
    num_layers: 1
    dropout: 0.5  # 更高 dropout 防止过拟合
    use_bn: true

training:
  num_epochs: 100
  batch_size: 64  # 全参数微调时减小 batch_size
  
  # 分层学习率（编码器较小，分类头较大）
  learning_rate_encoder: 1e-4
  learning_rate_classifier: 5e-4
  weight_decay: 1e-4
  
  # 学习率缩放
  encoder_wd_multiplier: 0.1  # 编码器权重衰减减弱
  
  warmup_epochs: 10
  scheduler: "cosine"
  min_lr: 1e-7
  
  optimizer: "adamw"
  betas: [0.9, 0.999]
  grad_clip_norm: 1.0
  use_amp: true
  
  early_stopping_patience: 20
  save_best_only: true
  
  # 自适应优化
  use_grad_accum: false  # 不需要，batch_size 已调整
  num_workers: 4

loss:
  loss_type: "ce"  # 交叉熵
  use_class_weights: true  # 处理类别不平衡
  label_smoothing: 0.1

evaluation:
  primary_metric: "auroc"
  metrics: [accuracy, f1, auroc, auprc, sensitivity, specificity]
  save_confusion_matrix: true
```

### 3.2 分层学习率配置

```python
# 在 finetune.py 中伪代码
encoder_params = [
    {'params': encoder.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5},
]
classifier_params = [
    {'params': classifier.parameters(), 'lr': 5e-4, 'weight_decay': 1e-4},
]
optimizer = torch.optim.AdamW(encoder_params + classifier_params)
```

### 3.3 期望结果

- **AUROC**: 0.80-0.90 (PhysioNet 2016，相比线性评估 +3-10%)
- **收敛周期**: 70-100 周期
- **训练时间**: ~4-8 小时

---

## 4️⃣ 监督学习基线（从零开始）

*直接从零训练 SinCNeXt-BM + 分类头，不使用预训练。用作消融实验。*

### 4.1 配置模板

```yaml
experiment:
  name: "supervised_baseline_physionet2016"
  task: "physionet2016"
  description: "从零开始的监督学习基线"

data:
  sample_rate: 5000
  target_length: 4000
  num_substructures: 4

model:
  encoder_type: "sincnet_eca_mamba"
  pretrained_checkpoint: null  # 无预训练
  
  # 编码器配置（与预训练相同）
  sinc_out_channels: 64
  sinc_kernel_size: 251
  local_dim: 128
  convnext_kernel_size: 7
  convnext_expansion: 4
  mamba_d_model: 128
  mamba_n_layers: 6
  drop_path_rate: 0.1
  use_bidirectional: true
  cycle_output_dim: 256
  pool_type: "asp"
  
  # 分类器（轻量化防止过拟合）
  classifier:
    hidden_dim: 128
    num_layers: 1
    dropout: 0.5
    use_bn: true
    use_ln: false

training:
  num_epochs: 100
  batch_size: 64
  learning_rate: 5e-4
  weight_decay: 1e-4
  
  warmup_epochs: 10
  scheduler: "cosine"
  min_lr: 1e-7
  
  optimizer: "adamw"
  grad_clip_norm: 1.0
  use_amp: true
  
  # 正则化（从零训练需要更强正则化）
  label_smoothing: 0.1
  use_class_weights: true
  use_balanced_sampling: true  # 类别平衡采样
  
  # 早停
  early_stopping_patience: 25
  save_best_only: true
  
  num_workers: 4

loss:
  loss_type: "ce"
  focal_gamma: 1.0  # 关闭 focal loss（标准交叉熵）

evaluation:
  primary_metric: "auroc"
  metrics: [accuracy, f1, auroc, auprc]
```

### 4.2 期望结果

- **AUROC**: 0.70-0.80 (PhysioNet 2016)
- **相对性能**: 相比微调 -10-20%（正常现象）
- **训练时间**: ~6-10 小时
- **参数量**: ~2.1M (与预训练编码器相同)

### 4.3 vs 微调对比

| 指标 | 监督基线 | 线性评估 | 全参数微调 |
|-----|--------|--------|---------|
| AUROC | 0.70-0.80 | 0.75-0.85 | 0.80-0.90 |
| 收敛周期 | 70-100 | 20-50 | 80-120 |
| 训练时间 | 6-10h | 1-2h | 4-8h |
| 过拟合风险 | ⚠️ 中等 | ✅ 低 | ⚠️ 中等 |
| 数据需求 | 大 | 小 | 中等 |

---

## 📊 超参数矩阵速查

### 学习率推荐表

| 场景 | 编码器 LR | 分类头 LR | 原因 |
|-----|---------|---------|------|
| 线性评估 | 0.0 (冻结) | 1e-3 | 编码器已优化，只训练新头 |
| 全参数微调 | 1e-4 | 5e-4 | 编码器保守更新，分类头激进 |
| 监督基线 | 5e-4 | 5e-4 | 端到端学习，统一学习率 |
| 大 batch (1k+) | +2-5× | +2-5× | 线性缩放规则 |
| 小 batch (<32) | -2-5× | -2-5× | 降低学习率避免不稳定 |

### Batch Size 和学习率的关系

```
LR_new = LR_base * sqrt(batch_size_new / batch_size_base)

例：
  基线: batch_size=128, lr=1e-3
  调大到: batch_size=256, lr = 1e-3 * sqrt(256/128) = 1.4e-3
  调小到: batch_size=64, lr = 1e-3 * sqrt(64/128) = 0.7e-3
```

### 正则化强度对应

| 过拟合现象 | 调整策略 | 优先级 |
|---------|--------|------|
| 验证集 AUROC 下降 | ↑ dropout (0.3→0.5) | ⭐⭐⭐ |
| 训练/验证差距 >15% | ↑ weight_decay (1e-4→5e-4) | ⭐⭐⭐ |
| 模型振荡 | ↑ drop_path_rate (0.1→0.2) | ⭐⭐ |
| 早期过拟合 | ↑ label_smoothing (0→0.1) | ⭐⭐ |
| 梯度爆炸 | ↓ grad_clip_norm(1.0→0.5) | ⭐⭐⭐ |

---

## 🔍 诊断表

### 训练异常及解决方案

| 症状 | 可能原因 | 快速修复 |
|------|--------|--------|
| 损失 NaN | 学习率过高 / 梯度爆炸 | ↓ LR (2e-3→1e-3) 或 ↑ grad_clip (1→0.5) |
| 损失平坦不动 | 学习率过低 / warmup不足 | ↑ LR (1e-3→2e-3) 或 ↑ warmup_epochs (5→15) |
| 显存溢出 | batch_size 或模型过大 | ↓ batch_size (256→128) 或 gradient_accumulation |
| 过拟合严重 (train auroc 0.95, val 0.70) | 正则化不足 | ↑ dropout, weight_decay, drop_path_rate 三管齐下 |
| 收敛慢 | 学习率偏保守 / 数据难度高 | ↑ LR + ↓ warmup_epochs，或检查数据质量 |

---

## 🛠️ 快速开始命令

```bash
# 1. 预训练
python scripts/pretrain.py \
  --config configs/pretrain.yaml \
  --seed 42 \
  --batch-size 256 \
  --num-epochs 100 \
  --learning-rate 2e-3

# 2. 线性评估
python scripts/finetune.py \
  --config configs/finetune.yaml \
  --task physionet2016 \
  --pretrained-checkpoint checkpoints/pretrain_sincnet_ep100.pt \
  --freeze-encoder \
  --num-epochs 50 \
  --learning-rate 1e-3

# 3. 全参数微调
python scripts/finetune.py \
  --config configs/finetune.yaml \
  --task physionet2016 \
  --pretrained-checkpoint checkpoints/pretrain_sincnet_ep100.pt \
  --num-epochs 100 \
  --learning-rate-encoder 1e-4 \
  --learning-rate-classifier 5e-4

# 4. 监督基线
python scripts/train_supervised_baseline.py \
  --task physionet2016 \
  --encoder-type sincnet_eca_mamba \
  --num-epochs 100 \
  --learning-rate 5e-4
```

---

## 📋 参数文件清单

- **预训练**: `configs/pretrain.yaml` ✅
- **微调**: `configs/finetune.yaml` ✅
- **监督基线**: `configs/supervised_baseline.yaml` ✅
- **任务配置**: `configs/tasks/*.yaml` (CircOR, PhysioNet2016, PASCAL, ...)

---

## 📝 记录&验证清单

首次运行时，建议逐项验证：

- [ ] 预训练: `configs/pretrain.yaml` 加载无误
- [ ] 预训练: 第 1 个 epoch 损失正常下降（10 → 5-6）
- [ ] 预训练: GPU 显存稳定 (<16GB for 256 batch)
- [ ] 微调: 预训练权重成功加载
- [ ] 微调: 线性评估在 20-30 周期收敛到 >0.70 AUROC
- [ ] 微调: 全参数微调到 >0.80 AUROC
- [ ] 监督基线: 从零训练到 0.70-0.80 AUROC
- [ ] 所有配置: 存档最终超参数到 `exp_log.yaml`

---

## 参考资源

1. **架构论文**: SinCNeXt-BM 基于以下工作：
   - SincNet (Ravanelli et al., 2018)
   - ConvNeXt (Liu et al., 2022)
   - Mamba (Gu & Dao, 2023)

2. **学习率缩放**: [Goyal et al., 2017 - Accurate Large Batch SGD]

3. **预训练最佳实践**: [Lightly blog on ViT pretraining]

---

**更新日期**: 2026-01-25  
**编写者**: PA-HCL Team
