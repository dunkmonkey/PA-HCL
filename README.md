# PA-HCL: 生理感知分层对比学习

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1](https://img.shields.io/badge/pytorch-2.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 面向心音生理结构的层次化自监督对比学习框架

这是一个自监督对比学习框架，利用心音固有的生理结构（心动周期和子结构）来学习可迁移的表示，用于下游临床任务。

## 🔬 概述

心音（PCG）信号具有区别于普通音频信号的独特生理结构。PA-HCL 通过以下方式利用这一领域知识：

1. **心动周期级对比学习**：学习对心率变化鲁棒的全局节律模式
2. **子结构级对比学习**：捕捉局部病理特征（杂音、异常音）
3. **CNN-Mamba 编码器**：结合局部特征提取与高效的长程依赖建模

## 📁 项目结构

```
PA-HCL/
├── configs/                    # 配置文件
│   ├── default.yaml           # 默认超参数
│   ├── pretrain.yaml          # 预训练配置
│   ├── finetune.yaml          # 微调配置
│   └── ablation.yaml          # 消融实验配置
├── data/
│   ├── raw/                   # 原始音频文件（按受试者组织）
│   └── processed/             # 处理后的周期片段
├── src/
│   ├── config.py              # 配置管理
│   ├── data/                  # 数据集与变换
│   │   ├── preprocessing.py   # 音频预处理
│   │   ├── transforms.py      # 数据增强
│   │   └── dataset.py         # PyTorch 数据集
│   ├── models/                # 编码器架构
│   │   ├── mamba.py           # Mamba (SSM) 实现
│   │   ├── encoder.py         # CNN-Mamba/Transformer 编码器
│   │   ├── heads.py           # 投影与分类头
│   │   └── pahcl.py           # 完整的 PA-HCL 模型
│   ├── losses/                # 对比损失函数
│   │   └── contrastive.py     # InfoNCE, 分层损失
│   ├── trainers/              # 训练逻辑
│   │   ├── pretrain_trainer.py   # 自监督预训练
│   │   └── downstream_trainer.py # 下游微调
│   └── utils/                 # 工具类
│       ├── logging.py         # 日志工具
│       ├── metrics.py         # 评估指标
│       └── seed.py            # 复现性
├── scripts/                   # 入口脚本
│   ├── preprocess.py          # 数据预处理
│   ├── pretrain.py            # 自监督预训练
│   ├── finetune.py            # 下游微调
│   ├── evaluate.py            # 模型评估
│   └── ablation.py            # 消融实验
├── tests/                     # 单元测试
├── checkpoints/               # 保存的模型
├── logs/                      # 训练日志
└── notebooks/                 # 分析笔记本
```

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/dunkmonkey/PA-HCL.git
cd PA-HCL

# 创建 conda 环境
conda create -n pahcl python=3.10
conda activate pahcl

# 安装 PyTorch (CUDA 11.8)
pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install -r requirements.txt

# 以开发模式安装包
pip install -e .
```

### 数据准备

按受试者结构组织原始数据：

```
/root/autodl-tmp/data/raw/
├── subject_0001/
│   ├── rec_01.wav
│   ├── rec_02.wav
│   └── ...
├── subject_0002/
│   └── ...
```

运行预处理：

```bash
python scripts/preprocess.py --config configs/default.yaml
```

### 自监督预训练

```bash
# 单 GPU
python scripts/pretrain.py --config configs/pretrain.yaml

# 多 GPU (DDP)
torchrun --nproc_per_node=4 scripts/pretrain.py \
    --config configs/pretrain.yaml \
    --distributed
```

### 下游微调

```bash
# 线性评估（冻结编码器）
python scripts/finetune.py \
    --config configs/finetune.yaml \
    --pretrained outputs/pahcl_pretrain/best_model.pt \
    --linear-eval

# 全量微调
python scripts/finetune.py \
    --config configs/finetune.yaml \
    --pretrained outputs/pahcl_pretrain/best_model.pt

# 小样本学习 (10% 数据)
python scripts/finetune.py \
    --config configs/finetune.yaml \
    --pretrained outputs/pahcl_pretrain/best_model.pt \
    --few-shot --shot-ratio 0.1
```

### 评估

```bash
python scripts/evaluate.py \
    --checkpoint outputs/downstream/best_model.pt \
    --data-dir /root/autodl-tmp/data/processed \
    --split test \
    --confusion-matrix
```

### 消融研究

```bash
# 运行所有消融实验
python scripts/ablation.py --config configs/ablation.yaml --all

# 运行特定消融
python scripts/ablation.py --config configs/ablation.yaml --encoder-ablation
python scripts/ablation.py --config configs/ablation.yaml --loss-ablation
```

## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试模块
pytest tests/test_mamba.py -v
pytest tests/test_encoder.py -v
pytest tests/test_heads.py -v
pytest tests/test_losses.py -v
pytest tests/test_pretrain.py -v
pytest tests/test_downstream.py -v
pytest tests/test_ablation.py -v

# 运行覆盖率检查
pytest tests/ -v --cov=src --cov-report=html
```

## ⚙️ 配置

所有超参数均通过 YAML 配置文件管理。关键设置：

| 参数 | 默认值 | 描述 |
|-----------|---------|-------------|
| `data.sample_rate` | 5000 | 音频采样率 (Hz) |
| `data.num_substructures` | 4 | 子结构数量 (K) |
| `model.encoder_type` | cnn_mamba | 编码器架构 |
| `model.mamba_d_model` | 256 | Mamba 隐藏层维度 |
| `model.mamba_n_layers` | 4 | Mamba 层数 |
| `loss.temperature` | 0.07 | InfoNCE 温度系数 |
| `loss.lambda_cycle` | 1.0 | 周期级损失权重 |
| `loss.lambda_sub` | 0.5 | 子结构损失权重 |
| `training.use_amp` | true | 启用混合精度 |

## 📊 实验

### 主要结果

| 方法 | 准确率 | AUROC | F1 |
|--------|----------|-------|-----|
| CNN (随机初始化) | - | - | - |
| wav2vec 2.0 | - | - | - |
| BYOL-A | - | - | - |
| **PA-HCL (本文)** | **-** | **-** | **-** |

### 消融研究

| 设置 | 周期级 | 子级 | AUROC |
|---------|-------------|-----------|-------|
| 基线 (仅 CNN) | ✗ | ✗ | - |
| 仅周期 | ✓ | ✗ | - |
| 仅子结构 | ✗ | ✓ | - |
| CNN-Transformer | ✓ | ✓ | - |
| **CNN-Mamba (本文)** | ✓ | ✓ | **-** |

## 🔧 关键组件

### CNN-Mamba 编码器
- **CNN 骨干**：用于局部特征提取的多尺度 1D 卷积
- **Mamba (SSM)**：用于 O(n) 长程依赖建模的状态空间模型

### 分层对比损失
- **周期级**：对齐增强后的心动周期表示
- **子结构级**：对齐对应的子结构（S1, 收缩期, S2, 舒张期）

```python
L_total = λ_cycle * L_cycle + λ_sub * L_sub
```

## 📝 引用

```bibtex
@article{pahcl2026,
  title={Physiology-Aware Hierarchical Contrastive Learning for Heart Sound Analysis},
  author={},
  journal={},
  year={2026}
}
```

## 📄 许可证

本项目采用 MIT 许可证。

## 🙏 致谢

- CirCor DigiScope 数据集
- PhysioNet 2016 挑战赛
