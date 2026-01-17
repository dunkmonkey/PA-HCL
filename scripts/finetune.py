#!/usr/bin/env python
"""
PA-HCL 下游微调脚本。

此脚本在预训练的 PA-HCL 模型上运行下游分类任务。

用法:
    # 线性评估 (冻结编码器)
    python scripts/finetune.py --config configs/finetune.yaml \
        --pretrained outputs/pahcl_pretrain/best_model.pt \
        --linear-eval
    
    # 全量微调
    python scripts/finetune.py --config configs/finetune.yaml \
        --pretrained outputs/pahcl_pretrain/best_model.pt
    
    # 小样本学习
    python scripts/finetune.py --config configs/finetune.yaml \
        --pretrained outputs/pahcl_pretrain/best_model.pt \
        --few-shot --shot-ratio 0.1

作者: PA-HCL 团队
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录添加到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.config import load_config, print_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logger


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="PA-HCL 下游微调",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 配置
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune.yaml",
        help="配置文件路径"
    )
    
    # 预训练模型
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="预训练检查点路径"
    )
    
    # 训练模式
    parser.add_argument(
        "--linear-eval",
        action="store_true",
        help="使用线性评估 (冻结编码器)"
    )
    parser.add_argument(
        "--few-shot",
        action="store_true",
        help="启用小样本学习"
    )
    parser.add_argument(
        "--shot-ratio",
        type=float,
        default=0.1,
        help="用于小样本学习的训练数据比例"
    )
    
    # 输出
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="输出目录"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="downstream",
        help="实验名称"
    )
    
    # 覆盖参数
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="类别数量 (覆盖配置)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="轮数"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="学习率"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批大小"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 应用覆盖
    if args.num_classes is not None:
        config.downstream.num_classes = args.num_classes
    if args.epochs is not None:
        config.downstream.num_epochs = args.epochs
    if args.lr is not None:
        config.downstream.learning_rate = args.lr
    if args.batch_size is not None:
        config.downstream.batch_size = args.batch_size
    
    # 设置
    logger = setup_logger(
        name="finetune",
        log_file=Path(args.output_dir) / args.experiment_name / "finetune.log"
    )
    
    logger.info("=" * 60)
    logger.info("PA-HCL 下游微调")
    logger.info("=" * 60)
    
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # 加载预训练模型
    from src.trainers.downstream_trainer import load_pretrained_encoder, DownstreamModel, DownstreamTrainer
    
    logger.info(f"Loading pretrained model from: {args.pretrained}")
    
    pretrained_model = load_pretrained_encoder(
        args.pretrained,
        device,
        config
    )
    
    # 创建下游模型
    encoder = pretrained_model.encoder if hasattr(pretrained_model, 'encoder') else pretrained_model
    
    model = DownstreamModel(
        encoder=encoder,
        num_classes=config.downstream.num_classes,
        encoder_dim=getattr(pretrained_model, 'encoder_dim', 256),
        hidden_dim=None if args.linear_eval else config.downstream.hidden_dim,
        dropout=config.downstream.dropout,
        freeze_encoder=args.linear_eval
    )
    
    logger.info(f"Mode: {'Linear Evaluation' if args.linear_eval else 'Full Fine-tuning'}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 加载数据集
    from src.data.dataset import PCGDownstreamDataset, create_dataloaders
    
    train_dataset = PCGDownstreamDataset(
        data_dir=config.data.processed_dir,
        split="train",
        label_map=config.downstream.label_map
    )
    
    val_dataset = PCGDownstreamDataset(
        data_dir=config.data.processed_dir,
        split="val",
        label_map=config.downstream.label_map
    )
    
    test_dataset = PCGDownstreamDataset(
        data_dir=config.data.processed_dir,
        split="test",
        label_map=config.downstream.label_map
    )
    
    # 小样本采样
    if args.few_shot:
        from torch.utils.data import Subset
        import numpy as np
        
        n_samples = int(len(train_dataset) * args.shot_ratio)
        indices = np.random.choice(len(train_dataset), n_samples, replace=False)
        train_dataset = Subset(train_dataset, indices)
        logger.info(f"Few-shot training with {n_samples} samples ({args.shot_ratio*100:.1f}%)")
    
    # 创建数据加载器
    train_loader = create_dataloaders(
        train_dataset,
        batch_size=config.downstream.batch_size,
        num_workers=config.training.num_workers
    )
    
    val_loader = create_dataloaders(
        val_dataset,
        batch_size=config.downstream.batch_size,
        num_workers=config.training.num_workers,
        shuffle=False
    )
    
    test_loader = create_dataloaders(
        test_dataset,
        batch_size=config.downstream.batch_size,
        num_workers=config.training.num_workers,
        shuffle=False
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    # 创建训练器
    trainer = DownstreamTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        num_epochs=config.downstream.num_epochs,
        learning_rate=config.downstream.learning_rate,
        weight_decay=config.downstream.weight_decay,
        warmup_epochs=config.downstream.warmup_epochs,
        use_amp=config.training.use_amp,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        seed=args.seed,
    )
    
    # 训练
    metrics = trainer.train()
    
    logger.info("=" * 60)
    logger.info("Final Test Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
