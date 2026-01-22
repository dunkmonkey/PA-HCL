#!/usr/bin/env python
"""
PA-HCL 预训练脚本。

此脚本运行使用分层对比学习的自监督预训练。

用法:
    # 单 GPU 训练
    python scripts/pretrain.py --config configs/pretrain.yaml
    
    # 多 GPU 训练 (DDP)
    torchrun --nproc_per_node=4 scripts/pretrain.py --config configs/pretrain.yaml --distributed
    
    # 从检查点恢复
    python scripts/pretrain.py --config configs/pretrain.yaml --resume outputs/pahcl_pretrain/epoch_50.pt

作者: PA-HCL 团队
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

# 将项目根目录添加到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.config import load_config, print_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logger


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="PA-HCL 预训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 配置
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pretrain.yaml",
        help="配置文件路径"
    )
    
    # 训练设置
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="检查点和日志的输出目录"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="pahcl_pretrain",
        help="实验名称"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="要恢复的检查点路径"
    )
    
    # 分布式训练
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="使用分布式数据并行训练"
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="分布式训练的本地排名 (由 torchrun 设置)"
    )
    
    # 日志记录
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="使用 Weights & Biases 进行日志记录"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="PA-HCL",
        help="W&B 项目名称"
    )
    
    # 覆盖参数
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="覆盖批大小"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="覆盖轮数"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="覆盖学习率"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    return parser.parse_args()


def main():
    """主入口点。"""
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 应用命令行覆盖
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    
    # 处理分布式训练
    if args.distributed:
        # 从环境获取本地排名 (由 torchrun 设置)
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        if local_rank == -1:
            raise ValueError(
                "分布式训练需要 LOCAL_RANK。"
                "请使用 'torchrun --nproc_per_node=N scripts/pretrain.py'"
            )
        
        # 初始化进程组
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        
        is_main = local_rank == 0
    else:
        local_rank = -1
        world_size = 1
        is_main = True
    
    # 设置日志
    logger = setup_logger(
        name="pretrain",
        log_file=Path(args.output_dir) / args.experiment_name / "train.log"
        if is_main else None
    )
    
    if is_main:
        logger.info("=" * 60)
        logger.info("PA-HCL 预训练")
        logger.info("=" * 60)
        print_config(config)
    
    # 设置种子
    set_seed(args.seed)
    
    # 构建数据集
    from src.data.dataset import PCGPretrainDataset, create_dataloaders, create_subject_wise_split
    from src.data.transforms import get_pretrain_transforms
    
    transforms = get_pretrain_transforms(
        sample_rate=config.data.sample_rate
    )
    
    # 加载完整数据集
    full_dataset = PCGPretrainDataset(
        data_dir=config.data.processed_dir,
        transform=transforms,
        num_substructures=config.data.num_substructures,
    )
    
    # 创建受试者级划分 (9:1 = 训练:验证)
    train_indices, val_indices, _ = create_subject_wise_split(
        full_dataset,
        train_ratio=0.9,
        val_ratio=0.1,
        test_ratio=0.0,
        seed=args.seed
    )
    
    if is_main:
        logger.info(f"Total samples: {len(full_dataset)}")
        logger.info(f"Training samples: {len(train_indices)} ({len(train_indices)/len(full_dataset)*100:.1f}%)")
        logger.info(f"Validation samples: {len(val_indices)} ({len(val_indices)/len(full_dataset)*100:.1f}%)")
        logger.info("Using subject-wise split to prevent data leakage")
    
    # 创建数据加载器（使用受试者级划分）
    train_loader, val_loader, _ = create_dataloaders(
        full_dataset,
        train_indices=train_indices,
        val_indices=val_indices,
        batch_size=config.training.batch_size // world_size,
        num_workers=config.training.num_workers,
        distributed=args.distributed,
    )
    
    # 构建模型
    from src.models.pahcl import build_pahcl_model
    
    model = build_pahcl_model(config)
    
    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # 保存配置文件
    if is_main:
        output_path = Path(args.output_dir) / args.experiment_name
        output_path.mkdir(parents=True, exist_ok=True)
        config_save_path = output_path / 'config.yaml'
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config.to_dict() if hasattr(config, 'to_dict') else dict(config), f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Configuration saved to {config_save_path}")
    
    # 创建训练器
    from src.trainers.pretrain_trainer import PretrainTrainer
    
    trainer = PretrainTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        num_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_epochs=config.training.warmup_epochs,
        min_lr=config.training.min_lr,
        temperature=config.loss.temperature,
        lambda_cycle=config.loss.lambda_cycle,
        lambda_sub=config.loss.lambda_sub,
        use_amp=config.training.use_amp,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        grad_clip_norm=config.training.grad_clip_norm,
        distributed=args.distributed,
        local_rank=local_rank,
        log_interval=config.training.log_interval,
        save_interval=config.training.save_interval,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        seed=args.seed,
    )
    
    # 如果指定则恢复
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 训练
    metrics = trainer.train()
    
    if is_main:
        logger.info("Training completed!")
        logger.info(f"Final train loss: {metrics['train_loss']:.4f}")
    
    # 清理分布式环境
    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
