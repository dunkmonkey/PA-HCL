#!/usr/bin/env python
"""
PA-HCL 监督学习基线训练脚本。

此脚本从零开始训练分类模型（不使用预训练权重），
用于验证编码器架构的有效性和作为对比基线。

特点:
    - 完全监督学习（不使用预训练）
    - 支持多种编码器架构（CNN-only, CNN-Transformer, CNN-Mamba）
    - 与预训练+微调结果对比，评估预训练的增益

用法:
    # 基础用法：在 PhysioNet 2016 上训练
    python scripts/train_supervised_baseline.py --task physionet2016
    
    # 测试不同编码器架构
    python scripts/train_supervised_baseline.py --task physionet2016 --encoder-type cnn_only
    python scripts/train_supervised_baseline.py --task physionet2016 --encoder-type cnn_transformer
    python scripts/train_supervised_baseline.py --task physionet2016 --encoder-type cnn_mamba
    
    # 使用自定义配置
    python scripts/train_supervised_baseline.py \\
        --config configs/supervised_baseline.yaml \\
        --task physionet2016 \\
        --experiment-name my_baseline
    
    # 启用 WandB 跟踪
    python scripts/train_supervised_baseline.py \\
        --task physionet2016 \\
        --wandb \\
        --wandb-project PA-HCL-Baselines

作者: PA-HCL 团队
"""

import argparse
import json
import sys
from pathlib import Path

# 将项目根目录添加到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config, print_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logger
from src.data.dataset import HeartSoundDataset
from src.models.encoder import Encoder
from src.models.heads import ClassificationHead


# 可用任务列表
AVAILABLE_TASKS = [
    "circor_murmur",
    "circor_outcome", 
    "physionet2016",
    "pascal"
]


class SupervisedModel(nn.Module):
    """
    监督学习分类模型。
    
    从零初始化的编码器 + 分类头，
    用于直接监督学习训练。
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        encoder_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        """
        参数:
            encoder: 编码器模块（从零初始化）
            num_classes: 分类类别数
            encoder_dim: 编码器输出维度
            hidden_dim: 分类头隐藏层维度
            dropout: Dropout 率
        """
        super().__init__()
        
        self.encoder = encoder
        self.classifier = ClassificationHead(
            input_dim=encoder_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            x: 输入信号 [B, 1, L] 或 [B, L]
            
        返回:
            logits [B, num_classes]
        """
        # 提取特征
        features = self.encoder(x)
        
        # 分类
        logits = self.classifier(features)
        
        return logits


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="PA-HCL 监督学习基线训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
可用任务:
    {', '.join(AVAILABLE_TASKS)}

示例:
    # 在 PhysioNet 2016 上训练 CNN-Mamba 基线
    python scripts/train_supervised_baseline.py --task physionet2016
    
    # 对比不同架构
    python scripts/train_supervised_baseline.py --task physionet2016 --encoder-type cnn_only
    python scripts/train_supervised_baseline.py --task physionet2016 --encoder-type cnn_transformer
    python scripts/train_supervised_baseline.py --task physionet2016 --encoder-type cnn_mamba
    
    # 启用 WandB 跟踪
    python scripts/train_supervised_baseline.py --task physionet2016 --wandb
        """
    )
    
    # 任务配置
    task_group = parser.add_argument_group("任务配置")
    task_group.add_argument(
        "--task",
        type=str,
        choices=AVAILABLE_TASKS,
        required=True,
        help="任务名称"
    )
    task_group.add_argument(
        "--config",
        type=str,
        default="configs/supervised_baseline.yaml",
        help="基础配置文件路径"
    )
    
    # 模型配置
    model_group = parser.add_argument_group("模型配置")
    model_group.add_argument(
        "--encoder-type",
        type=str,
        choices=["cnn_only", "cnn_transformer", "cnn_mamba"],
        default=None,
        help="编码器类型（覆盖配置文件）"
    )
    
    # 输出
    output_group = parser.add_argument_group("输出设置")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="outputs/supervised_baseline",
        help="输出根目录"
    )
    output_group.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="实验名称（默认使用任务名称）"
    )
    
    # 训练参数覆盖
    override_group = parser.add_argument_group("参数覆盖")
    override_group.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="训练轮数"
    )
    override_group.add_argument(
        "--lr",
        type=float,
        default=None,
        help="学习率"
    )
    override_group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批大小"
    )
    override_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    # 数据
    data_group = parser.add_argument_group("数据设置")
    data_group.add_argument(
        "--data-dir",
        type=str,
        default="/root/autodl-tmp/data/downstream",
        help="下游数据根目录"
    )
    data_group.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="数据加载工作进程数"
    )
    
    # 实验监控
    tracking_group = parser.add_argument_group("实验监控")
    tracking_group.add_argument(
        "--wandb",
        action="store_true",
        help="启用 WandB 实验跟踪"
    )
    tracking_group.add_argument(
        "--wandb-project",
        type=str,
        default="PA-HCL-Supervised",
        help="WandB 项目名称"
    )
    tracking_group.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB 实体/用户名"
    )
    tracking_group.add_argument(
        "--tensorboard",
        action="store_true",
        default=True,
        help="启用 TensorBoard 跟踪"
    )
    
    return parser.parse_args()


def load_task_config(args):
    """加载任务配置。"""
    from types import SimpleNamespace
    
    # 加载基础配置
    config = load_config(args.config)
    
    # 加载任务特定配置
    task_config_path = project_root / "configs" / "tasks" / f"{args.task}.yaml"
    if task_config_path.exists():
        task_config = load_config(str(task_config_path))
        
        # 合并配置（任务配置优先）
        if hasattr(task_config, 'classification'):
            if not hasattr(config, 'classification'):
                config.classification = SimpleNamespace()
            for key, value in vars(task_config.classification).items():
                setattr(config.classification, key, value)
        
        if hasattr(task_config, 'data'):
            for key, value in vars(task_config.data).items():
                setattr(config.data, key, value)
        
        if hasattr(task_config, 'task'):
            config.task = task_config.task
    
    return config


def create_dataloaders(config, args, logger):
    """创建训练和验证数据加载器。"""
    # 构建数据路径
    task_name = config.task.name if hasattr(config.task, 'name') else args.task
    data_dir = Path(args.data_dir) / task_name
    
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    
    if not train_csv.exists():
        raise FileNotFoundError(
            f"训练数据文件不存在: {train_csv}\n"
            f"请先运行数据准备脚本:\n"
            f"  python scripts/data_preparation/prepare_downstream_tasks.py --dataset {task_name}"
        )
    
    logger.info(f"加载训练数据: {train_csv}")
    logger.info(f"加载验证数据: {val_csv}")
    
    # 创建数据集
    train_dataset = HeartSoundDataset(
        csv_file=str(train_csv),
        sample_rate=config.data.sample_rate,
        target_length=config.data.target_length,
        augment=True,  # 训练时数据增强
    )
    
    val_dataset = HeartSoundDataset(
        csv_file=str(val_csv),
        sample_rate=config.data.sample_rate,
        target_length=config.data.target_length,
        augment=False,  # 验证时不增强
    )
    
    logger.info(f"训练样本数: {len(train_dataset)}")
    logger.info(f"验证样本数: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


def create_model(config, num_classes, logger):
    """创建监督学习模型（从零初始化）。"""
    logger.info("="*80)
    logger.info("创建监督学习模型（从零初始化）")
    logger.info("="*80)
    
    # 创建编码器（从零初始化）
    logger.info(f"编码器类型: {config.model.encoder_type}")
    
    encoder = Encoder(
        input_channels=1,
        encoder_type=config.model.encoder_type,
        cnn_channels=config.model.cnn_channels,
        cnn_kernel_sizes=config.model.cnn_kernel_sizes,
        cnn_strides=config.model.cnn_strides,
        mamba_d_model=getattr(config.model, 'mamba_d_model', 256),
        mamba_n_layers=getattr(config.model, 'mamba_n_layers', 4),
        mamba_d_state=getattr(config.model, 'mamba_d_state', 16),
        mamba_d_conv=getattr(config.model, 'mamba_d_conv', 4),
        mamba_expand_factor=getattr(config.model, 'mamba_expand_factor', 2),
        transformer_n_layers=getattr(config.model, 'transformer_n_layers', 4),
        transformer_n_heads=getattr(config.model, 'transformer_n_heads', 8),
        transformer_d_ff=getattr(config.model, 'transformer_d_ff', 1024),
        transformer_dropout=getattr(config.model, 'transformer_dropout', 0.1),
    )
    
    # 获取编码器输出维度
    if config.model.encoder_type == "cnn_only":
        encoder_dim = config.model.cnn_channels[-1]
    else:
        encoder_dim = config.model.mamba_d_model if hasattr(config.model, 'mamba_d_model') else 256
    
    # 创建完整模型
    model = SupervisedModel(
        encoder=encoder,
        num_classes=num_classes,
        encoder_dim=encoder_dim,
        hidden_dim=config.classifier.hidden_dim,
        dropout=config.classifier.dropout,
    )
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")
    logger.info(f"编码器输出维度: {encoder_dim}")
    logger.info(f"分类类别数: {num_classes}")
    
    return model


def main():
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = load_task_config(args)
    
    # 应用命令行覆盖
    if args.encoder_type is not None:
        config.model.encoder_type = args.encoder_type
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    
    # 设置实验名称
    if args.experiment_name is None:
        encoder_suffix = config.model.encoder_type.replace("_", "-")
        args.experiment_name = f"{args.task}_{encoder_suffix}"
    
    # 创建输出目录
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(
        name="supervised_baseline",
        log_file=output_dir / "train.log"
    )
    
    logger.info("="*80)
    logger.info("PA-HCL 监督学习基线训练")
    logger.info("="*80)
    logger.info(f"实验名称: {args.experiment_name}")
    logger.info(f"任务: {args.task}")
    logger.info(f"编码器类型: {config.model.encoder_type}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"随机种子: {args.seed}")
    
    # 打印配置
    print_config(config, logger)
    
    # 保存配置
    config_save_path = output_dir / "config.yaml"
    import yaml
    with open(config_save_path, 'w') as f:
        yaml.dump(vars(config), f, default_flow_style=False)
    logger.info(f"配置已保存到: {config_save_path}")
    
    # 创建数据加载器
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(
        config, args, logger
    )
    
    # 获取类别数
    if hasattr(config, 'classification'):
        num_classes = config.classification.num_classes
    elif hasattr(config, 'downstream'):
        num_classes = config.downstream.num_classes
    else:
        raise ValueError("配置中未找到 num_classes")
    
    # 创建模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config, num_classes, logger)
    model = model.to(device)
    
    # 导入训练器
    from src.trainers.downstream_trainer import DownstreamTrainer
    
    # 创建训练器
    trainer = DownstreamTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
        logger=logger,
        device=device,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        experiment_name=args.experiment_name,
        use_tensorboard=args.tensorboard,
    )
    
    # 开始训练
    logger.info("="*80)
    logger.info("开始训练")
    logger.info("="*80)
    
    try:
        best_metrics = trainer.train()
        
        logger.info("="*80)
        logger.info("训练完成")
        logger.info("="*80)
        logger.info("最佳验证集指标:")
        for metric_name, metric_value in best_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # 保存最终结果
        results = {
            "task": args.task,
            "encoder_type": config.model.encoder_type,
            "experiment_name": args.experiment_name,
            "seed": args.seed,
            "best_metrics": best_metrics,
            "config": vars(config),
        }
        
        results_path = output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"结果已保存到: {results_path}")
        
    except KeyboardInterrupt:
        logger.warning("训练被用户中断")
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}", exc_info=True)
        raise
    
    logger.info("程序结束")


if __name__ == "__main__":
    main()
