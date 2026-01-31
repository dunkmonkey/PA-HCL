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
from src.data.dataset import PCGDownstreamDataset
from src.models.encoder import build_encoder
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
        use_bn: bool = True,
        use_ln: bool = False,
        classifier_num_layers: int = 1,
    ):
        """
        参数:
            encoder: 编码器模块（从零初始化）
            num_classes: 分类类别数
            encoder_dim: 编码器输出维度
            hidden_dim: 分类头隐藏层维度
            dropout: Dropout 率
            use_bn: 是否使用 BatchNorm
            use_ln: 是否使用 LayerNorm
            classifier_num_layers: 分类头隐藏层数量
        """
        super().__init__()
        
        self.encoder = encoder
        self.freeze_encoder = False  # 监督学习不冻结编码器
        self.classifier = ClassificationHead(
            input_dim=encoder_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_bn=use_bn,
            use_ln=use_ln,
            num_layers=classifier_num_layers,
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
    override_group.add_argument(
        "--no-deterministic",
        action="store_true",
        help="禁用确定性模式以提升训练速度（可能影响结果可重现性）"
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
    from omegaconf import OmegaConf
    
    # 加载基础配置
    config = load_config(args.config)
    
    # 加载任务特定配置
    task_config_path = project_root / "configs" / "tasks" / f"{args.task}.yaml"
    if task_config_path.exists():
        task_config = load_config(str(task_config_path))
        
        # 使用 OmegaConf.merge 合并配置（任务配置优先）
        config = OmegaConf.merge(config, task_config)
    
    return config


def create_dataloaders(config, args, logger):
    """创建训练、验证和测试数据加载器。"""
    from torch.utils.data import WeightedRandomSampler
    
    # 构建数据路径
    task_name = config.task.name if hasattr(config.task, 'name') else args.task
    data_dir = Path(args.data_dir) / task_name
    
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"
    test_csv = data_dir / "test.csv"
    
    if not train_csv.exists():
        raise FileNotFoundError(
            f"训练数据文件不存在: {train_csv}\n"
            f"请先运行数据准备脚本:\n"
            f"  python scripts/data_preparation/prepare_downstream_tasks.py --dataset {task_name}"
        )
    
    logger.info(f"加载训练数据: {train_csv}")
    logger.info(f"加载验证数据: {val_csv}")
    logger.info(f"加载测试数据: {test_csv}")
    
    # 创建数据集（启用内存缓存）
    train_dataset = PCGDownstreamDataset(
        data_dir=data_dir,
        csv_path=train_csv,
        sample_rate=config.data.sample_rate,
        target_length=config.data.target_length,
        mode='train',
        cache_in_memory=True,
    )
    
    val_dataset = PCGDownstreamDataset(
        data_dir=data_dir,
        csv_path=val_csv,
        sample_rate=config.data.sample_rate,
        target_length=config.data.target_length,
        mode='val',
        cache_in_memory=True,
    )
    
    test_dataset = PCGDownstreamDataset(
        data_dir=data_dir,
        csv_path=test_csv,
        sample_rate=config.data.sample_rate,
        target_length=config.data.target_length,
        mode='val',  # 测试集使用 'val' 模式（不进行数据增强）
        cache_in_memory=True,
    )
    
    logger.info(f"训练样本数: {len(train_dataset)}")
    logger.info(f"验证样本数: {len(val_dataset)}")
    logger.info(f"测试样本数: {len(test_dataset)}")
    
    # 检查是否使用平衡采样
    use_balanced_sampling = getattr(config.training, 'use_balanced_sampling', False)
    
    # 创建数据加载器（优化配置）
    use_persistent = args.num_workers > 0
    
    # 训练数据加载器：根据配置选择是否使用 WeightedRandomSampler
    if use_balanced_sampling:
        logger.info("="*60)
        logger.info("启用类别平衡采样 (WeightedRandomSampler)")
        logger.info("="*60)
        
        # 获取每个样本的权重
        sample_weights = train_dataset.get_sample_weights()
        logger.info(f"  样本权重范围: [{sample_weights.min():.4f}, {sample_weights.max():.4f}]")
        
        # 打印类别分布和权重
        label_dist = train_dataset.get_label_distribution()
        class_weights = train_dataset.get_class_weights()
        for label, count in label_dist.items():
            label_idx = train_dataset.label_map[label]
            logger.info(f"  类别 '{label}' (idx={label_idx}): {count} 样本, 权重={class_weights[label_idx]:.4f}")
        
        # 创建加权采样器 (转换为 list 以满足类型要求)
        sampler = WeightedRandomSampler(
            weights=sample_weights.tolist(),
            num_samples=len(train_dataset),
            replacement=True  # 允许重复采样少数类
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            sampler=sampler,  # 使用加权采样器代替 shuffle
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4 if args.num_workers > 0 else None,
            persistent_workers=use_persistent,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4 if args.num_workers > 0 else None,
            persistent_workers=use_persistent,
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=use_persistent,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=use_persistent,
    )
    
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def create_model(config, num_classes, logger):
    """创建监督学习模型（从零初始化）。"""
    logger.info("="*80)
    logger.info("创建监督学习模型（从零初始化）")
    logger.info("="*80)
    
    # 创建编码器（从零初始化）
    logger.info(f"编码器类型: {config.model.encoder_type}")
    
    # 获取新增的模型参数
    attention_type = getattr(config.model, 'attention_type', 'none')
    drop_path_rate = getattr(config.model, 'drop_path_rate', 0.0)
    use_bidirectional = getattr(config.model, 'use_bidirectional', False)
    bidirectional_fusion = getattr(config.model, 'bidirectional_fusion', 'add')
    use_multiscale = getattr(config.model, 'use_multiscale', False)
    multiscale_kernel_sizes = getattr(config.model, 'multiscale_kernel_sizes', [3, 7, 15])
    
    logger.info(f"通道注意力类型: {attention_type}")
    logger.info(f"DropPath 率: {drop_path_rate}")
    logger.info(f"双向 Mamba: {use_bidirectional}")
    logger.info(f"多尺度卷积: {use_multiscale}")
    if use_multiscale:
        logger.info(f"  多尺度卷积核: {multiscale_kernel_sizes}")
    
    # 准备编码器参数
    encoder_kwargs = {
        'in_channels': 1,
    }
    
    # 根据编码器类型添加特定参数
    if config.model.encoder_type == "cnn_only":
        encoder_kwargs.update({
            'channels': config.model.cnn_channels,
            'kernel_sizes': config.model.cnn_kernel_sizes,
            'strides': config.model.cnn_strides,
            'drop_path_rate': drop_path_rate,
            'attention_type': attention_type,
            'use_multiscale': use_multiscale,
            'multiscale_kernel_sizes': multiscale_kernel_sizes,
        })
    elif config.model.encoder_type == "cnn_mamba":
        encoder_kwargs.update({
            'cnn_channels': config.model.cnn_channels,
            'cnn_kernel_sizes': config.model.cnn_kernel_sizes,
            'cnn_strides': config.model.cnn_strides,
            'mamba_d_model': getattr(config.model, 'mamba_d_model', 256),
            'mamba_n_layers': getattr(config.model, 'mamba_n_layers', 4),
            'mamba_d_state': getattr(config.model, 'mamba_d_state', 16),
            'mamba_expand': getattr(config.model, 'mamba_expand_factor', 2),
            # 新增参数
            'drop_path_rate': drop_path_rate,
            'attention_type': attention_type,
            'use_bidirectional': use_bidirectional,
            'bidirectional_fusion': bidirectional_fusion,
            'use_multiscale': use_multiscale,
            'multiscale_kernel_sizes': multiscale_kernel_sizes,
        })
    elif config.model.encoder_type == "cnn_transformer":
        encoder_kwargs.update({
            'cnn_channels': config.model.cnn_channels,
            'cnn_kernel_sizes': config.model.cnn_kernel_sizes,
            'cnn_strides': config.model.cnn_strides,
            'transformer_d_model': getattr(config.model, 'mamba_d_model', 256),  # 使用相同的维度
            'transformer_n_layers': getattr(config.model, 'transformer_n_layers', 4),
            'transformer_n_heads': getattr(config.model, 'transformer_n_heads', 8),
            'transformer_d_ff': getattr(config.model, 'transformer_d_ff', 1024),
            'transformer_dropout': getattr(config.model, 'transformer_dropout', 0.1),
        })
    
    encoder = build_encoder(
        encoder_type=config.model.encoder_type,
        **encoder_kwargs
    )
    
    # 获取编码器输出维度
    if config.model.encoder_type == "cnn_only":
        encoder_dim = config.model.cnn_channels[-1]
    else:
        encoder_dim = config.model.mamba_d_model if hasattr(config.model, 'mamba_d_model') else 256
    
    # 获取分类头参数
    classifier_hidden_dim = getattr(config.classifier, 'hidden_dim', 128)
    classifier_dropout = getattr(config.classifier, 'dropout', 0.3)
    classifier_use_bn = getattr(config.classifier, 'use_bn', True)
    classifier_use_ln = getattr(config.classifier, 'use_ln', False)
    classifier_num_layers = getattr(config.classifier, 'num_layers', 1)
    
    # 创建完整模型
    model = SupervisedModel(
        encoder=encoder,
        num_classes=num_classes,
        encoder_dim=encoder_dim,
        hidden_dim=classifier_hidden_dim,
        dropout=classifier_dropout,
        use_bn=classifier_use_bn,
        use_ln=classifier_use_ln,
        classifier_num_layers=classifier_num_layers,
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
    # 默认开启确定性模式以确保结果可重现，但会降低训练速度
    # 使用 --no-deterministic 参数可以禁用确定性模式以提升速度
    deterministic = not args.no_deterministic
    set_seed(args.seed, deterministic=deterministic)
    
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
    logger.info(f"确定性模式: {'启用' if deterministic else '禁用'}")
    if deterministic:
        logger.warning(
            "确定性模式已启用以确保结果可重现，但会显著降低训练速度。\n"
            "如需提升速度，可使用 --no-deterministic 参数禁用确定性模式。"
        )
    
    # 打印配置
    print_config(config)
    
    # 保存配置
    from omegaconf import OmegaConf
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        OmegaConf.save(config, f)
    logger.info(f"配置已保存到: {config_save_path}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_dataloaders(
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
    
    # 检查是否启用 GPU 增强
    use_gpu_augment = True  # 默认启用 GPU 增强以提升训练速度
    if hasattr(config, 'data_cache') and hasattr(config.data_cache, 'use_gpu_augment'):
        use_gpu_augment = config.data_cache.use_gpu_augment
    
    if use_gpu_augment:
        logger.info("GPU 批量增强已启用 - 数据增强将在 GPU 上批量执行")
    
    # 创建训练器
    trainer = DownstreamTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        output_dir=output_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        experiment_name=args.experiment_name,
        use_gpu_augment=use_gpu_augment,
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
        logger.info("最终测试集指标:")
        for metric_name, metric_value in best_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # 保存最终结果
        from omegaconf import OmegaConf
        results = {
            "task": args.task,
            "encoder_type": config.model.encoder_type,
            "experiment_name": args.experiment_name,
            "seed": args.seed,
            "test_metrics": best_metrics,
            "config": OmegaConf.to_container(config, resolve=True),
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
