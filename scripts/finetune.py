#!/usr/bin/env python
"""
PA-HCL 下游微调脚本。

此脚本在预训练的 PA-HCL 模型上运行下游分类任务。
支持多种任务类型和灵活的配置方式。

支持的任务:
    - circor_murmur: CirCor 杂音检测三分类
    - circor_outcome: CirCor 临床结果二分类
    - physionet2016: PhysioNet 2016 心音二分类
    - pascal: PASCAL 心音三分类

用法:
    # 使用任务名称自动加载配置
    python scripts/finetune.py --task circor_murmur \\
        --pretrained checkpoints/pretrain/best_model.pt
    
    # 线性评估 (冻结编码器)
    python scripts/finetune.py --task circor_murmur \\
        --pretrained checkpoints/pretrain/best_model.pt \\
        --linear-eval
    
    # 全量微调
    python scripts/finetune.py --task circor_outcome \\
        --pretrained checkpoints/pretrain/best_model.pt
    
    # 小样本学习
    python scripts/finetune.py --task physionet2016 \\
        --pretrained checkpoints/pretrain/best_model.pt \\
        --few-shot --shot-ratio 0.1
    
    # 使用自定义配置
    python scripts/finetune.py --task-config configs/tasks/circor_murmur.yaml \\
        --pretrained checkpoints/pretrain/best_model.pt

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
import numpy as np

from src.config import load_config, print_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logger


# 可用任务列表
AVAILABLE_TASKS = [
    "circor_murmur",
    "circor_outcome", 
    "physionet2016",
    "pascal"
]


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="PA-HCL 下游微调",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
可用任务:
    {', '.join(AVAILABLE_TASKS)}

示例:
    # CirCor 杂音检测
    python scripts/finetune.py --task circor_murmur --pretrained checkpoints/pretrain/best_model.pt
    
    # 线性评估模式
    python scripts/finetune.py --task circor_murmur --pretrained ... --linear-eval
    
    # 10% 少样本学习
    python scripts/finetune.py --task circor_murmur --pretrained ... --few-shot --shot-ratio 0.1
        """
    )
    
    # 任务配置
    task_group = parser.add_argument_group("任务配置")
    task_group.add_argument(
        "--task",
        type=str,
        choices=AVAILABLE_TASKS,
        help=f"任务名称，自动加载 configs/tasks/<task>.yaml"
    )
    task_group.add_argument(
        "--task-config",
        type=str,
        default=None,
        help="自定义任务配置文件路径"
    )
    task_group.add_argument(
        "--config",
        type=str,
        default="configs/finetune.yaml",
        help="基础配置文件路径（被任务配置覆盖）"
    )
    
    # 预训练模型
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="预训练检查点路径"
    )
    
    # 训练模式
    mode_group = parser.add_argument_group("训练模式")
    mode_group.add_argument(
        "--linear-eval",
        action="store_true",
        help="线性评估模式（冻结编码器）"
    )
    mode_group.add_argument(
        "--few-shot",
        action="store_true",
        help="启用小样本学习"
    )
    mode_group.add_argument(
        "--shot-ratio",
        type=float,
        default=0.1,
        help="小样本学习的训练数据比例"
    )
    
    # 输出
    output_group = parser.add_argument_group("输出设置")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="输出根目录"
    )
    output_group.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="实验名称（默认使用任务名称）"
    )
    
    # 覆盖参数
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
        default="PA-HCL",
        help="WandB 项目名称"
    )
    tracking_group.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="WandB 实体/用户名"
    )
    
    return parser.parse_args()


def load_task_config(args):
    """加载任务配置。"""
    # 确定配置文件路径
    if args.task_config:
        task_config_path = Path(args.task_config)
    elif args.task:
        task_config_path = project_root / "configs" / "tasks" / f"{args.task}.yaml"
    else:
        # 使用默认配置
        task_config_path = None
    
    # 加载基础配置
    base_config = load_config(args.config)
    
    # 如果有任务配置，加载并合并
    if task_config_path and task_config_path.exists():
        task_config = load_config(str(task_config_path))
        config = merge_configs(base_config, task_config)
    else:
        config = base_config
        task_config_path = Path(args.config)
    
    return config, task_config_path


def merge_configs(base, override):
    """递归合并配置，override 优先。"""
    from types import SimpleNamespace
    
    def to_dict(obj):
        if isinstance(obj, SimpleNamespace):
            return {k: to_dict(v) for k, v in vars(obj).items()}
        return obj
    
    def to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
        return d
    
    base_dict = to_dict(base)
    override_dict = to_dict(override)
    
    def deep_merge(b, o):
        result = b.copy() if isinstance(b, dict) else {}
        for key, value in o.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    merged = deep_merge(base_dict, override_dict)
    return to_namespace(merged)


def main():
    args = parse_args()
    
    # 加载配置
    config, task_config_path = load_task_config(args)
    
    # 获取任务名称
    if hasattr(config, 'task') and hasattr(config.task, 'name'):
        task_name = config.task.name
    elif args.task:
        task_name = args.task
    else:
        task_name = "default"
    
    # 应用命令行覆盖
    if hasattr(config, 'training'):
        training_cfg = config.training
    elif hasattr(config, 'downstream'):
        training_cfg = config.downstream
    else:
        from types import SimpleNamespace
        training_cfg = SimpleNamespace()
    
    if args.epochs is not None:
        training_cfg.num_epochs = args.epochs
    if args.lr is not None:
        training_cfg.learning_rate = args.lr
    if args.batch_size is not None:
        training_cfg.batch_size = args.batch_size
    
    # 设置实验名称
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        suffix = "_linear" if args.linear_eval else "_finetune"
        if args.few_shot:
            suffix += f"_fewshot{int(args.shot_ratio*100)}pct"
        experiment_name = f"{task_name}{suffix}"
    
    # 设置输出目录
    output_dir = Path(args.output_dir) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    logger = setup_logger(
        name="finetune",
        log_file=output_dir / "finetune.log"
    )
    
    logger.info("=" * 60)
    logger.info("PA-HCL 下游微调")
    logger.info("=" * 60)
    logger.info(f"任务: {task_name}")
    logger.info(f"任务配置: {task_config_path}")
    logger.info(f"模式: {'线性评估' if args.linear_eval else '全量微调'}")
    if args.few_shot:
        logger.info(f"小样本学习: {args.shot_ratio*100:.1f}%")
    logger.info(f"输出目录: {output_dir}")
    
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")
    
    # ============== 加载数据 ==============
    from src.data.dataset import create_task_dataloaders, downstream_collate_fn
    
    logger.info(f"\n加载数据: {args.data_dir}/{task_name}")
    
    # 获取训练参数
    batch_size = getattr(training_cfg, 'batch_size', 32)
    
    # 创建数据加载器
    try:
        data_loaders = create_task_dataloaders(
            task_name=task_name,
            base_dir=args.data_dir,
            batch_size=batch_size,
            num_workers=args.num_workers
        )
        
        train_loader = data_loaders['train']
        val_loader = data_loaders['val']
        test_loader = data_loaders['test']
        
        num_classes = data_loaders['num_classes']
        class_weights = data_loaders['class_weights']
        label_map = data_loaders['label_map']
        label_distribution = data_loaders.get('label_distribution', {})
        
    except FileNotFoundError as e:
        logger.error(f"数据加载失败: {e}")
        logger.error(f"请先运行数据准备脚本:")
        logger.error(f"  python scripts/data_preparation/prepare_downstream_tasks.py --dataset {task_name.split('_')[0]}")
        return
    
    logger.info(f"类别数: {num_classes}")
    logger.info(f"标签映射: {label_map}")
    logger.info(f"标签分布: {label_distribution}")
    logger.info(f"类别权重: {class_weights.tolist()}")
    logger.info(f"训练样本: {len(train_loader.dataset)}")
    logger.info(f"验证样本: {len(val_loader.dataset) if val_loader else 0}")
    logger.info(f"测试样本: {len(test_loader.dataset) if test_loader else 0}")
    
    # 小样本采样
    if args.few_shot:
        from torch.utils.data import Subset, DataLoader
        
        original_size = len(train_loader.dataset)
        n_samples = max(1, int(original_size * args.shot_ratio))
        
        # 受试者级下采样
        dataset = train_loader.dataset
        if hasattr(dataset, 'dataset'):  # Subset 包装
            base_dataset = dataset.dataset
            indices = list(dataset.indices) if hasattr(dataset, 'indices') else list(range(len(dataset)))
        else:
            base_dataset = dataset
            indices = list(range(len(dataset)))
        
        # 按受试者分组
        subject_to_indices = {}
        for idx in indices:
            if hasattr(base_dataset, 'get_subject_id'):
                subject_id = base_dataset.get_subject_id(idx)
            else:
                subject_id = str(idx)
            if subject_id not in subject_to_indices:
                subject_to_indices[subject_id] = []
            subject_to_indices[subject_id].append(idx)
        
        # 随机选择受试者
        subjects = list(subject_to_indices.keys())
        np.random.seed(args.seed)
        np.random.shuffle(subjects)
        
        selected_indices = []
        for subject in subjects:
            selected_indices.extend(subject_to_indices[subject])
            if len(selected_indices) >= n_samples:
                break
        
        selected_indices = selected_indices[:n_samples]
        
        # 创建新的训练集
        few_shot_dataset = Subset(base_dataset, selected_indices)
        train_loader = DataLoader(
            few_shot_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=downstream_collate_fn,
            pin_memory=True,
            drop_last=True
        )
        
        logger.info(f"小样本训练: {len(few_shot_dataset)} 样本 ({args.shot_ratio*100:.1f}%)")
    
    # ============== 加载预训练模型 ==============
    from src.trainers.downstream_trainer import load_pretrained_encoder, DownstreamModel, DownstreamTrainer
    
    logger.info(f"\n加载预训练模型: {args.pretrained}")
    
    pretrained_model = load_pretrained_encoder(
        args.pretrained,
        device,
        config,
        logger=logger  # Step 2: 传递 logger 以获取详细加载信息
    )
    
    # 获取编码器
    encoder = pretrained_model.encoder if hasattr(pretrained_model, 'encoder') else pretrained_model
    encoder_dim = getattr(pretrained_model, 'encoder_dim', 256)
    
    # 获取分类器配置
    if hasattr(config, 'classifier'):
        hidden_dim = getattr(config.classifier, 'hidden_dim', 128)
        dropout = getattr(config.classifier, 'dropout', 0.3)
    elif hasattr(config, 'downstream'):
        hidden_dim = getattr(config.downstream, 'hidden_dim', 128)
        dropout = getattr(config.downstream, 'dropout', 0.3)
    else:
        hidden_dim = 128
        dropout = 0.3
    
    # 线性评估时不使用隐藏层
    if args.linear_eval:
        hidden_dim = None
    
    # 创建下游模型
    model = DownstreamModel(
        encoder=encoder,
        num_classes=num_classes,
        encoder_dim=encoder_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        freeze_encoder=args.linear_eval
    )
    
    logger.info(f"编码器维度: {encoder_dim}")
    logger.info(f"分类器隐藏层: {hidden_dim}")
    logger.info(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # ============== 获取训练配置 ==============
    num_epochs = getattr(training_cfg, 'num_epochs', 100)
    learning_rate = getattr(training_cfg, 'learning_rate', 1e-3)
    weight_decay = getattr(training_cfg, 'weight_decay', 1e-4)
    warmup_epochs = getattr(training_cfg, 'warmup_epochs', 5)
    label_smoothing = getattr(training_cfg, 'label_smoothing', 0.1)
    use_class_weights = getattr(training_cfg, 'use_class_weights', True)
    early_stopping_patience = getattr(training_cfg, 'early_stopping_patience', 15)
    use_amp = getattr(training_cfg, 'use_amp', True)
    log_interval = getattr(training_cfg, 'log_interval', 20)
    save_interval = getattr(training_cfg, 'save_interval', 10)
    
    # 获取评估配置
    if hasattr(config, 'evaluation'):
        primary_metric = getattr(config.evaluation, 'primary_metric', 'f1_macro')
    else:
        primary_metric = 'f1_macro' if num_classes > 2 else 'auroc'
    
    logger.info(f"\n训练配置:")
    logger.info(f"  轮数: {num_epochs}")
    logger.info(f"  学习率: {learning_rate}")
    logger.info(f"  批大小: {batch_size}")
    logger.info(f"  使用类别权重: {use_class_weights}")
    logger.info(f"  主要指标: {primary_metric}")
    
    # ============== 创建训练器 ==============
    trainer = DownstreamTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        label_smoothing=label_smoothing,
        class_weights=class_weights if use_class_weights else None,
        use_amp=use_amp,
        early_stopping_patience=early_stopping_patience,
        log_interval=log_interval,
        save_interval=save_interval,
        output_dir=args.output_dir,
        experiment_name=experiment_name,
        seed=args.seed,
        task_name=task_name,
        primary_metric=primary_metric,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )
    
    # ============== 训练 ==============
    logger.info("\n开始训练...")
    metrics = trainer.train()
    
    # ============== 最终结果 ==============
    logger.info("\n" + "=" * 60)
    logger.info("最终测试结果:")
    logger.info("=" * 60)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 60)
    
    # 保存最终指标
    metrics_path = output_dir / "final_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (float, np.floating)) else v 
                   for k, v in metrics.items()}, f, indent=2)
    logger.info(f"指标已保存: {metrics_path}")
    
    logger.info("\n训练完成!")


if __name__ == "__main__":
    main()
