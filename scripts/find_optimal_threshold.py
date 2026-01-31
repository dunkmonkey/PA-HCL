#!/usr/bin/env python
"""
PA-HCL 最优阈值搜索工具。

此脚本用于在不重新训练模型的情况下，通过调整分类阈值来平衡
召回率(Recall)和特异性(Specificity)。

功能：
1. 加载已训练的模型和验证集
2. 计算验证集上的预测概率
3. 绘制 Precision-Recall 曲线和 ROC 曲线
4. 搜索最优阈值（支持多种策略：Youden's J, F1, Accuracy）
5. 输出新阈值下的各项指标

用法:
    # 基础用法：搜索最优阈值
    python scripts/find_optimal_threshold.py \\
        --checkpoint outputs/supervised_baseline/physionet2016_cnn-mamba/best_model.pt \\
        --task physionet2016
    
    # 指定优化目标
    python scripts/find_optimal_threshold.py \\
        --checkpoint path/to/model.pt \\
        --task physionet2016 \\
        --method f1  # youden, f1, accuracy, specificity
    
    # 手动指定阈值测试
    python scripts/find_optimal_threshold.py \\
        --checkpoint path/to/model.pt \\
        --task physionet2016 \\
        --threshold 0.65

作者: PA-HCL 团队
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录添加到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score
)

from src.config import load_config
from src.utils.seed import set_seed
from src.data.dataset import PCGDownstreamDataset
from src.models.encoder import build_encoder
from src.models.heads import ClassificationHead


class SupervisedModel(nn.Module):
    """监督学习分类模型（与训练脚本保持一致）。"""
    
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
        super().__init__()
        self.encoder = encoder
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
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits


def parse_args():
    parser = argparse.ArgumentParser(
        description="PA-HCL 最优阈值搜索工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型检查点路径"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="physionet2016",
        help="任务名称"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/supervised_baseline.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/root/autodl-tmp/data/downstream",
        help="数据目录"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="youden",
        choices=["youden", "f1", "accuracy", "specificity", "balanced"],
        help="阈值优化方法"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="手动指定阈值（跳过搜索）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认与模型同目录）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="批大小"
    )
    parser.add_argument(
        "--use-test",
        action="store_true",
        help="在测试集上评估（默认使用验证集搜索）"
    )
    
    return parser.parse_args()


def load_model_and_data(args):
    """加载模型和数据。"""
    print("=" * 60)
    print("加载模型和数据")
    print("=" * 60)
    
    # 加载配置
    config = load_config(args.config)
    
    # 加载任务配置
    task_config_path = project_root / "configs" / "tasks" / f"{args.task}.yaml"
    if task_config_path.exists():
        from omegaconf import OmegaConf
        task_config = load_config(str(task_config_path))
        config = OmegaConf.merge(config, task_config)
    
    # 获取类别数
    if hasattr(config, 'classification'):
        num_classes = config.classification.num_classes
    elif hasattr(config, 'downstream'):
        num_classes = config.downstream.num_classes
    else:
        num_classes = 2  # 默认二分类
    
    print(f"任务: {args.task}")
    print(f"类别数: {num_classes}")
    
    # 构建编码器参数
    encoder_kwargs = {
        'in_channels': 1,
        'cnn_channels': config.model.cnn_channels,
        'cnn_kernel_sizes': config.model.cnn_kernel_sizes,
        'cnn_strides': config.model.cnn_strides,
        'mamba_d_model': getattr(config.model, 'mamba_d_model', 256),
        'mamba_n_layers': getattr(config.model, 'mamba_n_layers', 4),
        'mamba_d_state': getattr(config.model, 'mamba_d_state', 16),
        'mamba_expand': getattr(config.model, 'mamba_expand_factor', 2),
        'drop_path_rate': getattr(config.model, 'drop_path_rate', 0.0),
        'attention_type': getattr(config.model, 'attention_type', 'none'),
        'use_bidirectional': getattr(config.model, 'use_bidirectional', False),
    }
    
    encoder = build_encoder(
        encoder_type=config.model.encoder_type,
        **encoder_kwargs
    )
    
    encoder_dim = config.model.mamba_d_model if hasattr(config.model, 'mamba_d_model') else 256
    
    # 创建模型
    model = SupervisedModel(
        encoder=encoder,
        num_classes=num_classes,
        encoder_dim=encoder_dim,
        hidden_dim=getattr(config.classifier, 'hidden_dim', 128),
        dropout=getattr(config.classifier, 'dropout', 0.3),
        use_bn=getattr(config.classifier, 'use_bn', True),
        use_ln=getattr(config.classifier, 'use_ln', False),
        classifier_num_layers=getattr(config.classifier, 'num_layers', 1),
    )
    
    # 加载权重
    print(f"加载检查点: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"模型已加载到: {device}")
    
    # 加载数据
    data_dir = Path(args.data_dir) / args.task
    csv_path = data_dir / ("test.csv" if args.use_test else "val.csv")
    
    print(f"加载数据: {csv_path}")
    
    dataset = PCGDownstreamDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        sample_rate=config.data.sample_rate,
        target_length=config.data.target_length,
        mode='val',
        cache_in_memory=True,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"样本数: {len(dataset)}")
    print(f"标签分布: {dataset.get_label_distribution()}")
    
    return model, loader, device, num_classes


def get_predictions(model, loader, device, num_classes):
    """获取模型在数据集上的预测概率。"""
    print("\n获取预测概率...")
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="推理中"):
            signals = batch['signal'].to(device)
            labels = batch['label']
            
            logits = model(signals)
            probs = torch.softmax(logits, dim=1)
            
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())
    
    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_probs)
    
    # 对于二分类，只取正类概率
    if num_classes == 2:
        y_score = y_probs[:, 1]
    else:
        y_score = y_probs
    
    return y_true, y_score, y_probs


def compute_metrics_at_threshold(y_true, y_score, threshold):
    """计算给定阈值下的各项指标。"""
    y_pred = (y_score >= threshold).astype(int)
    
    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),  # 灵敏度
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
    }
    
    # Youden's J
    metrics['youden_j'] = metrics['recall'] + metrics['specificity'] - 1
    
    # 平衡准确率
    metrics['balanced_accuracy'] = (metrics['recall'] + metrics['specificity']) / 2
    
    return metrics


def find_optimal_threshold(y_true, y_score, method="youden"):
    """搜索最优阈值。"""
    print(f"\n搜索最优阈值 (方法: {method})...")
    
    # 候选阈值
    thresholds = np.linspace(0.1, 0.9, 81)
    
    best_threshold = 0.5
    best_score = -np.inf
    all_results = []
    
    for thresh in thresholds:
        metrics = compute_metrics_at_threshold(y_true, y_score, thresh)
        all_results.append(metrics)
        
        if method == "youden":
            score = metrics['youden_j']
        elif method == "f1":
            score = metrics['f1']
        elif method == "accuracy":
            score = metrics['accuracy']
        elif method == "specificity":
            # 在保持 recall >= 0.85 的前提下最大化 specificity
            score = metrics['specificity'] if metrics['recall'] >= 0.85 else -1
        elif method == "balanced":
            score = metrics['balanced_accuracy']
        else:
            score = metrics['youden_j']
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, all_results


def plot_curves(y_true, y_score, output_dir, best_threshold=0.5):
    """绘制 PR 曲线和 ROC 曲线。"""
    print("\n绘制曲线...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Precision-Recall 曲线
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    # ROC 曲线
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    
    # 标记最优阈值点
    # 找到最接近 best_threshold 的 ROC 点
    idx = np.argmin(np.abs(roc_thresholds - best_threshold))
    plt.scatter([fpr[idx]], [tpr[idx]], c='red', s=100, zorder=5, 
                label=f'Optimal (thresh={best_threshold:.2f})')
    
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pr_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"曲线已保存到: {output_dir / 'pr_roc_curves.png'}")
    
    # 绘制阈值 vs 指标曲线
    thresholds = np.linspace(0.1, 0.9, 81)
    recalls = []
    specificities = []
    f1s = []
    accuracies = []
    
    for thresh in thresholds:
        metrics = compute_metrics_at_threshold(y_true, y_score, thresh)
        recalls.append(metrics['recall'])
        specificities.append(metrics['specificity'])
        f1s.append(metrics['f1'])
        accuracies.append(metrics['accuracy'])
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, recalls, 'b-', linewidth=2, label='Recall (Sensitivity)')
    plt.plot(thresholds, specificities, 'r-', linewidth=2, label='Specificity')
    plt.plot(thresholds, f1s, 'g-', linewidth=2, label='F1 Score')
    plt.plot(thresholds, accuracies, 'm-', linewidth=2, label='Accuracy')
    
    # 标记最优阈值
    plt.axvline(x=best_threshold, color='k', linestyle='--', alpha=0.7,
                label=f'Optimal threshold = {best_threshold:.2f}')
    
    plt.xlabel('Classification Threshold', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.title('Metrics vs Classification Threshold', fontsize=14)
    plt.legend(loc='center right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0.1, 0.9)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"阈值曲线已保存到: {output_dir / 'threshold_metrics.png'}")
    
    return pr_auc, roc_auc


def main():
    args = parse_args()
    set_seed(42)
    
    # 设置输出目录
    if args.output_dir is None:
        args.output_dir = Path(args.checkpoint).parent / "threshold_analysis"
    
    # 加载模型和数据
    model, loader, device, num_classes = load_model_and_data(args)
    
    # 获取预测
    y_true, y_score, y_probs = get_predictions(model, loader, device, num_classes)
    
    # AUROC
    auroc = roc_auc_score(y_true, y_score)
    print(f"\nAUROC: {auroc:.4f}")
    
    # 默认阈值 (0.5) 的指标
    print("\n" + "=" * 60)
    print("默认阈值 (0.5) 的指标:")
    print("=" * 60)
    default_metrics = compute_metrics_at_threshold(y_true, y_score, 0.5)
    for k, v in default_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # 搜索或使用指定阈值
    if args.threshold is not None:
        best_threshold = args.threshold
        print(f"\n使用手动指定阈值: {best_threshold}")
    else:
        best_threshold, all_results = find_optimal_threshold(y_true, y_score, args.method)
    
    # 最优阈值的指标
    print("\n" + "=" * 60)
    print(f"最优阈值 ({best_threshold:.2f}) 的指标:")
    print("=" * 60)
    optimal_metrics = compute_metrics_at_threshold(y_true, y_score, best_threshold)
    for k, v in optimal_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    # 对比改善
    print("\n" + "=" * 60)
    print("阈值移动带来的改善:")
    print("=" * 60)
    improvements = {
        'specificity': optimal_metrics['specificity'] - default_metrics['specificity'],
        'recall': optimal_metrics['recall'] - default_metrics['recall'],
        'f1': optimal_metrics['f1'] - default_metrics['f1'],
        'accuracy': optimal_metrics['accuracy'] - default_metrics['accuracy'],
        'balanced_accuracy': optimal_metrics['balanced_accuracy'] - default_metrics['balanced_accuracy'],
    }
    for k, v in improvements.items():
        sign = "+" if v >= 0 else ""
        print(f"  {k}: {sign}{v:.4f} ({sign}{v*100:.2f}%)")
    
    # 绘制曲线
    pr_auc, roc_auc = plot_curves(y_true, y_score, args.output_dir, best_threshold)
    
    # 保存结果
    import json
    results = {
        'checkpoint': str(args.checkpoint),
        'task': args.task,
        'method': args.method,
        'auroc': float(auroc),
        'auprc': float(pr_auc),
        'default_threshold': 0.5,
        'optimal_threshold': float(best_threshold),
        'default_metrics': {k: float(v) if isinstance(v, (float, np.floating)) else int(v) 
                           for k, v in default_metrics.items()},
        'optimal_metrics': {k: float(v) if isinstance(v, (float, np.floating)) else int(v) 
                           for k, v in optimal_metrics.items()},
        'improvements': {k: float(v) for k, v in improvements.items()},
    }
    
    output_path = Path(args.output_dir) / 'threshold_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_path}")
    
    print("\n" + "=" * 60)
    print("建议:")
    print("=" * 60)
    print(f"  推荐使用阈值: {best_threshold:.2f}")
    print(f"  预计 Specificity 提升: {improvements['specificity']*100:.1f}%")
    print(f"  预计 Recall 变化: {improvements['recall']*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
