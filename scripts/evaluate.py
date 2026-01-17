#!/usr/bin/env python
"""
PA-HCL 评估脚本。

对训练好的模型进行全面评估，提供详细的指标和分析。

用法:
    # 评估下游模型
    python scripts/evaluate.py --checkpoint outputs/downstream/best_model.pt \
        --data-dir data/processed --split test
    
    # 使用混淆矩阵进行评估
    python scripts/evaluate.py --checkpoint outputs/downstream/best_model.pt \
        --data-dir data/processed --split test --confusion-matrix
    
    # 交叉验证评估
    python scripts/evaluate.py --checkpoint outputs/downstream/best_model.pt \
        --data-dir data/processed --cross-val 5

作者: PA-HCL 团队
"""

import argparse
import json
import sys
from pathlib import Path

# 将项目根目录添加到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logger
from src.utils.metrics import (
    compute_classification_metrics,
    compute_confusion_matrix,
    get_classification_report,
    find_optimal_threshold
)


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="PA-HCL 模型评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必填
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="模型检查点路径"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="处理后的数据目录路径"
    )
    
    # 数据选项
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="要评估的数据集划分"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="评估批大小"
    )
    
    # 评估选项
    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="计算并保存混淆矩阵"
    )
    parser.add_argument(
        "--cross-val",
        type=int,
        default=0,
        help="交叉验证折数 (0 表示单次评估)"
    )
    parser.add_argument(
        "--find-threshold",
        action="store_true",
        help="查找最佳分类阈值"
    )
    
    # 输出
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="结果输出目录"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="将预测保存到文件"
    )
    
    # 其他
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (可选)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device):
    """从检查点加载模型。"""
    from src.trainers.downstream_trainer import DownstreamModel
    from src.models.pahcl import build_pahcl_model
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 检查是下游检查点还是预训练检查点
    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        config = None
    
    # 根据检查点类型构建模型
    model_state = checkpoint.get("model_state_dict", checkpoint)
    
    # 从状态字典键中推断模型类型
    if any("classifier" in k for k in model_state.keys()):
        # 下游模型
        encoder = build_pahcl_model(config)
        
        # 从状态字典获取编码器维度
        encoder_dim = 256  # 默认值
        
        # 从分类器输出获取类别数量
        for k, v in model_state.items():
            if "classifier" in k and "weight" in k and len(v.shape) == 2:
                num_classes = v.shape[0]
                if "linear" not in k:  # 输入层
                    encoder_dim = v.shape[1]
                break
        
        model = DownstreamModel(
            encoder=encoder,
            num_classes=num_classes,
            encoder_dim=encoder_dim
        )
        model.load_state_dict(model_state)
    else:
        # 预训练模型 - 包装分类器以进行评估
        model = build_pahcl_model(config)
        model.load_state_dict(model_state)
    
    model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> dict:
    """在数据集上评估模型。"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in data_loader:
        signals = batch["signal"].to(device)
        labels = batch["label"]
        
        logits = model(signals)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算指标
    metrics = compute_classification_metrics(all_labels, all_preds, all_probs)
    
    return {
        "metrics": metrics,
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs
    }


def main():
    args = parse_args()
    
    # 设置
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        name="evaluate",
        log_file=output_dir / "evaluate.log"
    )
    
    logger.info("=" * 60)
    logger.info("PA-HCL 模型评估")
    logger.info("=" * 60)
    
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # 如果提供则加载配置
    config = load_config(args.config) if args.config else None
    
    # 加载模型
    logger.info(f"Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # 加载数据集
    from src.data.dataset import PCGDownstreamDataset
    
    dataset = PCGDownstreamDataset(
        data_dir=args.data_dir,
        split=args.split
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    logger.info(f"Evaluating on {args.split} split: {len(dataset)} samples")
    
    # 评估
    results = evaluate_model(model, data_loader, device)
    metrics = results["metrics"]
    
    # 记录指标
    logger.info("\n" + "=" * 40)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 40)
    
    for name, value in sorted(metrics.items()):
        if isinstance(value, float):
            logger.info(f"  {name}: {value:.4f}")
    
    # 分类报告
    num_classes = len(np.unique(results["labels"]))
    class_names = [f"Class_{i}" for i in range(num_classes)]
    
    report = get_classification_report(
        results["labels"],
        results["predictions"],
        target_names=class_names
    )
    logger.info("\nClassification Report:\n" + report)
    
    # 混淆矩阵
    if args.confusion_matrix:
        cm = compute_confusion_matrix(
            results["labels"],
            results["predictions"],
            normalize="true"
        )
        
        logger.info("\nNormalized Confusion Matrix:")
        logger.info(np.array2string(cm, precision=3))
        
        # 保存混淆矩阵
        np.save(output_dir / "confusion_matrix.npy", cm)
    
    # 查找最佳阈值 (仅二分类)
    if args.find_threshold and num_classes == 2:
        opt_thresh, opt_f1 = find_optimal_threshold(
            results["labels"],
            results["probabilities"][:, 1]
        )
        logger.info(f"\nOptimal Threshold: {opt_thresh:.3f} (F1: {opt_f1:.4f})")
        metrics["optimal_threshold"] = opt_thresh
        metrics["optimal_f1"] = opt_f1
    
    # 保存结果
    results_dict = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_samples": len(dataset),
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v 
                   for k, v in metrics.items()}
    }
    
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_dir}")
    
    # 如果请求则保存预测
    if args.save_predictions:
        np.savez(
            output_dir / "predictions.npz",
            predictions=results["predictions"],
            labels=results["labels"],
            probabilities=results["probabilities"]
        )
        logger.info("Predictions saved to predictions.npz")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
