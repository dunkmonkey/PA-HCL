"""
PA-HCL 的评估指标。
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_auroc(
    y_true: Union[np.ndarray, List],
    y_score: Union[np.ndarray, List],
    multi_class: str = "ovr"
) -> float:
    """
    计算接收者操作特征曲线下面积 (AUROC)。
    
    参数:
        y_true: 真实标签
        y_score: 预测分数/概率
        multi_class: 多分类策略 ('ovr' 或 'ovo')
        
    返回:
        AUROC 分数
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # 检查是否只有一个类别
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        # 只有一个类别时无法计算AUROC
        return 0.0
    
    try:
        if len(y_score.shape) == 1 or y_score.shape[1] == 1:
            # 二分类：使用正类概率
            return roc_auc_score(y_true, y_score.ravel())
        else:
            # 多分类：使用完整概率矩阵
            return roc_auc_score(y_true, y_score, multi_class=multi_class, average='macro')
    except (ValueError, IndexError) as e:
        # 处理边界情况
        import warnings
        warnings.warn(f"AUROC计算失败: {str(e)}，返回0.0")
        return 0.0


def compute_auprc(
    y_true: Union[np.ndarray, List],
    y_score: Union[np.ndarray, List],
    num_classes: Optional[int] = None
) -> float:
    """
    计算精确率-召回率曲线下面积 (AUPRC)。
    
    参数:
        y_true: 真实标签
        y_score: 预测分数/概率
        num_classes: 类别数量（用于多分类）
        
    返回:
        AUPRC 分数
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    # 检查是否只有一个类别
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        return 0.0
    
    try:
        if len(y_score.shape) == 1 or y_score.shape[1] == 1:
            # 二分类：使用正类概率
            return average_precision_score(y_true, y_score.ravel())
        else:
            # 多分类：计算macro平均
            return average_precision_score(y_true, y_score, average='macro')
    except (ValueError, IndexError) as e:
        import warnings
        warnings.warn(f"AUPRC计算失败: {str(e)}，返回0.0")
        return 0.0


def compute_metrics(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    y_score: Optional[Union[np.ndarray, List]] = None,
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    计算综合分类指标。
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        y_score: 预测分数/概率（可选）
        num_classes: 类别数量
        
    返回:
        包含各种指标的字典
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    metrics = {}
    if num_classes is None:
        num_classes = len(np.unique(y_true))
    
    # 基础指标
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    
    # 每类指标和平均指标
    if num_classes == 2:
        # 二分类
        metrics["precision"] = precision_score(y_true, y_pred, average="binary", zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, average="binary", zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, average="binary", zero_division=0)
        metrics["specificity"] = compute_specificity(y_true, y_pred)
    else:
        # 多分类
        metrics["precision_macro"] = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics["recall_macro"] = recall_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics["f1_macro"] = f1_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        metrics["f1_weighted"] = f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        # 兼容通用字段
        metrics["precision"] = metrics["precision_macro"]
        metrics["recall"] = metrics["recall_macro"]
        metrics["f1"] = metrics["f1_macro"]
    
    # 基于分数的指标
    if y_score is not None:
        y_score = np.asarray(y_score)
        
        # 对于二分类，提取正类概率
        if num_classes == 2 and len(y_score.shape) == 2 and y_score.shape[1] == 2:
            y_score_binary = y_score[:, 1]  # 取正类概率
            metrics["auroc"] = compute_auroc(y_true, y_score_binary)
            metrics["auprc"] = compute_auprc(y_true, y_score_binary, num_classes)
        else:
            # 多分类或已经是正确格式
            metrics["auroc"] = compute_auroc(y_true, y_score)
            metrics["auprc"] = compute_auprc(y_true, y_score, num_classes)
    else:
        # 如果没有提供概率，设置为0.0（而不是完全不设置）
        metrics["auroc"] = 0.0
        metrics["auprc"] = 0.0
    
    return metrics


def compute_classification_metrics(
    y_true: Union[np.ndarray, List],
    y_pred: Union[np.ndarray, List],
    y_score: Optional[Union[np.ndarray, List]] = None,
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    兼容接口：计算分类相关指标。

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        y_score: 预测分数/概率（可选）
        num_classes: 类别数量

    返回:
        指标字典
    """
    return compute_metrics(y_true, y_pred, y_score, num_classes)


def compute_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    计算特异性 (真负率)。
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        
    返回:
        特异性分数
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def compute_sensitivity_specificity_curve(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算不同阈值下的灵敏度 (TPR) 和特异性。
    
    参数:
        y_true: 真实标签
        y_score: 预测分数
        
    返回:
        (thresholds, sensitivities, specificities) 的元组
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    sensitivities = tpr
    specificities = 1 - fpr
    return thresholds, sensitivities, specificities


def find_optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    method: str = "youden"
) -> Tuple[float, Dict[str, float]]:
    """
    查找最佳分类阈值。
    
    参数:
        y_true: 真实标签
        y_score: 预测分数
        method: 优化方法 ('youden' 或 'f1')
        
    返回:
        (optimal_threshold, metrics_at_threshold) 的元组
    """
    if method == "youden":
        # 最大化 Youden's J 统计量 (灵敏度 + 特异性 - 1)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
    elif method == "f1":
        # 最大化 F1 分数
        thresholds = np.linspace(0, 1, 100)
        f1_scores = []
        for thresh in thresholds:
            y_pred = (y_score >= thresh).astype(int)
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 在最佳阈值处计算指标
    y_pred_optimal = (y_score >= optimal_threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred_optimal, y_score)
    
    return optimal_threshold, metrics


class MetricTracker:
    """
    在训练/评估期间跟踪和聚合指标。
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """重置所有跟踪的指标。"""
        self.predictions = []
        self.scores = []
        self.labels = []
    
    def update(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        scores: Optional[torch.Tensor] = None
    ) -> None:
        """
        使用批次预测更新跟踪器。
        
        参数:
            preds: 预测标签
            labels: 真实标签
            scores: 预测分数/概率
        """
        self.predictions.extend(preds.cpu().numpy().tolist())
        self.labels.extend(labels.cpu().numpy().tolist())
        if scores is not None:
            self.scores.extend(scores.cpu().numpy().tolist())
    
    def compute(self, num_classes: int = 2) -> Dict[str, float]:
        """
        计算所有指标。
        
        参数:
            num_classes: 类别数量
            
        返回:
            指标字典
        """
        y_score = self.scores if self.scores else None
        return compute_metrics(
            self.labels,
            self.predictions,
            y_score,
            num_classes
        )
