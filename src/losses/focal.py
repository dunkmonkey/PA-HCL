"""
PA-HCL 的 Focal Loss 实现。

Focal Loss 通过降低易分类样本的权重来解决类别不平衡问题，
使模型更关注难分类的样本。

参考:
    Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection.

作者: PA-HCL 团队
"""

from typing import Optional, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss 用于解决类别不平衡问题。
    
    公式:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    其中:
        - p_t: 正确类别的预测概率
        - alpha: 类别权重（可选）
        - gamma: 聚焦参数，gamma > 0 时降低易分类样本权重
    
    gamma = 0 时退化为标准交叉熵损失。
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0
    ):
        """
        参数:
            gamma: 聚焦参数，推荐值 [0.5, 5]，常用 2.0
            alpha: 类别权重
                   - None: 不使用类别权重
                   - float: 二分类时正类权重，负类权重为 1-alpha
                   - List/Tensor: 多分类时每个类别的权重
            reduction: 损失聚合方式 ("none", "mean", "sum")
            label_smoothing: 标签平滑系数 [0, 1)
        """
        super().__init__()
        
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        # 处理 alpha 参数
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
            elif isinstance(alpha, torch.Tensor):
                self.register_buffer("alpha", alpha.float())
            else:
                # 二分类情况
                self.register_buffer("alpha", torch.tensor([1 - alpha, alpha], dtype=torch.float32))
        else:
            self.alpha = None
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Focal Loss。
        
        参数:
            inputs: 模型输出 logits [B, C] 或 [B, C, ...]
            targets: 真实标签 [B] 或 [B, ...]（整数类别索引）
            
        返回:
            Focal Loss 值
        """
        # 获取类别数
        num_classes = inputs.shape[1] if inputs.dim() > 1 else 2
        
        # 计算 log_softmax 以提高数值稳定性
        log_p = F.log_softmax(inputs, dim=1)
        
        # 计算 softmax 概率
        p = torch.exp(log_p)
        
        # 标签平滑
        if self.label_smoothing > 0:
            # 创建平滑后的 one-hot 标签
            smooth_targets = torch.zeros_like(log_p)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.label_smoothing)
            
            # 计算交叉熵: -sum(y * log(p))
            ce = -torch.sum(smooth_targets * log_p, dim=1)
            
            # 计算 p_t (正确类别的概率)
            p_t = torch.sum(smooth_targets * p, dim=1)
        else:
            # 获取正确类别的 log 概率
            ce = F.nll_loss(log_p, targets, reduction="none")
            
            # 获取正确类别的概率 p_t
            p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # 应用类别权重
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight
        
        # 计算最终损失
        loss = focal_weight * ce
        
        # 聚合
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BinaryFocalLoss(nn.Module):
    """
    二分类 Focal Loss。
    
    专门针对二分类任务优化，支持 BCEWithLogits 风格的接口。
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean"
    ):
        """
        参数:
            gamma: 聚焦参数
            alpha: 正类权重 (负类权重为 1-alpha)
            reduction: 损失聚合方式
        """
        super().__init__()
        
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        参数:
            inputs: logits [B] 或 [B, 1]
            targets: 二分类标签 [B] 或 [B, 1]，值为 0 或 1
            
        返回:
            Binary Focal Loss 值
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        
        # Sigmoid
        p = torch.sigmoid(inputs)
        
        # 计算 p_t
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # 计算 alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        
        # Focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        loss = focal_weight * bce
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    带标签平滑的交叉熵损失。
    
    标签平滑通过软化 one-hot 标签来防止过拟合，
    提高模型的泛化能力。
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = "mean"
    ):
        """
        参数:
            smoothing: 平滑系数，推荐值 [0.05, 0.2]
            reduction: 损失聚合方式
        """
        super().__init__()
        
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        参数:
            inputs: logits [B, C]
            targets: 类别标签 [B]
            
        返回:
            Label smoothing cross entropy loss
        """
        num_classes = inputs.shape[1]
        
        # 创建平滑标签
        smooth_targets = torch.zeros_like(inputs)
        smooth_targets.fill_(self.smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # 计算交叉熵
        log_p = F.log_softmax(inputs, dim=1)
        loss = -torch.sum(smooth_targets * log_p, dim=1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def get_classification_loss(
    loss_type: str = "ce",
    num_classes: int = 2,
    class_weights: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
    **kwargs
) -> nn.Module:
    """
    获取分类损失函数的工厂函数。
    
    参数:
        loss_type: "ce", "focal", "label_smoothing" 之一
        num_classes: 类别数
        class_weights: 类别权重张量
        gamma: Focal Loss 的 gamma 参数
        label_smoothing: 标签平滑系数
        **kwargs: 其他参数
        
    返回:
        损失函数模块
    """
    if loss_type == "ce" or loss_type == "cross_entropy":
        if label_smoothing > 0:
            return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    
    elif loss_type == "focal":
        alpha = None
        if class_weights is not None:
            alpha = class_weights / class_weights.sum()  # 归一化
        return FocalLoss(
            gamma=gamma,
            alpha=alpha,
            label_smoothing=label_smoothing
        )
    
    elif loss_type == "binary_focal":
        alpha = 0.25  # 默认值
        if class_weights is not None and len(class_weights) == 2:
            alpha = class_weights[1] / class_weights.sum()
        return BinaryFocalLoss(gamma=gamma, alpha=alpha)
    
    elif loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
