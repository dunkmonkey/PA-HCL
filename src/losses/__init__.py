"""
PA-HCL 的损失函数。

此包提供：
- InfoNCE: 标准对比损失
- HierarchicalContrastiveLoss: 周期 + 子结构级对比
- SupConLoss: 监督对比损失
- NTXentLoss: NT-Xent 损失 (SimCLR)
- FocalLoss: 用于类别不平衡的 Focal Loss
"""

from .contrastive import (
    InfoNCELoss,
    SubstructureContrastiveLoss,
    HierarchicalContrastiveLoss,
    SupConLoss,
    NTXentLoss,
    get_contrastive_loss,
)

from .focal import (
    FocalLoss,
    BinaryFocalLoss,
    LabelSmoothingCrossEntropy,
    get_classification_loss,
)

__all__ = [
    # Contrastive losses
    "InfoNCELoss",
    "SubstructureContrastiveLoss",
    "HierarchicalContrastiveLoss",
    "SupConLoss",
    "NTXentLoss",
    "get_contrastive_loss",
    # Classification losses
    "FocalLoss",
    "BinaryFocalLoss",
    "LabelSmoothingCrossEntropy",
    "get_classification_loss",
]
