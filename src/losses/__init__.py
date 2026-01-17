"""
PA-HCL 的损失函数。

此包提供：
- InfoNCE: 标准对比损失
- HierarchicalContrastiveLoss: 周期 + 子结构级对比
- SupConLoss: 监督对比损失
- NTXentLoss: NT-Xent 损失 (SimCLR)
"""

from .contrastive import (
    InfoNCELoss,
    SubstructureContrastiveLoss,
    HierarchicalContrastiveLoss,
    SupConLoss,
    NTXentLoss,
    get_contrastive_loss,
)

__all__ = [
    "InfoNCELoss",
    "SubstructureContrastiveLoss",
    "HierarchicalContrastiveLoss",
    "SupConLoss",
    "NTXentLoss",
    "get_contrastive_loss",
]
