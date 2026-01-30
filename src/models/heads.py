"""
PA-HCL 的投影头和分类头。

此模块提供：
- ProjectionHead: 用于对比学习的 MLP 投影头
- PredictionHead: 用于 BYOL 风格学习的预测器（可选）
- ClassificationHead: 用于下游任务的分类头

作者: PA-HCL 团队
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    用于对比学习的 MLP 投影头。
    
    将编码器输出投影到低维空间，并在其中计算对比损失。
    这遵循 SimCLR 设计，其中投影头在预训练后被丢弃。
    
    架构:
        Linear -> BN -> ReLU -> Linear -> BN (可选) -> Output
    
    隐藏层维度通常大于输出维度，以允许学习复杂的投影。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 128,
        num_layers: int = 2,
        use_bn: bool = True,
        use_bias: bool = True,
        last_bn: bool = False
    ):
        """
        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出投影维度
            num_layers: 线性层数量（2 或 3）
            use_bn: 是否使用 BatchNorm
            use_bias: 线性层是否使用偏置
            last_bn: 是否对最终输出应用 BN
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        
        if num_layers == 1:
            # Simple linear projection
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
        
        elif num_layers == 2:
            # Two-layer MLP (standard SimCLR style)
            layers.append(nn.Linear(input_dim, hidden_dim, bias=use_bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_dim, output_dim, bias=use_bias))
        
        elif num_layers == 3:
            # Three-layer MLP (BYOL style)
            layers.append(nn.Linear(input_dim, hidden_dim, bias=use_bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=use_bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            
            layers.append(nn.Linear(hidden_dim, output_dim, bias=use_bias))
        
        else:
            raise ValueError(f"num_layers must be 1, 2, or 3, got {num_layers}")
        
        # Optional final BatchNorm
        if last_bn:
            layers.append(nn.BatchNorm1d(output_dim, affine=False))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        投影特征。
        
        参数:
            x: 输入特征 [B, D] 或 [B, L, D]
            
        返回:
            投影后的特征，形状相同，最后一维 = output_dim
        """
        # Handle sequence input
        if x.dim() == 3:
            B, L, D = x.shape
            x = x.reshape(B * L, D)
            x = self.mlp(x)
            return x.reshape(B, L, -1)
        
        return self.mlp(x)


class PredictionHead(nn.Module):
    """
    用于非对称对比学习 (BYOL 风格) 的预测头。
    
    此头部由在线网络使用来预测目标网络的投影。
    它打破了两个网络之间的对称性。
    
    架构:
        Linear -> BN -> ReLU -> Linear
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 128
    ):
        """
        参数:
            input_dim: 投影特征的维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（通常应与投影维度匹配）
        """
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测目标投影。
        
        参数:
            x: 投影后的特征 [B, D]
            
        返回:
            预测值 [B, D]
        """
        return self.mlp(x)


class ClassificationHead(nn.Module):
    """
    用于下游任务的分类头。
    
    具有可选隐藏层和 dropout 的 MLP 分类器。
    用于预训练后的分类任务微调。
    
    新增特性：
    - 支持 LayerNorm（可选，替代 BatchNorm）
    - 支持更高的 dropout（缓解过拟合）
    - 支持双层 MLP（更强的非线性）
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_bn: bool = True,
        use_ln: bool = False,
        num_layers: int = 1
    ):
        """
        参数:
            input_dim: 编码器特征维度
            num_classes: 输出类别数
            hidden_dim: 隐藏层维度（None 表示线性分类器）
            dropout: Dropout 率
            use_bn: 隐藏层是否使用 BatchNorm
            use_ln: 是否使用 LayerNorm（与 use_bn 互斥，优先使用 ln）
            num_layers: 隐藏层数量（1 或 2）
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        if hidden_dim is None:
            # Linear classifier (for linear evaluation protocol)
            self.classifier = nn.Linear(input_dim, num_classes)
        else:
            # MLP classifier
            layers = [nn.Linear(input_dim, hidden_dim)]
            
            # 归一化层
            if use_ln:
                layers.append(nn.LayerNorm(hidden_dim))
            elif use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.extend([
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            
            # 可选的第二隐藏层
            if num_layers >= 2:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_ln:
                    layers.append(nn.LayerNorm(hidden_dim))
                elif use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.extend([
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
            
            layers.append(nn.Linear(hidden_dim, num_classes))
            self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入特征进行分类。
        
        参数:
            x: 输入特征 [B, D]
            
        返回:
            Logits [B, num_classes]
        """
        return self.classifier(x)


class AnomalyDetectionHead(nn.Module):
    """
    使用基于距离的方法进行异常检测的头部。
    
    此头部计算到学习原型（聚类中心）的距离来进行异常评分，
    而不是进行分类。
    """
    
    def __init__(
        self,
        input_dim: int,
        num_prototypes: int = 1,
        distance_type: str = "mahalanobis"
    ):
        """
        参数:
            input_dim: 编码器特征维度
            num_prototypes: 原型向量数量（1 表示单类）
            distance_type: 距离度量 ("euclidean", "mahalanobis", "cosine")
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_prototypes = num_prototypes
        self.distance_type = distance_type
        
        # Learnable prototypes (initialized during training)
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, input_dim))
        
        # For Mahalanobis distance, we need the inverse covariance
        if distance_type == "mahalanobis":
            # Will be set during training
            self.register_buffer(
                "inv_cov",
                torch.eye(input_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算作为到原型距离的异常分数。
        
        参数:
            x: 输入特征 [B, D]
            
        返回:
            异常分数 [B, num_prototypes]
        """
        if self.distance_type == "euclidean":
            # Euclidean distance to each prototype
            # x: [B, D], prototypes: [P, D]
            diff = x.unsqueeze(1) - self.prototypes.unsqueeze(0)  # [B, P, D]
            distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))  # [B, P]
        
        elif self.distance_type == "mahalanobis":
            # Mahalanobis distance
            diff = x.unsqueeze(1) - self.prototypes.unsqueeze(0)  # [B, P, D]
            # d = sqrt(diff @ inv_cov @ diff.T)
            distances = torch.sqrt(
                torch.einsum("bpd,de,bpe->bp", diff, self.inv_cov, diff)
            )
        
        elif self.distance_type == "cosine":
            # Cosine distance (1 - similarity)
            x_norm = F.normalize(x, dim=-1)
            proto_norm = F.normalize(self.prototypes, dim=-1)
            similarity = torch.mm(x_norm, proto_norm.t())  # [B, P]
            distances = 1 - similarity
        
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
        
        return distances
    
    def fit_prototypes(self, features: torch.Tensor) -> None:
        """
        使用训练特征上的 k-means 拟合原型。
        
        参数:
            features: 训练特征 [N, D]
        """
        # Simple k-means initialization
        # For production, use sklearn.cluster.KMeans
        with torch.no_grad():
            # Random initialization from features
            indices = torch.randperm(len(features))[:self.num_prototypes]
            self.prototypes.data = features[indices].clone()
            
            # If Mahalanobis, compute inverse covariance
            if self.distance_type == "mahalanobis":
                centered = features - features.mean(dim=0, keepdim=True)
                cov = (centered.t() @ centered) / (len(features) - 1)
                # Add small regularization for numerical stability
                cov = cov + 1e-6 * torch.eye(self.input_dim, device=cov.device)
                self.inv_cov.data = torch.inverse(cov)


class SubstructureProjectionHead(nn.Module):
    """
    专门用于子结构级特征的投影头。
    
    处理子结构特征的空间维度，并将其
    投影到适合对比学习的空间。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 64,
        pool_type: str = "mean"
    ):
        """
        参数:
            input_dim: 子结构特征的通道维度
            hidden_dim: 隐藏层维度
            output_dim: 输出投影维度
            pool_type: 空间池化类型 ("mean", "max", "adaptive")
        """
        super().__init__()
        
        self.pool_type = pool_type
        
        if pool_type == "adaptive":
            self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        投影子结构特征。
        
        参数:
            x: 子结构特征 [B, C, L] 或 [B, K, C, L]
            
        返回:
            投影后的特征 [B, D] 或 [B, K, D]
        """
        # Handle batched substructures
        if x.dim() == 4:
            B, K, C, L = x.shape
            x = x.reshape(B * K, C, L)
            squeezed = True
        else:
            squeezed = False
            B = x.shape[0]
        
        # Spatial pooling
        if self.pool_type == "mean":
            x = x.mean(dim=-1)  # [B*K, C]
        elif self.pool_type == "max":
            x = x.max(dim=-1)[0]  # [B*K, C]
        elif self.pool_type == "adaptive":
            x = self.pool(x).squeeze(-1)  # [B*K, C]
        
        # MLP projection
        x = self.mlp(x)
        
        if squeezed:
            x = x.reshape(B, K, -1)
        
        return x
