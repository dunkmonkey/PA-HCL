"""
PA-HCL 的通道注意力模块。

此模块提供：
- ECABlock: 高效通道注意力（无全连接层，更轻量）
- SEBlock: 标准 Squeeze-and-Excitation 注意力

参考:
    - ECA-Net: Efficient Channel Attention (Wang et al., 2020)
    - SE-Net: Squeeze-and-Excitation Networks (Hu et al., 2018)

作者: PA-HCL 团队
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ECABlock(nn.Module):
    """
    高效通道注意力模块 (Efficient Channel Attention, ECA)。
    
    ECA 使用 1D 卷积替代全连接层，实现跨通道交互，
    参数量极少但效果优异。适合对效率敏感的场景。
    
    结构:
        输入 [B, C, L]
        -> 全局平均池化 -> [B, C, 1]
        -> 1D 卷积 (kernel_size=k) -> [B, C, 1]
        -> Sigmoid -> 通道权重
        -> 与输入相乘 -> 输出
    """
    
    def __init__(
        self,
        channels: int,
        kernel_size: Optional[int] = None,
        gamma: int = 2,
        beta: int = 1
    ):
        """
        参数:
            channels: 输入通道数
            kernel_size: 1D 卷积核大小（None 则自适应计算）
            gamma: 自适应 kernel_size 计算参数
            beta: 自适应 kernel_size 计算参数
        """
        super().__init__()
        
        # 自适应计算 kernel_size
        # k = |log2(C) / gamma + beta / gamma|_odd
        if kernel_size is None:
            t = int(abs(math.log2(channels) / gamma + beta / gamma))
            kernel_size = t if t % 2 else t + 1  # 确保为奇数
            kernel_size = max(3, kernel_size)  # 最小为 3
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入张量 [B, C, L]
            
        返回:
            注意力加权后的张量 [B, C, L]
        """
        # 全局平均池化: [B, C, L] -> [B, C, 1]
        y = self.avg_pool(x)
        
        # 跨通道卷积: [B, C, 1] -> [B, 1, C] -> 卷积 -> [B, 1, C] -> [B, C, 1]
        y = y.transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y)         # [B, 1, C]
        y = y.transpose(-1, -2)  # [B, C, 1]
        
        # Sigmoid 激活
        y = self.sigmoid(y)
        
        # 通道加权
        return x * y


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 通道注意力模块。
    
    SE 模块通过全连接层学习通道间的依赖关系，
    效果稳定但参数量比 ECA 更大。
    
    结构:
        输入 [B, C, L]
        -> 全局平均池化 -> [B, C]
        -> FC (C -> C/r) -> ReLU -> FC (C/r -> C) -> Sigmoid
        -> 通道权重
        -> 与输入相乘 -> 输出
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        use_bias: bool = True
    ):
        """
        参数:
            channels: 输入通道数
            reduction: 降维比例（隐藏层维度 = channels // reduction）
            use_bias: 是否在全连接层使用偏置
        """
        super().__init__()
        
        hidden_dim = max(channels // reduction, 8)  # 至少 8 维
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=use_bias),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入张量 [B, C, L]
            
        返回:
            注意力加权后的张量 [B, C, L]
        """
        batch, channels, _ = x.shape
        
        # Squeeze: [B, C, L] -> [B, C]
        y = self.avg_pool(x).view(batch, channels)
        
        # Excitation: [B, C] -> [B, C]
        y = self.fc(y)
        
        # 重塑并加权: [B, C] -> [B, C, 1]
        y = y.view(batch, channels, 1)
        
        return x * y


class CBAMBlock(nn.Module):
    """
    卷积块注意力模块 (Convolutional Block Attention Module, CBAM)。
    
    结合通道注意力和空间注意力的混合注意力模块。
    注意：对于 1D 信号，空间注意力变为时间注意力。
    
    结构:
        输入 -> 通道注意力 -> 空间/时间注意力 -> 输出
    """
    
    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7
    ):
        """
        参数:
            channels: 输入通道数
            reduction: SE 模块降维比例
            kernel_size: 空间注意力卷积核大小
        """
        super().__init__()
        
        # 通道注意力 (使用 SE 风格)
        hidden_dim = max(channels // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels),
        )
        self.channel_gate_max = nn.Sequential(
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels),
        )
        
        # 空间/时间注意力
        self.spatial_gate = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入张量 [B, C, L]
            
        返回:
            注意力加权后的张量 [B, C, L]
        """
        # 通道注意力
        channel_avg = self.channel_gate(x)
        channel_max = self.channel_gate_max(x)
        channel_weight = torch.sigmoid(channel_avg + channel_max).unsqueeze(-1)
        x = x * channel_weight
        
        # 空间/时间注意力
        avg_out = x.mean(dim=1, keepdim=True)  # [B, 1, L]
        max_out = x.max(dim=1, keepdim=True)[0]  # [B, 1, L]
        spatial_input = torch.cat([avg_out, max_out], dim=1)  # [B, 2, L]
        spatial_weight = self.spatial_gate(spatial_input)  # [B, 1, L]
        x = x * spatial_weight
        
        return x


def get_attention_block(
    attention_type: str,
    channels: int,
    **kwargs
) -> nn.Module:
    """
    获取注意力模块的工厂函数。
    
    参数:
        attention_type: "eca", "se", "cbam", "none" 之一
        channels: 输入通道数
        **kwargs: 传递给注意力模块的其他参数
        
    返回:
        注意力模块（或 Identity）
    """
    if attention_type == "eca":
        return ECABlock(channels, **kwargs)
    elif attention_type == "se":
        return SEBlock(channels, **kwargs)
    elif attention_type == "cbam":
        return CBAMBlock(channels, **kwargs)
    elif attention_type == "none" or attention_type is None:
        return nn.Identity()
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
