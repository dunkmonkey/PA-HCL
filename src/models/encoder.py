"""
PA-HCL 的编码器架构。

此模块提供心音表示的主要编码器架构：
- CNNBackbone: 用于局部特征提取的多尺度一维卷积神经网络
- CNNMambaEncoder: 用于局部 + 全局建模的 CNN + Mamba 混合模型
- CNNTransformerEncoder: CNN + Transformer 替代方案

编码器旨在：
1. 提取多尺度局部特征（通过 CNN）
2. 建模长程依赖关系（通过 Mamba/Transformer）
3. 支持返回中间特征以进行子结构级对比学习

作者: PA-HCL 团队
"""

from typing import Dict, List, Optional, Tuple, Union

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import MambaEncoder, get_mamba_encoder, get_mamba_encoder_v2, DropPath
from .attention import get_attention_block, ECABlock


# ============== CNN 主干 ==============

class ConvBlock(nn.Module):
    """
    带有 BatchNorm 和激活函数的基本一维卷积块。
    
    结构: Conv1D -> BatchNorm -> Activation -> Dropout -> [可选 Attention]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 2,
        padding: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        attention_type: str = "none"
    ):
        """
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 卷积步幅（用于下采样）
            padding: 填充大小（如为 None 则自动计算）
            dropout: Dropout 率
            activation: 激活函数 ("relu", "gelu", "silu")
            attention_type: 注意力类型 ("none", "eca", "se", "cbam")
        """
        super().__init__()
        
        # 当 stride=1 时自动计算填充以实现 'same' 输出
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # BatchNorm 处理偏置
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # 激活函数
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "silu":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # 注意力模块
        self.attention = get_attention_block(attention_type, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入张量 [B, C_in, L]
            
        返回:
            输出张量 [B, C_out, L']
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.attention(x)
        return x


class ResidualConvBlock(nn.Module):
    """
    残差一维卷积块。
    
    结构:
        x -> Conv -> BN -> Act -> Conv -> BN -> DropPath -> + -> Act -> [可选 Attention] -> output
        |_______________________________________________|
                    (1x1 conv if needed)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        attention_type: str = "none"
    ):
        """
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 下采样步幅
            dropout: Dropout 率
            drop_path: DropPath 率（随机深度）
            attention_type: 注意力类型 ("none", "eca", "se", "cbam")
        """
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size, stride=1,
            padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # 捷径连接 (Shortcut connection)
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        # 注意力模块
        self.attention = get_attention_block(attention_type, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入张量 [B, C_in, L]
            
        返回:
            输出张量 [B, C_out, L']
        """
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.drop_path(out)
        out = out + identity
        out = self.act(out)
        out = self.attention(out)
        
        return out


class MultiScaleConvBlock(nn.Module):
    """
    多尺度并行卷积块，捕获不同时间尺度的特征。
    
    结构:
        x -> [Conv_k1, Conv_k2, Conv_k3, ...] -> Concat -> 1x1 Conv -> BN -> Act -> Output
        |___________________________________________________________|
                              (shortcut if needed)
    
    不同大小的卷积核并行处理输入，捕获：
    - 小卷积核 (3): 高频局部特征（如心脏杂音的高频成分）
    - 中卷积核 (7): 中频模式（如 S1/S2 心音）
    - 大卷积核 (15): 低频趋势（如心动周期整体形态）
    
    最终通过 1x1 卷积融合多尺度特征。
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = [3, 7, 15],
        stride: int = 2,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        attention_type: str = "none"
    ):
        """
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_sizes: 并行卷积核大小列表
            stride: 下采样步幅
            dropout: Dropout 率
            drop_path: DropPath 率（随机深度）
            attention_type: 注意力类型 ("none", "eca", "se", "cbam")
        """
        super().__init__()
        
        self.num_scales = len(kernel_sizes)
        
        # 每个分支的通道数（平分 out_channels）
        branch_channels = out_channels // self.num_scales
        # 确保总通道数正确（处理不能整除的情况）
        self.branch_channels = [branch_channels] * (self.num_scales - 1)
        self.branch_channels.append(out_channels - branch_channels * (self.num_scales - 1))
        
        # 并行卷积分支
        self.branches = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            padding = k // 2
            self.branches.append(nn.Sequential(
                nn.Conv1d(
                    in_channels, self.branch_channels[i],
                    kernel_size=k, stride=stride,
                    padding=padding, bias=False
                ),
                nn.BatchNorm1d(self.branch_channels[i]),
                nn.GELU()
            ))
        
        # 1x1 融合卷积
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # 捷径连接
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        # 注意力模块
        self.attention = get_attention_block(attention_type, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入张量 [B, C_in, L]
            
        返回:
            输出张量 [B, C_out, L']
        """
        identity = self.shortcut(x)
        
        # 多尺度并行卷积
        branch_outputs = [branch(x) for branch in self.branches]
        
        # 拼接多尺度特征
        out = torch.cat(branch_outputs, dim=1)
        
        # 1x1 融合
        out = self.fusion(out)
        out = self.dropout(out)
        
        # 残差连接 + DropPath
        out = self.drop_path(out)
        out = out + identity
        out = self.act(out)
        out = self.attention(out)
        
        return out


class CNNBackbone(nn.Module):
    """
    用于心音特征提取的多尺度一维 CNN 主干网络。
    
    此主干网络在对输入信号进行逐步下采样的同时
    增加特征通道数。它旨在捕获 PCG 信号中的
    多尺度时间模式。
    
    架构:
        输入 [B, 1, L] 
        -> 随通道增加和下采样的卷积层（可选注意力）
        -> 输出 [B, C_out, L']
    
    中间层输出可用于子结构级对比学习。
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        channels: List[int] = [32, 64, 128, 256],
        kernel_sizes: List[int] = [7, 5, 5, 3],
        strides: List[int] = [2, 2, 2, 2],
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        use_residual: bool = True,
        attention_type: str = "none",
        use_multiscale: bool = False,
        multiscale_kernel_sizes: List[int] = [3, 7, 15]
    ):
        """
        参数:
            in_channels: 输入通道数（原始音频为 1）
            channels: 每层的输出通道列表
            kernel_sizes: 每层的卷积核大小
            strides: 每层的步幅（决定下采样）
            dropout: Dropout 率
            drop_path_rate: 最大 DropPath 率（线性递增）
            use_residual: 是否使用残差连接
            attention_type: 注意力类型 ("none", "eca", "se", "cbam")
            use_multiscale: 是否使用多尺度并行卷积
            multiscale_kernel_sizes: 多尺度卷积核大小列表
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.num_layers = len(channels)
        self.use_multiscale = use_multiscale
        
        # 计算每层的 drop_path 率（线性递增）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(channels))]
        
        # 构建卷积层
        layers = []
        in_ch = in_channels
        
        for i, (out_ch, k, s) in enumerate(zip(channels, kernel_sizes, strides)):
            if use_multiscale and i > 0:
                # 从第二层开始使用多尺度卷积（第一层保留常规卷积做初始特征提取）
                layers.append(MultiScaleConvBlock(
                    in_ch, out_ch,
                    kernel_sizes=multiscale_kernel_sizes,
                    stride=s,
                    dropout=dropout,
                    drop_path=dpr[i],
                    attention_type=attention_type
                ))
            elif use_residual and i > 0:
                layers.append(ResidualConvBlock(
                    in_ch, out_ch,
                    kernel_size=k,
                    stride=s,
                    dropout=dropout,
                    drop_path=dpr[i],
                    attention_type=attention_type
                ))
            else:
                layers.append(ConvBlock(
                    in_ch, out_ch,
                    kernel_size=k,
                    stride=s,
                    dropout=dropout,
                    attention_type=attention_type
                ))
            in_ch = out_ch
        
        self.layers = nn.ModuleList(layers)
        self.out_channels = channels[-1]
        
        # 计算下采样因子
        self.downsample_factor = 1
        for s in strides:
            self.downsample_factor *= s
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        CNN 主干网络的前向传播。
        
        参数:
            x: 输入张量 [B, 1, L] 或 [B, L]
            return_intermediate: 如果为 True，也返回中间层输出
            
        返回:
            如果 return_intermediate:
                元组 (final_output, [intermediate_outputs])
            否则:
                最终输出张量 [B, C, L']
        """
        # 处理没有通道维度的输入
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, L] -> [B, 1, L]
        
        intermediates = []
        
        for layer in self.layers:
            x = layer(x)
            if return_intermediate:
                intermediates.append(x)
        
        if return_intermediate:
            return x, intermediates
        return x
    
    def get_output_length(self, input_length: int) -> int:
        """
        根据输入长度计算输出序列长度。
        
        参数:
            input_length: 输入序列长度
            
        返回:
            所有下采样后的输出序列长度
        """
        length = input_length
        for stride in [2, 2, 2, 2]:  # 默认步幅
            length = (length + stride - 1) // stride
        return length


# ============== CNN + Mamba 编码器 ==============

class CNNMambaEncoder(nn.Module):
    """
    用于心音表示的 CNN-Mamba 混合编码器。
    
    此架构结合了：
    1. 用于局部多尺度特征提取的 CNN 主干网络（可选注意力）
    2. 用于高效长程依赖建模的 Mamba (SSM)（支持双向）
    
    设计基本原理：
    - CNN 捕获具有平移等变性的局部模式（S1, S2, 杂音）
    - Mamba 高效地建模跨心动周期的随时间依赖关系
    - O(n) 的线性复杂度使其适用于长 PCG 信号
    
    新增特性：
    - 通道注意力 (ECA/SE/CBAM)
    - DropPath 正则化（随机深度）
    - 双向 Mamba（可选）
    """
    
    def __init__(
        self,
        # CNN 设置
        in_channels: int = 1,
        cnn_channels: List[int] = [32, 64, 128, 256],
        cnn_kernel_sizes: List[int] = [7, 5, 5, 3],
        cnn_strides: List[int] = [2, 2, 2, 2],
        cnn_dropout: float = 0.1,
        # Mamba 设置
        mamba_d_model: int = 256,
        mamba_n_layers: int = 4,
        mamba_d_state: int = 16,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.1,
        # 输出设置
        pool_type: str = "mean",
        # 新增：正则化和注意力
        drop_path_rate: float = 0.0,
        attention_type: str = "none",
        use_bidirectional: bool = False,
        bidirectional_fusion: str = "add",
        # 新增：多尺度卷积
        use_multiscale: bool = False,
        multiscale_kernel_sizes: List[int] = [3, 7, 15]
    ):
        """
        参数:
            in_channels: 输入通道数（原始音频为 1）
            cnn_channels: CNN 层通道大小
            cnn_kernel_sizes: CNN 卷积核大小
            cnn_strides: CNN 步幅
            cnn_dropout: CNN dropout 率
            mamba_d_model: Mamba 模型维度
            mamba_n_layers: Mamba 层数
            mamba_d_state: Mamba 状态维度
            mamba_expand: Mamba 扩展因子
            mamba_dropout: Mamba dropout 率
            pool_type: 最终输出的池化类型 ("mean", "max", "cls")
            drop_path_rate: DropPath 率（随机深度正则化）
            attention_type: 通道注意力类型 ("none", "eca", "se", "cbam")
            use_bidirectional: 是否使用双向 Mamba
            bidirectional_fusion: 双向融合方式 ("add", "concat", "gate")
            use_multiscale: 是否使用多尺度并行卷积
            multiscale_kernel_sizes: 多尺度卷积核大小列表
        """
        super().__init__()
        
        self.pool_type = pool_type
        
        # CNN backbone (支持注意力、DropPath 和多尺度卷积)
        self.cnn = CNNBackbone(
            in_channels=in_channels,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            strides=cnn_strides,
            dropout=cnn_dropout,
            drop_path_rate=drop_path_rate / 2,  # CNN 使用一半的 drop_path
            attention_type=attention_type,
            use_multiscale=use_multiscale,
            multiscale_kernel_sizes=multiscale_kernel_sizes
        )
        
        # 如果需要，投影到 Mamba 维度
        cnn_out_dim = cnn_channels[-1]
        if cnn_out_dim != mamba_d_model:
            self.proj = nn.Linear(cnn_out_dim, mamba_d_model)
        else:
            self.proj = nn.Identity()
        
        # Mamba 编码器 (使用 v2 版本支持更多选项)
        self.mamba = get_mamba_encoder_v2(
            d_model=mamba_d_model,
            n_layers=mamba_n_layers,
            d_state=mamba_d_state,
            expand=mamba_expand,
            dropout=mamba_dropout,
            drop_path_rate=drop_path_rate,
            bidirectional=use_bidirectional,
            fusion=bidirectional_fusion,
            use_official=False  # 为了兼容性使用纯 PyTorch
        )
        
        self.out_dim = mamba_d_model
        
        # 用于池化的可选 CLS 令牌
        if pool_type == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, mamba_d_model))
    
    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
        return_intermediate: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        通过编码器的前向传播。
        
        参数:
            x: 输入张量 [B, 1, L] 或 [B, L]
            return_sequence: 如果为 True，返回完整序列而不是池化后的结果
            return_intermediate: 如果为 True，返回中间 CNN 特征
            
        返回:
            如果 return_intermediate:
                包含 'cycle_features', 'sub_features', 'sequence' 的字典
            否则:
                池化输出 [B, D] 或序列 [B, L', D]
        """
        batch_size = x.shape[0]
        
        # 处理输入维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, L] -> [B, 1, L]
        
        # CNN 特征提取
        if return_intermediate:
            cnn_out, intermediates = self.cnn(x, return_intermediate=True)
        else:
            cnn_out = self.cnn(x)
        
        # 为 Mamba 重塑形状: [B, C, L'] -> [B, L', C]
        cnn_out = cnn_out.transpose(1, 2)
        
        # 投影到 Mamba 维度
        cnn_out = self.proj(cnn_out)
        
        # 如果使用 cls 池化，则添加 CLS 令牌
        if self.pool_type == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            cnn_out = torch.cat([cls_tokens, cnn_out], dim=1)
        
        # Mamba encoding
        mamba_out = self.mamba(cnn_out)
        
        # Pooling for final representation
        if self.pool_type == "cls":
            pooled = mamba_out[:, 0]  # CLS token
            sequence = mamba_out[:, 1:]
        elif self.pool_type == "mean":
            pooled = mamba_out.mean(dim=1)
            sequence = mamba_out
        elif self.pool_type == "max":
            pooled = mamba_out.max(dim=1)[0]
            sequence = mamba_out
        else:
            raise ValueError(f"Unknown pool_type: {self.pool_type}")
        
        if return_intermediate:
            # Return both cycle-level and substructure-level features
            # For sub-level, use the second-to-last CNN layer
            sub_features = intermediates[-2] if len(intermediates) >= 2 else intermediates[-1]
            return {
                "cycle_features": pooled,
                "sub_features": sub_features,  # [B, C, L']
                "sequence": sequence
            }
        
        if return_sequence:
            return sequence
        
        return pooled
    
    def get_sub_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取适合子结构级对比学习的特征。
        
        这些是在 Mamba 处理之前的中间 CNN 特征，
        它们更好地保留了局部空间信息。
        
        参数:
            x: 输入张量 [B, 1, L]
            
        返回:
            子结构特征 [B, C, L']
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        _, intermediates = self.cnn(x, return_intermediate=True)
        # Use second-to-last layer (before final downsampling)
        return intermediates[-2]


# ============== CNN + Transformer 编码器 (替代方案) ==============

class PositionalEncoding(nn.Module):
    """Transformer 的正弦位置编码。"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入张量 [B, L, D]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# Import math for PositionalEncoding
import math


class CNNTransformerEncoder(nn.Module):
    """
    CNN + Transformer 编码器作为 CNN + Mamba 的替代方案。
    
    用于比较 Mamba 与 Transformer 的消融研究。
    注意：Transformer 具有 O(n^2) 的复杂度，而 Mamba 为 O(n)。
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        cnn_channels: List[int] = [32, 64, 128, 256],
        cnn_kernel_sizes: List[int] = [7, 5, 5, 3],
        cnn_strides: List[int] = [2, 2, 2, 2],
        cnn_dropout: float = 0.1,
        transformer_d_model: int = 256,
        transformer_n_heads: int = 8,
        transformer_n_layers: int = 4,
        transformer_dim_ff: int = 1024,
        transformer_dropout: float = 0.1,
        pool_type: str = "mean"
    ):
        """
        参数:
            in_channels: 输入通道数
            cnn_channels: CNN 通道大小
            cnn_kernel_sizes: CNN 卷积核大小
            cnn_strides: CNN 步幅
            cnn_dropout: CNN dropout
            transformer_d_model: Transformer 模型维度
            transformer_n_heads: 注意力头数量
            transformer_n_layers: Transformer 层数
            transformer_dim_ff: 前馈层维度
            transformer_dropout: Transformer dropout
            pool_type: 池化类型
        """
        super().__init__()
        
        self.pool_type = pool_type
        
        # CNN backbone
        self.cnn = CNNBackbone(
            in_channels=in_channels,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            strides=cnn_strides,
            dropout=cnn_dropout
        )
        
        # Projection
        cnn_out_dim = cnn_channels[-1]
        if cnn_out_dim != transformer_d_model:
            self.proj = nn.Linear(cnn_out_dim, transformer_d_model)
        else:
            self.proj = nn.Identity()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            transformer_d_model, dropout=transformer_dropout
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_n_heads,
            dim_feedforward=transformer_dim_ff,
            dropout=transformer_dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_n_layers
        )
        
        self.out_dim = transformer_d_model
        
        if pool_type == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, transformer_d_model))
    
    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False
    ) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            x: 输入 [B, 1, L] 或 [B, L]
            return_sequence: 返回完整序列或池化结果
            
        返回:
            输出张量
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size = x.shape[0]
        
        # CNN
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)  # [B, L', C]
        cnn_out = self.proj(cnn_out)
        
        # Add CLS token
        if self.pool_type == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            cnn_out = torch.cat([cls_tokens, cnn_out], dim=1)
        
        # Positional encoding
        cnn_out = self.pos_encoder(cnn_out)
        
        # Transformer
        output = self.transformer(cnn_out)
        
        # Pooling
        if self.pool_type == "cls":
            pooled = output[:, 0]
        elif self.pool_type == "mean":
            if self.pool_type == "cls":
                pooled = output[:, 1:].mean(dim=1)
            else:
                pooled = output.mean(dim=1)
        elif self.pool_type == "max":
            pooled = output.max(dim=1)[0]
        
        if return_sequence:
            return output
        
        return pooled


# ============== SincNet 前端 ==============

class SincConv1d(nn.Module):
    """
    SincNet 卷积层：可学习带通滤波器。
    
    与标准卷积不同，SincNet 仅学习每个滤波器的上下截止频率，
    滤波器形状由 sinc 函数定义。这使得模型能够学习更有意义的
    频率特征，特别适合心音等频带结构明显的信号。
    
    参考:
        Ravanelli, M., & Bengio, Y. (2018). Speaker Recognition from Raw 
        Waveform with SincNet. SLT 2018.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 64,
        kernel_size: int = 251,
        stride: int = 2,
        padding: str = 'same',
        sample_rate: int = 5000,
        min_low_hz: float = 25.0,
        min_band_hz: float = 25.0
    ):
        """
        参数:
            in_channels: 输入通道数（PCG 通常为 1）
            out_channels: 滤波器数量
            kernel_size: 滤波器长度（应为奇数）
            stride: 步幅
            padding: 填充方式
            sample_rate: 采样率（Hz）
            min_low_hz: 最低截止频率下界
            min_band_hz: 最小带宽
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        
        # 计算填充
        if padding == 'same':
            self.padding = (kernel_size - 1) // 2
        else:
            self.padding = 0
        
        # 确保 kernel_size 为奇数
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        
        # 初始化可学习的截止频率
        # 使用 Mel-scale 分布初始化，覆盖心音关键频段 (25-400Hz)
        low_hz = 25.0
        high_hz = min(400.0, sample_rate / 2)
        
        # Mel-scale 初始化
        mel_low = self._hz_to_mel(low_hz)
        mel_high = self._hz_to_mel(high_hz)
        mel_points = np.linspace(mel_low, mel_high, out_channels + 1)
        hz_points = self._mel_to_hz(mel_points)
        
        # 初始化为升序的频带
        low_hz_init = hz_points[:-1]
        high_hz_init = hz_points[1:]
        
        # 可学习参数（实际学习的是归一化后的频率）
        self.low_hz_ = nn.Parameter(
            torch.FloatTensor(low_hz_init).view(-1, 1)
        )
        self.band_hz_ = nn.Parameter(
            torch.FloatTensor(high_hz_init - low_hz_init).view(-1, 1)
        )
        
        # 创建时间轴（用于 sinc 函数）
        n = torch.arange(0, kernel_size).float()
        self.register_buffer('window_', torch.hamming_window(kernel_size))
        self.register_buffer('n_', (n - (kernel_size - 1) / 2.0).view(1, -1) / sample_rate)
    
    def _hz_to_mel(self, hz):
        """将 Hz 转换为 Mel scale"""
        return 2595 * np.log10(1 + hz / 700)
    
    def _mel_to_hz(self, mel):
        """将 Mel scale 转换为 Hz"""
        return 700 * (10 ** (mel / 2595) - 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入信号 [B, 1, L]
        
        返回:
            滤波后的特征 [B, out_channels, L']
        """
        # 确保频率在有效范围内
        low_hz = self.min_low_hz + torch.abs(self.low_hz_)
        high_hz = torch.clamp(
            low_hz + self.min_band_hz + torch.abs(self.band_hz_),
            min=self.min_low_hz,
            max=self.sample_rate / 2
        )
        band = (high_hz - low_hz)[:, 0]
        
        # 计算归一化频率
        f_times_t_low = torch.matmul(low_hz, self.n_)   # [out_channels, kernel_size]
        f_times_t_high = torch.matmul(high_hz, self.n_) # [out_channels, kernel_size]
        
        # 带通滤波器 = high_pass - low_pass
        # sinc(x) = sin(x) / x，在 x=0 时值为 1
        eps = 1e-8
        low_pass = 2 * low_hz * torch.sinc(2 * f_times_t_low * low_hz)  
        high_pass = 2 * high_hz * torch.sinc(2 * f_times_t_high * high_hz)
        band_pass = high_pass - low_pass
        
        # 应用窗函数（Hamming）
        band_pass = band_pass * self.window_
        
        # 归一化
        band_pass = band_pass / (2 * band[:, None])
        
        # 添加输入通道维度 [out_channels, in_channels, kernel_size]
        filters = band_pass.unsqueeze(1)
        
        # 卷积
        out = F.conv1d(
            x,
            filters,
            stride=self.stride,
            padding=self.padding
        )
        
        # 应用绝对值（能量检测）
        out = torch.abs(out)
        
        return out


# ============== Attentive Statistics Pooling ==============

class AttentiveStatisticsPooling(nn.Module):
    """
    注意力统计池化 (Attentive Statistics Pooling, ASP)。
    
    通过可学习的注意力机制提取关键帧，并同时计算加权均值和标准差。
    相比简单的平均/最大池化，ASP 能够自适应地关注重要时刻。
    
    输出拼接均值和标准差: [B, T, D] -> [B, 2*D]
    
    参考:
        Okabe, K., et al. (2018). Attentive Statistics Pooling for Deep 
        Speaker Embedding. Interspeech 2018.
    """
    
    def __init__(self, input_dim: int, bottleneck_dim: Optional[int] = None):
        """
        参数:
            input_dim: 输入特征维度 D
            bottleneck_dim: 注意力网络的瓶颈维度（None 则使用 input_dim // 4）
        """
        super().__init__()
        
        self.input_dim = input_dim
        if bottleneck_dim is None:
            bottleneck_dim = max(input_dim // 4, 64)
        
        # 注意力网络: Linear -> Tanh -> Linear -> Softmax
        self.attention = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.Tanh(),
            nn.Linear(bottleneck_dim, 1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        参数:
            x: 输入序列 [B, T, D]
            mask: 可选的掩码 [B, T]，True 表示有效位置
        
        返回:
            统计特征 [B, 2*D]（拼接均值和标准差）
        """
        # 计算注意力权重
        attn_scores = self.attention(x)  # [B, T, 1]
        
        # 应用掩码（如果提供）
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        
        # Softmax 归一化
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, T, 1]
        
        # 加权均值
        mean = torch.sum(attn_weights * x, dim=1)  # [B, D]
        
        # 加权标准差
        # std = sqrt(E[(x - mean)^2])
        variance = torch.sum(attn_weights * (x - mean.unsqueeze(1)) ** 2, dim=1)
        std = torch.sqrt(variance + 1e-8)  # [B, D]
        
        # 拼接均值和标准差
        pooled = torch.cat([mean, std], dim=-1)  # [B, 2*D]
        
        return pooled


# ============== DepthwiseSeparable 卷积块 ==============

class DepthwiseSeparableConv1d(nn.Module):
    """
    深度可分离卷积：Depthwise + Pointwise。
    
    相比标准卷积大幅减少参数量和计算量，同时保持相近的表达能力。
    常用于轻量化网络设计。
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        use_groupnorm: bool = True,
        num_groups: int = 8
    ):
        super().__init__()
        
        # Depthwise 卷积（每个通道独立卷积）
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )
        
        # Pointwise 卷积（1x1 卷积，通道间交互）
        self.pointwise = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False
        )
        
        # 归一化
        if use_groupnorm:
            self.norm1 = nn.GroupNorm(num_groups, in_channels)
            self.norm2 = nn.GroupNorm(num_groups, out_channels)
        else:
            self.norm1 = nn.BatchNorm1d(in_channels)
            self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Depthwise
        x = self.depthwise(x)
        x = self.norm1(x)
        x = self.act(x)
        
        # Pointwise
        x = self.pointwise(x)
        x = self.norm2(x)
        x = self.act(x)
        
        return x


# ============== SincNet + ECA + ASP + Mamba 编码器 ==============

class SincNetECAMambaEncoder(nn.Module):
    """
    优化的心音编码器：SincNet + ECA + ASP + 轻量 Mamba。
    
    架构设计针对心音信号的特点：
    - Stage 0: SincNet 前端学习频带特征
    - Stage 1-2: 轻量卷积 + ECA 注意力提取局部模式
    - Stage 3: Token 投影准备序列建模
    - Stage 4: 轻量 Mamba 建模长程依赖
    - Stage 5: ASP 池化提取周期级特征 + 子结构切分
    
    接口保持与 CNNMambaEncoder 一致，可直接替换。
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        sinc_out_channels: int = 64,
        sinc_kernel_size: int = 251,
        sinc_stride: int = 2,
        stage1_channels: List[int] = [64, 96],
        stage2_channels: List[int] = [96, 128],
        mamba_d_model: int = 128,
        mamba_n_layers: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.1,
        use_groupnorm: bool = True,
        num_groups: int = 8,
        eca_kernel_size: int = 3,
        cycle_output_dim: int = 192,
        num_substructures: int = 4,
        pool_type: str = "asp",
        sample_rate: int = 5000,
        **kwargs  # 兼容旧参数
    ):
        """
        参数:
            in_channels: 输入通道数
            sinc_out_channels: SincNet 输出通道数
            sinc_kernel_size: SincNet 卷积核大小
            sinc_stride: SincNet 步幅
            stage1_channels: Stage 1 各层通道数 [64, 96]
            stage2_channels: Stage 2 各层通道数 [96, 128]
            mamba_d_model: Mamba 隐藏维度
            mamba_n_layers: Mamba 层数
            mamba_d_state: Mamba 状态维度
            mamba_d_conv: Mamba 卷积核大小
            mamba_expand: Mamba 扩展因子
            mamba_dropout: Mamba dropout
            use_groupnorm: 是否使用 GroupNorm（推荐）
            num_groups: GroupNorm 分组数
            eca_kernel_size: ECA 卷积核大小
            cycle_output_dim: 周期级特征输出维度
            num_substructures: 子结构数量
            pool_type: 池化类型（"asp" 或 "mean"）
            sample_rate: 采样率
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.mamba_d_model = mamba_d_model
        self.cycle_output_dim = cycle_output_dim
        self.num_substructures = num_substructures
        self.pool_type = pool_type
        self.out_dim = cycle_output_dim  # 接口兼容性
        
        # ============== Stage 0: SincNet 前端 ==============
        self.sinc_conv = SincConv1d(
            in_channels=in_channels,
            out_channels=sinc_out_channels,
            kernel_size=sinc_kernel_size,
            stride=sinc_stride,
            padding='same',
            sample_rate=sample_rate
        )
        
        # Normalization + Activation + AvgPool
        if use_groupnorm:
            self.sinc_norm = nn.GroupNorm(num_groups, sinc_out_channels)
        else:
            self.sinc_norm = nn.BatchNorm1d(sinc_out_channels)
        
        self.sinc_act = nn.SiLU()
        self.sinc_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # ============== Stage 1: Local CNN + ECA ==============
        # Block 1: 64 -> 96
        self.stage1_block1 = DepthwiseSeparableConv1d(
            in_channels=stage1_channels[0],
            out_channels=stage1_channels[1],
            kernel_size=7,
            stride=1,
            padding=3,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups
        )
        self.stage1_eca1 = ECABlock(stage1_channels[1], kernel_size=eca_kernel_size)
        
        # Block 2: 96 -> 96 (with downsampling)
        self.stage1_block2 = DepthwiseSeparableConv1d(
            in_channels=stage1_channels[1],
            out_channels=stage1_channels[1],
            kernel_size=5,
            stride=2,
            padding=2,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups
        )
        self.stage1_eca2 = ECABlock(stage1_channels[1], kernel_size=eca_kernel_size)
        
        # ============== Stage 2: Multi-scale refinement ==============
        # Block 3: 96 -> 128 (dilation=1)
        self.stage2_block1 = DepthwiseSeparableConv1d(
            in_channels=stage2_channels[0],
            out_channels=stage2_channels[1],
            kernel_size=5,
            stride=1,
            padding=2,
            dilation=1,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups
        )
        self.stage2_eca1 = ECABlock(stage2_channels[1], kernel_size=eca_kernel_size)
        
        # Block 4: 128 -> 128 (dilation=2)
        self.stage2_block2 = DepthwiseSeparableConv1d(
            in_channels=stage2_channels[1],
            out_channels=stage2_channels[1],
            kernel_size=5,
            stride=1,
            padding=4,
            dilation=2,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups
        )
        self.stage2_eca2 = ECABlock(stage2_channels[1], kernel_size=eca_kernel_size)
        
        # ============== Stage 3: Token projection ==============
        self.token_proj = nn.Conv1d(stage2_channels[1], mamba_d_model, kernel_size=1)
        
        # ============== Stage 4: Lightweight Mamba ==============
        self.mamba = get_mamba_encoder_v2(
            d_model=mamba_d_model,
            n_layers=mamba_n_layers,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=mamba_dropout,
            bidirectional=False,  # 单向即可，减少计算量
            use_official=False
        )
        
        # ============== Stage 5: Pooling heads ==============
        # 周期级头：ASP -> Linear
        if pool_type == "asp":
            self.asp = AttentiveStatisticsPooling(mamba_d_model)
            asp_out_dim = 2 * mamba_d_model  # ASP 输出均值+方差
        else:
            self.asp = None
            asp_out_dim = mamba_d_model
        
        self.cycle_head = nn.Linear(asp_out_dim, cycle_output_dim)
        
        # 子结构级头：共享 Linear
        self.sub_head = nn.Linear(mamba_d_model, mamba_d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
        return_intermediate: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播。
        
        参数:
            x: 输入信号 [B, 1, L]
            return_sequence: 是否返回完整序列
            return_intermediate: 是否返回中间特征（用于子结构对比学习）
        
        返回:
            - 默认: 周期级特征 [B, cycle_output_dim]
            - return_sequence=True: Mamba 输出序列 [B, T, mamba_d_model]
            - return_intermediate=True: 包含 cycle_features 和 sub_features 的字典
        """
        B = x.shape[0]
        
        # Stage 0: SincNet 前端
        x = self.sinc_conv(x)        # [B, 64, L/2]
        x = self.sinc_norm(x)
        x = self.sinc_act(x)
        x = self.sinc_pool(x)        # [B, 64, L/4]
        
        # Stage 1: Local CNN + ECA
        x = self.stage1_block1(x)    # [B, 96, L/4]
        x = self.stage1_eca1(x)
        x = self.stage1_block2(x)    # [B, 96, L/8]
        x = self.stage1_eca2(x)
        
        # Stage 2: Multi-scale refinement
        x = self.stage2_block1(x)    # [B, 128, L/8]
        x = self.stage2_eca1(x)
        x = self.stage2_block2(x)    # [B, 128, L/8]
        x = self.stage2_eca2(x)
        
        # 保存用于子结构对比学习的特征
        sub_features = x  # [B, 128, T']
        
        # Stage 3: Token projection
        x = self.token_proj(x)       # [B, 128, T']
        x = x.transpose(1, 2)        # [B, T', 128]
        
        # Stage 4: Mamba
        x = self.mamba(x)            # [B, T, 128]
        sequence = x
        
        if return_sequence:
            return sequence
        
        # Stage 5: Pooling
        # 周期级特征
        if self.pool_type == "asp" and self.asp is not None:
            cycle_pooled = self.asp(x)  # [B, 256]
        else:
            cycle_pooled = x.mean(dim=1)  # [B, 128]
        
        cycle_features = self.cycle_head(cycle_pooled)  # [B, 192]
        
        if return_intermediate:
            return {
                "cycle_features": cycle_features,
                "sub_features": sub_features,
                "sequence": sequence
            }
        
        return cycle_features
    
    def get_sub_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取子结构级特征（用于子结构对比学习）。
        
        参数:
            x: 输入信号 [B, 1, L]
        
        返回:
            子结构特征 [B, 128, T']（Stage 2 输出）
        """
        # Stage 0
        x = self.sinc_conv(x)
        x = self.sinc_norm(x)
        x = self.sinc_act(x)
        x = self.sinc_pool(x)
        
        # Stage 1
        x = self.stage1_block1(x)
        x = self.stage1_eca1(x)
        x = self.stage1_block2(x)
        x = self.stage1_eca2(x)
        
        # Stage 2
        x = self.stage2_block1(x)
        x = self.stage2_eca1(x)
        x = self.stage2_block2(x)
        x = self.stage2_eca2(x)
        
        return x  # [B, 128, T']


# ============== 工厂函数 ==============

def build_encoder(
    encoder_type: str = "cnn_mamba",
    **kwargs
) -> nn.Module:
    """
    基于类型构建编码器的工厂函数。
    
    参数:
        encoder_type: "cnn_only", "cnn_mamba", "cnn_transformer", "sincnet_eca_mamba" 之一
        **kwargs: 传递给编码器构造函数的其他参数
        
    返回:
        编码器模块
    """
    if encoder_type == "cnn_only":
        return CNNBackbone(**kwargs)
    elif encoder_type == "cnn_mamba":
        return CNNMambaEncoder(**kwargs)
    elif encoder_type == "cnn_transformer":
        return CNNTransformerEncoder(**kwargs)
    elif encoder_type == "sincnet_eca_mamba":
        return SincNetECAMambaEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
