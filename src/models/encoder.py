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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import MambaEncoder, get_mamba_encoder, get_mamba_encoder_v2, DropPath
from .attention import get_attention_block


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
        attention_type: str = "none"
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
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.channels = channels
        self.num_layers = len(channels)
        
        # 计算每层的 drop_path 率（线性递增）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(channels))]
        
        # 构建卷积层
        layers = []
        in_ch = in_channels
        
        for i, (out_ch, k, s) in enumerate(zip(channels, kernel_sizes, strides)):
            if use_residual and i > 0:
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
        bidirectional_fusion: str = "add"
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
        """
        super().__init__()
        
        self.pool_type = pool_type
        
        # CNN backbone (支持注意力和 DropPath)
        self.cnn = CNNBackbone(
            in_channels=in_channels,
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            strides=cnn_strides,
            dropout=cnn_dropout,
            drop_path_rate=drop_path_rate / 2,  # CNN 使用一半的 drop_path
            attention_type=attention_type
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


# ============== 工厂函数 ==============

def build_encoder(
    encoder_type: str = "cnn_mamba",
    **kwargs
) -> nn.Module:
    """
    基于类型构建编码器的工厂函数。
    
    参数:
        encoder_type: "cnn_only", "cnn_mamba", "cnn_transformer" 之一
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
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
