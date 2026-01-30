"""
PA-HCL 的 Mamba (选择性状态空间模型) 实现。

此模块为一维信号处理提供简化的 Mamba 实现。
要在生产环境中使用 CUDA 加速，请安装官方 mamba-ssm 包。

Mamba 架构提供：
- 线性复杂度 O(n) vs Transformer 的 O(n^2)
- 高效处理长序列（对 PCG 信号很重要）
- 用于上下文感知处理的选择性状态空间机制

参考:
    Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with 
    Selective State Spaces. arXiv preprint arXiv:2312.00752.

作者: PA-HCL 团队
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class DropPath(nn.Module):
    """
    随机深度 (Stochastic Depth / Drop Path) 正则化。
    
    在训练时随机丢弃整个残差分支，测试时不丢弃。
    这是一种有效的正则化方法，可以缓解深层网络的过拟合。
    
    参考:
        Huang et al. (2016). Deep Networks with Stochastic Depth.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        """
        参数:
            drop_prob: 丢弃概率，0 表示不丢弃
        """
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 残差分支的输出
            
        返回:
            可能被缩放或置零的张量
        """
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        # 保持概率
        keep_prob = 1 - self.drop_prob
        
        # 生成随机掩码 (按样本独立)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # 二值化
        
        # 缩放以保持期望值不变
        output = x.div(keep_prob) * random_tensor
        return output
    
    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class RMSNorm(nn.Module):
    """
    均方根层归一化 (Root Mean Square Layer Normalization)。
    
    比 LayerNorm 更高效，因为它不计算均值，
    并且已证明在 SSM 架构中效果良好。
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        参数:
            d_model: 模型维度
            eps: 用于数值稳定性的常数
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入张量，形状为 [B, L, D]
            
        返回:
            相同形状的归一化张量
        """
        # 计算均方根 (RMS)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化并缩放
        return x / rms * self.weight


class SelectiveSSM(nn.Module):
    """
    选择性状态空间模型 (Selective State Space Model, S6) - Mamba 的核心。
    
    这实现了选择性扫描机制，其中状态空间参数 (A, B, C) 是输入相关的，
    允许模型选择性地记住或忘记信息。
    
    离散化状态空间方程为：
        h_t = A_bar * h_{t-1} + B_bar * x_t
        y_t = C_t * h_t
    
    其中 A_bar, B_bar 是连续参数的离散化版本。
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        """
        参数:
            d_model: 输入/输出维度
            d_state: SSM 状态维度 (论文中的 N)
            d_conv: 局部卷积宽度
            expand: 内部维度的扩展因子
            dt_min: dt 初始化的最小值
            dt_max: dt 初始化的最大值
            dt_init: dt 初始化方法 ("random" 或 "constant")
            dt_scale: dt 的缩放因子
            bias: 是否在线性层中使用偏置
            conv_bias: 是否在卷积层中使用偏置
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # 输入投影：将输入投影到 (z, x, B, C, dt)
        # z: 门控, x: SSM 输入, B/C: 输入相关的 SSM 参数, dt: 步长
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        
        # 局部卷积用于捕获局部模式
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # 深度可分离卷积
            bias=conv_bias
        )
        
        # SSM 参数投影
        # x_proj: 将 x 投影到 (dt, B, C)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        
        # dt 投影 (从秩-1 到全秩)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # 初始化 dt 偏置以确保 dt 在 [dt_min, dt_max] 范围内
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_min)
        # softplus 的逆运算以获取偏置
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # A 参数 (状态转移矩阵对角线, 学习对数值)
        # 将 A 初始化为负值以保持稳定性
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner
        )
        self.A_log = nn.Parameter(torch.log(A))
        
        # D 参数 (跳跃连接)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        选择性 SSM 的前向传播。
        
        参数:
            x: 输入张量，形状为 [B, L, D]
            
        返回:
            输出张量，形状为 [B, L, D]
        """
        batch, seq_len, _ = x.shape
        
        # 输入投影
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x, z = xz.chunk(2, dim=-1)  # 每个 [B, L, d_inner]
        
        # 局部卷积 (用于短程依赖)
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :seq_len]  # 因果卷积, 裁剪到原始长度
        x = rearrange(x, "b d l -> b l d")
        
        # 应用 SiLU 激活
        x = F.silu(x)
        
        # SSM 计算
        y = self.ssm(x)
        
        # 使用 z 进行门控
        z = F.silu(z)
        output = y * z
        
        # 输出投影
        output = self.out_proj(output)
        
        return output
    
    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        """
        选择性状态空间模型计算。
        
        这是一个简化的顺序实现。
        对于 CUDA 优化的并行扫描，请使用官方的 mamba-ssm 包。
        
        参数:
            x: 输入张量 [B, L, d_inner]
            
        返回:
            输出张量 [B, L, d_inner]
        """
        batch, seq_len, d_inner = x.shape
        
        # 获取 A (负值以保持稳定性)
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # 投影 x 以获取 dt, B, C
        x_dbl = self.x_proj(x)  # [B, L, d_state*2 + 1]
        dt, B, C = torch.split(
            x_dbl, [1, self.d_state, self.d_state], dim=-1
        )
        
        # 通过投影和 softplus 处理 dt
        dt = self.dt_proj(dt)  # [B, L, d_inner]
        dt = F.softplus(dt)  # 确保为正值
        
        # 使用 dt 离散化 A 和 B
        # A_bar = exp(dt * A), B_bar = dt * B (简化的零阶保持)
        # 为数值稳定性，我们在对数空间中计算 A
        
        # 初始化状态
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        for t in range(seq_len):
            # 获取当前时间步的参数
            dt_t = dt[:, t, :]  # [B, d_inner]
            B_t = B[:, t, :]    # [B, d_state]
            C_t = C[:, t, :]    # [B, d_state]
            x_t = x[:, t, :]    # [B, d_inner]
            
            # 离散化: A_bar = exp(dt * A)
            # A 是 [d_inner, d_state], dt_t 是 [B, d_inner]
            A_bar = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # [B, d_inner, d_state]
            
            # B_bar = dt * B (简化版)
            B_bar = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # [B, d_inner, d_state]
            
            # 状态更新: h = A_bar * h + B_bar * x
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)  # [B, d_inner, d_state]
            
            # 输出: y = C * h + D * x
            y_t = torch.einsum("bdn,bn->bd", h, C_t)  # [B, d_inner]
            y_t = y_t + self.D * x_t
            
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)  # [B, L, d_inner]
        return y


class MambaBlock(nn.Module):
    """
    具有残差连接和归一化的单个 Mamba 块。
    
    结构:
        x -> Norm -> SelectiveSSM -> DropPath -> + -> output
        |_________________________________|
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        drop_path: float = 0.0,
    ):
        """
        参数:
            d_model: 模型维度
            d_state: SSM 状态维度
            d_conv: 局部卷积宽度
            expand: 扩展因子
            dropout: Dropout 率
            drop_path: DropPath 率（随机深度）
        """
        super().__init__()
        
        self.norm = RMSNorm(d_model)
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 输入张量 [B, L, D]
            
        返回:
            输出张量 [B, L, D]
        """
        # 预归一化残差
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        x = self.dropout(x)
        x = self.drop_path(x)
        return x + residual


class MambaEncoder(nn.Module):
    """
    用于序列编码的 Mamba 块堆栈。
    
    此编码器专为一维信号（如心音）设计，
    提供具有线性复杂度的高效长程建模。
    
    支持 DropPath（随机深度）正则化，概率从第一层线性递增到最后一层。
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
    ):
        """
        参数:
            d_model: 模型维度
            n_layers: Mamba 块的数量
            d_state: SSM 状态维度
            d_conv: 局部卷积宽度
            expand: 扩展因子
            dropout: Dropout 率
            drop_path_rate: 最大 DropPath 率（线性递增到此值）
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # 计算每层的 drop_path 率（线性递增）
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        
        # Mamba 块的堆叠
        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                drop_path=dpr[i]
            )
            for i in range(n_layers)
        ])
        
        # 最终归一化
        self.norm_f = RMSNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_layers: bool = False
    ) -> torch.Tensor:
        """
        前向传播通过所有 Mamba 层。
        
        参数:
            x: 输入张量 [B, L, D]
            return_all_layers: 如果为 True，返回所有层输出的列表
            
        返回:
            如果 return_all_layers:
                张量列表，每层一个 + 最终输出
            否则:
                最终输出张量 [B, L, D]
        """
        all_outputs = []
        
        for layer in self.layers:
            x = layer(x)
            if return_all_layers:
                all_outputs.append(x)
        
        x = self.norm_f(x)
        
        if return_all_layers:
            all_outputs.append(x)
            return all_outputs
        
        return x


# ==============可选：使用官方 Mamba 实现 ==============

def get_mamba_encoder(
    d_model: int = 256,
    n_layers: int = 4,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    dropout: float = 0.1,
    use_official: bool = False
) -> nn.Module:
    """
    获取 Mamba 编码器的工厂函数。
    
    如果 use_official=True 且可用，则尝试使用官方 mamba-ssm 实现，
    否则回退到纯 PyTorch 实现。
    
    参数:
        d_model: 模型维度
        n_layers: 层数
        d_state: 状态维度
        d_conv: 卷积宽度
        expand: 扩展因子
        dropout: Dropout 率
        use_official: 是否优先使用官方实现
        
    返回:
        MambaEncoder 模块
    """
    if use_official:
        try:
            from mamba_ssm import Mamba
            
            # 在我们的接口中封装官方 Mamba
            class OfficialMambaEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.ModuleList([
                        Mamba(
                            d_model=d_model,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                        )
                        for _ in range(n_layers)
                    ])
                    self.norm_f = RMSNorm(d_model)
                    self.dropout = nn.Dropout(dropout)
                
                def forward(self, x, return_all_layers=False):
                    all_outputs = []
                    for layer in self.layers:
                        x = layer(x)
                        x = self.dropout(x)
                        if return_all_layers:
                            all_outputs.append(x)
                    x = self.norm_f(x)
                    if return_all_layers:
                        all_outputs.append(x)
                        return all_outputs
                    return x
            
            return OfficialMambaEncoder()
            
        except ImportError:
            pass  # 回退到纯 PyTorch 实现
    
    return MambaEncoder(
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dropout=dropout
    )


# ============== 双向 Mamba 编码器 ==============

class BiMambaEncoder(nn.Module):
    """
    双向 Mamba 编码器 (Bidirectional Mamba Encoder)。
    
    使用两个独立的 Mamba 编码器分别处理正向和反向序列，
    然后将输出融合。可以更好地捕获心动周期的完整上下文。
    
    注意：训练时间约为单向 Mamba 的 1.5-2 倍，推荐仅在
    最终验证时启用，日常实验使用单向 Mamba。
    
    融合方式：
    - "add": 正向和反向输出相加
    - "concat": 拼接后通过线性层降维
    - "gate": 使用可学习门控融合
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        fusion: str = "add"
    ):
        """
        参数:
            d_model: 模型维度
            n_layers: 每个方向的 Mamba 层数
            d_state: SSM 状态维度
            d_conv: 局部卷积宽度
            expand: 扩展因子
            dropout: Dropout 率
            drop_path_rate: DropPath 率
            fusion: 融合方式 ("add", "concat", "gate")
        """
        super().__init__()
        
        self.d_model = d_model
        self.fusion = fusion
        
        # 正向 Mamba
        self.forward_mamba = MambaEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            drop_path_rate=drop_path_rate
        )
        
        # 反向 Mamba
        self.backward_mamba = MambaEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            drop_path_rate=drop_path_rate
        )
        
        # 融合层
        if fusion == "concat":
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        elif fusion == "gate":
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
        
        # 最终归一化
        self.norm_f = RMSNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        return_all_layers: bool = False
    ) -> torch.Tensor:
        """
        前向传播通过双向 Mamba。
        
        参数:
            x: 输入张量 [B, L, D]
            return_all_layers: 如果为 True，返回所有层输出
            
        返回:
            融合后的输出张量 [B, L, D]
        """
        # 正向处理
        forward_out = self.forward_mamba(x, return_all_layers=False)
        
        # 反向处理：翻转序列 -> Mamba -> 翻转回来
        x_reversed = torch.flip(x, dims=[1])
        backward_out = self.backward_mamba(x_reversed, return_all_layers=False)
        backward_out = torch.flip(backward_out, dims=[1])
        
        # 融合
        if self.fusion == "add":
            output = forward_out + backward_out
        elif self.fusion == "concat":
            concat = torch.cat([forward_out, backward_out], dim=-1)
            output = self.fusion_proj(concat)
        elif self.fusion == "gate":
            concat = torch.cat([forward_out, backward_out], dim=-1)
            gate = self.gate(concat)
            output = gate * forward_out + (1 - gate) * backward_out
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion}")
        
        output = self.norm_f(output)
        return output


def get_mamba_encoder_v2(
    d_model: int = 256,
    n_layers: int = 4,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    dropout: float = 0.1,
    drop_path_rate: float = 0.0,
    bidirectional: bool = False,
    fusion: str = "add",
    use_official: bool = False
) -> nn.Module:
    """
    获取 Mamba 编码器的工厂函数（v2 版本，支持更多选项）。
    
    参数:
        d_model: 模型维度
        n_layers: 层数
        d_state: 状态维度
        d_conv: 卷积宽度
        expand: 扩展因子
        dropout: Dropout 率
        drop_path_rate: DropPath 率
        bidirectional: 是否使用双向 Mamba
        fusion: 双向融合方式
        use_official: 是否优先使用官方实现
        
    返回:
        Mamba 编码器模块
    """
    if bidirectional:
        return BiMambaEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            fusion=fusion
        )
    
    if use_official:
        try:
            from mamba_ssm import Mamba
            
            class OfficialMambaEncoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
                    self.layers = nn.ModuleList([
                        Mamba(
                            d_model=d_model,
                            d_state=d_state,
                            d_conv=d_conv,
                            expand=expand,
                        )
                        for _ in range(n_layers)
                    ])
                    self.drop_paths = nn.ModuleList([
                        DropPath(dpr[i]) if dpr[i] > 0 else nn.Identity()
                        for i in range(n_layers)
                    ])
                    self.norm_f = RMSNorm(d_model)
                    self.dropout = nn.Dropout(dropout)
                
                def forward(self, x, return_all_layers=False):
                    all_outputs = []
                    for layer, drop_path in zip(self.layers, self.drop_paths):
                        residual = x
                        x = layer(x)
                        x = self.dropout(x)
                        x = drop_path(x) + residual
                        if return_all_layers:
                            all_outputs.append(x)
                    x = self.norm_f(x)
                    if return_all_layers:
                        all_outputs.append(x)
                        return all_outputs
                    return x
            
            return OfficialMambaEncoder()
            
        except ImportError:
            pass
    
    return MambaEncoder(
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        dropout=dropout,
        drop_path_rate=drop_path_rate
    )
