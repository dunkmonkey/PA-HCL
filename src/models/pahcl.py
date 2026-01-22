"""
用于预训练的完整 PA-HCL 模型。

此模块提供了完整的 PA-HCL 模型，结合了：
- 编码器 (CNN-Mamba)
- 投影头 (周期级和子结构级)
- 分层对比损失
- MoCo风格的动量编码器（周期级）

作者: PA-HCL 团队
"""

from typing import Dict, Optional, Tuple
import copy

import torch
import torch.nn as nn

from .encoder import CNNMambaEncoder, build_encoder
from .heads import ProjectionHead, SubstructureProjectionHead


class PAHCLModel(nn.Module):
    """
    生理感知分层对比学习模型 (PA-HCL)。
    
    此模型实现了心音自监督学习的完整 PA-HCL 架构：
    
    1. 编码器：CNN-Mamba 混合模型，用于局部和全局特征提取
    2. 周期投影头：投影周期级特征
    3. 子结构投影头：投影子结构级特征
    
    在预训练期间，处理两个增强视图，并在两个级别计算分层对比损失。
    """
    
    def __init__(
        self,
        # 编码器设置
        encoder_type: str = "cnn_mamba",
        in_channels: int = 1,
        cnn_channels: list = [32, 64, 128, 256],
        cnn_kernel_sizes: Optional[list] = None,
        cnn_strides: Optional[list] = None,
        cnn_dropout: float = 0.1,
        mamba_d_model: int = 256,
        mamba_n_layers: int = 4,
        mamba_d_state: int = 8,
        mamba_expand: int = 2,
        mamba_dropout: float = 0.1,
        pool_type: str = "mean",
        # 投影头设置
        cycle_proj_hidden: int = 128,
        cycle_proj_output: int = 32,
        cycle_proj_layers: int = 2,
        sub_proj_hidden: int = 64,
        sub_proj_output: int = 16,
        # 子结构设置
        num_substructures: int = 4,
        # MoCo 设置 (Step 2 & 3)
        use_moco: bool = False,
        moco_momentum: float = 0.999,
        queue_size: int = 8192,  # Step 3: 队列大小
    ):
        """
        参数:
            encoder_type: 编码器类型 ("cnn_mamba", "cnn_transformer", "cnn_only")
            in_channels: 输入通道数
            cnn_channels: CNN 层通道数
            cnn_kernel_sizes: CNN 卷积核大小
            cnn_strides: CNN 步幅
            cnn_dropout: CNN dropout 率
            mamba_d_model: Mamba 模型维度
            mamba_n_layers: Mamba 层数
            mamba_d_state: Mamba 状态维度
            mamba_expand: Mamba 扩展因子
            mamba_dropout: Mamba dropout 率
            pool_type: 编码器的池化类型
            cycle_proj_hidden: 周期投影头隐藏维度
            cycle_proj_output: 周期投影输出维度
            cycle_proj_layers: 投影层数
            sub_proj_hidden: 子结构投影隐藏维度
            sub_proj_output: 子结构投影输出维度
            num_substructures: 子结构数量 (K)
            use_moco: 是否使用 MoCo 风格的动量编码器（仅周期级）
            moco_momentum: 动量系数（默认 0.999）
            queue_size: 特征队列大小（默认 8192）
        """
        super().__init__()
        
        self.num_substructures = num_substructures
        self.use_moco = use_moco
        self.moco_momentum = moco_momentum
        self.queue_size = queue_size
        
        # 处理 CNN 参数默认值，使其与层数一致
        if cnn_kernel_sizes is None:
            if len(cnn_channels) == 2:
                cnn_kernel_sizes = [7, 5]
            elif len(cnn_channels) == 3:
                cnn_kernel_sizes = [7, 5, 3]
            else:
                cnn_kernel_sizes = [7, 5, 5, 3]
        if cnn_strides is None:
            cnn_strides = [2] * len(cnn_channels)

        # 构建编码器
        if encoder_type == "cnn_mamba":
            self.encoder = CNNMambaEncoder(
                in_channels=in_channels,
                cnn_channels=cnn_channels,
                cnn_kernel_sizes=cnn_kernel_sizes,
                cnn_strides=cnn_strides,
                cnn_dropout=cnn_dropout,
                mamba_d_model=mamba_d_model,
                mamba_n_layers=mamba_n_layers,
                mamba_d_state=mamba_d_state,
                mamba_expand=mamba_expand,
                mamba_dropout=mamba_dropout,
                pool_type=pool_type
            )
            encoder_out_dim = mamba_d_model
            # 子结构特征维度来自倒数第二层 CNN 层
            sub_feature_dim = cnn_channels[-2] if len(cnn_channels) >= 2 else cnn_channels[-1]
        else:
            self.encoder = build_encoder(encoder_type, **{
                "in_channels": in_channels,
                "cnn_channels": cnn_channels,
                "mamba_d_model": mamba_d_model,
                "mamba_n_layers": mamba_n_layers
            })
            encoder_out_dim = mamba_d_model
            sub_feature_dim = cnn_channels[-2]
        
        # 周期级投影头
        self.cycle_projector = ProjectionHead(
            input_dim=encoder_out_dim,
            hidden_dim=cycle_proj_hidden,
            output_dim=cycle_proj_output,
            num_layers=cycle_proj_layers
        )
        
        # Step 2: MoCo 风格的动量编码器（仅周期级）
        if self.use_moco:
            # 创建动量编码器（深拷贝主编码器）
            if encoder_type == "cnn_mamba":
                self.encoder_momentum = CNNMambaEncoder(
                    in_channels=in_channels,
                    cnn_channels=cnn_channels,
                    cnn_kernel_sizes=cnn_kernel_sizes,
                    cnn_strides=cnn_strides,
                    cnn_dropout=cnn_dropout,
                    mamba_d_model=mamba_d_model,
                    mamba_n_layers=mamba_n_layers,
                    mamba_d_state=mamba_d_state,
                    mamba_expand=mamba_expand,
                    mamba_dropout=mamba_dropout,
                    pool_type=pool_type
                )
            else:
                self.encoder_momentum = build_encoder(encoder_type, **{
                    "in_channels": in_channels,
                    "cnn_channels": cnn_channels,
                    "mamba_d_model": mamba_d_model,
                    "mamba_n_layers": mamba_n_layers
                })
            
            # 创建动量投影头
            self.cycle_projector_momentum = ProjectionHead(
                input_dim=encoder_out_dim,
                hidden_dim=cycle_proj_hidden,
                output_dim=cycle_proj_output,
                num_layers=cycle_proj_layers
            )
            
            # 初始化动量编码器参数（复制主编码器参数）
            self._init_momentum_encoder()
            
            # 动量编码器不需要梯度
            for param in self.encoder_momentum.parameters():
                param.requires_grad = False
            for param in self.cycle_projector_momentum.parameters():
                param.requires_grad = False
            
            # Step 3: 创建特征队列
            # 队列存储归一化的特征向量 [D, K]
            self.register_buffer(
                "queue",
                torch.randn(cycle_proj_output, queue_size)
            )
            # L2 归一化队列
            self.queue = nn.functional.normalize(self.queue, dim=0)
            
            # 队列指针，指示下一个要替换的位置
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        else:
            self.encoder_momentum = None
            self.cycle_projector_momentum = None
        
        # 子结构级投影头
        # 子结构特征来自 CNN 中间层，其通道数为 sub_feature_dim
        self.sub_projector = SubstructureProjectionHead(
            input_dim=sub_feature_dim,
            hidden_dim=sub_proj_hidden,
            output_dim=sub_proj_output
        )
        
        # 存储维度供下游使用
        self.encoder_dim = encoder_out_dim
        self.cycle_proj_dim = cycle_proj_output
        self.sub_proj_dim = sub_proj_output
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        单视图的前向传播（推理/下游任务）。
        
        参数:
            x: 输入信号 [B, 1, L] 或 [B, L]
            return_features: 如果为 True，返回编码器特征
            
        返回:
            包含 cycle_proj 和可选 features 的字典
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # 获取编码器输出
        features = self.encoder(x)  # [B, D]
        
        # 投影周期特征
        cycle_proj = self.cycle_projector(features)  # [B, proj_dim]
        
        result = {
            "cycle_proj": cycle_proj,
        }
        
        if return_features:
            result["features"] = features
        
        return result
    
    def forward_pretrain(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
        subs1: Optional[torch.Tensor] = None,
        subs2: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        双视图预训练的前向传播。
        
        Step 2: 支持 MoCo 模式
        - SimCLR 模式: 两个视图都用主编码器
        - MoCo 模式: view1 用主编码器（query），view2 用动量编码器（key）
        
        参数:
            view1: 第一个增强视图 [B, 1, L] (query)
            view2: 第二个增强视图 [B, 1, L] (key)
            subs1: 来自 view1 的预分割子结构 [B, K, L_sub] (可选)
            subs2: 来自 view2 的预分割子结构 [B, K, L_sub] (可选)
            
        返回:
            包含用于损失计算的所有投影的字典
        """
        batch_size = view1.shape[0]
        
        # 处理输入维度
        if view1.dim() == 2:
            view1 = view1.unsqueeze(1)
        if view2.dim() == 2:
            view2 = view2.unsqueeze(1)
        
        # 周期级: 编码两个视图
        # view1 (query) 使用主编码器
        cycle_z1 = self.encoder(view1)  # [B, D]
        cycle_proj1 = self.cycle_projector(cycle_z1)  # [B, proj_dim]
        
        # view2 (key) 使用动量编码器（如果启用 MoCo）
        if self.use_moco:
            # MoCo 模式: view2 用动量编码器，无梯度
            with torch.no_grad():
                cycle_z2 = self.encoder_momentum(view2)  # [B, D]
                cycle_proj2 = self.cycle_projector_momentum(cycle_z2)  # [B, proj_dim]
        else:
            # SimCLR 模式: view2 也用主编码器
            cycle_z2 = self.encoder(view2)
            cycle_proj2 = self.cycle_projector(cycle_z2)
        
        # 子结构级: 始终使用主编码器（保持简单性）
        # 获取子结构特征
        if subs1 is not None and subs2 is not None:
            # 使用提供的子结构
            # subs1/subs2: [B, K, L_sub] - 需要添加通道维度并编码
            sub_proj1 = self._encode_substructures(subs1)
            sub_proj2 = self._encode_substructures(subs2)
        else:
            # 从编码器中间特征提取子结构
            sub_feats1 = self.encoder.get_sub_features(view1)  # [B, C, L']
            sub_feats2 = self.encoder.get_sub_features(view2)
            
            # 分割成 K 个子结构
            sub_proj1 = self._split_and_project_subs(sub_feats1)
            sub_proj2 = self._split_and_project_subs(sub_feats2)
        
        return {
            "cycle_proj1": cycle_proj1,
            "cycle_proj2": cycle_proj2,
            "sub_proj1": sub_proj1,
            "sub_proj2": sub_proj2,
            "cycle_z1": cycle_z1,
            "cycle_z2": cycle_z2,
        }
    
    def _split_and_project_subs(self, sub_feats: torch.Tensor) -> torch.Tensor:
        """
        将特征图分割为子结构并投影。
        
        参数:
            sub_feats: 子结构特征 [B, C, L']
            
        返回:
            投影后的子结构 [B, K, D]
        """
        B, C, L = sub_feats.shape
        K = self.num_substructures
        
        # 确保 L 能被 K 整除
        sub_len = L // K
        
        # 重塑为 [B, K, C, sub_len]
        subs = sub_feats[:, :, :K * sub_len].reshape(B, C, K, sub_len)
        subs = subs.permute(0, 2, 1, 3)  # [B, K, C, sub_len]
        
        # 投影
        sub_proj = self.sub_projector(subs)  # [B, K, D]
        
        return sub_proj
    
    def _encode_substructures(self, subs: torch.Tensor) -> torch.Tensor:
        """
        编码预分割的子结构。
        
        参数:
            subs: 原始子结构信号 [B, K, L_sub]
            
        返回:
            投影后的子结构 [B, K, D]
        """
        B, K, L = subs.shape
        
        # 重塑以进行批量处理
        subs_flat = subs.reshape(B * K, 1, L)
        
        # 仅使用 CNN 主干获取特征（取倒数第二层以匹配子结构通道维度）
        _, intermediates = self.encoder.cnn(subs_flat, return_intermediate=True)
        sub_feats = intermediates[-2] if len(intermediates) >= 2 else intermediates[-1]

        # 重塑回来
        C_out = sub_feats.shape[1]
        L_out = sub_feats.shape[2]
        sub_feats = sub_feats.reshape(B, K, C_out, L_out)
        
        # 投影
        sub_proj = self.sub_projector(sub_feats)  # [B, K, D]
        
        return sub_proj
    
    def get_encoder_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        获取用于下游任务的编码器输出。
        
        参数:
            x: 输入信号 [B, 1, L]
            
        返回:
            编码器特征 [B, D]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.encoder(x)
    
    def _init_momentum_encoder(self):
        """
        初始化动量编码器参数（从主编码器复制）。
        Step 2: MoCo initialization
        """
        if not self.use_moco:
            return
        
        # 复制编码器参数
        for param_q, param_k in zip(
            self.encoder.parameters(),
            self.encoder_momentum.parameters()
        ):
            param_k.data.copy_(param_q.data)
        
        # 复制投影头参数
        for param_q, param_k in zip(
            self.cycle_projector.parameters(),
            self.cycle_projector_momentum.parameters()
        ):
            param_k.data.copy_(param_q.data)
    
    @torch.no_grad()
    def _momentum_update(self):
        """
        动量更新 key 编码器。
        Step 2: MoCo momentum update
        
        使用公式: θ_k = m * θ_k + (1 - m) * θ_q
        其中 m 是动量系数（默认 0.999）
        """
        if not self.use_moco:
            return
        
        m = self.moco_momentum
        
        # 更新编码器
        for param_q, param_k in zip(
            self.encoder.parameters(),
            self.encoder_momentum.parameters()
        ):
            param_k.data = m * param_k.data + (1.0 - m) * param_q.data
        
        # 更新投影头
        for param_q, param_k in zip(
            self.cycle_projector.parameters(),
            self.cycle_projector_momentum.parameters()
        ):
            param_k.data = m * param_k.data + (1.0 - m) * param_q.data
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """
        将新的特征向量入队，并移除旧的特征向量。
        Step 3: Queue management for MoCo
        
        参数:
            keys: 新的特征向量 [B, D]，来自动量编码器
        """
        if not self.use_moco:
            return
        
        batch_size = keys.shape[0]
        
        # 确保 batch size 不超过队列大小
        assert self.queue_size % batch_size == 0 or batch_size <= self.queue_size, \
            f"队列大小 {self.queue_size} 应该是 batch size {batch_size} 的倍数"
        
        ptr = int(self.queue_ptr)
        
        # 替换队列中的特征
        if ptr + batch_size <= self.queue_size:
            # 一次性替换
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # 需要循环（队列满了从头开始）
            remaining = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        
        # 更新指针
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr


def build_pahcl_model(config) -> PAHCLModel:
    """
    根据配置构建 PA-HCL 模型。
    
    参数:
        config: 包含模型参数的配置对象
        
    返回:
        PAHCLModel 实例
    """
    return PAHCLModel(
        encoder_type=getattr(config.model, 'encoder_type', 'cnn_mamba'),
        in_channels=1,
        cnn_channels=getattr(config.model, 'cnn_channels', [32, 64, 128, 256]),
        cnn_kernel_sizes=getattr(config.model, 'cnn_kernel_sizes', [7, 5, 5, 3]),
        cnn_strides=getattr(config.model, 'cnn_strides', [2, 2, 2, 2]),
        cnn_dropout=getattr(config.model, 'cnn_dropout', 0.1),
        mamba_d_model=getattr(config.model, 'mamba_d_model', 256),
        mamba_n_layers=getattr(config.model, 'mamba_n_layers', 4),
        mamba_d_state=getattr(config.model, 'mamba_d_state', 16),
        mamba_expand=getattr(config.model, 'mamba_expand', 2),
        mamba_dropout=getattr(config.model, 'mamba_dropout', 0.1),
        pool_type=getattr(config.model, 'pool_type', 'mean'),
        cycle_proj_hidden=getattr(config.model, 'proj_hidden_dim', 512),
        cycle_proj_output=getattr(config.model, 'proj_output_dim', 128),
        cycle_proj_layers=getattr(config.model, 'proj_num_layers', 2),
        sub_proj_hidden=getattr(config.model, 'sub_proj_hidden_dim', 256),
        sub_proj_output=getattr(config.model, 'sub_proj_output_dim', 64),
        num_substructures=getattr(config.data, 'num_substructures', 4),
        # Step 2 & 3: MoCo 参数
        use_moco=getattr(config.model, 'use_moco', False),
        moco_momentum=getattr(config.model, 'moco_momentum', 0.999),
        queue_size=getattr(config.model, 'queue_size', 8192),
    )

