"""
PA-HCL 的对比损失函数。

此模块提供分层对比学习损失：
- InfoNCE: 标准对比损失 (SimCLR 风格)
- HierarchicalContrastiveLoss: 周期级 + 子结构级对比
- SupConLoss: 监督对比损失（用于下游微调）

分层损失结合了：
1. 周期级：对比整个心动周期（全局模式）
2. 子结构级：对比对应的子结构（局部病理）

参考文献:
    Chen, T., et al. (2020). A Simple Framework for Contrastive Learning
    of Visual Representations. ICML.
    
    Khosla, P., et al. (2020). Supervised Contrastive Learning. NeurIPS.

作者: PA-HCL 团队
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    用于对比学习的 InfoNCE 损失。
    
    给定一批 N 个样本，每个样本有 2 个增强视图（共 2N 个），
    对于每个锚点，我们有 1 个正样本（同一样本的另一个视图）
    和 2(N-1) 个负样本（所有其他样本的视图）。
    
    Loss = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
    
    其中 (i, j) 是正样本对，k 遍历所有负样本。
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
        reduction: str = "mean"
    ):
        """
        参数:
            temperature: 温度缩放参数 (τ)
            normalize: 是否在计算相似度之前对特征进行 L2 归一化
            reduction: 如何减少损失 ("mean", "sum", "none")
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.reduction = reduction
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算 InfoNCE 损失。
        
        参数:
            z1: 来自第一个增强视图的特征 [B, D]
            z2: 来自第二个增强视图的特征 [B, D]
            labels: 监督对比变体的可选标签
            
        返回:
            标量损失值
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        # 归一化特征
        if self.normalize:
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
        
        # 拼接视图: [2B, D]
        features = torch.cat([z1, z2], dim=0)
        
        # 计算相似度矩阵: [2B, 2B]
        sim_matrix = torch.mm(features, features.t()) / self.temperature
        
        # 创建正样本对掩码
        # 对于样本 i，正样本位于 (i + batch_size) % (2 * batch_size)
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, device=device)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, device=device)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, device=device)
        
        # 掩盖自相似性 (对角线)
        self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix.masked_fill_(self_mask, float('-inf'))
        
        # 获取正样本相似度
        pos_sim = sim_matrix[pos_mask.bool()].view(2 * batch_size)
        
        # 计算 log-softmax (数值稳定)
        # 对于每个样本，分母包含所有非自身样本
        logsumexp = torch.logsumexp(sim_matrix, dim=1)
        
        # 损失: -log(exp(pos_sim) / sum(exp(all_sim))) = -pos_sim + logsumexp
        loss = -pos_sim + logsumexp
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SubstructureContrastiveLoss(nn.Module):
    """
    子结构级对比损失。
    
    在增强视图之间对齐相应的子结构，
    侧重于局部病理模式（S1，收缩期，S2，舒张期）。
    
    对于每个样本的 K 个子结构，我们对比：
    - view1 中的 sub_k 与 view2 中的 sub_k (正样本)
    - view1 中的 sub_k 与来自其他样本的所有其他 sub_k (负样本)
    
    这鼓励学习细粒度的局部表示。
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
        align_substructures: bool = True
    ):
        """
        参数:
            temperature: 温度缩放参数
            normalize: 是否对特征进行 L2 归一化
            align_substructures: 如果为 True，仅对比相同索引的子结构；
                                 如果为 False，对比所有子结构
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.align_substructures = align_substructures
    
    def forward(
        self,
        subs1: torch.Tensor,
        subs2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算子结构级对比损失。
        
        参数:
            subs1: 来自 view1 的子结构特征 [B, K, D]
            subs2: 来自 view2 的子结构特征 [B, K, D]
            
        返回:
            标量损失值
        """
        batch_size, num_subs, dim = subs1.shape
        device = subs1.device
        
        # 归一化
        if self.normalize:
            subs1 = F.normalize(subs1, dim=-1)
            subs2 = F.normalize(subs2, dim=-1)
        
        if self.align_substructures:
            # 分别对比对应的子结构
            # 对于每个子结构 k，将其视为独立的对比任务
            total_loss = 0.0
            
            for k in range(num_subs):
                # 从所有样本中获取第 k 个子结构: [B, D]
                z1_k = subs1[:, k]
                z2_k = subs2[:, k]
                
                # 拼接: [2B, D]
                features = torch.cat([z1_k, z2_k], dim=0)
                
                # 相似度: [2B, 2B]
                sim = torch.mm(features, features.t()) / self.temperature
                
                # 掩盖对角线
                mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
                sim.masked_fill_(mask, float('-inf'))
                
                # 正样本对: (i, i+B) 和 (i+B, i)
                labels = torch.cat([
                    torch.arange(batch_size, 2 * batch_size, device=device),
                    torch.arange(batch_size, device=device)
                ])
                
                # 交叉熵损失
                loss_k = F.cross_entropy(sim, labels)
                total_loss += loss_k
            
            return total_loss / num_subs
        
        else:
            # 对比所有子结构
            # 展平: [B*K, D]
            z1_flat = subs1.reshape(-1, dim)
            z2_flat = subs2.reshape(-1, dim)
            
            # 使用标准 InfoNCE
            features = torch.cat([z1_flat, z2_flat], dim=0)
            n = features.shape[0]
            
            sim = torch.mm(features, features.t()) / self.temperature
            
            # 掩盖对角线
            mask = torch.eye(n, dtype=torch.bool, device=device)
            sim.masked_fill_(mask, float('-inf'))
            
            # 正样本对
            bk = batch_size * num_subs
            labels = torch.cat([
                torch.arange(bk, n, device=device),
                torch.arange(bk, device=device)
            ])
            
            return F.cross_entropy(sim, labels)


class HierarchicalContrastiveLoss(nn.Module):
    """
    PA-HCL 的分层对比损失。
    
    结合了周期级和子结构级对比损失：
    
    L_total = λ_cycle * L_cycle + λ_sub * L_sub
    
    其中:
    - L_cycle: 对比整个心动周期（全局模式）
    - L_sub: 对比对应的子结构（局部模式）
    
    这种分层设计同时捕捉：
    1. 整体心律和模式
    2. 每个阶段的细粒度病理特征
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        lambda_cycle: float = 1.0,
        lambda_sub: float = 1.0,
        normalize: bool = True,
        align_substructures: bool = True
    ):
        """
        参数:
            temperature: 两种损失的温度
            lambda_cycle: 周期级损失的权重
            lambda_sub: 子结构级损失的权重
            normalize: 是否归一化特征
            align_substructures: 是否按索引对齐子结构
        """
        super().__init__()
        
        self.lambda_cycle = lambda_cycle
        self.lambda_sub = lambda_sub
        
        self.cycle_loss = InfoNCELoss(
            temperature=temperature,
            normalize=normalize
        )
        
        self.sub_loss = SubstructureContrastiveLoss(
            temperature=temperature,
            normalize=normalize,
            align_substructures=align_substructures
        )
    
    def forward(
        self,
        cycle_z1: torch.Tensor,
        cycle_z2: torch.Tensor,
        sub_z1: torch.Tensor,
        sub_z2: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算分层对比损失。
        
        参数:
            cycle_z1: 来自 view1 的周期级投影 [B, D]
            cycle_z2: 来自 view2 的周期级投影 [B, D]
            sub_z1: 来自 view1 的子结构投影 [B, K, D]
            sub_z2: 来自 view2 的子结构投影 [B, K, D]
            
        返回:
            (total_loss, loss_dict) 的元组，其中 loss_dict 包含
            用于记录的单个损失分量
        """
        # 周期级损失
        loss_cycle = self.cycle_loss(cycle_z1, cycle_z2)
        
        # 子结构级损失
        loss_sub = self.sub_loss(sub_z1, sub_z2)
        
        # 组合损失
        total_loss = self.lambda_cycle * loss_cycle + self.lambda_sub * loss_sub
        
        # 返回单独的损失用于日志记录
        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_cycle": loss_cycle.item(),
            "loss_sub": loss_sub.item()
        }
        
        return total_loss, loss_dict


class SupConLoss(nn.Module):
    """
    监督对比损失。
    
    InfoNCE 的扩展，使用标签信息来定义正样本对：
    具有相同标签的样本为正样本。
    
    对于使用标记数据进行微调很有用。
    
    参考文献:
        Khosla, P., et al. (2020). Supervised Contrastive Learning.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        contrast_mode: str = "all"
    ):
        """
        参数:
            temperature: 用于缩放的温度
            base_temperature: 用于归一化的基础温度
            contrast_mode: "one" 表示一个正样本，"all" 表示所有正样本
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.contrast_mode = contrast_mode
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算监督对比损失。
        
        参数:
            features: 特征表示 [B, D] 或 [B, n_views, D]
            labels: 真实标签 [B]
            
        返回:
            标量损失
        """
        device = features.device
        
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [B, 1, D]
        
        batch_size = features.shape[0]
        n_views = features.shape[1]
        
        # 检查标签
        if labels.shape[0] != batch_size:
            raise ValueError("Number of labels must match batch size")
        
        labels = labels.contiguous().view(-1, 1)
        
        # 掩码：正样本对是具有相同标签的样本
        mask = torch.eq(labels, labels.t()).float().to(device)
        
        # 计算对比
        contrast_feature = features.view(batch_size * n_views, -1)
        contrast_feature = F.normalize(contrast_feature, dim=1)
        
        # 相似度矩阵
        anchor_dot_contrast = torch.mm(contrast_feature, contrast_feature.t()) / self.temperature
        
        # 为了数值稳定性
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
        # 为 n_views 平铺掩码
        mask = mask.repeat(n_views, n_views)
        
        # 掩盖自对比
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size * n_views, device=device)
        mask = mask * logits_mask
        
        # 计算对数概率
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
        # 正样本对的对数似然均值
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.clamp(mask_pos_pairs, min=1)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        
        # 损失
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(n_views, batch_size).mean()
        
        return loss


class NTXentLoss(nn.Module):
    """
    归一化温度缩放交叉熵损失 (NT-Xent)。
    
    等同于 InfoNCE，但公式略有不同。
    在 SimCLR 论文中使用。
    """
    
    def __init__(
        self,
        temperature: float = 0.5,
        normalize: bool = True
    ):
        """
        参数:
            temperature: 温度参数
            normalize: 是否进行 L2 特征归一化
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 NT-Xent 损失。
        
        参数:
            z1: 来自视图 1 的特征 [B, D]
            z2: 来自视图 2 的特征 [B, D]
            
        返回:
            标量损失
        """
        batch_size = z1.shape[0]
        device = z1.device
        
        if self.normalize:
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
        
        # 拼接
        representations = torch.cat([z1, z2], dim=0)
        
        # 相似度矩阵
        similarity = torch.mm(representations, representations.t()) / self.temperature
        
        # 创建交叉熵标签
        # 对于 i 在 [0, B-1], 正样本位于 i+B
        # 对于 i 在 [B, 2B-1], 正样本位于 i-B
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(batch_size, device=device)
        ])
        
        # 掩盖对角线 (自相似性)
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        similarity = similarity.masked_fill(~mask, float('-inf'))
        
        # 实际上我们需要更仔细地处理这个问题
        # 标准的 NT-Xent 仅掩盖对角线
        # 让我们使用更清晰的实现
        
        sim = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        ) / self.temperature
        
        # 掩盖对角线
        sim.fill_diagonal_(float('-inf'))
        
        loss = self.criterion(sim, labels)
        
        return loss


# ============== Utility Functions ==============

def get_contrastive_loss(
    loss_type: str = "hierarchical",
    **kwargs
) -> nn.Module:
    """
    获取对比损失的工厂函数。
    
    参数:
        loss_type: 损失类型 ("infonce", "ntxent", "hierarchical", "supcon")
        **kwargs: 损失的其他参数
        
    返回:
        损失模块
    """
    if loss_type == "infonce":
        return InfoNCELoss(**kwargs)
    elif loss_type == "ntxent":
        return NTXentLoss(**kwargs)
    elif loss_type == "hierarchical":
        return HierarchicalContrastiveLoss(**kwargs)
    elif loss_type == "supcon":
        return SupConLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
