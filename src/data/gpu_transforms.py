"""
GPU 加速的数据增强模块。

此模块提供在 GPU 上批量执行的数据增强操作，
显著减少 CPU 负担并提高训练吞吐量。

所有增强都接受 [B, C, T] 形状的张量，并在 GPU 上执行。

设计原则:
1. 批量操作: 所有增强在 batch 维度并行执行
2. 纯 PyTorch: 不依赖 scipy/numpy，利用 CUDA 加速
3. 概率控制: 每个增强可独立控制应用概率
4. 可组合: 通过 GPUCompose 组合多个增强

作者: PA-HCL Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union


class GPUTransformBase(nn.Module):
    """
    GPU 增强变换的基类。
    
    所有 GPU 增强都继承此类，并实现 apply 方法。
    变换只在 training 模式下以概率 p 应用。
    """
    
    def __init__(self, p: float = 1.0):
        """
        参数:
            p: 应用此变换的概率 (0.0 到 1.0)
        """
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用变换。
        
        参数:
            x: 输入张量 [B, C, T] 或 [B, T]
            
        返回:
            变换后的张量，形状与输入相同
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # 只在 training 模式下应用增强
        if self.training and torch.rand(1, device=x.device).item() < self.p:
            return self.apply(x)
        return x
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """
        实际的变换逻辑（子类实现）。
        
        参数:
            x: 输入张量 [B, C, T]
            
        返回:
            变换后的张量 [B, C, T]
        """
        raise NotImplementedError


class GPUTimeShift(GPUTransformBase):
    """
    GPU 批量时间偏移（循环移位）。
    
    模拟心动周期边界时间的自然变化。
    每个样本独立采样偏移量。
    """
    
    def __init__(self, max_shift_ratio: float = 0.1, p: float = 0.8):
        """
        参数:
            max_shift_ratio: 最大偏移比例 (0.1 = ±10%)
            p: 应用概率
        """
        super().__init__(p)
        self.max_shift_ratio = max_shift_ratio
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        max_shift = int(T * self.max_shift_ratio)
        
        if max_shift == 0:
            return x
        
        # 每个样本独立的偏移量
        shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)
        
        # 批量 roll
        result = torch.empty_like(x)
        for i in range(B):
            result[i] = torch.roll(x[i], shifts[i].item(), dims=-1)
        
        return result


class GPUAmplitudeScale(GPUTransformBase):
    """
    GPU 批量幅度缩放。
    
    模拟录音设备灵敏度、传感器放置等变化。
    每个样本独立采样缩放因子。
    """
    
    def __init__(
        self, 
        scale_range: Tuple[float, float] = (0.8, 1.2), 
        p: float = 0.8
    ):
        """
        参数:
            scale_range: (min_scale, max_scale) 缩放范围
            p: 应用概率
        """
        super().__init__(p)
        self.scale_min = scale_range[0]
        self.scale_max = scale_range[1]
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # 每个样本独立的缩放因子 [B, 1, 1]
        scales = torch.empty(B, 1, 1, device=x.device).uniform_(
            self.scale_min, self.scale_max
        )
        return x * scales


class GPUGaussianNoise(GPUTransformBase):
    """
    GPU 批量高斯噪声添加。
    
    按指定的信噪比 (SNR) 范围添加随机噪声。
    每个样本独立采样 SNR。
    """
    
    def __init__(
        self, 
        snr_range: Tuple[float, float] = (20, 40), 
        p: float = 0.5
    ):
        """
        参数:
            snr_range: (min_snr_db, max_snr_db) 信噪比范围
            p: 应用概率
        """
        super().__init__(p)
        self.snr_min = snr_range[0]
        self.snr_max = snr_range[1]
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # 每个样本随机 SNR [B, 1, 1]
        target_snr = torch.empty(B, 1, 1, device=x.device).uniform_(
            self.snr_min, self.snr_max
        )
        
        # 计算信号功率 [B, 1, 1]
        signal_power = (x ** 2).mean(dim=-1, keepdim=True)
        
        # 计算噪声功率
        # SNR = 10 * log10(signal_power / noise_power)
        # noise_power = signal_power / 10^(SNR/10)
        noise_power = signal_power / (10 ** (target_snr / 10))
        
        # 生成并添加噪声
        noise = torch.randn_like(x) * torch.sqrt(noise_power + 1e-10)
        
        return x + noise


class GPUTimeMask(GPUTransformBase):
    """
    GPU 批量时间遮蔽。
    
    随机将时间段置零，类似于 SpecAugment 中的时间遮蔽。
    模拟信号缺失或传感器接触不良。
    """
    
    def __init__(
        self,
        max_mask_ratio: float = 0.1,
        max_mask_width: int = 150,  # 30ms @ 5000Hz
        num_masks: int = 2,
        p: float = 0.5
    ):
        """
        参数:
            max_mask_ratio: 最大遮蔽比例
            max_mask_width: 单个遮蔽块的最大宽度（样本数）
            num_masks: 遮蔽块数量
            p: 应用概率
        """
        super().__init__(p)
        self.max_mask_ratio = max_mask_ratio
        self.max_mask_width = max_mask_width
        self.num_masks = num_masks
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        max_total = int(T * self.max_mask_ratio)
        max_single = min(self.max_mask_width, max(1, max_total // self.num_masks))
        
        if max_single < 1:
            return x
        
        result = x.clone()
        
        for _ in range(self.num_masks):
            # 批量生成 mask 参数
            widths = torch.randint(1, max_single + 1, (B,), device=x.device)
            max_start = max(1, T - max_single)
            starts = torch.randint(0, max_start, (B,), device=x.device)
            
            # 应用 mask
            for i in range(B):
                w = widths[i].item()
                s = starts[i].item()
                end = min(s + w, T)
                result[i, :, s:end] = 0
        
        return result


class GPUFrequencyMask(GPUTransformBase):
    """
    GPU 批量频率遮蔽。
    
    在频域随机遮蔽频率带，类似于 SpecAugment。
    使用 FFT 实现，完全在 GPU 上执行。
    """
    
    def __init__(
        self, 
        max_mask_ratio: float = 0.15, 
        num_masks: int = 1, 
        p: float = 0.3
    ):
        """
        参数:
            max_mask_ratio: 最大遮蔽频带比例
            num_masks: 遮蔽块数量
            p: 应用概率
        """
        super().__init__(p)
        self.max_mask_ratio = max_mask_ratio
        self.num_masks = num_masks
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        
        # FFT
        x_fft = torch.fft.rfft(x, dim=-1)
        num_bins = x_fft.shape[-1]
        max_width = max(1, int(num_bins * self.max_mask_ratio))
        
        for _ in range(self.num_masks):
            # 随机 mask 宽度和位置
            widths = torch.randint(1, max_width + 1, (B,), device=x.device)
            max_start = max(1, num_bins - max_width)
            starts = torch.randint(0, max_start, (B,), device=x.device)
            
            # 应用 mask
            for i in range(B):
                w = widths[i].item()
                s = starts[i].item()
                end = min(s + w, num_bins)
                x_fft[i, :, s:end] = 0
        
        # iFFT
        return torch.fft.irfft(x_fft, n=T, dim=-1)


class GPUTimeStretch(GPUTransformBase):
    """
    GPU 批量时间拉伸/压缩。
    
    模拟心率变化导致的心动周期时长变化。
    使用 F.interpolate 实现高效的 GPU 重采样。
    
    注意: 由于不同样本可能有不同的拉伸比例，
    实现使用分组处理策略。
    """
    
    def __init__(
        self, 
        speed_range: Tuple[float, float] = (0.9, 1.1), 
        p: float = 0.5
    ):
        """
        参数:
            speed_range: (min_speed, max_speed) 速度范围
                        < 1.0 表示拉伸（减慢）
                        > 1.0 表示压缩（加快）
            p: 应用概率
        """
        super().__init__(p)
        self.speed_min = speed_range[0]
        self.speed_max = speed_range[1]
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        
        # 每个样本随机速度
        speeds = torch.empty(B, device=x.device).uniform_(
            self.speed_min, self.speed_max
        )
        
        result = torch.empty_like(x)
        
        for i in range(B):
            speed = speeds[i].item()
            new_length = int(T / speed)
            
            if new_length == T:
                result[i] = x[i]
                continue
            
            # 使用 interpolate 进行重采样
            # 输入需要是 [1, C, T]
            stretched = F.interpolate(
                x[i:i+1], 
                size=new_length, 
                mode='linear', 
                align_corners=False
            )
            
            # 裁剪或填充到原始长度
            if new_length > T:
                # 中心裁剪
                start = (new_length - T) // 2
                result[i] = stretched[0, :, start:start + T]
            else:
                # 中心填充
                pad_total = T - new_length
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                result[i] = F.pad(stretched[0], (pad_left, pad_right), mode='constant', value=0)
        
        return result


class GPUCompose(nn.Module):
    """
    组合多个 GPU 变换。
    
    变换按提供的顺序依次应用。
    
    示例:
        >>> transforms = GPUCompose([
        ...     GPUTimeShift(max_shift_ratio=0.1),
        ...     GPUGaussianNoise(snr_range=(20, 40)),
        ...     GPUAmplitudeScale(scale_range=(0.8, 1.2))
        ... ])
        >>> augmented = transforms(batch_tensor)  # [B, C, T]
    """
    
    def __init__(self, transforms: List[nn.Module]):
        """
        参数:
            transforms: GPU 变换列表
        """
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        顺序应用所有变换。
        
        参数:
            x: 输入张量 [B, C, T]
            
        返回:
            变换后的张量 [B, C, T]
        """
        for transform in self.transforms:
            x = transform(x)
        return x


class GPURandomApply(nn.Module):
    """
    以给定概率应用变换的包装器。
    """
    
    def __init__(self, transform: nn.Module, p: float = 0.5):
        """
        参数:
            transform: 要应用的变换
            p: 应用概率
        """
        super().__init__()
        self.transform = transform
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and torch.rand(1, device=x.device).item() < self.p:
            return self.transform(x)
        return x


def get_gpu_pretrain_transforms(
    config: Optional[dict] = None,
    sample_rate: int = 5000
) -> GPUCompose:
    """
    获取预训练用的 GPU 增强管道。
    
    参数:
        config: 增强配置字典（可选）
        sample_rate: 采样率
        
    返回:
        GPUCompose 实例
    """
    if config is None:
        config = {}
    
    # 解析配置
    time_shift_cfg = config.get('time_shift', {})
    amplitude_cfg = config.get('amplitude_scale', {})
    noise_cfg = config.get('gaussian_noise', {})
    time_mask_cfg = config.get('time_mask', {})
    freq_mask_cfg = config.get('freq_mask', {})
    speed_cfg = config.get('speed_perturb', {})
    
    transforms = []
    
    # 时间偏移
    if time_shift_cfg.get('enabled', True):
        transforms.append(GPUTimeShift(
            max_shift_ratio=time_shift_cfg.get('max_shift_ratio', 0.1),
            p=0.8
        ))
    
    # 时间拉伸（心率变化）
    if speed_cfg.get('enabled', True):
        speed_range = speed_cfg.get('speed_range', [0.9, 1.1])
        transforms.append(GPUTimeStretch(
            speed_range=tuple(speed_range),
            p=0.5
        ))
    
    # 幅度缩放
    if amplitude_cfg.get('enabled', True):
        transforms.append(GPUAmplitudeScale(
            scale_range=(
                amplitude_cfg.get('min_scale', 0.8),
                amplitude_cfg.get('max_scale', 1.2)
            ),
            p=0.8
        ))
    
    # 高斯噪声
    if noise_cfg.get('enabled', True):
        snr_range = noise_cfg.get('snr_range', [20, 40])
        transforms.append(GPUGaussianNoise(
            snr_range=tuple(snr_range),
            p=0.5
        ))
    
    # 时间遮蔽
    if time_mask_cfg.get('enabled', True):
        max_width_ms = time_mask_cfg.get('max_mask_width_ms', 30)
        max_width_samples = int(max_width_ms * sample_rate / 1000)
        transforms.append(GPUTimeMask(
            max_mask_ratio=time_mask_cfg.get('max_mask_ratio', 0.1),
            max_mask_width=max_width_samples,
            num_masks=2,
            p=0.5
        ))
    
    # 频率遮蔽
    if freq_mask_cfg.get('enabled', True):
        transforms.append(GPUFrequencyMask(
            max_mask_ratio=freq_mask_cfg.get('max_mask_ratio', 0.15),
            num_masks=1,
            p=0.3
        ))
    
    return GPUCompose(transforms)


def get_gpu_downstream_transforms(
    config: Optional[dict] = None,
    sample_rate: int = 5000,
    is_training: bool = True
) -> Optional[GPUCompose]:
    """
    获取下游任务用的 GPU 增强管道。
    
    参数:
        config: 增强配置字典（可选）
        sample_rate: 采样率
        is_training: 是否为训练模式
        
    返回:
        GPUCompose 实例，或 None（非训练模式）
    """
    if not is_training:
        return None
    
    # 下游任务使用较轻的增强
    transforms = [
        GPUTimeShift(max_shift_ratio=0.05, p=0.5),
        GPUAmplitudeScale(scale_range=(0.9, 1.1), p=0.5),
        GPUGaussianNoise(snr_range=(25, 40), p=0.3),
    ]
    
    return GPUCompose(transforms)


# ============== 用于在 collate 后应用 GPU 增强的辅助类 ==============

class GPUAugmentationWrapper:
    """
    封装 GPU 增强，用于在训练循环中使用。
    
    使用示例:
        gpu_aug = GPUAugmentationWrapper(config, device='cuda')
        
        for batch in dataloader:
            view1 = batch['view1'].to(device)
            view2 = batch['view2'].to(device)
            
            # 应用 GPU 增强
            view1, view2 = gpu_aug(view1, view2)
            
            # 继续训练...
    """
    
    def __init__(
        self,
        config: Optional[dict] = None,
        device: str = 'cuda',
        sample_rate: int = 5000
    ):
        """
        参数:
            config: 增强配置
            device: 目标设备
            sample_rate: 采样率
        """
        self.device = device
        self.transform = get_gpu_pretrain_transforms(config, sample_rate)
        self.transform = self.transform.to(device)
        self.transform.train()
    
    def __call__(
        self, 
        view1: torch.Tensor, 
        view2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对两个视图分别应用增强。
        
        参数:
            view1: 第一个视图 [B, C, T]
            view2: 第二个视图 [B, C, T]
            
        返回:
            (augmented_view1, augmented_view2)
        """
        return self.transform(view1), self.transform(view2)
    
    def train(self):
        """设置为训练模式。"""
        self.transform.train()
    
    def eval(self):
        """设置为评估模式（不应用增强）。"""
        self.transform.eval()
