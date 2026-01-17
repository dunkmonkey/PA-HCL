"""
心音 (PCG) 信号的数据增强变换。

此模块提供专为心脏信号设计的、受生理学约束的增强策略。
所有增强都尊重心音的时间和频谱特性。

关键设计原则:
1. 保持生理有效性（例如，心率在正常范围内）
2. 保持 S1/S2 时间关系
3. 基于临床变异性范围的增强幅度

作者: PA-HCL Team
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d


# ============== Base Classes ==============

class Transform(ABC):
    """所有变换的抽象基类。"""
    
    def __init__(self, p: float = 1.0):
        """
        参数:
            p: 应用此变换的概率
        """
        self.p = p
    
    @abstractmethod
    def apply(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        应用变换到信号。
        
        参数:
            signal_data: 输入信号
            sample_rate: 采样率 (Hz)
            
        返回:
            变换后的信号
        """
        pass
    
    def __call__(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """以概率 p 应用变换。"""
        if random.random() < self.p:
            return self.apply(signal_data, sample_rate)
        return signal_data
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class Compose:
    """
    组合多个变换。
    
    变换按提供的顺序依次应用。
    
    示例:
        >>> transforms = Compose([
        ...     TimeShift(max_shift_ratio=0.1),
        ...     GaussianNoise(snr_range=(20, 40)),
        ...     AmplitudeScale(scale_range=(0.8, 1.2))
        ... ])
        >>> augmented = transforms(signal, sample_rate=5000)
    """
    
    def __init__(self, transforms: List[Transform]):
        """
        参数:
            transforms: 要顺序应用的变换列表
        """
        self.transforms = transforms
    
    def __call__(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """顺序应用所有变换。"""
        for transform in self.transforms:
            signal_data = transform(signal_data, sample_rate)
        return signal_data
    
    def __repr__(self) -> str:
        transform_strs = [f"  {t}" for t in self.transforms]
        return "Compose([\n" + ",\n".join(transform_strs) + "\n])"


class RandomApply:
    """
    以给定的概率应用变换。
    
    用于概率性地应用任何变换的包装器。
    """
    
    def __init__(self, transform: Transform, p: float = 0.5):
        """
        参数:
            transform: 要应用的变换
            p: 应用变换的概率
        """
        self.transform = transform
        self.p = p
    
    def __call__(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        if random.random() < self.p:
            return self.transform(signal_data, sample_rate)
        return signal_data


class RandomChoice:
    """从列表中随机选择一个变换进行应用。"""
    
    def __init__(self, transforms: List[Transform]):
        """
        参数:
            transforms: 供选择的变换列表
        """
        self.transforms = transforms
    
    def __call__(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        transform = random.choice(self.transforms)
        return transform(signal_data, sample_rate)


# ============== Time Domain Augmentations ==============

class TimeShift(Transform):
    """
    随机在时间上偏移信号（循环移位）。
    
    这模拟了心动周期边界确切时间的变化，
    这种变化自然发生于心率变异性。
    
    注意:
        使用循环移位以保持信号长度并避免
        在边界引入人为的静音。
    """
    
    def __init__(
        self,
        max_shift_ratio: float = 0.1,
        p: float = 1.0
    ):
        """
        Args:
            max_shift_ratio: 最大偏移量占信号长度的比例
                            (0.1 = ±10% 的信号长度)
            p: 应用此变换的概率
        """
        super().__init__(p)
        self.max_shift_ratio = max_shift_ratio
    
    def apply(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        max_shift = int(len(signal_data) * self.max_shift_ratio)
        
        if max_shift == 0:
            return signal_data
        
        shift = random.randint(-max_shift, max_shift)
        return np.roll(signal_data, shift)


class AmplitudeScale(Transform):
    """
    随机缩放信号幅度。
    
    这模拟了以下变化:
    - 麦克风灵敏度
    - 患者身体成分（影响声音传输）
    - 传感器放置压力
    
    默认范围 (0.8-1.2) 对应于临床设置中观察到的
    典型录音变异性。
    """
    
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        p: float = 1.0
    ):
        """
        参数:
            scale_range: 幅度缩放的 (min_scale, max_scale)
            p: 应用此变换的概率
        """
        super().__init__(p)
        self.scale_range = scale_range
    
    def apply(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        scale = random.uniform(*self.scale_range)
        return signal_data * scale


class GaussianNoise(Transform):
    """
    在指定的 SNR 添加高斯白噪声。
    
    模拟临床录音中常见的环境噪声和传感器噪声。
    
    SNR 范围是均匀采样的，在训练期间提供多样化的噪音水平。
    """
    
    def __init__(
        self,
        snr_range: Tuple[float, float] = (20, 40),
        p: float = 1.0
    ):
        """
        参数:
            snr_range: dB 单位的 (min_snr, max_snr)
                      20 dB = 嘈杂的录音
                      40 dB = 干净的录音
            p: 应用此变换的概率
        """
        super().__init__(p)
        self.snr_range = snr_range
    
    def apply(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        # Sample target SNR
        target_snr = random.uniform(*self.snr_range)
        
        # Calculate signal power
        signal_power = np.mean(signal_data ** 2)
        
        # Calculate required noise power for target SNR
        # SNR = 10 * log10(signal_power / noise_power)
        # noise_power = signal_power / 10^(SNR/10)
        noise_power = signal_power / (10 ** (target_snr / 10))
        noise_std = np.sqrt(noise_power)
        
        # Generate and add noise
        noise = np.random.normal(0, noise_std, len(signal_data))
        return signal_data + noise


class TimeStretch(Transform):
    """
    时间拉伸/压缩信号（模拟心率变化）。
    
    这对心音来说是一个关键的增强，因为它模拟了自然的
    心率变异性 (HRV)。心率从 60 到 66 BPM 的变化对应于
    10% 的速度增加。
    
    实现使用线性插值以提高效率。
    """
    
    def __init__(
        self,
        speed_range: Tuple[float, float] = (0.9, 1.1),
        p: float = 1.0
    ):
        """
        参数:
            speed_range: (min_speed, max_speed) 因子
                        0.9 = 慢 10% (更低的心率)
                        1.1 = 快 10% (更高的心率)
            p: 应用此变换的概率
        """
        super().__init__(p)
        self.speed_range = speed_range
    
    def apply(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        speed = random.uniform(*self.speed_range)
        
        if abs(speed - 1.0) < 0.01:
            return signal_data
        
        # Calculate new length
        original_length = len(signal_data)
        new_length = int(original_length / speed)
        
        # Create interpolation indices
        old_indices = np.arange(original_length)
        new_indices = np.linspace(0, original_length - 1, new_length)
        
        # Interpolate
        stretched = np.interp(new_indices, old_indices, signal_data)
        
        # Restore original length (pad or truncate)
        if len(stretched) < original_length:
            # Pad with zeros (or repeat)
            pad_length = original_length - len(stretched)
            stretched = np.pad(stretched, (0, pad_length), mode='constant')
        elif len(stretched) > original_length:
            # Center crop
            start = (len(stretched) - original_length) // 2
            stretched = stretched[start:start + original_length]
        
        return stretched.astype(np.float32)


class RandomCrop(Transform):
    """
    随机裁剪信号的一部分。
    
    用于训练对录音边界处不完整心动周期
    具有鲁棒性的模型。
    """
    
    def __init__(
        self,
        crop_ratio: Tuple[float, float] = (0.8, 1.0),
        p: float = 0.5
    ):
        """
        参数:
            crop_ratio: 保留信号的 (min_ratio, max_ratio)
            p: 应用此变换的概率
        """
        super().__init__(p)
        self.crop_ratio = crop_ratio
    
    def apply(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        ratio = random.uniform(*self.crop_ratio)
        crop_length = int(len(signal_data) * ratio)
        
        if crop_length >= len(signal_data):
            return signal_data
        
        # Random start position
        max_start = len(signal_data) - crop_length
        start = random.randint(0, max_start)
        
        cropped = signal_data[start:start + crop_length]
        
        # Pad back to original length
        pad_left = start
        pad_right = len(signal_data) - start - crop_length
        
        return np.pad(cropped, (pad_left, pad_right), mode='constant')


class Reverse(Transform):
    """
    在时间上反转信号。
    
    虽然在生理上没有意义，但这可以帮助模型
    学习更鲁棒的局部特征。应少量使用，
    因为它会破坏 S1→S2 的时间顺序。
    """
    
    def __init__(self, p: float = 0.1):
        """
        参数:
            p: 应用概率（默认为低概率以保持结构）
        """
        super().__init__(p)
    
    def apply(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        return signal_data[::-1].copy()


# ============== Frequency Domain Augmentations ==============

class FrequencyMask(Transform):
    """
    掩盖随机频带 (SpecAugment 风格)。
    
    这有助于模型对来自不同听诊器类型的频率选择性
    噪声或滤波效应变得鲁棒。
    
    实现使用 FFT 在频域中工作。
    """
    
    def __init__(
        self,
        max_mask_ratio: float = 0.15,
        num_masks: int = 1,
        p: float = 1.0
    ):
        """
        参数:
            max_mask_ratio: 要掩盖的频率 bins 的最大比例
            num_masks: 要应用的频率掩码数量
            p: 应用此变换的概率
        """
        super().__init__(p)
        self.max_mask_ratio = max_mask_ratio
        self.num_masks = num_masks
    
    def apply(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        # Compute FFT
        fft = np.fft.rfft(signal_data)
        num_freq_bins = len(fft)
        
        for _ in range(self.num_masks):
            # Random mask width
            mask_width = random.randint(1, int(num_freq_bins * self.max_mask_ratio))
            
            # Random mask start position
            mask_start = random.randint(0, num_freq_bins - mask_width)
            
            # Apply mask (set to zero)
            fft[mask_start:mask_start + mask_width] = 0
        
        # Inverse FFT
        masked = np.fft.irfft(fft, n=len(signal_data))
        
        return masked.astype(np.float32)


class TimeMask(Transform):
    """
    掩盖随机时间段 (SpecAugment 风格)。
    
    默认情况下，最大掩码宽度限制为 30ms，
    这比典型的 S1/S2 持续时间 (~100-150ms) 短。
    这确保我们不会完全掩盖重要的心脏事件。
    
    此增强有助于模型从部分信息中学习
    并对短暂的信号丢失变得鲁棒。
    """
    
    def __init__(
        self,
        max_mask_ratio: float = 0.1,
        max_mask_width_ms: float = 30.0,
        num_masks: int = 2,
        p: float = 1.0
    ):
        """
        参数:
            max_mask_ratio: 要掩盖的信号的最大总比例
            max_mask_width_ms: 单个掩码的最大宽度 (毫秒)
                              (30ms 默认值确保 S1/S2 不会被完全掩盖)
            num_masks: 要应用的时间掩码数量
            p: 应用此变换的概率
        """
        super().__init__(p)
        self.max_mask_ratio = max_mask_ratio
        self.max_mask_width_ms = max_mask_width_ms
        self.num_masks = num_masks
    
    def apply(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        signal_length = len(signal_data)
        max_mask_samples = int(self.max_mask_width_ms * sample_rate / 1000)
        max_total_mask = int(signal_length * self.max_mask_ratio)
        
        # Limit single mask width to both constraints
        max_single_mask = min(max_mask_samples, max_total_mask // self.num_masks)
        
        if max_single_mask < 1:
            return signal_data
        
        result = signal_data.copy()
        
        for _ in range(self.num_masks):
            mask_width = random.randint(1, max_single_mask)
            mask_start = random.randint(0, signal_length - mask_width)
            
            # Set masked region to zero (or could use mean)
            result[mask_start:mask_start + mask_width] = 0
        
        return result


class LowPassFilter(Transform):
    """
    应用随机低通滤波。
    
    模拟不同听诊器频率响应的影响
    或穿过厚组织的闷响录音。
    """
    
    def __init__(
        self,
        cutoff_range: Tuple[float, float] = (200, 400),
        order: int = 4,
        p: float = 0.3
    ):
        """
        参数:
            cutoff_range: Hz 单位的 (min_cutoff, max_cutoff)
            order: 滤波器阶数
            p: 应用此变换的概率
        """
        super().__init__(p)
        self.cutoff_range = cutoff_range
        self.order = order
    
    def apply(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        cutoff = random.uniform(*self.cutoff_range)
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        
        if normalized_cutoff >= 1:
            return signal_data
        
        b, a = signal.butter(self.order, normalized_cutoff, btype='low')
        filtered = signal.filtfilt(b, a, signal_data)
        
        return filtered.astype(np.float32)


class HighPassFilter(Transform):
    """
    应用随机高通滤波。
    
    模拟激进的基线去除或某些
    数字听诊器中的高通滤波。
    """
    
    def __init__(
        self,
        cutoff_range: Tuple[float, float] = (20, 50),
        order: int = 4,
        p: float = 0.3
    ):
        """
        参数:
            cutoff_range: Hz 单位的 (min_cutoff, max_cutoff)
            order: 滤波器阶数
            p: 应用此变换的概率
        """
        super().__init__(p)
        self.cutoff_range = cutoff_range
        self.order = order
    
    def apply(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        cutoff = random.uniform(*self.cutoff_range)
        nyquist = sample_rate / 2
        normalized_cutoff = cutoff / nyquist
        
        if normalized_cutoff <= 0 or normalized_cutoff >= 1:
            return signal_data
        
        b, a = signal.butter(self.order, normalized_cutoff, btype='high')
        filtered = signal.filtfilt(b, a, signal_data)
        
        return filtered.astype(np.float32)


# ============== Advanced Augmentations ==============

class PitchShift(Transform):
    """
    改变音高而不改变持续时间。
    
    虽然心音没有音乐音高，但这种增强
    可以模拟不同患者之间不同的胸腔共振效果。
    
    实现: 重采样 + 时间拉伸以保持持续时间。
    """
    
    def __init__(
        self,
        shift_range: Tuple[float, float] = (-2, 2),
        p: float = 0.3
    ):
        """
        参数:
            shift_range: 以半音为单位的音高偏移
            p: 应用此变换的概率
        """
        super().__init__(p)
        self.shift_range = shift_range
    
    def apply(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        shift_semitones = random.uniform(*self.shift_range)
        
        if abs(shift_semitones) < 0.1:
            return signal_data
        
        # Calculate pitch shift ratio
        ratio = 2 ** (shift_semitones / 12)
        
        # Resample to shift pitch
        new_length = int(len(signal_data) / ratio)
        indices = np.linspace(0, len(signal_data) - 1, new_length)
        resampled = np.interp(indices, np.arange(len(signal_data)), signal_data)
        
        # Time stretch back to original length
        final_indices = np.linspace(0, len(resampled) - 1, len(signal_data))
        result = np.interp(final_indices, np.arange(len(resampled)), resampled)
        
        return result.astype(np.float32)


class AddColoredNoise(Transform):
    """
    添加有色噪声（粉红、棕色）而不是白噪声。
    
    有色噪声更好地模拟了现实世界的环境噪声，
    这些噪声通常在较低频率下具有更多能量。
    """
    
    def __init__(
        self,
        snr_range: Tuple[float, float] = (15, 30),
        noise_type: str = "pink",
        p: float = 0.3
    ):
        """
        参数:
            snr_range: (min_snr, max_snr) in dB
            noise_type: "pink" (1/f), "brown" (1/f^2), or "white"
            p: Probability of applying this transform
        """
        super().__init__(p)
        self.snr_range = snr_range
        self.noise_type = noise_type
    
    def apply(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        length = len(signal_data)
        
        # Generate colored noise in frequency domain
        if self.noise_type == "white":
            noise = np.random.normal(0, 1, length)
        else:
            # Generate white noise
            white = np.random.normal(0, 1, length)
            
            # Transform to frequency domain
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(length, 1/sample_rate)
            
            # Apply frequency weighting
            freqs[0] = 1  # Avoid division by zero
            if self.noise_type == "pink":
                # Pink noise: 1/f spectrum
                fft = fft / np.sqrt(freqs)
            elif self.noise_type == "brown":
                # Brown noise: 1/f^2 spectrum
                fft = fft / freqs
            
            # Transform back
            noise = np.fft.irfft(fft, n=length)
            noise = noise / np.std(noise)  # Normalize
        
        # Scale noise to target SNR
        target_snr = random.uniform(*self.snr_range)
        signal_power = np.mean(signal_data ** 2)
        noise_power = signal_power / (10 ** (target_snr / 10))
        noise = noise * np.sqrt(noise_power)
        
        return (signal_data + noise).astype(np.float32)


class SimulateRespiratoryNoise(Transform):
    """
    添加模拟的呼吸伪影。
    
    呼吸音是心音记录中常见的伪影，尤其是在听诊过程中。
    此类模拟低频呼吸干扰。
    """
    
    def __init__(
        self,
        intensity_range: Tuple[float, float] = (0.05, 0.2),
        breath_rate_range: Tuple[float, float] = (12, 20),
        p: float = 0.3
    ):
        """
        参数:
            intensity_range: 相对于信号的呼吸伪影振幅
            breath_rate_range: 每分钟呼吸次数
            p: 应用此变换的概率
        """
        super().__init__(p)
        self.intensity_range = intensity_range
        self.breath_rate_range = breath_rate_range
    
    def apply(self, signal_data: np.ndarray, sample_rate: int) -> np.ndarray:
        length = len(signal_data)
        duration = length / sample_rate
        
        # Generate respiratory pattern
        breath_rate = random.uniform(*self.breath_rate_range) / 60  # Hz
        intensity = random.uniform(*self.intensity_range)
        
        # Create low-frequency modulation (breath envelope)
        t = np.linspace(0, duration, length)
        # Use sum of sinusoids for more realistic breathing pattern
        breath_pattern = (
            0.6 * np.sin(2 * np.pi * breath_rate * t) +
            0.3 * np.sin(4 * np.pi * breath_rate * t + np.random.uniform(0, np.pi)) +
            0.1 * np.sin(6 * np.pi * breath_rate * t + np.random.uniform(0, np.pi))
        )
        
        # Add some noise to the breath pattern
        breath_noise = np.random.normal(0, 0.1, length)
        breath_noise = uniform_filter1d(breath_noise, size=int(sample_rate * 0.1))
        
        # Combine and scale
        respiratory_artifact = (breath_pattern + breath_noise) * intensity * np.std(signal_data)
        
        return (signal_data + respiratory_artifact).astype(np.float32)


# ============== Factory Functions ==============

def get_pretrain_transforms(
    config: Optional[Dict[str, Any]] = None,
    sample_rate: int = 5000
) -> Compose:
    """
    获取用于自监督预训练的标准增强管道。
    
    此管道旨在创建同一心动周期的多样化视图，同时保持生理有效性。
    
    参数:
        config: 包含增强设置的可选配置字典
        sample_rate: 采样率 (Hz)
        
    返回:
        Compose 变换管道
    """
    if config is None:
        config = {}
    
    transforms = []
    
    # Time shift - always useful, low risk
    if config.get("time_shift", {}).get("enabled", True):
        transforms.append(TimeShift(
            max_shift_ratio=config.get("time_shift", {}).get("max_shift_ratio", 0.1),
            p=0.8
        ))
    
    # Speed perturbation - simulates heart rate variation
    if config.get("speed_perturb", {}).get("enabled", True):
        speed_range = config.get("speed_perturb", {}).get("speed_range", [0.9, 1.1])
        transforms.append(TimeStretch(
            speed_range=tuple(speed_range),
            p=0.5
        ))
    
    # Amplitude scaling
    if config.get("amplitude_scale", {}).get("enabled", True):
        transforms.append(AmplitudeScale(
            scale_range=(
                config.get("amplitude_scale", {}).get("min_scale", 0.8),
                config.get("amplitude_scale", {}).get("max_scale", 1.2)
            ),
            p=0.8
        ))
    
    # Gaussian noise - common augmentation
    if config.get("gaussian_noise", {}).get("enabled", True):
        snr_range = config.get("gaussian_noise", {}).get("snr_range", [20, 40])
        transforms.append(GaussianNoise(
            snr_range=tuple(snr_range),
            p=0.5
        ))
    
    # Time masking - with physiological constraint
    if config.get("time_mask", {}).get("enabled", True):
        transforms.append(TimeMask(
            max_mask_ratio=config.get("time_mask", {}).get("max_mask_ratio", 0.1),
            max_mask_width_ms=config.get("time_mask", {}).get("max_mask_width_ms", 30),
            num_masks=2,
            p=0.5
        ))
    
    # Frequency masking
    if config.get("freq_mask", {}).get("enabled", True):
        transforms.append(FrequencyMask(
            max_mask_ratio=config.get("freq_mask", {}).get("max_mask_ratio", 0.15),
            num_masks=1,
            p=0.3
        ))
    
    return Compose(transforms)


def get_finetune_transforms(
    config: Optional[Dict[str, Any]] = None,
    sample_rate: int = 5000
) -> Compose:
    """
    获取用于下游微调的增强管道。
    
    使用比预训练更轻的增强，以避免破坏学习到的表示。
    
    参数:
        config: 可选配置字典
        sample_rate: 采样率 (Hz)
        
    返回:
        Compose 变换管道
    """
    transforms = [
        TimeShift(max_shift_ratio=0.05, p=0.5),
        AmplitudeScale(scale_range=(0.9, 1.1), p=0.5),
        GaussianNoise(snr_range=(25, 40), p=0.3),
    ]
    
    return Compose(transforms)


def get_eval_transforms() -> Compose:
    """
    获取用于评估的变换（无增强）。
    
    返回:
        空的 Compose 管道（恒等变换）
    """
    return Compose([])


# ============== Substructure-level Augmentation ==============

class SubstructureAugmentor:
    """
    对心动周期内的子结构应用增强。
    
    此类处理各个子结构之间增强的协调，确保在需要时的一致性。
    """
    
    def __init__(
        self,
        transform: Transform,
        consistent: bool = False
    ):
        """
        参数:
            transform: 应用于每个子结构的变换
            consistent: 如果为 True，则对所有子结构应用相同的随机参数
        """
        self.transform = transform
        self.consistent = consistent
    
    def __call__(
        self,
        substructures: List[np.ndarray],
        sample_rate: int
    ) -> List[np.ndarray]:
        """
        对子结构列表应用增强。
        
        参数:
            substructures: 子结构信号列表
            sample_rate: 采样率 (Hz)
            
        返回:
            增强后的子结构列表
        """
        if self.consistent:
            # Fix random seed for consistent augmentation
            seed = random.randint(0, 2**31)
            augmented = []
            for sub in substructures:
                random.seed(seed)
                np.random.seed(seed)
                augmented.append(self.transform(sub, sample_rate))
            return augmented
        else:
            # Independent augmentation for each substructure
            return [self.transform(sub, sample_rate) for sub in substructures]
