"""
心音 (PCG) 信号的信号预处理工具。

此模块提供以下功能:
- 信号滤波和归一化
- 心动周期检测和分割
- 子结构特征提取
- 信号质量评估

所有函数均设计为无需人工标注即可工作，
遵循论文中描述的“弱结构近似”原则。

作者: PA-HCL Team
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d


# ============== Signal Loading ==============

def load_audio(
    file_path: Union[str, Path],
    target_sr: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    加载音频文件并选择性地重新采样。
    
    参数:
        file_path: 音频文件路径 (.wav)
        target_sr: 目标采样率。如果为 None，则使用原始采样率。
        
    返回:
        (signal, sample_rate) 的元组
        
    抛出:
        FileNotFoundError: 如果文件不存在
        ValueError: 如果不支持文件格式
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    if file_path.suffix.lower() != ".wav":
        raise ValueError(f"Unsupported audio format: {file_path.suffix}")
    
    # Load WAV file
    sr, audio = wavfile.read(file_path)
    
    # Convert to float32 and normalize to [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128.0
    else:
        audio = audio.astype(np.float32)
    
    # Handle stereo: convert to mono by averaging channels
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # Resample if needed
    if target_sr is not None and sr != target_sr:
        audio = resample_signal(audio, sr, target_sr)
        sr = target_sr
    
    return audio, sr


def resample_signal(
    signal_data: np.ndarray,
    original_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    将信号重新采样到目标采样率。
    
    使用多相滤波进行高质量重采样。
    
    参数:
        signal_data: 输入信号
        original_sr: 原始采样率
        target_sr: 目标采样率
        
    返回:
        重采样后的信号
    """
    if original_sr == target_sr:
        return signal_data
    
    # Calculate resampling ratio
    gcd = np.gcd(original_sr, target_sr)
    up = target_sr // gcd
    down = original_sr // gcd
    
    # Use scipy's resample_poly for efficient polyphase resampling
    resampled = signal.resample_poly(signal_data, up, down)
    
    return resampled.astype(np.float32)


# ============== Signal Filtering ==============

def bandpass_filter(
    signal_data: np.ndarray,
    sample_rate: int,
    lowcut: float = 25.0,
    highcut: float = 400.0,
    order: int = 4
) -> np.ndarray:
    """
    应用巴特沃斯带通滤波器以去除噪声和基线漂移。
    
    选择默认频率范围 (25-400 Hz) 是为了:
    - 去除由呼吸引起的基线漂移 (< 25 Hz)
    - 保留心音成分 (S1, S2: 20-150 Hz)
    - 保留杂音频率 (最高 400 Hz)
    - 去除高频噪声 (> 400 Hz)
    
    参数:
        signal_data: 输入信号
        sample_rate: 采样率 (Hz)
        lowcut: 低截止频率 (Hz)
        highcut: 高截止频率 (Hz)
        order: 滤波器阶数（越高截止越陡峭，但振铃越多）
        
    返回:
        滤波后的信号
        
    注意:
        使用零相位滤波 (filtfilt) 以避免相位失真，
        这对于保留 PCG 中的时间关系很重要。
    """
    # Nyquist frequency
    nyquist = sample_rate / 2.0
    
    # Normalize cutoff frequencies
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Ensure frequencies are valid
    if low <= 0:
        low = 0.001
    if high >= 1:
        high = 0.999
    
    # Design Butterworth bandpass filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply zero-phase filtering to avoid phase distortion
    # This is crucial for maintaining temporal alignment of S1/S2
    filtered = signal.filtfilt(b, a, signal_data)
    
    return filtered.astype(np.float32)


def normalize_signal(
    signal_data: np.ndarray,
    method: str = "zscore"
) -> np.ndarray:
    """
    将信号归一化到标准范围。
    
    参数:
        signal_data: 输入信号
        method: 归一化方法
            - "zscore": 零均值，单位方差（推荐用于训练）
            - "minmax": 缩放到 [0, 1]
            - "peak": 基于绝对最大值缩放到 [-1, 1]
            
    返回:
        归一化后的信号
    """
    if method == "zscore":
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        if std < 1e-8:
            # Avoid division by zero for near-constant signals
            return np.zeros_like(signal_data)
        return (signal_data - mean) / std
    
    elif method == "minmax":
        min_val = np.min(signal_data)
        max_val = np.max(signal_data)
        if max_val - min_val < 1e-8:
            return np.zeros_like(signal_data)
        return (signal_data - min_val) / (max_val - min_val)
    
    elif method == "peak":
        peak = np.max(np.abs(signal_data))
        if peak < 1e-8:
            return np.zeros_like(signal_data)
        return signal_data / peak
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ============== Energy Envelope Computation ==============

def compute_energy_envelope(
    signal_data: np.ndarray,
    sample_rate: int,
    frame_length_ms: float = 20.0,
    hop_length_ms: float = 10.0,
    method: str = "shannon"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算心动周期检测的信号能量包络。
    
    能量包络突出了高能量区域 (S1, S2 音)，
    同时抑制了低能量区域 (收缩期、舒张期音程)。
    
    参数:
        signal_data: 输入信号（应先进行滤波）
        sample_rate: 采样率 (Hz)
        frame_length_ms: 帧长 (ms)
        hop_length_ms: 跳步长 (ms)
        method: 能量计算方法
            - "rms": 均方根能量
            - "shannon": 香农能量（推荐，增强峰值）
            - "hilbert": 希尔伯特包络
            
    返回:
        (envelope, time_axis) 的元组
        - envelope: 能量包络值
        - time_axis: 对应于包络值的样本时间
    """
    frame_length = int(frame_length_ms * sample_rate / 1000)
    hop_length = int(hop_length_ms * sample_rate / 1000)
    
    if method == "rms":
        # Root mean square energy
        # Simple but effective for general audio
        envelope = _compute_rms_envelope(signal_data, frame_length, hop_length)
        
    elif method == "shannon":
        # Shannon energy: -x^2 * log(x^2)
        # This method enhances medium-amplitude components (like S1/S2)
        # while suppressing very high and very low amplitude components
        envelope = _compute_shannon_envelope(signal_data, frame_length, hop_length)
        
    elif method == "hilbert":
        # Hilbert envelope (analytic signal magnitude)
        # Provides instantaneous amplitude
        envelope = _compute_hilbert_envelope(signal_data)
        # Downsample to match frame rate
        envelope = envelope[::hop_length]
        
    else:
        raise ValueError(f"Unknown envelope method: {method}")
    
    # Create time axis (in samples)
    time_axis = np.arange(len(envelope)) * hop_length
    
    return envelope.astype(np.float32), time_axis


def _compute_rms_envelope(
    signal_data: np.ndarray,
    frame_length: int,
    hop_length: int
) -> np.ndarray:
    """计算 RMS 能量包络。"""
    # Square the signal
    squared = signal_data ** 2
    
    # Apply moving average
    envelope = uniform_filter1d(squared, size=frame_length, mode='constant')
    
    # Take square root and downsample
    envelope = np.sqrt(envelope[::hop_length])
    
    return envelope


def _compute_shannon_envelope(
    signal_data: np.ndarray,
    frame_length: int,
    hop_length: int
) -> np.ndarray:
    """
    计算香农能量包络。
    
    香农能量定义为: E = -x^2 * log(x^2)
    具有增强中等振幅成分的特性，
    这对于在嘈杂的录音中检测 S1/S2 非常有益。
    """
    # Normalize to avoid numerical issues
    normalized = signal_data / (np.max(np.abs(signal_data)) + 1e-8)
    
    # Compute Shannon energy
    # Add small epsilon to avoid log(0)
    x_squared = normalized ** 2 + 1e-10
    shannon = -x_squared * np.log(x_squared)
    
    # Apply moving average
    envelope = uniform_filter1d(shannon, size=frame_length, mode='constant')
    
    # Downsample
    envelope = envelope[::hop_length]
    
    return envelope


def _compute_hilbert_envelope(signal_data: np.ndarray) -> np.ndarray:
    """计算希尔伯特包络（解析信号幅值）。"""
    analytic_signal = signal.hilbert(signal_data)
    envelope = np.abs(analytic_signal)
    return envelope


# ============== Peak Detection ==============

def detect_peaks(
    envelope: np.ndarray,
    sample_rate: int,
    hop_length_samples: int,
    min_peak_distance_ms: float = 300.0,
    threshold_factor: float = 0.3
) -> np.ndarray:
    """
    检测对应于心动周期的能量包络峰值。
    
    使用自适应阈值处理不同的信号质量和心率变异性。
    
    参数:
        envelope: 来自 compute_energy_envelope() 的能量包络
        sample_rate: 原始信号采样率
        hop_length_samples: 包络计算中使用的跳步长
        min_peak_distance_ms: 峰值之间的最小距离 (ms)
            （防止将 S1 和 S2 检测为单独的周期）
            默认 300ms 对应于最大 ~200 BPM 的心率
        threshold_factor: 峰值必须高于平均包络的比例
            
    返回:
        原始信号样本中的峰值位置数组
        
    注意:
        自适应阈值计算为:
        threshold = mean(envelope) + threshold_factor * (max - mean)
        这可以适当地处理安静和响亮的录音。
    """
    # Convert minimum distance to envelope frames
    envelope_sr = sample_rate / hop_length_samples
    min_distance = int(min_peak_distance_ms * envelope_sr / 1000)
    min_distance = max(1, min_distance)  # Ensure at least 1
    
    # Compute adaptive threshold
    # This adapts to the overall signal level
    mean_env = np.mean(envelope)
    max_env = np.max(envelope)
    threshold = mean_env + threshold_factor * (max_env - mean_env)
    
    # Find peaks using scipy
    peaks, properties = signal.find_peaks(
        envelope,
        height=threshold,
        distance=min_distance,
        prominence=threshold_factor * (max_env - mean_env)
    )
    
    # Convert back to original signal sample indices
    peak_samples = peaks * hop_length_samples
    
    return peak_samples


def detect_peaks_adaptive(
    envelope: np.ndarray,
    sample_rate: int,
    hop_length_samples: int,
    expected_hr_range: Tuple[float, float] = (40, 200)
) -> np.ndarray:
    """
    根据预期心率使用自适应参数检测峰值。
    
    这是一个更健壮的版本，会根据信号特性
    自动调整检测参数。
    
    参数:
        envelope: 能量包络
        sample_rate: 原始信号采样率
        hop_length_samples: 包络计算中使用的跳步长
        expected_hr_range: 预期心率范围 BPM (min, max)
        
    返回:
        原始信号样本中的峰值位置数组
    """
    envelope_sr = sample_rate / hop_length_samples
    
    # Calculate expected peak distances from heart rate range
    min_bpm, max_bpm = expected_hr_range
    # At max_bpm, minimum distance between beats
    min_distance_sec = 60.0 / max_bpm
    # At min_bpm, maximum distance between beats
    max_distance_sec = 60.0 / min_bpm
    
    min_distance_frames = int(min_distance_sec * envelope_sr)
    min_distance_frames = max(1, min_distance_frames)
    
    # Smooth envelope for more stable peak detection
    smoothed = uniform_filter1d(envelope, size=max(3, min_distance_frames // 4))
    
    # Multi-pass peak detection with adaptive threshold
    # Start with high threshold and lower if too few peaks found
    signal_duration = len(envelope) / envelope_sr
    expected_min_peaks = int(signal_duration * min_bpm / 60)
    expected_max_peaks = int(signal_duration * max_bpm / 60)
    
    best_peaks = None
    best_score = -np.inf
    
    for threshold_factor in [0.5, 0.4, 0.3, 0.2, 0.1]:
        mean_env = np.mean(smoothed)
        max_env = np.max(smoothed)
        threshold = mean_env + threshold_factor * (max_env - mean_env)
        
        peaks, _ = signal.find_peaks(
            smoothed,
            height=threshold,
            distance=min_distance_frames
        )
        
        # Score based on how close to expected number of peaks
        if len(peaks) >= expected_min_peaks * 0.5:
            # Prefer peaks that fall within expected range
            if expected_min_peaks <= len(peaks) <= expected_max_peaks * 1.5:
                score = 1.0 / (1.0 + abs(len(peaks) - (expected_min_peaks + expected_max_peaks) / 2))
            else:
                score = 0.5 / (1.0 + abs(len(peaks) - expected_max_peaks))
            
            if score > best_score:
                best_score = score
                best_peaks = peaks
    
    if best_peaks is None or len(best_peaks) < 2:
        # Fallback: use simple peak detection
        best_peaks = detect_peaks(
            envelope, sample_rate, hop_length_samples,
            min_peak_distance_ms=min_distance_sec * 1000,
            threshold_factor=0.2
        )
        # Convert back to envelope frames for consistency
        best_peaks = best_peaks // hop_length_samples
    
    # Convert to original signal samples
    peak_samples = best_peaks * hop_length_samples
    
    return peak_samples


# ============== Cardiac Cycle Extraction ==============

def extract_cycles(
    signal_data: np.ndarray,
    peak_positions: np.ndarray,
    sample_rate: int,
    min_duration_sec: float = 0.4,
    max_duration_sec: float = 1.5,
    target_length: Optional[int] = None,
    padding_mode: str = "zero"
) -> List[np.ndarray]:
    """
    根据检测到的峰值从信号中提取心动周期。
    
    每个周期定义为连续峰值之间的片段。
    过短或过长的周期会被过滤掉，因为它们
    可能代表检测错误或心律失常。
    
    参数:
        signal_data: 滤波后的信号
        peak_positions: 来自 detect_peaks() 的峰值位置（样本）
        sample_rate: 采样率 (Hz)
        min_duration_sec: 最小有效周期时间（秒）
        max_duration_sec: 最大有效周期时间（秒）
        target_length: 如果指定，所有周期都被填充/截断到此长度
        padding_mode: 长度标准化方式
            - "zero": 零填充短周期，截断长周期
            - "repeat": 重复短周期的信号
            - "resample": 将所有周期重采样到目标长度
            
    返回:
        心动周期信号列表
        
    注意:
        最小/最大持续时间过滤器作为一种质量控制形式:
        - 太短 (< 0.4s): 可能是误报检测
        - 太长 (> 1.5s): 可能是有漏搏或伪影
    """
    min_samples = int(min_duration_sec * sample_rate)
    max_samples = int(max_duration_sec * sample_rate)
    
    cycles = []
    
    # Extract segments between consecutive peaks
    for i in range(len(peak_positions) - 1):
        start = peak_positions[i]
        end = peak_positions[i + 1]
        
        # Boundary check
        if start < 0 or end > len(signal_data):
            continue
        
        cycle_length = end - start
        
        # Duration filtering
        if cycle_length < min_samples or cycle_length > max_samples:
            continue
        
        cycle = signal_data[start:end].copy()
        
        # Length normalization if target_length is specified
        if target_length is not None:
            cycle = _normalize_length(cycle, target_length, padding_mode)
        
        cycles.append(cycle)
    
    return cycles


def _normalize_length(
    signal_data: np.ndarray,
    target_length: int,
    mode: str = "zero"
) -> np.ndarray:
    """
    将信号标准化到目标长度。
    
    参数:
        signal_data: 输入信号
        target_length: 期望的输出长度
        mode: 标准化模式 ("zero", "repeat", 或 "resample")
        
    返回:
        确切目标长度的信号
    """
    current_length = len(signal_data)
    
    if current_length == target_length:
        return signal_data
    
    if mode == "zero":
        if current_length < target_length:
            # Zero-pad symmetrically
            pad_total = target_length - current_length
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return np.pad(signal_data, (pad_left, pad_right), mode='constant')
        else:
            # Center-crop
            start = (current_length - target_length) // 2
            return signal_data[start:start + target_length]
    
    elif mode == "repeat":
        if current_length < target_length:
            # Repeat signal to fill target length
            repeats = int(np.ceil(target_length / current_length))
            repeated = np.tile(signal_data, repeats)
            return repeated[:target_length]
        else:
            # Center-crop
            start = (current_length - target_length) // 2
            return signal_data[start:start + target_length]
    
    elif mode == "resample":
        # Resample to exact target length
        return signal.resample(signal_data, target_length).astype(np.float32)
    
    else:
        raise ValueError(f"Unknown padding mode: {mode}")


# ============== Substructure Extraction ==============

def split_substructures(
    cycle: np.ndarray,
    num_substructures: int = 4
) -> List[np.ndarray]:
    """
    将心动周期分割为 K 个相等的子结构片段。
    
    这实现了“弱结构近似”，我们均匀地划分周期，无需 S1/S2 标注。
    虽然与生理边界不完全对齐，但这在不同周期之间提供了统计上一致的表示。
    
    参数:
        cycle: 心动周期信号
        num_substructures: 子结构片段数量 (K)
        
    返回:
        K 个子结构信号的列表
        
    注意:
        对于典型的 K=4 的心动周期:
        - 片段 0: 近似覆盖 S1 区域
        - 片段 1: 近似覆盖收缩期
        - 片段 2: 近似覆盖 S2 区域
        - 片段 3: 近似覆盖舒张期
        
        这是一种统计近似，在聚合情况下对对比学习效果良好，
        即使单个周期有所不同。
    """
    length = len(cycle)
    substructures = []
    
    for i in range(num_substructures):
        start = i * length // num_substructures
        end = (i + 1) * length // num_substructures
        substructures.append(cycle[start:end].copy())
    
    return substructures


def split_substructures_with_overlap(
    cycle: np.ndarray,
    num_substructures: int = 4,
    overlap_ratio: float = 0.25
) -> List[np.ndarray]:
    """
    将周期分割为重叠的子结构，以实现更平滑的过渡。
    
    重叠有助于模型学习心动阶段之间的过渡特征
    （例如，从 S1 到收缩期的过渡）。
    
    参数:
        cycle: 心动周期信号
        num_substructures: 片段数量
        overlap_ratio: 相邻片段之间的重叠 (0 到 0.5)
        
    返回:
        重叠子结构信号的列表
    """
    length = len(cycle)
    base_length = length // num_substructures
    overlap_samples = int(base_length * overlap_ratio)
    
    substructures = []
    
    for i in range(num_substructures):
        # Calculate start and end with overlap
        nominal_start = i * base_length
        nominal_end = (i + 1) * base_length
        
        # Extend boundaries by overlap amount
        start = max(0, nominal_start - overlap_samples)
        end = min(length, nominal_end + overlap_samples)
        
        substructures.append(cycle[start:end].copy())
    
    return substructures


# ============== Quality Assessment ==============

def compute_snr(
    signal_data: np.ndarray,
    sample_rate: int,
    signal_band: Tuple[float, float] = (25, 150),
    noise_band: Tuple[float, float] = (200, 400)
) -> float:
    """
    估计 PCG 信号的信噪比。
    
    使用频带能量比作为 SNR 代理:
    - 信号频带: 25-150 Hz (主要心音频率)
    - 噪声频带: 200-400 Hz (通常被噪声主导)
    
    参数:
        signal_data: 输入信号
        sample_rate: 采样率 (Hz)
        signal_band: 信号能量的频率范围
        noise_band: 噪声能量的频率范围
        
    返回:
        以 dB 为单位的估计 SNR
    """
    # Compute power spectral density
    freqs, psd = signal.welch(signal_data, sample_rate, nperseg=min(len(signal_data), 1024))
    
    # Find frequency bins for signal and noise bands
    signal_mask = (freqs >= signal_band[0]) & (freqs <= signal_band[1])
    noise_mask = (freqs >= noise_band[0]) & (freqs <= noise_band[1])
    
    # Compute band energies
    signal_energy = np.sum(psd[signal_mask])
    noise_energy = np.sum(psd[noise_mask])
    
    # Avoid division by zero
    if noise_energy < 1e-10:
        return 60.0  # Return high SNR if no noise detected
    
    # Compute SNR in dB
    snr_db = 10 * np.log10(signal_energy / noise_energy)
    
    return snr_db


def assess_cycle_quality(
    cycle: np.ndarray,
    sample_rate: int,
    min_snr_db: float = 10.0
) -> Dict[str, Union[bool, float]]:
    """
    评估提取的心动周期的质量。
    
    参数:
        cycle: 心动周期信号
        sample_rate: 采样率 (Hz)
        min_snr_db: 最小可接受 SNR
        
    返回:
        包含质量指标的字典:
        - valid: 周期是否通过质量检查
        - snr_db: 估计 SNR
        - peak_ratio: 最大值与平均振幅之比（检测削波）
        - zero_ratio: 接近零样本的比例（检测丢失）
    """
    # Compute SNR
    snr_db = compute_snr(cycle, sample_rate)
    
    # Check for clipping (peak ratio too high suggests clipping)
    peak_ratio = np.max(np.abs(cycle)) / (np.mean(np.abs(cycle)) + 1e-8)
    
    # Check for dropout (too many near-zero samples)
    zero_threshold = 0.01 * np.max(np.abs(cycle))
    zero_ratio = np.sum(np.abs(cycle) < zero_threshold) / len(cycle)
    
    # Quality decision
    valid = (
        snr_db >= min_snr_db and
        peak_ratio < 50.0 and  # Not severely clipped
        zero_ratio < 0.3  # Less than 30% dropout
    )
    
    return {
        "valid": valid,
        "snr_db": snr_db,
        "peak_ratio": peak_ratio,
        "zero_ratio": zero_ratio
    }


# ============== Full Pipeline ==============

def process_recording(
    file_path: Union[str, Path],
    sample_rate: int = 5000,
    num_substructures: int = 4,
    target_cycle_length: int = 4000,
    min_cycle_duration: float = 0.4,
    max_cycle_duration: float = 1.5,
    min_snr_db: float = 10.0,
    filter_quality: bool = True
) -> Dict[str, Union[List[np.ndarray], Dict]]:
    """
    单个 PCG 录音的完整处理管道。
    
    此函数实现完整的预处理工作流:
    1. 加载和重新采样音频
    2. 应用带通滤波器
    3. 归一化信号
    4. 计算能量包络
    5. 检测心动周期峰值
    6. 提取和验证周期
    7. 将周期分割为子结构
    
    参数:
        file_path: 音频文件路径
        sample_rate: 目标采样率
        num_substructures: 子结构片段数量 (K)
        target_cycle_length: 每个周期的目标长度（样本数）
        min_cycle_duration: 最小有效周期持续时间（秒）
        max_cycle_duration: 最大有效周期持续时间（秒）
        min_snr_db: 质量过滤的最小 SNR
        filter_quality: 是否应用质量过滤
        
    返回:
        包含以下内容的字典:
        - cycles: 处理后的心动周期列表
        - substructures: 列表的列表，每个周期的子结构
        - metadata: 处理统计和参数
    """
    # Step 1: 加载音频
    audio, sr = load_audio(file_path, target_sr=sample_rate)
    
    # Step 2: 带通滤波
    filtered = bandpass_filter(audio, sr)
    
    # Step 3: 归一化
    normalized = normalize_signal(filtered, method="zscore")
    
    # Step 4: 计算能量包络
    envelope, time_axis = compute_energy_envelope(
        normalized, sr,
        frame_length_ms=20.0,
        hop_length_ms=10.0,
        method="shannon"
    )
    hop_length = int(10.0 * sr / 1000)  # 10ms hop
    
    # Step 5: 检测峰值
    peaks = detect_peaks_adaptive(envelope, sr, hop_length)
    
    # Step 6: 提取周期
    cycles_raw = extract_cycles(
        normalized, peaks, sr,
        min_duration_sec=min_cycle_duration,
        max_duration_sec=max_cycle_duration,
        target_length=target_cycle_length,
        padding_mode="zero"
    )
    
    # Step 7: 质量过滤和子结构提取
    cycles = []
    substructures = []
    quality_stats = {"total": len(cycles_raw), "passed": 0, "failed": 0}
    
    for cycle in cycles_raw:
        if filter_quality:
            quality = assess_cycle_quality(cycle, sr, min_snr_db)
            if not quality["valid"]:
                quality_stats["failed"] += 1
                continue
        
        cycles.append(cycle)
        subs = split_substructures(cycle, num_substructures)
        substructures.append(subs)
        quality_stats["passed"] += 1
    
    # Compile metadata
    metadata = {
        "file_path": str(file_path),
        "sample_rate": sr,
        "original_duration_sec": len(audio) / sr,
        "num_peaks_detected": len(peaks),
        "num_cycles_extracted": len(cycles),
        "target_cycle_length": target_cycle_length,
        "num_substructures": num_substructures,
        "quality_stats": quality_stats
    }
    
    return {
        "cycles": cycles,
        "substructures": substructures,
        "metadata": metadata
    }
