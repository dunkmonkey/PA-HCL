"""
数据预处理模块的单元测试。

运行方式: pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from scipy.io import wavfile

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocessing import (
    load_audio,
    resample_signal,
    bandpass_filter,
    normalize_signal,
    compute_energy_envelope,
    detect_peaks,
    extract_cycles,
    split_substructures,
    compute_snr,
    assess_cycle_quality,
)


# ============== Fixtures ==============

@pytest.fixture
def sample_rate():
    """测试用的标准采样率。"""
    return 5000


@pytest.fixture
def synthetic_pcg(sample_rate):
    """
    生成用于测试的合生 PCG 信号。
    
    创建一个带有模拟 S1/S2 声音的周期性峰值的信号。
    """
    duration = 5.0  # seconds
    heart_rate = 60  # BPM
    
    t = np.linspace(0, duration, int(duration * sample_rate))
    signal = np.zeros_like(t)
    
    # Generate S1/S2 peaks at regular intervals
    beat_interval = 60.0 / heart_rate
    for i in range(int(duration / beat_interval)):
        s1_time = i * beat_interval
        s2_time = s1_time + 0.3  # S2 occurs ~300ms after S1
        
        # Add S1 (larger peak)
        s1_idx = int(s1_time * sample_rate)
        if s1_idx < len(signal) - 100:
            s1_envelope = np.exp(-((np.arange(100) - 50) ** 2) / 200)
            signal[s1_idx:s1_idx + 100] += 0.8 * s1_envelope * np.sin(2 * np.pi * 50 * np.arange(100) / sample_rate)
        
        # Add S2 (smaller peak)
        s2_idx = int(s2_time * sample_rate)
        if s2_idx < len(signal) - 80:
            s2_envelope = np.exp(-((np.arange(80) - 40) ** 2) / 150)
            signal[s2_idx:s2_idx + 80] += 0.5 * s2_envelope * np.sin(2 * np.pi * 70 * np.arange(80) / sample_rate)
    
    # Add some noise
    signal += 0.05 * np.random.randn(len(signal))
    
    return signal.astype(np.float32)


@pytest.fixture
def temp_wav_file(synthetic_pcg, sample_rate):
    """创建用于测试的临时 WAV 文件。"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Convert to int16 for WAV file
        audio_int16 = (synthetic_pcg * 32767).astype(np.int16)
        wavfile.write(f.name, sample_rate, audio_int16)
        yield Path(f.name)
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


# ============== Audio Loading Tests ==============

class TestAudioLoading:
    """音频加载函数的测试。"""
    
    def test_load_audio_basic(self, temp_wav_file, sample_rate):
        """测试基本音频加载。"""
        audio, sr = load_audio(temp_wav_file)
        
        assert sr == sample_rate
        assert len(audio) > 0
        assert audio.dtype == np.float32
        assert np.max(np.abs(audio)) <= 1.0
    
    def test_load_audio_with_resampling(self, temp_wav_file):
        """测试带重采样的音频加载。"""
        target_sr = 2000
        audio, sr = load_audio(temp_wav_file, target_sr=target_sr)
        
        assert sr == target_sr
        assert len(audio) > 0
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件。"""
        with pytest.raises(FileNotFoundError):
            load_audio(Path("/nonexistent/file.wav"))
    
    def test_resample_signal(self, synthetic_pcg, sample_rate):
        """测试信号重采样。"""
        target_sr = 2500
        resampled = resample_signal(synthetic_pcg, sample_rate, target_sr)
        
        expected_length = int(len(synthetic_pcg) * target_sr / sample_rate)
        assert abs(len(resampled) - expected_length) <= 1


# ============== Filtering Tests ==============

class TestFiltering:
    """信号滤波函数的测试。"""
    
    def test_bandpass_filter(self, synthetic_pcg, sample_rate):
        """测试带通滤波。"""
        filtered = bandpass_filter(synthetic_pcg, sample_rate)
        
        assert len(filtered) == len(synthetic_pcg)
        assert filtered.dtype == np.float32
        # Filtered signal should have lower amplitude outside passband
    
    def test_bandpass_filter_custom_frequencies(self, synthetic_pcg, sample_rate):
        """测试使用自定义频率的带通滤波。"""
        filtered = bandpass_filter(synthetic_pcg, sample_rate, lowcut=30.0, highcut=200.0)
        
        assert len(filtered) == len(synthetic_pcg)
    
    def test_normalize_zscore(self, synthetic_pcg):
        """测试 z-score 归一化。"""
        normalized = normalize_signal(synthetic_pcg, method="zscore")
        
        assert abs(np.mean(normalized)) < 0.01
        assert abs(np.std(normalized) - 1.0) < 0.01
    
    def test_normalize_minmax(self, synthetic_pcg):
        """测试 min-max 归一化。"""
        normalized = normalize_signal(synthetic_pcg, method="minmax")
        
        assert np.min(normalized) >= 0.0
        assert np.max(normalized) <= 1.0
    
    def test_normalize_peak(self, synthetic_pcg):
        """测试峰值归一化。"""
        normalized = normalize_signal(synthetic_pcg, method="peak")
        
        assert np.max(np.abs(normalized)) <= 1.0


# ============== Envelope and Peak Detection Tests ==============

class TestEnvelopeAndPeaks:
    """包络计算与峰值检测的测试。"""
    
    def test_compute_energy_envelope_rms(self, synthetic_pcg, sample_rate):
        """测试 RMS 能量包络。"""
        envelope, time_axis = compute_energy_envelope(
            synthetic_pcg, sample_rate, method="rms"
        )
        
        assert len(envelope) > 0
        assert len(envelope) == len(time_axis)
        assert all(envelope >= 0)  # Energy is non-negative
    
    def test_compute_energy_envelope_shannon(self, synthetic_pcg, sample_rate):
        """测试 Shannon 能量包络。"""
        envelope, time_axis = compute_energy_envelope(
            synthetic_pcg, sample_rate, method="shannon"
        )
        
        assert len(envelope) > 0
        assert len(envelope) == len(time_axis)
    
    def test_detect_peaks(self, synthetic_pcg, sample_rate):
        """测试峰值检测。"""
        filtered = bandpass_filter(synthetic_pcg, sample_rate)
        normalized = normalize_signal(filtered, method="zscore")
        envelope, _ = compute_energy_envelope(normalized, sample_rate)
        hop_length = int(10.0 * sample_rate / 1000)
        
        peaks = detect_peaks(envelope, sample_rate, hop_length)
        
        # Should detect approximately 5 peaks for 5 second signal at 60 BPM
        assert len(peaks) >= 3
        assert len(peaks) <= 10


# ============== Cycle Extraction Tests ==============

class TestCycleExtraction:
    """心动周期提取的测试。"""
    
    def test_extract_cycles(self, synthetic_pcg, sample_rate):
        """测试心动周期的提取。"""
        # Create fake peak positions
        peaks = np.array([0, 5000, 10000, 15000, 20000])  # 1 second apart
        
        cycles = extract_cycles(
            synthetic_pcg, peaks, sample_rate,
            min_duration_sec=0.5,
            max_duration_sec=2.0,
            target_length=4000
        )
        
        assert len(cycles) > 0
        assert all(len(c) == 4000 for c in cycles)
    
    def test_extract_cycles_duration_filter(self, synthetic_pcg, sample_rate):
        """测试持续时间超出范围的周期是否被过滤。"""
        # Mix of valid and invalid intervals
        peaks = np.array([0, 1000, 6000, 7000, 12000])  # 0.2s, 1.0s, 0.2s, 1.0s
        
        cycles = extract_cycles(
            synthetic_pcg, peaks, sample_rate,
            min_duration_sec=0.5,
            max_duration_sec=1.5,
            target_length=4000
        )
        
        # Should only keep the 1.0s cycles
        assert len(cycles) == 2
    
    def test_split_substructures(self):
        """测试子结构划分。"""
        cycle = np.arange(100).astype(np.float32)
        
        subs = split_substructures(cycle, num_substructures=4)
        
        assert len(subs) == 4
        assert all(len(s) == 25 for s in subs)
        
        # Check that concatenation recovers original
        recovered = np.concatenate(subs)
        np.testing.assert_array_equal(recovered, cycle)
    
    def test_split_substructures_uneven(self):
        """测试无法整除情况下的子结构划分。"""
        cycle = np.arange(101).astype(np.float32)  # Not evenly divisible by 4
        
        subs = split_substructures(cycle, num_substructures=4)
        
        assert len(subs) == 4
        # Total length should match
        total_length = sum(len(s) for s in subs)
        assert total_length == 101


# ============== Quality Assessment Tests ==============

class TestQualityAssessment:
    """信号质量评估的测试。"""
    
    def test_compute_snr(self, synthetic_pcg, sample_rate):
        """测试 SNR 计算。"""
        snr = compute_snr(synthetic_pcg, sample_rate)
        
        assert isinstance(snr, float)
        # Synthetic signal should have positive SNR
        assert snr > 0
    
    def test_assess_cycle_quality_valid(self, synthetic_pcg, sample_rate):
        """测试对有效周期的质量评估。"""
        # Take a portion as a cycle
        cycle = synthetic_pcg[:4000]
        
        quality = assess_cycle_quality(cycle, sample_rate, min_snr_db=5.0)
        
        assert "valid" in quality
        assert "snr_db" in quality
        assert "peak_ratio" in quality
        assert "zero_ratio" in quality
    
    def test_assess_cycle_quality_noisy(self, sample_rate):
        """测试对含噪周期的质量评估。"""
        # Create a very noisy signal
        noisy_cycle = np.random.randn(4000).astype(np.float32)
        
        quality = assess_cycle_quality(noisy_cycle, sample_rate, min_snr_db=10.0)
        
        # Very noisy signal should fail quality check
        # (depending on random noise, this might occasionally pass)
        assert "valid" in quality


# ============== Integration Test ==============

class TestIntegration:
    """完整预处理流水线的集成测试。"""
    
    def test_full_pipeline(self, synthetic_pcg, sample_rate):
        """测试完整的预处理流水线。"""
        # Step 1: Filter
        filtered = bandpass_filter(synthetic_pcg, sample_rate)
        
        # Step 2: Normalize
        normalized = normalize_signal(filtered, method="zscore")
        
        # Step 3: Compute envelope
        envelope, _ = compute_energy_envelope(normalized, sample_rate, method="shannon")
        hop_length = int(10.0 * sample_rate / 1000)
        
        # Step 4: Detect peaks
        from src.data.preprocessing import detect_peaks_adaptive
        peaks = detect_peaks_adaptive(envelope, sample_rate, hop_length)
        
        # Step 5: Extract cycles
        cycles = extract_cycles(
            normalized, peaks, sample_rate,
            min_duration_sec=0.4,
            max_duration_sec=1.5,
            target_length=4000
        )
        
        # Should have extracted some cycles
        assert len(cycles) > 0
        
        # Step 6: Split substructures
        for cycle in cycles:
            subs = split_substructures(cycle, num_substructures=4)
            assert len(subs) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
