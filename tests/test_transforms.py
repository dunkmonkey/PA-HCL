"""
数据增强变换的单元测试。

运行方式: pytest tests/test_transforms.py -v
"""

import numpy as np
import pytest
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.transforms import (
    Compose,
    RandomApply,
    RandomChoice,
    TimeShift,
    AmplitudeScale,
    GaussianNoise,
    TimeStretch,
    RandomCrop,
    Reverse,
    FrequencyMask,
    TimeMask,
    LowPassFilter,
    HighPassFilter,
    PitchShift,
    AddColoredNoise,
    SimulateRespiratoryNoise,
    get_pretrain_transforms,
    get_finetune_transforms,
    get_eval_transforms,
    SubstructureAugmentor,
)


# ============== Fixtures ==============

@pytest.fixture
def sample_rate():
    """测试使用的标准采样率。"""
    return 5000


@pytest.fixture
def sample_signal(sample_rate):
    """生成用于测试的样本信号。"""
    duration = 0.8  # seconds
    t = np.linspace(0, duration, int(duration * sample_rate))
    # Create a signal with multiple frequency components
    signal = (
        0.5 * np.sin(2 * np.pi * 50 * t) +  # 50 Hz
        0.3 * np.sin(2 * np.pi * 100 * t) +  # 100 Hz
        0.2 * np.sin(2 * np.pi * 200 * t)    # 200 Hz
    )
    return signal.astype(np.float32)


# ============== Time Domain Transform Tests ==============

class TestTimeDomainTransforms:
    """时域增强的测试。"""
    
    def test_time_shift(self, sample_signal, sample_rate):
        """测试时间偏移变换。"""
        transform = TimeShift(max_shift_ratio=0.1, p=1.0)
        augmented = transform(sample_signal, sample_rate)
        
        assert len(augmented) == len(sample_signal)
        assert augmented.dtype == sample_signal.dtype
        # Signal should be different (shifted)
        assert not np.allclose(augmented, sample_signal)
    
    def test_time_shift_zero_shift(self, sample_signal, sample_rate):
        """测试最大位移为 0 的时间偏移。"""
        transform = TimeShift(max_shift_ratio=0.0, p=1.0)
        augmented = transform(sample_signal, sample_rate)
        
        # Should be identical
        np.testing.assert_array_equal(augmented, sample_signal)
    
    def test_amplitude_scale(self, sample_signal, sample_rate):
        """测试幅度缩放。"""
        transform = AmplitudeScale(scale_range=(0.5, 0.5), p=1.0)
        augmented = transform(sample_signal, sample_rate)
        
        # Should be exactly half amplitude
        np.testing.assert_allclose(augmented, sample_signal * 0.5)
    
    def test_amplitude_scale_range(self, sample_signal, sample_rate):
        """测试带范围的幅度缩放。"""
        transform = AmplitudeScale(scale_range=(0.8, 1.2), p=1.0)
        augmented = transform(sample_signal, sample_rate)
        
        # Amplitude should be within expected range
        original_max = np.max(np.abs(sample_signal))
        augmented_max = np.max(np.abs(augmented))
        scale = augmented_max / original_max
        
        assert 0.8 <= scale <= 1.2
    
    def test_gaussian_noise(self, sample_signal, sample_rate):
        """测试高斯噪声添加。"""
        transform = GaussianNoise(snr_range=(20, 20), p=1.0)
        augmented = transform(sample_signal, sample_rate)
        
        assert len(augmented) == len(sample_signal)
        # Should be different due to noise
        assert not np.allclose(augmented, sample_signal)
    
    def test_time_stretch_faster(self, sample_signal, sample_rate):
        """测试时间拉伸 (加快)。"""
        transform = TimeStretch(speed_range=(1.1, 1.1), p=1.0)
        augmented = transform(sample_signal, sample_rate)
        
        # Length should be preserved (internal resampling)
        assert len(augmented) == len(sample_signal)
    
    def test_time_stretch_slower(self, sample_signal, sample_rate):
        """测试时间拉伸 (减慢)。"""
        transform = TimeStretch(speed_range=(0.9, 0.9), p=1.0)
        augmented = transform(sample_signal, sample_rate)
        
        assert len(augmented) == len(sample_signal)
    
    def test_random_crop(self, sample_signal, sample_rate):
        """测试随机裁剪。"""
        transform = RandomCrop(crop_ratio=(0.8, 0.9), p=1.0)
        augmented = transform(sample_signal, sample_rate)
        
        # Length should be preserved (padded back)
        assert len(augmented) == len(sample_signal)
    
    def test_reverse(self, sample_signal, sample_rate):
        """测试信号反转。"""
        transform = Reverse(p=1.0)
        augmented = transform(sample_signal, sample_rate)
        
        np.testing.assert_array_equal(augmented, sample_signal[::-1])


# ============== Frequency Domain Transform Tests ==============

class TestFrequencyDomainTransforms:
    """频域增强的测试。"""
    
    def test_frequency_mask(self, sample_signal, sample_rate):
        """测试频率掩蔽。"""
        transform = FrequencyMask(max_mask_ratio=0.2, num_masks=1, p=1.0)
        augmented = transform(sample_signal, sample_rate)
        
        assert len(augmented) == len(sample_signal)
        assert augmented.dtype == np.float32
    
    def test_time_mask(self, sample_signal, sample_rate):
        """测试时间掩蔽。"""
        transform = TimeMask(max_mask_ratio=0.1, max_mask_width_ms=30, num_masks=2, p=1.0)
        augmented = transform(sample_signal, sample_rate)
        
        assert len(augmented) == len(sample_signal)
        # Some samples should be zero
        assert np.sum(augmented == 0) > 0
    
    def test_low_pass_filter(self, sample_signal, sample_rate):
        """测试低通滤波。"""
        transform = LowPassFilter(cutoff_range=(200, 200), p=1.0)
        augmented = transform(sample_signal, sample_rate)
        
        assert len(augmented) == len(sample_signal)
        assert augmented.dtype == np.float32
    
    def test_high_pass_filter(self, sample_signal, sample_rate):
        """测试高通滤波。"""
        transform = HighPassFilter(cutoff_range=(30, 30), p=1.0)
        augmented = transform(sample_signal, sample_rate)
        
        assert len(augmented) == len(sample_signal)
        assert augmented.dtype == np.float32


# ============== Advanced Transform Tests ==============

class TestAdvancedTransforms:
    """高级增强算子的测试。"""
    
    def test_pitch_shift(self, sample_signal, sample_rate):
        """测试音高偏移。"""
        transform = PitchShift(shift_range=(1, 1), p=1.0)  # +1 semitone
        augmented = transform(sample_signal, sample_rate)
        
        assert len(augmented) == len(sample_signal)
        assert augmented.dtype == np.float32
    
    def test_colored_noise_pink(self, sample_signal, sample_rate):
        """测试加入粉噪声。"""
        transform = AddColoredNoise(snr_range=(20, 20), noise_type="pink", p=1.0)
        augmented = transform(sample_signal, sample_rate)
        
        assert len(augmented) == len(sample_signal)
        assert not np.allclose(augmented, sample_signal)
    
    def test_colored_noise_brown(self, sample_signal, sample_rate):
        """测试加入棕色噪声。"""
        transform = AddColoredNoise(snr_range=(20, 20), noise_type="brown", p=1.0)
        augmented = transform(sample_signal, sample_rate)
        
        assert len(augmented) == len(sample_signal)
    
    def test_respiratory_noise(self, sample_signal, sample_rate):
        """测试呼吸噪声模拟。"""
        transform = SimulateRespiratoryNoise(intensity_range=(0.1, 0.1), p=1.0)
        augmented = transform(sample_signal, sample_rate)
        
        assert len(augmented) == len(sample_signal)
        assert not np.allclose(augmented, sample_signal)


# ============== Composition Tests ==============

class TestComposition:
    """变换组合的测试。"""
    
    def test_compose(self, sample_signal, sample_rate):
        """测试多个变换的组合。"""
        transforms = Compose([
            TimeShift(max_shift_ratio=0.1, p=1.0),
            AmplitudeScale(scale_range=(0.9, 1.1), p=1.0),
        ])
        augmented = transforms(sample_signal, sample_rate)
        
        assert len(augmented) == len(sample_signal)
    
    def test_random_apply(self, sample_signal, sample_rate):
        """测试随机应用包装器。"""
        transform = RandomApply(TimeShift(max_shift_ratio=0.1, p=1.0), p=0.5)
        
        # Run multiple times to test randomness
        results = [transform(sample_signal.copy(), sample_rate) for _ in range(10)]
        
        # At least some should be different
        all_same = all(np.allclose(r, sample_signal) for r in results)
        assert not all_same
    
    def test_random_choice(self, sample_signal, sample_rate):
        """测试在多个变换中随机选择。"""
        transforms = RandomChoice([
            AmplitudeScale(scale_range=(0.5, 0.5), p=1.0),
            AmplitudeScale(scale_range=(2.0, 2.0), p=1.0),
        ])
        
        results = [transforms(sample_signal.copy(), sample_rate) for _ in range(10)]
        
        # Should get a mix of half and double amplitude
        max_vals = [np.max(np.abs(r)) for r in results]
        assert min(max_vals) < np.max(np.abs(sample_signal))
        assert max(max_vals) > np.max(np.abs(sample_signal))


# ============== Factory Function Tests ==============

class TestFactoryFunctions:
    """变换工厂函数的测试。"""
    
    def test_get_pretrain_transforms(self, sample_signal, sample_rate):
        """测试预训练变换工厂函数。"""
        transforms = get_pretrain_transforms(sample_rate=sample_rate)
        augmented = transforms(sample_signal, sample_rate)
        
        assert len(augmented) == len(sample_signal)
    
    def test_get_finetune_transforms(self, sample_signal, sample_rate):
        """测试微调阶段变换工厂函数。"""
        transforms = get_finetune_transforms(sample_rate=sample_rate)
        augmented = transforms(sample_signal, sample_rate)
        
        assert len(augmented) == len(sample_signal)
    
    def test_get_eval_transforms(self, sample_signal, sample_rate):
        """测试评估阶段变换工厂函数 (应为恒等变换)。"""
        transforms = get_eval_transforms()
        augmented = transforms(sample_signal, sample_rate)
        
        np.testing.assert_array_equal(augmented, sample_signal)
    
    def test_pretrain_transforms_with_config(self, sample_signal, sample_rate):
        """测试带自定义配置的预训练变换。"""
        config = {
            "time_shift": {"enabled": True, "max_shift_ratio": 0.2},
            "amplitude_scale": {"enabled": True, "min_scale": 0.7, "max_scale": 1.3},
            "gaussian_noise": {"enabled": False},
        }
        transforms = get_pretrain_transforms(config=config, sample_rate=sample_rate)
        augmented = transforms(sample_signal, sample_rate)
        
        assert len(augmented) == len(sample_signal)


# ============== Substructure Augmentor Tests ==============

class TestSubstructureAugmentor:
    """子结构增强的测试。"""
    
    def test_substructure_augmentor_independent(self, sample_signal, sample_rate):
        """测试相互独立的子结构增强。"""
        # Split signal into substructures
        subs = [sample_signal[i*1000:(i+1)*1000] for i in range(4)]
        
        transform = AmplitudeScale(scale_range=(0.8, 1.2), p=1.0)
        augmentor = SubstructureAugmentor(transform, consistent=False)
        
        augmented_subs = augmentor(subs, sample_rate)
        
        assert len(augmented_subs) == len(subs)
        # Each substructure should be augmented independently
        # (different random scales)
    
    def test_substructure_augmentor_consistent(self, sample_signal, sample_rate):
        """测试一致性的子结构增强。"""
        # Split signal into substructures
        subs = [sample_signal[i*1000:(i+1)*1000] for i in range(4)]
        
        transform = AmplitudeScale(scale_range=(0.5, 0.5), p=1.0)  # Fixed scale
        augmentor = SubstructureAugmentor(transform, consistent=True)
        
        augmented_subs = augmentor(subs, sample_rate)
        
        assert len(augmented_subs) == len(subs)


# ============== Probability Tests ==============

class TestProbability:
    """变换概率行为的测试。"""
    
    def test_transform_probability_zero(self, sample_signal, sample_rate):
        """测试 p=0 (从不应用) 的变换。"""
        transform = AmplitudeScale(scale_range=(0.5, 0.5), p=0.0)
        
        for _ in range(10):
            augmented = transform(sample_signal.copy(), sample_rate)
            np.testing.assert_array_equal(augmented, sample_signal)
    
    def test_transform_probability_one(self, sample_signal, sample_rate):
        """测试 p=1 (总是应用) 的变换。"""
        transform = AmplitudeScale(scale_range=(0.5, 0.5), p=1.0)
        
        for _ in range(10):
            augmented = transform(sample_signal.copy(), sample_rate)
            np.testing.assert_allclose(augmented, sample_signal * 0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
