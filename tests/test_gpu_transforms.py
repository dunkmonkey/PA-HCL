"""
GPU 增强模块的单元测试。

测试所有 GPU 增强类的功能和性能。
"""

import pytest
import torch
import numpy as np

from src.data.gpu_transforms import (
    GPUTransformBase,
    GPUTimeShift,
    GPUAmplitudeScale,
    GPUGaussianNoise,
    GPUTimeMask,
    GPUFrequencyMask,
    GPUTimeStretch,
    GPUCompose,
    GPURandomApply,
    GPUAugmentationWrapper,
    get_gpu_pretrain_transforms,
    get_gpu_downstream_transforms,
)


class TestGPUTimeShift:
    """测试 GPU 时间偏移"""
    
    def test_shape_preserved(self):
        """输出形状应与输入相同"""
        x = torch.randn(8, 1, 4000)
        transform = GPUTimeShift(max_shift_ratio=0.1, p=1.0)
        transform.train()
        y = transform(x)
        assert y.shape == x.shape
    
    def test_no_shift_when_eval(self):
        """评估模式下不应用变换"""
        x = torch.randn(8, 1, 4000)
        transform = GPUTimeShift(max_shift_ratio=0.1, p=1.0)
        transform.eval()
        y = transform(x)
        assert torch.allclose(x, y)
    
    def test_shift_applied_when_train(self):
        """训练模式下应用变换"""
        x = torch.randn(8, 1, 4000)
        transform = GPUTimeShift(max_shift_ratio=0.1, p=1.0)
        transform.train()
        y = transform(x)
        # 由于是循环偏移，数据应该不同但能量相同
        assert not torch.allclose(x, y, atol=1e-6)


class TestGPUAmplitudeScale:
    """测试 GPU 幅度缩放"""
    
    def test_shape_preserved(self):
        """输出形状应与输入相同"""
        x = torch.randn(8, 1, 4000)
        transform = GPUAmplitudeScale(scale_range=(0.8, 1.2), p=1.0)
        transform.train()
        y = transform(x)
        assert y.shape == x.shape
    
    def test_scale_range(self):
        """缩放因子应在指定范围内"""
        x = torch.ones(100, 1, 100)  # 全1张量
        transform = GPUAmplitudeScale(scale_range=(0.5, 0.5), p=1.0)
        transform.train()
        y = transform(x)
        # 所有值应该是0.5
        assert torch.allclose(y, torch.ones_like(y) * 0.5)


class TestGPUGaussianNoise:
    """测试 GPU 高斯噪声"""
    
    def test_shape_preserved(self):
        """输出形状应与输入相同"""
        x = torch.randn(8, 1, 4000)
        transform = GPUGaussianNoise(snr_range=(20, 40), p=1.0)
        transform.train()
        y = transform(x)
        assert y.shape == x.shape
    
    def test_noise_added(self):
        """应添加噪声"""
        x = torch.randn(8, 1, 4000)
        transform = GPUGaussianNoise(snr_range=(20, 40), p=1.0)
        transform.train()
        y = transform(x)
        # 输出应该与输入不同
        assert not torch.allclose(x, y)


class TestGPUTimeMask:
    """测试 GPU 时间遮蔽"""
    
    def test_shape_preserved(self):
        """输出形状应与输入相同"""
        x = torch.randn(8, 1, 4000)
        transform = GPUTimeMask(max_mask_ratio=0.1, p=1.0)
        transform.train()
        y = transform(x)
        assert y.shape == x.shape
    
    def test_mask_applied(self):
        """应应用时间遮蔽（部分值为0）"""
        x = torch.ones(8, 1, 4000)
        transform = GPUTimeMask(max_mask_ratio=0.1, max_mask_width=100, p=1.0)
        transform.train()
        y = transform(x)
        # 应该有一些零值
        assert (y == 0).any()


class TestGPUFrequencyMask:
    """测试 GPU 频率遮蔽"""
    
    def test_shape_preserved(self):
        """输出形状应与输入相同"""
        x = torch.randn(8, 1, 4000)
        transform = GPUFrequencyMask(max_mask_ratio=0.15, p=1.0)
        transform.train()
        y = transform(x)
        assert y.shape == x.shape


class TestGPUTimeStretch:
    """测试 GPU 时间拉伸"""
    
    def test_shape_preserved(self):
        """输出形状应与输入相同"""
        x = torch.randn(8, 1, 4000)
        transform = GPUTimeStretch(speed_range=(0.9, 1.1), p=1.0)
        transform.train()
        y = transform(x)
        assert y.shape == x.shape


class TestGPUCompose:
    """测试 GPU 变换组合"""
    
    def test_compose_multiple(self):
        """应依次应用多个变换"""
        x = torch.randn(8, 1, 4000)
        transforms = GPUCompose([
            GPUTimeShift(max_shift_ratio=0.1, p=1.0),
            GPUAmplitudeScale(scale_range=(0.8, 1.2), p=1.0),
        ])
        transforms.train()
        y = transforms(x)
        assert y.shape == x.shape
    
    def test_empty_compose(self):
        """空组合应返回原始输入"""
        x = torch.randn(8, 1, 4000)
        transforms = GPUCompose([])
        y = transforms(x)
        assert torch.allclose(x, y)


class TestFactoryFunctions:
    """测试工厂函数"""
    
    def test_get_gpu_pretrain_transforms(self):
        """预训练增强管道应正常工作"""
        transforms = get_gpu_pretrain_transforms()
        x = torch.randn(8, 1, 4000)
        transforms.train()
        y = transforms(x)
        assert y.shape == x.shape
    
    def test_get_gpu_downstream_transforms_training(self):
        """下游任务训练增强应正常工作"""
        transforms = get_gpu_downstream_transforms(is_training=True)
        assert transforms is not None
        x = torch.randn(8, 1, 4000)
        transforms.train()
        y = transforms(x)
        assert y.shape == x.shape
    
    def test_get_gpu_downstream_transforms_eval(self):
        """下游任务评估不应返回增强"""
        transforms = get_gpu_downstream_transforms(is_training=False)
        assert transforms is None


class TestGPUAugmentationWrapper:
    """测试 GPU 增强包装器"""
    
    def test_dual_view_augmentation(self):
        """应对两个视图分别应用增强"""
        wrapper = GPUAugmentationWrapper(device='cpu')
        view1 = torch.randn(8, 1, 4000)
        view2 = torch.randn(8, 1, 4000)
        
        aug1, aug2 = wrapper(view1, view2)
        
        assert aug1.shape == view1.shape
        assert aug2.shape == view2.shape
    
    def test_train_eval_modes(self):
        """应支持训练/评估模式切换"""
        wrapper = GPUAugmentationWrapper(device='cpu')
        
        wrapper.train()
        assert wrapper.transform.training
        
        wrapper.eval()
        assert not wrapper.transform.training


class TestPerformance:
    """性能测试"""
    
    def test_batch_throughput(self):
        """批量处理性能测试"""
        import time
        
        batch_size = 64
        seq_len = 4000
        x = torch.randn(batch_size, 1, seq_len)
        
        transforms = get_gpu_pretrain_transforms()
        transforms.train()
        
        # 预热
        for _ in range(3):
            _ = transforms(x)
        
        # 计时
        n_iter = 10
        start = time.time()
        for _ in range(n_iter):
            _ = transforms(x)
        elapsed = time.time() - start
        
        throughput = batch_size * n_iter / elapsed
        
        # 应该达到至少 1000 samples/sec
        assert throughput > 1000, f"吞吐量过低: {throughput:.1f} samples/sec"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
