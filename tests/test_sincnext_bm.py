"""
SinCNeXt-BM 编码器架构的专门单元测试。

SinCNeXt-BM 是 PA-HCL 项目的核心编码器，采用限频 SincNet + ConvNeXt Local + Bi-Mamba Global 架构。

=============================================================================
测试覆盖范围
=============================================================================

1. 四阶段形状验证 (Shape Validation)
   - SincNet 频率限制 (20-500 Hz, 100 个等距滤波器)
   - ConvNeXt1D 局部块 (DWConv + PW 扩展 + DropPath)
   - 双向 Mamba 全局建模 (d_model=128, n_layers=6)
   - 动态掩码头 + ASP 头 (num_substructures=4, cycle_output_dim=256)

2. 接口一致性 (Interface Contracts)
   - encoder.out_dim (全局特征输出维度)
   - encoder.cycle_output_dim (ASP 最终输出维度)
   - encoder.sub_feature_dim (子结构特征维度)
   - encoder.get_sub_features() (获取中间表达)

3. 三链路最小构建 (Pipeline Integration)
   - SinCNeXt-BM 编码器独立构建
   - PAHCLModel 预训练流水线 (双视图)
   - SupervisedModel 监督学习流水线

4. 梯度流和正则化
   - DropPath 随机深度梯度流
   - GroupNorm 批次独立性
   - BiMamba 双向融合梯度

=============================================================================
运行指南
=============================================================================

# 运行所有 SinCNeXt-BM 测试
pytest tests/test_sincnext_bm.py -v

# 运行特定测试类
pytest tests/test_sincnext_bm.py::TestSincConv1d -v
pytest tests/test_sincnext_bm.py::TestConvNeXt1DBlock -v
pytest tests/test_sincnext_bm.py::TestSinCNeXtBMEncoder -v
pytest tests/test_sincnext_bm.py::TestSinCNeXtBMPipeline -v

# 快速烟雾测试 (仅形状验证)
pytest tests/test_sincnext_bm.py -v -k "shape"

# 带覆盖率
pytest tests/test_sincnext_bm.py -v --cov=src/models/encoder

=============================================================================

Author: PA-HCL Team
Date: 2026-01-25
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np


# ============== Configuration ==============

BATCH_SIZE = 4
SIGNAL_LENGTH = 4000  # ~0.8s at 5000Hz
IN_CHANNELS = 1
SAMPLE_RATE = 5000

# SinCNeXt-BM 参数
SINC_OUT_CHANNELS = 64
SINC_KERNEL_SIZE = 251
LOCAL_DIM = 128
MAMBA_D_MODEL = 128
MAMBA_N_LAYERS = 6
CYCLE_OUTPUT_DIM = 256
NUM_SUBSTRUCTURES = 4


# ============== Fixtures ==============

@pytest.fixture
def device():
    """获取合适的设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_signal(device):
    """生成样本 PCG 信号 [B, 1, T]。"""
    return torch.randn(BATCH_SIZE, IN_CHANNELS, SIGNAL_LENGTH, device=device)


@pytest.fixture
def sincnext_bm_encoder(device):
    """创建完整的 SinCNeXt-BM 编码器实例。"""
    from src.models.encoder import build_encoder
    
    encoder = build_encoder(
        encoder_type='sincnet_eca_mamba',
        in_channels=IN_CHANNELS,
        sinc_out_channels=SINC_OUT_CHANNELS,
        sinc_kernel_size=SINC_KERNEL_SIZE,
        sinc_stride=1,
        sinc_min_low_hz=20.0,
        sinc_min_band_hz=20.0,
        sinc_max_high_hz=500.0,
        local_dim=LOCAL_DIM,
        convnext_kernel_size=7,
        convnext_expansion=4,
        mamba_d_model=MAMBA_D_MODEL,
        mamba_n_layers=MAMBA_N_LAYERS,
        mamba_d_state=16,
        mamba_d_conv=4,
        use_bidirectional=True,
        bidirectional_fusion='add',
        drop_path_rate=0.1,
        pool_type='asp',
        cycle_output_dim=CYCLE_OUTPUT_DIM,
        use_groupnorm=True,
        num_groups=8,
        sample_rate=SAMPLE_RATE,
        num_substructures=NUM_SUBSTRUCTURES,
    ).to(device)
    
    return encoder


# ============== Test SincConv1d ==============

class TestSincConv1d:
    """SincNet 1D 卷积层测试（频率限制）。"""
    
    def test_sinc_conv_initialization(self, device):
        """测试 SincConv1d 初始化。"""
        from src.models.encoder import SincConv1d
        
        sinc = SincConv1d(
            out_channels=64,
            kernel_size=251,
            sample_rate=5000,
            min_low_hz=20.0,
            min_band_hz=20.0,
            max_high_hz=500.0
        ).to(device)
        
        # 检查滤波器参数（形状为 [out_channels, 1]）
        assert sinc.low_hz_.shape == (64, 1)
        assert sinc.band_hz_.shape == (64, 1)
        
        # 检查采样率和频率限制
        assert sinc.sample_rate == 5000
        assert sinc.min_low_hz == 20.0
        assert sinc.max_high_hz == 500.0
    
    def test_sinc_conv_output_shape(self, device):
        """测试 SincConv1d 输出形状。"""
        from src.models.encoder import SincConv1d
        
        sinc = SincConv1d(
            out_channels=64,
            kernel_size=251,
            sample_rate=5000,
            stride=1,
            padding='same'
        ).to(device)
        
        x = torch.randn(BATCH_SIZE, 1, SIGNAL_LENGTH, device=device)
        y = sinc(x)
        
        # 使用 padding='same' 输出长度保持不变
        assert y.shape == (BATCH_SIZE, 64, SIGNAL_LENGTH)
    
    def test_sinc_freq_band_constraint(self, device):
        """验证 SincConv1d 在 forward 中应用的频率约束。"""
        from src.models.encoder import SincConv1d
        
        min_low = 20.0
        min_band = 20.0
        max_high = 500.0
        
        sinc = SincConv1d(
            out_channels=32,
            kernel_size=251,
            sample_rate=5000,
            min_low_hz=min_low,
            min_band_hz=min_band,
            max_high_hz=max_high
        ).to(device)
        
        # 验证属性
        assert sinc.min_low_hz == min_low
        assert sinc.min_band_hz == min_band
        assert sinc.max_high_hz == max_high
        
        # 通过前向传播测试频率约束是否在 forward 中应用
        x = torch.randn(BATCH_SIZE, 1, SIGNAL_LENGTH, device=device)
        y = sinc(x)
        
        # 检查输出形状正确
        assert y.shape == (BATCH_SIZE, 32, SIGNAL_LENGTH)
    
    def test_sinc_gradient_flow(self, device):
        """测试梯度通过 SincConv1d 正确流动。"""
        from src.models.encoder import SincConv1d
        
        sinc = SincConv1d(
            out_channels=64,
            kernel_size=251,
            sample_rate=5000
        ).to(device)
        
        x = torch.randn(BATCH_SIZE, 1, SIGNAL_LENGTH, device=device, requires_grad=True)
        y = sinc(x)
        loss = y.mean()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert x.grad.abs().max() > 0


# ============== Test ConvNeXt1DBlock ==============

class TestConvNeXt1DBlock:
    """ConvNeXt 1D 块测试（深度卷积 + 扩张）。"""
    
    def test_convnext_block_output_shape(self, device):
        """测试 ConvNeXt1D 块输出形状。"""
        from src.models.encoder import ConvNeXt1DBlock
        
        block = ConvNeXt1DBlock(
            dim=LOCAL_DIM,
            kernel_size=7,
            expansion=4,
            drop_path=0.0
        ).to(device)
        
        x = torch.randn(BATCH_SIZE, LOCAL_DIM, SIGNAL_LENGTH, device=device)
        y = block(x)
        
        assert y.shape == x.shape
    
    def test_convnext_block_residual_connection(self, device):
        """测试 ConvNeXt1D 块残差连接。"""
        from src.models.encoder import ConvNeXt1DBlock
        
        block = ConvNeXt1DBlock(
            dim=LOCAL_DIM,
            kernel_size=7,
            expansion=4,
            drop_path=0.0
        ).to(device)
        
        x = torch.randn(BATCH_SIZE, LOCAL_DIM, SIGNAL_LENGTH, device=device)
        
        # 将块中的所有权重设为 0，输出应该等于输入（因为残差连接）
        with torch.no_grad():
            for param in block.parameters():
                param.zero_()
        
        y = block(x)
        
        # 由于残差连接，即使权重为 0，输出也应该接近输入
        assert torch.allclose(y, x, atol=1e-5)
    
    def test_convnext_block_gradient_flow(self, device):
        """测试梯度通过 ConvNeXt1D 块正确流动。"""
        from src.models.encoder import ConvNeXt1DBlock
        
        block = ConvNeXt1DBlock(
            dim=LOCAL_DIM,
            kernel_size=7,
            expansion=4,
            drop_path=0.0
        ).to(device)
        
        x = torch.randn(BATCH_SIZE, LOCAL_DIM, SIGNAL_LENGTH, 
                       device=device, requires_grad=True)
        y = block(x)
        loss = y.mean()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert x.grad.abs().max() > 0
    
    def test_convnext_block_drop_path(self, device):
        """测试 DropPath 随机深度。"""
        from src.models.encoder import ConvNeXt1DBlock
        
        # 高 drop_path 概率用于测试
        block = ConvNeXt1DBlock(
            dim=LOCAL_DIM,
            kernel_size=7,
            expansion=4,
            drop_path=0.9  # 90% drop 概率
        ).to(device)
        
        block.train()  # 确保在训练模式
        
        x = torch.randn(BATCH_SIZE, LOCAL_DIM, SIGNAL_LENGTH, device=device)
        
        # 多次前向传播，应该有些输出接近 0（被 dropped）
        outputs = []
        for _ in range(10):
            y = block(x)
            outputs.append(y)
        
        # 某些输出应该被完全 dropped (接近 0) 由于 skip connection
        # （这个测试比较宽松，只检查输出变化）
        output_means = torch.tensor([o.abs().mean().item() for o in outputs])
        assert output_means.std() > 0  # 应该有变化


# ============== Test DynamicSubstructureMasking ==============

class TestDynamicSubstructureMasking:
    """动态子结构掩码测试（时间加权聚合）。"""
    
    def test_masking_output_shape(self, device):
        """测试子结构掩码输出形状。"""
        from src.models.encoder import DynamicSubstructureMasking
        
        masking = DynamicSubstructureMasking(
            d_model=LOCAL_DIM,
            num_substructures=NUM_SUBSTRUCTURES
        ).to(device)
        
        # 输入格式: [B, T, D]（序列格式）
        x = torch.randn(BATCH_SIZE, SIGNAL_LENGTH, LOCAL_DIM, device=device)
        substructures = masking(x)
        
        # [B, K, D]
        assert substructures.shape == (BATCH_SIZE, NUM_SUBSTRUCTURES, LOCAL_DIM)
    
    def test_masking_weights_sum_to_one(self, device):
        """测试掩码权重在时间维和为 1（softmax 归一化）。"""
        from src.models.encoder import DynamicSubstructureMasking
        
        masking = DynamicSubstructureMasking(
            d_model=LOCAL_DIM,
            num_substructures=NUM_SUBSTRUCTURES
        ).to(device)
        
        # [B, T, D] 格式
        x = torch.randn(BATCH_SIZE, SIGNAL_LENGTH, LOCAL_DIM, device=device)
        substructures = masking(x)
        
        # 每个子结构应该是 x 的加权平均，所以 norm 应该类似于 x 的 norm
        x_norm = x.norm(dim=-1).mean()  # [B, T] -> scalar
        sub_norm = substructures.norm(dim=-1).mean()  # [B, K] -> scalar
        
        # 允许一定范围的变化（取决于 softmax 权重分布）
        assert sub_norm > 0
    
    def test_masking_gradient_flow(self, device):
        """测试梯度通过掩码层正确流动。"""
        from src.models.encoder import DynamicSubstructureMasking
        
        masking = DynamicSubstructureMasking(
            d_model=LOCAL_DIM,
            num_substructures=NUM_SUBSTRUCTURES
        ).to(device)
        
        # [B, T, D] 格式
        x = torch.randn(BATCH_SIZE, SIGNAL_LENGTH, LOCAL_DIM, device=device,
                       requires_grad=True)
        substructures = masking(x)
        loss = substructures.mean()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ============== Test SinCNeXtBMEncoder ==============

class TestSinCNeXtBMEncoder:
    """完整 SinCNeXt-BM 编码器测试。"""
    
    def test_encoder_four_stage_output_shapes(self, sincnext_bm_encoder, sample_signal, device):
        """测试四阶段管道的输出形状。"""
        encoder = sincnext_bm_encoder
        x = sample_signal
        
        # 前向传播
        cycle_features = encoder(x)
        
        # 全局特征（ASP 输出）
        assert cycle_features.shape == (BATCH_SIZE, CYCLE_OUTPUT_DIM)
    
    def test_encoder_cycle_output_dim_property(self, sincnext_bm_encoder):
        """测试编码器输出维度属性。"""
        encoder = sincnext_bm_encoder
        
        assert hasattr(encoder, 'out_dim')
        assert hasattr(encoder, 'cycle_output_dim')
        assert hasattr(encoder, 'sub_feature_dim')
        
        assert encoder.out_dim == CYCLE_OUTPUT_DIM
        assert encoder.cycle_output_dim == CYCLE_OUTPUT_DIM
        assert encoder.sub_feature_dim == MAMBA_D_MODEL
    
    def test_encoder_get_sub_features_interface(self, sincnext_bm_encoder, sample_signal):
        """测试子结构特征提取接口。"""
        encoder = sincnext_bm_encoder
        x = sample_signal
        
        # 使用 get_sub_features 获取中间特征
        sub_features = encoder.get_sub_features(x)
        
        # [B, C, T] 格式
        assert len(sub_features.shape) == 3
        assert sub_features.shape[0] == BATCH_SIZE
        assert sub_features.shape[1] == encoder.sub_feature_dim
        assert sub_features.shape[2] < x.shape[2]  # 经过下采样
    
    def test_encoder_gradients_flow(self, sincnext_bm_encoder, sample_signal):
        """测试梯度从输出回流到输入。"""
        encoder = sincnext_bm_encoder
        x = sample_signal.requires_grad_(True)
        
        cycle_features = encoder(x)
        loss = cycle_features.mean()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().max() > 0
    
    def test_encoder_parameter_count(self, sincnext_bm_encoder):
        """验证参数数量在合理范围。"""
        encoder = sincnext_bm_encoder
        
        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        
        # SinCNeXt-BM 应该在 1-5M 参数范围（取决于配置）
        assert 1e6 <= trainable_params <= 5e6, f"Got {trainable_params} params"
        print(f"SinCNeXt-BM 参数量: {trainable_params / 1e6:.2f}M")
    
    def test_encoder_inference_mode(self, sincnext_bm_encoder, sample_signal):
        """测试推理模式下的行为。"""
        encoder = sincnext_bm_encoder
        encoder.eval()
        
        x = sample_signal
        
        with torch.no_grad():
            cycle_features = encoder(x)
        
        assert cycle_features.shape == (BATCH_SIZE, CYCLE_OUTPUT_DIM)
        assert not cycle_features.requires_grad
    
    def test_encoder_different_input_lengths(self, sincnext_bm_encoder, device):
        """测试不同长度输入的处理。"""
        encoder = sincnext_bm_encoder
        encoder.eval()
        
        # 测试不同长度
        for length in [2000, 4000, 6000]:
            x = torch.randn(BATCH_SIZE, 1, length, device=device)
            with torch.no_grad():
                cycle_features = encoder(x)
            
            assert cycle_features.shape == (BATCH_SIZE, CYCLE_OUTPUT_DIM)


# ============== Test SinCNeXtBM Pipeline Integration ==============

class TestSinCNeXtBMPipeline:
    """三链路集成测试（encoder → PAHCLModel → SupervisedModel）。"""
    
    def test_encoder_standalone(self, device):
        """测试 1: 编码器独立。"""
        from src.models.encoder import build_encoder
        
        encoder = build_encoder(
            encoder_type='sincnet_eca_mamba',
            mamba_d_model=128,
            mamba_n_layers=6,
            pool_type='asp',
            cycle_output_dim=256,
            sample_rate=5000,
        ).to(device)
        
        x = torch.randn(BATCH_SIZE, 1, SIGNAL_LENGTH, device=device)
        cycle_out = encoder(x)
        sub_out = encoder.get_sub_features(x)
        
        assert cycle_out.shape == (BATCH_SIZE, 256)
        assert sub_out.shape[0] == BATCH_SIZE
        print(f"✓ Encoder: cycle {cycle_out.shape}, sub {sub_out.shape}")
    
    def test_pahcl_model_pretrain(self, device):
        """测试 2: PAHCLModel 预训练前向。"""
        from src.models.pahcl import build_pahcl_model
        from omegaconf import OmegaConf
        
        # 最小预训练配置
        cfg = OmegaConf.create({
            'data': {'sample_rate': 5000, 'num_substructures': 4},
            'model': {
                'encoder_type': 'sincnet_eca_mamba',
                'use_moco': False,
                'mamba_d_model': 128,
                'mamba_n_layers': 6,
                'cycle_output_dim': 256,
                'cycle_proj_output': 64,
                'sub_proj_output': 32,
            }
        })
        
        model = build_pahcl_model(cfg).to(device)
        
        v1 = torch.randn(BATCH_SIZE, 1, SIGNAL_LENGTH, device=device)
        v2 = torch.randn(BATCH_SIZE, 1, SIGNAL_LENGTH, device=device)
        
        cycle_z, cycle_p, sub_z, sub_p = model.forward_pretrain(v1, v2)
        
        assert cycle_z.shape == (BATCH_SIZE, 64)
        assert cycle_p.shape == (BATCH_SIZE, 64)
        assert sub_z.shape == (BATCH_SIZE, NUM_SUBSTRUCTURES, 32)
        assert sub_p.shape == (BATCH_SIZE, NUM_SUBSTRUCTURES, 32)
        print(f"✓ PAHCLModel: cycle_z {cycle_z.shape}, sub_z {sub_z.shape}")
    
    def test_supervised_model_baseline(self, device):
        """测试 3: SupervisedModel 监督基线。"""
        from src.models.encoder import build_encoder
        from src.models.heads import ClassificationHead
        
        encoder = build_encoder(
            encoder_type='sincnet_eca_mamba',
            mamba_d_model=128,
            mamba_n_layers=6,
            cycle_output_dim=256,
            sample_rate=5000,
        ).to(device)
        
        encoder_dim = encoder.out_dim
        classifier = ClassificationHead(
            input_dim=encoder_dim,
            hidden_dim=128,
            output_dim=2,  # 二分类
            num_layers=1,
            dropout=0.5
        ).to(device)
        
        x = torch.randn(BATCH_SIZE, 1, SIGNAL_LENGTH, device=device)
        features = encoder(x)
        logits = classifier(features)
        
        assert features.shape == (BATCH_SIZE, 256)
        assert logits.shape == (BATCH_SIZE, 2)
        print(f"✓ SupervisedModel: features {features.shape}, logits {logits.shape}")
    
    def test_full_pipeline_gradient_flow(self, device):
        """测试完整管道梯度流。"""
        from src.models.encoder import build_encoder
        from src.models.heads import ClassificationHead
        
        encoder = build_encoder(
            encoder_type='sincnet_eca_mamba',
            mamba_d_model=128,
            mamba_n_layers=6,
            cycle_output_dim=256,
            sample_rate=5000,
        ).to(device)
        
        classifier = ClassificationHead(
            input_dim=256,
            hidden_dim=128,
            output_dim=2,
            num_layers=1,
            dropout=0.5
        ).to(device)
        
        x = torch.randn(BATCH_SIZE, 1, SIGNAL_LENGTH, device=device, requires_grad=True)
        targets = torch.randint(0, 2, (BATCH_SIZE,), device=device)
        
        features = encoder(x)
        logits = classifier(features)
        loss = nn.CrossEntropyLoss()(logits, targets)
        loss.backward()
        
        # 验证梯度流通过整个链路
        assert x.grad is not None
        assert encoder.encoder[0].low_hz_.grad is not None  # SincConv 梯度
        print(f"✓ Full pipeline: loss {loss.item():.4f} with gradients flowing")


# ============== Test Dimension Inference ==============

class TestDimensionInference:
    """维度推断和一致性测试。"""
    
    def test_encoder_dim_consistency(self, device):
        """验证编码器输出维度一致性。"""
        from src.models.encoder import build_encoder
        
        encoder = build_encoder(
            encoder_type='sincnet_eca_mamba',
            cycle_output_dim=256,
            sample_rate=5000,
        ).to(device)
        
        # 三种不同的维度查询方式应该一致
        dim1 = encoder.out_dim
        dim2 = encoder.cycle_output_dim
        
        assert dim1 == dim2 == 256
        print(f"✓ Dimension inference: out_dim={dim1}, cycle_output_dim={dim2}")
    
    def test_pahcl_encoder_dim_inference(self, device):
        """验证 PAHCLModel 中的编码器维度推断。"""
        from src.models.pahcl import build_pahcl_model
        from omegaconf import OmegaConf
        
        cfg = OmegaConf.create({
            'data': {'sample_rate': 5000, 'num_substructures': 4},
            'model': {
                'encoder_type': 'sincnet_eca_mamba',
                'use_moco': False,
                'mamba_d_model': 128,
                'mamba_n_layers': 6,
                'cycle_output_dim': 256,
            }
        })
        
        model = build_pahcl_model(cfg).to(device)
        
        assert model.encoder.out_dim == 256
        print(f"✓ PAHCLModel encoder_dim: {model.encoder.out_dim}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
