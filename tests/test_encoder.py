"""
编码器架构的单元测试。

这是 PA-HCL 项目编码器模块的单元测试文件。

=============================================================================
测试运行指南 (Test Execution Guide)
=============================================================================

1. 环境准备:
   cd /workspaces/PA-HCL
   pip install -e .
   # 或确保以下依赖已安装:
   pip install torch einops pytest numpy

2. 运行所有编码器测试:
   pytest tests/test_encoder.py -v

3. 运行特定测试类:
   pytest tests/test_encoder.py::TestCNNBackbone -v
   pytest tests/test_encoder.py::TestCNNMambaEncoder -v
   pytest tests/test_encoder.py::TestCNNTransformerEncoder -v
   pytest tests/test_encoder.py::TestEncoderFactory -v

4. 运行带覆盖率的测试:
   pytest tests/test_encoder.py -v --cov=src/models/encoder

5. 快速烟雾测试 (仅测试核心功能):
   pytest tests/test_encoder.py -v -k "output_shape or forward"

6. GPU 测试 (如果可用):
   pytest tests/test_encoder.py -v  # 自动检测 CUDA

7. 预期结果:
   - 所有测试通过 (green)
   - CNNMambaEncoder: ~1-5M 参数
   - 下采样因子与配置一致 (默认 16x)

=============================================================================

Author: PA-HCL Team
"""

import pytest
import torch
import torch.nn as nn

# Test configuration
BATCH_SIZE = 4
SIGNAL_LENGTH = 4000  # ~0.8s at 5000Hz
IN_CHANNELS = 1


# ============== Fixtures ==============

@pytest.fixture
def device():
    """获取合适的设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_signal(device):
    """生成样本一维信号 (模拟 PCG)。"""
    return torch.randn(BATCH_SIZE, IN_CHANNELS, SIGNAL_LENGTH, device=device)


@pytest.fixture
def sample_signal_no_channel(device):
    """生成无通道维度的样本信号。"""
    return torch.randn(BATCH_SIZE, SIGNAL_LENGTH, device=device)


@pytest.fixture
def cnn_backbone(device):
    """创建 CNN 主干实例。"""
    from src.models.encoder import CNNBackbone
    return CNNBackbone(
        in_channels=1,
        channels=[32, 64, 128, 256],
        kernel_sizes=[7, 5, 5, 3],
        strides=[2, 2, 2, 2],
        dropout=0.0
    ).to(device)


@pytest.fixture
def cnn_mamba_encoder(device):
    """创建 CNN-Mamba 编码器实例。"""
    from src.models.encoder import CNNMambaEncoder
    return CNNMambaEncoder(
        in_channels=1,
        cnn_channels=[32, 64, 128, 256],
        cnn_strides=[2, 2, 2, 2],
        mamba_d_model=256,
        mamba_n_layers=2,
        mamba_dropout=0.0
    ).to(device)


@pytest.fixture
def cnn_transformer_encoder(device):
    """创建 CNN-Transformer 编码器实例。"""
    from src.models.encoder import CNNTransformerEncoder
    return CNNTransformerEncoder(
        in_channels=1,
        cnn_channels=[32, 64, 128, 256],
        cnn_strides=[2, 2, 2, 2],
        transformer_d_model=256,
        transformer_n_layers=2,
        transformer_dropout=0.0
    ).to(device)


# ============== Test ConvBlock ==============

class TestConvBlock:
    """基本 ConvBlock 的测试。"""
    
    def test_conv_block_output_shape(self, device):
        """测试 ConvBlock 输出维度。"""
        from src.models.encoder import ConvBlock
        
        block = ConvBlock(
            in_channels=1,
            out_channels=32,
            kernel_size=7,
            stride=2
        ).to(device)
        
        x = torch.randn(2, 1, 100, device=device)
        y = block(x)
        
        # With stride=2, length should be ~halved
        assert y.shape[0] == 2
        assert y.shape[1] == 32
        assert y.shape[2] == 50  # 100 / 2
    
    def test_activation_options(self, device):
        """测试不同的激活函数。"""
        from src.models.encoder import ConvBlock
        
        for act in ["relu", "gelu", "silu"]:
            block = ConvBlock(1, 32, activation=act).to(device)
            x = torch.randn(2, 1, 50, device=device)
            y = block(x)
            assert y.shape == (2, 32, 25)  # Default stride=2


# ============== Test ResidualConvBlock ==============

class TestResidualConvBlock:
    """残差卷积块的测试。"""
    
    def test_residual_block_same_channels(self, device):
        """测试具有相同输入/输出通道的残差块。"""
        from src.models.encoder import ResidualConvBlock
        
        block = ResidualConvBlock(64, 64, stride=1).to(device)
        x = torch.randn(2, 64, 100, device=device)
        y = block(x)
        
        assert y.shape == x.shape
    
    def test_residual_block_different_channels(self, device):
        """测试通道改变的残差块。"""
        from src.models.encoder import ResidualConvBlock
        
        block = ResidualConvBlock(32, 64, stride=2).to(device)
        x = torch.randn(2, 32, 100, device=device)
        y = block(x)
        
        assert y.shape == (2, 64, 50)
    
    def test_residual_shortcut_gradient(self, device):
        """测试梯度流经快捷连接。"""
        from src.models.encoder import ResidualConvBlock
        
        block = ResidualConvBlock(32, 64, stride=1).to(device)
        x = torch.randn(2, 32, 100, device=device, requires_grad=True)
        y = block(x)
        
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        # Gradient should be non-zero (information flows through shortcut)
        assert x.grad.abs().max() > 0


# ============== Test CNNBackbone ==============

class TestCNNBackbone:
    """CNN 主干的测试。"""
    
    def test_backbone_output_shape(self, cnn_backbone, sample_signal):
        """使用默认配置测试输出形状。"""
        output = cnn_backbone(sample_signal)
        
        # Downsample factor = 2*2*2*2 = 16
        expected_length = SIGNAL_LENGTH // 16
        expected_channels = 256
        
        assert output.shape == (BATCH_SIZE, expected_channels, expected_length)
    
    def test_backbone_auto_add_channel(self, cnn_backbone, sample_signal_no_channel):
        """测试自动通道维度处理。"""
        output = cnn_backbone(sample_signal_no_channel)
        
        expected_length = SIGNAL_LENGTH // 16
        assert output.shape == (BATCH_SIZE, 256, expected_length)
    
    def test_return_intermediate(self, cnn_backbone, sample_signal):
        """测试返回中间层特征。"""
        output, intermediates = cnn_backbone(sample_signal, return_intermediate=True)
        
        # Should return 4 intermediate outputs (one per layer)
        assert len(intermediates) == 4
        
        # Check intermediate shapes
        expected_lengths = [
            SIGNAL_LENGTH // 2,
            SIGNAL_LENGTH // 4,
            SIGNAL_LENGTH // 8,
            SIGNAL_LENGTH // 16
        ]
        expected_channels = [32, 64, 128, 256]
        
        for i, (feat, exp_len, exp_ch) in enumerate(
            zip(intermediates, expected_lengths, expected_channels)
        ):
            assert feat.shape == (BATCH_SIZE, exp_ch, exp_len), \
                f"Layer {i}: expected ({BATCH_SIZE}, {exp_ch}, {exp_len}), got {feat.shape}"
    
    def test_backbone_properties(self, cnn_backbone):
        """测试主干属性。"""
        assert cnn_backbone.out_channels == 256
        assert cnn_backbone.downsample_factor == 16
    
    def test_get_output_length(self, cnn_backbone):
        """测试输出长度计算。"""
        # This is a rough estimate
        length = cnn_backbone.get_output_length(SIGNAL_LENGTH)
        assert length > 0


# ============== Test CNNMambaEncoder ==============

class TestCNNMambaEncoder:
    """CNN-Mamba 混合编码器的测试。"""
    
    def test_encoder_output_shape_pooled(self, cnn_mamba_encoder, sample_signal):
        """测试池化输出形状。"""
        output = cnn_mamba_encoder(sample_signal)
        
        # Should be [B, D]
        assert output.shape == (BATCH_SIZE, 256)
    
    def test_encoder_output_shape_sequence(self, cnn_mamba_encoder, sample_signal):
        """测试序列输出形状。"""
        output = cnn_mamba_encoder(sample_signal, return_sequence=True)
        
        # Should be [B, L', D]
        expected_length = SIGNAL_LENGTH // 16
        assert output.shape == (BATCH_SIZE, expected_length, 256)
    
    def test_return_intermediate_features(self, cnn_mamba_encoder, sample_signal):
        """测试中间特征提取。"""
        result = cnn_mamba_encoder(sample_signal, return_intermediate=True)
        
        assert isinstance(result, dict)
        assert "cycle_features" in result
        assert "sub_features" in result
        assert "sequence" in result
        
        # Cycle features: pooled [B, D]
        assert result["cycle_features"].shape == (BATCH_SIZE, 256)
        
        # Sub features: CNN intermediate [B, C, L']
        assert result["sub_features"].dim() == 3
        assert result["sub_features"].shape[0] == BATCH_SIZE
    
    def test_get_sub_features(self, cnn_mamba_encoder, sample_signal):
        """测试专用子结构特征提取。"""
        sub_feats = cnn_mamba_encoder.get_sub_features(sample_signal)
        
        assert sub_feats.dim() == 3
        assert sub_feats.shape[0] == BATCH_SIZE
    
    def test_pool_types(self, device):
        """测试不同的池化策略。"""
        from src.models.encoder import CNNMambaEncoder
        
        x = torch.randn(2, 1, 1000, device=device)
        
        for pool_type in ["mean", "max", "cls"]:
            encoder = CNNMambaEncoder(
                cnn_channels=[32, 64, 128],
                cnn_strides=[2, 2, 2],
                mamba_d_model=128,
                mamba_n_layers=1,
                pool_type=pool_type
            ).to(device)
            
            output = encoder(x)
            assert output.shape == (2, 128), f"Pool type {pool_type} failed"
    
    def test_encoder_training_step(self, cnn_mamba_encoder, sample_signal):
        """测试完整的训练步骤。"""
        cnn_mamba_encoder.train()
        
        # Forward
        output = cnn_mamba_encoder(sample_signal)
        
        # Loss
        target = torch.randn_like(output)
        loss = nn.functional.mse_loss(output, target)
        
        # Backward
        loss.backward()
        
        # Check gradients
        grad_exists = any(
            p.grad is not None and p.grad.abs().max() > 0
            for p in cnn_mamba_encoder.parameters()
        )
        assert grad_exists, "No gradients found"


# ============== Test CNNTransformerEncoder ==============

class TestCNNTransformerEncoder:
    """CNN-Transformer 编码器的测试。"""
    
    def test_transformer_encoder_output(self, cnn_transformer_encoder, sample_signal):
        """测试 Transformer 编码器输出。"""
        output = cnn_transformer_encoder(sample_signal)
        
        assert output.shape == (BATCH_SIZE, 256)
    
    def test_transformer_sequence_output(self, cnn_transformer_encoder, sample_signal):
        """测试 Transformer 编码器的序列输出。"""
        output = cnn_transformer_encoder(sample_signal, return_sequence=True)
        
        # With CLS token, should have extra position
        # But default is mean pooling, so no extra
        assert output.dim() == 3
        assert output.shape[0] == BATCH_SIZE
        assert output.shape[2] == 256


# ============== Test Factory Function ==============

class TestEncoderFactory:
    """编码器工厂函数的测试。"""
    
    def test_build_cnn_only(self, device):
        """测试构建仅 CNN 编码器。"""
        from src.models.encoder import build_encoder, CNNBackbone
        
        encoder = build_encoder(
            encoder_type="cnn_only",
            in_channels=1,
            channels=[32, 64, 128]
        ).to(device)
        
        assert isinstance(encoder, CNNBackbone)
    
    def test_build_cnn_mamba(self, device):
        """测试构建 CNN-Mamba 编码器。"""
        from src.models.encoder import build_encoder, CNNMambaEncoder
        
        encoder = build_encoder(
            encoder_type="cnn_mamba",
            cnn_channels=[32, 64, 128],
            mamba_d_model=128,
            mamba_n_layers=2
        ).to(device)
        
        assert isinstance(encoder, CNNMambaEncoder)
    
    def test_build_cnn_transformer(self, device):
        """测试构建 CNN-Transformer 编码器。"""
        from src.models.encoder import build_encoder, CNNTransformerEncoder
        
        encoder = build_encoder(
            encoder_type="cnn_transformer",
            cnn_channels=[32, 64, 128],
            transformer_d_model=128,
            transformer_n_layers=2
        ).to(device)
        
        assert isinstance(encoder, CNNTransformerEncoder)
    
    def test_invalid_encoder_type(self):
        """测试无效编码器类型的错误处理。"""
        from src.models.encoder import build_encoder
        
        with pytest.raises(ValueError, match="Unknown encoder type"):
            build_encoder(encoder_type="invalid_type")


# ============== Integration Tests ==============

class TestEncoderIntegration:
    """编码器的集成测试。"""
    
    def test_pcg_signal_pipeline(self, device):
        """使用真实的 PCG 信号维度进行测试。"""
        from src.models.encoder import CNNMambaEncoder
        
        # Simulate 5 second PCG at 5000Hz
        sample_rate = 5000
        duration = 5
        signal_length = sample_rate * duration  # 25000 samples
        
        encoder = CNNMambaEncoder(
            cnn_channels=[32, 64, 128, 256],
            cnn_strides=[2, 2, 2, 2],
            mamba_d_model=256,
            mamba_n_layers=4
        ).to(device)
        
        x = torch.randn(2, 1, signal_length, device=device)
        
        # Should handle long signals
        output = encoder(x)
        assert output.shape == (2, 256)
    
    def test_cardiac_cycle_signal(self, device):
        """使用单个心动周期维度进行测试。"""
        from src.models.encoder import CNNMambaEncoder
        
        # Single cardiac cycle: ~0.4-1.5 seconds at 5000Hz
        # Conservative: 1 second = 5000 samples
        signal_length = 5000
        
        encoder = CNNMambaEncoder(
            cnn_channels=[32, 64, 128, 256],
            cnn_strides=[2, 2, 2, 2],
            mamba_d_model=256,
            mamba_n_layers=2
        ).to(device)
        
        x = torch.randn(8, 1, signal_length, device=device)
        
        output = encoder(x)
        assert output.shape == (8, 256)
        
        # Test substructure features
        sub_feats = encoder.get_sub_features(x)
        assert sub_feats.dim() == 3
    
    def test_model_parameter_count(self, device):
        """报告不同配置的模型大小。"""
        from src.models.encoder import CNNMambaEncoder, CNNTransformerEncoder
        
        configs = {
            "Small": {"cnn_channels": [32, 64, 128], "d_model": 128, "n_layers": 2},
            "Medium": {"cnn_channels": [32, 64, 128, 256], "d_model": 256, "n_layers": 4},
            "Large": {"cnn_channels": [64, 128, 256, 512], "d_model": 512, "n_layers": 6},
        }
        
        print("\n=== Model Parameter Counts ===")
        
        for name, cfg in configs.items():
            # Mamba
            mamba_enc = CNNMambaEncoder(
                cnn_channels=cfg["cnn_channels"],
                mamba_d_model=cfg["d_model"],
                mamba_n_layers=cfg["n_layers"]
            ).to(device)
            
            mamba_params = sum(p.numel() for p in mamba_enc.parameters())
            
            # Transformer
            transformer_enc = CNNTransformerEncoder(
                cnn_channels=cfg["cnn_channels"],
                transformer_d_model=cfg["d_model"],
                transformer_n_layers=cfg["n_layers"]
            ).to(device)
            
            transformer_params = sum(p.numel() for p in transformer_enc.parameters())
            
            print(f"\n{name}:")
            print(f"  CNN-Mamba: {mamba_params:,} params ({mamba_params * 4 / 1024 / 1024:.2f} MB)")
            print(f"  CNN-Transformer: {transformer_params:,} params ({transformer_params * 4 / 1024 / 1024:.2f} MB)")
    
    def test_gradient_checkpointing_memory(self, device):
        """测试梯度检查点的内存效率 (如果可用)。"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        from src.models.encoder import CNNMambaEncoder
        
        encoder = CNNMambaEncoder(
            cnn_channels=[64, 128, 256, 512],
            mamba_d_model=512,
            mamba_n_layers=6
        ).to(device)
        
        x = torch.randn(4, 1, 10000, device=device)
        
        # Get memory before forward
        torch.cuda.reset_peak_memory_stats()
        
        output = encoder(x)
        loss = output.sum()
        loss.backward()
        
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"\nPeak GPU memory: {peak_mem:.2f} MB")


# ============== Entry Point ==============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
