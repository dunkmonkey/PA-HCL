"""
Unit tests for Mamba (Selective State Space Model) module.

这是 PA-HCL 项目 Mamba 模块的单元测试文件。

=============================================================================
测试运行指南 (Test Execution Guide)
=============================================================================

1. 环境准备:
   cd /workspaces/PA-HCL
   pip install -e .
   # 或确保以下依赖已安装:
   pip install torch einops pytest numpy

2. 运行所有 Mamba 测试:
   pytest tests/test_mamba.py -v

3. 运行特定测试类:
   pytest tests/test_mamba.py::TestRMSNorm -v
   pytest tests/test_mamba.py::TestSelectiveSSM -v
   pytest tests/test_mamba.py::TestMambaBlock -v
   pytest tests/test_mamba.py::TestMambaEncoder -v

4. 运行带覆盖率的测试:
   pytest tests/test_mamba.py -v --cov=src/models/mamba

5. 调试单个测试:
   pytest tests/test_mamba.py::TestSelectiveSSM::test_ssm_output_shape -v -s

6. 预期结果:
   - 所有测试通过 (green)
   - 无 CUDA 内存泄漏警告
   - 输出维度与文档一致

=============================================================================

Author: PA-HCL Team
"""

import math
from typing import Tuple

import pytest
import torch
import torch.nn as nn

# 测试配置
BATCH_SIZE = 4
SEQ_LEN = 128
D_MODEL = 64
D_STATE = 16
D_CONV = 4
EXPAND = 2
N_LAYERS = 2


# ============== Fixtures ==============

@pytest.fixture
def device():
    """获取合适的设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_input(device):
    """生成样本输入张量。"""
    return torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL, device=device)


@pytest.fixture
def rms_norm(device):
    """创建 RMSNorm 实例。"""
    from src.models.mamba import RMSNorm
    return RMSNorm(D_MODEL).to(device)


@pytest.fixture
def selective_ssm(device):
    """创建 SelectiveSSM 实例。"""
    from src.models.mamba import SelectiveSSM
    return SelectiveSSM(
        d_model=D_MODEL,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND
    ).to(device)


@pytest.fixture
def mamba_block(device):
    """创建 MambaBlock 实例。"""
    from src.models.mamba import MambaBlock
    return MambaBlock(
        d_model=D_MODEL,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
        dropout=0.0  # 测试时禁用 dropout
    ).to(device)


@pytest.fixture
def mamba_encoder(device):
    """创建 MambaEncoder 实例。"""
    from src.models.mamba import MambaEncoder
    return MambaEncoder(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
        dropout=0.0
    ).to(device)


# ============== Test RMSNorm ==============

class TestRMSNorm:
    """RMS 根均方归一化层的测试。"""
    
    def test_output_shape(self, rms_norm, sample_input):
        """测试输出形状与输入形状一致。"""
        output = rms_norm(sample_input)
        assert output.shape == sample_input.shape, \
            f"Expected {sample_input.shape}, got {output.shape}"
    
    def test_normalization_scale(self, rms_norm, sample_input):
        """测试输出具有约为 1 的 RMS。"""
        output = rms_norm(sample_input)
        rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
        # RMS should be close to weight magnitude (initialized to 1)
        assert torch.allclose(rms.mean(), torch.tensor(1.0, device=rms.device), atol=0.1), \
            f"RMS mean should be ~1.0, got {rms.mean().item()}"
    
    def test_learnable_weight(self, device):
        """测试权重参数是否可学习。"""
        from src.models.mamba import RMSNorm
        norm = RMSNorm(D_MODEL).to(device)
        
        # Check weight is parameter
        assert isinstance(norm.weight, nn.Parameter)
        assert norm.weight.shape == (D_MODEL,)
        
        # Check gradient flows
        x = torch.randn(2, 10, D_MODEL, device=device, requires_grad=True)
        y = norm(x)
        loss = y.sum()
        loss.backward()
        
        assert norm.weight.grad is not None
        assert x.grad is not None
    
    def test_numerical_stability(self, device):
        """测试在极小数值下的数值稳定性。"""
        from src.models.mamba import RMSNorm
        norm = RMSNorm(D_MODEL, eps=1e-6).to(device)
        
        # 使用非常小的数值进行测试
        x_small = torch.randn(2, 10, D_MODEL, device=device) * 1e-8
        output = norm(x_small)
        
        # Should not have NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"


# ============== Test SelectiveSSM ==============

class TestSelectiveSSM:
    """选择性状态空间模型 (Selective SSM) 的测试。"""
    
    def test_ssm_output_shape(self, selective_ssm, sample_input):
        """测试输出形状与输入形状一致。"""
        output = selective_ssm(sample_input)
        assert output.shape == sample_input.shape, \
            f"Expected {sample_input.shape}, got {output.shape}"
    
    def test_ssm_parameters(self, selective_ssm):
        """测试所有 SSM 参数是否已初始化。"""
        # Check A_log
        assert hasattr(selective_ssm, 'A_log')
        expected_shape = (selective_ssm.d_inner, D_STATE)
        assert selective_ssm.A_log.shape == expected_shape
        
        # Check D (skip connection)
        assert hasattr(selective_ssm, 'D')
        assert selective_ssm.D.shape == (selective_ssm.d_inner,)
        
        # Check projections
        assert hasattr(selective_ssm, 'in_proj')
        assert hasattr(selective_ssm, 'out_proj')
    
    def test_causal_property(self, selective_ssm, device):
        """测试 SSM 是否因果 (时刻 t 的输出不依赖未来信息)。"""
        # Create two inputs that differ only at t=50
        x1 = torch.randn(1, 100, D_MODEL, device=device)
        x2 = x1.clone()
        x2[:, 50:, :] = torch.randn(1, 50, D_MODEL, device=device)
        
        # Forward pass
        y1 = selective_ssm(x1)
        y2 = selective_ssm(x2)
        
        # Outputs before t=50 should be identical (causality)
        assert torch.allclose(y1[:, :50], y2[:, :50], atol=1e-5), \
            "SSM is not causal: output depends on future inputs"
    
    def test_gradient_flow(self, selective_ssm, sample_input):
        """测试梯度是否流经所有参数。"""
        sample_input.requires_grad_(True)
        output = selective_ssm(sample_input)
        loss = output.sum()
        loss.backward()
        
        # Check gradient exists for input
        assert sample_input.grad is not None
        
        # Check gradients for key parameters
        assert selective_ssm.A_log.grad is not None, "No gradient for A_log"
        assert selective_ssm.D.grad is not None, "No gradient for D"
    
    def test_different_sequence_lengths(self, selective_ssm, device):
        """测试 SSM 在不同序列长度下是否正常工作。"""
        for seq_len in [32, 64, 128, 256]:
            x = torch.randn(2, seq_len, D_MODEL, device=device)
            y = selective_ssm(x)
            assert y.shape == (2, seq_len, D_MODEL)
    
    def test_d_inner_calculation(self, device):
        """测试 d_inner 是否计算正确。"""
        from src.models.mamba import SelectiveSSM
        ssm = SelectiveSSM(d_model=64, expand=2).to(device)
        assert ssm.d_inner == 128, f"Expected d_inner=128, got {ssm.d_inner}"


# ============== Test MambaBlock ==============

class TestMambaBlock:
    """带残差连接的 MambaBlock 测试。"""
    
    def test_block_output_shape(self, mamba_block, sample_input):
        """测试输出形状与输入形状一致。"""
        output = mamba_block(sample_input)
        assert output.shape == sample_input.shape
    
    def test_residual_connection(self, device):
        """测试残差连接是否有效。"""
        from src.models.mamba import MambaBlock
        block = MambaBlock(d_model=D_MODEL, dropout=0.0).to(device)
        
        # With zero initialization, output should be close to input
        # (not exactly equal due to learnable parameters)
        x = torch.randn(2, 10, D_MODEL, device=device)
        y = block(x)
        
        # Output should have similar magnitude due to residual
        assert y.std() > 0.1, "Output variance too low"
    
    def test_block_components(self, mamba_block):
        """测试 Block 是否包含所有必要组件。"""
        assert hasattr(mamba_block, 'norm'), "Missing normalization layer"
        assert hasattr(mamba_block, 'ssm'), "Missing SSM layer"
        assert hasattr(mamba_block, 'dropout'), "Missing dropout layer"


# ============== Test MambaEncoder ==============

class TestMambaEncoder:
    """完整 Mamba 编码器栈的测试。"""
    
    def test_encoder_output_shape(self, mamba_encoder, sample_input):
        """测试编码器输出形状。"""
        output = mamba_encoder(sample_input)
        assert output.shape == sample_input.shape
    
    def test_return_all_layers(self, mamba_encoder, sample_input):
        """测试返回各层中间输出。"""
        outputs = mamba_encoder(sample_input, return_all_layers=True)
        
        # Should return list with n_layers + 1 outputs
        assert isinstance(outputs, list)
        assert len(outputs) == N_LAYERS + 1, \
            f"Expected {N_LAYERS + 1} outputs, got {len(outputs)}"
        
        # All should have correct shape
        for i, out in enumerate(outputs):
            assert out.shape == sample_input.shape, \
                f"Layer {i} output shape mismatch"
    
    def test_encoder_with_different_configs(self, device):
        """在多种配置下测试编码器。"""
        from src.models.mamba import MambaEncoder
        
        configs = [
            {"d_model": 64, "n_layers": 2},
            {"d_model": 128, "n_layers": 4},
            {"d_model": 256, "n_layers": 6},
        ]
        
        for config in configs:
            encoder = MambaEncoder(**config).to(device)
            x = torch.randn(2, 32, config["d_model"], device=device)
            y = encoder(x)
            assert y.shape == x.shape
    
    def test_final_normalization(self, mamba_encoder):
        """测试编码器是否包含最终的归一化层。"""
        assert hasattr(mamba_encoder, 'norm_f')


# ============== Test Factory Function ==============

class TestFactoryFunction:
    """get_mamba_encoder 工厂函数的测试。"""
    
    def test_factory_default_params(self, device):
        """测试使用默认参数的工厂函数。"""
        from src.models.mamba import get_mamba_encoder
        encoder = get_mamba_encoder().to(device)
        
        x = torch.randn(2, 32, 256, device=device)
        y = encoder(x)
        assert y.shape == (2, 32, 256)
    
    def test_factory_custom_params(self, device):
        """测试使用自定义参数的工厂函数。"""
        from src.models.mamba import get_mamba_encoder
        encoder = get_mamba_encoder(
            d_model=128,
            n_layers=3,
            d_state=8,
            d_conv=3,
            expand=4
        ).to(device)
        
        x = torch.randn(2, 32, 128, device=device)
        y = encoder(x)
        assert y.shape == (2, 32, 128)
    
    def test_factory_pure_pytorch(self, device):
        """测试 use_official=False 时返回纯 PyTorch 实现。"""
        from src.models.mamba import get_mamba_encoder, MambaEncoder
        encoder = get_mamba_encoder(use_official=False).to(device)
        assert isinstance(encoder, MambaEncoder)


# ============== Integration Tests ==============

class TestMambaIntegration:
    """Mamba 模块的集成测试。"""
    
    def test_training_step(self, mamba_encoder, sample_input):
        """测试完整的训练步骤。"""
        mamba_encoder.train()
        
        # Forward pass
        output = mamba_encoder(sample_input)
        
        # Compute dummy loss
        target = torch.randn_like(output)
        loss = nn.functional.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        for name, param in mamba_encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_eval_mode(self, mamba_encoder, sample_input):
        """测试评估模式下的推理行为。"""
        mamba_encoder.eval()
        
        with torch.no_grad():
            output1 = mamba_encoder(sample_input)
            output2 = mamba_encoder(sample_input)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2)
    
    def test_model_size(self, device):
        """测试并报告模型参数量。"""
        from src.models.mamba import MambaEncoder
        
        encoder = MambaEncoder(d_model=256, n_layers=4).to(device)
        
        total_params = sum(p.numel() for p in encoder.parameters())
        trainable_params = sum(
            p.numel() for p in encoder.parameters() if p.requires_grad
        )
        
        print(f"\nMamba Encoder (d=256, L=4):")
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
        
        # Sanity check: should be reasonable size
        assert total_params < 50_000_000, "Model unexpectedly large"
        assert trainable_params == total_params, "Some params frozen unexpectedly"
    
    def test_long_sequence(self, device):
        """使用长序列进行测试 (模拟 PCG 信号)。"""
        from src.models.mamba import MambaEncoder
        
        # Simulate 5 seconds of PCG at 5000Hz / 16x downsampling
        # = 5000 * 5 / 16 ≈ 1562 time steps
        seq_len = 1562
        
        encoder = MambaEncoder(d_model=256, n_layers=4).to(device)
        x = torch.randn(2, seq_len, 256, device=device)
        
        # Should handle long sequences without OOM
        y = encoder(x)
        assert y.shape == (2, seq_len, 256)


# ============== Entry Point ==============

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
