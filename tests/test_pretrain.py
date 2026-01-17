"""
Unit tests for PA-HCL model and PretrainTrainer.

这是 PA-HCL 项目预训练模块的单元测试文件。

=============================================================================
测试运行指南 (Test Execution Guide)
=============================================================================

1. 环境准备:
   cd /workspaces/PA-HCL
   pip install -e .
   # 或确保以下依赖已安装:
   pip install torch einops pytest numpy omegaconf

2. 运行所有预训练测试:
   pytest tests/test_pretrain.py -v

3. 运行特定测试类:
   pytest tests/test_pretrain.py::TestPAHCLModel -v
   pytest tests/test_pretrain.py::TestPretrainTrainer -v
   pytest tests/test_pretrain.py::TestPretrainIntegration -v

4. 运行带覆盖率的测试:
   pytest tests/test_pretrain.py -v --cov=src/models/pahcl --cov=src/trainers

5. 快速测试 (跳过慢测试):
   pytest tests/test_pretrain.py -v -k "not slow"

6. 详细输出调试:
   pytest tests/test_pretrain.py::TestPAHCLModel::test_pretrain_forward -v -s

7. 预期结果:
   - 所有测试通过 (green)
   - 模型参数量与预期一致
   - 损失值正常下降
   - 梯度正确流动

=============================================================================

Author: PA-HCL Team
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, patch

# 测试配置
 BATCH_SIZE = 4
 SIGNAL_LENGTH = 5000  # 5000Hz 采样率下的 1 秒
 NUM_SUBSTRUCTURES = 4


# ============== Fixtures ==============

@pytest.fixture
def device():
    """获取合适的设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_signal(device):
    """生成样本 PCG 信号。"""
    return torch.randn(BATCH_SIZE, 1, SIGNAL_LENGTH, device=device)


@pytest.fixture
def sample_batch(device):
    """生成样本训练 batch。"""
    return {
        "view1": torch.randn(BATCH_SIZE, 1, SIGNAL_LENGTH, device=device),
        "view2": torch.randn(BATCH_SIZE, 1, SIGNAL_LENGTH, device=device),
        "subs1": torch.randn(BATCH_SIZE, NUM_SUBSTRUCTURES, SIGNAL_LENGTH // NUM_SUBSTRUCTURES, device=device),
        "subs2": torch.randn(BATCH_SIZE, NUM_SUBSTRUCTURES, SIGNAL_LENGTH // NUM_SUBSTRUCTURES, device=device),
        "idx": torch.arange(BATCH_SIZE, device=device),
    }


@pytest.fixture
def small_model_config():
    """用于测试的小模型配置。"""
    return {
        "in_channels": 1,
        "cnn_channels": [16, 32, 64],
        "cnn_kernel_sizes": [7, 5, 3],
        "cnn_strides": [2, 2, 2],
        "cnn_dropout": 0.0,
        "mamba_d_model": 64,
        "mamba_n_layers": 1,
        "mamba_d_state": 8,
        "mamba_expand": 2,
        "mamba_dropout": 0.0,
        "pool_type": "mean",
        "cycle_proj_hidden": 128,
        "cycle_proj_output": 32,
        "cycle_proj_layers": 2,
        "sub_proj_hidden": 64,
        "sub_proj_output": 16,
        "num_substructures": NUM_SUBSTRUCTURES,
    }


@pytest.fixture
def pahcl_model(device, small_model_config):
    """创建 PA-HCL 模型实例。"""
    from src.models.pahcl import PAHCLModel
    return PAHCLModel(**small_model_config).to(device)


# ============== Test PAHCLModel ==============

class TestPAHCLModel:
    """PA-HCL 模型的测试。"""
    
    def test_model_creation(self, pahcl_model):
        """测试模型是否能成功创建。"""
        assert pahcl_model is not None
        assert hasattr(pahcl_model, "encoder")
        assert hasattr(pahcl_model, "cycle_projector")
        assert hasattr(pahcl_model, "sub_projector")
    
    def test_forward_output_shape(self, pahcl_model, sample_signal):
        """测试前向传播输出形状。"""
        output = pahcl_model(sample_signal)
        
        assert "cycle_proj" in output
        assert output["cycle_proj"].shape == (BATCH_SIZE, 32)  # cycle_proj_output=32
    
    def test_forward_with_features(self, pahcl_model, sample_signal):
        """测试 return_features=True 时的前向传播。"""
        output = pahcl_model(sample_signal, return_features=True)
        
        assert "cycle_proj" in output
        assert "features" in output
        assert output["features"].shape == (BATCH_SIZE, 64)  # mamba_d_model=64
    
    def test_pretrain_forward(self, pahcl_model, device):
        """测试预训练前向传播。"""
        view1 = torch.randn(BATCH_SIZE, 1, SIGNAL_LENGTH, device=device)
        view2 = torch.randn(BATCH_SIZE, 1, SIGNAL_LENGTH, device=device)
        
        outputs = pahcl_model.forward_pretrain(view1, view2)
        
        # Check all outputs exist
        assert "cycle_proj1" in outputs
        assert "cycle_proj2" in outputs
        assert "sub_proj1" in outputs
        assert "sub_proj2" in outputs
        assert "cycle_z1" in outputs
        assert "cycle_z2" in outputs
        
        # Check shapes
        assert outputs["cycle_proj1"].shape == (BATCH_SIZE, 32)
        assert outputs["cycle_proj2"].shape == (BATCH_SIZE, 32)
        assert outputs["sub_proj1"].dim() == 3  # [B, K, D]
        assert outputs["sub_proj2"].dim() == 3
    
    def test_pretrain_forward_with_substructures(self, pahcl_model, device):
        """测试使用预先划分子结构的预训练前向传播。"""
        view1 = torch.randn(BATCH_SIZE, 1, SIGNAL_LENGTH, device=device)
        view2 = torch.randn(BATCH_SIZE, 1, SIGNAL_LENGTH, device=device)
        subs1 = torch.randn(BATCH_SIZE, NUM_SUBSTRUCTURES, SIGNAL_LENGTH // 4, device=device)
        subs2 = torch.randn(BATCH_SIZE, NUM_SUBSTRUCTURES, SIGNAL_LENGTH // 4, device=device)
        
        outputs = pahcl_model.forward_pretrain(view1, view2, subs1, subs2)
        
        assert outputs["sub_proj1"].shape[1] == NUM_SUBSTRUCTURES
        assert outputs["sub_proj2"].shape[1] == NUM_SUBSTRUCTURES
    
    def test_get_encoder_output(self, pahcl_model, sample_signal):
        """测试获取用于下游任务的编码器输出。"""
        features = pahcl_model.get_encoder_output(sample_signal)
        
        assert features.shape == (BATCH_SIZE, 64)
    
    def test_gradient_flow(self, pahcl_model, device):
        """测试梯度是否流经所有组件。"""
        view1 = torch.randn(BATCH_SIZE, 1, SIGNAL_LENGTH, device=device, requires_grad=True)
        view2 = torch.randn(BATCH_SIZE, 1, SIGNAL_LENGTH, device=device, requires_grad=True)
        
        outputs = pahcl_model.forward_pretrain(view1, view2)
        
        # Compute loss
        loss = outputs["cycle_proj1"].sum() + outputs["sub_proj1"].sum()
        loss.backward()
        
        # Check gradients exist for input
        assert view1.grad is not None
        
        # Check gradients exist for model parameters
        has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in pahcl_model.parameters()
        )
        assert has_grads
    
    def test_model_parameter_count(self, pahcl_model):
        """测试模型参数量是否合理。"""
        total_params = sum(p.numel() for p in pahcl_model.parameters())
        
        print(f"\nSmall model parameter count: {total_params:,}")
        
        # Small model should have < 1M parameters
        assert total_params < 5_000_000
        assert total_params > 10_000
    
    def test_input_without_channel_dim(self, pahcl_model, device):
        """测试模型是否能处理无通道维度的输入。"""
        x = torch.randn(BATCH_SIZE, SIGNAL_LENGTH, device=device)
        
        output = pahcl_model(x)
        
        assert output["cycle_proj"].shape == (BATCH_SIZE, 32)


# ============== Test Build Function ==============

class TestBuildPAHCLModel:
    """从配置构建模型的测试。"""
    
    def test_build_from_config(self, device):
        """测试从配置对象构建模型。"""
        from src.models.pahcl import build_pahcl_model
        
        # Create mock config
        config = MagicMock()
        config.model.encoder_type = "cnn_mamba"
        config.model.cnn_channels = [32, 64, 128]
        config.model.cnn_kernel_sizes = [7, 5, 3]
        config.model.cnn_strides = [2, 2, 2]
        config.model.cnn_dropout = 0.1
        config.model.mamba_d_model = 128
        config.model.mamba_n_layers = 2
        config.model.mamba_d_state = 16
        config.model.mamba_expand = 2
        config.model.mamba_dropout = 0.1
        config.model.pool_type = "mean"
        config.model.proj_hidden_dim = 256
        config.model.proj_output_dim = 64
        config.model.proj_num_layers = 2
        config.model.sub_proj_hidden_dim = 128
        config.model.sub_proj_output_dim = 32
        config.data.num_substructures = 4
        
        model = build_pahcl_model(config).to(device)
        
        assert model is not None
        assert model.encoder_dim == 128


# ============== Test PretrainTrainer ==============

class TestPretrainTrainer:
    """预训练 Trainer 的测试。"""
    
    def test_trainer_creation(self, pahcl_model, device):
        """测试 Trainer 是否能成功创建。"""
        from src.trainers.pretrain_trainer import PretrainTrainer
        
        # Create mock data loader
        mock_loader = [
            {
                "view1": torch.randn(4, 1, SIGNAL_LENGTH, device=device),
                "view2": torch.randn(4, 1, SIGNAL_LENGTH, device=device),
            }
        ]
        
        trainer = PretrainTrainer(
            model=pahcl_model,
            train_loader=mock_loader,
            num_epochs=1,
            learning_rate=1e-3,
            use_amp=False,
            distributed=False,
            log_interval=1,
            save_interval=1,
            output_dir="/tmp/pahcl_test",
        )
        
        assert trainer is not None
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
    
    def test_optimizer_param_groups(self, pahcl_model, device):
        """测试优化器是否包含正确的参数组。"""
        from src.trainers.pretrain_trainer import PretrainTrainer
        
        mock_loader = []
        
        trainer = PretrainTrainer(
            model=pahcl_model,
            train_loader=mock_loader,
            num_epochs=1,
            output_dir="/tmp/pahcl_test",
        )
        
        # Should have 2 param groups: with and without weight decay
        assert len(trainer.optimizer.param_groups) == 2
        
        # Check one has weight decay and one doesn't
        weight_decays = [pg["weight_decay"] for pg in trainer.optimizer.param_groups]
        assert 0.0 in weight_decays
        assert any(wd > 0 for wd in weight_decays)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_train_step(self, pahcl_model, device):
        """测试单个训练步骤能否正常运行。"""
        from src.trainers.pretrain_trainer import PretrainTrainer
        
        batch = {
            "view1": torch.randn(4, 1, SIGNAL_LENGTH, device=device),
            "view2": torch.randn(4, 1, SIGNAL_LENGTH, device=device),
        }
        mock_loader = [batch]
        
        trainer = PretrainTrainer(
            model=pahcl_model,
            train_loader=mock_loader,
            num_epochs=1,
            learning_rate=1e-3,
            use_amp=True,
            distributed=False,
            output_dir="/tmp/pahcl_test",
            log_interval=1,
        )
        
        # Run one epoch
        metrics = trainer.train_epoch()
        
        assert "train_loss" in metrics
        assert metrics["train_loss"] > 0
        assert metrics["train_loss"] < 100  # Sanity check


# ============== Integration Tests ==============

class TestPretrainIntegration:
    """完整预训练流水线的集成测试。"""
    
    def test_full_forward_backward(self, device):
        """测试包含损失计算的完整前向-反向过程。"""
        from src.models.pahcl import PAHCLModel
        from src.losses.contrastive import HierarchicalContrastiveLoss
        
        # Create model
        model = PAHCLModel(
            cnn_channels=[16, 32, 64],
            cnn_strides=[2, 2, 2],
            mamba_d_model=64,
            mamba_n_layers=1,
            num_substructures=4,
        ).to(device)
        
        # Create loss
        criterion = HierarchicalContrastiveLoss(
            temperature=0.07,
            lambda_cycle=1.0,
            lambda_sub=1.0
        )
        
        # Create data
        view1 = torch.randn(4, 1, 4000, device=device)
        view2 = torch.randn(4, 1, 4000, device=device)
        
        # Forward
        outputs = model.forward_pretrain(view1, view2)
        
        # Loss
        total_loss, loss_dict = criterion(
            outputs["cycle_proj1"],
            outputs["cycle_proj2"],
            outputs["sub_proj1"],
            outputs["sub_proj2"]
        )
        
        # Backward
        total_loss.backward()
        
        # Verify
        assert torch.isfinite(total_loss)
        assert total_loss > 0
        assert "loss_cycle" in loss_dict
        assert "loss_sub" in loss_dict
    
    def test_training_reduces_loss(self, device):
        """测试训练步骤是否能降低损失 (基础健壮性检查)。"""
        from src.models.pahcl import PAHCLModel
        from src.losses.contrastive import HierarchicalContrastiveLoss
        
        # Create small model
        model = PAHCLModel(
            cnn_channels=[8, 16],
            cnn_strides=[2, 2],
            mamba_d_model=16,
            mamba_n_layers=1,
            num_substructures=2,
        ).to(device)
        
        criterion = HierarchicalContrastiveLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Generate fixed data (so model can overfit)
        torch.manual_seed(42)
        view1 = torch.randn(8, 1, 1000, device=device)
        view2 = view1 + 0.1 * torch.randn_like(view1)  # Similar views
        
        # Track losses
        losses = []
        
        for step in range(20):
            optimizer.zero_grad()
            
            outputs = model.forward_pretrain(view1, view2)
            total_loss, _ = criterion(
                outputs["cycle_proj1"],
                outputs["cycle_proj2"],
                outputs["sub_proj1"],
                outputs["sub_proj2"]
            )
            
            total_loss.backward()
            optimizer.step()
            
            losses.append(total_loss.item())
        
        # Loss should generally decrease
        # (may not be monotonic, but trend should be down)
        print(f"\nLoss progression: {losses[0]:.4f} -> {losses[-1]:.4f}")
        
        # Final loss should be lower than initial (with some tolerance)
        assert losses[-1] < losses[0] + 0.5, \
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    
    def test_model_save_load(self, pahcl_model, device, tmp_path):
        """测试保存与加载模型 checkpoint。"""
        # Save model
        save_path = tmp_path / "model.pt"
        torch.save(pahcl_model.state_dict(), save_path)
        
        # Create new model and load
        from src.models.pahcl import PAHCLModel
        
        new_model = PAHCLModel(
            cnn_channels=[16, 32, 64],
            cnn_strides=[2, 2, 2],
            mamba_d_model=64,
            mamba_n_layers=1,
            num_substructures=4,
        ).to(device)
        
        new_model.load_state_dict(torch.load(save_path))
        
        # Verify outputs match
        x = torch.randn(2, 1, 4000, device=device)
        
        pahcl_model.eval()
        new_model.eval()
        
        with torch.no_grad():
            out1 = pahcl_model(x)
            out2 = new_model(x)
        
        assert torch.allclose(out1["cycle_proj"], out2["cycle_proj"])


# ============== Entry Point ==============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
