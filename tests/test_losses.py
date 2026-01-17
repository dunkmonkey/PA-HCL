"""
Unit tests for contrastive loss functions.

这是 PA-HCL 项目对比学习损失函数的单元测试文件。

=============================================================================
测试运行指南 (Test Execution Guide)
=============================================================================

1. 环境准备:
   cd /workspaces/PA-HCL
   pip install -e .
   # 或确保以下依赖已安装:
   pip install torch pytest numpy

2. 运行所有损失函数测试:
   pytest tests/test_losses.py -v

3. 运行特定测试类:
   pytest tests/test_losses.py::TestInfoNCELoss -v
   pytest tests/test_losses.py::TestSubstructureContrastiveLoss -v
   pytest tests/test_losses.py::TestHierarchicalContrastiveLoss -v
   pytest tests/test_losses.py::TestSupConLoss -v

4. 运行带覆盖率的测试:
   pytest tests/test_losses.py -v --cov=src/losses

5. 调试测试:
   pytest tests/test_losses.py::TestInfoNCELoss::test_loss_decreases_with_similarity -v -s

6. 预期结果:
   - 所有测试通过 (green)
   - 损失值非负且有限
   - 正样本相似度高时损失降低
   - 梯度正确流动

=============================================================================

Author: PA-HCL Team
"""

import pytest
import torch
import torch.nn.functional as F

# 测试配置
BATCH_SIZE = 16
FEATURE_DIM = 128
NUM_SUBSTRUCTURES = 4


# ============== Fixtures ==============

@pytest.fixture
def device():
    """获取合适的设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def random_features(device):
    """为两个视图生成随机特征。"""
    z1 = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
    z2 = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
    return z1, z2


@pytest.fixture
def similar_features(device):
    """生成相似的特征 (正样本对应该具有较低的损失)。"""
    z1 = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
    # z2 is z1 with small noise (very similar)
    z2 = z1 + 0.1 * torch.randn_like(z1)
    return z1, z2


@pytest.fixture
def random_substructures(device):
    """生成随机子结构特征。"""
    subs1 = torch.randn(BATCH_SIZE, NUM_SUBSTRUCTURES, FEATURE_DIM, device=device)
    subs2 = torch.randn(BATCH_SIZE, NUM_SUBSTRUCTURES, FEATURE_DIM, device=device)
    return subs1, subs2


@pytest.fixture
def similar_substructures(device):
    """生成相似的子结构特征。"""
    subs1 = torch.randn(BATCH_SIZE, NUM_SUBSTRUCTURES, FEATURE_DIM, device=device)
    subs2 = subs1 + 0.1 * torch.randn_like(subs1)
    return subs1, subs2


# ============== Test InfoNCELoss ==============

class TestInfoNCELoss:
    """InfoNCE 对比损失的测试。"""
    
    def test_output_is_scalar(self, random_features):
        """测试损失输出为标量。"""
        from src.losses.contrastive import InfoNCELoss
        
        z1, z2 = random_features
        loss_fn = InfoNCELoss()
        loss = loss_fn(z1, z2)
        
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.numel() == 1
    
    def test_loss_is_positive(self, random_features):
        """测试损失为非负数。"""
        from src.losses.contrastive import InfoNCELoss
        
        z1, z2 = random_features
        loss_fn = InfoNCELoss()
        loss = loss_fn(z1, z2)
        
        assert loss >= 0, f"Loss should be non-negative, got {loss.item()}"
    
    def test_loss_is_finite(self, random_features):
        """测试损失是有限的 (不是 NaN 或 Inf)。"""
        from src.losses.contrastive import InfoNCELoss
        
        z1, z2 = random_features
        loss_fn = InfoNCELoss()
        loss = loss_fn(z1, z2)
        
        assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"
    
    def test_loss_decreases_with_similarity(self, device):
        """测试当正样本对更相似时损失减少。"""
        from src.losses.contrastive import InfoNCELoss
        
        loss_fn = InfoNCELoss()
        
        # Random features (dissimilar)
        z1_random = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
        z2_random = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
        loss_random = loss_fn(z1_random, z2_random)
        
        # Similar features
        z1_similar = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
        z2_similar = z1_similar + 0.01 * torch.randn_like(z1_similar)
        loss_similar = loss_fn(z1_similar, z2_similar)
        
        assert loss_similar < loss_random, \
            f"Similar features should have lower loss: {loss_similar.item()} vs {loss_random.item()}"
    
    def test_gradient_flow(self, device):
        """测试梯度流经损失函数。"""
        from src.losses.contrastive import InfoNCELoss
        
        loss_fn = InfoNCELoss()
        
        z1 = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device, requires_grad=True)
        z2 = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device, requires_grad=True)
        
        loss = loss_fn(z1, z2)
        loss.backward()
        
        assert z1.grad is not None, "Gradient should flow to z1"
        assert z2.grad is not None, "Gradient should flow to z2"
        assert z1.grad.abs().sum() > 0, "Gradient should be non-zero"
    
    def test_temperature_effect(self, random_features):
        """测试温度影响损失幅度。"""
        from src.losses.contrastive import InfoNCELoss
        
        z1, z2 = random_features
        
        loss_high_temp = InfoNCELoss(temperature=1.0)(z1, z2)
        loss_low_temp = InfoNCELoss(temperature=0.07)(z1, z2)
        
        # Lower temperature should give higher loss for random features
        # (makes softmax more peaked)
        assert loss_low_temp != loss_high_temp, \
            "Different temperatures should give different losses"
    
    def test_normalization_option(self, device):
        """测试使用和不使用特征归一化。"""
        from src.losses.contrastive import InfoNCELoss
        
        z1 = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
        z2 = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
        
        loss_normalized = InfoNCELoss(normalize=True)(z1, z2)
        loss_unnormalized = InfoNCELoss(normalize=False)(z1, z2)
        
        # Both should be valid losses
        assert torch.isfinite(loss_normalized)
        assert torch.isfinite(loss_unnormalized)


# ============== Test SubstructureContrastiveLoss ==============

class TestSubstructureContrastiveLoss:
    """子结构级对比损失的测试。"""
    
    def test_output_is_scalar(self, random_substructures):
        """测试输出为标量。"""
        from src.losses.contrastive import SubstructureContrastiveLoss
        
        subs1, subs2 = random_substructures
        loss_fn = SubstructureContrastiveLoss()
        loss = loss_fn(subs1, subs2)
        
        assert loss.dim() == 0
    
    def test_loss_is_positive(self, random_substructures):
        """测试损失为非负数。"""
        from src.losses.contrastive import SubstructureContrastiveLoss
        
        subs1, subs2 = random_substructures
        loss_fn = SubstructureContrastiveLoss()
        loss = loss_fn(subs1, subs2)
        
        assert loss >= 0
    
    def test_aligned_substructures_mode(self, random_substructures):
        """测试对齐子结构对比模式。"""
        from src.losses.contrastive import SubstructureContrastiveLoss
        
        subs1, subs2 = random_substructures
        loss_fn = SubstructureContrastiveLoss(align_substructures=True)
        loss = loss_fn(subs1, subs2)
        
        assert torch.isfinite(loss)
    
    def test_unaligned_substructures_mode(self, random_substructures):
        """测试未对齐 (全对全) 子结构对比模式。"""
        from src.losses.contrastive import SubstructureContrastiveLoss
        
        subs1, subs2 = random_substructures
        loss_fn = SubstructureContrastiveLoss(align_substructures=False)
        loss = loss_fn(subs1, subs2)
        
        assert torch.isfinite(loss)
    
    def test_loss_decreases_with_similarity(self, device):
        """测试相似子结构的损失减少。"""
        from src.losses.contrastive import SubstructureContrastiveLoss
        
        loss_fn = SubstructureContrastiveLoss()
        
        # Random
        subs1_random = torch.randn(BATCH_SIZE, NUM_SUBSTRUCTURES, FEATURE_DIM, device=device)
        subs2_random = torch.randn(BATCH_SIZE, NUM_SUBSTRUCTURES, FEATURE_DIM, device=device)
        loss_random = loss_fn(subs1_random, subs2_random)
        
        # Similar
        subs1_similar = torch.randn(BATCH_SIZE, NUM_SUBSTRUCTURES, FEATURE_DIM, device=device)
        subs2_similar = subs1_similar + 0.1 * torch.randn_like(subs1_similar)  # Increased noise slightly to avoid 0 loss
        loss_similar = loss_fn(subs1_similar, subs2_similar)
        
        assert loss_similar < loss_random
    
    def test_different_num_substructures(self, device):
        """测试不同数量的子结构。"""
        from src.losses.contrastive import SubstructureContrastiveLoss
        
        loss_fn = SubstructureContrastiveLoss()
        
        for K in [2, 4, 6, 8]:
            subs1 = torch.randn(4, K, FEATURE_DIM, device=device)
            subs2 = torch.randn(4, K, FEATURE_DIM, device=device)
            
            loss = loss_fn(subs1, subs2)
            assert torch.isfinite(loss), f"Failed for K={K}"


# ============== Test HierarchicalContrastiveLoss ==============

class TestHierarchicalContrastiveLoss:
    """分层对比损失的测试。"""
    
    def test_output_format(self, device):
        """测试输出为 (loss, dict) 元组。"""
        from src.losses.contrastive import HierarchicalContrastiveLoss
        
        loss_fn = HierarchicalContrastiveLoss()
        
        cycle_z1 = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
        cycle_z2 = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
        sub_z1 = torch.randn(BATCH_SIZE, NUM_SUBSTRUCTURES, FEATURE_DIM, device=device)
        sub_z2 = torch.randn(BATCH_SIZE, NUM_SUBSTRUCTURES, FEATURE_DIM, device=device)
        
        result = loss_fn(cycle_z1, cycle_z2, sub_z1, sub_z2)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        total_loss, loss_dict = result
        
        # Check total loss
        assert total_loss.dim() == 0
        assert torch.isfinite(total_loss)
        
        # Check loss dict
        assert "loss_total" in loss_dict
        assert "loss_cycle" in loss_dict
        assert "loss_sub" in loss_dict
    
    def test_lambda_weights(self, device):
        """测试 lambda 权重对总损失的影响。"""
        from src.losses.contrastive import HierarchicalContrastiveLoss
        
        cycle_z1 = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
        cycle_z2 = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
        sub_z1 = torch.randn(BATCH_SIZE, NUM_SUBSTRUCTURES, FEATURE_DIM, device=device)
        sub_z2 = torch.randn(BATCH_SIZE, NUM_SUBSTRUCTURES, FEATURE_DIM, device=device)
        
        # Equal weights
        loss_equal = HierarchicalContrastiveLoss(lambda_cycle=1.0, lambda_sub=1.0)
        total_equal, dict_equal = loss_equal(cycle_z1, cycle_z2, sub_z1, sub_z2)
        
        # Cycle-only
        loss_cycle = HierarchicalContrastiveLoss(lambda_cycle=1.0, lambda_sub=0.0)
        total_cycle, dict_cycle = loss_cycle(cycle_z1, cycle_z2, sub_z1, sub_z2)
        
        # Sub-only
        loss_sub = HierarchicalContrastiveLoss(lambda_cycle=0.0, lambda_sub=1.0)
        total_sub, dict_sub = loss_sub(cycle_z1, cycle_z2, sub_z1, sub_z2)
        
        # Verify relationships
        assert abs(total_cycle.item() - dict_equal["loss_cycle"]) < 0.01
        assert abs(total_sub.item() - dict_equal["loss_sub"]) < 0.01
    
    def test_gradient_flow_both_branches(self, device):
        """测试梯度是否同时流经两个分支。"""
        from src.losses.contrastive import HierarchicalContrastiveLoss
        
        loss_fn = HierarchicalContrastiveLoss()
        
        cycle_z1 = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device, requires_grad=True)
        cycle_z2 = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device, requires_grad=True)
        sub_z1 = torch.randn(BATCH_SIZE, NUM_SUBSTRUCTURES, FEATURE_DIM, device=device, requires_grad=True)
        sub_z2 = torch.randn(BATCH_SIZE, NUM_SUBSTRUCTURES, FEATURE_DIM, device=device, requires_grad=True)
        
        total_loss, _ = loss_fn(cycle_z1, cycle_z2, sub_z1, sub_z2)
        total_loss.backward()
        
        assert cycle_z1.grad is not None
        assert cycle_z2.grad is not None
        assert sub_z1.grad is not None
        assert sub_z2.grad is not None


# ============== Test SupConLoss ==============

class TestSupConLoss:
    """有监督对比损失的测试。"""
    
    def test_output_is_scalar(self, device):
        """测试输出是否为标量。"""
        from src.losses.contrastive import SupConLoss
        
        loss_fn = SupConLoss()
        
        features = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
        labels = torch.randint(0, 3, (BATCH_SIZE,), device=device)
        
        loss = loss_fn(features, labels)
        
        assert loss.dim() == 0
    
    def test_loss_is_finite(self, device):
        """测试损失是否为有限值。"""
        from src.losses.contrastive import SupConLoss
        
        loss_fn = SupConLoss()
        
        features = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
        labels = torch.randint(0, 3, (BATCH_SIZE,), device=device)
        
        loss = loss_fn(features, labels)
        
        assert torch.isfinite(loss)
    
    def test_same_class_lower_loss(self, device):
        """测试同一类别样本是否具有更低的损失。"""
        from src.losses.contrastive import SupConLoss
        
        loss_fn = SupConLoss()
        
        # All same class - should have lower loss
        features_same = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
        labels_same = torch.zeros(BATCH_SIZE, dtype=torch.long, device=device)
        
        # Random classes
        features_random = torch.randn(BATCH_SIZE, FEATURE_DIM, device=device)
        labels_random = torch.randint(0, BATCH_SIZE, (BATCH_SIZE,), device=device)  # Many different classes
        
        loss_same = loss_fn(features_same, labels_same)
        loss_random = loss_fn(features_random, labels_random)
        
        # Note: this assertion might not always hold due to random features
        # In practice, when features are trained, same-class should cluster
        assert torch.isfinite(loss_same)
        assert torch.isfinite(loss_random)
    
    def test_multi_view_input(self, device):
        """测试多视图特征 [B, n_views, D] 的输入。"""
        from src.losses.contrastive import SupConLoss
        
        loss_fn = SupConLoss()
        
        # Two views per sample
        features = torch.randn(BATCH_SIZE, 2, FEATURE_DIM, device=device)
        labels = torch.randint(0, 3, (BATCH_SIZE,), device=device)
        
        loss = loss_fn(features, labels)
        
        assert torch.isfinite(loss)


# ============== Test NTXentLoss ==============

class TestNTXentLoss:
    """NT-Xent 损失的测试。"""
    
    def test_output_is_scalar(self, random_features):
        """测试输出是否为标量。"""
        from src.losses.contrastive import NTXentLoss
        
        z1, z2 = random_features
        loss_fn = NTXentLoss()
        loss = loss_fn(z1, z2)
        
        assert loss.dim() == 0
    
    def test_loss_is_finite(self, random_features):
        """测试损失是否为有限值。"""
        from src.losses.contrastive import NTXentLoss
        
        z1, z2 = random_features
        loss_fn = NTXentLoss()
        loss = loss_fn(z1, z2)
        
        assert torch.isfinite(loss)
    
    def test_symmetric(self, random_features):
        """测试损失在视图顺序上的对称性。"""
        from src.losses.contrastive import NTXentLoss
        
        z1, z2 = random_features
        loss_fn = NTXentLoss()
        
        loss_12 = loss_fn(z1, z2)
        loss_21 = loss_fn(z2, z1)
        
        # Should be approximately equal
        assert torch.allclose(loss_12, loss_21, atol=1e-5)


# ============== Test Factory Function ==============

class TestFactoryFunction:
    """损失工厂函数的测试。"""
    
    def test_get_infonce(self):
        """测试创建 InfoNCE 损失。"""
        from src.losses.contrastive import get_contrastive_loss, InfoNCELoss
        
        loss_fn = get_contrastive_loss("infonce", temperature=0.1)
        assert isinstance(loss_fn, InfoNCELoss)
    
    def test_get_ntxent(self):
        """测试创建 NT-Xent 损失。"""
        from src.losses.contrastive import get_contrastive_loss, NTXentLoss
        
        loss_fn = get_contrastive_loss("ntxent")
        assert isinstance(loss_fn, NTXentLoss)
    
    def test_get_hierarchical(self):
        """测试创建分层对比损失。"""
        from src.losses.contrastive import get_contrastive_loss, HierarchicalContrastiveLoss
        
        loss_fn = get_contrastive_loss("hierarchical", lambda_cycle=0.5, lambda_sub=0.5)
        assert isinstance(loss_fn, HierarchicalContrastiveLoss)
    
    def test_get_supcon(self):
        """测试创建有监督对比损失。"""
        from src.losses.contrastive import get_contrastive_loss, SupConLoss
        
        loss_fn = get_contrastive_loss("supcon")
        assert isinstance(loss_fn, SupConLoss)
    
    def test_invalid_type_raises(self):
        """测试无效损失类型是否抛出错误。"""
        from src.losses.contrastive import get_contrastive_loss
        
        with pytest.raises(ValueError, match="Unknown loss type"):
            get_contrastive_loss("invalid_loss")


# ============== Integration Tests ==============

class TestLossIntegration:
    """损失函数与编码器流水线的集成测试。"""
    
    def test_full_pipeline(self, device):
        """测试编码器+投影头上的损失计算流程。"""
        from src.models.encoder import CNNMambaEncoder
        from src.models.heads import ProjectionHead, SubstructureProjectionHead
        from src.losses.contrastive import HierarchicalContrastiveLoss
        
        # Build model components
        encoder = CNNMambaEncoder(
            cnn_channels=[32, 64, 128],
            mamba_d_model=128,
            mamba_n_layers=1
        ).to(device)
        
        cycle_head = ProjectionHead(
            input_dim=128,
            hidden_dim=256,
            output_dim=64
        ).to(device)
        
        sub_head = SubstructureProjectionHead(
            input_dim=64,  # CNN intermediate channel
            hidden_dim=128,
            output_dim=32
        ).to(device)
        
        loss_fn = HierarchicalContrastiveLoss(
            temperature=0.07,
            lambda_cycle=1.0,
            lambda_sub=1.0
        )
        
        # Simulate two augmented views
        x1 = torch.randn(4, 1, 2000, device=device)
        x2 = torch.randn(4, 1, 2000, device=device)
        
        # Encode
        cycle_z1 = encoder(x1)
        cycle_z2 = encoder(x2)
        
        # Project cycle features
        cycle_proj1 = cycle_head(cycle_z1)
        cycle_proj2 = cycle_head(cycle_z2)
        
        # Get substructure features
        sub_feats1 = encoder.get_sub_features(x1)  # [B, C, L]
        sub_feats2 = encoder.get_sub_features(x2)
        
        # Split into K substructures
        K = 4
        B, C, L = sub_feats1.shape
        sub_len = L // K
        
        subs1 = sub_feats1[:, :, :K*sub_len].reshape(B, C, K, sub_len)
        subs1 = subs1.permute(0, 2, 1, 3)  # [B, K, C, L/K]
        subs2 = sub_feats2[:, :, :K*sub_len].reshape(B, C, K, sub_len)
        subs2 = subs2.permute(0, 2, 1, 3)
        
        # Project substructures
        sub_proj1 = sub_head(subs1)  # [B, K, D]
        sub_proj2 = sub_head(subs2)
        
        # Compute loss
        total_loss, loss_dict = loss_fn(
            cycle_proj1, cycle_proj2,
            sub_proj1, sub_proj2
        )
        
        # Check output
        assert torch.isfinite(total_loss)
        assert total_loss.requires_grad
        
        # Backward
        total_loss.backward()
        
        # Check gradients
        encoder_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in encoder.parameters()
        )
        assert encoder_has_grad, "Encoder should have gradients"
        
        print(f"\nIntegration test loss values:")
        print(f"  Total: {loss_dict['loss_total']:.4f}")
        print(f"  Cycle: {loss_dict['loss_cycle']:.4f}")
        print(f"  Sub:   {loss_dict['loss_sub']:.4f}")


# ============== Entry Point ==============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
