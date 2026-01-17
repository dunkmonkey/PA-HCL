"""
Unit tests for projection and classification heads.

这是 PA-HCL 项目投影头和分类头模块的单元测试文件。

=============================================================================
测试运行指南 (Test Execution Guide)
=============================================================================

1. 环境准备:
   cd /workspaces/PA-HCL
   pip install -e .
   # 或确保以下依赖已安装:
   pip install torch pytest numpy

2. 运行所有 Heads 测试:
   pytest tests/test_heads.py -v

3. 运行特定测试类:
   pytest tests/test_heads.py::TestProjectionHead -v
   pytest tests/test_heads.py::TestPredictionHead -v
   pytest tests/test_heads.py::TestClassificationHead -v
   pytest tests/test_heads.py::TestAnomalyDetectionHead -v
   pytest tests/test_heads.py::TestSubstructureProjectionHead -v

4. 运行带覆盖率的测试:
   pytest tests/test_heads.py -v --cov=src/models/heads

5. 快速测试:
   pytest tests/test_heads.py -v -k "output_shape"

6. 预期结果:
   - 所有测试通过 (green)
   - 投影头正确降维
   - 分类头输出正确类别数

=============================================================================

Author: PA-HCL Team
"""

import pytest
import torch
import torch.nn as nn

# Test configuration
BATCH_SIZE = 8
INPUT_DIM = 256
HIDDEN_DIM = 512
OUTPUT_DIM = 128
NUM_CLASSES = 5


# ============== Fixtures ==============

@pytest.fixture
def device():
    """获取合适的设备。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_features(device):
    """生成样本编码器特征。"""
    return torch.randn(BATCH_SIZE, INPUT_DIM, device=device)


@pytest.fixture
def sample_sequence(device):
    """生成样本序列特征。"""
    seq_len = 32
    return torch.randn(BATCH_SIZE, seq_len, INPUT_DIM, device=device)


@pytest.fixture
def projection_head(device):
    """创建投影头实例。"""
    from src.models.heads import ProjectionHead
    return ProjectionHead(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=2
    ).to(device)


@pytest.fixture
def classification_head(device):
    """创建分类头实例。"""
    from src.models.heads import ClassificationHead
    return ClassificationHead(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM
    ).to(device)


# ============== Test ProjectionHead ==============

class TestProjectionHead:
    """MLP 投影头的测试。"""
    
    def test_output_shape(self, projection_head, sample_features):
        """测试输出维度是否正确。"""
        output = projection_head(sample_features)
        assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
    
    def test_sequence_input(self, projection_head, sample_sequence):
        """测试投影头处理序列输入。"""
        output = projection_head(sample_sequence)
        assert output.shape == (BATCH_SIZE, 32, OUTPUT_DIM)
    
    def test_different_num_layers(self, device):
        """测试不同的 MLP 深度。"""
        from src.models.heads import ProjectionHead
        
        x = torch.randn(4, INPUT_DIM, device=device)
        
        for num_layers in [1, 2, 3]:
            head = ProjectionHead(
                input_dim=INPUT_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=OUTPUT_DIM,
                num_layers=num_layers
            ).to(device)
            
            output = head(x)
            assert output.shape == (4, OUTPUT_DIM), \
                f"num_layers={num_layers} failed"
    
    def test_with_batch_norm(self, device):
        """使用 BatchNorm 测试投影头。"""
        from src.models.heads import ProjectionHead
        
        head = ProjectionHead(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            use_bn=True,
            last_bn=True
        ).to(device)
        
        x = torch.randn(8, INPUT_DIM, device=device)
        output = head(x)
        assert output.shape == (8, OUTPUT_DIM)
    
    def test_without_batch_norm(self, device):
        """不使用 BatchNorm 测试投影头。"""
        from src.models.heads import ProjectionHead
        
        head = ProjectionHead(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            use_bn=False
        ).to(device)
        
        x = torch.randn(8, INPUT_DIM, device=device)
        output = head(x)
        assert output.shape == (8, OUTPUT_DIM)
    
    def test_gradient_flow(self, projection_head, sample_features):
        """测试梯度流经投影头。"""
        sample_features.requires_grad_(True)
        output = projection_head(sample_features)
        
        loss = output.sum()
        loss.backward()
        
        assert sample_features.grad is not None
    
    def test_invalid_num_layers(self, device):
        """测试无效层数的错误。"""
        from src.models.heads import ProjectionHead
        
        with pytest.raises(ValueError, match="num_layers must be"):
            ProjectionHead(
                input_dim=INPUT_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=OUTPUT_DIM,
                num_layers=5
            )


# ============== Test PredictionHead ==============

class TestPredictionHead:
    """BYOL 风格预测头的测试。"""
    
    def test_output_shape(self, device):
        """测试预测头输出形状。"""
        from src.models.heads import PredictionHead
        
        head = PredictionHead(
            input_dim=OUTPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM
        ).to(device)
        
        x = torch.randn(8, OUTPUT_DIM, device=device)
        output = head(x)
        
        assert output.shape == (8, OUTPUT_DIM)
    
    def test_different_dimensions(self, device):
        """测试不同维度的预测头。"""
        from src.models.heads import PredictionHead
        
        for in_dim, out_dim in [(64, 64), (128, 64), (256, 128)]:
            head = PredictionHead(
                input_dim=in_dim,
                hidden_dim=256,
                output_dim=out_dim
            ).to(device)
            
            x = torch.randn(4, in_dim, device=device)
            output = head(x)
            assert output.shape == (4, out_dim)


# ============== Test ClassificationHead ==============

class TestClassificationHead:
    """分类头的测试。"""
    
    def test_output_shape(self, classification_head, sample_features):
        """测试输出是否为具有正确类别的 logits。"""
        output = classification_head(sample_features)
        assert output.shape == (BATCH_SIZE, NUM_CLASSES)
    
    def test_linear_classifier(self, device):
        """测试线性分类器 (无隐藏层)。"""
        from src.models.heads import ClassificationHead
        
        head = ClassificationHead(
            input_dim=INPUT_DIM,
            num_classes=NUM_CLASSES,
            hidden_dim=None  # Linear classifier
        ).to(device)
        
        x = torch.randn(8, INPUT_DIM, device=device)
        output = head(x)
        
        assert output.shape == (8, NUM_CLASSES)
    
    def test_mlp_classifier(self, device):
        """测试带有隐藏层的 MLP 分类器。"""
        from src.models.heads import ClassificationHead
        
        head = ClassificationHead(
            input_dim=INPUT_DIM,
            num_classes=NUM_CLASSES,
            hidden_dim=128
        ).to(device)
        
        x = torch.randn(8, INPUT_DIM, device=device)
        output = head(x)
        
        assert output.shape == (8, NUM_CLASSES)
    
    def test_multiclass_classification(self, classification_head, sample_features):
        """测试输出可用于多类分类。"""
        output = classification_head(sample_features)
        
        # Apply softmax for probabilities
        probs = torch.softmax(output, dim=-1)
        
        # Should sum to 1
        assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH_SIZE, device=probs.device))
    
    def test_binary_classification(self, device):
        """测试二元分类设置。"""
        from src.models.heads import ClassificationHead
        
        head = ClassificationHead(
            input_dim=INPUT_DIM,
            num_classes=2  # Binary
        ).to(device)
        
        x = torch.randn(8, INPUT_DIM, device=device)
        output = head(x)
        
        assert output.shape == (8, 2)


# ============== Test AnomalyDetectionHead ==============

class TestAnomalyDetectionHead:
    """异常检测头的测试。"""
    
    def test_euclidean_distance(self, device):
        """测试使用欧几里得距离的异常评分。"""
        from src.models.heads import AnomalyDetectionHead
        
        head = AnomalyDetectionHead(
            input_dim=INPUT_DIM,
            num_prototypes=1,
            distance_type="euclidean"
        ).to(device)
        
        x = torch.randn(8, INPUT_DIM, device=device)
        scores = head(x)
        
        assert scores.shape == (8, 1)
        assert (scores >= 0).all()  # Distances are non-negative
    
    def test_mahalanobis_distance(self, device):
        """测试使用马氏距离的异常评分。"""
        from src.models.heads import AnomalyDetectionHead
        
        head = AnomalyDetectionHead(
            input_dim=INPUT_DIM,
            num_prototypes=1,
            distance_type="mahalanobis"
        ).to(device)
        
        x = torch.randn(8, INPUT_DIM, device=device)
        scores = head(x)
        
        assert scores.shape == (8, 1)
    
    def test_cosine_distance(self, device):
        """测试使用余弦距离的异常评分。"""
        from src.models.heads import AnomalyDetectionHead
        
        head = AnomalyDetectionHead(
            input_dim=INPUT_DIM,
            num_prototypes=1,
            distance_type="cosine"
        ).to(device)
        
        x = torch.randn(8, INPUT_DIM, device=device)
        scores = head(x)
        
        assert scores.shape == (8, 1)
        # Cosine distance is in [0, 2]
        assert (scores >= 0).all()
        assert (scores <= 2).all()
    
    def test_multiple_prototypes(self, device):
        """使用多个原型向量进行测试。"""
        from src.models.heads import AnomalyDetectionHead
        
        head = AnomalyDetectionHead(
            input_dim=INPUT_DIM,
            num_prototypes=5,
            distance_type="euclidean"
        ).to(device)
        
        x = torch.randn(8, INPUT_DIM, device=device)
        scores = head(x)
        
        assert scores.shape == (8, 5)
    
    def test_fit_prototypes(self, device):
        """测试从训练数据拟合原型。"""
        from src.models.heads import AnomalyDetectionHead
        
        head = AnomalyDetectionHead(
            input_dim=INPUT_DIM,
            num_prototypes=3,
            distance_type="euclidean"
        ).to(device)
        
        # Simulate training features
        features = torch.randn(100, INPUT_DIM, device=device)
        
        # Fit prototypes
        head.fit_prototypes(features)
        
        # Prototypes should be updated
        assert head.prototypes.shape == (3, INPUT_DIM)


# ============== Test SubstructureProjectionHead ==============

class TestSubstructureProjectionHead:
    """子结构特定投影头的测试。"""
    
    def test_single_substructure(self, device):
        """使用单个子结构特征图进行测试。"""
        from src.models.heads import SubstructureProjectionHead
        
        head = SubstructureProjectionHead(
            input_dim=128,
            hidden_dim=256,
            output_dim=64
        ).to(device)
        
        # [B, C, L]
        x = torch.randn(8, 128, 32, device=device)
        output = head(x)
        
        assert output.shape == (8, 64)
    
    def test_batched_substructures(self, device):
        """使用批量子结构特征 [B, K, C, L] 进行测试。"""
        from src.models.heads import SubstructureProjectionHead
        
        head = SubstructureProjectionHead(
            input_dim=128,
            hidden_dim=256,
            output_dim=64
        ).to(device)
        
        # [B, K, C, L] where K=4 substructures
        x = torch.randn(8, 4, 128, 32, device=device)
        output = head(x)
        
        # Should return [B, K, D]
        assert output.shape == (8, 4, 64)
    
    def test_pool_types(self, device):
        """测试不同的池化策略。"""
        from src.models.heads import SubstructureProjectionHead
        
        x = torch.randn(8, 128, 32, device=device)
        
        for pool_type in ["mean", "max", "adaptive"]:
            head = SubstructureProjectionHead(
                input_dim=128,
                hidden_dim=256,
                output_dim=64,
                pool_type=pool_type
            ).to(device)
            
            output = head(x)
            assert output.shape == (8, 64), f"Pool type {pool_type} failed"


# ============== Integration Tests ==============

class TestHeadsIntegration:
    """带有编码器的头的集成测试。"""
    
    def test_contrastive_learning_pipeline(self, device):
        """测试完整的对比学习流程。"""
        from src.models.encoder import CNNMambaEncoder
        from src.models.heads import ProjectionHead
        
        # Create encoder and projection head
        encoder = CNNMambaEncoder(
            cnn_channels=[32, 64, 128],
            cnn_strides=[2, 2, 2],
            mamba_d_model=128,
            mamba_n_layers=1
        ).to(device)
        
        proj_head = ProjectionHead(
            input_dim=128,
            hidden_dim=256,
            output_dim=64
        ).to(device)
        
        # Forward pass
        x = torch.randn(4, 1, 2000, device=device)
        features = encoder(x)
        projections = proj_head(features)
        
        # Check shapes
        assert features.shape == (4, 128)
        assert projections.shape == (4, 64)
    
    def test_downstream_classification_pipeline(self, device):
        """测试完整的下游分类流程。"""
        from src.models.encoder import CNNMambaEncoder
        from src.models.heads import ClassificationHead
        
        encoder = CNNMambaEncoder(
            cnn_channels=[32, 64, 128],
            mamba_d_model=128,
            mamba_n_layers=1
        ).to(device)
        
        classifier = ClassificationHead(
            input_dim=128,
            num_classes=3,
            hidden_dim=64
        ).to(device)
        
        x = torch.randn(4, 1, 2000, device=device)
        features = encoder(x)
        logits = classifier(features)
        
        assert logits.shape == (4, 3)
    
    def test_substructure_contrastive_pipeline(self, device):
        """测试子结构级对比学习流程。"""
        from src.models.encoder import CNNMambaEncoder
        from src.models.heads import SubstructureProjectionHead
        
        encoder = CNNMambaEncoder(
            cnn_channels=[32, 64, 128],
            mamba_d_model=128,
            mamba_n_layers=1
        ).to(device)
        
        sub_head = SubstructureProjectionHead(
            input_dim=64,  # Second-to-last CNN layer
            hidden_dim=128,
            output_dim=32
        ).to(device)
        
        x = torch.randn(4, 1, 2000, device=device)
        sub_features = encoder.get_sub_features(x)  # [B, C, L']
        
        # Simulate splitting into K=4 substructures
        B, C, L = sub_features.shape
        K = 4
        sub_len = L // K
        substructures = sub_features[:, :, :K*sub_len].reshape(B, C, K, sub_len)
        substructures = substructures.permute(0, 2, 1, 3)  # [B, K, C, L/K]
        
        projections = sub_head(substructures)
        
        assert projections.shape == (4, K, 32)


# ============== Entry Point ==============

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
