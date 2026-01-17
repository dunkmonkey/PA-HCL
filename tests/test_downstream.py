"""
下游微调组件的测试。

运行测试:
    pytest tests/test_downstream.py -v

作者: PA-HCL 团队
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestDownstreamModel:
    """测试 DownstreamModel 类。"""
    
    def test_downstream_model_creation(self):
        """测试创建下游模型。"""
        import torch
        import torch.nn as nn
        from src.trainers.downstream_trainer import DownstreamModel
        
        # Simple mock encoder
        class MockEncoder(nn.Module):
            def __init__(self, output_dim=256):
                super().__init__()
                self.conv = nn.Conv1d(1, output_dim, 7, padding=3)
                self.pool = nn.AdaptiveAvgPool1d(1)
                
            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                x = self.conv(x)
                x = self.pool(x)
                return x.squeeze(-1)  # [B, C]
        
        encoder = MockEncoder(256)
        model = DownstreamModel(
            encoder=encoder,
            num_classes=5,
            encoder_dim=256,
            hidden_dim=128,
            dropout=0.1,
            freeze_encoder=False
        )
        
        # Check forward pass
        x = torch.randn(4, 1, 5000)
        logits = model(x)
        
        assert logits.shape == (4, 5)
        print("✓ DownstreamModel creation and forward pass work correctly")
    
    def test_frozen_encoder(self):
        """测试编码器冻结是否有效。"""
        import torch
        import torch.nn as nn
        from src.trainers.downstream_trainer import DownstreamModel
        
        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(100, 256)
                
            def forward(self, x):
                return self.linear(x.mean(dim=-1) if x.dim() == 2 else x.mean(dim=(1,2)))
        
        encoder = MockEncoder()
        model = DownstreamModel(
            encoder=encoder,
            num_classes=2,
            encoder_dim=256,
            freeze_encoder=True
        )
        
        # Check encoder params are frozen
        for param in model.encoder.parameters():
            assert not param.requires_grad
        
        # Check classifier params are trainable
        for param in model.classifier.parameters():
            assert param.requires_grad
        
        print("✓ Encoder freezing works correctly")
    
    def test_get_features(self):
        """测试特征提取。"""
        import torch
        import torch.nn as nn
        from src.trainers.downstream_trainer import DownstreamModel
        
        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.output_dim = 256
                
            def forward(self, x):
                B = x.shape[0]
                return torch.randn(B, self.output_dim)
        
        encoder = MockEncoder()
        model = DownstreamModel(
            encoder=encoder,
            num_classes=3,
            encoder_dim=256
        )
        
        x = torch.randn(4, 1, 5000)
        features = model.get_features(x)
        
        assert features.shape == (4, 256)
        print("✓ Feature extraction works correctly")


class TestDownstreamTrainer:
    """测试 DownstreamTrainer 类。"""
    
    def test_trainer_initialization(self):
        """测试训练器初始化。"""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from src.trainers.downstream_trainer import DownstreamModel, DownstreamTrainer
        
        # Mock encoder
        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(100, 256)
                
            def forward(self, x):
                return self.fc(x.view(x.shape[0], -1)[:, :100])
        
        encoder = MockEncoder()
        model = DownstreamModel(
            encoder=encoder,
            num_classes=2,
            encoder_dim=256
        )
        
        # Mock data
        signals = torch.randn(32, 1, 5000)
        labels = torch.randint(0, 2, (32,))
        
        # Create dataset with dict batches
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, signals, labels):
                self.signals = signals
                self.labels = labels
                
            def __len__(self):
                return len(self.signals)
                
            def __getitem__(self, idx):
                return {"signal": self.signals[idx], "label": self.labels[idx]}
        
        dataset = DictDataset(signals, labels)
        loader = DataLoader(dataset, batch_size=8)
        
        trainer = DownstreamTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            num_epochs=2,
            learning_rate=1e-4,
            use_amp=False,
            output_dir="/tmp/test_downstream"
        )
        
        assert trainer.model is not None
        assert trainer.optimizer is not None
        print("✓ DownstreamTrainer initialization works")
    
    def test_train_epoch(self):
        """测试单轮训练。"""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from src.trainers.downstream_trainer import DownstreamModel, DownstreamTrainer
        
        # Simple model
        class MockEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(1, 256)
                
            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                x = self.pool(x)
                return self.fc(x.squeeze(-1))
        
        encoder = MockEncoder()
        model = DownstreamModel(encoder=encoder, num_classes=2, encoder_dim=256)
        
        # Mock dataset
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, n_samples=32):
                self.signals = torch.randn(n_samples, 1, 1000)
                self.labels = torch.randint(0, 2, (n_samples,))
                
            def __len__(self):
                return len(self.signals)
                
            def __getitem__(self, idx):
                return {"signal": self.signals[idx], "label": self.labels[idx]}
        
        loader = DataLoader(DictDataset(32), batch_size=8)
        
        trainer = DownstreamTrainer(
            model=model,
            train_loader=loader,
            num_epochs=1,
            use_amp=False,
            output_dir="/tmp/test_train_epoch"
        )
        
        metrics = trainer.train_epoch()
        
        assert "train_loss" in metrics
        assert "accuracy" in metrics
        print(f"✓ Train epoch works - Loss: {metrics['train_loss']:.4f}, Acc: {metrics['accuracy']:.4f}")


class TestMetrics:
    """测试分类指标。"""
    
    def test_compute_classification_metrics(self):
        """测试指标计算。"""
        from src.utils.metrics import compute_metrics
        
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 1, 1, 1, 0]
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        
        assert metrics["accuracy"] == 0.6
        print(f"✓ Metrics computation works - Accuracy: {metrics['accuracy']}")
    
    def test_metrics_with_probabilities(self):
        """测试带有概率分数的指标。"""
        from src.utils.metrics import compute_metrics
        
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 1]
        y_score = [0.1, 0.3, 0.7, 0.8, 0.9]
        
        metrics = compute_metrics(y_true, y_pred, y_score, num_classes=2)
        
        assert "auroc" in metrics
        assert metrics["auroc"] > 0.5  # Should be good since predictions are correct
        print(f"✓ Metrics with probabilities work - AUROC: {metrics['auroc']:.4f}")
    
    def test_multiclass_metrics(self):
        """测试多分类指标。"""
        from src.utils.metrics import compute_metrics
        
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 1, 2]
        
        metrics = compute_metrics(y_true, y_pred, num_classes=3)
        
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        print(f"✓ Multi-class metrics work - F1_macro: {metrics['f1_macro']:.4f}")
    
    def test_optimal_threshold(self):
        """测试查找最佳阈值。"""
        from src.utils.metrics import find_optimal_threshold
        
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
        
        threshold, metrics = find_optimal_threshold(y_true, y_score, method="youden")
        
        assert 0 < threshold < 1
        assert "accuracy" in metrics
        print(f"✓ Optimal threshold finding works - Threshold: {threshold:.3f}")


class TestLoadPretrainedEncoder:
    """测试预训练编码器加载。"""
    
    def test_load_mock_checkpoint(self):
        """测试从检查点加载编码器。"""
        import torch
        import tempfile
        from pathlib import Path
        
        # Create mock checkpoint
        mock_state = {
            "epoch": 10,
            "model_state_dict": {},
            "config": None
        }
        
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(mock_state, f.name)
            checkpoint_path = f.name
        
        # Note: Full loading test requires complete model setup
        # This just tests the checkpoint file structure
        loaded = torch.load(checkpoint_path)
        
        assert "epoch" in loaded
        assert "model_state_dict" in loaded
        print("✓ Checkpoint structure is valid")
        
        # Cleanup
        Path(checkpoint_path).unlink()


class TestIntegration:
    """下游管道的集成测试。"""
    
    def test_full_pipeline_simulation(self):
        """模拟完整的下游训练管道。"""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from src.trainers.downstream_trainer import DownstreamModel, DownstreamTrainer
        
        print("\n" + "=" * 50)
        print("Downstream Pipeline Integration Test")
        print("=" * 50)
        
        # 1. Create mock pretrained encoder
        class MockPretrainedEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 64, 7, stride=2, padding=3)
                self.conv2 = nn.Conv1d(64, 256, 5, stride=2, padding=2)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.bn1 = nn.BatchNorm1d(64)
                self.bn2 = nn.BatchNorm1d(256)
                
            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                x = torch.relu(self.bn1(self.conv1(x)))
                x = torch.relu(self.bn2(self.conv2(x)))
                x = self.pool(x)
                return x.squeeze(-1)
        
        encoder = MockPretrainedEncoder()
        print("✓ Step 1: Created mock pretrained encoder")
        
        # 2. Create downstream model
        model = DownstreamModel(
            encoder=encoder,
            num_classes=5,
            encoder_dim=256,
            hidden_dim=128,
            dropout=0.2,
            freeze_encoder=False
        )
        print("✓ Step 2: Created downstream model")
        
        # 3. Create mock dataset
        class MockPCGDataset(torch.utils.data.Dataset):
            def __init__(self, n_samples=64):
                self.signals = torch.randn(n_samples, 1, 5000)
                self.labels = torch.randint(0, 5, (n_samples,))
                
            def __len__(self):
                return len(self.signals)
                
            def __getitem__(self, idx):
                return {"signal": self.signals[idx], "label": self.labels[idx]}
        
        train_dataset = MockPCGDataset(64)
        val_dataset = MockPCGDataset(16)
        
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
        print("✓ Step 3: Created data loaders")
        
        # 4. Create trainer
        trainer = DownstreamTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            learning_rate=1e-3,
            use_amp=False,
            log_interval=10,
            output_dir="/tmp/test_integration"
        )
        print("✓ Step 4: Initialized trainer")
        
        # 5. Run training epochs
        for epoch in range(2):
            train_metrics = trainer.train_epoch()
            val_metrics = trainer.validate()
            print(f"  Epoch {epoch+1}: Train Loss={train_metrics['train_loss']:.4f}, "
                  f"Val Acc={val_metrics.get('val_accuracy', 0):.4f}")
        
        print("✓ Step 5: Completed training epochs")
        print("=" * 50)
        print("All integration tests passed!")


def run_all_tests():
    """Run all downstream tests."""
    print("\n" + "=" * 60)
    print("PA-HCL Downstream Fine-tuning Tests")
    print("=" * 60 + "\n")
    
    # Model tests
    print("Testing DownstreamModel...")
    test_model = TestDownstreamModel()
    test_model.test_downstream_model_creation()
    test_model.test_frozen_encoder()
    test_model.test_get_features()
    
    # Trainer tests
    print("\nTesting DownstreamTrainer...")
    test_trainer = TestDownstreamTrainer()
    test_trainer.test_trainer_initialization()
    test_trainer.test_train_epoch()
    
    # Metrics tests
    print("\nTesting Metrics...")
    test_metrics = TestMetrics()
    test_metrics.test_compute_classification_metrics()
    test_metrics.test_metrics_with_probabilities()
    test_metrics.test_multiclass_metrics()
    test_metrics.test_optimal_threshold()
    
    # Checkpoint tests
    print("\nTesting Checkpoint Loading...")
    test_checkpoint = TestLoadPretrainedEncoder()
    test_checkpoint.test_load_mock_checkpoint()
    
    # Integration tests
    print("\nRunning Integration Tests...")
    test_integration = TestIntegration()
    test_integration.test_full_pipeline_simulation()
    
    print("\n" + "=" * 60)
    print("All downstream tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
