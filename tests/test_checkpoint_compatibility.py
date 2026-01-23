"""
PA-HCL 检查点兼容性测试。

测试下游微调能够正确加载不同模式（SimCLR/MoCo）的预训练权重。

作者: PA-HCL Team
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.pahcl import PAHCLModel, build_pahcl_model
from src.trainers.downstream_trainer import load_pretrained_encoder, DownstreamModel
from src.config import load_config
from types import SimpleNamespace


def create_mock_config(use_moco=False):
    """创建模拟配置。"""
    config = SimpleNamespace()
    
    # Model config
    config.model = SimpleNamespace(
        encoder_type='cnn_mamba',
        cnn_channels=[32, 64, 128, 256],
        cnn_kernel_sizes=[7, 5, 5, 3],
        cnn_strides=[2, 2, 2, 2],
        cnn_dropout=0.1,
        mamba_d_model=256,
        mamba_n_layers=4,
        mamba_d_state=16,
        mamba_expand=2,
        mamba_dropout=0.1,
        pool_type='mean',
        proj_hidden_dim=512,
        proj_output_dim=128,
        proj_num_layers=2,
        sub_proj_hidden_dim=256,
        sub_proj_output_dim=64,
        use_moco=use_moco,
        moco_momentum=0.999,
        queue_size=8192 if use_moco else 0,
    )
    
    # Data config
    config.data = SimpleNamespace(
        num_substructures=4,
        sample_rate=5000,
    )
    
    return config


def create_mock_checkpoint(use_moco=False):
    """创建模拟检查点。"""
    config = create_mock_config(use_moco=use_moco)
    model = build_pahcl_model(config)
    
    checkpoint = {
        'epoch': 100,
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_val_loss': 0.1234,
    }
    
    return checkpoint, config


class TestCheckpointCompatibility:
    """检查点兼容性测试套件。"""
    
    def test_load_simclr_checkpoint(self):
        """测试加载 SimCLR 模式预训练权重。"""
        # 创建 SimCLR 检查点
        checkpoint, config = create_mock_checkpoint(use_moco=False)
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
            torch.save(checkpoint, temp_path)
        
        try:
            # 加载预训练编码器
            device = torch.device('cpu')
            model = load_pretrained_encoder(temp_path, device, config=None)
            
            # 验证
            assert isinstance(model, PAHCLModel)
            assert hasattr(model, 'encoder')
            assert model.use_moco == False  # 下游任务不应该有 MoCo
            
            # 验证不包含动量编码器
            assert model.encoder_momentum is None
            assert model.cycle_projector_momentum is None
            
            print("✓ SimCLR 检查点加载成功")
        finally:
            os.unlink(temp_path)
    
    def test_load_moco_checkpoint(self):
        """测试加载 MoCo 模式预训练权重。"""
        # 创建 MoCo 检查点
        checkpoint, config = create_mock_checkpoint(use_moco=True)
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
            torch.save(checkpoint, temp_path)
        
        try:
            # 加载预训练编码器（应该自动过滤 MoCo 参数）
            device = torch.device('cpu')
            model = load_pretrained_encoder(temp_path, device, config=None)
            
            # 验证
            assert isinstance(model, PAHCLModel)
            assert hasattr(model, 'encoder')
            assert model.use_moco == False  # 下游任务强制禁用 MoCo
            
            # 验证不包含动量编码器（即使预训练模型有）
            assert model.encoder_momentum is None
            assert model.cycle_projector_momentum is None
            
            print("✓ MoCo 检查点加载成功（自动过滤动量编码器）")
        finally:
            os.unlink(temp_path)
    
    def test_downstream_with_simclr(self):
        """测试使用 SimCLR 权重进行下游微调。"""
        # 创建 SimCLR 检查点
        checkpoint, config = create_mock_checkpoint(use_moco=False)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
            torch.save(checkpoint, temp_path)
        
        try:
            # 加载预训练编码器
            device = torch.device('cpu')
            pretrained_model = load_pretrained_encoder(temp_path, device)
            
            # 获取编码器
            encoder = pretrained_model.encoder
            encoder_dim = pretrained_model.encoder_dim
            
            # 创建下游模型
            downstream_model = DownstreamModel(
                encoder=encoder,
                num_classes=3,
                encoder_dim=encoder_dim,
                hidden_dim=128,
                dropout=0.3,
                freeze_encoder=False
            )
            
            # 测试前向传播
            batch_size = 4
            seq_len = 4000
            x = torch.randn(batch_size, 1, seq_len)
            
            logits = downstream_model(x)
            
            # 验证输出
            assert logits.shape == (batch_size, 3)
            assert not torch.isnan(logits).any()
            
            print("✓ SimCLR 下游微调模型创建成功")
        finally:
            os.unlink(temp_path)
    
    def test_downstream_with_moco(self):
        """测试使用 MoCo 权重进行下游微调。"""
        # 创建 MoCo 检查点
        checkpoint, config = create_mock_checkpoint(use_moco=True)
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
            torch.save(checkpoint, temp_path)
        
        try:
            # 加载预训练编码器（关键：应该能正常加载）
            device = torch.device('cpu')
            pretrained_model = load_pretrained_encoder(temp_path, device)
            
            # 获取编码器
            encoder = pretrained_model.encoder
            encoder_dim = pretrained_model.encoder_dim
            
            # 创建下游模型
            downstream_model = DownstreamModel(
                encoder=encoder,
                num_classes=2,
                encoder_dim=encoder_dim,
                hidden_dim=None,  # 线性分类器
                dropout=0.1,
                freeze_encoder=True  # 线性评估
            )
            
            # 测试前向传播
            batch_size = 8
            seq_len = 4000
            x = torch.randn(batch_size, 1, seq_len)
            
            logits = downstream_model(x)
            
            # 验证输出
            assert logits.shape == (batch_size, 2)
            assert not torch.isnan(logits).any()
            
            # 验证编码器被冻结
            for param in encoder.parameters():
                assert not param.requires_grad
            
            print("✓ MoCo 下游微调模型创建成功（线性评估）")
        finally:
            os.unlink(temp_path)
    
    def test_state_dict_filtering(self):
        """测试 state_dict 过滤逻辑。"""
        # 创建 MoCo 检查点
        checkpoint, config = create_mock_checkpoint(use_moco=True)
        state_dict = checkpoint['model_state_dict']
        
        # 统计 MoCo 相关键
        moco_keys = ['encoder_momentum', 'cycle_projector_momentum', 'queue', 'queue_ptr']
        moco_count = sum(1 for key in state_dict.keys() 
                        if any(moco_key in key for moco_key in moco_keys))
        
        assert moco_count > 0, "MoCo 检查点应该包含动量编码器参数"
        
        # 过滤
        filtered_state_dict = {
            k: v for k, v in state_dict.items()
            if not any(moco_key in k for moco_key in moco_keys)
        }
        
        # 验证过滤结果
        assert len(filtered_state_dict) < len(state_dict)
        assert len(filtered_state_dict) == len(state_dict) - moco_count
        
        print(f"✓ 成功过滤 {moco_count} 个 MoCo 参数")
    
    def test_encoder_weights_preserved(self):
        """测试编码器权重是否正确保留。"""
        # 创建 MoCo 检查点
        checkpoint, config = create_mock_checkpoint(use_moco=True)
        
        # 获取原始编码器的某个参数
        original_state_dict = checkpoint['model_state_dict']
        original_encoder_param = None
        for key, value in original_state_dict.items():
            if 'encoder.' in key and 'encoder_momentum' not in key:
                original_encoder_param = (key, value.clone())
                break
        
        assert original_encoder_param is not None
        
        # 保存并加载
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
            torch.save(checkpoint, temp_path)
        
        try:
            device = torch.device('cpu')
            model = load_pretrained_encoder(temp_path, device)
            
            # 获取加载后的参数
            loaded_param = None
            for name, param in model.named_parameters():
                if name == original_encoder_param[0]:
                    loaded_param = param
                    break
            
            # 验证参数相同
            assert loaded_param is not None
            assert torch.allclose(loaded_param, original_encoder_param[1])
            
            print("✓ 编码器权重正确保留")
        finally:
            os.unlink(temp_path)


def run_tests():
    """运行所有测试。"""
    print("\n" + "="*60)
    print("PA-HCL 检查点兼容性测试")
    print("="*60 + "\n")
    
    test_suite = TestCheckpointCompatibility()
    
    tests = [
        ("加载 SimCLR 检查点", test_suite.test_load_simclr_checkpoint),
        ("加载 MoCo 检查点", test_suite.test_load_moco_checkpoint),
        ("SimCLR 下游微调", test_suite.test_downstream_with_simclr),
        ("MoCo 下游微调", test_suite.test_downstream_with_moco),
        ("State Dict 过滤", test_suite.test_state_dict_filtering),
        ("编码器权重保留", test_suite.test_encoder_weights_preserved),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"测试: {name}...")
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ 失败: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
        print()
    
    print("="*60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
