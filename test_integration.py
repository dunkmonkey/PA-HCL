#!/usr/bin/env python
"""
快速集成测试：验证新编码器在实际训练流程中可用。
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from src.models.pahcl import PAHCLModel, build_pahcl_model
from src.losses.contrastive import HierarchicalContrastiveLoss
from types import SimpleNamespace

def test_training_loop():
    """模拟训练循环"""
    print("=" * 60)
    print("集成测试: 模拟训练循环")
    print("=" * 60)
    
    # 创建模型配置
    config = SimpleNamespace(
        data=SimpleNamespace(
            num_substructures=4,
            sample_rate=5000
        ),
        model=SimpleNamespace(
            encoder_type="sincnet_eca_mamba",
            in_channels=1,
            mamba_d_model=128,
            mamba_n_layers=2,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_dropout=0.1,
            pool_type="asp",
            cycle_proj_hidden=512,
            cycle_proj_output=128,
            cycle_proj_layers=2,
            sub_proj_hidden=256,
            sub_proj_output=64,
            num_substructures=4,
            use_moco=False,
        ),
        loss=SimpleNamespace(
            temperature=0.07,
            lambda_cycle=1.0,
            lambda_sub=1.0,
            align_substructures=True
        )
    )
    
    # 构建模型
    print("\n1. 构建 PAHCLModel...")
    model = build_pahcl_model(config)
    print(f"   ✓ 模型创建成功")
    print(f"   - 编码器类型: {type(model.encoder).__name__}")
    print(f"   - 总参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建损失函数
    print("\n2. 创建损失函数...")
    criterion = HierarchicalContrastiveLoss(
        temperature=config.loss.temperature,
        lambda_cycle=config.loss.lambda_cycle,
        lambda_sub=config.loss.lambda_sub,
        align_substructures=config.loss.align_substructures
    )
    print(f"   ✓ 损失函数创建成功")
    
    # 创建优化器
    print("\n3. 创建优化器...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    print(f"   ✓ 优化器创建成功")
    
    # 模拟训练批次
    print("\n4. 模拟训练步骤...")
    model.train()
    
    batch_size = 4
    seq_len = 5000
    view1 = torch.randn(batch_size, 1, seq_len)
    view2 = torch.randn(batch_size, 1, seq_len)
    
    # 前向传播
    outputs = model.forward_pretrain(view1, view2)
    print(f"   ✓ 前向传播成功")
    print(f"   - 输出键: {list(outputs.keys())}")
    
    # 计算损失
    loss, loss_dict = criterion(
        cycle_z1=outputs['cycle_proj1'],
        cycle_z2=outputs['cycle_proj2'],
        sub_z1=outputs['sub_proj1'],
        sub_z2=outputs['sub_proj2']
    )
    print(f"   ✓ 损失计算成功")
    print(f"   - 总损失: {loss.item():.4f}")
    print(f"   - 周期损失: {loss_dict['loss_cycle']:.4f}")
    print(f"   - 子结构损失: {loss_dict['loss_sub']:.4f}")
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    print(f"   ✓ 反向传播成功")
    
    # 检查梯度
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"   ✓ 梯度检查: {has_grad}/{total_params} 参数有梯度")
    
    # 优化器步骤
    optimizer.step()
    print(f"   ✓ 优化器更新成功")
    
    print("\n" + "=" * 60)
    print("✅ 集成测试通过！新编码器可在实际训练中使用。")
    print("=" * 60)
    

def test_checkpoint_save_load():
    """测试检查点保存和加载"""
    print("\n" + "=" * 60)
    print("检查点测试: 保存和加载")
    print("=" * 60)
    
    # 创建模型
    model = PAHCLModel(
        encoder_type="sincnet_eca_mamba",
        mamba_d_model=128,
        mamba_n_layers=2,
        cycle_proj_output=128,
        sub_proj_output=64
    )
    
    # 保存检查点
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        checkpoint_path = f.name
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'encoder_type': 'sincnet_eca_mamba',
        'config': {
            'mamba_d_model': 128,
            'mamba_n_layers': 2
        }
    }, checkpoint_path)
    print(f"   ✓ 检查点已保存: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 创建新模型并加载权重
    model_new = PAHCLModel(
        encoder_type="sincnet_eca_mamba",
        mamba_d_model=128,
        mamba_n_layers=2,
        cycle_proj_output=128,
        sub_proj_output=64
    )
    model_new.load_state_dict(checkpoint['model_state_dict'])
    print(f"   ✓ 检查点已加载到新模型")
    
    # 设置为评估模式（消除 dropout 等随机性）
    model.eval()
    model_new.eval()
    
    # 测试前向传播
    x = torch.randn(2, 1, 5000)
    with torch.no_grad():
        out1 = model(x)
        out2 = model_new(x)
    
    # 比较输出（应该相同）
    diff = torch.abs(out1['cycle_proj'] - out2['cycle_proj']).max()
    print(f"   ✓ 输出差异: {diff.item():.10f}")
    assert diff < 1e-5, f"加载后的模型输出不一致！差异: {diff.item()}"
    
    # 清理
    import os
    os.unlink(checkpoint_path)
    
    print("\n✅ 检查点测试通过！")


if __name__ == "__main__":
    test_training_loop()
    test_checkpoint_save_load()
    
    print("\n" + "=" * 60)
    print("🎉 所有集成测试通过！可以开始使用新编码器训练模型。")
    print("=" * 60)
    print("\n推荐命令:")
    print("  python scripts/pretrain.py --config configs/pretrain.yaml")
    print("\n提示: 请先在配置文件中设置 encoder_type: 'sincnet_eca_mamba'")
