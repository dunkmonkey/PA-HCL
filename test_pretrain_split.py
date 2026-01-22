#!/usr/bin/env python
"""
测试预训练数据集划分是否正确。

此脚本验证:
1. 受试者级划分是否正确
2. 训练集和验证集没有重叠
3. 数据增强是否应用
"""

import sys
from pathlib import Path

# 将项目根目录添加到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data.dataset import PCGPretrainDataset, create_subject_wise_split
from src.data.transforms import get_pretrain_transforms

def test_split():
    """测试数据集划分"""
    
    # 加载配置
    config = load_config("configs/pretrain.yaml")
    
    # 创建增强
    transforms = get_pretrain_transforms(sample_rate=config.data.sample_rate)
    
    # 加载完整数据集
    print("加载数据集...")
    full_dataset = PCGPretrainDataset(
        data_dir=config.data.processed_dir,
        transform=transforms,
        num_substructures=config.data.num_substructures,
    )
    
    print(f"✓ 总样本数: {len(full_dataset)}")
    
    # 创建受试者级划分
    print("\n创建受试者级划分 (9:1)...")
    train_indices, val_indices, test_indices = create_subject_wise_split(
        full_dataset,
        train_ratio=0.9,
        val_ratio=0.1,
        test_ratio=0.0,
        seed=42
    )
    
    print(f"✓ 训练集样本: {len(train_indices)} ({len(train_indices)/len(full_dataset)*100:.1f}%)")
    print(f"✓ 验证集样本: {len(val_indices)} ({len(val_indices)/len(full_dataset)*100:.1f}%)")
    print(f"✓ 测试集样本: {len(test_indices)}")
    
    # 验证没有重叠
    print("\n检查数据泄露...")
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    
    assert len(train_set & val_set) == 0, "❌ 训练集和验证集有重叠!"
    assert len(train_set & test_set) == 0, "❌ 训练集和测试集有重叠!"
    assert len(val_set & test_set) == 0, "❌ 验证集和测试集有重叠!"
    print("✓ 无数据泄露: 训练/验证/测试集完全独立")
    
    # 验证受试者级划分
    print("\n检查受试者级划分...")
    train_subjects = set([full_dataset.get_subject_id(i) for i in train_indices])
    val_subjects = set([full_dataset.get_subject_id(i) for i in val_indices])
    
    assert len(train_subjects & val_subjects) == 0, "❌ 训练集和验证集有相同受试者!"
    print(f"✓ 训练集受试者数: {len(train_subjects)}")
    print(f"✓ 验证集受试者数: {len(val_subjects)}")
    print(f"✓ 受试者级划分正确: 无受试者跨集")
    
    # 测试数据增强
    print("\n测试数据增强...")
    sample = full_dataset[0]
    view1 = sample['view1']
    view2 = sample['view2']
    
    print(f"✓ View1 shape: {view1.shape}")
    print(f"✓ View2 shape: {view2.shape}")
    
    # 两个视图应该不同（因为随机增强）
    if (view1 != view2).any():
        print("✓ View1 和 View2 不同（随机增强生效）")
    else:
        print("⚠ View1 和 View2 相同（可能增强概率太低或数据相同）")
    
    print("\n" + "="*60)
    print("✓ 所有测试通过!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    try:
        test_split()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
