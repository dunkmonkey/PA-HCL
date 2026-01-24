#!/usr/bin/env python3
"""
测试 AUROC/AUPRC 计算修复
"""

import numpy as np
from src.utils.metrics import compute_classification_metrics

def test_binary_classification():
    """测试二分类"""
    print("=" * 60)
    print("测试二分类 AUROC/AUPRC 计算")
    print("=" * 60)
    
    # 模拟预测结果
    np.random.seed(42)
    n_samples = 100
    
    # 真实标签 (0 或 1)
    y_true = np.random.randint(0, 2, n_samples)
    
    # 预测标签
    y_pred = np.random.randint(0, 2, n_samples)
    
    # 预测概率 (形状: [n_samples, 2])
    y_probs = np.random.rand(n_samples, 2)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)  # 归一化
    
    print(f"样本数: {n_samples}")
    print(f"标签分布: {np.bincount(y_true)}")
    print(f"概率形状: {y_probs.shape}")
    print(f"概率范围: [{y_probs.min():.4f}, {y_probs.max():.4f}]")
    print()
    
    # 计算指标
    metrics = compute_classification_metrics(y_true, y_pred, y_probs, num_classes=2)
    
    print("计算结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 验证
    assert metrics['auroc'] > 0, f"二分类 AUROC 不应为 0，得到: {metrics['auroc']}"
    assert metrics['auprc'] > 0, f"二分类 AUPRC 不应为 0，得到: {metrics['auprc']}"
    print("\n✅ 二分类测试通过!")

def test_multiclass_classification():
    """测试多分类"""
    print("\n" + "=" * 60)
    print("测试多分类 AUROC/AUPRC 计算")
    print("=" * 60)
    
    # 模拟预测结果
    np.random.seed(42)
    n_samples = 100
    n_classes = 3
    
    # 真实标签 (0, 1, 或 2)
    y_true = np.random.randint(0, n_classes, n_samples)
    
    # 预测标签
    y_pred = np.random.randint(0, n_classes, n_samples)
    
    # 预测概率 (形状: [n_samples, 3])
    y_probs = np.random.rand(n_samples, n_classes)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)  # 归一化
    
    print(f"样本数: {n_samples}")
    print(f"类别数: {n_classes}")
    print(f"标签分布: {np.bincount(y_true)}")
    print(f"概率形状: {y_probs.shape}")
    print(f"概率范围: [{y_probs.min():.4f}, {y_probs.max():.4f}]")
    print()
    
    # 计算指标
    metrics = compute_classification_metrics(y_true, y_pred, y_probs, num_classes=n_classes)
    
    print("计算结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # 验证
    assert metrics['auroc'] > 0, f"多分类 AUROC 不应为 0，得到: {metrics['auroc']}"
    assert metrics['auprc'] > 0, f"多分类 AUPRC 不应为 0，得到: {metrics['auprc']}"
    print("\n✅ 多分类测试通过!")

def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 60)
    print("测试边界情况")
    print("=" * 60)
    
    # 情况1: 只有一个类别
    print("\n1. 只有一个类别的情况:")
    y_true = np.array([0, 0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0, 0])
    y_probs = np.array([[1.0, 0.0]] * 5)
    
    metrics = compute_classification_metrics(y_true, y_pred, y_probs, num_classes=2)
    print(f"  AUROC: {metrics['auroc']:.4f} (应为 0.0)")
    print(f"  AUPRC: {metrics['auprc']:.4f} (应为 0.0)")
    assert metrics['auroc'] == 0.0, "单类别 AUROC 应为 0.0"
    assert metrics['auprc'] == 0.0, "单类别 AUPRC 应为 0.0"
    print("  ✅ 通过")
    
    # 情况2: 完美预测
    print("\n2. 完美预测的情况:")
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1])
    y_probs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    
    metrics = compute_classification_metrics(y_true, y_pred, y_probs, num_classes=2)
    print(f"  AUROC: {metrics['auroc']:.4f} (应为 1.0)")
    print(f"  AUPRC: {metrics['auprc']:.4f} (应为 1.0)")
    print(f"  Accuracy: {metrics['accuracy']:.4f} (应为 1.0)")
    assert metrics['auroc'] == 1.0, "完美预测 AUROC 应为 1.0"
    assert metrics['accuracy'] == 1.0, "完美预测准确率应为 1.0"
    print("  ✅ 通过")
    
    print("\n✅ 所有边界情况测试通过!")

if __name__ == "__main__":
    try:
        test_binary_classification()
        test_multiclass_classification()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("🎉 所有测试通过！AUROC/AUPRC 计算已修复。")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
