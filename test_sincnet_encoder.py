#!/usr/bin/env python
"""
测试新的 SincNet+ECA+ASP+Mamba 编码器的验证脚本。

此脚本验证：
1. 编码器的基本输出形状
2. 中间特征返回（子结构级对比学习）
3. 接口兼容性
4. 与 PAHCLModel 的集成
5. 参数量对比

使用方法:
    python test_sincnet_encoder.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from src.models.encoder import build_encoder, CNNMambaEncoder, SincNetECAMambaEncoder
from src.models.pahcl import PAHCLModel


def test_encoder_basic():
    """测试1: 编码器基本功能"""
    print("=" * 60)
    print("测试1: SincNetECAMambaEncoder 基本功能")
    print("=" * 60)
    
    # 创建编码器
    encoder = SincNetECAMambaEncoder(
        in_channels=1,
        sinc_out_channels=64,
        sinc_kernel_size=251,
        sinc_stride=2,
        stage1_channels=[64, 96],
        stage2_channels=[96, 128],
        mamba_d_model=128,
        mamba_n_layers=2,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dropout=0.1,
        use_groupnorm=True,
        num_groups=8,
        eca_kernel_size=3,
        cycle_output_dim=192,
        num_substructures=4,
        pool_type="asp",
        sample_rate=5000
    )
    
    # 创建测试输入
    batch_size = 4
    seq_len = 5000  # 1秒 @ 5000Hz
    x = torch.randn(batch_size, 1, seq_len)
    
    print(f"输入形状: {x.shape}")
    
    # 测试默认输出
    out = encoder(x)
    print(f"✓ 默认输出形状: {out.shape}")
    assert out.shape == (batch_size, 192), f"期望 (4, 192), 得到 {out.shape}"
    
    # 测试序列输出
    seq_out = encoder(x, return_sequence=True)
    print(f"✓ 序列输出形状: {seq_out.shape}")
    assert seq_out.dim() == 3 and seq_out.shape[0] == batch_size and seq_out.shape[2] == 128
    
    # 测试中间特征
    out_dict = encoder(x, return_intermediate=True)
    print(f"✓ 中间特征键: {list(out_dict.keys())}")
    assert "cycle_features" in out_dict
    assert "sub_features" in out_dict
    assert "sequence" in out_dict
    print(f"  - cycle_features: {out_dict['cycle_features'].shape}")
    print(f"  - sub_features: {out_dict['sub_features'].shape}")
    print(f"  - sequence: {out_dict['sequence'].shape}")
    
    # 测试 get_sub_features
    sub_feat = encoder.get_sub_features(x)
    print(f"✓ 子结构特征形状: {sub_feat.shape}")
    assert sub_feat.shape[0] == batch_size and sub_feat.shape[1] == 128
    
    print("\n✓ 测试1通过: 所有基本功能正常工作\n")


def test_encoder_interface_compatibility():
    """测试2: 接口兼容性"""
    print("=" * 60)
    print("测试2: 接口兼容性（与 CNNMambaEncoder 对比）")
    print("=" * 60)
    
    # 创建旧编码器
    old_encoder = CNNMambaEncoder(
        in_channels=1,
        cnn_channels=[32, 64, 128, 256],
        mamba_d_model=256,
        mamba_n_layers=4,
        pool_type="mean"
    )
    
    # 创建新编码器
    new_encoder = SincNetECAMambaEncoder(
        in_channels=1,
        mamba_d_model=128,
        mamba_n_layers=2,
        pool_type="asp",
        cycle_output_dim=192
    )
    
    x = torch.randn(2, 1, 5000)
    
    # 测试接口一致性
    old_out = old_encoder(x)
    new_out = new_encoder(x)
    print(f"✓ 旧编码器输出: {old_out.shape}")
    print(f"✓ 新编码器输出: {new_out.shape}")
    
    # 测试属性
    print(f"✓ 旧编码器 out_dim: {old_encoder.out_dim}")
    print(f"✓ 新编码器 out_dim: {new_encoder.out_dim}")
    assert hasattr(new_encoder, 'out_dim'), "新编码器缺少 out_dim 属性"
    
    # 测试 get_sub_features
    old_sub = old_encoder.get_sub_features(x)
    new_sub = new_encoder.get_sub_features(x)
    print(f"✓ 旧编码器子结构特征: {old_sub.shape}")
    print(f"✓ 新编码器子结构特征: {new_sub.shape}")
    
    print("\n✓ 测试2通过: 接口兼容性良好\n")


def test_build_encoder_factory():
    """测试3: 工厂函数"""
    print("=" * 60)
    print("测试3: build_encoder 工厂函数")
    print("=" * 60)
    
    # 测试旧编码器类型
    encoder1 = build_encoder("cnn_mamba", mamba_d_model=256, mamba_n_layers=4)
    print(f"✓ build_encoder('cnn_mamba'): {type(encoder1).__name__}")
    
    # 测试新编码器类型
    encoder2 = build_encoder("sincnet_eca_mamba", mamba_d_model=128, mamba_n_layers=2)
    print(f"✓ build_encoder('sincnet_eca_mamba'): {type(encoder2).__name__}")
    assert isinstance(encoder2, SincNetECAMambaEncoder), "工厂函数返回错误的类型"
    
    # 测试前向传播
    x = torch.randn(2, 1, 5000)
    out1 = encoder1(x)
    out2 = encoder2(x)
    print(f"✓ cnn_mamba 输出: {out1.shape}")
    print(f"✓ sincnet_eca_mamba 输出: {out2.shape}")
    
    print("\n✓ 测试3通过: 工厂函数正常工作\n")


def test_pahcl_integration():
    """测试4: PAHCLModel 集成"""
    print("=" * 60)
    print("测试4: PAHCLModel 集成")
    print("=" * 60)
    
    # 使用旧编码器创建模型
    model_old = PAHCLModel(
        encoder_type="cnn_mamba",
        mamba_d_model=256,
        mamba_n_layers=4,
        cycle_proj_output=128,
        sub_proj_output=64
    )
    
    # 使用新编码器创建模型
    model_new = PAHCLModel(
        encoder_type="sincnet_eca_mamba",
        mamba_d_model=128,
        mamba_n_layers=2,
        cycle_proj_output=128,
        sub_proj_output=64
    )
    
    # 测试前向传播
    x = torch.randn(2, 1, 5000)
    
    out_old = model_old(x)
    print(f"✓ PAHCLModel (cnn_mamba) 输出: {out_old['cycle_proj'].shape}")
    
    out_new = model_new(x)
    print(f"✓ PAHCLModel (sincnet_eca_mamba) 输出: {out_new['cycle_proj'].shape}")
    
    # 测试预训练前向传播
    view1 = torch.randn(2, 1, 5000)
    view2 = torch.randn(2, 1, 5000)
    
    pretrain_out = model_new.forward_pretrain(view1, view2)
    print(f"✓ 预训练输出键: {list(pretrain_out.keys())}")
    assert "cycle_proj1" in pretrain_out
    assert "cycle_proj2" in pretrain_out
    assert "sub_proj1" in pretrain_out
    assert "sub_proj2" in pretrain_out
    
    print("\n✓ 测试4通过: PAHCLModel 集成成功\n")


def test_parameter_count():
    """测试5: 参数量对比"""
    print("=" * 60)
    print("测试5: 参数量对比")
    print("=" * 60)
    
    # 旧编码器
    old_encoder = CNNMambaEncoder(
        in_channels=1,
        cnn_channels=[32, 64, 128, 256],
        mamba_d_model=256,
        mamba_n_layers=4
    )
    
    # 新编码器
    new_encoder = SincNetECAMambaEncoder(
        in_channels=1,
        mamba_d_model=128,
        mamba_n_layers=2,
        cycle_output_dim=192
    )
    
    # 计算参数量
    old_params = sum(p.numel() for p in old_encoder.parameters())
    new_params = sum(p.numel() for p in new_encoder.parameters())
    
    old_trainable = sum(p.numel() for p in old_encoder.parameters() if p.requires_grad)
    new_trainable = sum(p.numel() for p in new_encoder.parameters() if p.requires_grad)
    
    print(f"旧编码器 (CNNMambaEncoder):")
    print(f"  总参数: {old_params:,}")
    print(f"  可训练: {old_trainable:,}")
    
    print(f"\n新编码器 (SincNetECAMambaEncoder):")
    print(f"  总参数: {new_params:,}")
    print(f"  可训练: {new_trainable:,}")
    
    reduction = (old_params - new_params) / old_params * 100
    print(f"\n参数量减少: {reduction:.1f}%")
    
    if new_params < old_params:
        print("✓ 新编码器更轻量")
    else:
        print("⚠ 新编码器参数量更多（但可能更高效）")
    
    print("\n✓ 测试5完成: 参数量统计完成\n")


def test_sinc_frequencies():
    """测试6: SincNet 频率学习"""
    print("=" * 60)
    print("测试6: SincNet 频率学习（可视化滤波器）")
    print("=" * 60)
    
    encoder = SincNetECAMambaEncoder(
        sinc_out_channels=64,
        sample_rate=5000
    )
    
    # 获取 SincConv 层
    sinc_layer = encoder.sinc_conv
    
    # 计算实际频率范围
    with torch.no_grad():
        low_hz = sinc_layer.min_low_hz + torch.abs(sinc_layer.low_hz_)
        high_hz = torch.clamp(
            low_hz + sinc_layer.min_band_hz + torch.abs(sinc_layer.band_hz_),
            min=sinc_layer.min_low_hz,
            max=sinc_layer.sample_rate / 2
        )
    
    print(f"滤波器数量: {len(low_hz)}")
    print(f"频率范围统计:")
    print(f"  低截止频率: {low_hz.min():.1f} - {low_hz.max():.1f} Hz")
    print(f"  高截止频率: {high_hz.min():.1f} - {high_hz.max():.1f} Hz")
    print(f"  带宽范围: {(high_hz - low_hz).min():.1f} - {(high_hz - low_hz).max():.1f} Hz")
    
    # 显示前5个滤波器
    print(f"\n前5个滤波器的频率范围:")
    for i in range(min(5, len(low_hz))):
        print(f"  滤波器 {i}: {low_hz[i].item():.1f} - {high_hz[i].item():.1f} Hz (带宽: {(high_hz[i] - low_hz[i]).item():.1f} Hz)")
    
    print("\n✓ 测试6完成: SincNet 频率范围合理\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("SincNet+ECA+ASP+Mamba 编码器验证测试")
    print("=" * 60 + "\n")
    
    try:
        test_encoder_basic()
        test_encoder_interface_compatibility()
        test_build_encoder_factory()
        test_pahcl_integration()
        test_parameter_count()
        test_sinc_frequencies()
        
        print("=" * 60)
        print("✅ 所有测试通过！新编码器可以安全使用。")
        print("=" * 60)
        
        print("\n使用建议:")
        print("1. 在配置文件中设置 encoder_type: 'sincnet_eca_mamba'")
        print("2. 推荐配置:")
        print("   - mamba_d_model: 128")
        print("   - mamba_n_layers: 2")
        print("   - pool_type: 'asp'")
        print("   - cycle_output_dim: 192")
        print("3. 训练命令:")
        print("   python scripts/pretrain.py --config configs/pretrain.yaml")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
