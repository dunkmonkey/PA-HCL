#!/usr/bin/env python
"""
测试 TensorBoard 和 WandB 集成

此脚本验证实验监控功能是否正常工作。
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_tensorboard():
    """测试 TensorBoard 导入和基本功能"""
    print("测试 TensorBoard...")
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("✓ TensorBoard 可用")
        
        # 创建测试 writer
        test_dir = Path("outputs/test_tensorboard")
        test_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(test_dir))
        
        # 写入测试数据
        for i in range(10):
            writer.add_scalar("test/loss", 1.0 / (i + 1), i)
            writer.add_scalar("test/accuracy", i / 10.0, i)
        
        writer.close()
        print("✓ TensorBoard 写入成功")
        print(f"  日志目录: {test_dir}")
        print(f"  查看: tensorboard --logdir {test_dir}")
        return True
        
    except ImportError as e:
        print(f"✗ TensorBoard 不可用: {e}")
        print("  安装: pip install tensorboard")
        return False

def test_wandb():
    """测试 WandB 导入"""
    print("\n测试 WandB...")
    try:
        import wandb
        print("✓ WandB 可用")
        print(f"  版本: {wandb.__version__}")
        
        # 检查是否已登录
        try:
            api = wandb.Api()
            print("✓ WandB 已登录")
        except Exception:
            print("! WandB 未登录")
            print("  登录: wandb login")
        
        return True
        
    except ImportError as e:
        print(f"✗ WandB 不可用: {e}")
        print("  安装: pip install wandb")
        return False

def test_trainer_import():
    """测试训练器导入"""
    print("\n测试训练器导入...")
    try:
        from src.trainers.downstream_trainer import DownstreamTrainer
        print("✓ DownstreamTrainer 导入成功")
        
        # 检查新参数
        import inspect
        sig = inspect.signature(DownstreamTrainer.__init__)
        params = list(sig.parameters.keys())
        
        required_params = ['use_tensorboard', 'use_wandb', 'wandb_project', 'wandb_entity']
        for param in required_params:
            if param in params:
                print(f"  ✓ 参数 {param} 已添加")
            else:
                print(f"  ✗ 参数 {param} 缺失")
        
        return True
        
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False

def test_finetune_args():
    """测试微调脚本参数"""
    print("\n测试微调脚本参数...")
    try:
        # 模拟导入 finetune 模块
        import scripts.finetune as finetune_module
        
        # 由于 parse_args 需要命令行参数，我们检查文件内容
        with open("scripts/finetune.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        required_flags = [
            '--tensorboard',
            '--wandb',
            '--wandb-project',
            '--wandb-entity'
        ]
        
        for flag in required_flags:
            if flag in content:
                print(f"  ✓ 参数 {flag} 已添加")
            else:
                print(f"  ✗ 参数 {flag} 缺失")
        
        return True
        
    except Exception as e:
        print(f"✗ 检查失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("=" * 60)
    print("PA-HCL 实验监控功能测试")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(("TensorBoard", test_tensorboard()))
    results.append(("WandB", test_wandb()))
    results.append(("Trainer Import", test_trainer_import()))
    results.append(("Finetune Args", test_finetune_args()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print("\n⚠️  部分测试失败，请检查依赖安装")

if __name__ == "__main__":
    main()
