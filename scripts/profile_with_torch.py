#!/usr/bin/env python
"""
使用 PyTorch Profiler 进行深度性能分析

此脚本使用 PyTorch 内置的 Profiler 工具来分析训练过程的详细性能，
包括 CPU 时间、CUDA 时间、内存使用等。结果可以在 TensorBoard 中可视化。

用法:
    python scripts/profile_with_torch.py --task circor_murmur
    
    # 查看结果
    tensorboard --logdir=./profiler_logs --host 0.0.0.0 --port 6006

作者: PA-HCL 团队
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import yaml


def profile_training(task: str, config_path: str, num_batches: int = 20):
    """使用 PyTorch Profiler 分析训练"""
    
    print(f"\n{'='*60}")
    print(f"PyTorch Profiler 深度性能分析")
    print(f"{'='*60}")
    print(f"任务: {task}")
    print(f"配置: {config_path}")
    print(f"分析批次: {num_batches}")
    print(f"{'='*60}\n")
    
    try:
        from src.trainers.downstream_trainer import DownstreamTrainer
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确保在项目根目录运行此脚本")
        sys.exit(1)
    
    # 加载配置
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 加载配置失败: {e}")
        sys.exit(1)
    
    # 设置任务
    if 'task' not in config:
        config['task'] = {}
    config['task']['name'] = task
    
    # 设置少量 epoch
    if 'downstream' not in config:
        config['downstream'] = {}
    config['downstream']['num_epochs'] = 1
    
    # 创建训练器
    try:
        trainer = DownstreamTrainer(config)
        print("✅ 训练器创建成功\n")
    except Exception as e:
        print(f"❌ 训练器创建失败: {e}")
        sys.exit(1)
    
    # 设置 profiler 输出目录
    profiler_dir = './profiler_logs'
    Path(profiler_dir).mkdir(exist_ok=True)
    
    print(f"开始性能分析 (分析 {num_batches} 个批次)...\n")
    
    # 使用 Profiler
    try:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir)
        ) as prof:
            
            trainer.model.train()
            
            for batch_idx, (data, target) in enumerate(trainer.train_loader):
                if batch_idx >= num_batches:
                    break
                
                # 数据传输
                with record_function("data_transfer"):
                    data = data.to(trainer.device)
                    target = target.to(trainer.device)
                
                # 前向传播
                with record_function("forward"):
                    if trainer.use_amp:
                        with torch.cuda.amp.autocast():
                            output = trainer.model(data)
                            loss = trainer.criterion(output, target)
                    else:
                        output = trainer.model(data)
                        loss = trainer.criterion(output, target)
                
                # 反向传播
                with record_function("backward"):
                    trainer.optimizer.zero_grad()
                    if trainer.use_amp:
                        trainer.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                
                # 优化器步骤
                with record_function("optimizer_step"):
                    if trainer.use_amp:
                        trainer.scaler.step(trainer.optimizer)
                        trainer.scaler.update()
                    else:
                        trainer.optimizer.step()
                
                # 通知 profiler 一个步骤完成
                prof.step()
                
                if (batch_idx + 1) % 5 == 0:
                    print(f"  已分析 {batch_idx + 1}/{num_batches} 个批次...")
        
        print("\n✅ 性能分析完成!\n")
        
        # 输出统计表格
        print("="*60)
        print("CPU 时间统计 (Top 10)")
        print("="*60)
        print(prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=10
        ))
        
        print("\n" + "="*60)
        print("CUDA 时间统计 (Top 10)")
        print("="*60)
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=10
        ))
        
        print("\n" + "="*60)
        print("内存使用统计 (Top 10)")
        print("="*60)
        print(prof.key_averages().table(
            sort_by="self_cpu_memory_usage",
            row_limit=10
        ))
        
        # 保存详细统计到文件
        stats_file = Path(profiler_dir) / "profiler_stats.txt"
        with open(stats_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("CPU 时间统计 (Top 20)\n")
            f.write("="*60 + "\n")
            f.write(prof.key_averages().table(
                sort_by="cpu_time_total",
                row_limit=20
            ))
            f.write("\n\n")
            
            f.write("="*60 + "\n")
            f.write("CUDA 时间统计 (Top 20)\n")
            f.write("="*60 + "\n")
            f.write(prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=20
            ))
            f.write("\n\n")
            
            f.write("="*60 + "\n")
            f.write("内存使用统计 (Top 20)\n")
            f.write("="*60 + "\n")
            f.write(prof.key_averages().table(
                sort_by="self_cpu_memory_usage",
                row_limit=20
            ))
        
        print(f"\n📄 详细统计已保存到: {stats_file.absolute()}")
        print(f"\n📊 TensorBoard 日志目录: {Path(profiler_dir).absolute()}")
        print("\n" + "="*60)
        print("查看 TensorBoard 可视化:")
        print("="*60)
        print(f"  tensorboard --logdir={profiler_dir} --host 0.0.0.0 --port 6006")
        print("  然后在浏览器中打开: http://localhost:6006")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ 性能分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='使用 PyTorch Profiler 进行深度性能分析',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python scripts/profile_with_torch.py --task circor_murmur
  
  # 分析更多批次
  python scripts/profile_with_torch.py --task circor_murmur --batches 50
  
  # 使用自定义配置
  python scripts/profile_with_torch.py --task physionet2016 \\
      --config configs/finetune.yaml
  
  # 查看结果
  tensorboard --logdir=./profiler_logs --host 0.0.0.0 --port 6006
        """
    )
    
    parser.add_argument(
        '--task',
        type=str,
        required=True,
        help='任务名称 (circor_murmur, circor_outcome, physionet2016, pascal)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/finetune.yaml',
        help='配置文件路径 (default: configs/finetune.yaml)'
    )
    
    parser.add_argument(
        '--batches',
        type=int,
        default=20,
        help='分析的批次数量 (default: 20)'
    )
    
    args = parser.parse_args()
    
    # 检查配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    
    # 执行分析
    profile_training(args.task, str(config_path), args.batches)


if __name__ == '__main__':
    main()
