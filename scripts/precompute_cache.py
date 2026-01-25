#!/usr/bin/env python
"""
数据预处理缓存脚本。

此脚本预加载所有数据并验证完整性，可选择预计算增强视图。
在训练前运行此脚本可以显著加速首次 epoch。

用法:
    # 仅验证和预热缓存
    python scripts/precompute_cache.py --data-dir /path/to/data
    
    # 预计算视图并保存
    python scripts/precompute_cache.py --data-dir /path/to/data --precompute-views --output views_cache.pt

作者: PA-HCL Team
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="预处理数据缓存脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="预处理后的数据目录（包含 .npy 文件）"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（可选）"
    )
    
    parser.add_argument(
        "--precompute-views",
        action="store_true",
        help="是否预计算增强视图"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="预计算视图的输出路径（.pt 文件）"
    )
    
    parser.add_argument(
        "--num-views",
        type=int,
        default=2,
        help="每个样本预计算的视图数量"
    )
    
    parser.add_argument(
        "--target-length",
        type=int,
        default=4000,
        help="目标信号长度"
    )
    
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=5000,
        help="采样率"
    )
    
    parser.add_argument(
        "--num-substructures",
        type=int,
        default=4,
        help="子结构数量"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅验证数据完整性，不加载到内存"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="输出数据统计信息"
    )
    
    return parser.parse_args()


def collect_cycle_files(data_dir: Path) -> List[Path]:
    """收集所有周期文件。"""
    cycle_files = []
    for ext in ["*.npy", "*.pt"]:
        cycle_files.extend(data_dir.rglob(ext))
    return sorted(set(cycle_files))


def validate_file(file_path: Path, target_length: int) -> Tuple[bool, str]:
    """验证单个文件的完整性。"""
    try:
        if file_path.suffix == ".npy":
            data = np.load(file_path)
        elif file_path.suffix == ".pt":
            data = torch.load(file_path).numpy()
        else:
            return False, f"不支持的格式: {file_path.suffix}"
        
        if not isinstance(data, np.ndarray):
            return False, "不是 numpy 数组"
        
        if data.ndim != 1:
            return False, f"维度错误: 期望 1D, 实际 {data.ndim}D"
        
        if len(data) == 0:
            return False, "空数组"
        
        if np.isnan(data).any():
            return False, "包含 NaN 值"
        
        if np.isinf(data).any():
            return False, "包含 Inf 值"
        
        return True, f"长度: {len(data)}"
        
    except Exception as e:
        return False, str(e)


def compute_statistics(cycle_files: List[Path]) -> Dict:
    """计算数据统计信息。"""
    lengths = []
    amplitudes = []
    
    for file_path in tqdm(cycle_files, desc="Computing statistics"):
        try:
            if file_path.suffix == ".npy":
                data = np.load(file_path)
            else:
                data = torch.load(file_path).numpy()
            
            lengths.append(len(data))
            amplitudes.append(np.abs(data).max())
        except:
            continue
    
    lengths = np.array(lengths)
    amplitudes = np.array(amplitudes)
    
    return {
        "num_samples": len(lengths),
        "length": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "std": float(lengths.std()),
        },
        "amplitude": {
            "min": float(amplitudes.min()),
            "max": float(amplitudes.max()),
            "mean": float(amplitudes.mean()),
        },
        "total_duration_hours": float(lengths.sum() / 5000 / 3600),  # 假设 5000 Hz
    }


def precompute_views(
    cycle_files: List[Path],
    target_length: int,
    sample_rate: int,
    num_substructures: int,
    num_views: int = 2,
    output_path: Optional[Path] = None
) -> Dict:
    """预计算增强视图。"""
    from src.data.transforms import get_pretrain_transforms
    from src.data.preprocessing import split_substructures
    
    transform = get_pretrain_transforms(sample_rate=sample_rate)
    
    views_cache = {}
    
    for idx, file_path in enumerate(tqdm(cycle_files, desc="Precomputing views")):
        try:
            if file_path.suffix == ".npy":
                cycle = np.load(file_path).astype(np.float32)
            else:
                cycle = torch.load(file_path).numpy().astype(np.float32)
            
            # 标准化长度
            if len(cycle) < target_length:
                pad_total = target_length - len(cycle)
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                cycle = np.pad(cycle, (pad_left, pad_right), mode='constant')
            elif len(cycle) > target_length:
                start = (len(cycle) - target_length) // 2
                cycle = cycle[start:start + target_length]
            
            # 预计算多个视图
            sample_views = []
            for _ in range(num_views):
                view = transform(cycle.copy(), sample_rate)
                subs = split_substructures(view, num_substructures)
                sample_views.append({
                    'view': view,
                    'subs': subs
                })
            
            views_cache[idx] = sample_views
            
        except Exception as e:
            print(f"Warning: Failed to process {file_path}: {e}")
            continue
    
    if output_path is not None:
        print(f"Saving views cache to {output_path}...")
        torch.save({
            'views_cache': views_cache,
            'file_paths': [str(f) for f in cycle_files],
            'config': {
                'target_length': target_length,
                'sample_rate': sample_rate,
                'num_substructures': num_substructures,
                'num_views': num_views,
            }
        }, output_path)
        print(f"Saved {len(views_cache)} samples to {output_path}")
    
    return views_cache


def main():
    args = parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: 数据目录不存在: {data_dir}")
        sys.exit(1)
    
    print(f"数据目录: {data_dir}")
    print("="*60)
    
    # 收集文件
    print("收集数据文件...")
    cycle_files = collect_cycle_files(data_dir)
    print(f"找到 {len(cycle_files)} 个周期文件")
    
    if len(cycle_files) == 0:
        print("Error: 未找到任何数据文件")
        sys.exit(1)
    
    # 验证文件
    if args.validate_only or args.stats:
        print("\n验证数据完整性...")
        valid_count = 0
        invalid_files = []
        
        for file_path in tqdm(cycle_files, desc="Validating"):
            is_valid, msg = validate_file(file_path, args.target_length)
            if is_valid:
                valid_count += 1
            else:
                invalid_files.append((file_path, msg))
        
        print(f"\n验证结果:")
        print(f"  有效文件: {valid_count}/{len(cycle_files)}")
        print(f"  无效文件: {len(invalid_files)}")
        
        if invalid_files and len(invalid_files) <= 10:
            print("\n无效文件列表:")
            for path, msg in invalid_files:
                print(f"  {path}: {msg}")
    
    # 统计信息
    if args.stats:
        print("\n计算统计信息...")
        stats = compute_statistics(cycle_files)
        print("\n数据统计:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # 预计算视图
    if args.precompute_views:
        output_path = Path(args.output) if args.output else data_dir / "views_cache.pt"
        
        print(f"\n预计算增强视图 (num_views={args.num_views})...")
        start_time = time.time()
        
        views_cache = precompute_views(
            cycle_files,
            target_length=args.target_length,
            sample_rate=args.sample_rate,
            num_substructures=args.num_substructures,
            num_views=args.num_views,
            output_path=output_path
        )
        
        elapsed = time.time() - start_time
        print(f"\n预计算完成，耗时: {elapsed:.1f}s")
        print(f"缓存文件: {output_path}")
        print(f"缓存大小: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # 预热缓存（加载所有文件到系统缓存）
    if not args.validate_only and not args.precompute_views:
        print("\n预热数据缓存...")
        for file_path in tqdm(cycle_files, desc="Warming cache"):
            try:
                if file_path.suffix == ".npy":
                    _ = np.load(file_path)
                else:
                    _ = torch.load(file_path)
            except:
                continue
        print("缓存预热完成")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
