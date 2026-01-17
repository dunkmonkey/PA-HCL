#!/usr/bin/env python
"""
PA-HCL 数据预处理脚本

此脚本将原始心音录音处理为可用于自监督预训练的心动周期片段。

处理流程：
1. 从按受试者组织的目录中加载原始 WAV 文件
2. 应用带通滤波 (25-400 Hz)
3. 信号归一化
4. 使用能量包络检测心动周期边界
5. 提取单个周期
6. 应用质量过滤 (基于 SNR)
7. 将周期拆分为子结构
8. 将处理后的数据保存为 .npy 文件

用法：
    python scripts/preprocess.py --config configs/default.yaml
    python scripts/preprocess.py --raw_dir data/raw --output_dir data/processed
    python scripts/preprocess.py --config configs/default.yaml --num_workers 8

作者: PA-HCL 团队
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 将项目根目录添加到路径以便导入
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data.preprocessing import (
    load_audio,
    bandpass_filter,
    normalize_signal,
    compute_energy_envelope,
    detect_peaks_adaptive,
    extract_cycles,
    split_substructures,
    assess_cycle_quality,
)


# ============== 数据类 ==============

@dataclass
class ProcessingStats:
    """单个录音的统计信息。"""
    file_path: str
    subject_id: str
    recording_id: str
    original_duration_sec: float
    sample_rate: int
    num_peaks_detected: int
    num_cycles_extracted: int
    num_cycles_passed_quality: int
    num_cycles_failed_quality: int
    processing_time_sec: float
    error: Optional[str] = None


@dataclass
class GlobalStats:
    """所有录音的汇总统计信息。"""
    total_recordings: int
    successful_recordings: int
    failed_recordings: int
    total_cycles_extracted: int
    total_cycles_passed: int
    total_cycles_failed: int
    total_processing_time_sec: float
    sample_rate: int
    target_cycle_length: int
    num_substructures: int


# ============== 处理函数 ==============

def process_single_recording(
    file_path: Path,
    output_dir: Path,
    sample_rate: int = 5000,
    target_cycle_length: int = 4000,
    num_substructures: int = 4,
    min_cycle_duration: float = 0.4,
    max_cycle_duration: float = 1.5,
    min_snr_db: float = 10.0,
    save_substructures: bool = False
) -> ProcessingStats:
    """
    处理单个心音录音。
    
    参数:
        file_path: WAV 文件路径
        output_dir: 基础输出目录
        sample_rate: 目标采样率
        target_cycle_length: 每个周期的目标长度
        num_substructures: 子结构数量 (K)
        min_cycle_duration: 最小有效周期时长
        max_cycle_duration: 最大有效周期时长
        min_snr_db: 质量过滤的最小 SNR
        save_substructures: 是否单独保存子结构
        
    返回:
        此录音的 ProcessingStats
    """
    start_time = time.time()
    
    # 从路径中提取受试者和录音 ID
    # 预期结构: raw_dir/subject_id/recording.wav
    try:
        parts = file_path.parts
        subject_id = parts[-2] if len(parts) >= 2 else "unknown"
        recording_id = file_path.stem
    except Exception:
        subject_id = "unknown"
        recording_id = file_path.stem
    
    try:
        # 第 1 步: 加载音频
        audio, sr = load_audio(file_path, target_sr=sample_rate)
        original_duration = len(audio) / sr
        
        # 第 2 步: 带通滤波
        filtered = bandpass_filter(audio, sr, lowcut=25.0, highcut=400.0)
        
        # 第 3 步: 归一化
        normalized = normalize_signal(filtered, method="zscore")
        
        # 第 4 步: 计算能量包络
        envelope, _ = compute_energy_envelope(
            normalized, sr,
            frame_length_ms=20.0,
            hop_length_ms=10.0,
            method="shannon"
        )
        hop_length = int(10.0 * sr / 1000)
        
        # 第 5 步: 检测峰值
        peaks = detect_peaks_adaptive(envelope, sr, hop_length)
        num_peaks = len(peaks)
        
        # 第 6 步: 提取周期
        cycles_raw = extract_cycles(
            normalized, peaks, sr,
            min_duration_sec=min_cycle_duration,
            max_duration_sec=max_cycle_duration,
            target_length=target_cycle_length,
            padding_mode="zero"
        )
        
        # 第 7 步: 质量过滤并保存
        # 创建输出目录结构
        subject_output_dir = output_dir / subject_id / recording_id
        subject_output_dir.mkdir(parents=True, exist_ok=True)
        
        cycles_passed = 0
        cycles_failed = 0
        
        for idx, cycle in enumerate(cycles_raw):
            # 质量评估
            quality = assess_cycle_quality(cycle, sr, min_snr_db)
            
            if not quality["valid"]:
                cycles_failed += 1
                continue
            
            cycles_passed += 1
            
            # 保存周期
            cycle_path = subject_output_dir / f"cycle_{idx:04d}.npy"
            np.save(cycle_path, cycle.astype(np.float32))
            
            # 可选：保存子结构
            if save_substructures:
                subs = split_substructures(cycle, num_substructures)
                for sub_idx, sub in enumerate(subs):
                    sub_path = subject_output_dir / f"cycle_{idx:04d}_sub_{sub_idx}.npy"
                    np.save(sub_path, sub.astype(np.float32))
        
        processing_time = time.time() - start_time
        
        return ProcessingStats(
            file_path=str(file_path),
            subject_id=subject_id,
            recording_id=recording_id,
            original_duration_sec=original_duration,
            sample_rate=sr,
            num_peaks_detected=num_peaks,
            num_cycles_extracted=len(cycles_raw),
            num_cycles_passed_quality=cycles_passed,
            num_cycles_failed_quality=cycles_failed,
            processing_time_sec=processing_time,
            error=None
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return ProcessingStats(
            file_path=str(file_path),
            subject_id=subject_id,
            recording_id=recording_id,
            original_duration_sec=0.0,
            sample_rate=sample_rate,
            num_peaks_detected=0,
            num_cycles_extracted=0,
            num_cycles_passed_quality=0,
            num_cycles_failed_quality=0,
            processing_time_sec=processing_time,
            error=str(e)
        )


def collect_wav_files(raw_dir: Path) -> List[Path]:
    """
    从原始数据目录收集所有 WAV 文件。
    
    参数:
        raw_dir: 原始数据目录路径
        
    返回:
        WAV 文件路径列表
    """
    wav_files = list(raw_dir.rglob("*.wav"))
    wav_files.extend(raw_dir.rglob("*.WAV"))
    
    # 排序以保证可复现性
    wav_files = sorted(set(wav_files))
    
    return wav_files


def process_all_recordings(
    raw_dir: Path,
    output_dir: Path,
    sample_rate: int = 5000,
    target_cycle_length: int = 4000,
    num_substructures: int = 4,
    min_cycle_duration: float = 0.4,
    max_cycle_duration: float = 1.5,
    min_snr_db: float = 10.0,
    save_substructures: bool = False,
    num_workers: int = 1
) -> GlobalStats:
    """
    处理原始数据目录中的所有录音。
    
    参数:
        raw_dir: 原始数据目录路径
        output_dir: 输出目录路径
        sample_rate: 目标采样率
        target_cycle_length: 目标周期长度
        num_substructures: 子结构数量
        min_cycle_duration: 最小周期时长
        max_cycle_duration: 最大周期时长
        min_snr_db: 最小 SNR 阈值
        save_substructures: 是否保存子结构
        num_workers: 并行工作进程数
        
    返回:
        包含汇总统计信息的 GlobalStats
    """
    # 收集所有 WAV 文件
    wav_files = collect_wav_files(raw_dir)
    
    if len(wav_files) == 0:
        raise ValueError(f"No WAV files found in {raw_dir}")
    
    print(f"Found {len(wav_files)} recordings to process")
    print(f"Output directory: {output_dir}")
    print(f"Using {num_workers} workers")
    print("-" * 50)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理录音
    all_stats: List[ProcessingStats] = []
    start_time = time.time()
    
    if num_workers == 1:
        # 顺序处理 (便于调试)
        for i, wav_file in enumerate(wav_files):
            stats = process_single_recording(
                wav_file, output_dir,
                sample_rate=sample_rate,
                target_cycle_length=target_cycle_length,
                num_substructures=num_substructures,
                min_cycle_duration=min_cycle_duration,
                max_cycle_duration=max_cycle_duration,
                min_snr_db=min_snr_db,
                save_substructures=save_substructures
            )
            all_stats.append(stats)
            
            # 进度更新
            if (i + 1) % 10 == 0 or i == len(wav_files) - 1:
                print(f"Processed {i + 1}/{len(wav_files)} recordings")
    else:
        # 并行处理
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    process_single_recording,
                    wav_file, output_dir,
                    sample_rate, target_cycle_length, num_substructures,
                    min_cycle_duration, max_cycle_duration, min_snr_db,
                    save_substructures
                ): wav_file
                for wav_file in wav_files
            }
            
            completed = 0
            for future in as_completed(futures):
                stats = future.result()
                all_stats.append(stats)
                completed += 1
                
                if completed % 10 == 0 or completed == len(wav_files):
                    print(f"Processed {completed}/{len(wav_files)} recordings")
    
    total_time = time.time() - start_time
    
    # 计算全局统计信息
    successful = [s for s in all_stats if s.error is None]
    failed = [s for s in all_stats if s.error is not None]
    
    global_stats = GlobalStats(
        total_recordings=len(wav_files),
        successful_recordings=len(successful),
        failed_recordings=len(failed),
        total_cycles_extracted=sum(s.num_cycles_extracted for s in successful),
        total_cycles_passed=sum(s.num_cycles_passed_quality for s in successful),
        total_cycles_failed=sum(s.num_cycles_failed_quality for s in successful),
        total_processing_time_sec=total_time,
        sample_rate=sample_rate,
        target_cycle_length=target_cycle_length,
        num_substructures=num_substructures
    )
    
    # 保存详细统计信息
    stats_path = output_dir / "processing_stats.json"
    with open(stats_path, 'w') as f:
        json.dump({
            "global_stats": asdict(global_stats),
            "per_recording_stats": [asdict(s) for s in all_stats]
        }, f, indent=2)
    
    # 打印摘要
    print("-" * 50)
    print("Processing Complete!")
    print(f"Total recordings: {global_stats.total_recordings}")
    print(f"  Successful: {global_stats.successful_recordings}")
    print(f"  Failed: {global_stats.failed_recordings}")
    print(f"Total cycles extracted: {global_stats.total_cycles_extracted}")
    print(f"  Passed quality filter: {global_stats.total_cycles_passed}")
    print(f"  Failed quality filter: {global_stats.total_cycles_failed}")
    print(f"Total processing time: {global_stats.total_processing_time_sec:.2f} seconds")
    print(f"Statistics saved to: {stats_path}")
    
    # 打印失败的录音（如果有）
    if failed:
        print("\nFailed recordings:")
        for s in failed[:10]:  # 显示前 10 个
            print(f"  {s.file_path}: {s.error}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    return global_stats


# ============== 主入口点 ==============

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Preprocess heart sound recordings for PA-HCL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 配置
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置 YAML 文件的路径"
    )
    
    # 数据路径 (覆盖配置)
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=None,
        help="原始数据目录的路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录的路径"
    )
    
    # 处理参数 (覆盖配置)
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=None,
        help="目标采样率 (Hz)"
    )
    parser.add_argument(
        "--target_length",
        type=int,
        default=None,
        help="目标周期长度 (样本数)"
    )
    parser.add_argument(
        "--num_substructures",
        type=int,
        default=None,
        help="子结构数量 (K)"
    )
    parser.add_argument(
        "--min_snr",
        type=float,
        default=None,
        help="用于质量过滤的最小 SNR (dB)"
    )
    
    # 执行选项
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="并行工作进程数 (默认: CPU 核心数)"
    )
    parser.add_argument(
        "--save_substructures",
        action="store_true",
        help="将子结构保存为单独的文件"
    )
    
    return parser.parse_args()


def main():
    """主函数。"""
    args = parse_args()
    
    # 加载配置
    if args.config is not None:
        config = load_config(args.config)
    else:
        config = load_config()  # 加载默认配置
    
    # 使用命令行参数覆盖
    raw_dir = Path(args.raw_dir) if args.raw_dir else Path(config.data.raw_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config.data.processed_dir)
    sample_rate = args.sample_rate if args.sample_rate else config.data.sample_rate
    target_length = args.target_length if args.target_length else config.data.cycle.target_length
    num_substructures = args.num_substructures if args.num_substructures else config.data.num_substructures
    min_snr = args.min_snr if args.min_snr else config.data.filter.min_snr_db
    num_workers = args.num_workers if args.num_workers else mp.cpu_count()
    
    # 验证路径
    if not raw_dir.exists():
        print(f"Error: Raw data directory does not exist: {raw_dir}")
        print("Please create the directory and add your WAV files with the following structure:")
        print("  data/raw/")
        print("  ├── subject_0001/")
        print("  │   ├── recording_01.wav")
        print("  │   └── recording_02.wav")
        print("  ├── subject_0002/")
        print("  │   └── ...")
        sys.exit(1)
    
    # 打印配置
    print("=" * 50)
    print("PA-HCL Data Preprocessing")
    print("=" * 50)
    print(f"Raw data directory: {raw_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Target cycle length: {target_length} samples ({target_length/sample_rate:.3f} sec)")
    print(f"Number of substructures: {num_substructures}")
    print(f"Minimum SNR: {min_snr} dB")
    print(f"Number of workers: {num_workers}")
    print("=" * 50)
    
    # 运行处理
    process_all_recordings(
        raw_dir=raw_dir,
        output_dir=output_dir,
        sample_rate=sample_rate,
        target_cycle_length=target_length,
        num_substructures=num_substructures,
        min_cycle_duration=config.data.cycle.min_duration,
        max_cycle_duration=config.data.cycle.max_duration,
        min_snr_db=min_snr,
        save_substructures=args.save_substructures,
        num_workers=num_workers
    )


if __name__ == "__main__":
    main()
