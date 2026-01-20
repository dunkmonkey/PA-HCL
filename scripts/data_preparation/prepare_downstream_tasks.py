#!/usr/bin/env python3
"""
PA-HCL 下游任务数据准备脚本

将预处理后的心音数据组织为各下游任务所需的格式，
支持多个数据集和多种分类任务。

支持的数据集和任务:
1. CirCor DigiScope (PhysioNet/CinC 2022):
   - 杂音检测三分类 (Present, Absent, Unknown)
   - 临床结果二分类 (Normal, Abnormal)
   
2. PhysioNet 2016 Challenge:
   - 心音二分类 (Normal, Abnormal)
   
3. PASCAL Heart Sound Challenge:
   - 心音三分类 (Normal, Murmur, Extrasystole)

数据组织原则:
- 受试者级物理隔离 (Subject-wise split)
- 预训练与微调数据分离
- 每个任务生成独立的目录结构和标签CSV

用法:
    # 准备所有数据集的所有任务
    python prepare_downstream_tasks.py --all --processed-dir data/processed --output-dir data/downstream

    # 准备特定数据集
    python prepare_downstream_tasks.py --dataset circor --task murmur
    python prepare_downstream_tasks.py --dataset circor --task outcome
    python prepare_downstream_tasks.py --dataset physionet2016
    python prepare_downstream_tasks.py --dataset pascal

作者: PA-HCL 团队
"""

import argparse
import csv
import json
import logging
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============== 数据类定义 ==============

@dataclass
class TaskConfig:
    """任务配置数据类"""
    task_name: str
    dataset: str
    task_type: str  # 'classification'
    num_classes: int
    label_map: Dict[str, int]
    class_weights: List[float] = field(default_factory=list)
    split_subjects: Dict[str, List[str]] = field(default_factory=dict)
    split_sizes: Dict[str, int] = field(default_factory=dict)
    primary_metric: str = "f1_macro"


@dataclass 
class SampleRecord:
    """样本记录数据类"""
    subject_id: str
    file_path: str
    label: int
    label_name: str
    location: str = ""
    split: str = ""


# ============== 工具函数 ==============

def setup_logging(verbose: bool = False):
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)


def compute_class_weights(
    labels: List[int],
    num_classes: int,
    method: str = "balanced",
    max_weight: float = 10.0
) -> List[float]:
    """
    计算类别权重用于处理不平衡数据
    
    Args:
        labels: 标签列表
        num_classes: 类别数量
        method: 计算方法 ("balanced" 或 "sqrt")
        max_weight: 权重上限（防止极端不平衡导致的不稳定）
    
    Returns:
        类别权重列表
    """
    counter = Counter(labels)
    total = len(labels)
    
    weights = []
    for i in range(num_classes):
        count = counter.get(i, 1)  # 避免除零
        if method == "balanced":
            # sklearn 风格: n_samples / (n_classes * n_samples_per_class)
            w = total / (num_classes * count)
        elif method == "sqrt":
            # 平方根平滑
            w = np.sqrt(total / count)
        else:
            w = 1.0
        
        # 应用权重上限
        w = min(w, max_weight)
        weights.append(round(w, 4))
    
    return weights


def split_subjects(
    subjects: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    按比例划分受试者
    
    Args:
        subjects: 受试者ID列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    
    Returns:
        {"train": [...], "val": [...], "test": [...]}
    """
    random.seed(seed)
    subjects = list(subjects)
    random.shuffle(subjects)
    
    n = len(subjects)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    return {
        "train": subjects[:n_train],
        "val": subjects[n_train:n_train + n_val],
        "test": subjects[n_train + n_val:]
    }


def link_or_copy_file(src: Path, dst: Path, copy: bool = False):
    """创建硬链接或复制文件"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)  # 硬链接节省空间
        except OSError:
            shutil.copy2(src, dst)  # 跨文件系统时回退到复制


def save_task_csv(
    records: List[SampleRecord],
    output_path: Path,
    class_weights: Optional[List[float]] = None
):
    """
    保存任务CSV文件
    
    CSV格式:
    subject_id,file_path,label,label_name,location,split
    """
    with open(output_path, 'w', newline='') as f:
        # 写入类别权重注释（如果提供）
        if class_weights:
            f.write(f"# class_weights: {class_weights}\n")
        
        writer = csv.DictWriter(
            f, 
            fieldnames=['subject_id', 'file_path', 'label', 'label_name', 'location', 'split']
        )
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def save_task_config(config: TaskConfig, output_path: Path):
    """保存任务配置JSON"""
    with open(output_path, 'w') as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)


# ============== CirCor 数据集处理 ==============

def prepare_circor_murmur(
    processed_dir: Path,
    raw_metadata_path: Path,
    output_dir: Path,
    seed: int = 42,
    copy_files: bool = False
) -> TaskConfig:
    """
    准备 CirCor 杂音检测三分类任务
    
    标签映射:
    - Present -> 0
    - Absent -> 1  
    - Unknown -> 2
    """
    logger = logging.getLogger(__name__)
    task_name = "circor_murmur"
    
    logger.info(f"准备任务: {task_name}")
    logger.info(f"处理后数据目录: {processed_dir}")
    logger.info(f"原始元数据: {raw_metadata_path}")
    
    # 标签映射
    label_map = {"Present": 0, "Absent": 1, "Unknown": 2}
    
    # 读取元数据
    df = pd.read_csv(raw_metadata_path)
    logger.info(f"读取 {len(df)} 条患者记录")
    
    # 提取 Patient ID 和 Murmur 标签
    # CirCor 的 Patient ID 列名可能是 'Patient ID' 或其他
    id_col = 'Patient ID' if 'Patient ID' in df.columns else df.columns[0]
    murmur_col = 'Murmur' if 'Murmur' in df.columns else None
    
    if murmur_col is None:
        raise ValueError("找不到 Murmur 列，请检查元数据文件格式")
    
    # 构建 subject -> label 映射
    subject_labels = {}
    for _, row in df.iterrows():
        subject_id = str(row[id_col])
        murmur = row[murmur_col]
        if pd.notna(murmur) and murmur in label_map:
            subject_labels[subject_id] = murmur
    
    logger.info(f"有效受试者数量: {len(subject_labels)}")
    
    # 统计标签分布
    label_counts = Counter(subject_labels.values())
    logger.info(f"标签分布: {dict(label_counts)}")
    
    # 受试者级划分
    subjects = list(subject_labels.keys())
    splits = split_subjects(subjects, seed=seed)
    
    logger.info(f"划分结果: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    
    # 收集所有样本记录
    all_records: Dict[str, List[SampleRecord]] = {"train": [], "val": [], "test": []}
    
    # 遍历处理后的数据目录
    processed_subjects = list(processed_dir.glob("subject_*"))
    
    for subject_path in tqdm(processed_subjects, desc="处理受试者"):
        subject_id = subject_path.name.replace("subject_", "")
        
        if subject_id not in subject_labels:
            continue
        
        label_name = subject_labels[subject_id]
        label = label_map[label_name]
        
        # 确定该受试者属于哪个 split
        split_name = None
        for s, ids in splits.items():
            if subject_id in ids:
                split_name = s
                break
        
        if split_name is None:
            continue
        
        # 遍历该受试者的所有录音
        for rec_dir in subject_path.iterdir():
            if not rec_dir.is_dir():
                continue
            
            location = rec_dir.name.replace("rec_", "")
            
            # 遍历所有周期文件
            for cycle_file in rec_dir.glob("cycle_*.npy"):
                # 目标路径
                rel_path = f"{subject_path.name}/{rec_dir.name}/{cycle_file.name}"
                dst_dir = output_dir / task_name / split_name / subject_path.name / rec_dir.name
                dst_path = dst_dir / cycle_file.name
                
                # 复制/链接文件
                link_or_copy_file(cycle_file, dst_path, copy=copy_files)
                
                # 记录样本
                record = SampleRecord(
                    subject_id=subject_id,
                    file_path=str(dst_path.relative_to(output_dir / task_name)),
                    label=label,
                    label_name=label_name,
                    location=location,
                    split=split_name
                )
                all_records[split_name].append(record)
    
    # 计算类别权重
    all_labels = [r.label for records in all_records.values() for r in records]
    class_weights = compute_class_weights(all_labels, num_classes=3)
    
    logger.info(f"类别权重: {class_weights}")
    
    # 保存 CSV 文件
    task_output = output_dir / task_name
    for split_name, records in all_records.items():
        csv_path = task_output / f"{split_name}.csv"
        save_task_csv(records, csv_path, class_weights if split_name == "train" else None)
        logger.info(f"保存 {csv_path.name}: {len(records)} 条记录")
    
    # 创建任务配置
    config = TaskConfig(
        task_name=task_name,
        dataset="circor",
        task_type="classification",
        num_classes=3,
        label_map=label_map,
        class_weights=class_weights,
        split_subjects=splits,
        split_sizes={s: len(r) for s, r in all_records.items()},
        primary_metric="f1_macro"
    )
    
    # 保存配置
    config_path = task_output / "task_config.json"
    save_task_config(config, config_path)
    logger.info(f"保存任务配置: {config_path}")
    
    return config


def prepare_circor_outcome(
    processed_dir: Path,
    raw_metadata_path: Path,
    output_dir: Path,
    seed: int = 42,
    copy_files: bool = False
) -> TaskConfig:
    """
    准备 CirCor 临床结果二分类任务
    
    标签映射:
    - Normal -> 0
    - Abnormal -> 1
    """
    logger = logging.getLogger(__name__)
    task_name = "circor_outcome"
    
    logger.info(f"准备任务: {task_name}")
    
    # 标签映射
    label_map = {"Normal": 0, "Abnormal": 1}
    
    # 读取元数据
    df = pd.read_csv(raw_metadata_path)
    
    id_col = 'Patient ID' if 'Patient ID' in df.columns else df.columns[0]
    outcome_col = 'Outcome' if 'Outcome' in df.columns else None
    
    if outcome_col is None:
        raise ValueError("找不到 Outcome 列，请检查元数据文件格式")
    
    # 构建 subject -> label 映射
    subject_labels = {}
    for _, row in df.iterrows():
        subject_id = str(row[id_col])
        outcome = row[outcome_col]
        if pd.notna(outcome) and outcome in label_map:
            subject_labels[subject_id] = outcome
    
    logger.info(f"有效受试者数量: {len(subject_labels)}")
    
    # 统计标签分布
    label_counts = Counter(subject_labels.values())
    logger.info(f"标签分布: {dict(label_counts)}")
    
    # 受试者级划分
    subjects = list(subject_labels.keys())
    splits = split_subjects(subjects, seed=seed)
    
    # 收集样本记录
    all_records: Dict[str, List[SampleRecord]] = {"train": [], "val": [], "test": []}
    
    processed_subjects = list(processed_dir.glob("subject_*"))
    
    for subject_path in tqdm(processed_subjects, desc="处理受试者"):
        subject_id = subject_path.name.replace("subject_", "")
        
        if subject_id not in subject_labels:
            continue
        
        label_name = subject_labels[subject_id]
        label = label_map[label_name]
        
        split_name = None
        for s, ids in splits.items():
            if subject_id in ids:
                split_name = s
                break
        
        if split_name is None:
            continue
        
        for rec_dir in subject_path.iterdir():
            if not rec_dir.is_dir():
                continue
            
            location = rec_dir.name.replace("rec_", "")
            
            for cycle_file in rec_dir.glob("cycle_*.npy"):
                dst_dir = output_dir / task_name / split_name / subject_path.name / rec_dir.name
                dst_path = dst_dir / cycle_file.name
                
                link_or_copy_file(cycle_file, dst_path, copy=copy_files)
                
                record = SampleRecord(
                    subject_id=subject_id,
                    file_path=str(dst_path.relative_to(output_dir / task_name)),
                    label=label,
                    label_name=label_name,
                    location=location,
                    split=split_name
                )
                all_records[split_name].append(record)
    
    # 计算类别权重
    all_labels = [r.label for records in all_records.values() for r in records]
    class_weights = compute_class_weights(all_labels, num_classes=2)
    
    logger.info(f"类别权重: {class_weights}")
    
    # 保存 CSV
    task_output = output_dir / task_name
    for split_name, records in all_records.items():
        csv_path = task_output / f"{split_name}.csv"
        save_task_csv(records, csv_path, class_weights if split_name == "train" else None)
        logger.info(f"保存 {csv_path.name}: {len(records)} 条记录")
    
    # 创建任务配置
    config = TaskConfig(
        task_name=task_name,
        dataset="circor",
        task_type="classification",
        num_classes=2,
        label_map=label_map,
        class_weights=class_weights,
        split_subjects=splits,
        split_sizes={s: len(r) for s, r in all_records.items()},
        primary_metric="auroc"
    )
    
    config_path = task_output / "task_config.json"
    save_task_config(config, config_path)
    
    return config


# ============== PhysioNet 2016 数据集处理 ==============

def prepare_physionet2016(
    processed_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    seed: int = 42,
    copy_files: bool = False
) -> TaskConfig:
    """
    准备 PhysioNet 2016 心音二分类任务
    
    标签映射:
    - Normal (-1 in original) -> 0
    - Abnormal (1 in original) -> 1
    """
    logger = logging.getLogger(__name__)
    task_name = "physionet2016"
    
    logger.info(f"准备任务: {task_name}")
    
    # 标签映射
    label_map = {"normal": 0, "abnormal": 1}
    original_label_map = {-1: "normal", 1: "abnormal"}
    
    # 查找 REFERENCE.csv 文件
    reference_files = list(raw_dir.rglob("REFERENCE*.csv"))
    if not reference_files:
        # 尝试查找 txt 格式
        reference_files = list(raw_dir.rglob("REFERENCE*.txt"))
    
    if not reference_files:
        raise FileNotFoundError(f"找不到 REFERENCE 文件: {raw_dir}")
    
    # 读取所有标签文件
    subject_labels = {}
    for ref_file in reference_files:
        logger.info(f"读取标签文件: {ref_file}")
        
        # 尝试不同的分隔符
        for sep in [',', '\t', ' ']:
            try:
                df = pd.read_csv(ref_file, sep=sep, header=None, names=['file', 'label'])
                if len(df) > 0 and df['label'].dtype in ['int64', 'float64']:
                    break
            except Exception:
                continue
        
        for _, row in df.iterrows():
            file_id = str(row['file']).replace('.wav', '')
            label_val = int(row['label'])
            
            if label_val in original_label_map:
                subject_labels[file_id] = original_label_map[label_val]
    
    logger.info(f"读取 {len(subject_labels)} 条标签")
    
    # 统计标签分布
    label_counts = Counter(subject_labels.values())
    logger.info(f"标签分布: {dict(label_counts)}")
    
    # 受试者级划分
    subjects = list(subject_labels.keys())
    splits = split_subjects(subjects, seed=seed)
    
    # 收集样本记录
    all_records: Dict[str, List[SampleRecord]] = {"train": [], "val": [], "test": []}
    
    # 遍历处理后的数据目录
    for subject_path in tqdm(list(processed_dir.glob("subject_*")), desc="处理受试者"):
        # 尝试匹配 subject_id
        subject_id = subject_path.name.replace("subject_", "")
        
        # PhysioNet 2016 的文件命名可能不同，尝试多种匹配
        matched_id = None
        for sid in subject_labels.keys():
            if sid in subject_id or subject_id in sid:
                matched_id = sid
                break
        
        if matched_id is None:
            continue
        
        label_name = subject_labels[matched_id]
        label = label_map[label_name]
        
        split_name = None
        for s, ids in splits.items():
            if matched_id in ids:
                split_name = s
                break
        
        if split_name is None:
            continue
        
        # 遍历周期文件
        for rec_dir in subject_path.iterdir():
            if rec_dir.is_dir():
                for cycle_file in rec_dir.glob("cycle_*.npy"):
                    dst_dir = output_dir / task_name / split_name / subject_path.name / rec_dir.name
                    dst_path = dst_dir / cycle_file.name
                    
                    link_or_copy_file(cycle_file, dst_path, copy=copy_files)
                    
                    record = SampleRecord(
                        subject_id=matched_id,
                        file_path=str(dst_path.relative_to(output_dir / task_name)),
                        label=label,
                        label_name=label_name,
                        location="",
                        split=split_name
                    )
                    all_records[split_name].append(record)
    
    # 计算类别权重
    all_labels = [r.label for records in all_records.values() for r in records]
    if len(all_labels) == 0:
        logger.warning("未找到任何匹配的样本，请检查数据目录结构")
        class_weights = [1.0, 1.0]
    else:
        class_weights = compute_class_weights(all_labels, num_classes=2)
    
    logger.info(f"类别权重: {class_weights}")
    
    # 保存 CSV
    task_output = output_dir / task_name
    for split_name, records in all_records.items():
        csv_path = task_output / f"{split_name}.csv"
        save_task_csv(records, csv_path, class_weights if split_name == "train" else None)
        logger.info(f"保存 {csv_path.name}: {len(records)} 条记录")
    
    # 创建任务配置
    config = TaskConfig(
        task_name=task_name,
        dataset="physionet2016",
        task_type="classification",
        num_classes=2,
        label_map=label_map,
        class_weights=class_weights,
        split_subjects=splits,
        split_sizes={s: len(r) for s, r in all_records.items()},
        primary_metric="auroc"
    )
    
    config_path = task_output / "task_config.json"
    save_task_config(config, config_path)
    
    return config


# ============== PASCAL 数据集处理 ==============

def prepare_pascal(
    processed_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    seed: int = 42,
    copy_files: bool = False
) -> Optional[TaskConfig]:
    """
    准备 PASCAL 心音三分类任务
    
    标签映射:
    - normal -> 0
    - murmur -> 1
    - extrasystole -> 2
    """
    logger = logging.getLogger(__name__)
    task_name = "pascal"
    
    logger.info(f"准备任务: {task_name}")
    
    # 标签映射
    label_map = {"normal": 0, "murmur": 1, "extrasystole": 2}
    
    # 查找标签文件
    label_files = list(raw_dir.rglob("*labels*.csv")) + list(raw_dir.rglob("*REFERENCE*.csv"))
    
    subject_labels = {}
    
    if label_files:
        for label_file in label_files:
            logger.info(f"读取标签文件: {label_file}")
            df = pd.read_csv(label_file)
            
            # 根据列名适配
            file_col = [c for c in df.columns if 'file' in c.lower() or 'name' in c.lower()]
            label_col = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower()]
            
            if file_col and label_col:
                for _, row in df.iterrows():
                    file_id = str(row[file_col[0]]).replace('.wav', '')
                    label_str = str(row[label_col[0]]).lower().strip()
                    
                    if label_str in label_map:
                        subject_labels[file_id] = label_str
    else:
        # 尝试基于目录结构推断标签
        logger.info("未找到标签文件，尝试从目录结构推断")
        for label_dir in raw_dir.iterdir():
            if label_dir.is_dir():
                label_name = label_dir.name.lower()
                if label_name in label_map:
                    for wav_file in label_dir.glob("*.wav"):
                        subject_labels[wav_file.stem] = label_name
    
    logger.info(f"读取 {len(subject_labels)} 条标签")
    
    if len(subject_labels) == 0:
        logger.warning("未找到任何标签，请检查数据目录结构")
        return None
    
    # 统计标签分布
    label_counts = Counter(subject_labels.values())
    logger.info(f"标签分布: {dict(label_counts)}")
    
    # 受试者级划分
    subjects = list(subject_labels.keys())
    splits = split_subjects(subjects, seed=seed)
    
    # 收集样本记录
    all_records: Dict[str, List[SampleRecord]] = {"train": [], "val": [], "test": []}
    
    for subject_path in tqdm(list(processed_dir.glob("subject_*")), desc="处理受试者"):
        subject_id = subject_path.name.replace("subject_", "")
        
        # 尝试匹配
        matched_id = None
        for sid in subject_labels.keys():
            if sid in subject_id or subject_id in sid:
                matched_id = sid
                break
        
        if matched_id is None:
            continue
        
        label_name = subject_labels[matched_id]
        label = label_map[label_name]
        
        split_name = None
        for s, ids in splits.items():
            if matched_id in ids:
                split_name = s
                break
        
        if split_name is None:
            continue
        
        for rec_dir in subject_path.iterdir():
            if rec_dir.is_dir():
                for cycle_file in rec_dir.glob("cycle_*.npy"):
                    dst_dir = output_dir / task_name / split_name / subject_path.name / rec_dir.name
                    dst_path = dst_dir / cycle_file.name
                    
                    link_or_copy_file(cycle_file, dst_path, copy=copy_files)
                    
                    record = SampleRecord(
                        subject_id=matched_id,
                        file_path=str(dst_path.relative_to(output_dir / task_name)),
                        label=label,
                        label_name=label_name,
                        location="",
                        split=split_name
                    )
                    all_records[split_name].append(record)
    
    # 计算类别权重
    all_labels = [r.label for records in all_records.values() for r in records]
    if len(all_labels) == 0:
        logger.warning("未找到任何匹配的样本")
        class_weights = [1.0, 1.0, 1.0]
    else:
        class_weights = compute_class_weights(all_labels, num_classes=3)
    
    logger.info(f"类别权重: {class_weights}")
    
    # 保存 CSV
    task_output = output_dir / task_name
    for split_name, records in all_records.items():
        csv_path = task_output / f"{split_name}.csv"
        save_task_csv(records, csv_path, class_weights if split_name == "train" else None)
        logger.info(f"保存 {csv_path.name}: {len(records)} 条记录")
    
    # 创建任务配置
    config = TaskConfig(
        task_name=task_name,
        dataset="pascal",
        task_type="classification",
        num_classes=3,
        label_map=label_map,
        class_weights=class_weights,
        split_subjects=splits,
        split_sizes={s: len(r) for s, r in all_records.items()},
        primary_metric="f1_macro"
    )
    
    config_path = task_output / "task_config.json"
    save_task_config(config, config_path)
    
    return config


# ============== 主函数 ==============

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PA-HCL 下游任务数据准备",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 准备 CirCor 杂音检测任务
    python prepare_downstream_tasks.py --dataset circor --task murmur \\
        --processed-dir data/processed/circor \\
        --raw-metadata data/raw/circor/training_data.csv
    
    # 准备所有 CirCor 任务
    python prepare_downstream_tasks.py --dataset circor --task all
    
    # 准备 PhysioNet 2016 任务
    python prepare_downstream_tasks.py --dataset physionet2016 \\
        --processed-dir data/processed/physionet2016 \\
        --raw-dir data/raw/physionet2016
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["circor", "physionet2016", "pascal", "all"],
        default="all",
        help="要准备的数据集 (默认: all)"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["murmur", "outcome", "classification", "all"],
        default="all",
        help="要准备的任务 (仅对 circor 有效)"
    )
    
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="预处理后数据的根目录"
    )
    
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="原始数据的根目录"
    )
    
    parser.add_argument(
        "--raw-metadata",
        type=str,
        default=None,
        help="原始元数据CSV路径 (用于 CirCor)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/downstream",
        help="输出目录"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    parser.add_argument(
        "--copy",
        action="store_true",
        help="复制文件而非创建硬链接"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细日志输出"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    logger = setup_logging(args.verbose)
    set_seed(args.seed)
    
    processed_dir = Path(args.processed_dir)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    
    logger.info("=" * 60)
    logger.info("PA-HCL 下游任务数据准备")
    logger.info("=" * 60)
    logger.info(f"预处理数据目录: {processed_dir}")
    logger.info(f"原始数据目录: {raw_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"随机种子: {args.seed}")
    logger.info("=" * 60)
    
    prepared_tasks = []
    
    # CirCor 数据集
    if args.dataset in ["circor", "all"]:
        circor_processed = processed_dir / "circor" if (processed_dir / "circor").exists() else processed_dir
        
        # 查找元数据文件
        if args.raw_metadata:
            circor_metadata = Path(args.raw_metadata)
        else:
            circor_raw = raw_dir / "circor"
            metadata_candidates = [
                circor_raw / "training_data.csv",
                circor_raw / "circor_metadata.csv",
                raw_dir / "training_data.csv",
            ]
            circor_metadata = None
            for candidate in metadata_candidates:
                if candidate.exists():
                    circor_metadata = candidate
                    break
        
        if circor_metadata and circor_metadata.exists():
            if args.task in ["murmur", "all"]:
                try:
                    config = prepare_circor_murmur(
                        circor_processed, circor_metadata, output_dir,
                        seed=args.seed, copy_files=args.copy
                    )
                    prepared_tasks.append(config.task_name)
                except Exception as e:
                    logger.error(f"准备 circor_murmur 失败: {e}")
            
            if args.task in ["outcome", "all"]:
                try:
                    config = prepare_circor_outcome(
                        circor_processed, circor_metadata, output_dir,
                        seed=args.seed, copy_files=args.copy
                    )
                    prepared_tasks.append(config.task_name)
                except Exception as e:
                    logger.error(f"准备 circor_outcome 失败: {e}")
        else:
            logger.warning(f"找不到 CirCor 元数据文件，跳过 CirCor 任务")
    
    # PhysioNet 2016 数据集
    if args.dataset in ["physionet2016", "all"]:
        physionet_processed = processed_dir / "physionet2016" if (processed_dir / "physionet2016").exists() else processed_dir
        physionet_raw = raw_dir / "physionet2016" if (raw_dir / "physionet2016").exists() else raw_dir
        
        if physionet_processed.exists():
            try:
                config = prepare_physionet2016(
                    physionet_processed, physionet_raw, output_dir,
                    seed=args.seed, copy_files=args.copy
                )
                prepared_tasks.append(config.task_name)
            except Exception as e:
                logger.error(f"准备 physionet2016 失败: {e}")
        else:
            logger.warning(f"找不到 PhysioNet 2016 处理后数据: {physionet_processed}")
    
    # PASCAL 数据集
    if args.dataset in ["pascal", "all"]:
        pascal_processed = processed_dir / "pascal" if (processed_dir / "pascal").exists() else processed_dir
        pascal_raw = raw_dir / "pascal" if (raw_dir / "pascal").exists() else raw_dir
        
        if pascal_processed.exists():
            try:
                config = prepare_pascal(
                    pascal_processed, pascal_raw, output_dir,
                    seed=args.seed, copy_files=args.copy
                )
                if config:
                    prepared_tasks.append(config.task_name)
            except Exception as e:
                logger.error(f"准备 pascal 失败: {e}")
        else:
            logger.warning(f"找不到 PASCAL 处理后数据: {pascal_processed}")
    
    # 总结
    logger.info("=" * 60)
    logger.info("数据准备完成!")
    logger.info(f"已准备的任务: {prepared_tasks}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 60)
    
    if prepared_tasks:
        logger.info("\n下一步: 运行微调训练")
        for task in prepared_tasks:
            logger.info(f"  python scripts/finetune.py --task {task}")
    
    return 0 if prepared_tasks else 1


if __name__ == "__main__":
    exit(main())
