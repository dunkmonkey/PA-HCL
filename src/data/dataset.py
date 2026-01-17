"""
PA-HCL 预训练和下游任务的数据集类。

此模块提供：
- PCGPretrainDataset: 用于带有分层视图的自监督预训练
- PCGDownstreamDataset: 用于监督微调和评估
- 数据整理工具

作者: PA-HCL 团队
"""

import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .preprocessing import (
    load_audio,
    bandpass_filter,
    normalize_signal,
    split_substructures,
)
from .transforms import Compose, get_pretrain_transforms, get_eval_transforms


# ============== Pretraining Dataset ==============

class PCGPretrainDataset(Dataset):
    """
    用于分层对比学习自监督预训练的数据集。
    
    对于每个样本，此数据集返回：
    - 两个增强的心动周期视图（用于周期级对比）
    - 每个视图的 K 个子结构（用于子结构级对比）
    
    数据组织:
        处理后的数据目录应具有以下结构:
        processed/
        ├── subject_0001/
        │   ├── rec_01/
        │   │   ├── cycle_000.npy
        │   │   ├── cycle_001.npy
        │   │   └── ...
        │   └── rec_02/
        │       └── ...
        ├── subject_0002/
        │   └── ...
        └── metadata.json  # 可选，包含处理信息
    
    每个 .npy 文件包含作为一个 1D numpy 数组的单个心动周期。
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        sample_rate: int = 5000,
        num_substructures: int = 4,
        target_length: int = 4000,
        transform: Optional[Compose] = None,
        augmentation_config: Optional[Dict] = None,
        cache_in_memory: bool = False
    ):
        """
        初始化预训练数据集。
        
        参数:
            data_dir: 处理后数据目录的路径
            sample_rate: 音频的采样率
            num_substructures: 子结构片段数量 (K)
            target_length: 每个周期的目标长度（样本数）
            transform: 增强管道。如果为 None，则使用默认的预训练变换
            augmentation_config: 增强配置
            cache_in_memory: 是否将所有数据缓存在内存中（更快但占用更多 RAM）
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.num_substructures = num_substructures
        self.target_length = target_length
        self.cache_in_memory = cache_in_memory
        
        # Set up augmentation pipeline
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_pretrain_transforms(augmentation_config, sample_rate)
        
        # Collect all cycle files
        self.cycle_files = self._collect_cycle_files()
        
        if len(self.cycle_files) == 0:
            raise ValueError(f"No cycle files found in {data_dir}")
        
        # Optional: cache data in memory
        self.cache = {}
        if cache_in_memory:
            self._load_all_to_memory()
    
    def _collect_cycle_files(self) -> List[Path]:
        """
        从数据目录收集所有心动周期文件。
        
        返回:
            周期文件路径列表
        """
        cycle_files = []
        
        # Support both .npy and .pt formats
        for ext in ["*.npy", "*.pt"]:
            cycle_files.extend(self.data_dir.rglob(ext))
        
        # Sort for reproducibility
        cycle_files = sorted(cycle_files)
        
        return cycle_files
    
    def _load_all_to_memory(self) -> None:
        """将所有周期文件加载到内存以加快访问速度。"""
        print(f"Loading {len(self.cycle_files)} cycles into memory...")
        for idx, file_path in enumerate(self.cycle_files):
            self.cache[idx] = self._load_cycle(file_path)
            if (idx + 1) % 1000 == 0:
                print(f"  Loaded {idx + 1}/{len(self.cycle_files)} cycles")
        print("Done loading data into memory.")
    
    def _load_cycle(self, file_path: Path) -> np.ndarray:
        """
        从文件加载单个心动周期。
        
        参数:
            file_path: 周期文件的路径
            
        返回:
            作为 numpy 数组的心动周期
        """
        if file_path.suffix == ".npy":
            cycle = np.load(file_path)
        elif file_path.suffix == ".pt":
            cycle = torch.load(file_path).numpy()
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Ensure float32
        cycle = cycle.astype(np.float32)
        
        # Ensure correct length
        if len(cycle) != self.target_length:
            cycle = self._normalize_length(cycle)
        
        return cycle
    
    def _normalize_length(self, cycle: np.ndarray) -> np.ndarray:
        """通过填充或截断将周期标准化为目标长度。"""
        if len(cycle) < self.target_length:
            # Zero-pad symmetrically
            pad_total = self.target_length - len(cycle)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            cycle = np.pad(cycle, (pad_left, pad_right), mode='constant')
        elif len(cycle) > self.target_length:
            # Center-crop
            start = (len(cycle) - self.target_length) // 2
            cycle = cycle[start:start + self.target_length]
        return cycle
    
    def __len__(self) -> int:
        """返回数据集中心动周期的数量。"""
        return len(self.cycle_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取带有两个增强视图的训练样本。
        
        参数:
            idx: 样本索引
            
        返回:
            包含以下内容的字典:
            - view1: 周期的第一个增强视图 [1, T]
            - view2: 周期的第二个增强视图 [1, T]
            - subs1: view1 的子结构 [K, T/K]
            - subs2: view2 的子结构 [K, T/K]
            - idx: 样本索引（用于调试）
        """
        # Load cycle (from cache or disk)
        if self.cache_in_memory and idx in self.cache:
            cycle = self.cache[idx].copy()  # Copy to avoid modifying cache
        else:
            cycle = self._load_cycle(self.cycle_files[idx])
        
        # Apply augmentations to create two views
        view1 = self.transform(cycle.copy(), self.sample_rate)
        view2 = self.transform(cycle.copy(), self.sample_rate)
        
        # Extract substructures from each view
        subs1 = split_substructures(view1, self.num_substructures)
        subs2 = split_substructures(view2, self.num_substructures)
        
        # Convert to tensors
        # Cycle views: add channel dimension [1, T]
        view1_tensor = torch.from_numpy(view1).float().unsqueeze(0)
        view2_tensor = torch.from_numpy(view2).float().unsqueeze(0)
        
        # Substructures: stack along first dimension [K, T/K]
        subs1_tensor = torch.stack([torch.from_numpy(s).float() for s in subs1])
        subs2_tensor = torch.stack([torch.from_numpy(s).float() for s in subs2])
        
        return {
            "view1": view1_tensor,
            "view2": view2_tensor,
            "subs1": subs1_tensor,
            "subs2": subs2_tensor,
            "idx": idx
        }
    
    def get_subject_id(self, idx: int) -> str:
        """
        获取给定样本索引的受试者 ID。
        
        用于按受试者划分数据。
        
        参数:
            idx: 样本索引
            
        返回:
            受试者 ID 字符串
        """
        file_path = self.cycle_files[idx]
        # Assuming structure: data_dir/subject_id/rec_id/cycle_xxx.npy
        parts = file_path.relative_to(self.data_dir).parts
        if len(parts) >= 1:
            return parts[0]  # First directory is subject_id
        return "unknown"


# ============== Downstream Dataset ==============

class PCGDownstreamDataset(Dataset):
    """
    用于监督下游任务（分类，异常检测）的数据集。
    
    支持:
    - 二分类（正常 vs 异常）
    - 多分类
    - 异常检测（仅使用正常样本进行训练）
    
    数据组织:
        数据目录应具有以下结构:
        
        选项 1 (基于文件的标签):
            data/
            ├── subject_0001/
            │   ├── rec_01.wav
            │   └── ...
            └── labels.csv  # 列: file_path, label
        
        选项 2 (基于目录的标签):
            data/
            ├── normal/
            │   ├── subject_0001_rec_01.wav
            │   └── ...
            └── abnormal/
                ├── subject_0002_rec_01.wav
                └── ...
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        labels_file: Optional[Union[str, Path]] = None,
        sample_rate: int = 5000,
        target_length: int = 4000,
        transform: Optional[Compose] = None,
        mode: str = "train",
        label_map: Optional[Dict[str, int]] = None
    ):
        """
        初始化下游数据集。
        
        参数:
            data_dir: 数据目录的路径
            labels_file: 带有标签的 CSV 文件路径（可选）
            sample_rate: 目标采样率
            target_length: 目标信号长度（样本数）
            transform: 增强管道
            mode: "train", "val", 或 "test"
            label_map: 从标签字符串到整数的映射
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.mode = mode
        
        # Set up transforms
        if transform is not None:
            self.transform = transform
        elif mode == "train":
            self.transform = get_pretrain_transforms(sample_rate=sample_rate)
        else:
            self.transform = get_eval_transforms()
        
        # Load data and labels
        self.samples = self._load_samples(labels_file)
        
        # Set up label mapping
        if label_map is not None:
            self.label_map = label_map
        else:
            self.label_map = self._create_label_map()
        
        self.num_classes = len(self.label_map)
    
    def _load_samples(
        self,
        labels_file: Optional[Path]
    ) -> List[Tuple[Path, str]]:
        """
        加载样本路径和标签。
        
        参数:
            labels_file: 标签 CSV 的路径
            
        返回:
            (file_path, label) 元组的列表
        """
        samples = []
        
        if labels_file is not None:
            # Load from CSV
            import csv
            labels_file = Path(labels_file)
            with open(labels_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file_path = self.data_dir / row['file_path']
                    if file_path.exists():
                        samples.append((file_path, row['label']))
        else:
            # Try directory-based organization
            for label_dir in self.data_dir.iterdir():
                if label_dir.is_dir():
                    label = label_dir.name
                    for file_path in label_dir.glob("*.wav"):
                        samples.append((file_path, label))
                    for file_path in label_dir.glob("*.npy"):
                        samples.append((file_path, label))
        
        if len(samples) == 0:
            raise ValueError(f"No samples found in {self.data_dir}")
        
        return samples
    
    def _create_label_map(self) -> Dict[str, int]:
        """创建从标签字符串到整数的映射。"""
        unique_labels = sorted(set(label for _, label in self.samples))
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def _load_signal(self, file_path: Path) -> np.ndarray:
        """加载和预处理信号文件。"""
        if file_path.suffix == ".wav":
            audio, sr = load_audio(file_path, target_sr=self.sample_rate)
            # Apply bandpass filter
            audio = bandpass_filter(audio, sr)
            # Normalize
            audio = normalize_signal(audio, method="zscore")
        elif file_path.suffix == ".npy":
            audio = np.load(file_path).astype(np.float32)
        elif file_path.suffix == ".pt":
            audio = torch.load(file_path).numpy().astype(np.float32)
        else:
            raise ValueError(f"Unsupported format: {file_path.suffix}")
        
        # Normalize length
        if len(audio) < self.target_length:
            pad = self.target_length - len(audio)
            audio = np.pad(audio, (0, pad), mode='constant')
        elif len(audio) > self.target_length:
            # Random crop for training, center crop for eval
            if self.mode == "train":
                start = random.randint(0, len(audio) - self.target_length)
            else:
                start = (len(audio) - self.target_length) // 2
            audio = audio[start:start + self.target_length]
        
        return audio
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取带标签的样本。
        
        参数:
            idx: 样本索引
            
        返回:
            包含以下内容的字典:
            - signal: 音频信号 [1, T]
            - label: 类标签 (整数)
            - idx: 样本索引
        """
        file_path, label_str = self.samples[idx]
        
        # Load signal
        signal = self._load_signal(file_path)
        
        # Apply augmentation
        if self.transform is not None:
            signal = self.transform(signal, self.sample_rate)
        
        # Convert to tensor
        signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)
        label_tensor = torch.tensor(self.label_map[label_str], dtype=torch.long)
        
        return {
            "signal": signal_tensor,
            "label": label_tensor,
            "idx": idx
        }
    
    def get_class_weights(self) -> torch.Tensor:
        """
        计算不平衡数据集的类别权重。
        
        返回:
            类别权重张量
        """
        label_counts = {}
        for _, label in self.samples:
            label_idx = self.label_map[label]
            label_counts[label_idx] = label_counts.get(label_idx, 0) + 1
        
        total = len(self.samples)
        weights = []
        for i in range(self.num_classes):
            count = label_counts.get(i, 1)
            weights.append(total / (self.num_classes * count))
        
        return torch.tensor(weights, dtype=torch.float32)


# ============== Data Collation ==============

def pretrain_collate_fn(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    预训练批次的整理函数。
    
    沿批次维度堆叠所有张量。
    
    参数:
        batch: 样本字典列表
        
    返回:
        带有以下形状的张量的批次字典:
        - view1, view2: [B, 1, T]
        - subs1, subs2: [B, K, T/K]
        - idx: [B]
    """
    return {
        "view1": torch.stack([sample["view1"] for sample in batch]),
        "view2": torch.stack([sample["view2"] for sample in batch]),
        "subs1": torch.stack([sample["subs1"] for sample in batch]),
        "subs2": torch.stack([sample["subs2"] for sample in batch]),
        "idx": torch.tensor([sample["idx"] for sample in batch])
    }


def downstream_collate_fn(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    下游任务批次的整理函数。
    
    参数:
        batch: 样本字典列表
        
    返回:
        带有张量的批次字典
    """
    return {
        "signal": torch.stack([sample["signal"] for sample in batch]),
        "label": torch.stack([sample["label"] for sample in batch]),
        "idx": torch.tensor([sample["idx"] for sample in batch])
    }


# ============== Data Splitting Utilities ==============

def create_subject_wise_split(
    dataset: Union[PCGPretrainDataset, PCGDownstreamDataset],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    创建遵守受试者边界的 训练/验证/测试 划分。
    
    这确保了划分之间没有数据泄漏 - 来自同一受试者的所有样本都在同一个划分中。
    
    参数:
        dataset: 数据集实例
        train_ratio: 训练比例
        val_ratio: 验证比例
        test_ratio: 测试比例
        seed: 用于重现性的随机种子
        
    返回:
        (train_indices, val_indices, test_indices) 的元组
    """
    # Get subject IDs for all samples
    subject_to_indices = {}
    for idx in range(len(dataset)):
        subject_id = dataset.get_subject_id(idx)
        if subject_id not in subject_to_indices:
            subject_to_indices[subject_id] = []
        subject_to_indices[subject_id].append(idx)
    
    # Shuffle subjects
    subjects = list(subject_to_indices.keys())
    random.seed(seed)
    random.shuffle(subjects)
    
    # Split subjects
    n_subjects = len(subjects)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)
    
    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:n_train + n_val]
    test_subjects = subjects[n_train + n_val:]
    
    # Collect indices for each split
    train_indices = []
    for subject in train_subjects:
        train_indices.extend(subject_to_indices[subject])
    
    val_indices = []
    for subject in val_subjects:
        val_indices.extend(subject_to_indices[subject])
    
    test_indices = []
    for subject in test_subjects:
        test_indices.extend(subject_to_indices[subject])
    
    return train_indices, val_indices, test_indices


def create_dataloaders(
    dataset: Dataset,
    train_indices: List[int],
    val_indices: List[int],
    test_indices: Optional[List[int]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    从索引划分创建 DataLoaders。
    
    参数:
        dataset: 完整数据集
        train_indices: 训练索引
        val_indices: 验证索引
        test_indices: 测试索引（可选）
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        collate_fn: 自定义整理函数
        pin_memory: 是否锁定内存（更快的 GPU 传输）
        
    返回:
        (train_loader, val_loader, test_loader) 的元组
    """
    from torch.utils.data import Subset
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    test_loader = None
    if test_indices is not None:
        test_dataset = Subset(dataset, test_indices)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=False
        )
    
    return train_loader, val_loader, test_loader
