"""
PA-HCL 预训练和下游任务的数据集类。

此模块提供：
- PCGPretrainDataset: 用于带有分层视图的自监督预训练
- PCGDownstreamDataset: 用于监督微调和评估
- 任务感知数据加载工具
- 数据整理工具

支持的下游任务：
- CirCor 杂音检测三分类 (circor_murmur)
- CirCor 临床结果二分类 (circor_outcome)
- PhysioNet 2016 心音二分类 (physionet2016)
- PASCAL 心音三分类 (pascal)

作者: PA-HCL 团队
"""

import csv
import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

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
        split: Optional[str] = None,
        sample_rate: int = 5000,
        num_substructures: int = 4,
        target_length: int = 4000,
        transform: Optional[Compose] = None,
        augmentation_config: Optional[Dict] = None,
        cache_in_memory: bool = True,
        use_gpu_augment: bool = False,
        view_cache_refresh_epochs: int = 5
    ):
        """
        初始化预训练数据集。
        
        参数:
            data_dir: 处理后数据目录的路径
            split: 数据划分标识（可选，当前实现不会据此过滤数据）
            sample_rate: 音频的采样率
            num_substructures: 子结构片段数量 (K)
            target_length: 每个周期的目标长度（样本数）
            transform: 增强管道。如果为 None，则使用默认的预训练变换
            augmentation_config: 增强配置
            cache_in_memory: 是否将所有数据缓存在内存中（默认 True，更快但占用更多 RAM）
            use_gpu_augment: 是否使用 GPU 增强（如果为 True，__getitem__ 返回原始数据，增强在 GPU 上进行）
            view_cache_refresh_epochs: 视图缓存刷新间隔（每 N 个 epoch 刷新预计算的增强视图）
        """
        self.data_dir = Path(data_dir)
        self.split = split
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
        
        # GPU augmentation mode
        self.use_gpu_augment = use_gpu_augment
        self.view_cache_refresh_epochs = view_cache_refresh_epochs
        
        # Optional: cache data in memory
        self.cache = {}
        if cache_in_memory:
            self._load_all_to_memory()
        
        # View cache for precomputed augmented views
        self.view_cache: Dict[int, Tuple[np.ndarray, np.ndarray, List, List]] = {}
        self.current_cache_epoch: int = -1
    
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
    
    def refresh_view_cache(self, epoch: int, force: bool = False) -> None:
        """
        刷新预计算的视图缓存。
        
        每 view_cache_refresh_epochs 个 epoch 重新预计算增强视图，
        以在训练效率和增强随机性之间取得平衡。
        
        参数:
            epoch: 当前 epoch 数
            force: 是否强制刷新（忽略刷新间隔）
        """
        # 检查是否需要刷新
        should_refresh = (
            force or 
            self.current_cache_epoch < 0 or
            (epoch - self.current_cache_epoch) >= self.view_cache_refresh_epochs
        )
        
        if not should_refresh:
            return
        
        # 如果使用 GPU 增强，不需要预计算视图缓存
        if self.use_gpu_augment:
            self.current_cache_epoch = epoch
            return
        
        print(f"Refreshing view cache at epoch {epoch}...")
        self.view_cache.clear()
        
        for idx in range(len(self.cycle_files)):
            # 获取原始周期
            if self.cache_in_memory and idx in self.cache:
                cycle = self.cache[idx].copy()
            else:
                cycle = self._load_cycle(self.cycle_files[idx])
            
            # 生成两个增强视图
            view1 = self.transform(cycle.copy(), self.sample_rate)
            view2 = self.transform(cycle.copy(), self.sample_rate)
            
            # 提取子结构
            subs1 = split_substructures(view1, self.num_substructures)
            subs2 = split_substructures(view2, self.num_substructures)
            
            self.view_cache[idx] = (view1, view2, subs1, subs2)
            
            if (idx + 1) % 1000 == 0:
                print(f"  Cached {idx + 1}/{len(self.cycle_files)} views")
        
        self.current_cache_epoch = epoch
        print(f"View cache refreshed: {len(self.view_cache)} samples cached.")
    
    def clear_view_cache(self) -> None:
        """清空视图缓存以释放内存。"""
        self.view_cache.clear()
        self.current_cache_epoch = -1
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取带有两个增强视图的训练样本。
        
        参数:
            idx: 样本索引
            
        返回:
            包含以下内容的字典:
            - view1: 周期的第一个增强视图 [1, T]
            - view2: 周期的第二个增强视图 [1, T]
            - subs1: view1 的子结构 [K, T/K] (如果不使用 GPU 增强)
            - subs2: view2 的子结构 [K, T/K] (如果不使用 GPU 增强)
            - idx: 样本索引（用于调试）
        """
        # GPU 增强模式: 只返回原始周期数据，增强在 GPU 上批量进行
        if self.use_gpu_augment:
            # 从缓存或磁盘加载原始周期
            if self.cache_in_memory and idx in self.cache:
                cycle = self.cache[idx].copy()
            else:
                cycle = self._load_cycle(self.cycle_files[idx])
            
            # 转为张量 [1, T]
            cycle_tensor = torch.from_numpy(cycle).float().unsqueeze(0)
            
            # 返回原始周期（GPU 增强将在训练循环中应用）
            return {
                "view1": cycle_tensor,
                "view2": cycle_tensor.clone(),  # 复制用于独立增强
                "subs1": None,  # GPU 增强后再分割
                "subs2": None,
                "idx": idx
            }
        
        # 优先从视图缓存读取
        if idx in self.view_cache:
            view1, view2, subs1, subs2 = self.view_cache[idx]
            
            # 转换为张量
            view1_tensor = torch.from_numpy(view1).float().unsqueeze(0)
            view2_tensor = torch.from_numpy(view2).float().unsqueeze(0)
            subs1_tensor = torch.stack([torch.from_numpy(s).float() for s in subs1])
            subs2_tensor = torch.stack([torch.from_numpy(s).float() for s in subs2])
            
            return {
                "view1": view1_tensor,
                "view2": view2_tensor,
                "subs1": subs1_tensor,
                "subs2": subs2_tensor,
                "idx": idx
            }
        
        # 回退: 从缓存或磁盘加载并实时增强
        if self.cache_in_memory and idx in self.cache:
            cycle = self.cache[idx].copy()
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
    - 多分类（杂音检测、PASCAL分类等）
    - 异常检测（仅使用正常样本进行训练）
    - 任务感知加载（从任务配置自动加载）
    
    数据组织:
        数据目录应具有以下结构:
        
        选项 1 (基于CSV的标签 - 推荐):
            data/downstream/circor_murmur/
            ├── train/
            │   └── subject_xxx/rec_xx/cycle_xxx.npy
            ├── val/
            ├── test/
            ├── train.csv
            ├── val.csv
            ├── test.csv
            └── task_config.json
        
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
        csv_path: Optional[Union[str, Path]] = None,
        sample_rate: int = 5000,
        target_length: int = 4000,
        transform: Optional[Compose] = None,
        mode: str = "train",
        split: Optional[str] = None,
        label_map: Optional[Dict[str, int]] = None,
        task_config: Optional[Dict] = None,
        # 兼容旧参数名
        labels_file: Optional[Union[str, Path]] = None,
        # 性能优化参数
        cache_in_memory: bool = True,
        use_gpu_augment: bool = False
    ):
        """
        初始化下游数据集。
        
        参数:
            data_dir: 数据目录的路径
            csv_path: 标签CSV文件路径（推荐使用此参数）
            sample_rate: 目标采样率
            target_length: 目标信号长度（样本数）
            transform: 增强管道
            mode: "train", "val", 或 "test"
            split: 数据划分别名（可选，等价于 mode）
            label_map: 从标签字符串到整数的映射
            task_config: 任务配置字典（从 task_config.json 加载）
            labels_file: 兼容旧参数名，等同于 csv_path
            cache_in_memory: 是否将所有信号缓存在内存中（默认 True）
            use_gpu_augment: 是否使用 GPU 增强（如果为 True，__getitem__ 返回原始数据）
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.target_length = target_length
        if split is not None:
            mode = split
        self.mode = mode
        self._class_weights = None
        self.task_config = task_config
        self.cache_in_memory = cache_in_memory
        self.use_gpu_augment = use_gpu_augment
        
        # 兼容旧参数名
        if csv_path is None and labels_file is not None:
            csv_path = labels_file
        
        # 从任务配置加载设置
        if task_config is not None:
            if label_map is None and "label_map" in task_config:
                label_map = task_config["label_map"]
            if "class_weights" in task_config:
                self._class_weights = torch.tensor(
                    task_config["class_weights"], dtype=torch.float32
                )
        
        # Set up transforms
        if transform is not None:
            self.transform = transform
        elif mode == "train":
            self.transform = get_pretrain_transforms(sample_rate=sample_rate)
        else:
            self.transform = get_eval_transforms()
        
        # Load data and labels
        self.samples, csv_class_weights = self._load_samples(csv_path)
        
        # 优先使用CSV中的类别权重
        if csv_class_weights is not None and self._class_weights is None:
            self._class_weights = torch.tensor(csv_class_weights, dtype=torch.float32)
        
        # Set up label mapping
        if label_map is not None:
            self.label_map = label_map
        else:
            self.label_map = self._create_label_map()
        
        # 创建反向映射 (int -> str)
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        
        self.num_classes = len(self.label_map)
        
        # 内存缓存
        self.signal_cache: Dict[int, np.ndarray] = {}
        if cache_in_memory:
            self._load_all_signals_to_memory()
    
    def _load_samples(
        self,
        csv_path: Optional[Union[str, Path]]
    ) -> Tuple[List[Tuple[Path, str, Optional[str]]], Optional[List[float]]]:
        """
        加载样本路径和标签。
        
        参数:
            csv_path: 标签 CSV 的路径
            
        返回:
            ((file_path, label, subject_id) 元组的列表, 类别权重)
        """
        samples = []
        class_weights = None
        
        if csv_path is not None:
            # Load from CSV (新格式)
            csv_path = Path(csv_path)
            with open(csv_path, 'r') as f:
                lines = f.readlines()
            
            # 解析注释行中的类别权重
            data_lines = []
            for line in lines:
                line = line.strip()
                if line.startswith('# class_weights:'):
                    # 解析类别权重: # class_weights: [1.0, 2.0, 3.0]
                    try:
                        weights_str = line.split(':', 1)[1].strip()
                        class_weights = json.loads(weights_str)
                    except (json.JSONDecodeError, IndexError):
                        pass
                elif line and not line.startswith('#'):
                    data_lines.append(line)
            
            # 解析 CSV 数据
            if data_lines:
                import io
                reader = csv.DictReader(io.StringIO('\n'.join(data_lines)))
                for row in reader:
                    # 支持多种列名格式
                    file_path_str = row.get('file_path', row.get('path', ''))
                    
                    # 尝试不同的标签列名
                    label = row.get('label_name', row.get('label', ''))
                    if label.isdigit():
                        # 如果 label 是数字，尝试获取 label_name
                        label = row.get('label_name', label)
                    
                    subject_id = row.get('subject_id', '')
                    
                    # 构建完整路径
                    file_path = self.data_dir / file_path_str
                    if not file_path.exists():
                        # 尝试其他路径组合
                        file_path = Path(file_path_str)
                        if not file_path.exists():
                            continue
                    
                    samples.append((file_path, str(label), subject_id))
        else:
            # Try directory-based organization
            for label_dir in self.data_dir.iterdir():
                if label_dir.is_dir():
                    label = label_dir.name
                    for file_path in label_dir.glob("*.wav"):
                        samples.append((file_path, label, ""))
                    for file_path in label_dir.glob("*.npy"):
                        samples.append((file_path, label, ""))
        
        if len(samples) == 0:
            raise ValueError(f"No samples found in {self.data_dir}")
        
        return samples, class_weights
    
    def _create_label_map(self) -> Dict[str, int]:
        """创建从标签字符串到整数的映射。"""
        unique_labels = sorted(set(label for _, label, *_ in self.samples))
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def _load_all_signals_to_memory(self) -> None:
        """将所有信号加载到内存以加快访问速度。"""
        print(f"Loading {len(self.samples)} signals into memory...")
        for idx in range(len(self.samples)):
            file_path = self.samples[idx][0]
            try:
                self.signal_cache[idx] = self._load_signal(file_path)
            except Exception as e:
                print(f"  Warning: Failed to load {file_path}: {e}")
            if (idx + 1) % 500 == 0:
                print(f"  Loaded {idx + 1}/{len(self.samples)} signals")
        print(f"Done loading {len(self.signal_cache)} signals into memory.")
    
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
        sample = self.samples[idx]
        file_path = sample[0]
        label_str = sample[1]
        subject_id = sample[2] if len(sample) > 2 else ""
        
        # 从缓存或磁盘加载信号
        if self.cache_in_memory and idx in self.signal_cache:
            signal = self.signal_cache[idx].copy()
        else:
            signal = self._load_signal(file_path)
        
        # GPU 增强模式: 跳过 CPU 增强，增强在 GPU 上批量进行
        if not self.use_gpu_augment and self.transform is not None:
            signal = self.transform(signal, self.sample_rate)
        
        # Convert to tensor
        signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)
        label_tensor = torch.tensor(self.label_map[label_str], dtype=torch.long)
        
        return {
            "signal": signal_tensor,
            "label": label_tensor,
            "idx": idx,
            "subject_id": subject_id
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
        sample = self.samples[idx]
        if len(sample) > 2 and sample[2]:
            return sample[2]
        
        # 从文件路径推断
        file_path = sample[0]
        parts = file_path.parts
        for part in parts:
            if part.startswith('subject_'):
                return part.replace('subject_', '')
        
        return "unknown"
    
    def get_class_weights(self) -> torch.Tensor:
        """
        获取类别权重（用于处理不平衡数据）。
        
        优先返回从配置/CSV加载的权重，否则动态计算。
        
        返回:
            类别权重张量
        """
        # 优先使用预加载的权重
        if self._class_weights is not None:
            return self._class_weights
        
        # 动态计算
        label_counts = {}
        for sample in self.samples:
            label_str = sample[1]
            label_idx = self.label_map[label_str]
            label_counts[label_idx] = label_counts.get(label_idx, 0) + 1
        
        total = len(self.samples)
        weights = []
        for i in range(self.num_classes):
            count = label_counts.get(i, 1)
            weights.append(total / (self.num_classes * count))
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_label_distribution(self) -> Dict[str, int]:
        """
        获取标签分布统计。
        
        返回:
            {标签名: 数量} 的字典
        """
        distribution = {}
        for sample in self.samples:
            label_str = sample[1]
            distribution[label_str] = distribution.get(label_str, 0) + 1
        return distribution
    
    @classmethod
    def from_task(
        cls,
        task_name: str,
        split: str,
        base_dir: Union[str, Path] = "data/downstream",
        **kwargs
    ) -> "PCGDownstreamDataset":
        """
        从任务名称创建数据集（便捷方法）。
        
        自动加载任务配置和对应的CSV文件。
        
        参数:
            task_name: 任务名称（如 'circor_murmur', 'physionet2016'）
            split: 数据划分（'train', 'val', 'test'）
            base_dir: 下游数据根目录
            **kwargs: 传递给构造函数的其他参数
            
        返回:
            PCGDownstreamDataset 实例
            
        示例:
            >>> dataset = PCGDownstreamDataset.from_task('circor_murmur', 'train')
        """
        base_dir = Path(base_dir)
        task_dir = base_dir / task_name
        
        # 加载任务配置
        config_path = task_dir / "task_config.json"
        task_config = None
        if config_path.exists():
            with open(config_path, 'r') as f:
                task_config = json.load(f)
        
        # 构建 CSV 路径
        csv_path = task_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")
        
        # 数据目录是任务目录
        data_dir = task_dir
        
        # 设置 mode
        mode = "train" if split == "train" else "eval"
        
        return cls(
            data_dir=data_dir,
            csv_path=csv_path,
            mode=mode,
            task_config=task_config,
            **kwargs
        )


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
    train_indices: Optional[List[int]] = None,
    val_indices: Optional[List[int]] = None,
    test_indices: Optional[List[int]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = True,
    shuffle: bool = True,
    distributed: bool = False,
    drop_last: Optional[bool] = None,
    prefetch_factor: int = 4,
    persistent_workers: bool = True
) -> Union[DataLoader, Tuple[DataLoader, DataLoader, Optional[DataLoader]]]:
    """
    创建 DataLoader（支持单数据集或按索引划分）。
    
    用法:
        - 仅传入 dataset：返回单个 DataLoader
        - 传入 train/val(/test) 索引：返回 (train_loader, val_loader, test_loader)
    
    参数:
        dataset: 完整数据集或已划分的数据集
        train_indices: 训练索引（可选）
        val_indices: 验证索引（可选）
        test_indices: 测试索引（可选）
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        collate_fn: 自定义整理函数
        pin_memory: 是否锁定内存（更快的 GPU 传输）
        shuffle: 是否打乱（单数据集模式生效）
        distributed: 是否使用分布式采样器
        drop_last: 是否丢弃不足一个 batch 的样本（若为 None，则按 shuffle 决定）
        prefetch_factor: 每个 worker 预取的 batch 数（提高吞吐量）
        persistent_workers: 是否保持 worker 进程存活（减少启动开销）
        
    返回:
        DataLoader 或 (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import Subset
    from torch.utils.data.distributed import DistributedSampler
    
    # 计算是否启用 persistent_workers (需要 num_workers > 0)
    use_persistent = persistent_workers and num_workers > 0
    use_prefetch = num_workers > 0
    
    # 索引划分模式
    if train_indices is not None or val_indices is not None:
        if train_indices is None or val_indices is None:
            raise ValueError("train_indices 和 val_indices 需同时提供")
        
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False if train_sampler is not None else True,
            sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=True,
            prefetch_factor=prefetch_factor if use_prefetch else None,
            persistent_workers=use_persistent
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
            prefetch_factor=prefetch_factor if use_prefetch else None,
            persistent_workers=use_persistent
        )
        
        test_loader = None
        if test_indices is not None:
            test_dataset = Subset(dataset, test_indices)
            test_sampler = DistributedSampler(test_dataset, shuffle=False) if distributed else None
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=test_sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                drop_last=False,
                prefetch_factor=prefetch_factor if use_prefetch else None,
                persistent_workers=use_persistent
            )
        
        return train_loader, val_loader, test_loader
    
    # 单数据集模式
    if drop_last is None:
        drop_last = shuffle
    
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sampler is not None else shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor if use_prefetch else None,
        persistent_workers=use_persistent
    )


# ============== Task-Aware Data Loading ==============

def create_task_dataloaders(
    task_name: str,
    base_dir: Union[str, Path] = "data/downstream",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    **dataset_kwargs
) -> Dict[str, DataLoader]:
    """
    为指定任务创建所有数据加载器。
    
    这是推荐的数据加载方式，自动处理:
    - 任务配置加载
    - CSV 文件解析
    - 类别权重计算
    
    参数:
        task_name: 任务名称 ('circor_murmur', 'circor_outcome', 'physionet2016', 'pascal')
        base_dir: 下游数据根目录
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        pin_memory: 是否锁定内存
        prefetch_factor: 每个 worker 预取的 batch 数
        persistent_workers: 是否保持 worker 进程存活
        **dataset_kwargs: 传递给 PCGDownstreamDataset 的额外参数
        
    返回:
        包含 'train', 'val', 'test' DataLoader 的字典，
        以及 'class_weights', 'num_classes', 'label_map' 等元信息
        
    示例:
        >>> loaders = create_task_dataloaders('circor_murmur', batch_size=32)
        >>> train_loader = loaders['train']
        >>> class_weights = loaders['class_weights']
    """
    base_dir = Path(base_dir)
    task_dir = base_dir / task_name
    
    if not task_dir.exists():
        raise FileNotFoundError(f"任务目录不存在: {task_dir}")
    
    # 加载任务配置
    config_path = task_dir / "task_config.json"
    task_config = None
    if config_path.exists():
        with open(config_path, 'r') as f:
            task_config = json.load(f)
    
    # 计算是否启用优化选项
    use_persistent = persistent_workers and num_workers > 0
    use_prefetch = num_workers > 0
    
    result = {
        'task_name': task_name,
        'task_config': task_config,
    }
    
    # 创建各个 split 的数据集和加载器
    for split in ['train', 'val', 'test']:
        csv_path = task_dir / f"{split}.csv"
        
        if not csv_path.exists():
            result[split] = None
            continue
        
        # 设置 mode
        mode = "train" if split == "train" else "eval"
        
        # 创建数据集
        dataset = PCGDownstreamDataset(
            data_dir=task_dir,
            csv_path=csv_path,
            mode=mode,
            task_config=task_config,
            **dataset_kwargs
        )
        
        # 创建数据加载器
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=downstream_collate_fn,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
            prefetch_factor=prefetch_factor if use_prefetch else None,
            persistent_workers=use_persistent
        )
        
        result[split] = loader
        
        # 从训练集获取元信息
        if split == 'train':
            result['class_weights'] = dataset.get_class_weights()
            result['num_classes'] = dataset.num_classes
            result['label_map'] = dataset.label_map
            result['label_distribution'] = dataset.get_label_distribution()
    
    return result


def get_available_tasks(base_dir: Union[str, Path] = "data/downstream") -> List[str]:
    """
    获取所有可用的下游任务列表。
    
    参数:
        base_dir: 下游数据根目录
        
    返回:
        任务名称列表
    """
    base_dir = Path(base_dir)
    tasks = []
    
    if not base_dir.exists():
        return tasks
    
    for task_dir in base_dir.iterdir():
        if task_dir.is_dir():
            config_path = task_dir / "task_config.json"
            train_csv = task_dir / "train.csv"
            if config_path.exists() or train_csv.exists():
                tasks.append(task_dir.name)
    
    return sorted(tasks)
