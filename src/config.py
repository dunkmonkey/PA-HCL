"""
PA-HCL 的配置管理。

支持：
- 具有继承功能的 YAML 配置文件
- 基于数据类（Dataclass）的类型检查
- 命令行覆盖
- 实验可复现性
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml
from omegaconf import OmegaConf, DictConfig


# ============== 数据类定义 ==============

@dataclass
class ExperimentConfig:
    """实验元数据配置。"""
    name: str = "pa-hcl"
    seed: int = 42
    output_dir: str = "./checkpoints"
    log_dir: str = "./logs"


@dataclass
class CycleConfig:
    """心动周期提取配置。"""
    min_duration: float = 0.4
    max_duration: float = 1.5
    target_length: int = 4000


@dataclass
class FilterConfig:
    """信号质量过滤配置。"""
    enabled: bool = True
    min_snr_db: float = 10.0


@dataclass
class DataConfig:
    """数据处理配置。"""
    raw_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"
    sample_rate: int = 5000
    num_substructures: int = 4
    cycle: CycleConfig = field(default_factory=CycleConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)


@dataclass
class AugmentationConfig:
    """数据增强配置。"""
    time_shift: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True, "max_shift_ratio": 0.1
    })
    amplitude_scale: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True, "min_scale": 0.8, "max_scale": 1.2
    })
    gaussian_noise: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True, "snr_range": [20, 40]
    })
    freq_mask: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True, "max_mask_ratio": 0.15
    })
    time_mask: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True, "max_mask_ratio": 0.1, "max_mask_width_ms": 30
    })
    speed_perturb: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True, "speed_range": [0.9, 1.1]
    })


@dataclass
class CNNConfig:
    """CNN 主干网络配置。"""
    in_channels: int = 1
    channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_sizes: List[int] = field(default_factory=lambda: [7, 5, 5, 3])
    strides: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    dropout: float = 0.1


@dataclass
class MambaConfig:
    """Mamba (SSM) 模块配置。"""
    d_model: int = 256
    n_layers: int = 4
    d_state: int = 16
    expand: int = 2
    dropout: float = 0.1


@dataclass
class ProjectionHeadConfig:
    """投影头配置。"""
    hidden_dim: int = 512
    output_dim: int = 128
    num_layers: int = 2
    use_bn: bool = True


@dataclass
class EncoderConfig:
    """编码器配置。"""
    type: str = "cnn_mamba"
    cnn: CNNConfig = field(default_factory=CNNConfig)
    mamba: MambaConfig = field(default_factory=MambaConfig)


@dataclass
class ModelConfig:
    """模型架构配置。"""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    projection_head: ProjectionHeadConfig = field(default_factory=ProjectionHeadConfig)
    cycle_feature_dim: int = 256
    sub_feature_dim: int = 128


@dataclass
class LossConfig:
    """损失函数配置。"""
    temperature: float = 0.07
    lambda_cycle: float = 1.0
    lambda_sub: float = 1.0
    dynamic_weight: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False, "warmup_epochs": 10
    })


@dataclass
class OptimizerConfig:
    """优化器配置。"""
    type: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    """学习率调度器配置。"""
    type: str = "cosine"
    warmup_epochs: int = 10
    min_lr: float = 1e-6


@dataclass
class PretrainConfig:
    """预训练配置。"""
    epochs: int = 100
    batch_size: int = 64
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    use_amp: bool = True
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    log_every_n_steps: int = 50
    save_every_n_epochs: int = 10


@dataclass
class FinetuneConfig:
    """微调配置。"""
    epochs: int = 50
    batch_size: int = 32
    encoder_lr: float = 1e-4
    head_lr: float = 1e-3
    freeze_encoder: bool = False
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class TrainingConfig:
    """训练配置。"""
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)


@dataclass
class DistributedConfig:
    """分布式训练配置。"""
    enabled: bool = False
    backend: str = "nccl"
    strategy: str = "ddp"


@dataclass
class HardwareConfig:
    """硬件配置。"""
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "cuda"


@dataclass
class Config:
    """主配置类。"""
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


# ============== 配置加载 ==============

def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """
    加载 YAML 配置文件，并可选择性地应用覆盖。
    
    参数:
        config_path: YAML 配置文件路径
        overrides: 要应用的覆盖字典
        
    返回:
        OmegaConf DictConfig 对象
    """
    # 加载默认配置
    default_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    
    if default_path.exists():
        default_cfg = OmegaConf.load(default_path)
    else:
        default_cfg = OmegaConf.structured(Config())
    
    # 加载指定配置
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            file_cfg = OmegaConf.load(config_path)
            # 处理继承
            if "defaults" in file_cfg:
                del file_cfg["defaults"]
            default_cfg = OmegaConf.merge(default_cfg, file_cfg)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # 应用覆盖
    if overrides is not None:
        override_cfg = OmegaConf.create(overrides)
        default_cfg = OmegaConf.merge(default_cfg, override_cfg)
    
    return default_cfg


def save_config(config: DictConfig, save_path: Union[str, Path]) -> None:
    """
    保存配置到 YAML 文件。
    
    参数:
        config: 要保存的配置
        save_path: 保存配置的路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w") as f:
        OmegaConf.save(config, f)


def print_config(config: DictConfig) -> None:
    """以可读格式打印配置。"""
    print(OmegaConf.to_yaml(config))
