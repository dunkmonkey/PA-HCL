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
from typing import Any, Dict, List, Optional, Tuple, Union, cast
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
            file_cfg_container = OmegaConf.to_container(file_cfg, resolve=False)
            if isinstance(file_cfg_container, dict) and "defaults" in file_cfg_container:
                file_cfg_container.pop("defaults", None)
                file_cfg = OmegaConf.create(file_cfg_container)
            default_cfg = OmegaConf.merge(default_cfg, file_cfg)
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # 应用覆盖
    if overrides is not None:
        override_cfg = OmegaConf.create(overrides)
        default_cfg = OmegaConf.merge(default_cfg, override_cfg)
    
    if not isinstance(default_cfg, DictConfig):
        default_cfg = OmegaConf.create(default_cfg)
    if not isinstance(default_cfg, DictConfig):
        raise TypeError("Loaded config is not a DictConfig")
    return cast(DictConfig, default_cfg)


def save_config(config: Any, save_path: Union[str, Path]) -> None:
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


def print_config(config: Any) -> None:
    """以可读格式打印配置。"""
    print(OmegaConf.to_yaml(config))


def _select_first(config: Any, keys: List[str], default: Any = None) -> Any:
    """按顺序返回第一个存在的配置键。"""
    for key in keys:
        value = OmegaConf.select(config, key, default=None)
        if value is not None:
            return value
    return default


def build_effective_run_config(config: Any, stage: str) -> DictConfig:
    """
    构建当前训练阶段的“有效配置视图”。

    该视图仅包含训练脚本实际会读取的关键参数，用于简洁日志输出与可追溯保存。
    """
    stage = stage.lower()

    common = {
        "experiment": {
            "name": _select_first(config, ["experiment.name"], "unknown"),
        },
        "seed": _select_first(config, ["seed", "experiment.seed"], 42),
        "task": {
            "name": _select_first(config, ["task.name"], None),
        },
        "data": {
            "sample_rate": _select_first(config, ["data.sample_rate"], 5000),
            "target_length": _select_first(config, ["data.target_length", "data.cycle.target_length"], 4000),
            "num_substructures": _select_first(config, ["data.num_substructures"], 4),
        },
        "data_cache": {
            "use_gpu_augment": _select_first(config, ["data_cache.use_gpu_augment"], True),
        },
        "model": {
            "encoder_type": _select_first(config, ["model.encoder_type", "model.encoder.type"], "cnn_mamba"),
            "cnn_channels": _select_first(config, ["model.cnn_channels", "model.encoder.cnn.channels"], None),
            "mamba_d_model": _select_first(config, ["model.mamba_d_model", "model.encoder.mamba.d_model"], None),
            "mamba_n_layers": _select_first(config, ["model.mamba_n_layers", "model.encoder.mamba.n_layers"], None),
        },
    }

    if stage == "pretrain":
        effective = {
            **common,
            "training": {
                "num_epochs": _select_first(config, ["training.num_epochs", "training.pretrain.epochs"], 100),
                "batch_size": _select_first(config, ["training.batch_size", "training.pretrain.batch_size"], 64),
                "learning_rate": _select_first(config, ["training.learning_rate", "training.pretrain.optimizer.lr"], 1e-3),
                "weight_decay": _select_first(config, ["training.weight_decay", "training.pretrain.optimizer.weight_decay"], 1e-4),
                "warmup_epochs": _select_first(config, ["training.warmup_epochs", "training.pretrain.scheduler.warmup_epochs"], 10),
                "min_lr": _select_first(config, ["training.min_lr", "training.pretrain.scheduler.min_lr"], 1e-6),
                "use_amp": _select_first(config, ["training.use_amp", "training.pretrain.use_amp"], True),
                "gradient_accumulation_steps": _select_first(
                    config,
                    ["training.gradient_accumulation_steps", "training.pretrain.gradient_accumulation_steps"],
                    1,
                ),
                "grad_clip_norm": _select_first(config, ["training.grad_clip_norm", "training.pretrain.grad_clip"], 1.0),
                "log_interval": _select_first(config, ["training.log_interval", "training.pretrain.log_every_n_steps"], 50),
                "save_interval": _select_first(config, ["training.save_interval", "training.pretrain.save_every_n_epochs"], 10),
            },
            "loss": {
                "temperature": _select_first(config, ["loss.temperature"], 0.07),
                "lambda_cycle": _select_first(config, ["loss.lambda_cycle"], 1.0),
                "lambda_sub": _select_first(config, ["loss.lambda_sub"], 1.0),
            },
        }
    elif stage == "finetune":
        effective = {
            **common,
            "training": {
                "num_epochs": _select_first(config, ["training.num_epochs", "downstream.num_epochs", "training.finetune.epochs"], 100),
                "batch_size": _select_first(config, ["training.batch_size", "downstream.batch_size", "training.finetune.batch_size"], 32),
                "learning_rate": _select_first(config, ["training.learning_rate", "downstream.learning_rate"], 1e-3),
                "weight_decay": _select_first(config, ["training.weight_decay", "downstream.weight_decay", "training.finetune.optimizer.weight_decay"], 1e-4),
                "warmup_epochs": _select_first(config, ["training.warmup_epochs", "downstream.warmup_epochs", "training.finetune.scheduler.warmup_epochs"], 5),
                "min_lr": _select_first(config, ["training.min_lr", "training.finetune.scheduler.min_lr"], 1e-7),
                "label_smoothing": _select_first(config, ["training.label_smoothing", "downstream.label_smoothing"], 0.1),
                "use_class_weights": _select_first(config, ["training.use_class_weights", "downstream.use_class_weights"], True),
                "early_stopping_patience": _select_first(config, ["training.early_stopping_patience", "downstream.early_stopping_patience"], 15),
                "use_amp": _select_first(config, ["training.use_amp"], True),
                "log_interval": _select_first(config, ["training.log_interval"], 20),
                "save_interval": _select_first(config, ["training.save_interval"], 10),
            },
            "classifier": {
                "hidden_dim": _select_first(config, ["classifier.hidden_dim", "downstream.hidden_dim"], 128),
                "dropout": _select_first(config, ["classifier.dropout", "downstream.dropout"], 0.3),
                "num_classes": _select_first(config, ["classification.num_classes", "downstream.num_classes"], None),
            },
            "evaluation": {
                "primary_metric": _select_first(config, ["evaluation.primary_metric"], None),
            },
        }
    elif stage == "supervised":
        effective = {
            **common,
            "training": {
                "num_epochs": _select_first(config, ["training.num_epochs"], 50),
                "batch_size": _select_first(config, ["training.batch_size"], 32),
                "learning_rate": _select_first(config, ["training.learning_rate"], 1e-3),
                "weight_decay": _select_first(config, ["training.weight_decay"], 1e-4),
                "warmup_epochs": _select_first(config, ["training.warmup_epochs"], 10),
                "min_lr": _select_first(config, ["training.min_lr"], 1e-7),
                "label_smoothing": _select_first(config, ["training.label_smoothing"], 0.1),
                "use_class_weights": _select_first(config, ["training.use_class_weights"], True),
                "use_balanced_sampling": _select_first(config, ["training.use_balanced_sampling"], False),
                "loss_type": _select_first(config, ["training.loss_type"], "ce"),
                "focal_gamma": _select_first(config, ["training.focal_gamma"], 2.0),
                "early_stopping_patience": _select_first(config, ["training.early_stopping_patience"], 20),
                "use_amp": _select_first(config, ["training.use_amp"], True),
                "log_interval": _select_first(config, ["training.log_interval"], 20),
                "save_interval": _select_first(config, ["training.save_interval"], 10),
            },
            "classifier": {
                "hidden_dim": _select_first(config, ["classifier.hidden_dim"], 128),
                "dropout": _select_first(config, ["classifier.dropout"], 0.3),
                "use_bn": _select_first(config, ["classifier.use_bn"], True),
                "use_ln": _select_first(config, ["classifier.use_ln"], False),
                "num_layers": _select_first(config, ["classifier.num_layers"], 1),
            },
            "evaluation": {
                "primary_metric": _select_first(config, ["evaluation.primary_metric"], None),
            },
        }
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    return OmegaConf.create(effective)


def _compact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """删除字典中值为 None 的键，递归处理子字典。"""
    compact: Dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            compact[key] = _compact_dict(value)
        elif value is not None:
            compact[key] = value
    return compact


def build_config_summary(config: Any, stage: str) -> DictConfig:
    """
    构建统一模板的配置摘要（固定顺序）：
    task -> model -> training -> loss -> data_cache
    """
    effective = build_effective_run_config(config, stage)
    effective_dict = OmegaConf.to_container(effective, resolve=True)
    if not isinstance(effective_dict, dict):
        raise TypeError("Effective config must be a dict-like structure")

    stage_lower = stage.lower()
    loss_section = effective_dict.get("loss")
    if not isinstance(loss_section, dict):
        if stage_lower == "finetune":
            loss_section = {
                "label_smoothing": _select_first(config, ["training.label_smoothing", "downstream.label_smoothing"], None),
                "use_class_weights": _select_first(config, ["training.use_class_weights", "downstream.use_class_weights"], None),
            }
        elif stage_lower == "supervised":
            loss_section = {
                "loss_type": _select_first(config, ["training.loss_type"], None),
                "focal_gamma": _select_first(config, ["training.focal_gamma"], None),
                "label_smoothing": _select_first(config, ["training.label_smoothing"], None),
                "use_class_weights": _select_first(config, ["training.use_class_weights"], None),
            }
        else:
            loss_section = {}

    task_section = _compact_dict(effective_dict.get("task", {})) if isinstance(effective_dict.get("task"), dict) else {}
    if "name" not in task_section:
        task_section["name"] = stage_lower

    summary = {
        "task": task_section,
        "model": _compact_dict(effective_dict.get("model", {})) if isinstance(effective_dict.get("model"), dict) else {},
        "training": _compact_dict(effective_dict.get("training", {})) if isinstance(effective_dict.get("training"), dict) else {},
        "loss": _compact_dict(loss_section),
        "data_cache": _compact_dict(effective_dict.get("data_cache", {})) if isinstance(effective_dict.get("data_cache"), dict) else {},
    }

    return OmegaConf.create(summary)
