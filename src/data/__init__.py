"""
PA-HCL 的数据处理模块。

此包提供:
- 信号预处理工具（滤波、归一化、周期检测）
- 数据增强变换（时域/频域）
- 预训练和下游任务的数据集类
"""

# 核心预处理函数（无 torch 依赖）
from .preprocessing import (
    # 音频加载
    load_audio,
    resample_signal,
    # 信号滤波
    bandpass_filter,
    normalize_signal,
    # 包络和峰值检测
    compute_energy_envelope,
    detect_peaks,
    detect_peaks_adaptive,
    # 周期提取
    extract_cycles,
    split_substructures,
    split_substructures_with_overlap,
    # 质量评估
    compute_snr,
    assess_cycle_quality,
    # 完整管道
    process_recording,
)

# Transform functions (no torch dependency)
from .transforms import (
    # Base classes
    Transform,
    Compose,
    RandomApply,
    RandomChoice,
    # Time domain augmentations
    TimeShift,
    AmplitudeScale,
    GaussianNoise,
    TimeStretch,
    RandomCrop,
    Reverse,
    # Frequency domain augmentations
    FrequencyMask,
    TimeMask,
    LowPassFilter,
    HighPassFilter,
    # Advanced augmentations
    PitchShift,
    AddColoredNoise,
    SimulateRespiratoryNoise,
    # Factory functions
    get_pretrain_transforms,
    get_finetune_transforms,
    get_eval_transforms,
    # Substructure augmentor
    SubstructureAugmentor,
)

# Lazy imports for torch-dependent modules
def __getattr__(name):
    """依赖 Torch 的类的延迟导入。"""
    torch_dependent = {
        'PCGPretrainDataset',
        'PCGDownstreamDataset',
        'pretrain_collate_fn',
        'downstream_collate_fn',
        'create_subject_wise_split',
        'create_dataloaders',
    }
    
    if name in torch_dependent:
        from . import dataset
        return getattr(dataset, name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
