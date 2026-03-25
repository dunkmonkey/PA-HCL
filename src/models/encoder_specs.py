"""编码器结构预设与日志工具。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


# 编码器结构参数统一硬编码在此处，避免多配置文件漂移。
_ENCODER_ARCH_SPECS: Dict[str, Dict[str, Any]] = {
    "cnn_mamba": {
        "cnn_channels": [32, 64, 128, 256],
        "cnn_kernel_sizes": [7, 5, 5, 3],
        "cnn_strides": [2, 2, 2, 2],
        "cnn_dropout": 0.1,
        "mamba_d_model": 256,
        "mamba_n_layers": 4,
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
        "mamba_dropout": 0.1,
        "pool_type": "mean",
        "drop_path_rate": 0.1,
        "attention_type": "eca",
        "use_bidirectional": False,
        "bidirectional_fusion": "add",
        "use_multiscale": False,
        "multiscale_kernel_sizes": [3, 7, 15],
    },
    "sincnet_eca_mamba": {
        "sinc_out_channels": 64,
        "sinc_kernel_size": 251,
        "sinc_stride": 1,
        "sinc_min_low_hz": 20.0,
        "sinc_min_band_hz": 20.0,
        "sinc_max_high_hz": 500.0,
        "local_dim": 128,
        "convnext_kernel_size": 7,
        "convnext_expansion": 4,
        "mamba_d_model": 128,
        "mamba_n_layers": 6,
        "mamba_d_state": 16,
        "mamba_d_conv": 4,
        "mamba_expand": 2,
        "mamba_dropout": 0.1,
        "drop_path_rate": 0.1,
        "cycle_output_dim": 256,
        "pool_type": "asp",
        "use_bidirectional": True,
        "bidirectional_fusion": "add",
        "use_groupnorm": True,
        "num_groups": 8,
    },
    "resnet34_1d": {
        "dropout": 0.0,
    },
}


def get_encoder_arch_spec(encoder_type: str) -> Dict[str, Any]:
    """返回编码器结构预设（深拷贝）。"""
    if encoder_type not in _ENCODER_ARCH_SPECS:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    return deepcopy(_ENCODER_ARCH_SPECS[encoder_type])


def get_encoder_build_kwargs(
    encoder_type: str,
    *,
    sample_rate: int = 5000,
    num_substructures: int = 4,
) -> Dict[str, Any]:
    """返回可直接传给 build_encoder 的参数。"""
    kwargs = get_encoder_arch_spec(encoder_type)
    kwargs["in_channels"] = 1

    if encoder_type == "cnn_mamba":
        # CNNMambaEncoder 当前不接收 mamba_d_conv 参数。
        kwargs.pop("mamba_d_conv", None)

    if encoder_type == "sincnet_eca_mamba":
        kwargs["sample_rate"] = sample_rate
        kwargs["num_substructures"] = num_substructures

    return kwargs


def format_encoder_arch_log(encoder_type: str, arch: Dict[str, Any] | None = None) -> str:
    """生成统一的编码器结构日志文本。"""
    arch_dict = get_encoder_arch_spec(encoder_type) if arch is None else arch
    lines = [f"encoder_type: {encoder_type}"]
    for key in sorted(arch_dict.keys()):
        lines.append(f"  - {key}: {arch_dict[key]}")
    return "\n".join(lines)
