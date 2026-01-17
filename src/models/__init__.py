"""
PA-HCL 的模型架构。

此包提供：
- 编码器架构 (CNN, CNN-Mamba, CNN-Transformer)
- 用于对比学习的投影头
- 用于下游任务的分类头
"""

from .mamba import (
    RMSNorm,
    SelectiveSSM,
    MambaBlock,
    MambaEncoder,
    get_mamba_encoder,
)

from .encoder import (
    ConvBlock,
    ResidualConvBlock,
    CNNBackbone,
    CNNMambaEncoder,
    CNNTransformerEncoder,
    PositionalEncoding,
    build_encoder,
)

from .heads import (
    ProjectionHead,
    PredictionHead,
    ClassificationHead,
    AnomalyDetectionHead,
    SubstructureProjectionHead,
)

from .pahcl import (
    PAHCLModel,
    build_pahcl_model,
)

__all__ = [
    # Mamba
    "RMSNorm",
    "SelectiveSSM",
    "MambaBlock",
    "MambaEncoder",
    "get_mamba_encoder",
    # Encoder
    "ConvBlock",
    "ResidualConvBlock",
    "CNNBackbone",
    "CNNMambaEncoder",
    "CNNTransformerEncoder",
    "PositionalEncoding",
    "build_encoder",
    # Heads
    "ProjectionHead",
    "PredictionHead",
    "ClassificationHead",
    "AnomalyDetectionHead",
    "SubstructureProjectionHead",
    # PA-HCL
    "PAHCLModel",
    "build_pahcl_model",
]
