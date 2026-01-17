"""
PA-HCL 的训练器。

此包提供以下训练工具:
- PretrainTrainer: 自监督预训练
- DownstreamTrainer: 下游微调
"""

from .pretrain_trainer import (
    PretrainTrainer,
    launch_distributed_training,
)
from .downstream_trainer import (
    DownstreamTrainer,
    DownstreamModel,
    load_pretrained_encoder,
)

__all__ = [
    "PretrainTrainer",
    "launch_distributed_training",
    "DownstreamTrainer",
    "DownstreamModel",
    "load_pretrained_encoder",
]
