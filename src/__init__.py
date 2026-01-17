"""
PA-HCL: 用于心音的生理感知分层对比学习

一个自监督对比学习框架，利用心音的固有生理结构（心动周期和子结构）
来学习可迁移到下游临床任务的表示。

作者: PA-HCL 团队
许可证: MIT
"""

__version__ = "0.1.0"
__author__ = "PA-HCL Team"

from .config import Config, load_config
