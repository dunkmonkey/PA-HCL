"""
用于重现性的随机种子实用程序。
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    在这所有库中设置随机种子以实现重现性。
    
    参数:
        seed: 随机种子值
        deterministic: 如果为 True，使用确定性算法（可能较慢）
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 对于 PyTorch >= 1.8
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def get_seed_worker(worker_id: int) -> None:
    """
    DataLoader 的工作进程初始化函数，以确保证可重现性。
    
    用法:
        DataLoader(..., worker_init_fn=get_seed_worker)
    
    参数:
        worker_id: 工作进程 ID
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    获取带有可选种子的 PyTorch 生成器。
    
    参数:
        seed: 生成器的可选种子
        
    返回:
        torch.Generator 实例
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    return g
