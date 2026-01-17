"""
PA-HCL 的日志实用程序。
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from rich.console import Console
from rich.logging import RichHandler


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    use_rich: bool = True
) -> logging.Logger:
    """
    获取配置好的日志记录器实例。
    
    参数:
        name: 日志记录器名称
        level: 日志级别
        log_file: 可选的日志文件路径
        use_rich: 是否对控制台输出使用 rich 格式
        
    返回:
        配置好的日志记录器实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复的处理器
    if logger.handlers:
        return logger
    
    # 控制台处理器
    if use_rich:
        console_handler = RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=False,
            rich_tracebacks=True
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
    console_handler.setLevel(level)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
    
    return logger


def setup_logging(
    log_dir: Union[str, Path],
    experiment_name: str,
    level: int = logging.INFO
) -> Path:
    """
    设置实验的日志记录。
    
    参数:
        log_dir: 日志的基础目录
        experiment_name: 实验名称
        level: 日志级别
        
    返回:
        日志文件的路径
    """
    log_dir = Path(log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / experiment_name / f"{timestamp}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 设置根日志记录器
    get_logger("pa_hcl", level=level, log_file=log_file)
    
    return log_file


class AverageMeter:
    """
    计算并存储平均值和当前值。
    
    用于跟踪训练指标。
    """
    
    def __init__(self, name: str = "metric"):
        self.name = name
        self.reset()
    
    def reset(self) -> None:
        """重置所有统计信息。"""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1) -> None:
        """
        用新值更新计量器。
        
        参数:
            val: 新值
            n: 样本数量
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"
