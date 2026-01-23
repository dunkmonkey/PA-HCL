"""
PA-HCL 的下游微调训练器。

此模块提供自监督预训练后用于下游分类任务的训练工具:
- 线性评估协议
- 全量微调
- 少样本学习

作者: PA-HCL Team
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# 实验监控工具（可选依赖）
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..models.pahcl import PAHCLModel
from ..models.heads import ClassificationHead
from ..utils.seed import set_seed
from ..utils.logging import setup_logger
from ..utils.metrics import compute_classification_metrics


class DownstreamModel(nn.Module):
    """
    下游分类模型。
    
    结合预训练编码器和分类头，
    用于心音分类等下游任务。
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        encoder_dim: int = 256,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ):
        """
        参数:
            encoder: 预训练编码器模块
            num_classes: 输出类别数量
            encoder_dim: 编码器输出维度
            hidden_dim: 隐藏层维度（None 表示线性分类器）
            dropout: Dropout 率
            freeze_encoder: 是否冻结编码器权重
        """
        super().__init__()
        
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        
        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = ClassificationHead(
            input_dim=encoder_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        参数:
            x: 输入信号 [B, 1, L] 或 [B, L]
            
        返回:
            Logits [B, num_classes]
        """
        # Get encoder features
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        
        # Handle PAHCLModel output
        if isinstance(features, dict):
            features = features.get("features", features.get("cycle_proj"))
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取不带分类的编码器特征。"""
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.encoder(x)
        else:
            features = self.encoder(x)
        
        if isinstance(features, dict):
            features = features.get("features", features.get("cycle_proj"))
        
        return features


class DownstreamTrainer:
    """
    下游分类任务的训练器。
    
    支持:
    - 线性评估（冻结编码器）
    - 全量微调
    - 小样本学习
    - 多任务支持
    """
    
    def __init__(
        self,
        model: DownstreamModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        config: Optional[Any] = None,
        # Training hyperparameters
        num_epochs: int = 50,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        min_lr: float = 1e-7,
        # Loss settings
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        # Training settings
        use_amp: bool = True,
        grad_clip_norm: float = 1.0,
        early_stopping_patience: int = 10,
        # Distributed training
        distributed: bool = False,
        local_rank: int = -1,
        # Logging and saving
        log_interval: int = 20,
        save_interval: int = 5,
        output_dir: str = "outputs",
        experiment_name: str = "downstream",
        # Reproducibility
        seed: int = 42,
        # Task-specific settings
        task_name: str = "default",
        primary_metric: str = "f1",
        # Experiment tracking
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: str = "PA-HCL",
        wandb_entity: Optional[str] = None,
    ):
        """
        初始化下游训练器。
        """
        self.config = config
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.label_smoothing = label_smoothing
        self.use_amp = use_amp
        self.grad_clip_norm = grad_clip_norm
        self.early_stopping_patience = early_stopping_patience
        self.distributed = distributed
        self.local_rank = local_rank
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.experiment_name = experiment_name
        self.seed = seed
        self.task_name = task_name
        self.primary_metric = primary_metric
        
        # Output directory
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger
        self.logger = setup_logger(
            name="DownstreamTrainer",
            log_file=self.output_dir / "train.log"
        )
        
        # 实验监控工具
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.tensorboard_writer = None
        
        # 初始化 TensorBoard
        if use_tensorboard and self.is_main_process:
            if TENSORBOARD_AVAILABLE:
                tb_dir = self.output_dir / "tensorboard"
                tb_dir.mkdir(parents=True, exist_ok=True)
                self.tensorboard_writer = SummaryWriter(log_dir=str(tb_dir))
                self.logger.info(f"TensorBoard 日志目录: {tb_dir}")
            else:
                self.logger.warning("TensorBoard 不可用。请安装: pip install tensorboard")
                self.use_tensorboard = False
        
        # 初始化 WandB
        if use_wandb and self.is_main_process:
            if WANDB_AVAILABLE:
                # 准备配置字典
                wandb_config = {
                    "task": task_name,
                    "experiment_name": experiment_name,
                    "num_epochs": num_epochs,
                    "learning_rate": learning_rate,
                    "batch_size": train_loader.batch_size,
                    "weight_decay": weight_decay,
                    "warmup_epochs": warmup_epochs,
                    "label_smoothing": label_smoothing,
                    "use_amp": use_amp,
                    "grad_clip_norm": grad_clip_norm,
                    "primary_metric": primary_metric,
                    "seed": seed,
                }
                
                # 添加完整配置
                if config is not None:
                    try:
                        from types import SimpleNamespace
                        def to_dict(obj):
                            if isinstance(obj, SimpleNamespace):
                                return {k: to_dict(v) for k, v in vars(obj).items()}
                            return obj
                        wandb_config["full_config"] = to_dict(config)
                    except Exception:
                        pass
                
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=experiment_name,
                    config=wandb_config,
                    dir=str(self.output_dir),
                    resume="allow"
                )
                
                # 监视模型
                wandb.watch(model, log="all", log_freq=100)
                self.logger.info(f"WandB 项目: {wandb_project}")
            else:
                self.logger.warning("WandB 不可用。请安装: pip install wandb")
                self.use_wandb = False
        
        # Device
        if distributed:
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Seed
        set_seed(seed)
        
        # Model
        self.model = model.to(self.device)
        if distributed:
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True  # Needed when encoder is frozen
            )
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None,
            label_smoothing=label_smoothing
        )
        
        # Optimizer (different lr for encoder and classifier if not frozen)
        self.optimizer = self._build_optimizer()
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=min_lr
        )
        
        # AMP scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = 0.0
        self.patience_counter = 0
        
        self.logger.info(f"Downstream trainer initialized. Device: {self.device}")
        self.logger.info(f"Task: {self.task_name}")
        self.logger.info(f"Primary metric: {self.primary_metric}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # 打印类别权重
        if class_weights is not None:
            self.logger.info(f"Class weights: {class_weights.tolist()}")
    
    @property
    def is_main_process(self) -> bool:
        return not self.distributed or self.local_rank == 0
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """构建具有分离学习率的优化器。"""
        # Get raw model
        model = self.model.module if isinstance(self.model, DDP) else self.model
        
        if model.freeze_encoder:
            # Only classifier parameters
            params = model.classifier.parameters()
            return AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            # Different learning rates for encoder and classifier
            encoder_params = list(model.encoder.parameters())
            classifier_params = list(model.classifier.parameters())
            
            param_groups = [
                {"params": encoder_params, "lr": self.learning_rate * 0.1},  # Lower lr for pretrained encoder
                {"params": classifier_params, "lr": self.learning_rate},
            ]
            
            return AdamW(param_groups, weight_decay=self.weight_decay)
    
    def train(self) -> Dict[str, float]:
        """运行完整的训练循环。"""
        self.logger.info("Starting downstream training...")
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Logging
            if self.is_main_process:
                self._log_epoch(train_metrics, val_metrics)
                
                # Early stopping check - use primary metric
                if val_metrics:
                    # 获取主要指标值
                    val_metric = self._get_primary_metric_value(val_metrics)
                    
                    if val_metric > self.best_val_metric:
                        self.best_val_metric = val_metric
                        self.patience_counter = 0
                        self.save_checkpoint("best_model.pt")
                        self.logger.info(f"★ 新的最佳模型! {self.primary_metric}: {val_metric:.4f}")
                    else:
                        self.patience_counter += 1
                    
                    if self.patience_counter >= self.early_stopping_patience:
                        self.logger.info(f"早停于 epoch {epoch+1}（{self.primary_metric} 无改善）")
                        break
                
                # Periodic checkpoint
                if (epoch + 1) % self.save_interval == 0:
                    self.save_checkpoint(f"epoch_{epoch+1}.pt")
        
        # Test evaluation
        test_metrics = {}
        if self.test_loader is not None:
            # Load best model
            best_path = self.output_dir / "best_model.pt"
            if best_path.exists():
                self.load_checkpoint(str(best_path))
            
            test_metrics = self.evaluate(self.test_loader)
            if self.is_main_process:
                self.logger.info(f"Test metrics: {test_metrics}")
        
        # 关闭监控工具
        if self.is_main_process:
            if self.use_tensorboard and self.tensorboard_writer is not None:
                self.tensorboard_writer.close()
                self.logger.info("TensorBoard writer 已关闭")
            
            if self.use_wandb:
                # 记录最终测试结果
                final_metrics = test_metrics if test_metrics else val_metrics
                wandb.log({f"final/{k}": v for k, v in final_metrics.items()})
                wandb.finish()
                self.logger.info("WandB 运行已完成")
        
        return test_metrics if test_metrics else val_metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch。"""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Get data
            signals = batch["signal"].to(self.device)
            labels = batch["label"].to(self.device)
            
            # Forward with AMP
            with autocast(enabled=self.use_amp):
                logits = self.model(signals)
                loss = self.criterion(logits, labels)
            
            # Backward
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.optimizer.step()
            
            self.global_step += 1
            
            # Accumulate
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Logging
            if self.is_main_process and (batch_idx + 1) % self.log_interval == 0:
                self.logger.info(
                    f"Epoch [{self.current_epoch+1}/{self.num_epochs}] "
                    f"Step [{batch_idx+1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Compute metrics
        metrics = compute_classification_metrics(all_labels, all_preds)
        metrics["train_loss"] = total_loss / len(self.train_loader)
        
        # 记录训练指标
        if self.is_main_process:
            # TensorBoard
            if self.use_tensorboard and self.tensorboard_writer is not None:
                for key, value in metrics.items():
                    self.tensorboard_writer.add_scalar(f"train/{key}", value, self.current_epoch)
                # 记录学习率
                for i, param_group in enumerate(self.optimizer.param_groups):
                    self.tensorboard_writer.add_scalar(f"lr/group_{i}", param_group['lr'], self.current_epoch)
            
            # WandB
            if self.use_wandb:
                wandb_metrics = {f"train/{k}": v for k, v in metrics.items()}
                wandb_metrics["epoch"] = self.current_epoch
                wandb_metrics["learning_rate"] = self.optimizer.param_groups[0]['lr']
                wandb.log(wandb_metrics, step=self.current_epoch)
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """运行验证。"""
        return self.evaluate(self.val_loader, prefix="val")
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        prefix: str = "test"
    ) -> Dict[str, float]:
        """在数据集上进行评估。"""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in data_loader:
            signals = batch["signal"].to(self.device)
            labels = batch["label"].to(self.device)
            
            with autocast(enabled=self.use_amp):
                logits = self.model(signals)
                loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
        
        # Compute metrics
        metrics = compute_classification_metrics(
            all_labels,
            all_preds,
            all_probs
        )
        metrics[f"{prefix}_loss"] = total_loss / len(data_loader)
        
        # Add prefix to all keys
        metrics = {f"{prefix}_{k}" if not k.startswith(prefix) else k: v for k, v in metrics.items()}
        
        # 记录验证/测试指标
        if self.is_main_process:
            # TensorBoard
            if self.use_tensorboard and self.tensorboard_writer is not None:
                for key, value in metrics.items():
                    # 去掉前缀以获取干净的指标名
                    clean_key = key.replace(f"{prefix}_", "")
                    self.tensorboard_writer.add_scalar(f"{prefix}/{clean_key}", value, self.current_epoch)
            
            # WandB
            if self.use_wandb:
                wandb_metrics = {k: v for k, v in metrics.items()}
                wandb_metrics["epoch"] = self.current_epoch
                wandb.log(wandb_metrics, step=self.current_epoch)
        
        return metrics
    
    def _log_epoch(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """记录 epoch 结果。"""
        # 基本训练指标
        log_str = (
            f"Epoch [{self.current_epoch+1}/{self.num_epochs}] "
            f"Train Loss: {train_metrics['train_loss']:.4f} "
            f"Acc: {train_metrics.get('accuracy', 0):.4f} "
            f"F1: {train_metrics.get('f1', 0):.4f}"
        )
        self.logger.info(log_str)
        
        if val_metrics:
            val_metric = self._get_primary_metric_value(val_metrics)
            self.logger.info(
                f"  Val Loss: {val_metrics.get('val_loss', 0):.4f} "
                f"Acc: {val_metrics.get('val_accuracy', 0):.4f} "
                f"F1: {val_metrics.get('val_f1', 0):.4f} "
                f"[{self.primary_metric}: {val_metric:.4f}]"
            )
    
    def _get_primary_metric_value(self, metrics: Dict[str, float]) -> float:
        """
        从指标字典中获取主要指标的值。
        
        支持的指标名称:
        - accuracy, acc
        - f1, f1_macro, f1_weighted
        - auroc, auc, roc_auc
        - auprc, ap, average_precision
        - precision, precision_macro
        - recall, recall_macro, sensitivity
        """
        metric = self.primary_metric.lower()
        
        # 尝试不同的键名变体
        possible_keys = [
            f"val_{metric}",
            metric,
            f"val_{metric.replace('_', '')}",
            metric.replace('_', ''),
        ]
        
        # 指标别名映射
        aliases = {
            'auroc': ['auc', 'roc_auc', 'auroc'],
            'f1_macro': ['f1', 'f1_macro'],
            'accuracy': ['acc', 'accuracy'],
            'auprc': ['ap', 'average_precision', 'auprc'],
        }
        
        for alias_key, alias_list in aliases.items():
            if metric in alias_list:
                for alias in alias_list:
                    possible_keys.extend([f"val_{alias}", alias])
        
        for key in possible_keys:
            if key in metrics:
                return metrics[key]
        
        # 默认回退到 val_f1 或 val_accuracy
        return metrics.get('val_f1', metrics.get('val_accuracy', 0.0))
    
    def save_checkpoint(self, filename: str):
        """保存检查点。"""
        checkpoint_path = self.output_dir / filename
        
        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, DDP)
            else self.model.state_dict()
        )
        
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_metric": self.best_val_metric,
            "task_name": self.task_name,
            "primary_metric": self.primary_metric,
        }
        
        # 保存配置（如果有）
        if self.config is not None:
            try:
                from types import SimpleNamespace
                def to_dict(obj):
                    if isinstance(obj, SimpleNamespace):
                        return {k: to_dict(v) for k, v in vars(obj).items()}
                    return obj
                checkpoint["config"] = to_dict(self.config)
            except Exception:
                pass
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点。"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        self.logger.info(f"Loaded checkpoint: {checkpoint_path}")


def load_pretrained_encoder(
    checkpoint_path: str,
    device: torch.device,
    config: Optional[Any] = None,
    logger: Optional[Any] = None
) -> nn.Module:
    """
    从 PA-HCL 检查点加载预训练编码器（下游微调专用）。
    
    智能处理 MoCo 相关参数：
    - 自动检测预训练模型是否使用 MoCo
    - 下游任务只加载主编码器权重（忽略动量编码器和队列）
    - 兼容 SimCLR 和 MoCo 预训练模型
    
    参数:
        checkpoint_path: 预训练检查点路径
        device: 目标设备
        config: 模型配置（如果为 None，则使用检查点配置）
        logger: 日志记录器（可选，用于详细输出）
        
    返回:
        加载的编码器模块（PAHCLModel，use_moco=False）
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取配置
    if config is None and "config" in checkpoint:
        config = checkpoint["config"]
    
    # 检测预训练模型是否使用了 MoCo
    state_dict = checkpoint["model_state_dict"]
    pretrain_used_moco = False
    
    # 优先使用元数据（建议A）
    if "moco_metadata" in checkpoint:
        pretrain_used_moco = checkpoint["moco_metadata"].get("use_moco", False)
        if logger:
            logger.info(f"从检查点元数据检测到 MoCo: {pretrain_used_moco}")
            if pretrain_used_moco:
                logger.info(f"  动量系数: {checkpoint['moco_metadata'].get('moco_momentum', 'N/A')}")
                logger.info(f"  队列大小: {checkpoint['moco_metadata'].get('queue_size', 'N/A')}")
    else:
        # 回退：通过检查 state_dict 键
        moco_keys = ['encoder_momentum', 'cycle_projector_momentum', 'queue', 'queue_ptr']
        for key in state_dict.keys():
            if any(moco_key in key for moco_key in moco_keys):
                pretrain_used_moco = True
                break
        if logger:
            logger.info(f"通过 state_dict 键检测到 MoCo: {pretrain_used_moco}")
    
    # 对于下游任务，始终创建非 MoCo 模型
    # 强制覆盖配置中的 use_moco 为 False
    if hasattr(config, 'model'):
        original_use_moco = getattr(config.model, 'use_moco', False)
        config.model.use_moco = False
        if logger and original_use_moco:
            logger.info("下游任务不需要 MoCo 动量编码器，已自动禁用")
    
    # 构建模型（不使用 MoCo）
    from ..models.pahcl import build_pahcl_model
    model = build_pahcl_model(config)
    
    # 加载权重
    if pretrain_used_moco:
        # 过滤 MoCo 相关参数
        filtered_state_dict = {}
        skipped_keys = []
        moco_keys = ['encoder_momentum', 'cycle_projector_momentum', 'queue', 'queue_ptr']
        
        for key, value in state_dict.items():
            # 跳过动量编码器和队列相关参数
            if any(moco_key in key for moco_key in moco_keys):
                skipped_keys.append(key)
            else:
                filtered_state_dict[key] = value
        
        # 加载过滤后的权重
        missing_keys, unexpected_keys = model.load_state_dict(
            filtered_state_dict, 
            strict=False
        )
        
        if logger:
            logger.info(f"已跳过 {len(skipped_keys)} 个 MoCo 相关参数（动量编码器/队列）")
            logger.info(f"成功加载 {len(filtered_state_dict)} 个主模型参数")
            if missing_keys:
                logger.warning(f"缺失的键 ({len(missing_keys)}): {missing_keys[:5]}...")
            if unexpected_keys:
                logger.warning(f"未预期的键 ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
        else:
            print(f"[INFO] 跳过 {len(skipped_keys)} 个 MoCo 参数，加载 {len(filtered_state_dict)} 个主模型参数")
    else:
        # 直接加载（SimCLR 模式）
        model.load_state_dict(state_dict, strict=True)
        if logger:
            logger.info("成功加载 SimCLR 预训练权重（严格模式）")
        else:
            print("[INFO] 成功加载 SimCLR 预训练权重")
    
    return model
