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
        
        # Output directory
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger
        self.logger = setup_logger(
            name="DownstreamTrainer",
            log_file=self.output_dir / "train.log"
        )
        
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
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
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
                
                # Early stopping check
                if val_metrics:
                    val_f1 = val_metrics.get("val_f1", val_metrics.get("val_accuracy", 0))
                    if val_f1 > self.best_val_metric:
                        self.best_val_metric = val_f1
                        self.patience_counter = 0
                        self.save_checkpoint("best_model.pt")
                    else:
                        self.patience_counter += 1
                    
                    if self.patience_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
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
        
        return metrics
    
    def _log_epoch(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """记录 epoch 结果。"""
        self.logger.info(
            f"Epoch [{self.current_epoch+1}/{self.num_epochs}] "
            f"Train Loss: {train_metrics['train_loss']:.4f} "
            f"Acc: {train_metrics.get('accuracy', 0):.4f} "
            f"F1: {train_metrics.get('f1', 0):.4f}"
        )
        
        if val_metrics:
            self.logger.info(
                f"Val Loss: {val_metrics.get('val_loss', 0):.4f} "
                f"Acc: {val_metrics.get('val_accuracy', 0):.4f} "
                f"F1: {val_metrics.get('val_f1', 0):.4f}"
            )
    
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
        }
        
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
    config: Optional[Any] = None
) -> nn.Module:
    """
    从 PA-HCL 检查点加载预训练编码器。
    
    参数:
        checkpoint_path: 预训练检查点路径
        device: 目标设备
        config: 模型配置（如果为 None，则使用检查点配置）
        
    返回:
        加载的编码器模块
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Build model
    from ..models.pahcl import build_pahcl_model
    
    if config is None and "config" in checkpoint:
        config = checkpoint["config"]
    
    model = build_pahcl_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model
