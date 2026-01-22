"""
PA-HCL 预训练的训练器。

此模块提供 PretrainTrainer 类，用于使用分层对比学习
对 PA-HCL 模型进行自监督预训练。

特性:
- 多 GPU 支持 (DDP, FSDP)
- 混合精度训练 (AMP)
- 梯度累积
- 学习率调度
- 检查点和恢复
- 日志记录 (TensorBoard, WandB)

作者: PA-HCL Team
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # Step 5: for normalize
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    OneCycleLR,
)

from ..models.pahcl import PAHCLModel, build_pahcl_model
from ..losses.contrastive import HierarchicalContrastiveLoss
from ..utils.seed import set_seed
from ..utils.logging import setup_logger


class PretrainTrainer:
    """
    PA-HCL 自监督预训练的训练器。
    
    此训练器处理:
    - 模型初始化和分发
    - 优化器和调度器设置
    - 混合精度训练循环
    - 验证和指标日志记录
    - 检查点和模型保存
    """
    
    def __init__(
        self,
        model: PAHCLModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Any] = None,
        # Training hyperparameters
        num_epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 10,
        min_lr: float = 1e-6,
        # Loss settings
        temperature: float = 0.07,
        lambda_cycle: float = 1.0,
        lambda_sub: float = 1.0,
        # Training settings
        use_amp: bool = True,
        gradient_accumulation_steps: int = 1,
        grad_clip_norm: float = 1.0,
        # Distributed training
        distributed: bool = False,
        local_rank: int = -1,
        # Logging and saving
        log_interval: int = 50,
        save_interval: int = 10,
        output_dir: str = "outputs",
        experiment_name: str = "pahcl_pretrain",
        use_wandb: bool = False,
        wandb_project: str = "PA-HCL",
        # Reproducibility
        seed: int = 42,
    ):
        """
        初始化训练器。
        
        参数:
            model: 要训练的 PA-HCL 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器 (可选)
            config: 配置对象
            num_epochs: 训练轮数
            learning_rate: 峰值学习率
            weight_decay: 优化器的权重衰减
            warmup_epochs: 热身轮数
            min_lr: 最小学习率
            temperature: 对比损失温度
            lambda_cycle: 周期级损失的权重
            lambda_sub: 子结构级损失的权重
            use_amp: 是否使用自动混合精度
            gradient_accumulation_steps: 梯度累积步数
            grad_clip_norm: 梯度裁剪的最大范数
            distributed: 是否使用分布式训练
            local_rank: 分布式训练的本地 rank
            log_interval: 每 N 步记录一次日志
            save_interval: 每 N 轮保存一次检查点
            output_dir: 输出目录
            experiment_name: 此实验的名称
            use_wandb: 是否使用 Weights & Biases 记录日志
            wandb_project: W&B 项目名称
            seed: 随机种子
        """
        self.config = config
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.temperature = temperature
        self.lambda_cycle = lambda_cycle
        self.lambda_sub = lambda_sub
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.grad_clip_norm = grad_clip_norm
        self.distributed = distributed
        self.local_rank = local_rank
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.seed = seed
        
        # Set up output directory
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.logger = setup_logger(
            name="PretrainTrainer",
            log_file=self.output_dir / "train.log"
        )
        
        # Set up TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.output_dir)) if self.is_main_process else None
        
        # Device setup
        if distributed:
            self.device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set seed
        set_seed(seed)
        
        # Model setup
        self.model = model.to(self.device)
        if distributed:
            self.model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False
            )
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Loss function (Step 4 & 5: 支持 MoCo)
        use_moco = getattr(model, 'use_moco', False) if hasattr(model, 'use_moco') else False
        self.criterion = HierarchicalContrastiveLoss(
            temperature=temperature,
            lambda_cycle=lambda_cycle,
            lambda_sub=lambda_sub,
            use_moco=use_moco  # Step 4: 传递 MoCo 标志
        )
        
        # Optimizer
        self.optimizer = self._build_optimizer()
        
        # Scheduler
        self.scheduler = self._build_scheduler()
        
        # AMP scaler
        self.scaler = GradScaler() if use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Initialize W&B if requested
        if use_wandb and self.is_main_process:
            self._init_wandb()
        
        self.logger.info(f"Trainer initialized. Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    @property
    def is_main_process(self) -> bool:
        """检查此进程是否为主进程（用于日志记录）。"""
        return not self.distributed or self.local_rank == 0
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """构建带有参数组的优化器。"""
        # Separate parameters with and without weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "bn" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        optimizer = AdamW(
            param_groups,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def _build_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """构建带有热身的学习率调度器。"""
        steps_per_epoch = len(self.train_loader) // self.gradient_accumulation_steps
        total_steps = self.num_epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.min_lr
        )
        
        # Combined scheduler
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        return scheduler
    
    def _init_wandb(self):
        """初始化 Weights & Biases 日志记录。"""
        try:
            import wandb
            
            # 准备完整的配置字典
            wandb_config = {
                # 训练参数
                "num_epochs": self.num_epochs,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "warmup_epochs": self.warmup_epochs,
                "min_lr": self.min_lr,
                "batch_size": self.train_loader.batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "grad_clip_norm": self.grad_clip_norm,
                # 损失参数
                "temperature": self.temperature,
                "lambda_cycle": self.lambda_cycle,
                "lambda_sub": self.lambda_sub,
                # 其他设置
                "use_amp": self.use_amp,
                "distributed": self.distributed,
                "seed": self.seed,
            }
            
            # 如果有config对象，添加模型配置
            if self.config is not None:
                if hasattr(self.config, 'model'):
                    wandb_config.update({
                        "encoder_dim": getattr(self.config.model, 'encoder_dim', None),
                        "mamba_d_model": getattr(self.config.model, 'd_model', None),
                        "mamba_n_layers": getattr(self.config.model, 'n_layers', None),
                        "projection_dim": getattr(self.config.model, 'projection_dim', None),
                        "use_moco": getattr(self.config.model, 'use_moco', False),
                    })
            
            wandb.init(
                project=self.wandb_project,
                name=self.experiment_name,
                config=wandb_config,
                # 设置同步tensorboard
                sync_tensorboard=True,
            )
            
            # 记录模型架构
            if hasattr(self.model, 'module'):
                wandb.watch(self.model.module, log="all", log_freq=100)
            else:
                wandb.watch(self.model, log="all", log_freq=100)
                
        except ImportError:
            self.logger.warning("wandb not installed, disabling W&B logging")
            self.use_wandb = False
        except Exception as e:
            self.logger.warning(f"Failed to initialize wandb: {e}")
            self.use_wandb = False
    
    def train(self) -> Dict[str, float]:
        """
        运行完整的训练循环。
        
        返回:
            包含最终训练指标的字典
        """
        self.logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.validate()
            
            # Logging
            if self.is_main_process:
                self._log_epoch(train_metrics, val_metrics)
                
                # Save checkpoint
                if (epoch + 1) % self.save_interval == 0:
                    self.save_checkpoint(f"epoch_{epoch+1}.pt")
                
                # Save best model
                if val_metrics and val_metrics.get("val_loss", float('inf')) < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best_model.pt")
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
        
        # Save final model
        if self.is_main_process:
            self.save_checkpoint("final_model.pt")
        
        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()
        
        return train_metrics
    
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个 epoch。
        
        返回:
            包含训练指标的字典
        """
        self.model.train()
        
        total_loss = 0.0
        total_loss_cycle = 0.0
        total_loss_sub = 0.0
        num_batches = 0
        
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            view1 = batch["view1"].to(self.device)
            view2 = batch["view2"].to(self.device)
            subs1 = batch.get("subs1")
            subs2 = batch.get("subs2")
            
            if subs1 is not None:
                subs1 = subs1.to(self.device)
            if subs2 is not None:
                subs2 = subs2.to(self.device)
            
            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                # Get model outputs
                if isinstance(self.model, DDP):
                    outputs = self.model.module.forward_pretrain(
                        view1, view2, subs1, subs2
                    )
                    model_module = self.model.module
                else:
                    outputs = self.model.forward_pretrain(
                        view1, view2, subs1, subs2
                    )
                    model_module = self.model
                
                # Step 5: 获取队列（如果使用 MoCo）
                queue = None
                if hasattr(model_module, 'use_moco') and model_module.use_moco:
                    queue = model_module.queue.clone().detach()
                
                # Compute loss
                loss, loss_dict = self.criterion(
                    outputs["cycle_proj1"],
                    outputs["cycle_proj2"],
                    outputs["sub_proj1"],
                    outputs["sub_proj2"],
                    queue=queue  # Step 5: 传递队列
                )
                
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm
                )
                
                # Optimizer step (must be before scheduler.step())
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Scheduler step (after optimizer.step())
                self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Step 5: MoCo 动量更新和队列管理
                if hasattr(model_module, 'use_moco') and model_module.use_moco:
                    # 动量更新编码器
                    model_module._momentum_update()
                    
                    # 队列更新：使用 cycle_proj2 (来自动量编码器)
                    # 需要归一化并入队
                    with torch.no_grad():
                        keys = F.normalize(outputs["cycle_proj2"], dim=1)
                        model_module._dequeue_and_enqueue(keys)
            
            # Accumulate metrics
            total_loss += loss_dict["loss_total"]
            total_loss_cycle += loss_dict["loss_cycle"]
            total_loss_sub += loss_dict["loss_sub"]
            num_batches += 1
            
            # Logging
            if self.is_main_process and (batch_idx + 1) % self.log_interval == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    f"Epoch [{self.current_epoch+1}/{self.num_epochs}] "
                    f"Step [{batch_idx+1}/{len(self.train_loader)}] "
                    f"Loss: {loss_dict['loss_total']:.4f} "
                    f"(cycle: {loss_dict['loss_cycle']:.4f}, sub: {loss_dict['loss_sub']:.4f}) "
                    f"LR: {lr:.2e}"
                )
        
        epoch_time = time.time() - epoch_start
        
        # Average metrics
        metrics = {
            "train_loss": total_loss / num_batches,
            "train_loss_cycle": total_loss_cycle / num_batches,
            "train_loss_sub": total_loss_sub / num_batches,
            "epoch_time": epoch_time,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        运行验证。
        
        返回:
            包含验证指标的字典
        """
        self.model.eval()
        
        total_loss = 0.0
        total_loss_cycle = 0.0
        total_loss_sub = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            view1 = batch["view1"].to(self.device)
            view2 = batch["view2"].to(self.device)
            subs1 = batch.get("subs1")
            subs2 = batch.get("subs2")
            
            if subs1 is not None:
                subs1 = subs1.to(self.device)
            if subs2 is not None:
                subs2 = subs2.to(self.device)
            
            with autocast(enabled=self.use_amp):
                if isinstance(self.model, DDP):
                    outputs = self.model.module.forward_pretrain(
                        view1, view2, subs1, subs2
                    )
                    model_module = self.model.module
                else:
                    outputs = self.model.forward_pretrain(
                        view1, view2, subs1, subs2
                    )
                    model_module = self.model
                
                # Step 5: 获取队列（验证时也需要）
                queue = None
                if hasattr(model_module, 'use_moco') and model_module.use_moco:
                    queue = model_module.queue.clone().detach()
                
                _, loss_dict = self.criterion(
                    outputs["cycle_proj1"],
                    outputs["cycle_proj2"],
                    outputs["sub_proj1"],
                    outputs["sub_proj2"],
                    queue=queue  # Step 5
                )
            
            total_loss += loss_dict["loss_total"]
            total_loss_cycle += loss_dict["loss_cycle"]
            total_loss_sub += loss_dict["loss_sub"]
            num_batches += 1
        
        metrics = {
            "val_loss": total_loss / num_batches,
            "val_loss_cycle": total_loss_cycle / num_batches,
            "val_loss_sub": total_loss_sub / num_batches,
        }
        
        return metrics
    
    def _log_epoch(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ):
        """记录 epoch 指标。"""
        # Console logging
        self.logger.info(
            f"Epoch [{self.current_epoch+1}/{self.num_epochs}] completed - "
            f"Train Loss: {train_metrics['train_loss']:.4f} "
            f"(cycle: {train_metrics['train_loss_cycle']:.4f}, "
            f"sub: {train_metrics['train_loss_sub']:.4f})"
        )
        
        if val_metrics:
            self.logger.info(
                f"Validation Loss: {val_metrics['val_loss']:.4f} "
                f"(cycle: {val_metrics['val_loss_cycle']:.4f}, "
                f"sub: {val_metrics['val_loss_sub']:.4f})"
            )
        
        # TensorBoard logging
        if self.writer is not None:
            # 使用 global_step 作为 x 轴以获得更好的可视化效果
            for k, v in train_metrics.items():
                if k != 'epoch_time':  # 跳过非损失指标
                    self.writer.add_scalar(f"train/{k}", v, self.global_step)
            for k, v in val_metrics.items():
                self.writer.add_scalar(f"val/{k}", v, self.global_step)
            # 记录学习率和 epoch
            self.writer.add_scalar("lr", self.optimizer.param_groups[0]['lr'], self.global_step)
            self.writer.add_scalar("epoch", self.current_epoch + 1, self.global_step)
            # 确保数据写入磁盘
            self.writer.flush()
        
        # W&B logging
        if self.use_wandb:
            try:
                import wandb
                
                log_dict = {
                    "epoch": self.current_epoch + 1,
                    "global_step": self.global_step,
                }
                
                # 添加训练指标
                for k, v in train_metrics.items():
                    log_dict[f"train/{k}"] = v
                
                # 添加验证指标
                for k, v in val_metrics.items():
                    log_dict[f"val/{k}"] = v
                
                # 记录学习率
                log_dict["learning_rate"] = self.optimizer.param_groups[0]['lr']
                
                # 如果有最佳验证损失，也记录
                if val_metrics:
                    log_dict["best_val_loss"] = self.best_val_loss
                
                wandb.log(log_dict, step=self.global_step)
            except Exception as e:
                self.logger.warning(f"W&B logging failed: {e}")
    
    def save_checkpoint(self, filename: str):
        """
        保存训练检查点。
        
        参数:
            filename: 检查点文件名
        """
        checkpoint_path = self.output_dir / filename
        
        # Get model state dict (handle DDP)
        if isinstance(self.model, DDP):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载训练检查点。
        
        参数:
            checkpoint_path: 检查点文件路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load training state
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        # Load scaler
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}")


def launch_distributed_training(
    rank: int,
    world_size: int,
    config,
    port: int = 12355
):
    """
    启动分布式训练。
    
    参数:
        rank: 进程 rank
        world_size: 进程总数
        config: 训练配置
        port: 通信端口
    """
    # Initialize process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    
    # Build model and data
    from ..data.dataset import PCGPretrainDataset, create_dataloaders
    
    model = build_pahcl_model(config)
    
    train_dataset = PCGPretrainDataset(
        data_dir=config.data.processed_dir,
        split="train"
    )
    
    train_loader = create_dataloaders(
        train_dataset,
        batch_size=config.training.batch_size // world_size,
        num_workers=config.training.num_workers,
        distributed=True
    )
    
    # Create trainer
    trainer = PretrainTrainer(
        model=model,
        train_loader=train_loader,
        config=config,
        distributed=True,
        local_rank=rank,
        **config.training
    )
    
    # Train
    trainer.train()
    
    # Cleanup
    dist.destroy_process_group()
