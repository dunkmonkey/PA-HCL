#!/usr/bin/env python
"""
PA-HCL 消融研究脚本。

此脚本运行消融实验以评估 PA-HCL 中不同组件的贡献。

消融研究：
1. 编码器变体 (仅 CNN, CNN-Transformer, CNN-Mamba)
2. 损失组件 (仅周期, 仅子结构, 分层)
3. 子结构数量 (K=2, 4, 8)
4. 温度参数
5. 数据增强策略

用法：
    # 运行所有消融实验
    python scripts/ablation.py --config configs/ablation.yaml --all
    
    # 运行特定消融
    python scripts/ablation.py --config configs/ablation.yaml --encoder-ablation
    
    # 使用特定 GPU 运行
    CUDA_VISIBLE_DEVICES=0 python scripts/ablation.py --config configs/ablation.yaml --loss-ablation

作者: PA-HCL 团队
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from copy import deepcopy

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch

from src.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logger


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="PA-HCL 消融研究",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ablation.yaml",
        help="消融配置的路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ablation",
        help="消融结果的输出目录"
    )
    
    # 消融选择
    parser.add_argument(
        "--all",
        action="store_true",
        help="运行所有消融实验"
    )
    parser.add_argument(
        "--encoder-ablation",
        action="store_true",
        help="运行编码器架构消融"
    )
    parser.add_argument(
        "--loss-ablation",
        action="store_true",
        help="运行损失组件消融"
    )
    parser.add_argument(
        "--substructure-ablation",
        action="store_true",
        help="运行子结构数量消融"
    )
    parser.add_argument(
        "--augmentation-ablation",
        action="store_true",
        help="运行数据增强消融"
    )
    parser.add_argument(
        "--temperature-ablation",
        action="store_true",
        help="运行温度消融"
    )
    
    # 训练设置
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=100,
        help="每次消融的预训练轮数"
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=50,
        help="每次消融的微调轮数"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--linear-eval",
        action="store_true",
        help="使用线性评估 (默认: 全量微调)"
    )
    
    return parser.parse_args()


class AblationRunner:
    """消融实验运行器。"""
    
    def __init__(self, base_config, output_dir: Path, logger, args):
        self.base_config = base_config
        self.output_dir = output_dir
        self.logger = logger
        self.args = args
        self.results = {}
    
    def run_single_experiment(
        self,
        exp_name: str,
        config_overrides: dict
    ) -> dict:
        """运行单个消融实验。"""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Running experiment: {exp_name}")
        self.logger.info(f"{'='*60}")
        
        # 创建实验配置
        exp_config = deepcopy(self.base_config)
        self._apply_overrides(exp_config, config_overrides)
        
        exp_dir = self.output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        config_path = exp_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self._config_to_dict(exp_config), f, indent=2)
        
        results = {
            "experiment": exp_name,
            "config_overrides": config_overrides,
            "status": "pending"
        }
        
        try:
            # 阶段 1: 预训练
            self.logger.info("Phase 1: Pretraining...")
            pretrain_results = self._run_pretraining(exp_config, exp_dir)
            results["pretrain"] = pretrain_results
            
            # 阶段 2: 下游评估
            self.logger.info("Phase 2: Downstream evaluation...")
            downstream_results = self._run_downstream(
                exp_config,
                exp_dir,
                pretrain_results.get("checkpoint_path")
            )
            results["downstream"] = downstream_results
            
            results["status"] = "completed"
            
        except Exception as e:
            self.logger.error(f"Experiment {exp_name} failed: {e}")
            results["status"] = "failed"
            results["error"] = str(e)
        
        # 保存结果
        results_path = exp_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _apply_overrides(self, config, overrides: dict):
        """递归应用配置覆盖。"""
        for key, value in overrides.items():
            if "." in key:
                # 嵌套键
                parts = key.split(".")
                obj = config
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        setattr(obj, part, type('', (), {})())
                        obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(config, key, value)
    
    def _config_to_dict(self, config) -> dict:
        """将配置对象转换为字典。"""
        if hasattr(config, "__dict__"):
            return {k: self._config_to_dict(v) for k, v in config.__dict__.items()
                   if not k.startswith("_")}
        elif isinstance(config, (list, tuple)):
            return [self._config_to_dict(v) for v in config]
        else:
            return config
    
    def _run_pretraining(self, config, exp_dir: Path) -> dict:
        """运行预训练阶段。"""
        from src.data.dataset import build_pretrain_dataloader
        from src.models.pahcl import build_pahcl_model
        from src.losses.contrastive import HierarchicalContrastiveLoss
        from src.trainers.pretrain_trainer import PretrainTrainer
        
        # 构建组件
        model = build_pahcl_model(config)
        
        loss_fn = HierarchicalContrastiveLoss(
            temperature=getattr(config.loss, 'temperature', 0.07),
            lambda_cycle=getattr(config.loss, 'lambda_cycle', 1.0),
            lambda_sub=getattr(config.loss, 'lambda_sub', 0.5)
        )
        
        # 尝试构建数据加载器 (如果数据不可用可能会失败)
        try:
            train_loader = build_pretrain_dataloader(config)
        except Exception as e:
            self.logger.warning(f"Could not build dataloader: {e}")
            self.logger.info("Simulating pretraining for ablation structure...")
            
            # 返回用于结构验证的模拟结果
            return {
                "epochs": self.args.pretrain_epochs,
                "final_loss": 0.0,
                "checkpoint_path": None,
                "simulated": True
            }
        
        # 创建训练器
        trainer = PretrainTrainer(
            model=model,
            loss_fn=loss_fn,
            train_loader=train_loader,
            num_epochs=self.args.pretrain_epochs,
            output_dir=str(exp_dir),
            experiment_name="pretrain"
        )
        
        # 训练
        final_loss = trainer.train()
        
        return {
            "epochs": self.args.pretrain_epochs,
            "final_loss": final_loss,
            "checkpoint_path": str(exp_dir / "pretrain" / "best_model.pt")
        }
    
    def _run_downstream(
        self,
        config,
        exp_dir: Path,
        pretrain_checkpoint: str = None
    ) -> dict:
        """运行下游评估阶段。"""
        from src.trainers.downstream_trainer import (
            DownstreamModel,
            DownstreamTrainer,
            load_pretrained_encoder
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载预训练编码器
        if pretrain_checkpoint and Path(pretrain_checkpoint).exists():
            encoder = load_pretrained_encoder(pretrain_checkpoint, device, config)
        else:
            self.logger.warning("No pretrain checkpoint, using random initialization")
            from src.models.pahcl import build_pahcl_model
            encoder = build_pahcl_model(config)
        
        # 创建下游模型
        model = DownstreamModel(
            encoder=encoder,
            num_classes=getattr(config.downstream, 'num_classes', 5),
            encoder_dim=getattr(config.model, 'mamba_d_model', 256),
            freeze_encoder=self.args.linear_eval
        )
        
        # 尝试构建数据加载器
        try:
            from src.data.dataset import build_downstream_dataloaders
            train_loader, val_loader, test_loader = build_downstream_dataloaders(config)
        except Exception as e:
            self.logger.warning(f"Could not build dataloaders: {e}")
            self.logger.info("Simulating downstream for ablation structure...")
            
            return {
                "epochs": self.args.finetune_epochs,
                "test_accuracy": 0.0,
                "test_f1": 0.0,
                "simulated": True
            }
        
        # 创建训练器
        trainer = DownstreamTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_epochs=self.args.finetune_epochs,
            output_dir=str(exp_dir),
            experiment_name="downstream"
        )
        
        # 训练和评估
        metrics = trainer.train()
        
        return {
            "epochs": self.args.finetune_epochs,
            "test_accuracy": metrics.get("test_accuracy", 0),
            "test_f1": metrics.get("test_f1", 0),
            "test_auroc": metrics.get("test_auroc", 0)
        }
    
    def run_encoder_ablation(self):
        """消融研究：编码器架构。"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("ENCODER ARCHITECTURE ABLATION")
        self.logger.info("=" * 60)
        
        encoder_variants = [
            ("cnn_only", {"model.encoder_type": "cnn_only"}),
            ("cnn_transformer", {"model.encoder_type": "cnn_transformer"}),
            ("cnn_mamba", {"model.encoder_type": "cnn_mamba"}),  # 完整 PA-HCL
        ]
        
        for name, overrides in encoder_variants:
            results = self.run_single_experiment(f"encoder_{name}", overrides)
            self.results[f"encoder_{name}"] = results
    
    def run_loss_ablation(self):
        """消融研究：损失组件。"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("LOSS COMPONENT ABLATION")
        self.logger.info("=" * 60)
        
        loss_variants = [
            ("cycle_only", {"loss.lambda_cycle": 1.0, "loss.lambda_sub": 0.0}),
            ("sub_only", {"loss.lambda_cycle": 0.0, "loss.lambda_sub": 1.0}),
            ("hierarchical_equal", {"loss.lambda_cycle": 1.0, "loss.lambda_sub": 1.0}),
            ("hierarchical_default", {"loss.lambda_cycle": 1.0, "loss.lambda_sub": 0.5}),
        ]
        
        for name, overrides in loss_variants:
            results = self.run_single_experiment(f"loss_{name}", overrides)
            self.results[f"loss_{name}"] = results
    
    def run_substructure_ablation(self):
        """消融研究：子结构数量。"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("SUBSTRUCTURE COUNT ABLATION")
        self.logger.info("=" * 60)
        
        sub_variants = [
            ("k2", {"data.num_substructures": 2}),   # S1-S2, 收缩期-舒张期
            ("k4", {"data.num_substructures": 4}),   # 默认: S1, 收缩期, S2, 舒张期
            ("k8", {"data.num_substructures": 8}),   # 更精细的分割
        ]
        
        for name, overrides in sub_variants:
            results = self.run_single_experiment(f"substructure_{name}", overrides)
            self.results[f"substructure_{name}"] = results
    
    def run_augmentation_ablation(self):
        """消融研究：数据增强。"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("AUGMENTATION ABLATION")
        self.logger.info("=" * 60)
        
        aug_variants = [
            ("none", {"augmentation.enable": False}),
            ("noise_only", {
                "augmentation.enable": True,
                "augmentation.time_warp": False,
                "augmentation.freq_mask": False
            }),
            ("time_warp_only", {
                "augmentation.enable": True,
                "augmentation.noise": False,
                "augmentation.freq_mask": False
            }),
            ("all", {
                "augmentation.enable": True,
                "augmentation.noise": True,
                "augmentation.time_warp": True,
                "augmentation.freq_mask": True
            }),
        ]
        
        for name, overrides in aug_variants:
            results = self.run_single_experiment(f"augmentation_{name}", overrides)
            self.results[f"augmentation_{name}"] = results
    
    def run_temperature_ablation(self):
        """消融研究：对比温度。"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TEMPERATURE ABLATION")
        self.logger.info("=" * 60)
        
        temp_variants = [
            ("t_0.01", {"loss.temperature": 0.01}),
            ("t_0.05", {"loss.temperature": 0.05}),
            ("t_0.07", {"loss.temperature": 0.07}),  # 默认
            ("t_0.1", {"loss.temperature": 0.1}),
            ("t_0.5", {"loss.temperature": 0.5}),
        ]
        
        for name, overrides in temp_variants:
            results = self.run_single_experiment(f"temperature_{name}", overrides)
            self.results[f"temperature_{name}"] = results
    
    def generate_report(self):
        """生成消融研究报告。"""
        report_path = self.output_dir / "ablation_report.md"
        
        with open(report_path, "w") as f:
            f.write("# PA-HCL 消融研究报告\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for category in ["encoder", "loss", "substructure", "augmentation", "temperature"]:
                category_results = {k: v for k, v in self.results.items() if k.startswith(category)}
                
                if category_results:
                    f.write(f"## {category.title()} Ablation\n\n")
                    f.write("| Variant | Status | Test Accuracy | Test F1 | Test AUROC |\n")
                    f.write("|---------|--------|---------------|---------|------------|\n")
                    
                    for name, result in category_results.items():
                        status = result.get("status", "unknown")
                        downstream = result.get("downstream", {})
                        acc = downstream.get("test_accuracy", "N/A")
                        f1 = downstream.get("test_f1", "N/A")
                        auroc = downstream.get("test_auroc", "N/A")
                        
                        if isinstance(acc, float):
                            acc = f"{acc:.4f}"
                        if isinstance(f1, float):
                            f1 = f"{f1:.4f}"
                        if isinstance(auroc, float):
                            auroc = f"{auroc:.4f}"
                        
                        f.write(f"| {name} | {status} | {acc} | {f1} | {auroc} |\n")
                    
                    f.write("\n")
        
        self.logger.info(f"Report saved to: {report_path}")
        
        # 也可以保存 JSON 结果
        results_path = self.output_dir / "all_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)


def main():
    args = parse_args()
    
    # 设置
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        name="ablation",
        log_file=output_dir / "ablation.log"
    )
    
    logger.info("=" * 60)
    logger.info("PA-HCL 消融研究")
    logger.info("=" * 60)
    
    set_seed(args.seed)
    
    # 加载基础配置
    base_config = load_config(args.config)
    
    # 创建运行器
    runner = AblationRunner(base_config, output_dir, logger, args)
    
    # 运行选定的消融
    if args.all or args.encoder_ablation:
        runner.run_encoder_ablation()
    
    if args.all or args.loss_ablation:
        runner.run_loss_ablation()
    
    if args.all or args.substructure_ablation:
        runner.run_substructure_ablation()
    
    if args.all or args.augmentation_ablation:
        runner.run_augmentation_ablation()
    
    if args.all or args.temperature_ablation:
        runner.run_temperature_ablation()
    
    # 生成报告
    runner.generate_report()
    
    logger.info("\n" + "=" * 60)
    logger.info("消融研究完成!")
    logger.info(f"结果保存至: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
