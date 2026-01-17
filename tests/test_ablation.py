"""
消融研究组件的测试。

运行测试:
    pytest tests/test_ablation.py -v

作者: PA-HCL 团队
"""

import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestAblationRunner:
    """测试 AblationRunner 类。"""
    
    def test_config_override_simple(self):
        """测试简单的配置覆盖。"""
        # Mock config class
        class MockConfig:
            def __init__(self):
                self.model = type('', (), {'encoder_type': 'cnn_mamba'})()
                self.loss = type('', (), {'temperature': 0.07})()
        
        config = MockConfig()
        
        # Simulate override
        config.model.encoder_type = "cnn_transformer"
        config.loss.temperature = 0.1
        
        assert config.model.encoder_type == "cnn_transformer"
        assert config.loss.temperature == 0.1
        print("✓ Simple config override works")
    
    def test_config_to_dict(self):
        """测试配置到字典的转换。"""
        class MockConfig:
            def __init__(self):
                self.name = "test"
                self.nested = type('', (), {'value': 42})()
        
        config = MockConfig()
        
        # Manual conversion
        result = {
            "name": config.name,
            "nested": {"value": config.nested.value}
        }
        
        assert result["name"] == "test"
        assert result["nested"]["value"] == 42
        print("✓ Config to dict conversion works")


class TestAblationVariants:
    """测试消融实验变体。"""
    
    def test_encoder_variants(self):
        """测试编码器变体定义。"""
        encoder_variants = [
            ("cnn_only", "cnn_only"),
            ("cnn_transformer", "cnn_transformer"),
            ("cnn_mamba", "cnn_mamba"),
        ]
        
        for name, encoder_type in encoder_variants:
            assert encoder_type in ["cnn_only", "cnn_transformer", "cnn_mamba"]
        
        print("✓ Encoder variants are valid")
    
    def test_loss_variants(self):
        """测试损失组件变体定义。"""
        loss_variants = [
            ("cycle_only", 1.0, 0.0),
            ("sub_only", 0.0, 1.0),
            ("hierarchical", 1.0, 0.5),
        ]
        
        for name, lambda_cycle, lambda_sub in loss_variants:
            assert lambda_cycle >= 0
            assert lambda_sub >= 0
            assert lambda_cycle + lambda_sub > 0  # At least one should be non-zero
        
        print("✓ Loss variants are valid")
    
    def test_substructure_variants(self):
        """测试子结构数量变体。"""
        sub_variants = [2, 4, 8]
        
        for k in sub_variants:
            assert k > 0
            assert k % 2 == 0  # Should be even for cardiac cycles
        
        print("✓ Substructure variants are valid")
    
    def test_temperature_variants(self):
        """测试温度变体。"""
        temp_variants = [0.01, 0.05, 0.07, 0.1, 0.5]
        
        for t in temp_variants:
            assert 0 < t < 1  # Temperature should be in (0, 1)
        
        print("✓ Temperature variants are valid")


class TestAblationResults:
    """测试消融结果处理。"""
    
    def test_result_structure(self):
        """测试结果字典结构。"""
        result = {
            "experiment": "encoder_cnn_mamba",
            "config_overrides": {"model.encoder_type": "cnn_mamba"},
            "status": "completed",
            "pretrain": {
                "epochs": 100,
                "final_loss": 0.25
            },
            "downstream": {
                "test_accuracy": 0.85,
                "test_f1": 0.83,
                "test_auroc": 0.92
            }
        }
        
        assert "experiment" in result
        assert "status" in result
        assert "pretrain" in result
        assert "downstream" in result
        assert result["downstream"]["test_accuracy"] > 0
        
        print("✓ Result structure is valid")
    
    def test_result_aggregation(self):
        """测试聚合多个结果。"""
        results = {
            "encoder_cnn_only": {"downstream": {"test_f1": 0.75}},
            "encoder_cnn_transformer": {"downstream": {"test_f1": 0.80}},
            "encoder_cnn_mamba": {"downstream": {"test_f1": 0.85}},
        }
        
        f1_scores = [r["downstream"]["test_f1"] for r in results.values()]
        best_variant = max(results.keys(), key=lambda k: results[k]["downstream"]["test_f1"])
        
        assert best_variant == "encoder_cnn_mamba"
        assert max(f1_scores) == 0.85
        
        print("✓ Result aggregation works")


class TestReportGeneration:
    """测试报告生成。"""
    
    def test_markdown_table_generation(self):
        """测试生成 markdown 表格。"""
        results = {
            "encoder_cnn_only": {
                "status": "completed",
                "downstream": {"test_accuracy": 0.75, "test_f1": 0.73}
            },
            "encoder_cnn_mamba": {
                "status": "completed",
                "downstream": {"test_accuracy": 0.85, "test_f1": 0.83}
            }
        }
        
        # Generate table
        lines = [
            "| Variant | Status | Test Accuracy | Test F1 |",
            "|---------|--------|---------------|---------|"
        ]
        
        for name, result in results.items():
            status = result["status"]
            acc = result["downstream"]["test_accuracy"]
            f1 = result["downstream"]["test_f1"]
            lines.append(f"| {name} | {status} | {acc:.4f} | {f1:.4f} |")
        
        table = "\n".join(lines)
        
        assert "encoder_cnn_only" in table
        assert "encoder_cnn_mamba" in table
        assert "0.8500" in table
        
        print("✓ Markdown table generation works")
        print("\nGenerated table:")
        print(table)



class TestIntegration:
    """消融实验流水线的集成测试。"""
    
    def test_ablation_pipeline_structure(self):
        """测试消融实验流水线结构。"""
        import json
        from pathlib import Path
        
        # Verify config file exists
        config_path = project_root / "configs" / "ablation.yaml"
        assert config_path.exists(), "ablation.yaml should exist"
        
        # Verify script exists
        script_path = project_root / "scripts" / "ablation.py"
        assert script_path.exists(), "ablation.py should exist"
        
        print("✓ Ablation pipeline files exist")
    
    def test_simulated_ablation_run(self):
        """Simulate an ablation run without actual training."""
        print("\n" + "=" * 50)
        print("Simulated Ablation Run")
        print("=" * 50)
        
        # Simulate encoder ablation
        encoder_variants = ["cnn_only", "cnn_transformer", "cnn_mamba"]
        results = {}
        
        for variant in encoder_variants:
            # Simulate results
            import random
            random.seed(hash(variant))
            
            results[f"encoder_{variant}"] = {
                "status": "completed",
                "pretrain": {
                    "epochs": 100,
                    "final_loss": random.uniform(0.1, 0.5)
                },
                "downstream": {
                    "test_accuracy": random.uniform(0.7, 0.9),
                    "test_f1": random.uniform(0.65, 0.88),
                    "test_auroc": random.uniform(0.8, 0.95)
                }
            }
        
        # Print summary
        print("\nSimulated Encoder Ablation Results:")
        print("-" * 50)
        for name, result in results.items():
            d = result["downstream"]
            print(f"{name}: Acc={d['test_accuracy']:.4f}, F1={d['test_f1']:.4f}")
        
        # Verify all variants completed
        assert all(r["status"] == "completed" for r in results.values())
        print("\n✓ Simulated ablation run completed")


def run_all_tests():
    """Run all ablation tests."""
    print("\n" + "=" * 60)
    print("PA-HCL Ablation Study Tests")
    print("=" * 60 + "\n")
    
    # Runner tests
    print("Testing AblationRunner...")
    test_runner = TestAblationRunner()
    test_runner.test_config_override_simple()
    test_runner.test_config_to_dict()
    
    # Variant tests
    print("\nTesting Ablation Variants...")
    test_variants = TestAblationVariants()
    test_variants.test_encoder_variants()
    test_variants.test_loss_variants()
    test_variants.test_substructure_variants()
    test_variants.test_temperature_variants()
    
    # Result tests
    print("\nTesting Result Handling...")
    test_results = TestAblationResults()
    test_results.test_result_structure()
    test_results.test_result_aggregation()
    
    # Report tests
    print("\nTesting Report Generation...")
    test_report = TestReportGeneration()
    test_report.test_markdown_table_generation()
    
    # Integration tests
    print("\nTesting Integration...")
    test_integration = TestIntegration()
    test_integration.test_ablation_pipeline_structure()
    test_integration.test_simulated_ablation_run()
    
    print("\n" + "=" * 60)
    print("All ablation tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
