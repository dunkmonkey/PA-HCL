#!/bin/bash
# PA-HCL 实验监控快速开始示例
# 
# 此脚本演示如何使用 TensorBoard 和 WandB 进行实验监控

set -e

echo "=========================================="
echo "PA-HCL 实验监控示例"
echo "=========================================="

# 配置
PRETRAINED="checkpoints/pretrain/best_model.pt"
TASK="circor_murmur"

# 检查预训练模型
if [ ! -f "$PRETRAINED" ]; then
    echo "错误: 预训练模型不存在: $PRETRAINED"
    echo "请先运行预训练或指定正确的模型路径"
    exit 1
fi

echo ""
echo "选择监控方式:"
echo "1) TensorBoard (本地轻量级)"
echo "2) WandB (云端功能完整)"
echo "3) 同时使用两者"
echo ""
read -p "请选择 (1-3): " choice

case $choice in
    1)
        echo ""
        echo "使用 TensorBoard 进行监控..."
        echo ""
        python scripts/finetune.py \
            --task $TASK \
            --pretrained $PRETRAINED \
            --tensorboard \
            --epochs 10 \
            --experiment-name "${TASK}_tensorboard_demo"
        
        echo ""
        echo "✓ 训练完成！"
        echo ""
        echo "查看结果:"
        echo "  tensorboard --logdir outputs/${TASK}_tensorboard_demo/tensorboard"
        echo ""
        echo "然后在浏览器打开: http://localhost:6006"
        ;;
        
    2)
        echo ""
        echo "使用 WandB 进行监控..."
        echo ""
        
        # 检查 WandB 登录状态
        if ! wandb status > /dev/null 2>&1; then
            echo "WandB 未登录，请先登录:"
            wandb login
        fi
        
        python scripts/finetune.py \
            --task $TASK \
            --pretrained $PRETRAINED \
            --wandb \
            --wandb-project PA-HCL-Demo \
            --epochs 10 \
            --experiment-name "${TASK}_wandb_demo"
        
        echo ""
        echo "✓ 训练完成！"
        echo ""
        echo "查看结果:"
        echo "  访问 WandB 控制台链接（已在训练日志中显示）"
        ;;
        
    3)
        echo ""
        echo "同时使用 TensorBoard 和 WandB..."
        echo ""
        
        # 检查 WandB 登录状态
        if ! wandb status > /dev/null 2>&1; then
            echo "WandB 未登录，请先登录:"
            wandb login
        fi
        
        python scripts/finetune.py \
            --task $TASK \
            --pretrained $PRETRAINED \
            --tensorboard \
            --wandb \
            --wandb-project PA-HCL-Demo \
            --epochs 10 \
            --experiment-name "${TASK}_dual_demo"
        
        echo ""
        echo "✓ 训练完成！"
        echo ""
        echo "查看结果:"
        echo "  TensorBoard: tensorboard --logdir outputs/${TASK}_dual_demo/tensorboard"
        echo "  WandB: 访问控制台链接（已在训练日志中显示）"
        ;;
        
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "示例完成！"
echo "=========================================="
