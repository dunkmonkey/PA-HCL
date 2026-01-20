#!/bin/bash
# ============================================================
# PA-HCL 下游任务数据准备和训练自动化脚本
# ============================================================
#
# 用法:
#   # 准备所有数据集
#   ./scripts/run_downstream.sh prepare
#
#   # 运行所有任务的微调
#   ./scripts/run_downstream.sh train
#
#   # 完整流程（准备 + 训练）
#   ./scripts/run_downstream.sh all
#
#   # 运行特定任务
#   ./scripts/run_downstream.sh train circor_murmur
#
# 作者: PA-HCL 团队
# ============================================================

set -e  # 遇到错误时退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 默认配置
PRETRAINED_MODEL="${PRETRAINED_MODEL:-checkpoints/pretrain/best_model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
DATA_DIR="${DATA_DIR:-data/downstream}"
RAW_DIR="${RAW_DIR:-data/raw}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
SEED="${SEED:-42}"

# 可用任务
TASKS=("circor_murmur" "circor_outcome" "physionet2016" "pascal")

# ============================================================
# 辅助函数
# ============================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "============================================================"
    echo -e "${GREEN}$1${NC}"
    echo "============================================================"
}

check_pretrained_model() {
    if [[ ! -f "$PRETRAINED_MODEL" ]]; then
        log_error "预训练模型不存在: $PRETRAINED_MODEL"
        log_info "请先运行预训练或设置 PRETRAINED_MODEL 环境变量"
        return 1
    fi
    log_success "找到预训练模型: $PRETRAINED_MODEL"
    return 0
}

check_data_dir() {
    local task=$1
    local task_dir="$DATA_DIR/$task"
    
    if [[ -d "$task_dir" && -f "$task_dir/train.csv" ]]; then
        return 0
    fi
    return 1
}

# ============================================================
# 数据准备
# ============================================================

prepare_data() {
    print_header "准备下游任务数据"
    
    local dataset="${1:-all}"
    
    log_info "数据集: $dataset"
    log_info "原始数据目录: $RAW_DIR"
    log_info "预处理数据目录: $PROCESSED_DIR"
    log_info "输出目录: $DATA_DIR"
    
    python scripts/data_preparation/prepare_downstream_tasks.py \
        --dataset "$dataset" \
        --raw-dir "$RAW_DIR" \
        --processed-dir "$PROCESSED_DIR" \
        --output-dir "$DATA_DIR" \
        --seed "$SEED" \
        --verbose
    
    log_success "数据准备完成"
}

# ============================================================
# 训练任务
# ============================================================

train_task() {
    local task=$1
    local mode="${2:-finetune}"  # finetune, linear, fewshot
    
    print_header "训练任务: $task ($mode)"
    
    # 检查预训练模型
    if ! check_pretrained_model; then
        return 1
    fi
    
    # 检查数据
    if ! check_data_dir "$task"; then
        log_error "任务数据不存在: $DATA_DIR/$task"
        log_info "请先运行: ./scripts/run_downstream.sh prepare"
        return 1
    fi
    
    # 构建命令
    local cmd="python scripts/finetune.py --task $task --pretrained $PRETRAINED_MODEL"
    cmd+=" --output-dir $OUTPUT_DIR --data-dir $DATA_DIR --seed $SEED"
    
    case "$mode" in
        linear)
            cmd+=" --linear-eval"
            ;;
        fewshot)
            cmd+=" --few-shot --shot-ratio 0.1"
            ;;
        fewshot1)
            cmd+=" --few-shot --shot-ratio 0.01"
            ;;
        *)
            # 全量微调
            ;;
    esac
    
    log_info "执行命令: $cmd"
    eval "$cmd"
    
    log_success "任务 $task ($mode) 训练完成"
}

train_all_tasks() {
    local mode="${1:-finetune}"
    
    print_header "训练所有任务 ($mode)"
    
    for task in "${TASKS[@]}"; do
        if check_data_dir "$task"; then
            log_info "开始训练: $task"
            train_task "$task" "$mode" || log_warning "任务 $task 失败，继续下一个..."
        else
            log_warning "跳过 $task (数据不存在)"
        fi
    done
    
    log_success "所有任务训练完成"
}

# ============================================================
# 结果汇总
# ============================================================

summarize_results() {
    print_header "汇总实验结果"
    
    local results_file="$OUTPUT_DIR/all_results_summary.csv"
    
    echo "task,mode,accuracy,f1,f1_macro,auroc" > "$results_file"
    
    for task in "${TASKS[@]}"; do
        for mode in "finetune" "linear"; do
            local experiment_name="${task}_${mode}"
            local metrics_file="$OUTPUT_DIR/${experiment_name}/final_metrics.json"
            
            if [[ -f "$metrics_file" ]]; then
                # 使用 Python 解析 JSON
                python -c "
import json
with open('$metrics_file') as f:
    m = json.load(f)
acc = m.get('test_accuracy', m.get('val_accuracy', 0))
f1 = m.get('test_f1', m.get('val_f1', 0))
f1_macro = m.get('test_f1_macro', m.get('val_f1_macro', f1))
auroc = m.get('test_auroc', m.get('val_auroc', 0))
print(f'$task,$mode,{acc:.4f},{f1:.4f},{f1_macro:.4f},{auroc:.4f}')
" >> "$results_file"
            fi
        done
    done
    
    log_info "结果已保存到: $results_file"
    
    # 打印表格
    echo ""
    echo "实验结果汇总:"
    echo "------------------------------------------------------------"
    column -t -s, "$results_file"
    echo "------------------------------------------------------------"
}

# ============================================================
# 主函数
# ============================================================

show_usage() {
    echo "用法: $0 <command> [options]"
    echo ""
    echo "命令:"
    echo "  prepare [dataset]    准备下游任务数据 (默认: all)"
    echo "  train [task] [mode]  训练任务 (mode: finetune/linear/fewshot)"
    echo "  train-all [mode]     训练所有任务"
    echo "  all                  完整流程 (准备 + 训练所有)"
    echo "  summarize            汇总实验结果"
    echo "  help                 显示此帮助信息"
    echo ""
    echo "可用任务:"
    echo "  ${TASKS[*]}"
    echo ""
    echo "环境变量:"
    echo "  PRETRAINED_MODEL    预训练模型路径 (默认: checkpoints/pretrain/best_model.pt)"
    echo "  OUTPUT_DIR          输出目录 (默认: outputs)"
    echo "  DATA_DIR            下游数据目录 (默认: data/downstream)"
    echo "  SEED                随机种子 (默认: 42)"
    echo ""
    echo "示例:"
    echo "  $0 prepare                          # 准备所有数据"
    echo "  $0 prepare circor                   # 仅准备 CirCor 数据"
    echo "  $0 train circor_murmur              # 训练杂音检测任务"
    echo "  $0 train circor_murmur linear       # 线性评估"
    echo "  $0 train circor_murmur fewshot      # 10% 少样本学习"
    echo "  $0 train-all                        # 训练所有任务"
    echo "  $0 all                              # 完整流程"
}

main() {
    local command="${1:-help}"
    shift || true
    
    case "$command" in
        prepare)
            prepare_data "$@"
            ;;
        train)
            if [[ -z "$1" ]]; then
                train_all_tasks
            else
                train_task "$@"
            fi
            ;;
        train-all)
            train_all_tasks "$@"
            ;;
        all)
            prepare_data
            train_all_tasks
            summarize_results
            ;;
        summarize)
            summarize_results
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            log_error "未知命令: $command"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
