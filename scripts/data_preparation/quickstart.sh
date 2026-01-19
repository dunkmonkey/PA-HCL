#!/bin/bash
# Quick Start Script for PA-HCL Data Preparation
# 
# This script provides a streamlined workflow for preparing datasets.
# Usage: ./quickstart.sh [circor|physionet2016|pascal|custom|all]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    print_success "Python 3 found: $(python3 --version)"
}

# Check if required packages are installed
check_dependencies() {
    print_info "Checking dependencies..."
    
    python3 -c "import pandas" 2>/dev/null || {
        print_warning "pandas not found. Installing..."
        pip install pandas
    }
    
    python3 -c "import tqdm" 2>/dev/null || {
        print_warning "tqdm not found. Installing..."
        pip install tqdm
    }
    
    python3 -c "import requests" 2>/dev/null || {
        print_warning "requests not found. Installing..."
        pip install requests
    }
    
    print_success "All dependencies installed"
}

# Download dataset
download_dataset() {
    local dataset=$1
    print_header "Downloading $dataset Dataset"
    
    python3 scripts/data_preparation/download_datasets.py \
        --dataset "$dataset" \
        --output-dir ./data/downloads \
        --keep-archive
    
    if [ $? -eq 0 ]; then
        print_success "Download complete"
        return 0
    else
        print_error "Download failed"
        return 1
    fi
}

# Prepare CirCor dataset
prepare_circor() {
    print_header "Preparing CirCor Dataset"
    
    local input_dir="./data/downloads/extracted/the-circor-digiscope-phonocardiogram-dataset-1.0.3"
    
    if [ ! -d "$input_dir" ]; then
        print_error "CirCor dataset not found at: $input_dir"
        print_info "Please download first with: ./quickstart.sh download circor"
        return 1
    fi
    
    python3 scripts/data_preparation/prepare_circor.py \
        --input-dir "$input_dir" \
        --output-dir ./data/raw/circor \
        --verbose
    
    if [ $? -eq 0 ]; then
        print_success "CirCor dataset prepared successfully"
        print_info "Output: ./data/raw/circor"
        return 0
    else
        print_error "CirCor preparation failed"
        return 1
    fi
}

# Prepare PhysioNet 2016 dataset
prepare_physionet2016() {
    print_header "Preparing PhysioNet 2016 Dataset"
    
    local input_dir="./data/downloads/extracted/classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0"
    
    if [ ! -d "$input_dir" ]; then
        print_error "PhysioNet 2016 dataset not found at: $input_dir"
        print_info "Please download first with: ./quickstart.sh download physionet2016"
        return 1
    fi
    
    python3 scripts/data_preparation/prepare_physionet2016.py \
        --input-dir "$input_dir" \
        --output-dir ./data/raw/physionet2016 \
        --include-validation \
        --verbose
    
    if [ $? -eq 0 ]; then
        print_success "PhysioNet 2016 dataset prepared successfully"
        print_info "Output: ./data/raw/physionet2016"
        return 0
    else
        print_error "PhysioNet 2016 preparation failed"
        return 1
    fi
}

# Prepare Pascal dataset
prepare_pascal() {
    print_header "Preparing Pascal Challenge Dataset"
    
    local input_dir="./data/downloads/extracted/heartchallenge"
    
    if [ ! -d "$input_dir" ]; then
        print_error "Pascal dataset not found at: $input_dir"
        print_warning "Pascal dataset requires manual download"
        print_info "Please visit: https://istethoscope.peterjbentley.com/heartchallenge/"
        print_info "After download, extract to: $input_dir"
        return 1
    fi
    
    python3 scripts/data_preparation/prepare_pascal.py \
        --input-dir "$input_dir" \
        --output-dir ./data/raw/pascal \
        --verbose
    
    if [ $? -eq 0 ]; then
        print_success "Pascal dataset prepared successfully"
        print_info "Output: ./data/raw/pascal"
        return 0
    else
        print_error "Pascal preparation failed"
        return 1
    fi
}

# Prepare custom dataset
prepare_custom() {
    print_header "Preparing Custom Dataset"
    
    local input_dir="./data/raw"
    
    # Check if Normal or Abnormal folders exist
    if [ ! -d "$input_dir/Normal" ] && [ ! -d "$input_dir/Abnormal" ]; then
        print_error "Custom dataset not found"
        print_info "Expected structure:"
        print_info "  data/raw/"
        print_info "  ├── Normal/"
        print_info "  │   └── *.wav"
        print_info "  └── Abnormal/"
        print_info "      └── *.wav"
        return 1
    fi
    
    python3 scripts/data_preparation/prepare_custom.py \
        --input-dir "$input_dir" \
        --output-dir ./data/raw/custom_organized \
        --verbose
    
    if [ $? -eq 0 ]; then
        print_success "Custom dataset prepared successfully"
        print_info "Output: ./data/raw/custom_organized"
        return 0
    else
        print_error "Custom dataset preparation failed"
        return 1
    fi
}

# Run preprocessing
run_preprocessing() {
    local dataset_dir=$1
    print_header "Running Preprocessing"
    
    if [ ! -d "$dataset_dir" ]; then
        print_error "Dataset directory not found: $dataset_dir"
        return 1
    fi
    
    python3 scripts/preprocess.py \
        --raw_dir "$dataset_dir" \
        --output_dir ./data/processed \
        --num_workers 4
    
    if [ $? -eq 0 ]; then
        print_success "Preprocessing complete"
        print_info "Processed data: ./data/processed"
        return 0
    else
        print_error "Preprocessing failed"
        return 1
    fi
}

# Show help
show_help() {
    echo "PA-HCL Data Preparation Quick Start"
    echo ""
    echo "Usage: $0 <command> [dataset]"
    echo ""
    echo "Commands:"
    echo "  check                    Check dependencies"
    echo "  download <dataset>       Download a public dataset"
    echo "  prepare <dataset>        Prepare a dataset for preprocessing"
    echo "  preprocess <dataset>     Run preprocessing on prepared dataset"
    echo "  full <dataset>           Run complete pipeline (download → prepare → preprocess)"
    echo "  help                     Show this help message"
    echo ""
    echo "Datasets:"
    echo "  circor                   CirCor DigiScope Dataset"
    echo "  physionet2016            PhysioNet 2016 Challenge Dataset"
    echo "  pascal                   Pascal Challenge Dataset (manual download required)"
    echo "  custom                   Your custom dataset"
    echo "  all                      All public datasets (for download only)"
    echo ""
    echo "Examples:"
    echo "  $0 check"
    echo "  $0 download circor"
    echo "  $0 prepare circor"
    echo "  $0 full circor"
    echo "  $0 prepare custom"
    echo ""
}

# Main logic
main() {
    local command=$1
    local dataset=$2
    
    # Set script directory
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd "$SCRIPT_DIR"
    
    case "$command" in
        check)
            print_header "Checking Environment"
            check_python
            check_dependencies
            print_success "Environment check complete"
            ;;
        
        download)
            if [ -z "$dataset" ]; then
                print_error "Please specify a dataset: circor, physionet2016, pascal, or all"
                exit 1
            fi
            check_python
            check_dependencies
            download_dataset "$dataset"
            ;;
        
        prepare)
            if [ -z "$dataset" ]; then
                print_error "Please specify a dataset: circor, physionet2016, pascal, or custom"
                exit 1
            fi
            check_python
            check_dependencies
            
            case "$dataset" in
                circor)
                    prepare_circor
                    ;;
                physionet2016)
                    prepare_physionet2016
                    ;;
                pascal)
                    prepare_pascal
                    ;;
                custom)
                    prepare_custom
                    ;;
                *)
                    print_error "Unknown dataset: $dataset"
                    exit 1
                    ;;
            esac
            ;;
        
        preprocess)
            if [ -z "$dataset" ]; then
                print_error "Please specify a dataset directory"
                exit 1
            fi
            check_python
            
            # Map dataset name to directory
            case "$dataset" in
                circor)
                    run_preprocessing "./data/raw/circor"
                    ;;
                physionet2016)
                    run_preprocessing "./data/raw/physionet2016"
                    ;;
                pascal)
                    run_preprocessing "./data/raw/pascal"
                    ;;
                custom)
                    run_preprocessing "./data/raw/custom_organized"
                    ;;
                *)
                    # Assume it's a directory path
                    run_preprocessing "$dataset"
                    ;;
            esac
            ;;
        
        full)
            if [ -z "$dataset" ]; then
                print_error "Please specify a dataset: circor, physionet2016, pascal, or custom"
                exit 1
            fi
            
            check_python
            check_dependencies
            
            case "$dataset" in
                circor|physionet2016)
                    download_dataset "$dataset" && \
                    prepare_"$dataset" && \
                    run_preprocessing "./data/raw/$dataset"
                    ;;
                pascal)
                    print_warning "Pascal requires manual download"
                    prepare_pascal && \
                    run_preprocessing "./data/raw/pascal"
                    ;;
                custom)
                    prepare_custom && \
                    run_preprocessing "./data/raw/custom_organized"
                    ;;
                *)
                    print_error "Unknown dataset: $dataset"
                    exit 1
                    ;;
            esac
            ;;
        
        help|--help|-h|"")
            show_help
            ;;
        
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
