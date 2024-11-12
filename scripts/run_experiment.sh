#!/bin/bash
# scripts/run_experiment.sh

# Configure GPU settings
export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Modify based on available GPUs

# Set up environment variables
export PROJECT_DIR="/home/ghazal/Documents/CS_Projects/CVPR2025/graph-augmentation/baseline+spatial"
export DATA_DIR="/users/gm00051/ChestX-ray14"
export CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
export OUTPUT_DIR="$PROJECT_DIR/outputs"

# Create required directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$OUTPUT_DIR"

# Function to select optimal GPU
select_gpu() {
    local gpu_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    local max_memory=0
    local selected_gpu=0

    IFS=$'\n' read -rd '' -a memory_array <<< "$gpu_memory"

    for i in "${!memory_array[@]}"; do
        if (( ${memory_array[$i]} > max_memory )); then
            max_memory=${memory_array[$i]}
            selected_gpu=$i
        fi
    done

    echo $selected_gpu
}

# Parse command line arguments
CONFIG_FILE=""
CHECKPOINT=""
GPU_COUNT=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate arguments
if [ -z "$CONFIG_FILE" ]; then
    echo "Config file must be specified with --config"
    exit 1
fi

# Select GPU(s)
if [ $GPU_COUNT -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=$(select_gpu)
    echo "Using GPU: $CUDA_VISIBLE_DEVICES"
else
    # For multi-GPU, use all available
    echo "Using multiple GPUs"
fi

# Start training
if [ -z "$CHECKPOINT" ]; then
    python scripts/train.py \
        --config "$CONFIG_FILE" \
        --device "cuda"
else
    python scripts/train.py \
        --config "$CONFIG_FILE" \
        --checkpoint "$CHECKPOINT" \
        --device "cuda"
fi

# Run evaluation
python scripts/evaluate.py \
    --config "$CONFIG_FILE" \
    --checkpoint "$CHECKPOINT_DIR/best_model.pt" \
    --device "cuda"

