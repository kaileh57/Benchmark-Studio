#!/bin/bash
# GGUF Benchmark Launcher
# ----------------------

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source ./venv/bin/activate
fi

# Set text colors
BLUE='\033[1;34m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
RESET='\033[0m'

echo -e "${BLUE}=====================================
    GGUF MODEL BENCHMARK LAUNCHER
=====================================${RESET}"
echo

# Get model path
read -p "Enter path to GGUF model file: " MODEL_PATH

# Get GPU layers
echo -n "Number of GPU layers (-1 for all) [default: -1]: "
read GPU_LAYERS
GPU_LAYERS=${GPU_LAYERS:-"-1"}

# Get context size
echo -n "Context size [default: 2048]: "
read CONTEXT_SIZE
CONTEXT_SIZE=${CONTEXT_SIZE:-"2048"}

# Get benchmarks
echo -n "Benchmarks to run (space-separated) [default: mmlu arc_easy hellaswag]: "
read BENCHMARKS
BENCHMARKS=${BENCHMARKS:-"mmlu arc_easy hellaswag"}

# Ask for verbose mode
echo -n "Enable verbose mode? (y/n) [default: n]: "
read VERBOSE_MODE
VERBOSE_MODE=${VERBOSE_MODE:-"n"}

# Set verbose flag if needed
VERBOSE_FLAG=""
if [[ $VERBOSE_MODE == "y" || $VERBOSE_MODE == "Y" ]]; then
    VERBOSE_FLAG="--verbose"
fi

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="benchmark_results/run_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Run the benchmark
echo
echo -e "${GREEN}Running benchmark with the following parameters:${RESET}"
echo "Model Path: $MODEL_PATH"
echo "GPU Layers: $GPU_LAYERS"
echo "Context Size: $CONTEXT_SIZE"
echo "Benchmarks: $BENCHMARKS"
echo "Output Directory: $OUTPUT_DIR"
echo "Verbose Mode: ${VERBOSE_MODE}"
echo

read -n 1 -s -r -p "Press any key to continue..."
echo

# Run the Python script with all parameters
python gguf_benchmark.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --n_gpu_layers "$GPU_LAYERS" \
    --context_size "$CONTEXT_SIZE" \
    --benchmarks $BENCHMARKS \
    $VERBOSE_FLAG

# Keep console open
echo
read -n 1 -s -r -p "Press any key to exit..."
echo
