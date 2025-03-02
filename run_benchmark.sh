#!/bin/bash
# GGUF Benchmark Launcher
# ----------------------

# Activate virtual environment
source ./venv/bin/activate

# Set text colors
echo -e "\033[1;34m====================================="
echo -e "    GGUF MODEL BENCHMARK LAUNCHER"
echo -e "====================================="
echo -e "\033[0m"
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

# Create output directory
OUTPUT_DIR="benchmark_results/run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Run the benchmark
echo
echo "Running benchmark with the following parameters:"
echo "Model Path: $MODEL_PATH"
echo "GPU Layers: $GPU_LAYERS"
echo "Context Size: $CONTEXT_SIZE"
echo "Benchmarks: $BENCHMARKS"
echo "Output Directory: $OUTPUT_DIR"
echo

read -n 1 -s -r -p "Press any key to continue..."
echo

# Run the Python script
python gguf_benchmark.py --model_path "$MODEL_PATH" --output_dir "$OUTPUT_DIR" --n_gpu_layers "$GPU_LAYERS" --context_size "$CONTEXT_SIZE" --benchmarks "$BENCHMARKS"

# Keep console open
echo
read -n 1 -s -r -p "Press any key to exit..."
echo
