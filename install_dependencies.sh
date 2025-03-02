#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================="
echo -e "GGUF Benchmark Dependencies"
echo -e "=============================${NC}"
echo

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}Python version: ${python_version}${NC}"

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment. Please install venv:${NC}"
        echo "sudo apt-get install python3-venv  # For Debian/Ubuntu"
        echo "sudo dnf install python3-venv      # For Fedora"
        exit 1
    fi
fi

# Activate the virtual environment
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to activate virtual environment.${NC}"
    exit 1
fi
echo -e "${GREEN}Virtual environment activated.${NC}"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install basic dependencies
echo -e "${YELLOW}Installing basic dependencies...${NC}"
pip install torch tqdm colorama plotly pandas

# Detect GPU and install appropriate llama-cpp-python version
echo -e "${YELLOW}Checking for NVIDIA GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected.${NC}"
    
    # Get CUDA version
    cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1)
    echo -e "${GREEN}CUDA driver version: ${cuda_version}${NC}"
    
    # Ask user which CUDA version to use for llama-cpp-python
    echo -e "${YELLOW}Which CUDA version would you like to use for llama-cpp-python?${NC}"
    echo "1) CUDA 11.7 (Compatible with older GPUs)"
    echo "2) CUDA 11.8"
    echo "3) CUDA 12.2"
    echo "4) CUDA 12.5 (Use the closest compatible version: 12.2)"
    echo "5) CPU only (No GPU acceleration)"
    read -p "Enter choice [1-5]: " cuda_choice
    
    case $cuda_choice in
        1)
            echo -e "${YELLOW}Installing llama-cpp-python with CUDA 11.7 support...${NC}"
            pip install llama-cpp-python --upgrade --force-reinstall --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117
            ;;
        2)
            echo -e "${YELLOW}Installing llama-cpp-python with CUDA 11.8 support...${NC}"
            pip install llama-cpp-python --upgrade --force-reinstall --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu118
            ;;
        3)
            echo -e "${YELLOW}Installing llama-cpp-python with CUDA 12.2 support...${NC}"
            pip install llama-cpp-python --upgrade --force-reinstall --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu122
            ;;
        4)
            echo -e "${YELLOW}Installing llama-cpp-python for CUDA 12.5 (using 12.2 wheels)...${NC}"
            echo -e "${YELLOW}Note: No specific CUDA 12.5 wheels are available, using 12.2 which is compatible${NC}"
            pip install llama-cpp-python --upgrade --force-reinstall --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu122
            ;;
        5)
            echo -e "${YELLOW}Installing CPU-only llama-cpp-python...${NC}"
            pip install llama-cpp-python --upgrade
            ;;
        *)
            echo -e "${RED}Invalid choice. Installing CPU-only version.${NC}"
            pip install llama-cpp-python --upgrade
            ;;
    esac
else
    echo -e "${YELLOW}No NVIDIA GPU detected. Installing CPU-only llama-cpp-python...${NC}"
    pip install llama-cpp-python --upgrade
fi

# Check if llama-cpp-python was installed correctly
if pip show llama-cpp-python > /dev/null 2>&1; then
    echo -e "${GREEN}llama-cpp-python installed successfully.${NC}"
else
    echo -e "${RED}llama-cpp-python installation failed.${NC}"
    exit 1
fi

# Install lm-eval
echo -e "${YELLOW}Installing lm-evaluation-harness...${NC}"
pip install lm-eval --upgrade

# Install any additional requirements
echo -e "${YELLOW}Installing additional requirements...${NC}"
pip install psutil --upgrade  # For better system monitoring

# Make the benchmark script executable
chmod +x run_benchmark.sh
chmod +x gguf_benchmark.py

echo -e "${GREEN}================================="
echo -e "All dependencies installed successfully!"
echo -e "==================================${NC}"
echo
echo -e "To run the benchmark, use: ${BLUE}./run_benchmark.sh${NC}"
echo -e "or: ${BLUE}python gguf_benchmark.py --model_path /path/to/model.gguf${NC}"
