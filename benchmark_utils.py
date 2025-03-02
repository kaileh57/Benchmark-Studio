#!/usr/bin/env python3
"""
Utility functions for GGUF benchmarking
"""

import os
import sys
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Progress bar and terminal colors
from tqdm import tqdm
import colorama
from colorama import Fore, Style

# Initialize colorama for Windows
colorama.init()

# Configure logging
def setup_logging(verbose: bool = False):
    """Configure logging with optional verbose mode"""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console)
    
    # Return the root logger
    return root_logger

def check_system_requirements() -> Dict[str, Any]:
    """Check system requirements and return system info"""
    # Check for CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_device_count = torch.cuda.device_count()
            cuda_device_name = torch.cuda.get_device_name(0)
        else:
            cuda_device_count = 0
            cuda_device_name = "N/A"
    except ImportError:
        cuda_available = False
        cuda_device_count = 0
        cuda_device_name = "N/A"
    
    # Check for llama-cpp-python
    try:
        from llama_cpp import Llama
        llama_cpp_available = True
    except ImportError:
        llama_cpp_available = False
    
    # Check for lm_eval
    try:
        from lm_eval import evaluator, tasks
        lm_eval_available = True
    except ImportError:
        lm_eval_available = False
    
    system_info = {
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_name": cuda_device_name,
        "llama_cpp_available": llama_cpp_available,
        "lm_eval_available": lm_eval_available,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    return system_info

def print_system_info(system_info: Dict[str, Any]) -> None:
    """Print system information with colors"""
    print(f"{Fore.YELLOW}Checking system requirements...{Style.RESET_ALL}")
    
    if system_info["cuda_available"]:
        print(f"{Fore.GREEN}✓ CUDA is available: {system_info['cuda_device_name']}{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}× CUDA is not available. The benchmark will run on CPU only.{Style.RESET_ALL}")
        print(f"{Fore.RED}  This will be significantly slower.{Style.RESET_ALL}")
    
    if system_info["llama_cpp_available"]:
        print(f"{Fore.GREEN}✓ llama-cpp-python is installed{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}× llama-cpp-python is required but not found.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  Please install it with CUDA support:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  pip install llama-cpp-python --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117{Style.RESET_ALL}")
    
    if system_info["lm_eval_available"]:
        print(f"{Fore.GREEN}✓ lm-evaluation-harness is installed{Style.RESET_ALL}")
    else:
        print(f"{Fore.RED}× lm-evaluation-harness is required but not found.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}  Please install it:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  pip install lm-eval{Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}✓ All checks completed{Style.RESET_ALL}")

class SpinnerAnimation:
    """An animated spinner for console output during long-running operations"""
    
    def __init__(self, desc="Processing"):
        """Initialize the spinner with a description"""
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.desc = desc
        self.index = 0
        self.start_time = None
        self.is_running = False
        self.thread = None
    
    def _spin(self):
        """Main spinner loop to be run in a thread"""
        import threading
        
        while self.is_running:
            elapsed = time.time() - self.start_time
            mins, secs = divmod(int(elapsed), 60)
            elapsed_str = f"{mins:02d}:{secs:02d}"
            
            char = self.spinner_chars[self.index % len(self.spinner_chars)]
            sys.stdout.write(f"\r{Fore.CYAN}{char} {self.desc} {Style.RESET_ALL}[{elapsed_str}]")
            sys.stdout.flush()
            self.index += 1
            time.sleep(0.1)
    
    def start(self):
        """Start the spinner animation"""
        self.is_running = True
        self.start_time = time.time()
        
        import threading
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop the spinner animation"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        sys.stdout.write("\r" + " " * 100 + "\r")  # Clear the line
        sys.stdout.flush()
    
    def __enter__(self):
        """Support for context manager usage"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up when exiting the context manager"""
        self.stop()

def format_time(seconds: float) -> str:
    """Format seconds into a human-readable time string"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins, secs = divmod(seconds, 60)
        return f"{int(mins)}m {int(secs)}s"
    else:
        hours, remainder = divmod(seconds, 3600)
        mins, secs = divmod(remainder, 60)
        return f"{int(hours)}h {int(mins)}m {int(secs)}s"

def save_results(results: Dict[str, Any], output_dir: Path, model_name: str, timestamp: str) -> Path:
    """Save benchmark results to JSON file"""
    results_file = output_dir / f"{model_name}_{timestamp}_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results_file
