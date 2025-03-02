#!/usr/bin/env python3
"""
Benchmark runner for GGUF models using lm-evaluation-harness
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from benchmark_utils import SpinnerAnimation, save_results
from model_adapter import LlamaCppAdapter

# Import colorama for terminal colors
import colorama
from colorama import Fore, Style

# For progress tracking
from tqdm import tqdm

# Initialize colorama
colorama.init()

logger = logging.getLogger(__name__)

class GGUFBenchmarkRunner:
    """
    Runner for benchmarking GGUF models with the lm-evaluation-harness
    """
    
    def __init__(self, 
                 model_path: str,
                 output_dir: str = "benchmark_results",
                 n_gpu_layers: int = -1,
                 context_size: int = 2048,
                 benchmarks: List[str] = None,
                 verbose: bool = False):
        """
        Initialize the benchmark runner
        
        Args:
            model_path: Path to the GGUF model file
            output_dir: Directory to save benchmark results
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            context_size: Context size for the model
            benchmarks: List of benchmarks to run
            verbose: Enable verbose logging
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_name = self.model_path.stem
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_gpu_layers = n_gpu_layers
        self.context_size = context_size
        self.verbose = verbose
        
        # Set up logging level based on verbose flag
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        # Model and results placeholders
        self.model = None
        self.results = {}
        
        # Default benchmarks to run if none specified
        self.default_benchmarks = ["mmlu", "hellaswag", "arc_easy"]
        self.benchmarks = benchmarks or self.default_benchmarks
        
        # Timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # System info will be populated when the model is loaded
        self.system_info = {
            "timestamp": self.timestamp,
            "model_path": str(self.model_path),
            "model_name": self.model_name,
            "n_gpu_layers": self.n_gpu_layers,
            "context_size": self.context_size,
        }
    
    def load_model(self) -> bool:
        """
        Load the GGUF model with GPU acceleration
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            from llama_cpp import Llama
            
            print(f"\n{Fore.CYAN}Loading model: {self.model_path}{Style.RESET_ALL}")
            print(f"GPU Layers: {self.n_gpu_layers}")
            
            # Create a progress bar for model loading
            loader = tqdm(
                total=100, 
                desc=f"{Fore.GREEN}Loading model{Style.RESET_ALL}",
                bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )
            
            # Use a background thread to update the progress bar
            import threading
            import random
            
            loading_complete = threading.Event()
            
            def update_progress():
                progress = 0
                while progress < 95 and not loading_complete.is_set():
                    sleep_time = random.uniform(0.1, 0.5)
                    time.sleep(sleep_time)
                    increment = random.uniform(0.5, 5)
                    progress += increment
                    progress = min(progress, 95)
                    loader.update(increment)
                    loader.refresh()
                
                if not loading_complete.is_set():
                    time.sleep(0.5)
            
            progress_thread = threading.Thread(target=update_progress)
            progress_thread.daemon = True
            progress_thread.start()
            
            try:
                # Actual model loading - catch any warnings or non-critical errors
                if self.verbose:
                    load_start = time.time()
                    # In verbose mode, show all output
                    self.model = Llama(
                        model_path=str(self.model_path),
                        n_gpu_layers=self.n_gpu_layers,
                        n_ctx=self.context_size,
                        verbose=self.verbose,
                        logits_all=True  # Need logits for benchmark
                    )
                    load_time = time.time() - load_start
                    logger.debug(f"Model loaded in {load_time:.2f} seconds")
                else:
                    # In normal mode, capture stdout/stderr
                    import contextlib
                    
                    @contextlib.contextmanager
                    def suppress_output():
                        # Suppress standard output and error
                        with open(os.devnull, 'w') as devnull:
                            old_stdout = sys.stdout
                            old_stderr = sys.stderr
                            sys.stdout = devnull
                            sys.stderr = devnull
                            try:
                                yield
                            finally:
                                sys.stdout = old_stdout
                                sys.stderr = old_stderr
                    
                    with suppress_output():
                        self.model = Llama(
                            model_path=str(self.model_path),
                            n_gpu_layers=self.n_gpu_layers,
                            n_ctx=self.context_size,
                            verbose=False,
                            logits_all=True  # Need logits for benchmark
                        )
                
                # Signal that loading is complete
                loading_complete.set()
                
                # Wait for progress thread to finish
                progress_thread.join(timeout=1.0)
                
                # Complete the progress bar
                remaining = 100 - loader.n
                loader.update(remaining)
                loader.close()
                
                # Update system info with model metadata
                import torch
                self.system_info.update({
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                })
                
                # Print successful load message
                print(f"{Fore.GREEN}✓ Model loaded successfully: {self.model_name}{Style.RESET_ALL}")
                return True
                
            except Exception as e:
                loading_complete.set()
                loader.close()
                print(f"{Fore.RED}✗ Error loading model: {e}{Style.RESET_ALL}")
                logger.error(f"Model loading error: {e}", exc_info=self.verbose)
                return False
                
        except ImportError:
            print(f"{Fore.RED}✗ llama-cpp-python not found. Please install it with CUDA support.{Style.RESET_ALL}")
            return False
    
    def run_benchmark(self, benchmark: str) -> Dict[str, Any]:
        """
        Run a single benchmark
        
        Args:
            benchmark: Name of benchmark to run
            
        Returns:
            Dict containing benchmark results
        """
        try:
            # Import here to avoid dependency issues if not installed
            from lm_eval import evaluator, tasks
            
            print(f"Initializing benchmark: {benchmark}")
            
            # Create spinner for tracking progress
            spinner = SpinnerAnimation(desc=f"Running {benchmark} benchmark")
            spinner.start()
            
            try:
                start_time = time.time()
                
                # Create adapter
                adapter = LlamaCppAdapter(self.model)
                
                # Register the adapter with lm-eval
                from lm_eval.api.registry import register_model
                register_model("llama-cpp-adapter", lambda **kwargs: adapter)
                
                # Run the evaluation with proper error handling
                # We'll use a try/except block to catch specific errors
                try:
                    # Set up additional args for evaluation
                    additional_args = {}
                    if self.verbose:
                        additional_args["verbosity"] = "DEBUG"
                    
                    # Run the evaluation with proper timeout handling
                    results = evaluator.simple_evaluate(
                        model="llama-cpp-adapter",
                        tasks=[benchmark],
                        batch_size=1,
                        device="cuda" if self.system_info.get("cuda_available", False) else "cpu",
                        **additional_args
                    )
                    
                    # Calculate elapsed time
                    elapsed_time = time.time() - start_time
                    
                    # Add elapsed time to results
                    results["elapsed_time"] = elapsed_time
                    
                    # Save results to temporary file
                    save_results(
                        {"system_info": self.system_info, 
                         "benchmarks": {benchmark: results}},
                        self.output_dir,
                        self.model_name,
                        self.timestamp
                    )
                    
                    # Stop the spinner
                    spinner.stop()
                    
                    # Print success message
                    print(f"{Fore.GREEN}✓ {benchmark} completed{Style.RESET_ALL}")
                    print(f"  Time taken: {elapsed_time:.2f} seconds")
                    
                    return results
                    
                except ValueError as e:
                    spinner.stop()
                    error_msg = str(e)
                    print(f"{Fore.RED}Error running benchmark {benchmark}: {error_msg}{Style.RESET_ALL}")
                    logger.error(f"Benchmark error: {error_msg}", exc_info=self.verbose)
                    return {"error": error_msg, "benchmark": benchmark, "elapsed_time": time.time() - start_time}
                    
                except Exception as e:
                    spinner.stop()
                    error_msg = str(e)
                    print(f"{Fore.RED}Error running benchmark {benchmark}: {error_msg}{Style.RESET_ALL}")
                    logger.error(f"Benchmark error: {error_msg}", exc_info=self.verbose)
                    return {"error": error_msg, "benchmark": benchmark, "elapsed_time": time.time() - start_time}
                
            except Exception as e:
                spinner.stop()
                error_msg = str(e)
                print(f"{Fore.RED}Error running benchmark {benchmark}: {error_msg}{Style.RESET_ALL}")
                logger.error(f"Benchmark error: {error_msg}", exc_info=self.verbose)
                return {"error": error_msg, "benchmark": benchmark}
                
        except ImportError:
            print(f"{Fore.RED}✗ lm-evaluation-harness not found. Please install it.{Style.RESET_ALL}")
            return {"error": "lm-evaluation-harness not installed", "benchmark": benchmark}
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """
        Run all specified benchmarks
        
        Returns:
            Dict containing all benchmark results
        """
        if self.model is None:
            success = self.load_model()
            if not success:
                return {"error": "Failed to load model"}
        
        self.results = {
            "system_info": self.system_info,
            "benchmarks": {}
        }
        
        total_start_time = time.time()
        
        # Create overall progress bar for all benchmarks
        print(f"\n{Fore.CYAN}Running {len(self.benchmarks)} benchmarks...{Style.RESET_ALL}")
        benchmarks_progress = tqdm(
            total=len(self.benchmarks),
            desc=f"{Fore.BLUE}Overall progress{Style.RESET_ALL}",
            bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        for i, benchmark in enumerate(self.benchmarks):
            try:
                # Create a separator for better visibility
                print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Running benchmark {i+1}/{len(self.benchmarks)}: {benchmark}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
                
                # Run the benchmark
                benchmark_results = self.run_benchmark(benchmark)
                self.results["benchmarks"][benchmark] = benchmark_results
                
                # Save intermediate results after each benchmark
                save_results(self.results, self.output_dir, self.model_name, self.timestamp)
                
                # Update overall progress
                benchmarks_progress.update(1)
                
            except Exception as e:
                print(f"{Fore.RED}✗ Error running benchmark {benchmark}: {e}{Style.RESET_ALL}")
                logger.error(f"Benchmark error: {e}", exc_info=self.verbose)
                self.results["benchmarks"][benchmark] = {"error": str(e)}
                benchmarks_progress.update(1)
        
        # Close the overall progress bar
        benchmarks_progress.close()
        
        total_elapsed_time = time.time() - total_start_time
        self.results["total_elapsed_time"] = total_elapsed_time
        
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}✓ All benchmarks completed in {total_elapsed_time:.2f} seconds{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        # Save final results
        save_results(self.results, self.output_dir, self.model_name, self.timestamp)
        
        return self.results
