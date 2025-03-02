#!/usr/bin/env python3
"""
GGUF Model Benchmark Runner
---------------------------
This program loads a GGUF model file, runs it on an NVIDIA GPU,
and benchmarks it on various LLM evaluation tasks.

Usage:
    python gguf_benchmark.py --model_path /path/to/model.gguf [options]

Requirements:
    pip install llama-cpp-python torch lm-eval tqdm plotly pandas colorama
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# Import colorama for terminal colors
import colorama
from colorama import Fore, Style

# Import our modules
from benchmark_utils import setup_logging, check_system_requirements, print_system_info
from benchmark_runner import GGUFBenchmarkRunner
from report_generator import BenchmarkReportGenerator

# Initialize colorama
colorama.init(autoreset=True)

def main():
    """Main function for the benchmark runner"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GGUF Model Benchmark Runner")
    parser.add_argument("--model_path", type=str, required=True, help="Path to GGUF model file")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", 
                       help="Directory to save benchmark results")
    parser.add_argument("--n_gpu_layers", type=int, default=-1, 
                       help="Number of layers to offload to GPU (-1 for all)")
    parser.add_argument("--context_size", type=int, default=2048, 
                       help="Context size for the model")
    parser.add_argument("--benchmarks", type=str, nargs="+", 
                       default=["mmlu", "arc_easy", "hellaswag"], 
                       help="Benchmarks to run")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose logging")
    parser.add_argument("--no_report", action="store_true",
                        help="Skip generating HTML report")
    
    args = parser.parse_args()
    
    # Setup logging based on verbose flag
    logger = setup_logging(args.verbose)
    
    # Ensure benchmarks is a list of individual benchmark names
    if args.benchmarks and len(args.benchmarks) == 1 and ' ' in args.benchmarks[0]:
        args.benchmarks = args.benchmarks[0].split()
    
    # Print welcome banner
    print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*25} GGUF BENCHMARK RUNNER {'='*25}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
    
    # Check system requirements
    system_info = check_system_requirements()
    print_system_info(system_info)
    
    if not system_info["llama_cpp_available"]:
        print(f"{Fore.RED}ERROR: llama-cpp-python is required but not found.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Please install it with CUDA support:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}pip install llama-cpp-python --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117{Style.RESET_ALL}")
        sys.exit(1)
    
    if not system_info["lm_eval_available"]:
        print(f"{Fore.RED}ERROR: lm-evaluation-harness is required but not found.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Please install it:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}pip install lm-eval{Style.RESET_ALL}")
        sys.exit(1)
    
    # Create timestamp-based output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Show configuration
        print(f"\n{Fore.CYAN}Benchmark Configuration:{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}Model Path:{Style.RESET_ALL} {args.model_path}")
        print(f"  {Fore.YELLOW}Output Directory:{Style.RESET_ALL} {output_dir}")
        print(f"  {Fore.YELLOW}GPU Layers:{Style.RESET_ALL} {args.n_gpu_layers}")
        print(f"  {Fore.YELLOW}Context Size:{Style.RESET_ALL} {args.context_size}")
        print(f"  {Fore.YELLOW}Benchmarks:{Style.RESET_ALL} {', '.join(args.benchmarks)}")
        print(f"  {Fore.YELLOW}Verbose Mode:{Style.RESET_ALL} {args.verbose}\n")
        
        # Prompt to confirm
        input(f"{Fore.YELLOW}Press Enter to start the benchmark...{Style.RESET_ALL}")
        
        # Initialize the benchmark runner
        print(f"\n{Fore.CYAN}Initializing benchmark...{Style.RESET_ALL}")
        benchmark = GGUFBenchmarkRunner(
            model_path=args.model_path,
            output_dir=output_dir,
            n_gpu_layers=args.n_gpu_layers,
            context_size=args.context_size,
            benchmarks=args.benchmarks,
            verbose=args.verbose
        )
        
        # Run all benchmarks
        results = benchmark.run_all_benchmarks()
        
        # Generate reports if not disabled
        if not args.no_report:
            # Get model name from path
            model_name = Path(args.model_path).stem
            
            # Create report generator
            report_generator = BenchmarkReportGenerator(
                results=results,
                output_dir=output_dir,
                model_name=model_name,
                timestamp=timestamp
            )
            
            # Generate detailed report
            report_path = report_generator.generate_detailed_report()
            
            # Try to open the report in a browser
            try:
                print(f"\n{Fore.CYAN}Opening report in web browser...{Style.RESET_ALL}")
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(report_path)}")
            except Exception as e:
                logger.warning(f"Could not open browser automatically: {e}")
                print(f"{Fore.YELLOW}Could not open browser automatically. Please open the report manually:{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}{report_path}{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ Benchmark completed successfully!{Style.RESET_ALL}")
        if not args.no_report:
            print(f"{Fore.GREEN}Detailed report available at: {report_path}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Benchmark interrupted by user.{Style.RESET_ALL}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n{Fore.RED}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.RED}✗ Error running benchmark: {e}{Style.RESET_ALL}")
        print(f"{Fore.RED}{'='*70}{Style.RESET_ALL}")
        
        if args.verbose:
            print(f"\n{Fore.RED}Error details:{Style.RESET_ALL}")
            traceback.print_exc()
        else:
            print(f"{Fore.YELLOW}Re-run with --verbose for more details{Style.RESET_ALL}")
        
        sys.exit(1)
    
    # Keep console open if running directly
    if __name__ == "__main__":
        input(f"\nPress any key to exit...")


if __name__ == "__main__":
    main()
