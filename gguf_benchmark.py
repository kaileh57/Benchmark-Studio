#!/usr/bin/env python3
"""
GGUF Model Benchmark Runner
---------------------------
This program loads a GGUF model file, runs it on an NVIDIA GPU,
and benchmarks it on various LLM evaluation tasks.

Usage:
    python gguf_benchmark.py --model_path /path/to/model.gguf [options]

Requirements:
    pip install llama-cpp-python torch lm-eval tqdm plotly pandas argparse colorama
"""

import os
import sys
import time
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# For progress bars and pretty output
from tqdm import tqdm
from tqdm.auto import trange
import colorama
from colorama import Fore, Style

# For HTML generation
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Initialize colorama for Windows
colorama.init()

# Check if CUDA is available
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        CUDA_DEVICE_COUNT = torch.cuda.device_count()
        CUDA_DEVICE_NAME = torch.cuda.get_device_name(0)
    else:
        CUDA_DEVICE_COUNT = 0
        CUDA_DEVICE_NAME = "N/A"
except ImportError:
    CUDA_AVAILABLE = False
    CUDA_DEVICE_COUNT = 0
    CUDA_DEVICE_NAME = "N/A"

# Import llama-cpp-python with CUDA support
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    print("Warning: llama-cpp-python not found. Please install with CUDA support:")
    print("pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir")


class GGUFBenchmark:
    """GGUF Model Benchmark Runner"""
    
    def __init__(self, model_path: str, output_dir: str = "benchmark_results",
                 n_gpu_layers: int = -1, context_size: int = 2048,
                 benchmarks: List[str] = None):
        """
        Initialize the benchmark runner
        
        Args:
            model_path: Path to the GGUF model file
            output_dir: Directory to save benchmark results
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            context_size: Context size for the model
            benchmarks: List of benchmarks to run
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_name = self.model_path.stem
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_gpu_layers = n_gpu_layers
        self.context_size = context_size
        
        self.model = None
        self.results = {}
        
        # Default benchmarks to run if none specified
        self.default_benchmarks = ["mmlu", "hellaswag", "arc_easy", "arc_challenge", 
                                  "truthfulqa_mc", "gsm8k"]
        
        self.benchmarks = benchmarks or self.default_benchmarks
        
        # Timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # System info
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        system_info = {
            "timestamp": self.timestamp,
            "cuda_available": CUDA_AVAILABLE,
            "cuda_device_count": CUDA_DEVICE_COUNT,
            "cuda_device_name": CUDA_DEVICE_NAME,
            "model_path": str(self.model_path),
            "model_name": self.model_name,
            "n_gpu_layers": self.n_gpu_layers,
            "context_size": self.context_size,
        }
        
        return system_info
    
    def load_model(self) -> None:
        """Load the GGUF model with GPU acceleration"""
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python not available. Cannot load model.")
        
        print(f"{Fore.CYAN}Loading model: {self.model_path}{Style.RESET_ALL}")
        print(f"GPU Layers: {self.n_gpu_layers}")
        
        # Dummy progress bar for model loading
        # This is an estimate since we can't track the actual loading process
        loader = tqdm(
            total=100, 
            desc=f"{Fore.GREEN}Loading model{Style.RESET_ALL}",
            bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        # Create a background thread to update the progress bar
        import threading
        import random
        import time
        
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
        
        loading_complete = threading.Event()
        progress_thread = threading.Thread(target=update_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        try:
            # Actual model loading
            self.model = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=self.n_gpu_layers,
                n_ctx=self.context_size,
                verbose=False
            )
            
            # Signal that loading is complete
            loading_complete.set()
            
            # Wait for progress thread to finish
            progress_thread.join(timeout=1.0)
            
            # Complete the progress bar
            remaining = 100 - loader.n
            loader.update(remaining)
            loader.close()
            
            print(f"{Fore.GREEN}✓ Model loaded successfully: {self.model_name}{Style.RESET_ALL}")
            
        except Exception as e:
            loading_complete.set()
            loader.close()
            print(f"{Fore.RED}✗ Error loading model: {e}{Style.RESET_ALL}")
            raise
    
    def create_lm_eval_adapter(self):
        """Create an adapter for the lm-eval harness"""
        # This is used to connect our model to EleutherAI's lm-evaluation-harness
        from lm_eval.api.model import LM
        from lm_eval.api.registry import register_model
        
        class LlamaCppAdapter(LM):
            def __init__(self, llama_model):
                self.model = llama_model
                
            def loglikelihood(self, requests):
                results = []
                for context, continuation in requests:
                    prompt = context
                    full_text = context + continuation
                    
                    # Get logits for the context
                    context_logits = self.model.eval(prompt)
                    
                    # Get logits for the context + continuation
                    full_logits = self.model.eval(full_text)
                    
                    # Calculate log likelihood
                    log_likelihood = full_logits - context_logits
                    
                    # Return is_greedy flag as well
                    is_greedy = True  # Simplified; would need to check if continuation has highest prob
                    
                    results.append((log_likelihood, is_greedy))
                
                return results
            
            def loglikelihood_rolling(self, requests):
                # This method can be more complex for efficiency
                # Simplified implementation for demonstration
                results = []
                for request in requests:
                    tokens = request
                    log_likelihood = 0.0
                    
                    for i in range(1, len(tokens)):
                        context = tokens[:i]
                        continuation = tokens[i:i+1]
                        
                        # Get token-by-token log likelihood
                        token_ll = self.model.eval(context + continuation) - self.model.eval(context)
                        log_likelihood += token_ll
                    
                    results.append(log_likelihood)
                
                return results

            def generate(self, context, max_tokens):
                return [self.model.generate(
                    prompt=ctx,
                    max_tokens=max_tokens,
                    temperature=0.0,  # Greedy decoding
                    stop=[]
                ) for ctx in context]
                
        # Register our adapter
        register_model("llama-cpp-adapter", LlamaCppAdapter)
        return LlamaCppAdapter(self.model)
    
    def run_lm_eval_benchmark(self, benchmark: str) -> Dict[str, Any]:
        """Run a benchmark using the EleutherAI lm-evaluation-harness"""
        try:
            from lm_eval import evaluator, tasks
            
            # Create a progress spinner since we can't directly track lm-eval progress
            spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            
            # Define a custom progress spinner
            class ProgressSpinner:
                def __init__(self, desc="Processing"):
                    self.desc = desc
                    self.index = 0
                    self.start_time = time.time()
                    self.is_running = False
                    self.thread = None
                
                def _spin(self):
                    while self.is_running:
                        elapsed = time.time() - self.start_time
                        mins, secs = divmod(int(elapsed), 60)
                        elapsed_str = f"{mins:02d}:{secs:02d}"
                        
                        char = spinner_chars[self.index % len(spinner_chars)]
                        sys.stdout.write(f"\r{Fore.CYAN}{char} {self.desc} {Style.RESET_ALL}[{elapsed_str}]")
                        sys.stdout.flush()
                        self.index += 1
                        time.sleep(0.1)
                
                def start(self):
                    self.is_running = True
                    import threading
                    self.thread = threading.Thread(target=self._spin)
                    self.thread.daemon = True
                    self.thread.start()
                
                def stop(self):
                    self.is_running = False
                    if self.thread:
                        self.thread.join(timeout=1.0)
                    sys.stdout.write("\r" + " " * 100 + "\r")  # Clear the line
                    sys.stdout.flush()
            
            # Start the progress spinner
            print(f"Initializing benchmark: {benchmark}")
            spinner = ProgressSpinner(desc=f"Running {benchmark} benchmark")
            spinner.start()
            
            start_time = time.time()
            
            # Create our adapter
            adapter = self.create_lm_eval_adapter()
            
            # Run the evaluation
            results = evaluator.simple_evaluate(
                model="llama-cpp-adapter",
                model_args=adapter,
                tasks=[benchmark],
                batch_size=1,
                device="cuda" if CUDA_AVAILABLE else "cpu",
                no_cache=True
            )
            
            # Stop the spinner
            spinner.stop()
            
            elapsed_time = time.time() - start_time
            
            # Add elapsed time to results
            results["elapsed_time"] = elapsed_time
            
            return results
        
        except Exception as e:
            print(f"{Fore.RED}Error running benchmark {benchmark}: {e}{Style.RESET_ALL}")
            return {"error": str(e), "benchmark": benchmark}
    
    def run_benchmarks(self) -> Dict[str, Any]:
        """Run all specified benchmarks"""
        if self.model is None:
            self.load_model()
        
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
            position=0,
            bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        
        for i, benchmark in enumerate(self.benchmarks):
            try:
                # Create a separator for better visibility
                print(f"\n{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Running benchmark {i+1}/{len(self.benchmarks)}: {benchmark}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}{'='*60}{Style.RESET_ALL}")
                
                # Run the benchmark
                benchmark_results = self.run_lm_eval_benchmark(benchmark)
                self.results["benchmarks"][benchmark] = benchmark_results
                
                # Save intermediate results after each benchmark
                self._save_results()
                
                # Display a summary of the benchmark results
                accuracy = benchmark_results.get("results", {}).get("accuracy", 
                          benchmark_results.get("results", {}).get("acc", 
                          benchmark_results.get("results", {}).get("exact_match", 0)))
                
                if accuracy:
                    accuracy_str = f"{accuracy*100:.2f}%"
                    print(f"{Fore.GREEN}✓ {benchmark} completed - Accuracy: {accuracy_str}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.GREEN}✓ {benchmark} completed{Style.RESET_ALL}")
                
                print(f"{Fore.GREEN}  Time taken: {benchmark_results.get('elapsed_time', 0):.2f} seconds{Style.RESET_ALL}")
                
                # Update overall progress
                benchmarks_progress.update(1)
                
            except Exception as e:
                print(f"{Fore.RED}✗ Error running benchmark {benchmark}: {e}{Style.RESET_ALL}")
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
        self._save_results()
        
        return self.results
    
    def _save_results(self) -> None:
        """Save benchmark results to JSON file"""
        results_file = self.output_dir / f"{self.model_name}_{self.timestamp}_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
    
    def generate_html_report(self) -> str:
        """Generate an HTML report with charts for benchmark results"""
        html_file = self.output_dir / f"{self.model_name}_{self.timestamp}_report.html"
        
        # Process results into a format suitable for visualization
        benchmark_names = []
        accuracy_scores = []
        elapsed_times = []
        
        for benchmark, results in self.results.get("benchmarks", {}).items():
            benchmark_names.append(benchmark)
            
            # Extract accuracy metrics - structure might vary by benchmark
            accuracy = results.get("results", {}).get("accuracy", 
                      results.get("results", {}).get("acc", 
                      results.get("results", {}).get("exact_match", 0)))
            
            accuracy_scores.append(accuracy * 100 if accuracy else 0)  # Convert to percentage
            elapsed_times.append(results.get("elapsed_time", 0))
        
        # Create a subplot with 2 rows
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Benchmark Accuracy (%)", "Benchmark Runtime (seconds)"),
            vertical_spacing=0.3
        )
        
        # Add accuracy bar chart
        fig.add_trace(
            go.Bar(
                x=benchmark_names,
                y=accuracy_scores,
                marker_color='royalblue',
                name="Accuracy (%)"
            ),
            row=1, col=1
        )
        
        # Add runtime bar chart
        fig.add_trace(
            go.Bar(
                x=benchmark_names,
                y=elapsed_times,
                marker_color='firebrick',
                name="Runtime (s)"
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title_text=f"Benchmark Results for {self.model_name}",
            height=800,
            showlegend=False,
        )
        
        # Add system info as annotations
        system_info_text = f"""
        <b>System Information:</b><br>
        Model: {self.model_name}<br>
        GPU: {self.system_info['cuda_device_name']}<br>
        GPU Layers: {self.system_info['n_gpu_layers']}<br>
        Context Size: {self.system_info['context_size']}<br>
        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=1.15,
            text=system_info_text,
            showarrow=False,
            font=dict(size=12),
            align="center",
        )
        
        # Write to HTML file
        with open(html_file, 'w') as f:
            f.write(fig.to_html(include_plotlyjs='cdn'))
        
        print(f"HTML report generated: {html_file}")
        return str(html_file)
    
    def generate_html_report_detailed(self) -> str:
        """Generate a more detailed HTML report with charts and tables"""
        html_file = self.output_dir / f"{self.model_name}_{self.timestamp}_detailed_report.html"
        
        # Show progress for report generation
        print(f"\n{Fore.CYAN}Generating detailed HTML report...{Style.RESET_ALL}")
        report_progress = tqdm(
            total=5,  # 5 steps in report generation
            desc=f"{Fore.BLUE}Report generation{Style.RESET_ALL}",
            bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt}"
        )
        
        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>GGUF Model Benchmark Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                }
                .system-info {
                    background-color: #f9f9f9;
                    border-radius: 10px;
                    padding: 20px;
                    margin-bottom: 30px;
                }
                .chart-container {
                    margin-bottom: 40px;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 30px;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
                tr:nth-child(even) {
                    background-color: #f9f9f9;
                }
                .benchmark-section {
                    margin-bottom: 40px;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 20px;
                }
                .footer {
                    text-align: center;
                    font-size: 0.8em;
                    color: #666;
                    margin-top: 30px;
                }
                .progress-bar {
                    height: 20px;
                    background-color: #e0e0e0;
                    border-radius: 10px;
                    margin-bottom: 10px;
                    overflow: hidden;
                }
                .progress-fill {
                    height: 100%;
                    background-color: #4CAF50;
                    border-radius: 10px;
                    transition: width 0.5s;
                }
                .good-score {
                    color: #388E3C;
                    font-weight: bold;
                }
                .medium-score {
                    color: #FFA000;
                    font-weight: bold;
                }
                .low-score {
                    color: #D32F2F;
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>GGUF Model Benchmark Report</h1>
                    <h2>{{model_name}}</h2>
                    <p>Generated on {{timestamp}}</p>
                </div>
                
                <div class="system-info">
                    <h3>System Information</h3>
                    <table>
                        <tr><th>Model Path</th><td>{{model_path}}</td></tr>
                        <tr><th>GPU Available</th><td>{{cuda_available}}</td></tr>
                        <tr><th>GPU Device</th><td>{{cuda_device_name}}</td></tr>
                        <tr><th>GPU Layers</th><td>{{n_gpu_layers}}</td></tr>
                        <tr><th>Context Size</th><td>{{context_size}}</td></tr>
                        <tr><th>Total Runtime</th><td>{{total_elapsed_time}} seconds</td></tr>
                    </table>
                </div>
                
                <div class="chart-container">
                    <h3>Benchmark Results Overview</h3>
                    <div id="accuracy-chart" style="height: 400px;"></div>
                </div>
                
                <div class="chart-container">
                    <h3>Benchmark Runtime</h3>
                    <div id="runtime-chart" style="height: 300px;"></div>
                </div>
                
                <h3>Detailed Benchmark Results</h3>
                {{benchmark_details}}
                
                <div class="footer">
                    <p>Generated by GGUF Model Benchmark Runner</p>
                </div>
            </div>
            
            {{chart_scripts}}
        </body>
        </html>
        """
        
        # Process results for visualization
        benchmark_names = []
        accuracy_scores = []
        elapsed_times = []
        
        for benchmark, results in self.results.get("benchmarks", {}).items():
            benchmark_names.append(benchmark)
            
            # Extract accuracy metrics
            accuracy = results.get("results", {}).get("accuracy", 
                      results.get("results", {}).get("acc", 
                      results.get("results", {}).get("exact_match", 0)))
            
            accuracy_scores.append(accuracy * 100 if accuracy else 0)  # Convert to percentage
            elapsed_times.append(results.get("elapsed_time", 0))
        
        # Update progress
        report_progress.update(1)
        report_progress.set_description(f"{Fore.BLUE}Processing benchmark data{Style.RESET_ALL}")
        
        # Generate benchmark details HTML with visual progress bars
        benchmark_details_html = ""
        for benchmark, results in self.results.get("benchmarks", {}).items():
            benchmark_details_html += f"""
            <div class="benchmark-section">
                <h4>{benchmark}</h4>
                <table>
                    <tr><th>Elapsed Time</th><td>{results.get('elapsed_time', 'N/A')} seconds</td></tr>
            """
            
            # Add all metrics from results
            if "results" in results:
                for metric, value in results["results"].items():
                    if isinstance(value, float):
                        # Format percentages
                        if 0 <= value <= 1:
                            # Add color coding and progress bar for percentage values
                            percentage = value * 100
                            color_class = "low-score"
                            if percentage >= 80:
                                color_class = "good-score"
                            elif percentage >= 50:
                                color_class = "medium-score"
                                
                            formatted_value = f"""
                            <div class="{color_class}">{percentage:.2f}%</div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: {percentage}%;"></div>
                            </div>
                            """
                        else:
                            formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    
                    benchmark_details_html += f"""
                    <tr><th>{metric}</th><td>{formatted_value}</td></tr>
                    """
            
            benchmark_details_html += """
                </table>
            </div>
            """
            
        # Update progress
        report_progress.update(1)
        report_progress.set_description(f"{Fore.BLUE}Creating charts{Style.RESET_ALL}")
        
        # Generate chart scripts
        chart_scripts = f"""
        <script>
            // Accuracy Chart
            var accuracyData = {{
                x: {json.dumps(benchmark_names)},
                y: {json.dumps(accuracy_scores)},
                type: 'bar',
                marker: {{
                    color: 'rgba(58, 71, 191, 0.6)',
                    line: {{
                        color: 'rgba(58, 71, 191, 1.0)',
                        width: 1
                    }}
                }},
                name: 'Accuracy (%)'
            }};
            
            var accuracyLayout = {{
                title: 'Benchmark Accuracy (%)',
                xaxis: {{
                    title: 'Benchmark'
                }},
                yaxis: {{
                    title: 'Accuracy (%)',
                    range: [0, 100]
                }}
            }};
            
            Plotly.newPlot('accuracy-chart', [accuracyData], accuracyLayout);
            
            // Runtime Chart
            var runtimeData = {{
                x: {json.dumps(benchmark_names)},
                y: {json.dumps(elapsed_times)},
                type: 'bar',
                marker: {{
                    color: 'rgba(191, 71, 58, 0.6)',
                    line: {{
                        color: 'rgba(191, 71, 58, 1.0)',
                        width: 1
                    }}
                }},
                name: 'Runtime (seconds)'
            }};
            
            var runtimeLayout = {{
                title: 'Benchmark Runtime (seconds)',
                xaxis: {{
                    title: 'Benchmark'
                }},
                yaxis: {{
                    title: 'Time (seconds)'
                }}
            }};
            
            Plotly.newPlot('runtime-chart', [runtimeData], runtimeLayout);
        </script>
        """
        
        # Fill template
        html_content = html_template.replace(
            "{{model_name}}", self.model_name
        ).replace(
            "{{timestamp}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ).replace(
            "{{model_path}}", str(self.model_path)
        ).replace(
            "{{cuda_available}}", str(self.system_info["cuda_available"])
        ).replace(
            "{{cuda_device_name}}", str(self.system_info["cuda_device_name"])
        ).replace(
            "{{n_gpu_layers}}", str(self.system_info["n_gpu_layers"])
        ).replace(
            "{{context_size}}", str(self.system_info["context_size"])
        ).replace(
            "{{total_elapsed_time}}", f"{self.results.get('total_elapsed_time', 0):.2f}"
        ).replace(
            "{{benchmark_details}}", benchmark_details_html
        ).replace(
            "{{chart_scripts}}", chart_scripts
        )
        
        # Update progress
        report_progress.update(1)
        report_progress.set_description(f"{Fore.BLUE}Writing HTML file{Style.RESET_ALL}")
        
        # Write to HTML file
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        # Update progress
        report_progress.update(1)
        report_progress.set_description(f"{Fore.BLUE}Finalizing report{Style.RESET_ALL}")
        time.sleep(0.5)  # Small delay for visual effect
        
        # Complete progress
        report_progress.update(1)
        report_progress.close()
        
        print(f"{Fore.GREEN}✓ Detailed HTML report generated: {html_file}{Style.RESET_ALL}")
        return str(html_file)


def main():
    """Main entry point"""
    # Initialize colorama
    colorama.init(autoreset=True)
    
    # Print welcome banner
    print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*25} GGUF BENCHMARK RUNNER {'='*25}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
    
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
    
    args = parser.parse_args()
    
    # Check system requirements with nice formatting
    print(f"{Fore.YELLOW}Checking system requirements...{Style.RESET_ALL}")
    
    if not CUDA_AVAILABLE:
        print(f"{Fore.RED}WARNING: CUDA is not available. The benchmark will run on CPU only.{Style.RESET_ALL}")
        print(f"{Fore.RED}This will be significantly slower.{Style.RESET_ALL}")
        response = input(f"{Fore.YELLOW}Continue without GPU? (y/n): {Style.RESET_ALL}")
        if response.lower() not in ["y", "yes"]:
            print(f"{Fore.RED}Exiting.{Style.RESET_ALL}")
            sys.exit(0)
    else:
        print(f"{Fore.GREEN}✓ CUDA is available: {CUDA_DEVICE_NAME}{Style.RESET_ALL}")
    
    if not LLAMA_CPP_AVAILABLE:
        print(f"{Fore.RED}ERROR: llama-cpp-python is required but not found.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Please install it with CUDA support:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}pip install llama-cpp-python --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu117{Style.RESET_ALL}")
        sys.exit(1)
    else:
        print(f"{Fore.GREEN}✓ llama-cpp-python is installed{Style.RESET_ALL}")
    
    print(f"{Fore.GREEN}✓ All requirements met{Style.RESET_ALL}\n")
    
    try:
        # Show configuration
        print(f"{Fore.CYAN}Benchmark Configuration:{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}Model Path:{Style.RESET_ALL} {args.model_path}")
        print(f"  {Fore.YELLOW}Output Directory:{Style.RESET_ALL} {args.output_dir}")
        print(f"  {Fore.YELLOW}GPU Layers:{Style.RESET_ALL} {args.n_gpu_layers}")
        print(f"  {Fore.YELLOW}Context Size:{Style.RESET_ALL} {args.context_size}")
        print(f"  {Fore.YELLOW}Benchmarks:{Style.RESET_ALL} {', '.join(args.benchmarks)}\n")
        
        # Prompt to confirm
        input(f"{Fore.YELLOW}Press Enter to start the benchmark...{Style.RESET_ALL}")
        
        # Create a spinner while creating the benchmark object
        print(f"\n{Fore.CYAN}Initializing benchmark...{Style.RESET_ALL}")
        # Create and run the benchmark
        benchmark = GGUFBenchmark(
            model_path=args.model_path,
            output_dir=args.output_dir,
            n_gpu_layers=args.n_gpu_layers,
            context_size=args.context_size,
            benchmarks=args.benchmarks
        )
        
        # Load the model
        benchmark.load_model()
        
        # Run the benchmarks
        benchmark.run_benchmarks()
        
        # Generate HTML report
        report_path = benchmark.generate_html_report_detailed()
        
        print(f"\n{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}✓ Benchmark completed successfully!{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Detailed report available at: {report_path}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'='*70}{Style.RESET_ALL}")
        
        # Try to open the report in a browser
        try:
            print(f"\n{Fore.CYAN}Opening report in web browser...{Style.RESET_ALL}")
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
        except Exception as e:
            print(f"{Fore.YELLOW}Could not open browser automatically: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Please open the report manually at: {report_path}{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"\n{Fore.RED}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.RED}✗ Error running benchmark: {e}{Style.RESET_ALL}")
        print(f"{Fore.RED}{'='*70}{Style.RESET_ALL}")
        
        import traceback
        print(f"\n{Fore.RED}Error details:{Style.RESET_ALL}")
        traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    main()
