#!/usr/bin/env python3
"""
HTML report generator for GGUF model benchmarking
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# For progress tracking
from tqdm import tqdm
import colorama
from colorama import Fore, Style

logger = logging.getLogger(__name__)

class BenchmarkReportGenerator:
    """
    Generates HTML reports from benchmark results
    """
    
    def __init__(self, results: Dict[str, Any], output_dir: Path, model_name: str, timestamp: str):
        """
        Initialize the report generator
        
        Args:
            results: Benchmark results dict
            output_dir: Directory to save reports
            model_name: Name of the model
            timestamp: Timestamp for the report
        """
        self.results = results
        self.output_dir = output_dir
        self.model_name = model_name
        self.timestamp = timestamp
    
    def generate_simple_report(self) -> str:
        """
        Generate a simple HTML report
        
        Returns:
            Path to generated HTML file
        """
        html_file = self.output_dir / f"{self.model_name}_{self.timestamp}_report.html"
        
        # Process results for visualization
        benchmark_names = []
        accuracy_scores = []
        elapsed_times = []
        
        for benchmark, results in self.results.get("benchmarks", {}).items():
            benchmark_names.append(benchmark)
            
            # Extract accuracy metrics
            accuracy = 0.0
            if "results" in results:
                for metric_name in ["accuracy", "acc", "exact_match"]:
                    if metric_name in results["results"]:
                        accuracy = results["results"][metric_name]
                        break
            
            accuracy_scores.append(accuracy * 100 if accuracy else 0)  # Convert to percentage
            elapsed_times.append(results.get("elapsed_time", 0))
        
        # Simple HTML template
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GGUF Model Benchmark Report - {self.model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #2c3e50; }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>GGUF Model Benchmark Report</h1>
                <h2>{self.model_name}</h2>
                <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <h3>System Information</h3>
                <table>
                    <tr><th>Model Path</th><td>{self.results.get("system_info", {}).get("model_path", "N/A")}</td></tr>
                    <tr><th>GPU Available</th><td>{self.results.get("system_info", {}).get("cuda_available", "N/A")}</td></tr>
                    <tr><th>GPU Device</th><td>{self.results.get("system_info", {}).get("cuda_device_name", "N/A")}</td></tr>
                    <tr><th>GPU Layers</th><td>{self.results.get("system_info", {}).get("n_gpu_layers", "N/A")}</td></tr>
                    <tr><th>Context Size</th><td>{self.results.get("system_info", {}).get("context_size", "N/A")}</td></tr>
                    <tr><th>Total Runtime</th><td>{self.results.get("total_elapsed_time", 0):.2f} seconds</td></tr>
                </table>
                
                <h3>Benchmark Results</h3>
                <table>
                    <tr>
                        <th>Benchmark</th>
                        <th>Accuracy (%)</th>
                        <th>Time (s)</th>
                    </tr>
        """
        
        # Add rows for each benchmark
        for i, benchmark in enumerate(benchmark_names):
            html_content += f"""
                    <tr>
                        <td>{benchmark}</td>
                        <td>{accuracy_scores[i]:.2f}%</td>
                        <td>{elapsed_times[i]:.2f}</td>
                    </tr>
            """
        
        # Close the table and HTML
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"{Fore.GREEN}✓ Simple HTML report generated: {html_file}{Style.RESET_ALL}")
        return str(html_file)
    
    def generate_detailed_report(self) -> str:
        """
        Generate a detailed HTML report with interactive charts
        
        Returns:
            Path to generated HTML file
        """
        html_file = self.output_dir / f"{self.model_name}_{self.timestamp}_detailed_report.html"
        
        # Show progress for report generation
        print(f"\n{Fore.CYAN}Generating detailed HTML report...{Style.RESET_ALL}")
        report_progress = tqdm(
            total=5,  # 5 steps in report generation
            desc=f"{Fore.BLUE}Finalizing report{Style.RESET_ALL}"
        )
        
        # Extract system info
        system_info = self.results.get("system_info", {})
        total_time = self.results.get("total_elapsed_time", 0)
        
        # Process results for visualization
        benchmark_names = []
        accuracy_scores = []
        elapsed_times = []
        
        for benchmark, results in self.results.get("benchmarks", {}).items():
            benchmark_names.append(benchmark)
            
            # Extract accuracy metrics
            accuracy = 0.0
            if "results" in results:
                for metric_name in ["accuracy", "acc", "exact_match"]:
                    if metric_name in results.get("results", {}):
                        accuracy = results["results"][metric_name]
                        break
            
            accuracy_scores.append(accuracy * 100 if accuracy else 0)  # Convert to percentage
            elapsed_times.append(results.get("elapsed_time", 0))
        
        report_progress.update(1)
        
        # Generate HTML template with interactive charts using Plotly
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>GGUF Model Benchmark Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
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
        
        report_progress.update(1)
        
        # Generate benchmark details HTML
        benchmark_details_html = ""
        for benchmark, results in self.results.get("benchmarks", {}).items():
            benchmark_details_html += f"""
            <div class="benchmark-section">
                <h4>{benchmark}</h4>
                <table>
                    <tr><th>Elapsed Time</th><td>{results.get('elapsed_time', 'N/A')} seconds</td></tr>
            """
            
            # Check if there was an error
            if "error" in results:
                benchmark_details_html += f"""
                    <tr><th>Error</th><td style="color: #D32F2F;">{results["error"]}</td></tr>
                """
            
            # Add all metrics from results
            elif "results" in results:
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
        
        report_progress.update(1)
        
        # Generate chart scripts with Plotly
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
        
        report_progress.update(1)
        
        # Fill template
        html_content = html_template.replace(
            "{{model_name}}", self.model_name
        ).replace(
            "{{timestamp}}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ).replace(
            "{{model_path}}", str(system_info.get("model_path", "N/A"))
        ).replace(
            "{{cuda_available}}", str(system_info.get("cuda_available", "N/A"))
        ).replace(
            "{{cuda_device_name}}", str(system_info.get("cuda_device_name", "N/A"))
        ).replace(
            "{{n_gpu_layers}}", str(system_info.get("n_gpu_layers", "N/A"))
        ).replace(
            "{{context_size}}", str(system_info.get("context_size", "N/A"))
        ).replace(
            "{{total_elapsed_time}}", f"{total_time:.2f}"
        ).replace(
            "{{benchmark_details}}", benchmark_details_html
        ).replace(
            "{{chart_scripts}}", chart_scripts
        )
        
        # Write to file
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        report_progress.update(1)
        report_progress.close()
        
        print(f"{Fore.GREEN}✓ Detailed HTML report generated: {html_file}{Style.RESET_ALL}")
        return str(html_file)
