@echo off
REM GGUF Benchmark Launcher
REM ----------------------

REM Activate virtual environment
call .\venv\Scripts\activate.bat

REM Set text colors
color 0B

echo =====================================
echo    GGUF MODEL BENCHMARK LAUNCHER
echo =====================================
echo.

REM Get model path
set /p MODEL_PATH="Enter path to GGUF model file: "

REM Get GPU layers
set /p GPU_LAYERS="Number of GPU layers (-1 for all) [default: -1]: "
if "%GPU_LAYERS%"=="" set GPU_LAYERS=-1

REM Get context size
set /p CONTEXT_SIZE="Context size [default: 2048]: "
if "%CONTEXT_SIZE%"=="" set CONTEXT_SIZE=2048

REM Get benchmarks
set /p BENCHMARKS="Benchmarks to run (space-separated) [default: mmlu arc_easy hellaswag]: "
if "%BENCHMARKS%"=="" set BENCHMARKS=mmlu arc_easy hellaswag

REM Create output directory
set OUTPUT_DIR=benchmark_results\run_%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set OUTPUT_DIR=%OUTPUT_DIR: =0%
mkdir %OUTPUT_DIR%

REM Run the benchmark
echo.
echo Running benchmark with the following parameters:
echo Model Path: %MODEL_PATH%
echo GPU Layers: %GPU_LAYERS%
echo Context Size: %CONTEXT_SIZE%
echo Benchmarks: %BENCHMARKS%
echo Output Directory: %OUTPUT_DIR%
echo.
echo Press any key to continue...
pause > nul

REM Run the Python script
python gguf_benchmark.py --model_path "%MODEL_PATH%" --output_dir "%OUTPUT_DIR%" --n_gpu_layers %GPU_LAYERS% --context_size %CONTEXT_SIZE% --benchmarks %BENCHMARKS%

REM Keep console open
echo.
echo Press any key to exit...
pause > nul
