#!/usr/bin/env python3
"""
Verify llama-cpp-python installation and CUDA support in WSL environment
"""

import os
import sys
import time
import importlib.util
import subprocess
import platform
from pathlib import Path

# ANSI color codes for terminal output
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
NC = "\033[0m"  # No Color
BOLD = "\033[1m"

def print_header(message):
    print(f"\n{BLUE}{BOLD}{'=' * 70}{NC}")
    print(f"{BLUE}{BOLD}{message.center(70)}{NC}")
    print(f"{BLUE}{BOLD}{'=' * 70}{NC}\n")

def print_section(message):
    print(f"\n{CYAN}{BOLD}{message}{NC}")
    print(f"{CYAN}{'-' * len(message)}{NC}")

def print_success(message):
    print(f"{GREEN}✓ {message}{NC}")

def print_warning(message):
    print(f"{YELLOW}⚠ {message}{NC}")

def print_error(message):
    print(f"{RED}✗ {message}{NC}")

def print_info(message):
    print(f"{BLUE}ℹ {message}{NC}")

def run_command(command):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            check=False  # Don't raise exception on non-zero exit
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def check_wsl_environment():
    """Check WSL-specific environment for CUDA compatibility"""
    print_section("Checking WSL Environment")
    
    # Check if we're actually running in WSL
    is_wsl = False
    try:
        # Check if /proc/version contains Microsoft or WSL
        with open('/proc/version', 'r') as f:
            proc_version = f.read().lower()
            is_wsl = 'microsoft' in proc_version or 'wsl' in proc_version
    except:
        pass
    
    if not is_wsl:
        print_warning("Not running in WSL environment, skipping WSL-specific checks")
        return
    
    print_success("Detected WSL environment")
    
    # Check WSL version
    stdout, stderr, _ = run_command("wsl --version 2>/dev/null || echo 'WSL version command not available'")
    if "WSL version command not available" in stdout:
        # Try alternative method
        stdout, stderr, _ = run_command("cat /proc/version")
        if stdout:
            print_info(f"WSL version info: {stdout}")
        else:
            print_warning("Could not determine WSL version")
    else:
        print_info(f"WSL version info: {stdout}")
    
    # Check if nvidia-smi is available
    stdout, stderr, exit_code = run_command("which nvidia-smi")
    if exit_code != 0:
        print_error("nvidia-smi not found. NVIDIA driver may not be installed in WSL")
        print_info("See: https://docs.nvidia.com/cuda/wsl-user-guide/index.html")
        return
    
    # Check GPU visibility in WSL
    stdout, stderr, exit_code = run_command("nvidia-smi")
    if exit_code != 0:
        print_error(f"nvidia-smi failed: {stderr}")
        print_info("NVIDIA driver may not be properly installed or configured in WSL")
        return
    
    # Extract relevant information from nvidia-smi output
    print_success("NVIDIA driver accessible in WSL")
    
    # Try to extract driver version
    driver_version = "Unknown"
    if "Driver Version:" in stdout:
        driver_version = stdout.split("Driver Version:")[1].split()[0]
    
    print_info(f"NVIDIA driver version: {driver_version}")
    
    # Check CUDA version via nvcc if available
    stdout, stderr, exit_code = run_command("nvcc --version")
    if exit_code != 0:
        print_warning("nvcc not found. CUDA toolkit may not be installed in WSL")
        print_info("This is OK if using pre-built wheels, but building from source requires CUDA toolkit")
    else:
        cuda_version = "Unknown"
        for line in stdout.splitlines():
            if "release" in line.lower() and "V" in line:
                cuda_version = line.split("V")[1].split()[0]
                break
        print_info(f"CUDA toolkit version: {cuda_version}")
    
    # Check CUDA libraries
    stdout, stderr, exit_code = run_command("ldconfig -p | grep -i cuda | wc -l")
    if exit_code == 0 and int(stdout.strip() or 0) > 0:
        print_success(f"Found {stdout.strip()} CUDA libraries in system path")
    else:
        print_warning("No CUDA libraries found in system path")
    
    # Check CUDA environment variables
    cuda_path = os.environ.get("CUDA_PATH", "Not set")
    cuda_home = os.environ.get("CUDA_HOME", "Not set")
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "Not set")
    
    print_info(f"CUDA_PATH: {cuda_path}")
    print_info(f"CUDA_HOME: {cuda_home}")
    print_info(f"LD_LIBRARY_PATH: {ld_library_path}")

def check_package_installation():
    """Check if llama-cpp-python is installed and get version info"""
    print_section("Checking llama-cpp-python Installation")
    
    # Check if the package is installed
    try:
        spec = importlib.util.find_spec("llama_cpp")
        if spec is None:
            print_error("llama-cpp-python is NOT installed")
            return False
        
        print_success("llama-cpp-python is installed")
        package_path = spec.origin
        print_info(f"Package location: {package_path}")
        
        # Try to get version information
        try:
            from llama_cpp import __version__
            print_info(f"Version: {__version__}")
        except (ImportError, AttributeError):
            try:
                result = subprocess.run([sys.executable, "-m", "pip", "show", "llama-cpp-python"], 
                                      capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        print_info(f"Version: {line.split(':')[1].strip()}")
                        break
            except Exception:
                print_warning("Could not determine version")
        
        # Check if it was built from source or installed as a wheel
        try:
            # Most wheels have 'cp' in their path for the Python version
            if 'cp3' in package_path and ('linux_x86_64' in package_path or 'win_amd64' in package_path):
                print_info("Installed from pre-built wheel")
                
                # Try to determine if it has CUDA support
                if 'cuda' in package_path.lower():
                    print_info("Wheel appears to include CUDA support")
            else:
                print_info("Likely built from source")
        except:
            pass
        
        return True
    
    except Exception as e:
        print_error(f"Error checking installation: {e}")
        return False

def check_cuda_support():
    """Check if llama-cpp-python has CUDA support"""
    print_section("Checking CUDA Support")
    
    # First, check if torch with CUDA is available as a reference
    try:
        import torch
        torch_cuda_available = torch.cuda.is_available()
        if torch_cuda_available:
            torch_cuda_device = torch.cuda.get_device_name(0)
            print_info(f"PyTorch CUDA: Available - {torch_cuda_device}")
            print_info(f"CUDA Version: {torch.version.cuda}")
            # Test CUDA by creating a small tensor
            try:
                _ = torch.ones(1).cuda()
                print_success("PyTorch CUDA test successful")
            except Exception as e:
                print_warning(f"PyTorch CUDA test failed: {e}")
        else:
            print_warning("PyTorch CUDA: Not available")
    except ImportError:
        print_warning("PyTorch not installed, skipping reference CUDA check")
    
    # Now check if llama-cpp-python has CUDA support
    try:
        from llama_cpp import Llama
        
        # Check if _STD_CUDA_GPU_LAYERS attribute exists which indicates CUDA support
        has_cuda_attr = hasattr(Llama, "_STD_CUDA_GPU_LAYERS")
        print_info(f"Has CUDA attributes: {has_cuda_attr}")
        
        # Look at the library file to check for CUDA symbols
        lib_file = getattr(Llama, "__file__", None)
        if lib_file:
            # Check if the library contains CUDA symbols using ldd
            stdout, stderr, exit_code = run_command(f"ldd {lib_file} | grep -i cuda")
            if exit_code == 0 and stdout:
                print_success("Library links to CUDA libraries:")
                for line in stdout.splitlines():
                    print_info(f"  {line.strip()}")
            else:
                print_warning("Library doesn't appear to link to CUDA libraries")
        
        # Try to create a minimal Llama object with n_gpu_layers set to check for CUDA errors
        print_info("Testing GPU initialization with minimal model...")
        try:
            # This doesn't actually load a model, it just tests the GPU layer initialization
            Llama(model_path=None, n_gpu_layers=1, verbose=True)
            print_error("CUDA test unexpected behavior: Initialization should have raised an exception since no model was provided")
        except ValueError as e:
            # Expected to fail with "failed to load model" which means the CUDA part was OK
            if "failed to load model" in str(e).lower():
                print_success("CUDA initialization test passed")
            else:
                print_error(f"Unexpected error: {e}")
        except Exception as e:
            print_error(f"CUDA test failed: {e}")
    
    except Exception as e:
        print_error(f"Error checking CUDA support: {e}")
        return False

def test_small_inference(model_path=None):
    """Test inference on a small model file if available"""
    print_section("Testing Inference")
    
    # If model path wasn't provided, try to find one
    if not model_path:
        # Try to locate a .gguf model file for testing
        search_paths = [
            ".",  # Current directory
            "./models",  # Models subdirectory
            os.path.expanduser("~/.cache/llama-cpp"),  # Common cache directory
            os.path.expanduser("~/.cache/lm-evaluation-harness"),  # lm-eval cache
        ]
        
        for path in search_paths:
            if not os.path.exists(path):
                continue
            
            for file in os.listdir(path):
                if file.endswith(".gguf"):
                    model_path = os.path.join(path, file)
                    break
            
            if model_path:
                break
    
    if not model_path or not os.path.exists(model_path):
        print_warning("No .gguf model file found for testing inference")
        print_info("To test inference, provide a path to a .gguf model file:")
        print_info("python verify_llama_cpp_wsl.py /path/to/model.gguf")
        return
    
    try:
        print_info(f"Using model: {model_path}")
        from llama_cpp import Llama
        
        # Try to import torch for memory checks
        has_torch = False
        try:
            import torch
            if torch.cuda.is_available():
                has_torch = True
                torch.cuda.empty_cache()
                before_mem = torch.cuda.memory_allocated(0)
                print_info(f"GPU memory before loading: {before_mem/1024**2:.2f} MB")
        except ImportError:
            print_warning("PyTorch not available for GPU memory checking")
        
        print_info("First testing with CPU only (n_gpu_layers=0)...")
        
        # Test CPU inference
        start_time = time.time()
        model_cpu = Llama(
            model_path=model_path,
            n_gpu_layers=0,  # CPU only
            n_ctx=512,
            verbose=False,
        )
        cpu_load_time = time.time() - start_time
        print_success(f"CPU model loaded in {cpu_load_time:.2f} seconds")
        
        # Simple CPU inference test
        start_time = time.time()
        cpu_result = model_cpu("Hello, my name is", max_tokens=5)
        cpu_infer_time = time.time() - start_time
        print_success(f"CPU inference completed in {cpu_infer_time:.2f} seconds")
        print_info(f"CPU result: {cpu_result}")
        
        # Free CPU model
        del model_cpu
        
        # Test GPU inference
        print_info("Now testing with GPU (n_gpu_layers=-1)...")
        
        # Check GPU memory before loading if available
        if has_torch:
            torch.cuda.empty_cache()
            before_gpu_mem = torch.cuda.memory_allocated(0)
        
        # Start with nvidia-smi to monitor GPU usage
        print_info("Checking GPU usage before model loading:")
        stdout, stderr, _ = run_command("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv")
        if stdout:
            print_info(stdout)
        
        # Load model with GPU layers
        start_time = time.time()
        try:
            # Set environment variable to force verbose GPU memory logging
            os.environ["LLAMA_CPP_VERBOSE"] = "1"
            
            model_gpu = Llama(
                model_path=model_path,
                n_gpu_layers=-1,  # All layers on GPU
                n_ctx=512,
                verbose=True,     # Enable verbose logging
            )
            gpu_load_time = time.time() - start_time
            print_success(f"GPU model loaded in {gpu_load_time:.2f} seconds")
            
            # Check GPU memory usage to confirm GPU offloading
            if has_torch:
                after_gpu_mem = torch.cuda.memory_allocated(0)
                gpu_mem_used = (after_gpu_mem - before_gpu_mem) / 1024**2  # MB
                print_info(f"GPU memory usage (PyTorch): {gpu_mem_used:.2f} MB")
                
                if gpu_mem_used < 10:  # Less than 10 MB suggests GPU is not being used
                    print_warning("Very little GPU memory is being used. Model may not be properly loaded to GPU.")
                else:
                    print_success(f"Confirmed {gpu_mem_used:.2f} MB GPU memory usage")
            
            # Check GPU usage after loading
            print_info("Checking GPU usage after model loading:")
            stdout, stderr, _ = run_command("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv")
            if stdout:
                print_info(stdout)
            
            # Simple GPU inference test
            start_time = time.time()
            gpu_result = model_gpu("Hello, my name is", max_tokens=5)
            gpu_infer_time = time.time() - start_time
            print_success(f"GPU inference completed in {gpu_infer_time:.2f} seconds")
            print_info(f"GPU result: {gpu_result}")
            
            # Compare performance
            if cpu_infer_time > 0 and gpu_infer_time > 0:
                speedup = cpu_infer_time / gpu_infer_time
                print_info(f"GPU speedup: {speedup:.2f}x faster than CPU")
                
                if speedup < 1.2:  # Less than 20% improvement suggests issues
                    print_warning("GPU acceleration appears minimal. Check CUDA configuration.")
                else:
                    print_success(f"GPU acceleration working as expected ({speedup:.2f}x speedup)")
            
            # Free GPU model
            del model_gpu
            if has_torch:
                torch.cuda.empty_cache()
        
        except Exception as e:
            print_error(f"Error during GPU inference test: {e}")
            print_warning("This suggests CUDA is not properly configured or not supported in your llama-cpp-python build")
        
    except Exception as e:
        print_error(f"Error during inference test: {e}")

def print_troubleshooting_tips():
    """Print WSL-specific troubleshooting tips"""
    print_section("Troubleshooting Tips for WSL")
    
    print_info("If CUDA is not working in WSL, try the following:")
    
    print(f"{BOLD}1. Verify Windows NVIDIA driver installation:{NC}")
    print("   - Open PowerShell as administrator and run: Get-CimInstance Win32_VideoController")
    print("   - Check NVIDIA Control Panel for driver version")
    
    print(f"\n{BOLD}2. Install NVIDIA CUDA driver for WSL:{NC}")
    print("   - Follow the official guide: https://docs.nvidia.com/cuda/wsl-user-guide/index.html")
    print("   - For Ubuntu WSL: sudo apt-get update && sudo apt-get install nvidia-cuda-toolkit")
    
    print(f"\n{BOLD}3. Verify WSL2 is being used (not WSL1):{NC}")
    print("   - Run in PowerShell: wsl --list --verbose")
    print("   - Convert to WSL2 if needed: wsl --set-version <distro> 2")
    
    print(f"\n{BOLD}4. Reinstall llama-cpp-python with CUDA support:{NC}")
    print("   - pip uninstall -y llama-cpp-python")
    print("   - pip install llama-cpp-python --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX2/cu118")
    print("   - Change cu118 to match your CUDA version (cu117, cu121, etc.)")
    
    print(f"\n{BOLD}5. Building from source in WSL:{NC}")
    print("   - Install CUDA toolkit: sudo apt install nvidia-cuda-toolkit")
    print("   - Set environment variables:")
    print("     export CMAKE_ARGS=\"-DLLAMA_CUBLAS=on\"")
    print("     export FORCE_CMAKE=1")
    print("   - Install: pip install llama-cpp-python --no-binary llama-cpp-python")
    
    print(f"\n{BOLD}6. Check if llama-cpp is linked against CUDA:{NC}")
    print("   - Find your llama_cpp shared library:")
    print("     python -c \"from llama_cpp import Llama; print(Llama.__file__)\"")
    print("   - Check CUDA dependencies:")
    print("     ldd <path_to_llama_cpp_so> | grep cuda")

def main():
    """Main function to run all checks"""
    print_header("LLAMA-CPP-PYTHON WSL VERIFICATION")
    
    # Check WSL environment first
    check_wsl_environment()
    
    # Check package installation
    package_installed = check_package_installation()
    if not package_installed:
        print_error("llama-cpp-python is not properly installed. Please install it first.")
        print_troubleshooting_tips()
        return
    
    # Check CUDA support
    check_cuda_support()
    
    # Test inference if model path is provided or can be found
    model_path = None
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if not os.path.exists(model_path):
            print_warning(f"Model file not found: {model_path}")
            model_path = None
    
    test_small_inference(model_path)
    
    # Show troubleshooting tips
    print_troubleshooting_tips()
    
    print_header("VERIFICATION COMPLETE")

if __name__ == "__main__":
    main()
