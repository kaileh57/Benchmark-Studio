#!/usr/bin/env python3
"""
Enhanced checker for lm-eval installation and available models
With WSL-specific diagnostics and more robust error handling
"""

import sys
import os
import importlib
import subprocess
import traceback
from importlib.util import find_spec
from pathlib import Path

# ANSI colors for better output visibility
GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
BLUE = "\033[0;34m"
CYAN = "\033[0;36m"
NC = "\033[0m"  # No Color
BOLD = "\033[1m"

def print_header(message):
    """Print a formatted header"""
    print(f"\n{BLUE}{BOLD}{'=' * 60}{NC}")
    print(f"{BLUE}{BOLD}{message.center(60)}{NC}")
    print(f"{BLUE}{BOLD}{'=' * 60}{NC}\n")

def print_section(message):
    """Print a section title"""
    print(f"\n{CYAN}{BOLD}{message}{NC}")
    print(f"{CYAN}{'-' * len(message)}{NC}")

def print_success(message):
    """Print a success message"""
    print(f"{GREEN}✅ {message}{NC}")

def print_warning(message):
    """Print a warning message"""
    print(f"{YELLOW}⚠️ {message}{NC}")

def print_error(message):
    """Print an error message"""
    print(f"{RED}❌ {message}{NC}")

def print_info(message):
    """Print an informational message"""
    print(f"{BLUE}ℹ️ {message}{NC}")

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

def check_wsl():
    """Check if running inside WSL"""
    try:
        with open('/proc/version', 'r') as f:
            version = f.read().lower()
            if 'microsoft' in version or 'wsl' in version:
                print_success("Running in WSL environment")
                return True
        return False
    except:
        return False

def check_pip_package(package_name):
    """Check if a package is installed using pip"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print_success(f"Package '{package_name}' is installed")
            version = None
            location = None
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    version = line.split(':')[1].strip()
                    print_info(f"Version: {version}")
                elif line.startswith('Location:'):
                    location = line.split(':')[1].strip()
                    print_info(f"Location: {location}")
            return True, version, location
        else:
            print_error(f"Package '{package_name}' is NOT installed")
            return False, None, None
    except Exception as e:
        print_error(f"Error checking pip package: {e}")
        return False, None, None

def check_importable(module_name):
    """Check if a module can be imported"""
    try:
        # First check if the module is findable using find_spec
        spec = find_spec(module_name)
        if spec is None:
            print_error(f"Module '{module_name}' cannot be found in sys.path")
            return None
        
        # Try to import the module
        module = importlib.import_module(module_name)
        print_success(f"Module '{module_name}' can be imported")
        
        # Get module details
        version = getattr(module, '__version__', 'Unknown')
        location = getattr(module, '__file__', 'Unknown')
        
        print_info(f"Version: {version}")
        print_info(f"Location: {location}")
        
        # Check if it's a package by looking for __path__
        if hasattr(module, '__path__'):
            print_info(f"Package path: {module.__path__}")
        
        return module
    except ImportError as e:
        print_error(f"Module '{module_name}' cannot be imported: {e}")
        return None
    except Exception as e:
        print_error(f"Error importing module '{module_name}': {e}")
        traceback.print_exc(file=sys.stdout)  # Print the full traceback for debugging
        return None

def check_available_models(lm_eval_module):
    """Check available models in lm-eval"""
    print_section("Checking available models in lm-eval")
    try:
        # Try different possible import paths for the registry
        try:
            from lm_eval.api.registry import ALL_MODELS
        except ImportError:
            try:
                from lm_eval.models.registry import ALL_MODELS
            except ImportError:
                try:
                    # Just get the registry directly from the module
                    registry_module = importlib.import_module('lm_eval.api.registry')
                    ALL_MODELS = getattr(registry_module, 'ALL_MODELS', {})
                except:
                    print_error("Could not import ALL_MODELS from any known location")
                    return False
        
        if not isinstance(ALL_MODELS, dict):
            print_warning(f"ALL_MODELS exists but is not a dictionary: {type(ALL_MODELS)}")
            return False
            
        print_success(f"Found {len(ALL_MODELS)} models in registry")
        
        # Print the first 12 models
        print_info("Available models:")
        for i, model_name in enumerate(sorted(ALL_MODELS.keys())):
            if i < 12:
                print(f"   - {model_name}")
            elif i == 12:
                print(f"   - ... and {len(ALL_MODELS) - 12} more")
                break
        
        # Check for specific model types
        key_models = ['gguf', 'ggml', 'llama_cpp', 'llama-cpp']
        found_models = [m for m in key_models if m in ALL_MODELS]
        
        if found_models:
            print_success(f"Found benchmark-compatible models: {', '.join(found_models)}")
            # Return a list of the found models
            return found_models
        else:
            print_error("No benchmark-compatible models found")
            print_warning("You need one of these model types for GGUF benchmarking: gguf, ggml, llama_cpp")
            return []
            
    except Exception as e:
        print_error(f"Error checking available models: {e}")
        traceback.print_exc(file=sys.stdout)
        return False

def check_available_tasks(lm_eval_module):
    """Check available tasks in lm-eval"""
    print_section("Checking available tasks in lm-eval")
    try:
        # Try different possible import paths
        try:
            from lm_eval.tasks import ALL_TASKS
        except ImportError:
            try:
                tasks_module = importlib.import_module('lm_eval.tasks')
                ALL_TASKS = getattr(tasks_module, 'ALL_TASKS', {})
            except:
                print_error("Could not import ALL_TASKS from any known location")
                return False
        
        if not isinstance(ALL_TASKS, dict):
            print_warning(f"ALL_TASKS exists but is not a dictionary: {type(ALL_TASKS)}")
            return False
            
        print_success(f"Found {len(ALL_TASKS)} tasks in registry")
        
        # Check for specific benchmark tasks
        key_tasks = ['mmlu', 'arc_easy', 'hellaswag', 'truthfulqa_mc', 'gsm8k', 'arc_challenge']
        found_tasks = [t for t in key_tasks if t in ALL_TASKS]
        
        if found_tasks:
            print_success(f"Found benchmark tasks: {', '.join(found_tasks)}")
            return found_tasks
        else:
            print_error("No common benchmark tasks found")
            print_warning("This suggests a problem with the lm-eval installation")
            return []
            
    except Exception as e:
        print_error(f"Error checking available tasks: {e}")
        traceback.print_exc(file=sys.stdout)
        return False

def try_benchmark_import(lm_eval_module):
    """Try to import and use the lm-eval evaluator"""
    print_section("Testing lm-eval imports")
    try:
        print_info("Attempting to import evaluator...")
        try:
            from lm_eval import evaluator
            print_success("evaluator module imported successfully")
        except ImportError as e:
            print_error(f"Could not import evaluator: {e}")
            return False
        
        print_info("Attempting to import models...")
        try:
            from lm_eval import models
            print_success("models module imported successfully")
        except ImportError as e:
            print_error(f"Could not import models: {e}")
        
        print_info("Attempting to access registry...")
        try:
            from lm_eval.api.registry import get_model
            print_success("registry.get_model imported successfully")
        except ImportError as e:
            print_error(f"Could not import get_model: {e}")
            try:
                # Try alternative paths
                from lm_eval.models.registry import get_model
                print_success("models.registry.get_model imported successfully")
            except ImportError:
                print_error("Could not import get_model from any known location")
                return False
                
        return True
    except Exception as e:
        print_error(f"Error testing imports: {e}")
        return False

def check_python_paths():
    """Check Python paths and environment"""
    print_section("Checking Python environment")
    
    # Print Python version
    print_info(f"Python version: {sys.version}")
    print_info(f"Python executable: {sys.executable}")
    
    # Check sys.path
    print_info("Python sys.path:")
    for i, path in enumerate(sys.path):
        print(f"   {i+1}. {path}")
    
    # Check key environment variables
    python_path = os.environ.get('PYTHONPATH', 'Not set')
    print_info(f"PYTHONPATH: {python_path}")
    
    virtual_env = os.environ.get('VIRTUAL_ENV', 'Not set')
    print_info(f"VIRTUAL_ENV: {virtual_env}")
    
    # Check for potential path issues
    if virtual_env != 'Not set':
        venv_site_packages = os.path.join(virtual_env, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
        if venv_site_packages in sys.path:
            print_success("Virtual environment site-packages is in sys.path")
        else:
            print_warning("Virtual environment site-packages is NOT in sys.path")
            print_info(f"Expected path: {venv_site_packages}")

def check_for_implementation_error():
    """Check for the specific implementation error in the benchmarking code"""
    print_section("Checking for benchmark implementation issues")
    
    try:
        # Check if we can import ALL_MODELS
        try:
            from lm_eval.api.registry import ALL_MODELS
            has_registry = True
        except ImportError:
            try:
                from lm_eval.models.registry import ALL_MODELS
                has_registry = True
            except ImportError:
                has_registry = False
        
        if not has_registry:
            print_error("Could not import model registry - this suggests installation issues")
            return
        
        # Check if the registry contains the 'gguf' model
        if 'gguf' in ALL_MODELS:
            print_success("'gguf' model is available in the registry")
            
            # Check if we can get the model type
            try:
                from lm_eval.api.registry import get_model
                model_type = get_model('gguf')
                print_success(f"Successfully retrieved 'gguf' model type: {model_type}")
            except Exception as e:
                print_error(f"Error getting 'gguf' model: {e}")
                
        else:
            print_warning("'gguf' model is NOT in registry")
            print_info("Available models that might work with GGUF files:")
            for model in ALL_MODELS:
                if any(term in model.lower() for term in ['gguf', 'ggml', 'llama']):
                    print(f"   - {model}")
            
            # Check for alternative model types that could be used
            if 'ggml' in ALL_MODELS:
                print_warning("'ggml' model exists and might work as an alternative to 'gguf'")
                print_info("Try modifying your benchmark code to use 'ggml' instead of 'gguf'")
            
            if 'llama_cpp' in ALL_MODELS or 'llama-cpp' in ALL_MODELS:
                llama_model = 'llama_cpp' if 'llama_cpp' in ALL_MODELS else 'llama-cpp'
                print_warning(f"'{llama_model}' exists and might work for GGUF files")
                print_info(f"Try modifying your benchmark code to use '{llama_model}' instead of 'gguf'")
        
        # Check if we can access the evaluator
        try:
            from lm_eval import evaluator
            print_success("lm_eval.evaluator is accessible")
            
            # Check the simple_evaluate method
            if hasattr(evaluator, 'simple_evaluate'):
                print_success("evaluator.simple_evaluate method is available")
            else:
                print_error("evaluator.simple_evaluate method is NOT available!")
                print_info("This suggests your lm-eval version is incompatible or corrupted")
        except Exception as e:
            print_error(f"Error accessing evaluator: {e}")
    
    except Exception as e:
        print_error(f"Error during implementation check: {e}")
        traceback.print_exc(file=sys.stdout)

def fix_suggestions(lm_eval_module, found_models, found_tasks):
    """Provide suggestions to fix common issues"""
    print_section("Suggestions to fix issues")
    
    has_issues = False
    
    # Check if we found the module but no models
    if lm_eval_module and not found_models:
        has_issues = True
        print_warning("Found lm_eval module but no compatible models")
        print_info("1. Try upgrading lm-eval to the latest version:")
        print(f"   pip install --upgrade lm-eval")
        print_info("2. Or install a specific version known to work with GGUF:")
        print(f"   pip install lm-eval==0.3.0")
    
    # Check if we found the module but no tasks
    if lm_eval_module and not found_tasks:
        has_issues = True
        print_warning("Found lm_eval module but no benchmark tasks")
        print_info("This suggests a corrupted or incomplete installation")
        print_info("Try reinstalling with the --force-reinstall flag:")
        print(f"   pip install --force-reinstall lm-eval")
    
    # If no module found at all
    if not lm_eval_module:
        has_issues = True
        print_warning("Could not import lm_eval module")
        print_info("1. Make sure you've installed it:")
        print(f"   pip install lm-eval")
        print_info("2. Check for path issues in your environment")
        print_info("3. Try installing with pip directly to the current Python:")
        print(f"   {sys.executable} -m pip install lm-eval")
    
    if not has_issues:
        print_success("No critical issues detected with lm-eval installation")
        print_info("If you're still having trouble, try modifying the benchmark code to match your environment")
    
def check_fix_benchmark_code():
    """Provide code snippets to fix common benchmark issues"""
    print_section("Code fix for benchmark script")
    
    print_info("If your benchmark script is failing with 'model not found' errors:")
    print_info("1. Locate the model registration in benchmark_runner.py")
    print_info("2. Replace the code that uses 'llama-cpp-adapter' with:")
    
    code_fix = """
    # Try to use different model types in order of preference
    model_types = ['gguf', 'ggml', 'llama_cpp', 'llama-cpp']
    model_type = None
    
    for type_name in model_types:
        try:
            from lm_eval.api.registry import ALL_MODELS
            if type_name in ALL_MODELS:
                model_type = type_name
                break
        except:
            pass
    
    if not model_type:
        raise ValueError("No compatible model type found in lm-eval registry")
    
    # Now use the found model type
    results = evaluator.simple_evaluate(
        model=model_type,  # Use the model type we found
        model_args=f"pretrained={self.model_path},n_gpu_layers={self.n_gpu_layers},n_ctx={self.context_size}",
        tasks=[benchmark],
        batch_size=1,
        device="cuda" if self.system_info.get("cuda_available", False) else "cpu",
        **additional_args
    )
    """
    
    # Print the code with some formatting
    for line in code_fix.strip().split('\n'):
        print(f"    {line}")

def main():
    """Main function to run all checks"""
    print_header("LM-EVAL DIAGNOSTIC TOOL")
    
    # Check if running in WSL
    is_wsl = check_wsl()
    
    # Check Python environment
    check_python_paths()
    
    # Check all possible package names
    print_section("Checking package installation")
    
    packages_to_check = ['lm-eval', 'lm_eval', 'lm-evaluation-harness']
    found_package = False
    package_info = {}
    
    for package in packages_to_check:
        installed, version, location = check_pip_package(package)
        if installed:
            found_package = True
            package_info[package] = {
                'version': version,
                'location': location
            }
    
    if not found_package:
        print_error("Could not find any variant of lm-eval package")
        print_info("Please install with: pip install lm-eval")
        return
    
    # Check module import
    lm_eval_module = check_importable('lm_eval')
    
    if lm_eval_module:
        # Check models and tasks
        found_models = check_available_models(lm_eval_module)
        found_tasks = check_available_tasks(lm_eval_module)
        
        # Test imports specifically needed for benchmarking
        try_benchmark_import(lm_eval_module)
        
        # Check for implementation error in benchmark code
        check_for_implementation_error()
        
        # Provide fix suggestions
        fix_suggestions(lm_eval_module, found_models, found_tasks)
        
        # Provide code fix
        check_fix_benchmark_code()
    
    print_header("DIAGNOSIS COMPLETE")
    
    # Final summary
    if lm_eval_module and (found_models or isinstance(found_models, list) and len(found_models) > 0):
        print_success("lm-eval appears to be properly installed")
        if not found_tasks:
            print_warning("Tasks might be missing - check your installation")
        print_info("If you're still having benchmark issues, modify your code using the suggestions above")
    else:
        print_warning("Problems detected with your lm-eval installation")
        print_info("Follow the suggestions above to fix the issues")

if __name__ == "__main__":
    main()

