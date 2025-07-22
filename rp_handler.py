import runpod
import os
import requests
import time
import json
import subprocess
import threading
import sys
from pathlib import Path

def robust_pytorch_install():
    """Install PyTorch with multiple fallback strategies"""
    try:
        print("🔧 Robust PyTorch GPU installation...")
        
        # Strategy 1: Try pre-installed PyTorch first
        try:
            import torch
            if torch.cuda.is_available():
                print("✅ PyTorch GPU already available!")
                return True, f"Pre-installed PyTorch {torch.__version__} with CUDA"
        except ImportError:
            print("📦 PyTorch not found, installing...")
        
        # Strategy 2: Install with conda (often more reliable)
        try:
            print("🐍 Trying conda install...")
            result = subprocess.run([
                "conda", "install", "pytorch", "torchvision", "torchaudio", 
                "pytorch-cuda=12.1", "-c", "pytorch", "-c", "nvidia", "-y"
            ], capture_output=True, timeout=300)
            
            if result.returncode == 0:
                # Test the installation
                test_result = subprocess.run([
                    sys.executable, "-c", 
                    "import torch; print('CUDA available:', torch.cuda.is_available())"
                ], capture_output=True, timeout=30)
                
                if "CUDA available: True" in test_result.stdout.decode():
                    return True, "Conda PyTorch GPU installation successful"
                    
        except Exception as e:
            print(f"❌ Conda failed: {e}")
        
        # Strategy 3: Pip with different mirrors
        pip_strategies = [
            {
                "name": "PyTorch Official",
                "cmd": [
                    sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", 
                    "--index-url", "https://download.pytorch.org/whl/cu121"
                ]
            },
            {
                "name": "Default PyPI",
                "cmd": [
                    sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"
                ]
            },
            {
                "name": "PyTorch Nightly",
                "cmd": [
                    sys.executable, "-m", "pip", "install", "--pre", "torch", "torchvision", "torchaudio", 
                    "--index-url", "https://download.pytorch.org/whl/nightly/cu121"
                ]
            }
        ]
        
        for strategy in pip_strategies:
            try:
                print(f"🔄 Trying {strategy['name']}...")
                
                # Set pip timeout and retries
                env = os.environ.copy()
                env["PIP_TIMEOUT"] = "60"
                env["PIP_RETRIES"] = "3"
                
                result = subprocess.run(
                    strategy["cmd"] + ["--timeout", "60"],
                    capture_output=True, 
                    timeout=300,
                    env=env
                )
                
                if result.returncode == 0:
                    # Test installation
                    test_result = subprocess.run([
                        sys.executable, "-c", 
                        "import torch; print(f'PyTorch {torch.__version__} CUDA: {torch.cuda.is_available()}')"
                    ], capture_output=True, timeout=30)
                    
                    test_output = test_result.stdout.decode()
                    if "CUDA: True" in test_output:
                        return True, f"{strategy['name']} successful: {test_output.strip()}"
                    elif "PyTorch" in test_output:
                        return True, f"{strategy['name']} installed (CPU fallback): {test_output.strip()}"
                
            except Exception as e:
                print(f"❌ {strategy['name']} failed: {e}")
                continue
        
        # Strategy 4: Use system PyTorch if available
        try:
            result = subprocess.run([
                "python3", "-c", "import torch; print('System PyTorch available')"
            ], capture_output=True, timeout=10)
            
            if result.returncode == 0:
                return True, "Using system PyTorch"
                
        except:
            pass
        
        return False, "All PyTorch installation strategies failed"
        
    except Exception as e:
        return False, f"PyTorch installation error: {str(e)}"

def download_essential_files():
    """Re-download essential files that got lost"""
    try:
        print("📥 Re-downloading essential files...")
        
        results = {}
        
        # Download Pepe LoRA
        lora_path = "/app/comfyui/models/loras/pepe.safetensors"
        if not os.path.exists(lora_path):
            try:
                print("📥 Downloading Pepe LoRA...")
                os.makedirs(os.path.dirname(lora_path), exist_ok=True)
                
                response = requests.get(
                    "https://huggingface.co/openfree/pepe/resolve/main/pepe.safetensors",
                    timeout=120, stream=True
                )
                response.raise_for_status()
                
                with open(lora_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                results["lora"] = f"✅ Downloaded {os.path.getsize(lora_path)} bytes"
                
            except Exception as e:
                results["lora"] = f"❌ Failed: {str(e)}"
        else:
            results["lora"] = "✅ Already exists"
        
        # Download SD 1.5 model if missing
        model_path = "/app/comfyui/models/checkpoints/sd15.safetensors"
        if not os.path.exists(model_path):
            try:
                print("📥 Re-downloading SD 1.5 model...")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                response = requests.get(
                    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
                    timeout=300, stream=True
                )
                response.raise_for_status()
                
                downloaded = 0
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if downloaded % (100*1024*1024) == 0:  # Progress every 100MB
                                print(f"📊 Downloaded {downloaded // (1024*1024)}MB...")
                
                results["model"] = f"✅ Downloaded {os.path.getsize(model_path)} bytes"
                
            except Exception as e:
                results["model"] = f"❌ Failed: {str(e)}"
        else:
            results["model"] = "✅ Already exists"
        
        return results
        
    except Exception as e:
        return {"error": str(e)}

def test_gpu_pytorch():
    """Test PyTorch GPU functionality"""
    try:
        # Basic import test
        try:
            import torch
            pytorch_version = torch.__version__
        except ImportError:
            return {
                "pytorch_available": False,
                "error": "PyTorch not installed",
                "cuda_available": False
            }
        
        # CUDA test
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        
        gpu_info = {}
        if cuda_available and device_count > 0:
            gpu_info = {
                "device_name": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "cuda_version": torch.version.cuda
            }
            
            # Test tensor creation
            try:
                test_tensor = torch.randn(100, 100).cuda()
                tensor_test = test_tensor.is_cuda
                gpu_info["tensor_test"] = "✅ GPU tensor creation successful"
            except Exception as e:
                gpu_info["tensor_test"] = f"❌ GPU tensor failed: {str(e)}"
        
        return {
            "pytorch_available": True,
            "pytorch_version": pytorch_version,
            "cuda_available": cuda_available,
            "device_count": device_count,
            "gpu_info": gpu_info,
            "ready": cuda_available and device_count > 0
        }
        
    except Exception as e:
        return {
            "pytorch_available": False,
            "error": str(e),
            "cuda_available": False
        }

def minimal_comfyui_test():
    """Test ComfyUI without starting full server"""
    try:
        print("🧪 Testing ComfyUI components...")
        
        # Change to ComfyUI directory
        original_cwd = os.getcwd()
        os.chdir("/app/comfyui")
        
        test_results = {}
        
        # Test basic imports
        try:
            import comfy.utils
            test_results["comfy_utils"] = "✅ Available"
        except Exception as e:
            test_results["comfy_utils"] = f"❌ {str(e)}"
        
        try:
            import nodes
            test_results["nodes"] = "✅ Available"
        except Exception as e:
            test_results["nodes"] = f"❌ {str(e)}"
        
        try:
            import execution
            test_results["execution"] = "✅ Available"
        except Exception as e:
            test_results["execution"] = f"❌ {str(e)}"
        
        # Test model loading capability
        try:
            import comfy.model_management
            test_results["model_management"] = "✅ Available"
        except Exception as e:
            test_results["model_management"] = f"❌ {str(e)}"
        
        os.chdir(original_cwd)
        
        successful = sum(1 for result in test_results.values() if result.startswith("✅"))
        total = len(test_results)
        
        return test_results, successful, total
        
    except Exception as e:
        return {"error": str(e)}, 0, 1

def start_comfyui_robust():
    """Start ComfyUI with robust error handling"""
    try:
        print("🚀 Starting ComfyUI with robust setup...")
        
        # Kill existing processes
        try:
            subprocess.run(["pkill", "-f", "main.py"], check=False, timeout=5)
            time.sleep(3)
        except:
            pass
        
        # Setup environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = "/app/comfyui"
        
        # Start server
        cmd = [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", "8188"]
        
        print(f"🔧 Starting: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            cwd="/app/comfyui",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        # Wait for startup with detailed logging
        for i in range(60):
            try:
                response = requests.get("http://localhost:8188/", timeout=3)
                if response.status_code == 200:
                    print("✅ ComfyUI server started!")
                    return True, f"Server running (PID: {process.pid})"
            except:
                pass
            
            # Check for process crash
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                error_msg = stderr.decode()[:500] if stderr else "No error output"
                return False, f"Process crashed: {error_msg}"
            
            if i % 10 == 0:
                print(f"⏳ Starting... ({i}/60 seconds)")
            
            time.sleep(1)
        
        return False, "Startup timeout after 60 seconds"
        
    except Exception as e:
        return False, f"Server start failed: {str(e)}"

def handler(event):
    """Robust GPU handler with fallback strategies"""
    print("🐸 PEPE WORKER v5.0 - ROBUST GPU EDITION! 💪")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'pepe the frog')
        
        print(f"📝 Processing: {prompt}")
        
        # Step 1: Install PyTorch with multiple strategies
        print("🔄 Step 1: Robust PyTorch installation...")
        pytorch_success, pytorch_msg = robust_pytorch_install()
        
        # Step 2: Test PyTorch GPU
        print("🔄 Step 2: Testing PyTorch GPU...")
        pytorch_test = test_gpu_pytorch()
        
        # Step 3: Download essential files
        print("🔄 Step 3: Ensuring essential files...")
        file_results = download_essential_files()
        
        # Step 4: Test ComfyUI components
        print("🔄 Step 4: Testing ComfyUI...")
        comfyui_results, comfyui_success, comfyui_total = minimal_comfyui_test()
        
        # Step 5: Start server if everything looks good
        server_success = False
        server_msg = "Not attempted"
        
        if pytorch_test.get("ready") and comfyui_success > 0:
            print("🔄 Step 5: Starting ComfyUI server...")
            server_success, server_msg = start_comfyui_robust()
        
        # Calculate overall readiness
        readiness_factors = [
            pytorch_test.get("ready", False),
            file_results.get("lora", "").startswith("✅"),
            file_results.get("model", "").startswith("✅"),
            comfyui_success >= 2,
            server_success
        ]
        
        overall_readiness = sum(readiness_factors) / len(readiness_factors) * 100
        
        return {
            "message": f"🐸 ROBUST GPU Pepe setup: {prompt}",
            "status": "success" if server_success else "partial",
            "pytorch": {
                "installation": pytorch_msg,
                "test_results": pytorch_test
            },
            "files": file_results,
            "comfyui": {
                "test_results": comfyui_results,
                "success_rate": f"{comfyui_success}/{comfyui_total}"
            },
            "server": server_msg,
            "overall_readiness": overall_readiness,
            "ready_for_generation": server_success,
            "next_steps": [
                "✅ PyTorch OK" if pytorch_test.get("ready") else "❌ Fix PyTorch",
                "✅ Files OK" if all(r.startswith("✅") for r in file_results.values() if isinstance(r, str)) else "❌ Download files",
                "✅ ComfyUI OK" if comfyui_success >= 2 else "❌ Fix ComfyUI",
                "✅ Server OK" if server_success else "❌ Start server",
                "🎯 READY FOR ROBUST GPU PEPE GENERATION!" if server_success else "🔧 Still robustly fixing..."
            ],
            "debug_info": {
                "pytorch_available": pytorch_test.get("pytorch_available", False),
                "cuda_available": pytorch_test.get("cuda_available", False),
                "files_ready": all(r.startswith("✅") for r in file_results.values() if isinstance(r, str)),
                "comfyui_ready": comfyui_success >= 2,
                "server_ready": server_success
            }
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
            "debug": "Handler exception in robust mode"
        }

if __name__ == '__main__':
    print("🚀 Starting ROBUST GPU-POWERED Pepe Worker v5.0...")
    runpod.serverless.start({'handler': handler})
