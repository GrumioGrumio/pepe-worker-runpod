import runpod
import os
import requests
import time
import json
import subprocess
import threading
import sys
from pathlib import Path

def fix_pytorch_gpu():
    """Install GPU-enabled PyTorch"""
    try:
        print("🔧 Installing GPU-enabled PyTorch...")
        
        # Uninstall CPU version first
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"
        ], capture_output=True, timeout=60)
        
        # Install GPU version
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121",
            "--force-reinstall"
        ], capture_output=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ GPU PyTorch installed successfully")
            return True, "GPU PyTorch installed"
        else:
            print(f"❌ PyTorch install failed: {result.stderr.decode()}")
            return False, f"PyTorch install failed: {result.stderr.decode()[:200]}"
            
    except Exception as e:
        return False, f"PyTorch installation error: {str(e)}"

def check_gpu_status():
    """Check GPU availability and status"""
    try:
        # Check nvidia-smi
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=10)
            if result.returncode == 0:
                nvidia_output = result.stdout.decode()
                gpu_info = "GPU detected via nvidia-smi"
            else:
                nvidia_output = "nvidia-smi failed"
                gpu_info = "No GPU detected"
        except:
            nvidia_output = "nvidia-smi not available"
            gpu_info = "No GPU tools"
        
        # Test PyTorch CUDA
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else "None"
            device_count = torch.cuda.device_count() if cuda_available else 0
            
            if cuda_available and device_count > 0:
                device_name = torch.cuda.get_device_name(0)
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                gpu_status = f"✅ {device_name} ({memory_total:.1f}GB)"
            else:
                gpu_status = "❌ No CUDA devices"
                device_name = "None"
                memory_total = 0
                
        except Exception as e:
            cuda_available = False
            cuda_version = "Error"
            device_count = 0
            gpu_status = f"❌ PyTorch error: {str(e)}"
            device_name = "Unknown"
            memory_total = 0
        
        return {
            "nvidia_smi": nvidia_output[:300],
            "cuda_available": cuda_available,
            "cuda_version": cuda_version,
            "device_count": device_count,
            "device_name": device_name,
            "memory_gb": memory_total,
            "status": gpu_status,
            "ready": cuda_available and device_count > 0
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "ready": False,
            "status": "❌ GPU check failed"
        }

def setup_gpu_environment():
    """Setup environment for GPU usage"""
    try:
        # Set CUDA environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
        os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5;8.0;8.6;8.9;9.0"
        
        # Remove CPU-only flags
        if "FORCE_CPU" in os.environ:
            del os.environ["FORCE_CPU"]
        
        # Add ComfyUI to path
        comfyui_path = "/app/comfyui"
        if comfyui_path not in sys.path:
            sys.path.insert(0, comfyui_path)
            
        return True, "GPU environment configured"
        
    except Exception as e:
        return False, f"Environment setup failed: {str(e)}"

def test_comfyui_gpu_imports():
    """Test ComfyUI imports with GPU"""
    try:
        import_results = {}
        
        # Test PyTorch GPU
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            device_count = torch.cuda.device_count()
            
            if cuda_available:
                # Test GPU tensor creation
                test_tensor = torch.randn(100, 100).cuda()
                gpu_test = test_tensor.is_cuda
                import_results["torch"] = f"✅ GPU ready ({device_count} devices)"
            else:
                import_results["torch"] = "❌ CUDA not available"
                
        except Exception as e:
            import_results["torch"] = f"❌ {str(e)}"
        
        # Test other imports
        try:
            import numpy as np
            import_results["numpy"] = "✅ Available"
        except Exception as e:
            import_results["numpy"] = f"❌ {str(e)}"
        
        try:
            from PIL import Image
            import_results["pillow"] = "✅ Available"
        except Exception as e:
            import_results["pillow"] = f"❌ {str(e)}"
        
        # Test ComfyUI imports
        try:
            original_cwd = os.getcwd()
            os.chdir("/app/comfyui")
            
            # Import ComfyUI modules
            try:
                import model_management
                import_results["model_management"] = "✅ Available"
            except Exception as e:
                import_results["model_management"] = f"❌ {str(e)}"
            
            try:
                import nodes
                import_results["nodes"] = "✅ Available"
            except Exception as e:
                import_results["nodes"] = f"❌ {str(e)}"
            
            try:
                import execution
                import_results["execution"] = "✅ Available"
            except Exception as e:
                import_results["execution"] = f"❌ {str(e)}"
            
            os.chdir(original_cwd)
            
        except Exception as e:
            import_results["comfyui_imports"] = f"❌ {str(e)}"
        
        successful = sum(1 for result in import_results.values() if result.startswith("✅"))
        total = len(import_results)
        
        return import_results, successful, total
        
    except Exception as e:
        return {"error": str(e)}, 0, 1

def start_comfyui_gpu():
    """Start ComfyUI with GPU acceleration"""
    try:
        print("🚀 Starting ComfyUI server (GPU mode)...")
        
        # Kill any existing processes
        try:
            subprocess.run(["pkill", "-f", "main.py"], check=False, timeout=5)
            time.sleep(3)
        except:
            pass
        
        # Prepare GPU environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = "/app/comfyui"
        
        # Remove CPU-only flags
        env.pop("FORCE_CPU", None)
        
        # Start server with GPU settings
        cmd = [
            sys.executable, "main.py",
            "--listen", "0.0.0.0",
            "--port", "8188",
            "--disable-auto-launch",
            "--enable-cors-header"
        ]
        
        print(f"🔧 Starting: {' '.join(cmd)}")
        print(f"🎯 Using GPU: {env.get('CUDA_VISIBLE_DEVICES')}")
        
        # Start process
        process = subprocess.Popen(
            cmd,
            cwd="/app/comfyui",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        # Wait for server startup
        print("⏳ Waiting for GPU server startup...")
        for i in range(90):  # 90 seconds for GPU initialization
            try:
                response = requests.get("http://localhost:8188/", timeout=5)
                if response.status_code == 200:
                    print("✅ ComfyUI GPU server started!")
                    return True, f"GPU server running (PID: {process.pid})"
                    
            except requests.exceptions.RequestException:
                pass
            
            # Check if process crashed
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                return False, f"Process crashed. STDERR: {stderr.decode()[:300]}"
            
            if i % 15 == 0:  # Progress every 15 seconds
                print(f"⏳ GPU initialization... ({i}/90 seconds)")
            
            time.sleep(1)
        
        # Timeout reached
        if process.poll() is None:
            stdout, stderr = process.communicate(timeout=5)
            return False, f"Startup timeout. STDERR: {stderr.decode()[:300]}"
        else:
            stdout, stderr = process.communicate()
            return False, f"Process died. STDERR: {stderr.decode()[:300]}"
        
    except Exception as e:
        return False, f"GPU server start failed: {str(e)}"

def create_gpu_workflow():
    """Create GPU-optimized workflow"""
    try:
        workflow = {
            "1": {
                "inputs": {
                    "ckpt_name": "sd15.safetensors"
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": "pepe the frog, detailed, high quality, cartoon style, green frog",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": "blurry, low quality, distorted, ugly",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "5": {
                "inputs": {
                    "seed": 42,
                    "steps": 20,  # Higher steps for GPU
                    "cfg": 7.5,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                },
                "class_type": "KSampler"
            },
            "6": {
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2]
                },
                "class_type": "VAEDecode"
            },
            "7": {
                "inputs": {
                    "filename_prefix": "pepe_gpu",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }
        
        return workflow, "GPU-optimized workflow created"
        
    except Exception as e:
        return None, f"Workflow creation failed: {str(e)}"

def comprehensive_gpu_status():
    """Get comprehensive GPU system status"""
    try:
        status = {}
        
        # GPU hardware check
        status["gpu"] = check_gpu_status()
        
        # File checks
        status["files"] = {
            "lora_exists": os.path.exists("/app/comfyui/models/loras/pepe.safetensors"),
            "model_exists": os.path.exists("/app/comfyui/models/checkpoints/sd15.safetensors"),
            "comfyui_main": os.path.exists("/app/comfyui/main.py"),
            "output_dir": os.path.exists("/app/comfyui/output")
        }
        
        # Import tests
        import_results, successful_imports, total_imports = test_comfyui_gpu_imports()
        status["imports"] = import_results
        status["import_success_rate"] = f"{successful_imports}/{total_imports}"
        
        # Server test
        try:
            response = requests.get("http://localhost:8188/", timeout=5)
            status["server"] = {
                "running": True,
                "status_code": response.status_code,
                "accessible": response.status_code == 200
            }
        except:
            status["server"] = {
                "running": False,
                "status_code": None,
                "accessible": False
            }
        
        # Calculate readiness
        readiness_factors = [
            status["gpu"]["ready"],
            status["files"]["lora_exists"],
            status["files"]["model_exists"],
            successful_imports >= 4,
            status["server"]["accessible"]
        ]
        
        status["overall_readiness"] = sum(readiness_factors) / len(readiness_factors) * 100
        status["ready_for_generation"] = all(readiness_factors)
        
        return status
        
    except Exception as e:
        return {"error": str(e), "overall_readiness": 0}

def handler(event):
    """GPU-optimized Pepe generation handler"""
    print("🐸 PEPE WORKER v4.0 - GPU POWER MODE! 🚀")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'pepe the frog')
        
        print(f"📝 Processing: {prompt}")
        print(f"💰 Using your paid GPU resources!")
        
        # Step 1: Fix PyTorch GPU
        print("🔄 Step 1: Installing GPU PyTorch...")
        pytorch_success, pytorch_msg = fix_pytorch_gpu()
        
        # Step 2: Setup GPU environment
        print("🔄 Step 2: Setting up GPU environment...")
        env_success, env_msg = setup_gpu_environment()
        
        # Step 3: Check GPU status
        print("🔄 Step 3: Checking GPU status...")
        gpu_status = check_gpu_status()
        
        # Step 4: Start GPU server
        print("🔄 Step 4: Starting GPU ComfyUI server...")
        server_success, server_msg = start_comfyui_gpu()
        
        # Step 5: Get comprehensive status
        print("🔄 Step 5: Getting comprehensive status...")
        system_status = comprehensive_gpu_status()
        
        return {
            "message": f"🐸 GPU-POWERED Pepe setup: {prompt}",
            "status": "success" if server_success else "partial",
            "gpu_info": {
                "hardware": gpu_status,
                "pytorch_gpu": pytorch_msg,
                "environment": env_msg
            },
            "server_status": server_msg,
            "system_status": system_status,
            "ready_for_generation": system_status.get("ready_for_generation", False),
            "readiness_percentage": system_status.get("overall_readiness", 0),
            "next_actions": [
                "✅ GPU detected" if gpu_status.get("ready") else "❌ Fix GPU",
                "✅ PyTorch GPU" if pytorch_success else "❌ Install GPU PyTorch",
                "✅ Environment OK" if env_success else "❌ Fix environment", 
                "✅ Server running" if server_success else "❌ Start server",
                "🎯 READY FOR GPU PEPE GENERATION!" if system_status.get("ready_for_generation") else "🔧 Still optimizing GPU setup..."
            ],
            "performance_mode": "GPU_ACCELERATED",
            "cost_optimization": "Using your paid GPU resources efficiently"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
            "debug": "Handler exception in GPU mode"
        }

if __name__ == '__main__':
    print("🚀 Starting GPU-POWERED Pepe Worker v4.0...")
    print("💰 Maximizing your RunPod GPU investment!")
    runpod.serverless.start({'handler': handler})
