import runpod
import os
import requests
import time
import json
import subprocess
import threading
import sys
from pathlib import Path
import shutil

def verify_fresh_environment():
    """Verify we have a clean, working environment"""
    try:
        print("🔍 Verifying fresh environment...")
        
        checks = {}
        
        # Check PyTorch GPU
        try:
            import torch
            checks["pytorch"] = {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
            }
        except Exception as e:
            checks["pytorch"] = {"error": str(e)}
        
        # Check disk space
        try:
            statvfs = os.statvfs("/")
            free_space_gb = (statvfs.f_bavail * statvfs.f_frsize) / (1024**3)
            total_space_gb = (statvfs.f_blocks * statvfs.f_frsize) / (1024**3)
            
            checks["disk_space"] = {
                "free_gb": free_space_gb,
                "total_gb": total_space_gb,
                "adequate": free_space_gb > 30  # Need 30GB+ for models
            }
        except Exception as e:
            checks["disk_space"] = {"error": str(e)}
        
        # Check CUDA libraries
        try:
            result = subprocess.run(["ldconfig", "-p"], capture_output=True, timeout=10)
            ldconfig_output = result.stdout.decode()
            
            cuda_libs = ["libcublas", "libcurand", "libcudnn"]
            found_libs = {lib: lib in ldconfig_output for lib in cuda_libs}
            
            checks["cuda_libraries"] = {
                "libraries": found_libs,
                "all_found": all(found_libs.values())
            }
        except Exception as e:
            checks["cuda_libraries"] = {"error": str(e)}
        
        # Overall health
        pytorch_ok = checks.get("pytorch", {}).get("cuda_available", False)
        disk_ok = checks.get("disk_space", {}).get("adequate", False)
        cuda_ok = checks.get("cuda_libraries", {}).get("all_found", False)
        
        checks["overall_health"] = {
            "pytorch_gpu": pytorch_ok,
            "disk_space": disk_ok,
            "cuda_libraries": cuda_ok,
            "ready_for_setup": pytorch_ok and disk_ok and cuda_ok
        }
        
        return checks
        
    except Exception as e:
        return {"error": str(e)}

def install_comfyui_fresh():
    """Install ComfyUI from scratch in fresh environment"""
    try:
        print("📦 Installing ComfyUI from scratch...")
        
        # Create app directory
        app_dir = "/workspace/comfyui"
        os.makedirs(app_dir, exist_ok=True)
        
        # Clone ComfyUI
        try:
            print("📥 Cloning ComfyUI repository...")
            result = subprocess.run([
                "git", "clone", "https://github.com/comfyanonymous/ComfyUI.git", app_dir
            ], capture_output=True, timeout=120)
            
            if result.returncode != 0:
                return False, f"Git clone failed: {result.stderr.decode()}"
            
            print("✅ ComfyUI repository cloned")
        except Exception as e:
            return False, f"Git clone exception: {str(e)}"
        
        # Install dependencies
        try:
            print("📦 Installing ComfyUI dependencies...")
            
            # Install required packages
            packages = [
                "xformers==0.0.22.post7",
                "transformers>=4.25.1",
                "tokenizers>=0.13.3", 
                "sentencepiece",
                "safetensors>=0.3.2",
                "aiohttp",
                "accelerate",
                "pyyaml",
                "Pillow",
                "scipy",
                "tqdm",
                "psutil",
                "kornia>=0.7.1",
                "spandrel"
            ]
            
            for package in packages:
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", package, "--no-cache-dir"
                    ], capture_output=True, timeout=180)
                    
                    if result.returncode == 0:
                        print(f"✅ Installed {package}")
                    else:
                        print(f"⚠️ Warning installing {package}: {result.stderr.decode()[:100]}")
                        
                except Exception as e:
                    print(f"❌ Failed to install {package}: {e}")
            
            return True, f"ComfyUI installed at {app_dir}"
            
        except Exception as e:
            return False, f"Dependency installation failed: {str(e)}"
        
    except Exception as e:
        return False, f"ComfyUI installation failed: {str(e)}"

def download_models_fresh():
    """Download models to fresh ComfyUI installation"""
    try:
        print("📥 Downloading models for fresh installation...")
        
        # Create model directories
        models_dir = "/workspace/comfyui/models"
        checkpoints_dir = os.path.join(models_dir, "checkpoints")
        loras_dir = os.path.join(models_dir, "loras")
        
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(loras_dir, exist_ok=True)
        
        downloads = {}
        
        # Download SD 1.5 model
        model_path = os.path.join(checkpoints_dir, "sd15.safetensors")
        if not os.path.exists(model_path):
            try:
                print("📥 Downloading SD 1.5 model (3.97GB)...")
                
                response = requests.get(
                    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
                    stream=True,
                    timeout=600
                )
                response.raise_for_status()
                
                downloaded = 0
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if downloaded % (500*1024*1024) == 0:  # Progress every 500MB
                                print(f"📊 Downloaded {downloaded // (1024*1024)}MB...")
                
                downloads["sd15_model"] = f"✅ Downloaded {os.path.getsize(model_path)} bytes"
                print("✅ SD 1.5 model downloaded")
                
            except Exception as e:
                downloads["sd15_model"] = f"❌ Failed: {str(e)}"
        else:
            downloads["sd15_model"] = "✅ Already exists"
        
        # Download Pepe LoRA
        lora_path = os.path.join(loras_dir, "pepe.safetensors")
        if not os.path.exists(lora_path):
            try:
                print("📥 Downloading Pepe LoRA (164MB)...")
                
                response = requests.get(
                    "https://huggingface.co/openfree/pepe/resolve/main/pepe.safetensors",
                    stream=True,
                    timeout=180
                )
                response.raise_for_status()
                
                with open(lora_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                downloads["pepe_lora"] = f"✅ Downloaded {os.path.getsize(lora_path)} bytes"
                print("✅ Pepe LoRA downloaded")
                
            except Exception as e:
                downloads["pepe_lora"] = f"❌ Failed: {str(e)}"
        else:
            downloads["pepe_lora"] = "✅ Already exists"
        
        return downloads
        
    except Exception as e:
        return {"error": str(e)}

def test_comfyui_fresh():
    """Test ComfyUI in fresh environment"""
    try:
        print("🧪 Testing ComfyUI fresh installation...")
        
        comfyui_path = "/workspace/comfyui"
        
        # Test imports
        original_cwd = os.getcwd()
        os.chdir(comfyui_path)
        
        # Add to Python path
        sys.path.insert(0, comfyui_path)
        
        test_results = {}
        
        # Test core imports
        imports_to_test = [
            "comfy.utils",
            "comfy.model_management", 
            "nodes",
            "execution"
        ]
        
        for import_name in imports_to_test:
            try:
                __import__(import_name)
                test_results[import_name] = "✅ Import successful"
            except Exception as e:
                test_results[import_name] = f"❌ {str(e)[:100]}"
        
        # Test GPU model loading capability
        try:
            import comfy.model_management as model_management
            
            # Check if GPU is being detected properly
            device = model_management.get_torch_device()
            test_results["gpu_device"] = f"✅ Device: {device}"
            
        except Exception as e:
            test_results["gpu_device"] = f"❌ {str(e)}"
        
        os.chdir(original_cwd)
        
        successful_imports = sum(1 for result in test_results.values() if result.startswith("✅"))
        total_imports = len(test_results)
        
        return {
            "results": test_results,
            "success_rate": f"{successful_imports}/{total_imports}",
            "ready": successful_imports >= 3
        }
        
    except Exception as e:
        return {"error": str(e), "ready": False}

def start_comfyui_fresh():
    """Start ComfyUI server in fresh environment"""
    try:
        print("🚀 Starting ComfyUI server in fresh environment...")
        
        comfyui_path = "/workspace/comfyui"
        
        # Kill any existing processes
        try:
            subprocess.run(["pkill", "-f", "main.py"], check=False)
            time.sleep(3)
        except:
            pass
        
        # Set up environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = comfyui_path
        
        # Start server
        cmd = [
            sys.executable, "main.py",
            "--listen", "0.0.0.0",
            "--port", "8188"
        ]
        
        print(f"🔧 Starting: {' '.join(cmd)}")
        print(f"📁 Working dir: {comfyui_path}")
        
        process = subprocess.Popen(
            cmd,
            cwd=comfyui_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        # Wait for startup
        for i in range(60):
            try:
                response = requests.get("http://localhost:8188/", timeout=3)
                if response.status_code == 200:
                    print("✅ ComfyUI server started successfully!")
                    return True, f"Server running (PID: {process.pid})"
            except:
                pass
            
            # Check for crash
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                return False, f"Server crashed: {stderr.decode()[:300]}"
            
            if i % 10 == 0:
                print(f"⏳ Starting server... ({i}/60)")
            
            time.sleep(1)
        
        return False, "Server startup timeout"
        
    except Exception as e:
        return False, f"Server start failed: {str(e)}"

def create_test_workflow():
    """Create a test workflow for Pepe generation"""
    try:
        workflow = {
            "1": {
                "inputs": {"ckpt_name": "sd15.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": "pepe the frog, high quality, detailed, cartoon style, green frog",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": "blurry, low quality, distorted",
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
                    "seed": 123456,
                    "steps": 25,
                    "cfg": 7.5,
                    "sampler_name": "euler",
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
                    "filename_prefix": "fresh_pepe",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }
        
        # Test the workflow
        try:
            response = requests.post(
                "http://localhost:8188/prompt",
                json={"prompt": workflow},
                timeout=10
            )
            
            if response.status_code == 200:
                return True, f"Test workflow queued: {response.json()}"
            else:
                return False, f"Workflow failed: {response.status_code}"
                
        except Exception as e:
            return False, f"Workflow test failed: {str(e)}"
        
    except Exception as e:
        return False, f"Workflow creation failed: {str(e)}"

def handler(event):
    """Fresh container setup handler"""
    print("🐸 PEPE WORKER v9.0 - FRESH START! 🌟")
    print("✨ Setting up everything from scratch on clean container...")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'pepe the frog')
        
        print(f"📝 Processing: {prompt}")
        
        # Step 1: Verify fresh environment
        print("🔄 Step 1: Verifying fresh environment...")
        env_check = verify_fresh_environment()
        
        # Step 2: Install ComfyUI
        print("🔄 Step 2: Installing ComfyUI...")
        comfyui_success, comfyui_msg = install_comfyui_fresh()
        
        # Step 3: Download models
        print("🔄 Step 3: Downloading models...")
        model_downloads = download_models_fresh()
        
        # Step 4: Test ComfyUI
        print("🔄 Step 4: Testing ComfyUI...")
        comfyui_test = test_comfyui_fresh()
        
        # Step 5: Start server
        server_success = False
        server_msg = "Not attempted"
        
        if comfyui_success and comfyui_test.get("ready"):
            print("🔄 Step 5: Starting server...")
            server_success, server_msg = start_comfyui_fresh()
        
        # Step 6: Test workflow if server is running
        workflow_success = False
        workflow_msg = "Not attempted"
        
        if server_success:
            print("🔄 Step 6: Testing workflow...")
            workflow_success, workflow_msg = create_test_workflow()
        
        return {
            "message": f"🐸 FRESH START Pepe setup: {prompt}",
            "status": "success" if workflow_success else "fresh_setup_in_progress",
            "environment_check": env_check,
            "comfyui_installation": {
                "success": comfyui_success,
                "message": comfyui_msg
            },
            "model_downloads": model_downloads,
            "comfyui_test": comfyui_test,
            "server": {
                "success": server_success,
                "message": server_msg
            },
            "workflow_test": {
                "success": workflow_success,
                "message": workflow_msg
            },
            "ready_for_generation": workflow_success,
            "fresh_setup_report": [
                f"🌟 Environment: {'READY' if env_check.get('overall_health', {}).get('ready_for_setup') else 'ISSUES'}",
                f"📦 ComfyUI: {'INSTALLED' if comfyui_success else 'FAILED'}",
                f"📥 Models: {'DOWNLOADED' if all('✅' in str(v) for v in model_downloads.values() if isinstance(v, str)) else 'MISSING'}",
                f"🧪 Tests: {'PASSED' if comfyui_test.get('ready') else 'FAILED'}",
                f"🚀 Server: {'RUNNING' if server_success else 'FAILED'}",
                f"🎯 {'PEPE GENERATOR READY!' if workflow_success else 'FRESH SETUP CONTINUING...'}"
            ],
            "installation_path": "/workspace/comfyui"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "fresh_setup_failed",
            "debug": "Fresh setup exception"
        }

if __name__ == '__main__':
    print("🚀 Starting FRESH Container Pepe Worker v9.0...")
    print("🌟 Clean slate setup with proper GPU support!")
    runpod.serverless.start({'handler': handler})
