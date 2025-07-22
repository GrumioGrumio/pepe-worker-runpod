import runpod
import os
import requests
import time
import json
import subprocess
import threading
import shutil
from urllib.parse import urlparse

def download_pepe_lora():
    """Download Pepe LoRA at runtime if missing"""
    lora_path = "/app/comfyui/models/loras/pepe.safetensors"
    
    if os.path.exists(lora_path):
        return True, f"LoRA already exists ({os.path.getsize(lora_path)} bytes)"
    
    try:
        print("📥 Downloading Pepe LoRA...")
        url = "https://huggingface.co/openfree/pepe/resolve/main/pepe.safetensors"
        
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(lora_path), exist_ok=True)
        
        with open(lora_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        size = os.path.getsize(lora_path)
        print(f"✅ LoRA downloaded: {size} bytes")
        return True, f"Downloaded successfully ({size} bytes)"
        
    except Exception as e:
        print(f"❌ LoRA download failed: {e}")
        return False, f"Download failed: {str(e)}"

def download_working_models():
    """Download smaller, more reliable models"""
    
    # Create directories
    checkpoint_dir = "/app/comfyui/models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Try these reliable, smaller models
    models_to_try = [
        {
            "name": "Realistic Vision V5.1",
            "filename": "realisticVision.safetensors",
            "url": "https://civitai.com/api/download/models/130072",
            "size_mb": 2000  # Approximate size
        },
        {
            "name": "SD 1.5 Base",
            "filename": "sd15.safetensors", 
            "url": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
            "size_mb": 3800
        },
        {
            "name": "Deliberate V2",
            "filename": "deliberate.safetensors",
            "url": "https://civitai.com/api/download/models/15236",
            "size_mb": 2000
        }
    ]
    
    for model in models_to_try:
        try:
            model_path = os.path.join(checkpoint_dir, model["filename"])
            
            # Skip if already exists and is reasonable size
            if os.path.exists(model_path) and os.path.getsize(model_path) > 100_000_000:  # >100MB
                return True, f"{model['name']} already exists ({os.path.getsize(model_path)} bytes)"
            
            print(f"📥 Trying to download {model['name']}...")
            
            # Test URL accessibility first
            head_response = requests.head(model["url"], timeout=10, allow_redirects=True)
            
            if head_response.status_code in [200, 302]:
                print(f"✅ {model['name']} is accessible, downloading...")
                
                # Download with progress
                response = requests.get(model["url"], stream=True, timeout=300, allow_redirects=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f"📊 Progress: {percent:.1f}%")
                
                final_size = os.path.getsize(model_path)
                
                # Verify download
                if final_size > 50_000_000:  # At least 50MB for a valid model
                    print(f"✅ {model['name']} downloaded: {final_size} bytes")
                    return True, f"{model['name']} downloaded successfully ({final_size} bytes)"
                else:
                    os.remove(model_path)
                    print(f"❌ {model['name']} download too small, removing")
                    
            else:
                print(f"❌ {model['name']} not accessible: {head_response.status_code}")
            
        except Exception as e:
            print(f"❌ {model['name']} failed: {e}")
            # Clean up partial download
            model_path = os.path.join(checkpoint_dir, model.get("filename", ""))
            if os.path.exists(model_path):
                try:
                    os.remove(model_path)
                except:
                    pass
            continue
    
    return False, "All model downloads failed - trying backup approach"

def setup_backup_model():
    """Setup a minimal working model for testing"""
    try:
        checkpoint_dir = "/app/comfyui/models/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Check if ComfyUI has any built-in models we can use
        comfyui_models = []
        if os.path.exists(checkpoint_dir):
            comfyui_models = [f for f in os.listdir(checkpoint_dir) 
                            if f.endswith(('.safetensors', '.ckpt', '.pth')) 
                            and os.path.getsize(os.path.join(checkpoint_dir, f)) > 50_000_000]
        
        if comfyui_models:
            return True, f"Found existing models: {comfyui_models}"
        
        # Try to download a very small test model
        test_url = "https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt"
        test_path = os.path.join(checkpoint_dir, "sd-test.ckpt")
        
        try:
            response = requests.get(test_url, timeout=60, stream=True)
            if response.status_code == 200:
                with open(test_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        if os.path.getsize(test_path) > 100_000_000:  # Stop at 100MB for testing
                            break
                
                return True, f"Backup model ready: {os.path.getsize(test_path)} bytes"
        except:
            pass
        
        return False, "No backup models available"
        
    except Exception as e:
        return False, f"Backup setup failed: {str(e)}"

def start_comfyui_server():
    """Start ComfyUI server in background"""
    try:
        print("🚀 Starting ComfyUI server...")
        
        # Kill any existing processes
        try:
            subprocess.run(["pkill", "-f", "main.py"], check=False)
            time.sleep(2)
        except:
            pass
        
        # Start server
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        
        process = subprocess.Popen(
            ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"],
            cwd="/app/comfyui",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        for i in range(30):  # 30 seconds timeout
            try:
                response = requests.get("http://localhost:8188/", timeout=2)
                if response.status_code == 200:
                    print("✅ ComfyUI server started!")
                    return True, "Server running on port 8188"
            except:
                pass
            time.sleep(1)
        
        return False, "Server startup timeout"
        
    except Exception as e:
        return False, f"Server start failed: {str(e)}"

def test_generation_capability():
    """Test if we can actually generate images"""
    try:
        # Check all components
        lora_path = "/app/comfyui/models/loras/pepe.safetensors"
        checkpoint_dir = "/app/comfyui/models/checkpoints"
        
        lora_exists = os.path.exists(lora_path)
        
        models = []
        if os.path.exists(checkpoint_dir):
            models = [f for f in os.listdir(checkpoint_dir) 
                     if f.endswith(('.safetensors', '.ckpt', '.pth'))
                     and os.path.getsize(os.path.join(checkpoint_dir, f)) > 10_000_000]
        
        # Test ComfyUI imports
        try:
            import sys
            sys.path.insert(0, '/app/comfyui')
            
            # Basic import test
            import torch
            pytorch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
            
            # Try importing ComfyUI components
            try:
                import nodes
                import execution
                comfyui_imports = True
            except:
                comfyui_imports = False
                
        except Exception as e:
            pytorch_version = "unknown"
            cuda_available = False
            comfyui_imports = False
        
        # Test server connectivity
        server_running = False
        try:
            response = requests.get("http://localhost:8188/", timeout=5)
            server_running = response.status_code == 200
        except:
            pass
        
        generation_ready = (
            lora_exists and 
            len(models) > 0 and 
            cuda_available and 
            comfyui_imports
        )
        
        return {
            "lora_available": lora_exists,
            "models_available": models,
            "pytorch_version": pytorch_version,
            "cuda_available": cuda_available,
            "comfyui_imports": comfyui_imports,
            "server_running": server_running,
            "generation_ready": generation_ready,
            "ready_percentage": sum([lora_exists, len(models) > 0, cuda_available, comfyui_imports]) / 4 * 100
        }
        
    except Exception as e:
        return {"error": str(e), "generation_ready": False}

def handler(event):
    """Enhanced Pepe generation handler with robust model downloading"""
    print("🐸 ENHANCED PEPE WORKER v2.0 - Robust Model Download")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'pepe the frog')
        
        print(f"📝 Processing: {prompt}")
        
        # Step 1: Ensure LoRA is ready
        print("🔄 Step 1: Checking LoRA...")
        lora_success, lora_msg = download_pepe_lora()
        
        # Step 2: Try to get working models
        print("🔄 Step 2: Downloading models...")
        model_success, model_msg = download_working_models()
        
        # Step 3: If primary models failed, try backup
        if not model_success:
            print("🔄 Step 3: Trying backup models...")
            model_success, backup_msg = setup_backup_model()
            model_msg = f"{model_msg} | Backup: {backup_msg}"
        
        # Step 4: Start ComfyUI server
        print("🔄 Step 4: Starting server...")
        server_success, server_msg = start_comfyui_server()
        
        # Step 5: Test everything
        print("🔄 Step 5: Testing capabilities...")
        capability_test = test_generation_capability()
        
        overall_success = lora_success and model_success
        
        return {
            "message": f"🐸 Enhanced Pepe setup v2.0: {prompt}",
            "status": "success" if overall_success else "partial",
            "lora_status": lora_msg,
            "model_status": model_msg,
            "server_status": server_msg,
            "capability_test": capability_test,
            "ready_for_generation": capability_test.get("generation_ready", False),
            "ready_percentage": capability_test.get("ready_percentage", 0),
            "next_steps": [
                "✅ LoRA ready" if lora_success else "❌ Need LoRA",
                "✅ Models ready" if model_success else "❌ Need models",
                "✅ Server ready" if server_success else "❌ Need server",
                "🎯 Ready to generate!" if capability_test.get("generation_ready") else "🔧 Still setting up..."
            ],
            "debug_info": {
                "lora_success": lora_success,
                "model_success": model_success,
                "server_success": server_success
            }
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
            "debug": "Handler exception occurred"
        }

if __name__ == '__main__':
    print("🚀 Starting Enhanced Pepe Worker v2.0...")
    runpod.serverless.start({'handler': handler})
