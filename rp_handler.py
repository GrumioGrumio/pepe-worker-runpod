import runpod
import os
import requests
import time
import json
import subprocess
import threading

def download_pepe_lora():
    """Download Pepe LoRA at runtime if missing"""
    lora_path = "/app/comfyui/models/loras/pepe.safetensors"
    
    if os.path.exists(lora_path):
        return True, f"LoRA already exists ({os.path.getsize(lora_path)} bytes)"
    
    try:
        print("📥 Downloading Pepe LoRA...")
        url = "https://huggingface.co/openfree/pepe/resolve/main/pepe.safetensors"
        
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(lora_path, 'wb') as f:
            f.write(response.content)
        
        size = os.path.getsize(lora_path)
        print(f"✅ LoRA downloaded: {size} bytes")
        return True, f"Downloaded successfully ({size} bytes)"
        
    except Exception as e:
        print(f"❌ LoRA download failed: {e}")
        return False, f"Download failed: {str(e)}"

def download_alternative_model():
    """Download a working alternative model (SD XL or similar)"""
    # Try multiple smaller, working models
    models_to_try = [
        {
            "name": "SD 1.5",
            "url": "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors",
            "path": "/app/comfyui/models/checkpoints/sd15.safetensors"
        },
        {
            "name": "SD 2.1",
            "url": "https://huggingface.co/stabilityai/stable-diffusion-2-1/resolve/main/v2-1_768-nonema-pruned.safetensors", 
            "path": "/app/comfyui/models/checkpoints/sd21.safetensors"
        }
    ]
    
    for model in models_to_try:
        try:
            model_path = model["path"]
            
            if os.path.exists(model_path):
                return True, f"{model['name']} already exists ({os.path.getsize(model_path)} bytes)"
            
            print(f"📥 Trying to download {model['name']}...")
            
            response = requests.head(model["url"], timeout=10)
            if response.status_code == 200:
                print(f"✅ {model['name']} is accessible, downloading...")
                
                response = requests.get(model["url"], stream=True, timeout=300)
                response.raise_for_status()
                
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                size = os.path.getsize(model_path)
                print(f"✅ {model['name']} downloaded: {size} bytes")
                return True, f"{model['name']} downloaded successfully ({size} bytes)"
            
        except Exception as e:
            print(f"❌ {model['name']} failed: {e}")
            continue
    
    return False, "All model downloads failed"

def test_simple_generation():
    """Test basic image generation without ComfyUI server"""
    try:
        # Check if we have the basic components
        lora_exists = os.path.exists("/app/comfyui/models/loras/pepe.safetensors")
        model_dir = "/app/comfyui/models/checkpoints"
        models = os.listdir(model_dir) if os.path.exists(model_dir) else []
        
        # Test if ComfyUI can be imported
        import sys
        sys.path.append('/app/comfyui')
        
        try:
            # Simple import test
            result = subprocess.run([
                "python", "-c", "import torch; print('PyTorch OK')"
            ], capture_output=True, timeout=5, cwd="/app/comfyui")
            
            pytorch_ok = "PyTorch OK" in result.stdout.decode()
            
        except Exception as e:
            pytorch_ok = False
        
        return {
            "lora_available": lora_exists,
            "models_available": models,
            "pytorch_working": pytorch_ok,
            "comfyui_path_exists": os.path.exists("/app/comfyui/main.py"),
            "generation_ready": lora_exists and len(models) > 0 and pytorch_ok
        }
        
    except Exception as e:
        return {"error": str(e)}

def handler(event):
    """Enhanced Pepe generation with working models"""
    print("🐸 ENHANCED PEPE WORKER - Alternative Models")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'pepe the frog')
        
        print(f"📝 Processing: {prompt}")
        
        # Step 1: Ensure LoRA is ready
        lora_success, lora_msg = download_pepe_lora()
        
        # Step 2: Try alternative models
        model_success, model_msg = download_alternative_model()
        
        # Step 3: Test generation readiness
        gen_test = test_simple_generation()
        
        return {
            "message": f"🐸 Enhanced Pepe setup: {prompt}",
            "status": "success",
            "lora_status": lora_msg,
            "model_status": model_msg,
            "generation_test": gen_test,
            "ready_for_generation": lora_success and model_success,
            "next_step": "Real image generation" if (lora_success and model_success) else "Alternative model approach",
            "alternatives": "Using SD models instead of FLUX (more reliable)"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

if __name__ == '__main__':
    print("🚀 Starting Enhanced Pepe Worker...")
    runpod.serverless.start({'handler': handler})
