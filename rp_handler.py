import runpod
import os
import requests
import time
import json
import base64
import subprocess
import threading
from PIL import Image
import io

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

def download_flux_model():
    """Download lightweight FLUX model"""
    model_path = "/app/comfyui/models/unet/flux1-schnell.safetensors"
    
    if os.path.exists(model_path):
        return True, f"FLUX model exists ({os.path.getsize(model_path)} bytes)"
    
    try:
        print("📥 Downloading FLUX-Schnell model (this may take a few minutes)...")
        # Using a smaller/faster FLUX variant
        url = "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/flux1-schnell.safetensors"
        
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        size = os.path.getsize(model_path)
        print(f"✅ FLUX model downloaded: {size} bytes")
        return True, f"Downloaded successfully ({size} bytes)"
        
    except Exception as e:
        print(f"❌ FLUX download failed: {e}")
        return False, f"Download failed: {str(e)}"

def start_comfyui_server():
    """Start ComfyUI server"""
    try:
        print("🚀 Starting ComfyUI server...")
        process = subprocess.Popen([
            "python", "/app/comfyui/main.py",
            "--listen", "0.0.0.0",
            "--port", "8188"
        ], cwd="/app/comfyui")
        
        # Wait for server to start
        for i in range(30):
            try:
                response = requests.get("http://127.0.0.1:8188", timeout=2)
                if response.status_code == 200:
                    print("✅ ComfyUI server is running!")
                    return True
            except:
                time.sleep(2)
        
        print("⚠️ ComfyUI server may not be ready yet")
        return False
        
    except Exception as e:
        print(f"❌ ComfyUI server error: {e}")
        return False

def generate_pepe_simple(prompt):
    """Generate Pepe using simple text-based approach"""
    try:
        # Create a simple workflow for FLUX + LoRA
        workflow = {
            "1": {
                "inputs": {"text": f"{prompt}, pepe the frog, meme style, green cartoon frog"},
                "class_type": "CLIPTextEncode"
            },
            "2": {
                "inputs": {"text": "blurry, low quality, realistic"},
                "class_type": "CLIPTextEncode"  
            },
            "3": {
                "inputs": {
                    "seed": int(time.time()),
                    "steps": 4,  # Fast for Schnell
                    "cfg": 1.0,
                    "width": 1024,
                    "height": 1024
                },
                "class_type": "EmptyLatentImage"
            }
        }
        
        # Try to queue the workflow
        response = requests.post("http://127.0.0.1:8188/prompt", 
                               json={"prompt": workflow}, 
                               timeout=10)
        
        if response.status_code == 200:
            return True, "Workflow queued successfully"
        else:
            return False, f"Queue failed: {response.status_code}"
            
    except Exception as e:
        return False, f"Generation error: {str(e)}"

def handler(event):
    """REAL Pepe generation handler"""
    print("🐸 REAL PEPE GENERATION WORKER!")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'pepe the frog')
        
        print(f"📝 Generating REAL Pepe: {prompt}")
        
        # Step 1: Ensure LoRA is ready
        lora_success, lora_msg = download_pepe_lora()
        if not lora_success:
            return {"error": f"LoRA failed: {lora_msg}", "status": "failed"}
        
        # Step 2: Check/download FLUX model
        flux_success, flux_msg = download_flux_model()
        
        # Step 3: Start ComfyUI server in background
        threading.Thread(target=start_comfyui_server, daemon=True).start()
        
        # Step 4: Try simple generation test
        time.sleep(5)  # Give server time to start
        gen_success, gen_msg = generate_pepe_simple(prompt)
        
        return {
            "message": f"🐸 REAL Pepe generation attempt: {prompt}",
            "status": "success",
            "lora_status": lora_msg,
            "flux_status": flux_msg,
            "generation_test": gen_msg,
            "ready_status": {
                "lora_ready": lora_success,
                "flux_ready": flux_success,
                "generation_attempted": gen_success
            },
            "next_step": "Full image generation pipeline" if (lora_success and flux_success) else "Fix missing components"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

if __name__ == '__main__':
    print("🚀 Starting REAL Pepe Generation Worker...")
    runpod.serverless.start({'handler': handler})
