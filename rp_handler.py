import runpod
import os
import requests
import time

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

def handler(event):
    """Pepe generation with runtime LoRA download"""
    print("🐸 Pepe Worker v4 - Runtime LoRA Download")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'pepe the frog')
        
        print(f"📝 Processing: {prompt}")
        
        # Check/download LoRA at runtime
        lora_success, lora_msg = download_pepe_lora()
        
        # Check ComfyUI
        comfyui_ready = os.path.exists('/app/comfyui/main.py')
        
        diagnostics = {
            "comfyui_ready": comfyui_ready,
            "lora_status": lora_msg,
            "lora_download_success": lora_success,
            "working_directory": os.getcwd(),
            "models_available": os.listdir('/app/comfyui/models') if os.path.exists('/app/comfyui/models') else []
        }
        
        if comfyui_ready and lora_success:
            status_msg = "🎉 ComfyUI + Pepe LoRA ready! Ready for FLUX model"
            next_step = "Add FLUX model and start real generation"
        else:
            status_msg = "⚠️ Setup incomplete"
            next_step = "Fix missing components"
        
        return {
            "message": f"Runtime setup: {prompt}",
            "status": "success",
            "diagnostics": diagnostics,
            "status_msg": status_msg,
            "next_step": next_step
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

if __name__ == '__main__':
    print("🚀 Starting Pepe Worker with Runtime Setup...")
    runpod.serverless.start({'handler': handler})
