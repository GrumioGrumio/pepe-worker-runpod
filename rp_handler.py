import runpod
import os
import subprocess

def handler(event):
    """Check ComfyUI + Pepe LoRA status"""
    print("🐸 Pepe Worker v3 - LoRA Check")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'pepe the frog')
        
        # Check LoRA file
        lora_path = "/app/comfyui/models/loras/pepe.safetensors"
        lora_exists = os.path.exists(lora_path)
        lora_size = os.path.getsize(lora_path) if lora_exists else 0
        
        # List all LoRA files
        lora_dir = "/app/comfyui/models/loras"
        lora_files = os.listdir(lora_dir) if os.path.exists(lora_dir) else []
        
        diagnostics = {
            "comfyui_exists": os.path.exists('/app/comfyui'),
            "comfyui_main_exists": os.path.exists('/app/comfyui/main.py'),
            "lora_directory": lora_dir,
            "lora_files": lora_files,
            "pepe_lora_exists": lora_exists,
            "pepe_lora_size": f"{lora_size} bytes" if lora_exists else "Not found",
            "models_directory": os.listdir('/app/comfyui/models') if os.path.exists('/app/comfyui/models') else []
        }
        
        # Status check
        if lora_exists and lora_size > 1000000:  # LoRA should be at least 1MB
            status_msg = "🎉 ComfyUI + Pepe LoRA ready! Next: Add FLUX model"
            next_step = "Add FLUX.1-dev model for actual generation"
        elif lora_exists:
            status_msg = f"⚠️ Pepe LoRA found but small ({lora_size} bytes) - might be incomplete"
            next_step = "Check LoRA download"
        else:
            status_msg = "❌ Pepe LoRA missing"
            next_step = "Fix LoRA download"
        
        return {
            "message": f"LoRA check: {prompt}",
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
    runpod.serverless.start({'handler': handler})
