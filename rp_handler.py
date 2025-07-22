import runpod
import os
import requests

def handler(event):
    """Simple diagnostic handler"""
    print("🐸 Pepe Diagnostic Worker")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'pepe the frog')
        
        # Quick diagnostics
        diagnostics = {
            "working_directory": os.getcwd(),
            "files_in_app": os.listdir('/app') if os.path.exists('/app') else "No /app directory",
            "comfyui_exists": os.path.exists('/app/comfyui'),
            "lora_path": "/app/comfyui/models/loras/pepe.safetensors",
            "lora_exists": os.path.exists('/app/comfyui/models/loras/pepe.safetensors') if os.path.exists('/app/comfyui/models/loras') else "LoRA directory missing"
        }
        
        return {
            "message": f"Received: {prompt}",
            "status": "success",
            "diagnostics": diagnostics,
            "recommendation": "Let's build this step by step"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
