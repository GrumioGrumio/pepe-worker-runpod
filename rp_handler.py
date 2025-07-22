import runpod
import os
import subprocess
import requests

def handler(event):
    """Enhanced diagnostic with ComfyUI check"""
    print("🐸 Pepe Worker v2 - ComfyUI Check")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'pepe the frog')
        
        # Enhanced diagnostics
        diagnostics = {
            "working_directory": os.getcwd(),
            "files_in_app": os.listdir('/app') if os.path.exists('/app') else [],
            "comfyui_exists": os.path.exists('/app/comfyui'),
            "comfyui_main_exists": os.path.exists('/app/comfyui/main.py'),
            "lora_directory_exists": os.path.exists('/app/comfyui/models/loras'),
            "comfyui_models": os.listdir('/app/comfyui/models') if os.path.exists('/app/comfyui/models') else []
        }
        
        # Try to start ComfyUI briefly to test
        comfyui_test = "Not tested"
        if os.path.exists('/app/comfyui/main.py'):
            try:
                # Just test if ComfyUI can import (don't actually start server)
                result = subprocess.run(
                    ["python", "/app/comfyui/main.py", "--help"],
                    capture_output=True, 
                    timeout=10,
                    cwd="/app/comfyui"
                )
                comfyui_test = "ComfyUI can run!" if result.returncode == 0 else f"Error: {result.stderr[:100]}"
            except Exception as e:
                comfyui_test = f"Test failed: {str(e)[:50]}"
        
        return {
            "message": f"Enhanced check: {prompt}",
            "status": "success", 
            "diagnostics": diagnostics,
            "comfyui_test": comfyui_test,
            "next_step": "Add Pepe LoRA download if ComfyUI works"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
