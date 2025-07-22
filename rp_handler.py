import runpod
import json
import base64
import requests
import time
import subprocess
import threading
import os

def start_comfyui():
    """Start ComfyUI in background"""
    try:
        print("🚀 Starting ComfyUI server...")
        subprocess.Popen([
            "python", "/app/comfyui/main.py", 
            "--listen", "0.0.0.0", 
            "--port", "8188"
        ])
        time.sleep(30)  # Give ComfyUI time to start
        print("✅ ComfyUI should be ready")
    except Exception as e:
        print(f"❌ ComfyUI start error: {e}")

def handler(event):
    """Enhanced Pepe handler with ComfyUI preparation"""
    print("🐸 Enhanced Pepe Worker Started!")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'pepe the frog')
        
        print(f"📝 Received prompt: {prompt}")
        
        # Check if ComfyUI is available
        try:
            response = requests.get("http://127.0.0.1:8188", timeout=5)
            comfyui_status = "ComfyUI running" if response.status_code == 200 else "ComfyUI not responding"
        except:
            comfyui_status = "ComfyUI not available (will start in background)"
            # Start ComfyUI in background for next request
            threading.Thread(target=start_comfyui, daemon=True).start()
        
        # Check if Pepe LoRA exists
        lora_path = "/app/comfyui/models/loras/pepe.safetensors"
        lora_status = "Pepe LoRA ready" if os.path.exists(lora_pa
