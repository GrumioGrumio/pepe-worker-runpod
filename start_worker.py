import subprocess
import time
import threading
import os
import sys

def start_comfyui():
    """Start ComfyUI server"""
    print("🚀 Starting ComfyUI server...")
    try:
        # Change to ComfyUI directory
        os.chdir('/comfyui')
        
        # Start ComfyUI with proper settings
        subprocess.run([
            sys.executable, "main.py", 
            "--listen", "0.0.0.0", 
            "--port", "8188",
            "--disable-auto-launch"
        ])
    except Exception as e:
        print(f"❌ ComfyUI start error: {e}")

def wait_for_comfyui():
    """Wait for ComfyUI to be ready"""
    import requests
    
    print("⏳ Waiting for ComfyUI to start...")
    max_attempts = 60
    
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://127.0.0.1:8188", timeout=5)
            if response.status_code == 200:
                print("✅ ComfyUI is ready!")
                return True
        except:
            pass
        
        print(f"⏳ Attempt {attempt + 1}/{max_attempts}...")
        time.sleep(5)
    
    print("❌ ComfyUI failed to start")
    return False

def start_runpod_handler():
    """Start RunPod handler after ComfyUI is ready"""
    if wait_for_comfyui():
        print("🚀 Starting RunPod handler...")
        os.chdir('/comfyui')
        subprocess.run([sys.executable, "rp_handler.py"])
    else:
        print("❌ Cannot start handler - ComfyUI not ready")

if __name__ == "__main__":
    print("🎭 Starting Pepe Worker Services...")
    
    # Start ComfyUI in background thread
    comfyui_thread = threading.Thread(target=start_comfyui)
    comfyui_thread.daemon = True
    comfyui_thread.start()
    
    # Start RunPod handler (this will wait for ComfyUI)
    start_runpod_handler()
