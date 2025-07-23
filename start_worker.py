import subprocess
import time
import threading
import os
import sys
import requests

def test_gpu():
    """Test GPU availability"""
    try:
        import torch
        print(f"🔧 PyTorch version: {torch.__version__}")
        print(f"🔧 CUDA available: {torch.cuda.is_available()}")
        print(f"🔧 CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
        return torch.cuda.is_available()
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def start_comfyui():
    """Start ComfyUI server"""
    print("🚀 Starting ComfyUI server...")
    try:
        # Change to ComfyUI directory (corrected path)
        comfyui_path = '/app/comfyui'
        os.chdir(comfyui_path)
        
        # Verify we're in the right place
        if not os.path.exists('main.py'):
            print(f"❌ main.py not found in {comfyui_path}")
            print(f"📁 Directory contents: {os.listdir('.')}")
            return
        
        print(f"✅ Found ComfyUI at {comfyui_path}")
        
        # Start ComfyUI with proper settings
        cmd = [
            sys.executable, "main.py", 
            "--listen", "0.0.0.0", 
            "--port", "8188",
            "--disable-auto-launch",
            "--verbose"
        ]
        
        print(f"🔧 Starting ComfyUI with: {' '.join(cmd)}")
        
        # Start ComfyUI process
        subprocess.run(cmd, cwd=comfyui_path)
        
    except Exception as e:
        print(f"❌ ComfyUI start error: {e}")
        import traceback
        traceback.print_exc()

def wait_for_comfyui():
    """Wait for ComfyUI to be ready"""
    print("⏳ Waiting for ComfyUI to start...")
    max_attempts = 60  # 5 minutes total
    
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://127.0.0.1:8188", timeout=5)
            if response.status_code == 200:
                print("✅ ComfyUI is ready!")
                
                # Additional check - test object_info endpoint
                try:
                    info_response = requests.get("http://127.0.0.1:8188/object_info", timeout=5)
                    if info_response.status_code == 200:
                        info = info_response.json()
                        print(f"✅ ComfyUI loaded {len(info)} node types")
                    else:
                        print(f"⚠️ object_info endpoint issue: {info_response.status_code}")
                except Exception as e:
                    print(f"⚠️ Could not check object_info: {e}")
                
                return True
        except Exception as e:
            if attempt % 10 == 0:  # Print every 10 attempts to reduce spam
                print(f"⏳ Attempt {attempt + 1}/{max_attempts} - ComfyUI not ready yet: {e}")
        
        time.sleep(5)
    
    print("❌ ComfyUI failed to start within timeout")
    return False

def start_runpod_handler():
    """Start RunPod handler after ComfyUI is ready"""
    if wait_for_comfyui():
        print("🚀 Starting RunPod handler...")
        
        # Make sure we're in the ComfyUI directory where the handler is
        comfyui_path = '/app/comfyui'
        os.chdir(comfyui_path)
        
        # Verify handler exists
        if not os.path.exists('rp_handler.py'):
            print(f"❌ rp_handler.py not found in {comfyui_path}")
            print(f"📁 Directory contents: {os.listdir('.')}")
            return
        
        print("✅ Found RunPod handler, starting...")
        
        try:
            subprocess.run([sys.executable, "rp_handler.py"], cwd=comfyui_path)
        except Exception as e:
            print(f"❌ RunPod handler error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Cannot start handler - ComfyUI not ready")

if __name__ == "__main__":
    print("🎭 Starting Pepe Worker Services...")
    print("="*50)
    
    # Test GPU first
    gpu_available = test_gpu()
    if not gpu_available:
        print("⚠️ Warning: GPU not available, ComfyUI may run slowly on CPU")
    
    print("="*50)
    
    # Start ComfyUI in background thread
    print("🔄 Starting ComfyUI in background...")
    comfyui_thread = threading.Thread(target=start_comfyui)
    comfyui_thread.daemon = True
    comfyui_thread.start()
    
    # Give ComfyUI a moment to start
    time.sleep(2)
    
    # Start RunPod handler (this will wait for ComfyUI)
    print("🔄 Starting RunPod handler...")
    start_runpod_handler()
    
    print("🎭 Worker services completed")
