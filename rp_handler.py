import runpod
import os
import requests
import time
import json
import subprocess
import base64
from PIL import Image
import io

def check_comfyui_status():
    """Check if ComfyUI server is running"""
    try:
        response = requests.get("http://localhost:8188/", timeout=5)
        return response.status_code == 200
    except:
        return False

def quick_setup_comfyui():
    """Quick setup if ComfyUI isn't running"""
    try:
        print("🔧 ComfyUI not running, starting quick setup...")
        
        # Check if files exist
        comfyui_path = "/workspace/comfyui"
        lora_path = "/workspace/comfyui/models/loras/pepe.safetensors"
        model_path = "/workspace/comfyui/models/checkpoints/sd15.safetensors"
        
        setup_needed = []
        
        if not os.path.exists(comfyui_path):
            setup_needed.append("ComfyUI installation")
        if not os.path.exists(lora_path):
            setup_needed.append("Pepe LoRA")
        if not os.path.exists(model_path):
            setup_needed.append("SD 1.5 model")
        
        if setup_needed:
            return False, f"Missing: {', '.join(setup_needed)} - need full fresh setup"
        
        # Try to start server
        print("🚀 Starting ComfyUI server...")
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = comfyui_path
        
        process = subprocess.Popen(
            ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"],
            cwd=comfyui_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        # Wait for startup
        for i in range(60):
            try:
                response = requests.get("http://localhost:8188/", timeout=3)
                if response.status_code == 200:
                    print("✅ ComfyUI server started!")
                    return True, "Server started successfully"
            except:
                pass
            
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                return False, f"Server crashed: {stderr.decode()[:200]}"
            
            time.sleep(1)
        
        return False, "Server startup timeout"
        
    except Exception as e:
        return False, f"Quick setup failed: {str(e)}"

def install_comfyui_fresh():
    """Install ComfyUI from scratch"""
    try:
        print("📦 Installing ComfyUI from scratch...")
        
        app_dir = "/workspace/comfyui"
        os.makedirs(app_dir, exist_ok=True)
        
        # Clone ComfyUI
        result = subprocess.run([
            "git", "clone", "https://github.com/comfyanonymous/ComfyUI.git", app_dir
        ], capture_output=True, timeout=120)
        
        if result.returncode != 0:
            return False, f"Git clone failed: {result.stderr.decode()}"
        
        # Install dependencies
        subprocess.run([
            "pip", "install", "-r", "requirements.txt"
        ], cwd=app_dir, capture_output=True, timeout=180)
        
        return True, f"ComfyUI installed at {app_dir}"
        
    except Exception as e:
        return False, f"ComfyUI installation failed: {str(e)}"

def download_models_fresh():
    """Download models if missing"""
    try:
        print("📥 Downloading models...")
        
        downloads = {}
        
        # Download SD 1.5 model
        model_path = "/workspace/comfyui/models/checkpoints/sd15.safetensors"
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            response = requests.get(
                "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
                stream=True, timeout=600
            )
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            
            downloads["model"] = f"✅ Downloaded {os.path.getsize(model_path)} bytes"
        else:
            downloads["model"] = "✅ Already exists"
        
        # Download Pepe LoRA
        lora_path = "/workspace/comfyui/models/loras/pepe.safetensors"
        if not os.path.exists(lora_path):
            os.makedirs(os.path.dirname(lora_path), exist_ok=True)
            
            response = requests.get(
                "https://huggingface.co/openfree/pepe/resolve/main/pepe.safetensors",
                stream=True, timeout=180
            )
            response.raise_for_status()
            
            with open(lora_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            downloads["lora"] = f"✅ Downloaded {os.path.getsize(lora_path)} bytes"
        else:
            downloads["lora"] = "✅ Already exists"
        
        return downloads
        
    except Exception as e:
        return {"error": str(e)}

def generate_pepe_with_lora(prompt):
    """Generate Pepe with LoRA - only if server is running"""
    try:
        if not check_comfyui_status():
            return {"error": "ComfyUI server not running"}
        
        print(f"🐸 Generating REAL Pepe with LoRA: {prompt}")
        
        # Enhanced workflow with LoRA
        workflow = {
            "1": {
                "inputs": {"ckpt_name": "sd15.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "lora_name": "pepe.safetensors",
                    "strength_model": 1.0,
                    "strength_clip": 1.0,
                    "model": ["1", 0],
                    "clip": ["1", 1]
                },
                "class_type": "LoraLoader"
            },
            "3": {
                "inputs": {
                    "text": f"pepe the frog, {prompt}, meme style, simple cartoon, green frog character, feels good man",
                    "clip": ["2", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "text": "blurry, low quality, distorted, realistic, photorealistic, human, anime, complex background, multiple characters",
                    "clip": ["2", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "5": {
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "6": {
                "inputs": {
                    "seed": int(time.time()) % 1000000,
                    "steps": 25,
                    "cfg": 8.0,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["2", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0]
                },
                "class_type": "KSampler"
            },
            "7": {
                "inputs": {
                    "samples": ["6", 0],
                    "vae": ["1", 2]
                },
                "class_type": "VAEDecode"
            },
            "8": {
                "inputs": {
                    "filename_prefix": f"real_pepe_{int(time.time())}",
                    "images": ["7", 0]
                },
                "class_type": "SaveImage"
            }
        }
        
        # Submit generation
        response = requests.post(
            "http://localhost:8188/prompt",
            json={"prompt": workflow},
            timeout=30
        )
        
        if response.status_code != 200:
            return {"error": f"Generation failed: {response.status_code}"}
        
        prompt_data = response.json()
        prompt_id = prompt_data.get("prompt_id")
        
        if not prompt_id:
            return {"error": "No prompt ID returned"}
        
        print(f"✅ Pepe generation queued: {prompt_id}")
        
        # Wait for completion
        for i in range(120):
            try:
                queue_response = requests.get("http://localhost:8188/queue", timeout=5)
                if queue_response.status_code == 200:
                    queue_data = queue_response.json()
                    
                    running = queue_data.get("queue_running", [])
                    pending = queue_data.get("queue_pending", [])
                    
                    still_processing = any(
                        item[1].get("prompt_id") == prompt_id 
                        for item in running + pending
                        if len(item) > 1 and isinstance(item[1], dict)
                    )
                    
                    if not still_processing:
                        print("🎉 Pepe generation completed!")
                        break
                
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️ Queue check error: {e}")
                time.sleep(1)
        
        # Find generated image
        output_dir = "/workspace/comfyui/output"
        image_files = []
        
        if os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        if time.time() - os.path.getctime(file_path) < 300:
                            image_files.append(file_path)
        
        if not image_files:
            return {"error": "No generated images found"}
        
        latest_image = max(image_files, key=os.path.getctime)
        
        # Convert to base64
        with open(latest_image, 'rb') as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        return {
            "success": True,
            "image_path": latest_image,
            "image_base64": img_base64,
            "lora_used": True,
            "prompt_id": prompt_id
        }
        
    except Exception as e:
        return {"error": f"Pepe generation failed: {str(e)}"}

def handler(event):
    """Auto-setup Pepe handler with LoRA"""
    print("🐸 AUTO-SETUP PEPE GENERATOR v12.0! 🔧")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'pepe the frog')
        action = input_data.get('action', 'generate')
        
        print(f"📝 Request: {prompt}")
        
        # Check if ComfyUI is running
        server_running = check_comfyui_status()
        print(f"🖥️ ComfyUI server: {'✅ Running' if server_running else '❌ Not running'}")
        
        if not server_running:
            print("🔧 Server not running, attempting setup...")
            
            # Try quick setup first
            quick_success, quick_msg = quick_setup_comfyui()
            
            if not quick_success:
                print("📦 Quick setup failed, doing full fresh install...")
                
                # Do full fresh setup
                comfyui_success, comfyui_msg = install_comfyui_fresh()
                if not comfyui_success:
                    return {"error": f"ComfyUI install failed: {comfyui_msg}"}
                
                model_downloads = download_models_fresh()
                if "error" in model_downloads:
                    return {"error": f"Model download failed: {model_downloads['error']}"}
                
                # Try to start server after fresh install
                quick_success, quick_msg = quick_setup_comfyui()
                
                if not quick_success:
                    return {
                        "error": "Full setup completed but server won't start",
                        "details": quick_msg,
                        "comfyui_install": comfyui_msg,
                        "downloads": model_downloads
                    }
        
        # Now try to generate with LoRA
        if action == 'status':
            return {
                "message": "🐸 Auto-Setup Pepe Generator",
                "server_running": check_comfyui_status(),
                "files": {
                    "comfyui": os.path.exists("/workspace/comfyui"),
                    "pepe_lora": os.path.exists("/workspace/comfyui/models/loras/pepe.safetensors"),
                    "sd15_model": os.path.exists("/workspace/comfyui/models/checkpoints/sd15.safetensors")
                }
            }
        
        # Generate Pepe
        generation_result = generate_pepe_with_lora(prompt)
        
        if generation_result.get("success"):
            return {
                "message": f"🐸 REAL PEPE GENERATED WITH LORA!",
                "prompt": prompt,
                "generation": generation_result,
                "auto_setup": "Server was automatically configured",
                "lora_used": "pepe.safetensors at full strength",
                "ready_for_generation": True
            }
        else:
            return {
                "error": "Pepe generation failed after setup",
                "details": generation_result,
                "server_status": check_comfyui_status()
            }
        
    except Exception as e:
        return {
            "error": str(e),
            "debug": "Auto-setup handler exception"
        }

if __name__ == '__main__':
    print("🚀 Starting Auto-Setup Pepe Generator with LoRA...")
    runpod.serverless.start({'handler': handler})
