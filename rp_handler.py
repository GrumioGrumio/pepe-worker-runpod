import runpod
import os
import requests
import time
import json
import subprocess
import sys
import base64

def setup_volume_correct_path():
    """Setup volume using the correct mount path /runpod-volume"""
    try:
        print("📦 Setting up volume at correct path: /runpod-volume")
        
        setup_status = {}
        
        # Use correct volume mount path
        volume_root = "/runpod-volume"
        comfyui_path = f"{volume_root}/comfyui"
        
        setup_status["volume_path"] = volume_root
        setup_status["volume_exists"] = os.path.exists(volume_root)
        
        if not os.path.exists(volume_root):
            return {"error": "Volume not mounted at /runpod-volume"}
        
        # Check if already set up
        if os.path.exists(comfyui_path) and os.path.exists(f"{comfyui_path}/main.py"):
            setup_status["already_setup"] = True
            print("✅ Volume already has ComfyUI!")
        else:
            setup_status["already_setup"] = False
            print("🔧 Fresh volume - installing ComfyUI...")
            
            # Install ComfyUI
            print("📥 Cloning ComfyUI...")
            
            result = subprocess.run([
                "git", "clone", "https://github.com/comfyanonymous/ComfyUI.git", comfyui_path
            ], capture_output=True, timeout=120)
            
            if result.returncode != 0:
                setup_status["comfyui_install"] = f"❌ Failed: {result.stderr.decode()}"
            else:
                setup_status["comfyui_install"] = "✅ ComfyUI cloned successfully"
                
                # Install dependencies
                print("📦 Installing dependencies...")
                pip_result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
                ], cwd=comfyui_path, capture_output=True, timeout=300)
                
                setup_status["dependencies"] = "✅ Dependencies installed" if pip_result.returncode == 0 else f"⚠️ Some failed: {pip_result.stderr.decode()[:200]}"
        
        # Create model directories
        models_dir = f"{comfyui_path}/models"
        checkpoints_dir = f"{models_dir}/checkpoints"
        loras_dir = f"{models_dir}/loras"
        
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(loras_dir, exist_ok=True)
        
        # Download SD 1.5 model
        model_path = f"{checkpoints_dir}/sd15.safetensors"
        if os.path.exists(model_path) and os.path.getsize(model_path) > 1000000000:
            setup_status["sd15_model"] = f"✅ Already exists ({os.path.getsize(model_path)} bytes)"
        else:
            print("📥 Downloading SD 1.5 model (4GB)...")
            try:
                response = requests.get(
                    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
                    stream=True, timeout=900
                )
                response.raise_for_status()
                
                downloaded = 0
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if downloaded % (500*1024*1024) == 0:
                                print(f"📊 SD model: {downloaded // (1024*1024)}MB...")
                
                setup_status["sd15_model"] = f"✅ Downloaded {os.path.getsize(model_path)} bytes"
                
            except Exception as e:
                setup_status["sd15_model"] = f"❌ Download failed: {str(e)}"
        
        # Download Pepe LoRA
        lora_path = f"{loras_dir}/pepe.safetensors"
        if os.path.exists(lora_path) and os.path.getsize(lora_path) > 100000000:
            setup_status["pepe_lora"] = f"✅ Already exists ({os.path.getsize(lora_path)} bytes)"
        else:
            print("📥 Downloading Pepe LoRA (171MB)...")
            try:
                response = requests.get(
                    "https://huggingface.co/openfree/pepe/resolve/main/pepe.safetensors",
                    stream=True, timeout=300
                )
                response.raise_for_status()
                
                with open(lora_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                setup_status["pepe_lora"] = f"✅ Downloaded {os.path.getsize(lora_path)} bytes"
                
            except Exception as e:
                setup_status["pepe_lora"] = f"❌ Download failed: {str(e)}"
        
        # Create output directory
        output_dir = f"{comfyui_path}/output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Test imports
        try:
            os.chdir(comfyui_path)
            sys.path.insert(0, comfyui_path)
            
            import comfy.utils
            import nodes
            setup_status["imports_test"] = "✅ ComfyUI imports working"
            
        except Exception as e:
            setup_status["imports_test"] = f"❌ Import test failed: {str(e)}"
        
        return setup_status
        
    except Exception as e:
        return {"error": f"Volume setup failed: {str(e)}"}

def start_comfyui_from_volume():
    """Start ComfyUI from volume"""
    try:
        print("🚀 Starting ComfyUI from volume...")
        
        comfyui_path = "/runpod-volume/comfyui"
        
        if not os.path.exists(f"{comfyui_path}/main.py"):
            return False, "ComfyUI not found in volume"
        
        # Kill existing
        try:
            subprocess.run(["pkill", "-f", "main.py"], check=False)
            time.sleep(2)
        except:
            pass
        
        # Start server
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = comfyui_path
        
        process = subprocess.Popen(
            [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", "8188"],
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
                    print("✅ ComfyUI server started from volume!")
                    return True, f"Server running (PID: {process.pid})"
            except:
                pass
            
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                return False, f"Server crashed: {stderr.decode()[:200]}"
            
            time.sleep(1)
        
        return False, "Server startup timeout"
        
    except Exception as e:
        return False, f"Server start failed: {str(e)}"

def generate_real_pepe_from_volume(prompt):
    """Generate real Pepe using volume-based ComfyUI"""
    try:
        print(f"🐸 Generating REAL Pepe from volume: {prompt}")
        
        # Workflow with Pepe LoRA
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
                    "text": f"pepe the frog, {prompt}, meme style, simple cartoon, green frog, feels good man",
                    "clip": ["2", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "text": "blurry, low quality, distorted, realistic, photorealistic, anime, complex background",
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
                    "filename_prefix": f"VOLUME_PEPE_{int(time.time())}",
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
        
        print(f"✅ REAL Pepe queued: {prompt_id}")
        
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
                        print("🎉 REAL Pepe generation completed!")
                        break
                
                if i % 15 == 0:
                    print(f"⏳ Generating REAL Pepe... ({i}/120s)")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️ Queue check error: {e}")
                time.sleep(1)
        
        # Find generated image
        output_dir = "/runpod-volume/comfyui/output"
        image_files = []
        
        if os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        if time.time() - os.path.getctime(file_path) < 300:
                            image_files.append(file_path)
        
        if not image_files:
            return {"error": "No REAL Pepe images found"}
        
        # Get latest image
        latest_image = max(image_files, key=os.path.getctime)
        
        # Convert to base64
        with open(latest_image, 'rb') as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        return {
            "success": True,
            "image_path": latest_image,
            "image_base64": img_base64,
            "image_size": len(img_data),
            "prompt_id": prompt_id,
            "lora_used": "pepe.safetensors (full strength)",
            "method": "REAL PEPE FROM VOLUME"
        }
        
    except Exception as e:
        return {"error": f"REAL Pepe generation failed: {str(e)}"}

def handler(event):
    """Corrected volume setup and Pepe generation"""
    print("🗄️ VOLUME SETUP v18.0 - CORRECTED PATH! 📦")
    print("🎯 Using correct mount path: /runpod-volume")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'wearing crown and sunglasses')
        action = input_data.get('action', 'setup_and_generate')
        
        # Step 1: Setup volume
        print("🔄 Step 1: Setting up volume...")
        setup_result = setup_volume_correct_path()
        
        if "error" in setup_result:
            return {
                "error": "Volume setup failed",
                "details": setup_result,
                "mount_path": "/runpod-volume"
            }
        
        # Step 2: Start ComfyUI
        print("🔄 Step 2: Starting ComfyUI...")
        server_success, server_msg = start_comfyui_from_volume()
        
        if not server_success:
            return {
                "error": "Failed to start ComfyUI from volume",
                "details": server_msg,
                "setup_result": setup_result
            }
        
        # Step 3: Generate REAL Pepe
        print(f"🔄 Step 3: Generating REAL Pepe: {prompt}")
        generation_result = generate_real_pepe_from_volume(prompt)
        
        if generation_result.get("success"):
            return {
                "message": "🗄️ VOLUME PEPE GENERATED! 🎉",
                "prompt": prompt,
                "setup": setup_result,
                "server": server_msg,
                "generation": generation_result,
                "volume_path": "/runpod-volume",
                "success": True,
                "note": "Volume is working! Future requests will be instant!"
            }
        else:
            return {
                "error": "REAL Pepe generation failed",
                "details": generation_result,
                "setup": setup_result,
                "server": server_msg,
                "volume_working": True
            }
        
    except Exception as e:
        return {
            "error": str(e),
            "debug": "Corrected volume handler exception"
        }

if __name__ == '__main__':
    print("🚀 Starting Corrected Volume Setup...")
    runpod.serverless.start({'handler': handler})
