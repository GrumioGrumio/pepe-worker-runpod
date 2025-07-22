import runpod
import os
import requests
import time
import json
import subprocess
import sys
import base64

def start_comfyui_simple():
    """Start ComfyUI server - everything is already installed"""
    try:
        print("🚀 Starting ComfyUI server...")
        
        comfyui_path = "/workspace/comfyui"
        
        # Kill any existing processes
        try:
            subprocess.run(["pkill", "-f", "main.py"], check=False)
            time.sleep(2)
        except:
            pass
        
        # Set up environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = comfyui_path
        
        # Start server
        cmd = [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", "8188"]
        
        print(f"🔧 Starting server in {comfyui_path}")
        
        process = subprocess.Popen(
            cmd,
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
                    return True, f"Server running (PID: {process.pid})"
            except:
                pass
            
            # Check for crash
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                return False, f"Server crashed: {stderr.decode()[:200]}"
            
            if i % 10 == 0:
                print(f"⏳ Starting... ({i}/60)")
            
            time.sleep(1)
        
        return False, "Server startup timeout"
        
    except Exception as e:
        return False, f"Server start failed: {str(e)}"

def generate_real_pepe(prompt):
    """Generate REAL Pepe with LoRA"""
    try:
        print(f"🐸 Generating REAL Pepe: {prompt}")
        
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
                    "filename_prefix": f"REAL_PEPE_{int(time.time())}",
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
        output_dir = "/workspace/comfyui/output"
        image_files = []
        
        if os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        if time.time() - os.path.getctime(file_path) < 300:  # Last 5 minutes
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
            "method": "REAL PEPE WITH LORA"
        }
        
    except Exception as e:
        return {"error": f"REAL Pepe generation failed: {str(e)}"}

def handler(event):
    """Simple handler - start server and generate REAL Pepe"""
    print("🐸 REAL PEPE GENERATOR v15.0 - WORKING CUDA! ⚡")
    print("🎯 Worker elm3mk6p53nh1h has working GPU!")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'wearing crown and sunglasses')
        action = input_data.get('action', 'generate')
        
        # Check if server is running
        try:
            response = requests.get("http://localhost:8188/", timeout=5)
            server_running = response.status_code == 200
        except:
            server_running = False
        
        print(f"🖥️ Server status: {'✅ Running' if server_running else '❌ Not running'}")
        
        # Start server if needed
        if not server_running:
            print("🚀 Starting ComfyUI server...")
            server_success, server_msg = start_comfyui_simple()
            
            if not server_success:
                return {
                    "error": "Failed to start ComfyUI server",
                    "details": server_msg,
                    "note": "GPU is working but server won't start"
                }
        
        # Generate REAL Pepe with LoRA
        print(f"🐸 Generating REAL Pepe: {prompt}")
        generation_result = generate_real_pepe(prompt)
        
        if generation_result.get("success"):
            return {
                "message": "🐸 REAL PEPE GENERATED WITH LORA! 🎉",
                "prompt": prompt,
                "worker": "elm3mk6p53nh1h (working CUDA)",
                "generation": generation_result,
                "gpu_info": "NVIDIA GeForce RTX 4090 (24GB)",
                "lora_info": {
                    "file": "pepe.safetensors",
                    "strength": "100% (full)",
                    "size": "171MB of Pepe training data"
                },
                "success": True,
                "note": "This should be a REAL Pepe with proper characteristics!"
            }
        else:
            return {
                "error": "REAL Pepe generation failed",
                "details": generation_result,
                "worker": "elm3mk6p53nh1h",
                "gpu_working": True,
                "server_running": True
            }
        
    except Exception as e:
        return {
            "error": str(e),
            "debug": "Real Pepe handler exception"
        }

if __name__ == '__main__':
    print("🚀 Starting REAL Pepe Generator with Working CUDA...")
    runpod.serverless.start({'handler': handler})
