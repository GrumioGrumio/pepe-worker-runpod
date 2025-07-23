import runpod
import os
import requests
import time
import json
import subprocess
import sys
import base64

def check_volume_setup():
    """Check if volume has everything needed"""
    try:
        volume_root = "/runpod-volume"
        comfyui_path = f"{volume_root}/comfyui"
        
        checks = {
            "volume_mounted": os.path.exists(volume_root),
            "comfyui_exists": os.path.exists(f"{comfyui_path}/main.py"),
            "sd15_model": os.path.exists(f"{comfyui_path}/models/checkpoints/sd15.safetensors"),
            "pepe_lora": os.path.exists(f"{comfyui_path}/models/loras/pepe.safetensors"),
            "output_dir": os.path.exists(f"{comfyui_path}/output")
        }
        
        if checks["sd15_model"]:
            checks["sd15_size"] = os.path.getsize(f"{comfyui_path}/models/checkpoints/sd15.safetensors")
        
        if checks["pepe_lora"]:
            checks["lora_size"] = os.path.getsize(f"{comfyui_path}/models/loras/pepe.safetensors")
        
        checks["ready"] = all(checks[key] for key in ["volume_mounted", "comfyui_exists", "sd15_model", "pepe_lora"])
        
        return checks
        
    except Exception as e:
        return {"error": str(e), "ready": False}

def start_comfyui_server():
    """Start ComfyUI server from volume"""
    try:
        print("🚀 Starting ComfyUI server from volume...")
        
        comfyui_path = "/runpod-volume/comfyui"
        
        # Kill any existing processes
        try:
            subprocess.run(["pkill", "-f", "main.py"], check=False, timeout=5)
            time.sleep(3)
        except:
            pass
        
        # Check if server is already running
        try:
            response = requests.get("http://localhost:8188/", timeout=3)
            if response.status_code == 200:
                print("✅ ComfyUI server already running!")
                return True, "Server already running"
        except:
            pass
        
        # Set up environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = comfyui_path
        
        # Start server with detailed logging
        cmd = [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", "8188", "--verbose"]
        
        print(f"🔧 Starting: {' '.join(cmd)}")
        print(f"📁 Working dir: {comfyui_path}")
        
        process = subprocess.Popen(
            cmd,
            cwd=comfyui_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        # Wait for startup with progress updates
        startup_logs = []
        
        for i in range(120):  # 2 minutes timeout
            try:
                # Check if server responds
                response = requests.get("http://localhost:8188/", timeout=2)
                if response.status_code == 200:
                    print("✅ ComfyUI server started successfully!")
                    return True, f"Server running (PID: {process.pid})"
                    
            except requests.exceptions.RequestException:
                pass  # Expected while starting
            
            # Check if process crashed
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                
                return False, {
                    "error": "Process crashed during startup",
                    "stdout": stdout.decode()[-500:] if stdout else "No stdout",
                    "stderr": stderr.decode()[-500:] if stderr else "No stderr",
                    "return_code": process.returncode
                }
            
            # Progress updates
            if i % 20 == 0:
                print(f"⏳ Server starting... ({i}/120 seconds)")
            
            time.sleep(1)
        
        # Timeout reached
        if process.poll() is None:
            # Process still running but not responding
            try:
                stdout, stderr = process.communicate(timeout=5)
                return False, {
                    "error": "Server timeout - running but not responding",
                    "stdout": stdout.decode()[-500:] if stdout else "No stdout",
                    "stderr": stderr.decode()[-500:] if stderr else "No stderr"
                }
            except:
                return False, "Server timeout - process hanging"
        else:
            # Process died
            stdout, stderr = process.communicate()
            return False, {
                "error": "Process died during startup",
                "stdout": stdout.decode()[-500:] if stdout else "No stdout", 
                "stderr": stderr.decode()[-500:] if stderr else "No stderr"
            }
        
    except Exception as e:
        return False, f"Server start exception: {str(e)}"

def test_server_endpoints():
    """Test that all server endpoints work"""
    try:
        tests = {}
        
        # Test main page
        try:
            response = requests.get("http://localhost:8188/", timeout=5)
            tests["main_page"] = f"✅ {response.status_code}"
        except Exception as e:
            tests["main_page"] = f"❌ {str(e)[:100]}"
        
        # Test queue endpoint
        try:
            response = requests.get("http://localhost:8188/queue", timeout=5)
            tests["queue"] = f"✅ {response.status_code}"
        except Exception as e:
            tests["queue"] = f"❌ {str(e)[:100]}"
        
        # Test history endpoint
        try:
            response = requests.get("http://localhost:8188/history", timeout=5)
            tests["history"] = f"✅ {response.status_code}"
        except Exception as e:
            tests["history"] = f"❌ {str(e)[:100]}"
        
        # Test model loading (check system stats)
        try:
            response = requests.get("http://localhost:8188/system_stats", timeout=5)
            tests["system_stats"] = f"✅ {response.status_code}"
        except Exception as e:
            tests["system_stats"] = f"❌ {str(e)[:100]}"
        
        all_working = all(test.startswith("✅") for test in tests.values())
        
        return {"tests": tests, "all_working": all_working}
        
    except Exception as e:
        return {"error": str(e), "all_working": False}

def generate_test_pepe(prompt="wearing a crown"):
    """Generate test Pepe with LoRA"""
    try:
        print(f"🐸 Generating test Pepe: {prompt}")
        
        # Test workflow with LoRA
        workflow = {
            "1": {
                "inputs": {"ckpt_name": "sd15.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "lora_name": "pepe.safetensors",
                    "strength_model": 0.8,  # Slightly reduced for stability
                    "strength_clip": 0.8,
                    "model": ["1", 0],
                    "clip": ["1", 1]
                },
                "class_type": "LoraLoader"
            },
            "3": {
                "inputs": {
                    "text": f"pepe the frog, {prompt}, meme style, simple cartoon, green frog",
                    "clip": ["2", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "text": "blurry, low quality, distorted, realistic, photorealistic",
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
                    "steps": 20,
                    "cfg": 7.5,
                    "sampler_name": "euler",
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
                    "filename_prefix": f"AUTO_PEPE_{int(time.time())}",
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
            return {"error": f"Submit failed: {response.status_code} - {response.text}"}
        
        prompt_data = response.json()
        prompt_id = prompt_data.get("prompt_id")
        
        if not prompt_id:
            return {"error": f"No prompt ID in response: {prompt_data}"}
        
        print(f"✅ Pepe generation queued: {prompt_id}")
        
        # Monitor generation with detailed progress
        for i in range(150):  # 2.5 minutes
            try:
                # Check queue status
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
                        
                        # Check for generated images
                        output_dir = "/runpod-volume/comfyui/output"
                        image_files = []
                        
                        if os.path.exists(output_dir):
                            for root, dirs, files in os.walk(output_dir):
                                for file in files:
                                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                        file_path = os.path.join(root, file)
                                        if time.time() - os.path.getctime(file_path) < 300:  # Last 5 minutes
                                            image_files.append(file_path)
                        
                        if image_files:
                            # Get most recent image
                            latest_image = max(image_files, key=os.path.getctime)
                            
                            # Convert to base64
                            with open(latest_image, 'rb') as img_file:
                                img_data = img_file.read()
                                img_base64 = base64.b64encode(img_data).decode('utf-8')
                            
                            return {
                                "success": True,
                                "prompt_id": prompt_id,
                                "image_path": latest_image,
                                "image_base64": img_base64,
                                "image_size": len(img_data),
                                "generation_time": f"~{i} seconds"
                            }
                        else:
                            return {"error": "Generation completed but no images found"}
                
                if i % 15 == 0:
                    print(f"⏳ Generating Pepe... ({i}/150s)")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️ Monitor error: {e}")
                time.sleep(1)
        
        return {"error": "Generation timeout after 2.5 minutes"}
        
    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}

def handler(event):
    """Auto-start handler - always gets ComfyUI running from volume"""
    print("🗄️ AUTO-START VOLUME HANDLER v20.0! 🚀")
    print("🎯 Always starts ComfyUI from volume on any worker")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'wearing a golden crown')
        
        # Step 1: Check volume setup
        print("🔄 Step 1: Checking volume setup...")
        volume_check = check_volume_setup()
        
        if not volume_check.get("ready"):
            return {
                "error": "Volume not properly set up",
                "volume_check": volume_check,
                "note": "Run the volume setup handler first"
            }
        
        # Step 2: Start ComfyUI server
        print("🔄 Step 2: Starting ComfyUI server...")
        server_success, server_result = start_comfyui_server()
        
        if not server_success:
            return {
                "error": "Failed to start ComfyUI server",
                "server_result": server_result,
                "volume_check": volume_check
            }
        
        # Step 3: Test server endpoints
        print("🔄 Step 3: Testing server endpoints...")
        endpoint_tests = test_server_endpoints()
        
        if not endpoint_tests.get("all_working"):
            return {
                "error": "Server started but endpoints not working",
                "endpoint_tests": endpoint_tests,
                "server_result": server_result
            }
        
        # Step 4: Generate test Pepe
        print(f"🔄 Step 4: Generating Pepe: {prompt}")
        generation_result = generate_test_pepe(prompt)
        
        if generation_result.get("success"):
            return {
                "message": "🗄️ AUTO-START SUCCESS! PEPE GENERATED! 🎉",
                "prompt": prompt,
                "volume_check": volume_check,
                "server_result": server_result,
                "endpoint_tests": endpoint_tests,
                "generation": generation_result,
                "success": True,
                "note": "Volume working perfectly! Future requests will be fast!"
            }
        else:
            return {
                "error": "Server working but generation failed",
                "generation_error": generation_result,
                "volume_check": volume_check,
                "server_result": server_result,
                "endpoint_tests": endpoint_tests
            }
        
    except Exception as e:
        return {
            "error": str(e),
            "debug": "Auto-start handler exception"
        }

if __name__ == '__main__':
    print("🚀 Starting Auto-Start Volume Handler...")
    runpod.serverless.start({'handler': handler})
