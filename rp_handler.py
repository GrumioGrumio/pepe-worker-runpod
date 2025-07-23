import runpod
import os
import requests
import time
import json
import subprocess
import sys
import base64
import glob
from pathlib import Path

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
        
        # Create output directory if it doesn't exist
        if not checks["output_dir"]:
            os.makedirs(f"{comfyui_path}/output", exist_ok=True)
            checks["output_dir"] = True
        
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

def find_latest_images(output_dir, time_threshold=300):
    """Find the most recent images in output directory"""
    try:
        image_files = []
        current_time = time.time()
        
        # Search patterns for different image extensions
        patterns = ['*.png', '*.jpg', '*.jpeg', '*.webp']
        
        for pattern in patterns:
            # Search recursively in output directory
            search_path = os.path.join(output_dir, "**", pattern)
            files = glob.glob(search_path, recursive=True)
            
            for file_path in files:
                try:
                    # Check if file was created recently
                    creation_time = os.path.getctime(file_path)
                    if current_time - creation_time < time_threshold:
                        image_files.append({
                            'path': file_path,
                            'created': creation_time,
                            'size': os.path.getsize(file_path)
                        })
                except (OSError, IOError):
                    continue
        
        # Sort by creation time (newest first)
        image_files.sort(key=lambda x: x['created'], reverse=True)
        
        return image_files
        
    except Exception as e:
        print(f"Error finding images: {e}")
        return []

def wait_for_generation_complete(prompt_id, timeout=180):
    """Wait for generation to complete and return status"""
    try:
        print(f"⏳ Waiting for generation {prompt_id} to complete...")
        
        for i in range(timeout):
            try:
                # Check queue status
                queue_response = requests.get("http://localhost:8188/queue", timeout=5)
                if queue_response.status_code != 200:
                    time.sleep(1)
                    continue
                
                queue_data = queue_response.json()
                running = queue_data.get("queue_running", [])
                pending = queue_data.get("queue_pending", [])
                
                # Check if our prompt is still in queue
                still_processing = False
                for item in running + pending:
                    if len(item) > 1 and isinstance(item[1], dict):
                        if item[1].get("prompt_id") == prompt_id:
                            still_processing = True
                            break
                
                if not still_processing:
                    print(f"✅ Generation {prompt_id} completed!")
                    return True, f"Completed in {i} seconds"
                
                # Progress updates
                if i % 15 == 0:
                    print(f"⏳ Still generating... ({i}/{timeout}s)")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️ Queue check error: {e}")
                time.sleep(2)
        
        return False, f"Timeout after {timeout} seconds"
        
    except Exception as e:
        return False, f"Wait error: {str(e)}"

def generate_test_pepe(prompt="wearing a crown"):
    """Generate test Pepe with LoRA - Enhanced version"""
    try:
        print(f"🐸 Generating test Pepe: {prompt}")
        
        # Get output directory
        output_dir = "/runpod-volume/comfyui/output"
        
        # Clear old images (optional - keeps storage clean)
        old_threshold = time.time() - 3600  # 1 hour ago
        old_images = find_latest_images(output_dir, time_threshold=3600)
        if len(old_images) > 10:  # Keep only recent images
            print(f"🧹 Cleaning {len(old_images) - 10} old images...")
            for img in old_images[10:]:
                try:
                    os.remove(img['path'])
                except:
                    pass
        
        # Record time before generation
        generation_start_time = time.time()
        
        # Enhanced workflow with better settings
        unique_filename = f"PEPE_{int(generation_start_time)}_{hash(prompt) % 10000}"
        
        workflow = {
            "1": {
                "inputs": {"ckpt_name": "sd15.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "lora_name": "pepe.safetensors",
                    "strength_model": 0.9,
                    "strength_clip": 0.9,
                    "model": ["1", 0],
                    "clip": ["1", 1]
                },
                "class_type": "LoraLoader"
            },
            "3": {
                "inputs": {
                    "text": f"pepe the frog, {prompt}, meme style, cartoon, green frog, masterpiece, high quality",
                    "clip": ["2", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "text": "blurry, low quality, distorted, realistic, photorealistic, ugly, deformed",
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
                    "seed": int(generation_start_time) % 1000000,
                    "steps": 25,  # Increased steps for better quality
                    "cfg": 8.0,   # Slightly higher CFG
                    "sampler_name": "euler_ancestral",  # Better sampler
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
                    "filename_prefix": unique_filename,
                    "images": ["7", 0]
                },
                "class_type": "SaveImage"
            }
        }
        
        # Submit generation
        print("📤 Submitting generation request...")
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
        
        # Wait for completion
        success, result = wait_for_generation_complete(prompt_id)
        
        if not success:
            return {"error": f"Generation failed: {result}"}
        
        # Find generated images
        print("🔍 Searching for generated images...")
        recent_images = find_latest_images(output_dir, time_threshold=300)
        
        if not recent_images:
            # Debug: List all files in output directory
            debug_info = {
                "output_dir_exists": os.path.exists(output_dir),
                "output_dir_contents": [],
                "all_files": []
            }
            
            try:
                if os.path.exists(output_dir):
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            full_path = os.path.join(root, file)
                            debug_info["all_files"].append({
                                "path": full_path,
                                "size": os.path.getsize(full_path),
                                "modified": os.path.getmtime(full_path)
                            })
                    
                    debug_info["output_dir_contents"] = os.listdir(output_dir)
            except Exception as e:
                debug_info["list_error"] = str(e)
            
            return {
                "error": "No recent images found after generation",
                "debug": debug_info,
                "prompt_id": prompt_id,
                "generation_time": result
            }
        
        # Get the most recent image
        latest_image = recent_images[0]
        image_path = latest_image['path']
        
        print(f"🎉 Found generated image: {image_path}")
        
        # Convert to base64
        try:
            with open(image_path, 'rb') as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            return {
                "success": True,
                "prompt_id": prompt_id,
                "image_path": image_path,
                "image_base64": img_base64,
                "image_size": len(img_data),
                "generation_result": result,
                "total_recent_images": len(recent_images),
                "filename": os.path.basename(image_path)
            }
            
        except Exception as e:
            return {
                "error": f"Failed to read generated image: {str(e)}",
                "image_path": image_path,
                "prompt_id": prompt_id
            }
        
    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}

def handler(event):
    """Auto-start handler - always gets ComfyUI running from volume"""
    print("🗄️ AUTO-START VOLUME HANDLER v21.0! 🚀")
    print("🎯 Enhanced image detection and debugging")
    
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
                "endpoint_tests": endpoint_tests,
                "debug": generation_result.get("debug", {})
            }
        
    except Exception as e:
        return {
            "error": str(e),
            "debug": "Auto-start handler exception"
        }

if __name__ == '__main__':
    print("🚀 Starting Enhanced Auto-Start Volume Handler...")
    runpod.serverless.start({'handler': handler})
