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
        
        # Get file sizes for verification
        if checks["sd15_model"]:
            checks["sd15_size"] = os.path.getsize(f"{comfyui_path}/models/checkpoints/sd15.safetensors")
            checks["sd15_size_gb"] = round(checks["sd15_size"] / (1024**3), 1)
        
        if checks["pepe_lora"]:
            checks["lora_size"] = os.path.getsize(f"{comfyui_path}/models/loras/pepe.safetensors")
            checks["lora_size_mb"] = round(checks["lora_size"] / (1024**2), 1)
        
        checks["ready"] = all(checks[key] for key in ["volume_mounted", "comfyui_exists", "sd15_model", "pepe_lora"])
        
        return checks
        
    except Exception as e:
        return {"error": str(e), "ready": False}

def start_comfyui_server():
    """Start ComfyUI server from volume with enhanced error handling"""
    try:
        print("🚀 Starting ComfyUI server from volume...")
        
        comfyui_path = "/runpod-volume/comfyui"
        
        if not os.path.exists(f"{comfyui_path}/main.py"):
            return False, "ComfyUI main.py not found in volume"
        
        # Kill any existing processes more thoroughly
        try:
            subprocess.run(["pkill", "-f", "main.py"], check=False, timeout=5)
            subprocess.run(["pkill", "-f", "comfyui"], check=False, timeout=5)
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
        
        # Set up environment properly
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = comfyui_path
        env["TORCH_CUDA_ARCH_LIST"] = "8.6"  # RTX 4000 series compatibility
        
        # Start server with optimal settings
        cmd = [
            sys.executable, "main.py", 
            "--listen", "0.0.0.0", 
            "--port", "8188", 
            "--verbose",
            "--force-fp16"  # Use half precision for better memory usage
        ]
        
        print(f"🔧 Starting: {' '.join(cmd)}")
        print(f"📁 Working dir: {comfyui_path}")
        
        # Start with better process handling
        process = subprocess.Popen(
            cmd,
            cwd=comfyui_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stdout and stderr
            start_new_session=True,
            bufsize=1,
            universal_newlines=True
        )
        
        startup_output = []
        
        # Wait for startup with better progress tracking
        for i in range(180):  # 3 minutes timeout for model loading
            # Check if server responds
            try:
                response = requests.get("http://localhost:8188/", timeout=2)
                if response.status_code == 200:
                    print("✅ ComfyUI server started successfully!")
                    return True, f"Server running (PID: {process.pid})"
            except requests.exceptions.RequestException:
                pass  # Expected while starting
            
            # Check if process crashed
            if process.poll() is not None:
                # Process ended, get output
                remaining_output = process.stdout.read() if process.stdout else ""
                startup_output.append(remaining_output)
                
                full_output = "".join(startup_output)
                
                return False, {
                    "error": "Process crashed during startup",
                    "output": full_output[-1000:],  # Last 1000 chars
                    "return_code": process.returncode
                }
            
            # Read any available output
            try:
                line = process.stdout.readline()
                if line:
                    startup_output.append(line)
                    # Look for key startup messages
                    if "Loading models" in line:
                        print("📦 Loading models...")
                    elif "Starting server" in line:
                        print("🌐 Starting web server...")
                    elif "Model loaded" in line:
                        print("✅ Model loaded!")
            except:
                pass
            
            # Progress updates
            if i % 30 == 0 and i > 0:
                print(f"⏳ Server starting... ({i}/180 seconds)")
            
            time.sleep(1)
        
        # Timeout reached
        if process.poll() is None:
            # Kill the hanging process
            try:
                process.terminate()
                time.sleep(5)
                if process.poll() is None:
                    process.kill()
            except:
                pass
            
            startup_output = "".join(startup_output)
            return False, {
                "error": "Server startup timeout after 3 minutes",
                "output": startup_output[-1000:] if startup_output else "No output captured"
            }
        
    except Exception as e:
        return False, f"Server start exception: {str(e)}"

def test_server_comprehensive():
    """Comprehensive server testing"""
    try:
        tests = {}
        
        # Test main page
        try:
            response = requests.get("http://localhost:8188/", timeout=10)
            tests["main_page"] = {"status": response.status_code, "ok": response.status_code == 200}
        except Exception as e:
            tests["main_page"] = {"status": "error", "error": str(e)[:100], "ok": False}
        
        # Test API endpoints
        endpoints = ["queue", "history", "system_stats", "object_info"]
        for endpoint in endpoints:
            try:
                response = requests.get(f"http://localhost:8188/{endpoint}", timeout=10)
                tests[endpoint] = {"status": response.status_code, "ok": response.status_code == 200}
                
                # Special handling for object_info to check loaded nodes
                if endpoint == "object_info" and response.status_code == 200:
                    try:
                        info = response.json()
                        tests[endpoint]["node_count"] = len(info) if isinstance(info, dict) else 0
                    except:
                        pass
                        
            except Exception as e:
                tests[endpoint] = {"status": "error", "error": str(e)[:100], "ok": False}
        
        all_working = all(test.get("ok", False) for test in tests.values())
        
        return {"tests": tests, "all_working": all_working}
        
    except Exception as e:
        return {"error": str(e), "all_working": False}

def test_simple_generation():
    """Test basic generation without LoRA first"""
    try:
        print("🧪 Testing simple generation (no LoRA)...")
        
        workflow = {
            "1": {
                "inputs": {"ckpt_name": "sd15.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": "a simple green frog, cartoon style, high quality",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": "blurry, bad quality, distorted",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "5": {
                "inputs": {
                    "seed": int(time.time()) % 1000000,
                    "steps": 15,  # Fewer steps for quick test
                    "cfg": 7.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                },
                "class_type": "KSampler"
            },
            "6": {
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2]
                },
                "class_type": "VAEDecode"
            },
            "7": {
                "inputs": {
                    "filename_prefix": f"SIMPLE_TEST_{int(time.time())}",
                    "images": ["6", 0]
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
            return {"success": False, "error": f"Submit failed: {response.status_code}"}
        
        prompt_data = response.json()
        prompt_id = prompt_data.get("prompt_id")
        
        if not prompt_id:
            return {"success": False, "error": "No prompt ID received"}
        
        print(f"✅ Simple test queued: {prompt_id}")
        
        # Wait for completion
        success, result = wait_for_completion(prompt_id, timeout=90)
        return {"success": success, "result": result, "prompt_id": prompt_id}
        
    except Exception as e:
        return {"success": False, "error": f"Simple test failed: {str(e)}"}

def wait_for_completion(prompt_id, timeout=180):
    """Wait for generation completion with better monitoring"""
    try:
        for i in range(timeout):
            try:
                # Check queue
                queue_response = requests.get("http://localhost:8188/queue", timeout=5)
                if queue_response.status_code == 200:
                    queue_data = queue_response.json()
                    
                    running = queue_data.get("queue_running", [])
                    pending = queue_data.get("queue_pending", [])
                    
                    still_processing = False
                    for item in running + pending:
                        if len(item) > 1 and isinstance(item[1], dict):
                            if item[1].get("prompt_id") == prompt_id:
                                still_processing = True
                                break
                    
                    if not still_processing:
                        return True, f"Completed in {i} seconds"
                
                # Progress updates
                if i % 15 == 0 and i > 0:
                    print(f"⏳ Generation progress... ({i}/{timeout}s)")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️ Monitor error: {e}")
                time.sleep(2)
        
        return False, f"Timeout after {timeout} seconds"
        
    except Exception as e:
        return False, f"Wait error: {str(e)}"

def find_generated_images(output_dir, time_threshold=600):
    """Find generated images with enhanced search"""
    try:
        current_time = time.time()
        image_files = []
        
        # Multiple image extensions
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.tiff']
        
        for ext in extensions:
            pattern = os.path.join(output_dir, "**", ext)
            files = glob.glob(pattern, recursive=True)
            
            for file_path in files:
                try:
                    # Skip placeholder files
                    if "_output_images_will_be_put_here" in file_path:
                        continue
                        
                    stat = os.stat(file_path)
                    
                    # Check if recent and has content
                    if (current_time - stat.st_ctime < time_threshold and 
                        stat.st_size > 1000):  # At least 1KB
                        
                        image_files.append({
                            'path': file_path,
                            'created': stat.st_ctime,
                            'modified': stat.st_mtime,
                            'size': stat.st_size,
                            'filename': os.path.basename(file_path)
                        })
                        
                except (OSError, IOError) as e:
                    print(f"Error checking file {file_path}: {e}")
                    continue
        
        # Sort by creation time (newest first)
        image_files.sort(key=lambda x: x['created'], reverse=True)
        
        return image_files
        
    except Exception as e:
        print(f"Error finding images: {e}")
        return []

def check_generation_history(prompt_id):
    """Check ComfyUI execution history for errors"""
    try:
        response = requests.get("http://localhost:8188/history", timeout=10)
        if response.status_code != 200:
            return {"error": f"History request failed: {response.status_code}"}
        
        history = response.json()
        
        if prompt_id in history:
            execution = history[prompt_id]
            status = execution.get("status", {})
            
            return {
                "found": True,
                "status": status.get("status_str", "unknown"),
                "completed": status.get("completed", False),
                "messages": status.get("messages", []),
                "outputs": list(execution.get("outputs", {}).keys()),
                "has_errors": len(status.get("messages", [])) > 0
            }
        else:
            return {"found": False, "available_ids": list(history.keys())[-3:]}
            
    except Exception as e:
        return {"error": f"History check failed: {str(e)}"}

def generate_pepe_with_lora(prompt="wearing a crown"):
    """Generate Pepe with LoRA using robust workflow"""
    try:
        print(f"🐸 Generating Pepe with LoRA: {prompt}")
        
        # First test simple generation
        simple_test = test_simple_generation()
        if not simple_test.get("success"):
            return {
                "error": "Basic generation test failed",
                "simple_test": simple_test,
                "recommendation": "ComfyUI server has issues with basic operations"
            }
        
        print("✅ Basic generation works, proceeding with LoRA...")
        
        output_dir = "/runpod-volume/comfyui/output"
        generation_start = time.time()
        unique_id = f"PEPE_{int(generation_start)}_{abs(hash(prompt)) % 10000}"
        
        # Enhanced LoRA workflow
        workflow = {
            "1": {
                "inputs": {"ckpt_name": "sd15.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "lora_name": "pepe.safetensors",
                    "strength_model": 0.7,  # Reduced for stability
                    "strength_clip": 0.7,
                    "model": ["1", 0],
                    "clip": ["1", 1]
                },
                "class_type": "LoraLoader"
            },
            "3": {
                "inputs": {
                    "text": f"pepe the frog, {prompt}, meme style, cartoon, green frog, simple background, high quality",
                    "clip": ["2", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "text": "blurry, low quality, distorted, realistic, photorealistic, complex background, anime",
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
                    "seed": int(generation_start) % 1000000,
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
                    "filename_prefix": unique_id,
                    "images": ["7", 0]
                },
                "class_type": "SaveImage"
            }
        }
        
        # Submit LoRA generation
        response = requests.post(
            "http://localhost:8188/prompt",
            json={"prompt": workflow},
            timeout=30
        )
        
        if response.status_code != 200:
            return {"error": f"LoRA generation submit failed: {response.status_code} - {response.text}"}
        
        prompt_data = response.json()
        prompt_id = prompt_data.get("prompt_id")
        
        if not prompt_id:
            return {"error": f"No prompt ID from LoRA generation: {prompt_data}"}
        
        print(f"✅ LoRA generation queued: {prompt_id}")
        
        # Wait for completion
        success, result = wait_for_completion(prompt_id, timeout=150)
        
        # Check execution history
        history = check_generation_history(prompt_id)
        
        if not success:
            return {
                "error": f"LoRA generation failed: {result}",
                "history": history,
                "simple_test_passed": True
            }
        
        # Look for generated images
        images = find_generated_images(output_dir)
        
        if not images:
            return {
                "error": "LoRA generation completed but no images found",
                "history": history,
                "debug": {
                    "output_dir": output_dir,
                    "output_exists": os.path.exists(output_dir),
                    "all_files": [f for f in os.listdir(output_dir) if os.path.exists(output_dir)] if os.path.exists(output_dir) else []
                }
            }
        
        # Get the newest image
        latest_image = images[0]
        image_path = latest_image['path']
        
        print(f"🎉 Found Pepe image: {latest_image['filename']}")
        
        # Convert to base64
        try:
            with open(image_path, 'rb') as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            return {
                "success": True,
                "prompt_id": prompt_id,
                "image_path": image_path,
                "image_base64": img_base64,
                "image_size": len(img_data),
                "filename": latest_image['filename'],
                "generation_time": result,
                "total_images_found": len(images),
                "lora_strength": 0.7,
                "history": history
            }
            
        except Exception as e:
            return {
                "error": f"Failed to read generated image: {str(e)}",
                "image_path": image_path,
                "image_exists": os.path.exists(image_path)
            }
        
    except Exception as e:
        return {"error": f"Pepe generation failed: {str(e)}"}

def handler(event):
    """Ultimate auto-start handler with comprehensive error handling"""
    print("🗄️ FINAL AUTO-START HANDLER v23.0! 🚀")
    print("🎯 Complete end-to-end solution with enhanced debugging")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'wearing a golden crown like a king')
        
        results = {"handler_version": "v23.0", "timestamp": int(time.time())}
        
        # Step 1: Volume check
        print("🔄 Step 1: Checking volume setup...")
        volume_check = check_volume_setup()
        results["volume_check"] = volume_check
        
        if not volume_check.get("ready"):
            return {
                "error": "Volume not ready - missing files",
                "details": volume_check,
                "note": "Ensure volume setup handler has been run successfully"
            }
        
        print(f"✅ Volume ready: SD model {volume_check.get('sd15_size_gb', '?')}GB, LoRA {volume_check.get('lora_size_mb', '?')}MB")
        
        # Step 2: Start server
        print("🔄 Step 2: Starting ComfyUI server...")
        server_success, server_result = start_comfyui_server()
        results["server_start"] = {"success": server_success, "result": server_result}
        
        if not server_success:
            return {
                "error": "Failed to start ComfyUI server",
                "server_details": server_result,
                "volume_check": volume_check,
                "note": "Server startup failed - check logs in server_details"
            }
        
        print(f"✅ Server started: {server_result}")
        
        # Step 3: Test endpoints
        print("🔄 Step 3: Testing server endpoints...")
        endpoint_tests = test_server_comprehensive()
        results["endpoint_tests"] = endpoint_tests
        
        if not endpoint_tests.get("all_working"):
            return {
                "error": "Server started but endpoints not responding properly",
                "endpoint_details": endpoint_tests,
                "server_result": server_result,
                "note": "Server running but API endpoints have issues"
            }
        
        print("✅ All endpoints working")
        
        # Step 4: Generate Pepe with LoRA
        print(f"🔄 Step 4: Generating Pepe with LoRA: '{prompt}'")
        generation_result = generate_pepe_with_lora(prompt)
        results["generation"] = generation_result
        
        if generation_result.get("success"):
            return {
                "message": "🎉 SUCCESS! PEPE GENERATED WITH LORA! 🐸👑",
                "prompt": prompt,
                "results": results,
                "image_info": {
                    "filename": generation_result["filename"],
                    "size_bytes": generation_result["image_size"],
                    "size_kb": round(generation_result["image_size"] / 1024, 1)
                },
                "success": True,
                "note": "Auto-start working perfectly! Future requests will be fast!"
            }
        else:
            return {
                "error": "Generation failed despite server working",
                "generation_details": generation_result,
                "results": results,
                "note": "Server and endpoints work, but generation has issues"
            }
        
    except Exception as e:
        return {
            "error": f"Handler exception: {str(e)}",
            "partial_results": locals().get("results", {}),
            "traceback": str(e)
        }

if __name__ == '__main__':
    print("🚀 Starting Final Auto-Start Handler...")
    runpod.serverless.start({'handler': handler})
