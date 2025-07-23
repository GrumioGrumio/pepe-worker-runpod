import runpod
import os
import requests
import time
import json
import base64
import glob

def download_models():
    """Download required models if they don't exist"""
    try:
        print("📦 Checking and downloading models...")
        
        models_dir = "/app/comfyui/models"
        checkpoints_dir = f"{models_dir}/checkpoints"
        loras_dir = f"{models_dir}/loras"
        
        # Create directories
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(loras_dir, exist_ok=True)
        
        results = {}
        
        # Download SD 1.5 model
        model_path = f"{checkpoints_dir}/sd15.safetensors"
        if os.path.exists(model_path) and os.path.getsize(model_path) > 3000000000:  # 3GB+
            results["sd15_model"] = f"✅ Already exists ({os.path.getsize(model_path)} bytes)"
        else:
            print("📥 Downloading SD 1.5 model (4GB)...")
            try:
                import requests
                response = requests.get(
                    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
                    stream=True, timeout=1800  # 30 minutes
                )
                response.raise_for_status()
                
                downloaded = 0
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if downloaded % (500*1024*1024) == 0:  # Every 500MB
                                print(f"📊 Downloaded: {downloaded // (1024*1024)}MB...")
                
                results["sd15_model"] = f"✅ Downloaded ({os.path.getsize(model_path)} bytes)"
                
            except Exception as e:
                results["sd15_model"] = f"❌ Download failed: {str(e)}"
        
        # Download Pepe LoRA
        lora_path = f"{loras_dir}/pepe.safetensors"
        if os.path.exists(lora_path) and os.path.getsize(lora_path) > 100000000:  # 100MB+
            results["pepe_lora"] = f"✅ Already exists ({os.path.getsize(lora_path)} bytes)"
        else:
            print("📥 Downloading Pepe LoRA (171MB)...")
            try:
                response = requests.get(
                    "https://huggingface.co/openfree/pepe/resolve/main/pepe.safetensors",
                    stream=True, timeout=600  # 10 minutes
                )
                response.raise_for_status()
                
                with open(lora_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                results["pepe_lora"] = f"✅ Downloaded ({os.path.getsize(lora_path)} bytes)"
                
            except Exception as e:
                results["pepe_lora"] = f"❌ Download failed: {str(e)}"
        
        return results
        
    except Exception as e:
        return {"error": f"Model download failed: {str(e)}"}

def test_comfyui_api():
    """Test ComfyUI API endpoints"""
    try:
        tests = {}
        
        # Test main endpoint
        try:
            response = requests.get("http://127.0.0.1:8188/", timeout=10)
            tests["main"] = {"status": response.status_code, "ok": response.status_code == 200}
        except Exception as e:
            tests["main"] = {"status": "error", "error": str(e), "ok": False}
        
        # Test queue
        try:
            response = requests.get("http://127.0.0.1:8188/queue", timeout=10)
            tests["queue"] = {"status": response.status_code, "ok": response.status_code == 200}
        except Exception as e:
            tests["queue"] = {"status": "error", "error": str(e), "ok": False}
        
        # Test object_info (available models/nodes)
        try:
            response = requests.get("http://127.0.0.1:8188/object_info", timeout=10)
            tests["object_info"] = {"status": response.status_code, "ok": response.status_code == 200}
            if response.status_code == 200:
                info = response.json()
                tests["object_info"]["node_count"] = len(info)
                
                # Check for required nodes
                required_nodes = ["CheckpointLoaderSimple", "LoraLoader", "KSampler", "SaveImage"]
                missing_nodes = [node for node in required_nodes if node not in info]
                tests["object_info"]["missing_nodes"] = missing_nodes
                tests["object_info"]["has_required_nodes"] = len(missing_nodes) == 0
                
        except Exception as e:
            tests["object_info"] = {"status": "error", "error": str(e), "ok": False}
        
        all_working = all(test.get("ok", False) for test in tests.values())
        
        return {"tests": tests, "all_working": all_working}
        
    except Exception as e:
        return {"error": str(e), "all_working": False}

def generate_simple_image(prompt="a green frog"):
    """Generate a simple image to test basic functionality"""
    try:
        print(f"🖼️ Generating simple image: {prompt}")
        
        # Simple workflow without LoRA
        workflow = {
            "1": {
                "inputs": {"ckpt_name": "sd15.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": prompt,
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": "blurry, low quality",
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
                    "steps": 20,
                    "cfg": 7.5,
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
                    "filename_prefix": f"SIMPLE_{int(time.time())}",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }
        
        # Submit workflow
        response = requests.post(
            "http://127.0.0.1:8188/prompt",
            json={"prompt": workflow},
            timeout=30
        )
        
        if response.status_code != 200:
            return {"error": f"Submit failed: {response.status_code} - {response.text}"}
        
        prompt_data = response.json()
        prompt_id = prompt_data.get("prompt_id")
        
        if not prompt_id:
            return {"error": f"No prompt ID: {prompt_data}"}
        
        print(f"✅ Simple generation queued: {prompt_id}")
        
        # Wait for completion
        for i in range(120):  # 2 minutes
            try:
                queue_response = requests.get("http://127.0.0.1:8188/queue", timeout=5)
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
                        print("🎉 Simple generation completed!")
                        break
                
                if i % 15 == 0:
                    print(f"⏳ Generating... ({i}/120s)")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️ Monitor error: {e}")
                time.sleep(1)
        
        # Find generated image
        output_dir = "/app/comfyui/output"
        recent_images = []
        
        if os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        if time.time() - os.path.getctime(file_path) < 300:  # Last 5 minutes
                            recent_images.append(file_path)
        
        if recent_images:
            # Get most recent
            latest_image = max(recent_images, key=os.path.getctime)
            
            # Convert to base64
            with open(latest_image, 'rb') as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            return {
                "success": True,
                "prompt_id": prompt_id,
                "image_path": latest_image,
                "image_base64": img_base64,
                "image_size": len(img_data)
            }
        else:
            return {"error": "Generation completed but no images found"}
        
    except Exception as e:
        return {"error": f"Simple generation failed: {str(e)}"}

def handler(event):
    """Main RunPod handler"""
    print("🎭 SIMPLE PEPE HANDLER v1.0! 🐸")
    
    try:
        input_data = event.get('input', {})
        action = input_data.get('action', 'generate')
        prompt = input_data.get('prompt', 'a cute green frog cartoon')
        
        if action == 'setup':
            # Download models
            model_results = download_models()
            
            # Test API
            api_tests = test_comfyui_api()
            
            return {
                "message": "Setup completed",
                "models": model_results,
                "api_tests": api_tests
            }
        
        elif action == 'test':
            # Test API and generate simple image
            api_tests = test_comfyui_api()
            
            if not api_tests.get("all_working"):
                return {
                    "error": "ComfyUI API not working",
                    "api_tests": api_tests
                }
            
            generation_result = generate_simple_image(prompt)
            
            return {
                "message": "Test completed",
                "api_tests": api_tests,
                "generation": generation_result
            }
        
        else:  # default generate action
            # Quick API test
            api_tests = test_comfyui_api()
            
            if not api_tests.get("all_working"):
                return {
                    "error": "ComfyUI API not working",
                    "api_tests": api_tests
                }
            
            # Generate image
            generation_result = generate_simple_image(prompt)
            
            if generation_result.get("success"):
                return {
                    "message": "🎉 IMAGE GENERATED! 🖼️",
                    "prompt": prompt,
                    "generation": generation_result,
                    "success": True
                }
            else:
                return {
                    "error": "Generation failed",
                    "details": generation_result,
                    "api_tests": api_tests
                }
        
    except Exception as e:
        return {
            "error": f"Handler exception: {str(e)}",
            "traceback": str(e)
        }

if __name__ == '__main__':
    print("🚀 Starting Simple Pepe Handler...")
    runpod.serverless.start({'handler': handler})
