import runpod
import os
import requests
import time
import json
import base64
from PIL import Image
import io

def check_files_detailed():
    """Check exactly what files exist"""
    try:
        file_check = {}
        
        # Check ComfyUI structure
        comfyui_path = "/workspace/comfyui"
        file_check["comfyui_exists"] = os.path.exists(comfyui_path)
        
        if os.path.exists(comfyui_path):
            file_check["comfyui_contents"] = os.listdir(comfyui_path)
        
        # Check models directory structure
        models_path = "/workspace/comfyui/models"
        if os.path.exists(models_path):
            file_check["models_structure"] = {}
            for subdir in ["checkpoints", "loras", "vae", "clip"]:
                subdir_path = os.path.join(models_path, subdir)
                if os.path.exists(subdir_path):
                    files = os.listdir(subdir_path)
                    file_check["models_structure"][subdir] = files
                else:
                    file_check["models_structure"][subdir] = "MISSING"
        
        # Check specific files
        important_files = {
            "main.py": "/workspace/comfyui/main.py",
            "sd15_model": "/workspace/comfyui/models/checkpoints/sd15.safetensors",
            "pepe_lora": "/workspace/comfyui/models/loras/pepe.safetensors"
        }
        
        for name, path in important_files.items():
            if os.path.exists(path):
                size = os.path.getsize(path)
                file_check[name] = f"✅ {size} bytes"
            else:
                file_check[name] = "❌ Missing"
        
        # Check output directory
        output_path = "/workspace/comfyui/output"
        if os.path.exists(output_path):
            output_files = []
            for root, dirs, files in os.walk(output_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root, file)
                        created = time.ctime(os.path.getctime(file_path))
                        size = os.path.getsize(file_path)
                        output_files.append({
                            "name": file,
                            "path": file_path,
                            "created": created,
                            "size": size
                        })
            
            file_check["output_files"] = output_files[:10]  # Last 10 files
        else:
            file_check["output_files"] = "Output directory missing"
        
        return file_check
        
    except Exception as e:
        return {"error": str(e)}

def test_basic_generation():
    """Test basic SD generation without LoRA first"""
    try:
        print("🧪 Testing basic SD generation (no LoRA)...")
        
        # Simple workflow without LoRA
        basic_workflow = {
            "1": {
                "inputs": {"ckpt_name": "sd15.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": "a simple green frog, cartoon style, clean background",
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": "blurry, low quality, distorted",
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
                    "seed": 12345,
                    "steps": 20,
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
                    "filename_prefix": f"basic_test_{int(time.time())}",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }
        
        # Submit basic generation
        response = requests.post(
            "http://localhost:8188/prompt",
            json={"prompt": basic_workflow},
            timeout=30
        )
        
        if response.status_code != 200:
            return {"error": f"Basic generation failed: {response.status_code}"}
        
        prompt_data = response.json()
        prompt_id = prompt_data.get("prompt_id")
        
        print(f"✅ Basic generation queued: {prompt_id}")
        
        # Wait for completion
        for i in range(60):
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
                        print("🎉 Basic generation completed!")
                        break
                
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️ Queue check error: {e}")
                time.sleep(1)
        
        return {"success": True, "prompt_id": prompt_id, "method": "basic_sd"}
        
    except Exception as e:
        return {"error": f"Basic generation failed: {str(e)}"}

def test_lora_generation():
    """Test LoRA generation after basic works"""
    try:
        print("🐸 Testing LoRA generation...")
        
        # LoRA workflow
        lora_workflow = {
            "1": {
                "inputs": {"ckpt_name": "sd15.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "lora_name": "pepe.safetensors",
                    "strength_model": 0.8,  # Reduced strength for testing
                    "strength_clip": 0.8,
                    "model": ["1", 0],
                    "clip": ["1", 1]
                },
                "class_type": "LoraLoader"
            },
            "3": {
                "inputs": {
                    "text": "pepe the frog, simple green frog character, meme style",
                    "clip": ["2", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "text": "blurry, low quality, distorted, realistic",
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
                    "seed": 54321,
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
                    "filename_prefix": f"lora_test_{int(time.time())}",
                    "images": ["7", 0]
                },
                "class_type": "SaveImage"
            }
        }
        
        # Submit LoRA generation
        response = requests.post(
            "http://localhost:8188/prompt",
            json={"prompt": lora_workflow},
            timeout=30
        )
        
        if response.status_code != 200:
            return {"error": f"LoRA generation failed: {response.status_code}"}
        
        prompt_data = response.json()
        prompt_id = prompt_data.get("prompt_id")
        
        print(f"✅ LoRA generation queued: {prompt_id}")
        
        # Wait for completion
        for i in range(90):
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
                        print("🎉 LoRA generation completed!")
                        break
                
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️ Queue check error: {e}")
                time.sleep(1)
        
        return {"success": True, "prompt_id": prompt_id, "method": "lora"}
        
    except Exception as e:
        return {"error": f"LoRA generation failed: {str(e)}"}

def get_latest_image():
    """Get the most recent generated image"""
    try:
        output_dir = "/workspace/comfyui/output"
        if not os.path.exists(output_dir):
            return None, "Output directory doesn't exist"
        
        image_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    image_files.append(file_path)
        
        if not image_files:
            return None, "No images found in output directory"
        
        # Get most recent
        latest_image = max(image_files, key=os.path.getctime)
        
        # Convert to base64
        with open(latest_image, 'rb') as img_file:
            img_data = img_file.read()
            img_base64 = base64.b64encode(img_data).decode('utf-8')
        
        return {
            "path": latest_image,
            "base64": img_base64,
            "size": len(img_data),
            "created": time.ctime(os.path.getctime(latest_image))
        }, "Success"
        
    except Exception as e:
        return None, f"Error getting image: {str(e)}"

def handler(event):
    """Debug handler to diagnose and test step by step"""
    print("🔍 DEBUG PEPE HANDLER v13.0 - STEP BY STEP! 🕵️")
    
    try:
        input_data = event.get('input', {})
        action = input_data.get('action', 'debug')
        
        # Step 1: Check all files
        print("📁 Step 1: Checking files...")
        file_check = check_files_detailed()
        
        # Step 2: Test basic generation
        print("🧪 Step 2: Testing basic generation...")
        basic_test = test_basic_generation()
        
        # Step 3: Get any existing images
        print("🖼️ Step 3: Checking for generated images...")
        latest_image, image_msg = get_latest_image()
        
        # Step 4: Test LoRA if basic worked
        lora_test = {"skipped": "Basic generation didn't work"}
        if basic_test.get("success"):
            print("🐸 Step 4: Testing LoRA generation...")
            lora_test = test_lora_generation()
            
            # Get image after LoRA test
            if lora_test.get("success"):
                latest_image, image_msg = get_latest_image()
        
        return {
            "message": "🔍 Debug Analysis Complete",
            "step1_files": file_check,
            "step2_basic_generation": basic_test,
            "step3_image_check": {
                "latest_image": latest_image,
                "message": image_msg
            },
            "step4_lora_generation": lora_test,
            "diagnosis": {
                "comfyui_running": True,  # We know it's running
                "files_present": file_check.get("pepe_lora", "❌").startswith("✅"),
                "basic_generation": basic_test.get("success", False),
                "lora_generation": lora_test.get("success", False)
            },
            "next_steps": [
                "✅ Server running",
                "🔍 Check file analysis above",
                "🧪 Check basic generation results", 
                "🐸 Check LoRA generation results",
                "🖼️ Check if any images were created"
            ]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "debug": "Debug handler exception"
        }

if __name__ == '__main__':
    print("🚀 Starting Debug Pepe Handler...")
    runpod.serverless.start({'handler': handler})
