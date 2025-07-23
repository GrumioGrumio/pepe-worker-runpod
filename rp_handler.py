import runpod
import os
import requests
import time
import json
import base64

def check_comfyui_status():
    """Check ComfyUI server status and queue"""
    try:
        status = {}
        
        # Check server
        try:
            response = requests.get("http://localhost:8188/", timeout=5)
            status["server_running"] = response.status_code == 200
        except:
            status["server_running"] = False
        
        # Check queue
        try:
            response = requests.get("http://localhost:8188/queue", timeout=5)
            if response.status_code == 200:
                queue_data = response.json()
                status["queue_running"] = len(queue_data.get("queue_running", []))
                status["queue_pending"] = len(queue_data.get("queue_pending", []))
            else:
                status["queue_error"] = f"Queue status: {response.status_code}"
        except Exception as e:
            status["queue_error"] = str(e)
        
        # Check history
        try:
            response = requests.get("http://localhost:8188/history", timeout=5)
            if response.status_code == 200:
                history_data = response.json()
                status["history_count"] = len(history_data)
                
                # Get recent history
                recent_items = []
                for item_id, item_data in list(history_data.items())[-3:]:  # Last 3 items
                    recent_items.append({
                        "id": item_id,
                        "status": item_data.get("status", {}),
                        "outputs": list(item_data.get("outputs", {}).keys())
                    })
                status["recent_history"] = recent_items
            else:
                status["history_error"] = f"History status: {response.status_code}"
        except Exception as e:
            status["history_error"] = str(e)
        
        return status
        
    except Exception as e:
        return {"error": str(e)}

def check_output_directory():
    """Check output directory for images"""
    try:
        output_info = {}
        
        output_dir = "/runpod-volume/comfyui/output"
        output_info["output_dir"] = output_dir
        output_info["output_exists"] = os.path.exists(output_dir)
        
        if os.path.exists(output_dir):
            # List all files
            all_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_info = {
                        "name": file,
                        "path": file_path,
                        "size": os.path.getsize(file_path),
                        "created": time.ctime(os.path.getctime(file_path)),
                        "age_minutes": (time.time() - os.path.getctime(file_path)) / 60
                    }
                    all_files.append(file_info)
            
            # Sort by creation time (newest first)
            all_files.sort(key=lambda x: os.path.getctime(x["path"]), reverse=True)
            
            output_info["total_files"] = len(all_files)
            output_info["recent_files"] = all_files[:10]  # Last 10 files
            
            # Check for images specifically
            image_files = [f for f in all_files if f["name"].lower().endswith(('.png', '.jpg', '.jpeg'))]
            output_info["image_count"] = len(image_files)
            output_info["recent_images"] = image_files[:5]  # Last 5 images
            
        else:
            output_info["error"] = "Output directory doesn't exist"
        
        return output_info
        
    except Exception as e:
        return {"error": str(e)}

def test_simple_generation():
    """Test a very simple generation workflow"""
    try:
        print("🧪 Testing simple generation workflow...")
        
        # Very basic workflow
        simple_workflow = {
            "1": {
                "inputs": {"ckpt_name": "sd15.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": "a simple green frog",
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
                    "seed": 12345,
                    "steps": 10,
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
                    "filename_prefix": f"simple_test_{int(time.time())}",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }
        
        # Submit simple generation
        response = requests.post(
            "http://localhost:8188/prompt",
            json={"prompt": simple_workflow},
            timeout=30
        )
        
        if response.status_code != 200:
            return {"error": f"Simple generation failed: {response.status_code}"}
        
        prompt_data = response.json()
        prompt_id = prompt_data.get("prompt_id")
        
        print(f"✅ Simple generation queued: {prompt_id}")
        
        # Wait for completion with more detailed monitoring
        for i in range(90):
            try:
                # Check queue
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
                        print("🎉 Simple generation completed!")
                        
                        # Check history for this prompt
                        history_response = requests.get("http://localhost:8188/history", timeout=5)
                        if history_response.status_code == 200:
                            history_data = history_response.json()
                            if prompt_id in history_data:
                                prompt_history = history_data[prompt_id]
                                return {
                                    "success": True,
                                    "prompt_id": prompt_id,
                                    "history": prompt_history,
                                    "outputs": prompt_history.get("outputs", {})
                                }
                        break
                
                if i % 10 == 0:
                    print(f"⏳ Simple generation... ({i}/90s)")
                
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️ Queue check error: {e}")
                time.sleep(1)
        
        return {"success": True, "prompt_id": prompt_id, "note": "Completed but couldn't get history"}
        
    except Exception as e:
        return {"error": f"Simple generation failed: {str(e)}"}

def get_latest_image():
    """Get the most recent image from output"""
    try:
        output_dir = "/runpod-volume/comfyui/output"
        
        if not os.path.exists(output_dir):
            return None, "Output directory doesn't exist"
        
        # Find all images
        image_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    image_files.append(file_path)
        
        if not image_files:
            return None, "No images found"
        
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
        return None, f"Error: {str(e)}"

def handler(event):
    """Debug generation and test simple workflow"""
    print("🔍 GENERATION DEBUG v19.0! 🕵️")
    print("🎯 ComfyUI is installed and running - let's debug generation")
    
    try:
        input_data = event.get('input', {})
        action = input_data.get('action', 'debug_and_test')
        
        # Step 1: Check ComfyUI status
        print("🔄 Step 1: Checking ComfyUI status...")
        comfyui_status = check_comfyui_status()
        
        # Step 2: Check output directory
        print("🔄 Step 2: Checking output directory...")
        output_info = check_output_directory()
        
        # Step 3: Test simple generation
        print("🔄 Step 3: Testing simple generation...")
        simple_test = test_simple_generation()
        
        # Step 4: Check for any new images
        print("🔄 Step 4: Looking for generated images...")
        latest_image, image_msg = get_latest_image()
        
        return {
            "message": "🔍 Generation Debug Complete",
            "comfyui_status": comfyui_status,
            "output_directory": output_info,
            "simple_generation_test": simple_test,
            "latest_image": {
                "found": latest_image is not None,
                "details": latest_image,
                "message": image_msg
            },
            "diagnosis": {
                "server_working": comfyui_status.get("server_running", False),
                "output_dir_exists": output_info.get("output_exists", False),
                "images_generated": output_info.get("image_count", 0),
                "simple_test_success": simple_test.get("success", False)
            },
            "recommendations": [
                "Check ComfyUI logs for generation errors",
                "Verify LoRA file is loading correctly", 
                "Test with simpler workflow first",
                "Check if images are being saved to different location"
            ]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "debug": "Generation debug handler exception"
        }

if __name__ == '__main__':
    print("🚀 Starting Generation Debug Handler...")
    runpod.serverless.start({'handler': handler})
