import runpod
import os
import requests
import time
import json
import base64
import glob
from PIL import Image

def setup_persistent_volume():
    """Set up persistent volume for models"""
    try:
        print("📦 Setting up persistent volume...")
        
        volume_path = "/runpod-volume"
        comfyui_models_path = "/app/comfyui/models"
        
        # Check if volume is mounted
        if not os.path.exists(volume_path):
            return {
                "error": "Volume not mounted",
                "note": "Make sure to attach a Network Volume to your pod at /runpod-volume"
            }
        
        # Create volume structure
        volume_models_path = f"{volume_path}/models"
        os.makedirs(f"{volume_models_path}/checkpoints", exist_ok=True)
        os.makedirs(f"{volume_models_path}/loras", exist_ok=True)
        os.makedirs(f"{volume_models_path}/vae", exist_ok=True)
        
        # Create symlink from ComfyUI models to persistent volume
        if os.path.exists(comfyui_models_path):
            if os.path.islink(comfyui_models_path):
                print("✅ Models symlink already exists")
            else:
                print("🔗 Creating models symlink...")
                # Remove default models directory
                import shutil
                shutil.rmtree(comfyui_models_path)
                # Create symlink to persistent volume
                os.symlink(volume_models_path, comfyui_models_path)
        else:
            print("🔗 Creating new models symlink...")
            os.symlink(volume_models_path, comfyui_models_path)
        
        # Verify setup
        checkpoints_dir = f"{comfyui_models_path}/checkpoints"
        loras_dir = f"{comfyui_models_path}/loras"
        
        return {
            "success": True,
            "volume_mounted": True,
            "models_symlinked": os.path.islink(comfyui_models_path),
            "checkpoints_dir": checkpoints_dir,
            "loras_dir": loras_dir,
            "checkpoints_exists": os.path.exists(checkpoints_dir),
            "loras_exists": os.path.exists(loras_dir)
        }
        
    except Exception as e:
        return {"error": f"Volume setup failed: {str(e)}"}

def check_volume_models():
    """Check what models are already in the persistent volume"""
    try:
        print("🔍 Checking volume for existing models...")
        
        models_dir = "/app/comfyui/models"
        checkpoints_dir = f"{models_dir}/checkpoints"
        loras_dir = f"{models_dir}/loras"
        
        status = {
            "models_dir_exists": os.path.exists(models_dir),
            "checkpoints_dir_exists": os.path.exists(checkpoints_dir),
            "loras_dir_exists": os.path.exists(loras_dir)
        }
        
        # Check checkpoint files
        if os.path.exists(checkpoints_dir):
            checkpoint_files = []
            for file in os.listdir(checkpoints_dir):
                if file.endswith('.safetensors') or file.endswith('.ckpt'):
                    file_path = os.path.join(checkpoints_dir, file)
                    size = os.path.getsize(file_path)
                    checkpoint_files.append({
                        "name": file,
                        "size": size,
                        "size_gb": round(size / (1024**3), 2)
                    })
            status["checkpoint_files"] = checkpoint_files
            status["has_sd15"] = any(f["size"] > 3000000000 for f in checkpoint_files)
        
        # Check LoRA files
        if os.path.exists(loras_dir):
            lora_files = []
            for file in os.listdir(loras_dir):
                if file.endswith('.safetensors'):
                    file_path = os.path.join(loras_dir, file)
                    size = os.path.getsize(file_path)
                    lora_files.append({
                        "name": file,
                        "size": size,
                        "size_mb": round(size / (1024**2), 1)
                    })
            status["lora_files"] = lora_files
            status["has_pepe"] = any("pepe" in f["name"].lower() for f in lora_files)
        
        return status
        
    except Exception as e:
        return {"error": f"Check failed: {str(e)}"}

def download_models_to_volume():
    """Download models to persistent volume"""
    try:
        print("📥 Downloading models to persistent volume...")
        
        models_dir = "/app/comfyui/models"
        checkpoints_dir = f"{models_dir}/checkpoints"
        loras_dir = f"{models_dir}/loras"
        
        results = {}
        
        # Download SD 1.5 if not exists
        sd15_path = f"{checkpoints_dir}/sd15.safetensors"
        if os.path.exists(sd15_path) and os.path.getsize(sd15_path) > 3000000000:
            results["sd15"] = f"✅ Already exists ({os.path.getsize(sd15_path)} bytes)"
        else:
            print("📥 Downloading SD 1.5...")
            try:
                url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"
                response = requests.get(url, stream=True, timeout=120)
                response.raise_for_status()
                
                downloaded = 0
                with open(sd15_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if downloaded % (500*1024*1024) == 0:
                                print(f"📊 SD1.5: {downloaded // (1024*1024)}MB...")
                
                final_size = os.path.getsize(sd15_path)
                if final_size > 3000000000:
                    results["sd15"] = f"✅ Downloaded ({final_size} bytes)"
                else:
                    results["sd15"] = f"❌ Incomplete ({final_size} bytes)"
                    
            except Exception as e:
                results["sd15"] = f"❌ Failed: {str(e)}"
        
        # Download Pepe LoRA if not exists
        pepe_path = f"{loras_dir}/pepe.safetensors"
        if os.path.exists(pepe_path) and os.path.getsize(pepe_path) > 100000000:
            results["pepe"] = f"✅ Already exists ({os.path.getsize(pepe_path)} bytes)"
        else:
            print("📥 Downloading Pepe LoRA...")
            try:
                url = "https://huggingface.co/openfree/pepe/resolve/main/pepe.safetensors"
                response = requests.get(url, stream=True, timeout=120)
                response.raise_for_status()
                
                with open(pepe_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                final_size = os.path.getsize(pepe_path)
                if final_size > 100000000:
                    results["pepe"] = f"✅ Downloaded ({final_size} bytes)"
                else:
                    results["pepe"] = f"❌ Incomplete ({final_size} bytes)"
                    
            except Exception as e:
                results["pepe"] = f"❌ Failed: {str(e)}"
        
        return results
        
    except Exception as e:
        return {"error": f"Download failed: {str(e)}"}

def wait_for_generation_and_get_image(prompt_id, timeout=60):
    """Wait for generation to complete and return the image as base64"""
    try:
        print(f"⏳ Waiting for generation {prompt_id} to complete...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check generation status
                response = requests.get(f"http://127.0.0.1:8188/history/{prompt_id}", timeout=10)
                
                if response.status_code == 200:
                    history = response.json()
                    
                    if prompt_id in history:
                        prompt_data = history[prompt_id]
                        
                        # Check if generation is complete
                        if "outputs" in prompt_data:
                            print("✅ Generation completed! Looking for output images...")
                            
                            # Find the SaveImage node output
                            for node_id, node_output in prompt_data["outputs"].items():
                                if "images" in node_output:
                                    images = node_output["images"]
                                    
                                    if images and len(images) > 0:
                                        # Get the first image
                                        image_info = images[0]
                                        filename = image_info["filename"]
                                        subfolder = image_info.get("subfolder", "")
                                        
                                        # Construct full path to the generated image
                                        if subfolder:
                                            image_path = f"/app/comfyui/output/{subfolder}/{filename}"
                                        else:
                                            image_path = f"/app/comfyui/output/{filename}"
                                        
                                        print(f"📸 Found image: {image_path}")
                                        
                                        # Check if file exists
                                        if os.path.exists(image_path):
                                            # Convert image to base64
                                            with open(image_path, "rb") as img_file:
                                                img_data = img_file.read()
                                                image_base64 = base64.b64encode(img_data).decode('utf-8')
                                                
                                            print(f"🎨 Successfully converted to base64: {len(image_base64)} chars")
                                            
                                            return {
                                                "success": True,
                                                "image_base64": image_base64,
                                                "image_path": image_path,
                                                "image_size": len(img_data),
                                                "filename": filename
                                            }
                                        else:
                                            print(f"❌ Image file not found: {image_path}")
                                            return {"error": f"Generated image file not found: {image_path}"}
                            
                            return {"error": "No images found in generation output"}
                
                # If not complete, wait a bit
                time.sleep(2)
                print(f"⏳ Still waiting... ({int(time.time() - start_time)}s)")
                
            except Exception as e:
                print(f"❌ Error checking status: {e}")
                time.sleep(2)
        
        return {"error": f"Generation timeout after {timeout}s"}
        
    except Exception as e:
        return {"error": f"Wait failed: {str(e)}"}

def generate_logo_with_volume(prompt="professional logo design"):
    """Generate logo using models from persistent volume and return base64"""
    try:
        print(f"🎨 Generating logo from volume: {prompt}")
        
        # Check ComfyUI can see models
        response = requests.get("http://127.0.0.1:8188/object_info", timeout=10)
        if response.status_code != 200:
            return {"error": "ComfyUI API not responding"}
        
        info = response.json()
        
        # Get available models
        available_checkpoints = []
        available_loras = []
        
        if "CheckpointLoaderSimple" in info:
            ckpt_info = info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
            available_checkpoints = ckpt_info
        
        if "LoraLoader" in info:
            lora_info = info["LoraLoader"]["input"]["required"]["lora_name"][0]
            available_loras = lora_info
        
        if not available_checkpoints:
            return {"error": "No checkpoints available"}
        
        # Use first checkpoint (should be our SD1.5)
        checkpoint_name = available_checkpoints[0]
        
        # Check if we have logo-related LoRAs
        logo_loras = [lora for lora in available_loras if any(word in lora.lower() for word in ['logo', 'design', 'helper'])]
        lora_to_use = logo_loras[0] if logo_loras else None
        
        # Create workflow based on available models
        if lora_to_use:
            # Use LoRA for better logo generation
            workflow = {
                "1": {
                    "inputs": {"ckpt_name": checkpoint_name},
                    "class_type": "CheckpointLoaderSimple"
                },
                "2": {
                    "inputs": {
                        "lora_name": lora_to_use,
                        "strength_model": 0.9,
                        "strength_clip": 0.9,
                        "model": ["1", 0],
                        "clip": ["1", 1]
                    },
                    "class_type": "LoraLoader"
                },
                "3": {
                    "inputs": {
                        "text": f"{prompt}, logo design, clean, professional, high quality, vector art, simple background",
                        "clip": ["2", 1]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "4": {
                    "inputs": {
                        "text": "blurry, low quality, realistic, photorealistic, complex background, messy, cluttered",
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
                        "filename_prefix": f"LOGO_{int(time.time())}",
                        "images": ["7", 0]
                    },
                    "class_type": "SaveImage"
                }
            }
        else:
            # Simple logo without LoRA
            workflow = {
                "1": {
                    "inputs": {"ckpt_name": checkpoint_name},
                    "class_type": "CheckpointLoaderSimple"
                },
                "2": {
                    "inputs": {
                        "text": f"{prompt}, logo design, clean, professional, high quality, vector art, simple background",
                        "clip": ["1", 1]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "3": {
                    "inputs": {
                        "text": "blurry, low quality, realistic, photorealistic, complex background, messy, cluttered",
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
                        "steps": 25,
                        "cfg": 8.0,
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
                        "filename_prefix": f"LOGO_{int(time.time())}",
                        "images": ["6", 0]
                    },
                    "class_type": "SaveImage"
                }
            }
        
        # Submit generation
        response = requests.post(
            "http://127.0.0.1:8188/prompt",
            json={"prompt": workflow},
            timeout=30
        )
        
        if response.status_code != 200:
            return {"error": f"Submit failed: {response.status_code}"}
        
        prompt_data = response.json()
        prompt_id = prompt_data.get("prompt_id")
        
        if not prompt_id:
            return {"error": "No prompt ID returned"}
        
        print(f"✅ Generation submitted with ID: {prompt_id}")
        
        # Wait for generation to complete and get the image
        image_result = wait_for_generation_and_get_image(prompt_id, timeout=90)
        
        if image_result.get("success"):
            return {
                "success": True,
                "prompt_id": prompt_id,
                "checkpoint_used": checkpoint_name,
                "lora_used": lora_to_use,
                "available_checkpoints": available_checkpoints,
                "available_loras": available_loras,
                "image_base64": image_result["image_base64"],
                "image_size": image_result["image_size"],
                "filename": image_result["filename"]
            }
        else:
            return {
                "error": image_result.get("error", "Unknown error"),
                "prompt_id": prompt_id,
                "checkpoint_used": checkpoint_name,
                "lora_used": lora_to_use,
                "available_checkpoints": available_checkpoints,
                "available_loras": available_loras
            }
        
    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}

def handler(event):
    """Enhanced handler with base64 image return"""
    print("🎨 LOGO GENERATOR HANDLER v2.0! 🖼️")
    
    try:
        input_data = event.get('input', {})
        action = input_data.get('action', 'generate')
        prompt = input_data.get('prompt', 'professional logo design')
        
        if action == 'setup':
            # Set up volume and download models
            volume_setup = setup_persistent_volume()
            if not volume_setup.get("success"):
                return {"error": "Volume setup failed", "details": volume_setup}
            
            model_check = check_volume_models()
            download_result = download_models_to_volume()
            
            return {
                "message": "📦 Volume setup completed",
                "volume_setup": volume_setup,
                "model_check": model_check,
                "download_result": download_result
            }
        
        elif action == 'generate':
            # Generate logo and return base64
            generation_result = generate_logo_with_volume(prompt)
            
            return {
                "message": "🐸 Generation completed",
                "prompt": prompt,
                "generation": generation_result
            }
        
        else:  # setup_and_generate
            # Complete flow with logo generation
            volume_setup = setup_persistent_volume()
            if not volume_setup.get("success"):
                return {"error": "Volume setup failed", "details": volume_setup}
            
            model_check = check_volume_models()
            download_result = download_models_to_volume()
            generation_result = generate_logo_with_volume(prompt)
            
            return {
                "message": "🎉 COMPLETE SETUP AND LOGO GENERATION!",
                "prompt": prompt,
                "volume_setup": volume_setup,
                "model_check": model_check,
                "download_result": download_result,
                "generation": generation_result,
                "success": generation_result.get("success", False)
            }
        
    except Exception as e:
        return {"error": f"Handler exception: {str(e)}"}

if __name__ == '__main__':
    print("🚀 Starting Logo Generation Handler...")
    runpod.serverless.start({'handler': handler})
