import runpod
import os
import requests
import time
import json
import base64
import glob

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

def generate_pepe_with_volume(prompt="wearing a crown"):
    """Generate Pepe using models from persistent volume"""
    try:
        print(f"🐸 Generating Pepe from volume: {prompt}")
        
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
        
        # Check if we have Pepe LoRA
        has_pepe = any("pepe" in lora.lower() for lora in available_loras)
        
        if has_pepe:
            # Full Pepe workflow with LoRA
            pepe_lora = next(lora for lora in available_loras if "pepe" in lora.lower())
            
            workflow = {
                "1": {
                    "inputs": {"ckpt_name": checkpoint_name},
                    "class_type": "CheckpointLoaderSimple"
                },
                "2": {
                    "inputs": {
                        "lora_name": pepe_lora,
                        "strength_model": 0.8,
                        "strength_clip": 0.8,
                        "model": ["1", 0],
                        "clip": ["1", 1]
                    },
                    "class_type": "LoraLoader"
                },
                "3": {
                    "inputs": {
                        "text": f"pepe the frog, {prompt}, meme style, cartoon, green frog",
                        "clip": ["2", 1]
                    },
                    "class_type": "CLIPTextEncode"
                },
                "4": {
                    "inputs": {
                        "text": "blurry, low quality, realistic, photorealistic",
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
                        "filename_prefix": f"VOLUME_PEPE_{int(time.time())}",
                        "images": ["7", 0]
                    },
                    "class_type": "SaveImage"
                }
            }
        else:
            # Simple frog without LoRA
            workflow = {
                "1": {
                    "inputs": {"ckpt_name": checkpoint_name},
                    "class_type": "CheckpointLoaderSimple"
                },
                "2": {
                    "inputs": {
                        "text": f"cute green frog, {prompt}, cartoon style",
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
                        "filename_prefix": f"VOLUME_FROG_{int(time.time())}",
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
        
        return {
            "success": True,
            "prompt_id": prompt_id,
            "checkpoint_used": checkpoint_name,
            "lora_used": pepe_lora if has_pepe else None,
            "available_checkpoints": available_checkpoints,
            "available_loras": available_loras
        }
        
    except Exception as e:
        return {"error": f"Generation failed: {str(e)}"}

def handler(event):
    """Persistent volume handler"""
    print("🗄️ PERSISTENT VOLUME HANDLER v1.0! 📦")
    
    try:
        input_data = event.get('input', {})
        action = input_data.get('action', 'setup_and_generate')
        prompt = input_data.get('prompt', 'wearing a golden crown')
        
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
            # Just generate with existing models
            generation_result = generate_pepe_with_volume(prompt)
            
            return {
                "message": "🐸 Generation completed",
                "prompt": prompt,
                "generation": generation_result
            }
        
        else:  # setup_and_generate
            # Complete flow
            volume_setup = setup_persistent_volume()
            if not volume_setup.get("success"):
                return {"error": "Volume setup failed", "details": volume_setup}
            
            model_check = check_volume_models()
            download_result = download_models_to_volume()
            generation_result = generate_pepe_with_volume(prompt)
            
            return {
                "message": "🎉 COMPLETE VOLUME SETUP AND GENERATION!",
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
    print("🚀 Starting Persistent Volume Handler...")
    runpod.serverless.start({'handler': handler})
