import runpod
import os
import requests
import time
import json
import base64
import glob

def diagnose_models():
    """Diagnose model file issues"""
    try:
        print("🔍 Diagnosing model files...")
        
        diagnosis = {}
        
        # Check ComfyUI model directories
        base_path = "/app/comfyui"
        models_path = f"{base_path}/models"
        
        diagnosis["base_path"] = base_path
        diagnosis["models_path"] = models_path
        diagnosis["base_exists"] = os.path.exists(base_path)
        diagnosis["models_exists"] = os.path.exists(models_path)
        
        # Check checkpoints directory
        checkpoints_path = f"{models_path}/checkpoints"
        diagnosis["checkpoints_path"] = checkpoints_path
        diagnosis["checkpoints_exists"] = os.path.exists(checkpoints_path)
        
        if os.path.exists(checkpoints_path):
            checkpoint_files = os.listdir(checkpoints_path)
            diagnosis["checkpoint_files"] = []
            
            for file in checkpoint_files:
                file_path = os.path.join(checkpoints_path, file)
                diagnosis["checkpoint_files"].append({
                    "name": file,
                    "size": os.path.getsize(file_path),
                    "size_gb": round(os.path.getsize(file_path) / (1024**3), 2)
                })
        else:
            diagnosis["checkpoint_files"] = "Directory does not exist"
        
        # Check LoRAs directory
        loras_path = f"{models_path}/loras"
        diagnosis["loras_path"] = loras_path
        diagnosis["loras_exists"] = os.path.exists(loras_path)
        
        if os.path.exists(loras_path):
            lora_files = os.listdir(loras_path)
            diagnosis["lora_files"] = []
            
            for file in lora_files:
                file_path = os.path.join(loras_path, file)
                diagnosis["lora_files"].append({
                    "name": file,
                    "size": os.path.getsize(file_path),
                    "size_mb": round(os.path.getsize(file_path) / (1024**2), 2)
                })
        else:
            diagnosis["lora_files"] = "Directory does not exist"
        
        # Get available models from ComfyUI API
        try:
            response = requests.get("http://127.0.0.1:8188/object_info", timeout=10)
            if response.status_code == 200:
                info = response.json()
                
                # Extract available checkpoints
                if "CheckpointLoaderSimple" in info:
                    checkpoint_loader = info["CheckpointLoaderSimple"]
                    if "input" in checkpoint_loader and "required" in checkpoint_loader["input"]:
                        ckpt_input = checkpoint_loader["input"]["required"]
                        if "ckpt_name" in ckpt_input:
                            available_checkpoints = ckpt_input["ckpt_name"][0]
                            diagnosis["available_checkpoints"] = available_checkpoints
                        else:
                            diagnosis["available_checkpoints"] = "ckpt_name not found in input"
                    else:
                        diagnosis["available_checkpoints"] = "No input/required section"
                else:
                    diagnosis["available_checkpoints"] = "CheckpointLoaderSimple not found"
                
                # Extract available LoRAs
                if "LoraLoader" in info:
                    lora_loader = info["LoraLoader"]
                    if "input" in lora_loader and "required" in lora_loader["input"]:
                        lora_input = lora_loader["input"]["required"]
                        if "lora_name" in lora_input:
                            available_loras = lora_input["lora_name"][0]
                            diagnosis["available_loras"] = available_loras
                        else:
                            diagnosis["available_loras"] = "lora_name not found in input"
                    else:
                        diagnosis["available_loras"] = "No input/required section"
                else:
                    diagnosis["available_loras"] = "LoraLoader not found"
                    
            else:
                diagnosis["api_error"] = f"object_info failed: {response.status_code}"
                
        except Exception as e:
            diagnosis["api_error"] = str(e)
        
        return diagnosis
        
    except Exception as e:
        return {"error": f"Diagnosis failed: {str(e)}"}

def fix_model_names():
    """Rename model files to standard names if needed"""
    try:
        print("🔧 Fixing model file names...")
        
        fixes = {}
        
        # Check and fix checkpoint name
        checkpoints_path = "/app/comfyui/models/checkpoints"
        if os.path.exists(checkpoints_path):
            files = os.listdir(checkpoints_path)
            
            # Look for SD 1.5 model (should be around 4GB)
            sd_file = None
            for file in files:
                file_path = os.path.join(checkpoints_path, file)
                if os.path.getsize(file_path) > 3000000000:  # 3GB+
                    sd_file = file
                    break
            
            if sd_file and sd_file != "sd15.safetensors":
                old_path = os.path.join(checkpoints_path, sd_file)
                new_path = os.path.join(checkpoints_path, "sd15.safetensors")
                
                try:
                    os.rename(old_path, new_path)
                    fixes["checkpoint"] = f"Renamed {sd_file} to sd15.safetensors"
                except Exception as e:
                    fixes["checkpoint"] = f"Failed to rename {sd_file}: {e}"
            elif sd_file == "sd15.safetensors":
                fixes["checkpoint"] = "Already named correctly"
            else:
                fixes["checkpoint"] = "No large model file found"
        
        # Check and fix LoRA name
        loras_path = "/app/comfyui/models/loras"
        if os.path.exists(loras_path):
            files = os.listdir(loras_path)
            
            # Look for Pepe LoRA (should be around 170MB)
            pepe_file = None
            for file in files:
                file_path = os.path.join(loras_path, file)
                size = os.path.getsize(file_path)
                if 150000000 < size < 200000000:  # 150-200MB
                    pepe_file = file
                    break
            
            if pepe_file and pepe_file != "pepe.safetensors":
                old_path = os.path.join(loras_path, pepe_file)
                new_path = os.path.join(loras_path, "pepe.safetensors")
                
                try:
                    os.rename(old_path, new_path)
                    fixes["lora"] = f"Renamed {pepe_file} to pepe.safetensors"
                except Exception as e:
                    fixes["lora"] = f"Failed to rename {pepe_file}: {e}"
            elif pepe_file == "pepe.safetensors":
                fixes["lora"] = "Already named correctly"
            else:
                fixes["lora"] = "No LoRA file found in expected size range"
        
        return fixes
        
    except Exception as e:
        return {"error": f"Fix failed: {str(e)}"}

def restart_comfyui():
    """Restart ComfyUI to refresh model list"""
    try:
        print("🔄 Restarting ComfyUI to refresh models...")
        
        # This won't actually restart in the serverless environment,
        # but we can try to force a model refresh
        
        # Try to clear any cached model info
        try:
            response = requests.post("http://127.0.0.1:8188/free", json={"unload_models": True}, timeout=10)
            if response.status_code == 200:
                return {"success": True, "message": "Models unloaded"}
            else:
                return {"success": False, "message": f"Unload failed: {response.status_code}"}
        except Exception as e:
            return {"success": False, "message": f"Restart attempt failed: {e}"}
            
    except Exception as e:
        return {"error": f"Restart failed: {str(e)}"}

def test_with_available_models():
    """Test generation using whatever models are actually available"""
    try:
        print("🧪 Testing with available models...")
        
        # Get available models
        response = requests.get("http://127.0.0.1:8188/object_info", timeout=10)
        if response.status_code != 200:
            return {"error": "Could not get object info"}
        
        info = response.json()
        
        # Get available checkpoints
        available_checkpoints = []
        if "CheckpointLoaderSimple" in info:
            checkpoint_loader = info["CheckpointLoaderSimple"]
            if "input" in checkpoint_loader and "required" in checkpoint_loader["input"]:
                ckpt_input = checkpoint_loader["input"]["required"]
                if "ckpt_name" in ckpt_input:
                    available_checkpoints = ckpt_input["ckpt_name"][0]
        
        if not available_checkpoints:
            return {"error": "No checkpoints available"}
        
        # Use the first available checkpoint
        checkpoint_name = available_checkpoints[0]
        print(f"Using checkpoint: {checkpoint_name}")
        
        # Simple workflow with available checkpoint
        workflow = {
            "1": {
                "inputs": {"ckpt_name": checkpoint_name},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": "a cute green frog, cartoon style",
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
                    "steps": 15,
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
                    "filename_prefix": f"WORKING_TEST_{int(time.time())}",
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
            return {"error": f"Submit failed: {response.status_code} - {response.text}"}
        
        prompt_data = response.json()
        prompt_id = prompt_data.get("prompt_id")
        
        return {
            "success": True,
            "checkpoint_used": checkpoint_name,
            "prompt_id": prompt_id,
            "message": "Test submitted successfully"
        }
        
    except Exception as e:
        return {"error": f"Test failed: {str(e)}"}

def handler(event):
    """Diagnostic handler to fix model issues"""
    print("🔍 MODEL DIAGNOSTIC HANDLER v1.0! 🕵️")
    
    try:
        input_data = event.get('input', {})
        action = input_data.get('action', 'diagnose')
        
        if action == 'diagnose':
            # Full diagnosis
            diagnosis = diagnose_models()
            
            return {
                "message": "🔍 Model diagnosis completed",
                "diagnosis": diagnosis,
                "recommendations": [
                    "Check if model files are in correct locations",
                    "Verify file names match expected names",
                    "Try fixing model names",
                    "Test with available models"
                ]
            }
        
        elif action == 'fix':
            # Diagnose, fix, and test
            diagnosis = diagnose_models()
            fixes = fix_model_names()
            restart_result = restart_comfyui()
            
            # Re-diagnose after fixes
            diagnosis_after = diagnose_models()
            
            return {
                "message": "🔧 Model fixes attempted",
                "diagnosis_before": diagnosis,
                "fixes_applied": fixes,
                "restart_result": restart_result,
                "diagnosis_after": diagnosis_after
            }
        
        elif action == 'test':
            # Test with whatever models are available
            test_result = test_with_available_models()
            
            return {
                "message": "🧪 Testing with available models",
                "test_result": test_result
            }
        
        else:
            return {"error": f"Unknown action: {action}"}
        
    except Exception as e:
        return {
            "error": f"Handler exception: {str(e)}"
        }

if __name__ == '__main__':
    print("🚀 Starting Model Diagnostic Handler...")
    runpod.serverless.start({'handler': handler})
