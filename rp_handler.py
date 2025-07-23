import runpod
import os
import requests
import time
import json
import subprocess
import sys
import base64
import glob

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
        
        # Kill existing processes
        try:
            subprocess.run(["pkill", "-f", "main.py"], check=False, timeout=5)
            time.sleep(3)
        except:
            pass
        
        # Check if already running
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
        
        # Start server
        cmd = [sys.executable, "main.py", "--listen", "0.0.0.0", "--port", "8188", "--verbose"]
        
        process = subprocess.Popen(
            cmd,
            cwd=comfyui_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        # Wait for startup
        for i in range(120):
            try:
                response = requests.get("http://localhost:8188/", timeout=2)
                if response.status_code == 200:
                    print("✅ ComfyUI server started successfully!")
                    return True, f"Server running (PID: {process.pid})"
            except:
                pass
            
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                return False, f"Process crashed: {stderr.decode()[-500:]}"
            
            if i % 20 == 0:
                print(f"⏳ Server starting... ({i}/120 seconds)")
            
            time.sleep(1)
        
        return False, "Server startup timeout"
        
    except Exception as e:
        return False, f"Server start exception: {str(e)}"

def get_available_models():
    """Get list of available models and LoRAs from ComfyUI"""
    try:
        response = requests.get("http://localhost:8188/object_info", timeout=10)
        if response.status_code != 200:
            return {"error": f"Object info request failed: {response.status_code}"}
        
        info = response.json()
        
        # Extract model info
        models_info = {}
        
        # Checkpoints
        if "CheckpointLoaderSimple" in info:
            checkpoint_info = info["CheckpointLoaderSimple"]
            if "input" in checkpoint_info and "required" in checkpoint_info["input"]:
                ckpt_input = checkpoint_info["input"]["required"]
                if "ckpt_name" in ckpt_input:
                    models_info["checkpoints"] = ckpt_input["ckpt_name"][0] if isinstance(ckpt_input["ckpt_name"], list) else []
        
        # LoRAs
        if "LoraLoader" in info:
            lora_info = info["LoraLoader"]
            if "input" in lora_info and "required" in lora_info["input"]:
                lora_input = lora_info["input"]["required"]
                if "lora_name" in lora_input:
                    models_info["loras"] = lora_input["lora_name"][0] if isinstance(lora_input["lora_name"], list) else []
        
        # Samplers
        if "KSampler" in info:
            ksampler_info = info["KSampler"]
            if "input" in ksampler_info and "required" in ksampler_info["input"]:
                sampler_input = ksampler_info["input"]["required"]
                if "sampler_name" in sampler_input:
                    models_info["samplers"] = sampler_input["sampler_name"][0] if isinstance(sampler_input["sampler_name"], list) else []
        
        return {"success": True, "models": models_info}
        
    except Exception as e:
        return {"error": f"Failed to get model info: {str(e)}"}

def validate_workflow(workflow):
    """Validate workflow against ComfyUI"""
    try:
        # Send workflow validation request
        response = requests.post(
            "http://localhost:8188/prompt",
            json={"prompt": workflow, "validate_only": True},  # Some ComfyUI versions support validation
            timeout=10
        )
        
        return {
            "status_code": response.status_code,
            "response_text": response.text[:500],
            "valid": response.status_code == 200
        }
        
    except Exception as e:
        return {"error": f"Validation failed: {str(e)}"}

def test_minimal_workflow():
    """Test the most minimal possible workflow"""
    try:
        print("🧪 Testing minimal workflow...")
        
        # Ultra-simple workflow - just load model and save empty image
        workflow = {
            "1": {
                "inputs": {"ckpt_name": "sd15.safetensors"},
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "width": 512,
                    "height": 512,
                    "batch_size": 1
                },
                "class_type": "EmptyLatentImage"
            },
            "3": {
                "inputs": {
                    "samples": ["2", 0],
                    "vae": ["1", 2]
                },
                "class_type": "VAEDecode"
            },
            "4": {
                "inputs": {
                    "filename_prefix": f"MINIMAL_TEST_{int(time.time())}",
                    "images": ["3", 0]
                },
                "class_type": "SaveImage"
            }
        }
        
        # Submit minimal workflow
        response = requests.post(
            "http://localhost:8188/prompt",
            json={"prompt": workflow},
            timeout=30
        )
        
        result = {
            "submit_status": response.status_code,
            "submit_response": response.text[:200] if response.text else "No response text"
        }
        
        if response.status_code != 200:
            return {"success": False, "error": "Submit failed", "details": result}
        
        try:
            prompt_data = response.json()
            prompt_id = prompt_data.get("prompt_id")
            result["prompt_id"] = prompt_id
        except:
            result["prompt_id"] = None
            return {"success": False, "error": "No JSON response", "details": result}
        
        if not prompt_id:
            return {"success": False, "error": "No prompt ID", "details": result}
        
        print(f"✅ Minimal workflow queued: {prompt_id}")
        
        # Monitor for completion
        for i in range(60):  # 1 minute timeout
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
                        result["completed"] = True
                        result["completion_time"] = i
                        break
                
                if i % 10 == 0:
                    print(f"⏳ Minimal test... ({i}/60s)")
                
                time.sleep(1)
                
            except Exception as e:
                result["monitor_error"] = str(e)
                break
        
        # Check history
        try:
            history_response = requests.get("http://localhost:8188/history", timeout=5)
            if history_response.status_code == 200:
                history = history_response.json()
                if prompt_id in history:
                    execution = history[prompt_id]
                    result["history_found"] = True
                    result["execution_status"] = execution.get("status", {})
                    result["outputs"] = list(execution.get("outputs", {}).keys())
                else:
                    result["history_found"] = False
                    result["available_history_ids"] = list(history.keys())[-3:]
            else:
                result["history_error"] = f"History request failed: {history_response.status_code}"
        except Exception as e:
            result["history_error"] = str(e)
        
        return {"success": result.get("completed", False), "details": result}
        
    except Exception as e:
        return {"success": False, "error": f"Minimal test failed: {str(e)}"}

def diagnose_generation_issue():
    """Comprehensive diagnosis of why generation fails"""
    try:
        print("🔍 Diagnosing generation issues...")
        
        diagnosis = {}
        
        # Step 1: Check available models
        print("🔍 Checking available models...")
        models_info = get_available_models()
        diagnosis["models"] = models_info
        
        # Step 2: Test minimal workflow
        print("🔍 Testing minimal workflow...")
        minimal_test = test_minimal_workflow()
        diagnosis["minimal_test"] = minimal_test
        
        # Step 3: Check queue status
        try:
            queue_response = requests.get("http://localhost:8188/queue", timeout=5)
            if queue_response.status_code == 200:
                queue_data = queue_response.json()
                diagnosis["queue_status"] = {
                    "running": len(queue_data.get("queue_running", [])),
                    "pending": len(queue_data.get("queue_pending", [])),
                    "queue_data": queue_data
                }
            else:
                diagnosis["queue_status"] = {"error": f"Queue check failed: {queue_response.status_code}"}
        except Exception as e:
            diagnosis["queue_status"] = {"error": str(e)}
        
        # Step 4: Check system stats
        try:
            stats_response = requests.get("http://localhost:8188/system_stats", timeout=5)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                diagnosis["system_stats"] = stats
            else:
                diagnosis["system_stats"] = {"error": f"Stats check failed: {stats_response.status_code}"}
        except Exception as e:
            diagnosis["system_stats"] = {"error": str(e)}
        
        # Step 5: Check output directory permissions
        output_dir = "/runpod-volume/comfyui/output"
        try:
            # Test write permissions
            test_file = os.path.join(output_dir, f"test_write_{int(time.time())}.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            diagnosis["output_permissions"] = {"writable": True}
        except Exception as e:
            diagnosis["output_permissions"] = {"writable": False, "error": str(e)}
        
        return diagnosis
        
    except Exception as e:
        return {"error": f"Diagnosis failed: {str(e)}"}

def handler(event):
    """Diagnostic handler to identify generation issues"""
    print("🔍 DIAGNOSTIC HANDLER v1.0! 🕵️")
    print("🎯 Finding out why generation fails despite working server")
    
    try:
        input_data = event.get('input', {})
        
        # Step 1: Basic checks
        print("🔄 Step 1: Volume and server checks...")
        volume_check = check_volume_setup()
        
        if not volume_check.get("ready"):
            return {
                "error": "Volume not ready",
                "volume_check": volume_check
            }
        
        # Step 2: Start server
        server_success, server_result = start_comfyui_server()
        
        if not server_success:
            return {
                "error": "Server failed to start",
                "server_result": server_result
            }
        
        # Step 3: Comprehensive diagnosis
        print("🔄 Step 3: Running comprehensive diagnosis...")
        diagnosis = diagnose_generation_issue()
        
        return {
            "message": "🔍 DIAGNOSTIC COMPLETE",
            "volume_check": volume_check,
            "server_result": server_result,
            "diagnosis": diagnosis,
            "recommendations": [
                "Check if models are loading correctly",
                "Verify workflow syntax",
                "Check file permissions",
                "Look for silent failures in ComfyUI logs"
            ]
        }
        
    except Exception as e:
        return {
            "error": f"Diagnostic failed: {str(e)}",
            "partial_results": locals().get("diagnosis", {})
        }

if __name__ == '__main__':
    print("🚀 Starting Diagnostic Handler...")
    runpod.serverless.start({'handler': handler})
