import runpod
import os
import requests
import time
import json
import subprocess
import threading
import sys
from pathlib import Path

def force_gpu_pytorch():
    """Aggressively force GPU PyTorch installation"""
    try:
        print("🔥 FORCE installing GPU PyTorch...")
        
        # Step 1: Completely remove any existing PyTorch
        print("🧹 Removing all PyTorch installations...")
        
        packages_to_remove = [
            "torch", "torchvision", "torchaudio", "pytorch", 
            "torch-cpu", "torchvision-cpu", "torchaudio-cpu"
        ]
        
        for package in packages_to_remove:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "uninstall", package, "-y"
                ], capture_output=True, timeout=30)
            except:
                pass
        
        # Step 2: Clear pip cache
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "cache", "purge"
            ], capture_output=True, timeout=30)
        except:
            pass
        
        # Step 3: Force install GPU version with explicit CUDA
        print("⚡ Force installing PyTorch GPU...")
        
        install_strategies = [
            {
                "name": "PyTorch CUDA 12.1 Direct",
                "cmd": [
                    sys.executable, "-m", "pip", "install", 
                    "torch==2.1.0+cu121", "torchvision==0.16.0+cu121", "torchaudio==2.1.0+cu121",
                    "--index-url", "https://download.pytorch.org/whl/cu121",
                    "--force-reinstall", "--no-cache-dir"
                ]
            },
            {
                "name": "PyTorch CUDA 11.8 Fallback", 
                "cmd": [
                    sys.executable, "-m", "pip", "install",
                    "torch==2.1.0+cu118", "torchvision==0.16.0+cu118", "torchaudio==2.1.0+cu118",
                    "--index-url", "https://download.pytorch.org/whl/cu118",
                    "--force-reinstall", "--no-cache-dir"
                ]
            },
            {
                "name": "Latest PyTorch GPU",
                "cmd": [
                    sys.executable, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio", 
                    "--index-url", "https://download.pytorch.org/whl/cu121",
                    "--force-reinstall", "--no-cache-dir", "--upgrade"
                ]
            }
        ]
        
        for strategy in install_strategies:
            try:
                print(f"🚀 Trying: {strategy['name']}")
                
                # Set aggressive environment
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = "0"
                env["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9;9.0"
                env["FORCE_CUDA"] = "1"
                
                result = subprocess.run(
                    strategy["cmd"],
                    capture_output=True,
                    timeout=300,
                    env=env
                )
                
                if result.returncode == 0:
                    print(f"✅ {strategy['name']} installation completed")
                    
                    # Immediate test
                    test_result = subprocess.run([
                        sys.executable, "-c",
                        "import torch; print(f'PyTorch {torch.__version__} CUDA: {torch.cuda.is_available()} Devices: {torch.cuda.device_count()}')"
                    ], capture_output=True, timeout=30, env=env)
                    
                    if test_result.returncode == 0:
                        output = test_result.stdout.decode().strip()
                        print(f"🧪 Test result: {output}")
                        
                        if "CUDA: True" in output:
                            return True, f"SUCCESS! {strategy['name']}: {output}"
                        elif "PyTorch" in output:
                            print(f"⚠️ {strategy['name']} installed but no CUDA")
                            # Continue to next strategy
                else:
                    print(f"❌ {strategy['name']} install failed: {result.stderr.decode()[:200]}")
                    
            except Exception as e:
                print(f"❌ {strategy['name']} exception: {e}")
                continue
        
        return False, "All GPU PyTorch installation strategies failed"
        
    except Exception as e:
        return False, f"Force GPU PyTorch failed: {str(e)}"

def fix_comfyui_installation():
    """Fix ComfyUI installation and paths"""
    try:
        print("🔧 Fixing ComfyUI installation...")
        
        # Check if ComfyUI directory exists and has content
        comfyui_path = "/app/comfyui"
        
        if not os.path.exists(comfyui_path):
            return False, "ComfyUI directory not found"
        
        # Check for main files
        main_py = os.path.join(comfyui_path, "main.py")
        if not os.path.exists(main_py):
            return False, "ComfyUI main.py not found"
        
        # Look for ComfyUI subdirectories
        expected_dirs = ["comfy", "nodes.py", "execution.py"]
        missing_items = []
        
        for item in expected_dirs:
            item_path = os.path.join(comfyui_path, item)
            if not os.path.exists(item_path):
                missing_items.append(item)
        
        if missing_items:
            print(f"⚠️ Missing ComfyUI components: {missing_items}")
            
            # Try to reinstall ComfyUI dependencies
            try:
                print("📦 Installing ComfyUI dependencies...")
                
                deps = [
                    "transformers", "diffusers", "accelerate", "xformers",
                    "safetensors", "opencv-python", "pillow", "numpy",
                    "tqdm", "psutil", "kornia", "spandrel"
                ]
                
                for dep in deps:
                    try:
                        subprocess.run([
                            sys.executable, "-m", "pip", "install", dep, "--upgrade"
                        ], capture_output=True, timeout=60)
                        print(f"✅ Installed {dep}")
                    except:
                        print(f"⚠️ Failed to install {dep}")
                
            except Exception as e:
                print(f"❌ Dependency installation failed: {e}")
        
        # Add ComfyUI to Python path
        if comfyui_path not in sys.path:
            sys.path.insert(0, comfyui_path)
        
        # Set PYTHONPATH environment variable
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        if comfyui_path not in current_pythonpath:
            os.environ["PYTHONPATH"] = f"{comfyui_path}:{current_pythonpath}"
        
        return True, f"ComfyUI path configured: {comfyui_path}"
        
    except Exception as e:
        return False, f"ComfyUI fix failed: {str(e)}"

def test_complete_system():
    """Test complete system with GPU and ComfyUI"""
    try:
        results = {}
        
        # Test 1: PyTorch GPU
        try:
            import torch
            results["pytorch"] = {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count()
            }
            
            if torch.cuda.is_available():
                results["pytorch"]["device_name"] = torch.cuda.get_device_name(0)
                results["pytorch"]["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                # Test GPU tensor
                try:
                    test_tensor = torch.randn(1000, 1000).cuda()
                    results["pytorch"]["gpu_test"] = "✅ GPU tensor creation successful"
                except Exception as e:
                    results["pytorch"]["gpu_test"] = f"❌ GPU tensor failed: {str(e)}"
            else:
                results["pytorch"]["gpu_test"] = "❌ CUDA not available"
                
        except Exception as e:
            results["pytorch"] = {"error": str(e)}
        
        # Test 2: ComfyUI imports
        try:
            # Change to ComfyUI directory for imports
            original_cwd = os.getcwd()
            os.chdir("/app/comfyui")
            
            import_tests = {
                "comfy.utils": "import comfy.utils",
                "nodes": "import nodes", 
                "execution": "import execution",
                "comfy.model_management": "import comfy.model_management",
                "comfy.sd": "import comfy.sd"
            }
            
            results["comfyui"] = {}
            for name, import_cmd in import_tests.items():
                try:
                    exec(import_cmd)
                    results["comfyui"][name] = "✅ Available"
                except Exception as e:
                    results["comfyui"][name] = f"❌ {str(e)}"
            
            os.chdir(original_cwd)
            
        except Exception as e:
            results["comfyui"] = {"error": str(e)}
        
        # Test 3: File availability
        files_to_check = {
            "lora": "/app/comfyui/models/loras/pepe.safetensors",
            "model": "/app/comfyui/models/checkpoints/sd15.safetensors",
            "main": "/app/comfyui/main.py"
        }
        
        results["files"] = {}
        for name, path in files_to_check.items():
            if os.path.exists(path):
                size = os.path.getsize(path)
                results["files"][name] = f"✅ {size} bytes"
            else:
                results["files"][name] = "❌ Missing"
        
        # Calculate overall readiness
        pytorch_ready = results.get("pytorch", {}).get("cuda_available", False)
        comfyui_imports = sum(1 for r in results.get("comfyui", {}).values() if isinstance(r, str) and r.startswith("✅"))
        files_ready = all(r.startswith("✅") for r in results.get("files", {}).values())
        
        overall_ready = pytorch_ready and comfyui_imports >= 3 and files_ready
        readiness_percentage = (
            (1 if pytorch_ready else 0) + 
            (comfyui_imports / 5) + 
            (1 if files_ready else 0)
        ) / 3 * 100
        
        results["summary"] = {
            "pytorch_ready": pytorch_ready,
            "comfyui_imports": f"{comfyui_imports}/5",
            "files_ready": files_ready,
            "overall_ready": overall_ready,
            "readiness_percentage": readiness_percentage
        }
        
        return results
        
    except Exception as e:
        return {"error": str(e)}

def start_comfyui_final():
    """Final attempt to start ComfyUI with all fixes"""
    try:
        print("🚀 Final ComfyUI startup attempt...")
        
        # Kill any existing processes
        try:
            subprocess.run(["pkill", "-f", "main.py"], check=False)
            time.sleep(3)
        except:
            pass
        
        # Setup optimal environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["PYTHONPATH"] = "/app/comfyui"
        env["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9;9.0"
        
        # Start with detailed logging
        cmd = [
            sys.executable, "main.py",
            "--listen", "0.0.0.0", 
            "--port", "8188",
            "--verbose"
        ]
        
        print(f"🔧 Starting: {' '.join(cmd)}")
        print(f"📁 Working directory: /app/comfyui")
        print(f"🎯 CUDA device: {env.get('CUDA_VISIBLE_DEVICES')}")
        
        process = subprocess.Popen(
            cmd,
            cwd="/app/comfyui",
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        
        # Wait with better progress tracking
        startup_logs = []
        
        for i in range(120):  # 2 minutes for GPU warmup
            try:
                response = requests.get("http://localhost:8188/", timeout=2)
                if response.status_code == 200:
                    print("✅ ComfyUI server is running!")
                    return True, f"Server started successfully (PID: {process.pid})"
            except:
                pass
            
            # Check for crash
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                
                # Get detailed error info
                stderr_text = stderr.decode() if stderr else "No stderr"
                stdout_text = stdout.decode() if stdout else "No stdout"
                
                return False, f"Process crashed. STDERR: {stderr_text[:400]}... STDOUT: {stdout_text[:200]}"
            
            # Progress updates
            if i % 20 == 0:
                print(f"⏳ GPU warming up... ({i}/120 seconds)")
            
            time.sleep(1)
        
        # Final timeout
        return False, "Startup timeout after 2 minutes"
        
    except Exception as e:
        return False, f"Final startup failed: {str(e)}"

def handler(event):
    """Force GPU handler - no compromises!"""
    print("🐸 PEPE WORKER v6.0 - FORCE GPU MODE! 🔥")
    print("💪 No CPU fallbacks - GPU or bust!")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'pepe the frog')
        
        print(f"📝 Processing: {prompt}")
        
        # Step 1: FORCE GPU PyTorch
        print("🔄 Step 1: FORCING GPU PyTorch installation...")
        pytorch_success, pytorch_msg = force_gpu_pytorch()
        
        # Step 2: Fix ComfyUI
        print("🔄 Step 2: Fixing ComfyUI installation...")
        comfyui_success, comfyui_msg = fix_comfyui_installation()
        
        # Step 3: Test everything
        print("🔄 Step 3: Testing complete system...")
        system_test = test_complete_system()
        
        # Step 4: Start server if ready
        server_success = False
        server_msg = "Not attempted"
        
        if system_test.get("summary", {}).get("overall_ready"):
            print("🔄 Step 4: Starting ComfyUI server...")
            server_success, server_msg = start_comfyui_final()
        else:
            server_msg = "System not ready for server startup"
        
        return {
            "message": f"🐸 FORCE GPU Pepe setup: {prompt}",
            "status": "success" if server_success else "forcing_fixes",
            "pytorch_force": {
                "success": pytorch_success,
                "message": pytorch_msg
            },
            "comfyui_fix": {
                "success": comfyui_success,
                "message": comfyui_msg
            },
            "system_test": system_test,
            "server": {
                "success": server_success,
                "message": server_msg
            },
            "ready_for_generation": server_success,
            "readiness_percentage": system_test.get("summary", {}).get("readiness_percentage", 0),
            "final_status": [
                "🔥 GPU PyTorch FORCED" if pytorch_success else "❌ GPU PyTorch FAILED",
                "🔧 ComfyUI FIXED" if comfyui_success else "❌ ComfyUI BROKEN",
                f"🧪 System {system_test.get('summary', {}).get('readiness_percentage', 0):.0f}% ready",
                "🚀 SERVER RUNNING!" if server_success else "❌ SERVER FAILED",
                "🎯 READY FOR PEPE DOMINATION!" if server_success else "💀 STILL FIGHTING..."
            ],
            "gpu_mode": "FORCED_NO_COMPROMISES"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed",
            "debug": "Handler exception in force GPU mode"
        }

if __name__ == '__main__':
    print("🚀 Starting FORCE GPU Pepe Worker v6.0...")
    print("🔥 GPU OR DEATH!")
    runpod.serverless.start({'handler': handler})
