import runpod
import os
import requests
import time
import json
import subprocess
import threading
import sys
from pathlib import Path

def fix_cuda_libraries():
    """Fix missing CUDA libraries"""
    try:
        print("🔧 Fixing CUDA libraries...")
        
        # Step 1: Check current CUDA installation
        cuda_paths = [
            "/usr/local/cuda",
            "/usr/local/cuda-12.7",
            "/usr/local/cuda-12.1", 
            "/usr/local/cuda-11.8",
            "/opt/cuda"
        ]
        
        cuda_found = False
        cuda_path = None
        
        for path in cuda_paths:
            if os.path.exists(path):
                cuda_found = True
                cuda_path = path
                print(f"✅ Found CUDA at: {path}")
                break
        
        if not cuda_found:
            print("❌ No CUDA installation found")
        
        # Step 2: Install CUDA toolkit
        try:
            print("📦 Installing CUDA toolkit...")
            
            # Try apt-get installation
            apt_commands = [
                ["apt-get", "update"],
                ["apt-get", "install", "-y", "nvidia-cuda-toolkit"],
                ["apt-get", "install", "-y", "libcublas-12-1", "libcublas-dev-12-1"],
                ["apt-get", "install", "-y", "libcudnn8", "libcudnn8-dev"],
                ["apt-get", "install", "-y", "cuda-libraries-12-1", "cuda-libraries-dev-12-1"]
            ]
            
            for cmd in apt_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, timeout=120)
                    if result.returncode == 0:
                        print(f"✅ {' '.join(cmd[:3])} successful")
                    else:
                        print(f"⚠️ {' '.join(cmd[:3])} warning: {result.stderr.decode()[:100]}")
                except Exception as e:
                    print(f"❌ {' '.join(cmd[:3])} failed: {e}")
                    
        except Exception as e:
            print(f"❌ CUDA toolkit install failed: {e}")
        
        # Step 3: Set up environment paths
        cuda_paths_to_try = [
            "/usr/local/cuda/lib64",
            "/usr/local/cuda-12.7/lib64",
            "/usr/local/cuda-12.1/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/lib"
        ]
        
        existing_paths = []
        for path in cuda_paths_to_try:
            if os.path.exists(path):
                existing_paths.append(path)
        
        if existing_paths:
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            new_ld_path = ":".join(existing_paths + [current_ld_path])
            os.environ["LD_LIBRARY_PATH"] = new_ld_path
            print(f"✅ Set LD_LIBRARY_PATH: {new_ld_path[:100]}...")
        
        # Step 4: Try conda CUDA installation
        try:
            print("🐍 Installing CUDA via conda...")
            conda_result = subprocess.run([
                "conda", "install", "cudatoolkit=12.1", "cudnn", "-c", "conda-forge", "-y"
            ], capture_output=True, timeout=180)
            
            if conda_result.returncode == 0:
                print("✅ Conda CUDA install successful")
            else:
                print(f"⚠️ Conda CUDA warning: {conda_result.stderr.decode()[:100]}")
                
        except Exception as e:
            print(f"❌ Conda CUDA failed: {e}")
        
        # Step 5: Test for libcublas
        library_test_result = test_cuda_libraries()
        
        return library_test_result
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def test_cuda_libraries():
    """Test if CUDA libraries are available"""
    try:
        print("🧪 Testing CUDA libraries...")
        
        # Libraries we need
        required_libs = [
            "libcublas.so",
            "libcurand.so", 
            "libcudnn.so",
            "libcufft.so"
        ]
        
        found_libs = {}
        
        # Search paths
        search_paths = [
            "/usr/local/cuda/lib64",
            "/usr/local/cuda-12.7/lib64",
            "/usr/local/cuda-12.1/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/lib"
        ]
        
        for lib in required_libs:
            found = False
            for search_path in search_paths:
                if os.path.exists(search_path):
                    # Look for the library file
                    for file in os.listdir(search_path):
                        if lib in file:
                            found_libs[lib] = f"✅ Found at {search_path}/{file}"
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                found_libs[lib] = "❌ Not found"
        
        # Test with ldconfig
        try:
            ldconfig_result = subprocess.run(["ldconfig", "-p"], capture_output=True, timeout=10)
            ldconfig_output = ldconfig_result.stdout.decode()
            
            for lib in required_libs:
                if lib in ldconfig_output:
                    if found_libs[lib].startswith("❌"):
                        found_libs[lib] = f"✅ Found in system (ldconfig)"
                        
        except Exception as e:
            print(f"⚠️ ldconfig test failed: {e}")
        
        success_count = sum(1 for status in found_libs.values() if status.startswith("✅"))
        
        return {
            "success": success_count >= 2,  # Need at least 2 libraries
            "libraries": found_libs,
            "found_count": f"{success_count}/{len(required_libs)}"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def install_pytorch_with_cuda_fix():
    """Install PyTorch after CUDA libraries are fixed"""
    try:
        print("🚀 Installing PyTorch with CUDA fix...")
        
        # Ensure environment is set
        cuda_paths = [
            "/usr/local/cuda/lib64",
            "/usr/local/cuda-12.1/lib64",
            "/usr/lib/x86_64-linux-gnu"
        ]
        
        existing_cuda_paths = [p for p in cuda_paths if os.path.exists(p)]
        if existing_cuda_paths:
            os.environ["LD_LIBRARY_PATH"] = ":".join(existing_cuda_paths) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        
        # Remove existing PyTorch
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "uninstall", 
                "torch", "torchvision", "torchaudio", "-y"
            ], capture_output=True, timeout=60)
        except:
            pass
        
        # Install with CUDA
        install_commands = [
            {
                "name": "PyTorch CUDA 12.1",
                "cmd": [
                    sys.executable, "-m", "pip", "install",
                    "torch==2.1.2+cu121", "torchvision==0.16.2+cu121", "torchaudio==2.1.2+cu121",
                    "--index-url", "https://download.pytorch.org/whl/cu121",
                    "--force-reinstall"
                ]
            },
            {
                "name": "Latest PyTorch GPU",
                "cmd": [
                    sys.executable, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu121"
                ]
            }
        ]
        
        for install_cmd in install_commands:
            try:
                print(f"📦 Trying {install_cmd['name']}...")
                
                result = subprocess.run(
                    install_cmd["cmd"], 
                    capture_output=True, 
                    timeout=300,
                    env=os.environ
                )
                
                if result.returncode == 0:
                    # Test the installation
                    test_result = subprocess.run([
                        sys.executable, "-c",
                        "import torch; print(f'PyTorch {torch.__version__} CUDA: {torch.cuda.is_available()}')"
                    ], capture_output=True, timeout=30, env=os.environ)
                    
                    if test_result.returncode == 0:
                        output = test_result.stdout.decode()
                        if "CUDA: True" in output:
                            return True, f"SUCCESS: {install_cmd['name']} - {output.strip()}"
                        else:
                            print(f"⚠️ {install_cmd['name']} installed but no CUDA: {output}")
                    else:
                        print(f"⚠️ {install_cmd['name']} test failed: {test_result.stderr.decode()}")
                else:
                    print(f"❌ {install_cmd['name']} install failed: {result.stderr.decode()[:200]}")
                    
            except Exception as e:
                print(f"❌ {install_cmd['name']} exception: {e}")
                continue
        
        return False, "All PyTorch installation attempts failed"
        
    except Exception as e:
        return False, f"PyTorch install error: {str(e)}"

def redownload_models():
    """Re-download models that got wiped"""
    try:
        print("📥 Re-downloading wiped models...")
        
        downloads = {}
        
        # Download LoRA
        lora_path = "/app/comfyui/models/loras/pepe.safetensors"
        if not os.path.exists(lora_path):
            try:
                os.makedirs(os.path.dirname(lora_path), exist_ok=True)
                print("📥 Downloading Pepe LoRA (171MB)...")
                
                response = requests.get(
                    "https://huggingface.co/openfree/pepe/resolve/main/pepe.safetensors",
                    stream=True, timeout=180
                )
                response.raise_for_status()
                
                with open(lora_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                downloads["lora"] = f"✅ Downloaded {os.path.getsize(lora_path)} bytes"
                
            except Exception as e:
                downloads["lora"] = f"❌ Failed: {e}"
        else:
            downloads["lora"] = "✅ Already exists"
        
        # Download model
        model_path = "/app/comfyui/models/checkpoints/sd15.safetensors"
        if not os.path.exists(model_path):
            try:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                print("📥 Downloading SD 1.5 model (4.2GB)... This will take a while...")
                
                response = requests.get(
                    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors",
                    stream=True, timeout=600
                )
                response.raise_for_status()
                
                downloaded = 0
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if downloaded % (200*1024*1024) == 0:  # Progress every 200MB
                                print(f"📊 Model download: {downloaded // (1024*1024)}MB...")
                
                downloads["model"] = f"✅ Downloaded {os.path.getsize(model_path)} bytes"
                
            except Exception as e:
                downloads["model"] = f"❌ Failed: {e}"
        else:
            downloads["model"] = "✅ Already exists"
        
        return downloads
        
    except Exception as e:
        return {"error": str(e)}

def comprehensive_final_test():
    """Final comprehensive test of everything"""
    try:
        print("🏁 Final comprehensive test...")
        
        results = {
            "cuda_libraries": test_cuda_libraries(),
            "pytorch": {},
            "comfyui": {},
            "files": {},
            "summary": {}
        }
        
        # Test PyTorch
        try:
            import torch
            results["pytorch"] = {
                "version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count()
            }
            
            if torch.cuda.is_available():
                results["pytorch"]["device_name"] = torch.cuda.get_device_name(0)
                try:
                    # Test GPU operation
                    test_tensor = torch.randn(100, 100).cuda()
                    results["pytorch"]["gpu_test"] = "✅ GPU tensor creation works"
                except Exception as e:
                    results["pytorch"]["gpu_test"] = f"❌ GPU tensor failed: {e}"
            
        except Exception as e:
            results["pytorch"] = {"error": str(e)}
        
        # Test ComfyUI imports
        try:
            os.chdir("/app/comfyui")
            
            comfyui_tests = ["comfy.utils", "nodes", "execution"]
            results["comfyui"] = {}
            
            for test in comfyui_tests:
                try:
                    __import__(test)
                    results["comfyui"][test] = "✅ Import successful"
                except Exception as e:
                    results["comfyui"][test] = f"❌ {str(e)[:100]}"
            
        except Exception as e:
            results["comfyui"] = {"error": str(e)}
        
        # Test files
        files_to_check = {
            "lora": "/app/comfyui/models/loras/pepe.safetensors",
            "model": "/app/comfyui/models/checkpoints/sd15.safetensors"
        }
        
        results["files"] = {}
        for name, path in files_to_check.items():
            if os.path.exists(path):
                results["files"][name] = f"✅ {os.path.getsize(path)} bytes"
            else:
                results["files"][name] = "❌ Missing"
        
        # Calculate summary
        cuda_ok = results["cuda_libraries"].get("success", False)
        pytorch_ok = results["pytorch"].get("cuda_available", False)
        comfyui_ok = sum(1 for r in results["comfyui"].values() if isinstance(r, str) and r.startswith("✅")) >= 2
        files_ok = all(r.startswith("✅") for r in results["files"].values())
        
        results["summary"] = {
            "cuda_libraries": cuda_ok,
            "pytorch_gpu": pytorch_ok,
            "comfyui_imports": comfyui_ok,
            "files_ready": files_ok,
            "overall_ready": cuda_ok and pytorch_ok and comfyui_ok and files_ok,
            "readiness_score": sum([cuda_ok, pytorch_ok, comfyui_ok, files_ok])
        }
        
        return results
        
    except Exception as e:
        return {"error": str(e)}

def handler(event):
    """CUDA Library Fix Handler"""
    print("🐸 PEPE WORKER v7.0 - CUDA LIBRARY SURGEON! 🔧")
    print("🎯 Fixing libcublas and CUDA libraries...")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'pepe the frog')
        
        print(f"📝 Processing: {prompt}")
        
        # Step 1: Fix CUDA libraries
        print("🔄 Step 1: Fixing CUDA libraries...")
        cuda_fix = fix_cuda_libraries()
        
        # Step 2: Install PyTorch with fixed CUDA
        print("🔄 Step 2: Installing PyTorch with CUDA...")
        pytorch_success, pytorch_msg = install_pytorch_with_cuda_fix()
        
        # Step 3: Re-download wiped models
        print("🔄 Step 3: Re-downloading models...")
        model_downloads = redownload_models()
        
        # Step 4: Final comprehensive test
        print("🔄 Step 4: Final system test...")
        final_test = comprehensive_final_test()
        
        return {
            "message": f"🐸 CUDA Library Surgery: {prompt}",
            "status": "success" if final_test.get("summary", {}).get("overall_ready") else "surgery_in_progress",
            "cuda_fix": cuda_fix,
            "pytorch": {
                "success": pytorch_success,
                "message": pytorch_msg
            },
            "model_downloads": model_downloads,
            "final_test": final_test,
            "ready_for_generation": final_test.get("summary", {}).get("overall_ready", False),
            "readiness_score": f"{final_test.get('summary', {}).get('readiness_score', 0)}/4",
            "surgery_report": [
                f"🔧 CUDA Libraries: {'FIXED' if cuda_fix.get('success') else 'FAILED'}",
                f"🚀 PyTorch GPU: {'WORKING' if pytorch_success else 'FAILED'}",
                f"📥 Models: {'DOWNLOADED' if all(r.startswith('✅') for r in model_downloads.values() if isinstance(r, str)) else 'MISSING'}",
                f"💻 ComfyUI: {'READY' if final_test.get('summary', {}).get('comfyui_imports') else 'BROKEN'}",
                f"🎯 {'PEPE GENERATION READY!' if final_test.get('summary', {}).get('overall_ready') else 'STILL PERFORMING SURGERY...'}"
            ],
            "next_operation": "START PEPE SERVER" if final_test.get("summary", {}).get("overall_ready") else "CONTINUE CUDA SURGERY"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "surgery_failed",
            "debug": "CUDA surgery exception"
        }

if __name__ == '__main__':
    print("🚀 Starting CUDA Library Surgeon v7.0...")
    print("🔧 Operating on libcublas.so...")
    runpod.serverless.start({'handler': handler})
