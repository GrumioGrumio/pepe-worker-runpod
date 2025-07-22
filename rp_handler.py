# GitHub Pepe Worker Files
## 📁 Files to create in your GitHub repository

### File 1: `rp_handler.py`
```python
import runpod
import json
import base64
import io
import requests
import os
import time
import websocket
import threading
from PIL import Image

class ComfyUIClient:
    def __init__(self):
        self.server_url = "http://127.0.0.1:8188"
        self.client_id = "runpod-client"
    
    def queue_prompt(self, workflow):
        """Queue a prompt in ComfyUI"""
        try:
            response = requests.post(f"{self.server_url}/prompt", json={
                "prompt": workflow,
                "client_id": self.client_id
            })
            if response.status_code == 200:
                return response.json()["prompt_id"]
            else:
                print(f"❌ Queue error: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Queue exception: {e}")
            return None
    
    def get_image(self, filename, subfolder="", folder_type="output"):
        """Get generated image from ComfyUI"""
        try:
            response = requests.get(f"{self.server_url}/view", params={
                "filename": filename,
                "subfolder": subfolder,
                "type": folder_type
            })
            if response.status_code == 200:
                return response.content
            return None
        except Exception as e:
            print(f"❌ Get image error: {e}")
            return None
    
    def wait_for_completion(self, prompt_id, timeout=300):
        """Wait for prompt completion and return image"""
        print(f"⏳ Waiting for completion of {prompt_id}...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check history for completion
                response = requests.get(f"{self.server_url}/history/{prompt_id}")
                
                if response.status_code == 200:
                    history = response.json()
                    
                    if prompt_id in history:
                        outputs = history[prompt_id].get("outputs", {})
                        
                        # Look for SaveImage node output
                        for node_id, output in outputs.items():
                            if "images" in output:
                                images = output["images"]
                                if images:
                                    filename = images[0]["filename"]
                                    subfolder = images[0].get("subfolder", "")
                                    
                                    print(f"📁 Found image: {filename}")
                                    
                                    # Get the actual image
                                    image_data = self.get_image(filename, subfolder)
                                    if image_data:
                                        return image_data
                
                time.sleep(2)
                
            except Exception as e:
                print(f"⚠️ Waiting error: {e}")
                time.sleep(5)
        
        print(f"⏰ Timeout after {timeout} seconds")
        return None

def create_pepe_workflow(prompt, lora_strength=0.8, steps=20, cfg=7.5):
    """Create ComfyUI workflow for FLUX + Pepe LoRA"""
    return {
        "1": {
            "inputs": {
                "ckpt_name": "flux1-dev.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "2": {
            "inputs": {
                "model": ["1", 0],
                "clip": ["1", 1],
                "lora_name": "pepe.safetensors", 
                "strength_model": lora_strength,
                "strength_clip": lora_strength
            },
            "class_type": "LoraLoader"
        },
        "3": {
            "inputs": {
                "text": prompt,
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "4": {
            "inputs": {
                "text": "blurry, low quality, distorted, ugly",
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "5": {
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        "6": {
            "inputs": {
                "seed": int(time.time()),
                "steps": steps,
                "cfg": cfg,
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
                "filename_prefix": "pepe_output",
                "images": ["7", 0]
            },
            "class_type": "SaveImage"
        }
    }

def handler(event):
    """Main handler for Pepe generation"""
    print("🐸 Pepe Worker Started!")
    
    try:
        # Extract input parameters
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'pepe the frog')
        lora_strength = input_data.get('lora_strength', 0.8)
        steps = input_data.get('steps', 20) 
        cfg_scale = input_data.get('cfg_scale', 7.5)
        
        print(f"📝 Prompt: {prompt}")
        print(f"🎛️ LoRA Strength: {lora_strength}")
        
        # Create ComfyUI client
        client = ComfyUIClient()
        
        # Create workflow
        workflow = create_pepe_workflow(prompt, lora_strength, steps, cfg_scale)
        
        # Queue the prompt
        prompt_id = client.queue_prompt(workflow)
        if not prompt_id:
            raise Exception("Failed to queue prompt in ComfyUI")
        
        print(f"📋 Queued prompt: {prompt_id}")
        
        # Wait for completion
        image_data = client.wait_for_completion(prompt_id)
        if not image_data:
            raise Exception("Failed to generate image")
        
        # Convert to base64
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        print(f"✅ Perfect Pepe generated! Size: {len(image_data)} bytes")
        
        return {
            "image": image_b64,
            "prompt": prompt,
            "lora_strength": lora_strength,
            "status": "success",
            "message": "Perfect Pepe generated via ComfyUI + FLUX + Pepe LoRA"
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error: {error_msg}")
        
        return {
            "error": error_msg,
            "status": "failed",
            "message": "Pepe generation failed"
        }

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
```

### File 2: `Dockerfile`
```dockerfile
# Use official RunPod ComfyUI image as base
FROM runpod/worker-comfyui:dev-cuda12.1

# Set working directory
WORKDIR /comfyui

# Install additional dependencies
RUN pip install runpod requests pillow websocket-client

# Download Pepe LoRA to the correct location
RUN cd /comfyui/models/loras && \
    wget -O pepe.safetensors "https://huggingface.co/openfree/pepe/resolve/main/pepe.safetensors"

# Verify LoRA downloaded
RUN ls -la /comfyui/models/loras/

# Copy our handler
COPY rp_handler.py /comfyui/
COPY start_worker.py /comfyui/

# Expose ports
EXPOSE 8188 8080

# Start the worker
CMD ["python", "start_worker.py"]
```

### File 3: `start_worker.py` 
```python
import subprocess
import time
import threading
import os
import sys

def start_comfyui():
    """Start ComfyUI server"""
    print("🚀 Starting ComfyUI server...")
    try:
        # Change to ComfyUI directory
        os.chdir('/comfyui')
        
        # Start ComfyUI with proper settings
        subprocess.run([
            sys.executable, "main.py", 
            "--listen", "0.0.0.0", 
            "--port", "8188",
            "--disable-auto-launch"
        ])
    except Exception as e:
        print(f"❌ ComfyUI start error: {e}")

def wait_for_comfyui():
    """Wait for ComfyUI to be ready"""
    import requests
    
    print("⏳ Waiting for ComfyUI to start...")
    max_attempts = 60
    
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://127.0.0.1:8188", timeout=5)
            if response.status_code == 200:
                print("✅ ComfyUI is ready!")
                return True
        except:
            pass
        
        print(f"⏳ Attempt {attempt + 1}/{max_attempts}...")
        time.sleep(5)
    
    print("❌ ComfyUI failed to start")
    return False

def start_runpod_handler():
    """Start RunPod handler after ComfyUI is ready"""
    if wait_for_comfyui():
        print("🚀 Starting RunPod handler...")
        os.chdir('/comfyui')
        subprocess.run([sys.executable, "rp_handler.py"])
    else:
        print("❌ Cannot start handler - ComfyUI not ready")

if __name__ == "__main__":
    print("🎭 Starting Pepe Worker Services...")
    
    # Start ComfyUI in background thread
    comfyui_thread = threading.Thread(target=start_comfyui)
    comfyui_thread.daemon = True
    comfyui_thread.start()
    
    # Start RunPod handler (this will wait for ComfyUI)
    start_runpod_handler()
```

### File 4: `test_input.json`
```json
{
    "input": {
        "prompt": "pepe the frog on a skateboard, meme style",
        "lora_strength": 0.8,
        "steps": 20,
        "cfg_scale": 7.5
    }
}
```

### File 5: `README.md`
```markdown
# Pepe Worker for RunPod Serverless

Perfect Pepe meme generation using ComfyUI + FLUX.1-dev + Pepe LoRA

## Features
- ✅ Real Pepe memes (not generic frogs!)
- ✅ FLUX.1-dev base model
- ✅ Pepe LoRA integration
- ✅ Serverless (pay-per-use)
- ✅ Auto-scaling

## Usage
Deploy to RunPod Serverless via GitHub integration.

Test input:
```json
{
    "input": {
        "prompt": "pepe the frog wearing sunglasses",
        "lora_strength": 0.8
    }
}
```

## Expected Output
Base64 encoded perfect Pepe meme image.
```

## 🚀 Next Steps:

1. **Create your GitHub repo** with the name `pepe-worker-runpod`
2. **Add these 5 files** via GitHub web interface
3. **Go back to RunPod** and use GitHub integration
4. **Deploy and test** your perfect Pepe generator!

**Ready to create the GitHub repository?** 🐸
