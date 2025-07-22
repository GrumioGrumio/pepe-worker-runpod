import runpod
import os
import json
import base64
import requests
import time
import subprocess
import threading

def start_comfyui_server():
    """Start ComfyUI server in background"""
    try:
        print("🚀 Starting ComfyUI server...")
        # Start ComfyUI server
        process = subprocess.Popen([
            "python", "/app/comfyui/main.py", 
            "--listen", "0.0.0.0", 
            "--port", "8188",
            "--disable-auto-launch"
        ], cwd="/app/comfyui")
        
        # Give it time to start
        time.sleep(20)
        print("✅ ComfyUI server should be running")
        return process
    except Exception as e:
        print(f"❌ ComfyUI server error: {e}")
        return None

def create_simple_pepe_workflow(prompt):
    """Create a simple workflow for Pepe generation"""
    return {
        "3": {
            "inputs": {
                "seed": int(time.time()),
                "steps": 8,  # Fast generation
                "cfg": 1.0,  # Simple settings
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler"
        },
        "4": {
            "inputs": {
                "ckpt_name": "flux1-schnell.safetensors"  # We'll add this
            },
            "class_type": "CheckpointLoaderSimple"
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
                "text": f"{prompt}, pepe the frog, meme style",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "7": {
            "inputs": {
                "text": "blurry, low quality",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEDecode"
        },
        "9": {
            "inputs": {
                "filename_prefix": "pepe_generation",
                "images": ["8", 0]
            },
            "class_type": "SaveImage"
        }
    }

def handler(event):
    """REAL Pepe generation handler"""
    print("🐸 REAL Pepe Generation Worker!")
    
    try:
        input_data = event.get('input', {})
        prompt = input_data.get('prompt', 'pepe the frog')
        
        print(f"📝 Generating Pepe: {prompt}")
        
        # Check if we have everything needed
        lora_exists = os.path.exists("/app/comfyui/models/loras/pepe.safetensors")
        comfyui_exists = os.path.exists("/app/comfyui/main.py")
        
        if not lora_exists or not comfyui_exists:
            return {
