# Use NVIDIA CUDA base image instead of python:3.10-slim
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python
RUN ln -s /usr/bin/python3.10 /usr/bin/python3

WORKDIR /app

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other requirements  
RUN pip install runpod requests pillow numpy opencv-python transformers accelerate safetensors

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/comfyui

WORKDIR /app/comfyui

# Install ComfyUI requirements
RUN pip install -r requirements.txt

# Install additional ComfyUI dependencies
RUN pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

# Create model directories
RUN mkdir -p models/checkpoints models/loras models/vae models/clip models/unet models/controlnet

# Set proper permissions
RUN chmod -R 755 /app

# Copy handler
COPY rp_handler.py /app/

WORKDIR /app

# Test GPU availability on container start
RUN python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" || echo "CUDA test will run at runtime"

CMD ["python", "rp_handler.py"]
