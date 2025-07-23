# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3.10-distutils \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python3.10 -m pip install --upgrade pip

# Create app directory
WORKDIR /app

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install base requirements  
RUN pip install runpod requests pillow numpy opencv-python transformers accelerate safetensors

# Clone ComfyUI to /app/comfyui (consistent with your handler)
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/comfyui

# Install ComfyUI requirements
WORKDIR /app/comfyui
RUN pip install -r requirements.txt

# Install additional ComfyUI dependencies
RUN pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121

# Create model directories with proper structure
RUN mkdir -p models/checkpoints models/loras models/vae models/clip models/unet models/controlnet models/embeddings

# Create output directory
RUN mkdir -p output

# Set proper permissions
RUN chmod -R 755 /app

# Copy the RunPod handler to ComfyUI directory (where it will be executed)
COPY rp_handler.py /app/comfyui/
COPY start_worker.py /app/

# Go back to app directory for startup
WORKDIR /app

# Test GPU and PyTorch installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')" || echo "CUDA test will run at runtime"

# Start the worker
CMD ["python", "start_worker.py"]
