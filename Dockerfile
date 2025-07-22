# Use Python base with CUDA support
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python packages
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install runpod requests pillow websocket-client

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/comfyui
WORKDIR /app/comfyui

# Install ComfyUI requirements
RUN pip3 install -r requirements.txt

# Create model directories
RUN mkdir -p models/checkpoints models/loras models/vae

# Download Pepe LoRA
RUN cd models/loras && \
    wget -O pepe.safetensors "https://huggingface.co/openfree/pepe/resolve/main/pepe.safetensors"

# Copy our handler files
COPY rp_handler.py /app/
COPY start_worker.py /app/

# Expose ports
EXPOSE 8188 8080

# Start the worker
CMD ["python3", "/app/start_worker.py"]
