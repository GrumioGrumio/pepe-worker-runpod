FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch (CPU version)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements  
RUN pip install runpod requests pillow

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /app/comfyui
WORKDIR /app/comfyui
RUN pip install -r requirements.txt

# Create model directories (no LoRA download here)
RUN mkdir -p models/checkpoints models/loras models/vae

# Copy handler
COPY rp_handler.py /app/

WORKDIR /app
CMD ["python", "rp_handler.py"]
