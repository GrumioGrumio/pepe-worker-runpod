# Use simple Python base
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install PyTorch CPU version (lighter for initial testing)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install runpod requests pillow websocket-client

# For now, create a simple test handler
COPY rp_handler.py /app/
COPY test_input.json /app/

# Simple test version first
CMD ["python", "/app/rp_handler.py"]
