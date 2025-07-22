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
