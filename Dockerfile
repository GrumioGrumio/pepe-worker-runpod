FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

# Install basic requirements only
RUN pip install runpod requests

# Copy handler
COPY rp_handler.py /app/

WORKDIR /app
CMD ["python", "rp_handler.py"]
