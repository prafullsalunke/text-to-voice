# Use official PyTorch image with CUDA 12.1 — includes torch pre-installed
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install system deps needed by soundfile (libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install app dependencies (torch already in base image; skip it here)
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi>=0.115.0 \
    "uvicorn[standard]>=0.32.0" \
    "pydantic-settings>=2.7.0" \
    "soundfile>=0.12.1" \
    "numpy>=1.26.0" \
    "httpx>=0.27.0"

# Install voxcpm (HuggingFace model library)
RUN pip install --no-cache-dir voxcpm

# Copy app source
COPY config.py synthesizer.py main.py ./

# Model weights are downloaded at runtime from HuggingFace.
# Mount a cache volume to avoid re-downloading on container restart.
ENV HF_HOME=/cache/huggingface

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
