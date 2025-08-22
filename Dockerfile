# Dockerfile for Teknofest CT Stroke Project

# 1. Base image with CUDA 11.8
FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn8-runtime

# 2. Set working directory
WORKDIR /app

# 3. Copy requirements file and install dependencies
COPY requirements_cuda.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements_cuda.txt

# 4. Copy the whole project
COPY . .

# 5. Hugging Face cache directory
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# 6. Default command
CMD ["python", "notebook/run_demo.py"]
