# Dockerfile for LegalTextDecoder deep learning project
# GPU-enabled image with Python 3.12 and uv package manager

# Use NVIDIA CUDA base image with Python 3.12 for GPU support
FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

# Install Python 3.12 and system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies using uv
# Use --system flag to install into the system Python instead of creating a venv
RUN uv sync --frozen --no-dev --no-cache

# Copy source code and configuration
COPY src/ src/
COPY notebook/ notebook/
COPY config.yaml .
COPY run.sh .

# Make run.sh executable
RUN chmod +x run.sh

# Create directories for data and output (to be mounted as volumes)
RUN mkdir -p /app/_data /app/data /app/output /app/logs /app/models /app/media

# Set environment variables for better Python behavior in containers
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the entrypoint to run the training pipeline by default
# You can override this with: docker run ... python -m src.04_inference
CMD ["bash", "/app/run.sh"]
