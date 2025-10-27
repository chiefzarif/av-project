# VisionGuard MLOps - Production Dockerfile with GPU Support
# Multi-stage build for optimized image size
# Use CUDA base image for GPU acceleration

# Stage 1: Build stage with dependencies
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as builder

# Install Python 3.9
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    python3-dev \
    && ln -s /usr/bin/python3.9 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage with GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python 3.9
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    && ln -s /usr/bin/python3.9 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python packages from builder (before creating user)
COPY --from=builder /usr/local /usr/local

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy application code
COPY app/ ./app/

# Copy model files if they exist (optional - will be downloaded on first run if missing)
# Note: Model files will be auto-downloaded by Ultralytics on first use if not present

# Create directories
RUN mkdir -p /app/coco /app/annotations /app/models

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH=/usr/local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
