# MyceliumFractalNet v4.1 Docker Image
# Multi-stage build for minimal production image

# Build stage
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install --no-cache-dir --user -e .

# Production stage
FROM python:3.10-slim

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash mfnuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /home/mfnuser/.local

# Copy source code
COPY --chown=mfnuser:mfnuser . .

# Add .local/bin to PATH
ENV PATH=/home/mfnuser/.local/bin:$PATH

# Switch to non-root user
USER mfnuser

# Expose port for API
EXPOSE 8000

# Health check - use lightweight /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

# Default command: run API server
# For validation, explicitly override: docker run ... python mycelium_fractal_net_v4_1.py --mode validate --seed 42
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
