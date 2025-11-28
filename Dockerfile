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

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy source code
COPY . .

# Add .local/bin to PATH
ENV PATH=/root/.local/bin:$PATH

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from mycelium_fractal_net import run_validation; run_validation()" || exit 1

# Default command: run validation
CMD ["python", "mycelium_fractal_net_v4_1.py", "--mode", "validate", "--seed", "42"]
