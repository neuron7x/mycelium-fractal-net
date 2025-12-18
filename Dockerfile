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

# Install dependencies system-wide to allow non-root runtime user
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package
RUN pip install --no-cache-dir -e .

# Production stage
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local /usr/local

# Copy source code
COPY . .

# Add non-root runtime user
RUN addgroup --system --gid 1000 mfn \
    && adduser --system --uid 1000 --ingroup mfn --home /app mfn \
    && chown -R mfn:mfn /app

# Ensure system Python/bin are available
ENV PATH=/usr/local/bin:/usr/local/sbin:$PATH

# Drop privileges for runtime
USER mfn

# Expose port for API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from mycelium_fractal_net import run_validation; run_validation()" || exit 1

# Default command: run validation
CMD ["python", "mycelium_fractal_net_v4_1.py", "--mode", "validate", "--seed", "42"]
