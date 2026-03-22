# MyceliumFractalNet v4.1 Docker Image
# Multi-stage build for minimal production image

# Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files for install
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ src/

# Install package (production deps only)
RUN pip install --no-cache-dir .

# Production stage
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source and configs
COPY src/ src/
COPY configs/ configs/

# Add non-root runtime user
RUN addgroup --system --gid 1000 mfn \
    && adduser --system --uid 1000 --ingroup mfn --home /app mfn \
    && chown -R mfn:mfn /app

ENV PATH=/usr/local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER mfn

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import mycelium_fractal_net; print('ok')" || exit 1

CMD ["mfn", "api", "--host", "0.0.0.0", "--port", "8000"]
