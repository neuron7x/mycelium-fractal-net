# MyceliumFractalNet v4.5 Docker Image
# Multi-stage build for minimal production image

ARG PYTHON_VERSION=3.12

# Build stage
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Production stage
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ src/
COPY configs/ configs/

RUN addgroup --system --gid 1000 mfn \
    && adduser --system --uid 1000 --ingroup mfn --home /app mfn \
    && chown -R mfn:mfn /app

ENV PATH=/usr/local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.title="MyceliumFractalNet" \
      org.opencontainers.image.version="4.5.0" \
      org.opencontainers.image.source="https://github.com/neuron7x/mycelium-fractal-net" \
      org.opencontainers.image.description="Morphogenetic field intelligence engine"

USER mfn

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import mycelium_fractal_net; print('ok')" || exit 1

CMD ["mfn", "api", "--host", "0.0.0.0", "--port", "8000"]
