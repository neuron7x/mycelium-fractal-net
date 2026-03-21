FROM mcr.microsoft.com/devcontainers/python:1-3.13-bookworm
RUN curl -LsSf https://astral.sh/uv/install.sh | sh &&     ln -s /root/.local/bin/uv /usr/local/bin/uv &&     apt-get update && apt-get install -y --no-install-recommends graphviz &&     rm -rf /var/lib/apt/lists/*
WORKDIR /workspaces/mfn
