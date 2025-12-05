.PHONY: help install install-dev test test-fast lint format clean proto docker k8s-apply grpc-server api-server

help:  ## Show this help message
@echo "MyceliumFractalNet v4.1 - Available Commands:"
@echo ""
@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package and dependencies
pip install -e .

install-dev:  ## Install package with dev dependencies
pip install -e ".[dev]"

test:  ## Run all tests with coverage
pytest --cov=mycelium_fractal_net --cov-report=term-missing -v

test-fast:  ## Run smoke tests only (fast)
pytest tests/smoke/ -q

test-grpc:  ## Run gRPC integration tests
pytest tests/grpc/ -v

test-integration:  ## Run integration tests
pytest tests/integration/ -v

lint:  ## Run linters (ruff + mypy)
ruff check .
mypy src/mycelium_fractal_net

format:  ## Format code with black and isort
black src/ tests/
isort src/ tests/

proto:  ## Generate protobuf code from protos/
./scripts/generate_proto.sh

clean:  ## Clean build artifacts and cache
rm -rf build/ dist/ *.egg-info
rm -rf .pytest_cache .coverage htmlcov
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

docker:  ## Build Docker image
docker build -t mycelium-fractal-net:v4.1 .

docker-run:  ## Run Docker container
docker run -p 8000:8000 -p 50051:50051 mycelium-fractal-net:v4.1

k8s-apply:  ## Apply Kubernetes configuration
kubectl apply -f k8s.yaml

k8s-delete:  ## Delete Kubernetes resources
kubectl delete -f k8s.yaml

grpc-server:  ## Start gRPC server
python grpc_server.py --port 50051

api-server:  ## Start REST API server
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

validate:  ## Run validation cycle
python mycelium_fractal_net_v4_1.py --mode validate --seed 42 --epochs 5

benchmark:  ## Run benchmarks
python benchmarks/benchmark_core.py
python benchmarks/benchmark_scalability.py

load-test:  ## Run Locust load tests
locust -f load_tests/locustfile.py --headless -u 10 -r 2 -t 30s --host http://localhost:8000

security-scan:  ## Run security scans
bandit -r src/ -ll
pip-audit --desc on

all: install-dev lint test  ## Install, lint, and test

.DEFAULT_GOAL := help
