# Makefile for MyceliumFractalNet gRPC

.PHONY: help proto-gen proto-clean test-grpc

help:
	@echo "MFN gRPC Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  proto-gen     - Generate Python stubs from .proto files"
	@echo "  proto-clean   - Remove generated proto files"
	@echo "  test-grpc     - Run gRPC tests"
	@echo "  test-all      - Run all tests"

proto-gen:
	@echo "Generating Python stubs from proto files..."
	python -m grpc_tools.protoc \
		-I. \
		--python_out=. \
		--grpc_python_out=. \
		grpc/proto/mfn.proto
	@echo "Moving generated files to src/mycelium_fractal_net/grpc/..."
	mv grpc/proto/mfn_pb2.py src/mycelium_fractal_net/grpc/
	mv grpc/proto/mfn_pb2_grpc.py src/mycelium_fractal_net/grpc/
	@echo "Proto generation complete!"

proto-clean:
	@echo "Cleaning generated proto files..."
	rm -f src/mycelium_fractal_net/grpc/mfn_pb2.py
	rm -f src/mycelium_fractal_net/grpc/mfn_pb2_grpc.py
	@echo "Clean complete!"

test-grpc:
	@echo "Running gRPC tests..."
	python -m pytest tests/test_grpc/ -v

test-all:
	@echo "Running all tests..."
	python -m pytest tests/ -v
