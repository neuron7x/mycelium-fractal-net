# Contributing to MyceliumFractalNet

Thank you for your interest in contributing to MyceliumFractalNet! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to example@example.com.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your changes
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip and virtualenv

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/mycelium-fractal-net.git
cd mycelium-fractal-net

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
pytest --version
ruff --version
```

### Environment Configuration

For local development, set:

```bash
export MFN_ENV=dev
export MFN_API_KEY=dev-key-for-testing
```

## Coding Standards

### Python Style

We follow PEP 8 with some modifications:

- **Line Length**: 100 characters (configured in ruff)
- **Imports**: Use `isort` for import sorting
- **Type Hints**: Required for all public functions and methods
- **Docstrings**: Google-style docstrings for all public APIs

### Tools

We use the following tools (enforced in CI):

- **ruff**: Fast Python linter (replaces flake8, isort checks)
- **mypy**: Static type checking
- **black**: Code formatting (optional, for consistency)

Run all checks:

```bash
# Lint
ruff check .

# Type check
mypy src/mycelium_fractal_net

# Run tests
pytest

# Run security tests
pytest tests/security/ -v
```

### Code Structure

```
src/mycelium_fractal_net/
├── core/           # Core simulation engine
├── crypto/         # Cryptographic operations (production)
├── security/       # Security utilities (deprecated encryption)
├── integration/    # API, middleware, adapters
├── types/          # Type definitions and schemas
├── numerics/       # Numerical methods
├── pipelines/      # Validation and federated pipelines
├── analytics/      # Runtime analytics (included in package)
└── experiments/    # Experimental features (dev-only)
```

### Naming Conventions

- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: Leading underscore `_private_method`

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=mycelium_fractal_net --cov-report=html

# Specific test file
pytest tests/security/test_encryption.py -v

# Specific test
pytest tests/security/test_encryption.py::test_encrypt_decrypt -v
```

### Test Structure

- Unit tests in `tests/` mirror source structure
- Integration tests in `tests/integration/`
- Security tests in `tests/security/`
- Performance tests in `tests/perf/`

### Writing Tests

```python
import pytest
from mycelium_fractal_net.crypto.symmetric import AESGCMCipher

def test_encryption_roundtrip():
    """Test that encryption and decryption work correctly."""
    cipher = AESGCMCipher()
    plaintext = b"sensitive data"
    
    ciphertext = cipher.encrypt(plaintext)
    decrypted = cipher.decrypt(ciphertext)
    
    assert decrypted == plaintext
```

### Test Requirements

- All new features must include tests
- Bug fixes should include regression tests
- Aim for >80% code coverage
- Security-critical code requires security tests

## Pull Request Process

### Before Submitting

1. **Create an issue** describing the bug or feature (if not already exists)
2. **Create a branch** from `main` or `develop`:
   ```bash
   git checkout -b feature/my-feature
   git checkout -b fix/my-bugfix
   ```
3. **Make your changes** following coding standards
4. **Add tests** for your changes
5. **Run all checks**:
   ```bash
   ruff check .
   mypy src/mycelium_fractal_net
   pytest --cov=mycelium_fractal_net
   ```
6. **Update documentation** if needed
7. **Update CHANGELOG.md** under `[Unreleased]` section

### Submitting PR

1. Push your branch to your fork
2. Create a pull request against `main` or `develop`
3. Fill out the PR template
4. Wait for CI checks to pass
5. Request review from maintainers

### PR Title Format

Use conventional commits format:

- `feat: add new feature`
- `fix: resolve bug in X`
- `docs: update contributing guide`
- `test: add tests for Y`
- `refactor: improve Z`
- `perf: optimize A`
- `security: fix vulnerability in B`
- `chore: update dependencies`

### Review Process

- At least one maintainer approval required
- All CI checks must pass
- No merge conflicts
- Code review feedback addressed

## Reporting Bugs

### Before Reporting

- Check existing issues to avoid duplicates
- Try latest version to see if bug is fixed
- Gather minimal reproduction steps

### Bug Report Template

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:
1. ...
2. ...

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.12]
- Package version: [e.g., 4.1.0]

**Additional context**
Any other relevant information.
```

## Suggesting Enhancements

We welcome feature suggestions! Please:

1. Check if feature already requested
2. Clearly describe the use case
3. Explain why it benefits users
4. Consider backwards compatibility
5. Propose API design if applicable

## Security Issues

**DO NOT** report security vulnerabilities via public issues.

See [SECURITY.md](SECURITY.md) for responsible disclosure process.

## Questions?

- Open a discussion on GitHub Discussions
- Check documentation in `docs/`
- Review existing issues and PRs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
