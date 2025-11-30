"""
Test layer boundaries for MyceliumFractalNet architecture.

This module validates that the architectural layer boundaries are maintained:
- core/ modules should NOT import infrastructure (FastAPI, uvicorn, etc.)
- integration/ should NOT contain business logic
- Clear dependency direction: core <- integration <- api/cli

Reference: docs/ARCHITECTURE.md
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Set

import pytest

# Infrastructure packages that core should NOT import (top-level names only)
# Note: torch is allowed (it's numerical computing), but torch.distributed would be disallowed
# However, we check top-level imports, so we focus on HTTP/web infrastructure
INFRASTRUCTURE_PACKAGES = {
    "fastapi",
    "uvicorn",
    "starlette",
    "httpx",
    "aiohttp",
    "flask",
    "django",
}

# Packages that are allowed in core (pure numerical/math)
ALLOWED_CORE_PACKAGES = {
    "numpy",
    "torch",  # torch is allowed for numerical computing
    "sympy",
    "scipy",
    "math",
    "dataclasses",
    "typing",
    "collections",
    "functools",
    "enum",
    "abc",
}


def get_imports_from_module(module_path: Path) -> Set[str]:
    """Extract import statements from a Python module file.
    
    Parameters
    ----------
    module_path : Path
        Path to the Python file.
        
    Returns
    -------
    Set[str]
        Set of imported module names (top-level only).
    """
    if not module_path.exists():
        return set()
    
    try:
        with open(module_path, "r") as f:
            source = f.read()
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    
    imports: Set[str] = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Get top-level package
                top_level = alias.name.split(".")[0]
                imports.add(top_level)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Get top-level package
                top_level = node.module.split(".")[0]
                imports.add(top_level)
    
    return imports


def get_all_imports_from_package(package_path: Path) -> dict[str, Set[str]]:
    """Get all imports from all Python files in a package.
    
    Parameters
    ----------
    package_path : Path
        Path to the package directory.
        
    Returns
    -------
    dict[str, Set[str]]
        Dictionary mapping file paths to their imports.
    """
    result = {}
    if not package_path.exists():
        return result
        
    for py_file in package_path.rglob("*.py"):
        imports = get_imports_from_module(py_file)
        result[str(py_file.relative_to(package_path))] = imports
    
    return result


class TestCoreLayerBoundaries:
    """Test that core/ layer doesn't import infrastructure packages."""
    
    @pytest.fixture
    def core_path(self) -> Path:
        """Get path to core package."""
        return Path(__file__).parent.parent / "src" / "mycelium_fractal_net" / "core"
    
    def test_core_does_not_import_fastapi(self, core_path: Path) -> None:
        """Test that core/ does not import FastAPI."""
        all_imports = get_all_imports_from_package(core_path)
        
        for file_path, imports in all_imports.items():
            assert "fastapi" not in imports, \
                f"core/{file_path} should not import fastapi"
    
    def test_core_does_not_import_uvicorn(self, core_path: Path) -> None:
        """Test that core/ does not import uvicorn."""
        all_imports = get_all_imports_from_package(core_path)
        
        for file_path, imports in all_imports.items():
            assert "uvicorn" not in imports, \
                f"core/{file_path} should not import uvicorn"
    
    def test_core_does_not_import_starlette(self, core_path: Path) -> None:
        """Test that core/ does not import starlette."""
        all_imports = get_all_imports_from_package(core_path)
        
        for file_path, imports in all_imports.items():
            assert "starlette" not in imports, \
                f"core/{file_path} should not import starlette"
    
    def test_core_infrastructure_free(self, core_path: Path) -> None:
        """Test that core/ is free of all infrastructure packages."""
        all_imports = get_all_imports_from_package(core_path)
        
        violations = []
        for file_path, imports in all_imports.items():
            for infra in INFRASTRUCTURE_PACKAGES:
                top_level = infra.split(".")[0]
                if top_level in imports:
                    violations.append(f"core/{file_path}: imports {infra}")
        
        if violations:
            pytest.fail(
                "Core layer should not import infrastructure packages:\n"
                + "\n".join(violations)
            )
    
    def test_core_does_not_import_integration(self, core_path: Path) -> None:
        """Test that core/ does not import from integration/ (no reverse dependency)."""
        all_imports = get_all_imports_from_package(core_path)
        
        for file_path, imports in all_imports.items():
            # Check for imports that look like they might be from integration
            # This is a heuristic check
            for imp in imports:
                assert "integration" not in imp.lower(), \
                    f"core/{file_path} should not import from integration layer"


class TestIntegrationLayerBoundaries:
    """Test integration layer constraints."""
    
    @pytest.fixture
    def integration_path(self) -> Path:
        """Get path to integration package."""
        return Path(__file__).parent.parent / "src" / "mycelium_fractal_net" / "integration"
    
    def test_integration_exists(self, integration_path: Path) -> None:
        """Test that integration/ package exists."""
        assert integration_path.exists(), "integration/ package should exist"
    
    def test_integration_has_schemas(self, integration_path: Path) -> None:
        """Test that integration/ has schemas module."""
        schemas_path = integration_path / "schemas.py"
        assert schemas_path.exists(), "integration/schemas.py should exist"
    
    def test_integration_has_adapters(self, integration_path: Path) -> None:
        """Test that integration/ has adapters module."""
        adapters_path = integration_path / "adapters.py"
        assert adapters_path.exists(), "integration/adapters.py should exist"


class TestAPILayerBoundaries:
    """Test that API layer follows proper dependency direction."""
    
    @pytest.fixture
    def api_path(self) -> Path:
        """Get path to api.py."""
        return Path(__file__).parent.parent / "api.py"
    
    def test_api_imports_from_integration(self, api_path: Path) -> None:
        """Test that api.py imports from integration layer."""
        imports = get_imports_from_module(api_path)
        
        # api.py should import from integration or mycelium_fractal_net
        has_integration_import = any(
            "mycelium_fractal_net" in imp or "integration" in imp 
            for imp in imports
        )
        assert has_integration_import or "mycelium_fractal_net" in imports, \
            "api.py should import from mycelium_fractal_net.integration"


class TestNoCyclicImports:
    """Test that there are no cyclic imports between layers."""
    
    def test_core_imports_work(self) -> None:
        """Test that core modules can be imported without cycles."""
        # Force fresh import
        modules_to_remove = [m for m in sys.modules if "mycelium_fractal_net.core" in m]
        for m in modules_to_remove:
            del sys.modules[m]
        
        # This should not raise ImportError
        from mycelium_fractal_net.core import (
            FractalGrowthEngine,
            MembraneEngine,
            ReactionDiffusionEngine,
        )
        
        assert MembraneEngine is not None
        assert ReactionDiffusionEngine is not None
        assert FractalGrowthEngine is not None
    
    def test_integration_imports_work(self) -> None:
        """Test that integration modules can be imported without cycles."""
        from mycelium_fractal_net.integration import (
            ServiceContext,
            ValidateRequest,
            ValidateResponse,
        )
        
        assert ServiceContext is not None
        assert ValidateRequest is not None
        assert ValidateResponse is not None
    
    def test_main_package_imports_work(self) -> None:
        """Test that main package can be imported without cycles."""
        import mycelium_fractal_net
        
        assert mycelium_fractal_net.__version__ == "4.1.0"


class TestDomainModuleStructure:
    """Test that domain modules in core/ have expected structure."""
    
    @pytest.fixture
    def core_path(self) -> Path:
        """Get path to core package."""
        return Path(__file__).parent.parent / "src" / "mycelium_fractal_net" / "core"
    
    def test_nernst_module_exists(self, core_path: Path) -> None:
        """Test nernst.py module exists."""
        assert (core_path / "nernst.py").exists()
    
    def test_turing_module_exists(self, core_path: Path) -> None:
        """Test turing.py module exists."""
        assert (core_path / "turing.py").exists()
    
    def test_fractal_module_exists(self, core_path: Path) -> None:
        """Test fractal.py module exists."""
        assert (core_path / "fractal.py").exists()
    
    def test_stdp_module_exists(self, core_path: Path) -> None:
        """Test stdp.py module exists."""
        assert (core_path / "stdp.py").exists()
    
    def test_federated_module_exists(self, core_path: Path) -> None:
        """Test federated.py module exists."""
        assert (core_path / "federated.py").exists()
    
    def test_stability_module_exists(self, core_path: Path) -> None:
        """Test stability.py module exists."""
        assert (core_path / "stability.py").exists()
