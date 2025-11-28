#!/usr/bin/env python3
"""
MyceliumFractalNet v4.1 - One-Click Validation Script

Run this script to validate the installation and functionality of
MyceliumFractalNet in approximately 30 seconds.

Usage:
    python validate.py [config_path]

Examples:
    python validate.py                    # Use default config
    python validate.py configs/small.json # Use small config
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mfn import validate_model


def main():
    """Run validation with optional config path."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    success = validate_model(config_path)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
