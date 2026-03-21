#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="artifacts/bootstrap"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/bootstrap_$(date +%F_%H-%M-%S).log"
exec > >(tee -a "$LOG") 2>&1

PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_PROFILE="${INSTALL_PROFILE:-dev,security,reporting}"
INSTALL_ML="${INSTALL_ML:-1}"
INSTALL_ACCEL="${INSTALL_ACCEL:-0}"
TORCH_CHANNEL="${TORCH_CHANNEL:-cpu}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

rm -rf .venv
"$PYTHON_BIN" -m venv --copies .venv
source .venv/bin/activate

python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[${INSTALL_PROFILE}]"

if [[ "$INSTALL_ACCEL" == "1" ]]; then
  python -m pip install -e ".[accel]"
fi

if [[ "$INSTALL_ML" == "1" ]]; then
  if [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" && "$TORCH_CHANNEL" == "cpu" ]]; then
    python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0,<3.0.0"
  else
    python -m pip install "torch>=2.0.0,<3.0.0"
  fi
fi

python scripts/dev_doctor.py
python - <<'PY'
import sys
print('TORCH_READY=', end='')
try:
    import torch
    print(f'1 version={torch.__version__}')
except Exception as exc:
    print(f'0 error={exc}')
    sys.exit(0)
PY

echo
echo "BOOTSTRAP_OK=1"
echo "LOG=$LOG"
echo "VENV=$ROOT/.venv"
echo "ACTIVATE: source .venv/bin/activate"
