#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${KHQUANT_INSTALL_PYTHON:-python3.12}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  python_bin=python3
fi
exec "$python_bin" "$repo_root/install.py" "$@"
