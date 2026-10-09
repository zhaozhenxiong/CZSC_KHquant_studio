#!/usr/bin/env bash
set -euo pipefail
cd /app
exec python -m my_strategy.cli "${@:-doctor}"
