#!/usr/bin/env bash
# KHQuant native launcher for macOS/Linux.
# Usage:
#   ./khquant-native.sh dashboard              # start web dashboard
#   ./khquant-native.sh backtest --stocks ...  # run CLI command

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="$repo_root/.env"

# Always run from the repository root so relative paths in .env resolve
# correctly regardless of where the user invoked this script.
cd "$repo_root"

if [ -f "$env_file" ]; then
  # Parse the installer's simple KEY=VALUE format without evaluating shell
  # syntax. A path containing $, backticks, or $(...) must remain literal.
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|[[:space:]]*'#'*) continue ;;
      *=*) ;;
      *) continue ;;
    esac
    key="${line%%=*}"
    value="${line#*=}"
    key="${key#"${key%%[![:space:]]*}"}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ ! "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      continue
    fi
    case "$key" in
      KHQUANT_*|PYTHONPATH|TZ|TUSHARE_TOKEN|HITHINK_FINANCE_API_KEY) ;;
      *) continue ;;
    esac
    if [ "${#value}" -ge 2 ]; then
      first="${value:0:1}"
      last="${value: -1}"
      if { [ "$first" = '"' ] && [ "$last" = '"' ]; } || \
         { [ "$first" = "'" ] && [ "$last" = "'" ]; }; then
        value="${value:1:${#value}-2}"
      fi
    fi
    printf -v "$key" '%s' "$value"
    export "$key"
  done < "$env_file"
fi

# KHQUANT_PROJECT_ROOT, KHQUANT_VENV_DIR and PYTHONPATH are repo-relative.
# Business storage paths are relative to the resolved application root.
_normalize_path() {
  local base="${1}"
  local val="${2:-}"
  if [ -z "$val" ]; then
    echo "$val"
  elif [ "${val:0:1}" = "/" ]; then
    echo "$val"
  else
    echo "$base/$val"
  fi
}

# Project and business paths are normalized once by my_strategy.runtime_env.
# Keeping their raw values here also preserves the legacy no-schema heuristic.

# Determine venv directory
if [ -n "${KHQUANT_VENV_DIR:-}" ]; then
  venv_dir="$(_normalize_path "$repo_root" "$KHQUANT_VENV_DIR")"
else
  venv_dir="$repo_root/.venv"
fi

if [ ! -f "$venv_dir/bin/python" ]; then
  echo "Python interpreter missing in $venv_dir" >&2
  exit 1
fi

PYTHONPATH="$(_normalize_path "$repo_root" "${PYTHONPATH:-./app}")"
export PYTHONPATH
export KHQUANT_DASHBOARD_PORT="${KHQUANT_DASHBOARD_PORT:-8124}"

if [ "${1:-}" = "dashboard" ]; then
  shift || true
  port="${KHQUANT_DASHBOARD_PORT}"
  echo "Starting KHQuant dashboard on http://localhost:$port ..."
  exec "$venv_dir/bin/python" -m my_strategy.cli dashboard --port "$port" "$@"
fi

exec "$venv_dir/bin/python" -m my_strategy.cli "$@"
