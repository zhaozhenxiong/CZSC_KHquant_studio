#!/usr/bin/env bash
# Install or remove KHQuant LaunchAgent services on macOS.
# Usage:
#   ./install.sh              # install and load services
#   ./install.sh --uninstall  # unload and remove services

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
launch_agents="$HOME/Library/LaunchAgents"

log_dir="$repo_root/.khquant/logs"
if [ ! -f "$repo_root/khquant-native.sh" ] || [ ! -f "$repo_root/app/AGENTS.md" ]; then
    echo "Unable to locate the KHQuant repository from $BASH_SOURCE" >&2
    exit 1
fi

mkdir -p "$launch_agents" "$log_dir"

plist_dashboard="$launch_agents/com.khquant.dashboard.plist"
plist_update="$launch_agents/com.khquant.daily-update.plist"

unload_if_loaded() {
    local plist="$1"
    local label
    label="$(basename "$plist" .plist)"
    if launchctl list "$label" >/dev/null 2>&1; then
        launchctl unload -w "$plist" 2>/dev/null || launchctl bootout "gui/$(id -u)" "$plist" 2>/dev/null || true
    fi
}

render_plist() {
    local source="$1"
    local destination="$2"
    python3 - "$source" "$destination" "$repo_root" <<'PY'
from __future__ import annotations

import html
import os
from pathlib import Path
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
repo_root = html.escape(sys.argv[3], quote=True)
rendered = source.read_text(encoding="utf-8").replace("__REPO_ROOT__", repo_root)
temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
temporary.write_text(rendered, encoding="utf-8")
os.replace(temporary, destination)
PY
}

install() {
    echo "Installing KHQuant LaunchAgents..."

    unload_if_loaded "$plist_dashboard"
    unload_if_loaded "$plist_update"

    render_plist "$repo_root/services/macos/com.khquant.dashboard.plist" "$plist_dashboard"
    render_plist "$repo_root/services/macos/com.khquant.daily-update.plist" "$plist_update"

    launchctl load -w "$plist_dashboard" 2>/dev/null || launchctl bootstrap "gui/$(id -u)" "$plist_dashboard"
    launchctl load -w "$plist_update" 2>/dev/null || launchctl bootstrap "gui/$(id -u)" "$plist_update"

    echo "Installed:"
    echo "  $plist_dashboard"
    echo "  $plist_update"
    echo "Logs: $log_dir"
}

uninstall() {
    echo "Removing KHQuant LaunchAgents..."
    unload_if_loaded "$plist_dashboard"
    unload_if_loaded "$plist_update"
    rm -f "$plist_dashboard" "$plist_update"
    echo "Removed."
}

case "${1:-}" in
    --uninstall|-u) uninstall ;;
    *) install ;;
esac
