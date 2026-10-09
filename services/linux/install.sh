#!/usr/bin/env bash
# Install or remove KHQuant systemd user services on Linux.
# Usage:
#   ./install.sh              # install, enable and start services
#   ./install.sh --uninstall  # stop, disable and remove services

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
unit_dir="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

if [ ! -f "$repo_root/khquant-native.sh" ] || [ ! -f "$repo_root/app/AGENTS.md" ]; then
    echo "Unable to locate the KHQuant repository from $BASH_SOURCE" >&2
    exit 1
fi

# Ensure the systemd user instance is available.
if ! systemctl --user status >/dev/null 2>&1; then
    echo "systemd user instance is not available." >&2
    echo "Install systemd and ensure \$XDG_RUNTIME_DIR is set." >&2
    exit 1
fi

mkdir -p "$unit_dir"

install_unit() {
    local name="$1"
    local unit_root sed_root temporary
    unit_root="${repo_root//\\/\\\\}"
    unit_root="${unit_root//\"/\\\"}"
    unit_root="${unit_root//%/%%}"
    sed_root="$(printf '%s' "$unit_root" | sed 's/[\\&|]/\\&/g')"
    temporary="$(mktemp "$unit_dir/.${name}.XXXXXX")"
    sed "s|__REPO_ROOT__|$sed_root|g" "$repo_root/services/linux/$name" > "$temporary"
    mv -f "$temporary" "$unit_dir/$name"
}

install() {
    echo "Installing KHQuant systemd user units to $unit_dir ..."
    install_unit khquant-dashboard.service
    install_unit khquant-daily-update.service
    install_unit khquant-daily-update.timer

    systemctl --user daemon-reload
    systemctl --user enable --now khquant-dashboard.service
    systemctl --user enable --now khquant-daily-update.timer

    echo "Installed and started. Check status with:"
    echo "  systemctl --user status khquant-dashboard.service"
    echo "  systemctl --user list-timers khquant-daily-update.timer"
}

uninstall() {
    echo "Removing KHQuant systemd user units..."
    systemctl --user stop khquant-dashboard.service khquant-daily-update.timer || true
    systemctl --user disable khquant-dashboard.service khquant-daily-update.timer || true
    rm -f "$unit_dir"/khquant-dashboard.service "$unit_dir"/khquant-daily-update.service "$unit_dir"/khquant-daily-update.timer
    systemctl --user daemon-reload
    echo "Removed."
}

case "${1:-}" in
    --uninstall|-u) uninstall ;;
    *) install ;;
esac
