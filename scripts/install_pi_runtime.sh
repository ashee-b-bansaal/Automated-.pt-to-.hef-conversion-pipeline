#!/usr/bin/env bash
set -euo pipefail
cat <<'EOF'
This installs the Raspberry Pi Hailo vision runtime packages.
Confirm the package for your hardware before continuing:
  Hailo-8 / Hailo-8L AI HAT+/AI Kit: hailo-all
  Hailo-10H AI HAT+ 2:              hailo-h10-all
EOF
PACKAGE="${1:-hailo-all}"
case "$PACKAGE" in hailo-all|hailo-h10-all) ;; *) echo "Unsupported package choice" >&2; exit 2;; esac
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y "$PACKAGE"
echo "Reboot, then run: hailortcli fw-control identify"
