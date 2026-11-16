#!/usr/bin/env bash
# Preserve checkpoints, logs, configs, and linger shared by other services.
set -Eeuo pipefail
trap 'echo "autoloop uninstall failed at line ${LINENO}" >&2' ERR
if [[ ${1:-} == --help || $# != 1 ]]; then
    echo "Usage: $0 <loop-id> (disables the service; preserves all training data)"
    exit 0
fi
loop_id=$1
[[ "$loop_id" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] || exit 2
systemctl --user disable --now "hasami-autoloop@${loop_id}.service"
echo "Service disabled. Configs, shared template, linger and training data were preserved."
