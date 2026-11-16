#!/usr/bin/env bash
# Install only when explicitly invoked; dry-run never changes the host.
set -Eeuo pipefail
trap 'echo "autoloop installer failed at line ${LINENO}" >&2' ERR
usage() { echo "Usage: $0 [--dry-run] [--apply-wsl-conf] <config.json>"; }
dry_run=0
apply_conf=0
config=""
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --apply-wsl-conf) apply_conf=1 ;;
        --help|-h) usage; exit 0 ;;
        -*) usage >&2; exit 2 ;;
        *) if [[ -n "$config" ]]; then usage >&2; exit 2; fi; config=$1 ;;
    esac
    shift
done
[[ -n "$config" ]] || { usage >&2; exit 2; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
training_python=${HASAMI_TRAIN_PYTHON:-$repo_root/.venv/bin/python}
command -v "$training_python" >/dev/null
cd -- "$repo_root"
args=("$config")
(( dry_run == 0 )) || args+=(--dry-run)
(( apply_conf == 0 )) || args+=(--apply-wsl-conf)
"$training_python" "$script_dir/install.py" "${args[@]}"
