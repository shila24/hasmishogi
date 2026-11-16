#!/usr/bin/env bash
# Restart a single training process; run from the repository root.
set -Eeuo pipefail

if [[ $# -eq 0 || ${1:-} == --help ]]; then
    echo "Usage: scripts/train_supervisor.sh <config.json> [train CLI args]"
    echo "Run from the repository root. Uses .venv/bin/python and --resume auto."
    exit 0
fi

training_config=$1
shift
training_python="${HASAMI_TRAIN_PYTHON:-.venv/bin/python}"
if ! command -v "${training_python}" >/dev/null 2>&1; then
    echo "Training interpreter not found: ${training_python}" >&2
    exit 1
fi

child_pid=""
stop_requested=0
forward_stop() {
    stop_requested=1
    if [[ -n ${child_pid} ]]; then
        # The child may have exited immediately before the signal arrived.
        kill -TERM "${child_pid}" 2>/dev/null || true
    fi
}
trap forward_stop TERM INT
trap 'echo "Supervisor error at line ${LINENO}" >&2' ERR

# Reuse CLI resolution so --set, --run-id and --output-dir select the same run.
run_dir=$("${training_python}" - "${training_config}" "$@" <<'PY'
import sys
from pathlib import Path
from hasami_learn.cli import build_parser, resolve_train_request

args = build_parser().parse_args(["train", "--config", sys.argv[1], *sys.argv[2:], "--resume", "auto"])
config, _ = resolve_train_request(args)
print(Path(config.run.output_dir) / config.run.run_id)
PY
)
mkdir -p -- "${run_dir}"
exec >>"${run_dir}/supervisor.log" 2>&1

backoff=5
while (( ! stop_requested )); do
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') starting training"
    "${training_python}" -m hasami_learn.cli train --config "${training_config}" "$@" --resume auto &
    child_pid=$!
    # Handle a signal delivered between the loop condition and PID assignment.
    if (( stop_requested )); then
        forward_stop
    fi
    child_status=0
    wait "${child_pid}" || child_status=$?
    if (( stop_requested )); then
        # A trapped signal interrupts wait; keep waiting for the graceful checkpoint.
        while kill -0 "${child_pid}" 2>/dev/null; do
            wait "${child_pid}" || true
        done
        echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') supervisor stopped"
        exit 0
    fi
    child_pid=""
    if (( child_status == 0 )) && "${training_python}" - "${run_dir}" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
try:
    state = json.loads((run / "trainer_state.json").read_text())
    config = json.loads((run / "config.json").read_text())
    complete = int(state["iteration"]) >= int(config["run"]["iterations"])
except (OSError, ValueError, KeyError, TypeError):
    complete = False
sys.exit(0 if complete else 1)
PY
    then
        echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') training complete"
        exit 0
    fi
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') exit=${child_status}; retry in ${backoff}s"
    sleep "${backoff}" &
    child_pid=$!
    if (( stop_requested )); then
        forward_stop
    fi
    wait "${child_pid}" || true
    child_pid=""
    backoff=$((backoff * 2))
    if (( backoff > 300 )); then
        backoff=300
    fi
done
