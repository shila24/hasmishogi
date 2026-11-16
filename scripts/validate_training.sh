#!/usr/bin/env bash
# Reproducible Linux/WSL2 validation: build, tests, riichi parity, benchmarks, CUDA smoke runs, power-loss resume.
# Usage (repository root, after the setup in docs/TRAINING_ja.md):
#   bash scripts/validate_training.sh --expect-commit <sha> [--quick] [--skip step,...] [--only step,...]
# <sha> is the reviewed public commit (git rev-parse HEAD); checkout fails without it.
# Writes validation-results/<UTC timestamp>/ (logs, steps.tsv, env.json) and, via scripts/validation_summary.py,
# sanitized summary.md / summary.json plus a .tar.gz to send back. It never starts production training.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV_PY="${VENV_PY:-$ROOT/.venv/bin/python}"
SYSTEM_PY="${SYSTEM_PY:-/usr/bin/python3}"
QUICK=0
SKIP=","
ONLY=""
EXPECT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --expect-commit) EXPECT="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --skip) SKIP=",$2,"; shift 2 ;;
    --only) ONLY=",$2,"; shift 2 ;;
    -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for tool in "$VENV_PY" "$SYSTEM_PY"; do
  [[ -x "$tool" ]] || { echo "missing interpreter: $tool (see docs/TRAINING_ja.md)" >&2; exit 2; }
done
command -v cmake >/dev/null || { echo "cmake not found" >&2; exit 2; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/validation-results/$STAMP"
mkdir -p "$OUT/logs"
: > "$OUT/steps.tsv"
SCRATCH="$(mktemp -d -t hasami-validation-XXXXXX)"
trap 'rm -rf "$SCRATCH"' EXIT

wanted() {
  local name="$1"
  [[ "$SKIP" == *",$name,"* ]] && return 1
  [[ -n "$ONLY" && "$ONLY" != *",$name,"* ]] && return 1
  return 0
}

# step <name> <command...>: run, log, record "<name>\t<ran|skipped>\t<exit>\t<seconds>".
step() {
  local name="$1"; shift
  if ! wanted "$name"; then
    printf '%s\tskipped\t0\t0\n' "$name" >> "$OUT/steps.tsv"; return 0
  fi
  echo "==> $name"
  local start end rc
  start=$(date +%s)
  "$@" > "$OUT/logs/$name.log" 2>&1
  rc=$?
  end=$(date +%s)
  printf '%s\tran\t%d\t%d\n' "$name" "$rc" "$((end - start))" >> "$OUT/steps.tsv"
  if [[ $rc -eq 0 ]]; then echo "    ok ($((end - start)) s)"; else echo "    FAILED (exit $rc, see logs/$name.log)"; fi
}

check_checkout() {  # the checkout is exactly the verified bundle commit, with no tracked modifications
  local head
  head="$(git rev-parse HEAD)"
  echo "HEAD $head"
  echo "expected ${EXPECT:-<not given>}"
  git status --porcelain --untracked-files=no
  [[ -n "$EXPECT" && "$head" == "$EXPECT"* && -z "$(git status --porcelain --untracked-files=no)" ]]
}

train_smoke() {  # train_smoke <config> <run-id> [--set ...]: one short run on CUDA into the scratch directory
  local config="$1" run_id="$2"; shift 2
  "$VENV_PY" -m hasami_learn.cli train --config "$config" --output-dir "$SCRATCH/runs" --run-id "$run_id" \
    --set device.device=cuda --set device.amp=auto "$@"
}

step checkout check_checkout
step env "$VENV_PY" scripts/validation_env.py
step build "$VENV_PY" scripts/build_native.py --build-dir build
if [[ $QUICK -eq 1 ]]; then
  printf 'sanitize\tskipped\t0\t0\n' >> "$OUT/steps.tsv"
else
  step sanitize "$VENV_PY" scripts/build_native.py --sanitize --build-dir build_sanitize
fi
# -v lists every skip with its reason; validation_summary.py fails the gate on any skip outside the policy.
step tests-native env HASAMI_REQUIRE_NATIVE=1 "$VENV_PY" -m unittest discover -s tests -v
step tests-depfree env HASAMI_REQUIRE_NATIVE=1 "$SYSTEM_PY" -m unittest discover -s tests -v
if [[ $QUICK -eq 1 ]]; then
  printf 'parity-scaled\tskipped\t0\t0\n' >> "$OUT/steps.tsv"
else
  step parity-scaled env RIICHI_PARITY_SCALE=10 HASAMI_REQUIRE_NATIVE=1 "$VENV_PY" -m unittest -v \
    tests.riichi.test_native_core tests.riichi.test_native_parity
fi
# Milestone 4: native shogi parity (2 000 scaled games) and the engine benchmark (>= 10x Python with encoding).
if [[ $QUICK -eq 1 ]]; then
  printf 'parity-scaled-shogi\tskipped\t0\t0\n' >> "$OUT/steps.tsv"
else
  step parity-scaled-shogi env SHOGI_PARITY_SCALE=10 HASAMI_REQUIRE_NATIVE=1 "$VENV_PY" -m unittest -v \
    tests.shogi_suite.test_native_parity tests.shogi_suite.test_native_bridge
fi
step benchmark-hasami "$VENV_PY" scripts/benchmark.py --depth 2 --rounds 5
step benchmark-riichi "$VENV_PY" scripts/benchmark_riichi.py
step benchmark-shogi "$VENV_PY" scripts/benchmark_shogi.py
step devices "$VENV_PY" -m hasami_learn.cli devices
step cuda-smoke-alphazero train_smoke configs/tiny/alphazero.json tiny-alphazero-cuda
step cuda-smoke-riichi train_smoke configs/riichi-ppo-league-tonpuu.json riichi-tonpuu-cuda-smoke \
  --set run.iterations=1 --set selfplay.games_per_iteration=8 --set evaluation.games=0
step cuda-smoke-shogi train_smoke configs/tiny/alphazero-shogi.json tiny-alphazero-shogi-cuda --set run.backend=cpp
# Autonomous loop (docs/TRAINING_ja.md): the tiny plan end to end on this machine, and the installer dry run.
autoloop_smoke() {
  local plan="$SCRATCH/autoloop-tiny.json"
  "$VENV_PY" - "$plan" "$SCRATCH/autoloop" <<'PY'
import json, sys
data = json.load(open("configs/autoloop/tiny-hasami.json"))
data["output_dir"] = sys.argv[2]
json.dump(data, open(sys.argv[1], "w"), indent=1)
PY
  "$VENV_PY" -m hasami_learn.autoloop run --config "$plan" && "$VENV_PY" -m hasami_learn.autoloop status --loop "$SCRATCH/autoloop/tiny-hasami"
}
step autoloop-smoke autoloop_smoke
step autoloop-install-dry-run bash scripts/autoloop/install_wsl.sh --dry-run configs/autoloop/tiny-hasami.json
# Simulated process death (SIGKILL) + resume. NOT a power-loss test; see docs/TRAINING_ja.md section 6.
step process-kill-resume "$VENV_PY" scripts/kill_resume_smoke.py --backend cpp
if command -v xvfb-run >/dev/null; then
  step gui-smoke xvfb-run -a "$SYSTEM_PY" scripts/gui_smoke.py --backend both
  step gui-smoke-shogi xvfb-run -a "$SYSTEM_PY" scripts/gui_smoke_shogi.py --backend both
else
  printf 'gui-smoke\tskipped\t0\t0\n' >> "$OUT/steps.tsv"
  printf 'gui-smoke-shogi\tskipped\t0\t0\n' >> "$OUT/steps.tsv"
fi
# Collection measurements (reported, not gating): isolated latency, batched versus reference,
# and the CUDA batch-size sweep that replaces the provisional CUDA default.
step benchmark-decisions "$VENV_PY" scripts/benchmark_decisions.py --devices cpu cuda
step benchmark-collection "$VENV_PY" scripts/benchmark_collection.py --tables 32 128 --devices cpu cuda
step benchmark-cuda-sweep "$VENV_PY" scripts/benchmark_collection.py --tables 32 128 --devices cuda \
  --modes batched --inference-batch 32 64 128 256
cp "$OUT/logs/env.log" "$OUT/env.json" 2>/dev/null || true

"$SYSTEM_PY" scripts/validation_summary.py "$OUT"
