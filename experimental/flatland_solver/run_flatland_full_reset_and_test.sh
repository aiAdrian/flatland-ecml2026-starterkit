#!/usr/bin/env bash
set -euo pipefail

SOLVER_DIR="$(dirname "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-/etc/pyenv/.pyenv/versions/flatland-ecml2026/bin/python}"

cleanup_paths=(
  "$SOLVER_DIR/runs"
  "$SOLVER_DIR/generated_envs"
  "$SOLVER_DIR/checkpoints"
)

printf '[cleanup] removing previous runs and generated data\n'
for path in "${cleanup_paths[@]}"; do
  if [[ -e "$path" ]]; then
    rm -rf "$path"
  fi
done

mkdir -p "$SOLVER_DIR"
cd "$SOLVER_DIR"

run_cmd() {
  printf '\n[run] %s\n' "$*"
  "$PYTHON_BIN" main.py "$@"
}

# Baselines first
run_cmd --mode eval --policy random --episodes 100 --max-episode-steps 20
run_cmd --mode eval --policy dla --episodes 100 --max-episode-steps 20

# BC train + eval
run_cmd --mode train --policy bc --obs-variant decision_point --episodes 200 --train-epochs 5 --max-episode-steps 300 --debug-checks
run_cmd --mode eval --policy bc --obs-variant decision_point --episodes 100 --max-episode-steps 300 --debug-checks

# MAPPO train + eval
run_cmd --mode train --policy mappo --obs-variant spawn_aware --episodes 200 --train-epochs 5 --max-episode-steps 300 --debug-checks
run_cmd --mode eval --policy mappo --obs-variant spawn_aware --episodes 100 --max-episode-steps 300 --debug-checks

printf '\n[done] all runs completed\n'
