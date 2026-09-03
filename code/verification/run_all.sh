#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [ -d "$ROOT/.venv" ]; then
  . "$ROOT/.venv/bin/activate"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -B code/verification/verify_environment.py
"$PYTHON_BIN" -B code/verification/verify_release.py
"$PYTHON_BIN" -B -m unittest discover -s code/tests -v
replay_dir="$(mktemp -d /tmp/three-axis-full-XXXXXX)"
trap 'rm -rf "$replay_dir"' EXIT
"$PYTHON_BIN" -B code/reproduction/replay_core_constructions.py \
  --output "$replay_dir/CORE_CONSTRUCTION_REPLAY.json"
"$PYTHON_BIN" -B code/reproduction/regenerate_all.py \
  --output-dir "$replay_dir/regenerated" --skip-construction-replay
cmp manuscript/figures/FIGURE_DATA.json "$replay_dir/regenerated/figures/FIGURE_DATA.json"
residue="$(find . \( -name '*.pyc' -o -name '__pycache__' \) -print)"
if [ -n "$residue" ]; then
  echo "FAIL cache residue"
  echo "$residue"
  exit 1
fi
echo "ALL_RELEASE_CHECKS_PASSED"
