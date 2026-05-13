#!/usr/bin/env bash
set -euo pipefail

# cleanup_workspace.sh
# Dry-run by default. Use --apply to actually delete files.

APPLY=0
REMOVE_VENV=0
REMOVE_RUNS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1 ;; 
    --remove-venv) REMOVE_VENV=1 ;; 
    --remove-runs) REMOVE_RUNS=1 ;; 
    -h|--help)
      cat <<EOF
Usage: $0 [--apply] [--remove-venv] [--remove-runs]

Find and optionally remove temporary Python artefacts:
- __pycache__, *.pyc, .pytest_cache, .mypy_cache, .ipynb_checkpoints
By default this script only prints what it would remove. Pass --apply to delete.
Use --remove-venv to also remove .venv, and --remove-runs to remove runs/ artifacts.
EOF
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "Workspace root: $ROOT_DIR"

declare -a patterns=("__pycache__" "*.pyc" ".pytest_cache" ".mypy_cache" ".ipynb_checkpoints")

for p in "${patterns[@]}"; do
  echo "Searching for: $p"
  if [[ $APPLY -eq 0 ]]; then
    find "$ROOT_DIR" -path "$ROOT_DIR/.venv" -prune -o -path "$ROOT_DIR/.git" -prune -o -name "$p" -print
  else
    find "$ROOT_DIR" -path "$ROOT_DIR/.venv" -prune -o -path "$ROOT_DIR/.git" -prune -o -name "$p" -print -exec rm -rf {} +
  fi
done

if [[ $REMOVE_RUNS -eq 1 ]]; then
  if [[ $APPLY -eq 1 ]]; then
    echo "Removing runs/ ..."
    rm -rf "$ROOT_DIR/runs"
  else
    echo "Would remove: $ROOT_DIR/runs"
  fi
fi

if [[ $REMOVE_VENV -eq 1 ]]; then
  if [[ $APPLY -eq 1 ]]; then
    echo "Removing .venv ..."
    rm -rf "$ROOT_DIR/.venv"
  else
    echo "Would remove: $ROOT_DIR/.venv"
  fi
fi

echo "Done."
