#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${STUDY_WORKSPACE:-/shared/g00107537/jupyter}"
PYTHON="${STUDY_PYTHON:-$WORKSPACE/environments/masa/bin/python}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -u "$REPO/experiments/run_study.py" --family msasl --workspace "$WORKSPACE" "$@"
