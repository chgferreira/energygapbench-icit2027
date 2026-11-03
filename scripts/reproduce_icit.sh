#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
analysis_python="${ANALYSIS_PYTHON:-python}"
"$analysis_python" analysis/scripts/reproduce_public.py "$@"
