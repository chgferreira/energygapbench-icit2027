#!/usr/bin/env bash
# Compatibility wrapper for the former --resume ID spelling.
set -euo pipefail
task_arguments=()
task_resume=''
while [[ $# -gt 0 ]]; do
  if [[ "$1" == --resume ]]; then task_resume="${2:?Missing experiment ID}"; shift 2
  else task_arguments+=("$1"); shift
  fi
done
if [[ -n "$task_resume" ]]; then
  if [[ ${#task_arguments[@]} -eq 0 ]]; then
    exec bash "$(dirname -- "${BASH_SOURCE[0]}")/server.sh" resume "$task_resume"
  fi
  exec bash "$(dirname -- "${BASH_SOURCE[0]}")/server.sh" resume "$task_resume" "${task_arguments[@]}"
fi
if [[ ${#task_arguments[@]} -eq 0 ]]; then
  exec bash "$(dirname -- "${BASH_SOURCE[0]}")/server.sh" run --preset full
fi
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/server.sh" run --preset full "${task_arguments[@]}"
