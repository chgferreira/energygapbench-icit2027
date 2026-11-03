#!/usr/bin/env bash
# Compatibility wrapper; starts a new study without deleting previous results.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/server.sh" run "$@"
