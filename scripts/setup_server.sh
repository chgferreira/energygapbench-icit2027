#!/usr/bin/env bash
# Compatibility wrapper; scripts/server.sh is the supported entry point.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/server.sh" setup "$@"
