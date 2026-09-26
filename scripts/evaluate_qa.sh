#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: $0 CONFIG.yaml CHECKPOINT" >&2
  exit 2
fi
exec aimzo-qa-eval --config "$1" --checkpoint "$2" --scope official
