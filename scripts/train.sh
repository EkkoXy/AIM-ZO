#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "usage: $0 CONFIG.yaml" >&2
  exit 2
fi
exec aimzo-train "$1"
