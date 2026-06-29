#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: av-baseline-report.sh <session-uuid>"
    echo "Example: av-baseline-report.sh 7ba63b1c-371a-40f2-855f-8451c1cfaaf1"
    exit 1
fi

python3 "${SCRIPT_DIR}/av-baseline-report.py" "$1"
