#!/usr/bin/env bash
# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

set -uo pipefail
MODE="${1:-once}"
SCHEDULED="${2:-}"
PYTHON_BIN="/opt/dns-ml/venv/bin/python"
SCORER="/opt/dns-ml/score_dns_heuristics_live.py"
LAST_RUN="/data/dns-ml/heuristics_scoring/cron_last_run.json"
mkdir -p /data/dns-ml/heuristics_scoring
STARTED="$(date -Iseconds)"
RC=0
"$PYTHON_BIN" "$SCORER" --mode "$MODE" || RC=$?
if [[ "$SCHEDULED" == "--scheduled" ]]; then
  FINISHED="$(date -Iseconds)"
  cat > "$LAST_RUN" <<EOF
{
  "started_at": "$STARTED",
  "finished_at": "$FINISHED",
  "return_code": $RC,
  "mode": "$MODE"
}
EOF
fi
exit "$RC"

