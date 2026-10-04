#!/usr/bin/env bash
# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

set -u
MODE="${1:-once}"
case "$MODE" in
  once|full|scheduled) ;;
  *) echo "Usage: $0 {once|full|scheduled}" >&2; exit 2 ;;
esac
SCORER_MODE="$MODE"
[ "$MODE" = "scheduled" ] && SCORER_MODE="once"
SCHEDULED_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

PYTHON_BIN="/opt/dns-ml/venv/bin/python"
SCORER="/opt/dns-ml/score_dns_isolation_forest_live.py"
ALERT_FILE="/data/dns-ml/alerts/dns_categorization_forest_alerts.jsonl"

export PYTHONPATH="/opt/dns-ml:/opt/dns-ml-gui${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="-1"
export DNS_IF_SCORING_DEBUG="${DNS_IF_SCORING_DEBUG:-1}"

"$PYTHON_BIN" "$SCORER" --mode "$SCORER_MODE"
RC=$?

if [ "$MODE" = "scheduled" ]; then
  mkdir -p /data/dns-ml/isolation_scoring
  cat > /data/dns-ml/isolation_scoring/last_cron_run.json <<EOF
{
  "started_at_utc": "$SCHEDULED_STARTED_UTC",
  "completed_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "return_code": $RC
}
EOF
fi

# Keep alert ingestion independent from the deep-learning/heuristic alert file.
# A successful/no-data run can safely ingest any new categorization alerts.
if [ "$RC" -eq 0 ] && [ -f "$ALERT_FILE" ]; then
  "$PYTHON_BIN" - "$ALERT_FILE" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "/opt/dns-ml-gui")
from src.ingest import ingest_alert_file
result = ingest_alert_file(Path(sys.argv[1]), incremental=True)
print(
    "[categorization-ingest] "
    f"read={result.get('total_read', 0)} inserted={result.get('inserted', 0)} "
    f"error={result.get('error')}"
)
PY
fi

exit "$RC"
