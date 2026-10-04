#!/bin/bash
# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

set -u

VENV_PYTHON="/opt/dns-ml/venv/bin/python"
SCORER="/opt/dns-ml/score_dns_autoencoder_live.py"
CURRENT_DIR="/opt/zeek/logs/current"
STATUS_FILE="/data/dns-ml/scoring/status.json"

mkdir -p /data/dns-ml/scoring /data/dns-ml/alerts /data/dns-ml/state /data/dns-ml/logs

find_live_log() {
    local candidate
    for candidate in \
        "$CURRENT_DIR/dns.log" \
        "$CURRENT_DIR/dns.jsonl" \
        "$CURRENT_DIR/dns.json"; do
        if [[ -f "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    find "$CURRENT_DIR" -maxdepth 1 -type f \
        \( -name 'dns*.log' -o -name 'dns*.jsonl' -o -name 'dns*.json' \) \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-
}

LIVE_LOG="$(find_live_log)"
if [[ -z "$LIVE_LOG" || ! -f "$LIVE_LOG" ]]; then
    waiting_at="$(date -u +%FT%TZ)"
    status_tmp="${STATUS_FILE}.tmp.$$"
    printf '%s\n' \
        '{' \
        '  "state": "waiting_input",' \
        '  "stage": "Waiting for DNS input",' \
        '  "progress_percent": 0,' \
        '  "message": "No live Zeek DNS records have been published yet. The next scheduled interval will check again automatically.",' \
        "  \"updated_at_utc\": \"$waiting_at\"," \
        '  "windows_scored": 0,' \
        '  "alerts_written": 0,' \
        '  "error": null' \
        '}' > "$status_tmp"
    mv -f "$status_tmp" "$STATUS_FILE"
    printf '[%s] [INFO] Waiting for Zeek DNS input in %s; the next scheduled interval will retry.\n' "$waiting_at" "$CURRENT_DIR"
    exit 0
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    printf '[%s] [!] Python environment not found: %s\n' "$(date -u +%FT%TZ)" "$VENV_PYTHON"
    exit 3
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="/opt/dns-ml:${PYTHONPATH:-}"
export ZEEK_DNS_LOG="$LIVE_LOG"
export DNS_SCORING_STATUS_PATH="$STATUS_FILE"
export DNS_SCORING_WINDOWS_PATH="/data/dns-ml/scoring/latest_scoring_windows.csv"
export DNS_SCORING_RESIDUALS_PATH="/data/dns-ml/scoring/latest_feature_residuals.csv"
export DNS_SCORING_LATENT_PATH="/data/dns-ml/scoring/latest_latent_vectors.csv"

printf '\n[%s] ===== Scheduled DNS scoring run started =====\n' "$(date -u +%FT%TZ)"
printf '[%s] Input: %s\n' "$(date -u +%FT%TZ)" "$LIVE_LOG"
"$VENV_PYTHON" "$SCORER" --once
rc=$?
printf '[%s] ===== Scheduled DNS scoring run finished rc=%s =====\n' "$(date -u +%FT%TZ)" "$rc"
exit "$rc"
