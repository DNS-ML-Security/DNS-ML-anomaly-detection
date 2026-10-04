#!/bin/bash
# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

set -euo pipefail

source /opt/dns-ml/venv/bin/activate
export PYTHONUNBUFFERED=1
export ZEEK_LOG_ROOT="${ZEEK_LOG_ROOT:-/opt/zeek/logs}"
export DNS_ZEEK_OUTPUT_DIR="${DNS_ZEEK_OUTPUT_DIR:-/data/dns-ml/zeek}"

exec python /opt/dns-ml/zeek_log_manager.py

