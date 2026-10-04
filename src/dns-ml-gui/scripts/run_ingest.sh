#!/bin/bash
# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

set -e

cd /opt/dns-ml-gui
source /opt/dns-ml/venv/bin/activate

export PYTHONPATH=/opt/dns-ml-gui

python scripts/ingest_alerts.py
