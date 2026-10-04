#!/usr/bin/env python3
# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from src.ingest import ingest_alert_file
from src.settings import ALERT_JSONL_PATH


def main():
    parser = argparse.ArgumentParser(description="Ingest DNS alert JSONL into SQLite.")
    parser.add_argument("--file", default=str(ALERT_JSONL_PATH), help="Path to dns_alerts.jsonl")
    parser.add_argument("--full", action="store_true", help="Read file from beginning.")

    args = parser.parse_args()

    result = ingest_alert_file(Path(args.file), incremental=not args.full)

    print("[+] Ingest result:")
    for key, value in result.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
