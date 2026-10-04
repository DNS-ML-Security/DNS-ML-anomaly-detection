#!/usr/bin/env python3
# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from feature_utils import (
    FEATURES,
    add_event_features,
    build_dns_windows,
    load_zeek_dns_json_files,
    validate_feature_matrix,
)

MERGED = Path("/data/dns-ml/zeek/merged_dns.jsonl")
OUT = Path("/data/dns-ml/features/dns_training_features.csv")
SUMMARY = Path(
    "/data/dns-ml/features/dns_training_feature_summary.json"
)
STATUS = Path(
    "/data/dns-ml/features/dns_training_feature_status.json"
)
WINDOW = os.getenv("DNS_WINDOW_SIZE", "5min")
MIN_WINDOWS = int(
    os.getenv(
        "DNS_MIN_WINDOWS_REQUIRED",
        "50",
    )
)


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )
    tmp.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)


def status(state, stage, percent, message, error=None):
    write_json(
        STATUS,
        {
            "state": state,
            "stage": stage,
            "progress_percent": percent,
            "message": message,
            "error": error,
            "updated_at_utc": now(),
        },
    )
    print(
        f"[{now()}] [{percent}%] {stage}: {message}",
        flush=True,
    )


def main():
    try:
        if (
            not MERGED.exists()
            or MERGED.stat().st_size <= 0
        ):
            raise RuntimeError(
                "Merged DNS dataset is missing or empty."
            )

        status(
            "running",
            "Loading merged JSON",
            10,
            "Loading verified merged DNS dataset.",
        )

        raw = load_zeek_dns_json_files(
            [str(MERGED)]
        )
        if raw.empty:
            raise RuntimeError(
                "Merged DNS dataset loaded zero records."
            )

        status(
            "running",
            "DNS event analysis",
            30,
            f"Analyzing {len(raw):,} DNS records.",
        )
        events = add_event_features(raw)
        if events.empty:
            raise RuntimeError(
                "No usable DNS events remained."
            )

        status(
            "running",
            "Window aggregation",
            55,
            f"Building {WINDOW} source-IP windows.",
        )
        windows = build_dns_windows(
            events,
            WINDOW,
        )

        if len(windows) < MIN_WINDOWS:
            raise RuntimeError(
                f"Only {len(windows):,} windows found; "
                f"minimum is {MIN_WINDOWS:,}. "
                f"Upload more normal DNS JSON data."
            )

        status(
            "running",
            "Feature validation",
            75,
            f"Validating {len(FEATURES)} features.",
        )
        valid = validate_feature_matrix(
            windows,
            FEATURES,
        )

        if (
            valid.empty
            or len(valid) != len(windows)
        ):
            raise RuntimeError(
                "Feature validation failed."
            )

        metadata = [
            column
            for column in (
                "timestamp",
                "src_ip",
            )
            if column in windows.columns
        ]

        output = (
            windows[metadata]
            .reset_index(drop=True)
            .copy()
        )

        for feature in FEATURES:
            output[feature] = (
                valid[feature]
                .reset_index(drop=True)
            )

        status(
            "running",
            "Saving features",
            90,
            f"Saving {len(output):,} training windows.",
        )

        OUT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        tmp = OUT.with_suffix(
            ".csv.tmp"
        )
        output.to_csv(
            tmp,
            index=False,
        )
        tmp.replace(OUT)

        timestamps = (
            pd.to_datetime(
                output["timestamp"],
                errors="coerce",
                utc=True,
            ).dropna()
            if "timestamp" in output
            else pd.Series(
                dtype="datetime64[ns, UTC]"
            )
        )

        summary = {
            "generated_at_utc": now(),
            "ready_for_ml": True,
            "source_merged_dataset": str(MERGED),
            "feature_output_path": str(OUT),
            "window_size": WINDOW,
            "raw_dns_records": len(raw),
            "usable_dns_events": len(events),
            "training_windows": len(output),
            "unique_source_ips": (
                int(
                    output["src_ip"]
                    .astype(str)
                    .nunique()
                )
                if "src_ip" in output
                else 0
            ),
            "feature_count": len(FEATURES),
            "features": list(FEATURES),
            "first_timestamp": (
                timestamps.min().isoformat()
                if not timestamps.empty
                else ""
            ),
            "last_timestamp": (
                timestamps.max().isoformat()
                if not timestamps.empty
                else ""
            ),
        }

        write_json(
            SUMMARY,
            summary,
        )

        status(
            "finished",
            "Ready for ML",
            100,
            (
                f"Extracted {len(output):,} windows "
                f"with {len(FEATURES)} features. "
                f"ML learning is unlocked."
            ),
        )

    except Exception as exc:
        status(
            "error",
            "Feature extraction failed",
            100,
            "Feature extraction failed.",
            str(exc),
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

