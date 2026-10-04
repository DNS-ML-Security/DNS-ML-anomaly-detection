#!/usr/bin/env python3
# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

"""
Merge Zeek dns.log files into one deduplicated JSONL training dataset.

Inputs:
    /opt/zeek/logs/current/dns.log
    All recursively discovered dns*.log and dns*.log.gz files under /opt/zeek/logs/

Outputs:
    /data/dns-ml/zeek/merged_dns.jsonl
    /data/dns-ml/zeek/stats.json
    /data/dns-ml/zeek/stats.sqlite
    /data/dns-ml/zeek/source_ip_stats.csv
    /data/dns-ml/zeek/merge_status.json
    /data/dns-ml/zeek/merge_manifest.json
    /data/dns-ml/zeek/merge.log
    /data/dns-ml/zeek/merge.pid

The original Zeek files are never modified or deleted.
Compressed .gz files are decompressed as streams, so no extracted copy is left behind.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

ZEEK_ROOT = Path(os.getenv("ZEEK_LOG_ROOT", "/opt/zeek/logs"))
OUTPUT_DIR = Path(os.getenv("DNS_ZEEK_OUTPUT_DIR", "/data/dns-ml/zeek"))

MERGED_PATH = OUTPUT_DIR / "merged_dns.jsonl"
STATS_PATH = OUTPUT_DIR / "stats.json"
DB_PATH = OUTPUT_DIR / "stats.sqlite"
SOURCE_IP_STATS_PATH = OUTPUT_DIR / "source_ip_stats.csv"
STATUS_PATH = OUTPUT_DIR / "merge_status.json"
MANIFEST_PATH = OUTPUT_DIR / "merge_manifest.json"
PID_PATH = OUTPUT_DIR / "merge.pid"

BATCH_SIZE = int(os.getenv("DNS_ZEEK_BATCH_SIZE", "2000"))
STATUS_EVERY = int(os.getenv("DNS_ZEEK_STATUS_EVERY", "5000"))
WINDOW_SECONDS = int(os.getenv("DNS_WINDOW_SECONDS", "300"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    tmp.replace(path)


def update_status(
    *,
    state: str,
    stage: str,
    progress: int,
    message: str,
    current_file: str = "",
    files_total: int = 0,
    files_processed: int = 0,
    lines_read: int = 0,
    records_inserted: int = 0,
    duplicates: int = 0,
    invalid_lines: int = 0,
    unique_src_ips_seen: int = 0,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "state": state,
        "stage": stage,
        "progress_percent": max(0, min(int(progress), 100)),
        "message": message,
        "current_file": current_file,
        "files_total": int(files_total),
        "files_processed": int(files_processed),
        "lines_read": int(lines_read),
        "records_inserted": int(records_inserted),
        "duplicates": int(duplicates),
        "invalid_lines": int(invalid_lines),
        "unique_src_ips_seen": int(unique_src_ips_seen),
        "error": error,
        "updated_at_utc": utc_now(),
    }
    if extra:
        payload.update(extra)
    atomic_json(STATUS_PATH, payload)


def discover_dns_files(root: Path = ZEEK_ROOT) -> List[Path]:
    """
    Discover supported Zeek DNS sources.

    Live Zeek operation remains compatible with:
      dns*.log
      dns*.log.gz

    GUI training uploads are JSON-only and are normalized/staged as:
      dns*.json
      dns*.jsonl

    The GUI uploader itself accepts only extensions configured in:
      /opt/dns-ml-gui/config/training_input.json
    """
    if not root.exists():
        return []

    candidates: List[Path] = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        name = path.name.lower()

        if not name.startswith("dns"):
            continue

        if (
            name.endswith(".log")
            or name.endswith(".log.gz")
            or name.endswith(".json")
            or name.endswith(".jsonl")
        ):
            candidates.append(path)

    current = root / "current" / "dns.log"

    if current.exists():
        candidates.append(current)

    # Resolve duplicate paths while preserving stable oldest-to-newest ordering.
    unique: Dict[str, Path] = {}

    for path in candidates:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)

        unique[key] = path

    return sorted(
        unique.values(),
        key=lambda path: (
            path.stat().st_mtime
            if path.exists()
            else 0.0,
            str(path),
        ),
    )




def source_manifest(files: List[Path]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for path in files:
        try:
            stat = path.stat()
            rows.append(
                {
                    "path": str(path),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "inode": int(stat.st_ino),
                }
            )
        except OSError as exc:
            rows.append(
                {
                    "path": str(path),
                    "size": None,
                    "mtime_ns": None,
                    "inode": None,
                    "error": str(exc),
                }
            )

    signature_raw = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return {
        "generated_at_utc": utc_now(),
        "root": str(ZEEK_ROOT),
        "files": rows,
        "file_count": len(rows),
        "signature": hashlib.sha256(signature_raw.encode("utf-8")).hexdigest(),
    }


def open_log(path: Path):
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def parse_zeek_separator(line: str) -> str:
    value = line[len("#separator"):].strip()
    if value.startswith(r"\x") and len(value) >= 4:
        try:
            return bytes.fromhex(value[2:4]).decode("latin1")
        except Exception:
            return "\t"
    return value or "\t"


def to_number(value: Any) -> Any:
    if value in (None, "", "-", "(empty)"):
        return None
    try:
        if isinstance(value, (int, float)):
            return value
        text = str(value)
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except Exception:
        return value


def to_bool(value: Any) -> Any:
    if value in (True, False):
        return value
    if value in (None, "", "-", "(empty)"):
        return None
    text = str(value).strip().lower()
    if text in {"t", "true", "1"}:
        return True
    if text in {"f", "false", "0"}:
        return False
    return value


def parse_vector(value: Any, numeric: bool = False) -> List[Any]:
    if value in (None, "", "-", "(empty)"):
        return []
    if isinstance(value, list):
        items = value
    else:
        items = str(value).split(",")

    result: List[Any] = []
    for item in items:
        item = item.strip() if isinstance(item, str) else item
        if item in ("", "-", "(empty)"):
            continue
        result.append(to_number(item) if numeric else item)
    return result


NUMERIC_FIELDS = {
    "ts",
    "id.orig_p",
    "id.resp_p",
    "trans_id",
    "rtt",
    "qclass",
    "qtype",
    "rcode",
}

BOOL_FIELDS = {"AA", "TC", "RD", "RA", "Z", "rejected"}


def normalize_event(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    query = record.get("query")
    src_ip = record.get("id.orig_h")
    ts = to_number(record.get("ts"))

    if ts is None or src_ip in (None, "", "-"):
        return None

    try:
        ts_float = float(ts)
    except Exception:
        return None

    normalized: Dict[str, Any] = {}
    for key, value in record.items():
        if key in NUMERIC_FIELDS:
            normalized[key] = to_number(value)
        elif key in BOOL_FIELDS:
            normalized[key] = to_bool(value)
        elif key == "answers":
            normalized[key] = parse_vector(value, numeric=False)
        elif key == "TTLs":
            normalized[key] = parse_vector(value, numeric=True)
        elif value in ("-", "(empty)"):
            normalized[key] = None
        else:
            normalized[key] = value

    normalized["ts"] = ts_float
    normalized["id.orig_h"] = str(src_ip)
    normalized["query"] = "" if query in (None, "-", "(empty)") else str(query)

    # Ensure fields used by the ML pipeline always exist.
    defaults = {
        "id.resp_h": "",
        "qtype_name": "",
        "rcode_name": "",
        "rtt": 0.0,
        "answers": [],
        "TTLs": [],
    }
    for key, default in defaults.items():
        if normalized.get(key) is None:
            normalized[key] = default

    return normalized


def iter_events(path: Path) -> Iterator[Tuple[Optional[Dict[str, Any]], bool]]:
    """
    Yield (event, valid_line).

    valid_line=False means a non-comment data line could not be parsed.
    """
    with open_log(path) as handle:
        separator = "\t"
        fields: Optional[List[str]] = None

        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue

            if line.startswith("#separator"):
                separator = parse_zeek_separator(line)
                continue

            if line.startswith("#fields"):
                fields = line.split(separator)[1:]
                if not fields:
                    fields = line.split()[1:]
                continue

            if line.startswith("#"):
                continue

            stripped = line.lstrip()
            if stripped.startswith("{"):
                try:
                    parsed = json.loads(line)
                    event = normalize_event(parsed)
                    yield event, event is not None
                except json.JSONDecodeError:
                    yield None, False
                continue

            if not fields:
                yield None, False
                continue

            values = line.split(separator)
            if len(values) < len(fields):
                values.extend(["-"] * (len(fields) - len(values)))
            elif len(values) > len(fields):
                values = values[: len(fields)]

            event = normalize_event(dict(zip(fields, values)))
            yield event, event is not None


def event_fingerprint(event: Dict[str, Any]) -> str:
    """
    Build a format-independent fingerprint from stable DNS transaction fields.

    When Zeek UID exists, it is the strongest deduplication key for overlap
    between current/dns.log and a rotated copy. A fallback key is used for
    records without UID.
    """
    uid = event.get("uid")
    if uid not in (None, "", "-"):
        stable = {
            "ts": round(float(event.get("ts", 0.0)), 6),
            "uid": uid,
            "id.orig_h": event.get("id.orig_h"),
            "id.resp_h": event.get("id.resp_h"),
            "query": event.get("query"),
            "qtype_name": event.get("qtype_name"),
            "rcode_name": event.get("rcode_name"),
        }
    else:
        stable = {
            "ts": round(float(event.get("ts", 0.0)), 6),
            "id.orig_h": event.get("id.orig_h"),
            "id.orig_p": event.get("id.orig_p"),
            "id.resp_h": event.get("id.resp_h"),
            "id.resp_p": event.get("id.resp_p"),
            "proto": event.get("proto"),
            "trans_id": event.get("trans_id"),
            "query": event.get("query"),
            "qtype_name": event.get("qtype_name"),
            "rcode_name": event.get("rcode_name"),
            "answers": event.get("answers") or [],
        }

    raw = json.dumps(
        stable,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def create_work_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute(
        """
        CREATE TABLE events (
            fingerprint TEXT PRIMARY KEY,
            ts REAL NOT NULL,
            src_ip TEXT NOT NULL,
            raw_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def export_merged(conn: sqlite3.Connection, destination: Path) -> None:
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        cursor = conn.execute(
            "SELECT raw_json FROM events ORDER BY ts ASC, rowid ASC"
        )
        for (raw_json,) in cursor:
            handle.write(raw_json)
            handle.write("\n")
    tmp.replace(destination)


def compute_stats(conn: sqlite3.Connection, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate aggregate source-IP coverage and actual wall-clock duration."""
    total_records, unique_src_ips, first_ts, last_ts = conn.execute(
        """
        SELECT COUNT(*), COUNT(DISTINCT src_ip), MIN(ts), MAX(ts)
        FROM events
        """
    ).fetchone()

    total_ip_windows = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT src_ip, CAST(ts / ? AS INTEGER) AS window_bucket
            FROM events
            GROUP BY src_ip, window_bucket
        )
        """,
        (WINDOW_SECONDS,),
    ).fetchone()[0]

    average_queries = (
        float(total_records) / float(unique_src_ips)
        if unique_src_ips else 0.0
    )

    actual_span_seconds = (
        max(0.0, float(last_ts) - float(first_ts))
        if first_ts is not None and last_ts is not None
        else 0.0
    )

    aggregate_minutes = float(total_ip_windows or 0) * (WINDOW_SECONDS / 60)
    aggregate_hours = aggregate_minutes / 60
    aggregate_days = aggregate_hours / 24

    actual_minutes = actual_span_seconds / 60
    actual_hours = actual_span_seconds / 3600
    actual_days = actual_span_seconds / 86400

    return {
        "generated_at_utc": utc_now(),
        "source_root": str(ZEEK_ROOT),
        "merged_path": str(MERGED_PATH),
        "source_file_count": int(manifest.get("file_count", 0)),
        "source_signature": manifest.get("signature"),
        "total_dns_records": int(total_records or 0),
        "total_source_ips": int(unique_src_ips or 0),
        "total_ip_windows": int(total_ip_windows or 0),
        "window_size_minutes": WINDOW_SECONDS / 60,
        "average_queries_per_source_ip": round(average_queries, 4),
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "aggregate_source_ip_coverage_minutes": round(aggregate_minutes, 4),
        "aggregate_source_ip_coverage_hours": round(aggregate_hours, 4),
        "aggregate_source_ip_coverage_days": round(aggregate_days, 4),
        "total_capture_minutes": round(aggregate_minutes, 4),
        "total_capture_hours": round(aggregate_hours, 4),
        "actual_capture_duration_seconds": round(actual_span_seconds, 4),
        "actual_capture_duration_minutes": round(actual_minutes, 4),
        "actual_capture_duration_hours": round(actual_hours, 4),
        "actual_capture_duration_days": round(actual_days, 4),
        "actual_wall_clock_span_seconds": round(actual_span_seconds, 4),
        "actual_wall_clock_span_hours": round(actual_hours, 4),
    }

def export_source_ip_stats(conn: sqlite3.Connection, destination: Path) -> None:
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    cursor = conn.execute(
        """
        SELECT
            src_ip,
            COUNT(*) AS dns_queries,
            COUNT(DISTINCT CAST(ts / ? AS INTEGER)) AS five_minute_windows,
            MIN(ts) AS first_ts,
            MAX(ts) AS last_ts
        FROM events
        GROUP BY src_ip
        ORDER BY dns_queries DESC, src_ip ASC
        """,
        (WINDOW_SECONDS,),
    )

    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "src_ip",
                "dns_queries",
                "five_minute_windows",
                "capture_minutes",
                "first_ts",
                "last_ts",
            ]
        )
        for src_ip, dns_queries, windows, first_ts, last_ts in cursor:
            writer.writerow(
                [
                    src_ip,
                    int(dns_queries),
                    int(windows),
                    int(windows) * (WINDOW_SECONDS / 60),
                    first_ts,
                    last_ts,
                ]
            )
    tmp.replace(destination)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")

    started = time.time()
    files: List[Path] = []
    conn: Optional[sqlite3.Connection] = None
    work_db = DB_PATH.with_suffix(DB_PATH.suffix + ".tmp")

    counters = {
        "lines_read": 0,
        "records_inserted": 0,
        "duplicates": 0,
        "invalid_lines": 0,
    }
    src_ips_seen: set[str] = set()

    try:
        update_status(
            state="running",
            stage="Discovering Zeek logs",
            progress=2,
            message=f"Scanning {ZEEK_ROOT} for DNS logs.",
        )

        files = discover_dns_files()
        manifest = source_manifest(files)

        if not files:
            raise RuntimeError(
                f"No supported dns*.log, dns*.log.gz, dns*.json or dns*.jsonl files were found under {ZEEK_ROOT}"
            )

        update_status(
            state="running",
            stage="Preparing merge database",
            progress=5,
            message=f"Discovered {len(files)} DNS log files.",
            files_total=len(files),
        )

        conn = create_work_db(work_db)
        batch: List[Tuple[str, float, str, str]] = []

        for file_index, path in enumerate(files, start=1):
            base_progress = 5 + int(((file_index - 1) / len(files)) * 80)
            update_status(
                state="running",
                stage="Reading and merging Zeek logs",
                progress=base_progress,
                message=f"Processing file {file_index} of {len(files)}.",
                current_file=str(path),
                files_total=len(files),
                files_processed=file_index - 1,
                unique_src_ips_seen=len(src_ips_seen),
                **counters,
            )

            for event, valid in iter_events(path):
                counters["lines_read"] += 1

                if not valid or event is None:
                    counters["invalid_lines"] += 1
                    continue

                fingerprint = event_fingerprint(event)
                raw_json = json.dumps(
                    event,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                batch.append(
                    (
                        fingerprint,
                        float(event["ts"]),
                        str(event["id.orig_h"]),
                        raw_json,
                    )
                )
                src_ips_seen.add(str(event["id.orig_h"]))

                if len(batch) >= BATCH_SIZE:
                    before = conn.total_changes
                    conn.executemany(
                        """
                        INSERT OR IGNORE INTO events
                        (fingerprint, ts, src_ip, raw_json)
                        VALUES (?, ?, ?, ?)
                        """,
                        batch,
                    )
                    conn.commit()
                    inserted = conn.total_changes - before
                    counters["records_inserted"] += inserted
                    counters["duplicates"] += len(batch) - inserted
                    batch.clear()

                if counters["lines_read"] % STATUS_EVERY == 0:
                    within_file = min(
                        79,
                        int((file_index / len(files)) * 79),
                    )
                    update_status(
                        state="running",
                        stage="Reading and merging Zeek logs",
                        progress=5 + within_file,
                        message=f"Reading {path.name}.",
                        current_file=str(path),
                        files_total=len(files),
                        files_processed=file_index - 1,
                        unique_src_ips_seen=len(src_ips_seen),
                        **counters,
                    )

            if batch:
                before = conn.total_changes
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO events
                    (fingerprint, ts, src_ip, raw_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    batch,
                )
                conn.commit()
                inserted = conn.total_changes - before
                counters["records_inserted"] += inserted
                counters["duplicates"] += len(batch) - inserted
                batch.clear()

            update_status(
                state="running",
                stage="Reading and merging Zeek logs",
                progress=5 + int((file_index / len(files)) * 80),
                message=f"Completed {path.name}.",
                current_file=str(path),
                files_total=len(files),
                files_processed=file_index,
                unique_src_ips_seen=len(src_ips_seen),
                **counters,
            )

        update_status(
            state="running",
            stage="Writing merged training dataset",
            progress=88,
            message="Exporting the deduplicated merged JSONL file.",
            files_total=len(files),
            files_processed=len(files),
            unique_src_ips_seen=len(src_ips_seen),
            **counters,
        )
        export_merged(conn, MERGED_PATH)

        update_status(
            state="running",
            stage="Calculating DNS statistics",
            progress=93,
            message="Calculating records, source IPs, capture windows and query averages.",
            files_total=len(files),
            files_processed=len(files),
            unique_src_ips_seen=len(src_ips_seen),
            **counters,
        )
        stats = compute_stats(conn, manifest)
        export_source_ip_stats(conn, SOURCE_IP_STATS_PATH)

        conn.execute("PRAGMA wal_checkpoint(FULL)")
        conn.close()
        conn = None

        # Move completed SQLite DB into place.
        for suffix in ("-wal", "-shm"):
            extra = Path(str(work_db) + suffix)
            if extra.exists():
                extra.unlink()
        work_db.replace(DB_PATH)

        atomic_json(STATS_PATH, stats)
        atomic_json(MANIFEST_PATH, manifest)

        elapsed = round(time.time() - started, 3)
        update_status(
            state="finished",
            stage="Completed",
            progress=100,
            message="Zeek DNS logs were merged successfully.",
            files_total=len(files),
            files_processed=len(files),
            unique_src_ips_seen=stats["total_source_ips"],
            extra={
                **counters,
                "elapsed_seconds": elapsed,
                "stats": stats,
            },
        )

        print(json.dumps({"status": "success", **stats}, indent=2))
        return 0

    except Exception as exc:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

        update_status(
            state="error",
            stage="Failed",
            progress=100,
            message="Zeek DNS log merge failed.",
            current_file="",
            files_total=len(files),
            files_processed=0,
            unique_src_ips_seen=len(src_ips_seen),
            error=str(exc),
            **counters,
        )
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    finally:
        try:
            PID_PATH.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
