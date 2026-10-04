# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.settings import DB_PATH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_uid TEXT UNIQUE NOT NULL,
            alert_name TEXT,
            alert_type TEXT,
            timestamp TEXT,
            window_end TEXT,
            src_ip TEXT,
            severity TEXT,
            threat_score INTEGER,
            status TEXT DEFAULT 'New',
            model_name TEXT,
            window_size TEXT,
            reconstruction_error REAL,
            threshold REAL,
            detection_methods_json TEXT,
            top_contributing_features_json TEXT,
            possible_causes_json TEXT,
            raw_json TEXT,
            analyst_note TEXT,
            created_at_utc TEXT,
            updated_at_utc TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS alert_domains (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_uid TEXT NOT NULL,
            query TEXT,
            count INTEGER,
            reasons_json TEXT,
            suspicion_score INTEGER,
            max_entropy REAL,
            avg_entropy REAL,
            max_query_len REAL,
            max_label_len REAL,
            ttl_min REAL,
            ttl_avg REAL,
            answer_count_max REAL,
            unique_answer_ips INTEGER,
            created_at_utc TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_state (
            source_path TEXT PRIMARY KEY,
            inode INTEGER,
            offset INTEGER,
            last_ingested_at_utc TEXT,
            last_error TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS rule_config (
            rule_name TEXT PRIMARY KEY,
            enabled INTEGER DEFAULT 1,
            description TEXT,
            rule_json TEXT,
            updated_at_utc TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS model_status (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at_utc TEXT
        )
        """
    )

    # Upgrade an older database without deleting data.
    existing_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(alerts)").fetchall()
    }
    for column_name, column_type in {
        "alert_type": "TEXT",
        "window_end": "TEXT",
    }.items():
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE alerts ADD COLUMN {column_name} {column_type}")


    # BEGIN DNS CATEGORIZATION ALERT SCHEMA
    # Additive migration only: existing Autoencoder/heuristic alerts are untouched.
    categorization_columns = {
        "isolation_score": "REAL",
        "isolation_threshold": "REAL",
        "score_margin": "REAL",
        "score_percentile": "REAL",
        "contamination": "REAL",
        "n_estimators": "INTEGER",
        "max_samples": "TEXT",
        "max_features": "TEXT",
    }
    current_alert_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(alerts)").fetchall()
    }
    for column_name, column_type in categorization_columns.items():
        if column_name not in current_alert_columns:
            cursor.execute(f"ALTER TABLE alerts ADD COLUMN {column_name} {column_type}")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_alerts_alert_type ON alerts(alert_type)"
    )
    # END DNS CATEGORIZATION ALERT SCHEMA

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_src_ip ON alerts(src_ip)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_domains_alert_uid ON alert_domains(alert_uid)")

    connection.commit()
    seed_default_rules(connection)
    connection.close()


def seed_default_rules(connection: sqlite3.Connection) -> None:
    rules = [
        (
            "heuristic_dga",
            "Many unique random-looking domains with high NXDOMAIN rate.",
            {
                "total_queries": ">= 20",
                "unique_query_ratio": ">= 0.90",
                "nxdomain_rate": ">= 0.50",
                "high_entropy_rate": ">= 0.50",
            },
        ),
        (
            "heuristic_tunnel",
            "Long or high-entropy queries and TXT abuse suggesting DNS tunneling.",
            {
                "total_queries": ">= 20",
                "query_shape": "txt_rate >= 0.40 OR long_query_rate >= 0.40 OR long_label_rate >= 0.30",
                "high_entropy_rate": ">= 0.40",
            },
        ),
        (
            "heuristic_fastflux",
            "Many answer IPs with low TTL suggesting fast-flux.",
            {
                "total_queries": ">= 10",
                "answer_behavior": "answer_count_avg >= 3 OR answer_count_max >= 5 OR unique_answer_ips >= 10",
                "ttl_behavior": "ttl_min > 0 AND (ttl_min <= 60 OR low_ttl_rate >= 0.30)",
            },
        ),
        (
            "heuristic_resolver_failure",
            "High SERVFAIL or NXDOMAIN rate.",
            {
                "total_queries": ">= 20",
                "failure_rate": "servfail_rate >= 0.50 OR nxdomain_rate >= 0.70",
            },
        ),
        (
            "heuristic_rogue_resolver",
            "A host uses an unauthorized DNS resolver.",
            {"approved_resolvers": "configured allowlist"},
        ),
    ]

    for name, description, rule in rules:
        connection.execute(
            """
            INSERT OR IGNORE INTO rule_config
            (rule_name, enabled, description, rule_json, updated_at_utc)
            VALUES (?, 1, ?, ?, ?)
            """,
            (name, description, json.dumps(rule, indent=2), utc_now()),
        )
    connection.commit()


def read_sql(query: str, params: tuple = ()) -> pd.DataFrame:
    connection = get_connection()
    try:
        return pd.read_sql_query(query, connection, params=params)
    finally:
        connection.close()


def execute(query: str, params: tuple = ()) -> None:
    connection = get_connection()
    try:
        connection.execute(query, params)
        connection.commit()
    finally:
        connection.close()


def get_alert_by_id(alert_id: int) -> Optional[Dict[str, Any]]:
    connection = get_connection()
    try:
        row = connection.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        return dict(row) if row else None
    finally:
        connection.close()


def get_domains_for_alert(alert_uid: str) -> pd.DataFrame:
    return read_sql(
        """
        SELECT query, count, suspicion_score, reasons_json, max_entropy,
               avg_entropy, max_query_len, max_label_len, ttl_min, ttl_avg,
               answer_count_max, unique_answer_ips
        FROM alert_domains
        WHERE alert_uid = ?
        ORDER BY suspicion_score DESC, count DESC
        """,
        (alert_uid,),
    )


def update_alert_status(alert_id: int, status: str, analyst_note: str) -> None:
    execute(
        """
        UPDATE alerts
        SET status = ?, analyst_note = ?, updated_at_utc = ?
        WHERE id = ?
        """,
        (status, analyst_note, utc_now(), alert_id),
    )


def _directory_usage(path: Path) -> tuple[int, int, int]:
    """Return file count, directory count, and total bytes without following links."""
    file_count = 0
    directory_count = 0
    total_bytes = 0

    if not path.exists():
        return file_count, directory_count, total_bytes

    for root, directory_names, file_names in os.walk(path, followlinks=False):
        directory_count += len(directory_names)
        root_path = Path(root)
        for filename in file_names:
            candidate = root_path / filename
            file_count += 1
            try:
                if not candidate.is_symlink():
                    total_bytes += candidate.stat().st_size
            except OSError:
                pass

    return file_count, directory_count, total_bytes


def clear_all_alerts(alert_directory: Path) -> Dict[str, Any]:
    """Permanently remove only alert data from SQLite and the alert directory.

    The alert directory itself is recreated immediately so the live scorer can
    continue writing future alerts. Model files, Zeek state, training data,
    scoring diagnostics, and configuration are not touched.
    """
    alert_directory = Path(alert_directory).expanduser()

    # Refuse to recursively purge through a symbolic link.
    if alert_directory.is_symlink():
        raise RuntimeError(
            f"Refusing to purge symbolic-link alert directory: {alert_directory}"
        )

    parent = alert_directory.parent
    parent.mkdir(parents=True, exist_ok=True)

    quarantine: Optional[Path] = None
    file_count = 0
    directory_count = 0
    total_bytes = 0

    if alert_directory.exists():
        if not alert_directory.is_dir():
            raise RuntimeError(
                f"Configured alert path is not a directory: {alert_directory}"
            )

        stat_result = alert_directory.stat()
        mode = stat_result.st_mode & 0o777
        uid = stat_result.st_uid
        gid = stat_result.st_gid

        quarantine = parent / (
            f".{alert_directory.name}.purge-{uuid.uuid4().hex}"
        )
        file_count, directory_count, total_bytes = _directory_usage(alert_directory)

        # Atomic directory swap: future writers immediately see a new empty
        # alert directory while the old contents are safely removed.
        alert_directory.rename(quarantine)
        alert_directory.mkdir(mode=mode or 0o750, parents=False, exist_ok=False)
        try:
            os.chown(alert_directory, uid, gid)
        except PermissionError:
            # Non-root deployments may not be permitted to restore ownership.
            pass
    else:
        alert_directory.mkdir(mode=0o750, parents=True, exist_ok=True)

    try:
        if quarantine is not None and quarantine.exists():
            shutil.rmtree(quarantine)
    except Exception:
        # Restore the old directory only when the new directory is still empty.
        try:
            if alert_directory.exists() and not any(alert_directory.iterdir()):
                alert_directory.rmdir()
                quarantine.rename(alert_directory)
        except Exception:
            pass
        raise

    connection = get_connection()
    alerts_deleted = 0
    domains_deleted = 0
    ingest_rows_deleted = 0

    try:
        connection.execute("BEGIN IMMEDIATE")

        alerts_deleted = int(
            connection.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        )
        domains_deleted = int(
            connection.execute("SELECT COUNT(*) FROM alert_domains").fetchone()[0]
        )

        connection.execute("DELETE FROM alert_domains")
        connection.execute("DELETE FROM alerts")

        directory_prefix = str(alert_directory.resolve()) + os.sep + "%"
        cursor = connection.execute(
            """
            DELETE FROM ingest_state
            WHERE source_path = ? OR source_path LIKE ?
            """,
            (str(alert_directory.resolve()), directory_prefix),
        )
        ingest_rows_deleted = max(int(cursor.rowcount or 0), 0)

        # Restart alert IDs from 1 without affecting sequences for unrelated
        # tables. sqlite_sequence exists because these tables use AUTOINCREMENT.
        try:
            connection.execute(
                "DELETE FROM sqlite_sequence WHERE name IN ('alerts', 'alert_domains')"
            )
        except sqlite3.OperationalError:
            pass

        connection.commit()

        # Reclaim deleted alert pages from the SQLite database.
        try:
            connection.execute("VACUUM")
        except sqlite3.OperationalError:
            pass
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    return {
        "alerts_deleted": alerts_deleted,
        "alert_domains_deleted": domains_deleted,
        "ingest_state_rows_deleted": ingest_rows_deleted,
        "files_deleted": file_count,
        "directories_deleted": directory_count,
        "bytes_deleted": total_bytes,
        "alert_directory": str(alert_directory),
        "completed_at_utc": utc_now(),
    }

