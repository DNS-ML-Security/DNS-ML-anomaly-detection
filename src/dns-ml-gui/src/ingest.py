# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, Tuple

from src.db import get_connection, init_db, utc_now
from src.settings import ALERT_JSONL_PATH


def make_alert_uid(alert: Dict) -> str:
    explicit = alert.get("alert_uid") or alert.get("id")
    if explicit:
        return str(explicit)

    methods = alert.get("detection_methods") or {}
    if not isinstance(methods, dict):
        methods = {}
    is_autoencoder = bool(
        str(alert.get("alert_type") or "") == "autoencoder_anomaly"
        or str(alert.get("detection_type") or "") == "deep_learning_autoencoder"
        or methods.get("autoencoder")
    )
    has_heuristic = any(
        bool(methods.get(name))
        for name in (
            "heuristic_dga",
            "heuristic_tunnel",
            "heuristic_fastflux",
            "heuristic_resolver_failure",
            "heuristic_rogue_resolver",
        )
    )
    if is_autoencoder and not has_heuristic:
        basis = f"autoencoder|{str(alert.get('timestamp') or '')}|{str(alert.get('src_ip') or '')}"
        return "AE-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20].upper()

    stable_fields = {
        "timestamp": alert.get("timestamp"),
        "window_end": alert.get("window_end"),
        "src_ip": alert.get("src_ip"),
        "alert_type": alert.get("alert_type"),
        "reconstruction_error": alert.get("reconstruction_error"),
        "isolation_score": alert.get("isolation_score"),
        "detection_methods": alert.get("detection_methods"),
        "suspicious_domains": (alert.get("suspicious_domains") or [])[:5],
    }
    raw = json.dumps(stable_fields, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def calc_threat_score(alert: Dict) -> int:
    methods = alert.get("detection_methods") or {}

    # Isolation Forest categorization scoring supplies a percentile-based
    # threat score. Preserve it rather than applying Autoencoder heuristics.
    if alert.get("alert_type") == "categorization_forest" or methods.get("categorization_forest"):
        try:
            explicit = int(round(float(alert.get("threat_score", 0) or 0)))
            if explicit > 0:
                return max(0, min(explicit, 100))
        except (TypeError, ValueError):
            pass
        try:
            percentile = float(alert.get("score_percentile", 0) or 0)
            if percentile > 0:
                return max(0, min(int(round(percentile)), 100))
        except (TypeError, ValueError):
            pass
        return 50

    score = 0
    score += 20 if methods.get("autoencoder") else 0
    score += 35 if methods.get("heuristic_dga") else 0
    score += 40 if methods.get("heuristic_tunnel") else 0
    score += 30 if methods.get("heuristic_fastflux") else 0
    score += 15 if methods.get("heuristic_resolver_failure") else 0
    score += 25 if methods.get("heuristic_rogue_resolver") else 0
    score += 15 if alert.get("suspicious_domains") else 0

    try:
        error = float(alert.get("reconstruction_error", 0))
        threshold = float(alert.get("threshold", 0))
        if threshold > 0 and error >= threshold * 2:
            score += 15
        elif threshold > 0 and error >= threshold * 1.5:
            score += 8
    except (TypeError, ValueError):
        pass
    return min(score, 100)

def normalize_severity(score: int, existing: str = None) -> str:
    if score >= 75:
        return "high"
    if score >= 50:
        return "medium"
    if score >= 25:
        return "low"
    return existing or "info"


def iter_jsonl_from_offset(path: Path, offset: int = 0) -> Tuple[Iterable[Dict], int]:
    records = []

    if not path.exists():
        return records, offset

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(offset)

        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        new_offset = f.tell()

    return records, new_offset


def get_ingest_state(conn, source_path: str):
    row = conn.execute(
        "SELECT inode, offset FROM ingest_state WHERE source_path = ?",
        (source_path,)
    ).fetchone()

    if not row:
        return None, 0

    return row["inode"], int(row["offset"] or 0)


def save_ingest_state(conn, source_path: str, inode, offset: int, error: str = None):
    conn.execute(
        """
        INSERT INTO ingest_state
        (source_path, inode, offset, last_ingested_at_utc, last_error)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source_path) DO UPDATE SET
            inode = excluded.inode,
            offset = excluded.offset,
            last_ingested_at_utc = excluded.last_ingested_at_utc,
            last_error = excluded.last_error
        """,
        (source_path, inode, offset, utc_now(), error)
    )


def insert_alert(connection, alert: Dict) -> bool:
    alert_uid = make_alert_uid(alert)
    threat_score = calc_threat_score(alert)
    severity = normalize_severity(threat_score, alert.get("severity"))

    cursor = connection.execute(
        """
        INSERT OR IGNORE INTO alerts
        (alert_uid, alert_name, alert_type, timestamp, window_end, src_ip,
         severity, threat_score, status, model_name, window_size,
         reconstruction_error, threshold, detection_methods_json,
         top_contributing_features_json, possible_causes_json, raw_json,
         isolation_score, isolation_threshold, score_margin, score_percentile,
         contamination, n_estimators, max_samples, max_features,
         created_at_utc, updated_at_utc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            alert_uid,
            alert.get("alert_name", "DNS anomaly detected"),
            alert.get("alert_type", "anomaly_candidate"),
            str(alert.get("timestamp", "")),
            str(alert.get("window_end", "")),
            str(alert.get("src_ip", "")),
            severity,
            threat_score,
            str(alert.get("status", "New") or "New"),
            alert.get("model_name", ""),
            alert.get("window_size", ""),
            float(alert.get("reconstruction_error", 0) or 0),
            float(alert.get("threshold", 0) or 0),
            json.dumps(alert.get("detection_methods") or {}),
            json.dumps(alert.get("top_contributing_features") or []),
            json.dumps(alert.get("possible_causes") or []),
            json.dumps(alert),
            float(alert.get("isolation_score", 0) or 0),
            float(alert.get("isolation_threshold", 0) or 0),
            float(alert.get("score_margin", 0) or 0),
            float(alert.get("score_percentile", 0) or 0),
            float(alert.get("contamination", 0) or 0),
            int(alert.get("n_estimators", 0) or 0),
            str(alert.get("max_samples", "")),
            str(alert.get("max_features", "")),
            utc_now(),
            utc_now(),
        ),
    )
    inserted = cursor.rowcount > 0

    if inserted:
        for domain in alert.get("suspicious_domains") or []:
            connection.execute(
                """
                INSERT INTO alert_domains
                (alert_uid, query, count, reasons_json, suspicion_score,
                 max_entropy, avg_entropy, max_query_len, max_label_len,
                 ttl_min, ttl_avg, answer_count_max, unique_answer_ips,
                 created_at_utc)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert_uid,
                    domain.get("query", ""),
                    int(domain.get("count", 0) or 0),
                    json.dumps(domain.get("reasons") or []),
                    int(domain.get("suspicion_score", 0) or 0),
                    float(domain.get("max_entropy", 0) or 0),
                    float(domain.get("avg_entropy", 0) or 0),
                    float(domain.get("max_query_len", 0) or 0),
                    float(domain.get("max_label_len", 0) or 0),
                    float(domain.get("ttl_min", 0) or 0),
                    float(domain.get("ttl_avg", 0) or 0),
                    float(domain.get("answer_count_max", 0) or 0),
                    int(domain.get("unique_answer_ips", 0) or 0),
                    utc_now(),
                ),
            )
    return inserted

def ingest_alert_file(path: Path = ALERT_JSONL_PATH, incremental: bool = True) -> Dict:
    init_db()
    path = Path(path)
    conn = get_connection()

    inserted = 0
    total = 0

    try:
        if not path.exists():
            save_ingest_state(conn, str(path), None, 0, f"File not found: {path}")
            conn.commit()
            return {"source": str(path), "total_read": 0, "inserted": 0, "error": f"File not found: {path}"}

        stat = os.stat(path)
        inode = stat.st_ino

        previous_inode, previous_offset = get_ingest_state(conn, str(path))

        if not incremental or previous_inode != inode:
            offset = 0
        else:
            offset = previous_offset

        records, new_offset = iter_jsonl_from_offset(path, offset)

        for alert in records:
            total += 1
            if insert_alert(conn, alert):
                inserted += 1

        save_ingest_state(conn, str(path), inode, new_offset, None)
        conn.commit()

        return {"source": str(path), "total_read": total, "inserted": inserted, "error": None}

    except Exception as e:
        save_ingest_state(conn, str(path), None, 0, str(e))
        conn.commit()
        return {"source": str(path), "total_read": total, "inserted": inserted, "error": str(e)}

    finally:
        conn.close()
