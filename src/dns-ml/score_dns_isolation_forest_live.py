#!/usr/bin/env python3
# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

"""CPU-only live Isolation Forest scoring for Zeek DNS.

This scorer is intentionally isolated from the Autoencoder / heuristic scorer.
It has its own cursor, buffer, lock, diagnostics and alert JSONL file.
It reuses the shared feature_utils.py feature pipeline and the trained Isolation
Forest artifacts created by train_dns_isolation_forest.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import fcntl
import joblib
import numpy as np
import pandas as pd

from feature_utils import (
    FEATURES,
    add_event_features,
    build_dns_windows,
    parse_json_lines,
    summarize_suspicious_domains,
    validate_feature_matrix,
)

MODEL_PATH = Path(os.getenv("DNS_IF_MODEL_PATH", "/data/dns-ml/models/dns_isolation_forest.joblib"))
SCALER_PATH = Path(os.getenv("DNS_IF_SCALER_PATH", "/data/dns-ml/models/dns_isolation_forest_scaler.joblib"))
CONFIG_PATH = Path(os.getenv("DNS_IF_CONFIG_PATH", "/data/dns-ml/models/dns_isolation_forest_config.json"))
TRAINING_SCORES_PATH = Path(os.getenv("DNS_IF_TRAINING_SCORES", "/data/dns-ml/runs/latest_isolation_forest_scores.csv"))

ZEEK_CURRENT_DIR = Path(os.getenv("DNS_ZEEK_CURRENT", "/opt/zeek/logs/current"))
SCORING_HOME = Path(os.getenv("DNS_IF_SCORING_HOME", "/data/dns-ml/isolation_scoring"))
STATUS_PATH = SCORING_HOME / "status.json"
LOG_PATH = SCORING_HOME / "live_scoring.log"
PID_PATH = SCORING_HOME / "scoring.pid"
LOCK_PATH = SCORING_HOME / "scoring.lock"
WINDOWS_PATH = SCORING_HOME / "latest_isolation_scoring_windows.csv"
SHAP_PATH = SCORING_HOME / "latest_isolation_scoring_shap.csv"

STATE_PATH = Path(os.getenv("DNS_IF_STATE_PATH", "/data/dns-ml/state/dns_isolation_forest_log_state.json"))
BUFFER_PATH = Path(os.getenv("DNS_IF_BUFFER_PATH", "/data/dns-ml/state/dns_isolation_forest_event_buffer.jsonl"))
ALERT_PATH = Path(os.getenv("DNS_IF_ALERT_PATH", "/data/dns-ml/alerts/dns_categorization_forest_alerts.jsonl"))

WINDOW_SIZE = os.getenv("DNS_IF_WINDOW_SIZE", "5min")
DEBUG_ENABLED = os.getenv("DNS_IF_SCORING_DEBUG", "1").strip().lower() not in {"0", "false", "no", "off"}
MAX_SHAP_ALERTS = int(os.getenv("DNS_IF_SCORING_SHAP_MAX", "1000"))
SUPPORTED_NAMES = ("dns.log", "dns.jsonl", "dns.json")
SUPPORTED_GLOBS = ("dns*.log", "dns*.jsonl", "dns*.json")

LOGGER = logging.getLogger("dns_isolation_scoring")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_logging() -> None:
    SCORING_HOME.mkdir(parents=True, exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.DEBUG if DEBUG_ENABLED else logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%dT%H:%M:%S%z")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def write_status(state: str, stage: str, progress: int, message: str, **extra: Any) -> None:
    payload = {
        "state": state,
        "stage": stage,
        "progress_percent": max(0, min(int(progress), 100)),
        "message": message,
        "updated_at_utc": utc_now(),
        "pid": os.getpid(),
        "model_type": "IsolationForest",
        "detection_type": "categorization_forest",
        "cpu_only": True,
        "debug_enabled": DEBUG_ENABLED,
        **extra,
    }
    atomic_json(STATUS_PATH, payload)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


class LiveInputPending(FileNotFoundError):
    """The scheduled scorer is healthy, but Zeek has not published DNS input yet."""


def find_live_log() -> Path:
    for name in SUPPORTED_NAMES:
        candidate = ZEEK_CURRENT_DIR / name
        if candidate.is_file():
            return candidate
    found: list[Path] = []
    for pattern in SUPPORTED_GLOBS:
        found.extend(p for p in ZEEK_CURRENT_DIR.glob(pattern) if p.is_file())
    if not found:
        raise LiveInputPending(
            f"No supported live Zeek DNS file found under {ZEEK_CURRENT_DIR}"
        )
    return max(found, key=lambda p: p.stat().st_mtime_ns)


def detect_format(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("#"):
                return "tsv"
            if line.startswith("["):
                return "json_array"
            if line.startswith("{"):
                return "json"
            return "text"
    return "empty"


def read_tsv_header(path: Path) -> list[str]:
    header: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            if raw.startswith("#"):
                header.append(raw)
                continue
            if raw.strip():
                break
    return header


def read_new_lines(path: Path, offset: int) -> tuple[list[str], int]:
    with path.open("rb") as handle:
        handle.seek(max(0, int(offset)))
        payload = handle.read()
        new_offset = handle.tell()
    if not payload:
        return [], new_offset
    text = payload.decode("utf-8", errors="ignore")
    return text.splitlines(keepends=True), new_offset


def safe_committed_offset(path: Path) -> tuple[int, dict[str, Any]]:
    state = read_json(STATE_PATH, {})
    stat = path.stat()
    if str(state.get("path") or "") != str(path):
        return 0, state
    if state.get("inode") is not None and int(state.get("inode")) != int(stat.st_ino):
        return 0, state
    offset = int(state.get("offset", 0) or 0)
    if offset < 0 or offset > stat.st_size:
        return 0, state
    return offset, state


def parse_lines(path: Path, lines: list[str], fmt: str) -> pd.DataFrame:
    if not lines:
        return pd.DataFrame()
    if fmt == "tsv":
        return parse_json_lines(read_tsv_header(path) + lines, keep_raw_line=False)
    if fmt == "json_array":
        # JSON arrays are not append-friendly. Reconstruct the full document.
        return parse_json_lines(lines, keep_raw_line=False)
    return parse_json_lines(lines, keep_raw_line=False)


def load_buffer() -> pd.DataFrame:
    if not BUFFER_PATH.exists() or BUFFER_PATH.stat().st_size == 0:
        return pd.DataFrame()
    try:
        frame = pd.read_json(BUFFER_PATH, lines=True)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        return frame
    except Exception as exc:
        LOGGER.warning("Unable to read Isolation Forest event buffer; ignoring it: %s", exc)
        return pd.DataFrame()


def write_buffer(frame: pd.DataFrame) -> None:
    BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = BUFFER_PATH.with_suffix(BUFFER_PATH.suffix + ".tmp")
    if frame.empty:
        tmp.write_text("", encoding="utf-8")
    else:
        frame.to_json(tmp, orient="records", lines=True, date_format="iso")
    tmp.replace(BUFFER_PATH)


def dedupe_events(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    keys = [key for key in ("uid", "timestamp", "src_ip", "query_clean", "qtype_norm", "rcode_norm") if key in frame.columns]
    return frame.drop_duplicates(subset=keys, keep="last") if keys else frame.drop_duplicates()


def restore_window_metadata(events: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    # Restore source-IP / 5-minute-window metadata if build_dns_windows()
    # returns only the 36 feature columns.
    result = windows.copy().reset_index(drop=True)

    if {"timestamp", "src_ip"}.issubset(result.columns):
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
        result["src_ip"] = result["src_ip"].astype(str)
        if result["timestamp"].isna().any():
            raise ValueError("Isolation scoring window metadata contains invalid timestamps.")
        return result

    if events.empty:
        raise ValueError("Cannot restore Isolation Forest scoring metadata from empty DNS events.")

    required = {"timestamp", "src_ip"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(
            "Cannot restore Isolation Forest scoring metadata; "
            f"DNS events are missing columns: {missing}"
        )

    grouped_obj = events.groupby(
        [pd.Grouper(key="timestamp", freq=WINDOW_SIZE), "src_ip"],
        dropna=False,
        observed=True,
    )
    grouped = grouped_obj.size().reset_index(name="__event_count")

    if len(grouped) != len(result):
        raise ValueError(
            "Isolation scoring metadata reconstruction failed: "
            f"feature rows={len(result):,}, "
            f"source-IP/5-minute groups={len(grouped):,}."
        )

    if "total_queries" in result.columns:
        expected_total = (
            pd.to_numeric(result["total_queries"], errors="coerce")
            .fillna(-1).round().astype("int64").to_numpy()
        )
        actual_total = grouped["__event_count"].astype("int64").to_numpy()
        if not np.array_equal(expected_total, actual_total):
            mismatch = np.flatnonzero(expected_total != actual_total)
            first = int(mismatch[0]) if len(mismatch) else -1
            raise ValueError(
                "Isolation scoring metadata alignment check failed for "
                f"total_queries at row {first}. Refusing to attach timestamp/src_ip."
            )

    if "unique_queries" in result.columns and "query_clean" in events.columns:
        unique_counts = grouped_obj["query_clean"].nunique().reset_index(name="__unique_queries")
        expected_unique = (
            pd.to_numeric(result["unique_queries"], errors="coerce")
            .fillna(-1).round().astype("int64").to_numpy()
        )
        actual_unique = unique_counts["__unique_queries"].astype("int64").to_numpy()
        if not np.array_equal(expected_unique, actual_unique):
            mismatch = np.flatnonzero(expected_unique != actual_unique)
            first = int(mismatch[0]) if len(mismatch) else -1
            raise ValueError(
                "Isolation scoring metadata alignment check failed for "
                f"unique_queries at row {first}. Refusing to attach timestamp/src_ip."
            )

    timestamps = pd.to_datetime(grouped["timestamp"], utc=True, errors="coerce")
    if timestamps.isna().any():
        raise ValueError("Isolation scoring metadata reconstruction produced invalid timestamps.")

    result.insert(0, "src_ip", grouped["src_ip"].astype(str).to_numpy())
    result.insert(0, "timestamp", timestamps.to_numpy())

    LOGGER.debug(
        "Restored missing timestamp/src_ip metadata for %d Isolation Forest source-IP/5-minute windows.",
        len(result),
    )
    return result

def load_model_artifacts() -> tuple[Any, Any, dict[str, Any]]:
    missing = [str(p) for p in (MODEL_PATH, SCALER_PATH, CONFIG_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Isolation Forest learning artifacts: " + ", ".join(missing))
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    config = read_json(CONFIG_PATH, {})
    expected = list(config.get("features") or FEATURES)
    if expected != list(FEATURES):
        raise ValueError("Isolation Forest feature order does not match the shared feature_utils.py FEATURES list.")
    if int(getattr(model, "n_features_in_", -1)) != len(FEATURES):
        raise ValueError("Isolation Forest model feature count is incompatible with the current 36-feature pipeline.")
    if int(getattr(scaler, "n_features_in_", -1)) != len(FEATURES):
        raise ValueError("Isolation Forest scaler feature count is incompatible with the current 36-feature pipeline.")
    return model, scaler, config


def training_score_reference() -> np.ndarray:
    if not TRAINING_SCORES_PATH.exists():
        return np.asarray([], dtype=float)
    try:
        frame = pd.read_csv(TRAINING_SCORES_PATH, usecols=["anomaly_score"])
        values = pd.to_numeric(frame["anomaly_score"], errors="coerce").dropna().to_numpy(dtype=float)
        return values[np.isfinite(values)]
    except Exception:
        return np.asarray([], dtype=float)


def score_percentile(score: float, reference: np.ndarray) -> float:
    if reference.size == 0:
        return float("nan")
    return float(100.0 * np.mean(reference <= float(score)))


def severity_from_percentile(percentile: float, score: float) -> tuple[str, int]:
    if np.isfinite(percentile):
        threat_score = int(max(0, min(100, round(percentile))))
        if percentile >= 99.9:
            return "critical", threat_score
        if percentile >= 99.0:
            return "high", threat_score
        if percentile >= 97.5:
            return "medium", threat_score
        return "low", threat_score
    # Fallback when the training score distribution is unavailable.
    threat_score = int(max(50, min(100, round(50 + 250 * max(0.0, score)))))
    return ("high" if threat_score >= 75 else "medium"), threat_score


def shap_for_anomalies(model: Any, x_scaled: np.ndarray, anomaly_indices: np.ndarray, windows: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
    if anomaly_indices.size == 0:
        return {}
    selected = anomaly_indices[:MAX_SHAP_ALERTS]
    output: dict[int, list[dict[str, Any]]] = {}
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        raw = np.asarray(explainer.shap_values(x_scaled[selected]), dtype=float)
        if raw.ndim == 3 and raw.shape[-1] == 1:
            raw = raw[..., 0]
        if raw.shape != (len(selected), len(FEATURES)):
            raise ValueError(f"Unexpected SHAP shape: {raw.shape}")
        values = -raw
        for local, global_index in enumerate(selected):
            order = np.argsort(np.abs(values[local]))[::-1][:8]
            output[int(global_index)] = [
                {
                    "feature": FEATURES[int(j)],
                    "feature_value": float(windows.iloc[int(global_index)][FEATURES[int(j)]]),
                    "shap_value": float(values[local, int(j)]),
                    "direction": "toward_anomaly" if values[local, int(j)] > 0 else "toward_normal",
                    "explanation_method": "tree_shap",
                }
                for j in order
            ]
    except Exception as exc:
        LOGGER.warning("SHAP alert explanation unavailable; using scaled-deviation fallback: %s", exc)

    for global_index in anomaly_indices:
        index = int(global_index)
        if index in output:
            continue
        order = np.argsort(np.abs(x_scaled[index]))[::-1][:8]
        output[index] = [
            {
                "feature": FEATURES[int(j)],
                "feature_value": float(windows.iloc[index][FEATURES[int(j)]]),
                "scaled_deviation": float(x_scaled[index, int(j)]),
                "direction": "unusual_magnitude",
                "explanation_method": "scaled_deviation_fallback",
            }
            for j in order
        ]
    return output


def possible_causes(top_features: list[dict[str, Any]]) -> list[str]:
    names = {str(item.get("feature")) for item in top_features}
    causes: list[str] = []
    if names & {"avg_entropy", "max_entropy", "high_entropy_rate", "avg_query_len", "max_query_len", "long_query_rate", "long_label_rate"}:
        causes.append("Unusual domain length, label structure, or entropy compared with the learned DNS baseline.")
    if names & {"nxdomain_rate", "servfail_rate", "noerror_rate"}:
        causes.append("Unusual DNS response-code behavior compared with the learned baseline.")
    if names & {"txt_rate", "mx_rate", "ptr_rate", "srv_rate"}:
        causes.append("Unusual DNS record-type mix compared with the learned baseline.")
    if names & {"unique_dns_servers", "avg_rtt", "max_rtt"}:
        causes.append("Resolver-selection or DNS response-time behavior differs from the learned baseline.")
    if names & {"ttl_avg", "ttl_min", "low_ttl_rate", "unique_answer_ips", "answer_count_avg", "answer_count_max"}:
        causes.append("DNS answer, TTL, or returned-IP behavior differs from the learned baseline.")
    if names & {"total_queries", "unique_queries", "unique_query_ratio", "repeated_query_ratio"}:
        causes.append("DNS query volume or repetition pattern differs from the learned baseline.")
    return causes or ["The complete 36-feature DNS window is isolated from the learned baseline."]


def context_domains(events: pd.DataFrame, src_ip: str, window_start: Any) -> list[dict[str, Any]]:
    domains = summarize_suspicious_domains(events, src_ip, window_start, WINDOW_SIZE, top_n=10)
    if domains:
        return domains

    start = pd.Timestamp(window_start)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    end = start + pd.Timedelta(WINDOW_SIZE)
    subset = events[(events["src_ip"].eq(src_ip)) & (events["timestamp"].ge(start)) & (events["timestamp"].lt(end))].copy()
    if subset.empty or "query_clean" not in subset.columns:
        return []
    counts = subset["query_clean"].replace("", np.nan).dropna().value_counts().head(10)
    return [
        {
            "query": str(query),
            "count": int(count),
            "reasons": ["isolation_window_context"],
            "suspicion_score": 0,
            "max_entropy": 0.0,
            "avg_entropy": 0.0,
            "max_query_len": 0.0,
            "max_label_len": 0.0,
            "ttl_min": 0.0,
            "ttl_avg": 0.0,
            "answer_count_max": 0.0,
            "unique_answer_ips": 0,
        }
        for query, count in counts.items()
    ]


def stable_alert_id(timestamp: Any, src_ip: str) -> str:
    basis = f"categorization_forest|{pd.Timestamp(timestamp).isoformat()}|{src_ip}"
    import hashlib

    return "CATF-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20].upper()


def append_alerts(alerts: list[dict[str, Any]]) -> int:
    if not alerts:
        return 0
    ALERT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Keep full-log rescoring idempotent. CATF-* IDs are stable for the same
    # source-IP/five-minute window, so an existing alert is not appended twice.
    existing_ids: set[str] = set()
    if ALERT_PATH.exists():
        try:
            with ALERT_PATH.open("r", encoding="utf-8", errors="ignore") as existing:
                for line in existing:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        value = payload.get("id") or payload.get("alert_uid")
                        if value:
                            existing_ids.add(str(value))
        except OSError:
            pass

    written = 0
    with ALERT_PATH.open("a", encoding="utf-8") as handle:
        for alert in alerts:
            alert_id = str(alert.get("id") or alert.get("alert_uid") or "")
            if alert_id and alert_id in existing_ids:
                continue
            handle.write(json.dumps(alert, default=str, separators=(",", ":")) + "\n")
            if alert_id:
                existing_ids.add(alert_id)
            written += 1
        handle.flush()
        os.fsync(handle.fileno())
    return written


def model_parameter_summary(model: Any, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "contamination": config.get("contamination", getattr(model, "contamination", None)),
        "n_estimators": config.get("n_estimators", getattr(model, "n_estimators", None)),
        "max_samples": config.get("max_samples", getattr(model, "max_samples_", getattr(model, "max_samples", None))),
        "max_features": config.get("max_features", getattr(model, "max_features", 1.0)),
        "feature_count": len(FEATURES),
        "window_size": config.get("window_size", WINDOW_SIZE),
        "anomaly_threshold": float(config.get("anomaly_threshold", 0.0) or 0.0),
    }


def write_scoring_windows(frame: pd.DataFrame, mode: str) -> None:
    SCORING_HOME.mkdir(parents=True, exist_ok=True)
    if mode == "full":
        frame.to_csv(WINDOWS_PATH, index=False)
        return
    if not WINDOWS_PATH.exists() or WINDOWS_PATH.stat().st_size == 0:
        frame.to_csv(WINDOWS_PATH, index=False)
        return
    try:
        existing = pd.read_csv(WINDOWS_PATH)
        combined = pd.concat([existing, frame], ignore_index=True, sort=False)
        keys = [key for key in ("timestamp", "src_ip") if key in combined.columns]
        if keys:
            combined = combined.drop_duplicates(keys, keep="last")
        combined.to_csv(WINDOWS_PATH, index=False)
    except Exception:
        frame.to_csv(WINDOWS_PATH, index=False)


def process(mode: str) -> dict[str, Any]:
    path = find_live_log()
    fmt = detect_format(path)
    model, scaler, config = load_model_artifacts()
    parameters = model_parameter_summary(model, config)
    threshold = float(parameters["anomaly_threshold"])

    write_status("running", "Reading Zeek DNS", 10, f"Reading {path}", run_mode=mode, live_log=str(path), **parameters)
    LOGGER.info("DNS CATEGORIZATION SCORING — ISOLATION FOREST")
    LOGGER.info("Mode: %s | CPU only: yes | Debug: %s", mode, DEBUG_ENABLED)
    LOGGER.info("Live DNS file: %s | format=%s", path, fmt)
    LOGGER.info("Critical Isolation Forest parameters: contamination=%s, n_estimators=%s, max_samples=%s, max_features=%s",
                parameters["contamination"], parameters["n_estimators"], parameters["max_samples"], parameters["max_features"])

    stat = path.stat()
    if mode == "full":
        offset = 0
        old_buffer = pd.DataFrame()
        lines, new_offset = read_new_lines(path, 0)
    else:
        offset, _ = safe_committed_offset(path)
        old_buffer = load_buffer()
        lines, new_offset = read_new_lines(path, offset)

    if not lines and old_buffer.empty:
        write_status("no_data", "No new records", 100, "No new DNS records were available after the categorization cursor.", run_mode=mode, alerts_written=0, windows_scored=0, **parameters)
        return {"alerts_written": 0, "windows_scored": 0, "new_records": 0}

    raw = parse_lines(path, lines, fmt)
    new_events = add_event_features(raw) if not raw.empty else pd.DataFrame()
    LOGGER.debug("New parsed DNS events: %d; previous categorization buffer: %d", len(new_events), len(old_buffer))

    if old_buffer.empty:
        combined_events = new_events
    elif new_events.empty:
        combined_events = old_buffer
    else:
        combined_events = pd.concat([old_buffer, new_events], ignore_index=True, sort=False)
    combined_events = dedupe_events(combined_events)
    if combined_events.empty:
        # Commit only parsed/consumed bytes; nothing useful was found.
        if mode != "full":
            atomic_json(STATE_PATH, {"path": str(path), "inode": stat.st_ino, "offset": new_offset, "updated_at_utc": utc_now()})
        write_status("no_data", "No valid DNS records", 100, "No valid Zeek DNS records were available for Isolation Forest scoring.", run_mode=mode, alerts_written=0, windows_scored=0, **parameters)
        return {"alerts_written": 0, "windows_scored": 0, "new_records": 0}

    write_status("running", "Preparing 5-minute windows", 30, "Building source-IP five-minute windows.", run_mode=mode, parsed_events=len(combined_events), **parameters)
    combined_events["timestamp"] = pd.to_datetime(combined_events["timestamp"], utc=True, errors="coerce")
    combined_events = combined_events.dropna(subset=["timestamp", "src_ip"])

    if mode == "full":
        completed_events = combined_events.copy()
        pending_events = pd.DataFrame(columns=combined_events.columns)
    else:
        current_window_start = pd.Timestamp.now(tz="UTC").floor(WINDOW_SIZE)
        completed_events = combined_events[combined_events["timestamp"] < current_window_start].copy()
        pending_events = combined_events[combined_events["timestamp"] >= current_window_start].copy()

    if completed_events.empty:
        if mode != "full":
            write_buffer(pending_events)
            atomic_json(STATE_PATH, {"path": str(path), "inode": stat.st_ino, "offset": new_offset, "updated_at_utc": utc_now()})
        write_status(
            "no_data",
            "Waiting for completed windows",
            100,
            "No new completed five-minute DNS windows were available for categorization scoring.",
            run_mode=mode,
            alerts_written=0,
            windows_scored=0,
            buffered_events=len(pending_events),
            **parameters,
        )
        return {"alerts_written": 0, "windows_scored": 0, "new_records": len(new_events)}

    windows = build_dns_windows(completed_events, window_size=WINDOW_SIZE)
    windows = restore_window_metadata(completed_events, windows)
    windows = validate_feature_matrix(windows, FEATURES)
    windows = restore_window_metadata(completed_events, windows)
    if windows.empty:
        raise ValueError("Feature extraction produced zero Isolation Forest scoring windows.")

    write_status("running", "Isolation Forest scoring", 55, f"Scoring {len(windows):,} five-minute windows.", run_mode=mode, windows_scored=len(windows), **parameters)
    LOGGER.debug(
        "Metadata ready for Isolation Forest scoring: windows=%d, first_window=%s, last_window=%s, source_ips=%d",
        len(windows),
        str(windows["timestamp"].min()),
        str(windows["timestamp"].max()),
        int(windows["src_ip"].nunique()),
    )
    x = windows[FEATURES].to_numpy(dtype=np.float64, copy=True)
    x_scaled = scaler.transform(x)
    decision = np.asarray(model.decision_function(x_scaled), dtype=float)
    anomaly_score = -decision
    prediction = np.asarray(model.predict(x_scaled), dtype=int)
    is_anomaly = prediction == -1
    score_margin = anomaly_score - threshold

    scored = windows.copy()
    scored["decision_function"] = decision
    scored["anomaly_score"] = anomaly_score
    scored["isolation_threshold"] = threshold
    scored["score_margin"] = score_margin
    scored["predicted_label"] = np.where(is_anomaly, "anomaly", "normal")
    scored["detection_type"] = "categorization_forest"
    scored["scoring_mode"] = mode

    reference = training_score_reference()
    scored["score_percentile"] = [score_percentile(value, reference) for value in anomaly_score]

    anomaly_indices = np.flatnonzero(is_anomaly)
    write_status("running", "Explaining anomalies", 72, f"Explaining {len(anomaly_indices):,} Isolation Forest anomalies.", run_mode=mode, windows_scored=len(windows), anomaly_windows=len(anomaly_indices), **parameters)
    explanations = shap_for_anomalies(model, x_scaled, anomaly_indices, windows)

    shap_rows: list[dict[str, Any]] = []
    for index, items in explanations.items():
        for item in items:
            shap_rows.append({"timestamp": str(windows.iloc[index]["timestamp"]), "src_ip": str(windows.iloc[index]["src_ip"]), "anomaly_score": float(anomaly_score[index]), **item})
    pd.DataFrame(shap_rows).to_csv(SHAP_PATH, index=False)

    alerts: list[dict[str, Any]] = []
    for index in anomaly_indices:
        row = windows.iloc[int(index)]
        timestamp = pd.Timestamp(row["timestamp"])
        src_ip = str(row["src_ip"])
        percentile = float(scored.iloc[int(index)]["score_percentile"])
        severity, threat_score = severity_from_percentile(percentile, float(anomaly_score[int(index)]))
        top_features = explanations.get(int(index), [])
        domains = context_domains(completed_events, src_ip, timestamp)
        alert_id = stable_alert_id(timestamp, src_ip)
        alerts.append(
            {
                "id": alert_id,
                "alert_uid": alert_id,
                "timestamp": timestamp.isoformat(),
                "window_end": (timestamp + pd.Timedelta(WINDOW_SIZE)).isoformat(),
                "src_ip": src_ip,
                "status": "New",
                "alert_name": "DNS categorization Isolation Forest anomaly",
                "alert_type": "categorization_forest",
                "detection_type": "categorization_forest",
                "model_name": "dns_isolation_forest",
                "window_size": WINDOW_SIZE,
                "severity": severity,
                "threat_score": threat_score,
                "isolation_score": float(anomaly_score[int(index)]),
                "isolation_threshold": threshold,
                "score_margin": float(score_margin[int(index)]),
                "score_percentile": percentile if np.isfinite(percentile) else None,
                "decision_function": float(decision[int(index)]),
                "predicted_label": "anomaly",
                "contamination": parameters["contamination"],
                "n_estimators": parameters["n_estimators"],
                "max_samples": parameters["max_samples"],
                "max_features": parameters["max_features"],
                "feature_count": len(FEATURES),
                "detection_methods": {"categorization_forest": True, "isolation_forest": True},
                "top_contributing_features": top_features,
                "possible_causes": possible_causes(top_features),
                "suspicious_domains": domains,
                "feature_values": {feature: float(row[feature]) for feature in FEATURES},
                "scoring_mode": mode,
                "generated_at_utc": utc_now(),
            }
        )

    write_status("running", "Saving results", 90, "Saving categorization scores and alerts.", run_mode=mode, windows_scored=len(windows), anomaly_windows=len(anomaly_indices), **parameters)
    write_scoring_windows(scored, mode)
    alerts_written = append_alerts(alerts)

    # State and pending buffer are committed only after successful scoring/output.
    if mode != "full":
        write_buffer(pending_events)
        atomic_json(
            STATE_PATH,
            {
                "path": str(path),
                "inode": stat.st_ino,
                "offset": new_offset,
                "updated_at_utc": utc_now(),
                "last_windows_scored": len(windows),
                "last_alerts_written": alerts_written,
            },
        )

    LOGGER.info("Scored %d Isolation Forest windows; anomalies=%d; categorization alerts written=%d", len(windows), int(is_anomaly.sum()), alerts_written)
    write_status(
        "finished",
        "Completed",
        100,
        f"Isolation Forest categorization scoring completed: {len(windows):,} windows, {int(is_anomaly.sum()):,} anomalies.",
        run_mode=mode,
        completed_at_utc=utc_now(),
        windows_scored=len(windows),
        anomaly_windows=int(is_anomaly.sum()),
        alerts_written=alerts_written,
        alert_jsonl=str(ALERT_PATH),
        scoring_windows=str(WINDOWS_PATH),
        **parameters,
    )
    return {"alerts_written": alerts_written, "windows_scored": len(windows), "anomaly_windows": int(is_anomaly.sum())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score live Zeek DNS with the learned Isolation Forest.")
    parser.add_argument("--mode", choices=("once", "full"), default="once")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging()
    SCORING_HOME.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        LOGGER.warning("Another Isolation Forest scoring pass is already running.")
        write_status("busy", "Already running", 0, "Another Isolation Forest categorization scoring pass is already running.", run_mode=args.mode)
        return 3

    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
    try:
        process(args.mode)
        return 0
    except LiveInputPending as exc:
        LOGGER.info("Isolation Forest is waiting for Zeek DNS input: %s", exc)
        write_status(
            "waiting_input",
            "Waiting for DNS input",
            0,
            "No live Zeek DNS records have been published yet. "
            "The next scheduled interval will check again automatically.",
            run_mode=args.mode,
            live_directory=str(ZEEK_CURRENT_DIR),
            alerts_written=0,
            windows_scored=0,
            error=None,
        )
        return 0
    except Exception as exc:
        LOGGER.error("Isolation Forest categorization scoring failed: %s: %s", type(exc).__name__, exc)
        LOGGER.debug("%s", traceback.format_exc())
        write_status("failed", "Failed", 100, f"Isolation Forest categorization scoring failed: {type(exc).__name__}: {exc}", run_mode=args.mode, error=str(exc))
        return 1
    finally:
        try:
            PID_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
