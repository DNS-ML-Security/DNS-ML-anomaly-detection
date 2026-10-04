#!/usr/bin/env python3
# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

"""Incrementally score Zeek dns.log with the DNS Autoencoder only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
import textwrap
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from feature_utils import (
    FEATURES,
    add_event_features,
    build_dns_windows,
    parse_utc_timestamps,
    calculate_reconstruction_error,
    classify_severity,
    parse_json_lines,
    summarize_suspicious_domains,
    validate_feature_matrix,
)


ZEEK_DNS_LOG = Path(os.getenv("ZEEK_DNS_LOG", "/opt/zeek/logs/current/dns.log"))
MODEL_DIR = Path(os.getenv("DNS_MODEL_DIR", "/data/dns-ml/models"))
ALERT_DIR = Path(os.getenv("DNS_ALERT_DIR", "/data/dns-ml/alerts"))
STATE_DIR = Path(os.getenv("DNS_STATE_DIR", "/data/dns-ml/state"))
LOG_DIR = Path(os.getenv("DNS_LOG_DIR", "/data/dns-ml/logs"))

MODEL_PATH = MODEL_DIR / "dns_autoencoder.keras"
SCALER_PATH = MODEL_DIR / "dns_scaler.joblib"
CONFIG_PATH = MODEL_DIR / "dns_autoencoder_config.json"
STATE_PATH = STATE_DIR / "dns_log_state.json"
BUFFER_PATH = STATE_DIR / "dns_event_buffer.jsonl"
ALERT_PATH = ALERT_DIR / "dns_alerts.jsonl"

SCORING_DIR = Path(os.getenv("DNS_SCORING_DIR", "/data/dns-ml/scoring"))
SCORING_STATUS_PATH = Path(
    os.getenv("DNS_SCORING_STATUS_PATH", str(SCORING_DIR / "status.json"))
)
SCORING_WINDOWS_PATH = Path(
    os.getenv(
        "DNS_SCORING_WINDOWS_PATH",
        str(SCORING_DIR / "latest_scoring_windows.csv"),
    )
)
SCORING_RESIDUALS_PATH = Path(
    os.getenv(
        "DNS_SCORING_RESIDUALS_PATH",
        str(SCORING_DIR / "latest_feature_residuals.csv"),
    )
)
SCORING_LATENT_PATH = Path(
    os.getenv(
        "DNS_SCORING_LATENT_PATH",
        str(SCORING_DIR / "latest_latent_vectors.csv"),
    )
)
SCORING_HISTORY_LIMIT = int(os.getenv("DNS_SCORING_HISTORY_LIMIT", "5000"))

POLL_SECONDS = int(os.getenv("DNS_SCORE_POLL_SECONDS", "10"))
DEBUG = os.getenv("DNS_DEBUG", "0") == "1"
APPROVED_DNS_SERVERS = {
    item.strip()
    for item in os.getenv("DNS_APPROVED_SERVERS", "").split(",")
    if item.strip()
}
ALERT_ON_AUTOENCODER_ONLY = os.getenv("DNS_ALERT_ON_AUTOENCODER_ONLY", "1") == "1"

_ALERT_FINGERPRINT_CACHE: set[str] | None = None


def stable_autoencoder_alert_id(timestamp, src_ip: str) -> str:
    basis = f"autoencoder|{pd.Timestamp(timestamp).isoformat()}|{str(src_ip)}"
    return "AE-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20].upper()

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def trace(message: str) -> None:
    print(f"[{now_utc()}] [+] {message}", flush=True)


def debug(message: str) -> None:
    if DEBUG:
        print(f"[{now_utc()}] [DEBUG] {message}", flush=True)


def warn(message: str) -> None:
    print(f"[{now_utc()}] [!] {message}", flush=True)


def print_information_table(
    title: str,
    rows: list[tuple[str, object]],
    *,
    label_width: int = 31,
    value_width: int = 88,
) -> None:
    """Print a readable two-column CLI table without extra packages."""
    border = "+-" + "-" * label_width + "-+-" + "-" * value_width + "-+"
    print("", flush=True)
    print(title, flush=True)
    print(border, flush=True)
    print(
        f"| {'Setting / measurement':<{label_width}} "
        f"| {'Current value':<{value_width}} |",
        flush=True,
    )
    print(border, flush=True)
    for label, raw_value in rows:
        label_text = str(label)
        value_text = str(raw_value) if raw_value is not None else "N/A"
        label_lines = textwrap.wrap(label_text, width=label_width) or [""]
        value_lines = textwrap.wrap(
            value_text, width=value_width, break_long_words=False, break_on_hyphens=False
        ) or [""]
        height = max(len(label_lines), len(value_lines))
        for index in range(height):
            left = label_lines[index] if index < len(label_lines) else ""
            right = value_lines[index] if index < len(value_lines) else ""
            print(
                f"| {left:<{label_width}} | {right:<{value_width}} |",
                flush=True,
            )
        print(border, flush=True)


def load_json(path: Path, default):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def save_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("rb") as handle:
            return sum(block.count(b"\n") for block in iter(lambda: handle.read(1024 * 1024), b""))
    except OSError:
        return 0


def _path_readiness(path: Path) -> str:
    if path.is_file():
        return f"READY - {path}"
    return f"MISSING - {path}"


def print_runtime_enablement_summary(run_once: bool, full_log: bool = False) -> None:
    """Restore the pre-GUI runtime and ML enablement information."""
    config = load_json(CONFIG_PATH, {})
    state = load_json(STATE_PATH, {})
    model_ready = all(path.is_file() for path in (MODEL_PATH, SCALER_PATH, CONFIG_PATH))
    log_ready = ZEEK_DNS_LOG.is_file() and os.access(ZEEK_DNS_LOG, os.R_OK)
    offset = int(state.get("offset", 0) or 0)
    file_size = ZEEK_DNS_LOG.stat().st_size if ZEEK_DNS_LOG.exists() else 0
    cursor_text = (
        f"{offset:,} of {file_size:,} bytes"
        if state.get("path") == str(ZEEK_DNS_LOG)
        else "No committed cursor for this file"
    )
    threshold = config.get("threshold")
    threshold_text = f"{float(threshold):.10f}" if threshold is not None else "Not available"
    rows = [
        (
            "Execution mode",
            "ONE-TIME FULL-LOG scoring from byte zero"
            if full_log
            else ("ONE-TIME incremental scoring pass" if run_once else f"CONTINUOUS polling every {POLL_SECONDS} seconds"),
        ),
        ("Live DNS logging input", "ENABLED / readable" if log_ready else "UNAVAILABLE / unreadable"),
        ("Machine-learning scoring", "ENABLED" if model_ready else "DISABLED - train the Autoencoder first"),
        ("Zeek DNS input", _path_readiness(ZEEK_DNS_LOG)),
        ("Autoencoder model", _path_readiness(MODEL_PATH)),
        ("Feature scaler", _path_readiness(SCALER_PATH)),
        ("Model configuration", _path_readiness(CONFIG_PATH)),
        ("Model name", config.get("model_name", "DNS Autoencoder")),
        ("Learning feature count", config.get("input_dim", len(FEATURES))),
        ("Configured time window", config.get("window_size", "5min")),
        ("Learned anomaly threshold", threshold_text),
        ("Autoencoder anomaly detection", "ENABLED" if model_ready else "DISABLED"),
        ("DNS heuristic detection", "DISABLED - moved to separate DNS heuristics scoring pipeline"),
        ("Autoencoder alerting", "ENABLED - Autoencoder only"),
        ("Approved DNS resolvers", ", ".join(sorted(APPROVED_DNS_SERVERS)) or "None configured"),
        (
            "Committed Zeek read cursor",
            f"IGNORED and preserved - {cursor_text}" if full_log else cursor_text,
        ),
        (
            "Buffered incomplete records",
            f"IGNORED and preserved - {_line_count(BUFFER_PATH):,}" if full_log else f"{_line_count(BUFFER_PATH):,}",
        ),
        ("Alert output", str(ALERT_PATH)),
        ("Scored-window diagnostics", str(SCORING_WINDOWS_PATH)),
        ("Feature residual diagnostics", str(SCORING_RESIDUALS_PATH)),
        ("Latent-space diagnostics", str(SCORING_LATENT_PATH)),
    ]
    print_information_table("DNS SCORING ENABLEMENT AND MACHINE-LEARNING SUMMARY", rows)


def prepare_new_zeek_lines(path: Path) -> tuple[list[str], dict | None]:
    """Read new log data without committing the file cursor yet.

    The cursor is committed only after a scoring pass succeeds. This prevents a
    failed pass from permanently skipping the records that caused the failure.
    """
    if not path.exists():
        debug(f"Zeek DNS log does not exist: {path}")
        return [], None

    state = load_json(STATE_PATH, {})
    stat = path.stat()
    inode = int(stat.st_ino)
    previous_inode = state.get("inode")
    previous_path = str(state.get("path") or "")
    offset = int(state.get("offset", 0) or 0)

    if previous_inode != inode or previous_path != str(path) or offset > stat.st_size:
        offset = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        handle.seek(offset)
        lines = handle.readlines()
        new_offset = handle.tell()

    pending_state = {
        "path": str(path),
        "inode": inode,
        "offset": new_offset,
        "previous_offset": offset,
        "updated_at_utc": now_utc(),
    }
    return lines, pending_state


def prepare_full_zeek_lines(path: Path) -> list[str]:
    """Read the active DNS log from byte zero without touching live state."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return handle.readlines()


def commit_zeek_state(pending_state: dict | None) -> None:
    if not pending_state:
        return
    payload = dict(pending_state)
    payload.pop("previous_offset", None)
    save_json_atomic(STATE_PATH, payload)


def load_buffer_lines() -> list[str]:
    if not BUFFER_PATH.exists():
        return []
    return BUFFER_PATH.read_text(encoding="utf-8", errors="ignore").splitlines(True)


def save_buffer_records(records: pd.DataFrame) -> None:
    BUFFER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if records.empty:
        BUFFER_PATH.write_text("", encoding="utf-8")
        return

    raw_columns = [column for column in records.columns if not column.startswith("_")]
    with open(BUFFER_PATH, "w", encoding="utf-8") as handle:
        for _, row in records[raw_columns].iterrows():
            payload = {}
            for key, value in row.items():
                if isinstance(value, pd.Timestamp):
                    continue
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                if isinstance(value, (np.integer, np.floating)):
                    value = value.item()
                if isinstance(value, float) and np.isnan(value):
                    value = None
                payload[key] = value
            handle.write(json.dumps(payload, default=str) + "\n")


def _parse_utc_timestamp(values: pd.Series) -> pd.Series:
    """Convert Zeek epoch seconds or ISO-8601 values into UTC timestamps."""
    return parse_utc_timestamps(values)


def _first_existing_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def normalize_event_metadata(
    events: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Guarantee canonical timestamp/src_ip columns for event-level data.

    This is intentionally compatible with older feature_utils.py revisions
    that used ts, window_start, id.orig_h, or index-based metadata.
    """
    frame = events.copy()
    if frame.empty:
        return frame

    timestamp_column = _first_existing_column(
        frame, ("timestamp", "ts", "window_start", "window_ts", "time")
    )
    if timestamp_column is not None:
        frame["timestamp"] = _parse_utc_timestamp(frame[timestamp_column])
    elif raw_df is not None and "ts" in raw_df.columns:
        aligned = raw_df.reindex(frame.index)
        frame["timestamp"] = _parse_utc_timestamp(aligned["ts"])
    else:
        raise KeyError(
            "Event timestamp metadata is missing. Expected one of: "
            "timestamp, ts, window_start, window_ts, time. "
            f"Available event columns: {sorted(map(str, frame.columns))}"
        )

    source_column = _first_existing_column(
        frame, ("src_ip", "id.orig_h", "source_ip", "orig_h", "client_ip")
    )
    if source_column is not None:
        frame["src_ip"] = frame[source_column].fillna("").astype(str)
    elif raw_df is not None and "id.orig_h" in raw_df.columns:
        aligned = raw_df.reindex(frame.index)
        frame["src_ip"] = aligned["id.orig_h"].fillna("").astype(str)
    else:
        raise KeyError(
            "Source-IP metadata is missing. Expected src_ip or id.orig_h. "
            f"Available event columns: {sorted(map(str, frame.columns))}"
        )

    frame = frame[
        frame["timestamp"].notna()
        & frame["src_ip"].astype(str).str.strip().ne("")
    ].copy()
    return frame


def derive_window_metadata(
    events: pd.DataFrame,
    window_size: str,
) -> pd.DataFrame:
    """Rebuild window timestamp/source-IP keys from event-level rows.

    Some older project revisions calculate the 36 feature columns correctly but
    return only FEATURES, dropping the two group keys.  This helper reproduces
    the exact pandas group order used by build_dns_windows() so the metadata can
    be attached safely by row position.
    """
    if events.empty:
        return pd.DataFrame(columns=["timestamp", "src_ip"])

    required = {"timestamp", "src_ip"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise KeyError(
            "Cannot reconstruct scoring-window metadata because event columns "
            f"are missing: {missing}. Available columns: "
            f"{sorted(map(str, events.columns))}"
        )

    event_keys = events[["timestamp", "src_ip"]].copy()
    event_keys["timestamp"] = _parse_utc_timestamp(event_keys["timestamp"])
    event_keys["src_ip"] = event_keys["src_ip"].fillna("").astype(str)
    event_keys = event_keys[
        event_keys["timestamp"].notna()
        & event_keys["src_ip"].str.strip().ne("")
    ].copy()

    metadata = (
        event_keys.groupby(
            [pd.Grouper(key="timestamp", freq=window_size), "src_ip"],
            dropna=False,
            observed=True,
            sort=True,
        )
        .size()
        .reset_index(name="_event_count")
    )
    metadata = metadata[["timestamp", "src_ip"]].copy()
    metadata["timestamp"] = _parse_utc_timestamp(metadata["timestamp"])
    metadata["src_ip"] = metadata["src_ip"].fillna("").astype(str)
    return metadata.reset_index(drop=True)


def normalize_window_metadata(
    windows: pd.DataFrame,
    *,
    events: pd.DataFrame | None = None,
    window_size: str | None = None,
    context: str = "scoring windows",
) -> pd.DataFrame:
    """Guarantee timestamp/src_ip columns for window-level data.

    Recovery order:
      1. Use ordinary columns.
      2. Recover named MultiIndex/index values.
      3. Reconstruct keys from the source events and attach by group order.
    """
    frame = windows.copy()
    if frame.empty:
        return frame

    # Named index levels are recoverable without making assumptions.
    index_names = [str(name) if name is not None else "" for name in frame.index.names]
    recoverable_index = any(
        name in {
            "timestamp", "ts", "window_start", "window_ts", "time",
            "src_ip", "id.orig_h", "source_ip", "orig_h", "client_ip",
        }
        for name in index_names
    )
    if recoverable_index:
        try:
            frame = frame.reset_index()
        except Exception:
            pass

    timestamp_column = _first_existing_column(
        frame,
        ("timestamp", "ts", "window_start", "window_ts", "time"),
    )
    source_column = _first_existing_column(
        frame,
        ("src_ip", "id.orig_h", "source_ip", "orig_h", "client_ip"),
    )

    # The local feature_utils.py can return only the 36 feature columns.  In
    # that case, reproduce the window group keys from completed_events.
    if timestamp_column is None or source_column is None:
        if events is None or not window_size:
            missing_names = []
            if timestamp_column is None:
                missing_names.append("timestamp")
            if source_column is None:
                missing_names.append("src_ip")
            raise KeyError(
                f"{context}: no {' or '.join(missing_names)} column was found. "
                f"Available columns: {sorted(map(str, frame.columns))}"
            )

        derived = derive_window_metadata(events, window_size)
        if len(derived) != len(frame):
            raise RuntimeError(
                f"{context}: cannot safely restore metadata because the number "
                f"of feature rows ({len(frame):,}) differs from the number of "
                f"source-IP/time groups ({len(derived):,})."
            )

        frame = frame.reset_index(drop=True)
        if timestamp_column is None:
            frame.insert(0, "timestamp", derived["timestamp"])
            timestamp_column = "timestamp"
        if source_column is None:
            insert_at = 1 if "timestamp" in frame.columns else 0
            frame.insert(insert_at, "src_ip", derived["src_ip"])
            source_column = "src_ip"

        trace(
            "Restored missing scoring-window metadata from completed Zeek "
            f"events for {len(frame):,} source-IP/time windows."
        )

    frame["timestamp"] = _parse_utc_timestamp(frame[timestamp_column])
    frame["src_ip"] = frame[source_column].fillna("").astype(str)
    frame = frame[
        frame["timestamp"].notna()
        & frame["src_ip"].astype(str).str.strip().ne("")
    ].copy()
    return frame.reset_index(drop=True)


def validate_windows_preserving_metadata(
    windows: pd.DataFrame,
    *,
    events: pd.DataFrame,
    window_size: str,
) -> pd.DataFrame:
    """Validate model features without losing timestamp or source IP."""
    canonical = normalize_window_metadata(
        windows,
        events=events,
        window_size=window_size,
        context="build_dns_windows output",
    )
    metadata = canonical[["timestamp", "src_ip"]].reset_index(drop=True)
    validated = validate_feature_matrix(canonical, FEATURES)

    if len(validated) != len(metadata):
        raise RuntimeError(
            "Feature validation changed the number of DNS windows: "
            f"metadata={len(metadata)}, validated={len(validated)}"
        )

    missing_features = [feature for feature in FEATURES if feature not in validated.columns]
    if missing_features:
        raise ValueError(
            "Validated scoring windows are missing model features: "
            f"{missing_features}"
        )

    features = validated[FEATURES].reset_index(drop=True)
    return pd.concat([metadata, features], axis=1)


def _deduplicate_raw_dns_records(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Prevent buffered records from being counted twice after a retry."""
    candidates = [
        "ts",
        "uid",
        "id.orig_h",
        "id.resp_h",
        "query",
        "qtype_name",
        "rcode_name",
    ]
    available = [column for column in candidates if column in raw_df.columns]
    if not available:
        return raw_df
    return raw_df.drop_duplicates(subset=available, keep="last").copy()


def _alert_fingerprint(alert: dict) -> str:
    material = "|".join(
        [
            str(alert.get("timestamp") or ""),
            str(alert.get("src_ip") or ""),
            str(alert.get("alert_type") or ""),
            str(alert.get("model_name") or ""),
        ]
    )
    return hashlib.sha256(material.encode("utf-8", errors="ignore")).hexdigest()


def _load_alert_fingerprint_cache() -> set[str]:
    global _ALERT_FINGERPRINT_CACHE
    if _ALERT_FINGERPRINT_CACHE is not None:
        return _ALERT_FINGERPRINT_CACHE

    fingerprints: set[str] = set()
    if ALERT_PATH.exists():
        try:
            with ALERT_PATH.open("r", encoding="utf-8", errors="replace") as handle:
                for raw in handle:
                    try:
                        item = json.loads(raw)
                    except Exception:
                        continue
                    if not isinstance(item, dict):
                        continue
                    fingerprints.add(
                        str(item.get("alert_fingerprint") or _alert_fingerprint(item))
                    )
        except OSError:
            pass
    _ALERT_FINGERPRINT_CACHE = fingerprints
    return fingerprints


def append_alert(alert: dict) -> bool:
    """Append a new alert once and skip duplicates during full-log rescoring."""
    ALERT_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = _alert_fingerprint(alert)
    cache = _load_alert_fingerprint_cache()
    if fingerprint in cache:
        return False
    alert["alert_fingerprint"] = fingerprint
    with ALERT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(alert, default=str) + "\n")
    cache.add(fingerprint)
    return True


def write_scoring_status(**updates) -> None:
    current = load_json(SCORING_STATUS_PATH, {})
    current.update(updates)
    current["updated_at_utc"] = now_utc()
    save_json_atomic(SCORING_STATUS_PATH, current)


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _append_bounded_csv(
    path: Path,
    frame: pd.DataFrame,
    *,
    dedupe_columns: list[str],
    limit: int = SCORING_HISTORY_LIMIT,
) -> None:
    if frame.empty:
        return
    try:
        previous = pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        previous = pd.DataFrame()
    combined = pd.concat([previous, frame], ignore_index=True, sort=False)
    available = [column for column in dedupe_columns if column in combined.columns]
    if available:
        combined = combined.drop_duplicates(subset=available, keep="last")
    if len(combined) > limit:
        combined = combined.tail(limit)
    _write_csv_atomic(path, combined)


def _select_latent_layer(model):
    """Return the intended bottleneck layer without depending on model.input."""
    for preferred in ("latent_layer", "latent", "bottleneck", "embedding"):
        try:
            return model.get_layer(preferred)
        except Exception:
            pass

    candidates = []
    for layer in model.layers[:-1]:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        try:
            output_shape = getattr(layer, "output_shape", None)
            if output_shape is None:
                output_shape = tuple(layer.output.shape)
            else:
                output_shape = tuple(output_shape)
            if len(output_shape) == 2 and output_shape[-1] is not None:
                candidates.append((int(output_shape[-1]), layer))
        except Exception:
            continue
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _build_latent_extractor(model, latent_layer, input_dim: int):
    """Build a latent-output model that works with Keras 3 loaded Sequential models.

    Keras 3 can load a Sequential model with valid weights and working ``predict()``
    while the legacy singular ``model.input`` property is still undefined.  Replaying
    the already-loaded layers on a fresh symbolic input avoids that property and keeps
    the exact trained layer objects and weights.
    """
    if latent_layer is None:
        return None

    scoring_input = tf.keras.Input(
        shape=(int(input_dim),),
        dtype=tf.float32,
        name="dns_scoring_features",
    )
    tensor = scoring_input
    latent_output = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        try:
            tensor = layer(tensor, training=False)
        except TypeError:
            tensor = layer(tensor)
        if layer is latent_layer or layer.name == latent_layer.name:
            latent_output = tensor
            break

    if latent_output is None:
        raise RuntimeError(
            f"Unable to reach latent layer {latent_layer.name!r} while replaying model layers."
        )

    return tf.keras.Model(
        inputs=scoring_input,
        outputs=latent_output,
        name="dns_latent_extractor",
    )


def _predict_latent_values(model, x_scaled: np.ndarray) -> np.ndarray | None:
    """Return latent vectors; keep the scoring pass alive if diagnostics cannot be built."""
    latent_layer = _select_latent_layer(model)
    if latent_layer is None:
        warn("No latent/bottleneck layer was found; latent diagnostics were skipped.")
        return None

    try:
        latent_model = _build_latent_extractor(
            model,
            latent_layer,
            input_dim=int(x_scaled.shape[1]),
        )
        values = np.asarray(latent_model.predict(x_scaled, verbose=0))
    except Exception as exc:
        warn(
            "Latent diagnostics were skipped, but DNS scoring will continue: "
            f"{type(exc).__name__}: {exc}"
        )
        return None

    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values


def save_scoring_diagnostics(
    windows: pd.DataFrame,
    x_scaled: np.ndarray,
    x_pred: np.ndarray,
    errors: np.ndarray,
    threshold: float,
    model,
    *,
    replace_existing: bool = False,
    scoring_mode: str = "incremental",
) -> None:
    if windows.empty:
        return

    windows = normalize_window_metadata(
        windows, context="scoring diagnostics windows"
    )
    metadata = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                windows["timestamp"], utc=True, errors="coerce"
            ).astype(str),
            "src_ip": windows["src_ip"].astype(str),
        }
    )

    scored = pd.concat(
        [metadata.reset_index(drop=True), windows[FEATURES].reset_index(drop=True)],
        axis=1,
    )
    scored["reconstruction_error"] = np.asarray(errors, dtype=float)
    scored["threshold"] = float(threshold)
    scored["threshold_ratio"] = np.where(
        threshold > 0, scored["reconstruction_error"] / threshold, np.nan
    )
    scored["autoencoder_anomaly"] = scored["reconstruction_error"] > threshold
    scored["scoring_mode"] = scoring_mode
    if replace_existing:
        _write_csv_atomic(SCORING_WINDOWS_PATH, scored)
    else:
        _append_bounded_csv(
            SCORING_WINDOWS_PATH,
            scored,
            dedupe_columns=["timestamp", "src_ip"],
        )

    residual_values = np.abs(np.asarray(x_scaled) - np.asarray(x_pred))
    residuals = metadata.copy()
    for index, feature in enumerate(FEATURES):
        residuals[f"residual__{feature}"] = residual_values[:, index]
    residuals["reconstruction_error"] = np.asarray(errors, dtype=float)
    residuals["scoring_mode"] = scoring_mode
    if replace_existing:
        _write_csv_atomic(SCORING_RESIDUALS_PATH, residuals)
    else:
        _append_bounded_csv(
            SCORING_RESIDUALS_PATH,
            residuals,
            dedupe_columns=["timestamp", "src_ip"],
        )

    latent_values = _predict_latent_values(model, np.asarray(x_scaled))
    if latent_values is not None:
        latent = metadata.copy()
        for index in range(latent_values.shape[1]):
            latent[f"latent_{index + 1:02d}"] = latent_values[:, index]
        latent["scoring_mode"] = scoring_mode
        if replace_existing:
            _write_csv_atomic(SCORING_LATENT_PATH, latent)
        else:
            _append_bounded_csv(
                SCORING_LATENT_PATH,
                latent,
                dedupe_columns=["timestamp", "src_ip"],
            )
        trace(
            f"Saved latent diagnostics: {latent_values.shape[0]:,} windows × "
            f"{latent_values.shape[1]:,} latent dimensions."
        )


class ArtifactCache:
    def __init__(self):
        self.signature = None
        self.model = None
        self.scaler = None
        self.config = None

    def _signature(self):
        return tuple(
            path.stat().st_mtime_ns if path.exists() else None
            for path in [MODEL_PATH, SCALER_PATH, CONFIG_PATH]
        )

    def load(self):
        signature = self._signature()
        if None in signature:
            raise FileNotFoundError(
                "Model artifacts are incomplete. Required: "
                f"{MODEL_PATH}, {SCALER_PATH}, {CONFIG_PATH}"
            )
        if signature != self.signature:
            trace("Loading or reloading model artifacts.")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            scaler = joblib.load(SCALER_PATH)
            config = load_json(CONFIG_PATH, {})
            expected_features = config.get("features")
            if expected_features != FEATURES:
                raise RuntimeError(
                    "Feature order mismatch between model config and feature_utils.py."
                )
            if int(config.get("input_dim", -1)) != len(FEATURES):
                raise RuntimeError("Model input dimension does not match feature count.")
            self.model, self.scaler, self.config = model, scaler, config
            self.signature = signature
            print_information_table(
                "LOADED AUTOENCODER MODEL",
                [
                    ("Model name", config.get("model_name", "DNS Autoencoder")),
                    ("Input features", len(FEATURES)),
                    ("Window size", config.get("window_size", "5min")),
                    ("Anomaly threshold", f"{float(config.get('threshold') or 0.0):.10f}"),
                    ("Model path", MODEL_PATH),
                    ("Scaler path", SCALER_PATH),
                    ("Configuration path", CONFIG_PATH),
                    ("Machine-learning state", "ENABLED / artifacts loaded successfully"),
                ],
            )
        return self.model, self.scaler, self.config


def top_contributing_features(x_scaled: np.ndarray, x_pred: np.ndarray, top_n: int = 10) -> list[dict]:
    absolute = np.abs(x_scaled - x_pred)
    indexes = np.argsort(absolute)[::-1][:top_n]
    return [
        {
            "feature": FEATURES[int(index)],
            "absolute_error": round(float(absolute[int(index)]), 8),
            "input_scaled": round(float(x_scaled[int(index)]), 8),
            "reconstructed_scaled": round(float(x_pred[int(index)]), 8),
        }
        for index in indexes
    ]


def possible_causes(methods: dict) -> list[str]:
    causes: list[str] = []
    if methods.get("heuristic_dga"):
        causes.append("DGA-like domain generation or malware domain discovery")
    if methods.get("heuristic_tunnel"):
        causes.append("DNS tunneling, command-and-control, or DNS exfiltration")
    if methods.get("heuristic_fastflux"):
        causes.append("Fast-flux infrastructure or legitimate CDN behavior requiring validation")
    if methods.get("heuristic_resolver_failure"):
        causes.append("Dead C2 domains, blocked domains, resolver failure, or DNS misconfiguration")
    if methods.get("heuristic_rogue_resolver"):
        causes.append("Unauthorized external DNS resolver, DoT, or incorrect endpoint DNS settings")
    if methods.get("autoencoder") and len(causes) == 0:
        causes.append("Behavior differs from the learned normal DNS baseline")
    return causes


def process_once(cache: ArtifactCache, *, full_log: bool = False) -> tuple[int, int]:
    if full_log:
        new_lines = prepare_full_zeek_lines(ZEEK_DNS_LOG)
        pending_state = None
        buffered_lines: list[str] = []
        trace(f"Full-log mode: read {len(new_lines):,} Zeek DNS lines from byte zero.")
        trace("Full-log mode: committed cursor and incomplete-event buffer are ignored and preserved.")
    else:
        new_lines, pending_state = prepare_new_zeek_lines(ZEEK_DNS_LOG)
        buffered_lines = load_buffer_lines()
        trace(f"New Zeek DNS lines read after cursor: {len(new_lines):,}")
        trace(f"Previously buffered incomplete lines: {len(buffered_lines):,}")
    all_lines = new_lines if full_log else buffered_lines + new_lines
    if not all_lines:
        trace("No new DNS records were found after the committed cursor.")
        print_information_table(
            "DNS SCORING PASS RESULT",
            [
                ("New Zeek lines", 0),
                ("Buffered lines", 0),
                ("Scored five-minute windows", 0),
                ("Alerts written", 0),
                ("Cursor action", "Unchanged - there was no new input"),
            ],
        )
        return 0, 0

    raw_df = parse_json_lines(all_lines)
    if raw_df.empty:
        warn("No valid DNS records were parsed from the new input.")
        commit_zeek_state(pending_state)
        return 0, 0

    parsed_count = int(len(raw_df))
    raw_df = _deduplicate_raw_dns_records(raw_df)
    deduplicated_count = int(len(raw_df))
    trace(
        f"Parsed DNS records: {parsed_count:,}; unique records after deduplication: "
        f"{deduplicated_count:,}."
    )
    events = normalize_event_metadata(add_event_features(raw_df), raw_df)
    if events.empty:
        warn("No usable DNS events remained after feature engineering.")
        commit_zeek_state(pending_state)
        return 0, 0

    model, scaler, config = cache.load()
    window_size = config.get("window_size", "5min")
    if full_log:
        # Full-log mode intentionally scores every record in the active file,
        # including the latest partial interval, and never changes the live buffer.
        completed_events = events.copy()
        incomplete_raw = raw_df.iloc[0:0].copy()
        trace(
            f"Full-log mode: all {len(completed_events):,} usable DNS events will be scored; "
            "the current partial interval is included."
        )
    else:
        current_window_start = pd.Timestamp.now(tz="UTC").floor(window_size)
        completed_mask = events["timestamp"] < current_window_start
        completed_events = events.loc[completed_mask].copy()

        incomplete_indexes = events.index[~completed_mask]
        incomplete_raw = raw_df.reindex(incomplete_indexes).dropna(how="all").copy()
        save_buffer_records(incomplete_raw)
        trace(
            f"Completed DNS events: {len(completed_events):,}; current-window events "
            f"buffered for the next pass: {len(incomplete_raw):,}."
        )

    if completed_events.empty:
        trace(
            "No new completed five-minute DNS windows were available. "
            "The current-window records were saved in the incomplete-event buffer."
        )
        commit_zeek_state(pending_state)
        print_information_table(
            "DNS SCORING PASS RESULT",
            [
                ("New Zeek lines", len(new_lines)),
                ("Previously buffered lines", len(buffered_lines)),
                ("Parsed unique DNS records", deduplicated_count),
                ("Completed DNS events", 0),
                ("Buffered current-window events", len(incomplete_raw)),
                ("Scored five-minute windows", 0),
                ("Alerts written", 0),
                (
                    "Cursor action",
                    "Unchanged - full-log mode preserves cursor and buffer"
                    if full_log
                    else "Committed; incomplete events preserved for the next pass",
                ),
            ],
        )
        return 0, 0

    built_windows = build_dns_windows(completed_events, window_size)
    windows = validate_windows_preserving_metadata(
        built_windows,
        events=completed_events,
        window_size=window_size,
    )
    if windows.empty:
        commit_zeek_state(pending_state)
        return 0, 0

    trace(
        f"Prepared {len(windows):,} scoring windows with metadata columns: "
        f"timestamp={windows['timestamp'].notna().sum():,}, "
        f"src_ip={windows['src_ip'].astype(str).str.strip().ne('').sum():,}."
    )

    x = windows[FEATURES].to_numpy(dtype="float32")
    x_scaled = scaler.transform(x).astype("float32")
    x_pred = model.predict(x_scaled, verbose=0)
    errors = calculate_reconstruction_error(x_scaled, x_pred)
    threshold = float(config.get("threshold", 0.0))
    above_threshold = int(np.sum(errors > threshold)) if threshold > 0 else 0
    trace(
        "Reconstruction error statistics: "
        f"min={float(np.min(errors)):.8f}, mean={float(np.mean(errors)):.8f}, "
        f"max={float(np.max(errors)):.8f}, threshold={threshold:.8f}, "
        f"above_threshold={above_threshold:,}."
    )

    write_scoring_status(
        state="saving",
        stage="Saving scoring diagnostics",
        progress_percent=82,
        message=f"Saving {len(windows):,} scored DNS windows and visualization data.",
        windows_scored=int(len(windows)),
    )
    save_scoring_diagnostics(
        windows,
        x_scaled,
        x_pred,
        errors,
        threshold,
        model,
        replace_existing=full_log,
        scoring_mode="full-log" if full_log else "incremental",
    )

    alerts_written = 0
    for position, (_, row) in enumerate(windows.iterrows()):
        error = float(errors[position])
        autoencoder_alert = threshold > 0 and error > threshold
        if not autoencoder_alert:
            continue

        start = pd.Timestamp(row["timestamp"])
        end = start + pd.Timedelta(window_size)
        methods = {"autoencoder": True}
        feature_values = {feature: round(float(row[feature]), 8) for feature in FEATURES}
        suspicious_domains = summarize_suspicious_domains(
            completed_events,
            src_ip=str(row["src_ip"]),
            window_start=start,
            window_size=window_size,
        )
        severity = classify_severity(error, threshold)

        alert_id = stable_autoencoder_alert_id(start, str(row["src_ip"]))
        alert = {
            "id": alert_id,
            "alert_uid": alert_id,
            "status": "New",
            "alert_name": "DNS autoencoder anomaly",
            "alert_type": "autoencoder_anomaly",
            "detection_type": "deep_learning_autoencoder",
            "timestamp": start.isoformat(),
            "window_end": end.isoformat(),
            "window_size": window_size,
            "src_ip": str(row["src_ip"]),
            "severity": severity,
            "model_name": config.get("model_name", "DNS Autoencoder"),
            "reconstruction_error": round(error, 10),
            "threshold": round(threshold, 10),
            "threshold_ratio": round(error / threshold, 6) if threshold > 0 else None,
            "detection_methods": methods,
            "features": feature_values,
            "top_contributing_features": top_contributing_features(
                x_scaled[position], x_pred[position]
            ),
            "suspicious_domains": suspicious_domains,
            "possible_causes": ["Behavior differs from the learned normal DNS baseline"],
            "generated_at_utc": now_utc(),
            "scoring_mode": "full-log" if full_log else "incremental",
        }
        alert.update(feature_values)
        if append_alert(alert):
            alerts_written += 1
            trace(
                f"Autoencoder alert: src={row['src_ip']} type=autoencoder_anomaly "
                f"severity={severity} error={error:.6f} threshold={threshold:.6f}"
            )
        else:
            trace(
                f"Duplicate Autoencoder alert skipped: src={row['src_ip']} "
                f"window={start.isoformat()}"
            )

    # Incremental mode commits only after successful diagnostics and alerts.
    # Full-log mode deliberately leaves the live cursor and buffer unchanged.
    if not full_log:
        commit_zeek_state(pending_state)
    print_information_table(
        "DNS SCORING PASS RESULT",
        [
            ("New Zeek lines", len(new_lines)),
            ("Previously buffered lines", len(buffered_lines)),
            ("Parsed DNS records", parsed_count),
            ("Unique DNS records", deduplicated_count),
            ("Completed DNS events", len(completed_events)),
            ("Buffered current-window events", len(incomplete_raw)),
            ("Scored five-minute windows", len(windows)),
            ("Reconstruction error minimum", f"{float(np.min(errors)):.10f}"),
            ("Reconstruction error mean", f"{float(np.mean(errors)):.10f}"),
            ("Reconstruction error maximum", f"{float(np.max(errors)):.10f}"),
            ("Learned anomaly threshold", f"{threshold:.10f}"),
            ("Windows above ML threshold", above_threshold),
            ("Alerts written", alerts_written),
            (
                "Cursor action",
                "Unchanged - full-log scoring ignored and preserved live state"
                if full_log
                else "Committed after successful diagnostics and alert processing",
            ),
        ],
    )
    return alerts_written, int(len(windows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Zeek DNS logs with the DNS Autoencoder only.")
    parser.add_argument("--once", action="store_true", help="Process available data once and exit.")
    parser.add_argument(
        "--full-log",
        action="store_true",
        help="Read the entire active DNS log from byte zero, ignore cursor/buffer state, and leave that state unchanged.",
    )
    args = parser.parse_args()
    if args.full_log and not args.once:
        parser.error("--full-log requires --once")

    ALERT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SCORING_DIR.mkdir(parents=True, exist_ok=True)
    cache = ArtifactCache()

    print_runtime_enablement_summary(run_once=args.once, full_log=args.full_log)
    trace(f"Watching DNS log: {ZEEK_DNS_LOG}")
    while True:
        started_at = now_utc()
        write_scoring_status(
            state="running",
            stage="Reading live Zeek DNS log",
            progress_percent=20,
            message=(
                f"Reading the complete DNS log from byte zero: {ZEEK_DNS_LOG}."
                if args.full_log
                else f"Reading new DNS records from {ZEEK_DNS_LOG}."
            ),
            run_mode="full-log" if args.full_log else "incremental",
            input_path=str(ZEEK_DNS_LOG),
            started_at_utc=started_at,
            error=None,
        )
        try:
            write_scoring_status(
                state="loading",
                stage="Loading model and building windows",
                progress_percent=45,
                message="Preparing completed source-IP five-minute windows.",
            )
            alerts_written, windows_scored = process_once(cache, full_log=args.full_log)
            final_state = "finished" if windows_scored else "no_data"
            message = (
                (
                    f"Full-log scoring completed: {windows_scored:,} windows scored and "
                    f"{alerts_written:,} new alerts written."
                    if args.full_log
                    else f"Scored {windows_scored:,} windows and wrote {alerts_written:,} alerts."
                )
                if windows_scored
                else (
                    "No usable DNS windows were available in the complete log."
                    if args.full_log
                    else "No new completed five-minute DNS windows were available."
                )
            )
            write_scoring_status(
                state=final_state,
                stage="Scoring completed",
                progress_percent=100,
                message=message,
                windows_scored=int(windows_scored),
                alerts_written=int(alerts_written),
                run_mode="full-log" if args.full_log else "incremental",
                completed_at_utc=now_utc(),
                error=None,
            )
            trace(message)
            debug(f"Alerts written in this pass: {alerts_written}")
        except KeyboardInterrupt:
            write_scoring_status(
                state="stopped",
                stage="Stopped",
                progress_percent=100,
                message="DNS scoring was stopped by the user.",
                completed_at_utc=now_utc(),
            )
            trace("Scoring stopped by user.")
            break
        except Exception as exc:
            write_scoring_status(
                state="error",
                stage="Scoring failed",
                progress_percent=100,
                message=f"Scoring pass failed: {exc}",
                error=str(exc),
                completed_at_utc=now_utc(),
            )
            warn(f"Scoring pass failed: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            if args.once:
                raise SystemExit(1)

        if args.once:
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
