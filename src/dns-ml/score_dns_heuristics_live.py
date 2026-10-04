#!/usr/bin/env python3
# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

"""Independent rule-based DNS heuristic scorer for Zeek DNS logs.

This scorer is intentionally separate from both Autoencoder scoring and
Isolation Forest categorization. It uses the shared feature_utils.py feature
pipeline, its own cursor/buffer/runtime directory/cron, and a dedicated alert
JSONL file.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

ZEEK_CURRENT_DIR = Path(os.getenv("DNS_ZEEK_CURRENT", "/opt/zeek/logs/current"))
SCORING_HOME = Path(os.getenv("DNS_HEUR_SCORING_HOME", "/data/dns-ml/heuristics_scoring"))
STATUS_PATH = SCORING_HOME / "status.json"
LOG_PATH = SCORING_HOME / "live_scoring.log"
PID_PATH = SCORING_HOME / "scoring.pid"
LOCK_PATH = SCORING_HOME / "scoring.lock"

STATE_PATH = Path(os.getenv("DNS_HEUR_STATE_PATH", "/data/dns-ml/state/dns_heuristics_log_state.json"))
BUFFER_PATH = Path(os.getenv("DNS_HEUR_BUFFER_PATH", "/data/dns-ml/state/dns_heuristics_event_buffer.jsonl"))
ALERT_PATH = Path(os.getenv("DNS_HEUR_ALERT_PATH", "/data/dns-ml/alerts/dns_heuristics_alerts.jsonl"))

WINDOW_SIZE = os.getenv("DNS_HEUR_WINDOW_SIZE", "5min")
DEBUG_ENABLED = os.getenv("DNS_HEUR_SCORING_DEBUG", "1").strip().lower() not in {"0", "false", "no", "off"}
APPROVED_DNS_SERVERS = {
    item.strip()
    for item in os.getenv("DNS_APPROVED_SERVERS", "").split(",")
    if item.strip()
}
SUPPORTED_NAMES = ("dns.log", "dns.jsonl", "dns.json")
SUPPORTED_GLOBS = ("dns*.log", "dns*.jsonl", "dns*.json")
LOGGER = logging.getLogger("dns_heuristics_scoring")


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


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def write_status(state: str, stage: str, progress: int, message: str, **extra: Any) -> None:
    atomic_json(
        STATUS_PATH,
        {
            "state": state,
            "stage": stage,
            "progress_percent": max(0, min(int(progress), 100)),
            "message": message,
            "updated_at_utc": utc_now(),
            "pid": os.getpid(),
            "detection_type": "heuristics",
            "window_size": WINDOW_SIZE,
            "debug_enabled": DEBUG_ENABLED,
            **extra,
        },
    )


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
    return payload.decode("utf-8", errors="ignore").splitlines(keepends=True), new_offset


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
        LOGGER.warning("Unable to read heuristic event buffer; ignoring it: %s", exc)
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
    if keys:
        return frame.drop_duplicates(subset=keys, keep="last")
    return frame


def restore_window_metadata(events: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    result = windows.copy().reset_index(drop=True)
    if {"timestamp", "src_ip"}.issubset(result.columns):
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
        result["src_ip"] = result["src_ip"].astype(str)
        return result

    grouped = (
        events.groupby(
            [pd.Grouper(key="timestamp", freq=WINDOW_SIZE), "src_ip"],
            dropna=False,
            observed=True,
        )
        .size()
        .reset_index(name="__event_count")
    )
    if len(grouped) != len(result):
        raise ValueError(
            "Heuristic scoring metadata reconstruction failed: "
            f"feature rows={len(result):,}, source-IP/time groups={len(grouped):,}."
        )
    if "total_queries" in result.columns:
        expected = pd.to_numeric(result["total_queries"], errors="coerce").fillna(-1).round().astype(int).to_numpy()
        actual = grouped["__event_count"].astype(int).to_numpy()
        if not np.array_equal(expected, actual):
            raise ValueError("Heuristic scoring metadata alignment failed: total_queries differs from grouped event counts.")
    result.insert(0, "src_ip", grouped["src_ip"].astype(str).to_numpy())
    result.insert(0, "timestamp", pd.to_datetime(grouped["timestamp"], utc=True, errors="coerce").to_numpy())
    return result


def context_domains(events: pd.DataFrame, src_ip: str, window_start: Any) -> list[dict[str, Any]]:
    try:
        domains = summarize_suspicious_domains(events, src_ip, window_start, WINDOW_SIZE, top_n=10)
        if domains:
            return domains
    except TypeError:
        domains = summarize_suspicious_domains(events, src_ip=src_ip, window_start=window_start, window_size=WINDOW_SIZE)
        if domains:
            return domains
    return []


def stable_alert_id(timestamp: Any, src_ip: str, fired: list[str]) -> str:
    basis = f"heuristics|{pd.Timestamp(timestamp).isoformat()}|{src_ip}|{'|'.join(sorted(fired))}"
    return "HEUR-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20].upper()


def append_alerts(alerts: list[dict[str, Any]]) -> int:
    if not alerts:
        return 0
    ALERT_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing_ids: set[str] = set()
    if ALERT_PATH.exists():
        try:
            with ALERT_PATH.open("r", encoding="utf-8", errors="ignore") as existing:
                for line in existing:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    value = payload.get("id") or payload.get("alert_uid") if isinstance(payload, dict) else None
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
            existing_ids.add(alert_id)
            written += 1
        handle.flush()
        os.fsync(handle.fileno())
    return written


def possible_causes(methods: dict[str, bool]) -> list[str]:
    causes: list[str] = []
    if methods.get("heuristic_dga"):
        causes.append("DGA-like domain generation or malware domain discovery")
    if methods.get("heuristic_tunnel"):
        causes.append("DNS tunneling, command-and-control, or DNS exfiltration")
    if methods.get("heuristic_fastflux"):
        causes.append("Fast-flux infrastructure or legitimate CDN behavior requiring validation")
    if methods.get("heuristic_resolver_failure"):
        causes.append("Dead domains, blocked domains, resolver failure, or DNS misconfiguration")
    if methods.get("heuristic_rogue_resolver"):
        causes.append("Unauthorized DNS resolver or incorrect endpoint DNS settings")
    return causes


def threat_from_methods(methods: dict[str, bool], has_domains: bool) -> tuple[str, int]:
    score = 0
    score += 35 if methods.get("heuristic_dga") else 0
    score += 40 if methods.get("heuristic_tunnel") else 0
    score += 30 if methods.get("heuristic_fastflux") else 0
    score += 15 if methods.get("heuristic_resolver_failure") else 0
    score += 25 if methods.get("heuristic_rogue_resolver") else 0
    score += 15 if has_domains else 0
    score = min(score, 100)
    if score >= 90:
        return "critical", score
    if score >= 70:
        return "high", score
    if score >= 40:
        return "medium", score
    return "low", score


def evaluate_rules(row: pd.Series, source_events: pd.DataFrame) -> tuple[dict[str, bool], list[str], list[str]]:
    heuristic_dga = bool(
        row["total_queries"] >= 20
        and row["unique_query_ratio"] >= 0.90
        and row["nxdomain_rate"] >= 0.50
        and row["high_entropy_rate"] >= 0.50
    )
    heuristic_tunnel = bool(
        row["total_queries"] >= 20
        and (
            row["txt_rate"] >= 0.40
            or row["long_query_rate"] >= 0.40
            or row["long_label_rate"] >= 0.30
        )
        and row["high_entropy_rate"] >= 0.40
    )
    heuristic_fastflux = bool(
        row["total_queries"] >= 10
        and (
            row["answer_count_avg"] >= 3
            or row["answer_count_max"] >= 5
            or row["unique_answer_ips"] >= 10
        )
        and row["ttl_min"] > 0
        and (row["ttl_min"] <= 60 or row["low_ttl_rate"] >= 0.30)
    )
    heuristic_resolver_failure = bool(
        row["total_queries"] >= 20
        and (row["servfail_rate"] >= 0.50 or row["nxdomain_rate"] >= 0.70)
    )

    observed_resolvers = {
        value for value in source_events.get("dns_server", pd.Series(dtype=str)).astype(str)
        if value and value != "nan"
    }
    rogue_resolvers = sorted(observed_resolvers - APPROVED_DNS_SERVERS)
    heuristic_rogue_resolver = bool(
        rogue_resolvers if APPROVED_DNS_SERVERS else row["unique_dns_servers"] >= 3
    )

    methods = {
        "heuristic_dga": heuristic_dga,
        "heuristic_tunnel": heuristic_tunnel,
        "heuristic_fastflux": heuristic_fastflux,
        "heuristic_resolver_failure": heuristic_resolver_failure,
        "heuristic_rogue_resolver": heuristic_rogue_resolver,
    }
    fired = [name for name, value in methods.items() if value]
    return methods, fired, rogue_resolvers


def process(mode: str) -> dict[str, Any]:
    path = find_live_log()
    fmt = detect_format(path)
    stat = path.stat()

    write_status("running", "Reading Zeek DNS", 10, f"Reading {path}", run_mode=mode, live_log=str(path))
    LOGGER.info("DNS HEURISTICS SCORING — RULE BASED")
    LOGGER.info("Mode: %s | Debug: %s", mode, DEBUG_ENABLED)
    LOGGER.info("Live DNS file: %s | format=%s", path, fmt)
    LOGGER.info("Rules enabled: DGA, DNS tunnel, fast-flux, resolver failure, rogue resolver")
    LOGGER.info("Approved DNS resolvers: %s", ", ".join(sorted(APPROVED_DNS_SERVERS)) or "None configured")

    if mode == "full":
        old_buffer = pd.DataFrame()
        lines, new_offset = read_new_lines(path, 0)
    else:
        offset, _ = safe_committed_offset(path)
        old_buffer = load_buffer()
        lines, new_offset = read_new_lines(path, offset)

    if not lines and old_buffer.empty:
        write_status("no_data", "No new records", 100, "No new DNS records were available after the heuristic cursor.", run_mode=mode, alerts_written=0, windows_scored=0)
        return {"alerts_written": 0, "windows_scored": 0, "new_records": 0}

    raw = parse_lines(path, lines, fmt)
    new_events = add_event_features(raw) if not raw.empty else pd.DataFrame()
    LOGGER.debug("New parsed DNS events: %d; previous heuristic buffer: %d", len(new_events), len(old_buffer))

    if old_buffer.empty:
        combined_events = new_events
    elif new_events.empty:
        combined_events = old_buffer
    else:
        combined_events = pd.concat([old_buffer, new_events], ignore_index=True, sort=False)
    combined_events = dedupe_events(combined_events)

    if combined_events.empty:
        if mode != "full":
            atomic_json(STATE_PATH, {"path": str(path), "inode": stat.st_ino, "offset": new_offset, "updated_at_utc": utc_now()})
        write_status("no_data", "No valid DNS records", 100, "No valid Zeek DNS records were available for heuristic scoring.", run_mode=mode, alerts_written=0, windows_scored=0)
        return {"alerts_written": 0, "windows_scored": 0, "new_records": 0}

    write_status("running", "Preparing 5-minute windows", 35, "Building source-IP five-minute windows.", run_mode=mode, parsed_events=len(combined_events))
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
        write_status("no_data", "Waiting for completed windows", 100, "No new completed five-minute DNS windows were available for heuristic scoring.", run_mode=mode, alerts_written=0, windows_scored=0, buffered_events=len(pending_events))
        return {"alerts_written": 0, "windows_scored": 0, "new_records": len(new_events)}

    windows = build_dns_windows(completed_events, window_size=WINDOW_SIZE)
    windows = restore_window_metadata(completed_events, windows)
    windows = validate_feature_matrix(windows, FEATURES)
    windows = restore_window_metadata(completed_events, windows)
    if windows.empty:
        raise ValueError("Feature extraction produced zero heuristic scoring windows.")

    write_status("running", "Evaluating heuristic rules", 65, f"Evaluating {len(windows):,} five-minute windows.", run_mode=mode, windows_scored=len(windows))

    alerts: list[dict[str, Any]] = []
    fired_windows = 0
    for _, row in windows.iterrows():
        start = pd.Timestamp(row["timestamp"])
        end = start + pd.Timedelta(WINDOW_SIZE)
        source_events = completed_events[
            completed_events["src_ip"].eq(row["src_ip"])
            & completed_events["timestamp"].ge(start)
            & completed_events["timestamp"].lt(end)
        ]
        methods, fired, rogue_resolvers = evaluate_rules(row, source_events)
        if not fired:
            continue

        fired_windows += 1
        domains = context_domains(completed_events, str(row["src_ip"]), start)
        severity, threat_score = threat_from_methods(methods, bool(domains))
        alert_id = stable_alert_id(start, str(row["src_ip"]), fired)
        alert = {
            "id": alert_id,
            "alert_uid": alert_id,
            "timestamp": start.isoformat(),
            "window_end": end.isoformat(),
            "src_ip": str(row["src_ip"]),
            "status": "New",
            "alert_name": "DNS heuristic rule detection",
            "alert_type": "heuristics_rule",
            "detection_type": "heuristics",
            "model_name": "DNS Heuristics",
            "window_size": WINDOW_SIZE,
            "severity": severity,
            "threat_score": threat_score,
            "detection_methods": methods,
            "heuristics_fired": fired,
            "rogue_dns_servers": rogue_resolvers,
            "suspicious_domains": domains,
            "possible_causes": possible_causes(methods),
            "features": {feature: float(row[feature]) for feature in FEATURES},
            "scoring_mode": mode,
            "generated_at_utc": utc_now(),
        }
        alert.update({feature: float(row[feature]) for feature in FEATURES})
        alerts.append(alert)
        LOGGER.info("Heuristic alert: src=%s fired=%s severity=%s", row["src_ip"], ",".join(fired), severity)

    write_status("running", "Saving heuristic alerts", 90, f"Saving {len(alerts):,} heuristic alerts.", run_mode=mode, windows_scored=len(windows), heuristic_windows=fired_windows)
    alerts_written = append_alerts(alerts)

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

    LOGGER.info("Evaluated %d heuristic windows; rule-positive windows=%d; alerts written=%d", len(windows), fired_windows, alerts_written)
    write_status(
        "finished",
        "Completed",
        100,
        f"Heuristic scoring completed: {len(windows):,} windows, {fired_windows:,} rule-positive windows.",
        run_mode=mode,
        completed_at_utc=utc_now(),
        windows_scored=len(windows),
        heuristic_windows=fired_windows,
        alerts_written=alerts_written,
        alert_jsonl=str(ALERT_PATH),
    )
    return {"alerts_written": alerts_written, "windows_scored": len(windows), "heuristic_windows": fired_windows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score live Zeek DNS with DNS heuristic rules.")
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
        LOGGER.warning("Another heuristic scoring pass is already running.")
        write_status("busy", "Already running", 0, "Another DNS heuristic scoring pass is already running.", run_mode=args.mode)
        return 3

    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
    try:
        process(args.mode)
        return 0
    except LiveInputPending as exc:
        LOGGER.info("DNS Heuristics is waiting for Zeek DNS input: %s", exc)
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
        LOGGER.error("DNS heuristic scoring failed: %s: %s", type(exc).__name__, exc)
        LOGGER.debug("%s", traceback.format_exc())
        write_status("failed", "Failed", 100, f"DNS heuristic scoring failed: {type(exc).__name__}: {exc}", run_mode=args.mode, error=str(exc))
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
