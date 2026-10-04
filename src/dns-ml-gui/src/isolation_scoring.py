# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from src.settings import (
    CATEGORIZATION_ALERT_JSONL_PATH,
    IF_SCORING_ACTION_STATUS_PATH,
    IF_SCORING_CRON_DISABLED_PATH,
    IF_SCORING_CRON_FILE,
    IF_SCORING_CRON_LAST_PATH,
    IF_SCORING_HOME,
    IF_SCORING_LIVE_LOG_PATH,
    IF_SCORING_PID_PATH,
    IF_SCORING_SCRIPT_PATH,
    IF_SCORING_STATE_PATH,
    IF_SCORING_BUFFER_PATH,
    IF_SCORING_STATUS_PATH,
    IF_SCORING_WINDOWS_PATH,
    IF_SCORING_WRAPPER_PATH,
    IF_MODEL_PATH,
    IF_SCALER_PATH,
    IF_CONFIG_PATH,
    ZEEK_CURRENT_DIR,
)

SUPPORTED_CURRENT_NAMES = ("dns.log", "dns.jsonl", "dns.json")
SUPPORTED_CURRENT_GLOBS = ("dns*.log", "dns*.jsonl", "dns*.json")
WINDOW_SIZE = "5min"
TERMINAL_STATES = {"finished", "completed", "success", "failed", "error", "stopped", "no_data", "busy"}

CARD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "check_logs": {
        "title": "Check Live Zeek Logs",
        "percent": 0,
        "state": "idle",
        "headline": "Not checked",
        "detail": "Checks the live Zeek DNS file and the separate categorization cursor.",
    },
    "run_once": {
        "title": "Run Logging Once",
        "percent": 0,
        "state": "idle",
        "headline": "Not run",
        "detail": "Scores new completed five-minute windows with Isolation Forest.",
    },
    "full_log": {
        "title": "Run Scoring From Scratch",
        "percent": 0,
        "state": "idle",
        "headline": "Not run",
        "detail": "Scores the entire current DNS log without changing the categorization cursor.",
    },
    "activate_task": {
        "title": "Activate Logging Task",
        "percent": 0,
        "state": "idle",
        "headline": "Not activated",
        "detail": "Installs a separate Isolation Forest cron task every five minutes.",
    },
    "check_task": {
        "title": "Check Task Status",
        "percent": 0,
        "state": "idle",
        "headline": "Not checked",
        "detail": "Checks the categorization cron service, last run, and next run.",
    },
    "deactivate_task": {
        "title": "Deactivate Logging Task",
        "percent": 0,
        "state": "idle",
        "headline": "Not deactivated",
        "detail": "Pauses only the Isolation Forest categorization scheduled task.",
    },
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def reset_live_log() -> None:
    IF_SCORING_LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = IF_SCORING_LIVE_LOG_PATH.with_suffix(IF_SCORING_LIVE_LOG_PATH.suffix + ".reset")
    tmp.write_text("", encoding="utf-8")
    tmp.replace(IF_SCORING_LIVE_LOG_PATH)


def read_live_log(max_lines: Optional[int] = None) -> str:
    if not IF_SCORING_LIVE_LOG_PATH.exists():
        return "No Isolation Forest categorization scoring output is available yet."
    try:
        lines = IF_SCORING_LIVE_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        if max_lines is not None:
            lines = lines[-max(1, int(max_lines)):]
        return "\n".join(lines)
    except Exception as exc:
        return f"Unable to read Isolation Forest categorization scoring output: {exc}"


def load_cards() -> Dict[str, Dict[str, Any]]:
    stored = _read_json(IF_SCORING_ACTION_STATUS_PATH, {})
    cards: Dict[str, Dict[str, Any]] = {}
    for key, default in CARD_DEFAULTS.items():
        merged = dict(default)
        if isinstance(stored.get(key), dict):
            merged.update(stored[key])
        cards[key] = merged
    return cards


def update_card(key: str, *, percent=None, state=None, headline=None, detail=None) -> Dict[str, Dict[str, Any]]:
    cards = load_cards()
    card = cards.setdefault(key, dict(CARD_DEFAULTS.get(key, {})))
    if percent is not None:
        card["percent"] = max(0, min(int(percent), 100))
    if state is not None:
        card["state"] = state
    if headline is not None:
        card["headline"] = headline
    if detail is not None:
        card["detail"] = detail
    card["updated_at_utc"] = now_utc()
    _write_json_atomic(IF_SCORING_ACTION_STATUS_PATH, cards)
    return cards


def find_live_zeek_dns_log() -> Optional[Path]:
    for name in SUPPORTED_CURRENT_NAMES:
        candidate = ZEEK_CURRENT_DIR / name
        if candidate.is_file():
            return candidate
    found: list[Path] = []
    for pattern in SUPPORTED_CURRENT_GLOBS:
        found.extend(p for p in ZEEK_CURRENT_DIR.glob(pattern) if p.is_file())
    return max(found, key=lambda p: p.stat().st_mtime_ns) if found else None


def _detect_format(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    return "Zeek TSV"
                if line.startswith("["):
                    return "Zeek JSON array"
                if line.startswith("{"):
                    return "Zeek JSON/JSONL"
                return "Unknown text"
    except Exception:
        return "Unreadable"
    return "Empty"


def _effective_offset(path: Path) -> tuple[int, str]:
    state = _read_json(IF_SCORING_STATE_PATH, {})
    try:
        stat = path.stat()
    except OSError:
        return 0, "Unable to inspect live DNS log"
    if str(state.get("path") or "") != str(path):
        return 0, "No categorization cursor exists for this file"
    if state.get("inode") is not None and int(state.get("inode")) != int(stat.st_ino):
        return 0, "DNS log inode changed; categorization starts at byte zero"
    offset = int(state.get("offset", 0) or 0)
    if offset < 0 or offset > stat.st_size:
        return 0, "Categorization cursor is invalid; starts at byte zero"
    return offset, "Categorization cursor is valid"


def _read_tail_lines(path: Path, offset: int) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, int(offset)))
            payload = handle.read()
        return payload.decode("utf-8", errors="ignore").splitlines()
    except OSError:
        return []


def _read_buffer_events() -> pd.DataFrame:
    if not IF_SCORING_BUFFER_PATH.exists() or IF_SCORING_BUFFER_PATH.stat().st_size == 0:
        return pd.DataFrame()
    try:
        frame = pd.read_json(IF_SCORING_BUFFER_PATH, lines=True)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        return frame
    except Exception:
        return pd.DataFrame()


def check_live_zeek_log() -> Dict[str, Any]:
    update_card("check_logs", percent=12, state="running", headline="Checking", detail="Inspecting /opt/zeek/logs/current ...")
    path = find_live_zeek_dns_log()
    if path is None:
        update_card("check_logs", percent=100, state="error", headline="No live DNS log", detail=f"No supported dns.log/dns.jsonl file found under {ZEEK_CURRENT_DIR}")
        return {"error": "No supported live Zeek DNS log was found."}

    try:
        sys.path.insert(0, "/opt/dns-ml") if "/opt/dns-ml" not in sys.path else None
        from feature_utils import add_event_features, parse_json_lines

        stat = path.stat()
        offset, cursor_message = _effective_offset(path)
        lines = _read_tail_lines(path, offset)
        nonempty = [line for line in lines if line.strip() and not line.lstrip().startswith("#")]
        raw = pd.DataFrame()
        if nonempty:
            if _detect_format(path) == "Zeek TSV":
                header = []
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    for raw_header in handle:
                        if raw_header.startswith("#"):
                            header.append(raw_header)
                            continue
                        if raw_header.strip():
                            break
                raw = parse_json_lines(header + [line + "\n" for line in nonempty])
            else:
                raw = parse_json_lines([line + "\n" for line in nonempty])
            new_valid_records = len(raw)
            new_events = add_event_features(raw) if not raw.empty else pd.DataFrame()
        else:
            new_valid_records = 0
            new_events = pd.DataFrame()

        buffered = _read_buffer_events()
        if buffered.empty:
            combined = new_events
        elif new_events.empty:
            combined = buffered
        else:
            combined = pd.concat([buffered, new_events], ignore_index=True, sort=False)

        complete_windows = 0
        current_records = len(buffered)
        if not combined.empty and {"timestamp", "src_ip"}.issubset(combined.columns):
            combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True, errors="coerce")
            combined = combined.dropna(subset=["timestamp", "src_ip"])
            cutoff = pd.Timestamp.now(tz="UTC").floor(WINDOW_SIZE)
            complete = combined[combined["timestamp"] < cutoff]
            pending = combined[combined["timestamp"] >= cutoff]
            if not complete.empty:
                complete_windows = int(
                    complete.groupby([pd.Grouper(key="timestamp", freq=WINDOW_SIZE), "src_ip"], observed=True).ngroups
                )
            current_records = int(len(pending))

        detail = (
            f"New DNS records: {new_valid_records:,} · Completed 5-minute windows: {complete_windows:,} · "
            f"Current/buffered records: {current_records:,}"
        )
        update_card("check_logs", percent=100, state="success", headline=f"{path.name} ready", detail=detail)
        return {
            "path": str(path),
            "format": _detect_format(path),
            "size_bytes": stat.st_size,
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "committed_offset": offset,
            "cursor_message": cursor_message,
            "new_text_records": len(nonempty),
            "new_dns_records": new_valid_records,
            "completed_five_minute_windows": complete_windows,
            "current_or_buffered_records": current_records,
            "error": None,
        }
    except Exception as exc:
        update_card("check_logs", percent=100, state="error", headline="Check failed", detail=f"{type(exc).__name__}: {exc}")
        return {"error": str(exc)}


def _pid_running(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def is_running() -> bool:
    # Keep Streamlit in the active/polling state for the complete manual
    # Isolation Forest scoring lifecycle, including wrapper startup time.
    try:
        if IF_SCORING_PID_PATH.exists():
            raw = IF_SCORING_PID_PATH.read_text(encoding="utf-8").strip()
            if raw:
                pid = int(raw)
                if _pid_running(pid):
                    return True
    except Exception:
        pass

    try:
        status = _read_json(IF_SCORING_STATUS_PATH, {})
        state = str(status.get("state", "")).strip().lower()
        status_pid = int(status.get("pid", 0) or 0)

        if state in {"starting", "launching", "running"}:
            if status_pid > 0 and _pid_running(status_pid):
                return True
    except Exception:
        pass

    return False


def read_status() -> Dict[str, Any]:
    return _read_json(IF_SCORING_STATUS_PATH, {})


def start_scoring(mode: str = "once") -> Dict[str, Any]:
    if mode not in {"once", "full"}:
        return {"error": f"Unsupported mode: {mode}"}

    if is_running():
        return {
            "error": "Isolation Forest categorization scoring is already running."
        }

    if not IF_SCORING_WRAPPER_PATH.exists():
        return {
            "error": f"Missing scoring wrapper: {IF_SCORING_WRAPPER_PATH}"
        }

    reset_live_log()

    card_key = "full_log" if mode == "full" else "run_once"

    update_card(
        card_key,
        percent=5,
        state="running",
        headline="Starting",
        detail="Launching Isolation Forest categorization scoring.",
    )

    IF_SCORING_HOME.mkdir(parents=True, exist_ok=True)
    IF_SCORING_PID_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        process = subprocess.Popen(
            [str(IF_SCORING_WRAPPER_PATH), mode],
            cwd="/opt/dns-ml",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Critical GUI fix: publish a PID immediately so the first Streamlit
        # rerun still sees an active process.
        IF_SCORING_PID_PATH.write_text(
            str(process.pid),
            encoding="utf-8",
        )

        # Publish a fresh running status immediately as well.
        _write_json_atomic(
            IF_SCORING_STATUS_PATH,
            {
                "state": "running",
                "stage": "Starting Isolation Forest scorer",
                "progress_percent": 5,
                "message": (
                    "Isolation Forest scoring process launched. "
                    "Waiting for scorer initialization and live Zeek DNS input."
                ),
                "run_mode": mode,
                "pid": process.pid,
                "launcher_pid": process.pid,
                "started_at_utc": now_utc(),
            },
        )

        update_card(
            card_key,
            percent=5,
            state="running",
            headline="Scorer launched",
            detail=f"PID {process.pid} · Waiting for live scoring output.",
        )

        return {
            "pid": process.pid,
            "mode": mode,
            "state": "running",
            "error": None,
        }

    except Exception as exc:
        try:
            IF_SCORING_PID_PATH.unlink(missing_ok=True)
        except Exception:
            pass

        update_card(
            card_key,
            percent=100,
            state="error",
            headline="Start failed",
            detail=f"{type(exc).__name__}: {exc}",
        )

        _write_json_atomic(
            IF_SCORING_STATUS_PATH,
            {
                "state": "failed",
                "stage": "Launch failed",
                "progress_percent": 100,
                "message": f"{type(exc).__name__}: {exc}",
                "run_mode": mode,
                "completed_at_utc": now_utc(),
            },
        )

        return {
            "error": f"{type(exc).__name__}: {exc}",
            "mode": mode,
        }


def sync_cards_from_status() -> Dict[str, Dict[str, Any]]:
    status = read_status()
    state = str(status.get("state", "")).lower()
    mode = str(status.get("run_mode", "once"))
    if not state:
        return load_cards()
    key = "full_log" if mode == "full" else "run_once"
    progress = int(status.get("progress_percent", 0) or 0)
    if state in {"running"}:
        update_card(key, percent=progress, state="running", headline=status.get("stage", "Running"), detail=status.get("message", ""))
    elif state in {"waiting_input", "waiting", "input_pending"}:
        update_card(
            key,
            percent=0,
            state="warning",
            headline="Waiting for DNS input",
            detail=status.get(
                "message",
                "No DNS records are available yet; the next scheduled interval will retry.",
            ),
        )
    elif state in {"finished", "completed", "success"}:
        update_card(
            key,
            percent=100,
            state="success",
            headline=f"{int(status.get('windows_scored', 0) or 0):,} windows scored",
            detail=f"Anomalies: {int(status.get('anomaly_windows', 0) or 0):,} · Alerts: {int(status.get('alerts_written', 0) or 0):,}",
        )
    elif state == "no_data":
        update_card(key, percent=100, state="warning", headline="No completed data", detail=status.get("message", "No new completed windows."))
    elif state in {"failed", "error"}:
        update_card(key, percent=100, state="error", headline="Scoring failed", detail=status.get("message", status.get("error", "Unknown error")))
    return load_cards()


def _cron_service_status() -> tuple[bool, str]:
    for service in ("cron", "crond"):
        try:
            completed = subprocess.run(["systemctl", "is-active", service], capture_output=True, text=True, timeout=5)
            if completed.returncode == 0 and completed.stdout.strip() == "active":
                return True, service
        except Exception:
            pass
    return False, "cron/crond"


def activate_cron() -> Dict[str, Any]:
    try:
        IF_SCORING_CRON_FILE.parent.mkdir(parents=True, exist_ok=True)
        IF_SCORING_HOME.mkdir(parents=True, exist_ok=True)
        content = (
            "SHELL=/bin/bash\n"
            "PATH=/opt/dns-ml/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
            f"*/5 * * * * root /usr/bin/flock -n {IF_SCORING_HOME / 'scheduled.lock'} "
            f"{IF_SCORING_WRAPPER_PATH} scheduled >> {IF_SCORING_HOME / 'cron.log'} 2>&1\n"
        )
        tmp = IF_SCORING_CRON_FILE.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(IF_SCORING_CRON_FILE)
        os.chmod(IF_SCORING_CRON_FILE, 0o644)
        update_card("activate_task", percent=100, state="success", headline="Categorization task active", detail="Isolation Forest scoring is scheduled every five minutes.")
        return {"error": None, "cron_file": str(IF_SCORING_CRON_FILE)}
    except Exception as exc:
        update_card("activate_task", percent=100, state="error", headline="Activation failed", detail=f"{type(exc).__name__}: {exc}")
        return {"error": str(exc)}


def check_cron_status() -> Dict[str, Any]:
    enabled = IF_SCORING_CRON_FILE.exists()
    service_active, service_name = _cron_service_status()
    scheduled_status = _read_json(IF_SCORING_CRON_LAST_PATH, {})
    last_run = scheduled_status.get("completed_at_utc") or scheduled_status.get("started_at_utc") or "Never"
    last_rc = scheduled_status.get("return_code")
    now = datetime.now(timezone.utc)
    minute = ((now.minute // 5) + 1) * 5
    if minute >= 60:
        next_run = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    else:
        next_run = now.replace(minute=minute, second=0, microsecond=0)

    if enabled and service_active:
        state, percent, headline = "success", 100, "Task running normally"
    elif enabled:
        state, percent, headline = "warning", 70, "Task installed; cron service not active"
    else:
        state, percent, headline = "warning", 20, "Categorization task is inactive"
    rc_text = "N/A" if last_rc is None else str(last_rc)
    detail = f"cron={service_name}:{'active' if service_active else 'inactive'} · last scheduled={last_run} (rc={rc_text}) · next≈{next_run.isoformat()}"
    update_card("check_task", percent=percent, state=state, headline=headline, detail=detail)
    return {"enabled": enabled, "cron_service_active": service_active, "cron_service": service_name, "last_scheduled_run": last_run, "last_return_code": last_rc, "next_run": next_run.isoformat(), "error": None}


def deactivate_cron() -> Dict[str, Any]:
    try:
        IF_SCORING_HOME.mkdir(parents=True, exist_ok=True)
        if IF_SCORING_CRON_FILE.exists():
            IF_SCORING_CRON_DISABLED_PATH.write_text(IF_SCORING_CRON_FILE.read_text(encoding="utf-8"), encoding="utf-8")
            IF_SCORING_CRON_FILE.unlink()
        update_card("deactivate_task", percent=100, state="success", headline="Categorization task inactive", detail="The Isolation Forest five-minute schedule is paused. Existing scores and alerts are retained.")
        return {"error": None}
    except Exception as exc:
        update_card("deactivate_task", percent=100, state="error", headline="Deactivation failed", detail=f"{type(exc).__name__}: {exc}")
        return {"error": str(exc)}


def load_scoring_windows() -> pd.DataFrame:
    if not IF_SCORING_WINDOWS_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(IF_SCORING_WINDOWS_PATH)
    except Exception:
        return pd.DataFrame()


def load_live_alerts(limit: int = 500) -> pd.DataFrame:
    if not CATEGORIZATION_ALERT_JSONL_PATH.exists():
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    try:
        lines = CATEGORIZATION_ALERT_JSONL_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()[-max(1, int(limit)):]
        for line in lines:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    except Exception:
        return pd.DataFrame()
    if not records:
        return pd.DataFrame()
    frame = pd.json_normalize(records)
    preferred = [
        "id", "timestamp", "src_ip", "severity", "threat_score", "status",
        "isolation_score", "isolation_threshold", "score_margin", "score_percentile",
        "contamination", "n_estimators", "max_samples", "max_features", "scoring_mode",
    ]
    columns = [column for column in preferred if column in frame.columns]
    return frame[columns] if columns else frame


def load_model_parameters() -> Dict[str, Any]:
    config = _read_json(IF_CONFIG_PATH, {})
    result: Dict[str, Any] = {
        "contamination": config.get("contamination"),
        "n_estimators": config.get("n_estimators"),
        "max_samples": config.get("max_samples"),
        "max_features": config.get("max_features"),
        "feature_count": config.get("feature_count", 36),
        "anomaly_threshold": config.get("anomaly_threshold", 0.0),
    }
    try:
        if IF_MODEL_PATH.exists():
            model = joblib.load(IF_MODEL_PATH)
            result["contamination"] = result["contamination"] if result["contamination"] is not None else getattr(model, "contamination", None)
            result["n_estimators"] = result["n_estimators"] if result["n_estimators"] is not None else getattr(model, "n_estimators", None)
            result["max_samples"] = result["max_samples"] if result["max_samples"] is not None else getattr(model, "max_samples_", getattr(model, "max_samples", None))
            result["max_features"] = result["max_features"] if result["max_features"] is not None else getattr(model, "max_features", 1.0)
    except Exception as exc:
        result["load_error"] = str(exc)
    result["model_ready"] = IF_MODEL_PATH.exists() and IF_SCALER_PATH.exists() and IF_CONFIG_PATH.exists()
    return result

# BEGIN CATEGORIZATION FLUSH LIVE OUTPUT PATCH
#
# DNS categorization scoring only.
#
# Requirement:
#   * Every control button flushes the previous categorization console.
#   * The new action writes its own fresh command/result output.
#   * Check Live Zeek Logs prints the new-record/readiness information.
#   * Existing Autoencoder / heuristic scoring is completely independent.

def _cat_live_log_path():
    path = globals().get("IF_SCORING_LIVE_LOG_PATH")
    if path is None:
        # Defensive fallback for older module revisions.
        return Path("/data/dns-ml/isolation_scoring/live_scoring.log")
    return Path(path)


def _cat_format_value(value):
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(value)
    if isinstance(value, (list, tuple, set)):
        try:
            return json.dumps(list(value), ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(value)
    if value is None:
        return "None"
    return str(value)


def _cat_flush_live_output(action_name: str) -> None:
    path = _cat_live_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    # "w" is intentional: every button must flush the previous console.
    with path.open("w", encoding="utf-8") as handle:
        handle.write("=" * 78 + "\n")
        handle.write("DNS CATEGORIZATION SCORING — ISOLATION FOREST\n")
        handle.write("=" * 78 + "\n")
        handle.write(f"[{now}] ACTION: {action_name}\n")
        handle.write("Previous Live recording command output: FLUSHED\n")
        handle.write("-" * 78 + "\n")


def _cat_append_live_output(title: str, result=None, extra_lines=None) -> None:
    path = _cat_live_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    lines = [
        "",
        f"[{now}] {title}",
        "-" * 78,
    ]

    if extra_lines:
        lines.extend(str(x) for x in extra_lines)

    if result is not None:
        if isinstance(result, dict):
            # Put the most useful live-Zeek/new-record keys first.
            preferred = [
                "error",
                "status",
                "message",
                "path",
                "file",
                "filename",
                "format",
                "size_bytes",
                "file_size",
                "modified_utc",
                "committed_offset",
                "cursor",
                "cursor_message",
                "new_text_lines",
                "new_text_records",
                "new_lines",
                "new_dns_records",
                "new_valid_dns_records",
                "completed_five_minute_windows",
                "completed_windows",
                "current_incomplete_window_records",
                "current_or_buffered_records",
                "buffered_records",
                "buffer_count",
                "pid",
                "mode",
                "cron_file",
                "enabled",
                "cron_service",
                "cron_service_active",
                "last_run",
                "next_run",
            ]
            emitted = set()

            for key in preferred:
                if key in result:
                    lines.append(f"{key}: {_cat_format_value(result.get(key))}")
                    emitted.add(key)

            for key, value in result.items():
                if key not in emitted:
                    lines.append(f"{key}: {_cat_format_value(value)}")
        else:
            lines.append(_cat_format_value(result))

    lines.append("")

    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


# Save the original categorization-control implementations once.
_cat_original_check_live_zeek_log = check_live_zeek_log
_cat_original_start_scoring = start_scoring
_cat_original_activate_cron = activate_cron
_cat_original_check_cron_status = check_cron_status
_cat_original_deactivate_cron = deactivate_cron


def check_live_zeek_log() -> Dict[str, Any]:
    action = "1 · Check Live Zeek Logs"
    _cat_flush_live_output(action)

    _cat_append_live_output(
        "STARTING LIVE ZEEK LOG CHECK",
        extra_lines=[
            f"Zeek live directory: {globals().get('ZEEK_CURRENT_DIR', '/opt/zeek/logs/current')}",
            "Mode: READ-ONLY",
            "Isolation Forest categorization cursor will be inspected but not advanced.",
            "",
            "Checking:",
            "  • current Zeek DNS file",
            "  • committed Isolation Forest cursor",
            "  • new text records after cursor",
            "  • new valid DNS records",
            "  • completed 5-minute source-IP windows",
            "  • incomplete/buffered records",
        ],
    )

    try:
        result = _cat_original_check_live_zeek_log()
    except Exception as exc:
        error_result = {
            "error": f"{type(exc).__name__}: {exc}",
        }
        _cat_append_live_output("CHECK LIVE ZEEK LOGS FAILED", error_result)
        raise

    _cat_append_live_output(
        "CHECK LIVE ZEEK LOGS RESULT — NEW RECORD READINESS",
        result=result,
    )
    return result


def start_scoring(mode: str = "once") -> Dict[str, Any]:
    action = (
        "3 · Run Scoring From Scratch"
        if mode == "full"
        else "2 · Run Logging Once"
    )

    # Flush BEFORE the new scorer starts.
    _cat_flush_live_output(action)

    _cat_append_live_output(
        "STARTING ISOLATION FOREST SCORING",
        extra_lines=[
            f"Scoring mode: {mode}",
            f"Live Zeek directory: {globals().get('ZEEK_CURRENT_DIR', '/opt/zeek/logs/current')}",
            f"Model: {globals().get('IF_MODEL_PATH', '/data/dns-ml/models/dns_isolation_forest.joblib')}",
            f"Scaler: {globals().get('IF_SCALER_PATH', '/data/dns-ml/models/dns_isolation_forest_scaler.joblib')}",
            f"Configuration: {globals().get('IF_CONFIG_PATH', '/data/dns-ml/models/dns_isolation_forest_config.json')}",
            "This run uses only the Isolation Forest categorization scoring pipeline.",
        ],
    )

    try:
        result = _cat_original_start_scoring(mode)
    except Exception as exc:
        error_result = {
            "error": f"{type(exc).__name__}: {exc}",
            "mode": mode,
        }
        _cat_append_live_output("ISOLATION FOREST SCORING LAUNCH FAILED", error_result)
        raise

    # Some existing start_scoring revisions clear the log internally.
    # Re-add the fresh launch information after the original launcher returns.
    _cat_append_live_output(
        "SCORING PROCESS LAUNCHED",
        result=result,
        extra_lines=[
            "The scorer process will append its new-record/scoring output below.",
        ],
    )
    return result


def activate_cron() -> Dict[str, Any]:
    action = "4 · Activate Logging Task"
    _cat_flush_live_output(action)

    _cat_append_live_output(
        "ACTIVATING ISOLATION FOREST LOGGING TASK",
        extra_lines=[
            "Target: independent Isolation Forest categorization cron/service",
            "Requested cadence: every five minutes",
        ],
    )

    try:
        result = _cat_original_activate_cron()
    except Exception as exc:
        error_result = {"error": f"{type(exc).__name__}: {exc}"}
        _cat_append_live_output("ACTIVATE LOGGING TASK FAILED", error_result)
        raise

    _cat_append_live_output("ACTIVATE LOGGING TASK RESULT", result=result)
    return result


def check_cron_status() -> Dict[str, Any]:
    action = "5 · Check Task Status"
    _cat_flush_live_output(action)

    _cat_append_live_output(
        "CHECKING ISOLATION FOREST TASK STATUS",
        extra_lines=[
            "Checking independent categorization cron/service only.",
        ],
    )

    try:
        result = _cat_original_check_cron_status()
    except Exception as exc:
        error_result = {"error": f"{type(exc).__name__}: {exc}"}
        _cat_append_live_output("CHECK TASK STATUS FAILED", error_result)
        raise

    _cat_append_live_output("CHECK TASK STATUS RESULT", result=result)
    return result


def deactivate_cron() -> Dict[str, Any]:
    action = "6 · Deactivate Logging Task"
    _cat_flush_live_output(action)

    _cat_append_live_output(
        "DEACTIVATING ISOLATION FOREST LOGGING TASK",
        extra_lines=[
            "Only the independent Isolation Forest categorization task will be disabled.",
            "Models, alerts, scoring history, cursor and buffer are retained.",
        ],
    )

    try:
        result = _cat_original_deactivate_cron()
    except Exception as exc:
        error_result = {"error": f"{type(exc).__name__}: {exc}"}
        _cat_append_live_output("DEACTIVATE LOGGING TASK FAILED", error_result)
        raise

    _cat_append_live_output("DEACTIVATE LOGGING TASK RESULT", result=result)
    return result

# END CATEGORIZATION FLUSH LIVE OUTPUT PATCH
