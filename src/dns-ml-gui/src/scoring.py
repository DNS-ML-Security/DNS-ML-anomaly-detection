# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

from __future__ import annotations
import hashlib

import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.settings import (
    ALERT_JSONL_PATH,
    MODEL_CONFIG_PATH,
    SCORING_ACTION_STATUS_PATH,
    SCORING_CRON_DISABLED_PATH,
    SCORING_CRON_FILE,
    SCORING_FEATURE_RESIDUALS_PATH,
    SCORING_LATENT_PATH,
    SCORING_LIVE_LOG_PATH,
    SCORING_LOCK_PATH,
    SCORING_PID_PATH,
    SCORING_SCRIPT_PATH,
    SCORING_STATUS_PATH,
    SCORING_WINDOWS_PATH,
    ZEEK_CURRENT_DIR,
)


SUPPORTED_CURRENT_NAMES = (
    "dns.log",
    "dns.jsonl",
    "dns.json",
)
SUPPORTED_CURRENT_GLOBS = (
    "dns*.log",
    "dns*.jsonl",
    "dns*.json",
)
TERMINAL_SCORING_STATES = {
    "finished",
    "completed",
    "success",
    "failed",
    "error",
    "stopped",
    "no_data",
}
SCORING_STATE_PATH = Path("/data/dns-ml/state/dns_log_state.json")
SCORING_BUFFER_PATH = Path("/data/dns-ml/state/dns_event_buffer.jsonl")
SCORING_WINDOW_SIZE = "5min"
CARD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "check_logs": {
        "title": "Check Live Zeek Logs",
        "percent": 0,
        "state": "idle",
        "headline": "Not checked",
        "detail": "Checks /opt/zeek/logs/current for a live DNS log.",
    },
    "run_once": {
        "title": "Run Logging Once",
        "percent": 0,
        "state": "idle",
        "headline": "Not run",
        "detail": "Scores new completed five-minute DNS windows once.",
    },
    "full_log": {
        "title": "Run Scoring From Scratch",
        "percent": 0,
        "state": "idle",
        "headline": "Not run",
        "detail": "Scores the entire current DNS log from byte zero while preserving the live cursor and buffer.",
    },
    "activate_task": {
        "title": "Activate Logging Task",
        "percent": 0,
        "state": "idle",
        "headline": "Not activated",
        "detail": "Installs a cron task that runs every five minutes.",
    },
    "check_task": {
        "title": "Check Logging Task Running Status",
        "percent": 0,
        "state": "idle",
        "headline": "Not checked",
        "detail": "Shows cron service, next run and last execution.",
    },
    "deactivate_task": {
        "title": "Deactivate Results Logging Task",
        "percent": 0,
        "state": "idle",
        "headline": "Not deactivated",
        "detail": "Pauses the scheduled five-minute scoring task.",
    },
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default):
    try:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    tmp.replace(path)


def append_scoring_log(message: str) -> None:
    SCORING_LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = now_utc()
    with SCORING_LIVE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message.rstrip()}\n")


def reset_scoring_log_for_manual_run() -> None:
    """Clear only the manual-run command output before a new run starts.

    The five-minute scheduled task is intentionally unchanged and continues to
    append to the same file.  This function is called only by the GUI's
    ``Run Logging Once`` action after confirming that no scorer is active.
    """
    SCORING_LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SCORING_LIVE_LOG_PATH.with_suffix(
        SCORING_LIVE_LOG_PATH.suffix + ".reset"
    )
    temporary.write_text("", encoding="utf-8")
    temporary.replace(SCORING_LIVE_LOG_PATH)


def read_scoring_log(max_lines: Optional[int] = None) -> str:
    if not SCORING_LIVE_LOG_PATH.exists():
        return "No DNS scoring command output is available yet."
    try:
        lines = SCORING_LIVE_LOG_PATH.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        if max_lines is not None:
            count = max(1, int(max_lines))
            lines = lines[-count:]
        return "\n".join(lines)
    except Exception as exc:
        return f"Unable to read DNS scoring output: {exc}"


def load_control_cards() -> Dict[str, Dict[str, Any]]:
    stored = _read_json(SCORING_ACTION_STATUS_PATH, {})
    cards: Dict[str, Dict[str, Any]] = {}
    for key, default in CARD_DEFAULTS.items():
        merged = dict(default)
        if isinstance(stored.get(key), dict):
            merged.update(stored[key])
        cards[key] = merged
    return cards


def update_control_card(
    key: str,
    *,
    percent: Optional[int] = None,
    state: Optional[str] = None,
    headline: Optional[str] = None,
    detail: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    cards = load_control_cards()
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
    _write_json_atomic(SCORING_ACTION_STATUS_PATH, cards)
    return cards


def find_live_zeek_dns_log() -> Optional[Path]:
    for name in SUPPORTED_CURRENT_NAMES:
        candidate = ZEEK_CURRENT_DIR / name
        if candidate.is_file():
            return candidate

    discovered: list[Path] = []
    for pattern in SUPPORTED_CURRENT_GLOBS:
        discovered.extend(path for path in ZEEK_CURRENT_DIR.glob(pattern) if path.is_file())
    if not discovered:
        return None
    return max(discovered, key=lambda path: path.stat().st_mtime_ns)


def _count_lines(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            count += block.count(b"\n")
    return count


def _detect_format(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    return "Zeek TSV"
                if line.startswith("{") or line.startswith("["):
                    return "Zeek JSON/JSONL"
                return "Unknown text"
    except Exception:
        return "Unreadable"
    return "Empty"



def _effective_committed_offset(path: Path) -> tuple[int, str]:
    """Return the safe byte offset for the current file without changing state."""
    state = _read_json(SCORING_STATE_PATH, {})
    try:
        stat = path.stat()
    except OSError:
        return 0, "Unable to inspect the current DNS log"

    committed_path = str(state.get("path") or "")
    committed_inode = state.get("inode")
    committed_offset = int(state.get("offset", 0) or 0)

    if committed_path != str(path):
        return 0, "No committed cursor exists for this DNS log"
    if committed_inode is not None and int(committed_inode) != int(stat.st_ino):
        return 0, "The DNS log inode changed; the next incremental run starts at byte zero"
    if committed_offset < 0 or committed_offset > int(stat.st_size):
        return 0, "The committed cursor is outside the current file; the next incremental run starts at byte zero"
    return committed_offset, "Committed cursor is valid"


def _read_text_lines_from_offset(path: Path, offset: int) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            handle.seek(max(0, int(offset)))
            return handle.readlines()
    except OSError:
        return []


def _read_buffer_lines() -> list[str]:
    try:
        if not SCORING_BUFFER_PATH.exists():
            return []
        return SCORING_BUFFER_PATH.read_text(
            encoding="utf-8", errors="ignore"
        ).splitlines(True)
    except OSError:
        return []


def _parse_dns_record_metadata(lines: list[str]) -> list[tuple[pd.Timestamp, str]]:
    """Parse only timestamp/source-IP metadata for readiness calculations.

    The active scoring pipeline consumes JSON/JSONL Zeek logs. This lightweight
    reader intentionally does not alter the live cursor or the incomplete-event
    buffer.
    """
    metadata: list[tuple[pd.Timestamp, str]] = []
    for raw_line in lines:
        text = raw_line.strip()
        if not text or not text.startswith("{"):
            continue
        try:
            payload = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue

        raw_ts = payload.get("ts", payload.get("timestamp"))
        src_ip = payload.get(
            "id.orig_h",
            payload.get("src_ip", payload.get("source_ip", "")),
        )
        src_ip = str(src_ip or "").strip()
        if raw_ts is None or not src_ip:
            continue

        timestamp = pd.NaT
        try:
            numeric = float(raw_ts)
            timestamp = pd.to_datetime(numeric, unit="s", utc=True, errors="coerce")
        except (TypeError, ValueError):
            timestamp = pd.to_datetime(raw_ts, utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue
        metadata.append((timestamp, src_ip))
    return metadata


def _calculate_live_log_readiness(path: Path) -> Dict[str, Any]:
    """Calculate cursor and five-minute-window readiness without side effects."""
    stat = path.stat()
    offset, cursor_message = _effective_committed_offset(path)
    new_lines = _read_text_lines_from_offset(path, offset)
    buffer_lines = _read_buffer_lines()

    new_metadata = _parse_dns_record_metadata(new_lines)
    buffered_metadata = _parse_dns_record_metadata(buffer_lines)
    combined_metadata = buffered_metadata + new_metadata

    current_window_start = pd.Timestamp.now(tz="UTC").floor(SCORING_WINDOW_SIZE)
    completed_keys: set[tuple[pd.Timestamp, str]] = set()
    incomplete_records = 0
    for timestamp, src_ip in combined_metadata:
        window_start = timestamp.floor(SCORING_WINDOW_SIZE)
        if timestamp < current_window_start:
            completed_keys.add((window_start, src_ip))
        else:
            incomplete_records += 1

    return {
        "committed_offset_bytes": int(offset),
        "file_size_bytes": int(stat.st_size),
        "new_bytes_after_cursor": max(0, int(stat.st_size) - int(offset)),
        "new_lines_after_cursor": len(new_lines),
        "new_dns_records_after_cursor": len(new_metadata),
        "buffered_dns_records": len(buffered_metadata),
        "completed_five_minute_windows": len(completed_keys),
        "current_incomplete_window_records": int(incomplete_records),
        "cursor_message": cursor_message,
    }

def check_live_zeek_log() -> Dict[str, Any]:
    update_control_card(
        "check_logs",
        percent=15,
        state="running",
        headline="Scanning current logs",
        detail=str(ZEEK_CURRENT_DIR),
    )
    append_scoring_log(f"Checking live Zeek DNS logs under {ZEEK_CURRENT_DIR}")

    if not ZEEK_CURRENT_DIR.exists():
        result = {
            "ok": False,
            "error": f"Directory not found: {ZEEK_CURRENT_DIR}",
        }
        update_control_card(
            "check_logs",
            percent=100,
            state="error",
            headline="Directory missing",
            detail=result["error"],
        )
        append_scoring_log(result["error"])
        return result

    path = find_live_zeek_dns_log()
    if path is None:
        result = {
            "ok": False,
            "error": (
                "No live DNS log was found. Accepted current names include "
                "dns.log, dns.jsonl and dns.json."
            ),
        }
        update_control_card(
            "check_logs",
            percent=100,
            state="error",
            headline="No DNS log found",
            detail=result["error"],
        )
        append_scoring_log(result["error"])
        return result

    update_control_card(
        "check_logs",
        percent=55,
        state="running",
        headline="Inspecting file",
        detail=str(path),
    )
    stat = path.stat()
    line_count = _count_lines(path)
    readiness = _calculate_live_log_readiness(path)
    result = {
        "ok": True,
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "line_count": int(line_count),
        "format": _detect_format(path),
        "modified_at_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        **readiness,
    }
    detail = (
        f"New records: {result['new_dns_records_after_cursor']:,} • "
        f"completed 5-minute windows: {result['completed_five_minute_windows']:,} • "
        f"current-window records: {result['current_incomplete_window_records']:,}"
    )
    update_control_card(
        "check_logs",
        percent=100,
        state="success",
        headline="Live DNS log ready",
        detail=detail,
    )
    append_scoring_log(
        f"Live DNS log ready: {path.name} • {result['format']} • "
        f"{line_count:,} lines • {stat.st_size:,} bytes"
    )
    append_scoring_log(
        "Committed cursor: "
        f"{result['committed_offset_bytes']:,} / {result['file_size_bytes']:,} bytes • "
        f"{result['cursor_message']}"
    )
    append_scoring_log(
        "New DNS records found after the committed cursor: "
        f"{result['new_dns_records_after_cursor']:,} valid records "
        f"({result['new_lines_after_cursor']:,} new text lines)."
    )
    append_scoring_log(
        "New completed five-minute source-IP DNS windows available: "
        f"{result['completed_five_minute_windows']:,}."
    )
    append_scoring_log(
        "Buffered/current incomplete DNS records awaiting a completed window: "
        f"{result['current_incomplete_window_records']:,}."
    )
    return result


def read_scoring_status() -> Dict[str, Any]:
    return _read_json(
        SCORING_STATUS_PATH,
        {
            "state": "not_started",
            "stage": "Not started",
            "progress_percent": 0,
            "message": "No DNS scoring run has started.",
        },
    )


def _get_pid() -> Optional[int]:
    try:
        raw = SCORING_PID_PATH.read_text(encoding="utf-8").strip()
        return int(raw) if raw else None
    except Exception:
        return None


def _process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def is_scoring_running() -> bool:
    status = read_scoring_status()
    state = str(status.get("state") or "").lower()
    if state in TERMINAL_SCORING_STATES:
        try:
            SCORING_PID_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        return False
    pid = _get_pid()
    if not pid:
        return False
    running = _process_running(pid)
    if not running:
        try:
            SCORING_PID_PATH.unlink(missing_ok=True)
        except Exception:
            pass
    return running


def _start_manual_scoring(*, full_log: bool) -> Dict[str, Any]:
    if is_scoring_running():
        return {"started": False, "error": "DNS scoring is already running."}

    card_key = "full_log" if full_log else "run_once"
    mode_name = "full-log" if full_log else "incremental"

    # Every manual run starts with a clean command-output window. Scheduled
    # cron executions remain append-only and are not affected by this reset.
    reset_scoring_log_for_manual_run()
    append_scoring_log(
        "===== New manual DNS scoring request: "
        + ("FULL LOG FROM BYTE ZERO" if full_log else "INCREMENTAL NEW DATA")
        + " ====="
    )

    log_result = check_live_zeek_log()
    if not log_result.get("ok"):
        update_control_card(
            card_key,
            percent=100,
            state="error",
            headline="Cannot start scoring",
            detail=log_result.get("error", "Live DNS log is unavailable."),
        )
        return {"started": False, "error": log_result.get("error")}

    if not SCORING_SCRIPT_PATH.exists():
        error = f"Scoring script not found: {SCORING_SCRIPT_PATH}"
        update_control_card(
            card_key, percent=100, state="error", headline="Script missing", detail=error
        )
        append_scoring_log(error)
        return {"started": False, "error": error}

    if not MODEL_CONFIG_PATH.exists():
        error = "Model artifacts are not ready. Complete Learning Navigator first."
        update_control_card(
            card_key, percent=100, state="error", headline="Model missing", detail=error
        )
        append_scoring_log(error)
        return {"started": False, "error": error}

    SCORING_LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    update_control_card(
        card_key,
        percent=5,
        state="running",
        headline="Starting full-log scoring" if full_log else "Starting DNS scoring",
        detail=log_result["path"],
    )
    _write_json_atomic(
        SCORING_STATUS_PATH,
        {
            "state": "starting",
            "stage": "Starting full-log scoring" if full_log else "Starting",
            "progress_percent": 5,
            "message": (
                "Launching full DNS-log scoring from byte zero. The live cursor and buffer will not be changed."
                if full_log
                else "Launching one incremental DNS scoring pass."
            ),
            "run_mode": mode_name,
            "input_path": log_result["path"],
            "started_at_utc": now_utc(),
        },
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"/opt/dns-ml:{env.get('PYTHONPATH', '')}"
    env["ZEEK_DNS_LOG"] = log_result["path"]
    env["DNS_SCORING_STATUS_PATH"] = str(SCORING_STATUS_PATH)
    env["DNS_SCORING_WINDOWS_PATH"] = str(SCORING_WINDOWS_PATH)
    env["DNS_SCORING_RESIDUALS_PATH"] = str(SCORING_FEATURE_RESIDUALS_PATH)
    env["DNS_SCORING_LATENT_PATH"] = str(SCORING_LATENT_PATH)

    command = [sys.executable, str(SCORING_SCRIPT_PATH), "--once"]
    if full_log:
        command.append("--full-log")

    with SCORING_LIVE_LOG_PATH.open("a", encoding="utf-8") as log_handle:
        log_handle.write(
            f"[{now_utc()}] ===== Manual {mode_name} DNS scoring run started =====\n"
        )
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(SCORING_SCRIPT_PATH.parent),
            env=env,
            start_new_session=True,
        )
    SCORING_PID_PATH.write_text(str(process.pid), encoding="utf-8")
    return {
        "started": True,
        "pid": process.pid,
        "mode": mode_name,
        "input_path": log_result["path"],
        "error": None,
    }


def start_scoring_once() -> Dict[str, Any]:
    """Score only records after the committed live cursor."""
    return _start_manual_scoring(full_log=False)


def start_full_log_scoring() -> Dict[str, Any]:
    """Score the entire active DNS log without changing cursor or buffer state."""
    return _start_manual_scoring(full_log=True)


def stop_scoring() -> Dict[str, Any]:
    pid = _get_pid()
    if not pid or not _process_running(pid):
        return {"stopped": False, "error": None, "message": "No active scoring run."}
    try:
        status = read_scoring_status()
        card_key = "full_log" if str(status.get("run_mode") or "").lower() == "full-log" else "run_once"
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        SCORING_PID_PATH.unlink(missing_ok=True)
        update_control_card(
            card_key,
            percent=100,
            state="stopped",
            headline="Scoring stopped",
            detail="The active manual scoring process was stopped.",
        )
        append_scoring_log("Manual DNS scoring process stopped by the user.")
        return {"stopped": True, "error": None}
    except Exception as exc:
        return {"stopped": False, "error": str(exc)}


def _cron_service_status() -> Dict[str, str]:
    for service in ("cron", "crond"):
        try:
            completed = subprocess.run(
                ["systemctl", "is-active", service],
                capture_output=True,
                text=True,
                timeout=6,
            )
            state = (completed.stdout or completed.stderr).strip()
            if state and state != "unknown":
                return {"service": service, "state": state}
        except Exception:
            continue
    return {"service": "cron/crond", "state": "unknown"}


def _next_five_minute_run(now: Optional[datetime] = None) -> datetime:
    current = now or datetime.now(timezone.utc)
    current = current.replace(second=0, microsecond=0)
    minutes_to_add = 5 - (current.minute % 5)
    if minutes_to_add == 0:
        minutes_to_add = 5
    return current + timedelta(minutes=minutes_to_add)


def _last_scoring_time() -> Optional[str]:
    status = read_scoring_status()
    for key in ("completed_at_utc", "finished_at_utc", "updated_at_utc", "started_at_utc"):
        value = status.get(key)
        if value:
            return str(value)
    if SCORING_LIVE_LOG_PATH.exists():
        return datetime.fromtimestamp(
            SCORING_LIVE_LOG_PATH.stat().st_mtime, tz=timezone.utc
        ).isoformat()
    return None


def activate_scoring_cron() -> Dict[str, Any]:
    update_control_card(
        "activate_task",
        percent=20,
        state="running",
        headline="Preparing cron task",
        detail="Creating a five-minute schedule.",
    )
    wrapper = Path("/opt/dns-ml-gui/scripts/run_dns_scoring_once.sh")
    if not wrapper.exists():
        error = f"Cron wrapper not found: {wrapper}"
        update_control_card(
            "activate_task", percent=100, state="error", headline="Wrapper missing", detail=error
        )
        return {"ok": False, "error": error}

    cron_content = (
        "SHELL=/bin/bash\n"
        "PATH=/opt/dns-ml/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        "*/5 * * * * root /usr/bin/flock -n "
        f"{SCORING_LOCK_PATH} {wrapper} >> {SCORING_LIVE_LOG_PATH} 2>&1\n"
    )
    try:
        SCORING_CRON_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SCORING_CRON_FILE.with_suffix(".tmp")
        tmp.write_text(cron_content, encoding="utf-8")
        os.chmod(tmp, 0o644)
        tmp.replace(SCORING_CRON_FILE)
        SCORING_CRON_DISABLED_PATH.unlink(missing_ok=True)
    except Exception as exc:
        error = f"Unable to install cron task: {exc}"
        update_control_card(
            "activate_task", percent=100, state="error", headline="Activation failed", detail=error
        )
        append_scoring_log(error)
        return {"ok": False, "error": error}

    service = _cron_service_status()
    next_run = _next_five_minute_run().isoformat()
    detail = f"Every 5 minutes • {service['service']}={service['state']} • next {next_run}"
    update_control_card(
        "activate_task",
        percent=100,
        state="success" if service["state"] == "active" else "warning",
        headline="Logging task activated",
        detail=detail,
    )
    append_scoring_log(f"Five-minute DNS scoring cron task activated: {detail}")
    return {
        "ok": True,
        "cron_file": str(SCORING_CRON_FILE),
        "service": service,
        "next_run_utc": next_run,
    }


def check_scoring_cron_status() -> Dict[str, Any]:
    update_control_card(
        "check_task",
        percent=30,
        state="running",
        headline="Checking schedule",
        detail="Inspecting cron configuration and service.",
    )
    active = SCORING_CRON_FILE.exists()
    service = _cron_service_status()
    next_run = _next_five_minute_run().isoformat() if active else None
    last_run = _last_scoring_time()
    normal = active and service["state"] == "active"
    detail = (
        f"task={'active' if active else 'inactive'} • "
        f"{service['service']}={service['state']} • "
        f"next={next_run or 'N/A'} • last={last_run or 'Never'}"
    )
    update_control_card(
        "check_task",
        percent=100,
        state="success" if normal else "warning",
        headline="Task running normally" if normal else "Task needs attention",
        detail=detail,
    )
    append_scoring_log(f"Cron status checked: {detail}")
    return {
        "ok": normal,
        "task_active": active,
        "service": service,
        "next_run_utc": next_run,
        "last_run_utc": last_run,
    }


def deactivate_scoring_cron() -> Dict[str, Any]:
    update_control_card(
        "deactivate_task",
        percent=35,
        state="running",
        headline="Pausing cron task",
        detail="Removing the active schedule while keeping a disabled copy.",
    )
    try:
        if SCORING_CRON_FILE.exists():
            SCORING_CRON_DISABLED_PATH.parent.mkdir(parents=True, exist_ok=True)
            SCORING_CRON_DISABLED_PATH.write_text(
                SCORING_CRON_FILE.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            SCORING_CRON_FILE.unlink()
        detail = f"Scheduled scoring paused. Disabled copy: {SCORING_CRON_DISABLED_PATH}"
        update_control_card(
            "deactivate_task",
            percent=100,
            state="success",
            headline="Logging task deactivated",
            detail=detail,
        )
        append_scoring_log(detail)
        return {"ok": True, "detail": detail}
    except Exception as exc:
        error = f"Unable to deactivate cron task: {exc}"
        update_control_card(
            "deactivate_task", percent=100, state="error", headline="Deactivation failed", detail=error
        )
        append_scoring_log(error)
        return {"ok": False, "error": error}


def sync_run_once_card_from_status() -> Dict[str, Dict[str, Any]]:
    status = read_scoring_status()
    state = str(status.get("state") or "not_started").lower()
    percent = int(status.get("progress_percent", 0) or 0)
    run_mode = str(status.get("run_mode") or "incremental").lower()
    card_key = "full_log" if run_mode == "full-log" else "run_once"
    mode_label = "Full-log scoring" if card_key == "full_log" else "Incremental scoring"

    if state in {"starting", "running", "loading", "scoring", "saving"}:
        update_control_card(
            card_key,
            percent=percent,
            state="running",
            headline=str(status.get("stage") or f"{mode_label} running"),
            detail=str(status.get("message") or "Processing current Zeek DNS records."),
        )
    elif state in {"waiting_input", "waiting", "input_pending"}:
        update_control_card(
            card_key,
            percent=0,
            state="warning",
            headline="Waiting for DNS input",
            detail=str(
                status.get("message")
                or "No DNS records are available yet; the next scheduled interval will retry."
            ),
        )
    elif state in {"finished", "completed", "success", "no_data"}:
        windows = int(status.get("windows_scored", 0) or 0)
        alerts = int(status.get("alerts_written", 0) or 0)
        update_control_card(
            card_key,
            percent=100,
            state="success",
            headline=f"{mode_label} completed",
            detail=f"{windows:,} windows scored • {alerts:,} new alerts written",
        )
    elif state in {"failed", "error"}:
        update_control_card(
            card_key,
            percent=100,
            state="error",
            headline=f"{mode_label} failed",
            detail=str(status.get("error") or status.get("message") or "Unknown error"),
        )
    return load_control_cards()


def load_scoring_windows() -> pd.DataFrame:
    try:
        return pd.read_csv(SCORING_WINDOWS_PATH) if SCORING_WINDOWS_PATH.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_feature_residuals() -> pd.DataFrame:
    try:
        return pd.read_csv(SCORING_FEATURE_RESIDUALS_PATH) if SCORING_FEATURE_RESIDUALS_PATH.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_latent_vectors() -> pd.DataFrame:
    try:
        return pd.read_csv(SCORING_LATENT_PATH) if SCORING_LATENT_PATH.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _stable_autoencoder_public_id(timestamp, src_ip: str) -> str:
    try:
        normalized = pd.Timestamp(timestamp).isoformat()
    except Exception:
        normalized = str(timestamp or "")
    basis = f"autoencoder|{normalized}|{str(src_ip or '')}"
    return "AE-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20].upper()

def load_live_alerts(limit: int = 500) -> pd.DataFrame:
    if not ALERT_JSONL_PATH.exists():
        return pd.DataFrame()

    records: list[dict] = []
    heuristic_names = {
        "heuristic_dga",
        "heuristic_tunnel",
        "heuristic_fastflux",
        "heuristic_resolver_failure",
        "heuristic_rogue_resolver",
    }

    try:
        with ALERT_JSONL_PATH.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue

                methods = item.get("detection_methods") or {}
                if not isinstance(methods, dict):
                    methods = {}
                has_heuristic = any(bool(methods.get(name)) for name in heuristic_names)
                is_autoencoder = (
                    str(item.get("alert_type") or "") == "autoencoder_anomaly"
                    or str(item.get("detection_type") or "") == "deep_learning_autoencoder"
                    or bool(methods.get("autoencoder"))
                )
                if not is_autoencoder or has_heuristic:
                    continue

                explicit = str(item.get("alert_uid") or item.get("id") or "").strip()
                public_id = explicit if explicit.startswith("AE-") else _stable_autoencoder_public_id(
                    item.get("timestamp"), str(item.get("src_ip") or "")
                )
                normalized = dict(item)
                normalized["Alert ID"] = public_id
                normalized["alert_id"] = public_id
                normalized["id"] = public_id
                normalized["alert_uid"] = public_id
                records.append(normalized)

        if not records:
            return pd.DataFrame()
        frame = pd.DataFrame(records[-max(1, int(limit)):])
        preferred = [
            "Alert ID",
            "timestamp",
            "src_ip",
            "severity",
            "alert_type",
            "reconstruction_error",
            "threshold",
            "threshold_ratio",
            "generated_at_utc",
        ]
        columns = [column for column in preferred if column in frame.columns]
        return frame[columns] if columns else frame
    except Exception:
        return pd.DataFrame()
