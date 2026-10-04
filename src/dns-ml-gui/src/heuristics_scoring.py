# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

ZEEK_CURRENT_DIR = Path("/opt/zeek/logs/current")
HEUR_SCORING_HOME = Path("/data/dns-ml/heuristics_scoring")
HEUR_STATUS_PATH = HEUR_SCORING_HOME / "status.json"
HEUR_ACTION_STATUS_PATH = HEUR_SCORING_HOME / "control_cards.json"
HEUR_LIVE_LOG_PATH = HEUR_SCORING_HOME / "live_scoring.log"
HEUR_PID_PATH = HEUR_SCORING_HOME / "scoring.pid"
HEUR_STATE_PATH = Path("/data/dns-ml/state/dns_heuristics_log_state.json")
HEUR_BUFFER_PATH = Path("/data/dns-ml/state/dns_heuristics_event_buffer.jsonl")
HEUR_ALERT_PATH = Path("/data/dns-ml/alerts/dns_heuristics_alerts.jsonl")
HEUR_SCORING_SCRIPT_PATH = Path("/opt/dns-ml/score_dns_heuristics_live.py")
HEUR_WRAPPER_PATH = Path("/opt/dns-ml-gui/scripts/run_dns_heuristics_scoring_once.sh")
HEUR_CRON_FILE = Path("/etc/cron.d/dns-ml-heuristics-scoring")
HEUR_CRON_DISABLED_PATH = HEUR_SCORING_HOME / "dns-ml-heuristics-scoring.cron.disabled"
HEUR_CRON_LAST_PATH = HEUR_SCORING_HOME / "cron_last_run.json"

SUPPORTED_CURRENT_NAMES = ("dns.log", "dns.jsonl", "dns.json")
SUPPORTED_CURRENT_GLOBS = ("dns*.log", "dns*.jsonl", "dns*.json")
WINDOW_SIZE = "5min"
TERMINAL_STATES = {"finished", "completed", "success", "failed", "error", "stopped", "no_data", "busy"}

CARD_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "check_logs": {"title": "Check Live Zeek Logs", "percent": 0, "state": "idle", "headline": "Not checked", "detail": "Checks live Zeek DNS and the independent heuristic cursor."},
    "run_once": {"title": "Run Logging Once", "percent": 0, "state": "idle", "headline": "Not run", "detail": "Evaluates new completed five-minute windows using heuristic rules only."},
    "full_log": {"title": "Run Scoring From Scratch", "percent": 0, "state": "idle", "headline": "Not run", "detail": "Evaluates the entire current DNS log without changing the heuristic cursor."},
    "activate_task": {"title": "Activate Logging Task", "percent": 0, "state": "idle", "headline": "Not activated", "detail": "Installs the independent heuristic cron task every five minutes."},
    "check_task": {"title": "Check Task Status", "percent": 0, "state": "idle", "headline": "Not checked", "detail": "Checks the heuristic cron/service, last run and next run."},
    "deactivate_task": {"title": "Deactivate Logging Task", "percent": 0, "state": "idle", "headline": "Not deactivated", "detail": "Disables only the scheduled heuristic task."},
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


def reset_live_log(action: str = "") -> None:
    HEUR_LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone().isoformat(timespec="seconds")
    content = ""
    if action:
        content = (
            "=" * 78 + "\n"
            "DNS HEURISTICS SCORING — RULE BASED\n"
            "=" * 78 + "\n"
            f"[{now}] ACTION: {action}\n"
            "Previous Live recording command output: FLUSHED\n"
            + "-" * 78 + "\n"
        )
    tmp = HEUR_LIVE_LOG_PATH.with_suffix(HEUR_LIVE_LOG_PATH.suffix + ".reset")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(HEUR_LIVE_LOG_PATH)


def append_live_log(message: str) -> None:
    HEUR_LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HEUR_LIVE_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(str(message).rstrip() + "\n")


def read_live_log(max_lines: Optional[int] = None) -> str:
    if not HEUR_LIVE_LOG_PATH.exists():
        return "No DNS heuristic scoring output is available yet."
    try:
        lines = HEUR_LIVE_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        if max_lines is not None:
            lines = lines[-max(1, int(max_lines)):]
        return "\n".join(lines)
    except Exception as exc:
        return f"Unable to read DNS heuristic scoring output: {exc}"


def load_cards() -> Dict[str, Dict[str, Any]]:
    stored = _read_json(HEUR_ACTION_STATUS_PATH, {})
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
    _write_json_atomic(HEUR_ACTION_STATUS_PATH, cards)
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
    state = _read_json(HEUR_STATE_PATH, {})
    stat = path.stat()
    if str(state.get("path") or "") != str(path):
        return 0, "No heuristic cursor exists for this file"
    if state.get("inode") is not None and int(state.get("inode")) != int(stat.st_ino):
        return 0, "DNS log inode changed; heuristic scoring starts at byte zero"
    offset = int(state.get("offset", 0) or 0)
    if offset < 0 or offset > stat.st_size:
        return 0, "Heuristic cursor is invalid; starts at byte zero"
    return offset, "Heuristic cursor is valid"


def check_live_zeek_log() -> Dict[str, Any]:
    reset_live_log("1 · Check Live Zeek Logs")
    update_card("check_logs", percent=15, state="running", headline="Scanning current logs", detail=str(ZEEK_CURRENT_DIR))
    append_live_log(f"Checking live Zeek DNS logs under {ZEEK_CURRENT_DIR}")

    path = find_live_zeek_dns_log()
    if path is None:
        error = f"No supported live Zeek DNS log was found under {ZEEK_CURRENT_DIR}"
        update_card("check_logs", percent=100, state="error", headline="No DNS log found", detail=error)
        append_live_log(error)
        return {"error": error}

    stat = path.stat()
    offset, cursor_message = _effective_offset(path)
    try:
        with path.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read()
        text = payload.decode("utf-8", errors="ignore")
        new_lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    except Exception:
        new_lines = []

    buffered_records = 0
    try:
        if HEUR_BUFFER_PATH.exists() and HEUR_BUFFER_PATH.stat().st_size:
            buffered_records = len(pd.read_json(HEUR_BUFFER_PATH, lines=True))
    except Exception:
        buffered_records = 0

    result = {
        "ok": True,
        "path": str(path),
        "format": _detect_format(path),
        "file_size_bytes": int(stat.st_size),
        "committed_offset_bytes": int(offset),
        "cursor_message": cursor_message,
        "new_lines_after_cursor": len(new_lines),
        "buffered_records": buffered_records,
        "modified_at_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }
    detail = f"{path.name} • {result['format']} • {len(new_lines):,} new lines • cursor {offset:,}/{stat.st_size:,} bytes"
    update_card("check_logs", percent=100, state="success", headline="Live DNS log ready", detail=detail)
    append_live_log(f"Live DNS log ready: {detail}")
    append_live_log(f"Heuristic cursor: {cursor_message}")
    append_live_log(f"New text records after cursor: {len(new_lines):,}")
    append_live_log(f"Buffered/current-window records: {buffered_records:,}")
    return result


def read_status() -> Dict[str, Any]:
    return _read_json(HEUR_STATUS_PATH, {"state": "not_started", "stage": "Not started", "progress_percent": 0, "message": "No DNS heuristic scoring run has started."})


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def is_running() -> bool:
    try:
        if HEUR_PID_PATH.exists():
            raw = HEUR_PID_PATH.read_text(encoding="utf-8").strip()
            if raw and _pid_running(int(raw)):
                return True
    except Exception:
        pass
    try:
        status = read_status()
        state = str(status.get("state", "")).lower()
        pid = int(status.get("pid", 0) or 0)
        if state in {"starting", "launching", "running"} and pid and _pid_running(pid):
            return True
    except Exception:
        pass
    return False


def start_scoring(mode: str = "once") -> Dict[str, Any]:
    if mode not in {"once", "full"}:
        return {"error": f"Unsupported mode: {mode}"}
    if is_running():
        return {"error": "DNS heuristic scoring is already running."}
    if not HEUR_WRAPPER_PATH.exists():
        return {"error": f"Missing scoring wrapper: {HEUR_WRAPPER_PATH}"}

    action = "3 · Run Scoring From Scratch" if mode == "full" else "2 · Run Logging Once"
    reset_live_log(action)
    card_key = "full_log" if mode == "full" else "run_once"
    update_card(card_key, percent=5, state="running", headline="Starting", detail="Launching DNS heuristic scoring.")

    HEUR_SCORING_HOME.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([str(HEUR_WRAPPER_PATH), mode], cwd="/opt/dns-ml", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    HEUR_PID_PATH.write_text(str(process.pid), encoding="utf-8")
    _write_json_atomic(
        HEUR_STATUS_PATH,
        {
            "state": "running",
            "stage": "Starting heuristic scorer",
            "progress_percent": 5,
            "message": "DNS heuristic scoring process launched.",
            "run_mode": mode,
            "pid": process.pid,
            "launcher_pid": process.pid,
            "started_at_utc": now_utc(),
        },
    )
    update_card(card_key, percent=5, state="running", headline="Scorer launched", detail=f"PID {process.pid} · Waiting for live output.")
    return {"pid": process.pid, "mode": mode, "state": "running", "error": None}


def sync_cards_from_status() -> Dict[str, Dict[str, Any]]:
    cards = load_cards()
    status = read_status()
    mode = str(status.get("run_mode") or "once")
    key = "full_log" if mode == "full" else "run_once"
    state = str(status.get("state") or "").lower()
    if state in {"running", "starting", "launching"}:
        update_card(key, percent=status.get("progress_percent", 5), state="running", headline=status.get("stage", "Running"), detail=status.get("message", "Heuristic scoring is running."))
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
    elif state in {"finished", "completed", "success", "no_data"}:
        update_card(key, percent=100, state="success", headline=status.get("stage", "Completed"), detail=status.get("message", "Heuristic scoring completed."))
    elif state in {"failed", "error"}:
        update_card(key, percent=100, state="error", headline="Failed", detail=status.get("message", "Heuristic scoring failed."))
    return load_cards()


def _cron_service_state() -> tuple[str, bool]:
    for name in ("cron", "crond"):
        try:
            result = subprocess.run(["systemctl", "is-active", name], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return name, True
        except Exception:
            pass
    return "cron/crond", False


def activate_cron() -> Dict[str, Any]:
    reset_live_log("4 · Activate Logging Task")
    HEUR_SCORING_HOME.mkdir(parents=True, exist_ok=True)
    cron_text = (
        "SHELL=/bin/bash\n"
        "PATH=/opt/dns-ml/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        "*/5 * * * * root /usr/bin/flock -n /data/dns-ml/heuristics_scoring/cron.lock "
        "/opt/dns-ml-gui/scripts/run_dns_heuristics_scoring_once.sh once --scheduled >/dev/null 2>&1\n"
    )
    try:
        HEUR_CRON_FILE.write_text(cron_text, encoding="utf-8")
        os.chmod(HEUR_CRON_FILE, 0o644)
        HEUR_CRON_DISABLED_PATH.write_text(cron_text, encoding="utf-8")
        update_card("activate_task", percent=100, state="success", headline="Task activated", detail="Independent heuristic scoring runs every five minutes.")
        append_live_log(f"Installed heuristic cron: {HEUR_CRON_FILE}")
        return {"enabled": True, "cron_file": str(HEUR_CRON_FILE), "error": None}
    except Exception as exc:
        update_card("activate_task", percent=100, state="error", headline="Activation failed", detail=str(exc))
        append_live_log(f"Activation failed: {exc}")
        return {"error": str(exc)}


def check_cron_status() -> Dict[str, Any]:
    reset_live_log("5 · Check Task Status")
    service, active = _cron_service_state()
    last = _read_json(HEUR_CRON_LAST_PATH, {})
    now = datetime.now().astimezone()
    minute = (now.minute // 5 + 1) * 5
    if minute >= 60:
        next_run = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    else:
        next_run = now.replace(minute=minute, second=0, microsecond=0)
    result = {
        "enabled": HEUR_CRON_FILE.exists(),
        "cron_file": str(HEUR_CRON_FILE),
        "cron_service": service,
        "cron_service_active": active,
        "last_run": last,
        "next_run": next_run.isoformat(),
    }
    update_card("check_task", percent=100, state="success" if result["enabled"] and active else "warning", headline="Task checked", detail=f"enabled={result['enabled']} • {service} active={active}")
    for key, value in result.items():
        append_live_log(f"{key}: {value}")
    return result


def deactivate_cron() -> Dict[str, Any]:
    reset_live_log("6 · Deactivate Logging Task")
    try:
        if HEUR_CRON_FILE.exists():
            text = HEUR_CRON_FILE.read_text(encoding="utf-8")
            HEUR_CRON_DISABLED_PATH.parent.mkdir(parents=True, exist_ok=True)
            HEUR_CRON_DISABLED_PATH.write_text(text, encoding="utf-8")
            HEUR_CRON_FILE.unlink()
        update_card("deactivate_task", percent=100, state="success", headline="Task deactivated", detail="Only the heuristic scheduled task was disabled.")
        append_live_log("Heuristic scheduled task deactivated. Models, alerts, cursor and buffer were retained.")
        return {"enabled": False, "error": None}
    except Exception as exc:
        update_card("deactivate_task", percent=100, state="error", headline="Deactivation failed", detail=str(exc))
        append_live_log(f"Deactivation failed: {exc}")
        return {"error": str(exc)}


def load_live_alerts(limit: int = 500) -> pd.DataFrame:
    if not HEUR_ALERT_PATH.exists():
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    try:
        with HEUR_ALERT_PATH.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    records.append(item)
        if not records:
            return pd.DataFrame()
        frame = pd.DataFrame(records[-max(1, int(limit)):])
        if "heuristics_fired" in frame.columns:
            frame["heuristics_fired"] = frame["heuristics_fired"].apply(lambda value: ", ".join(value) if isinstance(value, list) else str(value))
        preferred = ["id", "timestamp", "src_ip", "severity", "threat_score", "status", "heuristics_fired", "detection_type", "scoring_mode"]
        columns = [column for column in preferred if column in frame.columns]
        return frame[columns] if columns else frame
    except Exception:
        return pd.DataFrame()
