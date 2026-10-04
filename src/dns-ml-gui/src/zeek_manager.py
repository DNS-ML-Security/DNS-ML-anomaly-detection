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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.settings import (
    ZEEK_LOG_ROOT,
    ZEEK_MANAGER_SCRIPT,
    ZEEK_MERGED_PATH,
    ZEEK_STATS_PATH,
    ZEEK_SOURCE_IP_STATS_PATH,
    ZEEK_MERGE_STATUS_PATH,
    ZEEK_MERGE_MANIFEST_PATH,
    ZEEK_MERGE_LOG_PATH,
    ZEEK_MERGE_PID_PATH,
)


def read_json(path: Path, default):
    try:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def discover_source_files(root: Path = ZEEK_LOG_ROOT) -> List[Path]:
    if not root.exists():
        return []

    files: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.startswith("dns") and (
            name.endswith(".log") or name.endswith(".log.gz")
        ):
            files.append(path)

    current = root / "current" / "dns.log"
    if current.exists():
        files.append(current)

    unique: Dict[str, Path] = {}
    for path in files:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        unique[key] = path

    return sorted(unique.values(), key=lambda p: str(p))


def current_source_signature() -> Tuple[str, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []

    for path in discover_source_files():
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
            rows.append({"path": str(path), "error": str(exc)})

    raw = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest(), rows


def get_pid() -> Optional[int]:
    try:
        if not ZEEK_MERGE_PID_PATH.exists():
            return None
        value = ZEEK_MERGE_PID_PATH.read_text(encoding="utf-8").strip()
        return int(value) if value else None
    except Exception:
        return None


def is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def is_merge_running() -> bool:
    pid = get_pid()
    if not pid:
        return False

    running = is_process_running(pid)
    if not running:
        try:
            ZEEK_MERGE_PID_PATH.unlink(missing_ok=True)
        except Exception:
            pass
    return running


def read_merge_status() -> Dict[str, Any]:
    return read_json(
        ZEEK_MERGE_STATUS_PATH,
        {
            "state": "not_started",
            "stage": "Not started",
            "progress_percent": 0,
            "message": "Zeek log merge has not started.",
            "files_total": 0,
            "files_processed": 0,
            "lines_read": 0,
            "records_inserted": 0,
            "duplicates": 0,
            "invalid_lines": 0,
            "unique_src_ips_seen": 0,
            "error": None,
        },
    )


def read_stats() -> Dict[str, Any]:
    return read_json(ZEEK_STATS_PATH, {})


def read_manifest() -> Dict[str, Any]:
    return read_json(ZEEK_MERGE_MANIFEST_PATH, {})


def needs_merge() -> bool:
    if is_merge_running():
        return False

    if not ZEEK_MERGED_PATH.exists() or not ZEEK_STATS_PATH.exists():
        return True

    current_signature, _ = current_source_signature()
    manifest = read_manifest()
    return current_signature != manifest.get("signature")


def start_merge(force: bool = False) -> Dict[str, Any]:
    if is_merge_running():
        return {
            "started": False,
            "skipped": False,
            "error": "A Zeek DNS merge is already running.",
        }

    if not ZEEK_MANAGER_SCRIPT.exists():
        return {
            "started": False,
            "skipped": False,
            "error": f"Zeek manager script not found: {ZEEK_MANAGER_SCRIPT}",
        }

    if not force and not needs_merge():
        return {
            "started": False,
            "skipped": True,
            "error": None,
            "message": "The merged dataset is already current.",
        }

    ZEEK_MERGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["ZEEK_LOG_ROOT"] = str(ZEEK_LOG_ROOT)
    env["DNS_ZEEK_OUTPUT_DIR"] = str(ZEEK_MERGED_PATH.parent)

    log_handle = ZEEK_MERGE_LOG_PATH.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(ZEEK_MANAGER_SCRIPT)],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=str(ZEEK_MANAGER_SCRIPT.parent),
        env=env,
        start_new_session=True,
    )
    log_handle.close()

    # The child script also writes the PID. Writing it here closes the small race.
    ZEEK_MERGE_PID_PATH.write_text(str(process.pid), encoding="utf-8")

    return {
        "started": True,
        "skipped": False,
        "pid": process.pid,
        "error": None,
    }


def stop_merge() -> Dict[str, Any]:
    pid = get_pid()
    if not pid:
        return {"stopped": False, "error": "No Zeek merge process is running."}

    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        ZEEK_MERGE_PID_PATH.unlink(missing_ok=True)
        return {"stopped": True, "error": None}
    except Exception as exc:
        return {"stopped": False, "error": str(exc)}


def read_log_tail(max_lines: int = 200) -> str:
    if not ZEEK_MERGE_LOG_PATH.exists():
        return "No Zeek merge output is available yet."

    try:
        lines = ZEEK_MERGE_LOG_PATH.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception as exc:
        return f"Unable to read Zeek merge log: {exc}"


def artifact_health() -> Dict[str, bool]:
    return {
        "merged_dataset": ZEEK_MERGED_PATH.exists(),
        "stats": ZEEK_STATS_PATH.exists(),
        "source_ip_stats": ZEEK_SOURCE_IP_STATS_PATH.exists(),
        "manifest": ZEEK_MERGE_MANIFEST_PATH.exists(),
    }

