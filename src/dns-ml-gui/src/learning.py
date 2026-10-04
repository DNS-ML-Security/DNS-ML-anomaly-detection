# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.settings import (
    LEARNING_STATUS_PATH,
    LEARNING_LOG_PATH,
    LEARNING_HISTORY_PATH,
    LEARNING_RECONSTRUCTION_PATH,
    LEARNING_RESULT_PATH,
    LEARNING_PID_PATH,
    ZEEK_MERGED_PATH,
)
from src.training_upload import learning_input_ready, learning_gate_message
import time
from src.training_lifecycle import mark_training_started, reset_autoencoder_run_views


FEATURE_INPUT_PATH = Path("/data/dns-ml/features/dns_training_features.csv")

TRAIN_SCRIPT_PATH = Path(
    os.getenv("DNS_TRAIN_SCRIPT", "/opt/dns-ml/train_dns_autoencoder.py")
)


TERMINAL_LEARNING_STATES = frozenset(
    {
        "finished",
        "completed",
        "complete",
        "success",
        "failed",
        "error",
        "stopped",
        "cancelled",
        "canceled",
    }
)


def normalize_learning_state(value: Any) -> str:
    return str(value or "not_started").strip().lower()


def is_terminal_learning_state(value: Any) -> bool:
    return normalize_learning_state(value) in TERMINAL_LEARNING_STATES


def read_json(path: Path, default):
    try:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    tmp.replace(path)


def get_pid() -> Optional[int]:
    try:
        if not LEARNING_PID_PATH.exists():
            return None
        raw = LEARNING_PID_PATH.read_text(encoding="utf-8").strip()
        return int(raw) if raw else None
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


def _remove_stale_pid_file() -> None:
    try:
        LEARNING_PID_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def _pid_matches_training_process(pid: int) -> bool:
    """Avoid treating a recycled PID as the active training process."""
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    if not proc_cmdline.exists():
        # Non-Linux platforms do not necessarily expose /proc.
        return True

    try:
        raw = proc_cmdline.read_bytes().replace(b"\x00", b" ").decode(
            "utf-8", errors="replace"
        )
        if not raw.strip():
            return False
        return (
            str(TRAIN_SCRIPT_PATH) in raw
            or TRAIN_SCRIPT_PATH.name in raw
        )
    except Exception:
        return True


def is_learning_running() -> bool:
    # A completed/failed/stopped run is terminal even if an obsolete PID file
    # remains. This keeps the Stop Learning button disabled after completion.
    status = read_json(LEARNING_STATUS_PATH, {})
    if is_terminal_learning_state(status.get("state")):
        _remove_stale_pid_file()
        return False

    pid = get_pid()
    if not pid:
        return False

    running = is_process_running(pid) and _pid_matches_training_process(pid)
    if not running:
        _remove_stale_pid_file()
    return running


def start_learning() -> Dict[str, Any]:
    if is_learning_running():
        return {"started": False, "error": "Learning is already running."}

    # DNS GUI UPLOAD WORKFLOW ML GATE
    if not learning_input_ready():
        return {"started": False, "error": learning_gate_message()}

    if not TRAIN_SCRIPT_PATH.exists():
        return {
            "started": False,
            "error": f"Training script not found: {TRAIN_SCRIPT_PATH}",
        }

    if not ZEEK_MERGED_PATH.exists() or ZEEK_MERGED_PATH.stat().st_size == 0:
        return {
            "started": False,
            "error": (
                "The merged Zeek DNS dataset does not exist or is empty. "
                "Complete the Zeek Log Manager operation first."
            ),
        }

    for path in (
        LEARNING_STATUS_PATH,
        LEARNING_LOG_PATH,
        LEARNING_HISTORY_PATH,
        LEARNING_RECONSTRUCTION_PATH,
        LEARNING_RESULT_PATH,
    ):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.unlink(missing_ok=True)
        except Exception:
            pass

    write_json(
        LEARNING_STATUS_PATH,
        {
            "state": "starting",
            "stage": "Starting",
            "progress_percent": 1,
            "message": "Learning is starting from the shared precomputed feature dataset.",
            "training_source": str(FEATURE_INPUT_PATH),
            "error": None,
        },
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = (
        f"/opt/dns-ml:{env.get('PYTHONPATH', '')}"
    )
    env["DNS_PRECOMPUTED_FEATURES"] = str(FEATURE_INPUT_PATH)
    env["DNS_LEARNING_STATUS_PATH"] = str(LEARNING_STATUS_PATH)
    env["DNS_LEARNING_HISTORY_PATH"] = str(LEARNING_HISTORY_PATH)
    env["DNS_LEARNING_RECONSTRUCTION_PATH"] = str(
        LEARNING_RECONSTRUCTION_PATH
    )
    env["DNS_LEARNING_RESULT_PATH"] = str(LEARNING_RESULT_PATH)

    log_handle = LEARNING_LOG_PATH.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(TRAIN_SCRIPT_PATH)],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=str(TRAIN_SCRIPT_PATH.parent),
        env=env,
        start_new_session=True,
    )
    log_handle.close()

    LEARNING_PID_PATH.write_text(str(process.pid), encoding="utf-8")
    mark_training_started("autoencoder")
    return {
        "started": True,
        "pid": process.pid,
        "training_source": str(FEATURE_INPUT_PATH),
        "error": None,
    }


def stop_learning() -> Dict[str, Any]:
    pid = get_pid()
    running = bool(pid and is_process_running(pid))

    if running:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)

            deadline = time.time() + 5.0
            while time.time() < deadline and is_process_running(pid):
                time.sleep(0.2)

            if is_process_running(pid):
                os.killpg(pgid, signal.SIGKILL)
                time.sleep(0.2)
        except ProcessLookupError:
            pass
        except Exception as exc:
            return {"stopped": False, "error": str(exc)}

    try:
        LEARNING_PID_PATH.unlink(missing_ok=True)
    except Exception:
        pass

    reset_autoencoder_run_views()

    return {
        "stopped": True,
        "error": None,
        "message": (
            "Autoencoder learning was stopped/reset. Partial status, logs, "
            "charts and result tables were cleared. The shared feature dataset "
            "was preserved so Start Learning can be used again."
        ),
    }



def read_learning_status() -> Dict[str, Any]:
    return read_json(
        LEARNING_STATUS_PATH,
        {
            "state": "not_started",
            "stage": "Not started",
            "progress_percent": 0,
            "message": "No learning run has started.",
            "error": None,
        },
    )


def read_learning_result() -> Dict[str, Any]:
    return read_json(LEARNING_RESULT_PATH, {})


def _read_learning_log_file() -> str:
    if not LEARNING_LOG_PATH.exists():
        return "No learning output is available yet."

    return LEARNING_LOG_PATH.read_text(
        encoding="utf-8",
        errors="replace",
    )


def read_log_tail(max_lines: int = 300) -> str:
    """Read a finite live tail. This function never accepts ``None``."""
    try:
        line_count = int(max_lines)
    except (TypeError, ValueError):
        line_count = 300
    line_count = max(1, line_count)

    try:
        lines = _read_learning_log_file().splitlines()
        return "\n".join(lines[-line_count:])
    except Exception as exc:
        return f"Unable to read the live learning log: {exc}"


def read_complete_log() -> str:
    """Read the entire completed log without a nullable line-count argument."""
    try:
        return _read_learning_log_file()
    except Exception as exc:
        return f"Unable to read the completed learning log: {exc}"


def learning_log_signature() -> str:
    """Return a stable identifier used by the GUI to cache the final summary."""
    try:
        stat = LEARNING_LOG_PATH.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"
    except Exception:
        return "missing"


def load_history_df() -> pd.DataFrame:
    if not LEARNING_HISTORY_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(LEARNING_HISTORY_PATH)
    except Exception:
        return pd.DataFrame()


def load_reconstruction_df() -> pd.DataFrame:
    if not LEARNING_RECONSTRUCTION_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(LEARNING_RECONSTRUCTION_PATH)
    except Exception:
        return pd.DataFrame()
