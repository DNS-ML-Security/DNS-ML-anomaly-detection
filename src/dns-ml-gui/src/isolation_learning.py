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
from src.training_upload import learning_input_ready, learning_gate_message
import time
from src.training_lifecycle import mark_training_started, reset_isolation_run_views


RUN_DIR = Path(os.getenv("DNS_RUN_DIR", "/data/dns-ml/runs"))
MODEL_DIR = Path(os.getenv("DNS_MODEL_DIR", "/data/dns-ml/models"))
ZEEK_MERGED_PATH = Path(
    os.getenv("DNS_ZEEK_MERGED_PATH", "/data/dns-ml/zeek/merged_dns.jsonl")
)

FEATURE_INPUT_PATH = Path("/data/dns-ml/features/dns_training_features.csv")

TRAIN_SCRIPT_PATH = Path(
    os.getenv(
        "DNS_IF_TRAIN_SCRIPT",
        "/opt/dns-ml/train_dns_isolation_forest.py",
    )
)

IF_STATUS_PATH = RUN_DIR / "latest_isolation_forest_status.json"
IF_LOG_PATH = RUN_DIR / "latest_isolation_forest.log"
IF_RESULT_PATH = RUN_DIR / "latest_isolation_forest_result.json"
IF_PID_PATH = RUN_DIR / "latest_isolation_forest.pid"
IF_SCORES_PATH = RUN_DIR / "latest_isolation_forest_scores.csv"
IF_POINTS_PATH = RUN_DIR / "latest_isolation_forest_pca_points.csv"
IF_CONTOUR_PATH = RUN_DIR / "latest_isolation_forest_contour.csv"
IF_SHAP_SUMMARY_PATH = RUN_DIR / "latest_isolation_forest_shap_summary.csv"
IF_SHAP_VALUES_PATH = RUN_DIR / "latest_isolation_forest_shap_values.csv"
IF_SHAP_DEPENDENCY_PATH = RUN_DIR / "latest_isolation_forest_shap_dependency.csv"

IF_MODEL_PATH = MODEL_DIR / "dns_isolation_forest.joblib"
IF_SCALER_PATH = MODEL_DIR / "dns_isolation_forest_scaler.joblib"
IF_PCA_PATH = MODEL_DIR / "dns_isolation_forest_pca.joblib"
IF_CONFIG_PATH = MODEL_DIR / "dns_isolation_forest_config.json"

TERMINAL_STATES = frozenset(
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


def _normalize_state(value: Any) -> str:
    return str(value or "not_started").strip().lower()


def is_terminal_isolation_state(value: Any) -> bool:
    return _normalize_state(value) in TERMINAL_STATES


def _read_json(path: Path, default):
    try:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    temporary.replace(path)


def _get_pid() -> Optional[int]:
    try:
        raw = IF_PID_PATH.read_text(encoding="utf-8").strip()
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


def _pid_matches(pid: int) -> bool:
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.exists():
        return True
    try:
        text = cmdline.read_bytes().replace(b"\x00", b" ").decode(
            "utf-8", errors="replace"
        )
        return TRAIN_SCRIPT_PATH.name in text or str(TRAIN_SCRIPT_PATH) in text
    except Exception:
        return True


def _remove_stale_pid() -> None:
    try:
        IF_PID_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def read_isolation_status() -> Dict[str, Any]:
    return _read_json(
        IF_STATUS_PATH,
        {
            "state": "not_started",
            "stage": "Not started",
            "progress_percent": 0,
            "message": "No Isolation Forest learning run has started.",
            "error": None,
        },
    )


def read_isolation_result() -> Dict[str, Any]:
    return _read_json(IF_RESULT_PATH, {})


def is_isolation_learning_running() -> bool:
    status = read_isolation_status()
    if is_terminal_isolation_state(status.get("state")):
        _remove_stale_pid()
        return False

    pid = _get_pid()
    if not pid:
        return False
    running = _process_running(pid) and _pid_matches(pid)
    if not running:
        _remove_stale_pid()
    return running


def start_isolation_learning() -> Dict[str, Any]:
    if is_isolation_learning_running():
        return {"started": False, "error": "Isolation Forest learning is already running."}

    # DNS GUI UPLOAD WORKFLOW ML GATE
    if not learning_input_ready():
        return {"started": False, "error": learning_gate_message()}
    if not TRAIN_SCRIPT_PATH.exists():
        return {
            "started": False,
            "error": f"Isolation Forest training script not found: {TRAIN_SCRIPT_PATH}",
        }
    if not ZEEK_MERGED_PATH.exists() or ZEEK_MERGED_PATH.stat().st_size == 0:
        return {
            "started": False,
            "error": (
                "The merged Zeek DNS dataset is missing or empty. "
                "Complete the Zeek DNS Log Manager operation first."
            ),
        }

    for path in (
        IF_STATUS_PATH,
        IF_LOG_PATH,
        IF_RESULT_PATH,
        IF_SCORES_PATH,
        IF_POINTS_PATH,
        IF_CONTOUR_PATH,
        IF_SHAP_SUMMARY_PATH,
        IF_SHAP_VALUES_PATH,
        IF_SHAP_DEPENDENCY_PATH,
    ):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.unlink(missing_ok=True)
        except Exception:
            pass

    _write_json(
        IF_STATUS_PATH,
        {
            "state": "starting",
            "stage": "Starting",
            "progress_percent": 1,
            "message": "Starting CPU-only Isolation Forest learning in debug mode.",
            "training_source": str(FEATURE_INPUT_PATH),
            "cpu_only": True,
            "debug_enabled": True,
            "error": None,
        },
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"/opt/dns-ml:{env.get('PYTHONPATH', '')}"
    env["DNS_PRECOMPUTED_FEATURES"] = str(FEATURE_INPUT_PATH)
    env["DNS_IF_STATUS_PATH"] = str(IF_STATUS_PATH)
    env["DNS_IF_RESULT_PATH"] = str(IF_RESULT_PATH)
    env["DNS_IF_SCORES_PATH"] = str(IF_SCORES_PATH)
    env["DNS_IF_POINTS_PATH"] = str(IF_POINTS_PATH)
    env["DNS_IF_CONTOUR_PATH"] = str(IF_CONTOUR_PATH)
    env["DNS_IF_SHAP_SUMMARY_PATH"] = str(IF_SHAP_SUMMARY_PATH)
    env["DNS_IF_SHAP_VALUES_PATH"] = str(IF_SHAP_VALUES_PATH)
    env["DNS_IF_SHAP_DEPENDENCY_PATH"] = str(IF_SHAP_DEPENDENCY_PATH)
    env["IF_DEBUG"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "-1"

    log_handle = IF_LOG_PATH.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [sys.executable, str(TRAIN_SCRIPT_PATH)],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(TRAIN_SCRIPT_PATH.parent),
            env=env,
            start_new_session=True,
        )
    finally:
        log_handle.close()

    IF_PID_PATH.write_text(str(process.pid), encoding="utf-8")
    mark_training_started("isolation_forest")
    return {
        "started": True,
        "pid": process.pid,
        "training_source": str(FEATURE_INPUT_PATH),
        "error": None,
    }


def stop_isolation_learning() -> Dict[str, Any]:
    pid = _get_pid()
    running = bool(pid and _process_running(pid) and _pid_matches(pid))

    if running:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)

            deadline = time.time() + 5.0
            while time.time() < deadline and _process_running(pid):
                time.sleep(0.2)

            if _process_running(pid):
                os.killpg(pgid, signal.SIGKILL)
                time.sleep(0.2)
        except ProcessLookupError:
            pass
        except Exception as exc:
            return {"stopped": False, "noop": False, "error": str(exc)}

    _remove_stale_pid()
    reset_isolation_run_views()

    return {
        "stopped": True,
        "noop": False,
        "error": None,
        "message": (
            "Isolation Forest learning was stopped/reset. Partial status, logs, "
            "charts and result tables were cleared. The shared feature dataset "
            "was preserved so Start Learning can be used again."
        ),
    }



def _read_log() -> str:
    if not IF_LOG_PATH.exists():
        return "No Isolation Forest learning output is available yet."
    return IF_LOG_PATH.read_text(encoding="utf-8", errors="replace")


def read_isolation_log_tail(max_lines: int = 1500) -> str:
    try:
        count = max(1, int(max_lines))
    except (TypeError, ValueError):
        count = 1500
    try:
        return "\n".join(_read_log().splitlines()[-count:])
    except Exception as exc:
        return f"Unable to read the live Isolation Forest log: {exc}"


def read_complete_isolation_log() -> str:
    try:
        return _read_log()
    except Exception as exc:
        return f"Unable to read the completed Isolation Forest log: {exc}"


def isolation_log_signature() -> str:
    try:
        stat = IF_LOG_PATH.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"
    except Exception:
        return "missing"


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def load_isolation_scores() -> pd.DataFrame:
    return _load_csv(IF_SCORES_PATH)


def load_isolation_points() -> pd.DataFrame:
    return _load_csv(IF_POINTS_PATH)


def load_isolation_contour() -> pd.DataFrame:
    return _load_csv(IF_CONTOUR_PATH)


def load_isolation_shap_summary() -> pd.DataFrame:
    return _load_csv(IF_SHAP_SUMMARY_PATH)


def load_isolation_shap_values() -> pd.DataFrame:
    return _load_csv(IF_SHAP_VALUES_PATH)


def load_isolation_shap_dependency() -> pd.DataFrame:
    return _load_csv(IF_SHAP_DEPENDENCY_PATH)
