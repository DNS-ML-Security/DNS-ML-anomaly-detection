# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src import training_upload

DATA = Path("/data/dns-ml")
RUN_DIR = DATA / "runs"
MODEL_DIR = DATA / "models"
STATE_DIR = DATA / "training_lifecycle"
STATE_PATH = STATE_DIR / "state.json"

AE_MODEL_FILES = (
    MODEL_DIR / "dns_autoencoder.keras",
    MODEL_DIR / "dns_scaler.joblib",
    MODEL_DIR / "dns_autoencoder_config.json",
)
IF_MODEL_FILES = (
    MODEL_DIR / "dns_isolation_forest.joblib",
    MODEL_DIR / "dns_isolation_forest_scaler.joblib",
    MODEL_DIR / "dns_isolation_forest_pca.joblib",
    MODEL_DIR / "dns_isolation_forest_config.json",
)

AE_RUN_FILES = (
    "latest_training_status.json",
    "latest_training.log",
    "latest_training_history.csv",
    "latest_reconstruction_errors.csv",
    "latest_training_result.json",
    "latest_training.pid",
)
IF_RUN_FILES = (
    "latest_isolation_forest_status.json",
    "latest_isolation_forest.log",
    "latest_isolation_forest_result.json",
    "latest_isolation_forest.pid",
    "latest_isolation_forest_scores.csv",
    "latest_isolation_forest_pca_points.csv",
    "latest_isolation_forest_contour.csv",
    "latest_isolation_forest_shap_summary.csv",
    "latest_isolation_forest_shap_values.csv",
    "latest_isolation_forest_shap_dependency.csv",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state() -> Dict[str, Any]:
    return {
        "zeek_manager_frozen": False,
        "training_session_started": False,
        "started_models": [],
        "last_started_model": None,
        "updated_at_utc": _now(),
        "message": "No ML training session has started for the current prepared dataset.",
    }


def _read_json(path: Path, default):
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_state() -> Dict[str, Any]:
    result = _default_state()
    value = _read_json(STATE_PATH, {})
    if isinstance(value, dict):
        result.update(value)
    return result


def _save(**changes) -> Dict[str, Any]:
    state = read_state()
    state.update(changes)
    state["updated_at_utc"] = _now()
    _write_json(STATE_PATH, state)
    return state


def mark_training_started(model_name: str) -> Dict[str, Any]:
    state = read_state()
    models = list(state.get("started_models") or [])
    if model_name not in models:
        models.append(model_name)
    return _save(
        zeek_manager_frozen=True,
        training_session_started=True,
        started_models=models,
        last_started_model=model_name,
        message=(
            "ML training has started. Zeek DNS Log Manager is frozen until "
            "Purge All is used from either ML learning section."
        ),
    )


def zeek_manager_frozen() -> bool:
    return bool(read_state().get("zeek_manager_frozen"))


def _pid_running(pid_path: Path, token: str) -> bool:
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        cmd = Path(f"/proc/{pid}/cmdline")
        if cmd.exists():
            text = cmd.read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace"
            )
            if token not in text:
                return False
        return True
    except Exception:
        return False


def any_ml_running() -> bool:
    return (
        _pid_running(RUN_DIR / "latest_training.pid", "train_dns_autoencoder.py")
        or _pid_running(
            RUN_DIR / "latest_isolation_forest.pid",
            "train_dns_isolation_forest.py",
        )
    )


def _unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def reset_autoencoder_run_views() -> None:
    for name in AE_RUN_FILES:
        _unlink(RUN_DIR / name)


def reset_isolation_run_views() -> None:
    for name in IF_RUN_FILES:
        _unlink(RUN_DIR / name)


def reset_all_run_views() -> None:
    reset_autoencoder_run_views()
    reset_isolation_run_views()


def _remove_model_backups() -> None:
    backup_dir = MODEL_DIR / "backup"
    if backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)

    for pattern in (
        "dns_autoencoder.keras.*.bak",
        "dns_scaler.joblib.*.bak",
        "dns_autoencoder_config.json.*.bak",
        "dns_isolation_forest*.backup_*",
    ):
        for path in MODEL_DIR.glob(pattern):
            _unlink(path)


def clear_training_data() -> Dict[str, Any]:
    if any_ml_running():
        return {
            "cleared": False,
            "error": "Cannot clear training data while an ML learning process is running.",
        }

    if zeek_manager_frozen():
        return {
            "cleared": False,
            "error": (
                "Zeek DNS Log Manager is frozen because ML training has started. "
                "Use Purge All under either ML section to reset both models and "
                "unlock a new training-data cycle."
            ),
        }

    reset_all_run_views()
    result = training_upload.clear_training_cycle()
    if result.get("error"):
        return result

    _write_json(STATE_PATH, _default_state())
    return {
        "cleared": True,
        "error": None,
        "message": (
            "Training data, prepared features, and ML result views were cleared. "
            "A new JSON/JSONL upload cycle can begin."
        ),
    }


def purge_all() -> Dict[str, Any]:
    if any_ml_running():
        return {
            "purged": False,
            "error": (
                "Purge All is disabled while Autoencoder or Isolation Forest "
                "learning is running. Stop the active learning process first."
            ),
        }

    reset_all_run_views()

    for path in AE_MODEL_FILES + IF_MODEL_FILES:
        _unlink(path)
    _remove_model_backups()

    result = training_upload.clear_training_cycle()
    if result.get("error"):
        return {"purged": False, "error": result["error"]}

    _write_json(STATE_PATH, _default_state())

    return {
        "purged": True,
        "error": None,
        "message": (
            "Both ML model families, learning outputs, charts/tables, uploaded "
            "training data, merged training data and extracted features were "
            "purged. Zeek DNS Log Manager is unlocked."
        ),
    }

