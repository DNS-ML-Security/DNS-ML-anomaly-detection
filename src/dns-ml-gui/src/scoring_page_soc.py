# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

from __future__ import annotations

import html
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src import final_scoring
from src import dns_log_preparation
from src import heuristics_scoring
from src import isolation_scoring
from src import scoring
from src.db import clear_all_alerts, read_sql
from src.ui_helpers import animated_progress
from src.settings import (
    ALERT_JSONL_PATH,
    CATEGORIZATION_ALERT_JSONL_PATH,
    IF_CONFIG_PATH,
    IF_MODEL_PATH,
    IF_SCALER_PATH,
    IF_SCORING_ACTION_STATUS_PATH,
    IF_SCORING_BUFFER_PATH,
    IF_SCORING_CRON_DISABLED_PATH,
    IF_SCORING_CRON_FILE,
    IF_SCORING_CRON_LAST_PATH,
    IF_SCORING_LIVE_LOG_PATH,
    IF_SCORING_PID_PATH,
    IF_SCORING_STATE_PATH,
    IF_SCORING_STATUS_PATH,
    IF_SCORING_WINDOWS_PATH,
    MODEL_CONFIG_PATH,
    SCORING_ACTION_STATUS_PATH,
    SCORING_CRON_DISABLED_PATH,
    SCORING_CRON_FILE,
    SCORING_FEATURE_RESIDUALS_PATH,
    SCORING_LATENT_PATH,
    SCORING_LIVE_LOG_PATH,
    SCORING_LOCK_PATH,
    SCORING_PID_PATH,
    SCORING_STATUS_PATH,
    SCORING_WINDOWS_PATH,
    ZEEK_CURRENT_DIR,
)


AUTOENCODER_MODEL_PATH = Path("/data/dns-ml/models/dns_autoencoder.keras")
AUTOENCODER_SCALER_PATH = Path("/data/dns-ml/models/dns_scaler.joblib")
AUTOENCODER_STATE_PATH = Path("/data/dns-ml/state/dns_log_state.json")
AUTOENCODER_BUFFER_PATH = Path("/data/dns-ml/state/dns_event_buffer.jsonl")
HEURISTICS_SCRIPT_PATH = Path("/opt/dns-ml/score_dns_heuristics_live.py")
FINAL_MODULE_PATH = Path("/opt/dns-ml-gui/src/final_scoring.py")
FINAL_ORCHESTRATOR_PATH = Path("/opt/dns-ml-gui/scripts/run_dns_final_correlation.py")
FINAL_CRON_FILE = Path("/etc/cron.d/dns-ml-final-correlation")
FINAL_CRON_DISABLED_PATH = Path("/data/dns-ml/final_scoring/cron.disabled")
FINAL_PID_PATH = Path("/data/dns-ml/final_scoring/correlation.pid")
FINAL_LOCK_PATH = Path("/data/dns-ml/final_scoring/correlation.lock")
FINAL_LIVE_LOG_PATH = Path("/data/dns-ml/final_scoring/correlation.log")
LIVE_SCORING_STATE_PATH = SCORING_STATUS_PATH.parent / "live_scoring_state.json"
MANUAL_UPLOAD_ROOT = SCORING_STATUS_PATH.parent / "manual_uploads"
ZEEK_NODE_CONFIG_PATH = Path("/opt/zeek/etc/node.cfg")
ZEEKCTL_CANDIDATES = (Path("/opt/zeek/bin/zeekctl"), Path("/usr/local/bin/zeekctl"), Path("/usr/bin/zeekctl"))
VENV_PYTHON_PATH = Path("/opt/dns-ml/venv/bin/python")
TRAINING_FEATURE_PATH = Path("/data/dns-ml/features/dns_training_features.csv")
TRAINING_FEATURE_SUMMARY_PATH = Path("/data/dns-ml/features/dns_training_feature_summary.json")
AUTOENCODER_TRAINING_STATUS_PATH = Path("/data/dns-ml/runs/latest_training_status.json")
AUTOENCODER_TRAINING_RESULT_PATH = Path("/data/dns-ml/runs/latest_training_result.json")
AUTOENCODER_TRAINING_HISTORY_PATH = Path("/data/dns-ml/runs/latest_training_history.csv")
AUTOENCODER_RECONSTRUCTION_PATH = Path("/data/dns-ml/runs/latest_reconstruction_errors.csv")
ISOLATION_TRAINING_STATUS_PATH = Path("/data/dns-ml/runs/latest_isolation_forest_status.json")
ISOLATION_TRAINING_RESULT_PATH = Path("/data/dns-ml/runs/latest_isolation_forest_result.json")
ISOLATION_TRAINING_SCORES_PATH = Path("/data/dns-ml/runs/latest_isolation_forest_scores.csv")
ISOLATION_PCA_PATH = Path("/data/dns-ml/models/dns_isolation_forest_pca.joblib")
FINALIZED_LEARNING_STATES = {"finished", "completed", "success", "succeeded"}


def _inject_scoring_css() -> None:
    st.markdown(
        """
        <style>
        .dns-scoring-shell {
            --dns-bg-1: #0C1726;
            --dns-bg-2: #091421;
            --dns-bg-3: #08121E;
            --dns-card: #0E2A47;
            --dns-card-2: #10283D;
            --dns-border: #167FAF;
            --dns-cyan: #11D8F2;
            --dns-text: #F5F9FC;
            --dns-muted: #AFC8DC;
        }
        .dns-section-heading {
            margin: 1.25rem 0 .7rem 0;
            color: #F5F9FC;
            font-size: clamp(1.25rem, 1.9vw, 1.65rem);
            font-weight: 820;
        }
        .dns-runtime-heading-row {
            display: flex;
            align-items: center;
            gap: .72rem;
            min-height: 48px;
            margin: 1.15rem 0 .62rem 0;
        }
        .dns-runtime-heading-row .dns-section-heading {
            margin: 0;
        }
        .dns-runtime-indicator {
            --dns-radar-color: #64748B;
            --dns-radar-glow: rgba(100, 116, 139, .34);
            display: inline-flex;
            align-items: center;
            gap: .48rem;
            min-width: 0;
            color: var(--dns-radar-color);
        }
        .dns-live-radar {
            position: relative;
            flex: 0 0 auto;
            width: 40px;
            height: 40px;
            overflow: hidden;
            border: 1px solid color-mix(in srgb, var(--dns-radar-color) 82%, transparent);
            border-radius: 50%;
            background:
                linear-gradient(90deg, transparent 49%, color-mix(in srgb, var(--dns-radar-color) 26%, transparent) 50%, transparent 51%),
                linear-gradient(0deg, transparent 49%, color-mix(in srgb, var(--dns-radar-color) 26%, transparent) 50%, transparent 51%),
                repeating-radial-gradient(circle, transparent 0 7px, color-mix(in srgb, var(--dns-radar-color) 28%, transparent) 8px 9px),
                radial-gradient(circle, #0A2134 0 52%, #071521 100%);
            box-shadow:
                0 0 8px var(--dns-radar-glow),
                inset 0 0 10px color-mix(in srgb, var(--dns-radar-color) 18%, transparent);
        }
        .dns-live-radar::before {
            content: "";
            position: absolute;
            inset: 2px;
            border-radius: 50%;
            background: conic-gradient(
                from 0deg,
                transparent 0 72%,
                color-mix(in srgb, var(--dns-radar-color) 8%, transparent) 77%,
                color-mix(in srgb, var(--dns-radar-color) 72%, transparent) 100%
            );
            transform-origin: center;
        }
        .dns-live-radar::after {
            content: "";
            position: absolute;
            inset: -1px;
            border: 2px solid transparent;
            border-top-color: var(--dns-radar-color);
            border-right-color: color-mix(in srgb, var(--dns-radar-color) 34%, transparent);
            border-radius: 50%;
            filter: drop-shadow(0 0 4px var(--dns-radar-color));
        }
        .dns-live-radar-pulse {
            position: absolute;
            z-index: 2;
            top: 50%;
            left: 50%;
            width: 7px;
            height: 7px;
            border: 1px solid color-mix(in srgb, var(--dns-radar-color) 76%, white);
            border-radius: 50%;
            background: var(--dns-radar-color);
            box-shadow: 0 0 7px var(--dns-radar-color);
            transform: translate(-50%, -50%);
        }
        .dns-runtime-indicator.running {
            --dns-radar-color: #35E0A1;
            --dns-radar-glow: rgba(53, 224, 161, .46);
        }
        .dns-runtime-indicator.running .dns-live-radar::before {
            animation: dns-radar-sweep 1.8s linear infinite;
        }
        .dns-runtime-indicator.running .dns-live-radar::after {
            animation: dns-radar-ring 1.15s linear infinite;
        }
        .dns-runtime-indicator.running .dns-live-radar-pulse {
            animation: dns-radar-pulse 1.05s ease-in-out infinite;
        }
        .dns-runtime-indicator.paused {
            --dns-radar-color: #FFD166;
            --dns-radar-glow: rgba(255, 209, 102, .30);
        }
        .dns-runtime-indicator.degraded {
            --dns-radar-color: #FF6B7E;
            --dns-radar-glow: rgba(255, 107, 126, .34);
        }
        .dns-runtime-indicator.inactive {
            --dns-radar-color: #64748B;
            --dns-radar-glow: rgba(100, 116, 139, .25);
        }
        .dns-runtime-indicator-text {
            display: flex;
            flex-direction: column;
            min-width: 0;
            line-height: 1.08;
        }
        .dns-runtime-indicator-name {
            color: #AFC8DC;
            font-size: clamp(.58rem, .64vw, .68rem);
            font-weight: 760;
            letter-spacing: .05em;
            text-transform: uppercase;
        }
        .dns-runtime-indicator-state {
            margin-top: .2rem;
            color: var(--dns-radar-color);
            font-size: clamp(.72rem, .79vw, .84rem);
            font-weight: 850;
        }
        @keyframes dns-radar-ring {
            to { transform: rotate(360deg); }
        }
        @keyframes dns-radar-sweep {
            to { transform: rotate(360deg); }
        }
        @keyframes dns-radar-pulse {
            0%, 100% { opacity: .72; transform: translate(-50%, -50%) scale(.82); }
            50% { opacity: 1; transform: translate(-50%, -50%) scale(1.35); box-shadow: 0 0 12px var(--dns-radar-color); }
        }
        @media (prefers-reduced-motion: reduce) {
            .dns-runtime-indicator.running .dns-live-radar::before,
            .dns-runtime-indicator.running .dns-live-radar::after,
            .dns-runtime-indicator.running .dns-live-radar-pulse {
                animation: none;
            }
        }
        .dns-card-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            align-items: stretch;
            gap: .78rem;
            margin: .35rem 0 1rem 0;
        }
        .dns-card-grid.five {
            grid-template-columns: repeat(5, minmax(0, 1fr));
        }
        .dns-soc-card {
            min-width: 0;
            min-height: 0;
            height: 100%;
            padding: .68rem .78rem;
            border: 1px solid #167FAF;
            border-radius: 14px;
            background: linear-gradient(145deg, #10283D 0%, #0E2A47 100%);
            box-shadow: 0 8px 20px rgba(0, 0, 0, .20);
        }
        .dns-soc-label {
            color: #AFC8DC;
            font-size: clamp(.72rem, .9vw, .84rem);
            font-weight: 760;
            letter-spacing: .035em;
            text-transform: uppercase;
        }
        .dns-soc-value {
            margin-top: .42rem;
            color: #F5F9FC;
            font-size: clamp(1.05rem, 1.6vw, 1.42rem);
            font-weight: 840;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }
        .dns-soc-detail {
            margin-top: .38rem;
            color: #AFC8DC;
            font-size: clamp(.72rem, .86vw, .82rem);
            line-height: 1.38;
            overflow-wrap: anywhere;
        }
        .dns-state-success { color: #53E6B2; }
        .dns-state-warning { color: #FFD166; }
        .dns-state-error { color: #FF7A90; }
        .dns-state-info { color: #36D8FF; }
        .dns-progress-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(150px, 1fr));
            align-items: stretch;
            gap: clamp(.65rem, 1.3vw, 1.1rem);
            margin: .45rem 0 1rem 0;
        }
        .dns-progress-item {
            min-width: 0;
            height: 100%;
            padding: 1rem .7rem .85rem;
            border: 0;
            background: transparent;
            text-align: center;
        }
        .dns-progress-ring {
            --dns-progress: 0;
            --dns-ring: #11D8F2;
            position: relative;
            display: grid;
            place-items: center;
            width: clamp(106px, 8vw, 132px);
            aspect-ratio: 1;
            margin: 0 auto .8rem;
            border-radius: 50%;
            background:
                radial-gradient(circle, #091827 57%, transparent 59%),
                conic-gradient(var(--dns-ring) calc(var(--dns-progress) * 1%), #12334A 0);
            box-shadow:
                0 0 8px color-mix(in srgb, var(--dns-ring) 78%, transparent),
                0 0 24px color-mix(in srgb, var(--dns-ring) 42%, transparent),
                inset 0 0 14px rgba(255,255,255,.08);
        }
        .dns-progress-ring::after {
            content: "";
            position: absolute;
            inset: -5px;
            border: 1px solid color-mix(in srgb, var(--dns-ring) 48%, transparent);
            border-radius: 50%;
            opacity: .72;
        }
        .dns-progress-ring.active {
            animation: dns-ring-pulse 1.35s ease-in-out infinite alternate;
        }
        .dns-progress-value {
            position: relative;
            z-index: 1;
            color: #F5F9FC;
            font-size: clamp(1.35rem, 2vw, 1.85rem);
            font-weight: 850;
            letter-spacing: -.035em;
        }
        .dns-progress-title {
            color: #F5F9FC;
            font-size: clamp(.88rem, 1.05vw, 1rem);
            font-weight: 820;
            line-height: 1.25;
        }
        .dns-progress-state {
            margin-top: .34rem;
            font-size: clamp(.7rem, .78vw, .84rem);
            font-weight: 760;
            line-height: 1.28;
        }
        .dns-progress-detail {
            margin-top: .32rem;
            color: #AFC8DC;
            font-size: clamp(.64rem, .7vw, .75rem);
            line-height: 1.34;
            overflow-wrap: anywhere;
        }
        @keyframes dns-ring-pulse {
            from { filter: brightness(.88); transform: scale(.985); }
            to { filter: brightness(1.18); transform: scale(1.015); }
        }
        .dns-input-panel {
            padding: 1rem;
            margin: .4rem 0 1rem 0;
            border: 1px solid #168FC4;
            border-radius: 14px;
            background: linear-gradient(145deg, rgba(14,42,71,.96), rgba(9,20,33,.98));
        }
        .dns-path-caption {
            color: #AFC8DC;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: clamp(.64rem, .72vw, .78rem);
            overflow-wrap: anywhere;
        }
        .dns-alert-empty {
            padding: 1.2rem;
            border: 1px dashed #2B648F;
            border-radius: 13px;
            color: #AFC8DC;
            text-align: center;
            background: #091421;
        }
        .st-key-dns_live_operation_metrics [data-testid="stMetric"],
        .st-key-dns_final_alert_metrics [data-testid="stMetric"] {
            min-width: 0;
            min-height: 0 !important;
            height: 100% !important;
            padding: .58rem .68rem;
            border: 1px solid #167FAF;
            border-radius: 12px;
            background: linear-gradient(145deg, #10283D 0%, #0E2A47 100%);
            overflow: visible;
        }
        .st-key-dns_live_operation_metrics [data-testid="stMetricLabel"] p,
        .st-key-dns_final_alert_metrics [data-testid="stMetricLabel"] p {
            font-size: clamp(.66rem, .78vw, .8rem) !important;
            line-height: 1.22 !important;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
        }
        .st-key-dns_live_operation_metrics [data-testid="stMetricValue"],
        .st-key-dns_final_alert_metrics [data-testid="stMetricValue"] {
            font-size: clamp(.9rem, 1.18vw, 1.22rem) !important;
            line-height: 1.18 !important;
            white-space: normal !important;
            overflow-wrap: anywhere !important;
        }
        .dns-ml-info-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            align-items: stretch;
            gap: .78rem;
            margin: .45rem 0 1.1rem 0;
        }
        .dns-ml-info-card {
            min-width: 0;
            min-height: 0;
            height: 100%;
            padding: .68rem .75rem;
            border: 1px solid #167FAF;
            border-radius: 14px;
            background: linear-gradient(145deg, #10283D 0%, #0E2A47 100%);
            box-shadow: 0 8px 20px rgba(0, 0, 0, .20);
            text-align: center;
        }
        .dns-ml-info-number {
            color: #7FB8D7;
            font-size: clamp(.58rem, .64vw, .68rem);
            font-weight: 800;
            letter-spacing: .06em;
            text-transform: uppercase;
        }
        .dns-ml-info-title {
            margin-top: .28rem;
            color: #F5F9FC;
            font-size: clamp(.72rem, .84vw, .9rem);
            font-weight: 820;
            line-height: 1.28;
        }
        .dns-ml-info-ring {
            --dns-progress: 0;
            --dns-ring: #11D8F2;
            display: grid;
            place-items: center;
            width: 76px;
            aspect-ratio: 1;
            margin: .55rem auto;
            border-radius: 50%;
            background:
                radial-gradient(circle, #091827 56%, transparent 59%),
                conic-gradient(var(--dns-ring) calc(var(--dns-progress) * 1%), #12334A 0);
            box-shadow: 0 0 14px color-mix(in srgb, var(--dns-ring) 38%, transparent);
        }
        .dns-ml-info-ring span {
            color: #F5F9FC;
            font-size: clamp(.82rem, .94vw, 1rem);
            font-weight: 850;
        }
        .dns-ml-info-headline {
            color: #F5F9FC;
            font-size: clamp(.69rem, .78vw, .83rem);
            font-weight: 800;
            line-height: 1.3;
            overflow-wrap: anywhere;
        }
        .dns-ml-info-detail {
            margin-top: .3rem;
            color: #AFC8DC;
            font-size: clamp(.6rem, .67vw, .72rem);
            line-height: 1.34;
            overflow-wrap: anywhere;
        }
        @media (max-width: 1180px) {
            .dns-card-grid, .dns-card-grid.five {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .dns-progress-grid { grid-template-columns: repeat(3, minmax(150px, 1fr)); }
            .dns-ml-info-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 680px) {
            .dns-runtime-heading-row {
                align-items: flex-start;
                flex-wrap: wrap;
            }
            .dns-card-grid, .dns-card-grid.five {
                grid-template-columns: 1fr;
            }
            .dns-progress-grid { grid-template-columns: 1fr; }
            .dns-ml-info-grid { grid-template-columns: 1fr; }
            .dns-soc-card { min-height: 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _escape(value: Any) -> str:
    return html.escape(str(value if value not in (None, "") else "Not available"))


def _state_class(state: str) -> str:
    normalized = str(state or "").strip().lower()
    if normalized in {"ready", "pass", "running", "active", "scheduled", "finished", "success", "completed"}:
        return "dns-state-success"
    if normalized in {"failed", "error", "missing", "locked"} or "failed" in normalized:
        return "dns-state-error"
    if normalized in {"paused", "partial", "waiting", "no_data", "warning"} or any(
        token in normalized for token in ("waiting", "paused", "input gap", "partial")
    ):
        return "dns-state-warning"
    return "dns-state-info"


def _render_cards(cards: list[dict[str, Any]], *, five: bool = False) -> None:
    css_class = "dns-card-grid five" if five else "dns-card-grid"
    fragments = [f'<div class="{css_class}">']
    for card in cards:
        state = str(card.get("state") or card.get("value") or "")
        fragments.append(
            '<div class="dns-soc-card">'
            f'<div class="dns-soc-label">{_escape(card.get("label"))}</div>'
            f'<div class="dns-soc-value {_state_class(state)}">{_escape(card.get("value"))}</div>'
            f'<div class="dns-soc-detail">{_escape(card.get("detail"))}</div>'
            '</div>'
        )
    fragments.append("</div>")
    st.markdown("".join(fragments), unsafe_allow_html=True)


def _ring_color(state: str) -> str:
    normalized = str(state or "").strip().lower()
    if normalized in {"failed", "error"} or "failed" in normalized:
        return "#FF5F78"
    if normalized in {"paused", "warning"} or "waiting" in normalized:
        return "#FFD166"
    if normalized in {"finished", "completed", "calculated", "active", "success"}:
        return "#35E0A1"
    return "#11D8F2"


def _render_progress_rings(items: list[dict[str, Any]]) -> None:
    fragments = ['<div class="dns-progress-grid">']
    for item in items:
        try:
            progress = max(0, min(100, int(item.get("progress", 0) or 0)))
        except Exception:
            progress = 0
        state = str(item.get("state") or "Not activated")
        active_class = " active" if bool(item.get("active")) else ""
        ring_color = _ring_color(state)
        fragments.append(
            '<div class="dns-progress-item">'
            f'<div class="dns-progress-ring{active_class}" '
            f'style="--dns-progress:{progress};--dns-ring:{ring_color}">'
            f'<div class="dns-progress-value">{progress}%</div>'
            '</div>'
            f'<div class="dns-progress-title">{_escape(item.get("title"))}</div>'
            f'<div class="dns-progress-state {_state_class(state)}">{_escape(state)}</div>'
            f'<div class="dns-progress-detail">{_escape(item.get("detail"))}</div>'
            '</div>'
        )
    fragments.append("</div>")
    st.markdown("".join(fragments), unsafe_allow_html=True)


def _pid_file_running(path: Path, expected_name: str | None = None) -> bool:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        stat_fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        if len(stat_fields) > 2 and stat_fields[2] == "Z":
            return False
        if expected_name:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace"
            )
            return expected_name in command
        return Path(f"/proc/{pid}").exists()
    except Exception:
        return False


def _read_json(path: Path, default: Any) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value
    except Exception:
        return default


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_lifecycle_state() -> dict[str, Any]:
    value = _read_json(LIVE_SCORING_STATE_PATH, {})
    return value if isinstance(value, dict) else {}


def _write_lifecycle_state(**updates: Any) -> dict[str, Any]:
    state = _read_lifecycle_state()
    state.update(updates)
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(LIVE_SCORING_STATE_PATH, state)
    return state


def _run_command(command: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = "\n".join(
            part.strip() for part in (result.stdout, result.stderr) if part.strip()
        )
        return {
            "command": " ".join(command),
            "return_code": int(result.returncode),
            "output": output,
            "error": None if result.returncode == 0 else (output or f"Command returned {result.returncode}"),
        }
    except Exception as exc:
        return {
            "command": " ".join(command),
            "return_code": None,
            "output": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _notify_progress(
    callback: Callable[[int, str], None] | None,
    percent: int,
    message: str,
) -> None:
    if callback is not None:
        callback(max(0, min(100, int(percent))), str(message))


def _zeekctl_path() -> Path | None:
    for candidate in ZEEKCTL_CANDIDATES:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    discovered = shutil.which("zeekctl")
    return Path(discovered) if discovered else None


def _configured_zeek_interfaces() -> list[str]:
    try:
        source = ZEEK_NODE_CONFIG_PATH.read_text(encoding="utf-8")
    except Exception:
        return []
    return re.findall(r"(?m)^\s*interface\s*=\s*([^\s#]+)", source)


def _interface_inventory() -> tuple[list[dict[str, Any]], str | None]:
    ip_binary = shutil.which("ip")
    if not ip_binary:
        return [], "The ip utility is unavailable on this server."

    link_result = _run_command([ip_binary, "-j", "link", "show"])
    address_result = _run_command([ip_binary, "-j", "address", "show"])
    route_result = _run_command([ip_binary, "-j", "route", "show", "default"])
    failed = [item for item in (link_result, address_result, route_result) if item.get("error")]
    if failed:
        return [], str(failed[0].get("error"))

    try:
        links = json.loads(link_result.get("output") or "[]")
        addresses = json.loads(address_result.get("output") or "[]")
        routes = json.loads(route_result.get("output") or "[]")
    except Exception as exc:
        return [], f"Unable to parse network inventory: {type(exc).__name__}: {exc}"

    address_map = {str(item.get("ifname")): item for item in addresses if item.get("ifname")}
    default_interfaces = {
        str(item.get("dev")) for item in routes if item.get("dev")
    }
    configured = set(_configured_zeek_interfaces())
    rows: list[dict[str, Any]] = []
    for link in links:
        name = str(link.get("ifname") or "").strip()
        if not name:
            continue
        address_item = address_map.get(name, {})
        link_state = str(link.get("operstate") or address_item.get("operstate") or "UNKNOWN").upper()
        address_values: list[str] = []
        for item in address_item.get("addr_info") or []:
            local = str(item.get("local") or "").strip()
            family = str(item.get("family") or "")
            if not local or family not in {"inet", "inet6"}:
                continue
            address_values.append(f"{local}/{item.get('prefixlen', '')}".rstrip("/"))

        reasons: list[str] = []
        if name == "lo" or "LOOPBACK" in {str(flag).upper() for flag in link.get("flags") or []}:
            reasons.append("Loopback interface")
        if name in default_interfaces:
            reasons.append("Management/default-route interface")

        rows.append(
            {
                "Interface": name,
                "Link state": link_state,
                "IP address(es)": ", ".join(address_values) if address_values else "No IP address",
                "MAC": str(link.get("address") or "Not available"),
                "Default route": "YES" if name in default_interfaces else "NO",
                "Zeek configured": "YES" if name in configured else "NO",
                "Eligibility": "ELIGIBLE" if not reasons else "BLOCKED",
                "Explanation": (
                    "Dedicated capture candidate; link state and IP assignment do not affect eligibility"
                    if not reasons
                    else "; ".join(reasons)
                ),
            }
        )
    return rows, None


def _zeek_runtime_status() -> dict[str, Any]:
    zeekctl = _zeekctl_path()
    configured = _configured_zeek_interfaces()
    if zeekctl is None:
        return {
            "available": False,
            "running": False,
            "configured_interfaces": configured,
            "output": "zeekctl was not found",
        }
    result = _run_command([str(zeekctl), "status"], timeout=20)
    output = str(result.get("output") or "")
    return {
        "available": True,
        "running": result.get("return_code") == 0 and bool(re.search(r"\brunning\b", output, re.IGNORECASE)),
        "configured_interfaces": configured,
        "output": output or "No status output",
        "error": result.get("error"),
    }


def _restore_zeek_config(backup: Path, zeekctl: Path) -> None:
    if backup.is_file():
        shutil.copy2(backup, ZEEK_NODE_CONFIG_PATH)
        _run_command([str(zeekctl), "deploy"], timeout=90)


def _configure_and_start_zeek(
    interface: str,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    _notify_progress(progress_callback, 6, "Validating the selected capture interface")
    inventory, inventory_error = _interface_inventory()
    if inventory_error:
        return {"error": inventory_error}
    selected = next((row for row in inventory if row["Interface"] == interface), None)
    if selected is None:
        return {"error": f"Interface {interface!r} is no longer present."}
    if selected["Eligibility"] != "ELIGIBLE":
        return {"error": f"Interface {interface!r} is blocked: {selected['Explanation']}"}

    _notify_progress(progress_callback, 14, "Locating Zeek control utilities")
    zeekctl = _zeekctl_path()
    if zeekctl is None:
        return {"error": "zeekctl was not found; Zeek live capture cannot be managed."}
    if not ZEEK_NODE_CONFIG_PATH.is_file():
        return {"error": f"Zeek node configuration is missing: {ZEEK_NODE_CONFIG_PATH}"}

    try:
        _notify_progress(progress_callback, 22, "Preparing the Zeek interface configuration")
        source = ZEEK_NODE_CONFIG_PATH.read_text(encoding="utf-8")
        interface_matches = re.findall(r"(?m)^\s*interface\s*=\s*[^\s#]+", source)
        if len(interface_matches) != 1:
            return {
                "error": (
                    f"Zeek interface target count is {len(interface_matches)}; expected exactly one in "
                    f"{ZEEK_NODE_CONFIG_PATH}."
                )
            }
        patched = re.sub(
            r"(?m)^(\s*interface\s*=\s*)[^\s#]+",
            lambda match: match.group(1) + interface,
            source,
            count=1,
        )
        backup_dir = Path("/data/dns-ml/backups/dns-scoring-zeek-nodecfg")
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / f"node.cfg.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.bak"
        shutil.copy2(ZEEK_NODE_CONFIG_PATH, backup)
        stat = ZEEK_NODE_CONFIG_PATH.stat()
        temporary = ZEEK_NODE_CONFIG_PATH.with_suffix(".cfg.dns-scoring.tmp")
        temporary.write_text(patched, encoding="utf-8")
        os.chmod(temporary, stat.st_mode)
        os.chown(temporary, stat.st_uid, stat.st_gid)
        temporary.replace(ZEEK_NODE_CONFIG_PATH)

        _notify_progress(progress_callback, 38, "Checking the Zeek configuration")
        check = _run_command([str(zeekctl), "check"], timeout=60)
        if check.get("error"):
            _restore_zeek_config(backup, zeekctl)
            return {"error": f"Zeek configuration check failed: {check['error']}", "backup": str(backup)}
        _notify_progress(progress_callback, 52, "Starting Zeek live capture")
        deploy = _run_command([str(zeekctl), "deploy"], timeout=120)
        if deploy.get("error"):
            _restore_zeek_config(backup, zeekctl)
            return {"error": f"Zeek deploy failed: {deploy['error']}", "backup": str(backup)}
        _notify_progress(progress_callback, 68, "Zeek live capture is running")
        return {
            "error": None,
            "interface": interface,
            "node_config": str(ZEEK_NODE_CONFIG_PATH),
            "backup": str(backup),
            "check": check,
            "deploy": deploy,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _start_zeek_current_configuration(
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    _notify_progress(progress_callback, 8, "Locating Zeek control utilities")
    zeekctl = _zeekctl_path()
    if zeekctl is None:
        return {"error": "zeekctl was not found."}
    _notify_progress(progress_callback, 30, "Resuming Zeek live capture")
    result = _run_command([str(zeekctl), "deploy"], timeout=120)
    if not result.get("error"):
        _notify_progress(progress_callback, 62, "Zeek live capture resumed")
    return {**result, "error": result.get("error")}


def _stop_zeek_capture(
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    _notify_progress(progress_callback, 55, "Stopping Zeek live capture")
    zeekctl = _zeekctl_path()
    if zeekctl is None:
        return {"error": "zeekctl was not found."}
    result = _run_command([str(zeekctl), "stop"], timeout=90)
    output = str(result.get("output") or "")
    if result.get("error") and "not running" in output.lower():
        result["error"] = None
    if not result.get("error"):
        _notify_progress(progress_callback, 82, "Zeek live capture stopped")
    return result


def _mtime_text(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except Exception:
        return "Not available"


def _timestamp_from_status(status: dict[str, Any], fallback: Path | None = None) -> str:
    for key in (
        "completed_at_utc",
        "generated_at_utc",
        "updated_at_utc",
        "started_at_utc",
        "timestamp",
    ):
        if status.get(key):
            return str(status[key])
    return _mtime_text(fallback) if fallback is not None else "Not available"


def _nonempty(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def _learning_phase_row(label: str, status_path: Path) -> tuple[dict[str, str], bool]:
    status = _read_json(status_path, {})
    if not isinstance(status, dict):
        status = {}
    state = str(status.get("state") or "missing").strip().lower()
    progress_value = status.get("progress_percent")
    try:
        progress = int(progress_value) if progress_value is not None else None
    except Exception:
        progress = None
    finalized = _nonempty(status_path) and state in FINALIZED_LEARNING_STATES
    detail = f"{status_path} · state={state}"
    if progress is not None:
        detail += f" · progress={progress}%"
    return (
        {
            "Requirement": f"{label} learning phase finalized",
            "Status": "PASS" if finalized else "NOT FINALIZED",
            "Authoritative path or evidence source": detail,
        },
        finalized,
    )


def _model_prerequisites() -> tuple[list[dict[str, str]], bool]:
    checks = [
        ("Shared 36-feature training dataset", TRAINING_FEATURE_PATH),
        ("Shared training feature summary", TRAINING_FEATURE_SUMMARY_PATH),
        ("Autoencoder training status", AUTOENCODER_TRAINING_STATUS_PATH),
        ("Autoencoder training result", AUTOENCODER_TRAINING_RESULT_PATH),
        ("Autoencoder training history", AUTOENCODER_TRAINING_HISTORY_PATH),
        ("Autoencoder reconstruction evidence", AUTOENCODER_RECONSTRUCTION_PATH),
        ("Autoencoder model", AUTOENCODER_MODEL_PATH),
        ("Autoencoder scaler", AUTOENCODER_SCALER_PATH),
        ("Autoencoder learned configuration", MODEL_CONFIG_PATH),
        ("Isolation Forest training status", ISOLATION_TRAINING_STATUS_PATH),
        ("Isolation Forest training result", ISOLATION_TRAINING_RESULT_PATH),
        ("Isolation Forest training scores", ISOLATION_TRAINING_SCORES_PATH),
        ("Isolation Forest model", IF_MODEL_PATH),
        ("Isolation Forest scaler", IF_SCALER_PATH),
        ("Isolation Forest learned configuration", IF_CONFIG_PATH),
        ("Isolation Forest PCA artifact", ISOLATION_PCA_PATH),
        ("Heuristic scoring engine", HEURISTICS_SCRIPT_PATH),
        ("Final-correlation engine", FINAL_MODULE_PATH),
        ("Automatic final-correlation coordinator", FINAL_ORCHESTRATOR_PATH),
    ]
    rows: list[dict[str, str]] = []
    ready = True
    for requirement, path in checks:
        passed = _nonempty(path)
        if not passed:
            ready = False
        rows.append(
            {
                "Requirement": requirement,
                "Status": "PASS" if passed else "MISSING/INCOMPLETE",
                "Authoritative path or evidence source": str(path),
            }
        )
    for label, status_path in (
        ("Autoencoder", AUTOENCODER_TRAINING_STATUS_PATH),
        ("Isolation Forest", ISOLATION_TRAINING_STATUS_PATH),
    ):
        row, passed = _learning_phase_row(label, status_path)
        rows.append(row)
        if not passed:
            ready = False
    return rows, ready


def _scoring_gate_error() -> str | None:
    rows, ready = _model_prerequisites()
    if ready:
        return None
    failed = [row["Requirement"] for row in rows if row["Status"] != "PASS"]
    return (
        "Scoring is locked until both learning phases are finalized and every required training/model "
        "artifact exists. Missing or incomplete: " + ", ".join(failed)
    )


def _worker_snapshots() -> dict[str, dict[str, Any]]:
    auto_status = scoring.read_scoring_status()
    isolation_status = isolation_scoring.read_status()
    heuristic_status = heuristics_scoring.read_status()
    final_status = final_scoring.read_final_status()

    return {
        "autoencoder": {
            "name": "Autoencoder",
            "running": bool(scoring.is_scoring_running()),
            "scheduled": SCORING_CRON_FILE.exists(),
            "paused_copy": SCORING_CRON_DISABLED_PATH.exists(),
            "state": str(auto_status.get("state") or "not_started"),
            "stage": str(auto_status.get("stage") or "Waiting"),
            "progress": int(auto_status.get("progress_percent", 0) or 0),
            "latest": _timestamp_from_status(auto_status, SCORING_WINDOWS_PATH),
            "status": auto_status,
            "cron": str(SCORING_CRON_FILE),
            "input": str(ZEEK_CURRENT_DIR),
            "output": str(ALERT_JSONL_PATH),
        },
        "isolation": {
            "name": "Isolation Forest",
            "running": bool(isolation_scoring.is_running()),
            "scheduled": IF_SCORING_CRON_FILE.exists(),
            "paused_copy": IF_SCORING_CRON_DISABLED_PATH.exists(),
            "state": str(isolation_status.get("state") or "not_started"),
            "stage": str(isolation_status.get("stage") or "Waiting"),
            "progress": int(isolation_status.get("progress_percent", 0) or 0),
            "latest": _timestamp_from_status(isolation_status, IF_SCORING_WINDOWS_PATH),
            "status": isolation_status,
            "cron": str(IF_SCORING_CRON_FILE),
            "input": str(ZEEK_CURRENT_DIR),
            "output": str(CATEGORIZATION_ALERT_JSONL_PATH),
        },
        "heuristics": {
            "name": "DNS Heuristics",
            "running": bool(heuristics_scoring.is_running()),
            "scheduled": heuristics_scoring.HEUR_CRON_FILE.exists(),
            "paused_copy": heuristics_scoring.HEUR_CRON_DISABLED_PATH.exists(),
            "state": str(heuristic_status.get("state") or "not_started"),
            "stage": str(heuristic_status.get("stage") or "Waiting"),
            "progress": int(heuristic_status.get("progress_percent", 0) or 0),
            "latest": _timestamp_from_status(heuristic_status, heuristics_scoring.HEUR_ALERT_PATH),
            "status": heuristic_status,
            "cron": str(heuristics_scoring.HEUR_CRON_FILE),
            "input": str(ZEEK_CURRENT_DIR),
            "output": str(heuristics_scoring.HEUR_ALERT_PATH),
        },
        "final": {
            "name": "Final Correlation",
            "running": _pid_file_running(FINAL_PID_PATH, FINAL_ORCHESTRATOR_PATH.name),
            "scheduled": FINAL_CRON_FILE.exists(),
            "paused_copy": FINAL_CRON_DISABLED_PATH.exists(),
            "state": str(final_status.get("state") or "not_started"),
            "stage": str(final_status.get("stage") or "Waiting for detector alerts"),
            "progress": int(final_status.get("progress_percent", 100 if final_status.get("state") == "finished" else 0) or 0),
            "latest": _timestamp_from_status(final_status, final_scoring.FINAL_ALERTS_PATH),
            "status": final_status,
            "cron": str(FINAL_CRON_FILE),
            "input": (
                f"{CATEGORIZATION_ALERT_JSONL_PATH}; {ALERT_JSONL_PATH}; "
                f"{heuristics_scoring.HEUR_ALERT_PATH}"
            ),
            "output": str(final_scoring.FINAL_ALERTS_PATH),
        },
    }


def _worker_waiting_for_input(worker: dict[str, Any]) -> bool:
    state = str(worker.get("state") or "").strip().lower()
    if state in {"waiting_input", "waiting", "input_pending"}:
        return True
    if not worker.get("scheduled"):
        return False
    status = worker.get("status") or {}
    evidence = " ".join(
        str(value or "")
        for value in (status.get("message"), status.get("error"), worker.get("stage"))
    ).lower()
    return "no supported live zeek dns file" in evidence


def _display_worker_state(worker: dict[str, Any]) -> str:
    if worker.get("running"):
        return "Running"
    state = str(worker.get("state") or "").lower()
    if worker.get("name") == "Final Correlation":
        if state in {"failed", "error"}:
            return "Failed"
        if state in {"finished", "completed", "success"}:
            return "Calculated"
        if worker.get("paused_copy"):
            return "Paused"
        if worker.get("scheduled"):
            return "Waiting For Detectors"
        return "Not Calculated"
    if _worker_waiting_for_input(worker):
        return "Waiting For Input"
    if state in {"failed", "error"}:
        return "Failed"
    if state in {"finished", "completed", "success", "succeeded", "no_data"}:
        return "Completed"
    if worker.get("scheduled"):
        return "Scheduled"
    if worker.get("paused_copy"):
        return "Paused"
    return "Inactive"


def _runtime_summary(workers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    detector_workers = [workers[name] for name in ("autoencoder", "isolation", "heuristics")]
    running_count = sum(bool(item["running"]) for item in detector_workers)
    final_running = bool(workers["final"]["running"])
    pipeline_running_count = running_count + int(final_running)
    scheduled_count = sum(bool(item["scheduled"]) for item in detector_workers)
    paused_count = sum(bool(item["paused_copy"]) for item in detector_workers)
    failed_count = sum(
        str(item.get("state") or "").lower() in {"failed", "error"}
        and not _worker_waiting_for_input(item)
        for item in detector_workers
    )
    busy = bool(st.session_state.get("dns_soc_admin_busy"))
    lifecycle = _read_lifecycle_state()
    lifecycle_mode = str(lifecycle.get("mode") or "")
    lifecycle_state = str(lifecycle.get("state") or "")

    if busy:
        runtime = "Administrative operation"
    elif lifecycle_mode == "manual" and pipeline_running_count:
        runtime = "Manual one-shot processing"
    elif lifecycle_mode == "manual" and lifecycle_state in {"processing", "completed"}:
        runtime = "Manual one-shot complete"
    elif pipeline_running_count:
        runtime = "Processing"
    elif failed_count:
        runtime = "Failed"
    elif scheduled_count == 3:
        runtime = "Active"
    elif scheduled_count:
        runtime = "Partial Schedule"
    elif paused_count:
        runtime = "Paused"
    elif lifecycle_state == "paused":
        runtime = "Paused"
    else:
        runtime = "Inactive"

    activation_times = [
        _mtime_text(path)
        for path in (
            SCORING_CRON_FILE,
            IF_SCORING_CRON_FILE,
            heuristics_scoring.HEUR_CRON_FILE,
        )
        if path.exists()
    ]
    activated = str(lifecycle.get("activated_at_utc") or "")
    if not activated:
        activated = max(activation_times) if activation_times else "Not available"
    return {
        "runtime": runtime,
        "running_count": running_count,
        "final_running": final_running,
        "pipeline_running_count": pipeline_running_count,
        "scheduled_count": scheduled_count,
        "paused_count": paused_count,
        "failed_count": failed_count,
        "activated": activated,
        "busy": busy,
        "lifecycle": lifecycle,
        "lifecycle_mode": lifecycle_mode,
        "lifecycle_state": lifecycle_state,
    }


def _live_radar_state() -> dict[str, str]:
    """Return the operator-facing state for the compact live-scoring radar."""
    workers = _worker_snapshots()
    runtime = _runtime_summary(workers)
    lifecycle = runtime["lifecycle"]
    lifecycle_mode = runtime["lifecycle_mode"]
    lifecycle_state = runtime["lifecycle_state"]
    capture_enabled = bool(lifecycle.get("capture_enabled"))
    zeek_status = _zeek_runtime_status()

    ml_workers = [workers["autoencoder"], workers["isolation"]]
    unhealthy_ml = [
        worker["name"]
        for worker in ml_workers
        if (
            not (worker.get("scheduled") or worker.get("running"))
            or (
                str(worker.get("state") or "").strip().lower() in {"failed", "error"}
                and not _worker_waiting_for_input(worker)
            )
        )
    ]

    paused = lifecycle_state == "paused" or (
        capture_enabled
        and runtime["scheduled_count"] == 0
        and any(worker.get("paused_copy") for worker in ml_workers)
    )
    live_requested = (
        lifecycle_mode != "manual"
        and (
            capture_enabled
            or lifecycle_state in {"active", "failed"}
            or runtime["scheduled_count"] > 0
            or runtime["pipeline_running_count"] > 0
        )
    )

    if paused:
        return {
            "state": "paused",
            "label": "Paused",
            "detail": "Live Zeek capture and scoring are paused; animation is stopped.",
        }
    if not live_requested:
        return {
            "state": "inactive",
            "label": "Inactive",
            "detail": "Managed Live Zeek scoring is not active.",
        }

    degraded_reasons: list[str] = []
    if not capture_enabled:
        degraded_reasons.append("capture is not enabled")
    if not zeek_status.get("running"):
        degraded_reasons.append("Zeek capture is not running")
    if unhealthy_ml:
        degraded_reasons.append(f"unhealthy ML worker(s): {', '.join(unhealthy_ml)}")
    if lifecycle_state == "failed":
        degraded_reasons.append("the managed lifecycle reports failure")

    if degraded_reasons:
        return {
            "state": "degraded",
            "label": "Degraded",
            "detail": "Live scoring needs attention: " + "; ".join(degraded_reasons) + ".",
        }
    return {
        "state": "running",
        "label": "Running",
        "detail": (
            "Zeek capture, Autoencoder, and Isolation Forest are healthy. "
            f"DNS Heuristics: {_display_worker_state(workers['heuristics'])}."
        ),
    }


def _live_input() -> tuple[Path | None, str]:
    try:
        path = scoring.find_live_zeek_dns_log()
    except Exception:
        path = None
    if path is None:
        return None, "No supported dns.log, dns.jsonl, or dns.json file was found"
    try:
        stat = path.stat()
        detail = f"{stat.st_size:,} bytes · modified {_mtime_text(path)}"
    except Exception:
        detail = "File exists; metadata unavailable"
    return path, detail


def _render_top_status(ready: bool) -> None:
    workers = _worker_snapshots()
    runtime = _runtime_summary(workers)
    input_path, input_detail = _live_input()
    manual_input = str(runtime["lifecycle"].get("manual_input_path") or "")
    if runtime["lifecycle_mode"] == "manual" and manual_input:
        input_path = Path(manual_input)
        input_detail = "Staged manual DNS JSON/JSONL input"
    runtime_value = runtime["runtime"]
    if runtime["scheduled_count"] and input_path is None:
        runtime_value = "Scheduled · Input Gap"
    _render_cards(
        [
            {
                "label": "Runtime",
                "value": runtime_value,
                "detail": (
                    f"{runtime['pipeline_running_count']} pipeline stages processing · "
                    f"{runtime['scheduled_count']}/3 scheduled"
                ),
            },
            {
                "label": "Learning gate",
                "value": "Ready" if ready else "Locked",
                "detail": (
                    "Training data, finalized phases, and all artifacts are complete"
                    if ready
                    else "Training data, phase finalization, or required artifacts are incomplete"
                ),
            },
            {
                "label": "Activated",
                "value": runtime["activated"],
                "detail": "Most recent detector schedule activation",
            },
            {
                "label": "Input source",
                "value": input_path.name if input_path else "Not available",
                "detail": input_detail,
            },
        ]
    )
    st.caption(f"Authoritative live input directory: `{ZEEK_CURRENT_DIR}`")


def _render_health_cards() -> None:
    workers = _worker_snapshots()
    input_path, input_detail = _live_input()
    lifecycle = _read_lifecycle_state()
    zeek_status = _zeek_runtime_status()
    lifecycle_mode = str(lifecycle.get("mode") or "")
    lifecycle_state = str(lifecycle.get("state") or "")
    capture_enabled = bool(lifecycle.get("capture_enabled"))

    if lifecycle_mode == "manual":
        capture_value = "Manual Input"
        capture_detail = str(lifecycle.get("manual_input_path") or "Manual DNS upload staged")
        capture_progress = 100
        capture_active = lifecycle_state == "processing"
    elif capture_enabled and lifecycle_state == "paused":
        capture_value = "Paused"
        capture_detail = f"Interface: {lifecycle.get('capture_interface') or 'Not available'}"
        capture_progress = 100
        capture_active = False
    elif capture_enabled and zeek_status.get("running"):
        capture_value = "Active"
        capture_detail = f"Interface: {lifecycle.get('capture_interface') or 'Not available'} · Output: {ZEEK_CURRENT_DIR}"
        capture_progress = 100
        capture_active = True
    elif capture_enabled:
        capture_value = "Failed" if zeek_status.get("error") else "Inactive"
        capture_detail = str(zeek_status.get("error") or zeek_status.get("output") or "Zeek is not running")
        capture_progress = 0
        capture_active = False
    else:
        capture_value = "Not Selected"
        external_state = "running externally" if zeek_status.get("running") else "not lifecycle-managed"
        capture_detail = f"Zeek is {external_state} · Current DNS input: {input_path or 'Not available'}"
        capture_progress = 0
        capture_active = False

    ring_items: list[dict[str, Any]] = [
        {
            "title": "Zeek data capture",
            "state": capture_value,
            "progress": capture_progress,
            "active": capture_active,
            "detail": capture_detail,
        }
    ]
    for worker_key, title in (
        ("autoencoder", "Autoencoder"),
        ("isolation", "Isolation Forest"),
        ("heuristics", "DNS Heuristics"),
        ("final", "Final Correlation"),
    ):
        worker = workers[worker_key]
        status = worker.get("status") or {}
        display_state = _display_worker_state(worker)
        progress = max(0, min(100, int(worker.get("progress", 0) or 0)))
        terminal = str(worker.get("state") or "").lower() in {
            "finished", "completed", "success", "succeeded", "no_data"
        }
        if terminal:
            progress = 100
        if worker_key == "final":
            final_alerts = int(status.get("final_alerts", 0) or 0)
            correlated = int(status.get("correlated_windows", 0) or 0)
            if terminal:
                detail = f"{final_alerts:,} final alerts · {correlated:,} windows correlated"
            elif worker.get("running"):
                detail = str(worker.get("stage") or "Waiting for all detector stages")
            elif worker.get("scheduled"):
                detail = "Runs automatically after ML and heuristic scoring"
            else:
                detail = "Not activated"
        else:
            windows_scored = int(status.get("windows_scored", 0) or 0)
            alerts_written = int(status.get("alerts_written", 0) or 0)
            if windows_scored or terminal:
                detail = f"{windows_scored:,} windows scored · {alerts_written:,} alerts"
            elif display_state == "Waiting For Input":
                detail = "No DNS records are available yet; the next scoring interval will retry."
            elif worker.get("running"):
                detail = str(worker.get("stage") or "Scoring DNS windows")
            elif worker.get("scheduled"):
                detail = "Waiting for next scoring interval"
            else:
                detail = "Not activated"
        ring_items.append(
            {
                "title": title,
                "state": display_state,
                "progress": progress,
                "active": bool(worker.get("running")),
                "detail": detail,
            }
        )

    _render_progress_rings(ring_items)
    st.caption(
        "Service health refreshes every five seconds. Zeek configuration: "
        f"`{ZEEK_NODE_CONFIG_PATH}` · DNS input: `{ZEEK_CURRENT_DIR}` · Runtime evidence: `{LIVE_SCORING_STATE_PATH}`. "
        f"Current input evidence: {input_detail}. Final Correlation runs automatically after all detector stages finish."
    )


if hasattr(st, "fragment"):
    _render_health_fragment = st.fragment(run_every="5s")(_render_health_cards)
else:
    _render_health_fragment = _render_health_cards


def _record_action(title: str, results: dict[str, Any], *, error: bool = False) -> None:
    st.session_state["dns_soc_last_action"] = {
        "title": title,
        "results": results,
        "error": bool(error),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _has_error(result: Any) -> bool:
    return isinstance(result, dict) and bool(result.get("error"))


def _activate_final_correlation_cron() -> dict[str, Any]:
    if not FINAL_ORCHESTRATOR_PATH.is_file():
        return {"error": f"Automatic correlation coordinator is missing: {FINAL_ORCHESTRATOR_PATH}"}
    cron_text = (
        "SHELL=/bin/bash\n"
        "PATH=/opt/dns-ml/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        "*/5 * * * * root /bin/sleep 20; /usr/bin/flock -n "
        f"{FINAL_LOCK_PATH} /opt/dns-ml/venv/bin/python {FINAL_ORCHESTRATOR_PATH} --scheduled "
        f">> {FINAL_LIVE_LOG_PATH} 2>&1\n"
    )
    try:
        FINAL_CRON_FILE.parent.mkdir(parents=True, exist_ok=True)
        FINAL_CRON_DISABLED_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = FINAL_CRON_FILE.with_suffix(".tmp")
        temporary.write_text(cron_text, encoding="utf-8")
        os.chmod(temporary, 0o644)
        temporary.replace(FINAL_CRON_FILE)
        FINAL_CRON_DISABLED_PATH.unlink(missing_ok=True)
        return {"enabled": True, "cron_file": str(FINAL_CRON_FILE), "error": None}
    except Exception as exc:
        return {"error": f"Unable to activate automatic final correlation: {type(exc).__name__}: {exc}"}


def _pause_final_correlation_cron() -> dict[str, Any]:
    try:
        if FINAL_CRON_FILE.exists():
            FINAL_CRON_DISABLED_PATH.parent.mkdir(parents=True, exist_ok=True)
            FINAL_CRON_DISABLED_PATH.write_text(
                FINAL_CRON_FILE.read_text(encoding="utf-8"), encoding="utf-8"
            )
            FINAL_CRON_FILE.unlink()
        return {"enabled": False, "disabled_copy": str(FINAL_CRON_DISABLED_PATH), "error": None}
    except Exception as exc:
        return {"error": f"Unable to pause automatic final correlation: {type(exc).__name__}: {exc}"}


def _activate_all(
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for percent, name, label, operation in (
        (74, "Autoencoder", "Activating Autoencoder scoring", scoring.activate_scoring_cron),
        (80, "Isolation Forest", "Activating Isolation Forest scoring", isolation_scoring.activate_cron),
        (86, "DNS Heuristics", "Activating DNS Heuristics scoring", heuristics_scoring.activate_cron),
        (92, "Final Correlation", "Activating automatic Final Correlation", _activate_final_correlation_cron),
    ):
        _notify_progress(progress_callback, percent, label)
        results[name] = operation()
    return results


def _pause_all(
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for percent, name, operation in (
        (10, "Autoencoder", scoring.deactivate_scoring_cron),
        (22, "Isolation Forest", isolation_scoring.deactivate_cron),
        (34, "DNS Heuristics", heuristics_scoring.deactivate_cron),
        (46, "Final Correlation", _pause_final_correlation_cron),
    ):
        _notify_progress(progress_callback, percent, f"Pausing {name}")
        results[name] = operation()
    return results


def _check_all_inputs() -> dict[str, Any]:
    return {
        "Autoencoder": scoring.check_live_zeek_log(),
        "Isolation Forest": isolation_scoring.check_live_zeek_log(),
        "DNS Heuristics": heuristics_scoring.check_live_zeek_log(),
    }


def _start_all(mode: str) -> dict[str, Any]:
    if mode == "full":
        autoencoder = scoring.start_full_log_scoring()
    else:
        autoencoder = scoring.start_scoring_once()
    return {
        "Autoencoder": autoencoder,
        "Isolation Forest": isolation_scoring.start_scoring(mode),
        "DNS Heuristics": heuristics_scoring.start_scoring(mode),
    }


def _activate_continuous(
    interface: str,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    preparation_error = dns_log_preparation.preparation_conflict_reason()
    if preparation_error:
        return {"DNS log preparation": {"error": preparation_error}}
    _notify_progress(progress_callback, 2, "Checking live-scoring prerequisites")
    gate_error = _scoring_gate_error()
    if gate_error:
        return {"Learning gate": {"error": gate_error}}
    results: dict[str, Any] = {
        "Zeek data capture": _configure_and_start_zeek(interface, progress_callback)
    }
    if _has_error(results["Zeek data capture"]):
        return results

    schedule_results = _activate_all(progress_callback)
    results.update(schedule_results)
    if any(_has_error(item) for item in schedule_results.values()):
        rollback_results = _pause_all()
        _stop_zeek_capture()
        results["Schedule rollback"] = rollback_results
        _write_lifecycle_state(
            state="failed",
            mode="continuous",
            capture_enabled=True,
            capture_interface=interface,
            last_error="One or more detector schedules could not be activated.",
        )
        return results

    _write_lifecycle_state(
        state="active",
        mode="continuous",
        capture_enabled=True,
        capture_interface=interface,
        activated_at_utc=datetime.now(timezone.utc).isoformat(),
        last_error=None,
    )
    _notify_progress(progress_callback, 100, "Live Zeek capture and all scoring stages are active")
    return results


def _pause_lifecycle(
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    _notify_progress(progress_callback, 3, "Preparing to pause live scoring")
    lifecycle = _read_lifecycle_state()
    results = _pause_all(progress_callback)
    if bool(lifecycle.get("capture_enabled")):
        results["Zeek data capture"] = _stop_zeek_capture(progress_callback)
    error = any(_has_error(item) for item in results.values())
    _write_lifecycle_state(
        state="failed" if error else "paused",
        mode="continuous",
        last_error="One or more pause actions failed." if error else None,
    )
    _notify_progress(
        progress_callback,
        100,
        "Pause failed; review the action output" if error else "Live scoring and Zeek capture are paused",
    )
    return results


def _resume_lifecycle(
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    preparation_error = dns_log_preparation.preparation_conflict_reason()
    if preparation_error:
        return {"DNS log preparation": {"error": preparation_error}}
    _notify_progress(progress_callback, 2, "Checking live-scoring prerequisites")
    gate_error = _scoring_gate_error()
    if gate_error:
        return {"Learning gate": {"error": gate_error}}
    lifecycle = _read_lifecycle_state()
    results: dict[str, Any] = {}
    if bool(lifecycle.get("capture_enabled")):
        results["Zeek data capture"] = _start_zeek_current_configuration(progress_callback)
        if _has_error(results["Zeek data capture"]):
            return results
    schedule_results = _activate_all(progress_callback)
    results.update(schedule_results)
    error = any(_has_error(item) for item in results.values())
    _write_lifecycle_state(
        state="failed" if error else "active",
        mode="continuous",
        resumed_at_utc=datetime.now(timezone.utc).isoformat(),
        last_error="One or more resume actions failed." if error else None,
    )
    _notify_progress(
        progress_callback,
        100,
        "Resume failed; review the action output" if error else "Live scoring and Zeek capture resumed",
    )
    return results


def _parse_manual_timestamp(value: Any) -> pd.Timestamp | None:
    try:
        numeric = float(value)
        absolute = abs(numeric)
        if absolute < 1e11:
            unit = "s"
        elif absolute < 1e14:
            unit = "ms"
        elif absolute < 1e17:
            unit = "us"
        else:
            unit = "ns"
        parsed = pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    except (TypeError, ValueError, OverflowError):
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _validate_manual_dns_upload(uploaded_file: Any) -> tuple[bytes | None, str | None, str | None]:
    if uploaded_file is None:
        return None, None, "Select one Zeek DNS JSON or JSONL file."
    name = str(getattr(uploaded_file, "name", "") or "").lower()
    if not name.endswith((".json", ".jsonl")):
        return None, None, "Only .json and .jsonl files are accepted for manual DNS scoring."
    try:
        payload = uploaded_file.getvalue()
    except Exception as exc:
        return None, None, f"Unable to read the upload: {type(exc).__name__}: {exc}"
    if not payload or not payload.strip():
        return None, None, "The uploaded DNS file is empty."

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None, None, "The uploaded DNS file must be UTF-8 JSON/JSONL."

    first = next((character for character in text if not character.isspace()), "")
    records: list[dict[str, Any]] = []
    canonical_name = "dns.jsonl"
    normalized_payload = payload
    try:
        if first == "[":
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                return None, None, "The JSON document must contain an array of Zeek DNS records."
            all_records = [item for item in parsed if isinstance(item, dict)]
            records = all_records[:2000]
            normalized_payload = (
                "".join(json.dumps(item, ensure_ascii=False, default=str) + "\n" for item in all_records)
            ).encode("utf-8")
        elif first == "{":
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    return None, None, "Every JSONL line must be a JSON object."
                records.append(item)
                if len(records) >= 2000:
                    break
        else:
            return None, None, "The file does not begin with a JSON object or array."
    except json.JSONDecodeError as exc:
        return None, None, f"Invalid JSON/JSONL near line {exc.lineno}: {exc.msg}"

    if not records:
        return None, None, "No Zeek DNS record objects were found."
    has_dns_identity = any(
        ("ts" in item or "timestamp" in item)
        and ("id.orig_h" in item or "src_ip" in item or "source_ip" in item)
        for item in records
    )
    if not has_dns_identity:
        return None, None, "No record contains both a timestamp and a source-IP field required by the DNS scorers."

    parsed_timestamps = [
        _parse_manual_timestamp(item.get("ts", item.get("timestamp")))
        for item in records
        if "ts" in item or "timestamp" in item
    ]
    if not parsed_timestamps or any(value is None for value in parsed_timestamps):
        return None, None, "One or more sampled DNS records contain an invalid timestamp."
    minimum_supported = pd.Timestamp("2000-01-01T00:00:00Z")
    maximum_supported = pd.Timestamp("2100-01-01T00:00:00Z")
    if any(
        value < minimum_supported or value > maximum_supported
        for value in parsed_timestamps
        if value is not None
    ):
        return None, None, (
            "The DNS timestamp unit is invalid or outside the supported 2000-2100 range. "
            "Use Zeek epoch seconds or ISO-8601 timestamps."
        )
    return normalized_payload, canonical_name, None


def _terminate_started_processes(processes: dict[str, subprocess.Popen[Any]]) -> None:
    for process in processes.values():
        try:
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            pass


def _start_uploaded_manual_scoring(uploaded_file: Any) -> dict[str, Any]:
    gate_error = _scoring_gate_error()
    if gate_error:
        return {"error": gate_error}
    payload, canonical_name, validation_error = _validate_manual_dns_upload(uploaded_file)
    if validation_error:
        return {"error": validation_error}
    if payload is None or canonical_name is None:
        return {"error": "The validated DNS upload is unavailable."}

    workers = _worker_snapshots()
    runtime = _runtime_summary(workers)
    if runtime["pipeline_running_count"] or runtime["scheduled_count"]:
        return {"error": "Pause continuous scoring and wait for active detector workers before starting a manual cycle."}
    existing = _read_lifecycle_state()
    if str(existing.get("mode") or "") == "manual" and str(existing.get("state") or "") in {"processing", "completed"}:
        return {"error": "A manual cycle already exists. Use Deactivate and Delete before starting another."}

    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    job_dir = MANUAL_UPLOAD_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    input_path = job_dir / canonical_name
    temporary = job_dir / (canonical_name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(input_path)

    python_bin = VENV_PYTHON_PATH if VENV_PYTHON_PATH.is_file() else Path(sys.executable)
    scoring.reset_scoring_log_for_manual_run()
    isolation_scoring.reset_live_log()
    heuristics_scoring.reset_live_log("Manual uploaded DNS one-shot")
    for path in (SCORING_LIVE_LOG_PATH, IF_SCORING_LIVE_LOG_PATH, heuristics_scoring.HEUR_LIVE_LOG_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()
    base_env["PYTHONUNBUFFERED"] = "1"
    base_env["PYTHONPATH"] = f"/opt/dns-ml:/opt/dns-ml-gui:{base_env.get('PYTHONPATH', '')}"
    base_env["CUDA_VISIBLE_DEVICES"] = "-1"
    processes: dict[str, subprocess.Popen[Any]] = {}
    try:
        auto_env = dict(base_env)
        auto_env["ZEEK_DNS_LOG"] = str(input_path)
        auto_env["DNS_SCORING_STATUS_PATH"] = str(SCORING_STATUS_PATH)
        auto_env["DNS_SCORING_WINDOWS_PATH"] = str(SCORING_WINDOWS_PATH)
        auto_env["DNS_SCORING_RESIDUALS_PATH"] = str(SCORING_FEATURE_RESIDUALS_PATH)
        auto_env["DNS_SCORING_LATENT_PATH"] = str(SCORING_LATENT_PATH)
        with SCORING_LIVE_LOG_PATH.open("a", encoding="utf-8") as output:
            processes["Autoencoder"] = subprocess.Popen(
                [str(python_bin), str(scoring.SCORING_SCRIPT_PATH), "--once", "--full-log"],
                cwd="/opt/dns-ml",
                env=auto_env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        SCORING_PID_PATH.write_text(str(processes["Autoencoder"].pid), encoding="utf-8")

        detector_env = dict(base_env)
        detector_env["DNS_ZEEK_CURRENT"] = str(job_dir)
        with IF_SCORING_LIVE_LOG_PATH.open("a", encoding="utf-8") as output:
            processes["Isolation Forest"] = subprocess.Popen(
                [str(python_bin), str(isolation_scoring.IF_SCORING_SCRIPT_PATH), "--mode", "full"],
                cwd="/opt/dns-ml",
                env=detector_env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        IF_SCORING_PID_PATH.write_text(str(processes["Isolation Forest"].pid), encoding="utf-8")

        with heuristics_scoring.HEUR_LIVE_LOG_PATH.open("a", encoding="utf-8") as output:
            processes["DNS Heuristics"] = subprocess.Popen(
                [str(python_bin), str(HEURISTICS_SCRIPT_PATH), "--mode", "full"],
                cwd="/opt/dns-ml",
                env=detector_env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        heuristics_scoring.HEUR_PID_PATH.write_text(
            str(processes["DNS Heuristics"].pid), encoding="utf-8"
        )

        FINAL_LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        detector_processes = list(processes.items())
        _write_lifecycle_state(
            state="processing",
            mode="manual",
            capture_enabled=False,
            capture_interface=None,
            manual_job_id=job_id,
            manual_input_path=str(input_path),
            activated_at_utc=datetime.now(timezone.utc).isoformat(),
            process_ids={name: process.pid for name, process in detector_processes},
            last_error=None,
        )
        coordinator_command = [
            str(python_bin),
            str(FINAL_ORCHESTRATOR_PATH),
            "--manual",
            "--job-id",
            job_id,
        ]
        for detector_name, detector_process in detector_processes:
            coordinator_command.extend(["--pid", f"{detector_name}={detector_process.pid}"])
        with FINAL_LIVE_LOG_PATH.open("a", encoding="utf-8") as output:
            processes["Final Correlation"] = subprocess.Popen(
                coordinator_command,
                cwd="/opt/dns-ml-gui",
                env=base_env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        FINAL_PID_PATH.write_text(str(processes["Final Correlation"].pid), encoding="utf-8")
        _write_lifecycle_state(
            process_ids={name: process.pid for name, process in processes.items()},
        )
    except Exception as exc:
        _terminate_started_processes(processes)
        _write_lifecycle_state(
            state="failed",
            mode="manual",
            manual_job_id=job_id,
            manual_input_path=str(input_path),
            last_error=f"Unable to start complete scoring pipeline: {type(exc).__name__}: {exc}",
        )
        return {"error": f"Unable to start all manual scorers: {type(exc).__name__}: {exc}", "job_id": job_id}
    return {
        "error": None,
        "mode": "manual-full-log",
        "job_id": job_id,
        "input_path": str(input_path),
        "process_ids": {name: process.pid for name, process in processes.items()},
    }


def _purge_targets() -> list[Path]:
    return [
        SCORING_STATUS_PATH,
        SCORING_PID_PATH,
        SCORING_LOCK_PATH,
        SCORING_LIVE_LOG_PATH,
        SCORING_WINDOWS_PATH,
        SCORING_FEATURE_RESIDUALS_PATH,
        SCORING_LATENT_PATH,
        SCORING_ACTION_STATUS_PATH,
        AUTOENCODER_STATE_PATH,
        AUTOENCODER_BUFFER_PATH,
        SCORING_CRON_DISABLED_PATH,
        SCORING_STATUS_PATH.parent / "cron_last_run.json",
        SCORING_STATUS_PATH.parent / "cron.log",
        IF_SCORING_STATUS_PATH,
        IF_SCORING_PID_PATH,
        IF_SCORING_LIVE_LOG_PATH,
        IF_SCORING_WINDOWS_PATH,
        IF_SCORING_ACTION_STATUS_PATH,
        IF_SCORING_STATE_PATH,
        IF_SCORING_BUFFER_PATH,
        IF_SCORING_CRON_DISABLED_PATH,
        IF_SCORING_CRON_LAST_PATH,
        IF_SCORING_STATUS_PATH.parent / "scoring.lock",
        IF_SCORING_STATUS_PATH.parent / "cron.log",
        IF_SCORING_STATUS_PATH.parent / "latest_isolation_scoring_shap.csv",
        heuristics_scoring.HEUR_STATUS_PATH,
        heuristics_scoring.HEUR_ACTION_STATUS_PATH,
        heuristics_scoring.HEUR_LIVE_LOG_PATH,
        heuristics_scoring.HEUR_PID_PATH,
        heuristics_scoring.HEUR_STATE_PATH,
        heuristics_scoring.HEUR_BUFFER_PATH,
        heuristics_scoring.HEUR_CRON_DISABLED_PATH,
        heuristics_scoring.HEUR_CRON_LAST_PATH,
        heuristics_scoring.HEUR_SCORING_HOME / "scoring.lock",
        heuristics_scoring.HEUR_SCORING_HOME / "cron.log",
        final_scoring.FINAL_STATUS_PATH,
        FINAL_CRON_DISABLED_PATH,
        FINAL_PID_PATH,
        FINAL_LOCK_PATH,
        FINAL_LIVE_LOG_PATH,
        LIVE_SCORING_STATE_PATH,
    ]


def _stop_manual_workers(lifecycle: dict[str, Any]) -> dict[str, Any]:
    stopped: list[int] = []
    failed: list[str] = []
    allowed_scripts = (
        "score_dns_autoencoder_live.py",
        "score_dns_isolation_forest_live.py",
        "score_dns_heuristics_live.py",
        "run_dns_final_correlation.py",
    )
    raw_pids = lifecycle.get("process_ids") or {}
    for label, raw_pid in raw_pids.items():
        try:
            pid = int(raw_pid)
            command_line = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace"
            )
            if not any(script in command_line for script in allowed_scripts):
                failed.append(f"{label}: PID {pid} no longer belongs to a DNS scoring process")
                continue
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            stopped.append(pid)
        except FileNotFoundError:
            continue
        except ProcessLookupError:
            continue
        except Exception as exc:
            failed.append(f"{label}: {type(exc).__name__}: {exc}")

    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if not any(Path(f"/proc/{pid}").exists() for pid in stopped):
            break
        time.sleep(0.2)
    remaining = [pid for pid in stopped if Path(f"/proc/{pid}").exists()]
    for pid in remaining:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception as exc:
            failed.append(f"PID {pid}: {type(exc).__name__}: {exc}")
    return {"stopped_pids": stopped, "failed": failed, "error": None if not failed else "Some manual workers could not be stopped."}


def _purge_runtime() -> dict[str, Any]:
    workers = _worker_snapshots()
    runtime = _runtime_summary(workers)
    lifecycle = runtime["lifecycle"]
    manual_mode = runtime["lifecycle_mode"] == "manual"
    stop_result: dict[str, Any] | None = None
    if runtime["pipeline_running_count"] and manual_mode:
        stop_result = _stop_manual_workers(lifecycle)
        if stop_result.get("error"):
            return {"error": stop_result["error"], "worker_stop": stop_result}
    elif runtime["pipeline_running_count"]:
        return {"error": "A scoring worker is still running. Wait for it to finish before purging."}
    if runtime["scheduled_count"] or workers["final"]["scheduled"]:
        return {"error": "Pause all continuous scoring and correlation schedules before deleting data."}

    try:
        alert_purge = clear_all_alerts(ALERT_JSONL_PATH.parent)
    except Exception as exc:
        return {
            "error": f"Unable to delete all system alerts: {type(exc).__name__}: {exc}",
            "worker_stop": stop_result,
        }

    deleted: list[str] = []
    failed: list[str] = []
    for path in _purge_targets():
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
                deleted.append(str(path))
        except Exception as exc:
            failed.append(f"{path}: {type(exc).__name__}: {exc}")
    try:
        if MANUAL_UPLOAD_ROOT.is_dir():
            shutil.rmtree(MANUAL_UPLOAD_ROOT)
            deleted.append(str(MANUAL_UPLOAD_ROOT))
    except Exception as exc:
        failed.append(f"{MANUAL_UPLOAD_ROOT}: {type(exc).__name__}: {exc}")
    return {
        "deleted": deleted,
        "failed": failed,
        "alert_purge": alert_purge,
        "worker_stop": stop_result,
        "error": None if not failed else "All alerts were deleted, but some transient runtime files could not be removed.",
    }


def _purge_dialog_body() -> None:
    busy = bool(st.session_state.get("dns_soc_admin_busy"))
    st.warning(
        "This action is permanent. Continuous scoring must be paused first. For a manual one-shot cycle, "
        "confirmation stops the detector/correlation pipeline before deleting its data."
    )
    st.markdown(
        """
        **Deleted across the system:** Autoencoder alerts, Isolation Forest alerts, DNS Heuristic alerts, final
        correlated alerts, alert-domain rows, alert-ingestion state, and all alert files under `/data/dns-ml/alerts`.

        **Also deleted:** transient scoring status, command output, diagnostic CSVs, cursors, incomplete-window
        buffers, stale PID/lock files, disabled schedule copies, scheduler metadata, and staged manual DNS uploads.

        **Preserved:** `/opt/zeek/logs`, uploaded/training datasets, shared training features, learned
        models/scalers/configuration, learning results, rules, and the saved Final Correlation policy.
        """
    )
    authorized = st.checkbox(
        "I authorize permanent deletion of all DNS alerts and live-scoring runtime data.",
        key="dns_soc_purge_authorized",
        disabled=busy,
    )
    confirm_col, cancel_col = st.columns(2)
    with confirm_col:
        confirm = st.button(
            "Confirm Deactivate and Delete",
            key="dns_soc_confirm_purge",
            type="primary",
            use_container_width=True,
            disabled=busy or not authorized,
        )
    with cancel_col:
        cancel = st.button(
            "Cancel",
            key="dns_soc_cancel_purge",
            use_container_width=True,
            disabled=busy,
        )

    if cancel:
        st.rerun()
    if confirm:
        st.session_state["dns_soc_admin_busy"] = True
        try:
            result = _purge_runtime()
            _record_action("Deactivate and Delete", result, error=bool(result.get("error")))
            if not result.get("error"):
                for key in (
                    "dns_soc_enable_capture",
                    "dns_soc_selected_capture_interface",
                    "dns_soc_manual_dns_upload",
                    "dns_soc_purge_authorized",
                ):
                    st.session_state.pop(key, None)
        finally:
            st.session_state["dns_soc_admin_busy"] = False
        st.rerun()


if hasattr(st, "dialog"):
    _purge_dialog = st.dialog("Deactivate and Delete", width="large")(
        _purge_dialog_body
    )
else:
    def _purge_dialog() -> None:
        with st.expander("Deactivate and Delete", expanded=True):
            _purge_dialog_body()


def _interface_dialog_body() -> None:
    st.markdown(
        "The inventory is read directly from this DNS ML server. Any non-loopback interface that is different from "
        "the management/default-route interface can be selected. An assigned IP address is not required."
    )
    st.warning(
        "The management/default-route interface is blocked to protect GUI/SSH administration. Link state and IP "
        "assignment are displayed for information only and do not block a dedicated capture interface."
    )
    rows, error = _interface_inventory()
    if error:
        st.error(error)
        eligible: list[str] = []
    else:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=300)
        eligible = [str(row["Interface"]) for row in rows if row["Eligibility"] == "ELIGIBLE"]

    selected = st.selectbox(
        "Eligible dedicated capture interface",
        options=eligible,
        index=None,
        placeholder="Select an eligible interface",
        key="dns_soc_interface_choice",
        disabled=not eligible,
    )
    if not eligible and not error:
        st.error(
            "No eligible dedicated capture interface is available. Attach a non-loopback NIC that is different "
            "from the management/default-route interface; the capture NIC does not need an IP address."
        )

    select_col, close_col = st.columns(2)
    with select_col:
        if st.button(
            "Use selected capture interface",
            key="dns_soc_confirm_interface",
            type="primary",
            use_container_width=True,
            disabled=selected is None,
        ):
            st.session_state["dns_soc_selected_capture_interface"] = str(selected)
            st.rerun()
    with close_col:
        if st.button("Close", key="dns_soc_close_interface", use_container_width=True):
            st.session_state["dns_soc_enable_capture"] = False
            st.rerun()


def _manual_scoring_dialog_body() -> None:
    gate_rows, gate_ready = _model_prerequisites()
    st.warning(
        "Zeek live capture is not selected. Upload one complete Zeek DNS JSON/JSONL file directly in this window. "
        "The upload is staged privately and all three detector families run once in full-log mode."
    )
    st.markdown(
        "**GUI submission order**\n\n"
        "1. Select one complete Zeek DNS `.json` or `.jsonl` file.\n"
        "2. Click **Upload DNS log and initialize one-shot scoring**.\n"
        "3. Wait for Autoencoder, Isolation Forest, and DNS Heuristics to finish.\n"
        "4. Final Correlation runs automatically after all three detectors finish. Use **Deactivate and Delete** "
        "before another manual cycle."
    )
    upload = st.file_uploader(
        "Manual Zeek DNS file (JSON/JSONL)",
        type=["json", "jsonl"],
        key="dns_soc_manual_dns_upload",
        help="The file must contain Zeek DNS records with a timestamp and source-IP field.",
    )
    st.caption(
        f"Protected staging root: `{MANUAL_UPLOAD_ROOT}` · Canonical runtime input: `dns.jsonl` or `dns.json`. "
        "The upload and all generated alerts are permanently deleted by Deactivate and Delete; models and training evidence are preserved."
    )
    st.info(
        "A manual cycle cannot be restarted while it is processing or after it completes. Use Deactivate and Delete "
        "to clear its staged input and alerts, then unlock the next cycle."
    )
    if not gate_ready:
        failed = [row["Requirement"] for row in gate_rows if row["Status"] != "PASS"]
        st.error(
            "Manual scoring is locked until training is finalized. Missing or incomplete: "
            + ", ".join(failed)
        )
    start_col, close_col = st.columns(2)
    with start_col:
        if st.button(
            "Upload DNS log and initialize one-shot scoring",
            key="dns_soc_start_manual_upload",
            type="primary",
            use_container_width=True,
            disabled=(
                not gate_ready
                or upload is None
                or bool(st.session_state.get("dns_soc_admin_busy"))
            ),
        ):
            st.session_state["dns_soc_admin_busy"] = True
            try:
                result = _start_uploaded_manual_scoring(upload)
                _record_action("Initialize manual DNS one-shot scoring", result, error=bool(result.get("error")))
            finally:
                st.session_state["dns_soc_admin_busy"] = False
            if result.get("error"):
                st.error(str(result["error"]))
            else:
                st.rerun()
    with close_col:
        if st.button("Close", key="dns_soc_close_manual_upload", use_container_width=True):
            st.rerun()


if hasattr(st, "dialog"):
    _interface_dialog = st.dialog("🌐 Select Zeek capture interface", width="large")(
        _interface_dialog_body
    )
    _manual_scoring_dialog = st.dialog("📥 Initialize manual DNS one-shot scoring", width="large")(
        _manual_scoring_dialog_body
    )
else:
    def _interface_dialog() -> None:
        with st.expander("Select Zeek capture interface", expanded=True):
            _interface_dialog_body()

    def _manual_scoring_dialog() -> None:
        with st.expander("Initialize manual DNS one-shot scoring", expanded=True):
            _manual_scoring_dialog_body()


def _render_runtime_heading() -> None:
    radar = _live_radar_state()
    state = html.escape(radar["state"])
    label = html.escape(radar["label"])
    detail = html.escape(radar["detail"], quote=True)
    st.markdown(
        f"""
        <div class="dns-runtime-heading-row">
            <div class="dns-section-heading">Runtime controls</div>
            <div class="dns-runtime-indicator {state}" role="status" aria-label="Live scoring: {label}. {detail}" title="{detail}">
                <span class="dns-live-radar" aria-hidden="true"><span class="dns-live-radar-pulse"></span></span>
                <span class="dns-runtime-indicator-text">
                    <span class="dns-runtime-indicator-name">Live Zeek scoring</span>
                    <span class="dns-runtime-indicator-state">{label}</span>
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if hasattr(st, "fragment"):
    _render_runtime_heading_fragment = st.fragment(run_every="5s")(_render_runtime_heading)
else:
    _render_runtime_heading_fragment = _render_runtime_heading


def _render_live_scoring_controls(ready: bool) -> None:
    workers = _worker_snapshots()
    runtime = _runtime_summary(workers)
    lifecycle = runtime["lifecycle"]
    stale_runtime = (
        any(path.exists() for path in _purge_targets())
        or MANUAL_UPLOAD_ROOT.exists()
        or any(
            path.exists()
            for path in (
                ALERT_JSONL_PATH,
                CATEGORIZATION_ALERT_JSONL_PATH,
                heuristics_scoring.HEUR_ALERT_PATH,
                final_scoring.FINAL_ALERTS_PATH,
            )
        )
    )
    manual_cycle = runtime["lifecycle_mode"] == "manual" and runtime["lifecycle_state"] in {"processing", "completed"}
    lifecycle_locked = runtime["scheduled_count"] > 0 or runtime["pipeline_running_count"] > 0 or manual_cycle or runtime["lifecycle_state"] == "paused"
    preparation_lock = dns_log_preparation.preparation_conflict_reason()

    if "dns_soc_enable_capture" not in st.session_state:
        st.session_state["dns_soc_enable_capture"] = bool(lifecycle.get("capture_enabled"))
    if "dns_soc_selected_capture_interface" not in st.session_state and lifecycle.get("capture_interface"):
        st.session_state["dns_soc_selected_capture_interface"] = str(lifecycle["capture_interface"])
    if preparation_lock and not lifecycle_locked:
        st.session_state["dns_soc_enable_capture"] = False

    try:
        capture_panel = st.container(border=True)
    except TypeError:
        capture_panel = st.container()
    with capture_panel:
        capture_enabled = st.checkbox(
            "Enable Zeek live data capture",
            key="dns_soc_enable_capture",
            disabled=lifecycle_locked or runtime["busy"] or bool(preparation_lock),
            help=(
                "Enabled: select a dedicated interface and run continuous five-minute DNS scoring. Disabled: "
                "Activate opens the manual one-shot DNS upload window."
            ),
        )

    selected_interface = str(st.session_state.get("dns_soc_selected_capture_interface") or "")
    if capture_enabled and not selected_interface and not lifecycle_locked:
        _interface_dialog()

    if capture_enabled:
        if selected_interface:
            st.success(f"Selected dedicated Zeek capture interface: {selected_interface}")
        else:
            st.error("Select an eligible dedicated Zeek capture interface before activation.")
    else:
        st.caption("Live capture is not selected. Activate Live Scoring will open the manual DNS one-shot upload window.")

    if preparation_lock:
        st.warning(
            preparation_lock
            + " Manual one-shot file scoring remains available because it does not start a capture interface."
        )

    st.caption(
        f"Raw monitored directory: `{ZEEK_CURRENT_DIR}` · Zeek interface configuration: `{ZEEK_NODE_CONFIG_PATH}` · "
        f"Runtime evidence: `{LIVE_SCORING_STATE_PATH}`."
    )
    st.info(
        "Autoencoder, Isolation Forest, and DNS Heuristics independently consume the same Zeek DNS records and "
        "five-minute feature windows. Final Correlation starts automatically only after the three detector stages finish."
    )

    _render_runtime_heading_fragment()
    paused = runtime["scheduled_count"] == 0 and (
        runtime["paused_count"] > 0 or runtime["lifecycle_state"] == "paused"
    )
    inactive_fresh = (
        runtime["scheduled_count"] == 0
        and runtime["paused_count"] == 0
        and runtime["pipeline_running_count"] == 0
        and not manual_cycle
        and runtime["lifecycle_state"] not in {"paused", "active"}
    )
    activate_disabled = (
        not ready
        or not inactive_fresh
        or runtime["busy"]
        or (capture_enabled and not selected_interface)
        or (capture_enabled and bool(preparation_lock))
    )
    pause_disabled = (
        runtime["scheduled_count"] == 0
        or runtime["busy"]
        or runtime["lifecycle_mode"] == "manual"
    )
    resume_disabled = (
        not ready
        or not paused
        or runtime["pipeline_running_count"] > 0
        or runtime["busy"]
        or runtime["lifecycle_mode"] == "manual"
    )
    purge_allowed = (
        manual_cycle
        or (paused and runtime["pipeline_running_count"] == 0)
        or (inactive_fresh and stale_runtime)
        or (runtime["lifecycle_state"] == "failed" and runtime["scheduled_count"] == 0)
    )
    purge_disabled = runtime["busy"] or runtime["scheduled_count"] > 0 or not purge_allowed

    columns = st.columns(4)
    with columns[0]:
        if st.button(
            "Activate Live Scoring",
            key="dns_soc_activate",
            use_container_width=True,
            disabled=activate_disabled,
            help=(
                "With Zeek capture enabled, configures the selected interface and activates all three detector "
                "schedules. Without capture, opens the manual DNS upload window."
            ),
        ):
            if capture_enabled:
                operation_progress = animated_progress(1, "Starting live Zeek capture")
                results = _activate_continuous(
                    selected_interface,
                    lambda percent, message: operation_progress.progress(percent, message),
                )
                action_error = any(_has_error(item) for item in results.values())
                if action_error:
                    operation_progress.progress(100, "Activation failed; review the action output")
                _record_action("Activate Live Scoring", results, error=action_error)
                st.rerun()
            else:
                _manual_scoring_dialog()
    with columns[1]:
        if st.button(
            "Pause Live Scoring",
            key="dns_soc_pause",
            use_container_width=True,
            disabled=pause_disabled,
            help="Pauses all three schedules and managed Zeek capture while preserving cursors, buffers, scores, and alerts.",
        ):
            operation_progress = animated_progress(1, "Pausing detector schedules and Zeek capture")
            results = _pause_lifecycle(
                lambda percent, message: operation_progress.progress(percent, message)
            )
            action_error = any(_has_error(item) for item in results.values())
            _record_action("Pause Live Scoring", results, error=action_error)
            st.rerun()
    with columns[2]:
        if st.button(
            "Resume Live Scoring",
            key="dns_soc_resume",
            use_container_width=True,
            disabled=resume_disabled,
            help="Restarts the same managed capture lifecycle and continues from preserved detector-specific cursors.",
        ):
            operation_progress = animated_progress(1, "Resuming Zeek capture and detector schedules")
            results = _resume_lifecycle(
                lambda percent, message: operation_progress.progress(percent, message)
            )
            action_error = any(_has_error(item) for item in results.values())
            _record_action("Resume Live Scoring", results, error=action_error)
            st.rerun()
    with columns[3]:
        if st.button(
            "Deactivate and Delete",
            key="dns_soc_open_purge",
            use_container_width=True,
            disabled=purge_disabled,
            help=(
                "Continuous mode must be paused first. Permanently deletes all ML, heuristic, and final correlated "
                "alerts plus transient scoring data. Models, training evidence, and Zeek logs are preserved."
            ),
        ):
            _purge_dialog()

    if manual_cycle and runtime["pipeline_running_count"]:
        st.info(
            "Manual DNS one-shot scoring is processing the uploaded file. Activate, Pause, and Resume are locked. "
            "Final Correlation will run automatically after the detector stages finish."
        )
    elif manual_cycle:
        if runtime["lifecycle_state"] == "processing":
            _write_lifecycle_state(
                state="completed",
                completed_at_utc=datetime.now(timezone.utc).isoformat(),
            )
        st.success(
            "Manual DNS scoring and automatic Final Correlation are complete. Use Deactivate and Delete before "
            "starting another cycle."
        )
    if not ready:
        st.error(
            "Scoring is locked until the shared training data exists, Autoencoder and Isolation Forest learning "
            "are finalized, and every required training/model artifact is present."
        )
    if 0 < runtime["scheduled_count"] < 3:
        st.warning(
            f"Partial lifecycle detected: only {runtime['scheduled_count']} of 3 detector schedules are active. "
            "The exact affected services are shown in the five progress rings above."
        )


def _read_log_for_worker(name: str) -> str:
    if name == "Autoencoder":
        return scoring.read_scoring_log(1500)
    if name == "Isolation Forest":
        return isolation_scoring.read_live_log(1500)
    if name == "DNS Heuristics":
        return heuristics_scoring.read_live_log(1500)
    status = final_scoring.read_final_status()
    return json.dumps(status, indent=2, ensure_ascii=False, default=str) if status else "No Final Correlation status is available yet."


def _render_output_component(text: str) -> None:
    safe = html.escape(text or "No scoring output is available yet.")
    components.html(
        f"""
        <!doctype html><html><head><style>
        html,body{{margin:0;background:transparent;color:#dff8ff;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;}}
        #out{{height:350px;overflow:auto;box-sizing:border-box;padding:14px 16px;background:#060B13;border:1px solid #168FC4;border-radius:12px;scrollbar-color:#11D8F2 #10283D;}}
        pre{{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font-size:12.5px;line-height:1.45;}}
        </style></head><body><div id="out"><pre>{safe}</pre></div></body></html>
        """,
        height=368,
        scrolling=False,
    )


def _render_operation_output() -> None:
    st.markdown('<div class="dns-section-heading">Live operation output</div>', unsafe_allow_html=True)
    workers = _worker_snapshots()
    detector_workers = [workers[name] for name in ("autoencoder", "isolation", "heuristics")]
    running = [worker for worker in detector_workers if worker["running"]]
    if running:
        state = "Processing"
        stage = " · ".join(worker["stage"] for worker in running)
        progress = int(sum(worker["progress"] for worker in running) / len(running))
    else:
        state = _runtime_summary(workers)["runtime"]
        latest_worker = max(detector_workers, key=lambda item: item["latest"])
        stage = latest_worker["stage"]
        progress = max(worker["progress"] for worker in detector_workers)

    action = st.session_state.get("dns_soc_last_action") or {}
    with st.container(key="dns_live_operation_metrics"):
        columns = st.columns(4)
        columns[0].metric("State", state)
        columns[1].metric("Stage", stage)
        columns[2].metric("Progress", f"{max(0, min(progress, 100))}%")
        columns[3].metric("Execution result", "Failed" if action.get("error") else ("Completed" if action else "Waiting"))
    animated_progress(max(0, min(progress, 100)), stage)

    if action:
        message = st.error if action.get("error") else st.success
        message(f"{action.get('title')} · {action.get('timestamp')}")
        with st.expander("Latest unified action result"):
            st.json(action.get("results") or {})

    selected = st.selectbox(
        "Command output source",
        ["Autoencoder", "Isolation Forest", "DNS Heuristics", "Final Correlation"],
        key="dns_soc_output_source",
    )
    _render_output_component(_read_log_for_worker(selected))


if hasattr(st, "fragment"):
    _render_operation_output_fragment = st.fragment(run_every="3s")(_render_operation_output)
else:
    _render_operation_output_fragment = _render_operation_output


def _render_engine_io_map() -> None:
    st.markdown('<div class="dns-section-heading">Scoring engine input and output map</div>', unsafe_allow_html=True)
    st.caption(
        "Authoritative files and folders used by each live detector and by automatic Final Correlation. "
        "The table is read-only."
    )
    rows = [
        {
            "Engine": "Autoencoder",
            "Input files / folders": (
                f"Zeek: {ZEEK_CURRENT_DIR}\n"
                f"Model: {AUTOENCODER_MODEL_PATH}\n"
                f"Scaler: {AUTOENCODER_SCALER_PATH}\n"
                f"Configuration: {MODEL_CONFIG_PATH}\n"
                f"Cursor: {AUTOENCODER_STATE_PATH}\n"
                f"Buffer: {AUTOENCODER_BUFFER_PATH}"
            ),
            "Output files / folders": (
                f"Scored windows: {SCORING_WINDOWS_PATH}\n"
                f"Feature residuals: {SCORING_FEATURE_RESIDUALS_PATH}\n"
                f"Latent vectors: {SCORING_LATENT_PATH}\n"
                f"Alerts: {ALERT_JSONL_PATH}\n"
                f"Status: {SCORING_STATUS_PATH}\n"
                f"Live log: {SCORING_LIVE_LOG_PATH}"
            ),
        },
        {
            "Engine": "Isolation Forest",
            "Input files / folders": (
                f"Zeek: {ZEEK_CURRENT_DIR}\n"
                f"Model: {IF_MODEL_PATH}\n"
                f"Scaler: {IF_SCALER_PATH}\n"
                f"Configuration: {IF_CONFIG_PATH}\n"
                f"Cursor: {IF_SCORING_STATE_PATH}\n"
                f"Buffer: {IF_SCORING_BUFFER_PATH}"
            ),
            "Output files / folders": (
                f"Scored windows: {IF_SCORING_WINDOWS_PATH}\n"
                f"Alerts: {CATEGORIZATION_ALERT_JSONL_PATH}\n"
                f"Status: {IF_SCORING_STATUS_PATH}\n"
                f"Live log: {IF_SCORING_LIVE_LOG_PATH}\n"
                f"Runtime folder: {IF_SCORING_STATUS_PATH.parent}"
            ),
        },
        {
            "Engine": "DNS Heuristics",
            "Input files / folders": (
                f"Zeek: {ZEEK_CURRENT_DIR}\n"
                f"Rule engine: {heuristics_scoring.HEUR_SCORING_SCRIPT_PATH}\n"
                "Rule configuration: GUI SQLite table rule_config\n"
                f"Cursor: {heuristics_scoring.HEUR_STATE_PATH}\n"
                f"Buffer: {heuristics_scoring.HEUR_BUFFER_PATH}"
            ),
            "Output files / folders": (
                f"Alerts: {heuristics_scoring.HEUR_ALERT_PATH}\n"
                f"Status: {heuristics_scoring.HEUR_STATUS_PATH}\n"
                f"Live log: {heuristics_scoring.HEUR_LIVE_LOG_PATH}\n"
                f"Runtime folder: {heuristics_scoring.HEUR_SCORING_HOME}"
            ),
        },
        {
            "Engine": "Final Correlation",
            "Input files / folders": (
                f"Autoencoder alerts: {ALERT_JSONL_PATH}\n"
                f"Isolation alerts: {CATEGORIZATION_ALERT_JSONL_PATH}\n"
                f"Heuristic alerts: {heuristics_scoring.HEUR_ALERT_PATH}\n"
                f"Policy: {final_scoring.FINAL_CONFIG_PATH}"
            ),
            "Output files / folders": (
                f"Final alerts: {final_scoring.FINAL_ALERTS_PATH}\n"
                f"Status: {final_scoring.FINAL_STATUS_PATH}\n"
                f"Coordinator log: {FINAL_LIVE_LOG_PATH}\n"
                f"Runtime folder: {final_scoring.FINAL_HOME}"
            ),
        },
    ]
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        height=390,
        column_config={
            "Engine": st.column_config.TextColumn(width="small"),
            "Input files / folders": st.column_config.TextColumn(width="large"),
            "Output files / folders": st.column_config.TextColumn(width="large"),
        },
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not _nonempty(path):
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
    except Exception:
        return []
    return rows


def _frame_from_rows(rows: list[dict[str, Any]], limit: int = 500) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "timestamp" in frame.columns:
        frame["_timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.sort_values("_timestamp", ascending=False).drop(columns=["_timestamp"])
    return frame.head(limit).reset_index(drop=True)


def _preferred_table(frame: pd.DataFrame, preferred: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    for column in ("active_models", "heuristics_fired", "source_alert_ids", "detection_methods"):
        if column in display.columns:
            display[column] = display[column].apply(
                lambda value: json.dumps(value, ensure_ascii=False, default=str)
                if isinstance(value, (list, dict))
                else value
            )
    columns = [column for column in preferred if column in display.columns]
    return display[columns] if columns else display


def _render_alert_tab(
    title: str,
    frame: pd.DataFrame,
    worker_state: str,
    source_path: Path,
    status_path: Path,
    preferred: list[str],
    waiting_text: str,
    total_rows: int,
    metrics_key: str | None = None,
) -> None:
    st.markdown(f"#### {title}")
    if metrics_key:
        with st.container(key=metrics_key):
            metrics = st.columns(3)
            metrics[0].metric("Rows", f"{total_rows:,}")
            metrics[1].metric("Worker status", worker_state)
            metrics[2].metric("Last update", _mtime_text(source_path))
    else:
        metrics = st.columns(3)
        metrics[0].metric("Rows", f"{total_rows:,}")
        metrics[1].metric("Worker status", worker_state)
        metrics[2].metric("Last update", _mtime_text(source_path))
    if frame.empty:
        st.markdown(f'<div class="dns-alert-empty">{_escape(waiting_text)}</div>', unsafe_allow_html=True)
    else:
        st.dataframe(
            _preferred_table(frame, preferred),
            use_container_width=True,
            hide_index=True,
            height=430,
        )
    st.caption(f"{title} data source: `{source_path}` · Status source: `{status_path}`")


def _first_present(*values: Any, default: Any = "Not available") -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return default


def _display_value(value: Any) -> str:
    if value in (None, ""):
        return "Not available"
    if isinstance(value, bool):
        return "YES" if value else "NO"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _render_read_only_details(title: str, rows: list[tuple[str, Any, str]]) -> None:
    st.markdown(f"#### {title}")
    frame = pd.DataFrame(
        [
            {
                "Data item": label,
                "Current value": _display_value(value),
                "Evidence source": source,
            }
            for label, value, source in rows
        ]
    )
    st.dataframe(frame, use_container_width=True, hide_index=True, height=min(520, 40 + len(frame) * 35))


def _load_ml_card_information(family: str) -> dict[str, dict[str, Any]]:
    try:
        if family == "autoencoder":
            return scoring.sync_run_once_card_from_status()
        return isolation_scoring.sync_cards_from_status()
    except Exception:
        try:
            if family == "autoencoder":
                return scoring.load_control_cards()
            return isolation_scoring.load_cards()
        except Exception:
            return {}


def _runtime_card_state(display_state: str) -> str:
    normalized = str(display_state or "").strip().lower()
    if normalized == "running":
        return "running"
    if normalized in {"completed", "scheduled", "ready", "active"}:
        return "success"
    if normalized == "failed":
        return "error"
    if normalized in {"paused", "inactive"}:
        return "warning"
    return "idle"


def _recommended_ml_status_cards(
    family: str,
    worker: dict[str, Any],
) -> list[dict[str, Any]]:
    status = worker.get("status") or {}
    previous_cards = _load_ml_card_information(family)
    retained_full_log = dict(previous_cards.get("full_log") or {})

    if family == "autoencoder":
        artifact_paths = (AUTOENCODER_MODEL_PATH, AUTOENCODER_SCALER_PATH, MODEL_CONFIG_PATH)
        artifact_names = "model, scaler, and learned configuration"
    else:
        artifact_paths = (IF_MODEL_PATH, IF_SCALER_PATH, IF_CONFIG_PATH)
        artifact_names = "model, scaler, and learned configuration"
    available_artifacts = sum(_nonempty(path) for path in artifact_paths)
    artifact_progress = round(available_artifacts * 100 / len(artifact_paths))
    model_ready = available_artifacts == len(artifact_paths)

    display_state = _display_worker_state(worker)
    worker_progress = max(0, min(100, int(worker.get("progress", 0) or 0)))
    if display_state in {"Completed", "Scheduled"}:
        worker_progress = 100

    if not retained_full_log:
        retained_full_log = {
            "title": "Full-log scoring evidence",
            "percent": 0,
            "state": "idle",
            "headline": "No full-log run recorded",
            "detail": "This historical card is retained as requested.",
        }
    else:
        retained_full_log["title"] = "Full-log scoring evidence"

    if worker.get("scheduled"):
        schedule_headline = "Scheduled every five minutes"
        schedule_detail = f"Active schedule: {worker.get('cron')}"
        schedule_progress = 100
        schedule_state = "success"
    elif worker.get("paused_copy"):
        schedule_headline = "Schedule paused"
        schedule_detail = "A disabled schedule copy is available for Resume Live Scoring."
        schedule_progress = 50
        schedule_state = "warning"
    else:
        schedule_headline = "Manual mode only"
        schedule_detail = "No recurring scoring schedule is currently active."
        schedule_progress = 0
        schedule_state = "idle"

    latest = str(worker.get("latest") or "Not available")
    latest_available = latest != "Not available"
    windows = int(status.get("windows_scored", 0) or 0)
    alerts = int(status.get("alerts_written", 0) or 0)
    if family == "isolation":
        result_detail = f"{windows:,} windows · {int(status.get('anomaly_windows', 0) or 0):,} anomalies · {alerts:,} alerts"
    else:
        result_detail = f"{windows:,} windows · {alerts:,} Autoencoder alerts"
    result_state = "error" if display_state == "Failed" else ("success" if display_state == "Completed" else "idle")

    return [
        {
            "title": "Model readiness",
            "percent": artifact_progress,
            "state": "success" if model_ready else "error",
            "headline": "Ready" if model_ready else f"{available_artifacts}/{len(artifact_paths)} artifacts ready",
            "detail": f"Checks the saved {artifact_names}.",
        },
        {
            "title": "Scoring worker status",
            "percent": worker_progress,
            "state": _runtime_card_state(display_state),
            "headline": display_state,
            "detail": f"{worker.get('stage') or 'No current stage'} · {status.get('message') or 'No worker message'}",
        },
        retained_full_log,
        {
            "title": "Runtime schedule",
            "percent": schedule_progress,
            "state": schedule_state,
            "headline": schedule_headline,
            "detail": schedule_detail,
        },
        {
            "title": "Latest scoring update",
            "percent": 100 if latest_available else 0,
            "state": "success" if latest_available else "idle",
            "headline": "Timestamp available" if latest_available else "No update recorded",
            "detail": latest,
        },
        {
            "title": "Latest cycle result",
            "percent": 100 if display_state in {"Completed", "Failed"} else worker_progress,
            "state": result_state,
            "headline": display_state,
            "detail": result_detail,
        },
    ]


def _render_ml_card_information(family: str, title: str, worker: dict[str, Any]) -> None:
    cards = _recommended_ml_status_cards(family, worker)
    st.markdown(f"#### {title} scoring card information")
    st.caption(
        "Cards 1, 2, 4, 5, and 6 show recommended live model/runtime status. Card 3 retains the "
        "latest full-log scoring evidence. All cards are read-only; use the page-level Runtime controls."
    )
    fragments = ['<div class="dns-ml-info-grid">']
    for number, card in enumerate(cards, start=1):
        try:
            progress = max(0, min(100, int(card.get("percent", 0) or 0)))
        except Exception:
            progress = 0
        state = str(card.get("state") or "idle")
        fragments.append(
            '<div class="dns-ml-info-card">'
            f'<div class="dns-ml-info-number">Card Info Window {number}</div>'
            f'<div class="dns-ml-info-title">{_escape(card.get("title") or "Runtime status")}</div>'
            f'<div class="dns-ml-info-ring" style="--dns-progress:{progress};--dns-ring:{_ring_color(state)}">'
            f'<span>{progress}%</span></div>'
            f'<div class="dns-ml-info-headline {_state_class(state)}">{_escape(card.get("headline") or state.title())}</div>'
            f'<div class="dns-ml-info-detail">{_escape(card.get("detail") or "No operation evidence recorded yet.")}</div>'
            '</div>'
        )
    fragments.append("</div>")
    st.markdown("".join(fragments), unsafe_allow_html=True)


def _numeric_max(frame: pd.DataFrame, *columns: str) -> Any:
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not values.empty:
            return float(values.max())
    return "Not available"


def _latest_rows(frame: pd.DataFrame, preferred: list[str], limit: int = 25) -> pd.DataFrame:
    if frame.empty:
        return frame
    result = frame.copy()
    if "timestamp" in result.columns:
        parsed = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
        result = result.assign(_sort_timestamp=parsed).sort_values("_sort_timestamp", ascending=False)
        result = result.drop(columns=["_sort_timestamp"])
    else:
        result = result.iloc[::-1]
    columns = [column for column in preferred if column in result.columns]
    return (result[columns] if columns else result).head(limit).reset_index(drop=True)


def _render_autoencoder_scoring_readings(worker: dict[str, Any]) -> None:
    try:
        scored = scoring.load_scoring_windows()
    except Exception:
        scored = pd.DataFrame()
    status = worker.get("status") or {}
    config = _read_json(MODEL_CONFIG_PATH, {})
    threshold = _first_present(
        config.get("threshold"),
        config.get("reconstruction_threshold"),
        default="Not available",
    )
    anomaly_count: Any = status.get("anomaly_windows")
    if anomaly_count in (None, "") and "autoencoder_anomaly" in scored.columns:
        anomaly_count = int(scored["autoencoder_anomaly"].fillna(False).astype(bool).sum())
    _render_cards(
        [
            {"label": "Scored windows", "value": int(status.get("windows_scored", 0) or len(scored)), "detail": "Latest Autoencoder evaluation"},
            {"label": "Anomalous windows", "value": int(anomaly_count or 0), "detail": "Windows above the learned threshold"},
            {"label": "Maximum reconstruction error", "value": _numeric_max(scored, "reconstruction_error"), "detail": "Largest available live ML reading"},
            {"label": "Learned threshold", "value": threshold, "detail": "Saved Autoencoder decision boundary"},
        ]
    )
    latest = _latest_rows(
        scored,
        ["timestamp", "src_ip", "reconstruction_error", "threshold", "threshold_ratio", "autoencoder_anomaly", "scoring_mode"],
    )
    st.markdown("##### Latest Autoencoder ML window readings")
    if latest.empty:
        st.info("No Autoencoder window readings are available yet.")
    else:
        st.dataframe(latest, use_container_width=True, hide_index=True, height=320)


def _render_isolation_scoring_readings(worker: dict[str, Any]) -> None:
    try:
        scored = isolation_scoring.load_scoring_windows()
    except Exception:
        scored = pd.DataFrame()
    status = worker.get("status") or {}
    config = _read_json(IF_CONFIG_PATH, {})
    anomaly_count: Any = _first_present(
        status.get("anomaly_windows"),
        status.get("anomalies"),
        default=None,
    )
    if anomaly_count in (None, ""):
        if "predicted_label" in scored.columns:
            labels = scored["predicted_label"].astype(str).str.lower()
            anomaly_count = int(labels.isin({"anomaly", "-1"}).sum())
        elif "predicted_code" in scored.columns:
            anomaly_count = int((pd.to_numeric(scored["predicted_code"], errors="coerce") == -1).sum())
    _render_cards(
        [
            {"label": "Scored windows", "value": int(status.get("windows_scored", 0) or len(scored)), "detail": "Latest Isolation Forest evaluation"},
            {"label": "Anomalous windows", "value": int(anomaly_count or 0), "detail": "Windows classified as outliers"},
            {"label": "Maximum anomaly score", "value": _numeric_max(scored, "anomaly_score", "isolation_score"), "detail": "Largest available live ML reading"},
            {"label": "Contamination", "value": _first_present(config.get("contamination"), default="Not available"), "detail": "Saved expected anomaly proportion"},
        ]
    )
    latest = _latest_rows(
        scored,
        ["timestamp", "src_ip", "anomaly_score", "isolation_score", "isolation_threshold", "score_margin", "score_percentile", "predicted_label", "scoring_mode"],
    )
    st.markdown("##### Latest Isolation Forest ML window readings")
    if latest.empty:
        st.info("No Isolation Forest window readings are available yet.")
    else:
        st.dataframe(latest, use_container_width=True, hide_index=True, height=320)


def _render_autoencoder_details(worker: dict[str, Any]) -> None:
    training_status = _read_json(AUTOENCODER_TRAINING_STATUS_PATH, {})
    training_result = _read_json(AUTOENCODER_TRAINING_RESULT_PATH, {})
    model_config = _read_json(MODEL_CONFIG_PATH, {})
    scoring_status = worker.get("status") or {}
    features = model_config.get("features")
    feature_count = len(features) if isinstance(features, list) else _first_present(
        model_config.get("feature_count"), model_config.get("input_dim"), training_result.get("feature_count")
    )
    _render_read_only_details(
        "Autoencoder learning and scoring data",
        [
            ("Learning phase", str(training_status.get("state") or "missing").upper(), str(AUTOENCODER_TRAINING_STATUS_PATH)),
            ("Training windows", _first_present(training_result.get("training_windows"), model_config.get("training_windows"), training_status.get("training_windows")), str(AUTOENCODER_TRAINING_RESULT_PATH)),
            ("Feature count", feature_count, str(MODEL_CONFIG_PATH)),
            ("Window size", _first_present(model_config.get("window_size"), training_result.get("window_size"), default="5min"), str(MODEL_CONFIG_PATH)),
            ("Learned reconstruction threshold", _first_present(model_config.get("threshold"), model_config.get("reconstruction_threshold"), training_result.get("threshold")), str(MODEL_CONFIG_PATH)),
            ("Threshold quantile", _first_present(model_config.get("threshold_quantile"), training_result.get("threshold_quantile")), str(MODEL_CONFIG_PATH)),
            ("Training epochs", _first_present(training_result.get("epochs"), model_config.get("epochs"), training_status.get("epochs")), str(AUTOENCODER_TRAINING_RESULT_PATH)),
            ("Learning completed", _first_present(training_result.get("completed_at_utc"), training_status.get("completed_at_utc")), str(AUTOENCODER_TRAINING_RESULT_PATH)),
            ("Live scoring state", _display_worker_state(worker), str(SCORING_STATUS_PATH)),
            ("Latest scoring stage", worker.get("stage"), str(SCORING_STATUS_PATH)),
            ("Latest windows scored", scoring_status.get("windows_scored", 0), str(SCORING_STATUS_PATH)),
            ("Latest alerts written", scoring_status.get("alerts_written", 0), str(SCORING_STATUS_PATH)),
        ],
    )


def _render_isolation_details(worker: dict[str, Any]) -> None:
    training_status = _read_json(ISOLATION_TRAINING_STATUS_PATH, {})
    training_result = _read_json(ISOLATION_TRAINING_RESULT_PATH, {})
    model_config = _read_json(IF_CONFIG_PATH, {})
    scoring_status = worker.get("status") or {}
    _render_read_only_details(
        "Isolation Forest learning and scoring data",
        [
            ("Learning phase", str(training_status.get("state") or "missing").upper(), str(ISOLATION_TRAINING_STATUS_PATH)),
            ("Training windows", _first_present(training_result.get("training_windows"), model_config.get("training_windows"), training_status.get("training_windows")), str(ISOLATION_TRAINING_RESULT_PATH)),
            ("Feature count", _first_present(training_result.get("feature_count"), model_config.get("feature_count"), default=len(_read_json(IF_CONFIG_PATH, {}).get("features", [])) or "Not available"), str(IF_CONFIG_PATH)),
            ("Estimators", _first_present(model_config.get("n_estimators"), training_result.get("n_estimators")), str(IF_CONFIG_PATH)),
            ("Maximum samples", _first_present(model_config.get("max_samples"), training_result.get("max_samples")), str(IF_CONFIG_PATH)),
            ("Contamination", _first_present(model_config.get("contamination"), training_result.get("contamination")), str(IF_CONFIG_PATH)),
            ("Random state", _first_present(model_config.get("random_state"), training_result.get("random_state")), str(IF_CONFIG_PATH)),
            ("Learning completed", _first_present(training_result.get("completed_at_utc"), training_status.get("completed_at_utc")), str(ISOLATION_TRAINING_RESULT_PATH)),
            ("Live scoring state", _display_worker_state(worker), str(IF_SCORING_STATUS_PATH)),
            ("Latest scoring stage", worker.get("stage"), str(IF_SCORING_STATUS_PATH)),
            ("Latest windows scored", scoring_status.get("windows_scored", 0), str(IF_SCORING_STATUS_PATH)),
            ("Latest alerts written", scoring_status.get("alerts_written", 0), str(IF_SCORING_STATUS_PATH)),
        ],
    )


def _heuristic_rule_frame() -> pd.DataFrame:
    try:
        frame = read_sql(
            """
            SELECT rule_name, enabled, description, rule_json, updated_at_utc
            FROM rule_config
            ORDER BY rule_name
            """
        )
    except Exception:
        return pd.DataFrame()
    if not frame.empty:
        frame = frame.rename(
            columns={
                "rule_name": "Rule",
                "enabled": "Enabled",
                "description": "Description",
                "rule_json": "Thresholds / conditions",
                "updated_at_utc": "Updated UTC",
            }
        )
        frame["Enabled"] = frame["Enabled"].apply(lambda value: "YES" if bool(value) else "NO")
        frame["Thresholds / conditions"] = frame["Thresholds / conditions"].apply(
            lambda value: str(value or "").replace("\n", " ")
        )
    return frame


def _render_heuristic_details(worker: dict[str, Any]) -> None:
    scoring_status = worker.get("status") or {}
    rules = _heuristic_rule_frame()
    enabled_rules = int((rules["Enabled"] == "YES").sum()) if "Enabled" in rules else 0
    _render_read_only_details(
        "DNS Heuristics configuration and scoring data",
        [
            ("Detection engine", "Rule-based DNS heuristics (no model-training phase)", str(HEURISTICS_SCRIPT_PATH)),
            ("Window size", heuristics_scoring.WINDOW_SIZE, str(HEURISTICS_SCRIPT_PATH)),
            ("Configured rules", len(rules), "GUI database rule_config"),
            ("Enabled rules", enabled_rules, "GUI database rule_config"),
            ("Live scoring state", _display_worker_state(worker), str(heuristics_scoring.HEUR_STATUS_PATH)),
            ("Latest scoring stage", worker.get("stage"), str(heuristics_scoring.HEUR_STATUS_PATH)),
            ("Latest windows scored", scoring_status.get("windows_scored", 0), str(heuristics_scoring.HEUR_STATUS_PATH)),
            ("Rule-positive windows", scoring_status.get("heuristic_windows", 0), str(heuristics_scoring.HEUR_STATUS_PATH)),
            ("Latest alerts written", scoring_status.get("alerts_written", 0), str(heuristics_scoring.HEUR_STATUS_PATH)),
        ],
    )
    st.markdown("##### Active heuristic rule definitions")
    if rules.empty:
        st.info("No heuristic rule rows are available in the GUI database.")
    else:
        st.dataframe(rules, use_container_width=True, hide_index=True, height=min(390, 45 + len(rules) * 56))


def _render_final_policy_details(worker: dict[str, Any]) -> None:
    config = final_scoring.load_final_config()
    status = worker.get("status") or {}
    labels = {
        "correlation_minutes": "Correlation window (minutes)",
        "isolation_min_score": "Minimum Isolation score",
        "isolation_min_percentile": "Minimum Isolation percentile",
        "autoencoder_threshold_multiplier": "Autoencoder threshold multiplier",
        "heuristic_min_threat_score": "Minimum heuristic threat score",
        "weight_isolation": "Isolation weight (%)",
        "weight_autoencoder": "Autoencoder weight (%)",
        "weight_heuristics": "Heuristics weight (%)",
        "consensus_bonus_two": "Two-detector consensus bonus",
        "consensus_bonus_three": "Three-detector consensus bonus",
        "minimum_agreeing_models": "Minimum agreeing detector families",
        "final_alert_threshold": "Final alert threshold",
    }
    rows = [
        ("Correlation execution state", _display_worker_state(worker), str(final_scoring.FINAL_STATUS_PATH)),
        ("Correlated windows", status.get("correlated_windows", 0), str(final_scoring.FINAL_STATUS_PATH)),
        ("Final alerts generated", status.get("final_alerts", 0), str(final_scoring.FINAL_STATUS_PATH)),
        ("Two-detector candidates", status.get("two_model_candidates", 0), str(final_scoring.FINAL_STATUS_PATH)),
        ("Three-detector candidates", status.get("three_model_candidates", 0), str(final_scoring.FINAL_STATUS_PATH)),
    ]
    rows.extend((label, config.get(key), str(final_scoring.FINAL_CONFIG_PATH)) for key, label in labels.items())
    _render_read_only_details("Final correlation policy and execution data", rows)
    st.caption(
        "This policy is read-only on the consolidated scoring page. Final Correlation runs automatically after "
        "Autoencoder, Isolation Forest, and DNS Heuristics complete."
    )


def _render_unified_alerts() -> None:
    st.markdown('<div class="dns-section-heading">Unified live alert list</div>', unsafe_allow_html=True)
    workers = _worker_snapshots()
    final_rows = _read_jsonl(final_scoring.FINAL_ALERTS_PATH)
    auto_rows = _read_jsonl(ALERT_JSONL_PATH)
    isolation_rows = _read_jsonl(CATEGORIZATION_ALERT_JSONL_PATH)
    heuristic_rows = _read_jsonl(heuristics_scoring.HEUR_ALERT_PATH)
    three_model = sum(int(row.get("models_agreeing", 0) or 0) >= 3 for row in final_rows)

    _render_cards(
        [
            {"label": "Final correlated alerts", "value": f"{len(final_rows):,}", "detail": "Authoritative unified final output"},
            {"label": "Autoencoder alerts", "value": f"{len(auto_rows):,}", "detail": "Native reconstruction-error alerts"},
            {"label": "Isolation alerts", "value": f"{len(isolation_rows):,}", "detail": "Native Isolation Forest alerts"},
            {"label": "Heuristic alerts", "value": f"{len(heuristic_rows):,}", "detail": "Native rule-based alerts"},
            {"label": "Three-detector consensus", "value": f"{three_model:,}", "detail": "Final alerts supported by all families"},
        ],
        five=True,
    )

    final_frame = _frame_from_rows(final_rows)
    st.caption(f"Unified output: `{final_scoring.FINAL_ALERTS_PATH}` · Independent refresh interval: 5 seconds")

    tabs = st.tabs(
        [
            "Final Correlated Alerts",
            "Autoencoder Alerts",
            "Isolation Forest Alerts",
            "Heuristic Alerts",
        ]
    )
    with tabs[0]:
        _render_final_policy_details(workers["final"])
        _render_alert_tab(
            "Final Correlated Alerts",
            final_frame,
            _display_worker_state(workers["final"]),
            final_scoring.FINAL_ALERTS_PATH,
            final_scoring.FINAL_STATUS_PATH,
            ["id", "timestamp", "src_ip", "severity", "final_score", "models_agreeing", "active_models", "source_alert_ids"],
            "No final correlated alerts are available with the current correlation policy.",
            len(final_rows),
            metrics_key="dns_final_alert_metrics",
        )
    with tabs[1]:
        _render_ml_card_information("autoencoder", "Autoencoder", workers["autoencoder"])
        _render_autoencoder_scoring_readings(workers["autoencoder"])
        _render_autoencoder_details(workers["autoencoder"])
        _render_alert_tab(
            "Autoencoder Alerts",
            _frame_from_rows(auto_rows),
            _display_worker_state(workers["autoencoder"]),
            ALERT_JSONL_PATH,
            SCORING_STATUS_PATH,
            ["id", "alert_uid", "timestamp", "src_ip", "severity", "threat_score", "reconstruction_error", "threshold", "scoring_mode"],
            "The Autoencoder scoring cycle completed but did not produce a native alert for the evaluated windows.",
            len(auto_rows),
        )
    with tabs[2]:
        _render_ml_card_information("isolation", "Isolation Forest", workers["isolation"])
        _render_isolation_scoring_readings(workers["isolation"])
        _render_isolation_details(workers["isolation"])
        _render_alert_tab(
            "Isolation Forest Alerts",
            _frame_from_rows(isolation_rows),
            _display_worker_state(workers["isolation"]),
            CATEGORIZATION_ALERT_JSONL_PATH,
            IF_SCORING_STATUS_PATH,
            ["id", "timestamp", "src_ip", "severity", "threat_score", "isolation_score", "score_percentile", "predicted_label", "scoring_mode"],
            "The Isolation Forest scoring cycle completed but did not classify an evaluated window as anomalous.",
            len(isolation_rows),
        )
    with tabs[3]:
        _render_heuristic_details(workers["heuristics"])
        _render_alert_tab(
            "Heuristic Alerts",
            _frame_from_rows(heuristic_rows),
            _display_worker_state(workers["heuristics"]),
            heuristics_scoring.HEUR_ALERT_PATH,
            heuristics_scoring.HEUR_STATUS_PATH,
            ["id", "timestamp", "src_ip", "severity", "threat_score", "heuristics_fired", "scoring_mode"],
            "The DNS Heuristics scoring cycle completed but no enabled rule fired for the evaluated windows.",
            len(heuristic_rows),
        )


if hasattr(st, "fragment"):
    _render_alert_fragment = st.fragment(run_every="5s")(_render_unified_alerts)
else:
    _render_alert_fragment = _render_unified_alerts


def render_dns_scoring_soc_page(
    *,
    render_autoencoder: Callable[[], None],
    render_isolation: Callable[[], None],
    render_heuristics: Callable[[], None],
    render_final: Callable[[], None],
) -> None:
    _inject_scoring_css()
    st.markdown('<div class="dns-scoring-shell"></div>', unsafe_allow_html=True)

    prerequisite_rows, ready = _model_prerequisites()
    _render_top_status(ready)

    with st.expander("Learning prerequisites", expanded=not ready):
        st.dataframe(pd.DataFrame(prerequisite_rows), use_container_width=True, hide_index=True)
        if ready:
            st.success("All required learning and scoring artifacts are ready.")
        else:
            missing = [row["Requirement"] for row in prerequisite_rows if row["Status"] != "PASS"]
            st.error("Scoring controls are locked. Missing/incomplete: " + ", ".join(missing))
        st.caption(
            "Learning evidence is displayed read-only. Scoring lifecycle actions never modify the shared training feature CSV or learned artifacts."
        )

    st.markdown('<div class="dns-section-heading">Live scoring settings</div>', unsafe_allow_html=True)
    _render_health_fragment()
    _render_live_scoring_controls(ready)
    _render_engine_io_map()
    _render_operation_output_fragment()
    _render_alert_fragment()
