# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

import html
import json
import time
from datetime import datetime, timezone

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.db import (
    DB_PATH,
    clear_all_alerts,
    execute,
    get_alert_by_id,
    get_domains_for_alert,
    init_db,
    read_sql,
    update_alert_status,
)
from src.ingest import ingest_alert_file
from src.learning import (
    is_learning_running,
    load_history_df,
    load_reconstruction_df,
    learning_log_signature,
    is_terminal_learning_state,
    read_complete_log,
    read_learning_result,
    read_learning_status,
    read_log_tail,
    start_learning,
    stop_learning,
)
from src.settings import (
    ALERT_JSONL_PATH,
    APP_NAME,
    LEARNING_HISTORY_PATH,
    LEARNING_LOG_PATH,
    LEARNING_RECONSTRUCTION_PATH,
    MODEL_CONFIG_PATH,
    ZEEK_LOG_ROOT,
    ZEEK_MERGED_PATH,
    ZEEK_SOURCE_IP_STATS_PATH,
)
from src.ui_helpers import (
    animated_progress,
    dataframe_or_info,
    parse_json,
    render_solution_footer,
    severity_label,
    show_detection_methods,
)
from src.zeek_manager import (
    artifact_health,
    discover_source_files,
    is_merge_running,
    needs_merge,
    read_log_tail as read_zeek_log_tail,
    read_merge_status,
    read_stats as read_zeek_stats,
    start_merge,
    stop_merge,
)


from src.scoring import (
    activate_scoring_cron,
    check_live_zeek_log,
    check_scoring_cron_status,
    deactivate_scoring_cron,
    is_scoring_running,
    load_control_cards,
    load_feature_residuals,
    load_latent_vectors,
    load_live_alerts,
    load_scoring_windows,
    read_scoring_log,
    read_scoring_status,
    start_full_log_scoring,
    start_scoring_once,
    sync_run_once_card_from_status,
)



# BEGIN DNS ISOLATION FOREST LEARNING IMPORTS
from src.isolation_learning import (
    IF_CONFIG_PATH,
    IF_CONTOUR_PATH,
    IF_LOG_PATH,
    IF_MODEL_PATH,
    IF_PCA_PATH,
    IF_POINTS_PATH,
    IF_RESULT_PATH,
    IF_SCALER_PATH,
    IF_SCORES_PATH,
    IF_SHAP_DEPENDENCY_PATH,
    IF_SHAP_SUMMARY_PATH,
    IF_SHAP_VALUES_PATH,
    IF_STATUS_PATH,
    is_isolation_learning_running,
    is_terminal_isolation_state,
    isolation_log_signature,
    load_isolation_contour,
    load_isolation_points,
    load_isolation_scores,
    load_isolation_shap_dependency,
    load_isolation_shap_summary,
    load_isolation_shap_values,
    read_complete_isolation_log,
    read_isolation_log_tail,
    read_isolation_result,
    read_isolation_status,
    start_isolation_learning,
    stop_isolation_learning,
)
# END DNS ISOLATION FOREST LEARNING IMPORTS


# BEGIN DNS CATEGORIZATION SCORING IMPORTS
from src.settings import CATEGORIZATION_ALERT_JSONL_PATH
from src.isolation_scoring import (
    activate_cron as activate_categorization_cron,
    check_cron_status as check_categorization_cron_status,
    check_live_zeek_log as check_categorization_live_zeek_log,
    deactivate_cron as deactivate_categorization_cron,
    is_running as is_categorization_scoring_running,
    load_cards as load_categorization_cards,
    load_live_alerts as load_categorization_live_alerts,
    load_model_parameters as load_categorization_model_parameters,
    load_scoring_windows as load_categorization_scoring_windows,
    read_live_log as read_categorization_scoring_log,
    read_status as read_categorization_scoring_status,
    start_scoring as start_categorization_scoring,
    sync_cards_from_status as sync_categorization_cards,
)
from src.db import clear_all_alerts
# END DNS CATEGORIZATION SCORING IMPORTS

# BEGIN DNS HEURISTICS SCORING IMPORTS
from src.heuristics_scoring import (
    HEUR_ALERT_PATH,
    activate_cron as activate_heuristics_cron,
    check_cron_status as check_heuristics_cron_status,
    check_live_zeek_log as check_heuristics_live_zeek_log,
    deactivate_cron as deactivate_heuristics_cron,
    is_running as is_heuristics_scoring_running,
    load_live_alerts as load_heuristics_live_alerts,
    read_live_log as read_heuristics_scoring_log,
    read_status as read_heuristics_scoring_status,
    start_scoring as start_heuristics_scoring,
    sync_cards_from_status as sync_heuristics_cards,
)
# END DNS HEURISTICS SCORING IMPORTS

# BEGIN DNS FINAL CORRELATION IMPORTS
from src.final_scoring import (
    DEFAULT_CONFIG,
    FINAL_ALERTS_PATH,
    calculate_final_alerts,
    load_final_alerts,
    load_final_config,
    read_final_status,
    reset_final_runtime_state,
    save_final_config,
)
# END DNS FINAL CORRELATION IMPORTS

from src.scoring_page_soc import render_dns_scoring_soc_page
from src.dns_log_preparation_page import render_dns_log_preparation_page

from pathlib import Path
from src import training_upload as training_upload_manager
from src import training_lifecycle
from src.local_auth import (
    allowed_pages_for_user,
    render_account_toolbar,
    render_login_gate,
    render_user_management_page,
)
import os
DNS_CONSOLE_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "dns_console_logo.png"

st.set_page_config(
    page_title=APP_NAME,
    page_icon=str(DNS_CONSOLE_LOGO_PATH),
    layout="wide",
)

# DNS HIDE STREAMLIT BRANDING
st.markdown(
    """<style>
    footer,
    #MainMenu,
    [data-testid='stMainMenu'],
    [data-testid='stToolbar'],
    [data-testid='stToolbarActions'],
    [data-testid='stHeaderActionElements'],
    [data-testid='stAppDeployButton'],
    [data-testid='stStatusWidget'],
    [data-testid='stFooter'],
    [data-testid='stAppFooter'],
    [data-testid='stBottomBlockContainer'] footer,
    .viewerBadge_container__r5tak,
    .viewerBadge_link__qRIco {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
        max-height: 0 !important;
        overflow: hidden !important;
    }
    </style>""",
    unsafe_allow_html=True,
)

init_db()


st.markdown(
    """
    <style>
    :root {
        --soc-card: linear-gradient(135deg, #161d2b 0%, #0f1724 100%);
        --soc-border: rgba(148, 163, 184, 0.18);
        --soc-text: #f8fafc;
        --soc-muted: #94a3b8;
        --soc-accent: #38bdf8;
        --soc-success: #22c55e;
        --soc-warning: #f59e0b;
        --soc-danger: #ef4444;
    }

    /* Responsive typography shared by every DNS Console solution page. */
    .stApp h1,
    [data-testid="stHeadingWithActionElements"] h1 {
        font-size: clamp(1.55rem, 2.25vw, 2.15rem) !important;
        line-height: 1.16 !important;
        overflow-wrap: anywhere;
    }

    div[data-testid="stButton"] > button,
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        min-height: 2.55rem;
        padding: .5rem .72rem;
        font-size: clamp(.76rem, .88vw, .92rem) !important;
        line-height: 1.22 !important;
        white-space: normal !important;
        overflow-wrap: anywhere;
        text-align: center;
    }

    div[data-testid="stMetric"] {
        min-width: 0;
        min-height: 0 !important;
        height: 100% !important;
        overflow: visible;
    }

    div[data-testid="stMetricLabel"] p {
        font-size: clamp(.69rem, .78vw, .82rem) !important;
        line-height: 1.22 !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: clamp(.94rem, 1.26vw, 1.34rem) !important;
        line-height: 1.16 !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }

    .soc-hero {
        position: relative;
        overflow: hidden;
        padding: 22px 24px;
        margin-bottom: 18px;
        border-radius: 18px;
        background:
            radial-gradient(circle at 95% 0%, rgba(56,189,248,.24), transparent 34%),
            radial-gradient(circle at 5% 100%, rgba(34,197,94,.14), transparent 30%),
            linear-gradient(135deg, #111827 0%, #0b1220 100%);
        border: 1px solid var(--soc-border);
        box-shadow: 0 14px 35px rgba(0,0,0,.26);
    }

    .soc-hero-title {
        font-size: clamp(1.25rem, 1.8vw, 1.55rem);
        font-weight: 760;
        color: var(--soc-text);
        letter-spacing: .01em;
    }

    .soc-hero-subtitle {
        margin-top: 6px;
        color: var(--soc-muted);
        font-size: clamp(.78rem, .92vw, .96rem);
        max-width: 920px;
    }

    .soc-status-row {
        display: flex;
        align-items: center;
        gap: 9px;
        margin-top: 15px;
        color: #cbd5e1;
        font-size: .88rem;
    }

    .soc-dot {
        width: 10px;
        height: 10px;
        border-radius: 999px;
        background: var(--soc-success);
        box-shadow: 0 0 0 0 rgba(34,197,94,.6);
        animation: socPulse 1.7s infinite;
    }

    .soc-dot.waiting {
        background: var(--soc-warning);
        box-shadow: 0 0 0 0 rgba(245,158,11,.6);
    }

    .soc-dot.error {
        background: var(--soc-danger);
        animation: none;
    }

    @keyframes socPulse {
        0% { box-shadow: 0 0 0 0 currentColor; opacity: 1; }
        70% { box-shadow: 0 0 0 11px rgba(0,0,0,0); opacity: .82; }
        100% { box-shadow: 0 0 0 0 rgba(0,0,0,0); opacity: 1; }
    }

    div[data-testid="stMetric"] {
        background: var(--soc-card);
        border: 1px solid var(--soc-border);
        border-radius: 15px;
        padding: .62rem .75rem;
        box-shadow: 0 8px 20px rgba(0,0,0,.18);
    }

    div[data-testid="stMetric"] label {
        color: #a8b3c5 !important;
    }

    .soc-file {
        padding: 10px 13px;
        border-radius: 10px;
        background: rgba(15,23,42,.78);
        border: 1px solid rgba(148,163,184,.13);
        color: #cbd5e1;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-size: .82rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .soc-section-label {
        color: #7dd3fc;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }


    .soc-metric-grid {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        align-items: stretch;
        gap: 12px;
        margin: 10px 0 4px 0;
    }

    .soc-visual-metric {
        min-height: 0;
        height: 100%;
        padding: .72rem .68rem;
        border-radius: 16px;
        background: var(--soc-card);
        border: 1px solid var(--soc-border);
        box-shadow: 0 8px 20px rgba(0,0,0,.18);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        text-align: center;
    }

    .soc-visual-label {
        color: #a8b3c5;
        font-size: clamp(.68rem, .78vw, .82rem);
        font-weight: 650;
        line-height: 1.25;
    }

    .soc-visual-value {
        color: #f8fafc;
        font-size: clamp(1rem, 1.35vw, 1.48rem);
        font-weight: 760;
        line-height: 1.1;
        padding: .55rem 0 .45rem 0;
    }

    .soc-visual-note {
        color: #7f8da3;
        font-size: clamp(.62rem, .66vw, .69rem);
        line-height: 1.25;
        overflow-wrap: anywhere;
    }

    .soc-progress-ring {
        --progress: 0;
        position: relative;
        width: 92px;
        height: 92px;
        margin: 4px 0;
        border-radius: 50%;
        background:
            conic-gradient(
                #22d3ee calc(var(--progress) * 1%),
                rgba(71,85,105,.34) 0
            );
        box-shadow:
            0 0 18px rgba(34,211,238,.18),
            inset 0 0 12px rgba(0,0,0,.38);
        animation: socRingGlow 1.5s ease-in-out infinite alternate;
    }

    .soc-progress-ring::before {
        content: "";
        position: absolute;
        inset: 9px;
        border-radius: 50%;
        background: #0f1724;
        border: 1px solid rgba(34,211,238,.20);
    }

    .soc-progress-ring span {
        position: absolute;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #67e8f9;
        font-size: 1.22rem;
        font-weight: 760;
        z-index: 2;
        text-shadow: 0 0 12px rgba(34,211,238,.55);
    }

    @keyframes socRingGlow {
        from { filter: brightness(.92); transform: scale(.99); }
        to   { filter: brightness(1.15); transform: scale(1.01); }
    }

    @media (max-width: 1200px) {
        .soc-metric-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }

    @media (max-width: 760px) {
        .soc-metric-grid { grid-template-columns: 1fr; }
    }

    .scoring-control-grid {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        align-items: stretch;
        gap: 12px;
        margin: 12px 0 18px 0;
    }

    .scoring-control-card {
        min-height: 0;
        height: 100%;
        padding: .72rem .68rem;
        border-radius: 16px;
        background: var(--soc-card);
        border: 1px solid var(--soc-border);
        box-shadow: 0 8px 20px rgba(0,0,0,.18);
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }

    .scoring-card-number {
        color: #38bdf8;
        font-size: clamp(.6rem, .66vw, .70rem);
        font-weight: 800;
        letter-spacing: .12em;
        text-transform: uppercase;
    }

    .scoring-card-title {
        margin-top: 5px;
        color: #f8fafc;
        font-size: clamp(.7rem, .82vw, .88rem);
        font-weight: 720;
        line-height: 1.25;
    }

    .scoring-card-headline {
        margin-top: 8px;
        color: #67e8f9;
        font-size: clamp(.66rem, .74vw, .78rem);
        font-weight: 720;
    }

    .scoring-card-detail {
        margin-top: 5px;
        color: #94a3b8;
        font-size: clamp(.6rem, .64vw, .68rem);
        line-height: 1.35;
        overflow-wrap: anywhere;
    }

    .scoring-ring.success {
        background: conic-gradient(#22c55e calc(var(--progress) * 1%), rgba(71,85,105,.34) 0);
    }
    .scoring-ring.warning {
        background: conic-gradient(#f59e0b calc(var(--progress) * 1%), rgba(71,85,105,.34) 0);
    }
    .scoring-ring.error {
        background: conic-gradient(#ef4444 calc(var(--progress) * 1%), rgba(71,85,105,.34) 0);
    }

    .scoring-window {
        border: 1px solid rgba(148,163,184,.18);
        border-radius: 15px;
        padding: 14px 16px;
        background: linear-gradient(135deg, #111827 0%, #0b1220 100%);
        margin-bottom: 14px;
    }

    @media (max-width: 1250px) {
        .scoring-control-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
    @media (max-width: 760px) {
        .scoring-control-grid { grid-template-columns: 1fr; }
    }

    .block-container {
        padding-top: 1.8rem;
    }

    /* Shared animated percentage bar used by every project workflow. */
    .dns-animated-progress {
        width: 100%;
        margin: .48rem 0 .75rem;
    }

    .dns-animated-progress-track {
        position: relative;
        height: 1.48rem;
        overflow: hidden;
        border: 1px solid rgba(125, 211, 252, .46);
        border-radius: 999px;
        background: rgba(7, 18, 31, .92);
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, .46), 0 0 16px rgba(56, 189, 248, .10);
    }

    .dns-animated-progress-fill {
        position: absolute;
        inset: 0 auto 0 0;
        border-radius: inherit;
        background: linear-gradient(90deg, #2563eb 0%, #06b6d4 35%, #22c55e 68%, #f59e0b 100%);
        background-size: 220% 100%;
        transition: width .35s ease;
        box-shadow: 0 0 18px rgba(34, 211, 238, .42);
    }

    .dns-animated-progress.active .dns-animated-progress-fill {
        animation: dnsProgressFlow 1.45s linear infinite;
    }

    .dns-animated-progress-percent {
        position: absolute;
        inset: 0;
        z-index: 2;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #f8fafc;
        font-size: clamp(.68rem, .78vw, .8rem);
        font-weight: 850;
        letter-spacing: .025em;
        text-shadow: 0 1px 4px rgba(0, 0, 0, .9);
    }

    .dns-animated-progress-label {
        margin-top: .34rem;
        color: var(--soc-muted);
        font-size: clamp(.68rem, .78vw, .82rem);
        line-height: 1.3;
        overflow-wrap: anywhere;
    }

    @keyframes dnsProgressFlow {
        0% { background-position: 100% 0; filter: brightness(.95); }
        50% { filter: brightness(1.18); }
        100% { background-position: -120% 0; filter: brightness(.95); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_model_config():
    if not MODEL_CONFIG_PATH.exists():
        return None

    try:
        with MODEL_CONFIG_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


# BEGIN DNS PUBLIC ALERT ID HELPERS

import hashlib

def _stable_autoencoder_public_id_gui(timestamp, src_ip: str) -> str:
    try:
        normalized = pd.Timestamp(timestamp).isoformat()
    except Exception:
        normalized = str(timestamp or "")
    basis = f"autoencoder|{normalized}|{str(src_ip or '')}"
    return "AE-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20].upper()


def _public_alert_id_from_row(row) -> str:
    try:
        raw = parse_json(row.get("raw_json"), {})
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}

    candidates = [raw.get("alert_uid"), raw.get("id"), row.get("alert_uid")]
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value.startswith(("CATF-", "HEUR-", "AE-")):
            return value

    methods = parse_json(row.get("detection_methods_json"), {})
    if not isinstance(methods, dict):
        methods = {}
    alert_type = str(row.get("alert_type") or raw.get("alert_type") or "")
    detection_type = str(raw.get("detection_type") or "")
    is_autoencoder = bool(
        alert_type == "autoencoder_anomaly"
        or detection_type == "deep_learning_autoencoder"
        or methods.get("autoencoder")
    )
    has_heuristic = any(
        bool(methods.get(name))
        for name in (
            "heuristic_dga",
            "heuristic_tunnel",
            "heuristic_fastflux",
            "heuristic_resolver_failure",
            "heuristic_rogue_resolver",
        )
    )
    if is_autoencoder and not has_heuristic:
        return _stable_autoencoder_public_id_gui(row.get("timestamp"), row.get("src_ip"))

    for candidate in candidates:
        value = str(candidate or "").strip()
        if value:
            return value
    return f"DB-{row.get('id')}"


def _ensure_public_alert_ids(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    result = frame.copy()
    result["public_alert_id"] = result.apply(_public_alert_id_from_row, axis=1)
    return result

# END DNS PUBLIC ALERT ID HELPERS

@st.cache_data(ttl=10)
def load_alerts():
    frame = read_sql(
        """
        SELECT *
        FROM alerts
        ORDER BY timestamp DESC, id DESC
        """
    )
    return _ensure_public_alert_ids(frame)

def detection_type_from_json(value: str) -> str:
    """Return a readable comma-separated list of enabled detection methods."""
    methods = parse_json(value, {})
    if not isinstance(methods, dict):
        return "none"
    active = [str(key) for key, enabled in methods.items() if bool(enabled)]
    return ", ".join(active) if active else "none"


def human_duration(minutes: float) -> str:
    """Format a duration supplied in minutes for the Zeek statistics cards."""
    try:
        minutes = float(minutes or 0)
    except (TypeError, ValueError):
        minutes = 0.0

    if minutes < 0:
        minutes = 0.0
    if minutes < 60:
        return f"{minutes:,.0f} min"

    hours = minutes / 60.0
    if hours < 48:
        return f"{hours:,.2f} h"

    return f"{hours / 24.0:,.2f} days"

def render_zeek_stats_cards(stats: dict, status: dict, running: bool):
    """Render Zeek DNS preparation metrics with circular loading indicators."""
    total_records = stats.get(
        "total_dns_records",
        status.get("records_inserted", 0),
    )
    source_file_count = stats.get("source_file_count", 0)
    total_source_ips = stats.get(
        "total_source_ips",
        status.get("unique_src_ips_seen", 0),
    )
    average_queries = stats.get("average_queries_per_source_ip", 0)
    aggregate_minutes = stats.get(
        "aggregate_source_ip_coverage_minutes",
        stats.get("total_capture_minutes", 0),
    )
    actual_minutes = stats.get(
        "actual_capture_duration_minutes",
        float(stats.get("actual_wall_clock_span_seconds", 0) or 0) / 60,
    )

    progress = max(
        0,
        min(int(status.get("progress_percent", 0) or 0), 100),
    )
    stage = html.escape(str(status.get("stage", "Preparing Zeek data")))

    cards = [
        (
            "Total DNS logs",
            f"{int(source_file_count or 0):,}",
            "Unique Zeek DNS source files discovered",
        ),
        (
            "Total unique DNS records",
            f"{int(total_records or 0):,}",
            "Deduplicated DNS transactions stored by unique fingerprint",
        ),
        (
            "Total source IP addresses",
            f"{int(total_source_ips or 0):,}",
            "Unique DNS clients",
        ),
        (
            "DNS queries / source IP",
            f"{float(average_queries or 0):,.2f}",
            "Average across source IPs",
        ),
        (
            "Aggregate source-IP coverage",
            human_duration(aggregate_minutes),
            "Unique source-IP windows × 5 minutes",
        ),
        (
            "Actual capture duration",
            human_duration(actual_minutes),
            "Latest timestamp − earliest timestamp",
        ),
    ]

    fragments = ['<div class="soc-metric-grid">']
    for label, value, note in cards:
        if running:
            center = (
                f'<div class="soc-progress-ring" style="--progress:{progress}">'
                f'<span>{progress}%</span></div>'
            )
            footer = stage
        else:
            center = f'<div class="soc-visual-value">{html.escape(str(value))}</div>'
            footer = html.escape(str(note))

        fragments.append(
            '<div class="soc-visual-metric">'
            f'<div class="soc-visual-label">{html.escape(label)}</div>'
            f'{center}'
            f'<div class="soc-visual-note">{footer}</div>'
            '</div>'
        )

    fragments.append('</div>')
    st.markdown(''.join(fragments), unsafe_allow_html=True)

def make_source_ip_timestamps_readable(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert Zeek epoch timestamps to readable UTC values for the table."""
    result = frame.copy()
    for column in ("first_ts", "last_ts"):
        if column not in result.columns:
            continue

        original = result[column].astype(str)
        numeric = pd.to_numeric(result[column], errors="coerce")
        converted = pd.to_datetime(
            numeric,
            unit="s",
            utc=True,
            errors="coerce",
        ).dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        result[column] = converted.where(converted.notna(), original)

    return result


def render_learning_log_window(log_text: str, running: bool) -> None:
    """Show the command output with a permanent vertical scroll bar."""
    safe_log = html.escape(log_text or "No learning output is available yet.")
    scroll_target = "panel.scrollHeight" if running else "0"
    components.html(
        f"""
        <!doctype html>
        <html>
        <head>
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                background: transparent;
                color: #d7e2f0;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco,
                             Consolas, "Liberation Mono", monospace;
            }}
            #learning-log-panel {{
                height: 430px;
                overflow-y: scroll;
                overflow-x: auto;
                box-sizing: border-box;
                padding: 14px 16px;
                border: 1px solid rgba(148,163,184,.22);
                border-radius: 12px;
                background: #060b13;
                scrollbar-width: auto;
                scrollbar-color: #22d3ee #111827;
            }}
            #learning-log-panel::-webkit-scrollbar {{ width: 13px; }}
            #learning-log-panel::-webkit-scrollbar-track {{ background: #111827; }}
            #learning-log-panel::-webkit-scrollbar-thumb {{
                background: #22d3ee;
                border-radius: 8px;
                border: 3px solid #111827;
            }}
            pre {{
                margin: 0;
                white-space: pre;
                line-height: 1.43;
                font-size: 12.5px;
            }}
        </style>
        </head>
        <body>
            <div id="learning-log-panel"><pre>{safe_log}</pre></div>
            <script>
                const panel = document.getElementById("learning-log-panel");
                panel.scrollTop = {scroll_target};
            </script>
        </body>
        </html>
        """,
        height=452,
        scrolling=False,
    )
def _render_training_stage_cards(state: dict, merge_running: bool, feature_running: bool, ready: bool) -> None:
    uploaded_ok = state.get("phase") in {
        "uploaded", "merging", "merged_verified", "extracting_features", "features_ready"
    }
    merge_ok = bool(state.get("merge_verified"))
    features_ok = bool(state.get("features_extracted"))

    def card(label: str, value: str, detail: str = "", active: bool = False) -> str:
        active_class = " is-active" if active else ""
        detail_html = (
            f'<div class="dns-stage-detail">{html.escape(detail)}</div>'
            if detail else '<div class="dns-stage-detail">&nbsp;</div>'
        )
        return (
            f'<div class="dns-stage-card{active_class}">'
            f'<div class="dns-stage-label">{html.escape(label)}</div>'
            f'<div class="dns-stage-value">{html.escape(value)}</div>'
            f'{detail_html}</div>'
        )

    upload_value = "COMPLETE" if uploaded_ok else "WAITING"
    merge_value = "COMPLETE" if merge_ok else ("RUNNING" if merge_running else "WAITING")
    feature_value = "COMPLETE" if features_ok else ("RUNNING" if feature_running else "WAITING")
    gate_value = "UNLOCKED" if ready else "LOCKED"
    uploaded_detail = (
        f"{int(state.get('accepted_dns_files', 0) or 0):,} files · "
        f"{int(state.get('normalized_json_records', 0) or 0):,} records"
    )

    st.markdown(
        """
        <style>
        .dns-stage-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));align-items:stretch;gap:12px;margin:12px 0 16px}
        .dns-stage-card{min-height:0;height:100%;padding:.78rem .9rem;border:1px solid rgba(148,163,184,.22);border-radius:14px;background:#151d2b;display:flex;flex-direction:column;justify-content:flex-start;box-sizing:border-box;font-family:inherit}
        .dns-stage-card.is-active{border-color:rgba(34,211,238,.65);box-shadow:inset 0 -3px 0 #22d3ee}
        .dns-stage-label{color:#a9b7c9;font-size:clamp(.7rem,.82vw,.88rem);font-weight:500;line-height:1.25;overflow-wrap:anywhere}
        .dns-stage-value{color:#f8fafc;font-size:clamp(1.05rem,1.5vw,1.72rem);font-weight:750;line-height:1.08;letter-spacing:.01em;margin:10px 0 8px;overflow-wrap:anywhere}
        .dns-stage-detail{color:#6ee7b7;font-size:clamp(.64rem,.72vw,.78rem);font-weight:500;line-height:1.25;overflow-wrap:anywhere}
        @media(max-width:1000px){.dns-stage-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
        @media(max-width:620px){.dns-stage-grid{grid-template-columns:1fr}}
        </style>
        """,
        unsafe_allow_html=True,
    )

    block = (
        '<div class="dns-stage-grid">'
        + card("1 · JSON / JSONL upload", upload_value, uploaded_detail, state.get("phase") == "empty")
        + card("2 · Merge & verify", merge_value, "", merge_running)
        + card("3 · Feature extraction", feature_value, "", feature_running)
        + card("ML learning gate", gate_value, "", ready)
        + "</div>"
    )
    st.markdown(block, unsafe_allow_html=True)

def render_zeek_log_manager() -> bool:
    state = training_upload_manager.reconcile_workflow_state()
    phase = state.get("phase", "empty")
    merge_running = training_upload_manager.merge_is_running()
    feature_running = training_upload_manager.is_feature_extraction_running()
    busy = merge_running or feature_running
    ready = training_upload_manager.learning_input_ready()
    frozen = training_lifecycle.zeek_manager_frozen()

    merge_status = read_merge_status()
    stats = read_zeek_stats()
    input_spec = training_upload_manager.input_spec()

    labels = {
        "empty": "Waiting for JSON / JSONL upload",
        "uploaded": "Upload complete",
        "merging": "Merging & verifying",
        "merged_verified": "Merge verified",
        "extracting_features": "Feature extraction",
        "features_ready": "Ready for ML",
        "error": "Preparation error",
    }
    dot = "error" if phase == "error" else ("waiting" if busy else "")

    st.markdown(
        f"""<div class="soc-hero">
<div class="soc-section-label">Training-data preparation</div>
<div class="soc-hero-title">Zeek DNS Log Manager</div>
<div class="soc-hero-subtitle">Upload Zeek DNS JSON / JSONL files or a JSON dns.log, merge and verify all uploaded files, then calculate the shared ML feature dataset once. Autoencoder and Isolation Forest consume the same prepared feature file and do not repeat feature engineering.</div>
<div class="soc-status-row"><span class="soc-dot {dot}"></span><span>{labels.get(phase, phase)} — {state.get("message", "")}</span></div>
</div>""",
        unsafe_allow_html=True,
    )

    _render_training_stage_cards(state, merge_running, feature_running, ready)

    if frozen:
        st.warning(
            "Zeek DNS Log Manager is frozen because an ML training session has started. "
            "It remains read-only until **Purge All** is used under either ML section."
        )

    with st.expander("Training input/output specification", expanded=False):
        st.markdown(
            f"""
**Input policy:** `/opt/dns-ml-gui/config/training_input.json`  
Allowed: `{", ".join(input_spec.get("allowed_extensions", [".json", ".jsonl", ".log"]))}`
Archives: `disabled`  
All uploaded files merged: `enabled`

**Lifecycle documentation:** `/opt/dns-ml-gui/docs/ML_TRAINING_LIFECYCLE.md`

**Upload originals:** `/data/dns-ml/training_upload/incoming/`

**Normalized merge inputs:** `/data/dns-ml/training_upload/files/`

**Merged DNS:** `/data/dns-ml/zeek/merged_dns.jsonl`

**Shared feature dataset used by BOTH models:**  
`/data/dns-ml/features/dns_training_features.csv`
"""
        )

    cycle = int(state.get("cycle_id", 1) or 1)
    step1 = phase == "empty" and not busy and not frozen

    files = st.file_uploader(
        "1 · Select all Zeek DNS JSON / JSONL or JSON dns.log training files",
        type=["json", "jsonl", "log"],
        accept_multiple_files=True,
        disabled=not step1,
        key=f"zeek_training_json_upload_{cycle}",
        help=(
            "Select the complete training batch. After pressing Upload, "
            "server-side validation/staging progress remains visible until completion."
        ),
    )

    b1, b2, b3, b4 = st.columns(4)

    with b1:
        if st.button(
            "1 · Upload DNS log batch",
            use_container_width=True,
            disabled=(not step1 or not files),
            type="primary" if step1 else "secondary",
        ):
            progress = animated_progress(0, "Preparing uploaded files...")
            status_box = st.status("Uploading / validating training batch", expanded=True)

            def _upload_progress(done, total, filename, stage):
                total = max(int(total or 1), 1)
                done = max(0, min(int(done or 0), total))
                pct = done / total
                progress.progress(pct, text=f"{int(pct * 100)}% — {stage}: {filename}")
                status_box.write(f"{done}/{total} · {stage} · {filename}")

            result = training_upload_manager.save_uploaded_batch(
                files,
                progress_callback=_upload_progress,
            )
            if result.get("error"):
                status_box.update(label="Upload / validation failed", state="error")
                st.error(result["error"])
            else:
                progress.progress(1.0, text="100% — upload / validation complete")
                status_box.update(label="Upload / validation complete", state="complete")
                st.success(
                    f"Validated {result['original_files']} files containing "
                    f"{result['records']:,} JSON records."
                )
                st.rerun()

    step2 = phase == "uploaded" and not busy and not frozen
    with b2:
        if st.button(
            "2 · Merge all JSON / JSONL",
            use_container_width=True,
            disabled=not step2,
            type="primary" if step2 else "secondary",
        ):
            result = training_upload_manager.start_uploaded_merge()
            if result.get("error"):
                st.error(result["error"])
            else:
                st.success(f"Merge started for all {result['files']} uploaded files.")
                st.rerun()

    step3 = phase == "merged_verified" and not busy and not frozen
    with b3:
        if st.button(
            "3 · Analyze & extract features",
            use_container_width=True,
            disabled=not step3,
            type="primary" if step3 else "secondary",
        ):
            result = training_upload_manager.start_feature_extraction()
            if result.get("error"):
                st.error(result["error"])
            else:
                st.success(f"Feature extraction started with PID {result['pid']}.")
                st.rerun()

    step4 = (
        phase in {"features_ready", "error"}
        and not busy
        and not frozen
        and not training_lifecycle.any_ml_running()
    )
    with b4:
        if st.button(
            "4 · Clear training data",
            use_container_width=True,
            disabled=not step4,
            type="primary" if step4 else "secondary",
            help=(
                "Clears uploaded/prepared training data, extracted features and ML "
                "result views. Published models are preserved. After ML starts, "
                "use Purge All instead."
            ),
        ):
            result = training_lifecycle.clear_training_data()
            if result.get("error"):
                st.error(result["error"])
            else:
                st.success(result.get("message", "Training data cleared."))
                st.rerun()

    staged = training_upload_manager.uploaded_training_files()
    with st.expander(f"Uploaded/staged JSON / JSONL files ({len(staged)})"):
        if staged:
            st.dataframe(
                pd.DataFrame(
                    {
                        "file": [p.name for p in staged],
                        "size_bytes": [p.stat().st_size for p in staged],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No training upload is staged.")

    if phase in {"merging", "merged_verified", "extracting_features", "features_ready", "error"}:
        render_zeek_stats_cards(stats, merge_status, merge_running)

    if merge_running:
        pct = max(0, min(int(merge_status.get("progress_percent", 0) or 0), 100))
        animated_progress(pct, str(merge_status.get("stage", "Merging")))
    with st.expander("Zeek merge command output", expanded=merge_running):
        st.code(read_zeek_log_tail(250), language="text")

    feature_status = training_upload_manager.read_feature_status()
    if feature_running:
        pct = max(0, min(int(feature_status.get("progress_percent", 0) or 0), 100))
        animated_progress(pct, str(feature_status.get("stage", "Feature extraction")))

    if phase in {"extracting_features", "features_ready"}:
        with st.expander("Feature extraction command output", expanded=feature_running):
            st.code(training_upload_manager.read_feature_log_tail(), language="text")

    summary = training_upload_manager.read_feature_summary()
    if summary.get("ready_for_ml"):
        st.subheader("Shared ML feature dataset")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Training windows", f"{int(summary.get('training_windows', 0)):,}")
        f2.metric("Feature count", int(summary.get("feature_count", 0)))
        f3.metric("Source IPs", f"{int(summary.get('unique_source_ips', 0)):,}")
        f4.metric("Usable DNS events", f"{int(summary.get('usable_dns_events', 0)):,}")

        st.caption(
            "Feature engineering is performed here once. Both ML models consume "
            "/data/dns-ml/features/dns_training_features.csv directly."
        )

        feature_path = training_upload_manager.FEATURE_CSV
        if feature_path.exists():
            st.markdown("#### Feature dataset preview — first 20 rows only")
            try:
                preview = pd.read_csv(feature_path, nrows=20)
                st.dataframe(
                    preview,
                    use_container_width=True,
                    hide_index=True,
                    height=470,
                )
            except Exception as exc:
                st.warning(f"Could not read feature preview: {exc}")

    if ZEEK_SOURCE_IP_STATS_PATH.exists():
        with st.expander("DNS queries and capture windows by source IP"):
            try:
                frame = make_source_ip_timestamps_readable(
                    pd.read_csv(ZEEK_SOURCE_IP_STATS_PATH)
                )
                st.dataframe(frame, use_container_width=True, hide_index=True, height=360)
            except Exception as exc:
                st.warning(f"Could not read source-IP statistics: {exc}")

    if phase == "error":
        st.error(state.get("error") or state.get("message"))
    if not ready:
        st.warning(training_upload_manager.learning_gate_message())

    if busy:
        time.sleep(1.5)
        st.rerun()

    return training_upload_manager.learning_input_ready()






def show_learning_results():
    st.subheader("Polished learning result")

    result = read_learning_result()
    if result:
        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Training windows", f"{int(result.get('training_windows', 0)):,}")
        r2.metric("Feature count", result.get("feature_count", "N/A"))
        r3.metric(
            "Best validation loss",
            f"{float(result.get('best_val_loss', 0)):.6f}",
        )
        r4.metric(
            "Threshold",
            f"{float(result.get('threshold', 0)):.6f}",
        )
        r5.metric("Epochs trained", result.get("epochs_trained", "N/A"))

        with st.expander("Saved learning artifacts"):
            st.json(
                {
                    "training_source": result.get("training_files"),
                    "model_path": result.get("model_path"),
                    "scaler_path": result.get("scaler_path"),
                    "config_path": result.get("config_path"),
                    "history_path": str(LEARNING_HISTORY_PATH),
                    "reconstruction_errors_path": str(
                        LEARNING_RECONSTRUCTION_PATH
                    ),
                    "completed_at_utc": result.get("completed_at_utc"),
                }
            )
    else:
        st.info("No completed learning result is available yet.")

    history_df = load_history_df()

    left, right = st.columns(2)
    with left:
        st.subheader("Reconstruction Quality Against Epoch")
        if not history_df.empty and "epoch" in history_df:
            quality_cols = [
                column
                for column in (
                    "train_reconstruction_quality_pct",
                    "val_reconstruction_quality_pct",
                )
                if column in history_df.columns
            ]
            if quality_cols:
                st.line_chart(
                    history_df[["epoch"] + quality_cols].set_index("epoch")
                )
                st.caption(
                    "This is an Autoencoder reconstruction-quality proxy, "
                    "not supervised classification accuracy."
                )
            else:
                st.info("No reconstruction-quality columns were found.")
        else:
            st.info("No epoch history is available.")

    with right:
        st.subheader("Training and Validation Loss")
        if not history_df.empty and "epoch" in history_df:
            loss_cols = [
                column
                for column in ("loss", "val_loss")
                if column in history_df.columns
            ]
            if loss_cols:
                st.line_chart(
                    history_df[["epoch"] + loss_cols].set_index("epoch")
                )
            else:
                st.info("No loss columns were found.")
        else:
            st.info("No epoch history is available.")

    st.subheader("Reconstruction Error Distribution")
    reconstruction_df = load_reconstruction_df()

    if (
        not reconstruction_df.empty
        and "reconstruction_error" in reconstruction_df.columns
    ):
        chart_source = reconstruction_df.copy()
        chart_source["error_bin"] = pd.cut(
            chart_source["reconstruction_error"],
            bins=35,
        ).astype(str)

        histogram = (
            chart_source.groupby(["error_bin", "split"], observed=False)
            .size()
            .unstack(fill_value=0)
        )
        st.bar_chart(histogram)

        with st.expander("Raw reconstruction-error data"):
            st.dataframe(
                reconstruction_df.head(2000),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("No reconstruction-error distribution is available.")


def _clear_learning_log_snapshot() -> None:
    st.session_state.pop("learning_final_log_snapshot", None)
    st.session_state.pop("learning_final_log_signature", None)


def _stable_completed_learning_log() -> str:
    """Keep the completed summary unchanged across button reruns/read errors."""
    signature = learning_log_signature()
    cached_text = st.session_state.get("learning_final_log_snapshot")
    cached_signature = st.session_state.get("learning_final_log_signature")

    if cached_text and cached_signature == signature:
        return cached_text

    current_text = read_complete_log()
    read_failed = current_text.startswith(
        (
            "Unable to read the completed learning log:",
            "Unable to read the live learning log:",
        )
    )

    if read_failed and cached_text:
        return cached_text

    if not read_failed:
        st.session_state["learning_final_log_snapshot"] = current_text
        st.session_state["learning_final_log_signature"] = signature

    return current_text


def _render_autoencoder_learning_io() -> None:
    with st.expander("Autoencoder · input, outputs & learning configuration", expanded=False):
        st.markdown("**Input — precomputed once by Zeek DNS Log Manager**")
        st.code("/data/dns-ml/features/dns_training_features.csv", language="text")

        st.markdown("**Outputs**")
        st.code(
            """/data/dns-ml/models/dns_autoencoder.keras
/data/dns-ml/models/dns_scaler.joblib
/data/dns-ml/models/dns_autoencoder_config.json
/data/dns-ml/runs/latest_training_status.json
/data/dns-ml/runs/latest_training.log
/data/dns-ml/runs/latest_training_history.csv
/data/dns-ml/runs/latest_reconstruction_errors.csv
/data/dns-ml/runs/latest_training_result.json""",
            language="text",
        )

        rows = [
            ("DNS_TRAINING_EPOCHS", os.getenv("DNS_TRAINING_EPOCHS", "50"), "train_dns_autoencoder.py"),
            ("DNS_BATCH_SIZE", os.getenv("DNS_BATCH_SIZE", "256"), "train_dns_autoencoder.py"),
            ("DNS_THRESHOLD_QUANTILE", os.getenv("DNS_THRESHOLD_QUANTILE", "0.99"), "train_dns_autoencoder.py"),
            ("DNS_MIN_WINDOWS_REQUIRED", os.getenv("DNS_MIN_WINDOWS_REQUIRED", "50"), "train_dns_autoencoder.py"),
            ("DNS_EXPECTED_FEATURE_COUNT", os.getenv("DNS_EXPECTED_FEATURE_COUNT", "36"), "train_dns_autoencoder.py"),
            ("DNS_RANDOM_STATE", os.getenv("DNS_RANDOM_STATE", "42"), "train_dns_autoencoder.py"),
            ("DNS_KERAS_VERBOSE", os.getenv("DNS_KERAS_VERBOSE", "2"), "train_dns_autoencoder.py"),
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=["Parameter", "Current/default", "Defined/consumed in"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Persisted learned configuration: "
            "/data/dns-ml/models/dns_autoencoder_config.json"
        )

def render_learning_control(dataset_ready: bool):
    st.divider()
    st.markdown(
        """
        <div class="soc-hero">
            <div class="soc-section-label">Model learning</div>
            <div class="soc-hero-title">Autoencoder Learning Control</div>
            <div class="soc-hero-subtitle">
                Train the Autoencoder directly from the shared precomputed feature
                dataset. Feature engineering is not repeated here.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    running = is_learning_running()
    status = read_learning_status()
    state_name = str(status.get("state", "not_started")).strip().lower()
    can_stop = running or state_name in {"starting", "running", "training", "busy"}

    c1, c2, c3, c4 = st.columns([1.15, 1.15, 1.0, 1.15])

    with c1:
        if st.button(
            "▶ Start Learning",
            key="start_autoencoder_learning",
            use_container_width=True,
            disabled=running or not dataset_ready,
        ):
            result = start_learning()
            if result.get("error"):
                st.error(result["error"])
            else:
                st.success(f"Autoencoder learning started with PID {result['pid']}.")
                st.rerun()

    with c2:
        if st.button(
            "■ Stop Learning",
            key="stop_autoencoder_learning",
            use_container_width=True,
            disabled=not can_stop,
        ):
            result = stop_learning()
            if result.get("error"):
                st.error(result["error"])
            else:
                st.warning(result.get("message", "Autoencoder learning stopped and reset."))
                st.rerun()

    with c3:
        if st.button("↻ Refresh", key="refresh_autoencoder_learning", use_container_width=True):
            st.rerun()

    with c4:
        if st.button(
            "🧹 Purge All",
            key="purge_all_from_autoencoder",
            use_container_width=True,
            disabled=training_lifecycle.any_ml_running(),
            help=(
                "Global reset: purges BOTH ML model families, both learning result "
                "sets, all uploaded/prepared training data and features, then "
                "unlocks Zeek DNS Log Manager."
            ),
        ):
            result = training_lifecycle.purge_all()
            if result.get("error"):
                st.error(result["error"])
            else:
                st.warning(result["message"])
                st.rerun()

    _render_autoencoder_learning_io()

    if not dataset_ready:
        st.warning(
            "Start Learning is disabled until the Zeek DNS Log Manager has "
            "successfully produced the shared feature dataset."
        )

    status = read_learning_status()
    running = is_learning_running()
    progress = max(0, min(int(status.get("progress_percent", 0) or 0), 100))

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("State", status.get("state", "not_started"))
    s2.metric("Stage", status.get("stage", "Not started"))
    s3.metric("Progress", f"{progress}%")
    s4.metric("Execution", "CPU")

    animated_progress(progress, str(status.get("message", "")))

    if status.get("error"):
        st.error(status["error"])
    elif str(status.get("state", "")).lower() == "finished":
        st.success(status.get("message", "Learning completed."))
    elif running:
        st.info(status.get("message", "Learning is running."))

    with st.expander("Live Autoencoder learning command output", expanded=running):
        log_text = read_log_tail(350)
        if "render_learning_log_window" in globals():
            render_learning_log_window(log_text, running=running)
        else:
            st.code(log_text, language="text")

    if running:
        time.sleep(1.5)
        st.rerun()

    show_learning_results()




# BEGIN DNS ISOLATION FOREST LEARNING

def _clear_isolation_log_snapshot() -> None:
    st.session_state.pop("isolation_final_log_snapshot", None)
    st.session_state.pop("isolation_final_log_signature", None)


def _stable_completed_isolation_log() -> str:
    signature = isolation_log_signature()
    cached_text = st.session_state.get("isolation_final_log_snapshot")
    cached_signature = st.session_state.get("isolation_final_log_signature")
    if cached_text and cached_signature == signature:
        return cached_text

    current_text = read_complete_isolation_log()
    read_failed = current_text.startswith(
        (
            "Unable to read the completed Isolation Forest log:",
            "Unable to read the live Isolation Forest log:",
        )
    )
    if read_failed and cached_text:
        return cached_text
    if not read_failed:
        st.session_state["isolation_final_log_snapshot"] = current_text
        st.session_state["isolation_final_log_signature"] = signature
    return current_text


def _if_numeric(value, digits: int = 6) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def _render_isolation_contour(points: pd.DataFrame, contour: pd.DataFrame) -> None:
    st.subheader("Anomaly score contour plot — 2D PCA projection")
    if points.empty or contour.empty:
        st.info("Complete Isolation Forest learning to generate the PCA decision surface.")
        return

    required_grid = {"pca_1", "pca_2", "anomaly_score"}
    required_points = {"pca_1", "pca_2", "anomaly_score", "predicted_label"}
    if not required_grid.issubset(contour.columns) or not required_points.issubset(points.columns):
        st.warning("The saved Isolation Forest contour data is incomplete.")
        return

    try:
        import plotly.graph_objects as go

        grid = contour.pivot_table(
            index="pca_2",
            columns="pca_1",
            values="anomaly_score",
            aggfunc="mean",
        ).sort_index().sort_index(axis=1)
        point_sample = points
        if len(point_sample) > 5000:
            point_sample = point_sample.sample(5000, random_state=42)

        figure = go.Figure()
        figure.add_trace(
            go.Contour(
                x=grid.columns.to_numpy(dtype=float),
                y=grid.index.to_numpy(dtype=float),
                z=grid.to_numpy(dtype=float),
                contours={"showlabels": True},
                colorbar={"title": "Anomaly score"},
                name="Decision surface",
                hovertemplate="PCA 1=%{x:.4f}<br>PCA 2=%{y:.4f}<br>Score=%{z:.6f}<extra></extra>",
            )
        )
        for label, frame in point_sample.groupby("predicted_label", dropna=False):
            figure.add_trace(
                go.Scattergl(
                    x=frame["pca_1"],
                    y=frame["pca_2"],
                    mode="markers",
                    name=str(label),
                    marker={"size": 5, "opacity": 0.55},
                    customdata=frame[["src_ip", "anomaly_score"]].to_numpy()
                    if "src_ip" in frame.columns
                    else None,
                    hovertemplate=(
                        "PCA 1=%{x:.4f}<br>PCA 2=%{y:.4f}"
                        "<br>Source=%{customdata[0]}<br>Score=%{customdata[1]:.6f}<extra></extra>"
                        if "src_ip" in frame.columns
                        else "PCA 1=%{x:.4f}<br>PCA 2=%{y:.4f}<extra></extra>"
                    ),
                )
            )
        figure.update_layout(
            height=620,
            margin={"l": 20, "r": 20, "t": 20, "b": 20},
            xaxis={"title": "PCA component 1", "autorange": True},
            yaxis={"title": "PCA component 2", "autorange": True},
            legend={"orientation": "h"},
        )
        st.plotly_chart(figure, use_container_width=True, config={"responsive": True})
    except Exception as exc:
        st.warning(f"Plotly contour rendering was unavailable; using the dynamic heatmap fallback: {exc}")
        grid_chart = (
            alt.Chart(contour)
            .mark_rect()
            .encode(
                x=alt.X("pca_1:Q", bin=alt.Bin(maxbins=70), title="PCA component 1", scale=alt.Scale(zero=False)),
                y=alt.Y("pca_2:Q", bin=alt.Bin(maxbins=70), title="PCA component 2", scale=alt.Scale(zero=False)),
                color=alt.Color("mean(anomaly_score):Q", title="Anomaly score"),
                tooltip=[
                    alt.Tooltip("pca_1:Q", format=".4f"),
                    alt.Tooltip("pca_2:Q", format=".4f"),
                    alt.Tooltip("mean(anomaly_score):Q", format=".6f"),
                ],
            )
            .properties(height=560)
        )
        point_chart = (
            alt.Chart(points.sample(min(len(points), 4000), random_state=42))
            .mark_circle(size=28, opacity=0.45)
            .encode(
                x=alt.X("pca_1:Q", scale=alt.Scale(zero=False)),
                y=alt.Y("pca_2:Q", scale=alt.Scale(zero=False)),
                shape="predicted_label:N",
                tooltip=["timestamp:N", "src_ip:N", alt.Tooltip("anomaly_score:Q", format=".6f")],
            )
        )
        st.altair_chart(grid_chart + point_chart, use_container_width=True)

    st.caption(
        "Both axes are derived from the current saved PCA data and use automatic ranges on every page refresh. "
        "The zero anomaly-score boundary separates the model's predicted normal and anomaly regions."
    )


def _render_isolation_shap(
    summary: pd.DataFrame,
    values: pd.DataFrame,
    dependency: pd.DataFrame,
) -> None:
    st.subheader("SHAP explanations")
    left, right = st.columns(2)

    with left:
        st.markdown("#### SHAP summary")
        if values.empty or not {"feature", "shap_value", "feature_percentile"}.issubset(values.columns):
            st.info("No SHAP summary data is available.")
        else:
            order = (
                summary.sort_values("mean_abs_shap", ascending=False)["feature"].tolist()
                if not summary.empty and {"feature", "mean_abs_shap"}.issubset(summary.columns)
                else values["feature"].drop_duplicates().tolist()
            )
            summary_chart = (
                alt.Chart(values)
                .mark_circle(size=28, opacity=0.48)
                .encode(
                    x=alt.X(
                        "shap_value:Q",
                        title="SHAP contribution toward anomaly",
                        scale=alt.Scale(zero=False),
                    ),
                    y=alt.Y("feature:N", sort=order, title=None),
                    color=alt.Color(
                        "feature_percentile:Q",
                        title="Relative feature value",
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    tooltip=[
                        "feature:N",
                        alt.Tooltip("feature_value:Q", format=".6f"),
                        alt.Tooltip("shap_value:Q", format=".6f"),
                        alt.Tooltip("anomaly_score:Q", format=".6f"),
                        "predicted_label:N",
                    ],
                )
                .properties(height=max(430, len(order) * 28))
                .interactive()
            )
            st.altair_chart(summary_chart, use_container_width=True)

    with right:
        st.markdown("#### SHAP dependency")
        if dependency.empty or not {"feature_value", "shap_value", "anomaly_score"}.issubset(dependency.columns):
            st.info("No SHAP dependency data is available.")
        else:
            feature_name = str(dependency.get("feature", pd.Series(["Top feature"])).iloc[0])
            dependency_chart = (
                alt.Chart(dependency)
                .mark_circle(size=48, opacity=0.62)
                .encode(
                    x=alt.X(
                        "feature_value:Q",
                        title=feature_name,
                        scale=alt.Scale(zero=False),
                    ),
                    y=alt.Y(
                        "shap_value:Q",
                        title="SHAP contribution toward anomaly",
                        scale=alt.Scale(zero=False),
                    ),
                    color=alt.Color("anomaly_score:Q", title="Anomaly score"),
                    shape=alt.Shape("predicted_label:N", title="Prediction"),
                    tooltip=[
                        "timestamp:N",
                        "src_ip:N",
                        alt.Tooltip("feature_value:Q", format=".6f"),
                        alt.Tooltip("shap_value:Q", format=".6f"),
                        alt.Tooltip("anomaly_score:Q", format=".6f"),
                    ],
                )
                .properties(height=500)
                .interactive()
            )
            st.altair_chart(dependency_chart, use_container_width=True)
            st.caption(f"Dependency feature selected automatically from the largest mean absolute SHAP value: {feature_name}.")

    if not summary.empty:
        with st.expander("SHAP feature ranking"):
            st.dataframe(summary, use_container_width=True, hide_index=True, height=420)

    st.caption(
        "SHAP values are sign-adjusted from the Isolation Forest path-length output: positive values push the window toward anomaly, while negative values push it toward normal."
    )


def _render_isolation_score_histogram(scores: pd.DataFrame) -> None:
    st.subheader("Anomaly score distribution histogram")
    if scores.empty or "anomaly_score" not in scores.columns:
        st.info("No Isolation Forest score distribution is available.")
        return

    histogram = (
        alt.Chart(scores)
        .mark_bar(opacity=0.78)
        .encode(
            x=alt.X(
                "anomaly_score:Q",
                bin=alt.Bin(maxbins=55),
                title="Anomaly score (-decision_function)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y("count():Q", title="Five-minute DNS windows"),
            color=alt.Color("predicted_label:N", title="Prediction"),
            tooltip=[
                alt.Tooltip("count():Q", title="Windows"),
                alt.Tooltip("anomaly_score:Q", bin=True, title="Score range"),
                "predicted_label:N",
            ],
        )
        .properties(height=420)
        .interactive()
    )
    threshold = alt.Chart(pd.DataFrame({"threshold": [0.0]})).mark_rule(strokeDash=[6, 4]).encode(x="threshold:Q")
    st.altair_chart(histogram + threshold, use_container_width=True)
    st.caption("Higher values are more anomalous. The model decision boundary is anomaly score 0. Axes rescale from the current result data on refresh.")


def show_isolation_forest_results() -> None:
    st.subheader("Isolation Forest learning results")
    result = read_isolation_result()
    if not result:
        st.info("No completed Isolation Forest learning result is available yet.")
        return

    try:
        config = json.loads(IF_CONFIG_PATH.read_text(encoding="utf-8")) if IF_CONFIG_PATH.exists() else {}
    except Exception:
        config = {}

    scores = load_isolation_scores()

    st.markdown("#### Main Isolation Forest parameters")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("n_estimators", config.get("n_estimators", result.get("n_estimators", "N/A")))
    p1.caption("Total isolation trees in the ensemble.")
    p2.metric("max_samples", config.get("max_samples", result.get("max_samples", "N/A")))
    p2.caption("Samples drawn to train each tree.")
    p3.metric("contamination", config.get("contamination", result.get("contamination", "N/A")))
    p3.caption("Estimated outlier proportion used to set the decision boundary.")
    p4.metric("max_features", config.get("max_features", result.get("max_features", 1.0)))
    p4.caption("Features drawn to train each base tree.")

    # Fix the earlier GUI lookup bug. The trainer writes training_windows into
    # both result.json and config.json. The score CSV is a final safe fallback.
    training_windows = result.get("training_windows")
    if not training_windows:
        training_windows = config.get("training_windows")
    if not training_windows:
        try:
            training_windows = read_isolation_status().get("training_windows")
        except Exception:
            training_windows = None
    if not training_windows and not scores.empty:
        training_windows = len(scores)
    try:
        training_windows = int(training_windows or 0)
    except Exception:
        training_windows = 0

    feature_count = result.get("feature_count", config.get("feature_count", "N/A"))

    largest = result.get("score_max")
    p99 = None
    if not scores.empty and "anomaly_score" in scores.columns:
        score_series = pd.to_numeric(scores["anomaly_score"], errors="coerce").dropna()
        if not score_series.empty:
            largest = float(score_series.max())
            p99 = float(score_series.quantile(0.99))

    st.markdown("#### Learning output")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Training windows", f"{training_windows:,}")
    r2.metric("Feature count", feature_count)
    r3.metric("Largest anomaly score", _if_numeric(largest, 6))
    r4.metric("Score percentile (99th)", "N/A" if p99 is None else _if_numeric(p99, 6))

    st.caption(
        "Training windows is the number of five-minute source-IP windows actually used to fit the Isolation Forest. "
        "Largest anomaly score is the maximum -decision_function score observed in the learning run. "
        "Score percentile (99th) is the anomaly-score value below which 99% of training-window scores fall."
    )

    with st.expander("Saved Isolation Forest learning artifacts"):
        st.json(
            {
                "training_files": result.get("training_files"),
                "model_path": result.get("model_path"),
                "scaler_path": result.get("scaler_path"),
                "config_path": result.get("config_path"),
                "scores_path": result.get("scores_path"),
                "completed_at_utc": result.get("completed_at_utc"),
                "cpu_only": result.get("cpu_only"),
                "debug_enabled": result.get("debug_enabled"),
            }
        )

    _render_isolation_score_histogram(scores)

    with st.expander("Raw Isolation Forest window scores"):
        if scores.empty:
            st.info("No score rows are available.")
        else:
            st.dataframe(
                scores.sort_values("anomaly_score", ascending=False).head(3000),
                use_container_width=True,
                hide_index=True,
                height=520,
            )

def _render_isolation_learning_io() -> None:
    with st.expander("Isolation Forest · input, outputs & learning configuration", expanded=False):
        st.markdown("**Input — same precomputed feature dataset**")
        st.code("/data/dns-ml/features/dns_training_features.csv", language="text")

        st.markdown("**Outputs**")
        st.code(
            """/data/dns-ml/models/dns_isolation_forest.joblib
/data/dns-ml/models/dns_isolation_forest_scaler.joblib
/data/dns-ml/models/dns_isolation_forest_pca.joblib
/data/dns-ml/models/dns_isolation_forest_config.json
/data/dns-ml/runs/latest_isolation_forest_status.json
/data/dns-ml/runs/latest_isolation_forest.log
/data/dns-ml/runs/latest_isolation_forest_result.json
/data/dns-ml/runs/latest_isolation_forest_scores.csv""",
            language="text",
        )

        rows = [
            ("IF_N_ESTIMATORS", os.getenv("IF_N_ESTIMATORS", "300"), "train_dns_isolation_forest.py"),
            ("IF_MAX_SAMPLES", os.getenv("IF_MAX_SAMPLES", "auto"), "train_dns_isolation_forest.py"),
            ("IF_CONTAMINATION", os.getenv("IF_CONTAMINATION", "0.01"), "train_dns_isolation_forest.py"),
            ("IF_RANDOM_STATE", os.getenv("IF_RANDOM_STATE", "42"), "train_dns_isolation_forest.py"),
            ("IF_N_JOBS", os.getenv("IF_N_JOBS", "-1"), "train_dns_isolation_forest.py"),
            ("IF_MIN_WINDOWS", os.getenv("IF_MIN_WINDOWS", "50"), "train_dns_isolation_forest.py"),
            ("IF_SHAP_SAMPLE_SIZE", os.getenv("IF_SHAP_SAMPLE_SIZE", "500"), "train_dns_isolation_forest.py"),
            ("IF_PCA_FIT_SAMPLE_SIZE", os.getenv("IF_PCA_FIT_SAMPLE_SIZE", "50000"), "train_dns_isolation_forest.py"),
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=["Parameter", "Current/default", "Defined/consumed in"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Persisted learned configuration: "
            "/data/dns-ml/models/dns_isolation_forest_config.json"
        )

def render_isolation_forest_control(dataset_ready: bool) -> None:
    st.divider()
    st.markdown(
        """
        <div class="soc-hero">
            <div class="soc-section-label">Model learning</div>
            <div class="soc-hero-title">Isolation Forest Learning Control</div>
            <div class="soc-hero-subtitle">
                Train the CPU-only Isolation Forest from the same shared
                precomputed feature dataset. Feature engineering is not repeated.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status = read_isolation_status()
    running = is_isolation_learning_running()
    state_name = str(status.get("state", "not_started")).strip().lower()
    terminal = is_terminal_isolation_state(state_name)
    can_stop = running or (
        not terminal and state_name in {"starting", "running", "training", "busy"}
    )

    c1, c2, c3, c4 = st.columns([1.15, 1.15, 1.0, 1.15])

    with c1:
        if st.button(
            "▶ Start Learning",
            key="start_isolation_forest_learning",
            use_container_width=True,
            disabled=running or not dataset_ready,
        ):
            _clear_isolation_log_snapshot()
            result = start_isolation_learning()
            if result.get("error"):
                st.error(result["error"])
            else:
                st.success(f"Isolation Forest learning started with PID {result['pid']}.")
                st.rerun()

    with c2:
        if st.button(
            "■ Stop Learning",
            key="stop_isolation_forest_learning",
            use_container_width=True,
            disabled=not can_stop,
        ):
            result = stop_isolation_learning()
            if result.get("error"):
                st.error(result["error"])
            else:
                st.warning(result.get("message", "Isolation Forest learning stopped and reset."))
                st.rerun()

    with c3:
        if st.button("↻ Refresh", key="refresh_isolation_forest_learning", use_container_width=True):
            st.rerun()

    with c4:
        if st.button(
            "🧹 Purge All",
            key="purge_all_from_isolation",
            use_container_width=True,
            disabled=training_lifecycle.any_ml_running(),
            help=(
                "Global reset: purges BOTH ML model families, both learning result "
                "sets, all uploaded/prepared training data and features, then "
                "unlocks Zeek DNS Log Manager."
            ),
        ):
            result = training_lifecycle.purge_all()
            if result.get("error"):
                st.error(result["error"])
            else:
                st.warning(result["message"])
                st.rerun()

    _render_isolation_learning_io()

    if not dataset_ready:
        st.warning(
            "Isolation Forest learning is disabled until the shared feature "
            "dataset has been prepared by Zeek DNS Log Manager."
        )

    status = read_isolation_status()
    running = is_isolation_learning_running()
    terminal = is_terminal_isolation_state(status.get("state"))
    progress = max(0, min(int(status.get("progress_percent", 0) or 0), 100))

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("State", status.get("state", "not_started"))
    s2.metric("Stage", status.get("stage", "Not started"))
    s3.metric("Progress", f"{progress}%")
    s4.metric("Execution", "CPU · DEBUG")
    animated_progress(progress, str(status.get("message", "")))

    if status.get("error"):
        st.error(status["error"])
    elif str(status.get("state", "")).lower() == "finished":
        st.success(status.get("message", "Isolation Forest learning completed."))
    elif running:
        st.info(status.get("message", "Isolation Forest learning is running."))

    with st.expander(
        "Live Isolation Forest learning command output",
        expanded=(running or terminal),
    ):
        if running:
            log_text = read_isolation_log_tail(1500)
        elif terminal:
            log_text = _stable_completed_isolation_log()
        else:
            log_text = read_complete_isolation_log()

        if "render_learning_log_window" in globals():
            render_learning_log_window(log_text, running=running)
        else:
            st.code(log_text, language="text")

    if running:
        time.sleep(1.5)
        st.rerun()

    show_isolation_forest_results()



def _isolation_file_info(path) -> dict:
    from datetime import datetime, timezone
    from pathlib import Path
    import hashlib

    item = Path(path)
    info = {
        "artifact": item.name,
        "path": str(item),
        "exists": item.exists(),
        "size_bytes": None,
        "modified_utc": None,
        "sha256": None,
    }
    if not item.exists():
        return info
    try:
        stat = item.stat()
        info["size_bytes"] = stat.st_size
        info["modified_utc"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        if item.is_file():
            digest = hashlib.sha256()
            with item.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            info["sha256"] = digest.hexdigest()
    except Exception:
        pass
    return info


def render_isolation_model_integrity() -> None:
    st.divider()
    st.subheader("Isolation Forest model integrity")
    artifacts = [
        IF_MODEL_PATH,
        IF_SCALER_PATH,
        IF_PCA_PATH,
        IF_CONFIG_PATH,
        IF_RESULT_PATH,
        IF_STATUS_PATH,
        IF_LOG_PATH,
        IF_SCORES_PATH,
        IF_POINTS_PATH,
        IF_CONTOUR_PATH,
        IF_SHAP_SUMMARY_PATH,
        IF_SHAP_VALUES_PATH,
        IF_SHAP_DEPENDENCY_PATH,
    ]
    artifact_rows = [_isolation_file_info(path) for path in artifacts]
    artifact_frame = pd.DataFrame(artifact_rows)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Isolation Forest", "OK" if IF_MODEL_PATH.exists() else "Missing")
    c2.metric("Feature scaler", "OK" if IF_SCALER_PATH.exists() else "Missing")
    c3.metric("PCA projection", "OK" if IF_PCA_PATH.exists() else "Missing")
    c4.metric("SHAP diagnostics", "OK" if IF_SHAP_SUMMARY_PATH.exists() else "Missing")

    st.dataframe(artifact_frame, use_container_width=True, hide_index=True, height=470)

    result = read_isolation_result()
    try:
        config = json.loads(IF_CONFIG_PATH.read_text(encoding="utf-8")) if IF_CONFIG_PATH.exists() else {}
    except Exception:
        config = {}

    left, right = st.columns(2)
    with left:
        st.markdown("#### Isolation Forest configuration")
        if config:
            st.json(config)
        else:
            st.warning("The Isolation Forest configuration is missing or unreadable.")
    with right:
        st.markdown("#### Latest Isolation Forest result")
        if result:
            st.json(result)
        else:
            st.info("No completed Isolation Forest learning result is available.")

# END DNS ISOLATION FOREST LEARNING


def page_learning_navigator():
    st.title("🧭 Learning Navigator")
    st.caption(
        "Prepare Zeek DNS logs, train the CPU-only Isolation Forest and the "
        "Autoencoder independently, follow live progress and review each "
        "model's learning results."
    )

    dataset_ready = render_zeek_log_manager()
    render_isolation_forest_control(dataset_ready)
    render_learning_control(dataset_ready)

def filter_alerts(df: pd.DataFrame):
    if df.empty:
        return df

    df = df.copy()
    df["timestamp_dt"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True,
    )
    df["detection_type"] = df["detection_methods_json"].apply(
        detection_type_from_json
    )

    st.sidebar.subheader("Filters")
    severities = sorted(df["severity"].dropna().unique().tolist())
    statuses = sorted(df["status"].dropna().unique().tolist())
    src_ips = sorted(df["src_ip"].dropna().unique().tolist())

    selected_severity = st.sidebar.multiselect(
        "Severity",
        severities,
        default=severities,
    )
    selected_status = st.sidebar.multiselect(
        "Status",
        statuses,
        default=statuses,
    )
    selected_src_ips = st.sidebar.multiselect(
        "Source IP",
        src_ips,
        default=[],
    )

    hours = st.sidebar.selectbox(
        "Time range",
        [999999, 1, 6, 12, 24, 72, 168],
        index=0,
        format_func=(
            lambda value: "All time"
            if value == 999999
            else f"Last {value} hours"
        ),
    )

    filtered = df[
        df["severity"].isin(selected_severity)
        & df["status"].isin(selected_status)
    ].copy()

    if selected_src_ips:
        filtered = filtered[filtered["src_ip"].isin(selected_src_ips)]

    if hours != 999999:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
        filtered = filtered[filtered["timestamp_dt"] >= cutoff]

    return filtered



def render_scoring_control_cards(cards: dict) -> None:
    fragments = ['<div class="scoring-control-grid">']
    ordered = [
        "check_logs",
        "run_once",
        "full_log",
        "activate_task",
        "check_task",
        "deactivate_task",
    ]
    for number, key in enumerate(ordered, start=1):
        card = cards.get(key, {})
        percent = max(0, min(int(card.get("percent", 0) or 0), 100))
        state = str(card.get("state", "idle")).lower()
        ring_state = state if state in {"success", "warning", "error"} else ""
        fragments.append(
            '<div class="scoring-control-card">'
            f'<div class="scoring-card-number">Card Info Window {number}</div>'
            f'<div class="scoring-card-title">{html.escape(str(card.get("title", key)))}</div>'
            f'<div class="soc-progress-ring scoring-ring {ring_state}" '
            f'style="--progress:{percent}"><span>{percent}%</span></div>'
            f'<div class="scoring-card-headline">{html.escape(str(card.get("headline", "Waiting")))}</div>'
            f'<div class="scoring-card-detail">{html.escape(str(card.get("detail", "")))}</div>'
            '</div>'
        )
    fragments.append('</div>')
    st.markdown(''.join(fragments), unsafe_allow_html=True)


# BEGIN DNS SCORING LIVE OUTPUT FLUSH HELPER

_DNS_SCORING_LIVE_OUTPUT_PATHS = {
    "categorization": Path("/data/dns-ml/isolation_scoring/live_scoring.log"),
    "autoencoder": Path("/data/dns-ml/scoring/live_scoring.log"),
    "heuristics": Path("/data/dns-ml/heuristics_scoring/live_scoring.log"),
}


def _flush_dns_scoring_live_output(engine: str, action: str) -> None:
    path = _DNS_SCORING_LIVE_OUTPUT_PATHS.get(str(engine))
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    except Exception:
        stamp = datetime.now().astimezone().isoformat(timespec="seconds")

    titles = {
        "categorization": "DNS CATEGORIZATION SCORING — ISOLATION FOREST",
        "autoencoder": "DNS DEEP LEARNING SCORING — AUTOENCODER",
        "heuristics": "DNS HEURISTICS SCORING — RULE BASED",
    }
    title = titles.get(str(engine), "DNS SCORING")

    content = (
        "=" * 86 + "\n"
        + title + "\n"
        + "=" * 86 + "\n"
        + f"[{stamp}] ACTION: {action}\n"
        + "Previous Live recording command output: FLUSHED\n"
        + "-" * 86 + "\n"
    )

    temporary = path.with_suffix(path.suffix + ".gui-flush")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)

# END DNS SCORING LIVE OUTPUT FLUSH HELPER

def render_scoring_log_window(log_text: str, running: bool) -> None:
    safe_log = html.escape(log_text or "No DNS scoring output is available yet.")
    scroll_target = "panel.scrollHeight" if running else "0"
    components.html(
        f"""
        <!doctype html>
        <html><head><style>
        html,body{{margin:0;padding:0;background:transparent;color:#d7e2f0;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;}}
        #shell{{height:408px;box-sizing:border-box;border:1px solid rgba(148,163,184,.22);border-radius:12px;overflow:hidden;background:#060b13;}}
        #toolbar{{height:38px;display:flex;align-items:center;justify-content:flex-end;box-sizing:border-box;padding:4px 7px;border-bottom:1px solid rgba(148,163,184,.16);background:#0b1220;}}
        #fs{{width:31px;height:29px;border:1px solid rgba(148,163,184,.28);border-radius:7px;background:#111827;color:#e2e8f0;font-size:19px;line-height:1;display:flex;align-items:center;justify-content:center;cursor:pointer;}}
        #fs:hover{{border-color:#22d3ee;color:#67e8f9;background:#172033;}}
        #dns-scoring-log{{height:368px;overflow-y:scroll;overflow-x:auto;box-sizing:border-box;padding:14px 16px;background:#060b13;scrollbar-color:#22d3ee #111827;}}
        #dns-scoring-log::-webkit-scrollbar{{width:13px;height:13px}}
        #dns-scoring-log::-webkit-scrollbar-track{{background:#111827}}
        #dns-scoring-log::-webkit-scrollbar-thumb{{background:#22d3ee;border-radius:8px;border:3px solid #111827}}
        pre{{margin:0;white-space:pre;line-height:1.43;font-size:12.5px;}}
        :fullscreen,:-webkit-full-screen{{background:#060b13!important;}}
        :fullscreen #shell,:-webkit-full-screen #shell{{width:100vw;height:100vh;border:0;border-radius:0;}}
        :fullscreen #dns-scoring-log,:-webkit-full-screen #dns-scoring-log{{height:calc(100vh - 38px);}}
        </style></head>
        <body>
        <div id="shell">
          <div id="toolbar"><button id="fs" type="button" title="Full screen" aria-label="Full screen">⛶</button></div>
          <div id="dns-scoring-log"><pre>{safe_log}</pre></div>
        </div>
        <script>
        const panel=document.getElementById("dns-scoring-log");
        const button=document.getElementById("fs");
        const target=document.documentElement;
        panel.scrollTop={scroll_target};
        function active(){{return Boolean(document.fullscreenElement||document.webkitFullscreenElement);}}
        function icon(){{button.textContent=active()?"✕":"⛶";button.title=active()?"Exit full screen":"Full screen";}}
        async function toggle(){{
          try{{
            if(!active()){{
              if(target.requestFullscreen){{await target.requestFullscreen();}}
              else if(target.webkitRequestFullscreen){{target.webkitRequestFullscreen();}}
            }} else {{
              if(document.exitFullscreen){{await document.exitFullscreen();}}
              else if(document.webkitExitFullscreen){{document.webkitExitFullscreen();}}
            }}
          }}catch(error){{
            const popup=window.open("","_blank");
            if(popup){{popup.document.open();popup.document.write(document.documentElement.outerHTML);popup.document.close();}}
          }}
          icon();
        }}
        button.addEventListener("click",toggle);
        document.addEventListener("fullscreenchange",icon);
        document.addEventListener("webkitfullscreenchange",icon);
        icon();
        </script>
        </body></html>
        """,
        height=430,
        scrolling=False,
    )

def render_reconstruction_error_window(scored: pd.DataFrame) -> None:
    st.subheader("Reconstruction error")
    if scored.empty or "reconstruction_error" not in scored.columns:
        st.info("Run DNS scoring to generate reconstruction-error data.")
        return

    frame = scored.tail(500).copy()
    frame["timestamp"] = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    if frame.empty:
        st.info("No valid scoring timestamps are available.")
        return
    frame["window"] = frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC") + " | " + frame["src_ip"].astype(str)
    columns = ["window", "reconstruction_error"]
    if "threshold" in frame.columns:
        columns.append("threshold")
    long_frame = frame[columns].melt("window", var_name="series", value_name="value")
    chart = (
        alt.Chart(long_frame)
        .mark_line(point=True)
        .encode(
            x=alt.X("window:N", title="Scored source-IP window", sort=None, axis=alt.Axis(labels=False)),
            y=alt.Y("value:Q", title="MAE reconstruction error"),
            color=alt.Color("series:N", title=None),
            tooltip=["window:N", "series:N", alt.Tooltip("value:Q", format=".6f")],
        )
        .properties(height=340)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)


def render_top_reconstruction_errors(scored: pd.DataFrame) -> None:
    st.subheader("Top 10 reconstruction errors")
    if scored.empty or "reconstruction_error" not in scored.columns:
        st.info("Run DNS scoring to generate reconstruction-error results.")
        return

    frame = scored.copy()
    frame["reconstruction_error"] = pd.to_numeric(
        frame["reconstruction_error"], errors="coerce"
    )
    frame = frame.dropna(subset=["reconstruction_error"])
    if frame.empty:
        st.info("No valid reconstruction-error values are available.")
        return

    if "timestamp" in frame.columns:
        parsed = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame["timestamp"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    frame = frame.sort_values("reconstruction_error", ascending=False).head(10)
    preferred = [
        "timestamp",
        "src_ip",
        "reconstruction_error",
        "threshold",
        "threshold_ratio",
        "autoencoder_anomaly",
        "scoring_mode",
    ]
    columns = [column for column in preferred if column in frame.columns]
    display = frame[columns].copy() if columns else frame.head(10).copy()

    for column in ("reconstruction_error", "threshold", "threshold_ratio"):
        if column in display.columns:
            display[column] = pd.to_numeric(display[column], errors="coerce").round(8)

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        height=390,
    )
    st.caption(
        "Rows are ordered from the largest to the smallest reconstruction error. "
        "Full-log scoring replaces the diagnostic dataset before this table is recalculated."
    )


def render_feature_weight_heatmap(scored: pd.DataFrame, residuals: pd.DataFrame) -> None:
    st.subheader("Feature / weight correlation heatmap")
    if scored.empty or residuals.empty:
        st.info("Run DNS scoring to generate feature reconstruction-weight data.")
        return

    keys = [column for column in ("timestamp", "src_ip") if column in scored.columns and column in residuals.columns]
    if len(keys) < 2:
        st.info("The scoring diagnostics do not contain matching timestamp/source-IP keys.")
        return
    merged = scored.merge(residuals, on=keys, how="inner", suffixes=("", "_residual"))
    residual_columns = [column for column in residuals.columns if column.startswith("residual__")]
    rows = []
    mean_weights = {}
    correlations = {}
    for residual_column in residual_columns:
        feature = residual_column.removeprefix("residual__")
        if feature not in merged.columns:
            continue
        feature_values = pd.to_numeric(merged[feature], errors="coerce")
        weight_values = pd.to_numeric(merged[residual_column], errors="coerce")
        valid = feature_values.notna() & weight_values.notna()
        correlation = feature_values[valid].corr(weight_values[valid]) if valid.sum() >= 3 else 0.0
        correlations[feature] = 0.0 if pd.isna(correlation) else float(correlation)
        mean_weights[feature] = float(weight_values[valid].mean()) if valid.any() else 0.0

    if not mean_weights:
        st.info("No feature residual columns were found.")
        return
    maximum_weight = max(mean_weights.values()) or 1.0
    for feature in mean_weights:
        rows.extend(
            [
                {
                    "feature": feature,
                    "metric": "Feature ↔ residual correlation",
                    "value": correlations.get(feature, 0.0),
                    "raw_value": correlations.get(feature, 0.0),
                },
                {
                    "feature": feature,
                    "metric": "Relative reconstruction weight",
                    "value": mean_weights[feature] / maximum_weight,
                    "raw_value": mean_weights[feature],
                },
            ]
        )
    heat = pd.DataFrame(rows)
    chart = (
        alt.Chart(heat)
        .mark_rect()
        .encode(
            x=alt.X("metric:N", title=None),
            y=alt.Y("feature:N", title="DNS feature", sort="-x"),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="redyellowblue", domain=[-1, 1]), title="Normalized value"),
            tooltip=["feature:N", "metric:N", alt.Tooltip("raw_value:Q", format=".6f")],
        )
        .properties(height=max(420, len(mean_weights) * 15))
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Weight is the feature's mean absolute reconstruction residual; correlation compares each input feature with its own residual weight.")


def render_latent_heatmap(latent: pd.DataFrame) -> None:
    st.subheader("Latent space heatmap")
    latent_columns = [column for column in latent.columns if column.startswith("latent_")]
    if latent.empty or not latent_columns:
        st.info("Run DNS scoring to generate latent-layer vectors.")
        return
    frame = latent.tail(160).copy().reset_index(drop=True)
    frame["row"] = frame.index + 1
    frame["window"] = frame.get("timestamp", "").astype(str) + " | " + frame.get("src_ip", "").astype(str)
    long_frame = frame[["row", "window"] + latent_columns].melt(
        ["row", "window"], var_name="latent_dimension", value_name="activation"
    )
    chart = (
        alt.Chart(long_frame)
        .mark_rect()
        .encode(
            x=alt.X("latent_dimension:N", title="Latent dimension", sort=None),
            y=alt.Y("row:O", title="Latest scored windows", sort="descending", axis=alt.Axis(labels=False)),
            color=alt.Color("activation:Q", scale=alt.Scale(scheme="viridis"), title="Activation"),
            tooltip=["window:N", "latent_dimension:N", alt.Tooltip("activation:Q", format=".6f")],
        )
        .properties(height=420)
    )
    st.altair_chart(chart, use_container_width=True)


def render_detected_live_alerts(alerts: pd.DataFrame) -> None:
    st.subheader("Detected live alerts")
    if alerts.empty:
        st.info("No live DNS alerts have been detected yet.")
        return
    st.dataframe(alerts.iloc[::-1], use_container_width=True, hide_index=True, height=390)


def _refresh_alert_ingest() -> None:
    try:
        ingest_alert_file(ALERT_JSONL_PATH)
    except Exception:
        pass
    st.cache_data.clear()



# BEGIN DNS CATEGORIZATION SCORING GUI

def _render_categorization_cards(cards: dict) -> None:
    ordered = ["check_logs", "run_once", "full_log", "activate_task", "check_task", "deactivate_task"]
    fragments = ['<div class="scoring-control-grid">']
    for number, key in enumerate(ordered, start=1):
        card = cards.get(key, {})
        percent = max(0, min(int(card.get("percent", 0) or 0), 100))
        state = str(card.get("state", "idle")).lower()
        ring_state = state if state in {"success", "warning", "error"} else ""
        fragments.append(
            '<div class="scoring-control-card">'
            f'<div class="scoring-card-number">Card Info Window {number}</div>'
            f'<div class="scoring-card-title">{html.escape(str(card.get("title", key)))}</div>'
            f'<div class="soc-progress-ring scoring-ring {ring_state}" style="--progress:{percent}"><span>{percent}%</span></div>'
            f'<div class="scoring-card-headline">{html.escape(str(card.get("headline", "Waiting")))}</div>'
            f'<div class="scoring-card-detail">{html.escape(str(card.get("detail", "")))}</div>'
            '</div>'
        )
    fragments.append('</div>')
    st.markdown(''.join(fragments), unsafe_allow_html=True)


def _render_categorization_log(log_text: str, running: bool) -> None:
    safe_log = html.escape(log_text or "No DNS scoring command output is available yet.")
    scroll_target = "panel.scrollHeight" if running else "0"
    components.html(
        f"""
        <!doctype html>
        <html><head><style>
        html,body{{margin:0;padding:0;background:transparent;color:#d7e2f0;font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace;}}
        #shell{{height:408px;box-sizing:border-box;border:1px solid rgba(148,163,184,.22);border-radius:12px;overflow:hidden;background:#060b13;}}
        #toolbar{{height:38px;display:flex;align-items:center;justify-content:flex-end;box-sizing:border-box;padding:4px 7px;border-bottom:1px solid rgba(148,163,184,.16);background:#0b1220;}}
        #fs{{width:31px;height:29px;border:1px solid rgba(148,163,184,.28);border-radius:7px;background:#111827;color:#e2e8f0;font-size:19px;line-height:1;display:flex;align-items:center;justify-content:center;cursor:pointer;}}
        #fs:hover{{border-color:#22d3ee;color:#67e8f9;background:#172033;}}
        #cat-log{{height:368px;overflow-y:scroll;overflow-x:auto;box-sizing:border-box;padding:14px 16px;background:#060b13;scrollbar-color:#22d3ee #111827;}}
        #cat-log::-webkit-scrollbar{{width:13px;height:13px}}
        #cat-log::-webkit-scrollbar-track{{background:#111827}}
        #cat-log::-webkit-scrollbar-thumb{{background:#22d3ee;border-radius:8px;border:3px solid #111827}}
        pre{{margin:0;white-space:pre;line-height:1.43;font-size:12.5px;}}
        :fullscreen,:-webkit-full-screen{{background:#060b13!important;}}
        :fullscreen #shell,:-webkit-full-screen #shell{{width:100vw;height:100vh;border:0;border-radius:0;}}
        :fullscreen #cat-log,:-webkit-full-screen #cat-log{{height:calc(100vh - 38px);}}
        </style></head>
        <body>
        <div id="shell">
          <div id="toolbar"><button id="fs" type="button" title="Full screen" aria-label="Full screen">⛶</button></div>
          <div id="cat-log"><pre>{safe_log}</pre></div>
        </div>
        <script>
        const panel=document.getElementById("cat-log");
        const button=document.getElementById("fs");
        const target=document.documentElement;
        panel.scrollTop={scroll_target};
        function active(){{return Boolean(document.fullscreenElement||document.webkitFullscreenElement);}}
        function icon(){{button.textContent=active()?"✕":"⛶";button.title=active()?"Exit full screen":"Full screen";}}
        async function toggle(){{
          try{{
            if(!active()){{
              if(target.requestFullscreen){{await target.requestFullscreen();}}
              else if(target.webkitRequestFullscreen){{target.webkitRequestFullscreen();}}
            }} else {{
              if(document.exitFullscreen){{await document.exitFullscreen();}}
              else if(document.webkitExitFullscreen){{document.webkitExitFullscreen();}}
            }}
          }}catch(error){{
            const popup=window.open("","_blank");
            if(popup){{popup.document.open();popup.document.write(document.documentElement.outerHTML);popup.document.close();}}
          }}
          icon();
        }}
        button.addEventListener("click",toggle);
        document.addEventListener("fullscreenchange",icon);
        document.addEventListener("webkitfullscreenchange",icon);
        icon();
        </script>
        </body></html>
        """,
        height=430,
        scrolling=False,
    )

def _render_critical_isolation_parameters(parameters: dict) -> None:
    st.markdown("#### Critical Isolation Forest Parameters")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("contamination (Anomaly Rate)", parameters.get("contamination", "N/A"))
    p1.caption("Defines the expected proportion of outliers in the learned DNS population.")
    p2.metric("n_estimators (Number of Trees)", parameters.get("n_estimators", "N/A"))
    p2.caption("Number of randomized isolation trees in the ensemble.")
    p3.metric("max_samples (Subsample Size)", parameters.get("max_samples", "N/A"))
    p3.caption("Samples drawn to build each isolation tree.")
    p4.metric("max_features (Feature Subsampling)", parameters.get("max_features", "N/A"))
    p4.caption("Features made available to each tree; the model still uses the shared 36-feature schema.")

    final_cfg = _final_config_for_info()
    native_boundary = parameters.get("anomaly_threshold", 0.0)
    min_iso_score = float(final_cfg.get("isolation_min_score", 0.0) or 0.0)
    min_iso_percentile = float(final_cfg.get("isolation_min_percentile", 95.0) or 95.0)

    with st.container(border=True):
        st.markdown("#### Isolation Forest severity & final-correlation gates")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Native anomaly boundary", _if_numeric(native_boundary, 6))
        s2.metric("Minimum Isolation score", f"{min_iso_score:.6f}")
        s3.metric("Minimum Isolation percentile", f"{min_iso_percentile:.1f}")
        s4.metric("Native threat score", "≈ percentile")

        severity = pd.DataFrame(
            [
                {"Severity": "Low", "Native Isolation percentile": "< 97.5"},
                {"Severity": "Medium", "Native Isolation percentile": "97.5 to < 99.0"},
                {"Severity": "High", "Native Isolation percentile": "99.0 to < 99.9"},
                {"Severity": "Critical", "Native Isolation percentile": "≥ 99.9"},
            ]
        )
        st.dataframe(severity, use_container_width=True, hide_index=True)
        st.caption(
            "Native Isolation Forest severity is calculated from score_percentile: "
            "Critical ≥99.9 · High ≥99.0 · Medium ≥97.5 · otherwise Low. "
            "The native threat score is the rounded percentile. If the training score reference is unavailable, "
            "the scorer falls back to a score-derived threat value."
        )
        st.info(
            "Important: Minimum Isolation score and Minimum Isolation percentile are Final DNS Alert Correlation gates. "
            "They decide whether Isolation evidence is allowed into the final alert; they do NOT calculate or overwrite "
            "the native Isolation Forest severity."
        )

    if not parameters.get("model_ready"):
        st.warning("Isolation Forest learning artifacts are incomplete. Complete Isolation Forest learning before scoring.")
    if parameters.get("load_error"):
        st.warning(f"Model parameter inspection warning: {parameters['load_error']}")

def _render_categorization_top_scores(scored: pd.DataFrame) -> None:
    st.subheader("Top 10 Isolation Forest anomaly scores")
    if scored.empty or "anomaly_score" not in scored.columns:
        st.info("Run categorization scoring to generate Isolation Forest scores.")
        return
    frame = scored.copy()
    frame["anomaly_score"] = pd.to_numeric(frame["anomaly_score"], errors="coerce")
    frame = frame.dropna(subset=["anomaly_score"]).sort_values("anomaly_score", ascending=False).head(10)
    if "timestamp" in frame.columns:
        parsed = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame["timestamp"] = parsed.dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    preferred = ["timestamp", "src_ip", "anomaly_score", "isolation_threshold", "score_margin", "score_percentile", "predicted_label", "scoring_mode"]
    columns = [c for c in preferred if c in frame.columns]
    st.dataframe(frame[columns] if columns else frame, use_container_width=True, hide_index=True, height=390)


def _render_categorization_live_alerts(alerts: pd.DataFrame) -> None:
    st.subheader("Detected live alerts")
    if alerts.empty:
        st.info("No DNS categorization Isolation Forest alerts have been generated yet.")
        return
    st.dataframe(alerts.iloc[::-1], use_container_width=True, hide_index=True, height=410)


def render_dns_categorization_scoring() -> None:
    st.header("DNS categorization scoring")
    st.caption(
        "Independent CPU-only Isolation Forest scoring. This section uses its own cursor, buffer, cron task, "
        "diagnostic files and categorization alert JSONL, and does not modify the Autoencoder/heuristic scoring state."
    )

    parameters = load_categorization_model_parameters()
    _render_critical_isolation_parameters(parameters)

    running = is_categorization_scoring_running()
    cards = sync_categorization_cards()
    status = read_categorization_scoring_status()

    completion_signature = (
        f"{status.get('completed_at_utc')}|{status.get('alerts_written')}|"
        f"{status.get('state')}|{status.get('run_mode')}"
    )
    if not running and status.get("state") in {"finished", "completed", "success", "no_data"}:
        if st.session_state.get("last_categorization_ingest_signature") != completion_signature:
            try:
                ingest_alert_file(CATEGORIZATION_ALERT_JSONL_PATH, incremental=True)
            except Exception:
                pass
            st.session_state["last_categorization_ingest_signature"] = completion_signature
            st.cache_data.clear()

    _render_categorization_cards(cards)
    percentages = [int(card.get("percent", 0) or 0) for card in cards.values()]
    overall_percent = int(status.get("progress_percent", 0) or 0) if running else max(percentages or [0])
    overall_percent = max(0, min(overall_percent, 100))
    stage = status.get("stage", "Latest categorization control operation") if running else "Latest categorization control operation"
    animated_progress(overall_percent, stage)

    buttons = st.columns(6)
    clicked = False
    with buttons[0]:
        if st.button("1 · Check Live Zeek Logs", key="cat_check_logs", use_container_width=True, disabled=running):
            _flush_dns_scoring_live_output("categorization", '1 · Check Live Zeek Logs')
            result = check_categorization_live_zeek_log()
            st.session_state["categorization_check_result"] = result
            clicked = True
    with buttons[1]:
        if st.button("2 · Run Logging Once", key="cat_run_once", use_container_width=True, disabled=running or not parameters.get("model_ready")):
            _flush_dns_scoring_live_output("categorization", '2 · Run Logging Once')
            result = start_categorization_scoring("once")
            if result.get("error"):
                st.error(result["error"])
            clicked = True
    with buttons[2]:
        if st.button("3 · Run Scoring From Scratch", key="cat_full_log", use_container_width=True, disabled=running or not parameters.get("model_ready")):
            _flush_dns_scoring_live_output("categorization", '3 · Run Scoring From Scratch')
            result = start_categorization_scoring("full")
            if result.get("error"):
                st.error(result["error"])
            clicked = True
    with buttons[3]:
        if st.button("4 · Activate Logging Task", key="cat_activate", use_container_width=True, disabled=running):
            _flush_dns_scoring_live_output("categorization", '4 · Activate Logging Task')
            result = activate_categorization_cron()
            if result.get("error"):
                st.error(result["error"])
            clicked = True
    with buttons[4]:
        if st.button("5 · Check Task Status", key="cat_check_task", use_container_width=True):
            _flush_dns_scoring_live_output("categorization", '5 · Check Task Status')
            st.session_state["categorization_task_result"] = check_categorization_cron_status()
            clicked = True
    with buttons[5]:
        if st.button("6 · Deactivate Logging Task", key="cat_deactivate", use_container_width=True, disabled=running):
            _flush_dns_scoring_live_output("categorization", '6 · Deactivate Logging Task')
            result = deactivate_categorization_cron()
            if result.get("error"):
                st.error(result["error"])
            clicked = True

    if clicked:
        st.rerun()

    check_result = st.session_state.get("categorization_check_result")
    if check_result and not check_result.get("error"):
        with st.expander("Latest categorization live-log readiness details"):
            st.json(check_result)

    st.subheader("Live recording command output")
    _render_categorization_log(read_categorization_scoring_log(1500 if running else None), running)

    scored = load_categorization_scoring_windows()
    alerts = load_categorization_live_alerts(500)
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            _render_categorization_top_scores(scored)
    with right:
        with st.container(border=True):
            _render_categorization_live_alerts(alerts)

    if running:
        time.sleep(1.5)
        st.rerun()


def _categorization_alert_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    if "alert_type" in frame.columns:
        return frame["alert_type"].fillna("").astype(str).eq("categorization_forest")
    return frame["detection_methods_json"].fillna("").astype(str).str.contains("categorization_forest", regex=False)


def _render_alert_group_table(frame: pd.DataFrame, categorization: bool, key_prefix: str) -> None:
    if frame.empty:
        st.info("No alerts are available in this section for the current filters.")
        return
    display = frame.copy()
    display["severity"] = display["severity"].apply(severity_label)
    display["detection_type"] = display["detection_methods_json"].apply(detection_type_from_json)
    if categorization:
        preferred = ["id", "timestamp", "src_ip", "severity", "threat_score", "status", "isolation_score", "isolation_threshold", "score_percentile", "detection_type"]
    else:
        preferred = ["id", "timestamp", "src_ip", "severity", "threat_score", "status", "reconstruction_error", "threshold", "detection_type"]
    columns = [column for column in preferred if column in display.columns]
    dataframe_or_info(display[columns], "No alerts match the selected filters.")
    selected_id = st.selectbox("Select an alert ID", frame["id"].tolist(), key=f"{key_prefix}_alert_id")
    if st.button("Open selected alert", key=f"{key_prefix}_open_alert"):
        st.session_state["selected_alert_id"] = int(selected_id)
        st.info("Open the Alert Details page.")

# END DNS CATEGORIZATION SCORING GUI


# BEGIN DNS HEURISTICS SCORING GUI

def _heuristics_alert_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)
    heuristic_names = (
        "heuristic_dga",
        "heuristic_tunnel",
        "heuristic_fastflux",
        "heuristic_resolver_failure",
        "heuristic_rogue_resolver",
    )

    def is_heuristic_row(row) -> bool:
        alert_type = str(row.get("alert_type") or "")
        if alert_type in {"heuristics_rule", "rule_based_suspicious", "high_confidence_suspicious"}:
            return True
        methods = parse_json(row.get("detection_methods_json"), {})
        return isinstance(methods, dict) and any(bool(methods.get(name)) for name in heuristic_names)

    return frame.apply(is_heuristic_row, axis=1).astype(bool)


def _deep_learning_alert_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)
    cat = _categorization_alert_mask(frame)
    heur = _heuristics_alert_mask(frame)
    return ~(cat | heur)


def _render_heuristics_live_alerts(alerts: pd.DataFrame) -> None:
    st.subheader("Detected live alerts")
    if alerts.empty:
        st.info("No live DNS heuristic alerts have been detected yet.")
        return
    st.dataframe(alerts.iloc[::-1], use_container_width=True, hide_index=True, height=390)


def _render_alert_family_table(frame: pd.DataFrame, family: str, key_prefix: str) -> None:
    if frame.empty:
        st.info("No alerts are available in this section for the current filters.")
        return

    frame = _ensure_public_alert_ids(frame)
    display = frame.copy()
    display["severity"] = display["severity"].apply(severity_label)
    display["detection_type"] = display["detection_methods_json"].apply(detection_type_from_json)
    display["Alert ID"] = display["public_alert_id"]

    if family == "categorization":
        preferred = ["Alert ID", "timestamp", "src_ip", "severity", "threat_score", "status", "isolation_score", "isolation_threshold", "score_percentile", "detection_type"]
    elif family == "heuristics":
        preferred = ["Alert ID", "timestamp", "src_ip", "severity", "threat_score", "status", "model_name", "detection_type"]
    else:
        preferred = ["Alert ID", "timestamp", "src_ip", "severity", "threat_score", "status", "reconstruction_error", "threshold", "detection_type"]

    columns = [column for column in preferred if column in display.columns]
    dataframe_or_info(display[columns], "No alerts match the selected filters.")

    public_ids = frame["public_alert_id"].astype(str).tolist()
    selected_public_id = st.selectbox("Select an alert ID", public_ids, key=f"{key_prefix}_alert_id")
    if st.button("Open selected alert", key=f"{key_prefix}_open_alert"):
        st.session_state["selected_alert_id"] = str(selected_public_id)
        st.info("Open the Alert Details page.")

def render_dns_heuristics_scoring() -> None:
    st.header("DNS heuristics Scoring")
    st.caption(
        "Independent rule-based DNS scoring. This section uses its own cursor, buffer, cron task and alert JSONL. "
        "It does not load or execute the Autoencoder or Isolation Forest models."
    )
    _render_heuristics_severity_information()


    running = is_heuristics_scoring_running()
    cards = sync_heuristics_cards()
    status = read_heuristics_scoring_status()

    completion_signature = (
        f"{status.get('completed_at_utc')}|{status.get('alerts_written')}|"
        f"{status.get('state')}|{status.get('run_mode')}"
    )
    if not running and status.get("state") in {"finished", "completed", "success", "no_data"}:
        if st.session_state.get("last_heuristics_ingest_signature") != completion_signature:
            try:
                ingest_alert_file(HEUR_ALERT_PATH, incremental=True)
            except Exception:
                pass
            st.session_state["last_heuristics_ingest_signature"] = completion_signature
            st.cache_data.clear()

    _render_categorization_cards(cards)
    percentages = [int(card.get("percent", 0) or 0) for card in cards.values()]
    overall_percent = int(status.get("progress_percent", 0) or 0) if running else max(percentages or [0])
    overall_percent = max(0, min(overall_percent, 100))
    stage = status.get("stage", "Latest heuristic control operation") if running else "Latest heuristic control operation"
    animated_progress(overall_percent, stage)

    buttons = st.columns(6)
    clicked = False
    with buttons[0]:
        if st.button("1 · Check Live Zeek Logs", key="heur_check_logs", use_container_width=True, disabled=running):
            _flush_dns_scoring_live_output("heuristics", '1 · Check Live Zeek Logs')
            st.session_state["heuristics_check_result"] = check_heuristics_live_zeek_log()
            clicked = True
    with buttons[1]:
        if st.button("2 · Run Logging Once", key="heur_run_once", use_container_width=True, disabled=running):
            _flush_dns_scoring_live_output("heuristics", '2 · Run Logging Once')
            result = start_heuristics_scoring("once")
            if result.get("error"):
                st.error(result["error"])
            clicked = True
    with buttons[2]:
        if st.button("3 · Run Scoring From Scratch", key="heur_full", use_container_width=True, disabled=running):
            _flush_dns_scoring_live_output("heuristics", '3 · Run Scoring From Scratch')
            result = start_heuristics_scoring("full")
            if result.get("error"):
                st.error(result["error"])
            clicked = True
    with buttons[3]:
        if st.button("4 · Activate Logging Task", key="heur_activate", use_container_width=True, disabled=running):
            _flush_dns_scoring_live_output("heuristics", '4 · Activate Logging Task')
            result = activate_heuristics_cron()
            if result.get("error"):
                st.error(result["error"])
            clicked = True
    with buttons[4]:
        if st.button("5 · Check Task Status", key="heur_check_task", use_container_width=True):
            _flush_dns_scoring_live_output("heuristics", '5 · Check Task Status')
            st.session_state["heuristics_task_result"] = check_heuristics_cron_status()
            clicked = True
    with buttons[5]:
        if st.button("6 · Deactivate Logging Task", key="heur_deactivate", use_container_width=True, disabled=running):
            _flush_dns_scoring_live_output("heuristics", '6 · Deactivate Logging Task')
            result = deactivate_heuristics_cron()
            if result.get("error"):
                st.error(result["error"])
            clicked = True

    if clicked:
        st.rerun()

    check_result = st.session_state.get("heuristics_check_result")
    if check_result and not check_result.get("error"):
        with st.expander("Latest heuristic live-log readiness details"):
            st.json(check_result)

    st.subheader("Live recording command output")
    _render_categorization_log(read_heuristics_scoring_log(1500 if running else None), running)

    alerts = load_heuristics_live_alerts(500)
    with st.container(border=True):
        _render_heuristics_live_alerts(alerts)

    if running:
        time.sleep(1.5)
        st.rerun()

# END DNS HEURISTICS SCORING GUI

# BEGIN DNS FINAL CORRELATION GUI

def _render_final_alerts_window(location: str = "scoring") -> None:
    final_alerts = load_final_alerts(500)
    status = read_final_status()

    st.subheader("Final correlated DNS alerts")
    st.caption(
        "Second-stage SOC correlation of Isolation Forest, Autoencoder and DNS heuristic alerts "
        "for the same source IP and five-minute DNS window."
    )

    if status:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Correlated windows", f"{int(status.get('correlated_windows', 0)):,}")
        m2.metric("2-model candidates", f"{int(status.get('two_model_candidates', 0)):,}")
        m3.metric("3-model candidates", f"{int(status.get('three_model_candidates', 0)):,}")
        m4.metric("Final alerts", f"{int(status.get('final_alerts', 0)):,}")

    if final_alerts.empty:
        if location == "alerts":
            st.info(
                "No final correlated alerts are available. Run Final DNS Alert Correlation "
                "at the bottom of the DNS Scoring page after scoring the three detection pipelines."
            )
        else:
            st.info("No final correlated alerts have been generated with the current final-score parameters.")
        return

    preferred = [
        "timestamp",
        "src_ip",
        "severity",
        "threat_score",
        "models_agreeing",
        "active_models",
        "contributing_alert_ids",
        "contributing_alert_count",
        "isolation_score",
        "isolation_percentile",
        "reconstruction_error",
        "reconstruction_threshold",
        "reconstruction_ratio",
        "heuristic_threat_score",
        "heuristics_fired",
    ]
    columns = [column for column in preferred if column in final_alerts.columns]
    display = final_alerts[columns].copy() if columns else final_alerts.copy()
    if "severity" in display.columns:
        display["severity"] = display["severity"].apply(severity_label)
    if "contributing_alert_ids" in display.columns:
        display["contributing_alert_ids"] = display["contributing_alert_ids"].fillna("").astype(str)
    st.dataframe(display, use_container_width=True, hide_index=True, height=420)
    st.caption(f"Final alert file: {FINAL_ALERTS_PATH}")


def _recommended_final_profile_text() -> str:
    return (
        "Recommended defaults — correlation window: 5 min · "
        "Minimum Isolation score: 0.0 · Minimum Isolation percentile: 95 · "
        "Autoencoder threshold multiplier: 1.20 · Minimum heuristic score: 40 · "
        "Weights (Isolation / Autoencoder / Heuristics): 30 / 30 / 40 · "
        "2-model bonus: +15 · 3-model bonus: +25 · "
        "Minimum agreeing detector families: 2 · Final alert threshold: 65."
    )


def _reset_final_widget_state() -> None:
    for key in (
        "final_isolation_min_score",
        "final_isolation_min_percentile",
        "final_autoencoder_multiplier",
        "final_heuristic_min_score",
        "final_weight_isolation",
        "final_weight_autoencoder",
        "final_weight_heuristics",
        "final_consensus_bonus_two",
        "final_consensus_bonus_three",
        "final_minimum_models",
        "final_alert_threshold",
    ):
        st.session_state.pop(key, None)


def render_final_dns_alert_correlation() -> None:
    st.header("Final DNS Alert Correlation")
    st.caption(
        "Run this after Isolation Forest, Autoencoder and heuristic scoring. The final stage does not rerun the three detectors; "
        "it correlates their generated alerts and applies SOC-controlled thresholds to reduce false positives."
    )

    cfg = load_final_config()

    with st.expander("How the final threat score is calculated", expanded=True):
        st.markdown(
            """
**Correlation key:** same `src_ip` + same five-minute DNS window.

**Isolation evidence (I):** the Isolation Forest `score_percentile` on a 0–100 scale, but only when the raw Isolation score and percentile both pass the analyst gates. A percentile of 99 means the window is more anomalous than about 99% of the reference score distribution; it is not a 99% attack probability.

**Autoencoder evidence (A):** based on `reconstruction_error / learned_threshold`. The user threshold is a multiplier of the model's learned threshold. For SOC readability, an accepted ratio of 1.0 maps to 50 evidence points, 2.0 to 75, and 3.0 or higher to 100.

**Heuristic evidence (H):** the rule-based threat score already produced by the heuristic engine.

**Consensus:** two or three independent detector families agreeing on the same source/window adds a configurable bonus.
            """
        )
        st.code(
            "Base = (wI × I) + (wA × A) + (wH × H)\n"
            "Final threat score = min(100, Base + consensus bonus)\n"
            "Final alert = models_agreeing >= minimum_models AND final_score >= final_alert_threshold",
            language="text",
        )

    st.markdown("#### Detection gates")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        isolation_min_score = st.number_input(
            "Minimum Isolation score",
            value=float(cfg.get("isolation_min_score", 0.0)),
            step=0.01,
            format="%.6f",
            help="Final-correlation gate for the raw Isolation Forest anomaly score (-decision_function).",
            key="final_isolation_min_score",
        )
    with g2:
        isolation_min_percentile = st.number_input(
            "Minimum Isolation percentile",
            min_value=0.0,
            max_value=100.0,
            value=float(cfg.get("isolation_min_percentile", 95.0)),
            step=1.0,
            format="%.1f",
            help="Final-correlation gate for the Isolation percentile.",
            key="final_isolation_min_percentile",
        )
    with g3:
        autoencoder_multiplier = st.number_input(
            "Reconstruction error threshold multiplier",
            min_value=1.0,
            value=float(cfg.get("autoencoder_threshold_multiplier", 1.20)),
            step=0.05,
            format="%.2f",
            help="Final-correlation Autoencoder gate: reconstruction_error must be at least learned_threshold × this multiplier.",
            key="final_autoencoder_multiplier",
        )
    with g4:
        heuristic_min_score = st.number_input(
            "Minimum heuristic threat score",
            min_value=0.0,
            max_value=100.0,
            value=float(cfg.get("heuristic_min_threat_score", 40.0)),
            step=5.0,
            format="%.0f",
            help="Heuristic alerts below this score contribute no evidence to the final correlation.",
            key="final_heuristic_min_score",
        )

    st.markdown("#### Threat score pattern")
    w1, w2, w3, c2, c3 = st.columns(5)
    with w1:
        weight_isolation = st.number_input(
            "Isolation weight %",
            min_value=0.0,
            max_value=100.0,
            value=float(cfg.get("weight_isolation", 30.0)),
            step=5.0,
            key="final_weight_isolation",
        )
    with w2:
        weight_autoencoder = st.number_input(
            "Autoencoder weight %",
            min_value=0.0,
            max_value=100.0,
            value=float(cfg.get("weight_autoencoder", 30.0)),
            step=5.0,
            key="final_weight_autoencoder",
        )
    with w3:
        weight_heuristics = st.number_input(
            "Heuristics weight %",
            min_value=0.0,
            max_value=100.0,
            value=float(cfg.get("weight_heuristics", 40.0)),
            step=5.0,
            key="final_weight_heuristics",
        )
    with c2:
        consensus_bonus_two = st.number_input(
            "2-model bonus",
            min_value=0.0,
            max_value=50.0,
            value=float(cfg.get("consensus_bonus_two", 15.0)),
            step=5.0,
            key="final_consensus_bonus_two",
        )
    with c3:
        consensus_bonus_three = st.number_input(
            "3-model bonus",
            min_value=0.0,
            max_value=50.0,
            value=float(cfg.get("consensus_bonus_three", 25.0)),
            step=5.0,
            key="final_consensus_bonus_three",
        )

    f1, f2 = st.columns(2)
    with f1:
        configured_models = int(cfg.get("minimum_agreeing_models", 2))
        minimum_models = st.selectbox(
            "Minimum agreeing detector families",
            options=[1, 2, 3],
            index=[1, 2, 3].index(configured_models) if configured_models in [1, 2, 3] else 1,
            help="2 is the recommended starting point for lower false positives.",
            key="final_minimum_models",
        )
    with f2:
        final_threshold = st.number_input(
            "Final alert threat-score threshold",
            min_value=0.0,
            max_value=100.0,
            value=float(cfg.get("final_alert_threshold", 65.0)),
            step=5.0,
            format="%.0f",
            help="Only correlated windows whose final 0–100 score reaches this value become Final Alerts.",
            key="final_alert_threshold",
        )

    st.info(_recommended_final_profile_text())

    if st.button(
        "▶ Calculate Final Alert Score",
        key="calculate_final_dns_alert_score",
        type="primary",
        use_container_width=True,
    ):
        try:
            result = calculate_final_alerts(
                {
                    "correlation_minutes": 5,
                    "isolation_min_score": isolation_min_score,
                    "isolation_min_percentile": isolation_min_percentile,
                    "autoencoder_threshold_multiplier": autoencoder_multiplier,
                    "heuristic_min_threat_score": heuristic_min_score,
                    "weight_isolation": weight_isolation,
                    "weight_autoencoder": weight_autoencoder,
                    "weight_heuristics": weight_heuristics,
                    "consensus_bonus_two": consensus_bonus_two,
                    "consensus_bonus_three": consensus_bonus_three,
                    "minimum_agreeing_models": minimum_models,
                    "final_alert_threshold": final_threshold,
                }
            )
            st.session_state["final_dns_score_result"] = result
            st.rerun()
        except Exception as exc:
            st.error(f"Final alert calculation failed: {type(exc).__name__}: {exc}")

    if st.button(
        "↺ Reset All Fields to Recommended Defaults",
        key="reset_final_dns_alert_defaults",
        use_container_width=True,
        help="Restores the recommended SOC starting profile and saves it to the final-correlation configuration.",
    ):
        try:
            save_final_config(dict(DEFAULT_CONFIG))
            _reset_final_widget_state()
            st.session_state["final_dns_defaults_reset"] = True
            st.rerun()
        except Exception as exc:
            st.error(f"Unable to reset final-score defaults: {type(exc).__name__}: {exc}")

    if st.session_state.pop("final_dns_defaults_reset", False):
        st.success("Final DNS Alert Correlation fields were reset to the recommended default profile.")
        st.info(_recommended_final_profile_text())

    latest = st.session_state.pop("final_dns_score_result", None)
    if latest:
        st.success(
            f"Final correlation completed: {int(latest.get('correlated_windows', 0)):,} correlated windows evaluated, "
            f"{int(latest.get('final_alerts', 0)):,} final alerts produced."
        )
        with st.expander("Latest final-score calculation summary"):
            st.json(latest)

    _render_final_alerts_window(location="scoring")

# END DNS FINAL CORRELATION GUI

# BEGIN DNS SOC SEVERITY EXPLAINABILITY

def _final_config_for_info() -> dict:
    try:
        return load_final_config()
    except Exception:
        return {}


def _current_autoencoder_threshold_for_info():
    # Prefer the learned model configuration; fall back to the most recent
    # scoring diagnostic threshold if necessary.
    try:
        config = load_model_config() or {}
    except Exception:
        config = {}
    for key in ("threshold", "reconstruction_threshold", "anomaly_threshold"):
        try:
            value = float(config.get(key))
            if value > 0:
                return value
        except Exception:
            pass

    try:
        scored = load_scoring_windows()
        if not scored.empty and "threshold" in scored.columns:
            values = pd.to_numeric(scored["threshold"], errors="coerce").dropna()
            values = values[values > 0]
            if not values.empty:
                return float(values.iloc[-1])
    except Exception:
        pass
    return None


def _autoencoder_runtime_severity_table(threshold):
    # Derive the displayed severity bands from the currently installed
    # feature_utils.classify_severity() instead of hardcoding a possibly stale
    # formula. This keeps the GUI explanation synchronized with the scorer.
    if threshold is None or threshold <= 0:
        return pd.DataFrame()
    try:
        from feature_utils import classify_severity
    except Exception:
        return pd.DataFrame()

    samples = []
    # Native Autoencoder alerts are only created when error > threshold.
    # Scan ratios from just above 1x to 10x and collapse consecutive labels.
    for index in range(1, 9001):
        ratio = 1.0 + (index / 1000.0)
        try:
            label = str(classify_severity(float(threshold) * ratio, float(threshold))).strip().lower()
        except Exception:
            return pd.DataFrame()
        samples.append((ratio, label))

    if not samples:
        return pd.DataFrame()

    rows = []
    start_ratio, current = samples[0]
    previous_ratio = start_ratio
    for ratio, label in samples[1:]:
        if label != current:
            rows.append(
                {
                    "Severity": current.title(),
                    "Reconstruction error / threshold": f"{start_ratio:.3f}× to <{ratio:.3f}×",
                }
            )
            start_ratio = ratio
            current = label
        previous_ratio = ratio
    rows.append(
        {
            "Severity": current.title(),
            "Reconstruction error / threshold": f"≥ {start_ratio:.3f}× (through 10× scan)",
        }
    )
    return pd.DataFrame(rows)


def _render_autoencoder_severity_information() -> None:
    threshold = _current_autoencoder_threshold_for_info()
    final_cfg = _final_config_for_info()
    multiplier = float(final_cfg.get("autoencoder_threshold_multiplier", 1.20) or 1.20)
    effective = float(threshold) * multiplier if threshold is not None else None

    with st.container(border=True):
        st.markdown("#### Autoencoder severity & threshold information")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric(
            "Learned reconstruction threshold",
            "N/A" if threshold is None else f"{float(threshold):.8f}",
        )
        a2.metric("Native alert gate", "error > threshold")
        a3.metric("Final-correlation multiplier", f"{multiplier:.2f}×")
        a4.metric(
            "Effective final-correlation gate",
            "N/A" if effective is None else f"{effective:.8f}",
        )

        st.caption(
            "The Autoencoder first creates a native alert only when reconstruction_error > the learned threshold. "
            "Native severity is assigned by the currently installed feature_utils.classify_severity(error, threshold). "
            "The multiplier shown here belongs to Final DNS Alert Correlation; it does not change the learned model threshold."
        )

        severity_table = _autoencoder_runtime_severity_table(threshold)
        if not severity_table.empty:
            st.dataframe(severity_table, use_container_width=True, hide_index=True)
            st.caption(
                "Severity ranges above are derived at runtime from the installed classify_severity() function, "
                "so the explanation follows the code actually used by the scorer."
            )
        else:
            st.info(
                "The learned threshold or runtime severity function could not be inspected. "
                "The scorer still uses feature_utils.classify_severity(error, threshold)."
            )


def _render_heuristics_severity_information() -> None:
    final_cfg = _final_config_for_info()
    min_final = float(final_cfg.get("heuristic_min_threat_score", 40.0) or 40.0)
    with st.container(border=True):
        st.markdown("#### Heuristic threat score & severity information")
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Final-correlation minimum", f"{min_final:.0f}")
        h2.metric("Medium starts", "40")
        h3.metric("High starts", "70")
        h4.metric("Critical starts", "90")

        weights = pd.DataFrame(
            [
                {"Evidence": "DGA rule", "Threat points": 35},
                {"Evidence": "DNS tunneling rule", "Threat points": 40},
                {"Evidence": "Fast-flux rule", "Threat points": 30},
                {"Evidence": "Resolver failure rule", "Threat points": 15},
                {"Evidence": "Rogue resolver rule", "Threat points": 25},
                {"Evidence": "Suspicious-domain context", "Threat points": 15},
            ]
        )
        st.dataframe(weights, use_container_width=True, hide_index=True)
        st.caption(
            "Heuristic threat score is the sum of fired-rule evidence, capped at 100. "
            "Severity: Low <40 · Medium 40–69 · High 70–89 · Critical 90–100. "
            "The Final-correlation minimum shown above is a second-stage gate and does not change native heuristic severity."
        )

# END DNS SOC SEVERITY EXPLAINABILITY

def page_dns_scoring():
    render_dns_scoring_soc_page(
        render_autoencoder=page_deep_learning_heuristics_scoring,
        render_isolation=render_dns_categorization_scoring,
        render_heuristics=render_dns_heuristics_scoring,
        render_final=render_final_dns_alert_correlation,
    )


def page_deep_learning_heuristics_scoring():
    st.header("DNS Deep learning Scoring")
    st.caption(
        "Validate the live Zeek DNS log, run incremental or full-log scoring, manage the five-minute cron task, "
        "and inspect live Autoencoder diagnostics and alerts."
    )
    _render_autoencoder_severity_information()


    running = is_scoring_running()
    cards = sync_run_once_card_from_status()
    status = read_scoring_status()

    completion_signature = (
        f"{status.get('completed_at_utc')}|{status.get('alerts_written')}|"
        f"{status.get('state')}|{status.get('run_mode')}"
    )
    if not running and status.get("state") in {"finished", "completed", "success", "no_data"}:
        if st.session_state.get("last_scoring_ingest_signature") != completion_signature:
            _refresh_alert_ingest()
            st.session_state["last_scoring_ingest_signature"] = completion_signature

    card_placeholder = st.empty()
    with card_placeholder.container():
        render_scoring_control_cards(cards)

    percentages = [int(card.get("percent", 0) or 0) for card in cards.values()]
    overall_percent = int(status.get("progress_percent", 0) or 0) if running else max(percentages or [0])
    overall_percent = max(0, min(overall_percent, 100))
    overall_stage = status.get("stage", "Waiting") if running else "Latest control operation"
    animated_progress(overall_percent, overall_stage)

    buttons = st.columns(6)
    clicked = None
    with buttons[0]:
        if st.button("1 · Check Live Zeek Logs", use_container_width=True, disabled=running):
            _flush_dns_scoring_live_output("autoencoder", '1 · Check Live Zeek Logs')
            check_live_zeek_log()
            clicked = "check"
    with buttons[1]:
        if st.button("2 · Run Logging Once", use_container_width=True, disabled=running):
            _flush_dns_scoring_live_output("autoencoder", '2 · Run Logging Once')
            result = start_scoring_once()
            if result.get("error"):
                st.error(result["error"])
            clicked = "run"
    with buttons[2]:
        if st.button("3 · Run Scoring From Scratch", use_container_width=True, disabled=running):
            _flush_dns_scoring_live_output("autoencoder", '3 · Run Scoring From Scratch')
            result = start_full_log_scoring()
            if result.get("error"):
                st.error(result["error"])
            clicked = "full_log"
    with buttons[3]:
        if st.button("4 · Activate Logging Task", use_container_width=True, disabled=running):
            _flush_dns_scoring_live_output("autoencoder", '4 · Activate Logging Task')
            result = activate_scoring_cron()
            if result.get("error"):
                st.error(result["error"])
            clicked = "activate"
    with buttons[4]:
        if st.button("5 · Check Task Status", use_container_width=True):
            _flush_dns_scoring_live_output("autoencoder", '5 · Check Task Status')
            check_scoring_cron_status()
            clicked = "status"
    with buttons[5]:
        if st.button("6 · Deactivate Logging Task", use_container_width=True, disabled=running):
            _flush_dns_scoring_live_output("autoencoder", '6 · Deactivate Logging Task')
            result = deactivate_scoring_cron()
            if result.get("error"):
                st.error(result["error"])
            clicked = "deactivate"

    if clicked:
        _refresh_alert_ingest()
        st.rerun()

    st.divider()
    st.subheader("Live recording command output")
    log_text = read_scoring_log(1500 if running else None)
    render_scoring_log_window(log_text, running)

    scored = load_scoring_windows()
    residuals = load_feature_residuals()
    latent = load_latent_vectors()
    alerts = load_live_alerts(500)

    with st.container(border=True):
        render_top_reconstruction_errors(scored)

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render_reconstruction_error_window(scored)
    with right:
        with st.container(border=True):
            render_feature_weight_heatmap(scored, residuals)

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            render_latent_heatmap(latent)
    with right:
        with st.container(border=True):
            render_detected_live_alerts(alerts)

    if running:
        time.sleep(1.5)
        st.rerun()




def page_dashboard(alerts_df: pd.DataFrame):
    st.title("DNS Threat Monitoring Console")
    st.caption(
        "SOC dashboard for DNS Autoencoder and heuristic alert monitoring."
    )

    col_ingest, col_path = st.columns([1, 3])

    with col_ingest:
        if st.button("Ingest alerts now"):
            result = ingest_alert_file(ALERT_JSONL_PATH, incremental=True)
            st.success(
                f"Inserted {result['inserted']} new alerts from "
                f"{result['total_read']} records."
            )
            if result.get("error"):
                st.error(result["error"])
            st.cache_data.clear()

    with col_path:
        st.info(f"Alert source: `{ALERT_JSONL_PATH}`")

    if alerts_df.empty:
        st.warning("No alerts are stored in the GUI database.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total alerts", len(alerts_df))
    c2.metric(
        "High alerts",
        int((alerts_df["severity"].str.lower() == "high").sum()),
    )
    c3.metric(
        "Medium alerts",
        int((alerts_df["severity"].str.lower() == "medium").sum()),
    )
    c4.metric("Source IPs", alerts_df["src_ip"].nunique())
    c5.metric(
        "Average threat score",
        round(float(alerts_df["threat_score"].fillna(0).mean()), 1),
    )

    data = alerts_df.copy()
    data["timestamp_dt"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
        utc=True,
    )
    data["hour"] = data["timestamp_dt"].dt.floor("h")

    left, right = st.columns(2)
    with left:
        st.subheader("Alerts over time")
        timeline = data.groupby("hour").size().rename("alerts")
        st.line_chart(timeline)

    with right:
        st.subheader("Severity distribution")
        st.bar_chart(data["severity"].value_counts())

    left, right = st.columns(2)
    with left:
        st.subheader("Top suspicious source IPs")
        top_src = data["src_ip"].value_counts().head(10).reset_index()
        top_src.columns = ["src_ip", "alert_count"]
        dataframe_or_info(top_src, "No source-IP data.")

    with right:
        st.subheader("Detection methods")
        method_counts = {}
        for value in data["detection_methods_json"]:
            methods = parse_json(value, {})
            for key, enabled in methods.items():
                if enabled:
                    method_counts[key] = method_counts.get(key, 0) + 1

        if method_counts:
            method_df = pd.Series(method_counts).sort_values(ascending=False)
            st.bar_chart(method_df)
        else:
            st.info("No detection-method data.")

    # BEGIN DNS DASHBOARD TOP SUSPICIOUS DOMAINS
    st.subheader("Top 20 suspicious domains")
    st.caption(
        "Domains are ranked by maximum suspicion score, number of linked alerts, and observed query volume."
    )
    try:
        top_domains = read_sql(
            """
            SELECT
                RTRIM(LOWER(TRIM(ad.query)), '.') AS domain,
                COUNT(DISTINCT ad.alert_uid) AS linked_alerts,
                SUM(COALESCE(ad.count, 0)) AS observed_queries,
                MAX(COALESCE(ad.suspicion_score, 0)) AS max_suspicion_score,
                ROUND(MAX(COALESCE(ad.max_entropy, 0)), 4) AS max_entropy,
                MAX(a.timestamp) AS latest_alert_utc
            FROM alert_domains AS ad
            LEFT JOIN alerts AS a ON a.alert_uid = ad.alert_uid
            WHERE ad.query IS NOT NULL AND TRIM(ad.query) <> ''
            GROUP BY RTRIM(LOWER(TRIM(ad.query)), '.')
            ORDER BY max_suspicion_score DESC, linked_alerts DESC, observed_queries DESC, domain ASC
            LIMIT 20
            """
        )
    except Exception:
        top_domains = pd.DataFrame()
    dataframe_or_info(top_domains, "No suspicious-domain evidence is currently stored.")
    # END DNS DASHBOARD TOP SUSPICIOUS DOMAINS

    st.subheader("Latest alerts")
    latest = data.copy()
    latest["detection_type"] = latest["detection_methods_json"].apply(
        detection_type_from_json
    )
    dataframe_or_info(
        latest[
            [
                "id",
                "timestamp",
                "src_ip",
                "severity",
                "threat_score",
                "status",
                "reconstruction_error",
                "detection_type",
            ]
        ].head(20),
        "No alerts.",
    )


def page_alerts(alerts_df: pd.DataFrame):
    st.title("🚨 Alerts")
    _render_final_alerts_window(location="alerts")
    st.divider()

    signed_in_user = st.session_state.get("dns_auth_user") or {}
    can_purge_alerts = str(signed_in_user.get("role") or "") == "admin"

    purge_result = st.session_state.pop("alert_purge_result", None)
    if purge_result:
        st.success(
            "Alert purge completed: "
            f"{int(purge_result.get('alerts_deleted', 0)):,} database alerts, "
            f"{int(purge_result.get('alert_domains_deleted', 0)):,} domain rows, "
            f"and {int(purge_result.get('files_deleted', 0)):,} alert files removed."
        )

    if can_purge_alerts:
        control_col, note_col = st.columns([1.35, 4.65])
        with control_col:
            if st.button("🗑 Clear all alerts", key="clear_all_alerts_button", type="primary", use_container_width=True, help="Permanently delete all DNS alert families from the database and purge /data/dns-ml/alerts."):
                try:
                    result = clear_all_alerts(ALERT_JSONL_PATH.parent)
                    final_reset = reset_final_runtime_state()
                    result["final_correlation_reset"] = True
                    st.session_state["alert_purge_result"] = result
                    st.session_state.pop("selected_alert_id", None)
                    st.cache_data.clear()
                    st.rerun()
                except Exception as exc:
                    st.error(f"Unable to clear alerts: {type(exc).__name__}: {exc}")
        with note_col:
            st.warning(
                "This action is permanent. It removes final-correlation, categorization, deep-learning, and heuristic alert files/records, but does not change models, Zeek logs, learning data, scoring diagnostics, scoring cursors, or the saved final-score tuning configuration."
            )
    else:
        st.caption("Alert deletion is restricted to administrators.")

    if alerts_df.empty:
        st.info("No alerts are currently stored. Alert Details and Source IP Investigation remain empty until new alerts are generated.")
        return

    st.info(f"Database alert time range: {alerts_df['timestamp'].min()} → {alerts_df['timestamp'].max()}")
    filtered = filter_alerts(alerts_df)
    cat_mask = _categorization_alert_mask(filtered)
    heur_mask = _heuristics_alert_mask(filtered) & ~cat_mask
    deep_mask = ~(cat_mask | heur_mask)

    categorization = filtered[cat_mask].copy()
    deep = filtered[deep_mask].copy()
    heuristics = filtered[heur_mask].copy()

    st.subheader("DNS categorization Alerts")
    st.caption("Alerts generated only by the Isolation Forest categorization scoring pipeline.")
    st.write(f"Showing **{len(categorization)}** categorization alerts.")
    _render_alert_family_table(categorization, "categorization", "categorization")

    st.divider()
    st.subheader("DNS Deep learning Alerts")
    st.caption("Alerts generated by Autoencoder reconstruction scoring only.")
    st.write(f"Showing **{len(deep)}** deep-learning alerts.")
    _render_alert_family_table(deep, "deep", "deep")

    st.divider()
    st.subheader("DNS heuristics Alerts")
    st.caption("Alerts generated only by rule-based DNS heuristics: DGA, tunneling, fast-flux, resolver failure, and rogue resolver.")
    st.write(f"Showing **{len(heuristics)}** heuristic alerts.")
    _render_alert_family_table(heuristics, "heuristics", "heuristics")

def page_alert_details(alerts_df: pd.DataFrame):
    st.title("🔎 Alert Details")
    if alerts_df.empty:
        st.warning("No alerts were found.")
        return

    alerts_df = _ensure_public_alert_ids(alerts_df)
    public_ids = alerts_df["public_alert_id"].astype(str).tolist()
    selected_default = str(st.session_state.get("selected_alert_id", public_ids[0]))
    if selected_default not in public_ids:
        selected_default = public_ids[0]

    selected_public_id = st.selectbox(
        "Alert ID",
        public_ids,
        index=public_ids.index(selected_default),
    )
    selected_rows = alerts_df[
        alerts_df["public_alert_id"].astype(str).eq(str(selected_public_id))
    ]
    if selected_rows.empty:
        st.error("The selected alert could not be resolved.")
        return

    db_alert_id = int(selected_rows.iloc[0]["id"])
    alert = get_alert_by_id(db_alert_id)

    if not alert:
        st.error("The alert could not be found.")
        return

    st.caption(f"Alert ID: {selected_public_id}")

    methods = parse_json(alert.get("detection_methods_json"), {})
    top_features = parse_json(alert.get("top_contributing_features_json"), [])
    possible_causes = parse_json(alert.get("possible_causes_json"), [])
    raw = parse_json(alert.get("raw_json"), {})
    alert_type = str(alert.get("alert_type") or raw.get("alert_type") or "")
    is_cat = alert_type == "categorization_forest" or bool(methods.get("categorization_forest"))
    is_heur = (not is_cat) and (
        alert_type in {"heuristics_rule", "rule_based_suspicious", "high_confidence_suspicious"}
        or any(bool(methods.get(name)) for name in ("heuristic_dga", "heuristic_tunnel", "heuristic_fastflux", "heuristic_resolver_failure", "heuristic_rogue_resolver"))
    )

    if is_cat:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Source IP", alert.get("src_ip"))
        c2.metric("Severity", severity_label(alert.get("severity")))
        c3.metric("Threat score", alert.get("threat_score"))
        c4.metric("Isolation score", round(float(alert.get("isolation_score") or raw.get("isolation_score") or 0), 6))
        c5.metric("Isolation threshold", round(float(alert.get("isolation_threshold") or raw.get("isolation_threshold") or 0), 6))
        st.caption(f"Detection family: DNS categorization · {raw.get('id') or alert.get('alert_uid')}")
    elif is_heur:
        fired = [name for name in ("heuristic_dga", "heuristic_tunnel", "heuristic_fastflux", "heuristic_resolver_failure", "heuristic_rogue_resolver") if methods.get(name)]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Source IP", alert.get("src_ip"))
        c2.metric("Severity", severity_label(alert.get("severity")))
        c3.metric("Threat score", alert.get("threat_score"))
        c4.metric("Rules fired", len(fired))
        c5.metric("Detection type", "heuristics")
        st.caption("Detection family: DNS heuristics · " + (", ".join(fired) if fired else "historical heuristic alert"))
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Source IP", alert.get("src_ip"))
        c2.metric("Severity", severity_label(alert.get("severity")))
        c3.metric("Threat score", alert.get("threat_score"))
        c4.metric("Reconstruction error", round(float(alert.get("reconstruction_error") or 0), 4))
        c5.metric("Threshold", round(float(alert.get("threshold") or 0), 4))
        st.caption("Detection family: DNS Deep learning · Autoencoder")

    st.subheader("Detection methods")
    if is_cat:
        d1, d2 = st.columns(2)
        d1.metric("DNS categorization forest", "YES" if methods.get("categorization_forest") else "NO")
        d2.metric("Isolation Forest", "YES" if methods.get("isolation_forest") else "NO")
    else:
        show_detection_methods(methods)

    st.subheader("Suspicious domains")
    domains_df = get_domains_for_alert(alert["alert_uid"])
    if not domains_df.empty:
        domains_df["reasons"] = domains_df["reasons_json"].apply(lambda value: ", ".join(parse_json(value, [])))
        domains_df = domains_df.drop(columns=["reasons_json"])
    dataframe_or_info(domains_df, "No suspicious domains were recorded.")

    if not is_heur:
        st.subheader("Top contributing features")
        dataframe_or_info(pd.DataFrame(top_features), "No contributing-feature information was recorded.")

    st.subheader("Possible causes")
    if possible_causes:
        for cause in possible_causes:
            st.write(f"- {cause}")
    else:
        st.info("No possible causes were recorded.")

    st.divider()
    st.subheader("Analyst workflow")
    status_options = ["New", "Investigating", "Confirmed suspicious", "False positive", "Whitelisted", "Closed"]
    current_status = alert.get("status") or "New"
    if current_status not in status_options:
        current_status = "New"
    new_status = st.selectbox("Status", status_options, index=status_options.index(current_status))
    analyst_note = st.text_area("Analyst note", value=alert.get("analyst_note") or "", height=110)
    if st.button("Save analyst update"):
        update_alert_status(db_alert_id, new_status, analyst_note)
        st.success("The alert was updated.")
        st.cache_data.clear()

    with st.expander("Raw alert JSON"):
        st.json(raw)

def page_source_ip(alerts_df: pd.DataFrame):
    st.title("🖥️ Source IP Investigation")
    if alerts_df.empty:
        st.warning("No alerts were found.")
        return

    selected_ip = st.selectbox("Source IP", sorted(alerts_df["src_ip"].dropna().unique().tolist()))
    data = alerts_df[alerts_df["src_ip"] == selected_ip].copy()
    data["timestamp_dt"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    data["detection_type"] = data["detection_methods_json"].apply(detection_type_from_json)
    cat_mask = _categorization_alert_mask(data)
    heur_mask = _heuristics_alert_mask(data) & ~cat_mask
    deep_mask = ~(cat_mask | heur_mask)

    data["alert_family"] = "DNS Deep learning"
    data.loc[cat_mask, "alert_family"] = "DNS categorization"
    data.loc[heur_mask, "alert_family"] = "DNS heuristics"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("All alerts", len(data))
    c2.metric("Categorization", int(cat_mask.sum()))
    c3.metric("Deep learning", int(deep_mask.sum()))
    c4.metric("Heuristics", int(heur_mask.sum()))
    c5.metric("Maximum threat score", int(data["threat_score"].fillna(0).max()) if not data.empty else 0)

    st.subheader("Alert timeline")
    valid_time = data.dropna(subset=["timestamp_dt"])
    if not valid_time.empty:
        timeline = valid_time.groupby(valid_time["timestamp_dt"].dt.floor("h")).size().rename("alerts")
        st.line_chart(timeline)
    else:
        st.info("No valid alert timestamps are available.")

    st.subheader("Alerts for this source IP")
    preferred = ["id", "timestamp", "alert_family", "alert_type", "severity", "threat_score", "status", "isolation_score", "reconstruction_error", "detection_type"]
    columns = [column for column in preferred if column in data.columns]
    dataframe_or_info(data[columns], "No alerts were found for this source IP.")

def page_model_health():
    st.title("⚙️ Model Integrity")
    config = load_model_config()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Database", "OK" if DB_PATH.exists() else "Missing")
    c2.metric(
        "Alert JSONL",
        "OK" if ALERT_JSONL_PATH.exists() else "Missing",
    )
    c3.metric(
        "Model config",
        "OK" if MODEL_CONFIG_PATH.exists() else "Missing",
    )
    c4.metric(
        "Merged Zeek data",
        "OK" if ZEEK_MERGED_PATH.exists() else "Missing",
    )

    if config:
        st.subheader("Model configuration")
        st.json(config)
    else:
        st.warning("The model configuration is missing or unreadable.")

    st.subheader("Ingest state")
    dataframe_or_info(
        read_sql("SELECT * FROM ingest_state"),
        "No ingest state was found.",
    )

    render_isolation_model_integrity()

def page_rules():
    st.title("📏 Rule Configuration")

    rules_df = read_sql(
        """
        SELECT rule_name, enabled, description, rule_json, updated_at_utc
        FROM rule_config
        ORDER BY rule_name
        """
    )
    dataframe_or_info(rules_df, "No rules were found.")

    if rules_df.empty:
        return

    selected_rule = st.selectbox(
        "Rule",
        rules_df["rule_name"].tolist(),
    )
    selected_row = rules_df[
        rules_df["rule_name"] == selected_rule
    ].iloc[0]

    enabled = st.checkbox(
        "Enabled",
        value=bool(selected_row["enabled"]),
    )
    st.json(parse_json(selected_row["rule_json"], {}))

    if st.button("Save rule state"):
        execute(
            """
            UPDATE rule_config
            SET enabled = ?, updated_at_utc = ?
            WHERE rule_name = ?
            """,
            (
                1 if enabled else 0,
                datetime.now(timezone.utc).isoformat(),
                selected_rule,
            ),
        )
        st.success("The rule state was updated.")


def page_reports(alerts_df: pd.DataFrame):
    st.title("📄 Reports")

    if alerts_df.empty:
        st.warning("No alerts were found.")
        return

    export_df = alerts_df.copy()
    export_df["detection_type"] = export_df[
        "detection_methods_json"
    ].apply(detection_type_from_json)

    columns = [
        "id",
        "timestamp",
        "src_ip",
        "severity",
        "threat_score",
        "status",
        "reconstruction_error",
        "threshold",
        "detection_type",
        "created_at_utc",
    ]

    st.download_button(
        "Download alerts CSV",
        data=export_df[columns].to_csv(index=False).encode("utf-8"),
        file_name="dns_alert_report.csv",
        mime="text/csv",
    )

    st.subheader("Summary by severity")
    dataframe_or_info(
        alerts_df.groupby("severity").size().reset_index(name="count"),
        "No severity data.",
    )

    st.subheader("Summary by source IP")
    source_summary = (
        alerts_df.groupby("src_ip")
        .agg(
            alerts=("id", "count"),
            max_score=("threat_score", "max"),
            avg_score=("threat_score", "mean"),
        )
        .reset_index()
        .sort_values("alerts", ascending=False)
    )
    dataframe_or_info(source_summary, "No source-IP data.")



# BEGIN DNS CONSOLE TOP HEADER

import base64
import re
# BEGIN DNS CONSOLE HERO NAVIGATION UI

DNS_PAGES = [
    "Dashboard",
    "DNS Log Preparation",
    "Learning Navigator",
    "DNS Scoring",
    "Alerts",
    "Alert Details",
    "Source IP Investigation",
    "Model Integrity",
    "Rules",
]


def _dns_image_to_data_uri(path: Path) -> str:
    if not path.exists():
        return ""

    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/png")

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _dns_nav_key(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def render_dns_console_style() -> None:
    st.markdown(
        """
        <style>

        /* -----------------------------------------------------
           Page spacing
           ----------------------------------------------------- */
        .block-container {
            max-width: 100% !important;
            padding-top: 1.15rem !important;
            padding-left: 2.0rem !important;
            padding-right: 2.0rem !important;
            padding-bottom: 2rem !important;
        }

        /* -----------------------------------------------------
           Large hero header
           ----------------------------------------------------- */
        .dns-console-hero {
            position: relative;
            display: grid;
            grid-template-columns: minmax(190px, 250px) 1fr;
            align-items: center;
            gap: 2.4rem;
            min-height: 220px;
            padding: 22px 32px 22px 24px;
            border: 2px solid rgba(14, 165, 233, 0.65);
            border-radius: 22px;
            overflow: hidden;

            background:
                radial-gradient(
                    circle at 88% 45%,
                    rgba(14, 165, 233, 0.17),
                    transparent 27%
                ),
                linear-gradient(
                    110deg,
                    #071525 0%,
                    #0b2036 42%,
                    #113d61 100%
                );

            box-shadow:
                inset 0 0 40px rgba(14, 165, 233, 0.04),
                0 14px 40px rgba(0, 0, 0, 0.20);

            margin-bottom: 16px;
        }

        .dns-console-hero::before {
            content: "";
            position: absolute;
            right: -110px;
            top: -165px;
            width: 570px;
            height: 570px;
            border-radius: 50%;
            opacity: 0.55;

            background:
                repeating-radial-gradient(
                    circle at center,
                    rgba(14, 165, 233, 0.13) 0px,
                    rgba(14, 165, 233, 0.13) 1px,
                    transparent 2px,
                    transparent 20px
                ),
                radial-gradient(
                    circle at center,
                    transparent 0%,
                    transparent 47%,
                    rgba(14, 165, 233, 0.40) 47.5%,
                    transparent 48.5%,
                    transparent 62%,
                    rgba(14, 165, 233, 0.22) 62.5%,
                    transparent 63.5%
                );

            pointer-events: none;
        }

        .dns-console-hero::after {
            content: "";
            position: absolute;
            right: 170px;
            bottom: -80px;
            width: 550px;
            height: 190px;
            border-top: 1px solid rgba(34, 211, 238, 0.20);
            border-radius: 50%;
            transform: rotate(-13deg);

            box-shadow:
                0 -40px 0 -39px rgba(34, 211, 238, 0.20),
                0 -80px 0 -79px rgba(34, 211, 238, 0.13);

            pointer-events: none;
        }

        /* -----------------------------------------------------
           Enlarged logo panel
           ----------------------------------------------------- */
        .dns-console-logo-panel {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 175px;

            border: 1px solid rgba(56, 189, 248, 0.45);
            border-radius: 18px;

            background:
                radial-gradient(
                    circle at center,
                    rgba(14, 165, 233, 0.16),
                    transparent 67%
                ),
                rgba(5, 19, 34, 0.62);

            box-shadow:
                inset 0 0 30px rgba(14, 165, 233, 0.06);

            z-index: 2;
        }

        .dns-console-logo-panel img {
            width: 138px;
            max-width: 78%;
            height: auto;
            object-fit: contain;

            filter:
                drop-shadow(
                    0 10px 18px rgba(0, 0, 0, 0.25)
                );
        }

        /* -----------------------------------------------------
           Hero title
           ----------------------------------------------------- */
        .dns-console-title-area {
            position: relative;
            z-index: 2;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .dns-console-title {
            margin: 0;
            padding: 0;

            color: #f8fafc;

            font-size:
                clamp(
                    2.3rem,
                    3.4vw,
                    4rem
                );

            font-weight: 800;
            line-height: 1.05;
            letter-spacing: 0.015em;

            text-shadow:
                0 3px 10px rgba(0, 0, 0, 0.25);
        }

        .dns-console-subtitle {
            margin-top: 22px;

            color: #a9bdd2;

            font-size:
                clamp(
                    1rem,
                    1.35vw,
                    1.35rem
                );

            font-weight: 500;
            line-height: 1.45;
            max-width: 950px;
        }

        /* -----------------------------------------------------
           Main navigation container
           ----------------------------------------------------- */
        .st-key-dns_main_navigation {
            margin-top: 5px;
            margin-bottom: 20px;
        }

        .st-key-dns_main_navigation
        [data-testid="stHorizontalBlock"] {
            gap: 0.65rem !important;
        }

        /* -----------------------------------------------------
           Large rectangular navigation buttons
           ----------------------------------------------------- */
        .st-key-dns_main_navigation
        .stButton > button {
            width: 100%;
            min-height: 62px;
            padding: 0.55rem 0.65rem;

            border-radius: 10px;

            border:
                1px solid
                rgba(56, 189, 248, 0.48);

            background:
                linear-gradient(
                    180deg,
                    rgba(20, 62, 97, 0.92) 0%,
                    rgba(11, 39, 66, 0.95) 100%
                );

            color: #e7eef8;

            font-size: 0.96rem;
            font-weight: 700;
            letter-spacing: 0.01em;

            transition:
                transform 0.15s ease,
                border-color 0.15s ease,
                background 0.15s ease,
                box-shadow 0.15s ease;

            box-shadow:
                inset 0 1px 0 rgba(255, 255, 255, 0.035);
        }

        .st-key-dns_main_navigation
        .stButton > button:hover {
            transform: translateY(-2px);
            border-color: #38bdf8;

            background:
                linear-gradient(
                    180deg,
                    #174f79,
                    #0d3557
                );

            color: white;

            box-shadow:
                0 8px 18px rgba(14, 165, 233, 0.15);
        }

        /* Active page button = Streamlit primary button */
        .st-key-dns_main_navigation
        .stButton > button[kind="primary"],
        .st-key-dns_main_navigation
        .stButton > button[data-testid="stBaseButton-primary"] {
            color: white !important;

            border:
                1px solid #22d3ee !important;

            background:
                linear-gradient(
                    180deg,
                    #17608e 0%,
                    #12446a 100%
                ) !important;

            box-shadow:
                0 4px 0 #22d3ee,
                0 9px 20px rgba(34, 211, 238, 0.13) !important;
        }

        .dns-after-navigation {
            height: 1px;
            margin-top: 6px;

            border-bottom:
                1px solid rgba(148, 163, 184, 0.22);
        }

        /* -----------------------------------------------------
           Responsive behavior
           ----------------------------------------------------- */
        @media (max-width: 1050px) {
            .dns-console-hero {
                grid-template-columns: 185px 1fr;
                gap: 1.3rem;
                min-height: 190px;
            }

            .dns-console-logo-panel {
                min-height: 150px;
            }

            .dns-console-logo-panel img {
                width: 115px;
            }
        }

        @media (max-width: 720px) {
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }

            .dns-console-hero {
                grid-template-columns: 1fr;
                padding: 18px;
                text-align: center;
            }

            .dns-console-logo-panel {
                max-width: 180px;
                width: 100%;
                margin: 0 auto;
            }

            .dns-console-title-area {
                align-items: center;
            }

            .dns-console-subtitle {
                margin-top: 12px;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def render_dns_console_header() -> None:
    logo_uri = _dns_image_to_data_uri(DNS_CONSOLE_LOGO_PATH)

    if logo_uri:
        logo_html = f'<img src="{logo_uri}" alt="DNS Console">'
    else:
        logo_html = (
            '<div style="'
            'font-size:42px;'
            'font-weight:800;'
            'color:#67e8f9;'
            '">DNS</div>'
        )

    # Keep this HTML left-aligned. If it is indented, Streamlit Markdown
    # interprets it as a Markdown code block and shows the tags literally.
    hero_html = f"""<div class="dns-console-hero">
<div class="dns-console-logo-panel">
{logo_html}
</div>
<div class="dns-console-title-area">
<div class="dns-console-title">DNS Console</div>
<div class="dns-console-subtitle">DNS anomaly detection, scoring, correlation and investigation</div>
</div>
</div>"""

    st.markdown(
        hero_html,
        unsafe_allow_html=True,
    )



def render_dns_navigation(allowed_pages: list[str] | None = None) -> str:
    visible_pages = [page for page in DNS_PAGES if allowed_pages is None or page in allowed_pages]
    if not visible_pages:
        st.error("This account has no authorized GUI pages.")
        return ""
    if "dns_active_page" not in st.session_state:
        # Preserve the current old-radio selection on the first run when possible.
        st.session_state["dns_active_page"] = st.session_state.get(
            "dns_top_navigation",
            visible_pages[0],
        )

    current_page = st.session_state["dns_active_page"]

    if current_page not in visible_pages:
        current_page = visible_pages[0]
        st.session_state["dns_active_page"] = current_page

    width_by_page = {
        "Dashboard": 1.10,
        "DNS Log Preparation": 1.55,
        "Learning Navigator": 1.55,
        "DNS Scoring": 1.10,
        "Alerts": .90,
        "Alert Details": 1.15,
        "Source IP Investigation": 1.70,
        "Model Integrity": 1.25,
        "Rules": .85,
    }
    widths = [width_by_page[page] for page in visible_pages]

    with st.container(key="dns_main_navigation"):
        columns = st.columns(widths, gap="small")

        for column, page_name in zip(columns, visible_pages):
            is_active = current_page == page_name

            with column:
                clicked = st.button(
                    page_name,
                    key="dns_nav_" + _dns_nav_key(page_name),
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                )

                if clicked and page_name != current_page:
                    st.session_state["dns_active_page"] = page_name
                    st.rerun()

    st.markdown(
        '<div class="dns-after-navigation"></div>',
        unsafe_allow_html=True,
    )

    return st.session_state["dns_active_page"]

# END DNS CONSOLE HERO NAVIGATION UI

# END DNS CONSOLE TOP HEADER


def main():
    auth_user = render_login_gate(APP_NAME)
    if not auth_user:
        return

    render_dns_console_style()
    render_dns_console_header()
    if render_account_toolbar(auth_user):
        render_user_management_page(auth_user)
        render_solution_footer()
        return

    allowed_pages = allowed_pages_for_user(auth_user)
    page = render_dns_navigation(allowed_pages)
    if not page or (allowed_pages is not None and page not in allowed_pages):
        st.error("This account is not authorized to access the requested page.")
        render_solution_footer()
        return
    st.divider()

    alerts_df = load_alerts()

    if page == "Dashboard":
        page_dashboard(alerts_df)
    elif page == "DNS Log Preparation":
        render_dns_log_preparation_page()
    elif page == "Learning Navigator":
        page_learning_navigator()
    elif page == "DNS Scoring":
        page_dns_scoring()
    elif page == "Alerts":
        page_alerts(alerts_df)
    elif page == "Alert Details":
        page_alert_details(alerts_df)
    elif page == "Source IP Investigation":
        page_source_ip(alerts_df)
    elif page == "Model Integrity":
        page_model_health()
    elif page == "Rules":
        page_rules()

    render_solution_footer()

if __name__ == "__main__":
    main()
