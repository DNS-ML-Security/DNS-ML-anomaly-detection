# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src import dns_log_preparation as preparation
from src.ui_helpers import animated_progress


DURATION_OPTIONS = {
    "1 hour": 3600,
    "6 hours": 6 * 3600,
    "12 hours": 12 * 3600,
    "24 hours": 24 * 3600,
    "3 days": 3 * 24 * 3600,
    "7 days": 7 * 24 * 3600,
    "14 days": 14 * 24 * 3600,
    "21 days": 21 * 24 * 3600,
    "30 days (maximum)": 30 * 24 * 3600,
}
MAX_GUI_DOWNLOAD_BYTES = 256 * 1024**2


def _escape(value: Any) -> str:
    return html.escape(str(value if value not in (None, "") else "Not available"))


def _format_bytes(value: Any) -> str:
    try:
        number = float(value or 0)
    except Exception:
        number = 0.0
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024 or unit == "TiB":
            return f"{number:.2f} {unit}" if unit != "B" else f"{int(number)} B"
        number /= 1024
    return f"{number:.2f} TiB"


def _format_duration(seconds: Any) -> str:
    try:
        total = max(0, int(float(seconds or 0)))
    except Exception:
        total = 0
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .dns-prep-hero {
            padding: 1rem 1.15rem;
            margin-bottom: .9rem;
            border: 1px solid #168FC4;
            border-radius: 15px;
            background: linear-gradient(145deg, #10283D 0%, #0B1B2D 100%);
        }
        .dns-prep-title {
            color: #F5F9FC;
            font-size: clamp(1.25rem, 1.9vw, 1.7rem);
            font-weight: 850;
            line-height: 1.18;
        }
        .dns-prep-subtitle {
            margin-top: .35rem;
            color: #AFC8DC;
            font-size: clamp(.76rem, .9vw, .9rem);
            line-height: 1.45;
        }
        .dns-prep-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .7rem;
            margin: .4rem 0 .9rem;
        }
        .dns-prep-card {
            min-width: 0;
            padding: .64rem .72rem;
            border: 1px solid #167FAF;
            border-radius: 12px;
            background: linear-gradient(145deg, #10283D, #0E2A47);
        }
        .dns-prep-label {
            color: #8EB0C7;
            font-size: clamp(.62rem, .7vw, .72rem);
            font-weight: 780;
            letter-spacing: .045em;
            text-transform: uppercase;
        }
        .dns-prep-value {
            margin-top: .28rem;
            color: #F5F9FC;
            font-size: clamp(.86rem, 1.05vw, 1.08rem);
            font-weight: 830;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }
        .dns-prep-state-active, .dns-prep-state-completed { color: #35E0A1; }
        .dns-prep-state-paused { color: #FFD166; }
        .dns-prep-state-failed { color: #FF6B7E; }
        .dns-prep-path {
            padding: .6rem .7rem;
            border: 1px solid rgba(22, 127, 175, .7);
            border-radius: 10px;
            background: #081521;
            color: #B8D6E7;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: clamp(.65rem, .72vw, .76rem);
            overflow-wrap: anywhere;
        }
        @media (max-width: 900px) {
            .dns-prep-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 580px) {
            .dns-prep-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _state_css(state: str) -> str:
    normalized = str(state or "inactive").lower()
    if normalized in {"active", "starting", "pcap_uploading", "pcap_queued", "pcap_processing", "finalizing"}:
        return "dns-prep-state-active"
    if normalized == "completed":
        return "dns-prep-state-completed"
    if normalized == "paused":
        return "dns-prep-state-paused"
    if normalized == "failed":
        return "dns-prep-state-failed"
    return ""


def _render_cards(state: dict[str, Any]) -> None:
    current = str(state.get("state") or "inactive")
    mode = {"pcap": "PCAP conversion", "live": "Live learning capture"}.get(
        str(state.get("mode") or ""),
        "Not selected",
    )
    elapsed = preparation.effective_elapsed_seconds(state)
    cards = [
        ("State", current.replace("_", " ").title(), _state_css(current)),
        ("Mode", mode, ""),
        ("Captured / generated", _format_bytes(state.get("bytes_used")), ""),
        ("Active capture time", _format_duration(elapsed), ""),
    ]
    blocks = ['<div class="dns-prep-grid">']
    for label, value, css in cards:
        blocks.append(
            '<div class="dns-prep-card">'
            f'<div class="dns-prep-label">{_escape(label)}</div>'
            f'<div class="dns-prep-value {css}">{_escape(value)}</div>'
            '</div>'
        )
    blocks.append("</div>")
    st.markdown("".join(blocks), unsafe_allow_html=True)


def _render_output(state: dict[str, Any]) -> None:
    output = preparation.output_path_from_state(state)
    if output is None:
        return
    size = output.stat().st_size
    st.success(
        f"dns.log is ready: {int(state.get('dns_records', 0) or 0):,} records · "
        f"{_format_bytes(size)} · SHA-256 `{state.get('output_sha256') or 'Not available'}`"
    )
    st.markdown(f'<div class="dns-prep-path">{_escape(output)}</div>', unsafe_allow_html=True)
    if size <= MAX_GUI_DOWNLOAD_BYTES:
        with output.open("rb") as handle:
            st.download_button(
                "Download dns.log",
                data=handle,
                file_name="dns.log",
                mime="application/x-ndjson",
                use_container_width=True,
                key=f"dns_prep_download_{state.get('job_id')}",
            )
    else:
        st.info(
            "The consolidated dns.log is larger than the safe 256 MiB browser-download limit. "
            "Use the protected server path shown above, or copy it through your approved administrative transfer method."
        )


def _render_runtime_status() -> None:
    state = preparation.read_state()
    _render_cards(state)
    st.caption(str(state.get("message") or ""))
    if state.get("error"):
        st.error(str(state["error"]))
    current = str(state.get("state") or "inactive")
    if current in preparation.BUSY_STATES:
        animated_progress(
            max(0, min(100, int(state.get("progress_percent", 0) or 0))),
            str(state.get("message") or "DNS log preparation is running"),
        )
    if state.get("mode") == "live" and current in preparation.LIVE_CAPTURE_STATES:
        duration = max(1, int(state.get("duration_seconds", 0) or 0))
        elapsed = preparation.effective_elapsed_seconds(state)
        remaining = max(0, duration - elapsed)
        st.caption(
            f"Configured period: {_format_duration(duration)} · Active time: {_format_duration(elapsed)} · "
            f"Remaining: {_format_duration(remaining)} · Hard storage limit: 10 GiB"
        )
    previous = str(st.session_state.get("dns_prep_fragment_state") or "")
    st.session_state["dns_prep_fragment_state"] = current
    if previous in preparation.BUSY_STATES and current not in preparation.BUSY_STATES:
        st.rerun()


if hasattr(st, "fragment"):
    _render_runtime_status_fragment = st.fragment(run_every="3s")(_render_runtime_status)
else:
    _render_runtime_status_fragment = _render_runtime_status


def _render_pcap_controls() -> None:
    state = preparation.read_state()
    busy = str(state.get("state") or "") in preparation.BUSY_STATES
    scoring_conflict = preparation.live_scoring_conflict_reason()
    st.markdown("#### Convert one PCAP to Zeek JSON dns.log")
    st.caption(
        "The uploaded capture is stored with mode 0600, processed by Zeek in an isolated job directory, "
        "and deleted after conversion. Maximum GUI upload: 1 GiB. Starting a new job replaces the previous prepared output."
    )
    uploaded = st.file_uploader(
        "Packet capture",
        type=["pcap", "pcapng"],
        accept_multiple_files=False,
        disabled=busy or bool(scoring_conflict),
        key="dns_prep_pcap_upload",
    )
    if scoring_conflict:
        st.warning(scoring_conflict)
    if st.button(
        "Upload PCAP and generate dns.log",
        key="dns_prep_start_pcap",
        type="primary",
        use_container_width=True,
        disabled=busy or bool(scoring_conflict) or uploaded is None,
    ):
        progress = animated_progress(1, "Securing the uploaded PCAP")
        result = preparation.stage_pcap(
            uploaded,
            lambda percent, message: progress.progress(percent, message),
        )
        if result.get("error"):
            st.error(str(result["error"]))
        else:
            st.success("The PCAP is queued for persistent Zeek conversion.")
            st.rerun()


def _render_live_controls() -> None:
    state = preparation.read_state()
    current = str(state.get("state") or "inactive")
    mode = str(state.get("mode") or "")
    busy = current in preparation.BUSY_STATES
    scoring_conflict = preparation.live_scoring_conflict_reason()
    inventory, inventory_error = preparation.interface_inventory()
    eligible = [row["Interface"] for row in inventory if row["Eligibility"] == "ELIGIBLE"]

    st.markdown("#### Managed live learning capture")
    st.caption(
        "Capture runs independently of the browser and survives logout or server reboot. Zeek rotates logs hourly; "
        "the service consolidates every segment into one JSON dns.log when time expires, 10 GiB is reached, or Stop is selected. "
        "Starting a new job replaces the previous prepared output."
    )
    if inventory_error:
        st.error(inventory_error)
    elif inventory:
        st.dataframe(pd.DataFrame(inventory), use_container_width=True, hide_index=True)
    if scoring_conflict:
        st.warning(scoring_conflict)

    selected = st.selectbox(
        "Dedicated learning-capture interface",
        eligible or ["No eligible interface"],
        key="dns_prep_interface",
        disabled=busy or bool(scoring_conflict) or not eligible,
        help="The interface may be addressless; it must not be loopback or the management/default-route interface.",
    )
    duration_label = st.selectbox(
        "Preconfigured learning period",
        list(DURATION_OPTIONS),
        index=3,
        key="dns_prep_duration",
        disabled=busy or bool(scoring_conflict),
    )

    start_disabled = busy or bool(scoring_conflict) or not eligible
    suspend_disabled = not (mode == "live" and current in {"starting", "active"})
    resume_disabled = not (mode == "live" and current == "paused") or bool(scoring_conflict)
    stop_disabled = not (mode == "live" and current in {"starting", "active", "paused"})

    columns = st.columns(4)
    with columns[0]:
        if st.button(
            "Activate capture",
            key="dns_prep_activate_live",
            type="primary",
            use_container_width=True,
            disabled=start_disabled,
        ):
            result = preparation.start_live_capture(selected, DURATION_OPTIONS[duration_label])
            if result.get("error"):
                st.error(str(result["error"]))
            else:
                st.rerun()
    with columns[1]:
        if st.button(
            "Suspend capture",
            key="dns_prep_suspend_live",
            use_container_width=True,
            disabled=suspend_disabled,
        ):
            result = preparation.suspend_live_capture()
            if result.get("error"):
                st.error(str(result["error"]))
            else:
                st.rerun()
    with columns[2]:
        if st.button(
            "Resume capture",
            key="dns_prep_resume_live",
            use_container_width=True,
            disabled=resume_disabled,
        ):
            result = preparation.resume_live_capture()
            if result.get("error"):
                st.error(str(result["error"]))
            else:
                st.rerun()
    with columns[3]:
        if st.button(
            "Deactivate and consolidate",
            key="dns_prep_stop_live",
            use_container_width=True,
            disabled=stop_disabled,
            help="Stops capture and produces the final consolidated dns.log from every hourly segment.",
        ):
            result = preparation.stop_and_consolidate()
            if result.get("error"):
                st.error(str(result["error"]))
            else:
                st.rerun()


if hasattr(st, "fragment"):
    _render_live_controls_fragment = st.fragment(run_every="3s")(_render_live_controls)
else:
    _render_live_controls_fragment = _render_live_controls


def render_dns_log_preparation_page() -> None:
    _inject_css()
    st.markdown(
        """
        <div class="dns-prep-hero">
            <div class="dns-prep-title">DNS Log Preparation</div>
            <div class="dns-prep-subtitle">
                Generate a model-ready Zeek JSON dns.log from an offline PCAP or from a protected,
                time-bounded live learning capture. Preparation and Live Zeek scoring capture are mutually exclusive.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_runtime_status_fragment()

    current_state = preparation.read_state()
    _render_output(current_state)
    if str(current_state.get("state") or "") in {"completed", "failed"}:
        if st.button(
            "Reset preparation state",
            key="dns_prep_reset_state",
            use_container_width=True,
            help="Resets the control state. A completed dns.log remains at its displayed protected path.",
        ):
            result = preparation.reset_state()
            if result.get("error"):
                st.error(str(result["error"]))
            else:
                st.rerun()

    with st.expander("Recommended operating policy", expanded=True):
        st.markdown(
            """
- Use **PCAP conversion** for a bounded, already-approved packet capture. It is reproducible and does not touch a network interface.
- Use **live learning capture** only on an authorized SPAN/TAP interface that is different from the management/default-route interface.
- Suspend preserves elapsed time and captured segments. Resume creates a new segment without overwriting earlier data.
- Hourly rotation prevents one long open file. Finalization combines every DNS segment into one JSON `dns.log`.
- The persistent service resumes an active job after logout or reboot and stops automatically at **10 GiB** or the selected duration.
- While preparation is uploading, processing, active, suspended, or finalizing, **Live Zeek scoring capture is locked**. Manual file scoring remains separate.
            """
        )

    pcap_tab, live_tab = st.tabs(["PCAP → dns.log", "Live learning capture"])
    with pcap_tab:
        _render_pcap_controls()
    with live_tab:
        _render_live_controls_fragment()

    st.caption(
        f"Persistent state: `{preparation.STATE_PATH}` · Final outputs: `{preparation.OUTPUT_ROOT}` · "
        "Service: `dns-ml-learning-capture.service`"
    )
