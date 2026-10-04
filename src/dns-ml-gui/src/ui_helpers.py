# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

import html
import json
from typing import Any

import pandas as pd
import streamlit as st


class AnimatedProgress:
    """Reusable percentage-labelled progress bar with a shared SOC animation."""

    def __init__(self, value: float | int = 0, text: str = "") -> None:
        self._placeholder = st.empty()
        self._text = str(text or "")
        self.progress(value, text=text)

    @staticmethod
    def _percent(value: float | int) -> int:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0
        if isinstance(value, float) and 0.0 <= numeric <= 1.0:
            numeric *= 100.0
        return max(0, min(100, int(round(numeric))))

    def progress(self, value: float | int, text: str | None = None) -> "AnimatedProgress":
        percent = self._percent(value)
        if text is not None:
            self._text = str(text or "")
        label = html.escape(self._text or "Operation progress")
        active = " active" if 0 < percent < 100 else ""
        self._placeholder.markdown(
            (
                f'<div class="dns-animated-progress{active}" role="progressbar" '
                f'aria-label="{label}" aria-valuemin="0" aria-valuemax="100" '
                f'aria-valuenow="{percent}">'
                '<div class="dns-animated-progress-track">'
                f'<div class="dns-animated-progress-fill" style="width:{percent}%"></div>'
                f'<div class="dns-animated-progress-percent">{percent}%</div>'
                '</div>'
                f'<div class="dns-animated-progress-label">{label}</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )
        return self

    def empty(self) -> None:
        self._placeholder.empty()


def animated_progress(value: float | int = 0, text: str = "") -> AnimatedProgress:
    return AnimatedProgress(value, text)


def render_solution_footer() -> None:
    """Render the project ownership and source-license notice."""
    st.markdown(
        """
        <style>
        .dns-solution-footer {
            margin: 1.35rem 0 .35rem;
            padding: .68rem .9rem;
            border-top: 1px solid rgba(56, 189, 248, .28);
            color: #8FAFC5;
            font-size: clamp(.68rem, .76vw, .79rem);
            font-weight: 400;
            line-height: 1.45;
            text-align: center;
            overflow-wrap: anywhere;
        }
        .dns-solution-footer strong {
            color: #DCEAF5;
            font-weight: 650;
        }
        .dns-solution-footer a {
            color: #38BDF8;
            font-weight: 600;
            text-decoration: none;
        }
        .dns-solution-footer a:hover,
        .dns-solution-footer a:focus {
            color: #67E8F9;
            text-decoration: underline;
        }
        </style>
        <div class="dns-solution-footer">
          <strong>© 2026 A.Mekky</strong> — DNS-ML ·
          <a href="https://github.com/DNS-ML-Security/DNS-ML-anomaly-detection"
             target="_blank" rel="noopener noreferrer"><strong>DNS-ML-anomaly-detection</strong></a>
          · DNS ML Responsible-Use Source-Available License 1.0 ·
          Modifications must preserve the ownership, license, and modification-responsibility notices.
        </div>
        """,
        unsafe_allow_html=True,
    )


SEVERITY_ICON = {
    "high": "🔴",
    "medium": "🟠",
    "low": "🟡",
    "info": "🔵",
}


def parse_json(value: Any, default):
    if value is None:
        return default

    if isinstance(value, (dict, list)):
        return value

    try:
        return json.loads(value)
    except Exception:
        return default


def severity_label(severity: str) -> str:
    severity = (severity or "info").lower()
    return f"{SEVERITY_ICON.get(severity, '⚪')} {severity.upper()}"


def show_detection_methods(methods: dict):
    cols = st.columns(6)

    names = [
        ("autoencoder", "Autoencoder"),
        ("heuristic_dga", "DGA"),
        ("heuristic_tunnel", "Tunnel"),
        ("heuristic_fastflux", "Fast-flux"),
        ("heuristic_resolver_failure", "Resolver failure"),
        ("heuristic_rogue_resolver", "Rogue resolver"),
    ]

    for col, (key, label) in zip(cols, names):
        value = bool(methods.get(key))
        col.metric(label, "YES" if value else "NO")


def dataframe_or_info(df: pd.DataFrame, message: str):
    if df is None or df.empty:
        st.info(message)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)
