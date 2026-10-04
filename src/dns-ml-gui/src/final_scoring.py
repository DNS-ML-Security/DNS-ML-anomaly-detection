# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

FINAL_HOME = Path(os.getenv("DNS_FINAL_SCORING_HOME", "/data/dns-ml/final_scoring"))
FINAL_CONFIG_PATH = FINAL_HOME / "config.json"
FINAL_STATUS_PATH = FINAL_HOME / "status.json"
FINAL_ALERTS_PATH = Path(
    os.getenv("DNS_FINAL_ALERTS_PATH", "/data/dns-ml/alerts/dns_final_alerts.jsonl")
)

ISOLATION_ALERT_PATH = Path(
    os.getenv(
        "DNS_FINAL_ISOLATION_ALERTS",
        "/data/dns-ml/alerts/dns_categorization_forest_alerts.jsonl",
    )
)
AUTOENCODER_ALERT_PATH = Path(
    os.getenv("DNS_FINAL_AUTOENCODER_ALERTS", "/data/dns-ml/alerts/dns_alerts.jsonl")
)
HEURISTICS_ALERT_PATH = Path(
    os.getenv("DNS_FINAL_HEURISTICS_ALERTS", "/data/dns-ml/alerts/dns_heuristics_alerts.jsonl")
)

HEURISTIC_NAMES = (
    "heuristic_dga",
    "heuristic_tunnel",
    "heuristic_fastflux",
    "heuristic_resolver_failure",
    "heuristic_rogue_resolver",
)

DEFAULT_CONFIG: dict[str, Any] = {
    "correlation_minutes": 5,
    "isolation_min_score": 0.0,
    "isolation_min_percentile": 95.0,
    "autoencoder_threshold_multiplier": 1.20,
    "heuristic_min_threat_score": 40.0,
    "weight_isolation": 30.0,
    "weight_autoencoder": 30.0,
    "weight_heuristics": 40.0,
    "consensus_bonus_two": 15.0,
    "consensus_bonus_three": 25.0,
    "minimum_agreeing_models": 2,
    "final_alert_threshold": 65.0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def load_final_config() -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if FINAL_CONFIG_PATH.exists():
        try:
            stored = json.loads(FINAL_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                config.update(stored)
        except Exception:
            pass
    return config


def save_final_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    merged.update(config or {})
    merged["correlation_minutes"] = max(1, int(merged.get("correlation_minutes", 5)))
    merged["minimum_agreeing_models"] = max(
        1, min(3, int(merged.get("minimum_agreeing_models", 2)))
    )
    for key in (
        "isolation_min_score",
        "isolation_min_percentile",
        "autoencoder_threshold_multiplier",
        "heuristic_min_threat_score",
        "weight_isolation",
        "weight_autoencoder",
        "weight_heuristics",
        "consensus_bonus_two",
        "consensus_bonus_three",
        "final_alert_threshold",
    ):
        merged[key] = float(merged.get(key, DEFAULT_CONFIG[key]))
    merged["updated_at_utc"] = utc_now()
    _atomic_json(FINAL_CONFIG_PATH, merged)
    return merged


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
    return records


def _methods(alert: dict[str, Any]) -> dict[str, Any]:
    value = alert.get("detection_methods") or {}
    return value if isinstance(value, dict) else {}


def _is_isolation(alert: dict[str, Any]) -> bool:
    methods = _methods(alert)
    return bool(
        str(alert.get("alert_type") or "") == "categorization_forest"
        or str(alert.get("detection_type") or "") == "categorization_forest"
        or methods.get("categorization_forest")
        or methods.get("isolation_forest")
    )


def _is_autoencoder(alert: dict[str, Any]) -> bool:
    methods = _methods(alert)
    has_heuristic = any(bool(methods.get(name)) for name in HEURISTIC_NAMES)
    is_ae = bool(
        str(alert.get("alert_type") or "") == "autoencoder_anomaly"
        or str(alert.get("detection_type") or "") == "deep_learning_autoencoder"
        or methods.get("autoencoder")
    )
    return is_ae and not has_heuristic


def _is_heuristics(alert: dict[str, Any]) -> bool:
    methods = _methods(alert)
    return bool(
        str(alert.get("alert_type") or "") == "heuristics_rule"
        or str(alert.get("detection_type") or "") == "heuristics"
        or any(bool(methods.get(name)) for name in HEURISTIC_NAMES)
    )


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if pd.isna(result):
            return default
        return result
    except Exception:
        return default


def _window_start(alert: dict[str, Any], minutes: int) -> pd.Timestamp | None:
    value = alert.get("timestamp")
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.floor(f"{minutes}min")
    except Exception:
        return None


def _source_alert_id(alert: dict[str, Any]) -> str:
    return str(alert.get("id") or alert.get("alert_uid") or "")


def _isolation_component(alert: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    raw_score = _number(alert.get("isolation_score"), -999999.0)
    percentile = _number(alert.get("score_percentile"), -1.0)
    if percentile < 0:
        percentile = _number(alert.get("threat_score"), 0.0)
    percentile = max(0.0, min(percentile, 100.0))

    active = bool(
        raw_score >= float(config["isolation_min_score"])
        and percentile >= float(config["isolation_min_percentile"])
    )
    return {
        "active": active,
        "evidence": percentile if active else 0.0,
        "isolation_score": raw_score if raw_score > -999998 else None,
        "isolation_percentile": percentile,
        "source_alert_id": _source_alert_id(alert),
    }


def _autoencoder_component(alert: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    error = _number(alert.get("reconstruction_error"), 0.0)
    threshold = _number(alert.get("threshold"), 0.0)
    ratio = error / threshold if threshold > 0 else 0.0
    min_ratio = max(1.0, float(config["autoencoder_threshold_multiplier"]))
    active = bool(threshold > 0 and ratio >= min_ratio)

    # SOC-friendly normalization:
    # ratio 1.0 -> 50, ratio 2.0 -> 75, ratio >=3.0 -> 100.
    evidence = 0.0
    if active:
        evidence = max(0.0, min(100.0, 50.0 + 25.0 * (ratio - 1.0)))

    return {
        "active": active,
        "evidence": evidence,
        "reconstruction_error": error,
        "reconstruction_threshold": threshold,
        "reconstruction_ratio": ratio,
        "source_alert_id": _source_alert_id(alert),
    }


def _heuristic_component(alert: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    score = _number(alert.get("threat_score"), 0.0)
    score = max(0.0, min(score, 100.0))
    active = bool(score >= float(config["heuristic_min_threat_score"]))
    fired = alert.get("heuristics_fired") or [
        name for name in HEURISTIC_NAMES if bool(_methods(alert).get(name))
    ]
    if not isinstance(fired, list):
        fired = [str(fired)]
    return {
        "active": active,
        "evidence": score if active else 0.0,
        "heuristic_threat_score": score,
        "heuristics_fired": fired,
        "source_alert_id": _source_alert_id(alert),
    }


def _severity(score: float) -> str:
    if score >= 90:
        return "critical"
    if score >= 75:
        return "high"
    if score >= 60:
        return "medium"
    return "low"


def _stable_final_id(src_ip: str, window_start: pd.Timestamp) -> str:
    basis = f"final-correlation|{src_ip}|{window_start.isoformat()}"
    return "FINAL-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:20].upper()


def _write_final_alerts(alerts: list[dict[str, Any]]) -> None:
    FINAL_ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = FINAL_ALERTS_PATH.with_suffix(FINAL_ALERTS_PATH.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for alert in alerts:
            handle.write(json.dumps(alert, separators=(",", ":"), ensure_ascii=False, default=str))
            handle.write("\n")
    tmp.replace(FINAL_ALERTS_PATH)


def calculate_final_alerts(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = save_final_config(config or load_final_config())
    minutes = int(cfg["correlation_minutes"])

    isolation_raw = [a for a in _read_jsonl(ISOLATION_ALERT_PATH) if _is_isolation(a)]
    autoencoder_raw = [a for a in _read_jsonl(AUTOENCODER_ALERT_PATH) if _is_autoencoder(a)]
    heuristics_raw = [a for a in _read_jsonl(HEURISTICS_ALERT_PATH) if _is_heuristics(a)]

    grouped: dict[tuple[str, pd.Timestamp], dict[str, list[dict[str, Any]]]] = {}

    def add_family(family: str, records: list[dict[str, Any]]) -> None:
        for alert in records:
            src_ip = str(alert.get("src_ip") or "").strip()
            start = _window_start(alert, minutes)
            if not src_ip or start is None:
                continue
            bucket = grouped.setdefault(
                (src_ip, start),
                {"isolation": [], "autoencoder": [], "heuristics": []},
            )
            bucket[family].append(alert)

    add_family("isolation", isolation_raw)
    add_family("autoencoder", autoencoder_raw)
    add_family("heuristics", heuristics_raw)

    wi = max(0.0, float(cfg["weight_isolation"]))
    wa = max(0.0, float(cfg["weight_autoencoder"]))
    wh = max(0.0, float(cfg["weight_heuristics"]))
    weight_total = wi + wa + wh
    if weight_total <= 0:
        raise ValueError("At least one final-score weight must be greater than zero.")
    wi, wa, wh = wi / weight_total, wa / weight_total, wh / weight_total

    results: list[dict[str, Any]] = []
    two_model_candidates = 0
    three_model_candidates = 0
    suppressed_by_consensus = 0
    suppressed_by_score = 0

    for (src_ip, start), families in sorted(grouped.items(), key=lambda item: item[0][1]):
        iso_components = [_isolation_component(a, cfg) for a in families["isolation"]]
        ae_components = [_autoencoder_component(a, cfg) for a in families["autoencoder"]]
        heur_components = [_heuristic_component(a, cfg) for a in families["heuristics"]]

        iso = max(iso_components, key=lambda x: x["evidence"], default={"active": False, "evidence": 0.0})
        ae = max(ae_components, key=lambda x: x["evidence"], default={"active": False, "evidence": 0.0})
        heur = max(heur_components, key=lambda x: x["evidence"], default={"active": False, "evidence": 0.0})

        active_models = []
        if iso.get("active"):
            active_models.append("Isolation Forest")
        if ae.get("active"):
            active_models.append("Autoencoder")
        if heur.get("active"):
            active_models.append("Heuristics")

        model_count = len(active_models)
        if model_count == 2:
            two_model_candidates += 1
        elif model_count == 3:
            three_model_candidates += 1

        base_score = (
            wi * float(iso.get("evidence", 0.0))
            + wa * float(ae.get("evidence", 0.0))
            + wh * float(heur.get("evidence", 0.0))
        )

        bonus = 0.0
        if model_count >= 3:
            bonus = float(cfg["consensus_bonus_three"])
        elif model_count == 2:
            bonus = float(cfg["consensus_bonus_two"])

        final_score = max(0.0, min(100.0, base_score + bonus))

        if model_count < int(cfg["minimum_agreeing_models"]):
            suppressed_by_consensus += 1
            continue
        if final_score < float(cfg["final_alert_threshold"]):
            suppressed_by_score += 1
            continue

        source_ids = [
            value
            for value in (
                iso.get("source_alert_id"),
                ae.get("source_alert_id"),
                heur.get("source_alert_id"),
            )
            if value
        ]

        result = {
            "id": _stable_final_id(src_ip, start),
            "alert_uid": _stable_final_id(src_ip, start),
            "alert_name": "DNS final correlated alert",
            "alert_type": "final_correlated_alert",
            "detection_type": "final_correlation",
            "timestamp": start.isoformat(),
            "window_end": (start + pd.Timedelta(minutes=minutes)).isoformat(),
            "src_ip": src_ip,
            "status": "New",
            "severity": _severity(final_score),
            "threat_score": int(round(final_score)),
            "final_score": round(final_score, 4),
            "weighted_base_score": round(base_score, 4),
            "consensus_bonus": round(bonus, 4),
            "models_agreeing": model_count,
            "active_models": active_models,
            "isolation_evidence": round(float(iso.get("evidence", 0.0)), 4),
            "isolation_score": iso.get("isolation_score"),
            "isolation_percentile": iso.get("isolation_percentile"),
            "autoencoder_evidence": round(float(ae.get("evidence", 0.0)), 4),
            "reconstruction_error": ae.get("reconstruction_error"),
            "reconstruction_threshold": ae.get("reconstruction_threshold"),
            "reconstruction_ratio": round(float(ae.get("reconstruction_ratio", 0.0)), 4) if ae else None,
            "heuristic_evidence": round(float(heur.get("evidence", 0.0)), 4),
            "heuristic_threat_score": heur.get("heuristic_threat_score"),
            "heuristics_fired": heur.get("heuristics_fired", []),
            "weights": {
                "isolation": round(wi, 6),
                "autoencoder": round(wa, 6),
                "heuristics": round(wh, 6),
            },
            "thresholds": {
                "isolation_min_score": cfg["isolation_min_score"],
                "isolation_min_percentile": cfg["isolation_min_percentile"],
                "autoencoder_threshold_multiplier": cfg["autoencoder_threshold_multiplier"],
                "heuristic_min_threat_score": cfg["heuristic_min_threat_score"],
                "minimum_agreeing_models": cfg["minimum_agreeing_models"],
                "final_alert_threshold": cfg["final_alert_threshold"],
            },
            "source_alert_ids": source_ids,
            "generated_at_utc": utc_now(),
        }
        results.append(result)

    results.sort(key=lambda item: (str(item.get("timestamp")), float(item.get("final_score", 0))), reverse=True)
    _write_final_alerts(results)

    status = {
        "state": "finished",
        "generated_at_utc": utc_now(),
        "source_alert_counts": {
            "isolation": len(isolation_raw),
            "autoencoder": len(autoencoder_raw),
            "heuristics": len(heuristics_raw),
        },
        "correlated_windows": len(grouped),
        "two_model_candidates": two_model_candidates,
        "three_model_candidates": three_model_candidates,
        "suppressed_by_consensus": suppressed_by_consensus,
        "suppressed_by_final_score": suppressed_by_score,
        "final_alerts": len(results),
        "final_alerts_path": str(FINAL_ALERTS_PATH),
        "config_path": str(FINAL_CONFIG_PATH),
        "config": cfg,
    }
    _atomic_json(FINAL_STATUS_PATH, status)
    return status



def reset_final_runtime_state() -> dict[str, Any]:
    # Clear derived final-correlation runtime state without changing analyst config.
    try:
        FINAL_ALERTS_PATH.unlink(missing_ok=True)
    except TypeError:
        if FINAL_ALERTS_PATH.exists():
            FINAL_ALERTS_PATH.unlink()

    cfg = load_final_config()
    status = {
        "state": "cleared",
        "generated_at_utc": utc_now(),
        "source_alert_counts": {
            "isolation": 0,
            "autoencoder": 0,
            "heuristics": 0,
        },
        "correlated_windows": 0,
        "two_model_candidates": 0,
        "three_model_candidates": 0,
        "suppressed_by_consensus": 0,
        "suppressed_by_final_score": 0,
        "final_alerts": 0,
        "final_alerts_path": str(FINAL_ALERTS_PATH),
        "config_path": str(FINAL_CONFIG_PATH),
        "config": cfg,
        "reset_reason": "Clear All Alerts",
    }
    _atomic_json(FINAL_STATUS_PATH, status)
    return status

def _format_contributing_alert_ids(value: Any) -> str:
    if isinstance(value, dict):
        preferred = ["isolation", "autoencoder", "heuristics"]
        parts: list[str] = []
        seen: set[str] = set()
        for key in preferred:
            raw = value.get(key)
            if raw in (None, "", [], {}):
                continue
            if isinstance(raw, list):
                ids = ", ".join(str(item) for item in raw if str(item).strip())
            else:
                ids = str(raw).strip()
            if not ids:
                continue
            parts.append(f"{key}: {ids}")
            seen.add(str(key))
        for key, raw in value.items():
            if str(key) in seen or raw in (None, "", [], {}):
                continue
            if isinstance(raw, list):
                ids = ", ".join(str(item) for item in raw if str(item).strip())
            else:
                ids = str(raw).strip()
            if ids:
                parts.append(f"{key}: {ids}")
        return " | ".join(parts)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item).strip())
    if value in (None, ""):
        return ""
    return str(value)


def _count_contributing_alert_ids(value: Any) -> int:
    if isinstance(value, dict):
        total = 0
        for raw in value.values():
            if raw in (None, "", [], {}):
                continue
            if isinstance(raw, list):
                total += sum(1 for item in raw if str(item).strip())
            else:
                total += 1 if str(raw).strip() else 0
        return total
    if isinstance(value, list):
        return sum(1 for item in value if str(item).strip())
    return 1 if value not in (None, "") and str(value).strip() else 0


def load_final_alerts(limit: int = 500) -> pd.DataFrame:
    rows = _read_jsonl(FINAL_ALERTS_PATH)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if "timestamp" in frame.columns:
        frame["timestamp_dt"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.sort_values(["timestamp_dt", "final_score"], ascending=[False, False])
    if "active_models" in frame.columns:
        frame["active_models"] = frame["active_models"].apply(
            lambda value: ", ".join(value) if isinstance(value, list) else str(value)
        )
    if "heuristics_fired" in frame.columns:
        frame["heuristics_fired"] = frame["heuristics_fired"].apply(
            lambda value: ", ".join(value) if isinstance(value, list) else str(value)
        )
    if "source_alert_ids" in frame.columns:
        frame["contributing_alert_ids"] = frame["source_alert_ids"].apply(_format_contributing_alert_ids)
        frame["contributing_alert_count"] = frame["source_alert_ids"].apply(_count_contributing_alert_ids)
    return frame.head(max(1, int(limit))).reset_index(drop=True)


def read_final_status() -> dict[str, Any]:
    if not FINAL_STATUS_PATH.exists():
        return {}
    try:
        value = json.loads(FINAL_STATUS_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}
