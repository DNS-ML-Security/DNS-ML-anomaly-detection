#!/usr/bin/env python3
# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

"""Train a CPU-only Isolation Forest for Zeek DNS five-minute windows.

This learning pipeline is intentionally independent from live DNS scoring and
alert generation. It reuses feature_utils.py and writes only Isolation Forest
model and run artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# Explicitly disable accelerator discovery. Scikit-learn IsolationForest is CPU
# based, but these variables also prevent optional libraries from attempting to
# initialize CUDA while this process is running.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from feature_utils import (
    FEATURES,
    add_event_features,
    build_dns_windows,
    load_zeek_dns_json_files,
    validate_feature_matrix,
)


MODEL_DIR = Path(os.getenv("DNS_MODEL_DIR", "/data/dns-ml/models"))
RUN_DIR = Path(os.getenv("DNS_RUN_DIR", "/data/dns-ml/runs"))
TRAINING_DIR = Path(os.getenv("DNS_TRAINING_DIR", "/data/dns-ml/training"))
DEFAULT_MERGED_PATH = Path(
    os.getenv("DNS_ZEEK_MERGED_PATH", "/data/dns-ml/zeek/merged_dns.jsonl")
)

MODEL_PATH = Path(
    os.getenv(
        "DNS_IF_MODEL_PATH",
        str(MODEL_DIR / "dns_isolation_forest.joblib"),
    )
)
SCALER_PATH = Path(
    os.getenv(
        "DNS_IF_SCALER_PATH",
        str(MODEL_DIR / "dns_isolation_forest_scaler.joblib"),
    )
)
PCA_PATH = Path(
    os.getenv(
        "DNS_IF_PCA_PATH",
        str(MODEL_DIR / "dns_isolation_forest_pca.joblib"),
    )
)
CONFIG_PATH = Path(
    os.getenv(
        "DNS_IF_CONFIG_PATH",
        str(MODEL_DIR / "dns_isolation_forest_config.json"),
    )
)

STATUS_PATH = Path(
    os.getenv(
        "DNS_IF_STATUS_PATH",
        str(RUN_DIR / "latest_isolation_forest_status.json"),
    )
)
RESULT_PATH = Path(
    os.getenv(
        "DNS_IF_RESULT_PATH",
        str(RUN_DIR / "latest_isolation_forest_result.json"),
    )
)
SCORES_PATH = Path(
    os.getenv(
        "DNS_IF_SCORES_PATH",
        str(RUN_DIR / "latest_isolation_forest_scores.csv"),
    )
)
POINTS_PATH = Path(
    os.getenv(
        "DNS_IF_POINTS_PATH",
        str(RUN_DIR / "latest_isolation_forest_pca_points.csv"),
    )
)
CONTOUR_PATH = Path(
    os.getenv(
        "DNS_IF_CONTOUR_PATH",
        str(RUN_DIR / "latest_isolation_forest_contour.csv"),
    )
)
SHAP_SUMMARY_PATH = Path(
    os.getenv(
        "DNS_IF_SHAP_SUMMARY_PATH",
        str(RUN_DIR / "latest_isolation_forest_shap_summary.csv"),
    )
)
SHAP_VALUES_PATH = Path(
    os.getenv(
        "DNS_IF_SHAP_VALUES_PATH",
        str(RUN_DIR / "latest_isolation_forest_shap_values.csv"),
    )
)
SHAP_DEPENDENCY_PATH = Path(
    os.getenv(
        "DNS_IF_SHAP_DEPENDENCY_PATH",
        str(RUN_DIR / "latest_isolation_forest_shap_dependency.csv"),
    )
)

MIN_WINDOWS = int(os.getenv("IF_MIN_WINDOWS", "50"))
DEFAULT_CONTAMINATION = float(os.getenv("IF_CONTAMINATION", "0.01"))
DEFAULT_N_ESTIMATORS = int(os.getenv("IF_N_ESTIMATORS", "300"))
DEFAULT_MAX_SAMPLES = os.getenv("IF_MAX_SAMPLES", "auto")
DEFAULT_RANDOM_STATE = int(os.getenv("IF_RANDOM_STATE", "42"))
DEFAULT_N_JOBS = int(os.getenv("IF_N_JOBS", "-1"))
DEFAULT_SHAP_SAMPLE = int(os.getenv("IF_SHAP_SAMPLE_SIZE", "500"))
DEFAULT_SILHOUETTE_SAMPLE = int(os.getenv("IF_SILHOUETTE_SAMPLE_SIZE", "5000"))
DEFAULT_PCA_FIT_SAMPLE = int(os.getenv("IF_PCA_FIT_SAMPLE_SIZE", "50000"))
DEFAULT_CONTOUR_GRID_SIZE = int(os.getenv("IF_CONTOUR_GRID_SIZE", "70"))
DEBUG_ENABLED = os.getenv("IF_DEBUG", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

LOGGER = logging.getLogger("dns_isolation_forest")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_status(
    state: str,
    stage: str,
    progress: int,
    message: str,
    *,
    error: str | None = None,
    **extra: Any,
) -> None:
    payload: dict[str, Any] = {
        "state": state,
        "stage": stage,
        "progress_percent": max(0, min(int(progress), 100)),
        "message": message,
        "error": error,
        "updated_at_utc": utc_now(),
        "pid": os.getpid(),
        "cpu_only": True,
        "debug_enabled": DEBUG_ENABLED,
    }
    payload.update(extra)
    atomic_write_json(STATUS_PATH, payload)


def configure_logging() -> None:
    level = logging.DEBUG if DEBUG_ENABLED else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stdout,
        force=True,
    )
    LOGGER.setLevel(level)


def print_heading(title: str) -> None:
    line = "=" * max(78, len(title) + 8)
    print(f"\n{line}\n{title}\n{line}", flush=True)


def print_table(title: str, rows: Sequence[tuple[str, Any]]) -> None:
    print_heading(title)
    label_width = max([len(str(label)) for label, _ in rows] + [10])
    value_width = max([len(str(value)) for _, value in rows] + [10])
    border = f"+-{'-' * label_width}-+-{'-' * value_width}-+"
    print(border)
    print(f"| {'Metric':<{label_width}} | {'Value':<{value_width}} |")
    print(border)
    for label, value in rows:
        print(f"| {str(label):<{label_width}} | {str(value):<{value_width}} |")
    print(border, flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_input_paths(raw: str | None) -> list[Path]:
    candidates: list[Path] = []
    if raw:
        normalized = raw.replace(",", os.pathsep)
        candidates.extend(Path(item.strip()) for item in normalized.split(os.pathsep) if item.strip())
    elif DEFAULT_MERGED_PATH.exists():
        candidates.append(DEFAULT_MERGED_PATH)
    elif TRAINING_DIR.exists():
        patterns = ("*.jsonl", "*.json", "*.log", "*.jsonl.gz", "*.json.gz", "*.log.gz")
        for pattern in patterns:
            candidates.extend(sorted(TRAINING_DIR.glob(pattern)))

    unique: dict[str, Path] = {}
    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved.is_file() and resolved.stat().st_size > 0:
            unique[str(resolved)] = resolved
    return list(unique.values())


def parse_max_samples(value: str) -> str | int | float:
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    try:
        if "." in text:
            number = float(text)
            if 0 < number <= 1:
                return number
        number_int = int(text)
        if number_int > 0:
            return number_int
    except ValueError:
        pass
    raise ValueError("IF_MAX_SAMPLES must be 'auto', a positive integer, or a float in (0, 1].")


def stratified_indices(labels: np.ndarray, limit: int, random_state: int) -> np.ndarray:
    total = len(labels)
    if total <= limit:
        return np.arange(total)

    rng = np.random.default_rng(random_state)
    selected: list[np.ndarray] = []
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        group = np.flatnonzero(labels == label)
        allocation = max(2, int(round(limit * (count / total))))
        allocation = min(allocation, len(group))
        selected.append(rng.choice(group, size=allocation, replace=False))

    indices = np.unique(np.concatenate(selected))
    if len(indices) > limit:
        indices = rng.choice(indices, size=limit, replace=False)
    elif len(indices) < limit:
        remaining = np.setdiff1d(np.arange(total), indices, assume_unique=False)
        extra_size = min(limit - len(indices), len(remaining))
        if extra_size:
            indices = np.concatenate(
                [indices, rng.choice(remaining, size=extra_size, replace=False)]
            )
    return np.sort(indices)


def calculate_silhouette(
    x_scaled: np.ndarray,
    predicted_labels: np.ndarray,
    sample_limit: int,
    random_state: int,
) -> tuple[float | None, int, str | None]:
    cluster_labels = np.where(predicted_labels == -1, 1, 0)
    unique, counts = np.unique(cluster_labels, return_counts=True)
    if len(unique) < 2:
        return None, 0, "Silhouette is undefined because only one predicted group exists."
    if np.min(counts) < 2:
        return None, 0, "Silhouette is undefined because one predicted group has fewer than two windows."

    indices = stratified_indices(cluster_labels, sample_limit, random_state)
    sampled_labels = cluster_labels[indices]
    sampled_unique, sampled_counts = np.unique(sampled_labels, return_counts=True)
    if len(sampled_unique) < 2 or np.min(sampled_counts) < 2:
        return None, len(indices), "The silhouette sample did not retain two valid groups."

    value = float(silhouette_score(x_scaled[indices], sampled_labels, metric="euclidean"))
    return value, len(indices), None


def fit_pca(
    x_scaled: np.ndarray,
    fit_limit: int,
    random_state: int,
) -> PCA:
    if len(x_scaled) > fit_limit:
        rng = np.random.default_rng(random_state)
        fit_indices = rng.choice(len(x_scaled), size=fit_limit, replace=False)
        fit_data = x_scaled[fit_indices]
    else:
        fit_data = x_scaled
    pca = PCA(n_components=2, svd_solver="auto", random_state=random_state)
    pca.fit(fit_data)
    return pca


def build_contour_frame(
    model: IsolationForest,
    pca: PCA,
    projected: np.ndarray,
    grid_size: int,
) -> pd.DataFrame:
    x_low, x_high = np.quantile(projected[:, 0], [0.01, 0.99])
    y_low, y_high = np.quantile(projected[:, 1], [0.01, 0.99])
    x_padding = max((x_high - x_low) * 0.10, 1e-6)
    y_padding = max((y_high - y_low) * 0.10, 1e-6)
    x_values = np.linspace(x_low - x_padding, x_high + x_padding, grid_size)
    y_values = np.linspace(y_low - y_padding, y_high + y_padding, grid_size)
    xx, yy = np.meshgrid(x_values, y_values)
    grid_2d = np.column_stack([xx.ravel(), yy.ravel()])
    reconstructed_scaled = pca.inverse_transform(grid_2d)
    anomaly_scores = -model.decision_function(reconstructed_scaled)
    return pd.DataFrame(
        {
            "pca_1": grid_2d[:, 0],
            "pca_2": grid_2d[:, 1],
            "anomaly_score": anomaly_scores,
            "is_anomaly_region": anomaly_scores > 0,
        }
    )


def build_shap_outputs(
    model: IsolationForest,
    x_scaled: np.ndarray,
    windows: pd.DataFrame,
    anomaly_scores: np.ndarray,
    predicted_labels: np.ndarray,
    sample_size: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str | None]:
    try:
        import shap
    except Exception as exc:  # pragma: no cover - depends on target environment
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"SHAP import failed: {exc}"

    indices = stratified_indices(predicted_labels, min(sample_size, len(x_scaled)), random_state)
    x_sample = x_scaled[indices]
    sampled_windows = windows.iloc[indices].reset_index(drop=True)

    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x_sample)
        values_array = np.asarray(values, dtype=float)
        if values_array.ndim == 3 and values_array.shape[-1] == 1:
            values_array = values_array[..., 0]
        if values_array.shape != x_sample.shape:
            raise ValueError(
                f"Unexpected SHAP shape {values_array.shape}; expected {x_sample.shape}."
            )

        # Isolation Forest's TreeExplainer output is path-length oriented.
        # Shorter paths indicate stronger anomaly evidence, therefore the sign
        # is inverted so positive values consistently mean 'pushes anomalous'.
        anomaly_direction_values = -values_array

        mean_abs = np.mean(np.abs(anomaly_direction_values), axis=0)
        summary = pd.DataFrame(
            {
                "feature": FEATURES,
                "mean_abs_shap": mean_abs,
                "mean_shap": np.mean(anomaly_direction_values, axis=0),
                "rank": pd.Series(mean_abs).rank(method="dense", ascending=False).astype(int),
            }
        ).sort_values(["mean_abs_shap", "feature"], ascending=[False, True])

        top_features = summary.head(min(15, len(FEATURES)))["feature"].tolist()
        feature_to_index = {feature: index for index, feature in enumerate(FEATURES)}
        long_rows: list[pd.DataFrame] = []
        for feature in top_features:
            index = feature_to_index[feature]
            raw_values = pd.to_numeric(sampled_windows[feature], errors="coerce").fillna(0.0)
            percentile = raw_values.rank(pct=True, method="average")
            frame = pd.DataFrame(
                {
                    "sample_id": np.arange(len(indices)),
                    "feature": feature,
                    "feature_value": raw_values.to_numpy(dtype=float),
                    "feature_percentile": percentile.to_numpy(dtype=float),
                    "shap_value": anomaly_direction_values[:, index],
                    "anomaly_score": anomaly_scores[indices],
                    "predicted_label": np.where(predicted_labels[indices] == -1, "anomaly", "normal"),
                }
            )
            long_rows.append(frame)
        shap_long = pd.concat(long_rows, ignore_index=True) if long_rows else pd.DataFrame()

        top_feature = str(summary.iloc[0]["feature"])
        top_index = feature_to_index[top_feature]
        dependency = pd.DataFrame(
            {
                "timestamp": sampled_windows.get("timestamp", pd.Series([""] * len(indices))).astype(str),
                "src_ip": sampled_windows.get("src_ip", pd.Series([""] * len(indices))).astype(str),
                "feature": top_feature,
                "feature_value": pd.to_numeric(sampled_windows[top_feature], errors="coerce").fillna(0.0),
                "shap_value": anomaly_direction_values[:, top_index],
                "anomaly_score": anomaly_scores[indices],
                "predicted_label": np.where(predicted_labels[indices] == -1, "anomaly", "normal"),
            }
        )
        return summary, shap_long, dependency, None
    except Exception as exc:
        LOGGER.exception("SHAP diagnostics failed")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), f"SHAP diagnostics failed: {exc}"


def replace_artifact(staged: Path, destination: Path, backup_stamp: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        backup = destination.with_name(f"{destination.name}.backup_{backup_stamp}")
        shutil.copy2(destination, backup)
    os.replace(staged, destination)


def save_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)



def ensure_isolation_window_metadata(
    windows: pd.DataFrame,
    events: pd.DataFrame,
    window_size: str,
) -> pd.DataFrame:
    """Restore five-minute grouping metadata omitted by older feature_utils.py.

    Some project versions return only the 36 model features from
    build_dns_windows(). Isolation Forest still needs the source IP and window
    timestamp for diagnostics, charts and saved score rows. Rebuild those keys
    with the exact same pandas grouping order and verify alignment against the
    total_queries feature before attaching them.
    """
    result = windows.reset_index(drop=True).copy()

    if {"timestamp", "src_ip"}.issubset(result.columns):
        result["timestamp"] = pd.to_datetime(
            result["timestamp"], utc=True, errors="coerce"
        )
        result["src_ip"] = result["src_ip"].fillna("").astype(str)
        if result["timestamp"].isna().any():
            raise ValueError(
                "Isolation Forest windows contain invalid timestamp values."
            )
        return result

    required_event_columns = {"timestamp", "src_ip"}
    missing_event_columns = sorted(required_event_columns.difference(events.columns))
    if missing_event_columns:
        raise ValueError(
            "Cannot restore Isolation Forest window metadata. "
            f"Event columns missing: {missing_event_columns}. "
            f"Available columns: {list(events.columns)}"
        )

    metadata_source = events[["timestamp", "src_ip"]].copy()
    metadata_source["timestamp"] = pd.to_datetime(
        metadata_source["timestamp"], utc=True, errors="coerce"
    )
    metadata_source["src_ip"] = metadata_source["src_ip"].fillna("").astype(str)
    metadata_source = metadata_source[
        metadata_source["timestamp"].notna()
        & metadata_source["src_ip"].ne("")
    ].copy()

    if metadata_source.empty:
        raise ValueError(
            "Cannot restore Isolation Forest window metadata because no valid "
            "timestamp/source-IP events remain after normalization."
        )

    # This mirrors feature_utils.build_dns_windows():
    # group by a timestamp Grouper and source IP, with pandas' default sorted
    # group-key order.
    grouped_metadata = (
        metadata_source.groupby(
            [
                pd.Grouper(key="timestamp", freq=window_size),
                "src_ip",
            ],
            dropna=False,
            observed=True,
            sort=True,
        )
        .size()
        .reset_index(name="_event_count")
    )

    if len(grouped_metadata) != len(result):
        raise ValueError(
            "Isolation Forest window metadata alignment failed: "
            f"feature rows={len(result):,}, reconstructed groups="
            f"{len(grouped_metadata):,}. The feature utility grouping logic "
            "does not match the trainer's five-minute grouping logic."
        )

    # total_queries is the exact group size in feature_utils.py. Comparing it
    # row by row prevents silently attaching a timestamp/source IP to the wrong
    # feature vector.
    if "total_queries" in result.columns:
        feature_counts = pd.to_numeric(
            result["total_queries"], errors="coerce"
        ).fillna(-1).astype("int64").to_numpy()
        metadata_counts = grouped_metadata["_event_count"].astype("int64").to_numpy()
        mismatches = int((feature_counts != metadata_counts).sum())
        if mismatches:
            raise ValueError(
                "Isolation Forest metadata order validation failed: "
                f"{mismatches:,} rows have a total_queries/group-size mismatch. "
                "Metadata was not attached to avoid corrupting diagnostics."
            )

    feature_only = result.drop(
        columns=[column for column in ("timestamp", "src_ip") if column in result.columns],
        errors="ignore",
    )
    restored = pd.concat(
        [
            grouped_metadata[["timestamp", "src_ip"]].reset_index(drop=True),
            feature_only.reset_index(drop=True),
        ],
        axis=1,
    )

    LOGGER.debug(
        "Restored missing Isolation Forest window metadata for %d "
        "source-IP/%s windows.",
        len(restored),
        window_size,
    )
    return restored

def train(args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.time()
    feature_input = Path(
        os.getenv(
            "DNS_PRECOMPUTED_FEATURES",
            "/data/dns-ml/features/dns_training_features.csv",
        )
    )
    feature_summary_path = Path(
        "/data/dns-ml/features/dns_training_feature_summary.json"
    )

    if not feature_input.exists() or feature_input.stat().st_size == 0:
        raise FileNotFoundError(
            f"Precomputed DNS feature dataset is missing or empty: {feature_input}"
        )

    write_status(
        "running",
        "Loading precomputed features",
        8,
        f"Loading shared feature dataset: {feature_input}",
        training_files=[str(feature_input)],
    )

    windows = pd.read_csv(feature_input)
    if windows.empty:
        raise ValueError("Precomputed feature dataset contains zero windows.")

    # DNS_METADATA_PRESERVATION_HOTFIX_V5_3_IF
    metadata_columns = ["timestamp", "src_ip"]
    missing_metadata = [
        column for column in metadata_columns if column not in windows.columns
    ]
    if missing_metadata:
        raise RuntimeError(
            "Shared feature CSV is missing metadata columns: "
            + ", ".join(missing_metadata)
        )

    metadata = windows[metadata_columns].copy().reset_index(drop=True)
    validated_windows = validate_feature_matrix(
        windows,
        FEATURES,
    ).reset_index(drop=True)

    if len(metadata) != len(validated_windows):
        raise RuntimeError(
            "Metadata/feature row mismatch: "
            f"{len(metadata)} metadata rows versus "
            f"{len(validated_windows)} validated feature rows"
        )
    if metadata["timestamp"].fillna("").astype(str).str.strip().eq("").any():
        raise RuntimeError("Shared feature CSV contains blank timestamp values")
    if metadata["src_ip"].fillna("").astype(str).str.strip().eq("").any():
        raise RuntimeError("Shared feature CSV contains blank src_ip values")

    windows = pd.concat(
        [metadata, validated_windows[FEATURES]],
        axis=1,
    )
    if len(windows) < MIN_WINDOWS:
        raise ValueError(
            f"Only {len(windows):,} precomputed DNS windows are available. "
            f"Minimum required is {MIN_WINDOWS:,}."
        )

    input_paths = [feature_input]

    feature_summary = {}
    try:
        if feature_summary_path.exists():
            feature_summary = json.loads(
                feature_summary_path.read_text(encoding="utf-8")
            )
    except Exception:
        feature_summary = {}

    LOGGER.info(
        "Loaded %d precomputed windows x %d shared features. "
        "Feature engineering/window aggregation are NOT repeated.",
        len(windows),
        len(FEATURES),
    )

    print_table(
        "ISOLATION FOREST PRECOMPUTED TRAINING DATASET",
        [
            ("Feature CSV", feature_input),
            ("Training windows", f"{len(windows):,}"),
            ("Features per window", len(FEATURES)),
            ("Window size", feature_summary.get("window_size", args.window_size)),
            (
                "First window",
                str(windows["timestamp"].min()) if "timestamp" in windows else "N/A",
            ),
            (
                "Last window",
                str(windows["timestamp"].max()) if "timestamp" in windows else "N/A",
            ),
            (
                "Source IP addresses",
                int(windows["src_ip"].astype(str).nunique())
                if "src_ip" in windows
                else 0,
            ),
        ],
    )
    x = windows[FEATURES].to_numpy(dtype=np.float64, copy=True)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    max_samples = parse_max_samples(args.max_samples)
    model = IsolationForest(
        n_estimators=args.n_estimators,
        max_samples=max_samples,
        contamination=args.contamination,
        max_features=1.0,
        bootstrap=False,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbose=1 if DEBUG_ENABLED else 0,
    )

    print_table(
        "ISOLATION FOREST CONFIGURATION",
        [
            ("CPU-only", "YES"),
            ("Debug mode", "ENABLED" if DEBUG_ENABLED else "DISABLED"),
            ("Estimators", args.n_estimators),
            ("Maximum samples", max_samples),
            ("Contamination", args.contamination),
            ("CPU workers", args.n_jobs),
            ("Random state", args.random_state),
        ],
    )

    write_status(
        "running",
        "Training Isolation Forest",
        42,
        f"Fitting {args.n_estimators:,} isolation trees on CPU.",
        training_windows=int(len(windows)),
        feature_count=int(len(FEATURES)),
    )
    LOGGER.info("Fitting IsolationForest on %d windows x %d features", len(windows), len(FEATURES))
    model.fit(x_scaled)

    write_status(
        "running",
        "Calculating anomaly scores",
        58,
        "Calculating score distribution and unsupervised separation metrics.",
    )
    decision_function = model.decision_function(x_scaled)
    anomaly_scores = -decision_function
    raw_scores = model.score_samples(x_scaled)
    predicted_labels = model.predict(x_scaled)
    is_anomaly = predicted_labels == -1

    normal_scores = anomaly_scores[~is_anomaly]
    anomaly_group_scores = anomaly_scores[is_anomaly]
    if len(normal_scores) and len(anomaly_group_scores):
        score_distribution_gap = float(
            np.median(anomaly_group_scores) - np.median(normal_scores)
        )
        gap_definition = "median(predicted anomaly score) - median(predicted normal score)"
    else:
        score_distribution_gap = float(np.quantile(anomaly_scores, 0.95) - np.median(anomaly_scores))
        gap_definition = "95th percentile anomaly score - median anomaly score (single predicted group fallback)"

    silhouette_value, silhouette_samples_used, silhouette_note = calculate_silhouette(
        x_scaled,
        predicted_labels,
        args.silhouette_sample_size,
        args.random_state,
    )

    scores_frame = windows[["timestamp", "src_ip"]].copy()
    scores_frame["raw_score_samples"] = raw_scores
    scores_frame["decision_function"] = decision_function
    scores_frame["anomaly_score"] = anomaly_scores
    scores_frame["predicted_label"] = np.where(is_anomaly, "anomaly", "normal")
    scores_frame["predicted_code"] = predicted_labels
    scores_frame["threshold"] = 0.0

    write_status(
        "running",
        "Building PCA decision surface",
        68,
        "Projecting the 36-feature windows into two dimensions and calculating the contour surface.",
    )
    pca = fit_pca(x_scaled, args.pca_fit_sample_size, args.random_state)
    projected = pca.transform(x_scaled)
    points_frame = scores_frame.copy()
    points_frame["pca_1"] = projected[:, 0]
    points_frame["pca_2"] = projected[:, 1]
    contour_frame = build_contour_frame(model, pca, projected, args.contour_grid_size)

    write_status(
        "running",
        "Calculating SHAP explanations",
        80,
        "Calculating CPU-based SHAP summary and dependency datasets.",
    )
    shap_summary, shap_values_long, shap_dependency, shap_warning = build_shap_outputs(
        model,
        x_scaled,
        windows,
        anomaly_scores,
        predicted_labels,
        args.shap_sample_size,
        args.random_state,
    )
    if shap_warning:
        LOGGER.warning(shap_warning)

    write_status(
        "running",
        "Saving model artifacts",
        91,
        "Validating and publishing the Isolation Forest model and learning diagnostics.",
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix="dns_if_stage_", dir=str(MODEL_DIR)))
    staged_model = stage_dir / MODEL_PATH.name
    staged_scaler = stage_dir / SCALER_PATH.name
    staged_pca = stage_dir / PCA_PATH.name
    staged_config = stage_dir / CONFIG_PATH.name

    joblib.dump(model, staged_model)
    joblib.dump(scaler, staged_scaler)
    joblib.dump(pca, staged_pca)

    model_validation = joblib.load(staged_model)
    scaler_validation = joblib.load(staged_scaler)
    pca_validation = joblib.load(staged_pca)
    if int(getattr(model_validation, "n_features_in_", -1)) != len(FEATURES):
        raise ValueError("Staged Isolation Forest feature count validation failed.")
    if int(getattr(scaler_validation, "n_features_in_", -1)) != len(FEATURES):
        raise ValueError("Staged scaler feature count validation failed.")
    if int(getattr(pca_validation, "n_features_in_", -1)) != len(FEATURES):
        raise ValueError("Staged PCA feature count validation failed.")

    completed_at = utc_now()
    config = {
        "model_type": "IsolationForest",
        "model_name": "dns_isolation_forest",
        "cpu_only": True,
        "debug_enabled": DEBUG_ENABLED,
        "features": FEATURES,
        "feature_count": len(FEATURES),
        "window_size": args.window_size,
        "n_estimators": args.n_estimators,
        "max_samples": max_samples,
        "contamination": args.contamination,
        "n_jobs": args.n_jobs,
        "random_state": args.random_state,
        "anomaly_score_definition": "-decision_function; higher values are more anomalous",
        "anomaly_threshold": 0.0,
        "score_distribution_gap": score_distribution_gap,
        "score_distribution_gap_definition": gap_definition,
        "silhouette_score": silhouette_value,
        "silhouette_sample_size": silhouette_samples_used,
        "silhouette_note": silhouette_note,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "shap_direction": "Tree SHAP path-length contribution multiplied by -1; positive values push toward anomaly",
        "training_windows": len(windows),
        "training_files": [str(path) for path in input_paths],
        "trained_at_utc": completed_at,
        "sklearn_version": __import__("sklearn").__version__,
        "python_version": sys.version,
    }
    atomic_write_json(staged_config, config)

    backup_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        replace_artifact(staged_model, MODEL_PATH, backup_stamp)
        replace_artifact(staged_scaler, SCALER_PATH, backup_stamp)
        replace_artifact(staged_pca, PCA_PATH, backup_stamp)
        replace_artifact(staged_config, CONFIG_PATH, backup_stamp)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    save_csv_atomic(scores_frame, SCORES_PATH)
    save_csv_atomic(points_frame, POINTS_PATH)
    save_csv_atomic(contour_frame, CONTOUR_PATH)
    if not shap_summary.empty:
        save_csv_atomic(shap_summary, SHAP_SUMMARY_PATH)
    else:
        SHAP_SUMMARY_PATH.unlink(missing_ok=True)
    if not shap_values_long.empty:
        save_csv_atomic(shap_values_long, SHAP_VALUES_PATH)
    else:
        SHAP_VALUES_PATH.unlink(missing_ok=True)
    if not shap_dependency.empty:
        save_csv_atomic(shap_dependency, SHAP_DEPENDENCY_PATH)
    else:
        SHAP_DEPENDENCY_PATH.unlink(missing_ok=True)

    elapsed = time.time() - start_time
    result = {
        "state": "finished",
        "model_type": "IsolationForest",
        "training_windows": int(len(windows)),
        "feature_count": int(len(FEATURES)),
        "training_window_features": f"{len(windows):,} windows × {len(FEATURES)} features",
        "source_ip_count": int(windows["src_ip"].nunique()),
        "predicted_normal_windows": int((~is_anomaly).sum()),
        "predicted_anomaly_windows": int(is_anomaly.sum()),
        "predicted_anomaly_rate": float(is_anomaly.mean()),
        "score_min": float(np.min(anomaly_scores)),
        "score_mean": float(np.mean(anomaly_scores)),
        "score_median": float(np.median(anomaly_scores)),
        "score_max": float(np.max(anomaly_scores)),
        "score_distribution_gap": score_distribution_gap,
        "score_distribution_gap_definition": gap_definition,
        "silhouette_score": silhouette_value,
        "silhouette_sample_size": silhouette_samples_used,
        "silhouette_note": silhouette_note,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "top_shap_feature": (
            str(shap_summary.iloc[0]["feature"]) if not shap_summary.empty else None
        ),
        "shap_warning": shap_warning,
        "training_files": [str(path) for path in input_paths],
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "pca_path": str(PCA_PATH),
        "config_path": str(CONFIG_PATH),
        "scores_path": str(SCORES_PATH),
        "pca_points_path": str(POINTS_PATH),
        "contour_path": str(CONTOUR_PATH),
        "shap_summary_path": str(SHAP_SUMMARY_PATH),
        "shap_values_path": str(SHAP_VALUES_PATH),
        "shap_dependency_path": str(SHAP_DEPENDENCY_PATH),
        "model_sha256": file_sha256(MODEL_PATH),
        "elapsed_seconds": elapsed,
        "completed_at_utc": completed_at,
        "cpu_only": True,
        "debug_enabled": DEBUG_ENABLED,
    }
    atomic_write_json(RESULT_PATH, result)

    print_table(
        "ISOLATION FOREST LEARNING RESULTS",
        [
            ("Training windows", f"{len(windows):,}"),
            ("Feature count", len(FEATURES)),
            ("Training window features", result["training_window_features"]),
            ("Predicted normal windows", f"{result['predicted_normal_windows']:,}"),
            ("Predicted anomaly windows", f"{result['predicted_anomaly_windows']:,}"),
            ("Predicted anomaly rate", f"{result['predicted_anomaly_rate']:.4%}"),
            ("Score distribution gap", f"{score_distribution_gap:.8f}"),
            ("Silhouette score", "N/A" if silhouette_value is None else f"{silhouette_value:.6f}"),
            ("PCA explained variance", f"{sum(pca.explained_variance_ratio_):.4%}"),
            ("Top SHAP feature", result["top_shap_feature"] or "Unavailable"),
            ("Elapsed", f"{elapsed:.2f} seconds"),
        ],
    )
    print_table(
        "SAVED ISOLATION FOREST ARTIFACTS",
        [
            ("Model", MODEL_PATH),
            ("Scaler", SCALER_PATH),
            ("PCA", PCA_PATH),
            ("Configuration", CONFIG_PATH),
            ("Latest result", RESULT_PATH),
            ("Window scores", SCORES_PATH),
            ("PCA points", POINTS_PATH),
            ("Contour grid", CONTOUR_PATH),
            ("SHAP summary", SHAP_SUMMARY_PATH),
            ("SHAP values", SHAP_VALUES_PATH),
            ("SHAP dependency", SHAP_DEPENDENCY_PATH),
        ],
    )

    write_status(
        "finished",
        "Completed",
        100,
        "Isolation Forest learning completed successfully.",
        training_windows=int(len(windows)),
        feature_count=int(len(FEATURES)),
        score_distribution_gap=score_distribution_gap,
        silhouette_score=silhouette_value,
        result_path=str(RESULT_PATH),
        completed_at_utc=completed_at,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-files",
        default=os.getenv("DNS_TRAINING_FILES"),
        help="Path-separated list of Zeek DNS training files.",
    )
    parser.add_argument("--window-size", default=os.getenv("DNS_WINDOW_SIZE", "5min"))
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--max-samples", default=DEFAULT_MAX_SAMPLES)
    parser.add_argument("--contamination", type=float, default=DEFAULT_CONTAMINATION)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--shap-sample-size", type=int, default=DEFAULT_SHAP_SAMPLE)
    parser.add_argument(
        "--silhouette-sample-size",
        type=int,
        default=DEFAULT_SILHOUETTE_SAMPLE,
    )
    parser.add_argument("--pca-fit-sample-size", type=int, default=DEFAULT_PCA_FIT_SAMPLE)
    parser.add_argument("--contour-grid-size", type=int, default=DEFAULT_CONTOUR_GRID_SIZE)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    if args.n_estimators < 10:
        raise ValueError("n_estimators must be at least 10.")
    if not 0 < args.contamination <= 0.5:
        raise ValueError("contamination must be greater than 0 and at most 0.5.")
    if args.shap_sample_size < 10:
        raise ValueError("SHAP sample size must be at least 10.")
    if args.silhouette_sample_size < 10:
        raise ValueError("Silhouette sample size must be at least 10.")
    if args.contour_grid_size < 20:
        raise ValueError("Contour grid size must be at least 20.")
    parse_max_samples(args.max_samples)


def main() -> int:
    configure_logging()
    args = build_parser().parse_args()
    try:
        validate_arguments(args)
        write_status(
            "starting",
            "Starting",
            1,
            "Starting CPU-only Isolation Forest learning in debug mode.",
        )
        train(args)
        return 0
    except KeyboardInterrupt:
        LOGGER.warning("Isolation Forest learning was interrupted by the user.")
        write_status(
            "stopped",
            "Stopped",
            0,
            "Isolation Forest learning was interrupted.",
        )
        return 130
    except Exception as exc:
        LOGGER.exception("Isolation Forest learning failed")
        write_status(
            "failed",
            "Failed",
            0,
            "Isolation Forest learning failed.",
            error=f"{type(exc).__name__}: {exc}",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
