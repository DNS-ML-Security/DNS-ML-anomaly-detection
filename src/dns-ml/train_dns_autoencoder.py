#!/usr/bin/env python3
# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

"""Train the DNS Autoencoder and publish GUI learning artifacts."""

from __future__ import annotations

import glob
import json
import os
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from feature_utils import (
    FEATURES,
    add_event_features,
    build_dns_windows,
    calculate_reconstruction_error,
    load_zeek_dns_json_files,
    validate_feature_matrix,
)


TRAINING_DIR = Path(os.getenv("DNS_TRAINING_DIR", "/data/dns-ml/training"))
MODEL_DIR = Path(os.getenv("DNS_MODEL_DIR", "/data/dns-ml/models"))
FEATURE_DIR = Path(os.getenv("DNS_FEATURE_DIR", "/data/dns-ml/features"))
RUN_DIR = Path(os.getenv("DNS_RUN_DIR", "/data/dns-ml/runs"))

MODEL_PATH = MODEL_DIR / "dns_autoencoder.keras"
SCALER_PATH = MODEL_DIR / "dns_scaler.joblib"
CONFIG_PATH = MODEL_DIR / "dns_autoencoder_config.json"
BACKUP_DIR = MODEL_DIR / "backup"

STATUS_PATH = Path(os.getenv("DNS_LEARNING_STATUS_PATH", RUN_DIR / "latest_training_status.json"))
HISTORY_PATH = Path(os.getenv("DNS_LEARNING_HISTORY_PATH", RUN_DIR / "latest_training_history.csv"))
RECONSTRUCTION_PATH = Path(
    os.getenv("DNS_LEARNING_RECONSTRUCTION_PATH", RUN_DIR / "latest_reconstruction_errors.csv")
)
RESULT_PATH = Path(os.getenv("DNS_LEARNING_RESULT_PATH", RUN_DIR / "latest_training_result.json"))

WINDOW_SIZE = os.getenv("DNS_WINDOW_SIZE", "5min")
THRESHOLD_QUANTILE = float(os.getenv("DNS_THRESHOLD_QUANTILE", "0.99"))
MIN_WINDOWS_REQUIRED = int(os.getenv("DNS_MIN_WINDOWS_REQUIRED", "50"))
EXPECTED_FEATURE_COUNT = int(os.getenv("DNS_EXPECTED_FEATURE_COUNT", str(len(FEATURES))))
EPOCHS = int(os.getenv("DNS_TRAINING_EPOCHS", "50"))
BATCH_SIZE = int(os.getenv("DNS_BATCH_SIZE", "256"))
RANDOM_STATE = int(os.getenv("DNS_RANDOM_STATE", "42"))
KERAS_VERBOSE = int(os.getenv("DNS_KERAS_VERBOSE", "2"))
SHOW_MODEL_SUMMARY = os.getenv("DNS_SHOW_MODEL_SUMMARY", "1") == "1"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{now_utc()}] {message}", flush=True)


def print_console_table(title: str, rows: list[tuple[str, object]]) -> None:
    """Print a compact two-column table to the terminal and GUI training log."""
    if not rows:
        return

    label_width = max(len(str(label)) for label, _ in rows)
    value_width = max(len(str(value)) for _, value in rows)
    label_width = max(label_width, len("Metric"))
    value_width = max(value_width, len("Value"))

    border = f"+{'-' * (label_width + 2)}+{'-' * (value_width + 2)}+"
    print("", flush=True)
    print(title, flush=True)
    print(border, flush=True)
    print(f"| {'Metric'.ljust(label_width)} | {'Value'.ljust(value_width)} |", flush=True)
    print(border, flush=True)
    for label, value in rows:
        print(
            f"| {str(label).ljust(label_width)} | {str(value).ljust(value_width)} |",
            flush=True,
        )
    print(border, flush=True)
    print("", flush=True)


def print_history_table(history: tf.keras.callbacks.History) -> None:
    """Print epoch-level training and validation losses as a terminal table."""
    losses = history.history.get("loss", [])
    val_losses = history.history.get("val_loss", [])
    if not losses:
        return

    table = pd.DataFrame(
        {
            "Epoch": np.arange(1, len(losses) + 1),
            "Training Loss": losses,
            "Validation Loss": val_losses,
            "Training Quality %": [reconstruction_quality_pct(v) for v in losses],
            "Validation Quality %": [reconstruction_quality_pct(v) for v in val_losses],
        }
    )

    print("", flush=True)
    print("TRAINING HISTORY TABLE", flush=True)
    print(
        table.to_string(
            index=False,
            formatters={
                "Training Loss": lambda value: f"{value:.6f}",
                "Validation Loss": lambda value: f"{value:.6f}",
                "Training Quality %": lambda value: f"{value:.4f}",
                "Validation Quality %": lambda value: f"{value:.4f}",
            },
        ),
        flush=True,
    )
    print("", flush=True)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def update_status(
    state: str,
    stage: str,
    progress: int,
    message: str,
    error: str | None = None,
    extra: dict | None = None,
) -> None:
    payload = {
        "state": state,
        "stage": stage,
        "progress_percent": int(max(0, min(progress, 100))),
        "message": message,
        "error": error,
        "updated_at_utc": now_utc(),
    }
    if extra:
        payload.update(extra)
    write_json_atomic(STATUS_PATH, payload)


def parse_explicit_training_files() -> list[str]:
    """Return GUI-supplied training files from DNS_TRAINING_FILES."""
    raw = os.getenv("DNS_TRAINING_FILES", "").strip()
    if not raw:
        return []

    normalized = raw.replace("\n", os.pathsep).replace(",", os.pathsep)
    files = [
        item.strip()
        for item in normalized.split(os.pathsep)
        if item.strip()
    ]
    return sorted(dict.fromkeys(files))


def get_training_files() -> list[str]:
    explicit = parse_explicit_training_files()
    if explicit:
        missing = [path for path in explicit if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(
                "Explicit training files do not exist: " + ", ".join(missing)
            )
        return explicit

    patterns = [
        "*.json",
        "*.jsonl",
        "*.log",
        "*.json.gz",
        "*.jsonl.gz",
        "*.log.gz",
    ]
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(str(TRAINING_DIR / pattern)))
    return sorted(set(files))

def build_autoencoder(input_dim: int, num_samples: int) -> tf.keras.Model:
    if num_samples < 1000:
        layers = [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(12, activation="relu", name="latent_layer"),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(input_dim, activation="linear"),
        ]
    else:
        layers = [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(96, activation="relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(48, activation="relu"),
            tf.keras.layers.Dense(18, activation="relu", name="latent_layer"),
            tf.keras.layers.Dense(48, activation="relu"),
            tf.keras.layers.Dense(96, activation="relu"),
            tf.keras.layers.Dense(input_dim, activation="linear"),
        ]

    model = tf.keras.Sequential(layers, name="dns_dense_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mae")
    return model



def show_model_architecture_table(model: tf.keras.Model) -> None:
    """Print the old terminal-style layer table plus the Keras summary."""
    rows: list[dict] = []
    for index, layer in enumerate(model.layers, start=1):
        try:
            output_shape = tuple(layer.output.shape)
        except Exception:
            output_shape = "N/A"

        try:
            activation = layer.get_config().get("activation", "-")
        except Exception:
            activation = "-"

        rows.append(
            {
                "No.": index,
                "Layer name": layer.name,
                "Layer type": layer.__class__.__name__,
                "Output shape": str(output_shape),
                "Parameters": f"{layer.count_params():,}",
                "Activation": activation,
            }
        )

    print("", flush=True)
    print("AUTOENCODER LAYER TABLE", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print("", flush=True)
    print("KERAS MODEL SUMMARY", flush=True)
    model.summary(print_fn=lambda line: print(line, flush=True))
    print("", flush=True)
def reconstruction_quality_pct(loss_value: float) -> float:
    """Visualization proxy only; this is not supervised classification accuracy."""
    return round(100.0 / (1.0 + max(float(loss_value), 0.0)), 4)


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs
        self.rows: list[dict] = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = float(logs.get("loss", 0.0))
        val_loss = float(logs.get("val_loss", 0.0))
        row = {
            "epoch": int(epoch + 1),
            "loss": loss,
            "val_loss": val_loss,
            "train_reconstruction_quality_pct": reconstruction_quality_pct(loss),
            "val_reconstruction_quality_pct": reconstruction_quality_pct(val_loss),
            "updated_at_utc": now_utc(),
        }
        self.rows.append(row)
        HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(HISTORY_PATH, index=False)

        progress = min(80, 45 + int(((epoch + 1) / max(self.total_epochs, 1)) * 35))
        update_status(
            state="running",
            stage="Training epochs",
            progress=progress,
            message=(
                f"Epoch {epoch + 1}/{self.total_epochs} completed | "
                f"loss={loss:.6f} | val_loss={val_loss:.6f}"
            ),
            extra={
                "current_epoch": int(epoch + 1),
                "total_epochs": int(self.total_epochs),
                "loss": loss,
                "val_loss": val_loss,
            },
        )
        if KERAS_VERBOSE == 0:
            log(
                f"Epoch {epoch + 1}/{self.total_epochs} - "
                f"loss={loss:.6f} - val_loss={val_loss:.6f}"
            )


def backup_current_artifacts() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for source in [MODEL_PATH, SCALER_PATH, CONFIG_PATH]:
        if source.exists():
            destination = BACKUP_DIR / f"{source.name}.{timestamp}.bak"
            shutil.copy2(source, destination)
            log(f"Backed up current artifact: {destination}")


def publish_artifacts(model: tf.keras.Model, scaler: StandardScaler, config: dict) -> None:
    """Stage and validate artifacts before replacing current files."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    staging_dir = MODEL_DIR / ".staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    staged_model = staging_dir / MODEL_PATH.name
    staged_scaler = staging_dir / SCALER_PATH.name
    staged_config = staging_dir / CONFIG_PATH.name

    model.save(staged_model)
    joblib.dump(scaler, staged_scaler)
    write_json_atomic(staged_config, config)

    # Validate staged files before touching the active set.
    loaded_model = tf.keras.models.load_model(staged_model, compile=False)
    loaded_scaler = joblib.load(staged_scaler)
    with open(staged_config, "r", encoding="utf-8") as handle:
        loaded_config = json.load(handle)
    if int(loaded_model.input_shape[-1]) != len(FEATURES):
        raise RuntimeError("Staged model input dimension does not match feature count.")
    if int(getattr(loaded_scaler, "n_features_in_", -1)) != len(FEATURES):
        raise RuntimeError("Staged scaler feature count does not match feature count.")
    if loaded_config.get("features") != FEATURES:
        raise RuntimeError("Staged configuration feature order does not match feature_utils.FEATURES.")

    backup_current_artifacts()
    os.replace(staged_model, MODEL_PATH)
    os.replace(staged_scaler, SCALER_PATH)
    os.replace(staged_config, CONFIG_PATH)
    shutil.rmtree(staging_dir, ignore_errors=True)


def main() -> None:
    try:
        RUN_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        FEATURE_DIR.mkdir(parents=True, exist_ok=True)

        update_status("running", "Initialization", 2, "Training process initialized.")
        log("DNS Autoencoder training started.")
        log("Training input: /data/dns-ml/features/dns_training_features.csv")
        log(f"Window size: {WINDOW_SIZE}")
        log(f"Expected feature count: {EXPECTED_FEATURE_COUNT}")

        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

        feature_input_path = Path(
            os.getenv(
                "DNS_PRECOMPUTED_FEATURES",
                "/data/dns-ml/features/dns_training_features.csv",
            )
        )
        feature_summary_path = FEATURE_DIR / "dns_training_feature_summary.json"

        update_status(
            "running",
            "Loading precomputed features",
            15,
            f"Loading shared feature dataset: {feature_input_path}",
            extra={"training_files": [str(feature_input_path)]},
        )

        if not feature_input_path.exists() or feature_input_path.stat().st_size == 0:
            raise RuntimeError(
                f"Precomputed feature dataset is missing or empty: {feature_input_path}"
            )

        windows_df = pd.read_csv(feature_input_path)
        if windows_df.empty:
            raise RuntimeError("Precomputed feature dataset contains zero windows.")

        if len(windows_df) < MIN_WINDOWS_REQUIRED:
            raise RuntimeError(
                f"Only {len(windows_df)} precomputed windows found. "
                f"Minimum required is {MIN_WINDOWS_REQUIRED}."
            )

        # DNS_METADATA_PRESERVATION_HOTFIX_V5_3_AE
        metadata_columns = ["timestamp", "src_ip"]
        missing_metadata = [
            column for column in metadata_columns if column not in windows_df.columns
        ]
        if missing_metadata:
            raise RuntimeError(
                "Shared feature CSV is missing metadata columns: "
                + ", ".join(missing_metadata)
            )

        windows_df = windows_df.reset_index(drop=True)
        feature_matrix_df = validate_feature_matrix(
            windows_df,
            FEATURES,
        ).reset_index(drop=True)

        if len(windows_df) != len(feature_matrix_df):
            raise RuntimeError(
                "Metadata/feature row mismatch: "
                f"{len(windows_df)} metadata rows versus "
                f"{len(feature_matrix_df)} validated feature rows"
            )
        if windows_df["timestamp"].fillna("").astype(str).str.strip().eq("").any():
            raise RuntimeError("Shared feature CSV contains blank timestamp values")
        if windows_df["src_ip"].fillna("").astype(str).str.strip().eq("").any():
            raise RuntimeError("Shared feature CSV contains blank src_ip values")
        if len(FEATURES) != EXPECTED_FEATURE_COUNT:
            raise RuntimeError(
                f"Feature count mismatch: actual={len(FEATURES)}, "
                f"expected={EXPECTED_FEATURE_COUNT}"
            )

        summary = {}
        try:
            if feature_summary_path.exists():
                summary = json.loads(feature_summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}

        source_raw_records = int(
            summary.get("raw_dns_records", len(windows_df)) or len(windows_df)
        )
        source_usable_events = int(
            summary.get("usable_dns_events", len(windows_df)) or len(windows_df)
        )
        source_ip_count = int(
            summary.get(
                "unique_source_ips",
                windows_df["src_ip"].astype(str).nunique()
                if "src_ip" in windows_df.columns
                else 0,
            )
            or 0
        )

        log(f"Precomputed feature windows loaded: {len(windows_df):,}")
        log(f"Feature columns consumed: {len(FEATURES)}")
        log(
            "Feature engineering/window aggregation are NOT repeated "
            "by the Autoencoder trainer."
        )

        print_console_table(
            "PRECOMPUTED TRAINING FEATURE DATASET",
            [
                ("Feature CSV", feature_input_path),
                ("Training windows", f"{len(windows_df):,}"),
                ("Source raw DNS records", f"{source_raw_records:,}"),
                ("Source usable DNS events", f"{source_usable_events:,}"),
                ("Unique source IPs", f"{source_ip_count:,}"),
                ("Feature count", len(FEATURES)),
                ("Window size", summary.get("window_size", WINDOW_SIZE)),
            ],
        )
        update_status("running", "Scaling features", 42, "Scaling training feature matrix.")
        x = feature_matrix_df[FEATURES].to_numpy(dtype="float32")
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x).astype("float32")

        x_train, x_val = train_test_split(
            x_scaled,
            test_size=0.2,
            random_state=RANDOM_STATE,
            shuffle=True,
        )
        log(f"Training matrix shape: {x_train.shape}")
        log(f"Validation matrix shape: {x_val.shape}")

        print_console_table(
            "TRAINING / VALIDATION SPLIT",
            [
                ("Training windows", f"{len(x_train):,}"),
                ("Validation windows", f"{len(x_val):,}"),
                ("Input dimensions", x_train.shape[1]),
                ("Batch size", min(BATCH_SIZE, max(8, len(x_train)))),
                ("Epochs requested", EPOCHS),
                ("Random state", RANDOM_STATE),
            ],
        )

        update_status("running", "Model training", 45, "Training Autoencoder model.")
        model = build_autoencoder(input_dim=len(FEATURES), num_samples=len(x_train))

        if SHOW_MODEL_SUMMARY:
            show_model_architecture_table(model)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True
        )
        progress_callback = TrainingProgressCallback(total_epochs=EPOCHS)
        history = model.fit(
            x_train,
            x_train,
            epochs=EPOCHS,
            batch_size=min(BATCH_SIZE, max(8, len(x_train))),
            shuffle=True,
            validation_data=(x_val, x_val),
            callbacks=[early_stop, progress_callback],
            verbose=KERAS_VERBOSE,
        )

        print_history_table(history)

        epochs_trained = len(history.history.get("loss", []))
        best_val_loss = float(min(history.history.get("val_loss", [0.0])))
        log(f"Training finished after {epochs_trained} epochs.")
        log(f"Best validation loss: {best_val_loss:.6f}")

        update_status("running", "Reconstruction scoring", 82, "Calculating reconstruction errors.")
        train_pred = model.predict(x_train, verbose=0)
        val_pred = model.predict(x_val, verbose=0)
        train_errors = calculate_reconstruction_error(x_train, train_pred)
        val_errors = calculate_reconstruction_error(x_val, val_pred)

        threshold = float(np.quantile(val_errors, THRESHOLD_QUANTILE))
        threshold_95 = float(np.quantile(val_errors, 0.95))
        threshold_99 = float(np.quantile(val_errors, 0.99))
        threshold_mean_3std = float(np.mean(val_errors) + 3 * np.std(val_errors))

        reconstruction_df = pd.concat(
            [
                pd.DataFrame({"split": "train", "reconstruction_error": train_errors}),
                pd.DataFrame({"split": "validation", "reconstruction_error": val_errors}),
            ],
            ignore_index=True,
        )
        reconstruction_df.to_csv(RECONSTRUCTION_PATH, index=False)
        log(f"Reconstruction error file saved: {RECONSTRUCTION_PATH}")
        log(f"Threshold: {threshold:.6f}")
        log(f"Threshold 95: {threshold_95:.6f}")
        log(f"Threshold 99: {threshold_99:.6f}")
        log(f"Threshold mean + 3 std: {threshold_mean_3std:.6f}")

        print_console_table(
            "RECONSTRUCTION ERROR AND THRESHOLD SUMMARY",
            [
                ("Train error mean", f"{np.mean(train_errors):.6f}"),
                ("Train error std", f"{np.std(train_errors):.6f}"),
                ("Validation error mean", f"{np.mean(val_errors):.6f}"),
                ("Validation error std", f"{np.std(val_errors):.6f}"),
                ("Selected quantile", THRESHOLD_QUANTILE),
                ("Selected threshold", f"{threshold:.6f}"),
                ("95th percentile", f"{threshold_95:.6f}"),
                ("99th percentile", f"{threshold_99:.6f}"),
                ("Mean + 3 STD", f"{threshold_mean_3std:.6f}"),
            ],
        )

        update_status("running", "Publishing model", 92, "Validating and publishing model artifacts.")
        config = {
            "model_name": "DNS_Dense_Autoencoder_36_Features_CPU",
            "feature_count": len(FEATURES),
            "input_dim": len(FEATURES),
            "features": FEATURES,
            "training_windows": int(len(windows_df)),
            "training_records": int(source_raw_records),
            "unique_source_ips": int(source_ip_count),
            "window_size": WINDOW_SIZE,
            "threshold_quantile": THRESHOLD_QUANTILE,
            "threshold": threshold,
            "threshold_95": threshold_95,
            "threshold_99": threshold_99,
            "threshold_mean_3std": threshold_mean_3std,
            "loss": "mae",
            "epochs_requested": EPOCHS,
            "epochs_trained": int(epochs_trained),
            "best_val_loss": best_val_loss,
            "created_at_utc": now_utc(),
            "model_path": str(MODEL_PATH),
            "scaler_path": str(SCALER_PATH),
            "config_path": str(CONFIG_PATH),
            "history_path": str(HISTORY_PATH),
            "reconstruction_errors_path": str(RECONSTRUCTION_PATH),
            "fast_flux_features_enabled": True,
            "cpu_only": True,
        }
        publish_artifacts(model, scaler, config)

        result = {**config, "completed_at_utc": now_utc(), "status": "success"}
        write_json_atomic(RESULT_PATH, result)
        update_status("finished", "Completed", 100, "Learning completed successfully.", extra=result)

        print_console_table(
            "FINAL TRAINING RESULT",
            [
                ("Status", "SUCCESS"),
                ("Epochs trained", epochs_trained),
                ("Best validation loss", f"{best_val_loss:.6f}"),
                ("Training windows", f"{len(windows_df):,}"),
                ("Features", len(FEATURES)),
                ("Anomaly threshold", f"{threshold:.6f}"),
                ("Model", MODEL_PATH),
                ("Scaler", SCALER_PATH),
                ("Configuration", CONFIG_PATH),
            ],
        )

        log(f"Model saved: {MODEL_PATH}")
        log(f"Scaler saved: {SCALER_PATH}")
        log(f"Config saved: {CONFIG_PATH}")
        log("DNS Autoencoder training completed successfully.")

    except Exception as exc:
        error_message = str(exc)
        log(f"ERROR: {error_message}")
        traceback.print_exc()
        update_status("error", "Failed", 100, "Learning failed.", error=error_message)
        raise


if __name__ == "__main__":
    main()
