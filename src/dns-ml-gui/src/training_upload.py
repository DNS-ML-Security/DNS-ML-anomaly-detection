# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CONFIG_PATH = Path("/opt/dns-ml-gui/config/training_input.json")
DOCUMENTATION_PATH = Path(
    "/opt/dns-ml-gui/docs/TRAINING_DATA_INPUT_OUTPUT.md"
)

HOME = Path("/data/dns-ml/training_upload")
INCOMING = HOME / "incoming"
FILES = HOME / "files"
STATE = HOME / "workflow_state.json"
MANIFEST = HOME / "upload_manifest.json"

ZEEK_OUT = Path("/data/dns-ml/zeek")
MERGED = ZEEK_OUT / "merged_dns.jsonl"
MERGE_STATUS = ZEEK_OUT / "merge_status.json"
MERGE_PID = ZEEK_OUT / "merge.pid"
ZEEK_MANAGER = Path("/opt/dns-ml/zeek_log_manager.py")

FEATURE_DIR = Path("/data/dns-ml/features")
FEATURE_CSV = FEATURE_DIR / "dns_training_features.csv"
FEATURE_SUMMARY = FEATURE_DIR / "dns_training_feature_summary.json"
FEATURE_STATUS = FEATURE_DIR / "dns_training_feature_status.json"
FEATURE_LOG = FEATURE_DIR / "dns_training_feature_extract.log"
FEATURE_PID = FEATURE_DIR / "dns_training_feature_extract.pid"
FEATURE_SCRIPT = Path("/opt/dns-ml/extract_dns_training_features.py")

TRAINING_DIR = Path("/data/dns-ml/training")
RUN_DIR = Path("/data/dns-ml/runs")

EMPTY = "empty"
UPLOADED = "uploaded"
MERGING = "merging"
MERGED_OK = "merged_verified"
EXTRACTING = "extracting_features"
READY = "features_ready"
ERROR = "error"

DEFAULT_INPUT_CONFIG = {
    "input_format": "json",
    "allowed_extensions": [".json", ".jsonl", ".log"],
    "allow_archives": False,
    "merge_all_uploaded_files": True,
    "require_all_files_valid": True,
    "uploaded_original_directory": str(INCOMING),
    "normalized_input_directory": str(FILES),
    "merged_output": str(MERGED),
    "feature_output": str(FEATURE_CSV),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except Exception:
        return default


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def input_spec():
    result = dict(DEFAULT_INPUT_CONFIG)
    value = read_json(CONFIG_PATH, {})
    if isinstance(value, dict):
        result.update(value)

    # Archives are intentionally unsupported for this workflow.
    result["allow_archives"] = False
    result["merge_all_uploaded_files"] = True

    allowed = result.get("allowed_extensions") or [".json", ".jsonl", ".log"]
    normalized = []
    for ext in allowed:
        ext = str(ext).strip().lower()
        if not ext.startswith("."):
            ext = "." + ext
        normalized.append(ext)
    result["allowed_extensions"] = normalized or [".json", ".jsonl", ".log"]
    return result


def default_state():
    return {
        "phase": EMPTY,
        "cycle_id": 1,
        "busy": False,
        "message": "Upload the complete Zeek DNS JSON/JSONL or JSON dns.log training batch.",
        "uploaded_original_files": 0,
        "accepted_dns_files": 0,
        "normalized_json_records": 0,
        "merge_verified": False,
        "features_extracted": False,
        "ready_for_ml": False,
        "error": None,
        "updated_at_utc": now(),
    }


def read_workflow_state():
    result = default_state()
    value = read_json(STATE, {})
    if isinstance(value, dict):
        result.update(value)
    return result


def update_state(**values):
    state = read_workflow_state()
    state.update(values)
    state["updated_at_utc"] = now()
    write_json(STATE, state)
    return state


def pid_running(path, token):
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
        cmd = Path(f"/proc/{pid}/cmdline")
        if cmd.exists():
            text = (
                cmd.read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
            if token not in text:
                return False
        return True
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def merge_is_running():
    return pid_running(MERGE_PID, ZEEK_MANAGER.name)


def is_feature_extraction_running():
    return pid_running(FEATURE_PID, FEATURE_SCRIPT.name)


def ml_process_active():
    return (
        pid_running(
            RUN_DIR / "latest_training.pid",
            "train_dns_autoencoder.py",
        )
        or pid_running(
            RUN_DIR / "latest_isolation_forest.pid",
            "train_dns_isolation_forest.py",
        )
    )


def safe_name(name):
    return (
        re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            Path(str(name)).name,
        ).strip("._")
        or "upload.json"
    )


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def allowed_json_name(name):
    spec = input_spec()
    lower = str(name).lower()
    return any(
        lower.endswith(ext)
        for ext in spec["allowed_extensions"]
    )


def staged_name(index):
    return f"dns_upload_{index:06d}.json"


def save_uploaded_stream(uploaded, destination):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    try:
        uploaded.seek(0)
    except Exception:
        pass

    with destination.open("wb") as output:
        while True:
            chunk = uploaded.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)

    try:
        uploaded.seek(0)
    except Exception:
        pass


def _write_records(records, destination):
    count = 0
    with destination.open(
        "w",
        encoding="utf-8",
    ) as output:
        for record in records:
            if not isinstance(record, dict):
                raise RuntimeError(
                    "Every JSON record must be an object/dictionary."
                )
            output.write(
                json.dumps(
                    record,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def normalize_json_file(source, destination):
    """
    Validate JSON and normalize it to one JSON object per line.

    Accepted source content:
      - Zeek JSON/NDJSON: one object per line
      - a single JSON object
      - a JSON array containing objects
    """
    if not source.exists() or source.stat().st_size <= 0:
        raise RuntimeError(
            f"JSON input is empty: {source.name}"
        )

    # First try line-oriented JSON. This is the normal Zeek JSON form.
    line_records = []
    line_mode_valid = True

    try:
        with source.open(
            "r",
            encoding="utf-8-sig",
        ) as handle:
            for line_number, raw in enumerate(handle, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError:
                    line_mode_valid = False
                    break

                if not isinstance(value, dict):
                    line_mode_valid = False
                    break

                line_records.append(value)
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"{source.name} is not valid UTF-8 JSON: {exc}"
        ) from exc

    if line_mode_valid and line_records:
        return _write_records(
            line_records,
            destination,
        )

    # Fallback for standard pretty-printed JSON object/array.
    try:
        with source.open(
            "r",
            encoding="utf-8-sig",
        ) as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{source.name} is not valid JSON "
            f"(line {exc.lineno}, column {exc.colno}): {exc.msg}"
        ) from exc

    if isinstance(value, dict):
        records = [value]
    elif isinstance(value, list):
        records = value
    else:
        raise RuntimeError(
            f"{source.name}: top-level JSON must be an object "
            f"or an array of objects."
        )

    if not records:
        raise RuntimeError(
            f"{source.name}: JSON contains zero records."
        )

    return _write_records(
        records,
        destination,
    )


def save_uploaded_batch(uploaded, progress_callback=None):
    state = reconcile_workflow_state()

    if state["phase"] != EMPTY:
        return {
            "saved": False,
            "error": (
                "A training upload cycle already exists. "
                "Clear it before starting a new JSON batch."
            ),
        }

    selected = list(uploaded or [])
    if progress_callback:
        progress_callback(0, len(selected), "batch", "starting")
    if not selected:
        return {
            "saved": False,
            "error": "Select at least one JSON or JSONL file.",
        }

    spec = input_spec()
    invalid_extensions = [
        getattr(item, "name", "unnamed")
        for item in selected
        if not allowed_json_name(
            getattr(item, "name", "")
        )
    ]

    if invalid_extensions:
        return {
            "saved": False,
            "error": (
                "Only JSON/JSONL training files are accepted. "
                "Rejected: "
                + ", ".join(invalid_extensions)
            ),
        }

    shutil.rmtree(
        HOME,
        ignore_errors=True,
    )
    INCOMING.mkdir(parents=True)
    FILES.mkdir(parents=True)

    originals = []
    accepted = []
    total_records = 0

    try:
        for index, uploaded_file in enumerate(
            selected,
            start=1,
        ):
            if progress_callback:
                progress_callback(
                    index - 1, len(selected),
                    getattr(uploaded_file, "name", f"file_{index}"),
                    "validating / staging",
                )
            original_name = safe_name(
                getattr(
                    uploaded_file,
                    "name",
                    f"upload_{index}.json",
                )
            )

            original_path = (
                INCOMING
                / f"{index:04d}_{original_name}"
            )
            save_uploaded_stream(
                uploaded_file,
                original_path,
            )

            staged_path = FILES / staged_name(index)
            record_count = normalize_json_file(
                original_path,
                staged_path,
            )
            total_records += record_count

            original_row = {
                "original_name": original_name,
                "original_path": str(original_path),
                "original_size_bytes": original_path.stat().st_size,
                "original_sha256": sha256(original_path),
                "staged_path": str(staged_path),
                "staged_size_bytes": staged_path.stat().st_size,
                "staged_sha256": sha256(staged_path),
                "normalized_json_records": record_count,
            }
            originals.append(original_row)
            accepted.append(str(staged_path))
            if progress_callback:
                progress_callback(
                    index, len(selected), original_name, "file complete"
                )

        manifest = {
            "created_at_utc": now(),
            "input_policy_path": str(CONFIG_PATH),
            "input_format": spec["input_format"],
            "allowed_extensions": spec["allowed_extensions"],
            "allow_archives": False,
            "merge_all_uploaded_files": True,
            "original_upload_count": len(originals),
            "staged_json_file_count": len(accepted),
            "normalized_json_records": total_records,
            "files": originals,
        }
        write_json(MANIFEST, manifest)

        update_state(
            phase=UPLOADED,
            busy=False,
            message=(
                f"JSON/JSONL upload complete: {len(originals)} files, "
                f"{total_records:,} normalized JSON records. "
                f"All files are staged for one merge."
            ),
            uploaded_original_files=len(originals),
            accepted_dns_files=len(accepted),
            normalized_json_records=total_records,
            merge_verified=False,
            features_extracted=False,
            ready_for_ml=False,
            error=None,
        )

        return {
            "saved": True,
            "original_files": len(originals),
            "accepted_dns_files": len(accepted),
            "records": total_records,
            "error": None,
        }

    except Exception as exc:
        update_state(
            phase=ERROR,
            busy=False,
            message="JSON/JSONL upload/validation failed.",
            ready_for_ml=False,
            error=str(exc),
        )
        return {
            "saved": False,
            "error": str(exc),
        }


def uploaded_training_files():
    if not FILES.exists():
        return []

    return sorted(
        [
            path
            for path in FILES.iterdir()
            if path.is_file()
            and path.name.lower().startswith("dns")
            and path.name.lower().endswith(".json")
        ],
        key=lambda path: path.name,
    )


def clear_zeek_outputs():
    for name in (
        "merged_dns.jsonl",
        "stats.json",
        "stats.sqlite",
        "source_ip_stats.csv",
        "merge_status.json",
        "merge_manifest.json",
        "merge.log",
        "merge.pid",
    ):
        try:
            (ZEEK_OUT / name).unlink(
                missing_ok=True
            )
        except Exception:
            pass


def start_uploaded_merge():
    state = reconcile_workflow_state()

    if state["phase"] != UPLOADED:
        return {
            "started": False,
            "error": (
                "Step 2 is enabled only after the JSON upload "
                "batch is validated."
            ),
        }

    files = uploaded_training_files()
    if not files:
        return {
            "started": False,
            "error": "No staged JSON/JSONL training files were found.",
        }

    clear_zeek_outputs()
    ZEEK_OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["ZEEK_LOG_ROOT"] = str(FILES)
    env["DNS_ZEEK_OUTPUT_DIR"] = str(ZEEK_OUT)

    merge_log = (
        ZEEK_OUT / "merge.log"
    ).open(
        "w",
        encoding="utf-8",
    )

    try:
        process = subprocess.Popen(
            [
                sys.executable,
                str(ZEEK_MANAGER),
            ],
            stdout=merge_log,
            stderr=subprocess.STDOUT,
            cwd=str(ZEEK_MANAGER.parent),
            env=env,
            start_new_session=True,
        )
    finally:
        merge_log.close()

    MERGE_PID.write_text(
        str(process.pid),
        encoding="utf-8",
    )

    update_state(
        phase=MERGING,
        busy=True,
        message=(
            f"Merging all {len(files)} staged JSON files "
            f"into one deduplicated training dataset."
        ),
        merge_verified=False,
        features_extracted=False,
        ready_for_ml=False,
        error=None,
    )

    return {
        "started": True,
        "pid": process.pid,
        "files": len(files),
        "error": None,
    }


def verify_merge(status):
    expected = len(
        uploaded_training_files()
    )
    total = int(
        status.get(
            "files_total",
            0,
        )
        or 0
    )
    processed = int(
        status.get(
            "files_processed",
            0,
        )
        or 0
    )
    invalid = int(
        status.get(
            "invalid_lines",
            0,
        )
        or 0
    )
    records = int(
        status.get(
            "records_inserted",
            0,
        )
        or 0
    )

    if status.get("state") != "finished":
        return (
            False,
            status.get("error")
            or "Merge did not finish successfully.",
        )

    if expected <= 0:
        return (
            False,
            "No staged JSON/JSONL files exist.",
        )

    if total != expected:
        return (
            False,
            (
                f"Not all uploaded JSON/JSONL files entered the merge: "
                f"staged={expected}, manager discovered={total}."
            ),
        )

    if processed != total:
        return (
            False,
            (
                f"Not all uploaded JSON/JSONL files were processed: "
                f"{processed}/{total}."
            ),
        )

    if records <= 0:
        return (
            False,
            "Merge produced zero usable DNS records.",
        )

    if invalid > 0:
        return (
            False,
            (
                f"Merge found {invalid} invalid DNS JSON records. "
                f"Every uploaded JSON file must verify cleanly."
            ),
        )

    if not MERGED.exists() or MERGED.stat().st_size <= 0:
        return (
            False,
            "Merged DNS dataset is missing or empty.",
        )

    return (
        True,
        (
            f"Verified and merged all {processed} JSON files "
            f"into {records:,} unique DNS records."
        ),
    )


def start_feature_extraction():
    state = reconcile_workflow_state()

    if state["phase"] != MERGED_OK:
        return {
            "started": False,
            "error": (
                "Step 3 is enabled only after all JSON files "
                "are merged and verified."
            ),
        }

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in (
        FEATURE_CSV,
        FEATURE_SUMMARY,
        FEATURE_STATUS,
        FEATURE_LOG,
        FEATURE_PID,
    ):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = (
        f"/opt/dns-ml:{env.get('PYTHONPATH', '')}"
    )

    feature_log = FEATURE_LOG.open(
        "w",
        encoding="utf-8",
    )

    try:
        process = subprocess.Popen(
            [
                sys.executable,
                str(FEATURE_SCRIPT),
            ],
            stdout=feature_log,
            stderr=subprocess.STDOUT,
            cwd=str(FEATURE_SCRIPT.parent),
            env=env,
            start_new_session=True,
        )
    finally:
        feature_log.close()

    FEATURE_PID.write_text(
        str(process.pid),
        encoding="utf-8",
    )

    update_state(
        phase=EXTRACTING,
        busy=True,
        message=(
            "Analyzing the merged JSON dataset and extracting ML features."
        ),
        features_extracted=False,
        ready_for_ml=False,
        error=None,
    )

    return {
        "started": True,
        "pid": process.pid,
        "error": None,
    }


def read_feature_summary():
    return read_json(
        FEATURE_SUMMARY,
        {},
    )


def read_feature_status():
    return read_json(
        FEATURE_STATUS,
        {},
    )


def read_feature_log_tail(lines=300):
    try:
        return "\n".join(
            FEATURE_LOG.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()[-lines:]
        )
    except Exception:
        return (
            "No feature-extraction output is available yet."
        )


def reconcile_workflow_state():
    state = read_workflow_state()

    if (
        state["phase"] == MERGING
        and not merge_is_running()
    ):
        valid, message = verify_merge(
            read_json(
                MERGE_STATUS,
                {},
            )
        )

        return update_state(
            phase=MERGED_OK if valid else ERROR,
            busy=False,
            message=(
                message
                if valid
                else "JSON merge verification failed."
            ),
            merge_verified=valid,
            features_extracted=False,
            ready_for_ml=False,
            error=None if valid else message,
        )

    if (
        state["phase"] == EXTRACTING
        and not is_feature_extraction_running()
    ):
        status = read_feature_status()
        summary = read_feature_summary()

        valid = bool(
            status.get("state") == "finished"
            and summary.get("ready_for_ml") is True
            and FEATURE_CSV.exists()
            and FEATURE_CSV.stat().st_size > 0
        )

        return update_state(
            phase=READY if valid else ERROR,
            busy=False,
            message=(
                (
                    f"Feature extraction complete: "
                    f"{int(summary.get('training_windows', 0)):,} "
                    f"windows. ML learning unlocked."
                )
                if valid
                else "Feature extraction failed."
            ),
            merge_verified=True,
            features_extracted=valid,
            ready_for_ml=valid,
            error=(
                None
                if valid
                else status.get(
                    "error",
                    "Feature extraction failed.",
                )
            ),
        )

    return state


def learning_input_ready():
    state = reconcile_workflow_state()
    summary = read_feature_summary()

    return bool(
        state["phase"] == READY
        and state.get("merge_verified")
        and state.get("features_extracted")
        and summary.get("ready_for_ml")
        and MERGED.exists()
        and MERGED.stat().st_size > 0
        and FEATURE_CSV.exists()
        and FEATURE_CSV.stat().st_size > 0
    )


def learning_gate_message():
    state = reconcile_workflow_state()
    phase = state["phase"]

    return {
        EMPTY: (
            "ML learning is locked. Step 1: upload the complete "
            "JSON training batch."
        ),
        UPLOADED: (
            "ML learning is locked. Step 2: merge and verify "
            "all uploaded JSON files."
        ),
        MERGING: (
            "ML learning is locked while all JSON files are "
            "being merged and verified."
        ),
        MERGED_OK: (
            "ML learning is locked. Step 3: analyze the merged "
            "JSON dataset and extract features."
        ),
        EXTRACTING: (
            "ML learning is locked while feature extraction is running."
        ),
        READY: "ML learning input is ready.",
        ERROR: (
            "ML learning is locked: "
            + str(
                state.get("error")
                or state.get("message")
            )
        ),
    }.get(
        phase,
        (
            "ML learning is locked until JSON upload, merge "
            "verification, and feature extraction complete."
        ),
    )


def clear_training_cycle():
    state = reconcile_workflow_state()

    if (
        merge_is_running()
        or is_feature_extraction_running()
    ):
        return {
            "cleared": False,
            "error": (
                "Cannot clear while data preparation is running."
            ),
        }

    if ml_process_active():
        return {
            "cleared": False,
            "error": (
                "Cannot clear while an ML learning process is running."
            ),
        }

    cycle = (
        int(
            state.get(
                "cycle_id",
                1,
            )
            or 1
        )
        + 1
    )

    shutil.rmtree(
        HOME,
        ignore_errors=True,
    )
    HOME.mkdir(
        parents=True,
        exist_ok=True,
    )

    clear_zeek_outputs()

    for path in (
        FEATURE_CSV,
        FEATURE_SUMMARY,
        FEATURE_STATUS,
        FEATURE_LOG,
        FEATURE_PID,
    ):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass

    if TRAINING_DIR.exists():
        for path in TRAINING_DIR.iterdir():
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except Exception:
                pass

    state = default_state()
    state["cycle_id"] = cycle
    state["message"] = (
        "Training cycle cleared. Ready for a new JSON/JSONL upload batch."
    )
    write_json(
        STATE,
        state,
    )

    return {
        "cleared": True,
        "cycle_id": cycle,
        "error": None,
    }
