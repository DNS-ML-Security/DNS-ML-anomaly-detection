#!/usr/bin/env python3
# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

"""Run final DNS correlation only after detector scoring has completed."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GUI_ROOT = Path(os.getenv("DNS_GUI_ROOT", "/opt/dns-ml-gui"))
if str(GUI_ROOT) not in sys.path:
    sys.path.insert(0, str(GUI_ROOT))

from src import final_scoring  # noqa: E402


FINAL_HOME = Path(os.getenv("DNS_FINAL_SCORING_HOME", "/data/dns-ml/final_scoring"))
STATUS_PATH = final_scoring.FINAL_STATUS_PATH
PID_PATH = FINAL_HOME / "correlation.pid"
LOCK_PATH = FINAL_HOME / "correlation.lock"
LIFECYCLE_PATH = Path(
    os.getenv("DNS_LIVE_SCORING_STATE_PATH", "/data/dns-ml/scoring/live_scoring_state.json")
)
DETECTORS = {
    "Autoencoder": (
        Path(os.getenv("DNS_SCORING_STATUS_PATH", "/data/dns-ml/scoring/status.json")),
        Path(os.getenv("DNS_SCORING_PID_PATH", "/data/dns-ml/scoring/manual_scoring.pid")),
        "score_dns_autoencoder_live.py",
    ),
    "Isolation Forest": (
        Path(os.getenv("DNS_IF_SCORING_STATUS_PATH", "/data/dns-ml/isolation_scoring/status.json")),
        Path(os.getenv("DNS_IF_SCORING_PID_PATH", "/data/dns-ml/isolation_scoring/scoring.pid")),
        "score_dns_isolation_forest_live.py",
    ),
    "DNS Heuristics": (
        Path(os.getenv("DNS_HEUR_STATUS_PATH", "/data/dns-ml/heuristics_scoring/status.json")),
        Path(os.getenv("DNS_HEUR_PID_PATH", "/data/dns-ml/heuristics_scoring/scoring.pid")),
        "score_dns_heuristics_live.py",
    ),
}
TERMINAL_STATES = {"finished", "completed", "success", "succeeded", "no_data"}
FAILED_STATES = {"failed", "error"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_status(state: str, stage: str, progress: int, message: str, **extra: Any) -> None:
    write_json_atomic(
        STATUS_PATH,
        {
            "state": state,
            "stage": stage,
            "progress_percent": max(0, min(100, int(progress))),
            "message": message,
            "updated_at_utc": utc_now(),
            **extra,
        },
    )


def pid_running(pid: int, expected_script: str | None = None) -> bool:
    try:
        stat_fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        if len(stat_fields) > 2 and stat_fields[2] == "Z":
            return False
        if expected_script:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(
                "utf-8", errors="replace"
            )
            return expected_script in command
        return True
    except Exception:
        return False


def detector_is_running(pid_path: Path, script_name: str) -> bool:
    try:
        return pid_running(int(pid_path.read_text(encoding="utf-8").strip()), script_name)
    except Exception:
        return False


def parse_pid_specs(values: list[str]) -> list[tuple[str, int]]:
    parsed: list[tuple[str, int]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --pid value: {value!r}")
        label, raw_pid = value.rsplit("=", 1)
        parsed.append((label.strip() or "Detector", int(raw_pid)))
    return parsed


def wait_for_manual_detectors(pid_specs: list[tuple[str, int]], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while True:
        running = [(label, pid) for label, pid in pid_specs if pid_running(pid)]
        if not running:
            return
        if time.monotonic() >= deadline:
            labels = ", ".join(f"{label}({pid})" for label, pid in running)
            raise TimeoutError(f"Timed out waiting for detector processes: {labels}")
        write_status(
            "waiting",
            "Waiting for ML and heuristic scoring",
            15,
            "Final Correlation is queued until Autoencoder, Isolation Forest, and DNS Heuristics finish.",
            running_detectors=[label for label, _ in running],
        )
        time.sleep(2)


def wait_for_scheduled_detectors(timeout: int) -> None:
    cycle_floor = time.time() - 120
    deadline = time.monotonic() + timeout
    stable_idle_since: float | None = None
    while True:
        running: list[str] = []
        fresh: list[str] = []
        for label, (status_path, pid_path, script_name) in DETECTORS.items():
            status = read_json(status_path)
            state = str(status.get("state") or "").lower()
            status_pid = int(status.get("pid", 0) or 0)
            status_running = state in {"starting", "launching", "loading", "running", "scoring", "saving"}
            if (status_running and status_pid and pid_running(status_pid, script_name)) or detector_is_running(pid_path, script_name):
                running.append(label)
            try:
                if status_path.stat().st_mtime >= cycle_floor:
                    fresh.append(label)
            except OSError:
                pass

        if len(fresh) == len(DETECTORS) and not running:
            if stable_idle_since is None:
                stable_idle_since = time.monotonic()
            elif time.monotonic() - stable_idle_since >= 4:
                return
        else:
            stable_idle_since = None

        if time.monotonic() >= deadline:
            missing = sorted(set(DETECTORS) - set(fresh))
            raise TimeoutError(
                "Timed out waiting for the scheduled detector cycle; "
                f"running={running or 'none'}, fresh status missing={missing or 'none'}"
            )
        write_status(
            "waiting",
            "Waiting for scheduled detector cycle",
            15,
            "Final Correlation is queued behind the three detector schedules.",
            running_detectors=running,
            fresh_detector_statuses=fresh,
        )
        time.sleep(2)


def validate_detector_results() -> dict[str, dict[str, Any]]:
    statuses = {label: read_json(paths[0]) for label, paths in DETECTORS.items()}
    failures = {
        label: str(status.get("message") or status.get("error") or status.get("state"))
        for label, status in statuses.items()
        if str(status.get("state") or "").lower() in FAILED_STATES
    }
    incomplete = {
        label: str(status.get("state") or "missing")
        for label, status in statuses.items()
        if str(status.get("state") or "").lower() not in TERMINAL_STATES | FAILED_STATES
    }
    if failures:
        raise RuntimeError(f"Detector failure prevents final correlation: {failures}")
    if incomplete:
        raise RuntimeError(f"Detector status is not finalized: {incomplete}")
    return statuses


def update_lifecycle(job_id: str | None, *, state: str, error: str | None = None) -> None:
    if not job_id:
        return
    lifecycle = read_json(LIFECYCLE_PATH)
    if str(lifecycle.get("manual_job_id") or "") != job_id:
        return
    lifecycle.update(
        {
            "state": state,
            "final_correlation_completed_at_utc": utc_now() if state == "completed" else None,
            "completed_at_utc": utc_now() if state == "completed" else lifecycle.get("completed_at_utc"),
            "last_error": error,
            "updated_at_utc": utc_now(),
        }
    )
    write_json_atomic(LIFECYCLE_PATH, lifecycle)


def run(args: argparse.Namespace) -> int:
    FINAL_HOME.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
    lock_handle = None
    try:
        if args.manual:
            lock_handle = LOCK_PATH.open("a+")
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"[{utc_now()}] Final Correlation is already running.", flush=True)
                return 0
            wait_for_manual_detectors(parse_pid_specs(args.pid), args.timeout)
        else:
            # A scheduled coordinator must finish or fail before the next
            # five-minute cycle instead of holding the cron lock indefinitely.
            wait_for_scheduled_detectors(min(args.timeout, 270))

        detector_statuses = validate_detector_results()
        write_status(
            "running",
            "Calculating final DNS alerts",
            82,
            "ML and heuristic scoring are finalized; aggregating detector alerts now.",
            detector_states={label: status.get("state") for label, status in detector_statuses.items()},
        )
        result = final_scoring.calculate_final_alerts(final_scoring.load_final_config())
        result.update(
            {
                "state": "finished",
                "stage": "Final correlation complete",
                "progress_percent": 100,
                "message": "ML and heuristic alerts were aggregated automatically.",
                "pipeline_sequence": [
                    "Autoencoder",
                    "Isolation Forest",
                    "DNS Heuristics",
                    "Final Correlation",
                ],
                "updated_at_utc": utc_now(),
            }
        )
        write_json_atomic(STATUS_PATH, result)
        update_lifecycle(args.job_id, state="completed")
        print(
            f"[{utc_now()}] Final Correlation completed: "
            f"windows={result.get('correlated_windows', 0)} alerts={result.get('final_alerts', 0)}",
            flush=True,
        )
        return 0
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        write_status(
            "failed",
            "Automatic Final Correlation failed",
            100,
            error,
            traceback=traceback.format_exc(),
        )
        update_lifecycle(args.job_id, state="failed", error=error)
        print(f"[{utc_now()}] {error}", file=sys.stderr, flush=True)
        return 1
    finally:
        try:
            if PID_PATH.read_text(encoding="utf-8").strip() == str(os.getpid()):
                PID_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        if lock_handle is not None:
            lock_handle.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--manual", action="store_true")
    mode.add_argument("--scheduled", action="store_true")
    parser.add_argument("--job-id")
    parser.add_argument("--pid", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.manual and not args.pid:
        parser.error("--manual requires at least one --pid NAME=PID")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
