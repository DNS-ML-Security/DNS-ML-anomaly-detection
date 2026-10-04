# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

from __future__ import annotations

import fcntl
import gzip
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator


PREPARATION_HOME = Path(
    os.getenv("DNS_LOG_PREPARATION_HOME", "/data/dns-ml/dns_preparation")
)
STATE_PATH = PREPARATION_HOME / "state.json"
LOCK_PATH = PREPARATION_HOME / "control.lock"
PCAP_ROOT = PREPARATION_HOME / "pcap"
PCAP_INCOMING = PCAP_ROOT / "incoming"
PCAP_JOBS = PCAP_ROOT / "jobs"
LIVE_ROOT = PREPARATION_HOME / "live"
OUTPUT_ROOT = PREPARATION_HOME / "output"

LIVE_SCORING_STATE_PATH = Path(
    os.getenv(
        "DNS_LIVE_SCORING_STATE_PATH",
        "/data/dns-ml/scoring/live_scoring_state.json",
    )
)
LIVE_SCORING_CRON_FILES = (
    Path("/etc/cron.d/dns-ml-scoring"),
    Path("/etc/cron.d/dns-ml-isolation-scoring"),
    Path("/etc/cron.d/dns-ml-heuristics-scoring"),
    Path("/etc/cron.d/dns-ml-final-correlation"),
)

ZEEK_CANDIDATES = (
    Path("/opt/zeek/bin/zeek"),
    Path("/usr/local/bin/zeek"),
    Path("/usr/bin/zeek"),
)

HARD_MAX_CAPTURE_BYTES = 10 * 1024**3
MAX_CAPTURE_BYTES = min(
    HARD_MAX_CAPTURE_BYTES,
    max(1, int(os.getenv("DNS_LEARNING_CAPTURE_MAX_BYTES", str(HARD_MAX_CAPTURE_BYTES)))),
)
MAX_PCAP_UPLOAD_BYTES = int(
    os.getenv("DNS_PCAP_UPLOAD_MAX_BYTES", str(1024**3))
)
ROTATION_INTERVAL = os.getenv("DNS_LEARNING_CAPTURE_ROTATION", "1hr")
POLL_SECONDS = max(1.0, float(os.getenv("DNS_LEARNING_CAPTURE_POLL_SECONDS", "2")))

BUSY_STATES = {
    "pcap_uploading",
    "pcap_queued",
    "pcap_processing",
    "starting",
    "active",
    "paused",
    "finalizing",
}
LIVE_CAPTURE_STATES = {"starting", "active", "paused", "finalizing"}
PCAP_MAGIC_VALUES = {
    b"\xa1\xb2\xc3\xd4",
    b"\xd4\xc3\xb2\xa1",
    b"\xa1\xb2\x3c\x4d",
    b"\x4d\x3c\xb2\xa1",
    b"\x0a\x0d\x0d\x0a",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_utc(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _default_state() -> dict[str, Any]:
    return {
        "state": "inactive",
        "mode": None,
        "job_id": None,
        "message": "Choose PCAP conversion or a managed live learning capture.",
        "progress_percent": 0,
        "interface": None,
        "duration_seconds": 0,
        "active_elapsed_seconds": 0.0,
        "active_started_at_utc": None,
        "created_at_utc": None,
        "updated_at_utc": now_utc(),
        "capture_pid": None,
        "segment_directory": None,
        "session_directory": None,
        "input_path": None,
        "original_filename": None,
        "bytes_used": 0,
        "maximum_bytes": MAX_CAPTURE_BYTES,
        "output_path": None,
        "output_bytes": 0,
        "dns_records": 0,
        "output_sha256": None,
        "stop_reason": None,
        "error": None,
    }


def _ensure_directories() -> None:
    for path in (
        PREPARATION_HOME,
        PCAP_INCOMING,
        PCAP_JOBS,
        LIVE_ROOT,
        OUTPUT_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


@contextmanager
def _control_lock() -> Iterator[None]:
    _ensure_directories()
    with LOCK_PATH.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_state_unlocked() -> dict[str, Any]:
    state = _default_state()
    try:
        value = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            state.update(value)
    except Exception:
        pass
    state["maximum_bytes"] = MAX_CAPTURE_BYTES
    return state


def _write_state_unlocked(state: dict[str, Any]) -> dict[str, Any]:
    state = {**_default_state(), **state}
    state["maximum_bytes"] = MAX_CAPTURE_BYTES
    state["updated_at_utc"] = now_utc()
    temporary = STATE_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(state, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    temporary.replace(STATE_PATH)
    return state


def read_state() -> dict[str, Any]:
    with _control_lock():
        return _load_state_unlocked()


def update_state(**updates: Any) -> dict[str, Any]:
    with _control_lock():
        state = _load_state_unlocked()
        state.update(updates)
        return _write_state_unlocked(state)


def _remove_protected_tree(raw_path: Any, allowed_root: Path) -> None:
    if not raw_path:
        return
    try:
        path = Path(str(raw_path)).resolve()
        path.relative_to(allowed_root.resolve())
        target = path if path.is_dir() else path.parent
        shutil.rmtree(target, ignore_errors=True)
    except Exception:
        pass


def _path_within(path: Path, allowed_root: Path) -> bool:
    try:
        path.resolve().relative_to(allowed_root.resolve())
        return True
    except Exception:
        return False


def _clear_prior_outputs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_ROOT.iterdir():
        try:
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            continue


def reset_state() -> dict[str, Any]:
    with _control_lock():
        state = _load_state_unlocked()
        if str(state.get("state") or "") in BUSY_STATES:
            return {
                **state,
                "error": "The preparation job must finish or be stopped before its state can be reset.",
            }
        if state.get("state") == "failed":
            _remove_protected_tree(state.get("input_path"), PCAP_INCOMING)
            job_id = str(state.get("job_id") or "")
            if re.fullmatch(r"[a-f0-9]{32}", job_id):
                _remove_protected_tree(PCAP_JOBS / job_id, PCAP_JOBS)
            _remove_protected_tree(state.get("session_directory"), LIVE_ROOT)
        return _write_state_unlocked(_default_state())


def preparation_conflict_reason() -> str | None:
    state = read_state()
    current = str(state.get("state") or "inactive")
    if current not in BUSY_STATES:
        return None
    mode = "PCAP conversion" if state.get("mode") == "pcap" else "live learning capture"
    return (
        f"DNS {mode} is {current.replace('_', ' ')}. Live Zeek scoring capture is locked "
        "until preparation completes or the learning capture is stopped and consolidated."
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def live_scoring_conflict_reason() -> str | None:
    lifecycle = _read_json(LIVE_SCORING_STATE_PATH)
    mode = str(lifecycle.get("mode") or "")
    state = str(lifecycle.get("state") or "")
    capture_enabled = bool(lifecycle.get("capture_enabled"))
    scheduled = any(path.exists() for path in LIVE_SCORING_CRON_FILES)
    if mode == "continuous" and (
        capture_enabled or scheduled or state in {"active", "paused", "failed"}
    ):
        return (
            "Managed Live Zeek scoring is active, paused, or reserved. Use DNS Scoring to "
            "pause and then deactivate/delete that runtime before preparing learning DNS logs."
        )
    if scheduled:
        return (
            "One or more live-scoring schedules are still installed. Deactivate the DNS Scoring "
            "runtime before preparing learning DNS logs."
        )
    return None


def _zeek_path() -> Path | None:
    for candidate in ZEEK_CANDIDATES:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _run_json_command(command: list[str], timeout: int = 10) -> tuple[Any, str | None]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        if completed.returncode != 0:
            return None, (completed.stderr or completed.stdout or "command failed").strip()
        return json.loads(completed.stdout or "[]"), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def interface_inventory() -> tuple[list[dict[str, str]], str | None]:
    links, link_error = _run_json_command(["ip", "-j", "link", "show"])
    if link_error:
        return [], f"Unable to inventory server interfaces: {link_error}"
    routes, route_error = _run_json_command(["ip", "-j", "route", "show", "default"])
    if route_error:
        return [], f"Unable to identify the management/default-route interface: {route_error}"
    default_interfaces = {
        str(row.get("dev") or "")
        for row in (routes or [])
        if isinstance(row, dict) and row.get("dev")
    }
    rows: list[dict[str, str]] = []
    for link in links or []:
        if not isinstance(link, dict):
            continue
        name = str(link.get("ifname") or "")
        if not name:
            continue
        reasons: list[str] = []
        if name == "lo" or str(link.get("link_type") or "").lower() == "loopback":
            reasons.append("loopback interface")
        if name in default_interfaces:
            reasons.append("management/default-route interface")
        rows.append(
            {
                "Interface": name,
                "Link state": str(link.get("operstate") or "UNKNOWN").upper(),
                "MAC": str(link.get("address") or "Not available"),
                "Management": "YES" if name in default_interfaces else "NO",
                "Eligibility": "ELIGIBLE" if not reasons else "BLOCKED",
                "Explanation": (
                    "Dedicated learning-capture candidate; an IP address is not required"
                    if not reasons
                    else "; ".join(reasons)
                ),
            }
        )
    return rows, None


def _validate_interface(interface: str) -> str | None:
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", str(interface or "")):
        return "Select a valid capture interface."
    rows, error = interface_inventory()
    if error:
        return error
    selected = next((row for row in rows if row["Interface"] == interface), None)
    if selected is None:
        return f"The selected interface {interface!r} is no longer present."
    if selected["Eligibility"] != "ELIGIBLE":
        return f"Interface {interface!r} is blocked: {selected['Explanation']}."
    return None


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(value).name).strip("._") or "capture.pcap"


def stage_pcap(
    uploaded: Any,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, Any]:
    if uploaded is None:
        return {"error": "Select one PCAP or PCAPNG file."}
    original_name = _safe_name(str(getattr(uploaded, "name", "capture.pcap")))
    if not original_name.lower().endswith((".pcap", ".pcapng")):
        return {"error": "Only .pcap and .pcapng files are accepted."}
    scoring_error = live_scoring_conflict_reason()
    if scoring_error:
        return {"error": scoring_error}

    job_id = uuid.uuid4().hex
    incoming_dir = PCAP_INCOMING / job_id
    incoming_dir.mkdir(parents=True, exist_ok=False)
    destination = incoming_dir / original_name
    temporary = destination.with_suffix(destination.suffix + ".uploading")

    with _control_lock():
        current = _load_state_unlocked()
        if str(current.get("state") or "") in BUSY_STATES:
            shutil.rmtree(incoming_dir, ignore_errors=True)
            return {"error": "Another DNS preparation job is active or suspended."}
        _write_state_unlocked(
            {
                **_default_state(),
                "state": "pcap_uploading",
                "mode": "pcap",
                "job_id": job_id,
                "message": "Saving the uploaded packet capture securely.",
                "progress_percent": 5,
                "created_at_utc": now_utc(),
                "input_path": str(destination),
                "original_filename": original_name,
            }
        )

    bytes_written = 0
    try:
        try:
            uploaded.seek(0)
        except Exception:
            pass
        with temporary.open("wb") as output:
            while True:
                chunk = uploaded.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_PCAP_UPLOAD_BYTES:
                    raise RuntimeError(
                        f"The PCAP exceeds the {MAX_PCAP_UPLOAD_BYTES / 1024**3:.0f} GiB GUI upload limit."
                    )
                output.write(chunk)
                if progress_callback:
                    percent = min(20, 5 + int(bytes_written / MAX_PCAP_UPLOAD_BYTES * 15))
                    progress_callback(percent, f"Uploaded {bytes_written / 1024**2:.1f} MiB")
        if bytes_written == 0:
            raise RuntimeError("The uploaded packet capture is empty.")
        with temporary.open("rb") as captured:
            if captured.read(4) not in PCAP_MAGIC_VALUES:
                raise RuntimeError("The uploaded file does not contain a recognized PCAP or PCAPNG header.")
        os.chmod(temporary, 0o600)
        temporary.replace(destination)
        with _control_lock():
            state = _load_state_unlocked()
            if state.get("job_id") != job_id or state.get("state") != "pcap_uploading":
                raise RuntimeError("The preparation state changed while the PCAP was uploading.")
            _clear_prior_outputs()
            state.update(
                state="pcap_queued",
                message="PCAP saved. The persistent preparation service will generate dns.log.",
                progress_percent=25,
                bytes_used=bytes_written,
                error=None,
            )
            state = _write_state_unlocked(state)
        return {**state, "error": None}
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        shutil.rmtree(incoming_dir, ignore_errors=True)
        state = update_state(
            state="failed",
            message="PCAP upload or staging failed.",
            progress_percent=0,
            error=f"{type(exc).__name__}: {exc}",
        )
        return {**state, "error": state["error"]}


def start_live_capture(interface: str, duration_seconds: int) -> dict[str, Any]:
    interface_error = _validate_interface(interface)
    if interface_error:
        return {"error": interface_error}
    scoring_error = live_scoring_conflict_reason()
    if scoring_error:
        return {"error": scoring_error}
    duration = int(duration_seconds or 0)
    if duration < 300 or duration > 30 * 24 * 3600:
        return {"error": "The learning period must be between 5 minutes and 30 days."}
    if _zeek_path() is None:
        return {"error": "The Zeek executable was not found."}

    job_id = uuid.uuid4().hex
    session_directory = LIVE_ROOT / job_id
    (session_directory / "segments").mkdir(parents=True, exist_ok=False)
    with _control_lock():
        current = _load_state_unlocked()
        if str(current.get("state") or "") in BUSY_STATES:
            shutil.rmtree(session_directory, ignore_errors=True)
            return {"error": "Another DNS preparation job is already active or suspended."}
        _clear_prior_outputs()
        return _write_state_unlocked(
            {
                **_default_state(),
                "state": "starting",
                "mode": "live",
                "job_id": job_id,
                "message": "The persistent service is starting the learning capture.",
                "progress_percent": 1,
                "interface": interface,
                "duration_seconds": duration,
                "active_elapsed_seconds": 0.0,
                "active_started_at_utc": now_utc(),
                "created_at_utc": now_utc(),
                "session_directory": str(session_directory),
                "error": None,
            }
        )


def _checkpoint_elapsed(state: dict[str, Any]) -> dict[str, Any]:
    started = _parse_utc(state.get("active_started_at_utc"))
    if started is not None:
        state["active_elapsed_seconds"] = float(state.get("active_elapsed_seconds", 0) or 0) + max(
            0.0,
            (datetime.now(timezone.utc) - started).total_seconds(),
        )
    state["active_started_at_utc"] = None
    return state


def suspend_live_capture() -> dict[str, Any]:
    with _control_lock():
        state = _load_state_unlocked()
        if state.get("mode") != "live" or state.get("state") not in {"starting", "active"}:
            return {**state, "error": "No active learning capture can be suspended."}
        state = _checkpoint_elapsed(state)
        state.update(
            state="paused",
            message="Learning capture is suspended. Existing segments are preserved.",
            capture_pid=None,
            error=None,
        )
        return _write_state_unlocked(state)


def resume_live_capture() -> dict[str, Any]:
    scoring_error = live_scoring_conflict_reason()
    if scoring_error:
        return {"error": scoring_error}
    with _control_lock():
        state = _load_state_unlocked()
        if state.get("mode") != "live" or state.get("state") != "paused":
            return {**state, "error": "No suspended learning capture can be resumed."}
        interface_error = _validate_interface(str(state.get("interface") or ""))
        if interface_error:
            return {**state, "error": interface_error}
        state.update(
            state="starting",
            message="Resuming learning capture in a new protected segment.",
            active_started_at_utc=now_utc(),
            capture_pid=None,
            error=None,
        )
        return _write_state_unlocked(state)


def stop_and_consolidate() -> dict[str, Any]:
    with _control_lock():
        state = _load_state_unlocked()
        if state.get("mode") != "live" or state.get("state") not in {
            "starting",
            "active",
            "paused",
        }:
            return {**state, "error": "No active or suspended learning capture can be stopped."}
        if state.get("state") in {"starting", "active"}:
            state = _checkpoint_elapsed(state)
        state.update(
            state="finalizing",
            message="Stopping capture and consolidating all hourly DNS segments.",
            stop_reason="user_stop",
            capture_pid=None,
            error=None,
        )
        return _write_state_unlocked(state)


def effective_elapsed_seconds(state: dict[str, Any] | None = None) -> float:
    state = state or read_state()
    elapsed = float(state.get("active_elapsed_seconds", 0) or 0)
    if state.get("state") in {"starting", "active"}:
        started = _parse_utc(state.get("active_started_at_utc"))
        if started is not None:
            elapsed += max(0.0, (datetime.now(timezone.utc) - started).total_seconds())
    return elapsed


def output_path_from_state(state: dict[str, Any] | None = None) -> Path | None:
    state = state or read_state()
    raw = str(state.get("output_path") or "")
    if not raw:
        return None
    path = Path(raw)
    try:
        path.resolve().relative_to(OUTPUT_ROOT.resolve())
    except Exception:
        return None
    return path if path.is_file() else None


def _path_size(root: Path) -> int:
    total = 0
    try:
        for path in root.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _process_running(pid: Any, expected: str = "zeek") -> bool:
    try:
        value = int(pid)
        os.kill(value, 0)
        command = Path(f"/proc/{value}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
        return expected in command
    except Exception:
        return False


def _terminate_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(timeout=5)
        except Exception:
            pass


def _dns_sources(root: Path) -> list[Path]:
    sources: list[Path] = []
    if not root.exists():
        return sources
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        is_dns_log = name == "dns.log" or (
            (name.startswith("dns.") or name.startswith("dns_"))
            and (name.endswith(".log") or name.endswith(".log.gz") or name.endswith(".gz"))
        )
        if is_dns_log:
            sources.append(path)
    return sorted(sources, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def _append_source(source: Path, output: Any) -> int:
    opener = gzip.open if source.name.lower().endswith(".gz") else open
    lines = 0
    last_byte = b""
    with opener(source, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            lines += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if last_byte and last_byte != b"\n":
        output.write(b"\n")
        lines += 1
    return lines


def _complete_output(
    *,
    state: dict[str, Any],
    source_root: Path,
    allowed_source_root: Path,
    output_name: str,
    message_prefix: str,
) -> dict[str, Any]:
    if not _path_within(source_root, allowed_source_root):
        raise RuntimeError(f"Refusing to consolidate an unprotected source path: {source_root}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / output_name
    if not _path_within(output_path, OUTPUT_ROOT):
        raise RuntimeError(f"Refusing to write an unprotected output path: {output_path}")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    sources = _dns_sources(source_root)
    records = 0
    with temporary.open("wb") as output:
        for source in sources:
            records += _append_source(source, output)
    os.chmod(temporary, 0o600)
    temporary.replace(output_path)
    output_bytes = output_path.stat().st_size
    digest = _sha256(output_path)
    shutil.rmtree(source_root, ignore_errors=True)
    return update_state(
        state="completed",
        message=f"{message_prefix} {records:,} DNS records are ready.",
        progress_percent=100,
        active_started_at_utc=None,
        capture_pid=None,
        segment_directory=None,
        session_directory=None,
        input_path=None,
        bytes_used=output_bytes,
        output_path=str(output_path),
        output_bytes=output_bytes,
        dns_records=records,
        output_sha256=digest,
        error=None,
    )


def _mark_failed(message: str, error: str) -> dict[str, Any]:
    return update_state(
        state="failed",
        message=message,
        progress_percent=0,
        active_started_at_utc=None,
        capture_pid=None,
        error=error,
    )


def _run_pcap_job(state: dict[str, Any]) -> None:
    zeek = _zeek_path()
    if zeek is None:
        _mark_failed("PCAP conversion failed.", "The Zeek executable was not found.")
        return
    input_path = Path(str(state.get("input_path") or ""))
    if not _path_within(input_path, PCAP_INCOMING) or not input_path.is_file():
        _mark_failed("PCAP conversion failed.", f"Staged PCAP is missing: {input_path}")
        return
    job_id = str(state.get("job_id") or uuid.uuid4().hex)
    job_root = PCAP_JOBS / job_id
    work = job_root / "work"
    shutil.rmtree(job_root, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    log_path = job_root / "zeek-output.txt"
    update_state(
        state="pcap_processing",
        message="Zeek is reading the PCAP and generating JSON DNS records.",
        progress_percent=35,
        error=None,
    )
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [str(zeek), "-C", "-r", str(input_path), "LogAscii::use_json=T"],
            cwd=str(work),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while process.poll() is None:
            bytes_used = _path_size(work)
            update_state(
                capture_pid=process.pid,
                bytes_used=bytes_used,
                progress_percent=60,
                message=f"Zeek is processing the PCAP; generated {bytes_used / 1024**2:.1f} MiB so far.",
            )
            if bytes_used >= MAX_CAPTURE_BYTES:
                _terminate_process(process)
                _mark_failed(
                    "PCAP conversion stopped at the safety limit.",
                    "Generated Zeek logs reached the 10 GiB maximum.",
                )
                return
            time.sleep(POLL_SECONDS)
    if process.returncode != 0:
        tail = ""
        try:
            tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
        except Exception:
            pass
        _mark_failed(
            "PCAP conversion failed.",
            f"Zeek exited with status {process.returncode}. {tail}".strip(),
        )
        return
    update_state(
        state="finalizing",
        message="PCAP analysis completed; preparing the downloadable dns.log.",
        progress_percent=85,
        capture_pid=None,
    )
    try:
        _complete_output(
            state=read_state(),
            source_root=work,
            allowed_source_root=PCAP_JOBS,
            output_name=f"pcap_{job_id}_dns.log",
            message_prefix="PCAP conversion completed.",
        )
    except Exception as exc:
        _mark_failed("PCAP DNS consolidation failed.", f"{type(exc).__name__}: {exc}")
        return
    input_path.unlink(missing_ok=True)
    shutil.rmtree(input_path.parent, ignore_errors=True)
    shutil.rmtree(job_root, ignore_errors=True)


def _spawn_live_zeek(state: dict[str, Any]) -> subprocess.Popen[Any] | None:
    zeek = _zeek_path()
    if zeek is None:
        _mark_failed("Learning capture failed.", "The Zeek executable was not found.")
        return None
    interface = str(state.get("interface") or "")
    error = _validate_interface(interface)
    if error:
        _mark_failed("Learning capture failed.", error)
        return None
    session_value = str(state.get("session_directory") or "")
    if not session_value:
        _mark_failed("Learning capture failed.", "The protected session directory is missing.")
        return None
    session_root = Path(session_value)
    if not _path_within(session_root, LIVE_ROOT):
        _mark_failed("Learning capture failed.", "The protected session directory is invalid.")
        return None
    segment = session_root / "segments" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    segment.mkdir(parents=True, exist_ok=False)
    log_handle = (segment / "zeek-output.txt").open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [
                str(zeek),
                "-C",
                "-i",
                interface,
                "LogAscii::use_json=T",
                f"Log::default_rotation_interval={ROTATION_INTERVAL}",
            ],
            cwd=str(segment),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_handle.close()
    update_state(
        state="active",
        message="Learning DNS capture is active; hourly Zeek segments are protected and monitored.",
        capture_pid=process.pid,
        segment_directory=str(segment),
        active_started_at_utc=now_utc(),
        error=None,
    )
    return process


def _finalize_live_capture(state: dict[str, Any]) -> None:
    session_value = str(state.get("session_directory") or "")
    session_root = Path(session_value) if session_value else Path("/")
    if not _path_within(session_root, LIVE_ROOT) or not session_root.exists():
        _mark_failed("DNS consolidation failed.", f"Session directory is missing: {session_root}")
        return
    job_id = str(state.get("job_id") or uuid.uuid4().hex)
    update_state(
        state="finalizing",
        message="Consolidating all current and hourly archived DNS logs.",
        progress_percent=95,
        capture_pid=None,
        active_started_at_utc=None,
    )
    try:
        _complete_output(
            state=read_state(),
            source_root=session_root,
            allowed_source_root=LIVE_ROOT,
            output_name=f"live_{job_id}_dns.log",
            message_prefix="Live learning capture consolidated.",
        )
    except Exception as exc:
        _mark_failed("DNS consolidation failed.", f"{type(exc).__name__}: {exc}")


def _reconcile_service_startup() -> None:
    with _control_lock():
        state = _load_state_unlocked()
        current = str(state.get("state") or "")
        if current == "pcap_uploading":
            state.update(
                state="failed",
                message="The server restarted before the PCAP upload completed.",
                capture_pid=None,
                error="Upload was interrupted. Reset the preparation state and upload the PCAP again.",
            )
        elif current == "pcap_processing":
            state.update(
                state="pcap_queued",
                message="Restart detected; PCAP conversion will restart safely.",
                capture_pid=None,
                error=None,
            )
        elif current in {"starting", "active"} and state.get("mode") == "live":
            state.update(
                state="starting",
                message="Service restart detected; live learning capture will resume in a new segment.",
                active_started_at_utc=now_utc(),
                capture_pid=None,
                segment_directory=None,
                error=None,
            )
        elif current == "finalizing" and state.get("mode") == "pcap":
            state.update(
                state="pcap_queued",
                message="Restart detected during PCAP finalization; conversion will restart safely.",
                capture_pid=None,
                error=None,
            )
        _write_state_unlocked(state)


def run_supervisor() -> None:
    _ensure_directories()
    _reconcile_service_startup()
    stopping = False
    live_process: subprocess.Popen[Any] | None = None

    def _request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    while not stopping:
        state = read_state()
        current = str(state.get("state") or "inactive")
        mode = str(state.get("mode") or "")

        if mode == "pcap" and current == "pcap_queued":
            _run_pcap_job(state)
            continue

        if mode == "live" and current in {"starting", "active"}:
            scoring_error = live_scoring_conflict_reason()
            if scoring_error:
                _terminate_process(live_process)
                live_process = None
                _mark_failed("Learning capture stopped to prevent a scoring conflict.", scoring_error)
                continue
            if live_process is None or live_process.poll() is not None:
                if current == "active" and live_process is not None and live_process.returncode not in (None, 0):
                    _mark_failed(
                        "Learning capture failed.",
                        f"Zeek exited unexpectedly with status {live_process.returncode}.",
                    )
                    live_process = None
                    continue
                live_process = _spawn_live_zeek(state)
                if live_process is None:
                    continue
                state = read_state()

            with _control_lock():
                latest = _load_state_unlocked()
                if latest.get("state") in {"starting", "active"}:
                    latest = _checkpoint_elapsed(latest)
                    latest["active_started_at_utc"] = now_utc()
                    session_root = Path(str(latest.get("session_directory") or ""))
                    bytes_used = _path_size(session_root)
                    latest["bytes_used"] = bytes_used
                    duration = max(1, int(latest.get("duration_seconds", 0) or 0))
                    elapsed = float(latest.get("active_elapsed_seconds", 0) or 0)
                    time_percent = int(min(100, elapsed / duration * 100))
                    size_percent = int(min(100, bytes_used / MAX_CAPTURE_BYTES * 100))
                    latest["progress_percent"] = max(time_percent, size_percent)
                    latest["capture_pid"] = live_process.pid
                    latest["message"] = (
                        f"Learning capture active: {elapsed / 3600:.2f} h captured; "
                        f"{bytes_used / 1024**3:.2f} GiB stored."
                    )
                    reached_duration = elapsed >= duration
                    reached_size = bytes_used >= MAX_CAPTURE_BYTES
                    if reached_duration or reached_size:
                        latest["state"] = "finalizing"
                        latest["active_started_at_utc"] = None
                        latest["capture_pid"] = None
                        latest["stop_reason"] = "duration_complete" if reached_duration else "size_limit"
                        latest["message"] = (
                            "Configured learning period completed; consolidating DNS logs."
                            if reached_duration
                            else "The 10 GiB safety limit was reached; consolidating DNS logs."
                        )
                    _write_state_unlocked(latest)
            state = read_state()
            if state.get("state") == "finalizing":
                _terminate_process(live_process)
                live_process = None
                _finalize_live_capture(state)
                continue

        elif live_process is not None:
            _terminate_process(live_process)
            live_process = None
            update_state(capture_pid=None, segment_directory=None)

        if mode == "live" and current == "finalizing":
            _finalize_live_capture(state)
            continue

        time.sleep(POLL_SECONDS)

    if live_process is not None:
        _terminate_process(live_process)
    with _control_lock():
        state = _load_state_unlocked()
        if state.get("mode") == "live" and state.get("state") in {"starting", "active"}:
            state = _checkpoint_elapsed(state)
            state.update(
                state="starting",
                message="Capture service stopped; it will resume automatically after service or server restart.",
                capture_pid=None,
                segment_directory=None,
            )
            _write_state_unlocked(state)


if __name__ == "__main__":
    run_supervisor()
