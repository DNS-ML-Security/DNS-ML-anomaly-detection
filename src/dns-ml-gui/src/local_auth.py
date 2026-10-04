# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import html
import os
import re
import secrets
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import streamlit as st
except ModuleNotFoundError:  # CLI bootstrap can run with a minimal system Python.
    st = None  # type: ignore[assignment]

try:
    from streamlit_cookies_controller import CookieController
except ModuleNotFoundError:  # CLI bootstrap does not require the browser component.
    CookieController = None  # type: ignore[assignment,misc]


AUTH_DB_PATH = Path(os.getenv("DNS_GUI_AUTH_DB", "/data/dns-ml/gui_auth/users.db"))
PBKDF2_ITERATIONS = int(os.getenv("DNS_GUI_PASSWORD_ITERATIONS", "600000"))
MAX_FAILED_ATTEMPTS = int(os.getenv("DNS_GUI_MAX_FAILED_ATTEMPTS", "5"))
LOCKOUT_SECONDS = int(os.getenv("DNS_GUI_LOCKOUT_SECONDS", "900"))
SESSION_IDLE_SECONDS = int(os.getenv("DNS_GUI_SESSION_IDLE_SECONDS", "28800"))
AUTH_COOKIE_NAME = os.getenv("DNS_GUI_AUTH_COOKIE_NAME", "dns_ml_auth_session")
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,63}$")
PRODUCT_VERSION = os.getenv("DNS_PRODUCT_VERSION", "v1.0.0-rc.1")
BUILD_CLASSIFICATION = os.getenv(
    "DNS_BUILD_CLASSIFICATION", "Official release candidate"
).replace("_", " ")
BUILD_FINGERPRINT = os.getenv(
    "DNS_BUILD_FINGERPRINT", "Not recorded for this installation"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    AUTH_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(AUTH_DB_PATH.parent, 0o700)
    except OSError:
        pass
    connection = sqlite3.connect(AUTH_DB_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=15000")
    return connection


def init_auth_db() -> None:
    connection = _connect()
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'user', 'soc')),
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                password_iterations INTEGER NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                must_change_password INTEGER NOT NULL DEFAULT 0,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until_epoch INTEGER,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                last_login_at_utc TEXT
            );
            CREATE TABLE IF NOT EXISTS auth_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                actor_username TEXT,
                action TEXT NOT NULL,
                target_username TEXT,
                outcome TEXT NOT NULL,
                detail TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_auth_audit_timestamp
                ON auth_audit(timestamp_utc);
            CREATE TABLE IF NOT EXISTS auth_sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at_epoch INTEGER NOT NULL,
                last_seen_epoch INTEGER NOT NULL,
                expires_at_epoch INTEGER NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_auth_sessions_user
                ON auth_sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_auth_sessions_expiry
                ON auth_sessions(expires_at_epoch);
            """
        )
        connection.commit()
    finally:
        connection.close()
    try:
        os.chmod(AUTH_DB_PATH, 0o600)
    except OSError:
        pass


def _audit(
    connection: sqlite3.Connection,
    action: str,
    outcome: str,
    *,
    actor: str | None = None,
    target: str | None = None,
    detail: str | None = None,
) -> None:
    connection.execute(
        """
        INSERT INTO auth_audit
        (timestamp_utc, actor_username, action, target_username, outcome, detail)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (_utc_now(), actor, action, target, outcome, detail),
    )


def password_policy_error(password: str) -> str | None:
    value = str(password or "")
    if not value:
        return "Password cannot be empty."
    return None


def username_policy_error(username: str) -> str | None:
    if not USERNAME_PATTERN.fullmatch(str(username or "").strip()):
        return "Username must be 3–64 characters and may use letters, numbers, dot, underscore, and hyphen."
    return None


def _password_record(password: str, salt_hex: str | None = None, iterations: int | None = None) -> tuple[str, str, int]:
    salt = bytes.fromhex(salt_hex) if salt_hex else secrets.token_bytes(32)
    rounds = int(iterations or PBKDF2_ITERATIONS)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return digest.hex(), salt.hex(), rounds


def _verify_password(password: str, row: sqlite3.Row) -> bool:
    candidate, _, _ = _password_record(
        password,
        str(row["password_salt"]),
        int(row["password_iterations"]),
    )
    return hmac.compare_digest(candidate, str(row["password_hash"]))


def user_count() -> int:
    init_auth_db()
    connection = _connect()
    try:
        return int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])
    finally:
        connection.close()


def bootstrap_admin(username: str, password: str, display_name: str = "DNS Administrator") -> dict[str, Any]:
    init_auth_db()
    username = str(username or "").strip()
    display_name = str(display_name or username).strip() or username
    error = username_policy_error(username) or password_policy_error(password)
    if error:
        return {"ok": False, "error": error}
    connection = _connect()
    try:
        connection.execute("BEGIN IMMEDIATE")
        if int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]) > 0:
            connection.rollback()
            return {"ok": True, "created": False, "message": "An account already exists; bootstrap was skipped."}
        digest, salt, rounds = _password_record(password)
        now = _utc_now()
        connection.execute(
            """
            INSERT INTO users
            (username, display_name, role, password_hash, password_salt,
             password_iterations, active, must_change_password, created_at_utc, updated_at_utc)
            VALUES (?, ?, 'admin', ?, ?, ?, 1, 0, ?, ?)
            """,
            (username, display_name, digest, salt, rounds, now, now),
        )
        _audit(connection, "bootstrap_admin", "success", actor=username, target=username)
        connection.commit()
        return {"ok": True, "created": True, "message": f"Administrator {username} created."}
    except sqlite3.IntegrityError:
        connection.rollback()
        return {"ok": False, "error": "The requested administrator username already exists."}
    finally:
        connection.close()


def _public_user(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(row["id"]),
        "username": str(row["username"]),
        "display_name": str(row["display_name"]),
        "role": str(row["role"]),
        "active": bool(row["active"]),
        "must_change_password": bool(row["must_change_password"]),
        "last_login_at_utc": row["last_login_at_utc"],
    }


def role_label(role: str) -> str:
    return {
        "admin": "Administrator",
        "user": "Standard User",
        "soc": "SOC Analyst",
    }.get(str(role or ""), "Unknown Role")


def allowed_pages_for_user(user: dict[str, Any]) -> list[str] | None:
    """Return None for full application access or an explicit page allowlist."""
    if str(user.get("role") or "") == "soc":
        return ["Alerts", "Alert Details", "Source IP Investigation"]
    return None


def authenticate(username: str, password: str) -> tuple[dict[str, Any] | None, str | None]:
    init_auth_db()
    normalized = str(username or "").strip()
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
            (normalized,),
        ).fetchone()
        now_epoch = int(time.time())
        if row is None:
            _password_record(str(password or ""), secrets.token_hex(32), PBKDF2_ITERATIONS)
            _audit(connection, "login", "failed", actor=normalized, target=normalized, detail="unknown user")
            connection.commit()
            return None, "Invalid username or password."
        if not bool(row["active"]):
            _audit(connection, "login", "failed", actor=normalized, target=str(row["username"]), detail="disabled account")
            connection.commit()
            return None, "Invalid username or password."
        locked_until = int(row["locked_until_epoch"] or 0)
        if locked_until > now_epoch:
            remaining = max(1, locked_until - now_epoch)
            _audit(connection, "login", "blocked", actor=normalized, target=str(row["username"]), detail="account lockout")
            connection.commit()
            return None, f"Account temporarily locked. Try again in approximately {remaining // 60 + 1} minute(s)."
        if not _verify_password(str(password or ""), row):
            attempts = int(row["failed_attempts"] or 0) + 1
            new_lock = now_epoch + LOCKOUT_SECONDS if attempts >= MAX_FAILED_ATTEMPTS else None
            connection.execute(
                "UPDATE users SET failed_attempts = ?, locked_until_epoch = ?, updated_at_utc = ? WHERE id = ?",
                (0 if new_lock else attempts, new_lock, _utc_now(), int(row["id"])),
            )
            _audit(connection, "login", "failed", actor=normalized, target=str(row["username"]), detail=f"failed attempt {attempts}")
            connection.commit()
            return None, "Invalid username or password."
        now = _utc_now()
        connection.execute(
            "UPDATE users SET failed_attempts = 0, locked_until_epoch = NULL, last_login_at_utc = ?, updated_at_utc = ? WHERE id = ?",
            (now, now, int(row["id"])),
        )
        _audit(connection, "login", "success", actor=str(row["username"]), target=str(row["username"]))
        connection.commit()
        refreshed = connection.execute("SELECT * FROM users WHERE id = ?", (int(row["id"]),)).fetchone()
        return _public_user(refreshed), None
    finally:
        connection.close()


def _load_active_user(user_id: int) -> dict[str, Any] | None:
    connection = _connect()
    try:
        row = connection.execute("SELECT * FROM users WHERE id = ? AND active = 1", (int(user_id),)).fetchone()
        return _public_user(row) if row else None
    finally:
        connection.close()


def _token_hash(token: str) -> str:
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


def _create_auth_session(user_id: int) -> str:
    token = secrets.token_urlsafe(48)
    now_epoch = int(time.time())
    connection = _connect()
    try:
        connection.execute("DELETE FROM auth_sessions WHERE expires_at_epoch <= ?", (now_epoch,))
        connection.execute(
            """
            INSERT INTO auth_sessions
            (token_hash, user_id, created_at_epoch, last_seen_epoch, expires_at_epoch)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                _token_hash(token),
                int(user_id),
                now_epoch,
                now_epoch,
                now_epoch + SESSION_IDLE_SECONDS,
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return token


def _restore_auth_session(token: str) -> dict[str, Any] | None:
    value = str(token or "").strip()
    if not value:
        return None
    now_epoch = int(time.time())
    connection = _connect()
    try:
        connection.execute("DELETE FROM auth_sessions WHERE expires_at_epoch <= ?", (now_epoch,))
        row = connection.execute(
            """
            SELECT users.*
            FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            WHERE auth_sessions.token_hash = ?
              AND auth_sessions.expires_at_epoch > ?
              AND users.active = 1
            """,
            (_token_hash(value), now_epoch),
        ).fetchone()
        if row is None:
            connection.commit()
            return None
        connection.execute(
            """
            UPDATE auth_sessions
            SET last_seen_epoch = ?, expires_at_epoch = ?
            WHERE token_hash = ?
            """,
            (now_epoch, now_epoch + SESSION_IDLE_SECONDS, _token_hash(value)),
        )
        connection.commit()
        return _public_user(row)
    finally:
        connection.close()


def _revoke_auth_session(token: str | None) -> None:
    value = str(token or "").strip()
    if not value:
        return
    connection = _connect()
    try:
        connection.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (_token_hash(value),))
        connection.commit()
    finally:
        connection.close()


def _cookie_controller():
    if CookieController is None or st is None:
        return None
    # Recreate the controller on every Streamlit run. Its component updates the
    # stable session-state key asynchronously after reading the browser. Caching
    # the Python object kept its initial empty cookie dictionary and prevented a
    # refreshed browser session from being restored.
    return CookieController(key="dns_auth_browser_cookies")


def _set_auth_cookie(controller, token: str) -> None:
    if controller is None or not token:
        return
    controller.set(
        AUTH_COOKIE_NAME,
        token,
        path="/",
        expires=datetime.now(timezone.utc) + timedelta(seconds=SESSION_IDLE_SECONDS),
        max_age=SESSION_IDLE_SECONDS,
        same_site="strict",
    )


def _login_logo_data_uri() -> str:
    logo_path = Path(__file__).resolve().parent.parent / "assets" / "dns_console_logo.png"
    try:
        payload = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{payload}"
    except OSError:
        return ""


def _remove_auth_cookie(controller) -> None:
    if controller is None:
        return
    try:
        controller.remove(AUTH_COOKIE_NAME)
    except Exception:
        pass


def _clear_session(controller=None) -> None:
    token = st.session_state.pop("dns_auth_session_token", None)
    _revoke_auth_session(token)
    _remove_auth_cookie(controller)
    for key in (
        "dns_auth_user",
        "dns_auth_last_activity",
        "dns_auth_cookie_refreshed",
        "dns_auth_manage_open",
    ):
        st.session_state.pop(key, None)


def _change_password(user_id: int, current_password: str, new_password: str, *, actor: str) -> str | None:
    policy = password_policy_error(new_password)
    if policy:
        return policy
    connection = _connect()
    try:
        row = connection.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
        if row is None or not _verify_password(current_password, row):
            _audit(connection, "change_password", "failed", actor=actor, target=actor, detail="current password mismatch")
            connection.commit()
            return "Current password is incorrect."
        if hmac.compare_digest(current_password, new_password):
            return "New password must be different from the current password."
        digest, salt, rounds = _password_record(new_password)
        connection.execute(
            """
            UPDATE users SET password_hash = ?, password_salt = ?, password_iterations = ?,
                must_change_password = 0, failed_attempts = 0, locked_until_epoch = NULL,
                updated_at_utc = ? WHERE id = ?
            """,
            (digest, salt, rounds, _utc_now(), int(user_id)),
        )
        _audit(connection, "change_password", "success", actor=actor, target=actor)
        connection.commit()
        return None
    finally:
        connection.close()


def render_login_gate(app_name: str = "DNS ML GUI") -> dict[str, Any] | None:
    init_auth_db()
    controller = _cookie_controller()
    session_user = st.session_state.get("dns_auth_user")
    if not session_user and controller is not None:
        try:
            browser_token = controller.get(AUTH_COOKIE_NAME)
        except Exception:
            browser_token = None
        restored = _restore_auth_session(browser_token) if browser_token else None
        if restored:
            st.session_state["dns_auth_user"] = restored
            st.session_state["dns_auth_session_token"] = browser_token
            st.session_state["dns_auth_last_activity"] = time.time()
            st.session_state["dns_auth_manage_open"] = False
            session_user = restored
        elif browser_token:
            _remove_auth_cookie(controller)
    last_activity = float(st.session_state.get("dns_auth_last_activity", 0) or 0)
    if session_user and time.time() - last_activity <= SESSION_IDLE_SECONDS:
        refreshed = _load_active_user(int(session_user.get("id", 0)))
        if refreshed:
            st.session_state["dns_auth_user"] = refreshed
            st.session_state["dns_auth_last_activity"] = time.time()
            session_token = str(st.session_state.get("dns_auth_session_token") or "")
            cookie_refreshed = float(st.session_state.get("dns_auth_cookie_refreshed", 0) or 0)
            if session_token and time.time() - cookie_refreshed >= 300:
                try:
                    _set_auth_cookie(controller, session_token)
                    st.session_state["dns_auth_cookie_refreshed"] = time.time()
                except Exception:
                    pass
            if refreshed.get("must_change_password"):
                st.title("Password change required")
                st.warning("An administrator assigned a temporary password. Set a new password before continuing.")
                with st.form("dns_forced_password_change", clear_on_submit=True):
                    current = st.text_input("Temporary password", type="password")
                    new = st.text_input("New password", type="password")
                    confirm = st.text_input("Confirm new password", type="password")
                    submitted = st.form_submit_button("Update password", use_container_width=True)
                if submitted:
                    if new != confirm:
                        st.error("The new passwords do not match.")
                    else:
                        error = _change_password(int(refreshed["id"]), current, new, actor=str(refreshed["username"]))
                        if error:
                            st.error(error)
                        else:
                            st.success("Password updated. Continue to the DNS Console.")
                            st.session_state["dns_auth_user"] = _load_active_user(int(refreshed["id"]))
                            st.rerun()
                return None
            return refreshed
    if session_user:
        _clear_session(controller)

    logo_uri = _login_logo_data_uri()
    logo_html = (
        f'<img class="dns-login-logo" src="{logo_uri}" alt="DNS Console logo">'
        if logo_uri
        else '<div class="dns-login-logo-fallback">DNS</div>'
    )
    st.markdown(
        """
        <style>
        footer, [data-testid="stFooter"], [data-testid="stAppFooter"] {display:none!important;visibility:hidden!important;}
        .dns-login-brand {text-align:center;margin:4vh auto 1rem;}
        .dns-login-logo {width:112px;height:112px;object-fit:contain;filter:drop-shadow(0 10px 24px rgba(34,211,238,.24));}
        .dns-login-logo-fallback {width:92px;height:92px;margin:auto;display:grid;place-items:center;border:1px solid #38bdf8;border-radius:24px;color:#67e8f9;font-size:2rem;font-weight:900;background:#0b1f34;}
        .dns-login-title {font-size:clamp(1.55rem,2.2vw,2.05rem);font-weight:850;color:#F5F9FC;text-align:center;margin-top:.65rem;letter-spacing:.015em;}
        .dns-login-subtitle {color:#AFC8DC;text-align:center;margin:.35rem 0 .2rem;font-size:clamp(.78rem,.95vw,.96rem);}
        .dns-login-security {color:#6f91aa;text-align:center;font-size:.73rem;letter-spacing:.045em;text-transform:uppercase;}
        div[data-testid="stForm"] {padding:1.1rem 1.15rem 1.2rem;border:1px solid rgba(56,189,248,.34);border-radius:18px;background:linear-gradient(145deg,rgba(16,40,61,.97),rgba(8,20,34,.98));box-shadow:0 20px 48px rgba(0,0,0,.34),0 0 30px rgba(14,165,233,.07);}
        div[data-testid="stForm"] label p {color:#dceaf5;font-weight:720;}
        div[data-testid="stForm"] input {border-radius:10px;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="dns-login-brand">'
        f'{logo_html}'
        '<div class="dns-login-title">DNS Console</div>'
        f'<div class="dns-login-subtitle">{html.escape(str(app_name))}</div>'
        '<div class="dns-login-security">Local authenticated SOC access</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if user_count() == 0:
        st.error("No local administrator exists. Run the v6.11 installer interactively or provide DNS_GUI_ADMIN_USERNAME and DNS_GUI_ADMIN_PASSWORD.")
        return None
    left, center, right = st.columns([1.25, 1, 1.25])
    with center:
        with st.form("dns_local_login", clear_on_submit=False):
            username = st.text_input("Username", autocomplete="username")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            submitted = st.form_submit_button("Sign in", use_container_width=True, type="primary")
        if submitted:
            user, error = authenticate(username, password)
            if error:
                st.error(error)
            else:
                session_token = _create_auth_session(int(user["id"]))
                if controller is not None:
                    _set_auth_cookie(controller, session_token)
                st.session_state["dns_auth_user"] = user
                st.session_state["dns_auth_session_token"] = session_token
                st.session_state["dns_auth_last_activity"] = time.time()
                st.session_state["dns_auth_cookie_refreshed"] = time.time()
                st.session_state["dns_auth_manage_open"] = False
                st.rerun()
    return None


def render_account_toolbar(user: dict[str, Any]) -> bool:
    spacer, identity, manage, logout = st.columns([5.8, 1.8, 1.25, .8])
    with identity:
        st.caption(f"Signed in: **{user.get('display_name')}** · {role_label(str(user.get('role')))}")
    with manage:
        manage_clicked = st.button("⚙️ Account", key="dns_open_account_management", use_container_width=True)
    with logout:
        logout_clicked = st.button("Sign out", key="dns_local_logout", use_container_width=True)
    if manage_clicked:
        st.session_state["dns_auth_manage_open"] = not bool(st.session_state.get("dns_auth_manage_open"))
        st.rerun()
    if logout_clicked:
        _clear_session(_cookie_controller())
        st.rerun()
    return bool(st.session_state.get("dns_auth_manage_open"))


def _list_users() -> list[dict[str, Any]]:
    connection = _connect()
    try:
        rows = connection.execute(
            """
            SELECT id, username, display_name, role, active, must_change_password,
                   failed_attempts, locked_until_epoch, created_at_utc, last_login_at_utc
            FROM users ORDER BY username COLLATE NOCASE
            """
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def _create_user(actor: str, username: str, display_name: str, role: str, password: str) -> str | None:
    username = str(username or "").strip()
    display_name = str(display_name or username).strip() or username
    error = username_policy_error(username) or password_policy_error(password)
    if error:
        return error
    if role not in {"admin", "user", "soc"}:
        return "Invalid account role."
    digest, salt, rounds = _password_record(password)
    connection = _connect()
    try:
        now = _utc_now()
        connection.execute(
            """
            INSERT INTO users
            (username, display_name, role, password_hash, password_salt, password_iterations,
             active, must_change_password, created_at_utc, updated_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, 1, 1, ?, ?)
            """,
            (username, display_name, role, digest, salt, rounds, now, now),
        )
        _audit(connection, "create_user", "success", actor=actor, target=username, detail=f"role={role}")
        connection.commit()
        return None
    except sqlite3.IntegrityError:
        connection.rollback()
        return "That username already exists."
    finally:
        connection.close()


def _set_user_active(actor: str, target_id: int, active: bool) -> str | None:
    connection = _connect()
    try:
        target = connection.execute("SELECT username, role, active FROM users WHERE id = ?", (target_id,)).fetchone()
        if target is None:
            return "User was not found."
        if not active and str(target["username"]).lower() == actor.lower():
            return "You cannot disable your own signed-in account."
        if not active and str(target["role"]) == "admin":
            active_admins = int(connection.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND active=1").fetchone()[0])
            if active_admins <= 1:
                return "The last active administrator cannot be disabled."
        connection.execute(
            "UPDATE users SET active = ?, failed_attempts = 0, locked_until_epoch = NULL, updated_at_utc = ? WHERE id = ?",
            (1 if active else 0, _utc_now(), target_id),
        )
        if not active:
            connection.execute("DELETE FROM auth_sessions WHERE user_id = ?", (target_id,))
        _audit(connection, "set_user_active", "success", actor=actor, target=str(target["username"]), detail=f"active={active}")
        connection.commit()
        return None
    finally:
        connection.close()


def _admin_reset_password(actor: str, target_id: int, password: str) -> str | None:
    error = password_policy_error(password)
    if error:
        return error
    digest, salt, rounds = _password_record(password)
    connection = _connect()
    try:
        target = connection.execute("SELECT username FROM users WHERE id = ?", (target_id,)).fetchone()
        if target is None:
            return "User was not found."
        connection.execute(
            """
            UPDATE users SET password_hash = ?, password_salt = ?, password_iterations = ?,
                must_change_password = 1, failed_attempts = 0, locked_until_epoch = NULL,
                updated_at_utc = ? WHERE id = ?
            """,
            (digest, salt, rounds, _utc_now(), target_id),
        )
        connection.execute("DELETE FROM auth_sessions WHERE user_id = ?", (target_id,))
        _audit(connection, "reset_password", "success", actor=actor, target=str(target["username"]))
        connection.commit()
        return None
    finally:
        connection.close()


def _cli_set_password(username: str, password: str) -> str | None:
    """Set an existing local user's password without enabling first-login rotation."""
    error = password_policy_error(password)
    if error:
        return error
    connection = _connect()
    try:
        target = connection.execute(
            "SELECT id, username FROM users WHERE username = ? COLLATE NOCASE",
            (str(username or "").strip(),),
        ).fetchone()
        if target is None:
            return "User was not found."
        digest, salt, rounds = _password_record(password)
        connection.execute(
            """
            UPDATE users SET password_hash = ?, password_salt = ?, password_iterations = ?,
                must_change_password = 0, failed_attempts = 0, locked_until_epoch = NULL,
                updated_at_utc = ? WHERE id = ?
            """,
            (digest, salt, rounds, _utc_now(), int(target["id"])),
        )
        connection.execute("DELETE FROM auth_sessions WHERE user_id = ?", (int(target["id"]),))
        _audit(
            connection,
            "cli_set_password",
            "success",
            actor="local-cli",
            target=str(target["username"]),
        )
        connection.commit()
        return None
    finally:
        connection.close()


def render_user_management_page(user: dict[str, Any]) -> None:
    st.title("⚙️ Account management")
    if st.button("← Return to DNS Console", key="dns_close_account_management"):
        st.session_state["dns_auth_manage_open"] = False
        st.rerun()
    tab_names = ["My profile", "About / Legal"]
    if user.get("role") == "admin":
        tab_names.append("User administration")
    tabs = st.tabs(tab_names)
    with tabs[0]:
        st.subheader("My profile")
        st.write(f"**Username:** {user.get('username')}")
        st.write(f"**Display name:** {user.get('display_name')}")
        st.write(f"**Role:** {role_label(str(user.get('role')))}")
        st.write(f"**Last login:** {user.get('last_login_at_utc') or 'Not available'}")
        with st.form("dns_profile_password_change", clear_on_submit=True):
            current = st.text_input("Current password", type="password")
            new = st.text_input(
                "New password",
                type="password",
                help="Any non-empty password is accepted; no complexity rules are enforced.",
            )
            confirm = st.text_input("Confirm new password", type="password")
            submitted = st.form_submit_button("Change my password", type="primary")
        if submitted:
            if new != confirm:
                st.error("The new passwords do not match.")
            else:
                error = _change_password(int(user["id"]), current, new, actor=str(user["username"]))
                if error:
                    st.error(error)
                else:
                    st.success("Your password was changed.")

    with tabs[1]:
        st.subheader("DNS ML Anomaly Detection")
        st.write(f"**Version:** `{PRODUCT_VERSION}`")
        st.write(f"**Build classification:** {BUILD_CLASSIFICATION}")
        st.write(f"**Build fingerprint:** `{BUILD_FINGERPRINT}`")
        st.write("**Copyright © 2026 Ahmed Mekky. All rights reserved.**")
        st.write(
            "**Author:** Ahmed Mekky — Deputy General Manager, "
            "Head of Incident Response and Digital Forensics"
        )
        st.write("**Contact:** ahmedmekkyf13@gmail.com")
        st.info(
            "Approved scope: authorized defensive cybersecurity, DNS monitoring, "
            "anomaly detection, threat-intelligence correlation, incident response, "
            "and authorized security testing."
        )
        st.warning(
            "This software does not authorize interception, capture, scanning, or "
            "monitoring without permission. Modified builds must retain the original "
            "ownership notice, identify their changes, and must not be represented "
            "as an official Ahmed Mekky release."
        )
        st.caption(
            "Use, modification, and redistribution are governed by the LICENSE and "
            "NOTICE supplied with the corresponding source release."
        )

    if user.get("role") != "admin":
        return
    with tabs[2]:
        st.subheader("Create local user")
        with st.form("dns_create_local_user", clear_on_submit=True):
            username = st.text_input("New username")
            display_name = st.text_input("Display name")
            role = st.selectbox(
                "Role",
                ["user", "soc", "admin"],
                format_func=role_label,
                help="SOC Analyst can access only Alerts, Alert Details, and Source IP Investigation.",
            )
            password = st.text_input(
                "Temporary password",
                type="password",
                help="Any non-empty password is accepted; no complexity rules are enforced.",
            )
            confirm = st.text_input("Confirm temporary password", type="password")
            create_clicked = st.form_submit_button("Create user", type="primary")
        if create_clicked:
            if password != confirm:
                st.error("The temporary passwords do not match.")
            else:
                error = _create_user(str(user["username"]), username, display_name, role, password)
                if error:
                    st.error(error)
                else:
                    st.success(f"User {username} created. A password change is required at first login.")
                    st.rerun()

        st.subheader("Existing users")
        users = _list_users()
        st.dataframe(
            [
                {
                    "Username": item["username"],
                    "Display name": item["display_name"],
                    "Role": role_label(str(item["role"])),
                    "Active": bool(item["active"]),
                    "Password change required": bool(item["must_change_password"]),
                    "Last login UTC": item["last_login_at_utc"],
                }
                for item in users
            ],
            use_container_width=True,
            hide_index=True,
        )
        labels = {
            f"{item['username']} · {role_label(str(item['role']))} · {'active' if item['active'] else 'disabled'}": item
            for item in users
        }
        selected_label = st.selectbox("Manage user", list(labels))
        selected = labels[selected_label]
        action_left, action_right = st.columns(2)
        with action_left:
            desired_active = not bool(selected["active"])
            if st.button(
                "Enable account" if desired_active else "Disable account",
                key="dns_toggle_user_active",
                use_container_width=True,
            ):
                error = _set_user_active(str(user["username"]), int(selected["id"]), desired_active)
                if error:
                    st.error(error)
                else:
                    st.success("Account status updated.")
                    st.rerun()
        with action_right:
            st.caption("Password reset forces a change at the user's next login.")
        with st.form("dns_admin_password_reset", clear_on_submit=True):
            reset_password = st.text_input(
                "New temporary password",
                type="password",
                help="Any non-empty password is accepted; no complexity rules are enforced.",
            )
            reset_confirm = st.text_input("Confirm temporary password", type="password")
            reset_clicked = st.form_submit_button("Reset selected user password")
        if reset_clicked:
            if reset_password != reset_confirm:
                st.error("The temporary passwords do not match.")
            else:
                error = _admin_reset_password(str(user["username"]), int(selected["id"]), reset_password)
                if error:
                    st.error(error)
                else:
                    st.success("Temporary password assigned. The user must change it at next login.")


def _main_cli() -> int:
    parser = argparse.ArgumentParser(description="DNS GUI local authentication administration")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--has-users", action="store_true")
    parser.add_argument("--bootstrap-admin", action="store_true")
    parser.add_argument(
        "--set-password",
        action="store_true",
        help="Set an existing user's password using DNS_GUI_PASSWORD or a hidden interactive prompt.",
    )
    parser.add_argument("--username", default=os.getenv("DNS_GUI_ADMIN_USERNAME", ""))
    parser.add_argument("--display-name", default=os.getenv("DNS_GUI_ADMIN_DISPLAY_NAME", "DNS Administrator"))
    args = parser.parse_args()
    init_auth_db()
    if args.has_users:
        return 0 if user_count() > 0 else 1
    if args.bootstrap_admin:
        password = os.getenv("DNS_GUI_ADMIN_PASSWORD", "")
        result = bootstrap_admin(args.username, password, args.display_name)
        if result.get("ok"):
            print(result.get("message"))
            return 0
        print(result.get("error") or "Unable to create administrator.", file=sys.stderr)
        return 2
    if args.set_password:
        if not str(args.username or "").strip():
            print("--username is required with --set-password.", file=sys.stderr)
            return 2
        password = os.getenv("DNS_GUI_PASSWORD")
        if password is None:
            if not sys.stdin.isatty():
                print("Set DNS_GUI_PASSWORD or run from an interactive terminal.", file=sys.stderr)
                return 2
            password = getpass.getpass("New password: ")
            confirmation = getpass.getpass("Confirm new password: ")
            if password != confirmation:
                print("The passwords do not match.", file=sys.stderr)
                return 2
        error = _cli_set_password(args.username, password)
        if error:
            print(error, file=sys.stderr)
            return 2
        print(f"Password updated for {args.username}.")
        return 0
    if args.init:
        print(AUTH_DB_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main_cli())
