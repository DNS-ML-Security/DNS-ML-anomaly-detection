#!/usr/bin/env bash
# DNS ML Anomaly Detection - Version 1 release-candidate installer
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

set -Eeuo pipefail
IFS=$'\n\t'
umask 077

PRODUCT="DNS ML Anomaly Detection"
VERSION="v1.0.0-rc.1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_SOURCE="$SCRIPT_DIR/src/dns-ml"
GUI_SOURCE="$SCRIPT_DIR/src/dns-ml-gui"
ENGINE_TARGET="/opt/dns-ml"
GUI_TARGET="/opt/dns-ml-gui"
DATA_ROOT="/data/dns-ml"
AUTH_DB="$DATA_ROOT/gui_auth/users.db"
SERVICE_NAME="dns-ml-gui.service"
CAPTURE_SERVICE_NAME="dns-ml-learning-capture.service"
LOG_PATH="/var/log/dns-ml-install.log"
APT_LOCK_TIMEOUT="${DNS_ML_APT_LOCK_TIMEOUT:-900}"
ZEEK_KEYRING="/etc/apt/keyrings/security_zeek.gpg"
ZEEK_SOURCE_LIST="/etc/apt/sources.list.d/security_zeek.list"
LEGACY_ZEEK_KEYRING="/etc/apt/trusted.gpg.d/security_zeek.gpg"
LEGACY_ZEEK_SOURCE_LIST="/etc/apt/sources.list.d/security:zeek.list"

info() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

apt_get() {
    apt-get -o "DPkg::Lock::Timeout=${APT_LOCK_TIMEOUT}" "$@"
}

repair_legacy_zeek_repo_access() {
    local path
    for path in "$LEGACY_ZEEK_KEYRING" "$LEGACY_ZEEK_SOURCE_LIST"; do
        if [[ -e "$path" ]]; then
            chmod 0644 "$path"
        fi
    done
}

on_error() {
    local status=$?
    printf '[ERROR] Installation stopped at line %s with status %s. Review %s.\n' \
        "${BASH_LINENO[0]:-unknown}" "$status" "$LOG_PATH" >&2
    exit "$status"
}
trap on_error ERR

[[ "$EUID" -eq 0 ]] || die "Run with sudo: sudo bash install_dns_ml_anomaly_detection.sh"
[[ -f "$ENGINE_SOURCE/feature_utils.py" ]] || die "Engine source is missing: $ENGINE_SOURCE"
[[ -f "$GUI_SOURCE/app.py" ]] || die "GUI source is missing: $GUI_SOURCE"
[[ -f "$SCRIPT_DIR/packaging/systemd/$SERVICE_NAME" ]] || die "Service definition is missing."
[[ -f "$SCRIPT_DIR/packaging/systemd/$CAPTURE_SERVICE_NAME" ]] || die "Learning-capture service definition is missing."

source /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]] \
    || die "This release candidate supports Ubuntu Server 24.04 LTS only."

touch "$LOG_PATH"
chmod 0600 "$LOG_PATH"
exec > >(tee -a "$LOG_PATH") 2>&1

info "$PRODUCT installer $VERSION"
info "No capture-interface checks or changes will be performed."

existing_admin=0
if [[ -s "$AUTH_DB" ]]; then
    if python3 - "$AUTH_DB" <<'PY' >/dev/null 2>&1
import sqlite3
import sys

try:
    connection = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    count = int(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0])
    connection.close()
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if count > 0 else 1)
PY
    then
        existing_admin=1
        info "Existing GUI users detected; the authentication database will be preserved."
    fi
fi

ADMIN_PASSWORD=""
if [[ "$existing_admin" -eq 0 ]]; then
    [[ -t 0 ]] || die "A terminal is required to enter the initial GUI administrator password."
    while :; do
        read -r -s -p "Enter the initial GUI administrator password: " ADMIN_PASSWORD
        printf '\n'
        read -r -s -p "Confirm the initial GUI administrator password: " ADMIN_CONFIRM
        printf '\n'
        if [[ -z "$ADMIN_PASSWORD" ]]; then
            warn "The password cannot be empty."
        elif [[ "$ADMIN_PASSWORD" != "$ADMIN_CONFIRM" ]]; then
            warn "The passwords do not match."
        else
            break
        fi
    done
    unset ADMIN_CONFIRM
fi

export DEBIAN_FRONTEND=noninteractive
info "Installing required Ubuntu packages."
info "Package-manager lock wait limit: ${APT_LOCK_TIMEOUT} seconds."
repair_legacy_zeek_repo_access
apt_get update
apt_get install -y --no-install-recommends \
    ca-certificates \
    cron \
    curl \
    gnupg \
    iproute2 \
    jq \
    libgomp1 \
    libpcap0.8 \
    python3 \
    python3-pip \
    python3-venv

if [[ ! -x /opt/zeek/bin/zeek ]]; then
    info "Installing the supported Zeek 8.0 LTS package."
    zeek_repo_tmp="$(mktemp -d /tmp/dns-ml-zeek-repo.XXXXXXXX)"
    curl -fsSL \
        https://download.opensuse.org/repositories/security:/zeek/xUbuntu_24.04/Release.key \
        -o "$zeek_repo_tmp/Release.key"
    gpg --dearmor --yes \
        --output "$zeek_repo_tmp/security_zeek.gpg" \
        "$zeek_repo_tmp/Release.key"
    install -d -m 0755 /etc/apt/keyrings
    install -m 0644 "$zeek_repo_tmp/security_zeek.gpg" "$ZEEK_KEYRING"
    printf '%s\n' \
        "deb [signed-by=$ZEEK_KEYRING] https://download.opensuse.org/repositories/security:/zeek/xUbuntu_24.04/ /" \
        >"$zeek_repo_tmp/security_zeek.list"
    install -m 0644 "$zeek_repo_tmp/security_zeek.list" "$ZEEK_SOURCE_LIST"
    rm -f -- \
        "$zeek_repo_tmp/Release.key" \
        "$zeek_repo_tmp/security_zeek.gpg" \
        "$zeek_repo_tmp/security_zeek.list"
    rmdir -- "$zeek_repo_tmp"
    rm -f -- "$LEGACY_ZEEK_KEYRING" "$LEGACY_ZEEK_SOURCE_LIST"
    apt_get update
    apt_get install -y --no-install-recommends zeek-8.0
else
    info "Existing Zeek installation detected; it will be preserved."
fi

info "Creating application and runtime directories."
install -d -m 0755 "$ENGINE_TARGET" "$GUI_TARGET"
install -d -m 0750 \
    "$DATA_ROOT" \
    "$DATA_ROOT/alerts" \
    "$DATA_ROOT/dns_preparation" \
    "$DATA_ROOT/dns_preparation/live" \
    "$DATA_ROOT/dns_preparation/output" \
    "$DATA_ROOT/dns_preparation/pcap/incoming" \
    "$DATA_ROOT/dns_preparation/pcap/jobs" \
    "$DATA_ROOT/features" \
    "$DATA_ROOT/final_scoring" \
    "$DATA_ROOT/gui" \
    "$DATA_ROOT/heuristics_scoring" \
    "$DATA_ROOT/isolation_scoring" \
    "$DATA_ROOT/models" \
    "$DATA_ROOT/runs" \
    "$DATA_ROOT/scoring" \
    "$DATA_ROOT/state" \
    "$DATA_ROOT/training" \
    "$DATA_ROOT/training_upload/incoming" \
    "$DATA_ROOT/training_upload/files" \
    "$DATA_ROOT/zeek"
install -d -m 0700 "$DATA_ROOT/gui_auth"

info "Installing approved application source."
tar -C "$ENGINE_SOURCE" --exclude='__pycache__' --exclude='*.pyc' -cf - . \
    | tar -C "$ENGINE_TARGET" --no-same-owner -xf -

# Preserve an existing local GUI configuration during reinstall. The release
# template is installed only when config.json does not already exist.
tar -C "$GUI_SOURCE" \
    --exclude='./config.json' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -cf - . \
    | tar -C "$GUI_TARGET" --no-same-owner -xf -
if [[ ! -f "$GUI_TARGET/config.json" ]]; then
    install -m 0640 "$GUI_SOURCE/config.json" "$GUI_TARGET/config.json"
fi

find "$ENGINE_TARGET" "$GUI_TARGET" -type d -exec chmod 0755 {} +
find "$ENGINE_TARGET" "$GUI_TARGET" -type f -exec chmod 0644 {} +
find "$ENGINE_TARGET" "$GUI_TARGET" -type f -name '*.sh' -exec chmod 0755 {} +

info "Creating the Python virtual environment and installing dependencies."
if [[ ! -x "$ENGINE_TARGET/venv/bin/python" ]]; then
    python3 -m venv "$ENGINE_TARGET/venv"
fi
PIP_DISABLE_PIP_VERSION_CHECK=1 "$ENGINE_TARGET/venv/bin/python" -m pip install --upgrade pip setuptools wheel
PIP_DISABLE_PIP_VERSION_CHECK=1 "$ENGINE_TARGET/venv/bin/python" -m pip install \
    -r "$ENGINE_TARGET/requirements.txt" \
    -r "$GUI_TARGET/requirements.txt"

if [[ "$existing_admin" -eq 0 ]]; then
    info "Creating the initial local GUI administrator: admin"
    env \
        DNS_GUI_AUTH_DB="$AUTH_DB" \
        DNS_GUI_ADMIN_USERNAME="admin" \
        DNS_GUI_ADMIN_PASSWORD="$ADMIN_PASSWORD" \
        PYTHONPATH="$GUI_TARGET" \
        "$ENGINE_TARGET/venv/bin/python" \
        "$GUI_TARGET/src/local_auth.py" \
        --bootstrap-admin \
        --username admin
    unset ADMIN_PASSWORD
fi

chmod 0700 "$DATA_ROOT/gui_auth"
[[ ! -e "$AUTH_DB" ]] || chmod 0600 "$AUTH_DB"

info "Installing and starting system services."
install -m 0644 \
    "$SCRIPT_DIR/packaging/systemd/$SERVICE_NAME" \
    "/etc/systemd/system/$SERVICE_NAME"
install -m 0644 \
    "$SCRIPT_DIR/packaging/systemd/$CAPTURE_SERVICE_NAME" \
    "/etc/systemd/system/$CAPTURE_SERVICE_NAME"

release_fingerprint="unpackaged-release-candidate"
if [[ -f "$SCRIPT_DIR/release/SOURCE_TREE.sha256" ]]; then
    release_fingerprint="$(sha256sum "$SCRIPT_DIR/release/SOURCE_TREE.sha256" | awk '{print $1}')"
fi
cat >/etc/default/dns-ml-gui <<EOF
DNS_PRODUCT_VERSION=$VERSION
DNS_BUILD_CLASSIFICATION=Official_release_candidate
DNS_BUILD_FINGERPRINT=$release_fingerprint
EOF
chmod 0644 /etc/default/dns-ml-gui

systemctl daemon-reload
systemctl enable --now cron.service
systemctl enable "$CAPTURE_SERVICE_NAME"
systemctl restart "$CAPTURE_SERVICE_NAME"
systemctl enable "$SERVICE_NAME"
systemctl restart "$SERVICE_NAME"

for _ in $(seq 1 20); do
    if systemctl is-active --quiet "$SERVICE_NAME" \
        && systemctl is-active --quiet "$CAPTURE_SERVICE_NAME"; then
        break
    fi
    sleep 1
done
systemctl is-active --quiet "$SERVICE_NAME" \
    || die "$SERVICE_NAME did not become active. Review: journalctl -u $SERVICE_NAME"
systemctl is-active --quiet "$CAPTURE_SERVICE_NAME" \
    || die "$CAPTURE_SERVICE_NAME did not become active. Review: journalctl -u $CAPTURE_SERVICE_NAME"

server_address="$(hostname -I 2>/dev/null | awk '{print $1}')"
server_address="${server_address:-SERVER_ADDRESS}"

printf '\n%s\n' '=================================================================='
printf '%s installation completed.\n' "$PRODUCT"
printf 'Version      : %s\n' "$VERSION"
printf 'GUI username : admin\n'
printf 'GUI address  : http://%s:8779\n' "$server_address"
printf 'GUI service  : %s\n' "$(systemctl is-active "$SERVICE_NAME")"
printf 'Prep service : %s\n' "$(systemctl is-active "$CAPTURE_SERVICE_NAME")"
printf 'Zeek setup   : Select the authorized capture interface later in the GUI.\n'
printf '%s\n\n' '=================================================================='
