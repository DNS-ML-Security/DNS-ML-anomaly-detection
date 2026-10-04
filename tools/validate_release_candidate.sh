#!/usr/bin/env bash
# DNS ML Anomaly Detection release-candidate validation
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.

set -Eeuo pipefail
IFS=$'\n\t'

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FAILURES=0

pass() { printf '[PASS] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*" >&2; FAILURES=$((FAILURES + 1)); }

required=(
    README.md LICENSE NOTICE AUTHORS.md SECURITY.md CHANGELOG.md
    CONTRIBUTIONS.md THIRD_PARTY_NOTICES.md VERSION .gitignore
    DNS-ML-anomaly-detection.gif
    install_dns_ml_anomaly_detection.sh
    packaging/systemd/dns-ml-gui.service
    packaging/systemd/dns-ml-learning-capture.service
    src/dns-ml/feature_utils.py
    src/dns-ml/train_dns_autoencoder.py
    src/dns-ml/train_dns_isolation_forest.py
    src/dns-ml-gui/app.py
    src/dns-ml-gui/src/dns_log_preparation.py
    src/dns-ml-gui/src/dns_log_preparation_page.py
    src/dns-ml-gui/src/local_auth.py
)

for relative in "${required[@]}"; do
    [[ -f "$ROOT/$relative" ]] || fail "Required file is missing: $relative"
done
[[ "$FAILURES" -eq 0 ]] && pass "Required repository files are present"

forbidden_names="$(
    find "$ROOT" -type f \( \
        -iname '*.db' -o -iname '*.db-*' -o -iname '*.sqlite' -o -iname '*.sqlite3' \
        -o -iname '*.log' -o -iname '*.jsonl' -o -iname '*.ndjson' \
        -o -iname '*.pcap' -o -iname '*.pcapng' -o -iname '*.csv' -o -iname '*.tsv' \
        -o -iname '*.joblib' -o -iname '*.keras' -o -iname '*.h5' \
        -o -iname '*.pem' -o -iname '*.key' -o -iname '*.p12' -o -iname '*.pfx' \
        -o -iname '.env' -o -iname '.env.*' \
    \) -printf '%P\n' | sort
)"
if [[ -n "$forbidden_names" ]]; then
    fail "Forbidden artifacts detected"
    printf '%s\n' "$forbidden_names" >&2
else
    pass "No forbidden data, model, credential, or runtime artifact types detected"
fi

scan_pattern() {
    local description="$1"
    local pattern="$2"
    local matches
    matches="$(grep -RIlE --binary-files=without-match -- "$pattern" "$ROOT" 2>/dev/null || true)"
    if [[ -n "$matches" ]]; then
        fail "$description"
        while IFS= read -r match; do
            printf '  %s\n' "${match#"$ROOT/"}" >&2
        done <<<"$matches"
    else
        pass "$description: none detected"
    fi
}

scan_pattern \
    "Private IPv4 literals" \
    '(^|[^0-9])(10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|192\.168\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3})([^0-9]|$)'
scan_pattern \
    "Private keys or recognizable access tokens" \
    '(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN ([A-Z0-9]+ )?PRIVATE KEY-----)'

shell_failed=0
while IFS= read -r -d '' shell_file; do
    bash -n "$shell_file" || shell_failed=1
done < <(find "$ROOT" -type f -name '*.sh' -print0)
[[ "$shell_failed" -eq 0 ]] && pass "Shell syntax" || fail "Shell syntax"

python_failed=0
python_cache="$(mktemp -d /tmp/dns-ml-validation-pycache.XXXXXXXX)"
while IFS= read -r -d '' python_file; do
    PYTHONPYCACHEPREFIX="$python_cache" python3 -m py_compile "$python_file" \
        || python_failed=1
done < <(find "$ROOT" -type f -name '*.py' -print0)
[[ "$python_failed" -eq 0 ]] && pass "Python syntax" || fail "Python syntax"

if [[ "$FAILURES" -ne 0 ]]; then
    printf '[FAIL] Release-candidate validation completed with %s failure(s).\n' "$FAILURES" >&2
    exit 1
fi

pass "Release-candidate validation completed successfully"
