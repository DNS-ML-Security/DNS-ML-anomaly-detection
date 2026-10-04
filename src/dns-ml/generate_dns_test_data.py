#!/usr/bin/env python3
# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.


import json
import random
import string
import time
from pathlib import Path
from typing import Dict, List


# ============================================================
# Output files
# ============================================================

TRAINING_OUT = Path("/data/dns-ml/training/synthetic_normal_dns.jsonl")
TEST_OUT = Path("/data/dns-ml/test_suspicious_dns.log")
SUMMARY_OUT = Path("/data/dns-ml/test_data_generation_summary.json")


# ============================================================
# Main generation controls
# ============================================================

SEED = 42
random.seed(SEED)

# Normal training data:
# 50 IPs x 4 days x 288 five-minute windows/day = 57,600 windows
NORMAL_DAYS = 4
NORMAL_SOURCE_IP_COUNT = 50

# Suspicious test data duration
SUSPICIOUS_TEST_MINUTES = 90

# Internal DNS resolvers
INTERNAL_DNS_SERVERS = [
    "192.0.2.1",
    "192.0.2.2",
]

# External / rogue resolvers used only in suspicious test data
ROGUE_DNS_SERVERS = [
    "8.8.8.8",
    "8.8.4.4",
    "1.1.1.1",
    "1.0.0.1",
    "9.9.9.9",
    "208.67.222.222",
]


# ============================================================
# Normal enterprise source IPs
# ============================================================

NORMAL_SRC_IPS = [
    f"10.10.{subnet}.{host}"
    for subnet in range(5, 10)
    for host in range(10, 20)
][:NORMAL_SOURCE_IP_COUNT]


# ============================================================
# Suspicious source IPs
# ============================================================

ATTACK_IPS = {
    "dga": "192.0.2.3",
    "dns_tunnel": "192.0.2.4",
    "c2_beacon": "192.0.2.5",
    "fast_flux": "192.0.2.6",
    "resolver_failure": "192.0.2.7",
    "rogue_resolver": "192.0.2.8",
}


# ============================================================
# Legitimate-looking enterprise / daily browsing domains
# ============================================================

POPULAR_LEGITIMATE_DOMAINS = [
    # Search / web / browser ecosystems
    "google.com",
    "googleapis.com",
    "gstatic.com",
    "googleusercontent.com",
    "googlevideo.com",
    "youtube.com",
    "ytimg.com",
    "bing.com",
    "duckduckgo.com",
    "mozilla.org",
    "firefox.com",

    # Microsoft / enterprise
    "microsoft.com",
    "windowsupdate.com",
    "office.com",
    "office365.com",
    "microsoftonline.com",
    "login.microsoftonline.com",
    "teams.microsoft.com",
    "sharepoint.com",
    "onedrive.live.com",
    "outlook.com",
    "live.com",
    "msftconnecttest.com",
    "msftncsi.com",
    "azure.com",
    "azureedge.net",
    "trafficmanager.net",
    "skype.com",

    # Apple
    "apple.com",
    "icloud.com",
    "mzstatic.com",
    "apple-dns.net",
    "cdn-apple.com",

    # Cloud / CDN / developer
    "amazonaws.com",
    "amazon.com",
    "cloudfront.net",
    "aws.amazon.com",
    "github.com",
    "githubusercontent.com",
    "githubassets.com",
    "gitlab.com",
    "docker.com",
    "docker.io",
    "cloudflare.com",
    "cloudflare-dns.com",
    "fastly.net",
    "akamaihd.net",
    "edgesuite.net",
    "edgekey.net",
    "cdn.jsdelivr.net",
    "jsdelivr.net",
    "npmjs.com",
    "pypi.org",
    "python.org",
    "ubuntu.com",
    "canonical.com",
    "debian.org",
    "redhat.com",

    # Social / communications
    "facebook.com",
    "fbcdn.net",
    "instagram.com",
    "whatsapp.com",
    "whatsapp.net",
    "linkedin.com",
    "licdn.com",
    "twitter.com",
    "x.com",
    "tiktok.com",
    "snapchat.com",
    "telegram.org",
    "slack.com",
    "zoom.us",
    "webex.com",
    "discord.com",

    # News / business / services
    "cnn.com",
    "bbc.com",
    "reuters.com",
    "nytimes.com",
    "bloomberg.com",
    "forbes.com",
    "medium.com",
    "wikipedia.org",
    "wikimedia.org",
    "stackoverflow.com",
    "cloud.google.com",

    # Regional / common portals
    "yahoo.com",
    "aol.com",
    "paypal.com",
    "stripe.com",
    "dropbox.com",
    "box.com",
    "adobe.com",
    "salesforce.com",
    "servicenow.com",
    "atlassian.com",
    "jira.com",
    "confluence.atlassian.com",
]


NORMAL_SUBDOMAIN_PREFIXES = [
    "www",
    "api",
    "cdn",
    "static",
    "assets",
    "img",
    "images",
    "login",
    "auth",
    "account",
    "mail",
    "smtp",
    "autodiscover",
    "download",
    "updates",
    "edge",
    "mobile",
    "app",
    "portal",
    "graph",
    "events",
    "telemetry",
    "config",
    "client",
    "gateway",
    "sync",
    "media",
    "video",
    "docs",
    "drive",
]


# ============================================================
# Synthetic but realistic suspicious domains
# Use reserved example.com/.net/.org so no real malicious infra.
# ============================================================

ATTACK_PARENT_DOMAINS = [
    "cdn-update.example.com",
    "asset-sync.example.net",
    "service-gateway.example.org",
    "edge-cache.example.com",
    "cloud-session.example.net",
    "api-health.example.org",
]


def random_string(length: int, charset: str = None) -> str:
    if charset is None:
        charset = string.ascii_lowercase + string.digits
    return "".join(random.choices(charset, k=length))


def random_wordlike_label() -> str:
    words = [
        "cdn", "edge", "api", "sync", "asset", "static", "img", "login",
        "client", "update", "session", "health", "gateway", "media",
        "cache", "cloud", "node", "service", "config", "portal"
    ]

    return random.choice(words) + "-" + random.choice(words) + "-" + str(random.randint(10, 999))


def normal_legitimate_domain() -> str:
    """
    Generate legitimate-looking normal user/enterprise DNS names.
    This is randomized but still uses known common domains.
    """
    base = random.choice(POPULAR_LEGITIMATE_DOMAINS)

    mode = random.random()

    if mode < 0.45:
        return base

    if mode < 0.75:
        return f"{random.choice(NORMAL_SUBDOMAIN_PREFIXES)}.{base}"

    if mode < 0.90:
        return f"{random.choice(NORMAL_SUBDOMAIN_PREFIXES)}-{random.randint(1, 20)}.{base}"

    # CDN-like normal labels, short and not too random
    return f"{random.choice(NORMAL_SUBDOMAIN_PREFIXES)}.{random_wordlike_label()}.{base}"


def dga_domain() -> str:
    """
    DGA-looking but safe.
    Looks like generated hostnames under reserved example domains.
    """
    label = random_string(random.randint(18, 28))
    parent = random.choice([
        "update-check.example.com",
        "device-sync.example.net",
        "cdn-status.example.org",
        "client-gateway.example.com",
    ])
    return f"{label}.{parent}"


def dns_tunnel_domain() -> str:
    """
    DNS tunneling-looking query:
    long high-entropy labels under a safe reserved parent domain.
    """
    label1 = random_string(random.randint(45, 58))
    label2 = random_string(random.randint(35, 52))
    parent = random.choice([
        "data-upload.example.com",
        "session-sync.example.net",
        "edge-relay.example.org",
    ])
    return f"{label1}.{label2}.{parent}"


def c2_beacon_domain() -> str:
    """
    C2 beaconing often uses repeated legitimate-looking endpoints.
    """
    return random.choice([
        "api.checkin.edge-cache.example.com",
        "cdn.config.asset-sync.example.net",
        "client.update.service-gateway.example.org",
        "telemetry.session.cloud-session.example.net",
    ])


def fast_flux_domain() -> str:
    """
    Fast-flux-like behavior:
    many answers, low TTL, same parent, rotating nodes.
    """
    node = random.choice(["node", "cdn", "edge", "cache", "pop"])
    return f"{node}-{random.randint(1000, 9999)}.edge-cache.example.com"


def resolver_failure_domain() -> str:
    """
    Suspicious infrastructure / resolver failure style.
    Not random-looking at first glance, but frequently fails.
    """
    return random.choice([
        "api.service-gateway.example.org",
        "updates.asset-sync.example.net",
        "status.cdn-update.example.com",
        "health.cloud-session.example.net",
    ])


def rogue_resolver_domain() -> str:
    """
    Legit-looking domains but queried through public rogue resolvers.
    """
    return normal_legitimate_domain()


def random_answer_ips(count: int) -> List[str]:
    answers = []
    for _ in range(count):
        answers.append(
            f"{random.randint(23, 223)}."
            f"{random.randint(1, 254)}."
            f"{random.randint(1, 254)}."
            f"{random.randint(1, 254)}"
        )
    return answers


def zeek_dns_event(
    ts: int,
    src_ip: str,
    query: str,
    dns_server: str,
    qtype: str = "A",
    rcode: str = "NOERROR",
    rtt: float = 0.02,
    answers: List[str] = None,
    ttls: List[float] = None,
) -> Dict:
    if answers is None:
        answers = []

    if ttls is None:
        ttls = []

    return {
        "ts": ts,
        "uid": random_string(12),
        "id.orig_h": src_ip,
        "id.orig_p": random.randint(30000, 65000),
        "id.resp_h": dns_server,
        "id.resp_p": 53,
        "proto": "udp",
        "query": query,
        "qtype_name": qtype,
        "rcode_name": rcode,
        "rtt": round(float(rtt), 5),
        "answers": answers,
        "TTLs": ttls,
    }


def write_jsonl(path: Path, events: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def generate_normal_training() -> List[Dict]:
    """
    Generate several days of legitimate enterprise DNS traffic.
    All domains are legitimate-looking and normal.
    """
    events = []

    now = int(time.time())
    start = now - (NORMAL_DAYS * 24 * 60 * 60)

    total_minutes = NORMAL_DAYS * 24 * 60

    for minute in range(total_minutes):
        base_ts = start + (minute * 60)

        for src_ip in NORMAL_SRC_IPS:
            # Normal enterprise behavior:
            # enough activity to ensure most 5-min windows exist
            query_count = random.randint(2, 7)

            # Business-hour traffic is slightly heavier.
            hour_utc = time.gmtime(base_ts).tm_hour
            if 7 <= hour_utc <= 18:
                query_count += random.randint(1, 4)

            for _ in range(query_count):
                query = normal_legitimate_domain()

                qtype = random.choices(
                    ["A", "AAAA", "CNAME", "TXT", "MX", "PTR", "SRV"],
                    weights=[62, 22, 8, 3, 2, 2, 1],
                    k=1,
                )[0]

                rcode = random.choices(
                    ["NOERROR", "NXDOMAIN", "SERVFAIL"],
                    weights=[97, 2, 1],
                    k=1,
                )[0]

                dns_server = random.choices(
                    INTERNAL_DNS_SERVERS,
                    weights=[80, 20],
                    k=1,
                )[0]

                if rcode == "NOERROR":
                    answers = random_answer_ips(random.choice([1, 1, 1, 2, 3]))
                    ttls = [float(random.choice([60, 120, 300, 600, 1800, 3600])) for _ in answers]
                else:
                    answers = []
                    ttls = []

                rtt = random.uniform(0.004, 0.08)

                events.append(
                    zeek_dns_event(
                        ts=base_ts + random.randint(0, 59),
                        src_ip=src_ip,
                        query=query,
                        dns_server=dns_server,
                        qtype=qtype,
                        rcode=rcode,
                        rtt=rtt,
                        answers=answers,
                        ttls=ttls,
                    )
                )

    return events


def add_background_normal_test_traffic(events: List[Dict], start: int):
    """
    Add normal traffic into suspicious test file so the scorer sees both normal and bad.
    """
    normal_test_ips = NORMAL_SRC_IPS[:10]

    for minute in range(SUSPICIOUS_TEST_MINUTES):
        base_ts = start + (minute * 60)

        for src_ip in normal_test_ips:
            for _ in range(random.randint(1, 4)):
                query = normal_legitimate_domain()
                rcode = random.choices(["NOERROR", "NXDOMAIN"], weights=[98, 2], k=1)[0]
                answers = random_answer_ips(random.choice([1, 1, 2])) if rcode == "NOERROR" else []
                ttls = [float(random.choice([120, 300, 600, 1800])) for _ in answers]

                events.append(
                    zeek_dns_event(
                        ts=base_ts + random.randint(0, 59),
                        src_ip=src_ip,
                        query=query,
                        dns_server=random.choice(INTERNAL_DNS_SERVERS),
                        qtype=random.choices(["A", "AAAA", "CNAME", "TXT"], weights=[70, 20, 8, 2], k=1)[0],
                        rcode=rcode,
                        rtt=random.uniform(0.004, 0.07),
                        answers=answers,
                        ttls=ttls,
                    )
                )


def add_dga_attack(events: List[Dict], start: int):
    """
    DGA: many unique random-looking domains, high NXDOMAIN rate.
    Should trigger heuristic_dga_alert.
    """
    src_ip = ATTACK_IPS["dga"]

    for i in range(350):
        ts = start + random.randint(5 * 60, 25 * 60)
        events.append(
            zeek_dns_event(
                ts=ts,
                src_ip=src_ip,
                query=dga_domain(),
                dns_server=random.choice(INTERNAL_DNS_SERVERS),
                qtype=random.choice(["A", "AAAA"]),
                rcode=random.choices(["NXDOMAIN", "SERVFAIL"], weights=[90, 10], k=1)[0],
                rtt=random.uniform(0.04, 0.15),
            )
        )


def add_dns_tunnel_attack(events: List[Dict], start: int):
    """
    DNS tunneling: long TXT queries, high entropy, long labels.
    """
    src_ip = ATTACK_IPS["dns_tunnel"]

    for i in range(280):
        ts = start + random.randint(20 * 60, 45 * 60)
        events.append(
            zeek_dns_event(
                ts=ts,
                src_ip=src_ip,
                query=dns_tunnel_domain(),
                dns_server=random.choice(INTERNAL_DNS_SERVERS),
                qtype=random.choices(["TXT", "A"], weights=[80, 20], k=1)[0],
                rcode=random.choices(["NOERROR", "NXDOMAIN"], weights=[60, 40], k=1)[0],
                rtt=random.uniform(0.06, 0.25),
                answers=random_answer_ips(1),
                ttls=[60.0],
            )
        )


def add_c2_beacon_attack(events: List[Dict], start: int):
    """
    C2 beaconing: repeated queries to same endpoint at regular intervals.
    Domain is not weird at first look.
    """
    src_ip = ATTACK_IPS["c2_beacon"]
    domain = c2_beacon_domain()

    # Every 30 seconds for 60 minutes
    for offset in range(10 * 60, 70 * 60, 30):
        ts = start + offset
        events.append(
            zeek_dns_event(
                ts=ts,
                src_ip=src_ip,
                query=domain,
                dns_server=random.choice(INTERNAL_DNS_SERVERS),
                qtype="A",
                rcode="NOERROR",
                rtt=random.uniform(0.02, 0.09),
                answers=random_answer_ips(1),
                ttls=[300.0],
            )
        )

    # Add small amount of normal cover traffic
    for i in range(80):
        ts = start + random.randint(10 * 60, 70 * 60)
        events.append(
            zeek_dns_event(
                ts=ts,
                src_ip=src_ip,
                query=normal_legitimate_domain(),
                dns_server=random.choice(INTERNAL_DNS_SERVERS),
                qtype="A",
                rcode="NOERROR",
                rtt=random.uniform(0.005, 0.06),
                answers=random_answer_ips(1),
                ttls=[600.0],
            )
        )


def add_fast_flux_attack(events: List[Dict], start: int):
    """
    Fast-flux / suspicious CDN-like behavior:
    many successful answers, many rotating IPs, very low TTL.
    The current 30-feature model does not fully use answers/TTL,
    but raw Zeek fields are present for future features.
    """
    src_ip = ATTACK_IPS["fast_flux"]

    for i in range(220):
        ts = start + random.randint(35 * 60, 65 * 60)
        answer_count = random.randint(4, 8)

        events.append(
            zeek_dns_event(
                ts=ts,
                src_ip=src_ip,
                query=fast_flux_domain(),
                dns_server=random.choice(INTERNAL_DNS_SERVERS),
                qtype="A",
                rcode="NOERROR",
                rtt=random.uniform(0.03, 0.18),
                answers=random_answer_ips(answer_count),
                ttls=[float(random.choice([15, 20, 30])) for _ in range(answer_count)],
            )
        )


def add_resolver_failure_attack(events: List[Dict], start: int):
    """
    Resolver failure / suspicious infrastructure:
    legitimate-looking domains with high SERVFAIL.
    """
    src_ip = ATTACK_IPS["resolver_failure"]

    for i in range(250):
        ts = start + random.randint(45 * 60, 80 * 60)

        events.append(
            zeek_dns_event(
                ts=ts,
                src_ip=src_ip,
                query=resolver_failure_domain(),
                dns_server=random.choice(INTERNAL_DNS_SERVERS),
                qtype=random.choice(["A", "AAAA", "CNAME"]),
                rcode=random.choices(["SERVFAIL", "NXDOMAIN"], weights=[85, 15], k=1)[0],
                rtt=random.uniform(0.10, 0.40),
            )
        )


def add_rogue_resolver_attack(events: List[Dict], start: int):
    """
    Rogue resolver usage:
    same host uses many public resolvers instead of approved internal DNS.
    """
    src_ip = ATTACK_IPS["rogue_resolver"]

    for i in range(250):
        ts = start + random.randint(15 * 60, 75 * 60)

        query = rogue_resolver_domain()
        rcode = random.choices(["NOERROR", "NXDOMAIN"], weights=[95, 5], k=1)[0]
        answers = random_answer_ips(random.choice([1, 2])) if rcode == "NOERROR" else []
        ttls = [float(random.choice([60, 120, 300])) for _ in answers]

        events.append(
            zeek_dns_event(
                ts=ts,
                src_ip=src_ip,
                query=query,
                dns_server=random.choice(ROGUE_DNS_SERVERS),
                qtype=random.choice(["A", "AAAA"]),
                rcode=rcode,
                rtt=random.uniform(0.02, 0.12),
                answers=answers,
                ttls=ttls,
            )
        )


def generate_suspicious_test() -> List[Dict]:
    """
    Generate suspicious test log with multiple attack scenarios.
    """
    events = []

    now = int(time.time())

    # Put events in the past so score_dns_autoencoder.py sees completed windows.
    start = now - (2 * 60 * 60)

    add_background_normal_test_traffic(events, start)
    add_dga_attack(events, start)
    add_dns_tunnel_attack(events, start)
    add_c2_beacon_attack(events, start)
    add_fast_flux_attack(events, start)
    add_resolver_failure_attack(events, start)
    add_rogue_resolver_attack(events, start)

    return events


def summarize_events(events: List[Dict]) -> Dict:
    if not events:
        return {}

    timestamps = [event["ts"] for event in events]
    src_ips = sorted(set(event["id.orig_h"] for event in events))

    per_src = {}
    for ip in src_ips:
        per_src[ip] = sum(1 for event in events if event["id.orig_h"] == ip)

    return {
        "total_records": len(events),
        "first_timestamp_epoch": min(timestamps),
        "last_timestamp_epoch": max(timestamps),
        "duration_seconds": max(timestamps) - min(timestamps),
        "source_ip_count": len(src_ips),
        "records_per_source_ip": per_src,
    }


def main():
    TRAINING_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEST_OUT.parent.mkdir(parents=True, exist_ok=True)

    print("[+] Generating synthetic normal enterprise DNS training data...")
    normal_events = generate_normal_training()
    write_jsonl(TRAINING_OUT, normal_events)

    print("[+] Generating synthetic suspicious DNS test data...")
    suspicious_events = generate_suspicious_test()
    write_jsonl(TEST_OUT, suspicious_events)

    summary = {
        "seed": SEED,
        "training_file": str(TRAINING_OUT),
        "suspicious_test_file": str(TEST_OUT),
        "normal_training_summary": summarize_events(normal_events),
        "suspicious_test_summary": summarize_events(suspicious_events),
        "suspicious_scenarios": {
            "dga": ATTACK_IPS["dga"],
            "dns_tunnel": ATTACK_IPS["dns_tunnel"],
            "c2_beacon": ATTACK_IPS["c2_beacon"],
            "fast_flux": ATTACK_IPS["fast_flux"],
            "resolver_failure": ATTACK_IPS["resolver_failure"],
            "rogue_resolver": ATTACK_IPS["rogue_resolver"],
        },
        "notes": [
            "Use synthetic_normal_dns.jsonl for training only.",
            "Use synthetic_suspicious_dns.log for scoring and alert testing only.",
            "Suspicious domains are synthetic and use reserved example.com/example.net/example.org namespaces.",
            "The current 30-feature model sees fast-flux partially. Full fast-flux detection is stronger if TTL and answer-count features are added.",
        ],
    }

    with SUMMARY_OUT.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[+] Done.")
    print(f"[+] Normal training file: {TRAINING_OUT}")
    print(f"[+] Suspicious test file: {TEST_OUT}")
    print(f"[+] Summary file: {SUMMARY_OUT}")
    print(f"[+] Normal records: {len(normal_events)}")
    print(f"[+] Suspicious test records: {len(suspicious_events)}")
    print("[+] Suspicious scenario source IPs:")
    for name, ip in ATTACK_IPS.items():
        print(f"    {name}: {ip}")


if __name__ == "__main__":
    main()
