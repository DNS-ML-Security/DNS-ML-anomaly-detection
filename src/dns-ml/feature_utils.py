# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

import gzip
import json
import math
import os
from collections import Counter
from typing import Iterable, List

import numpy as np
import pandas as pd


FEATURES = [
    # Volume behavior
    "total_queries",
    "unique_queries",
    "unique_query_ratio",
    "repeated_query_ratio",

    # DNS response behavior
    "nxdomain_rate",
    "servfail_rate",
    "noerror_rate",

    # Query type behavior
    "txt_rate",
    "a_rate",
    "aaaa_rate",
    "mx_rate",
    "cname_rate",
    "ptr_rate",
    "srv_rate",

    # Query string length behavior
    "avg_query_len",
    "max_query_len",
    "long_query_rate",

    # Entropy / randomness behavior
    "avg_entropy",
    "max_entropy",
    "high_entropy_rate",

    # Domain label structure
    "avg_label_count",
    "max_label_count",
    "avg_max_label_len",
    "max_max_label_len",
    "long_label_rate",

    # Character composition
    "digit_ratio_avg",
    "hyphen_ratio_avg",

    # Resolver / timing behavior
    "unique_dns_servers",
    "avg_rtt",
    "max_rtt",

    # Answer / TTL behavior for fast-flux / suspicious CDN behavior
    "answer_count_avg",
    "answer_count_max",
    "ttl_avg",
    "ttl_min",
    "unique_answer_ips",
    "low_ttl_rate",
]


REQUIRED_ZEEK_COLUMNS = [
    "ts",
    "id.orig_h",
    "id.resp_h",
    "query",
    "qtype_name",
    "rcode_name",
    "rtt",
    "answers",
    "TTLs",
]


def shannon_entropy(value: str) -> float:
    if not isinstance(value, str) or value == "":
        return 0.0

    counts = Counter(value)
    total = len(value)

    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return float(entropy)


def clean_domain(query: str) -> str:
    if not isinstance(query, str):
        return ""

    query = query.strip().lower()

    if query.endswith("."):
        query = query[:-1]

    return query


def safe_ratio(numerator, denominator) -> float:
    try:
        denominator = float(denominator)
        if denominator == 0:
            return 0.0
        return float(numerator) / denominator
    except Exception:
        return 0.0


def open_text_file(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")

    return open(path, "r", encoding="utf-8", errors="ignore")


def parse_json_lines(lines: Iterable[str], keep_raw_line: bool = False) -> pd.DataFrame:
    records = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        try:
            record = json.loads(line)

            if keep_raw_line:
                record["__raw_line"] = line

            records.append(record)

        except json.JSONDecodeError:
            continue

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def load_zeek_dns_json_file(path: str, keep_raw_line: bool = False) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"DNS log file not found: {path}")

    with open_text_file(path) as f:
        return parse_json_lines(f, keep_raw_line=keep_raw_line)


def load_zeek_dns_json_files(paths: List[str], keep_raw_line: bool = False) -> pd.DataFrame:
    frames = []

    for path in paths:
        print(f"[+] Reading DNS log file: {path}", flush=True)

        df = load_zeek_dns_json_file(path, keep_raw_line=keep_raw_line)

        if not df.empty:
            frames.append(df)
            print(f"    Loaded rows: {len(df)}", flush=True)
        else:
            print("    No valid JSON rows found.", flush=True)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in REQUIRED_ZEEK_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    return df


def normalize_list(value) -> List:
    """
    Normalize Zeek answers / TTLs to Python list.

    Handles:
    - list
    - NaN
    - string
    - missing value
    """
    if isinstance(value, list):
        return value

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []

        # Try JSON list string
        if value.startswith("[") and value.endswith("]"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass

        # Zeek TSV-style separator fallback
        if "," in value:
            return [item.strip() for item in value.split(",") if item.strip()]

        return [value]

    return []


def numeric_list(value) -> List[float]:
    items = normalize_list(value)
    numbers = []

    for item in items:
        try:
            numbers.append(float(item))
        except Exception:
            continue

    return numbers


def avg_numeric_list(value) -> float:
    values = numeric_list(value)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def min_numeric_list(value) -> float:
    values = numeric_list(value)
    if not values:
        return 0.0
    return float(min(values))


def unique_answer_count(series: pd.Series) -> int:
    unique_answers = set()

    for value in series:
        for answer in normalize_list(value):
            if answer is not None and str(answer).strip():
                unique_answers.add(str(answer).strip())

    return len(unique_answers)


def parse_utc_timestamps(values: pd.Series) -> pd.Series:
    """Normalize Zeek epoch values or ISO timestamps without double conversion.

    Pandas 3 may expose timezone-aware datetimes at microsecond resolution.
    Converting those datetimes to numeric and then treating the result as epoch
    seconds can move records hundreds of thousands of years into the future.
    Detect datetime input first and infer the unit only for genuinely numeric
    values.
    """
    original = pd.Series(values, copy=True)
    if pd.api.types.is_datetime64_any_dtype(original.dtype):
        return pd.to_datetime(original, utc=True, errors="coerce")

    parsed = pd.Series(pd.NaT, index=original.index, dtype="datetime64[ns, UTC]")
    numeric = pd.to_numeric(original, errors="coerce")
    absolute = numeric.abs()
    units = (
        ("s", numeric.notna() & absolute.lt(1e11)),
        ("ms", numeric.notna() & absolute.ge(1e11) & absolute.lt(1e14)),
        ("us", numeric.notna() & absolute.ge(1e14) & absolute.lt(1e17)),
        ("ns", numeric.notna() & absolute.ge(1e17)),
    )
    for unit, mask in units:
        if mask.any():
            parsed.loc[mask] = pd.to_datetime(
                numeric.loc[mask], unit=unit, utc=True, errors="coerce"
            )

    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            original.loc[missing], utc=True, errors="coerce"
        )
    return parsed


def add_event_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = ensure_required_columns(df)
    df = df.copy()

    df["timestamp"] = parse_utc_timestamps(df["ts"])

    df = df.dropna(subset=["timestamp"])

    df["src_ip"] = df["id.orig_h"].fillna("unknown").astype(str)
    df["dns_server"] = df["id.resp_h"].fillna("unknown").astype(str)

    df["query_clean"] = df["query"].apply(clean_domain)
    df["query_len"] = df["query_clean"].str.len().fillna(0).astype(float)

    df["labels"] = df["query_clean"].apply(
        lambda value: value.split(".") if isinstance(value, str) and value else []
    )

    df["label_count"] = df["labels"].apply(len).astype(float)

    df["max_label_len"] = df["labels"].apply(
        lambda labels: max([len(label) for label in labels], default=0)
    ).astype(float)

    df["entropy"] = df["query_clean"].apply(shannon_entropy).astype(float)

    df["digit_count"] = df["query_clean"].apply(
        lambda value: sum(char.isdigit() for char in value) if isinstance(value, str) else 0
    ).astype(float)

    df["hyphen_count"] = df["query_clean"].apply(
        lambda value: value.count("-") if isinstance(value, str) else 0
    ).astype(float)

    df["digit_ratio"] = df.apply(
        lambda row: safe_ratio(row["digit_count"], row["query_len"]),
        axis=1,
    )

    df["hyphen_ratio"] = df.apply(
        lambda row: safe_ratio(row["hyphen_count"], row["query_len"]),
        axis=1,
    )

    df["qtype_name"] = df["qtype_name"].fillna("UNKNOWN").astype(str).str.upper()
    df["rcode_name"] = df["rcode_name"].fillna("UNKNOWN").astype(str).str.upper()

    # Response behavior
    df["is_nxdomain"] = (df["rcode_name"] == "NXDOMAIN").astype(int)
    df["is_servfail"] = (df["rcode_name"] == "SERVFAIL").astype(int)
    df["is_noerror"] = (df["rcode_name"] == "NOERROR").astype(int)

    # Query type behavior
    df["is_txt"] = (df["qtype_name"] == "TXT").astype(int)
    df["is_a"] = (df["qtype_name"] == "A").astype(int)
    df["is_aaaa"] = (df["qtype_name"] == "AAAA").astype(int)
    df["is_mx"] = (df["qtype_name"] == "MX").astype(int)
    df["is_cname"] = (df["qtype_name"] == "CNAME").astype(int)
    df["is_ptr"] = (df["qtype_name"] == "PTR").astype(int)
    df["is_srv"] = (df["qtype_name"] == "SRV").astype(int)

    # Suspicious string indicators
    df["is_long_query"] = (df["query_len"] >= 80).astype(int)
    df["is_high_entropy"] = (df["entropy"] >= 4.0).astype(int)
    df["is_long_label"] = (df["max_label_len"] >= 50).astype(int)

    df["rtt"] = pd.to_numeric(df["rtt"], errors="coerce").fillna(0.0)

    # Answer / TTL event-level features
    df["answer_list"] = df["answers"].apply(normalize_list)
    df["ttl_list"] = df["TTLs"].apply(normalize_list)

    df["answer_count"] = df["answer_list"].apply(len).astype(float)
    df["ttl_avg_event"] = df["ttl_list"].apply(avg_numeric_list).astype(float)
    df["ttl_min_event"] = df["ttl_list"].apply(min_numeric_list).astype(float)

    # Low TTL only counts when answers exist and TTL exists.
    df["is_low_ttl"] = df.apply(
        lambda row: int(row["answer_count"] > 0 and row["ttl_min_event"] > 0 and row["ttl_min_event"] <= 60),
        axis=1,
    )

    return df


def build_dns_windows(df: pd.DataFrame, window_size: str = "5min") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "src_ip"] + FEATURES)

    df = df.copy()

    if "timestamp" not in df.columns:
        raise ValueError("Missing timestamp column. Run add_event_features() first.")

    # Normalize again at the shared aggregation boundary. If an older caller
    # double-converted an already parsed timestamp, recover from the original
    # Zeek epoch-seconds field before Pandas constructs the time-bin range.
    normalized = parse_utc_timestamps(df["timestamp"])
    minimum_supported = pd.Timestamp("2000-01-01T00:00:00Z")
    maximum_supported = pd.Timestamp("2100-01-01T00:00:00Z")
    reasonable = normalized.between(minimum_supported, maximum_supported)
    if not reasonable.all() and "ts" in df.columns:
        raw_normalized = parse_utc_timestamps(df["ts"])
        normalized = normalized.where(reasonable, raw_normalized)
        reasonable = normalized.between(minimum_supported, maximum_supported)

    invalid_count = int((normalized.isna() | ~reasonable).sum())
    if invalid_count:
        raise ValueError(
            f"DNS window aggregation rejected {invalid_count:,} invalid timestamp(s). "
            "Expected Zeek epoch seconds/milliseconds/microseconds/nanoseconds or ISO-8601 values "
            "between 2000-01-01 and 2100-01-01."
        )
    df["timestamp"] = normalized

    timestamp_span = df["timestamp"].max() - df["timestamp"].min()
    if timestamp_span > pd.Timedelta(days=3660):
        raise ValueError(
            "DNS input spans more than ten years. Refusing window allocation because this usually "
            "indicates an incorrect timestamp unit."
        )

    grouped = (
        df.groupby(
            [
                pd.Grouper(key="timestamp", freq=window_size),
                "src_ip",
            ]
        )
        .agg(
            total_queries=("query_clean", "count"),
            unique_queries=("query_clean", "nunique"),

            nxdomain_count=("is_nxdomain", "sum"),
            servfail_count=("is_servfail", "sum"),
            noerror_count=("is_noerror", "sum"),

            txt_count=("is_txt", "sum"),
            a_count=("is_a", "sum"),
            aaaa_count=("is_aaaa", "sum"),
            mx_count=("is_mx", "sum"),
            cname_count=("is_cname", "sum"),
            ptr_count=("is_ptr", "sum"),
            srv_count=("is_srv", "sum"),

            avg_query_len=("query_len", "mean"),
            max_query_len=("query_len", "max"),
            long_query_count=("is_long_query", "sum"),

            avg_entropy=("entropy", "mean"),
            max_entropy=("entropy", "max"),
            high_entropy_count=("is_high_entropy", "sum"),

            avg_label_count=("label_count", "mean"),
            max_label_count=("label_count", "max"),
            avg_max_label_len=("max_label_len", "mean"),
            max_max_label_len=("max_label_len", "max"),
            long_label_count=("is_long_label", "sum"),

            digit_ratio_avg=("digit_ratio", "mean"),
            hyphen_ratio_avg=("hyphen_ratio", "mean"),

            unique_dns_servers=("dns_server", "nunique"),

            avg_rtt=("rtt", "mean"),
            max_rtt=("rtt", "max"),

            answer_count_avg=("answer_count", "mean"),
            answer_count_max=("answer_count", "max"),
            ttl_avg=("ttl_avg_event", "mean"),
            ttl_min=("ttl_min_event", "min"),
            unique_answer_ips=("answers", unique_answer_count),
            low_ttl_count=("is_low_ttl", "sum"),
        )
        .reset_index()
    )

    grouped["unique_query_ratio"] = grouped.apply(
        lambda row: safe_ratio(row["unique_queries"], row["total_queries"]),
        axis=1,
    )

    grouped["repeated_query_ratio"] = 1.0 - grouped["unique_query_ratio"]

    grouped["nxdomain_rate"] = grouped.apply(
        lambda row: safe_ratio(row["nxdomain_count"], row["total_queries"]),
        axis=1,
    )

    grouped["servfail_rate"] = grouped.apply(
        lambda row: safe_ratio(row["servfail_count"], row["total_queries"]),
        axis=1,
    )

    grouped["noerror_rate"] = grouped.apply(
        lambda row: safe_ratio(row["noerror_count"], row["total_queries"]),
        axis=1,
    )

    grouped["txt_rate"] = grouped.apply(
        lambda row: safe_ratio(row["txt_count"], row["total_queries"]),
        axis=1,
    )

    grouped["a_rate"] = grouped.apply(
        lambda row: safe_ratio(row["a_count"], row["total_queries"]),
        axis=1,
    )

    grouped["aaaa_rate"] = grouped.apply(
        lambda row: safe_ratio(row["aaaa_count"], row["total_queries"]),
        axis=1,
    )

    grouped["mx_rate"] = grouped.apply(
        lambda row: safe_ratio(row["mx_count"], row["total_queries"]),
        axis=1,
    )

    grouped["cname_rate"] = grouped.apply(
        lambda row: safe_ratio(row["cname_count"], row["total_queries"]),
        axis=1,
    )

    grouped["ptr_rate"] = grouped.apply(
        lambda row: safe_ratio(row["ptr_count"], row["total_queries"]),
        axis=1,
    )

    grouped["srv_rate"] = grouped.apply(
        lambda row: safe_ratio(row["srv_count"], row["total_queries"]),
        axis=1,
    )

    grouped["long_query_rate"] = grouped.apply(
        lambda row: safe_ratio(row["long_query_count"], row["total_queries"]),
        axis=1,
    )

    grouped["high_entropy_rate"] = grouped.apply(
        lambda row: safe_ratio(row["high_entropy_count"], row["total_queries"]),
        axis=1,
    )

    grouped["long_label_rate"] = grouped.apply(
        lambda row: safe_ratio(row["long_label_count"], row["total_queries"]),
        axis=1,
    )

    grouped["low_ttl_rate"] = grouped.apply(
        lambda row: safe_ratio(row["low_ttl_count"], row["total_queries"]),
        axis=1,
    )

    grouped = grouped.replace([np.inf, -np.inf], np.nan).fillna(0)

    output_columns = ["timestamp", "src_ip"] + FEATURES

    for column in output_columns:
        if column not in grouped.columns:
            grouped[column] = 0

    return grouped[output_columns].copy()


def calculate_reconstruction_error(x_true: np.ndarray, x_pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(x_true - x_pred), axis=1)


def classify_severity(error: float, threshold: float) -> str:
    if error >= threshold * 2.0:
        return "high"

    if error >= threshold * 1.5:
        return "medium"

    return "low"


def validate_feature_matrix(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    missing = [feature for feature in features if feature not in df.columns]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = df[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce").fillna(0)

    return X


def domain_suspicion_reasons(row: pd.Series) -> List[str]:
    reasons = []

    query_len = float(row.get("query_len", 0))
    entropy = float(row.get("entropy", 0))
    max_label_len = float(row.get("max_label_len", 0))
    label_count = float(row.get("label_count", 0))
    digit_ratio = float(row.get("digit_ratio", 0))
    hyphen_ratio = float(row.get("hyphen_ratio", 0))
    answer_count = float(row.get("answer_count", 0))
    ttl_min_event = float(row.get("ttl_min_event", 0))

    qtype = str(row.get("qtype_name", "")).upper()
    rcode = str(row.get("rcode_name", "")).upper()

    if query_len >= 80:
        reasons.append("long_query")

    if entropy >= 4.0:
        reasons.append("high_entropy")

    if max_label_len >= 50:
        reasons.append("long_label")

    if label_count >= 5:
        reasons.append("deep_subdomain")

    if digit_ratio >= 0.30:
        reasons.append("high_digit_ratio")

    if hyphen_ratio >= 0.15:
        reasons.append("high_hyphen_ratio")

    if qtype == "TXT":
        reasons.append("txt_record")

    if rcode == "NXDOMAIN":
        reasons.append("nxdomain")

    if rcode == "SERVFAIL":
        reasons.append("servfail")

    if answer_count >= 4:
        reasons.append("many_answers")

    if answer_count > 0 and ttl_min_event > 0 and ttl_min_event <= 60:
        reasons.append("low_ttl")

    return reasons


def summarize_suspicious_domains(
    events: pd.DataFrame,
    src_ip: str,
    window_start,
    window_size: str,
    top_n: int = 20,
) -> List[dict]:
    if events.empty:
        return []

    window_start = pd.Timestamp(window_start)
    window_end = window_start + pd.Timedelta(window_size)

    subset = events[
        (events["src_ip"] == src_ip)
        & (events["timestamp"] >= window_start)
        & (events["timestamp"] < window_end)
    ].copy()

    if subset.empty:
        return []

    subset["suspicion_reasons"] = subset.apply(domain_suspicion_reasons, axis=1)

    suspicious = subset[
        subset["suspicion_reasons"].apply(lambda reasons: len(reasons) > 0)
    ].copy()

    if suspicious.empty:
        return []

    rows = []

    for query, group in suspicious.groupby("query_clean"):
        all_reasons = sorted(
            set(
                reason
                for reasons in group["suspicion_reasons"]
                for reason in reasons
            )
        )

        unique_answers = set()
        for value in group["answers"]:
            for answer in normalize_list(value):
                if answer is not None and str(answer).strip():
                    unique_answers.add(str(answer).strip())

        score = 0
        score += 2 if "long_query" in all_reasons else 0
        score += 2 if "high_entropy" in all_reasons else 0
        score += 2 if "long_label" in all_reasons else 0
        score += 1 if "deep_subdomain" in all_reasons else 0
        score += 1 if "high_digit_ratio" in all_reasons else 0
        score += 1 if "high_hyphen_ratio" in all_reasons else 0
        score += 2 if "txt_record" in all_reasons else 0
        score += 1 if "nxdomain" in all_reasons else 0
        score += 1 if "servfail" in all_reasons else 0
        score += 2 if "many_answers" in all_reasons else 0
        score += 2 if "low_ttl" in all_reasons else 0

        rows.append(
            {
                "query": query,
                "count": int(len(group)),
                "qtypes": sorted(group["qtype_name"].dropna().astype(str).unique().tolist()),
                "rcodes": sorted(group["rcode_name"].dropna().astype(str).unique().tolist()),
                "max_entropy": float(group["entropy"].max()),
                "avg_entropy": float(group["entropy"].mean()),
                "max_query_len": float(group["query_len"].max()),
                "max_label_len": float(group["max_label_len"].max()),
                "avg_digit_ratio": float(group["digit_ratio"].mean()),
                "answer_count_max": float(group["answer_count"].max()),
                "unique_answer_ips": int(len(unique_answers)),
                "ttl_min": float(group["ttl_min_event"].min()),
                "ttl_avg": float(group["ttl_avg_event"].mean()),
                "reasons": all_reasons,
                "suspicion_score": int(score),
            }
        )

    rows = sorted(
        rows,
        key=lambda item: (
            item["suspicion_score"],
            item["count"],
            item["unique_answer_ips"],
            item["max_entropy"],
            item["max_query_len"],
        ),
        reverse=True,
    )

    return rows[:top_n]