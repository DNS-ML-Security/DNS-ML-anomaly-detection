# DNS ML Anomaly Detection
# Copyright (c) 2026 Ahmed Mekky. All rights reserved.
# Use and modification are governed by the repository LICENSE file.

import json
import os
from pathlib import Path


GUI_HOME = Path(os.getenv("DNS_GUI_HOME", "/opt/dns-ml-gui"))
CONFIG_PATH = Path(os.getenv("DNS_GUI_CONFIG", GUI_HOME / "config.json"))


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


CONFIG = load_config()

APP_NAME = CONFIG.get("app_name", "DNS Threat Monitoring Console")
DB_PATH = Path(
    os.getenv(
        "DNS_GUI_DB",
        CONFIG.get("database_path", "/data/dns-ml/gui/dns_gui.db"),
    )
)
ALERT_JSONL_PATH = Path(
    os.getenv(
        "DNS_ALERT_FILE",
        CONFIG.get("alert_jsonl_path", "/data/dns-ml/alerts/dns_alerts.jsonl"),
    )
)
MODEL_CONFIG_PATH = Path(
    os.getenv(
        "DNS_MODEL_CONFIG",
        CONFIG.get(
            "model_config_path",
            "/data/dns-ml/models/dns_autoencoder_config.json",
        ),
    )
)

LEARNING_STATUS_PATH = Path(
    CONFIG.get(
        "learning_status_path",
        "/data/dns-ml/runs/latest_training_status.json",
    )
)
LEARNING_LOG_PATH = Path(
    CONFIG.get(
        "learning_log_path",
        "/data/dns-ml/runs/latest_training.log",
    )
)
LEARNING_HISTORY_PATH = Path(
    CONFIG.get(
        "learning_history_path",
        "/data/dns-ml/runs/latest_training_history.csv",
    )
)
LEARNING_RECONSTRUCTION_PATH = Path(
    CONFIG.get(
        "learning_reconstruction_path",
        "/data/dns-ml/runs/latest_reconstruction_errors.csv",
    )
)
LEARNING_RESULT_PATH = Path(
    CONFIG.get(
        "learning_result_path",
        "/data/dns-ml/runs/latest_training_result.json",
    )
)
LEARNING_PID_PATH = Path(
    CONFIG.get(
        "learning_pid_path",
        "/data/dns-ml/runs/latest_training.pid",
    )
)

ZEEK_LOG_ROOT = Path(
    os.getenv(
        "ZEEK_LOG_ROOT",
        CONFIG.get("zeek_log_root", "/opt/zeek/logs"),
    )
)
ZEEK_MANAGER_SCRIPT = Path(
    CONFIG.get(
        "zeek_manager_script",
        "/opt/dns-ml/zeek_log_manager.py",
    )
)
ZEEK_MERGED_PATH = Path(
    CONFIG.get(
        "zeek_merged_path",
        "/data/dns-ml/zeek/merged_dns.jsonl",
    )
)
ZEEK_STATS_PATH = Path(
    CONFIG.get(
        "zeek_stats_path",
        "/data/dns-ml/zeek/stats.json",
    )
)
ZEEK_SOURCE_IP_STATS_PATH = Path(
    CONFIG.get(
        "zeek_source_ip_stats_path",
        "/data/dns-ml/zeek/source_ip_stats.csv",
    )
)
ZEEK_MERGE_STATUS_PATH = Path(
    CONFIG.get(
        "zeek_merge_status_path",
        "/data/dns-ml/zeek/merge_status.json",
    )
)
ZEEK_MERGE_MANIFEST_PATH = Path(
    CONFIG.get(
        "zeek_merge_manifest_path",
        "/data/dns-ml/zeek/merge_manifest.json",
    )
)
ZEEK_MERGE_LOG_PATH = Path(
    CONFIG.get(
        "zeek_merge_log_path",
        "/data/dns-ml/zeek/merge.log",
    )
)
ZEEK_MERGE_PID_PATH = Path(
    CONFIG.get(
        "zeek_merge_pid_path",
        "/data/dns-ml/zeek/merge.pid",
    )
)

APPROVED_DNS_SERVERS = set(CONFIG.get("approved_dns_servers", []))

# DNS Scoring Navigator
ZEEK_CURRENT_DIR = Path(
    CONFIG.get("zeek_current_dir", "/opt/zeek/logs/current")
)
SCORING_SCRIPT_PATH = Path(
    CONFIG.get(
        "scoring_script_path",
        "/opt/dns-ml/score_dns_autoencoder_live.py",
    )
)
SCORING_HOME = Path(
    CONFIG.get("scoring_home", "/data/dns-ml/scoring")
)
SCORING_STATUS_PATH = Path(
    CONFIG.get("scoring_status_path", SCORING_HOME / "status.json")
)
SCORING_PID_PATH = Path(
    CONFIG.get("scoring_pid_path", SCORING_HOME / "manual_scoring.pid")
)
SCORING_LOCK_PATH = Path(
    CONFIG.get("scoring_lock_path", SCORING_HOME / "scoring.lock")
)
SCORING_LIVE_LOG_PATH = Path(
    CONFIG.get("scoring_live_log_path", SCORING_HOME / "live_scoring.log")
)
SCORING_WINDOWS_PATH = Path(
    CONFIG.get(
        "scoring_windows_path",
        SCORING_HOME / "latest_scoring_windows.csv",
    )
)
SCORING_FEATURE_RESIDUALS_PATH = Path(
    CONFIG.get(
        "scoring_feature_residuals_path",
        SCORING_HOME / "latest_feature_residuals.csv",
    )
)
SCORING_LATENT_PATH = Path(
    CONFIG.get(
        "scoring_latent_path",
        SCORING_HOME / "latest_latent_vectors.csv",
    )
)
SCORING_ACTION_STATUS_PATH = Path(
    CONFIG.get(
        "scoring_action_status_path",
        SCORING_HOME / "control_cards.json",
    )
)
SCORING_CRON_FILE = Path(
    CONFIG.get("scoring_cron_file", "/etc/cron.d/dns-ml-scoring")
)
SCORING_CRON_DISABLED_PATH = Path(
    CONFIG.get(
        "scoring_cron_disabled_path",
        SCORING_HOME / "dns-ml-scoring.cron.disabled",
    )
)

# BEGIN DNS CATEGORIZATION ISOLATION SCORING SETTINGS
CATEGORIZATION_ALERT_JSONL_PATH = Path(
    CONFIG.get(
        "categorization_alert_jsonl_path",
        "/data/dns-ml/alerts/dns_categorization_forest_alerts.jsonl",
    )
)
IF_MODEL_PATH = Path(CONFIG.get("if_model_path", "/data/dns-ml/models/dns_isolation_forest.joblib"))
IF_SCALER_PATH = Path(CONFIG.get("if_scaler_path", "/data/dns-ml/models/dns_isolation_forest_scaler.joblib"))
IF_CONFIG_PATH = Path(CONFIG.get("if_config_path", "/data/dns-ml/models/dns_isolation_forest_config.json"))
IF_SCORING_HOME = Path(CONFIG.get("if_scoring_home", "/data/dns-ml/isolation_scoring"))
IF_SCORING_SCRIPT_PATH = Path(CONFIG.get("if_scoring_script_path", "/opt/dns-ml/score_dns_isolation_forest_live.py"))
IF_SCORING_WRAPPER_PATH = Path(CONFIG.get("if_scoring_wrapper_path", "/opt/dns-ml-gui/scripts/run_dns_isolation_scoring_once.sh"))
IF_SCORING_STATUS_PATH = Path(CONFIG.get("if_scoring_status_path", IF_SCORING_HOME / "status.json"))
IF_SCORING_PID_PATH = Path(CONFIG.get("if_scoring_pid_path", IF_SCORING_HOME / "scoring.pid"))
IF_SCORING_LIVE_LOG_PATH = Path(CONFIG.get("if_scoring_live_log_path", IF_SCORING_HOME / "live_scoring.log"))
IF_SCORING_WINDOWS_PATH = Path(CONFIG.get("if_scoring_windows_path", IF_SCORING_HOME / "latest_isolation_scoring_windows.csv"))
IF_SCORING_ACTION_STATUS_PATH = Path(CONFIG.get("if_scoring_action_status_path", IF_SCORING_HOME / "control_cards.json"))
IF_SCORING_STATE_PATH = Path(CONFIG.get("if_scoring_state_path", "/data/dns-ml/state/dns_isolation_forest_log_state.json"))
IF_SCORING_BUFFER_PATH = Path(CONFIG.get("if_scoring_buffer_path", "/data/dns-ml/state/dns_isolation_forest_event_buffer.jsonl"))
IF_SCORING_CRON_FILE = Path(CONFIG.get("if_scoring_cron_file", "/etc/cron.d/dns-ml-isolation-scoring"))
IF_SCORING_CRON_DISABLED_PATH = Path(CONFIG.get("if_scoring_cron_disabled_path", IF_SCORING_HOME / "dns-ml-isolation-scoring.cron.disabled"))
IF_SCORING_CRON_LAST_PATH = Path(CONFIG.get("if_scoring_cron_last_path", IF_SCORING_HOME / "last_cron_run.json"))
# END DNS CATEGORIZATION ISOLATION SCORING SETTINGS
