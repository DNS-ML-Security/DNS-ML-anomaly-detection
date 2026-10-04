# DNS ML Anomaly Detection

DNS ML Anomaly Detection is an on-premises defensive cybersecurity platform
that prepares Zeek DNS telemetry, trains anomaly-detection models, evaluates
live or uploaded DNS activity, and presents correlated findings to SOC users.

The solution converts approved PCAP or live network capture into Zeek JSON
`dns.log`, creates five-minute behavioral windows per source IP, and evaluates
those windows with an Autoencoder, Isolation Forest, and DNS-specific
heuristics. The detector results are automatically consolidated into final
correlated alerts for investigation through the authenticated web interface.

Version 1 includes persistent learning capture, model-training controls, live
and one-shot scoring, role-based local authentication, alert investigation,
runtime health monitoring, and safe pause, resume, deactivate, and purge
operations. Processing and operational data remain on the deployed server.

![Demo Animation](DNS-ML-anomaly-detection.gif)

## Capabilities

- Imports newline-delimited Zeek DNS JSON records.
- Converts an uploaded PCAP/PCAPNG into a downloadable Zeek JSON `dns.log`.
- Runs a persistent, time-bounded live learning capture with suspend, resume,
  hourly rotation, reboot recovery, automatic consolidation, and a 10 GiB cap.
- Builds five-minute feature windows by source IP.
- Trains and evaluates an Autoencoder and Isolation Forest.
- Applies DNS heuristic rules in parallel with ML scoring.
- Automatically calculates final correlated alerts.
- Supports manual sample scoring and live Zeek capture workflows.
- Provides local authentication and administrator, standard-user, and
  alerts-only SOC roles.
- Supports activation, pause, resume, deactivation, and runtime purge controls.

## Installation candidate

### Public GitHub installation

After the repository visibility is changed to public, a fresh Ubuntu Server
24.04 VM can install Git, clone the repository without GitHub authentication,
and run the installer:

```bash
sudo apt update
sudo apt install -y git

git clone https://github.com/DNS-ML-Security/DNS-ML-anomaly-detection.git
cd DNS-ML-anomaly-detection

sudo bash install_dns_ml_anomaly_detection.sh
```

No GitHub account, GitHub CLI installation, web authorization, personal access
token, or repository credential is required after the repository becomes
public. The DNS ML application does not request or store GitHub credentials.

The clone command creates a directory named exactly
`DNS-ML-anomaly-detection`. Do not use `dns_ml_anomaly_detection`. If the
repository has already been cloned, do not run the clone command again. Resume
with:

```bash
cd DNS-ML-anomaly-detection
git pull --ff-only
sudo bash install_dns_ml_anomaly_detection.sh
```

Ubuntu may be running `unattended-upgrades` during installation. The installer
waits safely for the `apt`/`dpkg` lock for up to 15 minutes. Do not delete lock
files or terminate the automatic-update process.

If an earlier release-candidate run stopped with a Zeek `NO_PUBKEY` or
unreadable-key warning, pull the latest `main` branch and rerun the installer.
It repairs the legacy repository files and installs the Zeek signing key under
`/etc/apt/keyrings` with permissions readable by Ubuntu's `_apt` user.

After the repository is cloned, the installer requests only the initial local
GUI administrator password. The initial GUI username is `admin`.

The only application configuration requested during a fresh installation is:

```text
Enter the initial GUI administrator password:
Confirm the initial GUI administrator password:
```

Any non-empty matching password is accepted. The installer stores only a
salted PBKDF2 password hash in the local authentication database.

The installer does not inspect, select, or configure a Zeek capture interface.
After installation, an administrator configures capture through the GUI. The
selected interface may be addressless, but it must be operational,
non-loopback, and different from the management/default-route interface.

## Access

After installation:

```text
http://SERVER_ADDRESS:8779
```

Restrict this port to authorized management sources. Production approval
requires TLS termination or an authenticated reverse proxy.

## Validation

Run the repository checks before committing or packaging:

```bash
bash tools/validate_release_candidate.sh
```

Clean-VM acceptance must validate installation, login, reboot persistence,
training, manual scoring, live capture, alert correlation, runtime controls,
purge behavior, roles, and a second installer execution.

## Release status

| Item | Value |
|---|---|
| Version | `v1.0.0-rc.1` |
| Distribution | Public source-available repository after the planned visibility change |
| Approval | Pending clean-VM, security, legal, and management review |
| Supported validation OS | Ubuntu Server 24.04 LTS |
| GUI port | `8779/tcp` |
| Initial GUI username | `admin` |

Use, modification, and
redistribution remain governed by `LICENSE`, `NOTICE`, and the required legal
and intellectual-property approvals.

## Workflows

### 1. Fresh installation

1. Clone the public `DNS-ML-anomaly-detection` repository directly with Git.
2. Enter the cloned `DNS-ML-anomaly-detection` directory.
3. Run `sudo bash install_dns_ml_anomaly_detection.sh`.
4. Enter and confirm the initial `admin` GUI password.
5. The installer deploys source, creates the virtual environment, installs
   dependencies, initializes authentication, installs systemd, and starts the
   GUI on port `8779`.
6. Configure the authorized Zeek capture interface later through the GUI.

### 2. Prepare a Zeek JSON dns.log

Use the second GUI page, **DNS Log Preparation**, and select one workflow:

1. Upload an approved `.pcap` or `.pcapng`; the persistent service processes it
   with Zeek and provides the corresponding JSON `dns.log`.
2. Select a dedicated capture interface and a preconfigured learning period,
   from 1 hour through a maximum of 30 days, then activate live capture.
   Suspend and resume preserve the job. Stop, time completion, or the 10 GiB
   safety cap consolidates all hourly DNS segments.
3. Active or suspended preparation survives GUI logout and server reboot.
4. Live Zeek scoring capture is locked until preparation finishes or the live
   learning capture is stopped and consolidated.

### 3. Training and model finalization

1. Upload approved Zeek DNS JSON/JSONL training files or the prepared JSON
   `dns.log`.
2. Merge, normalize, deduplicate, and validate the records.
3. Extract the shared 36-feature, five-minute windows by source IP.
4. Train the Autoencoder and Isolation Forest from the same finalized feature
   dataset.
5. Verify both model artifacts and training results.
6. Finalize the learning lifecycle to unlock DNS scoring.

Feature engineering and window aggregation are performed once. Each trainer
consumes `/data/dns-ml/features/dns_training_features.csv` and does not repeat
those operations.

### 4. Live scoring and automatic correlation

1. Select an operational, non-loopback capture interface that is different
   from the management/default-route interface; an IP address is not required.
2. Activate live scoring from **DNS Scoring → Runtime controls**.
3. Zeek publishes DNS JSON under `/opt/zeek/logs/current/`.
4. Autoencoder, Isolation Forest, and DNS heuristics evaluate the same current
   DNS input.
5. Final correlation runs after detector evaluation and produces the unified
   final alerts automatically.
6. The GUI refreshes runtime health, detector outputs, and correlated alerts.

Pause retains runtime data while stopping scheduled evaluation. Resume
restores the schedules. **Deactivate and purge** stops live operation and
removes generated detector and final-correlation alerts and runtime state;
trained models and finalized training data are retained.

### 5. Manual one-shot scoring

1. Keep live capture inactive.
2. Provide the supported Zeek DNS JSON sample through the scoring interface.
3. Start one-shot scoring.
4. The three detector families evaluate the staged sample.
5. Final correlation runs automatically after detector completion.
6. Review each native alert tab and **Final Correlated Alerts**.

### 6. Authentication and roles

1. The initial local administrator signs in with username `admin`.
2. The administrator can create administrators, standard users, and SOC
   analysts.
3. SOC analysts are limited to Alerts, Alert Details, and Source IP
   Investigation.
4. A normal page refresh restores the active browser session using an opaque
   token; only its SHA-256 hash is stored locally.
5. Sign-out, account disabling, password reset, or session expiry revokes the
   corresponding session.

### 7. Repository update and repeat installation

```bash
cd DNS-ML-anomaly-detection
git pull --ff-only
sha256sum --check release/SOURCE_TREE.sha256
sudo bash install_dns_ml_anomaly_detection.sh
```

An existing authentication database and GUI configuration are preserved. The
installer refreshes approved application source, dependencies, service files,
and the running GUI.

### 8. Release validation

```bash
sha256sum --check release/SOURCE_TREE.sha256
bash tools/validate_release_candidate.sh
```

Version 1 approval additionally requires clean-VM installation, reboot,
training, manual scoring, live capture, role restriction, alert purge,
reinstallation, security, legal, and management testing.

## Repository structure

```text
DNS-ML-anomaly-detection/
├── src/dns-ml/                 ML, feature, Zeek, and scoring engine
├── src/dns-ml-gui/             Streamlit GUI and controller modules
├── packaging/systemd/          GUI and persistent capture service definitions
├── tools/                      Release and validation utilities
├── docs/                       Architecture, paths, release, and approval records
├── install_dns_ml_anomaly_detection.sh
├── LICENSE
├── NOTICE
├── AUTHORS.md
├── SECURITY.md
├── CONTRIBUTIONS.md
├── CHANGELOG.md
└── VERSION
```

## Software packages and versions

The release candidate installs the following platform and application
dependencies. A value of **installer-resolved** means that the current
requirements file does not pin an exact version; `pip` or Ubuntu resolves the
compatible version available at installation time.

| Package or platform | Version or constraint | Purpose |
|---|---|---|
| Ubuntu Server | `24.04 LTS` | Supported operating system for Version 1 validation |
| Python | Ubuntu 24.04 system `python3` (`3.12` series) | Engine, GUI, training, scoring, and administration runtime |
| Zeek | `8.0 LTS` | Authorized DNS telemetry capture and JSON log generation |
| Streamlit | Installer-resolved; current POC uses `1.60.0` | Web GUI on `8779/tcp` |
| streamlit-cookies-controller | `0.0.4` | Browser session restoration after a normal page refresh |
| pandas | Installer-resolved | Tabular input, features, scoring outputs, and GUI tables |
| NumPy | Installer-resolved | Numerical feature and model operations |
| scikit-learn | Installer-resolved | Isolation Forest, scaling, PCA, and ML metrics |
| TensorFlow CPU | Installer-resolved | Autoencoder training and inference without GPU discovery |
| Joblib | Installer-resolved | Model and scaler serialization |
| SHAP | `>=0.46` | Isolation Forest explanation outputs |
| Altair | Installer-resolved | GUI charts |
| Plotly | `>=5.20` | Interactive GUI visualizations |
| `ca-certificates`, `cron`, `curl`, `gnupg`, `iproute2`, `jq` | Ubuntu repository versions | Installation, scheduling, repository trust, interface inventory, and diagnostics |
| `libgomp1`, `libpcap0.8` | Ubuntu repository versions | ML parallel runtime and packet-capture support |
| `python3-pip`, `python3-venv` | Ubuntu repository versions | Isolated Python environment and dependency installation |

The authoritative dependency declarations are
`src/dns-ml/requirements.txt` and `src/dns-ml-gui/requirements.txt`. Record the
exact versions installed on an acceptance VM with:

```bash
/opt/dns-ml/venv/bin/python --version
/opt/zeek/bin/zeek --version
/opt/dns-ml/venv/bin/python -m pip freeze | sort
```

Third-party packages remain the property of their respective owners and are
governed by their own licenses. See `THIRD_PARTY_NOTICES.md`.

## Server physical requirements

The following sizing is intended for the Version 1 on-premises deployment.
Actual capacity must be adjusted for DNS event rate, log-retention period,
training-dataset size, concurrent GUI users, and the monitored link speed.

| Resource | Minimum POC | Recommended deployment | Notes |
|---|---:|---:|---|
| CPU architecture | `x86_64/amd64` | `x86_64/amd64` | Ubuntu Server 24.04 LTS and Zeek 8.0 compatible processor |
| CPU | 4 physical or virtual cores | 8 or more cores | Training and the three scoring engines are CPU-based and benefit from additional cores |
| Memory | 16 GB RAM | 32 GB RAM | 32 GB is recommended when training both models and retaining large scoring tables |
| System and application storage | 80 GB SSD | 150 GB or larger SSD/NVMe | Includes Ubuntu, Zeek, Python environment, application source, models, logs, and runtime data |
| Free space before installation | 40 GB | 100 GB or more | Additional capacity is required when retaining long Zeek histories or large training datasets |
| Management network interface | 1 × 1 GbE | 1 × 1 GbE or faster | Used for SSH, GUI access, GitHub, Ubuntu repositories, and administration |
| Dedicated capture interface | 1 separate NIC | 1 separate NIC sized for the monitored link | Receives authorized SPAN, mirror, or network-TAP traffic; it may be addressless |
| Capture-link capacity | Match the monitored DNS traffic | Match or exceed the mirrored link rate | Use 10 GbE capture hardware when monitoring traffic that can exceed 1 GbE |
| GPU | Not required | Not required | Autoencoder and Isolation Forest operation is configured for CPU execution |
| Time synchronization | Required | NTP-synchronized | Accurate timestamps are required for five-minute windows and final correlation |
| Power and resilience | Standard POC availability | Redundant power and protected storage where required | Select according to organizational availability and recovery requirements |

The capture NIC must be operational, non-loopback, and different from the
management/default-route interface. It does not require an assigned IP
address. The installer deliberately does not inspect or select interfaces;
the authorized capture interface is configured later through the GUI.

For virtual deployment, reserve the recommended CPU and memory instead of
allowing heavy overcommit, use SSD-backed virtual disks, and attach the capture
adapter to the approved mirror or monitoring network in promiscuous mode where
the virtualization platform requires it.

## Complete project file inventory

This inventory covers the complete Version 1 repository, including the demo
animation supplied with the source repository.

| Project location | Files | Responsibility |
|---|---|---|
| Repository root | `.gitignore`<br>`AUTHORS.md`<br>`CHANGELOG.md`<br>`CONTRIBUTIONS.md`<br>`LICENSE`<br>`NOTICE`<br>`README.md`<br>`SECURITY.md`<br>`THIRD_PARTY_NOTICES.md`<br>`VERSION` | Source-control exclusions, ownership, release history, security, licensing, and primary documentation |
| Repository media | `DNS-ML-anomaly-detection.gif` | README demonstration animation |
| Installation | `install_dns_ml_anomaly_detection.sh` | One-command Ubuntu installation and repeatable upgrade entry point |
| Architecture and release documents | `docs/APPROVAL_RECORD.md`<br>`docs/ARCHITECTURE.md`<br>`docs/FILE_DESTINATIONS.md`<br>`docs/KNOWN_LIMITATIONS.md`<br>`docs/RELEASE_CHECKLIST.md` | Design, approval, destination, limitation, and checklist documentation |
| Service packaging | `packaging/systemd/dns-ml-gui.service`<br>`packaging/systemd/dns-ml-learning-capture.service` | Streamlit GUI and persistent DNS preparation systemd units |
| Release evidence | `release/EXCLUDED_ARTIFACTS.txt`<br>`release/RELEASE_RECORD_TEMPLATE.md`<br>`release/REPOSITORY_PREPARATION_REPORT.md`<br>`release/SANITIZED_PRIVATE_ADDRESSES.txt`<br>`release/SOURCE_COLLECTION_REPORT.txt`<br>`release/SOURCE_TREE.sha256` | Sanitization, release approval, collection evidence, and source fingerprints |
| Release validation | `tools/validate_release_candidate.sh` | Required-file, secret, private-address, shell, and Python checks |
| ML engine | `src/dns-ml/extract_dns_training_features.py`<br>`src/dns-ml/feature_utils.py`<br>`src/dns-ml/generate_dns_test_data.py`<br>`src/dns-ml/requirements.txt`<br>`src/dns-ml/score_dns_autoencoder_live.py`<br>`src/dns-ml/score_dns_heuristics_live.py`<br>`src/dns-ml/score_dns_isolation_forest_live.py`<br>`src/dns-ml/train_dns_autoencoder.py`<br>`src/dns-ml/train_dns_isolation_forest.py`<br>`src/dns-ml/zeek_log_manager.py` | Feature extraction, shared features, synthetic test generation, model training, detector scoring, and Zeek preparation |
| GUI entry and configuration | `src/dns-ml-gui/app.py`<br>`src/dns-ml-gui/config.json`<br>`src/dns-ml-gui/requirements.txt`<br>`src/dns-ml-gui/.streamlit/config.toml`<br>`src/dns-ml-gui/config/training_input.json` | GUI entry point, paths, dependencies, Streamlit settings, and training-input template |
| GUI assets and documentation | `src/dns-ml-gui/assets/dns_console_logo.png`<br>`src/dns-ml-gui/docs/ML_TRAINING_LIFECYCLE.md`<br>`src/dns-ml-gui/docs/TRAINING_DATA_INPUT_OUTPUT.md` | Product identity and operator guidance |
| GUI command wrappers | `src/dns-ml-gui/scripts/ingest_alerts.py`<br>`src/dns-ml-gui/scripts/init_db.py`<br>`src/dns-ml-gui/scripts/run_dns_final_correlation.py`<br>`src/dns-ml-gui/scripts/run_dns_heuristics_scoring_once.sh`<br>`src/dns-ml-gui/scripts/run_dns_isolation_scoring_once.sh`<br>`src/dns-ml-gui/scripts/run_dns_learning_capture_service.py`<br>`src/dns-ml-gui/scripts/run_dns_scoring_once.sh`<br>`src/dns-ml-gui/scripts/run_gui.sh`<br>`src/dns-ml-gui/scripts/run_ingest.sh`<br>`src/dns-ml-gui/scripts/run_zeek_merge.sh` | Service startup, persistent DNS preparation, alert ingestion, detector execution, final correlation, and Zeek merge wrappers |
| GUI application modules | `src/dns-ml-gui/src/__init__.py`<br>`src/dns-ml-gui/src/db.py`<br>`src/dns-ml-gui/src/dns_log_preparation.py`<br>`src/dns-ml-gui/src/dns_log_preparation_page.py`<br>`src/dns-ml-gui/src/final_scoring.py`<br>`src/dns-ml-gui/src/heuristics_scoring.py`<br>`src/dns-ml-gui/src/ingest.py`<br>`src/dns-ml-gui/src/isolation_learning.py`<br>`src/dns-ml-gui/src/isolation_scoring.py`<br>`src/dns-ml-gui/src/learning.py`<br>`src/dns-ml-gui/src/local_auth.py`<br>`src/dns-ml-gui/src/scoring.py`<br>`src/dns-ml-gui/src/scoring_page_soc.py`<br>`src/dns-ml-gui/src/settings.py`<br>`src/dns-ml-gui/src/training_lifecycle.py`<br>`src/dns-ml-gui/src/training_upload.py`<br>`src/dns-ml-gui/src/ui_helpers.py`<br>`src/dns-ml-gui/src/zeek_manager.py` | Database, authentication, DNS-log preparation, learning, detector controls, scoring UI, upload lifecycle, correlation, settings, and Zeek management |

Runtime data, credentials, authentication databases, Zeek logs, models,
alerts, captures, and training datasets are deliberately excluded from this
inventory because they must never be committed to the repository.

## Installed and runtime file destinations

### Installed application and operating-system files

| Installed destination | Created from | Purpose |
|---|---|---|
| `/opt/dns-ml/` | `src/dns-ml/` | ML engine, training, scoring, features, and Zeek utilities |
| `/opt/dns-ml/venv/` | Installer-created Python virtual environment | Shared engine and GUI Python runtime |
| `/opt/dns-ml-gui/` | `src/dns-ml-gui/` | Streamlit GUI, controller modules, scripts, assets, and local configuration |
| `/etc/systemd/system/dns-ml-gui.service` | `packaging/systemd/dns-ml-gui.service` | GUI service definition |
| `/etc/systemd/system/dns-ml-learning-capture.service` | `packaging/systemd/dns-ml-learning-capture.service` | Persistent PCAP conversion and live learning-capture supervisor |
| `/etc/default/dns-ml-gui` | Installer-generated environment file | Product version, release classification, and build fingerprint |
| `/etc/apt/keyrings/security_zeek.gpg` | Zeek repository signing key | Verifies the Zeek Ubuntu repository |
| `/etc/apt/sources.list.d/security_zeek.list` | Installer-generated APT source | Zeek 8.0 package repository |
| `/etc/cron.d/dns-ml-scoring` | Created by live-scoring activation | Autoencoder scoring schedule |
| `/etc/cron.d/dns-ml-isolation-scoring` | Created by live-scoring activation | Isolation Forest scoring schedule |
| `/etc/cron.d/dns-ml-heuristics-scoring` | Created by live-scoring activation | DNS heuristic scoring schedule |
| `/etc/cron.d/dns-ml-final-correlation` | Created by live-scoring activation | Final correlation schedule |
| `/var/log/dns-ml-install.log` | Installer | Restricted installation audit and diagnostic log |

## Runtime inputs, models, outputs, and state

| Runtime destination | Data owner | Purpose |
|---|---|---|
| `/opt/zeek/logs/current/` | Zeek | Authoritative live DNS input directory |
| `/data/dns-ml/dns_preparation/state.json` | DNS preparation service | Persistent preparation mode, progress, duration, storage, and recovery state |
| `/data/dns-ml/dns_preparation/pcap/` | DNS preparation service | Protected temporary PCAP staging and isolated conversion jobs |
| `/data/dns-ml/dns_preparation/live/` | DNS preparation service | Protected hourly live-capture segments; deleted after successful consolidation |
| `/data/dns-ml/dns_preparation/output/` | DNS preparation service | Final consolidated JSON `dns.log` outputs |
| `/data/dns-ml/gui_auth/users.db` | Local authentication | Users, password hashes, roles, audit events, and revocable browser-session hashes |
| `/data/dns-ml/gui/dns_gui.db` | GUI | Alert database, rule configuration, and investigation records |
| `/data/dns-ml/training_upload/incoming/` | Training upload | Temporary incoming Zeek JSON/JSONL files |
| `/data/dns-ml/training_upload/files/` | Training upload | Accepted and normalized training inputs |
| `/data/dns-ml/zeek/merged_dns.jsonl` | Zeek manager | Deduplicated merged training input |
| `/data/dns-ml/zeek/stats.json`<br>`/data/dns-ml/zeek/source_ip_stats.csv` | Zeek manager | Dataset and source-IP statistics |
| `/data/dns-ml/zeek/merge_status.json`<br>`merge_manifest.json`<br>`merge.log`<br>`merge.pid` | Zeek manager | Merge progress, evidence, diagnostics, and process state |
| `/data/dns-ml/features/dns_training_features.csv` | Feature extractor | Shared 36-feature, five-minute training windows |
| `/data/dns-ml/features/dns_training_feature_summary.json` | Feature extractor | Feature-dataset summary and provenance |
| `/data/dns-ml/models/dns_autoencoder.keras`<br>`dns_scaler.joblib`<br>`dns_autoencoder_config.json` | Autoencoder training | Learned Autoencoder, scaler, and model configuration |
| `/data/dns-ml/models/dns_isolation_forest.joblib`<br>`dns_isolation_forest_scaler.joblib`<br>`dns_isolation_forest_pca.joblib`<br>`dns_isolation_forest_config.json` | Isolation Forest training | Learned forest, scaler, PCA, and configuration |
| `/data/dns-ml/models/backup/` | Training lifecycle | Recoverable prior model artifacts |
| `/data/dns-ml/runs/` | Both learning pipelines | Training status, PID, logs, result JSON, history, score, PCA, and SHAP CSV outputs |
| `/data/dns-ml/training_lifecycle/state.json` | Learning gate | Finalization state and model-readiness evidence |
| `/data/dns-ml/scoring/` | Autoencoder scoring | Status, lock, PID, live log, control cards, windows, residuals, and latent vectors |
| `/data/dns-ml/isolation_scoring/` | Isolation Forest scoring | Status, PID, live log, control cards, windows, and scheduler evidence |
| `/data/dns-ml/heuristics_scoring/` | DNS heuristics | Status, logs, control state, windows, and scheduler evidence |
| `/data/dns-ml/final_scoring/` | Final correlation | Correlation configuration, status, execution evidence, and final output |
| `/data/dns-ml/alerts/dns_alerts.jsonl` | Autoencoder | Native Autoencoder alerts |
| `/data/dns-ml/alerts/dns_categorization_forest_alerts.jsonl` | Isolation Forest | Native Isolation Forest alerts |
| `/data/dns-ml/alerts/` | Heuristics and correlation | Native heuristic and final correlated alert streams |
| `/data/dns-ml/state/` | Scoring engines | Incremental cursors and event buffers used to avoid reprocessing |

Core learning and scoring paths are centralized in
`src/dns-ml-gui/src/settings.py`; preparation paths are defined in
`src/dns-ml-gui/src/dns_log_preparation.py`. The expanded destination
reference is in `docs/FILE_DESTINATIONS.md`.

## Data policy

Do not commit Zeek logs, PCAPs, models, training data, generated alerts,
runtime state, authentication databases, credentials, private keys, tokens,
internal addresses, or sensitive screenshots.

## Roadmap: version 2 (future expansion)

- **More models:** Combine additional machine-learning algorithms with AI
  techniques.
- **GPU support:** Add optional GPU deployment for faster training and scoring.

## Legal and responsible use

Copyright © 2026 Ahmed Mekky. All rights reserved.

Use, modification, and redistribution are governed by `LICENSE` and `NOTICE`.
Complete the required legal and intellectual-property review before changing
repository visibility. The approved scope remains authorized defensive
cybersecurity evaluation, monitoring, anomaly detection, and security testing.

## Security limitations

Review `docs/KNOWN_LIMITATIONS.md` and `SECURITY.md` before deployment. In
particular, the GUI service currently performs privileged Zeek and scheduler
operations and therefore requires focused hardening review before production.
