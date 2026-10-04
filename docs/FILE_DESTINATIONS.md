# File destinations

## Installed source

| Source package | Installed destination | Purpose |
|---|---|---|
| `src/dns-ml/*` | `/opt/dns-ml/` | ML, feature, training, scoring, and Zeek utilities |
| `src/dns-ml-gui/*` | `/opt/dns-ml-gui/` | Streamlit GUI and controller code |
| `packaging/systemd/dns-ml-gui.service` | `/etc/systemd/system/dns-ml-gui.service` | GUI service |
| `packaging/systemd/dns-ml-learning-capture.service` | `/etc/systemd/system/dns-ml-learning-capture.service` | Persistent DNS-log preparation service |

## Runtime data

| Path | Purpose | Source controlled? |
|---|---|---|
| `/data/dns-ml/gui_auth/users.db` | Local users and authentication audit | No |
| `/data/dns-ml/gui/dns_gui.db` | GUI alert database | No |
| `/data/dns-ml/training_upload/` | Temporary uploaded training inputs | No |
| `/data/dns-ml/dns_preparation/state.json` | Persistent preparation and reboot-recovery state | No |
| `/data/dns-ml/dns_preparation/pcap/` | Protected temporary PCAP inputs and isolated conversion jobs | No |
| `/data/dns-ml/dns_preparation/live/` | Hourly live learning-capture segments | No |
| `/data/dns-ml/dns_preparation/output/` | Consolidated JSON `dns.log` outputs | No |
| `/data/dns-ml/features/` | Precomputed training feature windows | No |
| `/data/dns-ml/models/` | Trained model and scaler artifacts | No |
| `/data/dns-ml/runs/` | Training status, results, and logs | No |
| `/data/dns-ml/scoring/` | Autoencoder scoring state and outputs | No |
| `/data/dns-ml/isolation_scoring/` | Isolation Forest scoring outputs | No |
| `/data/dns-ml/heuristics_scoring/` | Heuristic scoring outputs | No |
| `/data/dns-ml/final_scoring/` | Correlation state and output | No |
| `/data/dns-ml/alerts/` | Detector and correlated alert streams | No |
| `/data/dns-ml/state/` | Cursors and event buffers | No |
| `/data/dns-ml/zeek/` | Merged and summarized Zeek data | No |
| `/opt/zeek/logs/current/` | Live Zeek logs | No |

## Scheduled controls

The GUI creates and removes the following files when live scoring is activated,
paused, resumed, or deactivated:

- `/etc/cron.d/dns-ml-scoring`
- `/etc/cron.d/dns-ml-isolation-scoring`
- `/etc/cron.d/dns-ml-heuristics-scoring`
- `/etc/cron.d/dns-ml-final-correlation`
