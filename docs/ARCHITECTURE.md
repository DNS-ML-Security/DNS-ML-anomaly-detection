# Version 1 architecture

## Processing flow

```text
Zeek DNS JSON
    |
    v
Five-minute feature windows by source IP
    |
    +--> Autoencoder scoring --------+
    +--> Isolation Forest scoring ---+--> Final correlation --> GUI alerts
    +--> DNS heuristic scoring ------+
```

DNS training input can first be prepared through an isolated PCAP-conversion
job or a persistent live learning capture. Both paths produce one consolidated
JSON `dns.log` before the existing merge and feature-extraction workflow.

## Components

| Component | Responsibility | Primary location |
|---|---|---|
| Zeek | Authorized DNS telemetry capture | `/opt/zeek` |
| ML engine | Feature extraction, training, and scoring | `/opt/dns-ml` |
| GUI | Learning, scoring, alerts, authentication, and administration | `/opt/dns-ml-gui` |
| Runtime data | Models, state, alerts, uploads, and results | `/data/dns-ml` |
| GUI service | Streamlit lifecycle | `dns-ml-gui.service` |
| Learning-capture service | Persistent PCAP conversion, live capture, hourly rotation, restart recovery, 10 GiB enforcement, and DNS consolidation | `dns-ml-learning-capture.service` |
| Scheduler | Five-minute detector and correlation jobs | `/etc/cron.d/dns-ml-*` |

## Capture-interface policy

Installation does not select or validate a network interface. Capture is
configured after installation through the GUI. Runtime selection permits an
addressless passive interface, provided it is operational, non-loopback, and
not the management/default-route interface.

## Preparation and scoring exclusion

The DNS Log Preparation lifecycle and managed Live Zeek scoring capture use a
shared exclusion policy. PCAP upload/processing and live learning states
(`active`, `paused`, or `finalizing`) block scoring-interface activation.
Conversely, an active, paused, or scheduled continuous-scoring lifecycle blocks
new preparation jobs. PCAP processing and live learning capture write only
under `/data/dns-ml/dns_preparation`; they never publish training traffic into
the scoring directory `/opt/zeek/logs/current`.

## Trust boundaries

- The management interface and GUI are administrative trust boundaries.
- Zeek input is untrusted and must be treated as hostile data.
- Uploaded files are untrusted and are limited to approved PCAP/PCAPNG inputs
  or supported Zeek JSON formats.
- Model and alert artifacts must remain local unless an approved integration is
  explicitly configured.
- The authentication database contains password hashes and must never enter
  source control or a release archive.
