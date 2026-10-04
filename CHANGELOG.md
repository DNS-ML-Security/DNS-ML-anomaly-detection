# Changelog

All notable changes to the approved source line are recorded here.

## [v1.0.0-rc.1] - 2026-10-01

### Added

- Version 1 release-candidate source baseline.
- Autoencoder, Isolation Forest, DNS heuristic, and final-correlation scoring.
- Zeek JSON ingestion, feature extraction, training, and live-scoring controls.
- Local GUI authentication with administrator, user, and SOC analyst roles.
- Source-only collection, private-address sanitization, and integrity records.
- Draft ownership, responsible-use, security, and contribution policies.
- Fresh Ubuntu Server installer candidate.
- Compact Live Zeek scoring radar beside Runtime controls, with a rotating
  ring, radar sweep, and green health pulse while managed capture and both ML
  workers are healthy; paused, inactive, and degraded states remain static.
- Second-page DNS Log Preparation workflow for PCAP-to-JSON-dns.log conversion
  and persistent live learning capture with preconfigured periods, suspend and
  resume controls, hourly rotation, reboot recovery, automatic consolidation,
  mutual exclusion from Live Zeek scoring capture, and a 10 GiB safety limit.
- Extended the selectable live-learning periods to 14, 21, and 30 days, with
  30 days enforced as the maximum configured period.
- Added a compact global GUI footer with the project ownership, GitHub
  repository link, source-license name, and modification-responsibility notice.
- Expanded the README solution overview, prepared direct public-repository
  installation instructions without GitHub authentication, and added the
  Version 2 roadmap for additional AI/ML models and optional GPU support.

### Security

- Removed runtime data, authentication databases, models, logs, captures,
  credentials, private keys, backups, and internal address literals.
- Replaced POC private addresses with documentation-only TEST-NET addresses.

### Fixed

- Treat an empty live Zeek interval as a healthy waiting state instead of an
  Isolation Forest or DNS Heuristics failure; scheduled scorers retry
  automatically when Zeek publishes DNS records.
- Added visible staged progress for starting, pausing, and resuming managed
  Zeek capture and detector schedules.
- Replaced legacy single-color linear progress bars with a shared animated,
  color-changing bar that always displays its percentage.
- Restored authenticated browser sessions after refresh by rebuilding the
  browser-cookie controller from its stable component state on every run.
- Added the DNS Console logo and a dedicated SOC login-card presentation.
- Prevented the DNS Scoring page from overriding the application background,
  navigation-button styling, and shared page theme.

### Known limitations

- Clean-VM installation and full acceptance testing remain pending.
- The GUI service currently requires elevated privileges for Zeek lifecycle and
  `/etc/cron.d` management; this requires focused security review.
- Dependency versions are not yet fully locked with hashes.
- Legal and employer intellectual-property approval remain pending.
