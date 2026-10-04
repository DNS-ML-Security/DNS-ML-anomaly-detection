# Repository preparation report

## Source provenance

| Item | Value |
|---|---|
| Collected version | `v1.0.0-rc.1` |
| Collection time | 2026-10-01 11:09 UTC |
| Original archive | `DNS-ML-anomaly-detection_v1.0.0-rc.1_source.tar.gz` |
| Original archive SHA-256 | `d135efd661808c20d24678eefd8fac521a2a0e15c9864071c40c7f99a9ffba47` |
| Source environment | `/opt/dns-ml` and `/opt/dns-ml-gui` |
| Repository status | Private release candidate |

The outer archive checksum passed. Every application-source entry in the
collector manifest also passed. The original collection report was appended
after that internal manifest was generated, causing only the report's own
internal checksum to become stale. The obsolete internal manifest was removed
and is superseded by the release-candidate source-tree fingerprints.

## Repository preparation changes

- Removed obsolete operational handoff and source-export scripts.
- Added ownership, draft licensing, security, contribution, and third-party
  notice files.
- Added repository documentation, known limitations, and approval checklist.
- Added a fresh Ubuntu Server installer candidate.
- Added an automated release validation gate.
- Added copyright headers to first-party Python and shell source files.
- Added an authenticated About / Legal panel in the GUI.
- Removed placeholder approved-resolver addresses; the unconfigured default is
  now empty rather than treating documentation addresses as real resolvers.
- Added restrictive systemd sandbox settings while retaining the current
  privileged runtime required for Zeek and scheduler management.

## Automated results

- Archive traversal check: PASS
- Uploaded archive SHA-256: PASS
- Forbidden artifact scan: PASS
- Private network literal scan: PASS
- Recognizable token/private-key scan: PASS
- Shell syntax validation: PASS
- Python syntax validation: PASS

Clean-VM functional, security, legal, and management reviews remain pending.

