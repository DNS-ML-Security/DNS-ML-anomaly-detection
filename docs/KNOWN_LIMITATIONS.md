# Known limitations — v1.0.0-rc.1

1. The release candidate has not yet completed clean Ubuntu Server acceptance.
2. The GUI service currently runs with elevated privileges because it modifies
   Zeek configuration and `/etc/cron.d` runtime schedules. The systemd unit
   applies filesystem and process restrictions, but a dedicated privileged
   helper should be evaluated before production approval.
3. The GUI is served over HTTP on port 8779. TLS termination is not included.
4. Python dependencies are constrained only by the supplied requirements and
   are not yet locked with hashes.
5. Training data and trained models are intentionally excluded. Each new
   installation must complete approved training before scoring is enabled.
6. Capture-interface selection is intentionally deferred to the GUI and must be
   validated operationally by the administrator.
7. Legal, employer intellectual-property, and third-party-license reviews are
   pending.
8. The formal solution-design document and sanitized GUI screenshot package
   are pending completion.
9. The learning-capture supervisor runs as root to open an authorized passive
   capture interface. Its unit is hardened and writes only to the preparation
   runtime directory, but production approval should evaluate Linux capture
   capabilities or a dedicated unprivileged capture account.
10. Browser download is limited to consolidated `dns.log` files of 256 MiB or
    less to avoid excessive Streamlit memory use. Larger outputs remain at the
    protected server path for an approved administrative transfer.
