# Security policy

## Supported versions

`v1.0.0-rc.1` is a private release candidate for controlled validation. It is
not yet approved for production or public distribution.

## Reporting a vulnerability

Do not disclose suspected vulnerabilities, credentials, internal addresses,
captured traffic, or customer information in a public issue.

Report security concerns privately to:

- Ahmed Mekky
- ahmedmekkyf13@gmail.com

Include the affected version, component, reproduction conditions, potential
impact, and whether operational or sensitive data may be involved. Do not send
real DNS logs, packet captures, credentials, or personal information unless a
secure transfer method has been agreed.

## Deployment expectations

- Run only on an authorized monitoring network.
- Restrict GUI port `8779` to approved management sources.
- Place TLS termination or an authenticated reverse proxy in front of the GUI
  before production exposure.
- Protect `/data/dns-ml/gui_auth` and all model, alert, and training data.
- Never commit runtime databases, logs, captures, credentials, or model files.
- Review the known limitations before approval.

