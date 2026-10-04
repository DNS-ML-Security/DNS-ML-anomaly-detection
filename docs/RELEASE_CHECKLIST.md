# Version 1 release checklist

## Source and security

- [x] Operational data excluded
- [x] Credentials, keys, tokens, and authentication databases excluded
- [x] Private network literals replaced with documentation addresses
- [x] Source syntax validated
- [x] Ownership and draft license files added
- [ ] Dependency lock and SBOM approved
- [ ] Privileged GUI design approved
- [ ] Independent security review completed

## Clean-VM acceptance

- [ ] Fresh Ubuntu Server 24.04 installation
- [ ] Only initial administrator password requested
- [ ] GUI login succeeds
- [ ] Services start after reboot
- [ ] PCAP/PCAPNG conversion produces a valid JSON `dns.log`
- [ ] Live learning capture suspends, resumes, rotates, and consolidates
- [ ] Active learning capture resumes safely after server reboot
- [ ] Learning capture stops and consolidates at the 10 GiB limit
- [ ] DNS preparation and Live Zeek scoring capture lock each other
- [ ] Autoencoder training succeeds
- [ ] Isolation Forest training succeeds
- [ ] Manual scoring succeeds
- [ ] Live Zeek capture succeeds
- [ ] Addressless capture interface succeeds
- [ ] ML and heuristic alerts correlate automatically
- [ ] Pause, resume, deactivate, and purge succeed
- [ ] Purge removes every detector and correlated alert
- [ ] SOC role sees alerts-only pages
- [ ] Second installer execution preserves data and users

## Approval

- [ ] Technical review
- [ ] Functional testing
- [ ] Security review
- [ ] Source-cleanliness review
- [ ] Documentation review
- [ ] Intellectual-property and legal review
- [ ] Management approval
- [ ] Final release fingerprints
- [ ] Final `v1.0.0` tag
