# Contracts

This directory contains the method, dataset, endpoint, and campaign contracts
needed to interpret the released ledgers.

- `FINAL_ENDPOINT_CONTRACT.txt` defines the release-wide endpoint boundary.
- Method specifications define Conventional, F-, Ball, and Kang–Lim RCESCC
  Ensemble Mapper with project-specified metric-medoid Silhouette hardening.
- Dataset and Axis-III specifications define inputs and fidelity rules.
- JSON campaign contracts govern C5–C9. Where a release copy differs from the
  executed digest, `CONTRACT_SOURCE_BINDINGS.json` records the mapping.
- `SPEC_SOURCE_BINDINGS.json` maps executed specification digests to the
  release copies.

Campaign 5 admits only finite-sample F-Mapper Tier-A structural comparison;
Tier B remains deferred and Tier C disabled. FCM centroid initialization is
deterministic (`numpy.linspace` over the observed lens range); `fcm_seed` is
not an experimental factor.
