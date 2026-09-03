# Environment notes

Two recorded software families are shipped.

- `requirements.txt` is the portable final-release verification/reproduction environment: CPython 3.12.x with exact NumPy, SciPy, scikit-learn, joblib, threadpoolctl, narwhals, matplotlib, and GUDHI versions. `code/verification/verify_environment.py` checks these pins and fails closed before the release verification scripts run.
- `requirements_axis3_legacy.txt` is the exact software record associated with the Conventional III-1 and III-2 evidence (CPython 3.13.5, including NetworkX and GUDHI).
- `ENVIRONMENT_LOCK_MANIFEST.json` is the CPython 3.12.3 production lock bound by later campaign ledgers. It is shipped because those ledgers record its digest. A bundled site-packages tree is not included.

Create the portable environment explicitly with `python3.12`; do not rely on a system `python3` alias that may point to another minor version.

Adding GUDHI to the portable 3.12 environment makes the advertised persistence modules importable after the README install. It does not claim that every released row was produced under that combined portable pin; historical environment provenance remains recorded separately.
