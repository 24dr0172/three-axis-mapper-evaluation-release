# Mapper Evaluation — Final Reproducibility Release

This is the v4.1 scientific reproducibility package for **Evaluating Mapper: A Three-Axis Critical Review**, covering stability, local cluster quality, and topological agreement. It contains definitions, an installable `mapper_framework` package, contracts, row-level evidence, scientific recomputation checks, construction-level replay checks, deterministic figure generation, v4.1 derived analyses, and the FCM solver-sensitivity audit. The publication manuscript and bibliography are distributed separately; `manuscript/README.md` provides the protocol crosswalk, while generated figures and plotted-data metadata are under `manuscript/figures/`. Third-party reference PDFs are intentionally not bundled.

The authors are the principal investigators who conceived the study, designed the evaluation study and analyses, and directed the research programme. AI tools were used for drafting assistance, code review, discrepancy detection, audit assistance, cross-checking the code, verifying numerical reproducibility, and validating the release structure. The authors approved every scientific decision and independently verified the evidence outputs reported here. No public archive DOI is claimed; none has been issued.

**Version convention.** The reproducibility release is version **4.1.0**. The embedded installable `mapper_framework` package/API remains version **3.0.0**. These numbers refer to different objects: release revisions can add audits, derived summaries, figures, and documentation without changing the public package API.

## Install

The portable verification environment is **CPython 3.12.x** with exact package versions recorded in `environment/requirements.txt`. The verification entrypoints fail closed if the interpreter or any required distribution differs.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r environment/requirements.txt
python -m pip install -e . --no-deps
```

Check the environment explicitly with:

```bash
python -B code/verification/verify_environment.py
```

## Fast verification

For integrity, claim/evidence checks, unit tests, representative construction replay, and exact manuscript-figure-data regeneration without the expensive corrected Axis-I campaign replay:

```bash
bash code/verification/run_quick.sh
```

A successful run ends with:

```text
QUICK_RELEASE_CHECKS_PASSED
```

The v4.1 revision adds derived analyses, the FCM solver-sensitivity audit, and claim-tier language. Verify that additive layer with:

```bash
PYTHONPATH=src python3 -B revision_v4_1/verify_release_v4_1.py
```

A successful run ends with `V4_1_RELEASE_CHECKS_PASSED`. `revision_v4_1/CHANGELOG_V4_1.txt` lists disclosure changes, derived analyses, and robustness runs. PDF compilation is not required on hosts without `pdflatex`.

## Full scientific replay

For the same checks **plus the full corrected Axis-I replay**:

```bash
bash code/verification/run_all.sh
```

A successful run ends with:

```text
ALL_RELEASE_CHECKS_PASSED
```

The full command is intentionally more expensive because it reconstructs the released corrected Axis-I campaigns before comparing the regenerated manuscript outputs.

## Regenerate manuscript outputs and keep them

```bash
python -B code/reproduction/regenerate_all.py \
  --output-dir ../three_axis_mapper_final_release_reproduced
```

This creates fresh PDF/SVG/PNG figures, `FIGURE_DATA.json`, a complete claim-verification transcript, a full corrected Axis-I replay under `axis1_corrected/`, the core construction replay report, and a SHA-256 regeneration report. It never overwrites released evidence.

## Publication-package boundary

The manuscript source and bibliography are distributed with the standalone submission. This release does not depend on that sibling package. Its `manuscript/` directory contains the compact protocol crosswalk and generated figure assets with their data metadata.

## Evaluation quantities represented in the release

The package keeps distinct quantities that answer different questions. Observation-level common-cover distances require corresponding observations and a shared indexed cover. Local Silhouette and Davies--Bouldin values are computed only where the required pullback partition exists and are interpreted together with construction status, coverage, DBSCAN noise/retention, and the number of eligible pullbacks. Reference-based topological comparisons require a justified reference on the same domain and scalar lens; the finite graph filtration used for extended persistence is stated explicitly. Ball Mapper is treated through lens-free structural comparisons rather than being forced into a scalar-Reeb reference comparison.

## Reproducibility scope

The release recomputes the manuscript-level numerical claims and figures from the included evidence and reruns representative core constructions from released inputs, parameters, and seeds. The released row-level evidence is the authority for campaign-level analyses; representative construction replay and evidence-ledger recomputation are identified separately.

Use the row-level evidence, contracts, protocol crosswalk, and verification entrypoints in this package when auditing scientific results. The publication manuscript and bibliography are shipped with the standalone submission. `SHA256SUMS.txt` is the sole release-integrity authority for this release tree.

## Citation and licensing

`CITATION.cff` contains package citation metadata. `LICENSE` records the copyright/licensing status of the release; it does not silently impose an open-source license on the authors.
