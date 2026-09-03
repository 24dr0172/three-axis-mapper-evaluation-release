# Manuscript experiment map and protocol summary

This file is the manuscript-level index for the reviewer-ready release. It is intended to make the experimental protocol auditable without requiring the reader to reconstruct it from the code, ledgers, contracts, and figure metadata.

The mapping below answers three questions for every reported result:

1. Which paper section, table, or figure uses it?
2. Which campaign, ledger, contract, or derived artifact contains the result?
3. Which data, sampling rule, preprocessing rule, mapper settings, replicate structure, and uncertainty unit define it?

The compact summaries here are navigational. The JSON contracts and row-level ledgers linked below remain the authoritative specifications and evidence records. No result is imputed when its contract marks it undefined, ineligible, a coverage gap, or a solver failure.

All mathematical notation in this README uses ordinary Markdown and Unicode so it renders consistently in GitHub, local viewers, and repository browsers. The contracts and source manuscript remain the authoritative locations for full formal notation.

## How to navigate the release

- [Paper figures and source mapping](figures/README.md)
- [Experimental design table](../revision_v4_1/MASTER_EXPERIMENTAL_DESIGN_TABLE.csv)
- [Derived evidence registry](../revision_v4_1/DERIVED_EVIDENCE_REGISTRY.csv)
- [Campaign replay provenance](../revision_v4_1/derived/CAMPAIGN_REPLAY_PROVENANCE_V4_1.csv)
- [Environment preflight record](../revision_v4_1/derived/ENVIRONMENT_PREFLIGHT_V4_1.json)
- [Canonical contracts](../contracts/)
- [Executed campaign evidence](../evidence/)
- [Frozen and derived inputs](../data/)
- [Replay plans](../replay_inputs/)
- [Figure-generation code](../code/figures/generate_manuscript_figures.py)
- [Reproduction code](../code/reproduction/)
- [Verification scripts](../code/verification/)

For a quick local check from the release root:

```bash
bash code/verification/run_quick.sh
```

The full release is evidence-led: the released row-level ledgers are authoritative for campaign-level analyses. Representative construction replay and evidence-ledger recomputation are identified separately.

## Shared protocol conventions

### Canonical datasets and lenses

- **Circle.** `N = 1000` in the executed campaign configurations, sampled uniformly on `S¹`. The scalar lens is height, `circle_height_y`. The analytic same-filter Reeb reference is `β₀ = 1, β₁ = 1`.
- **Tripod.** `N = 1000` total points are sampled uniformly along the three line segments with endpoints `(-1, 1)`, `(1, 1)`, and `(0, -1)`. The implementation allocates `floor(N / 3)` points per arm and assigns the remainder to the first arms, giving `(334, 333, 333)` for `N = 1000`. The scalar lens is `tripod_height_y`. The exact piecewise-linear reference is `β₀ = 1, β₁ = 0`, with three leaves and one branch.
- **Swiss roll.** `φ(u, v) = (u cos u, v, u sin u)`, with `u ∈ [1.5π, 4.5π]` and `v ∈ [0, 21]`; the rectangular hole `u ∈ [2.5π, 3.5π]`, `v ∈ [7, 14]` is excised. The canonical release campaigns draw `N_base = 2500` base points, remove the hole points, and retain the realized post-excision sample size. Exact-size rejection sampling is an optional generator mode, not the mode used for the reported release campaigns. The radial lens is `radial_xz = sqrt(x² + z²)`. No intrinsic Swiss-roll Betti/Reeb target is asserted in the paper.
- **Digits.** The fixed `digits_1797x64_scaled16` release dataset has shape `1797 × 64`. The released PCA artifact is [N1_DIGITS_PCA_ARTIFACT.npz](../data/N1_DIGITS_PCA_ARTIFACT.npz); the lens is frozen signed PC1, `f(x) = (x − μ₀)ᵀ w₁`. PCA is fitted once on the clean artifact and never refitted on a perturbed realization.

### Preprocessing and perturbation

- Any learned transform, including PCA or standardization, is fitted on the clean `X₀` and then applied unchanged to noisy data. It is never refit after perturbation.
- Coordinate noise uses `Xσ,r = X₀ + N(0, σ² I_d)`, generated from the row-level `perturbation_seed`.
- When a contract specifies matched comparisons, the same clean sample, noise draw, filter perturbation, or landmark selection is reused across the compared factor levels.
- Noise label `−1` is excluded from node supports. Local quality is computed only on eligible pullbacks and retained non-noise observations; retention and noise are reported separately.

### Seeds and nesting

The ledgers distinguish the following namespaces:

- `data_seed`: clean data generation or sampling;
- `perturbation_seed`: coordinate noise or filter perturbation;
- `resample_seed`: subsampling indexes;
- `construction_seed`: internal method randomness or ordering;
- `selection_seed`: algorithmic selection, never shared with evaluation;
- `evaluation_seed`: bootstrap or analysis resampling.

The exact values are retained in the row-level ledgers. For sealed F-Mapper runs, `fcm_seed = 42` is retained for traceability but is inert because initialization is deterministic (`numpy.linspace`). FCM uses fuzzifier `m = 2`, `tol = 1e−7`, `max_iter = 300`, and deterministic linspace initialization unless a campaign entry below states otherwise.

Parameter-grid levels, noise doses, cover levels, and method labels are controlled factors, not independent replicate units. The replicate or resampling unit is stated campaign by campaign below; uncertainty procedures resample that unit rather than individual observations or overlapping pullbacks.

### Mapper and comparison rules

- **Conventional Mapper.** Closed regular interval covers, usually `n_intervals = 10` and `overlap = 0.30`; campaign-specific grids are listed below. DBSCAN uses the campaign-specific `eps` and `min_samples = 3`. The full-membership nerve 2-skeleton is reconstructed from retained supports.
- **F-Mapper.** Fuzzy cover parameters are explicitly recorded as `c` and `tau`; the standard grid is `c ∈ {5, 8, 10, 15, 20}` and `tau ∈ {0.10, 0.15, 0.20, 0.30, 0.40}`. Failures and coverage gaps remain in the denominator and are not silently retried or retuned.
- **Ball Mapper.** Uses the frozen landmark protocol and metric covers. It has no scalar-Reeb score and no DBSCAN step. The structure metric is edge Jaccard or retained-edge fraction.
- **Same-filter Reeb comparisons.** Only Circle and Tripod have intrinsic reference targets, using their height lenses. Swiss and Digits are not assigned an intrinsic Reeb target.
- **Graph versus nerve.** These are kept distinct; no clique completion is used. Topology statements are finite-sample statements tied to the released construction.
- **Common-cover distance.** `D_COMMON_ID` is used when both constructions have a common eligible cover. `D_M^NA` is the noise-aware extension when explicit outside-cover or DBSCAN-noise states occur. C6A uses ordinary `D_M`, so its coverage gaps make the primary endpoint undefined; C1 uses `D_M^NA`, which compares the same special states explicitly. Undefined comparisons remain typed undefined values.
- **Type-restricted Wasserstein companion.** For each of the four extended-persistence types, finite diagram points are matched with the `L∞` ground cost and diagonal cost `|death − birth| / 2`; the four typewise 1-Wasserstein costs are then summed. This is the manuscript quantity `W₁type`, not an untyped Wasserstein distance obtained by matching across persistence types.

### Quantities named in the figures

- **Retained fraction:** `r_N = |I_s| / |I_0|`, where `I_0` is the reference observation-ID set and `I_s` is the subsample. This is the controlled sampling level (0.50 or 0.80 in the fixed-cover experiment), not a performance score.
- **Joint common-cover distance:** the exact observation-level mismatch fraction minimized jointly over all allowed local label permutations. It is `D_M` for ordinary labels and `D_M^NA` when the explicit outside-cover or DBSCAN-noise states occur. It ranges from 0 to 1; larger values mean more observation assignments remain unmatched. “Joint” does not mean an average of separately optimized pullback distances.
- **Retained-edge fraction:** `r_E = |E_0 ∩ E_s| / |E_0|` for a nonempty clean/reference edge set `E_0` and subsample edge set `E_s` on fixed landmark identities. This is a directed, study-specific preservation ratio: it detects lost reference edges but does not penalize added edges.
- **Edge-Jaccard dissimilarity:** `d_J(E_0,E_s) = 1 − |E_0 ∩ E_s| / |E_0 ∪ E_s|`. This is the complement of the classical Jaccard set-similarity coefficient, applied to edge sets only when node identities correspond. An empty union is typed undefined/ineligible, not assigned zero.
- **Joint recovery proportion:** the number of attempted runs in which both the graph and membership nerve recover the stated `(β₀, β₁)` reference invariants, divided by all attempted runs. A conditional version restricts the denominator to eligible comparisons, so its eligible denominator must be reported. This is invariant recovery, not graph isomorphism or complete Reeb-graph recovery.

## Paper-to-release experiment map

The paper labels below follow the current standalone submission manuscript.

### §2 Review scope and critical-appraisal method

This narrative-review section defines the purposive, non-database-complete source-identification procedure, the 55-source substantive corpus, and the common appraisal questions. It contains no experimental result; the campaigns below provide finite illustrations after the mathematical review.

### §3 Case studies and study design

The case-study protocol and benchmark roles are documented by the [master experimental design table](../revision_v4_1/MASTER_EXPERIMENTAL_DESIGN_TABLE.csv), the dataset contracts, and the campaign ledgers listed below. The [campaign replay record](../revision_v4_1/derived/CAMPAIGN_REPLAY_PROVENANCE_V4_1.csv) identifies whether each reported quantity is checked by ledger recomputation, representative construction replay, deterministic reconstruction, verified metadata, or a derived analysis on fixed evidence.

### §4 Axis I: stability

#### Fixed-cover subsampling

- **Paper role:** fixed-cover stability comparison and the fixed-cover panel in Fig. 2.
- **Data and constructions:** Circle, Swiss, and Digits; conventional and F-Mapper campaigns use a clean reference cover frozen before applying 50% and 80% retention subsamples. Circle uses `N = 1000`; Swiss uses the canonical `N_base = 2500` sample-then-excise protocol with realized `N` recorded; Digits uses the frozen `1797 × 64` PCA artifact.
- **F-Mapper campaign:** [C4 construction ledger](../evidence/campaigns/c4_fmapper/C4R_CONSTRUCTION_LEDGER.jsonl), with Circle `eps = 0.15`, Swiss `eps = 1.015739105123552`, and Digits `eps = 1.455055840852852`, all `min_samples = 3`; `c = 8`, `tau = 0.10`; 20 defined comparisons per dataset and retention cell.
- **Conventional campaign:** [CIA2 construction ledger](../evidence/campaigns/conventional_fixed_cover/CIA2_CONSTRUCTION_LEDGER.jsonl), with 10 closed regular intervals and 0.30 overlap, frozen reference intervals, and `eps = 0.15` for all three datasets, `min_samples = 3`. Digits has 0/20 defined comparisons because the clean reference is degenerate and all points are unassigned.
- **Ball comparison:** the corresponding Ball ensemble is in [C4V2_BALL_LEDGER.jsonl](../evidence/campaigns/c4_ball_ensemble/C4V2_BALL_LEDGER.jsonl). It follows the metric-cover/landmark protocol, not the scalar-cover/DBSCAN protocol.
- **Sampling and uncertainty:** retention indexes are controlled by the row-level resampling seeds; comparisons are paired within the defined replicate/cell. Medians and defined-comparison counts are reported; the denominator is not replaced by eligible-only counts.

#### M6A2 filtered-graph noise comparison

- **Paper role:** §4 Swiss Conventional/F-Mapper filtered-graph comparison and its Wasserstein companion table.
- **Evidence:** [per-run metrics](../evidence/baseline/m6a2/per_run_metrics.csv) and [bootstrap summaries](../evidence/baseline/m6a2/bootstrap_intervals.csv).
- **Accounting:** 302 rows: one clean reference per method and 30 replicates for each method at five nonzero noise levels. One F-Mapper row at `sigma = 0.10` is non-convergent, leaving 29 valid distances in that cell.
- **Interpretation:** The methods share replication IDs and perturbation seeds. Paired Conventional-minus-F-Mapper bottleneck intervals exclude zero at `sigma ∈ {0.01, 0.03}` but include zero at the three larger doses. The Wasserstein mean is lower for F-Mapper at `sigma ∈ {0.05, 0.08, 0.10}`. These are diagnostic-dependent finite-design summaries, not a uniform method ranking.

#### C7 Circle noise-by-cover/cluster interaction

- **Paper role:** §4 cover-by-clustering interaction and Fig. 6A.
- **Campaign:** [C7 interaction fit](../evidence/campaigns/c7/C7_INTERACTION_FIT.json), with input [Track-C ledger](../evidence/baseline/track_c/IB5_INTERACTIONS_COMPARISON_LEDGER.jsonl).
- **Data:** Circle, `N = 1000`, uniform sampling, height lens `circle_height_y`.
- **Cover grid:** `n_intervals ∈ {6, 10, 15, 20}`, overlap 0.30; clean realized pullbacks are frozen within replicate.
- **Cluster grid:** `eps ∈ {0.08, 0.15, 0.25}`, `min_samples = 3`.
- **Noise:** positive coordinate-noise doses `alpha ∈ {0.05, 0.10, 0.20, 0.30}`, with `epsilon0 = 0.9092268858543847` and `sigma = alpha × epsilon0`; the clean reference is not one of the 48 perturbed cells.
- **Nesting:** 10 replicate streams, 48 perturbed cells, 600 constructions, and 480 eligible comparisons. `β₁₂` is the specified bilinear product-term summary computed with the exact joint common-cover endpoint; the categorical interaction contrasts change sign across `eps` levels and are reported as a robustness check.
- **Bootstrap:** stream-level percentile bootstrap, `B = 10,000`, evaluation seed `20260829`; the secondary CR1 analysis clusters by replicate stream.

### §5 Axis II: local quality and construction reliability

#### C3 Swiss/Digits F-Mapper reliability grid

- **Paper role:** §5 construction-reliability experiments and Fig. 3, Swiss/Digits portion.
- **Campaign:** [C3 construction ledger](../evidence/campaigns/c3/C3_CONSTRUCTION_LEDGER.jsonl).
- **Data:** Swiss has 500 records in total, with 20 realizations in each of the 25 parameter cells and the radial `x,z` lens. Digits has one realization in each parameter cell, using fixed signed PC1 and the released PCA artifact.
- **Grid:** `c ∈ {5, 8, 10, 15, 20}` and `tau ∈ {0.10, 0.15, 0.20, 0.30, 0.40}`.
- **DBSCAN:** Swiss `eps = 1.015739105123552`, Digits `eps = 1.455055840852852`, both `min_samples = 3`.
- **Accounting:** each endpoint is classified as success, FCM non-convergence, or coverage gap. The failure status is part of the result, not a reason to discard the row.

#### C8 Circle/Tripod F-Mapper reliability and quality--topology grid

- **Paper role:** §5 F-Mapper reliability and quality--topology discussion, Fig. 3C, and Fig. 5C--D.
- **Campaign:** [C8 construction ledger](../evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl) and [C8 contract](../contracts/C8_FMAPPER_CIRCLE_GRID_AND_AXIS3_REPLICATION_V1.json).
- **Data:** Circle and Tripod, `N = 1000`, uniform/per-arm sampling as defined above, both with height lenses; Circle has 500 constructions and Tripod has 30 primary-cell constructions.
- **Grid:** `c ∈ {5, 8, 10, 15, 20}`, `tau ∈ {0.10, 0.15, 0.20, 0.30, 0.40}`; primary cell is `c = 10, tau = 0.10`.
- **Clusterer:** Axis-III `eps = 0.15`, `min_samples = 3`; this is not the Swiss scale rule.
- **Solver and seeds:** `tol = 1e−7`, `max_iter = 300`, deterministic linspace initialization; `fcm_seed = 42` is inert. Data seeds are derived from `SeedManager(master_seed = 42).derive_seed('data', stream_index)`.
- **Denominator:** all attempted constructions remain in the denominator; eligible-only conditional summaries are shown separately. There are no retries or parameter retuning after a failure.
- **Finite quality--topology comparison:** pooled differences and rank association are descriptive summaries of the designed grid. The release retains the earlier `B = 10,000`, evaluation-seed `20260830` row-bootstrap artifact for provenance, but the manuscript does not report it as a confidence interval because grid rows with shared factors and replicate structure are not exchangeable population draws.

#### C1 Swiss conventional noise response

- **Paper role:** §5 Swiss local-quality experiment, Fig. 4 at `alpha = 0.05`, and Fig. 7 response across all perturbation levels.
- **Campaign:** [C1 final accounting](../evidence/campaigns/c1/FINAL_ACCOUNTING.json) and [C1 replay plan](../replay_inputs/axis1_corrections/C1_PLAN.jsonl).
- **Data and cover:** Swiss canonical `N_base = 2500` sample-then-excise construction with realized `N`, radial `x,z` lens, 10 regular closed intervals with 0.30 overlap; clean realized pullbacks are frozen within replicate.
- **Clusterer:** `eps = 1.015739105123552`, `min_samples = 3`.
- **Noise:** `epsilon0 = 11.634735480276268`, `sigma = alpha × epsilon0`, with `alpha ∈ {0.05, 0.10, 0.20, 0.30}` in addition to 30 clean references. There are 30 paired constructions at each positive perturbation level, 150 constructions total.
- **Analysis:** The originally specified `alpha = 0.05` confirmatory pattern was not observed; Fig. 4 and the complete perturbation-level series are used only descriptively. The four-level response, macro pullback-level DBSCAN noise, and conditional Silhouette values use the released ledger; Fig. 7 uses the [derived C1 perturbation-response artifact](../revision_v4_1/derived/C1_FULL_DOSE_RESPONSE_V4_1.json). Because observations can occur in several pullbacks, the macro noise fraction is not a unique-observation exclusion rate.
- **Uncertainty:** paired quality summaries over the 30 matched pairs at each perturbation level; no bootstrap is used for the derived Figure 7.

### §6 Axis III: topological agreement

#### Direct Circle/Tripod baselines (III-1)

- **Paper role:** §6 direct same-filter baselines.
- **Evidence:** [Circle result](../evidence/baseline/axis3_iii1/CIRCLE_RESULT.json) and [Tripod result](../evidence/baseline/axis3_iii1/TRIPOD_RESULT.json).
- **Protocol:** `N = 1000`; height lens; 10 regular intervals with 0.30 overlap; Axis-III DBSCAN `eps = 0.15`, `min_samples = 3`; one finite case per domain; same-filter analytic/PL reference.

#### III-2 recovery landscape

- **Paper role:** §6 recovery landscape and Fig. 5A--B.
- **Evidence:** [run-level results](../evidence/baseline/axis3_iii2/RUN_LEVEL_REQUALIFIED.csv), [III-2B recovery summary](../evidence/baseline/axis3_iii2/III2B_RECOVERY_SUMMARY.csv), [III-2B persistence summary](../evidence/baseline/axis3_iii2/III2B_PERSISTENCE_SUMMARY.csv), and [III-2A cell results](../evidence/baseline/axis3_iii2/III2A_CELL_RESULTS.csv).
- **Protocol:** Circle and Tripod with height lenses and the Axis-III clusterer `eps = 0.15`, `min_samples = 3`. III-2A uses `N = 1000`; III-2B uses the sample-size grid below.
- **III-2A:** the `5 × 5` full cover landscape has 50 cover cells.
- **III-2B:** `N ∈ {100, 300, 600, 1200}`, a reduced `3 × 3` cover grid, and 30 matched realizations per cell; 2,160 eligible runs and 2,208 retained metadata rows.
- **Endpoint:** joint graph-and-nerve invariant recovery, with 46/50 cells and 1,201/2,160 runs recovered. Recovery frequencies use exact numerators/denominators and Wilson 95% intervals; no bootstrap is substituted for the specified binomial interval.

#### C8/C5 F-Mapper topological agreement

- **Paper role:** §6 F-Mapper Axis-III comparison, including the C5 case study and the C8 grid.
- **C8:** uses the Circle/Tripod grid above, with primary `c = 10, tau = 0.10`, `eps = 0.15`, `min_samples = 3`, and all-attempted denominator accounting.
- **C5:** [C5 ledger](../evidence/campaigns/c5/C5TA_LEDGER.jsonl) contains one Circle and one Tripod case at `c = 10`, `tau = 0.10`, `eps = 0.15`, height lens. Tripod non-convergence under the frozen solver is retained; frequencies are supplied by the larger C8 campaign rather than by silently retrying C5.

#### Ball Mapper structural comparison

- **Paper role:** §6 Ball Mapper comparison and the Ball panels of Figs. 2 and 6.
- **Swiss campaign:** [C6B ledger](../evidence/campaigns/c6b/C6B_LEDGER.jsonl) and [C6B contract](../contracts/C6B_BALL_SWISS_NOISE_X_RADIUS_V1.json).
- **Swiss protocol:** hole-excised Swiss sample, realized `N`, metric cover, `epsilon0 = 0.8143977317769947`, radius multipliers `k ∈ {1.0, 1.5, 2.0}`, and radii `[0.8143977317769947, 1.221596597665492, 1.6287954635539894]`. There are 20 replicates and `alpha ∈ {0, 0.05, 0.10, 0.20, 0.30}`. Landmarks are selected once from the clean sample per radius and the exact IDs, coordinates, and radius are reused under noise; there is no noisy landmark refit.
- **Circle campaign:** [C9 ledger](../evidence/campaigns/c9/C9_LEDGER.jsonl) and [C9 contract](../contracts/C9_BALL_CIRCLE_NOISE_X_RADIUS_V1.json).
- **Circle protocol:** `N = 1000`, `epsilon0 = 0.9092268858543847`, radius multipliers `k ∈ {1, 1.5, 2}`, radii `[0.9092268858543847, 1.3638403287815771, 1.8184537717087694]`, and `alpha ∈ {0, 0.05, 0.10, 0.20, 0.30}`. There are 20 replicates and 300 constructions. Landmarks are frozen from clean data.
- **Metric and bootstrap:** edge Jaccard or retained-edge fraction; primary difference-in-differences over replicates. C9 uses `B = 20,000`, evaluation seed `20260829`; the saturated `k = 2` arm is zero/vacuous and is not presented as evidence of an effect. C6B uses a paired percentile bootstrap over replicate-level contrasts, `B = 20,000`, evaluation seed `20260828`, as recorded in its run accounting and contract.

### §7 Integration across axes

The integration section combines the campaign-level evidence; it does not introduce a new hidden experiment. The supporting analyses are indexed in the [derived evidence registry](../revision_v4_1/DERIVED_EVIDENCE_REGISTRY.csv) and the [replay provenance file](../revision_v4_1/derived/CAMPAIGN_REPLAY_PROVENANCE_V4_1.csv). In particular:

- IV-1, IV-2, IV-3, and IV-5 use the explicitly linked stability, quality, topological-agreement, ensemble, and failure evidence above.
- C7 supplies the stability-by-structure interaction rather than a separate unreported dataset.
- C6A supplies the Swiss conventional noise-by-cover comparison. Its contract is [C6A_CONVENTIONAL_SWISS_NOISE_X_COVER_V1.json](../contracts/C6A_CONVENTIONAL_SWISS_NOISE_X_COVER_V1.json); the paired primary estimand is not measurable because out-of-cover noisy values create coverage-gap ineligibility. Typed nulls are retained and no imputation is performed.
- C6A uses 20 replicates, 200 constructions, `n_intervals ∈ {10, 15}`, `alpha ∈ {0, 0.05, 0.10, 0.20, 0.30}`, frozen clean covers within replicate, and the same data/noise draw across cover levels. The primary contrast at `alpha = 0.05` is `D(n = 15) − D(n = 10)`.
- The release-level integration therefore distinguishes evidence that is measurable, descriptive, failure-accounting, or formally undefined; these states should not be collapsed into a single success rate.

## Figure-to-data map

The complete figure map and source hashes are in [manuscript/figures/README.md](figures/README.md), [FIGURE_DATA.json](figures/FIGURE_DATA.json), and [FIGURE_DATA_V4_1.json](figures/FIGURE_DATA_V4_1.json). The short map is:

- **Fig. 1:** conceptual three-axis framework; no data campaign.
- **Fig. 2:** fixed-cover subsampling: C4 F-Mapper, CIA2 Conventional, and C4V2 Ball.
- **Fig. 3:** F-Mapper construction reliability: C3 Swiss/Digits and C8 Circle.
- **Fig. 4:** C1 Swiss quality response at the `alpha = 0.05` descriptive slice.
- **Fig. 5:** AXIS3 III-2 conventional recovery and C8 F-Mapper grid.
- **Fig. 6:** noise-by-structure interactions: C7 Circle (A), C6A Swiss conventional (B), C6B Swiss Ball (C), and C9 Circle Ball (D).
- **Fig. 7:** C1 response across four perturbation levels from the frozen [derived response artifact](../revision_v4_1/derived/C1_FULL_DOSE_RESPONSE_V4_1.json); no new constructions.

Figures are generated by [generate_manuscript_figures.py](../code/figures/generate_manuscript_figures.py). Plot summaries do not replace the row-level ledgers; they are derived views whose source hashes are recorded in the figure metadata.

## Replicate and bootstrap reference

For quick reviewer reference, the principal nesting and uncertainty units are:

- **C4/CIA2 fixed-cover subsampling:** 20 planned comparisons per dataset and retention cell; paired within the defined comparison cell; Digits Conventional retains 0/20 defined comparisons.
- **C1:** 30 clean references plus 30 paired constructions at each of four positive perturbation levels; paired construction/replicate unit; descriptive medians and paired quality summaries.
- **C3:** 500 Swiss records in total, comprising 20 realizations in each of 25 grid cells, and one Digits realization per cell; construction status is part of the denominator.
- **C7:** 10 replicate streams; bootstrap resamples streams, not individual observations or pullbacks; `B = 10,000`, seed `20260829`.
- **C6A:** 20 replicates and paired cover-level draws; primary estimand is undefined under the contract because of coverage gaps; no imputation.
- **C6B:** 20 Swiss replicates; paired metric-cover comparisons with frozen landmarks; percentile bootstrap resamples replicate-level contrasts, `B = 20,000`, seed `20260828`.
- **C8:** 500 Circle constructions and 30 Tripod primary-cell constructions; the grid is finite and descriptive. The earlier row-bootstrap artifact (`B = 10,000`, seed `20260830`) is retained only as supplementary provenance and is not used for a manuscript confidence interval.
- **C9:** 20 Circle replicates across 15 noise-by-radius cells; bootstrap resamples replicates, `B = 20,000`, seed `20260829`.
- **III-2:** 30 matched realizations per III-2B cell; recovery is reported as an exact frequency with Wilson 95% intervals.
- **M6A2 supporting baseline:** 30 matched replicate IDs and perturbation seeds per method/noise condition, except 29 complete pairs at `sigma = 0.10`; paired method-difference intervals use 5,000 replicate-level bootstrap resamples with seed `20260830`.

## Failure, eligibility, and denominator policy

The release distinguishes:

- a construction that successfully produces the required object;
- a solver failure, such as FCM non-convergence;
- a coverage gap or other ineligible endpoint;
- a comparison that is formally undefined under the contract;
- an eligible comparison with a finite metric.

These states are retained in the ledgers and summarized separately. “Eligible-only” summaries are conditional descriptions and must not be read as replacement denominators for all attempted constructions. This policy is especially important for Digits Conventional in CIA2, Tripod under the frozen C5 solver, and the C6A coverage-gap response.

## Reproduction and audit trail

The release can be checked in three layers:

1. **Specification:** contracts define datasets, lenses, sampling, preprocessing, parameters, nesting, and typed special states.
2. **Evidence:** campaign ledgers record row-level constructions, seeds, statuses, eligibility, and metrics.
3. **Derivation:** registered scripts and JSON/CSV artifacts produce manuscript summaries and figures from the evidence layer.

Use [run_quick.sh](../code/verification/run_quick.sh) for the fast verification path and [run_all.sh](../code/verification/run_all.sh) for the full release verification. The [environment preflight record](../revision_v4_1/derived/ENVIRONMENT_PREFLIGHT_V4_1.json) records the sealed execution environment and dependency checks.

This manuscript README is therefore the compact protocol summary and crosswalk requested by the reviewer. It does not supersede the contracts, ledgers, or derived-artifact provenance, and it does not claim a public archive DOI; the repository/archive URL can be added to the manuscript once available.
