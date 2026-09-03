# Manuscript figures

Each figure is supplied as PDF, editable SVG, and 300-dpi PNG.
`FIGURE_DATA.json` records plotted summaries and controlling-ledger hashes.
Regenerate all figures from the release root with:

```bash
python3 -B code/figures/generate_manuscript_figures.py
```

1. `fig01_three_axis_framework`: stability under a stated perturbation and
   correspondence, evaluated through the observation-, edge-, or filtered-graph
   endpoint appropriate to the construction; local cluster validation with
   coverage, noise, and construction status; and topological agreement with a
   justified reference under a fixed domain and scalar lens using graph/nerve
   invariants or extended persistence on the stated finite filtration.
2. `fig02_axis1_fixed_cover_subsampling`: joint common-cover F-Mapper and
   Conventional fixed-cover distances (noise-aware where special states occur)
   plus Ball retained-edge fractions; panel endpoints are not ranked on a common
   scale.
3. `fig03_fmapper_construction_reliability`: Swiss/Digits and Circle
   construction-success rates over the stated cover-cardinality and threshold grid.
4. `fig04_c1_swiss_quality_response`: 30 paired clean/noisy Silhouette values
   and their relation to the joint common-cover distance; no general
   benefit-of-noise claim.
5. `fig05_conventional_axis3_recovery_surface`: Conventional recovery surfaces
   and F-Mapper finite-sample recovery; no convergence theorem.
6. `fig06_noise_by_structure_interactions`: the Circle cover--clustering
   interaction, Conventional fixed-cover unmeasurability, the Ball Mapper Swiss
   interaction, and the saturated Ball Mapper Circle arm.

v4.1 additive figure, generated from frozen C1 ledgers by
`revision_v4_1/generate_phase3_derived.py` (not by the six-figure script above):

7. `fig07_c1_perturbation_response`: $D_M^{\mathrm{NA}}$, paired Silhouette
   change, the median pullback-level DBSCAN noise/retention summary, and the number of defined
   comparisons at each perturbation level. The rising conditional Silhouette is
   therefore reported together with the corresponding pullback-level noise.
