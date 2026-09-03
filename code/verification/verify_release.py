#!/usr/bin/env python3
"""Verify release integrity, cleanliness, and every indexed scientific claim."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def jl(rel: str) -> list[dict]:
    return [json.loads(x) for x in (ROOT / rel).read_text().splitlines() if x.strip()]


def ct(rel: str) -> list[dict]:
    with (ROOT / rel).open(newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# Public compatibility name used by the release test module.
sha256 = digest


def check(ok: bool, label: str) -> None:
    if not ok:
        raise AssertionError(label)
    print(f"PASS {label}")


def verify_manifest(ignore_runtime_caches: bool = False) -> None:
    manifest = ROOT / "SHA256SUMS.txt"
    entries = {}
    for line in manifest.read_text().splitlines():
        expected, rel = line.split(None, 1)
        rel = rel.strip()
        if rel in entries:
            raise AssertionError(f"duplicate manifest path: {rel}")
        entries[rel] = expected
    def runtime_cache(path: Path) -> bool:
        rel = path.relative_to(ROOT)
        return ("__pycache__" in rel.parts or ".pytest_cache" in rel.parts or path.suffix == ".pyc")
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
              if p.is_file() and p != manifest and not (ignore_runtime_caches and runtime_cache(p))}
    check(set(entries) == actual, "manifest path set equals release tree")
    check(all(digest(ROOT / rel) == want for rel, want in entries.items()),
          "all root-manifest hashes match")
    print(f"MANIFEST entries={len(entries)}")


def verify_clean_boundary(ignore_runtime_caches: bool = False) -> None:
    bad_parts = {"external_reviews", "outputs", "checks", "__pycache__", "site-packages",
                 "wheels", "staging", "receipts"}
    bad_names = ("unsigned", "draft", "incident", "handoff", "self_audit",
                 "prior_package", "execution_stdout", "candidate", "pycache")
    violations = []
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT)
        runtime_cache = ("__pycache__" in rel.parts or ".pytest_cache" in rel.parts or p.suffix == ".pyc")
        if ignore_runtime_caches and runtime_cache:
            continue
        if bad_parts.intersection(rel.parts):
            violations.append(rel.as_posix())
        if p.is_file() and (any(x in p.name.lower() for x in bad_names)
                            or p.suffix == ".pyc" or p.name.endswith("~")):
            violations.append(rel.as_posix())
    check(not violations, f"no process packages or build residue: {violations[:5]}")
    check(not [p for p in ROOT.rglob("SHA256SUMS.txt") if p.parent != ROOT],
          "root manifest is sole checksum authority")
    text_suffixes = {".md", ".txt", ".json", ".jsonl", ".csv", ".yaml", ".py", ".sh", ".tex", ".cff"}
    bad_text = ("/home/shivam", "external_reviews/", "claude", "grok",
                "chatgpt", "antigravity", "pasted-text", "authorizer_identity",
                "signer_identity", "signed_by")
    traced = []
    for p in ROOT.rglob("*"):
        if p.is_file() and p.suffix in text_suffixes and p.name not in {"SHA256SUMS.txt", "verify_release.py"}:
            if any(x in p.read_text(errors="replace").lower() for x in bad_text):
                traced.append(p.relative_to(ROOT).as_posix())
    check(not traced, f"no machine paths or cross-model briefing residue: {traced[:5]}")
    check(not [p for p in ROOT.rglob("*") if p.is_dir() and not any(p.iterdir())],
          "release has no empty directories")
    allowed_markdown = {
        "README.md",
        "contracts/README.md",
        "environment/README.md",
        "manuscript/README.md",
        "manuscript/figures/README.md",
    }
    actual_markdown = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*.md")}
    check(actual_markdown == allowed_markdown,
          f"Markdown is limited to README files: {sorted(actual_markdown ^ allowed_markdown)}")
    forbidden_plan_text = ("final experimental rebuild plan", "rebuild plan",
                           "13 august 2026")
    named_plan = []
    for p in ROOT.rglob("*"):
        if (p.is_file() and p.suffix in text_suffixes
                and p.name != "verify_release.py"):
            text = p.read_text(errors="replace").lower()
            if any(term in text for term in forbidden_plan_text):
                named_plan.append(p.relative_to(ROOT).as_posix())
    check(not named_plan, f"obsolete rebuild-plan wording removed: {named_plan[:5]}")

    disclosure = (ROOT / "README.md").read_text()
    required = ("principal investigators", "conceived the study",
                "designed the evaluation study and analyses",
                "directed the research programme", "cross-checking the code",
                "verifying numerical reproducibility")
    disclosure_flat = " ".join(disclosure.lower().split())
    check(all(token in disclosure_flat for token in required) and "ai" in disclosure_flat,
          "authorship and concise AI-assistance disclosure present")

    spec_bindings = json.loads((ROOT / "contracts/SPEC_SOURCE_BINDINGS.json").read_text())
    evidence_text = "\n".join(
        p.read_text(errors="replace") for p in (ROOT / "evidence").rglob("*") if p.is_file()
    )
    check(len(spec_bindings) == 8
          and all(digest(ROOT / "contracts" / name) == binding["release_sha256"]
                  and binding["executed_sha256"] in evidence_text
                  for name, binding in spec_bindings.items()),
          "clean specification copies preserve executed-source digest bindings")

    bound = {}
    for ledger in sorted((ROOT / "evidence/campaigns").rglob("*.jsonl")):
        for row in jl(ledger.relative_to(ROOT).as_posix()):
            if row.get("contract_id") and row.get("contract_sha256"):
                bound[row["contract_id"]] = row["contract_sha256"]
                break
    unresolved = []
    portable = json.loads((ROOT / "contracts/CONTRACT_SOURCE_BINDINGS.json").read_text())
    for contract_id, want in sorted(bound.items()):
        target = ROOT / "contracts" / f"{contract_id}.json"
        direct = target.is_file() and digest(target) == want
        binding = portable.get(contract_id, {})
        normalized = (target.is_file() and binding.get("executed_sha256") == want
                      and binding.get("release_sha256") == digest(target))
        if not (direct or normalized):
            unresolved.append(contract_id)
    check(bound and not unresolved, f"all ledger-bound campaign contracts resolve: {unresolved}")
    check(set(portable) == {
            "C5_FMAPPER_TIER_A_ELIGIBILITY_AMENDMENT_V1",
            "C6A_CONVENTIONAL_SWISS_NOISE_X_COVER_V1",
            "C6B_BALL_SWISS_NOISE_X_RADIUS_V1",
            "C7_TRACKC_INTERACTION_ANALYSIS_V3_CORRECTED_ENDPOINT",
            "C8_FMAPPER_CIRCLE_GRID_AND_AXIS3_REPLICATION_V1",
            "C9_BALL_CIRCLE_NOISE_X_RADIUS_V1",
          },
          "portable contract normalization is explicit and hash-bound")
    check((ROOT / "contracts/IV1_COVER_ARM_ANALYSIS_SPEC.txt").is_file()
          and not (ROOT / "contracts/IV1_COVER_ARM_ANALYSIS_CONTRACT.txt").exists(),
          "IV-1 ships as a final executed analysis specification")

    manuscript_files = {
        p.relative_to(ROOT / "manuscript").as_posix()
        for p in (ROOT / "manuscript").rglob("*") if p.is_file()
    }
    publication_index = {"README.md"}
    figure_files = manuscript_files - publication_index
    check(publication_index.issubset(manuscript_files)
          and figure_files
          and all(path.startswith("figures/") for path in figure_files),
          "release manuscript boundary contains the protocol crosswalk and figures only")

    readme_text = (ROOT / "README.md").read_text(errors="replace")
    check("python3.12 -m venv" in readme_text
          and "run_quick.sh" in readme_text and "run_all.sh" in readme_text,
          "README uses explicit Python 3.12 and distinguishes fast verification from full replay")
    check((ROOT / "code/verification/verify_environment.py").is_file(),
          "strict environment preflight is bundled")
    check((ROOT / "CITATION.cff").is_file() and (ROOT / "LICENSE").is_file(),
          "citation metadata and explicit licensing status are bundled")

    fixed = {
        "src/mapper_framework/support/d_common_id.py": "5ddeac40a45bd2e32b670b51359d95ad8bcf1c4ebbf96a55cb24be062a4ab864",
        "src/mapper_framework/support/ia2_core.py": "06dab819039c1543660071f15d92da44228842f4016822cd2de8d08ff4ccd401",
        "environment/ENVIRONMENT_LOCK_MANIFEST.json": "46c58532e16925920e54a363e9727f7fa0afa908e2f5f1317d9f057be8ed11d9",
    }
    check(all(digest(ROOT / rel) == want for rel, want in fixed.items()),
          "shared metric modules and environment lock match release bindings")
    bindings = json.loads((ROOT / "src/mapper_framework/support/SOURCE_BINDINGS.json").read_text())
    check(bindings["ia2_core.py"]["executed_sha256"]
          == "e2b6df143ec4fa42e335949620c869249bdb47a7631f37e30ab86a9eff0fa484"
          and bindings["ia2_core.py"]["release_sha256"] == fixed["src/mapper_framework/support/ia2_core.py"],
          "portable I-A2 helper binds the corrected endpoint and executed-source provenance")
    check(bindings["d_common_id.py"]["release_sha256"]
          == fixed["src/mapper_framework/support/d_common_id.py"],
          "exact joint Mapper-distance source binding is current")
    check(bindings["local_quality.py"]["executed_sha256"]
          == "3c2e7e1dffe6195959f15d06835708e492c8b2257ff5bdb5ac482e44ca94d089"
          and bindings["local_quality.py"]["release_sha256"]
          == digest(ROOT / "src/mapper_framework/local_quality.py"),
          "portable local-quality source preserves its executed digest")
    fmapper_source = (ROOT / "src/mapper_framework/f_mapper.py").read_text()
    fmapper_spec = (ROOT / "contracts/f_mapper_spec_v1.0.0.txt").read_text()
    check(fmapper_source.count("rng = np.random.default_rng(seed)") == 1
          and fmapper_source.count("rng.") == 0
          and "initial_centroids = np.linspace(min_val, max_val, c)" in fmapper_source
          and "evenly spaced" in fmapper_spec.lower()
          and "not an experimental factor" in fmapper_spec
          and not (ROOT / "contracts/FMAPPER_FCM_SEED_ERRATUM_2026-08-29.txt").exists(),
          "F-Mapper specification states deterministic linspace initialization")
    check((ROOT / "pyproject.toml").is_file()
          and (ROOT / "src/mapper_framework/__init__.py").is_file(),
          "Mapper framework is shipped as an installable src-layout package")
    stale_governance = []
    for p in (ROOT / "contracts").glob("*.txt"):
        lowered = p.read_text(errors="replace").lower()
        if "pending external" in lowered or "candidate specification" in lowered:
            stale_governance.append(p.name)
    check(not stale_governance, f"release specifications have frozen status: {stale_governance}")

    narrative = [
        ROOT / "README.md",
        ROOT / "contracts/README.md",
        ROOT / "environment/README.md",
    ]
    narrative_hits = []
    for path in narrative:
        lowered = path.read_text(errors="replace").lower()
        if "confirmatory" in lowered or "preregister" in lowered:
            narrative_hits.append(path.relative_to(ROOT).as_posix())
    check(not narrative_hits, f"narrative files omit confirmatory/preregister language: {narrative_hits}")

    dataset_spec = (ROOT / "contracts/dataset_contracts_v1.0.0.txt").read_text()
    check("Open Protocol Decisions" not in dataset_spec,
          "dataset specification has no Open Protocol Decisions register")

    forbidden_release_phrases = (
        "frozen_prospective",
        "preregister",
        "cross-model briefing",
        "execution not authorized",
    )
    phrase_hits = []
    stale_paths = []
    for path in sorted((ROOT / "contracts").rglob("*")):
        if not path.is_file() or path.suffix not in {".txt", ".json", ".md"}:
            continue
        text = path.read_text(errors="replace")
        lowered = text.lower()
        if any(phrase in lowered for phrase in forbidden_release_phrases):
            phrase_hits.append(path.name)
        if "final_release/evidence/" in text:
            stale_paths.append(path.name)
        for match in re.findall(r"([A-Za-z0-9_./-]+\.md)", text):
            txt_name = Path(match).name.replace(".md", ".txt")
            if (ROOT / "contracts" / txt_name).is_file():
                stale_paths.append(f"{path.name}:{match}")
    check(not phrase_hits, f"contracts omit prospective/briefing residue: {phrase_hits[:5]}")
    check(not stale_paths, f"contracts use released paths and extensions: {stale_paths[:5]}")


def verify_figures() -> None:
    stems = [
        "fig01_three_axis_framework",
        "fig02_axis1_fixed_cover_subsampling",
        "fig03_fmapper_construction_reliability",
        "fig04_c1_swiss_quality_response",
        "fig05_conventional_axis3_recovery_surface",
        "fig06_noise_by_structure_interactions",
    ]
    figure_dir = ROOT / "manuscript/figures"
    check(all((figure_dir / f"{stem}.{suffix}").stat().st_size > 1000
              for stem in stems for suffix in ("pdf", "svg", "png")),
          "seven figures are present as nonempty PDF, SVG, and PNG files")
    data = json.loads((figure_dir / "FIGURE_DATA.json").read_text())
    check(all(digest(ROOT / rel) == want for rel, want in data["source_sha256"].items()),
          "figure data are bound to the current released ledgers")
    check(data["figure_04"]["n_pairs"] == 30
          and sum(cell["success"] for cell in data["figure_03"].values()) == 726
          and sum(cell["joint_recovered"] for cell
                  in data["figure_05"]["conventional"].values()) == 1201
          and data["figure_05"]["fmapper_primary"]["circle"]["exact"] == 14
          and data["figure_05"]["fmapper_primary"]["tripod"]["exact"] == 21,
          "figure summaries reproduce C1, C3/C8, and Axis-III denominators")
    check(data["figure_06"]["c6a"]["0.05"]["median"] == 119.0
          and data["figure_06"]["c6a"]["0.3"]["median"] == 626.5
          and math.isclose(data["figure_06"]["c7"]["beta_12"]["point_estimate"],
                           -0.023328450168527137, rel_tol=0, abs_tol=1e-14)
          and all(data["figure_06"]["c9"][f"alpha={a}|k=2.0"]["fraction_nonzero"] == 0
                  for a in (.05, .10, .20, .30)),
          "interaction figure reproduces C7, C6-a, and C9 diagnostics")


def verify_quality_survival() -> None:
    quality = jl("evidence/baseline/quality_survival/REPLICATION_1300_QS_QUALITY_LEDGER.jsonl")
    paired = jl("evidence/baseline/quality_survival/REPLICATION_1300_QS_PAIRED_LEDGER.jsonl")
    circle = [r for r in paired if r["dataset_id"] == "unit_circle_S1" and r["alpha"] == 0.05]
    fields = ("d_common_id", "clean_silhouette_macro", "perturbed_silhouette_macro",
              "clean_silhouette_coverage_adjusted_01", "perturbed_silhouette_coverage_adjusted_01")
    complete = [r for r in circle if all(r.get(f) is not None for f in fields)]
    signature = [r for r in complete if r["d_common_id"] > 0
                 and r["perturbed_coverage_fraction"] < r["clean_coverage_fraction"]
                 and r["perturbed_silhouette_macro"] > r["clean_silhouette_macro"]
                 and r["perturbed_silhouette_coverage_adjusted_01"]
                 < r["clean_silhouette_coverage_adjusted_01"]]
    check((len(quality), len(paired), len(complete), len(signature)) == (150, 120, 10, 0),
          "QS historical accounting retained; composite 0/10 is non-adjudicating")


def verify_exact_mapper_distance() -> None:
    """Check the joint objective on a case where the old upper estimate differs."""
    from mapper_framework.support.d_common_id import (
        d_common_id,
        hash_filter_definition,
        hash_realized_cover,
    )
    a = {0: {0: 0, 1: 0, 2: 0, 3: 1},
         1: {0: 0, 1: 0, 2: 0, 3: 1}}
    b = {0: {0: 0, 1: 0, 2: 1, 3: 0},
         1: {0: 0, 1: 1, 2: 0, 3: 1}}
    ids = {0, 1, 2, 3}
    cover_hash = hash_realized_cover(a, ids)
    filter_hash = hash_filter_definition("test filter")
    result = d_common_id(
        a, b, common_ids=ids,
        cover_id_A="fixed", cover_id_B="fixed",
        filter_id_A="f", filter_id_B="f",
        cover_hash_A=cover_hash, cover_hash_B=cover_hash,
        filter_hash_A=filter_hash, filter_hash_B=filter_hash,
    )
    missing = d_common_id(
        a, b, common_ids=ids,
        cover_id_A="fixed", cover_id_B="fixed",
        filter_id_A="f", filter_id_B="f",
    )
    check(result["distance"] == .5
          and result["optimizer"] == "scipy_milp_exact_joint",
          "exact joint optimizer resolves the per-bin-upper-estimate counterexample")
    check(missing["status"] == "rejected"
          and missing["reason"] == "missing_cover_hash_A",
          "Mapper-distance provenance hashes fail closed")


def verify_track_c() -> None:
    con = jl("evidence/baseline/track_c/IB5_INTERACTIONS_CONSTRUCTION_LEDGER.jsonl")
    cmp = jl("evidence/baseline/track_c/IB5_INTERACTIONS_COMPARISON_LEDGER.jsonl")
    cells = {(r["n_intervals"], r["eps"], r["alpha"]) for r in cmp}
    check(len(con) == 600 and len(cmp) == 480 and all(r["eligible"] for r in cmp),
          "Track C Circle factorial 600 constructions / 480 eligible comparisons")
    check(len(cells) == 48 and all(sum((r["n_intervals"], r["eps"], r["alpha"]) == c
                                      for r in cmp) == 10 for c in cells),
          "Track C has 48 cells with 10 comparisons each")
    metric_hash = digest(ROOT / "src/mapper_framework/support/d_common_id.py")
    check(all(r["metric_definition"] == "exact_joint_definition_9"
              and r["exact_metric_sha256"] == metric_hash
              and len(r["realized_cover_sha256"]) == 64
              and len(r["filter_definition_sha256"]) == 64 for r in cmp),
          "Track C exact endpoint is bound to realized-cover and filter hashes")


def verify_c7() -> None:
    """Independently refit the specified Track C interaction and bootstrap."""
    rows = [r for r in jl("evidence/baseline/track_c/IB5_INTERACTIONS_COMPARISON_LEDGER.jsonl")
            if r["eligible"] and r["d_common_id"] is not None]
    released = json.loads((ROOT / "evidence/campaigns/c7/C7_INTERACTION_FIT.json").read_text())
    diagnostic = json.loads((ROOT / "evidence/campaigns/c7/C7_ROBUSTNESS_DIAGNOSTICS.json").read_text())
    contract = ROOT / "contracts/C7_TRACKC_INTERACTION_ANALYSIS_V1.json"
    portable = json.loads((ROOT / "contracts/CONTRACT_SOURCE_BINDINGS.json").read_text())
    binding = portable.get("C7_TRACKC_INTERACTION_ANALYSIS_V3_CORRECTED_ENDPOINT", {})
    want = released["analysis_contract_sha256"]
    contract_ok = (
        digest(contract) == want
        or (binding.get("executed_sha256") == want
            and binding.get("release_sha256") == digest(contract))
    )
    check(contract_ok
          and digest(ROOT / "evidence/baseline/track_c/IB5_INTERACTIONS_COMPARISON_LEDGER.jsonl")
          == released["input_ledger_sha256"],
          "C7 release-local analysis contract and input binding")
    alpha_levels = (.10, .20, .30)
    X, y, groups = [], [], []
    for row in rows:
        nc = float(row["n_intervals"]) - 10.0
        ec = float(row["eps"]) - .15
        X.append([nc, ec, nc * ec]
                 + [float(abs(float(row["alpha"]) - a) < 1e-12) for a in alpha_levels])
        y.append(float(row["d_common_id"]))
        groups.append(int(row["replicate_index"]))
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)

    def fit(x, response, cluster):
        xw, yw = x.copy(), response.copy()
        for group in np.unique(cluster):
            mask = cluster == group
            xw[mask] -= xw[mask].mean(axis=0)
            yw[mask] -= yw[mask].mean()
        return np.linalg.lstsq(xw, yw, rcond=None)[0]

    beta = fit(X, y, groups)
    streams = np.unique(groups)
    where = {int(s): np.flatnonzero(groups == s) for s in streams}
    rng = np.random.default_rng(20260829)
    boot = np.empty(10000)
    for b in range(len(boot)):
        pick = rng.choice(streams, size=len(streams), replace=True)
        xb, yb, gb = [], [], []
        for copy_id, stream in enumerate(pick):
            index = where[int(stream)]
            xb.append(X[index]); yb.append(y[index])
            gb.append(np.full(len(index), copy_id))
        boot[b] = fit(np.vstack(xb), np.concatenate(yb), np.concatenate(gb))[2]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    stored = released["beta_12"]
    numerical = json.loads((ROOT / "contracts/NUMERICAL_REPORTING_CONTRACT.json").read_text())
    tol = numerical["cross_environment_absolute_tolerance"]
    check(len(rows) == 480 and math.isclose(beta[2], stored["point_estimate"], rel_tol=0, abs_tol=tol)
          and math.isclose(float(lo), stored["primary_interval"]["lo_2.5"], rel_tol=0, abs_tol=tol)
          and math.isclose(float(hi), stored["primary_interval"]["hi_97.5"], rel_tol=0, abs_tol=tol)
          and released["terminal_state"] == "T2" and hi < 0,
          "C7 specified interaction refit and stream bootstrap")

    def mean(n, eps):
        values = [r["d_common_id"] for r in rows
                  if r["n_intervals"] == n and r["eps"] == eps]
        return float(np.mean(values))
    corner = (mean(20, .25) - mean(20, .08)) - (mean(6, .25) - mean(6, .08))
    check(math.isclose(corner, diagnostic["model_free_corner_did"]["point_estimate"],
                       rel_tol=0, abs_tol=tol)
          and diagnostic["model_free_corner_did"]["stream_bootstrap_95"]["hi"] < 0,
          "C7 post-hoc model-free corner diagnostic is independently negative")

    categorical = diagnostic["categorical_factor_interaction_contrasts"]
    check(categorical["reference_n_intervals"] == 6 and categorical["reference_eps"] == .08
          and len(categorical["rows"]) == 6,
          "C7 categorical-factor robustness metadata")
    cat_index = {(r["n_intervals"], r["eps"]): r for r in categorical["rows"]}
    by_stream = {int(s): [r for r in rows if int(r["replicate_index"]) == int(s)] for s in streams}
    rng = np.random.default_rng(20260829)
    picks = rng.choice(streams, size=(10000, len(streams)), replace=True)
    for n in (10, 15, 20):
        for eps in (.15, .25):
            per_stream = np.asarray([
                ((np.mean([r["d_common_id"] for r in by_stream[int(s)] if r["n_intervals"] == n and r["eps"] == eps])
                  - np.mean([r["d_common_id"] for r in by_stream[int(s)] if r["n_intervals"] == n and r["eps"] == .08]))
                 - (np.mean([r["d_common_id"] for r in by_stream[int(s)] if r["n_intervals"] == 6 and r["eps"] == eps])
                    - np.mean([r["d_common_id"] for r in by_stream[int(s)] if r["n_intervals"] == 6 and r["eps"] == .08])))
                for s in streams
            ], dtype=float)
            boot_cat = per_stream[picks].mean(axis=1)
            c_lo, c_hi = np.percentile(boot_cat, [2.5, 97.5])
            stored_cat = cat_index[(n, eps)]
            check(math.isclose(float(per_stream.mean()), stored_cat["point_estimate"], rel_tol=0, abs_tol=tol)
                  and math.isclose(float(c_lo), stored_cat["stream_bootstrap_95"]["lo"], rel_tol=0, abs_tol=tol)
                  and math.isclose(float(c_hi), stored_cat["stream_bootstrap_95"]["hi"], rel_tol=0, abs_tol=tol),
                  f"C7 categorical interaction contrast n={n}, eps={eps}")
    check(cat_index[(10, .15)]["stream_bootstrap_95"]["lo"] < 0 < cat_index[(10, .15)]["stream_bootstrap_95"]["hi"]
          and all(cat_index[(n, .25)]["stream_bootstrap_95"]["hi"] < 0 for n in (10, 15, 20)),
          "C7 categorical robustness shows non-uniform interaction across eps levels")


def verify_c8() -> None:
    rows = jl("evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl")
    accounting = json.loads((ROOT / "evidence/campaigns/c8/C8_RUN_ACCOUNTING.json").read_text())
    circle = [r for r in rows if r["dataset_id"] == "unit_circle"]
    tripod = [r for r in rows if r["dataset_id"] == "branching_tripod_y"]
    primary_circle = [r for r in circle if r["n_intervals"] == 10 and r["threshold"] == .10]
    check(len(rows) == 530 and Counter(r["status"] for r in rows)
          == Counter(success=408, fcm_non_convergence=111, coverage_gap=11)
          and accounting["attempted_constructions"] == 530
          and {r["local_quality_sha256"] for r in rows}
          == {"3c2e7e1dffe6195959f15d06835708e492c8b2257ff5bdb5ac482e44ca94d089"},
          "C8 530-attempt status accounting")
    check((sum(r["tier_a_exact_agreement"] for r in primary_circle),
           sum(r["tier_a_eligible"] for r in primary_circle)) == (14, 14)
          and (sum(r["tier_a_exact_agreement"] for r in tripod),
               sum(r["tier_a_eligible"] for r in tripod)) == (21, 21),
          "C8 primary Circle 14/20 and Tripod 21/30 with 100% conditional invariant agreement")
    eligible_circle = [r for r in circle if r["tier_a_eligible"]]
    check(len(eligible_circle) == 387
          and sum(r["tier_a_exact_agreement"] for r in circle) == 315
          and sum(r["tier_a_eligible"] and not r["tier_a_exact_agreement"] for r in circle) == 72,
          "C8 full Circle grid distinguishes eligibility from Tier-A invariant agreement")
    by_c = {c: sum(r["status"] == "success" for r in circle if r["n_intervals"] == c)
            for c in (5, 8, 10, 15, 20)}
    check(by_c == {5: 98, 8: 89, 10: 77, 15: 66, 20: 57},
          "C8 Circle construction success declines across cover cardinality")
    strict = [r for r in circle if r["n_intervals"] == 10 and r["threshold"] == .40]
    check(sum(r["tier_a_eligible"] for r in strict) == 16
          and sum(r["tier_a_exact_agreement"] for r in strict) == 0,
          "C8 strict-threshold cell prevents global 100%-conditional claim")


def verify_c9() -> None:
    rows = jl("evidence/campaigns/c9/C9_LEDGER.jsonl")
    accounting = json.loads((ROOT / "evidence/campaigns/c9/C9_RUN_ACCOUNTING.json").read_text())
    diagnostic = json.loads((ROOT / "evidence/campaigns/c9/C9_DEGENERACY_DIAGNOSTICS.json").read_text())
    identity = [r for r in rows if r["alpha"] == 0.0]
    k2 = [r for r in rows if r["radius_multiplier"] == 2.0]
    check(len(rows) == 300 and all(r["status"] == "success" and r["eligible"] for r in rows)
          and len(identity) == 60 and all(r["edge_jaccard_distance"] == 0.0 for r in identity),
          "C9 300 defined rows and 60 identity checks")
    check(len(k2) == 100 and {(r["clean_V"], r["clean_E"]) for r in k2} == {(2, 1)}
          and {r["edge_jaccard_distance"] for r in k2} == {0.0},
          "C9 k=2 arm is empirically saturated in the executed design")
    index = {(r["replicate_index"], r["alpha"], r["radius_multiplier"]):
             r["edge_jaccard_distance"] for r in rows}
    did = np.asarray([(index[(i, .10, 2.0)] - index[(i, .05, 2.0)])
                      - (index[(i, .10, 1.0)] - index[(i, .05, 1.0)])
                      for i in range(20)])
    primary = accounting["primary_estimand"]
    rng = np.random.default_rng(primary["bootstrap_seed"])
    boot = np.empty(primary["bootstrap_B"])
    for b in range(len(boot)):
        boot[b] = np.median(did[rng.integers(0, len(did), len(did))])
    lo, hi = np.quantile(boot, [.025, .975])
    check(float(np.median(did)) == primary["median_difference_in_differences"] == 0.0
          and (float(lo), float(hi)) == (0.0, 0.0)
          and diagnostic["did_fraction_exactly_zero"] == .85,
          "C9 zero-width interval reproduced and typed as discrete saturation")


def verify_c1_c3() -> None:
    rows = [r for r in jl("evidence/campaigns/c1/C1_PAIRED_QUALITY_LEDGER.jsonl") if r["alpha"] == 0.05]
    acc = json.loads((ROOT / "evidence/campaigns/c1/FINAL_ACCOUNTING.json").read_text())
    dvals = [r["d_common_id"] for r in jl("evidence/campaigns/c1/C1_COMPARISON_LEDGER.jsonl")
             if r["alpha"] == 0.05]
    check(len(rows) == 30
          and statistics.median(r["silhouette_clean"] for r in rows) == acc["median_clean_silhouette"]
          and statistics.median(r["silhouette_perturbed"] for r in rows) == acc["median_perturbed_silhouette"]
          and statistics.median(dvals) == acc["median_d_common_id"]
          and acc["terminal_state"] == "PRIMARY_PATTERN_NOT_OBSERVED",
          "C1 primary medians and corrected terminal state")
    comparisons = jl("evidence/campaigns/c1/C1_COMPARISON_LEDGER.jsonl")
    metric_hash = digest(ROOT / "src/mapper_framework/support/d_common_id.py")
    check(len(comparisons) == 120 and all(r["eligible"]
          and r["metric_definition"] == "exact_joint_definition_9"
          and r["exact_metric_sha256"] == metric_hash
          and len(r["realized_cover_sha256"]) == 64
          and len(r["filter_definition_sha256"]) == 64 for r in comparisons),
          "C1 has 120 exact fixed-realized-cover distances with provenance hashes")
    dose = json.loads((ROOT / "revision_v4_1/derived/C1_FULL_DOSE_RESPONSE_V4_1.json").read_text())
    expected_noise = {"0.05": .2503346954680561, "0.10": .5578188663107546,
                      "0.20": .8491631250779191, "0.30": .9446321276569616}
    check(all(math.isclose(dose["doses"][a]["median_noise_fraction_macro_perturbed"], v,
                           rel_tol=0, abs_tol=1e-15)
              and dose["doses"][a]["n_eligible"] == 30
              for a, v in expected_noise.items())
          and all(math.isclose(dose["doses"][a]["median_noise_fraction_macro_clean"],
                               .055306906950664335, rel_tol=0, abs_tol=1e-15)
                  for a in expected_noise),
          "C1 conditional-quality summaries carry DBSCAN noise/retention denominators")
    units = jl("evidence/campaigns/c2/C2_UNIT_ACCOUNTING_LEDGER.jsonl")
    arms = jl("evidence/campaigns/c2/C2_ARM_LEDGER.jsonl")
    c2 = json.loads((ROOT / "evidence/campaigns/c2/FINAL_ACCOUNTING.json").read_text())
    check(len(units) == 260 and len(arms) == 780
          and all(r["unit_status"] == "selection_stage_failed" and r["n_arms_eligible"] == 0 for r in units)
          and c2["record_origin"].startswith("deterministic_reconstruction"),
          "C2 260 failures and 780 explicitly reconstructed arms")
    c3 = jl("evidence/campaigns/c3/C3_CONSTRUCTION_LEDGER.jsonl")
    check(len(c3) == 525 and Counter(r["status"] for r in c3)
          == Counter(success=339, fcm_non_convergence=181, coverage_gap=5),
          "C3 status counts 339/181/5")


def verify_axis1_repairs() -> None:
    fmapper = jl("evidence/campaigns/c4_fmapper/C4R_CONSTRUCTION_LEDGER.jsonl")
    for dataset in ("swiss_roll_with_hole", "digits_1797x64_scaled16"):
        vals = {f: [r["d_common_id_distance"] for r in fmapper
                    if r["arm"] == "fixed_cover" and r["dataset_id"] == dataset
                    and r["fraction"] == f and r["d_common_id_distance"] is not None]
                for f in (0.5, 0.8)}
        check(statistics.median(vals[0.8]) < statistics.median(vals[0.5]),
              f"C4 F-Mapper {dataset}: 80% median below 50%")
    expected_fmapper = {
        ("swiss_roll_with_hole", .5): .6285714285714286,
        ("swiss_roll_with_hole", .8): .3087005019520357,
        ("digits_1797x64_scaled16", .5): .27783964365256125,
        ("digits_1797x64_scaled16", .8): .08066759388038942,
    }
    for (dataset, fraction), expected in expected_fmapper.items():
        values = [r["d_common_id_distance"] for r in fmapper
                  if r["arm"] == "fixed_cover" and r["dataset_id"] == dataset
                  and r["fraction"] == fraction and r["d_common_id_distance"] is not None]
        check(len(values) == 20 and statistics.median(values) == expected,
              f"C4 exact F-Mapper median {dataset} {fraction}")
    ball = jl("evidence/campaigns/c4_ball_ensemble/C4V2_BALL_LEDGER.jsonl")
    for f, want in ((0.5, 0.6669435215946844), (0.8, 0.8853820598006644)):
        vals = [r["retained_edge_fraction"] for r in ball
                if r["dataset_id"] == "swiss_roll_with_hole" and r["fraction"] == f]
        check(statistics.median(vals) == want, f"Ball Swiss retained-edge median {f}")
    ens = jl("evidence/campaigns/c4_ball_ensemble/C4V2_ENSEMBLE_LEDGER.jsonl")
    circle = [r for r in ens if r["dataset_id"] == "unit_circle_S1"]
    other = [r for r in ens if r["dataset_id"] != "unit_circle_S1"]
    check(len(circle) == 41 and all(r["status"] == "success" for r in circle)
          and len(other) == 82 and all(r["status"] != "success" for r in other),
          "C4 Ensemble constructibility 41/41 Circle and 0/82 other")
    acc = json.loads((ROOT / "evidence/campaigns/c4_ball_ensemble/FINAL_ACCOUNTING.json").read_text())
    check(acc["fit_transform_calls_total"] == 6396 and acc["mapper_operations_total"] == 6516,
          "C4 operation accounting 6396 / 6516")
    fixed = [r for r in jl("evidence/campaigns/conventional_fixed_cover/CIA2_CONSTRUCTION_LEDGER.jsonl")
             if r["arm"] == "fixed_cover"]
    def ds(dataset, fraction):
        return [r["d_common_id_distance"] for r in fixed if r["dataset_id"] == dataset
                and r["fraction"] == fraction and r["d_common_id_distance"] is not None]
    c50, c80 = ds("unit_circle_S1", .5), ds("unit_circle_S1", .8)
    check(len(c50) == 20 and sum(x == 0 for x in c50) == 19 and max(c50) == .084
          and len(c80) == 20 and set(c80) == {0.0}, "Conventional Circle fixed-cover values")
    check(len(ds("swiss_roll_with_hole", .5)) == 8 and len(ds("swiss_roll_with_hole", .8)) == 20
          and not ds("digits_1797x64_scaled16", .5) and not ds("digits_1797x64_scaled16", .8),
          "Conventional Swiss/Digits fixed-cover denominators")
    exact_hash = digest(ROOT / "src/mapper_framework/support/d_common_id.py")
    promoted = [r for r in fmapper + fixed if r.get("d_common_id_distance") is not None]
    check(promoted and all(r["metric_definition"] == "exact_joint_definition_9"
          and r["exact_metric_sha256"] == exact_hash
          and len(r["realized_cover_sha256"]) == 64
          and len(r["filter_definition_sha256"]) == 64 for r in promoted),
          "all promoted fixed-cover distances bind exact code, cover, and filter")


def verify_c5_c6() -> None:
    c5 = jl("evidence/campaigns/c5/C5TA_LEDGER.jsonl")
    circle = next(r for r in c5 if r["dataset_id"] == "unit_circle")
    tripod = next(r for r in c5 if r["dataset_id"] == "branching_tripod_y")
    check(circle["eligible"] and all(circle[f] == 1 for f in
          ("beta0_graph_mapper", "beta0_nerve_mapper", "beta1_graph_mapper", "beta1_nerve_mapper"))
          and circle["delta_beta1_graph_to_reference"] == 0, "C5 Circle finite-case agreement")
    check(tripod["status"] == "fcm_non_convergence" and not tripod["eligible"]
          and tripod["beta0_graph_mapper"] is None, "C5 Tripod typed non-convergence")
    c6a = jl("evidence/campaigns/c6a/C6A_LEDGER.jsonl")
    med = {a: statistics.median(r["n_outside_frozen_cover"] for r in c6a if r["alpha"] == a)
           for a in (.05, .10, .20, .30)}
    pa = json.loads((ROOT / "evidence/campaigns/c6a/C6A_RUN_ACCOUNTING.json").read_text())["primary_estimand"]
    check(Counter(r["status"] for r in c6a) == Counter(success=40, coverage_gap=160)
          and med == {.05: 119, .10: 233, .20: 445, .30: 626.5}
          and pa["n_complete_pairs"] == 0 and pa["terminal_state"] == "PRIMARY_ESTIMAND_NOT_MEASURABLE",
          "C6-a gaps, diagnostics, and unmeasurable estimand")
    c6b = jl("evidence/campaigns/c6b/C6B_LEDGER.jsonl")
    idx = {(r["replicate_index"], r["alpha"], r["radius_multiplier"]): r["edge_jaccard_distance"]
           for r in c6b}
    did = [(idx[(i, .10, 2.0)] - idx[(i, .05, 2.0)]) -
           (idx[(i, .10, 1.0)] - idx[(i, .05, 1.0)]) for i in range(20)]
    pb = json.loads((ROOT / "evidence/campaigns/c6b/C6B_RUN_ACCOUNTING.json").read_text())["primary_estimand"]
    import numpy as np
    arr, rng = np.asarray(did), np.random.default_rng(pb["bootstrap_seed"])
    boot = np.empty(pb["bootstrap_B"])
    for i in range(pb["bootstrap_B"]):
        boot[i] = np.median(arr[rng.integers(0, len(arr), len(arr))])
    lo, hi = float(np.quantile(boot, .025)), float(np.quantile(boot, .975))
    check(len(c6b) == 300 and all(r["edge_jaccard_distance"] is not None for r in c6b)
          and statistics.median(did) == pb["median_difference_in_differences"]
          and (lo, hi) == (pb["ci95_low"], pb["ci95_high"]) and lo < 0 < hi,
          "C6-b exact paired-bootstrap DiD interval")


def verify_axis3_conventional() -> None:
    c = json.loads((ROOT / "evidence/baseline/axis3_iii1/CIRCLE_RESULT.json").read_text())
    t = json.loads((ROOT / "evidence/baseline/axis3_iii1/TRIPOD_RESULT.json").read_text())
    check(c["scientific_evidence_eligible"] and c["status"] == "success"
          and (c["beta0_graph_mapper"], c["beta1_graph_mapper"]) == (1, 1)
          and c["type_restricted_bottleneck_distance"] == 0.09683077738086054,
          "III-1 Conventional Circle direct result")
    check(t["scientific_evidence_eligible"] and t["status"] == "success"
          and (t["beta0_graph_mapper"], t["beta1_graph_mapper"]) == (1, 0)
          and (t["n_leaves_mapper"], t["n_branch_vertices_mapper"]) == (3, 1)
          and t["non_degree2_signature_mapper"] == t["non_degree2_signature_reference"]
          and t["type_restricted_bottleneck_distance"] == 0.14742780962289204,
          "III-1 Conventional Tripod direct result")
    audit = ct("evidence/baseline/axis3_iii2/RUN_LEVEL_REQUALIFIED.csv")
    a = ct("evidence/baseline/axis3_iii2/III2A_CELL_RESULTS.csv")
    b = ct("evidence/baseline/axis3_iii2/III2B_RECOVERY_SUMMARY.csv")
    check(len(audit) == 2208 and all(r["same_filter_contract_requalified"] == "True"
                                      and r["scientific_evidence_eligible_requalified"] == "True" for r in audit),
          "III-2 2208 run records explicitly requalified")
    check(len(a) == 50 and sum(r["joint_recovered"] == "True" for r in a) == 46,
          "III-2A 46/50 joint-recovered cells")
    check(len(b) == 72 and all(sum(int(r[k]) for r in b) == 2160
                               for k in ("n_planned", "n_attempts_executed", "n_success", "n_eligible"))
          and sum(int(r["k_joint_recovered"]) for r in b) == 1201,
          "III-2B 1201/2160 joint recoveries in 72 cells")


def verify_n1_r10_ball() -> None:
    n1a = jl("evidence/baseline/n1/N1_C_RETROSPECTIVE_LEDGER.jsonl")
    n1b = jl("evidence/baseline/n1/N1_C_DIGITS_ATTEMPT_LEDGER.jsonl")
    check((len(n1a), len(n1b)) == (604, 302)
          and Counter(r["status"] for r in n1a + n1b) == Counter(success=811, fcm_non_convergence=95),
          "N1 906 units: 811 success / 95 FCM non-convergence")
    clean = jl("evidence/baseline/r10/R10_V4_CLEAN_CONSTRUCTION_LEDGER.jsonl")
    noisy = jl("evidence/baseline/r10/R10_V4_NOISY_CONSTRUCTION_LEDGER.jsonl")
    acc = json.loads((ROOT / "evidence/baseline/r10/R10_V4_RUN_ACCOUNTING.json").read_text())
    check((len(clean), len(noisy)) == (42, 240) and sum(r["eligible"] for r in clean + noisy) == 205
          and Counter(r["status"] for r in noisy) == Counter(success=193, degenerate_output=47)
          and acc["planned"] == acc["attempted"] == 282 and acc["journal"]["n_run_completed"] == 1,
          "R10 282 attempts / 205 eligible / one completed run")
    ball = jl("evidence/baseline/ball_parameter/BALL_CONSTRUCTION_LEDGER.jsonl")
    check(len(ball) == 45 and all(r["status"] == "success" for r in ball)
          and sum(r["scientific_evidence_eligible"] for r in ball) == 44,
          "Ball parameter baseline 45 successes / 44 eligible")


def verify_e2m() -> None:
    ens = jl("evidence/baseline/e2m_ensemble/ENSEMBLE_AXIS1_CONFIRMATORY_LEDGER.jsonl")
    ii1 = jl("evidence/baseline/e2m_ii1/II1_OPTION_A_CONSTRUCTION_LEDGER.jsonl")
    iii3 = jl("evidence/baseline/e2m_iii3/III3_CIRCLE_ONLY_LEDGER.jsonl")
    check(len(ens) == 50 and Counter(r["status"] for r in ens) == Counter(success=21, SELECTION_FAILED=29)
          and sum(r["eligible"] for r in ens) == 21, "E2M Ensemble 21 eligible / 29 selection-failed")
    check(len(ii1) == 12 and sum(r["eligible"] for r in ii1) == 9
          and Counter(r["status"] for r in ii1) == Counter(success=9, fcm_non_convergence=2, degenerate_output=1),
          "E2M II-1 9 eligible / 3 typed failures")
    check(len(iii3) == 3 and all(r["eligible"] and r["same_filter_contract"]
                                and r["fidelity_distance"] == 0.0 for r in iii3),
          "E2M III-3 three eligible same-filter Circle cases")


def verify_synthesis() -> None:
    iv1 = ct("evidence/baseline/iv1_cover/dataset_associations.csv")
    want = {("unit_circle_S1", "silhouette_macro"): -.3272727272727273,
            ("unit_circle_S1", "davies_bouldin_macro"): .3272727272727273,
            ("swiss_roll_with_hole", "silhouette_macro"): .19090909090909092,
            ("swiss_roll_with_hole", "davies_bouldin_macro"): -.3272727272727273,
            ("digits_1797x64_scaled16", "silhouette_macro"): .6454545454545455,
            ("digits_1797x64_scaled16", "davies_bouldin_macro"): -.6454545454545455}
    got = {(r["dataset_id"], r["quality_name"]): float(r["spearman_primary_D_vs_Q"]) for r in iv1}
    check(got == want and all(r["n_evaluated_pairs"] == "11" and r["p_value_reported"] == "False" for r in iv1),
          "IV-1 six dataset-specific descriptive associations")
    iv2 = json.loads((ROOT / "evidence/baseline/iv2/IV2_ASSOCIATION_ROWS.json").read_text())
    check(len(iv2) == 4 and all(r["n_complete_pairs"] == 12 for r in iv2)
          and {r["rho_s_descriptive"] for r in iv2} == {-1.0, 1.0}, "IV-2 four Circle associations")
    iv3 = jl("evidence/baseline/iv3/IV3_PER_UNIT_DERIVED_LEDGER.jsonl")
    ia = json.loads((ROOT / "evidence/baseline/iv3/IV3_ANALYSIS_ACCOUNTING.json").read_text())
    check(len(iv3) == 32 and all(r["scientific_evidence_eligible"] for r in iv3)
          and len({r["design_stratum_id"] for r in iv3}) == 4
          and ia["analysis_type"] == "retrospective_analysis_only" and ia["zero_imputation_count"] == 0,
          "IV-3 32 eligible rows in four retrospective strata")
    q, d = ct("evidence/baseline/iv5/matched_quality_table.csv"), ct("evidence/baseline/iv5/spoke_distance_table.csv")
    assoc = []
    for p in sorted((ROOT / "evidence/baseline/iv5").glob("IV5_COVER_ARM_ASSOCIATIONS_*.jsonl")):
        assoc.extend(jl(p.relative_to(ROOT).as_posix()))
    check(len(q) == len(d) == 33 and len(assoc) == 6
          and sum(r["is_ib2_anchor"] == "True" and r["anchor_binary64_equal"] == "True" for r in d) == 12
          and all(r["n_evaluated_pairs"] == 11 and not r["p_value_reported"] for r in assoc),
          "IV-5 33 spokes / 12 exact anchors / six associations")
    result = json.loads((ROOT / "evidence/campaigns/ib4f/RESULT.json").read_text())
    check(result["source_ledger_sha256"] == digest(ROOT / "evidence/campaigns/c3/C3_CONSTRUCTION_LEDGER.jsonl")
          and result["n_source_rows"] == 525
          and result["likelihood_ratio_test"]["p_value"] == .13977340161971297
          and result["max_abs_residual_statistic"]["calibrated_p_value"] == .128043597820109
          and result["verdict"] == "EXPLORATORY_NO_CONFIRMATORY_INTERACTION",
          "IB4F calibrated analysis bound to C3 ledger")


def verify_legacy_baseline() -> None:
    ib1 = jl("evidence/baseline/ib1_iii4/IB1_III4_EVIDENCE_RECORDS.jsonl")
    tripod = [r for r in ib1 if r["benchmark_id"] == "branching_tripod_Y"]
    lower = [r for r in tripod if r["direction_id"] == "tripod_hat_lower_leaf"]
    check(len(ib1) == 34 and len(tripod) == 17 and all(r["scientific_evidence_eligible"] for r in ib1)
          and all(r["beta0_graph"] == 1 for r in tripod)
          and all(r["delta_leaves"] == 0 and r["delta_branch_vertices"] == 0 for r in lower),
          "IB1/III-4 34 eligible Circle/Tripod rows")
    m6 = jl("evidence/baseline/m6a2/run_ledger.jsonl")
    by = {}
    for r in m6:
        if r.get("primary_stability_distance") is not None:
            k = (r["identity"]["method_id"], r["identity"]["noise_condition_id"])
            by.setdefault(k, []).append(r["primary_stability_distance"])
    noise = sorted({r["identity"]["noise_condition_id"] for r in m6} - {"sigma_0.00"})
    check(len(m6) == 302 and len(noise) == 5
          and all(statistics.mean(by[("conventional", n)]) < statistics.mean(by[("f_mapper", n)]) for n in noise),
          "M6-A2 302-run nonzero-noise mean ordering")
    ib3 = jl("evidence/baseline/ib3_ii3/IB3_II3_CONSTRUCTION_LEDGER.jsonl")
    digits = [r for r in ib3 if r["dataset_id"] == "digits_1797x64_scaled16" and r["observed_eps"] is not None]
    low = min(r["observed_eps"] for r in digits)
    swiss = {r["observed_min_samples"]: r["silhouette_macro"] for r in ib3
             if r["dataset_id"] == "swiss_roll_with_hole" and r["observed_eps"] == 1.015739105123552
             and r["observed_min_samples"] is not None}
    check(len(ib3) == 42 and {r["graph_beta1"] for r in digits if r["observed_eps"] == low} == {0}
          and max(r["graph_beta1"] for r in digits) > 0
          and [swiss[k] for k in sorted(swiss)] == sorted(swiss.values()), "IB3/II-3 42-row response")
    ib2 = jl("evidence/baseline/ib2_ii2/construction_records.jsonl")
    check(len(ib2) == 36 and all(r["status"] == "success" for r in ib2)
          and Counter(r["dataset_id"] for r in ib2)
          == Counter({"unit_circle_S1": 12, "swiss_roll_with_hole": 12, "digits_1797x64_scaled16": 12}),
          "IB2/II-2 36 successful constructions")


def main() -> None:
    verify_manifest()
    verify_clean_boundary()
    verify_figures()
    verify_exact_mapper_distance()
    verify_quality_survival()
    verify_track_c()
    verify_c7()
    verify_c1_c3()
    verify_axis1_repairs()
    verify_c5_c6()
    verify_c8()
    verify_c9()
    verify_axis3_conventional()
    verify_n1_r10_ball()
    verify_e2m()
    verify_synthesis()
    verify_legacy_baseline()
    print("ALL_FINAL_RELEASE_CHECKS_PASSED")


if __name__ == "__main__":
    main()
