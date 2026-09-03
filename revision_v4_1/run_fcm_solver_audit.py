#!/usr/bin/env python3
"""FCM_SOLVER_SENSITIVITY_AUDIT_V4_1

New robustness arm. Does not edit C3/C8 ledgers or hashes.
Varies only the FCM stopping rule. Same frozen datasets, linspace init,
fuzzifier=2.0. Downstream Mapper quality/fidelity are not recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REV = Path(__file__).resolve().parent
OUT = REV / "derived"
TEX = ROOT / "manuscript" / "tables"
CAMPAIGN = "FCM_SOLVER_SENSITIVITY_AUDIT_V4_1"
ENV_CAMPAIGN = "ENV_EQUIVALENCE_CHECK_V4_1"
TOLS = (("1e-4", 1e-4), ("1e-5", 1e-5), ("1e-6", 1e-6), ("1e-7", 1e-7))
FROZEN_TOL = 1e-7
FROZEN_MAX_ITER = 300
DIAG_MAX_ITER = 1000
ALPHA = 2.0
EXPECTED_UNITS = 1055
EXPECTED_UNIQUE_JOBS = 1035
EXPECTED_DIGITS_POINTS_SHA256 = (
    "23a53393488b04f92efc45ebc92767e2dcfdf206ba37a928f497d261e42267d2"
)
SKLEARN_CANDIDATES = [
    ROOT.parent / "final_release_candidate_2026-08-28/environment_lock/site-packages",
    ROOT.parent / "archive/final_release_reconstructed_2026-08-28/environment_lock/site-packages",
]


def sha_arr(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f8").tobytes(order="C")).hexdigest()


def ensure_sklearn() -> str:
    try:
        import sklearn  # noqa: F401

        return sklearn.__version__
    except Exception:
        pass
    for path in SKLEARN_CANDIDATES:
        if path.is_dir():
            sys.path.insert(0, str(path))
            try:
                import sklearn  # noqa: F401

                return sklearn.__version__
            except Exception:
                sys.path.pop(0)
                for k in list(sys.modules):
                    if k == "sklearn" or k.startswith("sklearn."):
                        del sys.modules[k]
    raise RuntimeError(
        "Digits FCM audit requires scikit-learn to rebuild the frozen PCA lens"
    )


def run_fcm_vec(y, c, alpha=2.0, max_iter=300, stop_tol=1e-7):
    """Sealed FCM membership updates, vectorized. Returns (ok_finite, diffs)."""
    y = np.asarray(y, dtype=float).ravel()
    n = len(y)
    if c == 1:
        return True, [0.0]
    if float(np.max(y)) == float(np.min(y)):
        return False, []
    centroids = np.linspace(float(np.min(y)), float(np.max(y)), c)
    exp = 2.0 / (alpha - 1.0)

    def membership(centroids):
        dists = np.abs(y[:, None] - centroids[None, :])
        zero = dists == 0.0
        any_zero = zero.any(axis=1)
        u = np.zeros((n, c), dtype=float)
        if np.any(~any_zero):
            d = dists[~any_zero]
            ratio = (d[:, :, None] / d[:, None, :]) ** exp
            u[~any_zero] = 1.0 / ratio.sum(axis=2)
        for i in np.flatnonzero(any_zero):
            z = np.flatnonzero(zero[i])
            u[i, z] = 1.0 / len(z)
        return u

    u = membership(centroids)
    diffs = []
    for _it in range(1, max_iter + 1):
        u_prev = u
        u_alpha = u ** alpha
        denom = np.sum(u_alpha, axis=0)
        denom = np.where(denom == 0.0, 1e-12, denom)
        centroids = np.sum(u_alpha * y[:, None], axis=0) / denom
        u = membership(centroids)
        diff = float(np.max(np.abs(u - u_prev)))
        diffs.append(diff)
        if diff < stop_tol:
            break
    if not np.all(np.isfinite(u)):
        return False, diffs
    if not np.all(np.abs(u.sum(axis=1) - 1.0) <= 1e-4):
        return False, diffs
    return True, diffs


def first_hit(diffs, tol) -> int | None:
    for i, d in enumerate(diffs, start=1):
        if d < tol:
            return i
    return None


def classify(ok_finite, diffs):
    out = {}
    prefix = diffs[:FROZEN_MAX_ITER]
    for name, tol in TOLS:
        hit = first_hit(prefix, tol)
        out[f"converged_tol_{name}_maxiter_300"] = bool(ok_finite and hit is not None)
        out[f"first_iter_tol_{name}"] = hit
    out["n_iter_recorded"] = len(diffs)
    out["last_diff"] = diffs[-1] if diffs else None
    out["last_diff_at_300"] = prefix[-1] if prefix else None
    out["converged_frozen_1e-7_300"] = bool(
        ok_finite and first_hit(prefix, FROZEN_TOL) is not None
    )
    return out


def jl(rel: str) -> list[dict]:
    return [json.loads(x) for x in (ROOT / rel).read_text().splitlines() if x.strip()]


def load_lens_fn():
    sklearn_version = ensure_sklearn()
    sys.path.insert(0, str(ROOT / "src"))
    from mapper_framework.dataset_generators import (
        generate_branching_tripod,
        generate_clean_circle,
        generate_swiss_roll_with_hole,
    )

    digits_lens = None
    digits_meta = {"sklearn_version": sklearn_version}

    def digits():
        nonlocal digits_lens
        if digits_lens is not None:
            return digits_lens
        artifact = np.load(ROOT / "data/N1_DIGITS_PCA_ARTIFACT.npz")
        from sklearn.datasets import load_digits

        x = np.ascontiguousarray(load_digits().data.astype("<f8", copy=True) / 16.0)
        got = sha_arr(x)
        if got != EXPECTED_DIGITS_POINTS_SHA256:
            raise RuntimeError(f"digits points digest mismatch: {got}")
        digits_lens = (x - artifact["pca_mean"]) @ artifact["signed_pc1_component"]
        digits_meta["points_sha256"] = got
        digits_meta["n"] = int(len(digits_lens))
        return digits_lens

    cache = {}

    def lens_for(row: dict):
        ds = row["dataset_id"]
        seed = row.get("data_seed")
        key = (ds, seed)
        if key in cache:
            return cache[key]
        if ds == "swiss_roll_with_hole":
            y = generate_swiss_roll_with_hole(
                N=2500, data_seed=int(seed), resample_to_exact_N=False, lens_id="radial_xz"
            ).lens
        elif ds in {"unit_circle", "unit_circle_S1"}:
            n = int(row.get("n_observations") or 1000)
            y = generate_clean_circle(N=n, data_seed=int(seed), sampling="uniform").lens
        elif ds in {"branching_tripod_y", "branching_tripod_Y"}:
            n = int(row.get("n_observations") or 1000)
            y = generate_branching_tripod(N=n, data_seed=int(seed)).lens
        elif ds == "digits_1797x64_scaled16":
            y = digits()
        else:
            raise ValueError(ds)
        cache[key] = np.asarray(y, dtype=float).ravel()
        return cache[key]

    return lens_for, digits_meta


def job_key(plan: dict):
    seed = plan.get("data_seed")
    return (plan["dataset_id"], seed if seed is not None else "NA", int(plan["n_intervals"]))


def sealed_converged(y, c, max_iter=300, tol=1e-7) -> bool:
    from mapper_framework.f_mapper import run_fcm_1d

    _u, _cent, conv = run_fcm_1d(y, c, alpha=ALPHA, tol=tol, max_iter=max_iter, seed=42)
    return bool(conv)


def smoke_vs_sealed(jobs, lens_for, n=6) -> dict:
    """Compare vectorized FCM to sealed run_fcm_1d on a few unique jobs."""
    want = {
        ("C3", "swiss_roll_with_hole", 5),
        ("C3", "swiss_roll_with_hole", 20),
        ("C3", "digits_1797x64_scaled16", 5),
        ("C3", "digits_1797x64_scaled16", 20),
        ("C8", "unit_circle", 10),
        ("C8", "branching_tripod_y", 10),
    }
    seen_triples = set()
    checks = []
    for campaign, uid, plan, led in jobs:
        ds = plan["dataset_id"]
        c = int(plan["n_intervals"])
        triple = (campaign, ds, c)
        if triple not in want or triple in seen_triples:
            continue
        seen_triples.add(triple)
        y = lens_for(plan)
        t0 = time.time()
        ok, diffs = run_fcm_vec(y, c, alpha=ALPHA, max_iter=FROZEN_MAX_ITER, stop_tol=FROZEN_TOL)
        vec_conv = bool(ok and first_hit(diffs, FROZEN_TOL) is not None)
        t_vec = time.time() - t0
        t0 = time.time()
        sealed = sealed_converged(y, c)
        t_sealed = time.time() - t0
        rec = {
            "unit_id": uid,
            "dataset_id": ds,
            "n_intervals": c,
            "vectorized_converged": vec_conv,
            "sealed_converged": sealed,
            "frozen_fcm_converged": bool(led.get("fcm_converged")),
            "match_sealed": vec_conv == sealed,
            "match_frozen": vec_conv == bool(led.get("fcm_converged")),
            "seconds_vectorized": round(t_vec, 4),
            "seconds_sealed": round(t_sealed, 4),
        }
        checks.append(rec)
        print("smoke", rec, flush=True)
        if len(checks) >= n:
            break
    if not checks or not all(r["match_sealed"] and r["match_frozen"] for r in checks):
        raise SystemExit(f"vectorized FCM disagrees with sealed/frozen on smoke checks: {checks}")
    return {"n_checks": len(checks), "all_match": True, "checks": checks}


def write_env_skip() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    py = sys.version.split()[0]
    obj = {
        "campaign_id": ENV_CAMPAIGN,
        "status": "skipped_interpreter_unavailable",
        "reason": (
            "CPython 3.13.5 is not provisioned on this host; only "
            f"{py} is available. Plan section 6.2: not blocking if 3.12 "
            "cannot be paired with 3.13.5."
        ),
        "python_available": [py],
        "python_missing": ["3.13.5"],
        "mapper_constructions_generated": 0,
        "label": "equivalence check skipped; provenance disclosure retained",
    }
    path = OUT / f"{ENV_CAMPAIGN}_SKIP.json"
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    return path


DS_SHORT = {
    "digits_1797x64_scaled16": "Digits",
    "swiss_roll_with_hole": "Swiss",
    "branching_tripod_y": "Tripod",
    "unit_circle": "Circle",
}


def write_tex(summary: dict) -> Path:
    TEX.mkdir(parents=True, exist_ok=True)
    nc = summary["frozen_nonconverged"]
    rec = summary["reclassified_to_converged_vs_frozen"]
    by = summary["by_tol_nonconverged"]
    rec1000 = summary["frozen_fail_then_converged_maxiter_1000"]
    remain1000 = nc - rec1000
    lines = [
        r"% Auto-generated by run_fcm_solver_audit.py. Do not hand-edit.",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{FCM solver-sensitivity audit (\texttt{FCM\_SOLVER\_SENSITIVITY\_AUDIT\_V4\_1}) on the frozen C3/C8 lens values. Same datasets, deterministic \texttt{linspace} initialization, and fuzzifier $2.0$; only the stopping rule varies. This is a robustness/sensitivity reclassification, not a replacement for the frozen confirmatory campaigns. Downstream quality and fidelity summaries are not recomputed.}",
        r"\label{tab:fcmsolv}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"Source & Data & $c$ & $n$ & Frozen nonconv. & $10^{-4}$ & $10^{-5}$ & $10^{-6}$ \\",
        r"\midrule",
    ]
    for b in summary["buckets"]:
        ds = DS_SHORT.get(b["dataset_id"], b["dataset_id"].replace("_", r"\_"))
        lines.append(
            f"{b['source_campaign']} & {ds} & {b['n_intervals']} & {b['n_units']} & "
            f"{b['frozen_nonconverged']} & {b['nonconverged_1e-4']} & "
            f"{b['nonconverged_1e-5']} & {b['nonconverged_1e-6']} \\\\"
        )
    lines.extend(
        [
            r"\midrule",
            (
                f"All &  &  & {summary['n_units']} & {nc} & "
                f"{by['1e-4']} & {by['1e-5']} & {by['1e-6']} \\\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\par\vspace{0.4em}{\scriptsize Frozen non-convergence is "
                r"$\mathrm{tol}=10^{-7}$, $\mathrm{max\_iter}=300$. Columns "
                r"$10^{-k}$ count records that remain non-convergent at that "
                f"tolerance and 300 iterations. Of {nc} frozen failures, "
                f"{rec['1e-4']}/{rec['1e-5']}/{rec['1e-6']} reclassify as "
                r"convergent at $10^{-4}$/$10^{-5}$/$10^{-6}$; "
                f"{rec1000} of the frozen failures converge at "
                r"$\mathrm{tol}=10^{-7}$ if $\mathrm{max\_iter}=1000$ "
                f"({remain1000} remain). "
                r"Audit $10^{-7}/300$ matches the ledger "
                r"\texttt{fcm\_converged} flag on every record.}"
            ),
            r"\end{table}",
            "",
        ]
    )
    path = TEX / "fcm_solver_sensitivity.tex"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lens_for, digits_meta = load_lens_fn()
    c3_plan = {r["unit_id"]: r for r in jl("evidence/campaigns/c3/PLAN.jsonl")}
    c3_led = {r["unit_id"]: r for r in jl("evidence/campaigns/c3/C3_CONSTRUCTION_LEDGER.jsonl")}
    c8_plan = {r["unit_id"]: r for r in jl("evidence/campaigns/c8/PLAN.jsonl")}
    c8_led = {r["unit_id"]: r for r in jl("evidence/campaigns/c8/C8_CONSTRUCTION_LEDGER.jsonl")}

    jobs = []
    for uid, plan in c3_plan.items():
        jobs.append(("C3", uid, plan, c3_led[uid]))
    for uid, plan in c8_plan.items():
        jobs.append(("C8", uid, plan, c8_led[uid]))
    if len(jobs) != EXPECTED_UNITS:
        raise SystemExit(f"expected {EXPECTED_UNITS} C3+C8 units, got {len(jobs)}")

    smoke = smoke_vs_sealed(jobs, lens_for)
    if args.smoke:
        print(json.dumps(smoke, indent=2))
        print("SMOKE_OK")
        return

    grouped: dict = defaultdict(list)
    for item in jobs:
        grouped[job_key(item[2])].append(item)
    if len(grouped) != EXPECTED_UNIQUE_JOBS:
        raise SystemExit(f"expected {EXPECTED_UNIQUE_JOBS} unique FCM jobs, got {len(grouped)}")

    jsonl_path = OUT / f"{CAMPAIGN}.jsonl"
    rows = []
    mismatches = 0
    t0 = time.time()
    with jsonl_path.open("w") as handle:
        for i, (key, items) in enumerate(grouped.items(), start=1):
            frozen_flags = {bool(it[3].get("fcm_converged")) for it in items}
            if len(frozen_flags) != 1:
                raise SystemExit(f"mixed frozen FCM flags for job {key}: {frozen_flags}")
            frozen = frozen_flags.pop()
            plan0 = items[0][2]
            y = lens_for(plan0)
            c = int(plan0["n_intervals"])
            max_iter = DIAG_MAX_ITER if not frozen else FROZEN_MAX_ITER
            ok_finite, diffs = run_fcm_vec(
                y, c, alpha=ALPHA, max_iter=max_iter, stop_tol=FROZEN_TOL
            )
            cls = classify(ok_finite, diffs)
            extra = {}
            if not frozen:
                extra["converged_tol_1e-7_maxiter_1000"] = bool(
                    ok_finite and first_hit(diffs, FROZEN_TOL) is not None
                )
                extra["first_iter_tol_1e-7_maxiter_1000"] = first_hit(diffs, FROZEN_TOL)
                extra["n_iter_maxiter_1000"] = len(diffs)
            match = cls["converged_frozen_1e-7_300"] == frozen
            if not match:
                mismatches += 1
            for campaign, uid, plan, led in items:
                rec = {
                    "campaign_id": CAMPAIGN,
                    "source_campaign": campaign,
                    "unit_id": uid,
                    "dataset_id": plan["dataset_id"],
                    "n_intervals": c,
                    "threshold": plan.get("threshold"),
                    "data_seed": plan.get("data_seed"),
                    "n_observations": int(len(y)),
                    "unique_job": {"dataset_id": key[0], "data_seed": key[1], "n_intervals": key[2]},
                    "frozen_fcm_converged": frozen,
                    "frozen_status": led.get("status"),
                    "audit_matches_frozen_1e-7_300": match,
                    **cls,
                    **extra,
                    "label": "robustness_sensitivity_not_confirmatory",
                }
                rows.append(rec)
                handle.write(json.dumps(rec) + "\n")
            if i % 25 == 0 or i == len(grouped):
                elapsed = time.time() - t0
                print(
                    f"progress jobs {i}/{len(grouped)} units {len(rows)} "
                    f"mismatches_vs_frozen={mismatches} elapsed_s={elapsed:.1f}",
                    flush=True,
                )

    if mismatches:
        raise SystemExit(
            f"audit 1e-7/300 disagrees with frozen FCM flag on {mismatches} units"
        )

    def bucket_key(r):
        return (r["source_campaign"], r["dataset_id"], r["n_intervals"])

    summary_rows = []
    grouped_rows = defaultdict(list)
    for r in rows:
        grouped_rows[bucket_key(r)].append(r)
    for key, grp in sorted(grouped_rows.items()):
        rec = {
            "source_campaign": key[0],
            "dataset_id": key[1],
            "n_intervals": key[2],
            "n_units": len(grp),
            "frozen_nonconverged": sum(not r["frozen_fcm_converged"] for r in grp),
        }
        for name, _tol in TOLS:
            k = f"converged_tol_{name}_maxiter_300"
            rec[f"nonconverged_{name}"] = sum(not r[k] for r in grp)
            rec[f"reclassified_to_converged_{name}"] = sum(
                (not r["frozen_fcm_converged"]) and r[k] for r in grp
            )
        rec["converged_1e-7_maxiter_1000_among_frozen_fail"] = sum(
            r.get("converged_tol_1e-7_maxiter_1000") is True for r in grp
        )
        summary_rows.append(rec)

    primary = [r for r in rows if r["source_campaign"] == "C8" and r["n_intervals"] == 10
               and abs(float(r["threshold"]) - 0.10) < 1e-12]
    primary_by_ds = defaultdict(list)
    for r in primary:
        primary_by_ds[r["dataset_id"]].append(r)

    def primary_block(grp):
        n = len(grp)
        frozen_success = sum(r["frozen_status"] == "success" for r in grp)
        frozen_fail = sum(not r["frozen_fcm_converged"] for r in grp)
        return {
            "n": n,
            "frozen_success": frozen_success,
            "frozen_fcm_nonconvergence": frozen_fail,
            "fcm_converged_if_tol": {
                name: sum(r[f"converged_tol_{name}_maxiter_300"] for r in grp)
                for name, _ in TOLS
            },
            "frozen_fail_then_converged_maxiter_1000": sum(
                r.get("converged_tol_1e-7_maxiter_1000") is True for r in grp
            ),
        }

    overall = {
        "campaign_id": CAMPAIGN,
        "mapper_constructions_generated": 0,
        "label": "robustness/sensitivity audit; not a replacement for C3/C8 confirmatory campaigns",
        "n_units": len(rows),
        "n_unique_fcm_jobs": len(grouped),
        "frozen_match_count": len(rows) - mismatches,
        "tols": [name for name, _ in TOLS],
        "frozen_nonconverged": sum(not r["frozen_fcm_converged"] for r in rows),
        "by_tol_nonconverged": {
            name: sum(not r[f"converged_tol_{name}_maxiter_300"] for r in rows)
            for name, _ in TOLS
        },
        "reclassified_to_converged_vs_frozen": {
            name: sum((not r["frozen_fcm_converged"]) and r[f"converged_tol_{name}_maxiter_300"] for r in rows)
            for name, _ in TOLS
        },
        "frozen_fail_then_converged_maxiter_1000": sum(
            r.get("converged_tol_1e-7_maxiter_1000") is True for r in rows
        ),
        "buckets": summary_rows,
        "c8_primary_cell_c10_tau0.10": {
            ds: primary_block(grp) for ds, grp in sorted(primary_by_ds.items())
        },
        "downstream_quality_fidelity": (
            "Not recomputed. Newly convergent units at a looser stopping rule "
            "would enter the construction pipeline; that requires new Mapper "
            "constructions, which this arm does not generate. C3/C8 quality "
            "and fidelity summaries remain those of the frozen solver criterion."
        ),
        "smoke_vs_sealed": smoke,
        "digits_lens": digits_meta,
        "python": sys.version.split()[0],
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    (OUT / f"{CAMPAIGN}_SUMMARY.json").write_text(json.dumps(overall, indent=2, sort_keys=True) + "\n")
    write_tex(overall)
    write_env_skip()
    print(json.dumps({k: overall[k] for k in (
        "n_units", "n_unique_fcm_jobs", "frozen_nonconverged", "by_tol_nonconverged",
        "reclassified_to_converged_vs_frozen", "frozen_fail_then_converged_maxiter_1000",
        "c8_primary_cell_c10_tau0.10",
    )}, indent=2))
    print("AUDIT_WRITTEN")


if __name__ == "__main__":
    main()
