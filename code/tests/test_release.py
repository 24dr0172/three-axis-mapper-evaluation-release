from __future__ import annotations

import importlib.util
import itertools
import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERIFY = ROOT / "code/verification/verify_release.py"


def load_verifier():
    spec = importlib.util.spec_from_file_location("verify_release", VERIFY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FinalReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verifier = load_verifier()

    def test_manifest(self):
        self.verifier.verify_manifest(ignore_runtime_caches=True)

    def test_clean_boundary(self):
        self.verifier.verify_clean_boundary(ignore_runtime_caches=True)

    def test_figures(self):
        self.verifier.verify_figures()

    def test_quality_survival(self):
        self.verifier.verify_quality_survival()

    def test_track_c(self):
        self.verifier.verify_track_c()

    def test_c7(self):
        self.verifier.verify_c7()

    def test_c1_through_c3(self):
        self.verifier.verify_c1_c3()

    def test_axis1_repairs(self):
        self.verifier.verify_axis1_repairs()

    def test_c5_c6(self):
        self.verifier.verify_c5_c6()

    def test_c8(self):
        self.verifier.verify_c8()

    def test_c9(self):
        self.verifier.verify_c9()

    def test_axis3_conventional(self):
        self.verifier.verify_axis3_conventional()

    def test_n1_r10_ball(self):
        self.verifier.verify_n1_r10_ball()

    def test_e2m(self):
        self.verifier.verify_e2m()

    def test_synthesis(self):
        self.verifier.verify_synthesis()

    def test_legacy_baseline(self):
        self.verifier.verify_legacy_baseline()

    def test_no_empty_directories(self):
        """Empty directories are structural residue and must not ship."""
        root = self.verifier.ROOT
        empty = [p.relative_to(root).as_posix()
                 for p in root.rglob("*") if p.is_dir() and not any(p.iterdir())]
        self.assertEqual(empty, [], f"empty directories present: {empty[:5]}")

    def test_authorship_disclosure_present(self):
        """The concise authorship and AI-assistance disclosure is present."""
        root = self.verifier.ROOT
        text = (root / "README.md").read_text(errors="replace")
        text = " ".join(text.lower().split())
        for token in ("principal investigators", "conceived the study",
                      "designed the evaluation study and analyses",
                      "directed the research programme", "cross-checking the code",
                      "verifying numerical reproducibility", "AI"):
            self.assertIn(token.lower(), text, f"disclosure missing {token!r}")
        manuscript_files = [p for p in (root / "manuscript").rglob("*") if p.is_file()]
        self.assertTrue(manuscript_files)
        self.assertTrue(all("figures" in p.relative_to(root / "manuscript").parts
                            for p in manuscript_files))

    def test_ledger_bound_contracts_are_shipped(self):
        """Every contract_sha256 bound by a ledger row must resolve against a shipped contract."""
        import json
        root = self.verifier.ROOT
        bound = {}
        for ledger in sorted((root / "evidence/campaigns").rglob("*.jsonl")):
            with ledger.open() as stream:
                for line in stream:
                    if line.strip():
                        row = json.loads(line)
                        if row.get("contract_id") and row.get("contract_sha256"):
                            bound[row["contract_id"]] = row["contract_sha256"]
                        break
        self.assertTrue(bound, "no ledger-bound contracts discovered")
        bindings = json.loads((root / "contracts/CONTRACT_SOURCE_BINDINGS.json").read_text())
        for contract_id, digest in bound.items():
            target = root / "contracts" / f"{contract_id}.json"
            self.assertTrue(target.is_file(), f"missing shipped contract {contract_id}")
            release_digest = self.verifier.sha256(target)
            normalized = bindings.get(contract_id, {})
            self.assertTrue(release_digest == digest or
                            (normalized.get("executed_sha256") == digest and
                             normalized.get("release_sha256") == release_digest),
                            f"contract digest mismatch for {contract_id}")

    def test_installable_src_layout_imports(self):
        """The public package imports from its standard src layout."""
        sys.path.insert(0, str(ROOT / "src"))
        try:
            import mapper_framework
            self.assertEqual(mapper_framework.__version__, "3.0.0")
            try:
                from mapper_framework.ball_mapper import BallMapper
                from mapper_framework.conventional import ConventionalMapper
                from mapper_framework.f_mapper import FMapper
                self.assertTrue(all(x is not None for x in (BallMapper, ConventionalMapper, FMapper)))
            except ImportError as e:
                if "sklearn" in str(e):
                    raise unittest.SkipTest("sklearn runtime dependency not installed in execution environment")
                raise
        finally:
            sys.path.pop(0)

    def test_persistence_modules_import(self):
        """Advertised persistence modules import after the documented install."""
        sys.path.insert(0, str(ROOT / "src"))
        try:
            import mapper_framework.extended_persistence
            import mapper_framework.stability_metrics
            self.assertTrue(hasattr(mapper_framework.extended_persistence,
                                    "compute_graph_extended_persistence"))
            self.assertTrue(hasattr(mapper_framework.stability_metrics, "METRIC_REGISTRY"))
        except ImportError as e:
            missing = str(e).lower()
            if "gudhi" in missing:
                raise unittest.SkipTest("gudhi is not installed in this execution environment")
            if "sklearn" in missing:
                raise unittest.SkipTest("sklearn runtime dependency not installed in execution environment")
            raise
        finally:
            sys.path.pop(0)


    def test_persistence_metric_terminology_and_release_version_explained(self):
        """Publication-facing metadata does not conflate bottleneck and Reeb interleaving distances."""
        registry = (ROOT / "src/mapper_framework/stability_metrics.py").read_text().lower()
        rules = (ROOT / "src/mapper_framework/acceptance_rules.py").read_text().lower()
        self.assertNotIn("bottleneck interleaving distance", registry)
        self.assertNotIn("bottleneck interleaving distance", rules)
        readme = (ROOT / "README.md").read_text().lower()
        self.assertIn("release is version **4.1.0**", readme)
        self.assertIn("package/api remains version **3.0.0**", readme)

    def test_numerical_reporting_contract(self):
        """Cross-environment float drift is bounded without weakening exact counts."""
        contract = json.loads((ROOT / "contracts/NUMERICAL_REPORTING_CONTRACT.json").read_text())
        tolerance = contract["cross_environment_absolute_tolerance"]
        self.assertTrue(math.isclose(1.0, 1.0 + 1e-15, rel_tol=0, abs_tol=tolerance))
        self.assertFalse(math.isclose(1.0, 1.0 + 1e-12, rel_tol=0, abs_tol=tolerance))

    def test_archive_filenames_are_ascii_portable(self):
        """Every release path survives conservative ZIP extraction tools."""
        non_ascii = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                     if not p.relative_to(ROOT).as_posix().isascii()]
        self.assertEqual(non_ascii, [])

    def test_exact_joint_mapper_distance_and_hash_gates(self):
        """The Axis-I endpoint is joint across bins and fails closed on provenance."""
        sys.path.insert(0, str(ROOT / "src"))
        try:
            from mapper_framework.support.d_common_id import (
                d_common_id,
                hash_filter_definition,
                hash_realized_cover,
            )
            bins_a = {
                0: {0: 0, 1: 0, 2: 0, 3: 1},
                1: {0: 0, 1: 0, 2: 0, 3: 1},
            }
            bins_b = {
                0: {0: 0, 1: 0, 2: 1, 3: 0},
                1: {0: 0, 1: 1, 2: 0, 3: 1},
            }
            ids = {0, 1, 2, 3}
            cover_hash = hash_realized_cover(bins_a, ids)
            self.assertEqual(cover_hash, hash_realized_cover(bins_b, ids))
            filter_hash = hash_filter_definition("f(x,y) = y")
            keywords = dict(
                common_ids=ids,
                cover_id_A="fixed",
                cover_id_B="fixed",
                filter_id_A="height",
                filter_id_B="height",
                cover_hash_A=cover_hash,
                cover_hash_B=cover_hash,
                filter_hash_A=filter_hash,
                filter_hash_B=filter_hash,
            )
            result = d_common_id(bins_a, bins_b, **keywords)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["distance"], 0.5)
            self.assertEqual(result["optimizer"], "scipy_milp_exact_joint")
            missing = d_common_id(
                bins_a,
                bins_b,
                common_ids=ids,
                cover_id_A="fixed",
                cover_id_B="fixed",
                filter_id_A="height",
                filter_id_B="height",
            )
            self.assertEqual(missing["reason"], "missing_cover_hash_A")
            corrupted = d_common_id(
                bins_a, bins_b, **{**keywords, "cover_hash_B": "0" * 64}
            )
            self.assertEqual(corrupted["reason"], "cover_mismatch")
        finally:
            sys.path.pop(0)

    def test_exact_joint_optimizer_matches_exhaustive_small_case(self):
        """An independent permutation enumeration agrees with the MILP."""
        sys.path.insert(0, str(ROOT / "src"))
        try:
            from mapper_framework.support.d_common_id import (
                d_common_id,
                hash_filter_definition,
                hash_realized_cover,
            )
            a = {0: {0: 0, 1: 0, 2: 1}, 1: {0: 0, 1: 1, 2: 1}}
            b = {0: {0: 1, 1: 0, 2: 0}, 1: {0: 0, 1: 0, 2: 1}}
            ids = {0, 1, 2}
            exhaustive = 1.0
            for p0, p1 in itertools.product(itertools.permutations((0, 1)), repeat=2):
                maps = ({0: p0[0], 1: p0[1]}, {0: p1[0], 1: p1[1]})
                mismatches = sum(
                    a[0][x] != maps[0][b[0][x]]
                    or a[1][x] != maps[1][b[1][x]]
                    for x in ids
                )
                exhaustive = min(exhaustive, mismatches / len(ids))
            cover_hash = hash_realized_cover(a, ids)
            result = d_common_id(
                a,
                b,
                common_ids=ids,
                cover_id_A="c",
                cover_id_B="c",
                filter_id_A="f",
                filter_id_B="f",
                cover_hash_A=cover_hash,
                cover_hash_B=cover_hash,
                filter_hash_A=hash_filter_definition("f"),
                filter_hash_B=hash_filter_definition("f"),
            )
            self.assertEqual(result["distance"], exhaustive)
        finally:
            sys.path.pop(0)


if __name__ == "__main__":
    unittest.main()
