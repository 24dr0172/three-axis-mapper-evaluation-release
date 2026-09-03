"""Canonical Seed Manager for Multi-Tier Randomness Isolation."""

import hashlib
from typing import Dict, Optional


class SeedManager:
    """Deterministic, isolated seed derivation manager across six canonical namespaces."""

    NAMESPACES = {
        "data": "data_generation",
        "perturbation": "measurement_perturbation",
        "resample": "subsampling_bootstrap",
        "construction": "algorithm_internal",
        "selection": "candidate_selection",
        "evaluation": "posthoc_evaluation",
    }

    def __init__(self, master_seed: int = 42):
        if master_seed < 0:
            raise ValueError(f"master_seed must be non-negative, got {master_seed}")
        self.master_seed = master_seed

    def derive_seed(self, namespace: str, stream_index: int = 0) -> int:
        """Derive a deterministic 31-bit integer seed for a specific namespace and stream index.

        Uses SHA-256 domain separation to guarantee orthogonality between namespaces.
        """
        if namespace not in self.NAMESPACES:
            raise ValueError(f"Unknown namespace '{namespace}'. Valid namespaces: {list(self.NAMESPACES.keys())}")
        if stream_index < 0:
            raise ValueError(f"stream_index must be non-negative, got {stream_index}")

        domain_str = f"{self.master_seed}__{self.NAMESPACES[namespace]}__{stream_index}"
        hash_digest = hashlib.sha256(domain_str.encode("utf-8")).hexdigest()
        derived = int(hash_digest[:8], 16) % 2147483647  # 2^31 - 1
        return derived

    def get_run_seeds(
        self,
        replication_id: int,
        perturbation_stream: int = 0,
        construction_stream: int = 0,
        selection_stream: int = 0,
        evaluation_stream: int = 0,
    ) -> Dict[str, int]:
        """Derive all seed fields for a single experimental replication."""
        return {
            "data_seed": self.derive_seed("data", stream_index=replication_id),
            "perturbation_seed": self.derive_seed("perturbation", stream_index=perturbation_stream),
            "resample_seed": self.derive_seed("resample", stream_index=replication_id),
            "construction_seed": self.derive_seed("construction", stream_index=construction_stream),
            "selection_seed": self.derive_seed("selection", stream_index=selection_stream),
            "evaluation_seed": self.derive_seed("evaluation", stream_index=evaluation_stream),
        }
