"""Synthetic Point Cloud Dataset Generators and Seed Hierarchy Controls."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class SyntheticDataset:
    """Immutable representation of a synthetic point cloud dataset."""

    dataset_id: str
    space_name: str
    points: np.ndarray
    lens: np.ndarray
    lens_id: str
    data_seed: Optional[int]
    perturbation_seed: Optional[int]
    resample_seed: Optional[int]
    point_order: Optional[Tuple[int, ...]]
    noise_sigma: float
    is_perturbed: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


def generate_clean_circle(
    N: int = 100,
    radius: float = 1.0,
    data_seed: Optional[int] = None,
    sampling: str = "uniform",
) -> SyntheticDataset:
    """Generate clean finite sample X_0 from Unit Circle S^1.

    Parameters
    ----------
    N : int
        Sample size.
    radius : float
        Radius of circle (default 1.0).
    data_seed : Optional[int]
        Independent random seed for data generation.
    sampling : str
        'uniform' (random uniform angles) or 'regular' (evenly spaced angles).
    """
    if N <= 0:
        raise ValueError(f"Sample size N must be > 0, got {N}")

    rng = np.random.default_rng(data_seed)
    if sampling == "regular":
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    else:
        theta = rng.uniform(0, 2 * np.pi, size=N)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    points = np.column_stack([x, y])
    lens = y.copy()  # Standard height lens

    return SyntheticDataset(
        dataset_id=f"clean_circle_N{N}_seed{data_seed}",
        space_name="unit_circle_S1",
        points=points,
        lens=lens,
        lens_id="height",
        data_seed=data_seed,
        perturbation_seed=None,
        resample_seed=None,
        point_order=tuple(range(N)),
        noise_sigma=0.0,
        is_perturbed=False,
        metadata={
            "radius": radius,
            "sampling": sampling,
            "dimension": 2,
            "N": N,
            "canonical_reeb_reference": "circle_analytic_reeb",
        },
    )


def generate_branching_tripod(
    N: int = 99,
    data_seed: Optional[int] = None,
) -> SyntheticDataset:
    """Generate clean finite sample X_0 from Branching Y-Graph (Tripod).

    Parameters
    ----------
    N : int
        Total sample size. Distributed equally among the 3 arms.
    data_seed : Optional[int]
        Independent random seed.
    """
    if N < 3:
        raise ValueError(f"Tripod sample size N must be >= 3, got {N}")

    rng = np.random.default_rng(data_seed)
    n_per_arm = N // 3
    remainder = N % 3

    # Arm 1: (0, 0) to (-1, 1)
    t1 = rng.uniform(0, 1, size=n_per_arm + (1 if remainder > 0 else 0))
    p1 = np.column_stack([-t1, t1])

    # Arm 2: (0, 0) to (1, 1)
    t2 = rng.uniform(0, 1, size=n_per_arm + (1 if remainder > 1 else 0))
    p2 = np.column_stack([t2, t2])

    # Arm 3: (0, 0) to (0, -1)
    t3 = rng.uniform(0, 1, size=n_per_arm)
    p3 = np.column_stack([np.zeros_like(t3), -t3])

    points = np.vstack([p1, p2, p3])
    lens = points[:, 1].copy()  # Height lens y in [-1, 1]

    return SyntheticDataset(
        dataset_id=f"clean_tripod_N{len(points)}_seed{data_seed}",
        space_name="branching_tripod_Y",
        points=points,
        lens=lens,
        lens_id="height",
        data_seed=data_seed,
        perturbation_seed=None,
        resample_seed=None,
        point_order=tuple(range(len(points))),
        noise_sigma=0.0,
        is_perturbed=False,
        metadata={
            "dimension": 2,
            "N": len(points),
            "n_arms": 3,
            "canonical_reeb_reference": "tripod_exact_pl_reeb",
        },
    )


def generate_swiss_roll_with_hole(
    N: int = 2500,
    data_seed: Optional[int] = None,
    resample_to_exact_N: bool = False,
    lens_id: str = "radial_xz",
) -> SyntheticDataset:
    """Generate clean finite sample X_0 from Swiss Roll with rectangular hole removed.

    Parameters
    ----------
    N : int
        Base sample size N_swiss_base (default 2500), or exact post-hole sample size if resample_to_exact_N=True.
    data_seed : Optional[int]
        Independent random seed for data generation.
    resample_to_exact_N : bool
        If False (canonical legacy-anchored protocol), draws N base points and excises points in the hole
        [2.5*pi, 3.5*pi] x [7, 14], returning the realized post-hole points (N_swiss_realized).
        If True, resamples until exactly N post-hole points are collected.
    lens_id : str
        Lens identifier. "radial_xz" for approved M6-A2 ambient radial lens f(x,y,z)=sqrt(x^2+z^2),
        or "unrolled_u" for historical M6-A latent coordinate.
    """
    if N <= 0:
        raise ValueError(f"Sample size N must be > 0, got {N}")

    rng = np.random.default_rng(data_seed)
    u_min, u_max = 1.5 * np.pi, 4.5 * np.pi
    v_min, v_max = 0.0, 21.0

    if not resample_to_exact_N:
        # Canonical protocol: sample N base points, then excise hole
        u_base = rng.uniform(u_min, u_max, size=N)
        v_base = rng.uniform(v_min, v_max, size=N)

        hole_mask = (u_base >= 2.5 * np.pi) & (u_base <= 3.5 * np.pi) & (v_base >= 7.0) & (v_base <= 14.0)
        valid_idx = np.where(~hole_mask)[0]

        u_valid = u_base[valid_idx]
        v_valid = v_base[valid_idx]
        x = u_valid * np.cos(u_valid)
        y = v_valid
        z = u_valid * np.sin(u_valid)

        points = np.column_stack([x, y, z])
        if lens_id == "radial_xz":
            lens = np.sqrt(x ** 2 + z ** 2)
        elif lens_id == "unrolled_u":
            lens = u_valid.copy()
        elif lens_id == "height":
            lens = y.copy()
        else:
            raise ValueError(
                f"Unsupported lens_id '{lens_id}' for Swiss Roll with hole. "
                f"Supported: 'radial_xz', 'unrolled_u', 'height'"
            )
        n_realized = len(points)

        return SyntheticDataset(
            dataset_id=f"swiss_roll_hole_Nbase{N}_realized{n_realized}_seed{data_seed}",
            space_name="swiss_roll_with_hole",
            points=points,
            lens=lens,
            lens_id=lens_id,
            data_seed=data_seed,
            perturbation_seed=None,
            resample_seed=None,
            point_order=tuple(range(n_realized)),
            noise_sigma=0.0,
            is_perturbed=False,
            metadata={
                "dimension": 3,
                "sample_size_requested": N,
                "sample_size_realized": n_realized,
                "N_swiss_base": N,
                "N_swiss_realized": n_realized,
                "hole_u_range": [2.5 * np.pi, 3.5 * np.pi],
                "hole_v_range": [7.0, 14.0],
                "hole_removed_count": N - n_realized,
            },
        )
    else:
        points_list = []
        lens_list = []
        while len(points_list) < N:
            batch_size = (N - len(points_list)) * 2 + 50
            u_cand = rng.uniform(u_min, u_max, size=batch_size)
            v_cand = rng.uniform(v_min, v_max, size=batch_size)

            hole_mask = (u_cand >= 2.5 * np.pi) & (u_cand <= 3.5 * np.pi) & (v_cand >= 7.0) & (v_cand <= 14.0)
            valid_idx = np.where(~hole_mask)[0]

            for idx in valid_idx:
                u = u_cand[idx]
                v = v_cand[idx]
                x = u * np.cos(u)
                y = v
                z = u * np.sin(u)
                points_list.append([x, y, z])
                if lens_id == "radial_xz":
                    lens_list.append(np.sqrt(x ** 2 + z ** 2))
                elif lens_id == "unrolled_u":
                    lens_list.append(u)
                elif lens_id == "height":
                    lens_list.append(y)
                else:
                    raise ValueError(
                        f"Unsupported lens_id '{lens_id}' for Swiss Roll with hole. "
                        f"Supported: 'radial_xz', 'unrolled_u', 'height'"
                    )
                if len(points_list) == N:
                    break

        points = np.array(points_list, dtype=float)
        lens = np.array(lens_list, dtype=float)

        return SyntheticDataset(
            dataset_id=f"swiss_roll_hole_N{N}_seed{data_seed}",
            space_name="swiss_roll_with_hole",
            points=points,
            lens=lens,
            lens_id=lens_id,
            data_seed=data_seed,
            perturbation_seed=None,
            resample_seed=None,
            point_order=tuple(range(N)),
            noise_sigma=0.0,
            is_perturbed=False,
            metadata={
                "dimension": 3,
                "sample_size_requested": N,
                "sample_size_realized": N,
                "N_swiss_base": N,
                "N_swiss_realized": N,
                "hole_u_range": [2.5 * np.pi, 3.5 * np.pi],
                "hole_v_range": [7.0, 14.0],
            },
        )


from mapper_framework.exceptions import ConfigurationInvalidError


def apply_coordinate_noise(
    dataset: SyntheticDataset,
    noise_sigma: float,
    perturbation_seed: int,
) -> SyntheticDataset:
    """Apply i.i.d. Gaussian noise N(0, sigma^2 * I_d) to dataset coordinates."""
    if noise_sigma < 0:
        raise ValueError(f"noise_sigma must be >= 0, got {noise_sigma}")

    if noise_sigma == 0.0:
        return dataset

    rng = np.random.default_rng(perturbation_seed)
    noise = rng.normal(0.0, noise_sigma, size=dataset.points.shape)
    noisy_points = dataset.points + noise

    # Update lens according to canonical ambient filter function definition
    if dataset.lens_id == "height":
        noisy_lens = noisy_points[:, 1].copy()
    elif dataset.lens_id == "radial_xz":
        # Ambient radial filter f(x, y, z) = sqrt(x^2 + z^2) for Swiss Roll M6-A2
        noisy_lens = np.sqrt(noisy_points[:, 0] ** 2 + noisy_points[:, 2] ** 2)
    elif dataset.lens_id == "unrolled_u":
        # Historical M6-A unrolled latent coordinate (preserved historical lens per A4)
        noisy_lens = dataset.lens.copy()
    elif dataset.lens_id in ("digits_pca1_frozen", "frozen_pca1", "frozen_pca1_clean_digits"):
        # Recompute lens values on X_noisy using frozen clean PCA mean and signed component
        pca_mean = dataset.metadata.get("pca_mean")
        pca_comp = dataset.metadata.get("pca_component")
        if pca_comp is None:
            pca_comp = dataset.metadata.get("signed_pc1_component")
        if pca_mean is None or pca_comp is None:
            # Try loading from materialized artifact if available
            from pathlib import Path
            p_art = Path(__file__).resolve().parent.parent.parent / "results" / "N1" / "materialization" / "N1_DIGITS_PCA_ARTIFACT.npz"
            if p_art.exists():
                art = np.load(p_art)
                pca_mean = art["pca_mean"]
                pca_comp = art["signed_pc1_component"]
        if pca_mean is None or pca_comp is None:
            raise ConfigurationInvalidError(
                f"Frozen PCA lens '{dataset.lens_id}' requires 'pca_mean' and 'pca_component' in dataset.metadata or materialized artifact."
            )
        noisy_lens = np.asarray(noisy_points - pca_mean, dtype=float) @ np.asarray(pca_comp, dtype=float)
    else:
        raise ConfigurationInvalidError(
            f"Unknown or unsupported lens_id '{dataset.lens_id}' for coordinate noise evaluation. "
            f"Supported: 'height', 'radial_xz', 'unrolled_u', 'digits_pca1_frozen', 'frozen_pca1'"
        )


    return SyntheticDataset(
        dataset_id=f"{dataset.dataset_id}_sigma{noise_sigma}_pseed{perturbation_seed}",
        space_name=dataset.space_name,
        points=noisy_points,
        lens=noisy_lens,
        lens_id=dataset.lens_id,
        data_seed=dataset.data_seed,
        perturbation_seed=perturbation_seed,
        resample_seed=dataset.resample_seed,
        point_order=dataset.point_order,
        noise_sigma=noise_sigma,
        is_perturbed=True,
        metadata={
            **dataset.metadata,
            "base_dataset_id": dataset.dataset_id,
            "noise_sigma": noise_sigma,
            "perturbation_factor": "coordinate_gaussian_noise",
        },
    )


def apply_subsampling(
    dataset: SyntheticDataset,
    sample_size: int,
    resample_seed: int,
) -> SyntheticDataset:
    """Generate subsampled realization from dataset X_0 without replacement."""
    N = len(dataset.points)
    if sample_size > N:
        raise ValueError(f"Requested sample_size ({sample_size}) exceeds dataset size ({N})")

    rng = np.random.default_rng(resample_seed)
    selected_indices = rng.choice(N, size=sample_size, replace=False)
    selected_indices.sort()

    sub_points = dataset.points[selected_indices]
    sub_lens = dataset.lens[selected_indices]

    return SyntheticDataset(
        dataset_id=f"{dataset.dataset_id}_subsample{sample_size}_rseed{resample_seed}",
        space_name=dataset.space_name,
        points=sub_points,
        lens=sub_lens,
        lens_id=dataset.lens_id,
        data_seed=dataset.data_seed,
        perturbation_seed=dataset.perturbation_seed,
        resample_seed=resample_seed,
        point_order=tuple(range(sample_size)),
        noise_sigma=dataset.noise_sigma,
        is_perturbed=dataset.is_perturbed,
        metadata={
            **dataset.metadata,
            "subsample_indices": selected_indices.tolist(),
            "original_N": N,
            "subsample_N": sample_size,
        },
    )
