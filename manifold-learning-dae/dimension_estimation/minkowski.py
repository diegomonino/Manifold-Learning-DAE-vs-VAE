"""
minkowski.py
------------
Minkowski (box-counting) dimension estimation.

For a point cloud C near a d-dimensional manifold M in R^D:
    log N(r) ~ log V - d * log r

where N(r) is the number of occupied cubes of side r.
The dimension d is estimated as the slope of log N vs -log r
in the linear regime.

For large D, we enumerate cubes by hashing point coordinates
(iterating over data points) rather than constructing a full grid.
"""

import numpy as np
from typing import Tuple


def count_occupied_cubes(data: np.ndarray, r: float) -> int:
    """
    Count the number of distinct D-dimensional cubes of side r
    that contain at least one data point.

    Efficient implementation: assigns each point to a cube via
    floor(x / r) and counts unique cube indices.

    Args:
        data: Array of shape (N, D).
        r:    Cube side length.

    Returns:
        Number of occupied cubes.
    """
    # Map each point to its cube index
    cube_indices = np.floor(data / r).astype(np.int64)
    # Count unique rows (each row = a cube index vector)
    # Use a set of tuples for exact counting
    unique_cubes = set(map(tuple, cube_indices))
    return len(unique_cubes)


def minkowski_dimension(
    data: np.ndarray,
    r_min: float = None,
    r_max: float = None,
    n_scales: int = 20
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate the Minkowski (box-counting) dimension of a point cloud.

    Computes N(r) for a range of scales r, then fits a line to
    log N vs -log r in the region where the relationship is linear.

    Args:
        data:     Array of shape (N, D).
        r_min:    Minimum cube side. Defaults to 2x median nearest-neighbor distance.
        r_max:    Maximum cube side. Defaults to data diameter / 2.
        n_scales: Number of scale values to evaluate.

    Returns:
        Tuple of:
            - dim_estimate: estimated dimension (float)
            - neg_log_r:    array of -log(r) values
            - log_N:        array of log(N(r)) values
    """
    from .metrics import nearest_neighbor_distances

    N, D = data.shape

    # Determine scale range if not provided
    if r_min is None:
        dists = nearest_neighbor_distances(data)
        r_min = 2.0 * np.median(dists)
    if r_max is None:
        data_range = data.max(axis=0) - data.min(axis=0)
        r_max = np.max(data_range) / 2.0

    # Ensure valid range
    r_min = max(r_min, 1e-10)
    if r_max <= r_min:
        r_max = r_min * 10

    # Logarithmically spaced scales
    r_values = np.logspace(np.log10(r_min), np.log10(r_max), n_scales)

    log_N = []
    neg_log_r = []

    for r in r_values:
        n_cubes = count_occupied_cubes(data, r)
        if n_cubes > 1:  # log(1) = 0, not useful
            log_N.append(np.log(n_cubes))
            neg_log_r.append(-np.log(r))

    log_N = np.array(log_N)
    neg_log_r = np.array(neg_log_r)

    # Fit a line: log_N = intercept + slope * neg_log_r
    # The slope is our dimension estimate
    if len(neg_log_r) < 3:
        raise ValueError("Not enough valid scale points for linear fit. "
                         "Try adjusting r_min or r_max.")

    coeffs = np.polyfit(neg_log_r, log_N, deg=1)
    dim_estimate = coeffs[0]  # slope

    return dim_estimate, neg_log_r, log_N


if __name__ == "__main__":
    from sklearn.datasets import make_swiss_roll

    print("== Swiss Roll (expected dim ~2) ==")
    X, _ = make_swiss_roll(n_samples=5000, noise=0.0, random_state=42)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    dim, neg_lr, lN = minkowski_dimension(X)
    print(f"  Estimated Minkowski dimension: {dim:.2f}")

    print("\n== Linear H_3,30 (expected dim ~3) ==")
    rng = np.random.RandomState(42)
    Z = rng.uniform(0, 1, size=(10000, 3))
    A_raw = rng.randn(30, 3)
    A, _ = np.linalg.qr(A_raw)
    A = A[:, :3]
    X_lin = Z @ A.T
    dim_lin, _, _ = minkowski_dimension(X_lin)
    print(f"  Estimated Minkowski dimension: {dim_lin:.2f}")
