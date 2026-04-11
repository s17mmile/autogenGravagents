import math
from typing import List, Tuple, Dict

# Small epsilon to avoid division by zero or taking roots of non-physical values
EPS = 1e-12


def fret_distance(E: float, R0: float = 50.0, eps: float = EPS) -> float:
    """Compute the FRET distance r from efficiency E and reference R0.
    E must be between 0 and 1 (exclusive). The value is clamped for numerical stability.
    Returns r in angstroms by default.
    """
    # Robust clamping to avoid invalid math near 0 or 1
    E = max(min(E, 1.0 - eps), eps)
    return R0 * ((1.0 - E) / E) ** (1.0 / 6.0)


def compute_delta_r(E_left: float, E_right: float, R0: float = 50.0) -> Dict[str, float]:
    """Compute left and right distances and the delta_r between them.

    Args:
        E_left: FRET efficiency for the left state (0 < E_left < 1).
        E_right: FRET efficiency for the right state (0 < E_right < 1).
        R0: Förster radius in angstroms (default 50.0 A).

    Returns:
        dict with keys: r_left, r_right, delta_r
    """
    r_left = fret_distance(E_left, R0)
    r_right = fret_distance(E_right, R0)
    delta_r = r_right - r_left
    return {"r_left": r_left, "r_right": r_right, "delta_r": delta_r}


def _dr_dE(E: float, R0: float) -> float:
    """Derivative of r with respect to E: dr/dE.
    Uses the analytical form dr/dE = - (R0/6) * ((1-E)/E)^(-5/6) / E^2, with clamping for stability.
    """
    eps = EPS
    E = max(min(E, 1.0 - eps), eps)
    g = (1.0 - E) / E
    return - (R0 / 6.0) * (g ** (-5.0 / 6.0)) / (E * E)


def compute_delta_r_with_uncertainty(
    E_left: float, dE_left: float, E_right: float, dE_right: float, R0: float = 50.0
) -> Dict[str, float]:
    """Compute delta_r and its uncertainty from two E values with their uncertainties.

    Assumes independent errors in E_left and E_right.
    Returns a dictionary with r_left, r_right, delta_r, d_r_left, d_r_right, delta_r_unc.
    """
    r_left = fret_distance(E_left, R0)
    r_right = fret_distance(E_right, R0)
    delta_r = r_right - r_left

    dr_left = _dr_dE(E_left, R0)
    dr_right = _dr_dE(E_right, R0)

    d_r_left = abs(dr_left) * dE_left
    d_r_right = abs(dr_right) * dE_right
    delta_r_unc = math.sqrt(d_r_left ** 2 + d_r_right ** 2)

    return {
        "r_left": r_left,
        "r_right": r_right,
        "delta_r": delta_r,
        "d_r_left": d_r_left,
        "d_r_right": d_r_right,
        "delta_r_unc": delta_r_unc,
    }


def compute_delta_r_batch(pairs: List[Tuple[float, float]], R0: float = 50.0) -> List[Dict[str, float]]:
    """Compute delta_r for a batch of (E_left, E_right) pairs.

    Returns a list of dictionaries as produced by compute_delta_r.
    """
    results: List[Dict[str, float]] = []
    for E_left, E_right in pairs:
        results.append(compute_delta_r(E_left, E_right, R0))
    return results


if __name__ == "__main__":
    # Example usage: replace with actual E_left and E_right values
    E_left = 0.25
    E_right = 0.75
    R0 = 50.0

    res = compute_delta_r(E_left, E_right, R0)
    print("E_left = {0:.6f}, E_right = {1:.6f}".format(E_left, E_right))
    print("r_left = {0:.6f} A".format(res["r_left"]))
    print("r_right = {0:.6f} A".format(res["r_right"]))
    print("Delta r = {0:.6f} A".format(res["delta_r"]))

    # Uncertainty example (optional)
    dE_left = 0.01
    dE_right = 0.01
    res_unc = compute_delta_r_with_uncertainty(E_left, dE_left, E_right, dE_right, R0)
    print("Delta r uncertainty = {0:.6f} A".format(res_unc["delta_r_unc"]))

    # Batch example
    pairs = [(0.25, 0.75), (0.30, 0.60)]
    batch = compute_delta_r_batch(pairs, R0)
    for i, b in enumerate(batch, start=1):
        print("Pair {0}: delta_r = {1:.6f} A".format(i, b["delta_r"]))
