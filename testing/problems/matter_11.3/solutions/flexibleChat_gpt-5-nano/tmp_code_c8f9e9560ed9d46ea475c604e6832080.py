from collections import defaultdict
from typing import List, Tuple

# Global cache to store precomputed level structures for (ne, model_id)
_LEVEL_CACHE = {}


def precompute_levels_for_ne(ne: int, model_id: str = 'cube_infinite', max_n_init: int = 6):
    """
    Precompute the degenerate energy levels for a 3D cubic infinite-well model.
    Returns a tuple: (levels, i_last, ds) where:
      - levels: list of (s, g) with s = n_x^2 + n_y^2 + n_z^2 and g the degeneracy
      - i_last: index in levels corresponding to the last occupied s for 'ne' electrons
      - ds: s_{i_last+1} - s_{i_last} (difference to the next unoccupied level)
    This function grows the grid size until enough states exist to hold 'ne' electrons
    (two electrons per spatial orbital due to spin).
    """
    key = (ne, model_id)
    if key in _LEVEL_CACHE:
        return _LEVEL_CACHE[key]

    max_n = max(1, max_n_init)
    while True:
        degeneracies = defaultdict(int)
        for nx in range(1, max_n + 1):
            nx2 = nx * nx
            for ny in range(1, max_n + 1):
                ny2 = ny * ny
                for nz in range(1, max_n + 1):
                    s = nx2 + ny2 + nz * nz
                    degeneracies[s] += 1
        levels = sorted((s, g) for s, g in degeneracies.items())  # list of (s, g)

        cum = 0
        i_last = None
        for i, (s, g) in enumerate(levels):
            cap = 2 * g  # two electrons per spatial orbital due to spin
            if cum + cap >= ne:
                i_last = i
                break
            cum += cap

        if i_last is not None and i_last + 1 < len(levels):
            ds = levels[i_last + 1][0] - levels[i_last][0]
            _LEVEL_CACHE[key] = (levels, i_last, ds)
            return levels, i_last, ds

        # Not enough states yet; extend search space
        max_n *= 2
        if max_n > 150:
            raise RuntimeError("Unable to precompute levels for given Ne within bounds")


def get_levels(ne: int) -> Tuple[List[Tuple[int, int]], int, int]:
    """Return cached or freshly computed levels for a given Ne."""
    return precompute_levels_for_ne(ne)


def first_excitation_sweep(ne: int, L_min: float, L_max: float, n_points: int = 41) -> List[Tuple[float, float, float, float]]:
    """
    Sweep L values and compute Delta_E and lambda for a fixed Ne using the precomputed levels.
    Returns a list of tuples: (L, Delta_E_J, Delta_E_eV, lambda_nm)
    """
    levels, i_last, ds = get_levels(ne)

    h = 6.62607015e-34
    c = 299792458.0
    m_e = 9.10938356e-31
    def U(L: float) -> float:
        return (h**2) / (8.0 * m_e * L**2)

    results: List[Tuple[float, float, float, float]] = []
    for i in range(n_points):
        t = i / (n_points - 1) if n_points > 1 else 0.0
        L = L_min + t * (L_max - L_min)
        Delta_E = ds * U(L)
        lambda_nm = (h * c) / Delta_E * 1e9
        Delta_E_eV = Delta_E / 1.602176634e-19
        results.append((L, Delta_E, Delta_E_eV, lambda_nm))
    return results


def main():
    Ne = 60
    L_min = 0.5e-9
    L_max = 0.9e-9
    n_points = 41

    results = first_excitation_sweep(Ne, L_min, L_max, n_points=n_points)

    # Find best matches to 730 nm
    target_nm = 730.0
    best = min(results, key=lambda x: abs(x[3] - target_nm))

    print("L sweep results for Ne = {} over L from {:.3f} nm to {:.3f} nm".format(
        Ne, L_min * 1e9, L_max * 1e9))
    print("Best match: L = {:.6f} nm, lambda = {:.0f} nm, Delta_E = {:.3e} J ({:.3f} eV)".format(
        best[0] * 1e9, best[3], best[1], best[2]))
    print("Top results (L_nm, lambda_nm, Delta_E_eV):")
    sorted_by_lambda = sorted(results, key=lambda x: abs(x[3] - target_nm))
    for i in range(min(5, len(sorted_by_lambda))):
        L, De, De_eV, lam = sorted_by_lambda[i]
        print("L={:.3f} nm, lambda={:.0f} nm, Delta_E={:.3e} J ({:.3f} eV)".format(
            L * 1e9, lam, De, De_eV))


if __name__ == "__main__":
    main()
