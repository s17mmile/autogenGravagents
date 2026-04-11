import math
from itertools import permutations
from typing import Dict, Tuple


def d_spacing(h: int, k: int, l: int, a: float, b: float, c: float) -> float:
    """Compute the interplanar spacing d_hkl for an orthorhombic lattice.

    Parameters:
    - h, k, l: Miller indices (integers)
    - a, b, c: lattice parameters in nanometers (must be positive)

    Returns:
    - d in nanometers (float)
    """
    if a <= 0 or b <= 0 or c <= 0:
        raise ValueError("Lattice parameters a, b, c must be positive.")
    term = (h * h) / (a * a) + (k * k) / (b * b) + (l * l) / (c * c)
    return 1.0 / math.sqrt(term)


def d_values_for_permutations(h: int, k: int, l: int, a: float, b: float, c: float) -> Dict[Tuple[int, int, int], float]:
    """Return a mapping from each permutation of (h, k, l) to its interplanar spacing d.

    Note: For a != b != c, these are not symmetry-equivalent planes but simply index permutations.
    """
    result: Dict[Tuple[int, int, int], float] = {}
    for perm in set(permutations((h, k, l))):
        result[perm] = d_spacing(perm[0], perm[1], perm[2], a, b, c)
    return result


def main():
    # Preset lattice parameters in nanometers
    a = 0.82
    b = 0.94
    c = 0.75

    h, k, l = 1, 2, 3

    d_123 = d_spacing(h, k, l, a, b, c)
    print("d_123 (1,2,3) =", f"{d_123:.6f}", "nm")

    perm_results = d_values_for_permutations(h, k, l, a, b, c)
    print("Permutations of (1,2,3) and their d values (nm):")
    for perm in sorted(perm_results.keys()):
        print(f"{perm} -> {perm_results[perm]:.6f}")


if __name__ == "__main__":
    main()
