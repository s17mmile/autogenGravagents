from collections import defaultdict
from typing import Tuple


def first_excitation_wavelength_cube_box(L: float = 0.7e-9, Ne: int = 60) -> Tuple[float, float]:
    """
    Compute the first electronic excitation energy (Delta_E) and the corresponding
    wavelength (lambda) for a particle-in-a-cube model with infinite walls.

    Parameters:
    - L: side length of the cube in meters (default 0.7e-9 m)
    - Ne: number of electrons to fill (default 60). Each spatial orbital can hold 2
          electrons due to spin degeneracy, so total capacity per level is 2*g[s].

    Returns:
    - Delta_E: energy gap to the next unoccupied level (in joules)
    - lambda_nm: wavelength corresponding to Delta_E in nanometers

    Model details:
    E(n_x,n_y,n_z) = (h^2 / (8 m_e L^2)) * (n_x^2 + n_y^2 + n_z^2), with n_i >= 1.
    The first excitation is the gap between the last occupied level and the next level.
    """
    h = 6.62607015e-34
    c = 299792458.0
    m_e = 9.10938356e-31
    U = h**2 / (8.0 * m_e * L**2)  # energy scale factor in joules

    # Dynamically grow the search space until Ne electrons can be accommodated
    max_n = 6
    while True:
        degeneracies = defaultdict(int)
        for nx in range(1, max_n + 1):
            nx2 = nx * nx
            for ny in range(1, max_n + 1):
                ny2 = ny * ny
                for nz in range(1, max_n + 1):
                    s = nx2 + ny2 + nz * nz
                    degeneracies[s] += 1

        levels = sorted(degeneracies.items())  # list of (s, g) pairs
        cum = 0
        s_last = None
        s_next = None
        for i, (s, g) in enumerate(levels):
            cap = 2 * g  # two electrons per spatial orbital due to spin
            if cum + cap >= Ne:
                s_last = s
                if i + 1 < len(levels):
                    s_next = levels[i + 1][0]
                else:
                    s_next = None
                break
            cum += cap

        if s_last is None or s_next is None:
            # Not enough states yet; extend the search space
            max_n *= 2
            continue

        Delta_E = U * (s_next - s_last)
        lambda_nm = (h * c) / Delta_E * 1e9
        return Delta_E, lambda_nm


def main():
    Delta_E, lambda_nm = first_excitation_wavelength_cube_box()
    Delta_E_eV = Delta_E / 1.602176634e-19
    print("Delta_E = {:.6e} J ({:.6f} eV)".format(Delta_E, Delta_E_eV))
    print("lambda = {:.0f} nm".format(lambda_nm))


if __name__ == "__main__":
    main()
