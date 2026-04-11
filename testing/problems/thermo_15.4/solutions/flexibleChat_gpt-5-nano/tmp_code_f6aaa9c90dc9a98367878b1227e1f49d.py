import math
from typing import Tuple


def temperature_two_level(U_J: float, wavenumber_cm1: float, NA: float = 6.02214076e23,
                        h: float = 6.62607015e-34, c: float = 299792458.0, kB: float = 1.380649e-23,
                        g0: int = 1, g1: int = 1) -> Tuple[float, float]:
    """Compute temperature for a mole of two-level systems with degeneracies g0 (lower) and g1 (upper).

    U_J: target internal energy in joules (per mole)
    wavenumber_cm1: energy spacing expressed as wavenumber in cm^-1
    NA: Avogadro constant
    h, c, kB: Planck constant, speed of light, Boltzmann constant
    g0, g1: degeneracies of the lower and upper levels, respectively

    For degeneracies, per-particle partition function Z1 = g0 + g1 * exp(-beta * DeltaE)
    and U per particle U1 = DeltaE * g1 * exp(-beta * DeltaE) / Z1.
    The closed-form solution gives exp(beta * DeltaE) = (g1/g0) * ((N_A * DeltaE)/U_J - 1).
    Then T = DeltaE / (k_B * ln(exp(beta DeltaE))).

    Returns (T, DeltaE).
    """
    if U_J <= 0:
        raise ValueError("U_J must be positive.")
    if g0 <= 0 or g1 <= 0:
        raise ValueError("g0 and g1 must be positive integers.")

    # Delta E from wavenumber: DeltaE = h * c * (wn * 100)
    deltaE = h * c * (wavenumber_cm1 * 100.0)

    # Maximum energy per mole for given degeneracies: U_max = N_A * DeltaE * g1 / (g0 + g1)
    U_max = NA * deltaE * (g1 / (g0 + g1))
    if not (0 < U_J <= U_max):
        raise ValueError(
            f"U_J must satisfy 0 < U_J <= U_max (physical bound). U_max = {U_max:.12g}, got {U_J:.12g}."
        )

    # Closed-form: A = (N_A * DeltaE) / U_J, x = exp(beta DeltaE) = (g1/g0) * (A - 1)
    A = (NA * deltaE) / U_J
    x = (g1 / g0) * (A - 1.0)
    if x <= 0.0:
        raise ValueError("Invalid inputs yield non-positive exp(beta DeltaE). Check U_J, wavenumber, and degeneracies.")

    beta_delta = math.log(x)
    if beta_delta == 0.0:
        # Infinite temperature limit
        return math.inf, deltaE

    T = deltaE / (kB * beta_delta)
    return T, deltaE


def main():
    # Original case: g0 = g1 = 1
    U_J = 3000.0           # J per mole
    wavenumber_cm1 = 1000.0  # cm^-1

    T, DeltaE = temperature_two_level(U_J, wavenumber_cm1)
    print("DeltaE (J):", DeltaE)
    print("Temperature (K):", T)

    # Degenerate example: g0 = 2, g1 = 3
    T2, DeltaE2 = temperature_two_level(U_J, wavenumber_cm1, g0=2, g1=3)
    print("\nDegenerate case: DeltaE (J):", DeltaE2)
    print("Degenerate case: Temperature (K):", T2)


if __name__ == "__main__":
    main()
