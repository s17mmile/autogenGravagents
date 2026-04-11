import math
from typing import Tuple, Dict


def temperature_two_level_with_diagnostics(
    U_J: float,
    wavenumber_cm1: float,
    NA: float = 6.02214076e23,
    h: float = 6.62607015e-34,
    c: float = 299792458.0,
    kB: float = 1.380649e-23,
    g0: int = 1,
    g1: int = 1,
    tol: float = 1e-8,
) -> Tuple[float, float, Dict[str, object]]:
    """Compute temperature for a mole of two-level systems with degeneracies g0 (lower)
    and g1 (upper).

    Returns a triple: (T, DeltaE, info)
      - T: temperature in Kelvin (float, or inf for infinite temperature limit)
      - DeltaE: energy spacing in joules per molecule
      - info: diagnostic dictionary with keys:
          'beta_delta': value of beta * DeltaE
          'x': exp(beta DeltaE)
          'U_max': physical maximum energy per mole
          'status': 'finite' or 'infinite'
          'near_infinite': True/False indicating beta_delta near zero

    U_J must satisfy 0 < U_J <= U_max, where U_max = NA * DeltaE * g1 / (g0 + g1).
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
        raise ValueError("Invalid inputs yield non-positive exp(beta DeltaE).")

    beta_delta = math.log(x)

    info: Dict[str, object] = {
        'beta_delta': beta_delta,
        'x': x,
        'U_max': U_max,
    }

    if abs(beta_delta) < tol:
        # Infinite temperature limit
        info['status'] = 'infinite'
        info['near_infinite'] = True
        T = math.inf
        return T, deltaE, info

    T = deltaE / (kB * beta_delta)
    info['status'] = 'finite'
    info['near_infinite'] = False
    return T, deltaE, info


def main():
    # Original case: g0 = g1 = 1
    U_J = 3000.0           # J per mole
    wavenumber_cm1 = 1000.0  # cm^-1

    T, DeltaE, info = temperature_two_level_with_diagnostics(U_J, wavenumber_cm1)
    print("DeltaE (J):", DeltaE)
    print("Temperature (K):", T)
    print("Info:", info)

    # Degenerate example: g0 = 2, g1 = 3
    T2, DeltaE2, info2 = temperature_two_level_with_diagnostics(U_J, wavenumber_cm1, g0=2, g1=3)
    print("\nDegenerate case: DeltaE (J):", DeltaE2)
    print("Degenerate case: Temperature (K):", T2)
    print("Degenerate case Info:", info2)


if __name__ == "__main__":
    main()
