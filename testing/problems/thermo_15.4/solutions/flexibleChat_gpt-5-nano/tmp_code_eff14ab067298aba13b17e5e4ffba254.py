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
    max_T_surrogate: float = 1e6,
) -> Tuple[float, float, Dict[str, object]]:
    """Compute temperature for a mole of two-level systems with degeneracies g0 and g1.

    Returns a triple: (T, DeltaE, info)
      - T: temperature in Kelvin (float, or inf for infinite temperature limit, or surrogate finite if beta_delta near zero)
      - DeltaE: energy spacing in joules per molecule
      - info: diagnostic dictionary with keys:
          'beta_delta': value of beta * DeltaE
          'x': exp(beta * DeltaE)
          'U_max': physical maximum energy per mole
          'status': 'finite' or 'infinite' or 'surrogate'
          'near_infinite': True/False indicating beta_delta near zero

     U_J must satisfy 0 < U_J <= U_max, where U_max = NA * DeltaE * g1 /(g0 + g1).
    """
    if U_J <= 0:
        raise ValueError("U_J must be positive.")
    if g0 <= 0 or g1 <= 0:
        raise ValueError("g0 and g1 must be positive integers.")

    deltaE = h * c * (wavenumber_cm1 * 100.0)
    U_max = NA * deltaE * (g1 / (g0 + g1))
    if not (0 < U_J <= U_max):
        raise ValueError(
            f"U_J must satisfy 0 < U_J <= U_max (physical bound). U_max = {U_max:.12g}, got {U_J:.12g}."
        )

    A = (NA * deltaE) / U_J
    x = (g1 / g0) * (A - 1.0)
    if x <= 0.0:
        raise ValueError("Invalid inputs yield non-positive exp(beta DeltaE).")

    beta_delta = math.log(x)
    info = {
        'beta_delta': beta_delta,
        'x': x,
        'U_max': U_max,
    }

    if abs(beta_delta) < tol:
        # Finite surrogate for infinite temperature to avoid inf in outputs
        T = max_T_surrogate
        info['status'] = 'surrogate'
        info['near_infinite'] = True
        return T, deltaE, info

    T = deltaE / (kB * beta_delta)
    info['status'] = 'finite'
    info['near_infinite'] = False
    return T, deltaE, info


def main():
    U_J = 3000.0
    wavenumber_cm1 = 1000.0
    T, DeltaE, info = temperature_two_level_with_diagnostics(U_J, wavenumber_cm1)
    print("DeltaE (J):", DeltaE)
    print("Temperature (K):", T)
    print("Info:", info)


if __name__ == "__main__":
    main()
