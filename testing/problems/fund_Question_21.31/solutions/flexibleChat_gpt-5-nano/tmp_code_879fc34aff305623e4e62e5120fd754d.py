"""Earth intercepts protons current calculator

Given a flux of protons per second per square meter, this module computes:
- Earth's surface area from its radius
- total proton rate hitting Earth (protons per second)
- intercepted current in amperes (A) via I = N_dot * e

Units:
- area_m2: square meters (m^2)
- proton_rate: protons per second (s^-1)
- current_A: amperes (A)
"""
from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass
class InterceptResult:
    area_m2: float
    proton_rate: float
    current_A: float

__all__ = ["InterceptResult", "compute_area", "compute_intercepted_current"]


def compute_area(radius_m: float) -> float:
    """Return Earth's surface area given the radius in meters.

    radius_m: radius in meters (must be positive)
    Returns: area in square meters (m^2)
    """
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")
    return 4.0 * math.pi * (radius_m ** 2)


def compute_intercepted_current(
    flux_pps_per_m2: float = 1500.0,
    radius_m: float = 6.371e6,
    e_charge: float = 1.602176634e-19,
) -> InterceptResult:
    """Compute the intercepted current and related quantities for Earth.

    flux_pps_per_m2: protons per second per square meter (pps/m^2)
    radius_m: Earth's radius in meters
    e_charge: elementary charge in coulombs
    Returns an InterceptResult with area_m2, proton_rate, and current_A
    """
    if flux_pps_per_m2 < 0:
        raise ValueError("flux_pps_per_m2 must be non-negative")

    area = compute_area(radius_m)
    proton_rate = flux_pps_per_m2 * area  # protons per second
    current_A = proton_rate * e_charge       # coulombs per second (A)
    return InterceptResult(area, proton_rate, current_A)


if __name__ == "__main__":
    result = compute_intercepted_current()
    print(f"Earth surface area (m2): {result.area_m2:.6e}")
    print(f"Total proton rate (protons/s): {result.proton_rate:.6e}")
    print(f"Intercepted current (A): {result.current_A:.6e}")

    # Inline sanity checks using math.isclose for robustness
    area_expected = 4.0 * math.pi * (6.371e6 ** 2)
    proton_rate_expected = 1500.0 * area_expected
    current_expected = proton_rate_expected * 1.602176634e-19

    assert math.isclose(result.area_m2, area_expected, rel_tol=1e-12)
    assert math.isclose(result.proton_rate, proton_rate_expected, rel_tol=1e-12)
    assert math.isclose(result.current_A, current_expected, rel_tol=1e-12)
    print("Tests passed for default parameters.")