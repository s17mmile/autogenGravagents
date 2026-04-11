import math
from dataclasses import dataclass
from typing import Tuple

# Internal constants (SI units)
_U_TO_KG = 1.66053906660e-27  # atomic mass unit to kg
_C = 299792458.0               # speed of light in m/s


def compute_k(m1_u: float, m2_u: float, nu_tilde_cm: float) -> Tuple[float, float]:
    """Compute the bond force constant k for a diatomic molecule using the harmonic model.

    Args:
        m1_u: Mass of atom 1 in atomic mass units (amu).
        m2_u: Mass of atom 2 in amu.
        nu_tilde_cm: Vibrational band wavenumber in cm^-1 (nu tilde).

    Returns:
        A tuple (k_N_per_m, k_dyn_per_cm) where k is in Newtons per meter and dynes per centimeter.
    """
    if m1_u <= 0.0 or m2_u <= 0.0:
        raise ValueError("Masses must be positive numbers.")
    if nu_tilde_cm <= 0.0:
        raise ValueError("nu_tilde must be a positive number.")

    mu = (m1_u * m2_u) / (m1_u + m2_u) * _U_TO_KG
    nu_tilde_m = nu_tilde_cm * 100.0  # convert to m^-1
    omega_e = 2.0 * math.pi * _C * nu_tilde_m
    k = mu * omega_e**2
    k_dyn_per_cm = k * 1000.0  # 1 N/m = 1000 dyn/cm
    return k, k_dyn_per_cm


@dataclass(frozen=True)
class IsotopeData:
    m1_u: float
    m2_u: float
    nu_tilde_cm: float


# Minimal isotopologue mapping. Extend with more entries as needed.
_ISOTOPOLOGUES = {
    "12C16O": IsotopeData(m1_u=12.0, m2_u=16.0, nu_tilde_cm=2143.0),
}


def isotopologue_k(name: str) -> Tuple[float, float]:
    """Compute k for a named isotopologue using a predefined data mapping.

    Returns a tuple (k_N_per_m, k_dyn_per_cm).
    """
    data = _ISOTOPOLOGUES.get(name)
    if data is None:
        raise ValueError(f"Unknown isotopologue '{name}'. Add it to the _ISOTOPOLOGUES mapping to use this function.")
    return compute_k(data.m1_u, data.m2_u, data.nu_tilde_cm)


def main():
    # Demonstration for the original problem: 12C16O with nu_tilde = 2143 cm^-1
    m1_u, m2_u, nu_tilde_cm = 12.0, 16.0, 2143.0
    k, k_dyn = compute_k(m1_u, m2_u, nu_tilde_cm)
    print("k (N/m) = {0:.6e}".format(k))
    print("k (dyn/cm) = {0:.6e}".format(k_dyn))

    # Reduced mass for educational reference
    mu = (m1_u * m2_u) / (m1_u + m2_u) * _U_TO_KG
    print("reduced mass mu (kg) = {0:.6e}".format(mu))

    # Demonstrate isotopologue lookup
    k1, k1_dyn = isotopologue_k("12C16O")
    print("isotopologue_k -> k (N/m) = {0:.6e}".format(k1))
    print("isotopologue_k -> k (dyn/cm) = {0:.6e}".format(k1_dyn))


if __name__ == "__main__":
    main()