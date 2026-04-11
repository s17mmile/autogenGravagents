from dataclasses import dataclass

@dataclass
class IonizationEnergy:
    IE_eV_reduced: float
    IE_kJmol_reduced: float
    IE_eV_approx: float
    IE_kJmol_approx: float


def ionization_energy_bohr_like(Z: int, n: int = 1, A: int = 4) -> IonizationEnergy:
    """Compute Bohr-model ionization energy for a hydrogen-like ion with reduced mass correction.

    Z: nuclear charge number (integer >= 1)
    n: principal quantum number (integer >= 1)
    A: mass number of the nucleus (default 4 for Helium-4)

    Returns an IonizationEnergy dataclass containing:
      - IE_eV_reduced: ionization energy in eV with reduced-mass correction
      - IE_kJmol_reduced: same energy in kJ/mol
      - IE_eV_approx: ionization energy in eV with mu_over_me = 1 (approximate)
      - IE_kJmol_approx: same as kJ/mol for the approximate value
    """
    if not isinstance(Z, int) or Z < 1:
        raise ValueError("Z must be an integer >= 1")
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be an integer >= 1")
    if not isinstance(A, int) or A < 1:
        raise ValueError("A must be an integer >= 1")

    # Physical constants (SI)
    m_e = 9.10938356e-31        # electron mass, kg
    m_u = 1.66053906660e-27     # atomic mass unit, kg
    M = A * m_u                   # nucleus mass, kg

    # Reduced mass and ratio to electron mass
    mu = (m_e * M) / (m_e + M)
    mu_over_me = mu / m_e           # dimensionless

    # Bohr model parameters with precise constants
    eV_core = 13.605693122994      # ground-state energy scale in eV
    eV_to_kJmol = 96.485261        # CODATA: 1 eV per particle = 96.485261 kJ/mol

    # Reduced-mass corrected energy (n-th level)
    E1_eV_reduced = - mu_over_me * (Z**2) * eV_core / (n**2)
    IE_eV_reduced = - E1_eV_reduced
    IE_kJmol_reduced = IE_eV_reduced * eV_to_kJmol

    # Approximate energy with mu_over_me = 1
    E1_eV_approx = - (Z**2) * eV_core / (n**2)
    IE_eV_approx = - E1_eV_approx
    IE_kJmol_approx = IE_eV_approx * eV_to_kJmol

    return IonizationEnergy(
        IE_eV_reduced=IE_eV_reduced,
        IE_kJmol_reduced=IE_kJmol_reduced,
        IE_eV_approx=IE_eV_approx,
        IE_kJmol_approx=IE_kJmol_approx,
    )


if __name__ == "__main__":
    Z, n, A = 2, 1, 4  # He_plus: Helium-4 nucleus, ground state
    ie = ionization_energy_bohr_like(Z, n, A)
    print("Bohr-model ionization energy for He_plus:")
    print(f"  Reduced-mass:   {ie.IE_eV_reduced:.6f} eV, {ie.IE_kJmol_reduced:.0f} kJ/mol")
    print(f"  Approx mu_over_me=1: {ie.IE_eV_approx:.6f} eV, {ie.IE_kJmol_approx:.0f} kJ/mol")