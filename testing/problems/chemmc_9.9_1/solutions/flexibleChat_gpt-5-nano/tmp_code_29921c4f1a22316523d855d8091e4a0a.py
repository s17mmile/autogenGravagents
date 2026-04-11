import math
from typing import Tuple

# Conversion factor: 1 eV in joules
J_per_eV: float = 1.602176634e-19


def hartree_eJ_eV(
    m_e: float = 9.10938356e-31,
    charge: float = 1.602176634e-19,
    eps0: float = 8.8541878128e-12,
    hbar: float = 1.054571817e-34
) -> Tuple[float, float]:
    """Compute the Hartree energy in joules and in electron volts.

    Eh = m_e * (charge^4) / (16 * pi^2 * eps0^2 * hbar^2)
    Returns (Eh_J, Eh_eV).
    """
    Eh_J = m_e * (charge**4) / (16.0 * math.pi**2 * (eps0**2) * (hbar**2))
    Eh_eV = Eh_J / J_per_eV
    return Eh_J, Eh_eV


if __name__ == "__main__":
    Eh_J, Eh_eV = hartree_eJ_eV()
    print("Hartree energy in joules: {:.12e}".format(Eh_J))
    print("Hartree energy in eV: {:.12f}".format(Eh_eV))
    # Validation against the widely cited value (within a tight tolerance)
    assert abs(Eh_eV - 27.211386245988) < 1e-12
