import math
from typing import Tuple

# Conversion factor: 1 eV in joules
J_per_eV: float = 1.602176634e-19


def hartree_eJ_eV(
    m_e: float = 9.10938356e-31,
    electron_charge: float = 1.602176634e-19,
    eps0: float = 8.8541878128e-12,
    hbar: float = 1.054571817e-34
) -> Tuple[float, float]:
    """Compute the Hartree energy in joules and in electron volts.

    Eh = m_e * (electron_charge^4) / (16 * pi^2 * eps0^2 * hbar^2)
    Returns (Eh_J, Eh_eV).
    """
    # Basic input validation
    if not (m_e > 0.0):
        raise ValueError("m_e must be positive.")
    if not (electron_charge > 0.0):
        raise ValueError("electron_charge must be positive.")
    if not (eps0 > 0.0):
        raise ValueError("eps0 must be positive.")
    if not (hbar > 0.0):
        raise ValueError("hbar must be positive.")

    Eh_J = m_e * (electron_charge**4) / (16.0 * math.pi**2 * (eps0**2) * (hbar**2))
    Eh_eV = Eh_J / J_per_eV
    return Eh_J, Eh_eV


def is_valid_hartree_eV(Eh_eV: float, expected: float = 27.211386245988,
                       rel_tol: float = 1e-6, abs_tol: float = 1e-12) -> bool:
    """Return True if Eh_eV matches expected within tolerances."""
    return math.isclose(Eh_eV, expected, rel_tol=rel_tol, abs_tol=abs_tol)


def assert_valid_hartree_eV(Eh_eV: float, expected: float = 27.211386245988,
                          rel_tol: float = 1e-6, abs_tol: float = 1e-12) -> None:
    """Raise ValueError if Eh_eV is not within tolerances."""
    if not is_valid_hartree_eV(Eh_eV, expected, rel_tol, abs_tol):
        raise ValueError("Hartree energy in eV is not within the allowed tolerance of the expected value.")


if __name__ == "__main__":
    Eh_J, Eh_eV = hartree_eJ_eV()
    print("Hartree energy in joules: {:.12e}".format(Eh_J))
    print("Hartree energy in eV: {:.12f}".format(Eh_eV))
    print("Validation against expected value: {}".format(is_valid_hartree_eV(Eh_eV)))
    # To enforce strict runtime checks, uncomment the next line
    # assert_valid_hartree_eV(Eh_eV)
