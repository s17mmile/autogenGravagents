# filename: calculate_molar_energy.py

import math
from scipy.constants import epsilon_0, elementary_charge, Avogadro

def calculate_molar_energy() -> float:
    """Calculate the molar energy required to reverse the dipole direction of an H2O molecule
    located 100 pm from a Li+ ion.

    Returns:
        float: Molar energy in Joules per mole.
    """
    # Constants
    dipole_moment_D = 1.85  # Debye
    debye_to_Cm = 3.33564e-30  # 1 Debye in C*m
    dipole_moment = dipole_moment_D * debye_to_Cm  # C*m

    distance_pm = 100  # pm
    distance_m = distance_pm * 1e-12  # m

    charge_C = elementary_charge  # Charge of Li+ ion in Coulombs

    # Calculate interaction energy U (Joules) for one molecule
    # U = - (1 / (4 * pi * epsilon_0)) * (q * mu) / r^2
    prefactor = 1 / (4 * math.pi * epsilon_0)
    U = - prefactor * (charge_C * dipole_moment) / (distance_m ** 2)

    # Energy to reverse dipole direction is twice the magnitude of U
    energy_per_molecule = 2 * abs(U)

    # Convert to molar energy
    molar_energy = energy_per_molecule * Avogadro  # Joules per mole

    return molar_energy

if __name__ == '__main__':
    energy = calculate_molar_energy()
    print(f'Molar energy required to reverse the dipole direction: {energy:.3e} J/mol')
