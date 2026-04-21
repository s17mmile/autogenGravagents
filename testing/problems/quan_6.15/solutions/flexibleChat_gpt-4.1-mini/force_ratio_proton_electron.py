# filename: force_ratio_proton_electron.py

from scipy.constants import epsilon_0, elementary_charge, G, proton_mass, electron_mass

import math

def calculate_force_ratio():
    """Calculate the ratio of electrical to gravitational force between a proton and an electron."""
    # Coulomb's constant k_e = 1 / (4 * pi * epsilon_0)
    k_e = 1 / (4 * math.pi * epsilon_0)

    # Calculate ratio
    ratio = (k_e * elementary_charge ** 2) / (G * proton_mass * electron_mass)
    return ratio

if __name__ == '__main__':
    ratio = calculate_force_ratio()
    print(f"Ratio of electrical to gravitational force between proton and electron: {ratio:.3e}")
