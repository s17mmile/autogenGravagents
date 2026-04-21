# filename: calculate_cl2_radius.py
import math

# Constants
B_L_PER_MOL = 0.0562  # van der Waals constant b for Cl2 in liters per mole
AVOGADRO_NUMBER = 6.02214076e23  # more precise CODATA 2019 value

# Convert liters to cubic meters
L_TO_M3 = 1e-3


def calculate_cl2_radius(b_l_per_mol=B_L_PER_MOL, n_a=AVOGADRO_NUMBER):
    """Calculate approximate radius of Cl2 molecule in picometers."""
    b_m3_per_mol = b_l_per_mol * L_TO_M3
    volume_per_molecule = b_m3_per_mol / n_a
    radius_m = (3 * volume_per_molecule / (4 * math.pi)) ** (1/3)
    radius_pm = radius_m * 1e12  # convert meters to picometers
    return radius_pm


if __name__ == '__main__':
    radius = calculate_cl2_radius()
    print(f"Approximate radius of Cl2 molecule: {radius:.2f} pm")
