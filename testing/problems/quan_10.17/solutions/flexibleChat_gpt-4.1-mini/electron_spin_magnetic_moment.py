# filename: electron_spin_magnetic_moment.py
import math
from scipy.constants import physical_constants

def calculate_spin_magnetic_moment():
    """Calculate the magnitude of the electron's spin magnetic moment in J/T."""
    g_s = 2.002319  # electron spin g-factor
    mu_B = physical_constants['Bohr magneton'][0]  # Bohr magneton in J/T
    s = 0.5  # spin quantum number for electron

    magnitude = g_s * mu_B * math.sqrt(s * (s + 1))
    return magnitude

if __name__ == "__main__":
    mu_s = calculate_spin_magnetic_moment()
    print(f"Magnitude of electron spin magnetic moment: {mu_s:.6e} J/T")
