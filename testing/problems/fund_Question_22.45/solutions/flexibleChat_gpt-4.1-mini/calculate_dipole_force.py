# filename: calculate_dipole_force.py
import math

def calculate_electrostatic_force(p, r):
    """
    Calculate the magnitude of the electrostatic force on an electron due to an electric dipole on its axial line.

    Parameters:
    p (float): Dipole moment in Coulomb-meters (C*m)
    r (float): Distance from the center of the dipole in meters (m), must be > 0

    Returns:
    float: Magnitude of the electrostatic force in newtons (N)
    """
    if r <= 0:
        raise ValueError("Distance r must be positive and non-zero.")

    epsilon_0 = 8.854e-12  # Permittivity of free space in C^2/(N*m^2)
    e = 1.602e-19          # Charge of electron in Coulombs

    E = (1 / (4 * math.pi * epsilon_0)) * (2 * p) / (r ** 3)
    F = e * E
    return F

# Given values
p = 3.6e-29  # Dipole moment in C*m
r = 25e-9    # Distance from dipole center in meters

force = calculate_electrostatic_force(p, r)
print(f"Magnitude of electrostatic force on the electron: {force:.3e} N")
