import numpy as np
from scipy.integrate import quad

# Constants
proton_radius_cm = 1e-13  # Radius of the proton in cm
proton_radius_m = proton_radius_cm * 1e-2  # Convert to meters
epsilon_0 = 8.854187817e-12  # Vacuum permittivity in F/m
e = 1.602176634e-19  # Charge of the electron in C
a_0 = 5.29e-11  # Bohr radius in meters

# Potential energy function inside the nucleus
def V(r):
    """Calculate potential energy inside the nucleus."""
    return - (e**2 * r**2) / (4 * np.pi * epsilon_0 * proton_radius_m**3)

# Ground state wave function squared
def psi_squared(r):
    """Calculate the square of the ground state wave function of hydrogen."""
    return (1 / (np.pi * a_0**3)) * np.exp(-2 * r / a_0)

# Integrand for the energy shift
def integrand(r):
    """Integrand for the energy shift calculation."""
    return psi_squared(r) * V(r) * r**2

# Calculate the energy shift using numerical integration
integral, error = quad(integrand, 0, proton_radius_m)
Delta_E = - (e**2 / (4 * np.pi * epsilon_0 * proton_radius_m**3)) * (4 * integral)

# Output the energy shift and error estimate
print(f'Estimated energy shift due to finite proton size: {Delta_E:.3e} J, with error estimate: {error:.3e}')