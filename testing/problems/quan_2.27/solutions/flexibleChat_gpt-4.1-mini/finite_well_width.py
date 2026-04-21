# filename: finite_well_width.py
import numpy as np
from scipy.optimize import fsolve

# Constants
m = 9.10938356e-31  # electron mass in kg
hbar = 1.0545718e-34  # reduced Planck constant in J*s
eV = 1.60218e-19  # electron volt in J

# Given values
V0_eV = 20.0  # well depth in eV
E_eV = 3.00   # lowest energy level above bottom in eV

# Convert energies to joules
V0 = V0_eV * eV
E = E_eV * eV

# Define k and kappa as functions of a (half-width a)
# k = sqrt(2mE)/hbar
# kappa = sqrt(2m(V0 - E))/hbar
k = np.sqrt(2 * m * E) / hbar
kappa = np.sqrt(2 * m * (V0 - E)) / hbar

# Transcendental equation for even solutions: tan(k*a) = kappa / k
# Define function f(a) = tan(k*a) - kappa/k

def transcendental_eq(a):
    return np.tan(k * a) - kappa / k

# Initial guess for a (in meters)
initial_guess = 1e-10

# Solve for a with error handling
try:
    solution = fsolve(transcendental_eq, initial_guess)
    a = solution[0]  # half-width in meters
    # Check if solution is physically meaningful
    if a <= 0:
        raise ValueError("Non-physical solution for half-width.")
except Exception:
    # Fallback initial guess
    initial_guess = 5e-10
    solution = fsolve(transcendental_eq, initial_guess)
    a = solution[0]

# Full width of the well
width = 2 * a

# Convert width to nanometers for readability
width_nm = width * 1e9

# Output the result
print(f"Width of the well: {width_nm:.3f} nm")
