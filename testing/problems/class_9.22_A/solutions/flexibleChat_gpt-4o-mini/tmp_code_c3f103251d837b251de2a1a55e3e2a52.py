import numpy as np
from scipy.optimize import fsolve

# Constants
v_d_i = 14900  # Initial speed of the deuteron in m/s
psi = 10  # Scattering angle of the deuteron in degrees

# Convert angle to radians
psi_rad = np.radians(psi)

# Function to calculate the difference based on the equations
# This function will return the squared differences for momentum conservation

def equations(v_n_f):
    # Calculate final speed of the deuteron
    if v_n_f >= v_d_i:
        return np.inf  # Prevent complex numbers by returning a large number
    v_d_f = np.sqrt(v_d_i**2 - v_n_f**2)  # Final speed of the deuteron
    # Assuming phi is derived from momentum equations
    phi_rad = np.arctan2(2 * v_d_f * np.sin(psi_rad), 29800 - 2 * v_d_f * np.cos(psi_rad))
    momentum_y = 2 * v_d_f * np.sin(psi_rad) - v_n_f * np.sin(phi_rad)
    momentum_x = 29800 - (2 * v_d_f * np.cos(psi_rad) + v_n_f * np.cos(phi_rad))
    return momentum_y**2 + momentum_x**2

# Initial guess for v_n_f
v_n_f_initial_guess = 0

# Numerical method to find the root
v_n_f_solution = fsolve(equations, v_n_f_initial_guess)

# Output the final speed of the neutron with units
print(f'Final speed of the neutron: {v_n_f_solution[0]:.2f} m/s')