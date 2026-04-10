import numpy as np
from scipy.optimize import fsolve

# Constants
v_d_i = 14.9  # initial speed of deuteron in km/s
psi = 10  # scattering angle of deuteron in degrees
m_d = 2  # mass of deuteron (arbitrary units)
m_n = 1  # mass of neutron (arbitrary units)

# Convert angle to radians
psi_rad = np.radians(psi)

# Define the equations based on conservation of momentum and kinetic energy
def equations(vars):
    v_d_f, v_n_f, theta = vars
    eq1 = m_d * v_d_i - m_d * v_d_f * np.cos(psi_rad) - m_n * v_n_f * np.cos(theta)
    eq2 = 0.5 * m_d * v_d_i**2 - 0.5 * m_d * v_d_f**2 - 0.5 * m_n * v_n_f**2
    eq3 = v_n_f * np.sin(theta) - m_d * v_d_f * np.sin(psi_rad)
    return [eq1, eq2, eq3]

# Initial guesses for final speeds and angle
initial_guess = [10.0, 10.0, np.radians(45)]  # v_d_f, v_n_f, theta in radians

# Solve the equations
v_d_f, v_n_f, theta = fsolve(equations, initial_guess)

# Final speed of the deuteron
print(f'Final speed of the deuteron: {v_d_f:.2f} km/s')