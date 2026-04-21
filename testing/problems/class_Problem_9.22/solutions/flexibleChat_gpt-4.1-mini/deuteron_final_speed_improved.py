# filename: deuteron_final_speed_improved.py
import numpy as np
from scipy.optimize import fsolve

# Define the system of equations based on conservation laws
# vars: [v_d (final speed of deuteron), theta (neutron scattering angle in radians)]
def equations(vars, u_initial, psi_rad):
    v_d, theta = vars
    # Momentum conservation in x-direction
    eq1 = u_initial - v_d * (np.cos(psi_rad) + np.sin(psi_rad) / np.tan(theta))
    # Kinetic energy conservation
    eq2 = (2 * u_initial**2) / v_d**2 - 2 - 4 * (np.sin(psi_rad)**2) / (np.sin(theta)**2)
    return [eq1, eq2]

# Given initial speed of deuteron (km/s) and scattering angle psi (degrees)
u_initial = 14.9  # km/s
psi_degrees = 10
psi_radians = np.radians(psi_degrees)

# Initial guess for [v_d, theta]
initial_guess = [10.0, np.radians(30)]

# Solve the nonlinear system
solution = fsolve(equations, initial_guess, args=(u_initial, psi_radians))
v_d_final, theta_rad = solution

# Validate solution: speeds should be positive and angles within physical range
if v_d_final <= 0 or theta_rad <= 0 or theta_rad >= np.pi:
    print("Warning: Non-physical solution found.")

# Convert neutron scattering angle to degrees
theta_degrees = np.degrees(theta_rad)

# Output results
print(f"Final speed of the deuteron: {v_d_final:.4f} km/s")
print(f"Scattering angle of the neutron: {theta_degrees:.2f} degrees")
