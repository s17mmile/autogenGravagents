# filename: neutron_final_speed_improved.py
import math
from scipy.optimize import fsolve

# Given constants
u_initial = 14.9  # initial speed of deuteron in km/s
psi_deg = 10  # scattering angle of deuteron in degrees
psi = math.radians(psi_deg)  # convert to radians

# Masses
m = 1  # neutron mass (arbitrary units)
M = 2 * m  # deuteron mass

# Define the system of equations to solve for v_d (speed of deuteron after collision) and v_n (speed of neutron after collision)
# Variables: x = [v_d, v_n]
# We will try both signs of cos(theta) to resolve ambiguity

def equations(x, cos_theta_sign):
    v_d, v_n = x
    # From y-momentum: 0 = M v_d sin(psi) - m v_n sin(theta)
    # => v_n sin(theta) = 2 v_d sin(psi)
    # sin(theta) = 2 v_d sin(psi) / v_n
    if v_n == 0:
        return [1e6, 1e6]  # avoid division by zero
    sin_theta = 2 * v_d * math.sin(psi) / v_n
    if abs(sin_theta) > 1:
        # No physical solution
        return [1e6, 1e6]
    cos_theta = cos_theta_sign * math.sqrt(1 - sin_theta**2)

    # x-momentum equation
    eq1 = 2 * nu_initial - 2 * v_d * math.cos(psi) - v_n * cos_theta

    # kinetic energy conservation
    eq2 = 2 * nu_initial**2 - 2 * v_d**2 - v_n**2

    return [eq1, eq2]

# Initial guess for v_d and v_n
initial_guess = [nu_initial * 0.9, nu_initial * 0.9]

# Try positive cos(theta)
solution_pos = fsolve(equations, initial_guess, args=(1,))
v_d_pos, v_n_pos = solution_pos

# Validate solution
valid_pos = False
if v_d_pos > 0 and v_n_pos > 0:
    sin_theta_pos = 2 * v_d_pos * math.sin(psi) / v_n_pos
    if abs(sin_theta_pos) <= 1:
        valid_pos = True

# Try negative cos(theta)
solution_neg = fsolve(equations, initial_guess, args=(-1,))
v_d_neg, v_n_neg = solution_neg

valid_neg = False
if v_d_neg > 0 and v_n_neg > 0:
    sin_theta_neg = 2 * v_d_neg * math.sin(psi) / v_n_neg
    if abs(sin_theta_neg) <= 1:
        valid_neg = True

# Choose physically valid solution with positive speeds
if valid_pos and valid_neg:
    # Choose solution with neutron speed closer to initial deuteron speed (heuristic)
    if abs(v_n_pos - nu_initial) < abs(v_n_neg - nu_initial):
        v_d_sol, v_n_sol, cos_theta_sign = v_d_pos, v_n_pos, 1
    else:
        v_d_sol, v_n_sol, cos_theta_sign = v_d_neg, v_n_neg, -1
elif valid_pos:
    v_d_sol, v_n_sol, cos_theta_sign = v_d_pos, v_n_pos, 1
elif valid_neg:
    v_d_sol, v_n_sol, cos_theta_sign = v_d_neg, v_n_neg, -1
else:
    raise ValueError('No physical solution found for the given parameters.')

# Calculate neutron scattering angle theta in degrees
sin_theta_sol = 2 * v_d_sol * math.sin(psi) / v_n_sol
cos_theta_sol = cos_theta_sign * math.sqrt(1 - sin_theta_sol**2)
theta_rad = math.atan2(sin_theta_sol, cos_theta_sol)
theta_deg = math.degrees(theta_rad)

# Save results to a file
with open('final_collision_results.txt', 'w') as f:
    f.write(f'Initial deuteron speed: {nu_initial:.4f} km/s\n')
    f.write(f'Deuteron scattering angle (psi): {psi_deg} degrees\n')
    f.write(f'Final deuteron speed: {v_d_sol:.4f} km/s\n')
    f.write(f'Final neutron speed: {v_n_sol:.4f} km/s\n')
    f.write(f'Neutron scattering angle (theta): {theta_deg:.4f} degrees\n')
