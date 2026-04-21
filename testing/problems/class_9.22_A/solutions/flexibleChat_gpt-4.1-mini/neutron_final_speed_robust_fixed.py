# filename: neutron_final_speed_robust_fixed.py
import math
from scipy.optimize import root

# Given constants
u_initial = 14.9  # initial speed of deuteron in km/s
psi_deg = 10  # scattering angle of deuteron in degrees
psi = math.radians(psi_deg)  # convert to radians

# Masses
m = 1  # neutron mass (arbitrary units)
M = 2 * m  # deuteron mass

# Rename variable for consistency
u_initial = 14.9

# Use consistent variable name u_initial
u_initial = 14.9

# Define the system of equations to solve for v_d (speed of deuteron after collision) and v_n (speed of neutron after collision)
# Variables: x = [v_d, v_n]
# We will try both signs of cos(theta) to resolve ambiguity

def equations(x, cos_theta_sign):
    v_d, v_n = x
    if v_n == 0:
        return [1e6, 1e6]  # avoid division by zero
    sin_theta = 2 * v_d * math.sin(psi) / v_n
    if abs(sin_theta) > 1:
        return [1e6, 1e6]  # no physical solution
    cos_theta = cos_theta_sign * math.sqrt(1 - sin_theta**2)

    eq1 = 2 * u_initial - 2 * v_d * math.cos(psi) - v_n * cos_theta
    eq2 = 2 * u_initial**2 - 2 * v_d**2 - v_n**2
    return [eq1, eq2]

# Try multiple initial guesses to improve convergence
initial_guesses = [
    [u_initial * 0.9, u_initial * 0.9],
    [u_initial * 0.5, u_initial * 1.0],
    [u_initial * 1.0, u_initial * 0.5],
    [u_initial * 1.1, u_initial * 1.1]
]

solutions = []
for cos_sign in [1, -1]:
    for guess in initial_guesses:
        sol = root(equations, guess, args=(cos_sign,), method='hybr')
        if sol.success:
            v_d, v_n = sol.x
            if v_d > 0 and v_n > 0:
                sin_theta = 2 * v_d * math.sin(psi) / v_n
                if abs(sin_theta) <= 1:
                    solutions.append((v_d, v_n, cos_sign, sol.fun))

# Select best solution (smallest residual norm)
if not solutions:
    raise ValueError('No physical solution found for the given parameters.')

best_solution = min(solutions, key=lambda s: sum(abs(x) for x in s[3]))
v_d_sol, v_n_sol, cos_theta_sign, _ = best_solution

sin_theta_sol = 2 * v_d_sol * math.sin(psi) / v_n_sol
cos_theta_sol = cos_theta_sign * math.sqrt(1 - sin_theta_sol**2)
theta_rad = math.atan2(sin_theta_sol, cos_theta_sol)
theta_deg = math.degrees(theta_rad)

# Output results
output = (
    f'Initial deuteron speed: {u_initial:.4f} km/s\n'
    f'Deuteron scattering angle (psi): {psi_deg} degrees\n'
    f'Final deuteron speed: {v_d_sol:.4f} km/s\n'
    f'Final neutron speed: {v_n_sol:.4f} km/s\n'
    f'Neutron scattering angle (theta): {theta_deg:.4f} degrees\n'
)
print(output)

with open('final_collision_results.txt', 'w') as f:
    f.write(output)
