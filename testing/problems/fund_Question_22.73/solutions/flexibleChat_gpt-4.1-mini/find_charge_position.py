# filename: find_charge_position.py
import numpy as np
from scipy.optimize import fsolve

def equations(vars):
    x_p, y_p = vars
    # Points where E is given (converted to meters)
    p1 = np.array([3.0, 3.0]) * 1e-2  # m
    p2 = np.array([2.0, 0.0]) * 1e-2  # m

    # Electric fields given
    E1_vec = 7.2 * np.array([4.0, 3.0])  # N/C
    E2_vec = np.array([100.0, 0.0])      # N/C

    # Vectors from charge to points (in meters)
    r1 = p1 - np.array([x_p, y_p])
    r2 = p2 - np.array([x_p, y_p])

    r1_mag = np.linalg.norm(r1)
    r2_mag = np.linalg.norm(r2)

    epsilon = 1e-12  # small number to avoid division by zero

    # Calculate kq from E1 components
    kq_x = E1_vec[0] * r1_mag**3 / (r1[0] + epsilon)
    kq_y = E1_vec[1] * r1_mag**3 / (r1[1] + epsilon)

    # Equation 1: consistency of kq from x and y components at point 1
    eq1 = kq_x - kq_y

    # Equation 2: y component of r2 should be zero because E2 is along x only
    eq2 = r2[1]

    return [eq1, eq2]

# Initial guess for (x_p, y_p) in meters
initial_guess = [0.0, 0.0]

solution = fsolve(equations, initial_guess)
x_p_solution = solution[0]
y_p_solution = solution[1]

# Convert back to cm for output
x_p_cm = x_p_solution * 100
y_p_cm = y_p_solution * 100

print(f"The x coordinate of the particle is approximately {x_p_cm:.4f} cm.")
print(f"The y coordinate of the particle is approximately {y_p_cm:.4f} cm.")
