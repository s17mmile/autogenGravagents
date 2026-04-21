# filename: equilibrium_three_charges.py
import numpy as np
from scipy.optimize import fsolve

# Constants
L = 0.09  # Separation between particle 1 and 2 in meters

# Define the system of equations for equilibrium on particles 1 and 3
# Variables: x3 (position of particle 3), x (charge ratio q3/q)
def reduced_equations(vars):
    x3, x = vars  # x3 in meters, x is q3/q
    
    r_13 = x3
    r_23 = L - x3
    
    # Forces on particle 1 (at 0):
    F_12 = 4 / L**2  # force from particle 2
    F_13 = x / r_13**2  # force from particle 3
    eq1 = -F_12 - F_13  # net force on particle 1 (right positive)
    
    # Forces on particle 3 (at x3):
    F_31 = 1 / r_13**2  # force from particle 1
    F_32 = 4 / r_23**2  # force from particle 2
    eq3 = F_31 - F_32  # net force on particle 3 (right positive)
    
    return (eq1, eq3)

# Initial guess: x3 = L/3, x = 1
initial_guess = (L/3, 1.0)

solution = fsolve(reduced_equations, initial_guess)
x3_sol, x_sol = solution

# Check net force on particle 2 for consistency
r_23_sol = L - x3_sol
F_21_sol = 1 / L**2
F_23_sol = x_sol / r_23_sol**2
eq2_sol = F_21_sol + F_23_sol

print(f"Equilibrium position of particle 3: {x3_sol*100:.2f} cm")
print(f"Charge ratio q3/q: {x_sol:.2f}")
print(f"Net force on particle 2 (should be close to zero): {eq2_sol:.2e} (arbitrary units)")

# Note: The solution assumes particle 3 lies between particles 1 and 2 and all charges are positive.
# Constants k and q^2 cancel out in equilibrium conditions, so they are omitted.