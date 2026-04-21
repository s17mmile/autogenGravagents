# filename: solve_time_for_y.py
import numpy as np
from scipy.optimize import fsolve

def equation(T):
    # Transcendental equation derived from y(T) = 3.99:
    # 1 + T = 0.2237 * exp(T)
    return 1 + T - 0.2237 * np.exp(T)

# Initial guess for T
T_initial_guess = 1.0

# Solve for T using fsolve
T_solution, info, ier, mesg = fsolve(equation, T_initial_guess, full_output=True)

if ier == 1 and T_solution[0] >= 0:
    print(f"The time T at which y(T) = 3.99 is approximately: {T_solution[0]:.6f}")
else:
    print("Root finding did not converge or found a negative time solution.")
