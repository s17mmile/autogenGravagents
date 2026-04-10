import numpy as np
from scipy.optimize import fsolve

def equation(t):
    """Calculate the difference between the left-hand side and right-hand side of the equation."""
    return 32 * (1/4 * np.cos(2 * t) + 2 * np.sin(2 * t)) - 788 * np.exp(-1/4 * t)

# Initial guess for t
initial_guess = 0

# Use fsolve to find the root
solution, info, ier, msg = fsolve(equation, initial_guess, full_output=True)

if ier == 1:
    # Output the first value of t where y = 12
    first_intersection_t = solution[0]
    print(f'The first intersection occurs at t = {first_intersection_t:.4f}')
else:
    print('The solution did not converge:', msg)