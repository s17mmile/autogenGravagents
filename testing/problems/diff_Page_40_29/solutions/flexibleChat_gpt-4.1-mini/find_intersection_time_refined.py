# filename: find_intersection_time_refined.py
import numpy as np
from scipy.optimize import root_scalar

def equation(t):
    # Coefficients from the solution
    A = 8
    B = 64
    R = np.sqrt(A**2 + B**2)  # amplitude
    alpha = np.arctan2(B, A)  # angle for single sinusoid representation
    # Left side: sin(2t + alpha)
    left = np.sin(2*t + alpha)
    # Right side: RHS constant times exp(-t/4)
    rhs_const = 788 / 65  # from original problem
    right = rhs_const * np.exp(-t/4)
    return left - right

# Starting search interval based on reasoning
t_start = 10.0
t_end = 20.0
step = 0.1
prev_val = equation(t_start)
root_t = None

# Scan to find sign change intervals
for t in np.arange(t_start + step, t_end + step, step):
    curr_val = equation(t)
    if prev_val * curr_val <= 0:  # sign change detected
        sol = root_scalar(equation, bracket=[t - step, t], method='brentq')
        if sol.converged:
            root_t = sol.root
            break
    prev_val = curr_val

if root_t is not None:
    print(f'The first intersection with y=12 occurs at t approximately {root_t:.5f}.')
else:
    print('No root found in the interval [10, 20]. Consider expanding the search range.')
