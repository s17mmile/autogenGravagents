# filename: find_first_time_abs_u_equals_10.py
import numpy as np
from scipy.optimize import root_scalar

# Constants from the solution
omega = np.sqrt(23) / 6

# Define the solution u(t)
def u(t):
    A = 2
    B = -2 / np.sqrt(23)
    return np.exp(t / 6) * (A * np.cos(omega * t) + B * np.sin(omega * t))

# Define the function whose root we want to find: |u(t)| - 10 = 0
def func(t):
    return np.abs(u(t)) - 10

# We search for the first positive root
# Start searching from a small positive number
# We choose an interval [t_low, t_high] where the function changes sign
# Increase t_high exponentially until a sign change is detected or limit reached

t_low = 0.1

t_high = 1.0
max_t_high = 1000  # safeguard to avoid infinite loop

try:
    while func(t_low) * func(t_high) > 0:
        t_high *= 2
        if t_high > max_t_high:
            raise RuntimeError("Could not find an interval where the function changes sign.")

    # Use root_scalar to find the root in the interval
    sol = root_scalar(func, bracket=[t_low, t_high], method='brentq')

    if sol.converged:
        first_time = sol.root
        print(f'First time t > 0 where |u(t)| = 10: {first_time}')
    else:
        first_time = None
        print('Root finding did not converge.')

except Exception as e:
    first_time = None
    print(f'An error occurred during root finding: {e}')

# Save the result to a file
with open('first_time_abs_u_equals_10.txt', 'w') as f:
    if first_time is not None:
        f.write(f'First time t > 0 where |u(t)| = 10: {first_time}\n')
    else:
        f.write('Root finding did not converge or an error occurred.')
