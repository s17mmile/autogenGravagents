# filename: find_min_beta_auto_bracket.py
import numpy as np
from scipy.optimize import brentq

def func(beta):
    return (beta + 6)**3 / (beta + 4)**2 - 27

# Automatically find an interval where func changes sign
beta_start = 0.1
beta_end = 50
beta_step = 0.1

beta_lower = None
beta_upper = None

beta_values = np.arange(beta_start, beta_end, beta_step)
for i in range(len(beta_values) - 1):
    a = beta_values[i]
    b = beta_values[i+1]
    if func(a) * func(b) < 0:
        beta_lower = a
        beta_upper = b
        break

if beta_lower is not None and beta_upper is not None:
    beta_root = brentq(func, beta_lower, beta_upper)
    print(f"The smallest value of beta for which y_m >= 4 is approximately: {beta_root:.6f}")
else:
    print("Failed to find an interval where the function changes sign. Try increasing the search range.")
