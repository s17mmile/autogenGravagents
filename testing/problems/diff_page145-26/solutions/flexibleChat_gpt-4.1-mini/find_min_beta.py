# filename: find_min_beta.py
import numpy as np
from scipy.optimize import brentq

def func(beta):
    """Function whose root corresponds to the smallest beta satisfying (beta+6)^3/(beta+4)^2 = 27."""
    return (beta + 6)**3 / (beta + 4)**2 - 27

# Define interval to bracket the root
beta_lower = 0.1
beta_upper = 10

try:
    beta_root = brentq(func, beta_lower, beta_upper)
    print(f"The smallest value of beta for which y_m >= 4 is approximately: {beta_root:.6f}")
except ValueError as e:
    print(f"Root finding failed: {e}")
