# filename: solve_time_for_y_bisect.py
import numpy as np
from scipy.optimize import bisect

def equation(T):
    # Transcendental equation: 1 + T = 0.2237 * exp(T)
    return 1 + T - 0.2237 * np.exp(T)

# We need to find an interval [a, b] where equation(a) and equation(b) have opposite signs
# Test some values to find such an interval

# Check values at 0 and 5
# equation(0) = 1 + 0 - 0.2237*exp(0) = 1 - 0.2237 = 0.7763 > 0
# equation(5) = 1 + 5 - 0.2237*exp(5) = 6 - 0.2237*148.413 = 6 - 33.2 = -27.2 < 0

# So the root lies between 0 and 5

T_solution = bisect(equation, 0, 5)

print(f"The time T at which y(T) = 3.99 is approximately: {T_solution:.6f}")
