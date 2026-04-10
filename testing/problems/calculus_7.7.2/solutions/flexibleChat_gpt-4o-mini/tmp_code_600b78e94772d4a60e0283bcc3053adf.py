import math

# Constants
error_tolerance = 0.0001

# Error estimation for Trapezoidal Rule
# E_T = (b-a)^3 / (12*n^2) * |f''(xi)|
# f''(x) = 2/x^3, max at x=1, |f''(1)| = 2
b = 2
a = 1
max_f_double_prime = 2

# Calculate n for Trapezoidal Rule
n_trapezoidal = math.ceil(math.sqrt((b - a)**3 / (12 * error_tolerance)))

# Error estimation for Midpoint Rule
# E_M = (b-a)^3 / (24*n^2) * |f''(xi)|
# Calculate n for Midpoint Rule
n_midpoint = math.ceil(math.sqrt((b - a)**3 / (24 * error_tolerance)))

# Determine the larger n
n_required = max(n_trapezoidal, n_midpoint)

print(f'To ensure accuracy within {error_tolerance}, n should be at least: {n_required}')