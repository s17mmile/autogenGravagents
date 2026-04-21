# filename: calculate_n_for_integration_accuracy.py

import math

def calculate_min_n_for_accuracy(error_tolerance=0.0001):
    """
    Calculate the minimum number of subintervals n required for the Trapezoidal and Midpoint Rules
    to approximate the integral of 1/x from 1 to 2 with an error less than error_tolerance.

    Parameters:
    error_tolerance (float): Desired maximum error tolerance.

    Returns:
    tuple: (n_trapezoidal, n_midpoint) minimum n values for each method.
    """
    a, b = 1, 2
    max_f2 = 2  # max of |f''(x)| on [1,2] for f(x) = 1/x

    interval_length_cubed = (b - a)**3

    n_trap = math.ceil(math.sqrt((interval_length_cubed * max_f2) / (12 * error_tolerance)))
    n_mid = math.ceil(math.sqrt((interval_length_cubed * max_f2) / (24 * error_tolerance)))

    return n_trap, n_mid

if __name__ == '__main__':
    n_trapezoidal, n_midpoint = calculate_min_n_for_accuracy()
    print(f"Minimum n for Trapezoidal Rule: {n_trapezoidal}")
    print(f"Minimum n for Midpoint Rule: {n_midpoint}")
