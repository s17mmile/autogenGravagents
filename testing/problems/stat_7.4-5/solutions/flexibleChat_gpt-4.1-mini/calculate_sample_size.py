# filename: calculate_sample_size.py
import scipy.stats as stats
import math

def required_sample_size(confidence_level, s, E, n0, max_iter=100):
    """
    Calculate the required sample size for estimating a population mean with a given confidence level,
    sample standard deviation, and maximum allowable error using iterative t-distribution approach.

    Parameters:
    confidence_level (float): Desired confidence level (e.g., 0.98 for 98%).
    s (float): Sample standard deviation.
    E (float): Maximum allowable error (margin of error).
    n0 (int): Initial sample size from preliminary data.
    max_iter (int): Maximum number of iterations to prevent infinite loops.

    Returns:
    int: Required sample size rounded up to the nearest integer.
    """
    if not (0 < confidence_level < 1):
        raise ValueError('confidence_level must be between 0 and 1')
    if s <= 0 or E <= 0:
        raise ValueError('Standard deviation and error must be positive')
    if not (isinstance(n0, int) and n0 > 0):
        raise ValueError('Initial sample size n0 must be a positive integer')

    alpha = 1 - confidence_level
    df = n0 - 1
    t_value = stats.t.ppf(1 - alpha/2, df)
    n = (t_value * s / E) ** 2
    n = math.ceil(n)

    for _ in range(max_iter):
        df = n - 1
        t_value_new = stats.t.ppf(1 - alpha/2, df)
        n_new = (t_value_new * s / E) ** 2
        n_new = math.ceil(n_new)
        if n_new == n:
            break
        n = n_new
    else:
        print('Warning: Maximum iterations reached without convergence')

    return n

# Given data
confidence_level = 0.98
s = 1.723
E = 0.25
n0 = 32

sample_size = required_sample_size(confidence_level, s, E, n0)
print(f'Required sample size: {sample_size}')

# Save result to a file
with open('required_sample_size.txt', 'w') as f:
    f.write(f'Required sample size: {sample_size}\n')
