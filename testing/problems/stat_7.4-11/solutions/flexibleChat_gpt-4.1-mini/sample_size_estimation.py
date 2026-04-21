# filename: sample_size_estimation.py
import math
from scipy.stats import norm

def calculate_sample_size(p, margin_error, confidence_level):
    """Calculate the minimum sample size needed for a confidence interval of a proportion.

    Args:
        p (float): Estimated proportion.
        margin_error (float): Desired margin of error.
        confidence_level (float): Confidence level (e.g., 0.95 for 95%).

    Returns:
        int: Minimum required sample size.
    """
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    n = (z**2 * p * (1 - p)) / (margin_error**2)
    return math.ceil(n)

# Given values
p = 0.89
margin_error = 0.10
confidence_level = 0.95

required_sample_size = calculate_sample_size(p, margin_error, confidence_level)
print(f"Required sample size n: {required_sample_size}")
