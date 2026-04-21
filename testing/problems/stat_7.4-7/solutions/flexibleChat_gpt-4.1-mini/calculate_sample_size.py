# filename: calculate_sample_size.py
import math
from scipy.stats import norm

def calculate_sample_size(error_margin=0.03, confidence_level=0.95, p=0.5):
    """
    Calculate the required sample size for estimating a population proportion with a specified margin of error and confidence level.
    
    Parameters:
    error_margin (float): Maximum allowable error (margin of error) in the estimate.
    confidence_level (float): Desired confidence level (between 0 and 1).
    p (float): Estimated population proportion (default 0.5 for maximum sample size).
    
    Returns:
    int: Required sample size rounded up to the nearest whole number.
    """
    if not (0 < error_margin < 1):
        raise ValueError("error_margin must be between 0 and 1.")
    if not (0 < confidence_level < 1):
        raise ValueError("confidence_level must be between 0 and 1.")
    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1.")

    # Calculate z-score for the given confidence level
    alpha = 1 - confidence_level
    z = norm.ppf(1 - alpha/2)

    # Calculate sample size
    n = (z**2 * p * (1 - p)) / (error_margin**2)

    return math.ceil(n)

sample_size = calculate_sample_size()
print(f"Required sample size: {sample_size}")
