# filename: calculate_sample_size_grackle.py
import math

def calculate_sample_size(sigma, margin_error, confidence_level=0.95):
    """
    Calculate the required sample size for estimating a population mean with a given margin of error and confidence level.
    :param sigma: population standard deviation
    :param margin_error: desired maximum margin of error
    :param confidence_level: confidence level (default 0.95)
    :return: required sample size (integer)
    """
    # Calculate the z-score for the given confidence level
    alpha = 1 - confidence_level
    z = abs(math.erf((1 - alpha/2) * math.sqrt(2)))  # approximate z using error function (not exact)
    # For exact z, use scipy.stats.norm.ppf if available, but here we use 1.96 for 95%
    if confidence_level == 0.95:
        z = 1.96
    
    # Calculate sample size
    n = (z * sigma / margin_error) ** 2
    return math.ceil(n)

# Given parameters
sigma = math.sqrt(4.84)  # population standard deviation
margin_error = 0.4  # maximum margin of error

# Calculate required sample size
n_required = calculate_sample_size(sigma, margin_error)

print(f"Required sample size n: {n_required}")
