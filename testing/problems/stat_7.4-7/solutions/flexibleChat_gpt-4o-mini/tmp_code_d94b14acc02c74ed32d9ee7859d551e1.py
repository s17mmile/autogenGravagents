import math

def calculate_sample_size(Z, E, estimated_proportion):
    if E <= 0:
        raise ValueError('Maximum error must be greater than zero.')
    n = (Z**2 * estimated_proportion * (1 - estimated_proportion)) / (E**2)
    return math.ceil(n)  # Round up to the nearest whole number

# Constants
Z = 1.96  # Z-score for 95% confidence
E = 0.03  # Maximum error
estimated_proportion = 0.5  # Estimated proportion (max variability)

# Calculate and output the required sample size
required_sample_size = calculate_sample_size(Z, E, estimated_proportion)
print(f'Required sample size: {required_sample_size}')