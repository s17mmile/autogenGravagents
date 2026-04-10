import math

def calculate_sample_size(confidence_level, margin_of_error, p):
    if margin_of_error <= 0:
        raise ValueError('Margin of error must be greater than zero.')
    
    # Z-score for the given confidence level
    z_score = 1.96  # For 95% confidence level
    
    # Sample size calculation
    n = (z_score ** 2 * p * (1 - p)) / (margin_of_error ** 2)
    
    # Round up to the nearest whole number
    return math.ceil(n)

# Parameters
confidence_level = 0.95
margin_of_error = 0.10
p = 0.5  # Conservative estimate for maximum variability

# Calculate required sample size
required_sample_size = calculate_sample_size(confidence_level, margin_of_error, p)

# Output the required sample size
print(f'Required sample size: {required_sample_size}')