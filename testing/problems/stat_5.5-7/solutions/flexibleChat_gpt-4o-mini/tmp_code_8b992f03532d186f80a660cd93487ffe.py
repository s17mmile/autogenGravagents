import numpy as np
from scipy.stats import norm

# Parameters for the distributions
mean_1pound = 1.18
stddev_1pound = 0.07
mean_3pound = 3.22
stddev_3pound = 0.09

# Distribution of the sum of three 1-pound bags
mean_sum_1pound = 3 * mean_1pound  # Mean of Y
stddev_sum_1pound = np.sqrt(3) * stddev_1pound  # Standard deviation of Y

# Distribution of the weight of one 3-pound bag
mean_weight_3pound = mean_3pound  # Mean of W
stddev_weight_3pound = stddev_3pound  # Standard deviation of W

# Distribution of the difference Z = Y - W
mean_diff = mean_sum_1pound - mean_weight_3pound  # Mean of Z
stddev_diff = np.sqrt(stddev_sum_1pound**2 + stddev_weight_3pound**2)  # Standard deviation of Z

# Error handling for standard deviation
if stddev_diff == 0:
    raise ValueError('Standard deviation of the difference cannot be zero.')

# Calculate the probability P(Y > W) = P(Z > 0)
# Standardize Z
z_standard = -mean_diff / stddev_diff

# Calculate the probability using the cumulative distribution function
probability = norm.sf(z_standard)  # P(Z > 0)

# Output the result
print(f'The probability that the sum of three 1-pound bags exceeds the weight of one 3-pound bag is approximately {probability:.3f} or {probability * 100:.2f}%.')