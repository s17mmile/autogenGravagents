# filename: compute_sample_mean_probability.py
from scipy.stats import norm

# Given parameters
mu = 77  # population mean
sigma = 5  # population standard deviation (sqrt(25))
n = 16  # sample size

# Sampling distribution parameters
mu_bar = mu
sigma_bar = sigma / (n ** 0.5)

# Bounds
lower_bound = 74.2
upper_bound = 78.4

# Calculate the cumulative probabilities for the bounds
cdf_lower = norm.cdf(lower_bound, loc=mu_bar, scale=sigma_bar)
cdf_upper = norm.cdf(upper_bound, loc=mu_bar, scale=sigma_bar)

# Probability that sample mean lies between the bounds
probability = cdf_upper - cdf_lower

print(f"P({lower_bound} < X_bar < {upper_bound}) = {probability:.6f}")
