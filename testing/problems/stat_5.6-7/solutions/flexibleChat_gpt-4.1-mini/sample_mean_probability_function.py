# filename: sample_mean_probability_function.py
from scipy.stats import norm
import math

def sample_mean_probability(mu, sigma, n, lower_bound, upper_bound):
    """
    Calculate the probability that the sample mean \u03C1(X_bar) lies between lower_bound and upper_bound
    for a sample of size n from a population with mean mu and standard deviation sigma.
    """
    # Validate inputs
    if n <= 0:
        raise ValueError("Sample size n must be positive.")
    if sigma <= 0:
        raise ValueError("Standard deviation sigma must be positive.")
    if lower_bound > upper_bound:
        raise ValueError("Lower bound must be less than or equal to upper bound.")

    # Calculate standard error
    standard_error = sigma / math.sqrt(n)

    # Calculate Z-scores for the bounds
    z_lower = (lower_bound - mu) / standard_error
    z_upper = (upper_bound - mu) / standard_error

    # Calculate the probability P(lower_bound <= X_bar <= upper_bound)
    probability = norm.cdf(z_upper) - norm.cdf(z_lower)
    return probability

# Given parameters
mu = 54.030
sigma = 5.8
n = 47
lower_bound = 52.761
upper_bound = 54.453

# Calculate and print the probability
prob = sample_mean_probability(mu, sigma, n, lower_bound, upper_bound)
print(f"P({lower_bound} <= X_bar <= {upper_bound}) = {prob:.4f}")
