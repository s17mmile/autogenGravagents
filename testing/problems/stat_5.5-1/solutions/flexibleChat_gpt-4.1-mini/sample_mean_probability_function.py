# filename: sample_mean_probability_function.py
from scipy.stats import norm

def sample_mean_probability(mu, sigma, n, lower_bound, upper_bound):
    """
    Calculate the probability that the sample mean \u03C1 lies between lower_bound and upper_bound
    for a sample of size n from a normal distribution N(mu, sigma^2).

    Parameters:
    - mu: population mean
    - sigma: population standard deviation (must be non-negative)
    - n: sample size (must be positive integer)
    - lower_bound: lower bound for sample mean
    - upper_bound: upper bound for sample mean

    Returns:
    - probability: P(lower_bound < X_bar < upper_bound)
    """
    if n <= 0:
        raise ValueError("Sample size n must be positive.")
    if sigma < 0:
        raise ValueError("Standard deviation sigma must be non-negative.")
    if lower_bound > upper_bound:
        raise ValueError("Lower bound must be less than or equal to upper bound.")

    sigma_bar = sigma / (n ** 0.5)
    cdf_lower = norm.cdf(lower_bound, loc=mu, scale=sigma_bar)
    cdf_upper = norm.cdf(upper_bound, loc=mu, scale=sigma_bar)
    probability = cdf_upper - cdf_lower
    return probability

# Example usage with given parameters
if __name__ == "__main__":
    mu = 77
    sigma = 5
    n = 16
    lower_bound = 77
    upper_bound = 79.5

    prob = sample_mean_probability(mu, sigma, n, lower_bound, upper_bound)
    print(f"P({lower_bound} < X_bar < {upper_bound}) = {prob:.4f}")
