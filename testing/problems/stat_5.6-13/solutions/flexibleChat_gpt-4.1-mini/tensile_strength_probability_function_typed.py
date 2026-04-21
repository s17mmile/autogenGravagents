# filename: tensile_strength_probability_function_typed.py
from scipy.stats import norm

def probability_sample_mean_greater_than(x_bar: float, mu: float, sigma: float, n: int) -> float:
    """Calculate the probability that the sample mean is greater than x_bar.

    Args:
        x_bar (float): The sample mean threshold.
        mu (float): Population mean.
        sigma (float): Population standard deviation (must be positive).
        n (int): Sample size (must be positive).

    Returns:
        float: Probability that sample mean > x_bar.
    """
    if sigma <= 0:
        raise ValueError("Population standard deviation must be positive.")
    if n <= 0:
        raise ValueError("Sample size must be positive.")

    standard_error = sigma / (n ** 0.5)
    z = (x_bar - mu) / standard_error
    probability = 1 - norm.cdf(z)
    return probability

# Given parameters
mu = 30
sigma = 3
n = 100
x_bar = 29.5

prob = probability_sample_mean_greater_than(x_bar, mu, sigma, n)
print(f"Probability that the sample mean is greater than {x_bar}: {prob:.4f}")
