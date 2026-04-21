# filename: binomial_probability_lower_bound.py
import scipy.stats as stats
from math import ceil, floor

def binomial_probability_lower_bound(n, p, delta):
    """
    Calculate a lower bound for P(|Y/n - p| < delta) where Y ~ Binomial(n, p)
    using normal approximation with continuity correction.
    Also computes exact binomial probability for comparison.

    Parameters:
    n (int): number of trials, must be positive
    p (float): success probability, 0 < p < 1
    delta (float): deviation threshold, 0 < delta < 1

    Returns:
    tuple: (normal_approx_prob, exact_binom_prob)
    """
    # Input validation
    if not (isinstance(n, int) and n > 0):
        raise ValueError("n must be a positive integer")
    if not (0 < p < 1):
        raise ValueError("p must be between 0 and 1 (exclusive)")
    if not (0 < delta < 1):
        raise ValueError("delta must be between 0 and 1 (exclusive)")

    mu = n * p
    sigma = (n * p * (1 - p)) ** 0.5

    # Calculate bounds for Y
    lower_bound = n * (p - delta)
    upper_bound = n * (p + delta)

    # Apply continuity correction by adjusting bounds by 0.5
    lower_cc = lower_bound + 0.5
    upper_cc = upper_bound - 0.5

    # Normal approximation probability
    prob_lower = stats.norm.cdf(upper_cc, loc=mu, scale=sigma)
    prob_upper = stats.norm.cdf(lower_cc, loc=mu, scale=sigma)
    normal_approx_prob = prob_lower - prob_upper

    # Exact binomial probability
    # P(lower_bound < Y < upper_bound) = P(ceil(lower_bound) <= Y <= floor(upper_bound))
    lower_int = ceil(lower_bound)
    upper_int = floor(upper_bound)
    exact_binom_prob = stats.binom.cdf(upper_int, n, p) - stats.binom.cdf(lower_int - 1, n, p)

    return normal_approx_prob, exact_binom_prob


if __name__ == '__main__':
    n = 100
    p = 0.25
    delta = 0.05

    normal_prob, exact_prob = binomial_probability_lower_bound(n, p, delta)

    print(f"Estimated probability using normal approximation with continuity correction: {normal_prob:.4f}")
    print(f"Exact binomial probability: {exact_prob:.4f}")
