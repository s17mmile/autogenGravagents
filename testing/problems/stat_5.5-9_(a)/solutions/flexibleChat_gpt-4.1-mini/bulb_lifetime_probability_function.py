# filename: bulb_lifetime_probability_function.py
from scipy.stats import norm
import math

def probability_bulb_A_exceeds_B_by_at_least(t=15, mu_X=800, var_X=14400, mu_Y=850, var_Y=2500):
    """Calculate P(X - Y >= t) where X ~ N(mu_X, var_X), Y ~ N(mu_Y, var_Y)"""
    # Validate inputs
    if var_X <= 0 or var_Y <= 0:
        raise ValueError("Variances must be positive.")
    sigma_X = math.sqrt(var_X)
    sigma_Y = math.sqrt(var_Y)
    mu_Z = mu_X - mu_Y
    sigma_Z = math.sqrt(sigma_X**2 + sigma_Y**2)
    z_score = (t - mu_Z) / sigma_Z
    probability = 1 - norm.cdf(z_score)
    return probability

# Example usage
if __name__ == '__main__':
    prob = probability_bulb_A_exceeds_B_by_at_least(15)
    print(f"Probability that the bulb from company A lasts at least 15 hours longer than the bulb from company B: {prob:.4f}")
