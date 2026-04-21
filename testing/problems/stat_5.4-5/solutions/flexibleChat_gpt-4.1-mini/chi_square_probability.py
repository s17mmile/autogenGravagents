# filename: chi_square_probability.py
from scipy.stats import chi2

# Degrees of freedom
k = 7

# Given bounds
lower_bound = 1.69
upper_bound = 14.07

if __name__ == "__main__":
    # Calculate the cumulative distribution function values
    cdf_lower = chi2.cdf(lower_bound, k)
    cdf_upper = chi2.cdf(upper_bound, k)

    # Calculate the probability P(1.69 < W < 14.07)
    # W follows a chi-square distribution with 7 degrees of freedom
    probability = cdf_upper - cdf_lower

    print(f"P({lower_bound} < W < {upper_bound}) = {probability:.6f}")
