# filename: t_distribution_quantile_function.py
from scipy.stats import t

def t_quantile(df, alpha):
    """Calculate the lower tail quantile for Student's t-distribution.
    df: degrees of freedom
    alpha: cumulative probability for the quantile (e.g., 0.01 for 1%)
    Returns the quantile value."""
    return t.ppf(alpha, df)

# Parameters
degrees_of_freedom = 17
significance_level = 0.01

# Calculate quantile
quantile_value = t_quantile(degrees_of_freedom, significance_level)

print(f"t_0.01({degrees_of_freedom}) = {quantile_value}")
