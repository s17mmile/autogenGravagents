from scipy.stats import f


def calculate_f_critical_value(df1, df2, alpha):
    """
    Calculate the critical value of the F-distribution.
    
    Parameters:
    df1 (int): Degrees of freedom for the numerator.
    df2 (int): Degrees of freedom for the denominator.
    alpha (float): Significance level.
    
    Returns:
    float: Critical value of the F-distribution.
    """
    if df1 <= 0 or df2 <= 0:
        raise ValueError('Degrees of freedom must be positive.')
    return f.ppf(alpha, df1, df2)

# Degrees of freedom

df1 = 8  # numerator degrees of freedom

df2 = 4  # denominator degrees of freedom

# Significance level (0.01 for the 1st percentile)
alpha = 0.01

# Calculate the critical value for F-distribution
critical_value = calculate_f_critical_value(df1, df2, alpha)

# Print the critical value
print(critical_value)