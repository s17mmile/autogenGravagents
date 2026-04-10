import scipy.stats as stats

def calculate_probability(lower_bound, upper_bound, df):
    if df <= 0:
        raise ValueError('Degrees of freedom must be positive.')
    p1 = stats.chi2.cdf(upper_bound, df)
    p2 = stats.chi2.cdf(lower_bound, df)
    return p1 - p2

df = 7  # Degrees of freedom
lower_bound = 1.69
upper_bound = 14.07

# Calculate the probability
probability = calculate_probability(lower_bound, upper_bound, df)
print(f'Probability that {lower_bound} < W < {upper_bound}: {probability}')