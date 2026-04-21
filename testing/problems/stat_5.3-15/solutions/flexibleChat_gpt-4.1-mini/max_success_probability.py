# filename: max_success_probability.py
from scipy.stats import binom

"""
Calculate the probability that the maximum number of successes among three drugs,
each with 10 trials and success probability 0.7, exceeds a given threshold.
"""

# Parameters
n = 10
p = 0.7
threshold = 8

# Calculate P(X <= threshold) for X ~ Binomial(n, p)
prob_leq_threshold = binom.cdf(threshold, n, p)

# Calculate P(max(X1, X2, X3) > threshold) = 1 - P(max(X1, X2, X3) <= threshold)
# Since X1, X2, X3 are i.i.d., P(max <= threshold) = (P(X <= threshold))^3
prob_max_exceeds = 1 - prob_leq_threshold**3

print(f"Probability that the maximum number of successes exceeds {threshold}: {prob_max_exceeds:.4%}")
