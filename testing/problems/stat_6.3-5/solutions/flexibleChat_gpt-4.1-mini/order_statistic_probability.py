# filename: order_statistic_probability.py
from scipy.stats import binom

# Number of observations
n = 8
# Probability that a single observation is less than 27.3 (70th percentile)
p = 0.7
# Order statistic index
k = 7

# S is the number of observations less than 27.3
# P(Y_7 < 27.3) = P(S >= 7) = P(S=7) + P(S=8)
prob_Y7_less_27_3 = binom.pmf(k, n, p) + binom.pmf(n, n, p)

print(f"P(Y_7 < 27.3) = {prob_Y7_less_27_3:.6f}")
