# filename: compute_dropout_probability.py
import numpy as np
from scipy.stats import binom

n = 25
p = 0.5

# Precompute binomial pmf for X and Y
pmf_X = binom.pmf(np.arange(n+1), n, p)
pmf_Y = binom.pmf(np.arange(n+1), n, p)

prob = 0.0

# k corresponds to Z = Y - X + 25, range 27 to 50 for Y >= X + 2
for k in range(27, 51):
    # i must satisfy 0 <= i <= 25 and 0 <= i + k - 25 <= 25
    i_min = max(0, 25 - k)  # lower bound for i
    i_max = min(n, 50 - k)  # upper bound for i
    for i in range(i_min, i_max + 1):
        j = i + k - 25  # index for Y
        # Validate indices are within bounds
        assert 0 <= i <= n
        assert 0 <= j <= n
        prob += pmf_X[i] * pmf_Y[j]

print(f"Probability P(Y >= X + 2) = {prob:.6f}")
