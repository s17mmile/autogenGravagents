# filename: binomial_defectives_prob_validated.py
from scipy.stats import binom

# Parameters
n = 50  # number of articles
p = 0.01  # probability of defective

# Input validation
if not (isinstance(n, int) and n > 0):
    raise ValueError("n must be a positive integer")
if not (0 <= p <= 1):
    raise ValueError("p must be between 0 and 1 inclusive")

# Calculate probabilities for Y = 0, 1, 2, 3
probabilities = {k: binom.pmf(k, n, p) for k in range(4)}

# Print the results
for k, prob in probabilities.items():
    print(f"P(Y = {k}) = {prob:.6f}")

# Also return the dictionary for further use if needed
probabilities
