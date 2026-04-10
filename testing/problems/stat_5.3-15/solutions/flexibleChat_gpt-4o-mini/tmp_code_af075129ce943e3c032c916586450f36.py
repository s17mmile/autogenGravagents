import numpy as np
from scipy.stats import binom

# Constants
n = 10  # total number of trials
p = 0.7  # probability of success

# Function to calculate the probability that max successes exceed 8
def probability_max_success_exceeds_eight(n, p):
    prob_max_leq_8 = 0.0
    # Iterate through all possible distributions of patients among the drugs
    for n1 in range(0, n + 1):
        for n2 in range(0, n - n1 + 1):
            n3 = n - n1 - n2
            if n3 < 0:
                continue
            # Calculate P(X1 <= 8), P(X2 <= 8), P(X3 <= 8) using CDF
            p1 = binom.cdf(8, n1, p)
            p2 = binom.cdf(8, n2, p)
            p3 = binom.cdf(8, n3, p)
            # Multiply probabilities for the current distribution
            prob_max_leq_8 += p1 * p2 * p3 * (1 / 3) ** (n1 + n2 + n3)

    # Probability that max successes exceed 8
    prob_max_exceeds_8 = 1 - prob_max_leq_8
    return prob_max_exceeds_8

# Calculate the probability
result = probability_max_success_exceeds_eight(n, p)
print(f'The probability that the maximum number of successes exceeds 8 is: {result:.4f}')