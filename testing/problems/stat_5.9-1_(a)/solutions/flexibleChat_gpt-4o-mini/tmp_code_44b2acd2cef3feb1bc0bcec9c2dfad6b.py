import scipy.stats as stats

# Parameters
n = 50  # number of articles
p = 0.01  # probability of an article being defective

# Calculate probabilities for Y = 0, 1, 2, 3
probabilities = {k: stats.binom.pmf(k, n, p) for k in range(4)}

# Print individual probabilities
for k in range(4):
    print(f'P(Y = {k}) = {probabilities[k]:.5f}')

# Calculate total probability for Y = 0, 1, 2, 3
total_probability = sum(probabilities.values())
print(f'Total Probability P(Y = 0, 1, 2, 3) = {total_probability:.5f}')