import math

# Function to calculate combinations with input validation

def combinations(n, k):
    if k > n:
        return 0  # Not enough items to choose from
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

# Calculate the number of ways to choose 3 kings from 4
kings = combinations(4, 3)

# Calculate the number of ways to choose 2 queens from 4
queens = combinations(4, 2)

# Calculate the total number of ways to choose 5 cards from 52
 total_outcomes = combinations(52, 5)

# Calculate the number of favorable outcomes
favorable_outcomes = kings * queens

# Calculate the probability
probability = favorable_outcomes / total_outcomes

# Print the result with percentage formatting
percentage_probability = probability * 100
print(f'Probability of drawing 3 kings and 2 queens: {probability:.10f} (or {percentage_probability:.10f}%)')