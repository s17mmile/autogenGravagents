import math

def calculate_combinations(n, k):
    if k > n:
        return 0  # Return 0 if k is greater than n
    return math.comb(n, k)  # Use math.comb for efficiency

# Constants for the problem
n = 52  # Total number of cards
k = 13  # Number of cards to choose

# Calculate the number of combinations
number_of_hands = calculate_combinations(n, k)

# Print the result
print(number_of_hands)