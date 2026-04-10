import math

def calculate_combinations(n, k):
    """
    Calculate the number of combinations of n items taken k at a time.
    :param n: Total number of items (must be non-negative integer)
    :param k: Number of items to choose (must be non-negative integer)
    :return: Number of combinations
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError('Invalid input: n must be >= k and both must be non-negative.')
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))

# Constants for the problem
n = 52  # Total number of cards
k = 5   # Number of cards to choose

# Calculate the number of combinations
number_of_hands = calculate_combinations(n, k)

# Print the result
print(number_of_hands)  # Output: 2598960

# Test case to verify the function
assert calculate_combinations(52, 5) == 2598960, "Test case failed!"