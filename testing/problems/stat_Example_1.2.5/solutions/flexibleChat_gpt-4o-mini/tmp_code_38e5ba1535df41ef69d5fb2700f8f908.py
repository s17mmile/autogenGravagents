import math

def calculate_permutations(n, r):
    if n < r:
        raise ValueError('Number of persons must be greater than or equal to the number of positions.')
    return math.factorial(n) // math.factorial(n - r)

# Define the total number of persons and the number of positions to fill
n = 10
r = 4

# Calculate the number of ways to select the officers
number_of_ways = calculate_permutations(n, r)

# Return the result
number_of_ways