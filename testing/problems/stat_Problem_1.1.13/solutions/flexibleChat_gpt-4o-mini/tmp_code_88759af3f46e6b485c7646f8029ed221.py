import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Number of random points to simulate
num_points = 1000000

# Generate random points along the line segment [0, 1]
random_points = np.random.uniform(0, 1, num_points)

# Check conditions for valid segments using vectorized operations
longer_segment_condition = (random_points >= 2 * (1 - random_points)) | ((1 - random_points) >= 2 * random_points)

# Count the number of valid occurrences
count_valid = np.sum(longer_segment_condition)

# Calculate the probability
probability = count_valid / num_points

# Print the result with context
print(f'Estimated Probability that the longer segment is at least twice the shorter segment: {probability}')