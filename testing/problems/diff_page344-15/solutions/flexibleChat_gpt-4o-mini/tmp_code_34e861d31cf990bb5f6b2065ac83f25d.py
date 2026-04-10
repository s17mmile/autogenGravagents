import math

def calculate_k(peak_value, gamma):
    amplitude_factor = 4 / math.sqrt(15)
    k = peak_value / amplitude_factor
    return k

# Constants
gamma = 1/2
peak_value = 2

# Calculate k
k = calculate_k(peak_value, gamma)

# Output result with formatting
print(f'The value of k for which the response has a peak value of 2 is: {k:.2f}')