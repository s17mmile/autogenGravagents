import math

# Constants
INITIAL_CONDITION_CONSTANT = 0.5 / 3.5  # K value from initial condition
VALUE_TO_REACH = 3.98
FACTOR = 7
DENOMINATOR = 0.02

# Calculate the value of T using the derived formula
# e^(2/3 * T^2) = (VALUE_TO_REACH * FACTOR) / DENOMINATOR
exponential_value = (VALUE_TO_REACH * FACTOR) / DENOMINATOR

# Ensure the exponential value is positive before taking the logarithm
if exponential_value <= 0:
    raise ValueError('The calculated exponential value must be positive.')

# Calculate T^2
T_squared = (3 / 2) * math.log(exponential_value)

# Calculate T
T = math.sqrt(T_squared)

# Print the result with context
print(f'The time T at which the solution first reaches the value 3.98 is approximately: {T:.2f} time units.')