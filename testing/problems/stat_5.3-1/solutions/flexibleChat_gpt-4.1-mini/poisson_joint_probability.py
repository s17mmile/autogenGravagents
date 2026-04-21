# filename: poisson_joint_probability.py
import math

# Function to calculate joint Poisson probability
# for independent Poisson variables X1 and X2
# with means lambda_1 and lambda_2,
# and observed values k1 and k2 respectively.
def joint_poisson_probability(lambda_1, k1, lambda_2, k2):
    P_X1_k1 = math.exp(-lambda_1) * (lambda_1 ** k1) / math.factorial(k1)
    P_X2_k2 = math.exp(-lambda_2) * (lambda_2 ** k2) / math.factorial(k2)
    return P_X1_k1 * P_X2_k2

# Given parameters
lambda_1 = 2
lambda_2 = 3
k1 = 3
k2 = 5

# Calculate joint probability
P_joint = joint_poisson_probability(lambda_1, k1, lambda_2, k2)

# Exact formula value for verification
exact_value = (27/10) * math.exp(-5)

# Print results
print(f"P(X1=3) * P(X2=5) = {P_joint:.6f}")
print(f"Exact formula value: {exact_value:.6f}")

# Assert to verify both values are equal within a tolerance
assert abs(P_joint - exact_value) < 1e-12, "Computed value does not match exact formula!"