import numpy as np

# Define the function based on the rearranged equation
def equation(T, target_value=3.99):
    return target_value * ((1 + T)**4 + 4 * np.exp(4 * T)) - 4 * np.exp(4 * T)

# Define specific T values to evaluate
T_values = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

# Calculate corresponding y values
results = {T: equation(T) for T in T_values}

# Print the results
for T, value in results.items():
    print(f'T({T}) = {value}')