import numpy as np

# Define the function based on the rearranged equation
def equation(T, target_value=3.99):
    return target_value * ((1 + T)**4 + 4 * np.exp(4 * T)) - 4 * np.exp(4 * T)

# Define specific T values to evaluate from 5 to 10
T_values_extended = np.linspace(5, 10, 100)

# Calculate corresponding y values
results_extended = {T: equation(T) for T in T_values_extended}

# Print the results
for T, value in results_extended.items():
    print(f'T({T}) = {value}')