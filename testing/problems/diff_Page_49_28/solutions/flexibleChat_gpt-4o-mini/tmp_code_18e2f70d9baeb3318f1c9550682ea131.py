import numpy as np
import matplotlib.pyplot as plt

# Define the function based on the rearranged equation
def equation(T, target_value=3.99):
    return target_value * ((1 + T)**4 + 4 * np.exp(4 * T)) - 4 * np.exp(4 * T)

# Generate T values
T_values = np.linspace(0, 5, 100)
# Calculate corresponding y values
y_values = equation(T_values)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(T_values, y_values, label='Equation Value')
plt.axhline(0, color='red', linestyle='--', label='y=0')
plt.title('Plot of the Equation')
plt.xlabel('T')
plt.ylabel('Equation Value')
plt.legend()
plt.grid()
plt.show()