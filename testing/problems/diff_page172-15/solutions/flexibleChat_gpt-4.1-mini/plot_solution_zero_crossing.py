# filename: plot_solution_zero_crossing.py
import numpy as np
import matplotlib.pyplot as plt

# Define the solution function
def y(t):
    return (1 - 2.5 * t) * np.exp(-1.5 * t)

# Generate t values from 0 to 1 (to include zero crossing at 0.4)
t_values = np.linspace(0, 1, 400)

# Compute y values
y_values = y(t_values)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(t_values, y_values, label='y(t) = (1 - 2.5 t) e^{-3t/2}')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # zero line
plt.axvline(0.4, color='red', linestyle='--', label='Zero crossing at t=0.4')
plt.title('Solution of the ODE and Zero Crossing')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('solution_plot.png')
# Note: plt.show() is omitted as per instructions
