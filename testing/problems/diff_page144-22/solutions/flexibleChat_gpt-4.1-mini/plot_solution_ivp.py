# filename: plot_solution_ivp.py
import numpy as np
import matplotlib.pyplot as plt

# Define the solution y(t) for given beta
def y(t, beta):
    C1 = 1 + beta
    C2 = 1 - beta
    return C1 * np.exp(t / 2) + C2 * np.exp(-t / 2)

# Time values from 0 to 10
t = np.linspace(0, 10, 400)

# Beta values to plot
beta_values = [-1, 0]

plt.figure(figsize=(8, 5))
for beta in beta_values:
    y_vals = y(t, beta)
    plt.plot(t, y_vals, label=f'beta = {beta}')

plt.title('Solution y(t) of 4 y" - y = 0 with y(0)=2 and y\'(0)=beta')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.legend()
plt.grid(True)
plt.savefig('solution_plot.png')
