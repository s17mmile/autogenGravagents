import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Constants
alpha = 1/6
beta = np.sqrt(23)/6
C1 = 2
C2 = 2/np.sqrt(23)

# Function to calculate |u(t)|
def calculate_u_t(t):
    """
    Calculate the value of |u(t)| based on the given formula.
    """
    return np.exp(alpha * t) * (C1 * np.cos(beta * t) + C2 * np.sin(beta * t))

# Function to find the root where |u(t)| = 10
def equation(t):
    """
    Define the equation to find the time when |u(t)| equals 10.
    """
    return np.abs(calculate_u_t(t)) - 10

# Initial guess for fsolve
initial_guess = 1.0
try:
    first_time = fsolve(equation, initial_guess)[0]
    if first_time <= 0:
        raise ValueError('The found time must be greater than zero.')
except Exception as e:
    print('Error finding the root:', e)
    first_time = None

# Print the first time at which |u(t)| = 10
if first_time is not None:
    print('The first time at which |u(t)| = 10 is:', first_time)

# Plotting |u(t)|
t_values = np.linspace(0, 10, 400)
abs_u_values = np.abs(calculate_u_t(t_values))

plt.plot(t_values, abs_u_values, label='|u(t)|')
plt.axhline(y=10, color='r', linestyle='--', label='y=10')
plt.title('Plot of |u(t)| vs Time')
plt.xlabel('Time (t)')
plt.ylabel('|u(t)|')
plt.legend()
plt.grid()
plt.savefig('u_t_plot.png')