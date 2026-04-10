import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit

# Constants
V = 100  # volume of the tank in gallons
S0 = 50  # initial amount of salt in oz
inflow_rate = 2  # inflow rate in gal/min
outflow_rate = 2  # outflow rate in gal/min

# Define the differential equation
def model(S, t):
    C_in = (1/4) * (1 + (1/2) * np.sin(t))  # salt concentration in oz/gal
    salt_inflow = C_in * inflow_rate  # oz/min
    salt_outflow = (S / V) * outflow_rate  # oz/min
    dSdt = salt_inflow - salt_outflow
    return dSdt

# Time points
t = np.linspace(0, 200, 1000)  # simulate for 200 minutes

# Solve the differential equation
S = odeint(model, S0, t)

# Define a function to fit the solution
def fit_func(t, A, B, C):
    return A + B * np.sin(t) + C * np.cos(t)

# Fit the solution to the assumed form
params, _ = curve_fit(fit_func, t, S.flatten())
A, B, C = params

# Calculate amplitude
amplitude = np.sqrt(B**2 + C**2)

# Save the plot
plt.plot(t, S, label='Salt Concentration Over Time')
plt.title('Salt Concentration in Tank Over Time')
plt.xlabel('Time (min)')
plt.ylabel('Salt Concentration (oz)')
plt.legend()
plt.grid()
plt.savefig('salt_concentration_plot.png')

# Output the amplitude
print('Amplitude of oscillation:', amplitude)