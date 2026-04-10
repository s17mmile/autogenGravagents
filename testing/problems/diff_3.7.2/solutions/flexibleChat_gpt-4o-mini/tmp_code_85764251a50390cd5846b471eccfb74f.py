import numpy as np
from scipy.integrate import odeint

# Constants
weight = 10  # lb
stretch = 2 / 12  # ft (2 inches to feet)
additional_displacement = 2 / 12  # ft (2 inches to feet)
initial_velocity = 1  # ft/s

g = 32.2  # ft/s^2 (acceleration due to gravity)

# Step 1: Calculate mass in slugs
mass = weight / g  # slugs

# Step 2: Calculate spring constant k using Hooke's Law
k = weight / stretch  # lb/ft

# Step 3: Define the differential equation for SHM
def model(y, t):
    position, velocity = y
    dydt = [velocity, -k/mass * position]
    return dydt

# Step 4: Initial conditions
initial_position = stretch + additional_displacement  # total displacement from equilibrium
initial_conditions = [initial_position, initial_velocity]

t = np.linspace(0, 10, 100)  # time array from 0 to 10 seconds

# Step 5: Solve the differential equation
solution = odeint(model, initial_conditions, t)
position = solution[:, 0]

# Step 6: Calculate amplitude
C1 = initial_position  # initial position
C2 = initial_velocity / (k/mass)  # corrected initial velocity for C2
amplitude = np.sqrt(C1**2 + C2**2)

# Output the amplitude
print(f'Amplitude: {amplitude}')