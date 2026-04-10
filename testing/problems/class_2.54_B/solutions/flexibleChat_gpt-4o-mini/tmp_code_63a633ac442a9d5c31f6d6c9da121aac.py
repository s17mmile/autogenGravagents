import numpy as np
import matplotlib.pyplot as plt

# Constants
m = 0.5  # mass of the potato in kg
k = 0.01  # air resistance coefficient in s^-1
g = 9.81  # acceleration due to gravity in m/s^2
v0 = 120  # initial velocity in m/s

# Time parameters
dt = 0.01  # time step in seconds
T = 30  # total time in seconds
n = int(T / dt)  # number of time steps

t = np.linspace(0, T, n)

# Arrays to store velocity and height
v = np.zeros(n)
v[0] = v0  # initial velocity
h = np.zeros(n)  # initial height

# Numerical integration using Euler method
for i in range(1, n):
    # Calculate the net force
    F_net = m * g - k * v[i-1] * np.abs(v[i-1])
    # Update velocity
    v[i] = v[i-1] + (F_net / m) * dt
    # Update height
    h[i] = h[i-1] + v[i-1] * dt
    # Stop if the potato reaches the ground
    if h[i] < 0:
        h[i] = 0
        break

# Maximum height
max_height = np.max(h)
print(f'Maximum height reached by the potato: {max_height:.2f} meters')

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(t, h, label='Height (m)')
plt.title('Height of Potato Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.axhline(0, color='red', linestyle='--')
plt.legend()
plt.grid()
plt.show()