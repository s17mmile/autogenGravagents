import numpy as np
import matplotlib.pyplot as plt

# Constants
m = 0.5  # mass of the potato in kg
k = 0.01  # air resistance coefficient in s^-1
g = 9.81  # acceleration due to gravity in m/s^2
v0 = 120  # initial velocity in m/s

# Time parameters
dt = 0.05  # time step in seconds (increased)
T = 5  # total time in seconds
n = int(T / dt)  # number of time steps

t = np.linspace(0, T, n)

# Arrays to store velocity and height
v = np.zeros(n)
v[0] = v0  # initial velocity
h = np.zeros(n)  # initial height

# Numerical integration using Runge-Kutta method
for i in range(1, n):
    # Calculate the forces
    F_net = m * g - k * v[i-1] * np.abs(v[i-1])
    a = F_net / m  # acceleration
    # Runge-Kutta method
    k1_v = a * dt
    k1_h = v[i-1] * dt
    k2_v = (F_net / m) * dt
    k2_h = (v[i-1] + 0.5 * k1_v) * dt
    k3_v = (F_net / m) * dt
    k3_h = (v[i-1] + 0.5 * k2_v) * dt
    k4_v = (F_net / m) * dt
    k4_h = (v[i-1] + k3_v) * dt
    v[i] = v[i-1] + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
    h[i] = h[i-1] + (k1_h + 2*k2_h + 2*k3_h + k4_h) / 6
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