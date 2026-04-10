import numpy as np
import matplotlib.pyplot as plt

# Constants
m = 0.25  # mass in kg
h = 30.0  # height in meters
g = 9.81  # acceleration due to gravity in m/s^2
resistance_coefficient = 0.2  # resistance coefficient

# Potential energy at height h
potential_energy = m * g * h

# Work done against resistance as a function of velocity
# W = integral of resistance from 0 to v
# W = 0.1 * v^2
# Setting up the energy equation
# mgh = (1/2)mv^2 + 0.1v^2
# 73.575 = 0.125v^2 + 0.1v^2
# 73.575 = 0.225v^2
# v^2 = 73.575 / 0.225
final_velocity = np.sqrt(potential_energy / (0.5 * resistance_coefficient))

# Print the final velocity with units
print(f'The velocity of the mass when it hits the ground is approximately {final_velocity:.2f} m/s')

# Time simulation for velocity profile
time = np.linspace(0, 5, 100)  # simulate for 5 seconds
velocity_profile = final_velocity * (1 - np.exp(-resistance_coefficient * time / m))

# Plotting the velocity profile
plt.figure(figsize=(10, 6))
plt.plot(time, velocity_profile, label='Velocity Profile', color='blue')
plt.title('Velocity of Mass Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid()
plt.savefig('velocity_profile.png')