import numpy as np
import matplotlib.pyplot as plt

# Constants
v_e = 4000  # Exhaust velocity in m/s
m_0 = 10**5  # Total mass of the rocket in kg
m_f = 0.9 * m_0  # Mass of fuel in kg
burn_time = 100  # Burn time in seconds
r = 0.2  # Radius of the rocket in meters

# Derived constants
mass_flow_rate = m_f / burn_time  # Mass flow rate in kg/s

g = 9.81  # Acceleration due to gravity in m/s^2

# Simulation parameters
max_time = 15  # Further reduced maximum simulation time in seconds
time_step = 3  # Increased time step for better performance
num_steps = max_time // time_step  # Number of steps

# Initialize variables
height = 0  # Initial height in meters
velocity = 0  # Initial velocity in m/s
mass = m_0  # Initial mass in kg
thrust = mass_flow_rate * v_e  # Constant thrust during burn

# Lists to store time and height for plotting
time_list = np.zeros(num_steps)
height_list = np.zeros(num_steps)

# Simulation loop
for t in range(num_steps):
    current_time = t * time_step
    if current_time < burn_time:
        # Calculate net force without drag
        net_force = thrust - mass * g
    else:
        net_force = -mass * g  # Only gravity after burn

    # Update velocity and height
    acceleration = net_force / mass
    velocity += acceleration * time_step
    height += velocity * time_step

    # Update mass if fuel is burning
    if current_time < burn_time:
        mass -= mass_flow_rate * time_step
        mass = max(mass, 0)  # Ensure mass does not go below zero

    # Store values for plotting
    time_list[t] = current_time
    height_list[t] = height

# Plotting results
plt.figure(figsize=(12, 6))
plt.plot(time_list, height_list, label='Height (m)')
plt.title('Rocket Height vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Final height reached
max_height_km = height / 1000
max_time_reached = current_time
print(f'Maximum height reached: {max_height_km:.2f} km at time {max_time_reached:.2f} s')