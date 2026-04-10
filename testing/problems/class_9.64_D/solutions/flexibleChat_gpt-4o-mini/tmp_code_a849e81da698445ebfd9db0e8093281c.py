import numpy as np
import matplotlib.pyplot as plt

# Constants
v_e = 4000  # Exhaust velocity in m/s
m_0 = 10**5  # Total mass of the rocket in kg
m_f = 0.9 * m_0  # Mass of fuel in kg
burn_time = 100  # Burn time in seconds
r = 0.2  # Radius of the rocket in meters
c_w = 0.2  # Drag coefficient

# Derived constants
A = np.pi * r**2  # Cross-sectional area in m^2
mass_flow_rate = m_f / burn_time  # Mass flow rate in kg/s

g = 9.81  # Acceleration due to gravity in m/s^2

# Simulation parameters
max_time = 30  # Further reduced maximum simulation time in seconds
time_step = 2  # Increased time step for better performance
num_steps = max_time // time_step  # Number of steps

# Initialize variables
height = 0  # Initial height in meters
velocity = 0  # Initial velocity in m/s
mass = m_0  # Initial mass in kg

# Lists to store time, height, and velocity for plotting
time_list = np.zeros(num_steps)
height_list = np.zeros(num_steps)
velocity_list = np.zeros(num_steps)

# Simulation loop
for t in range(num_steps):
    current_time = t * time_step
    if current_time < burn_time:
        thrust = mass_flow_rate * v_e  # Thrust during burn
    else:
        thrust = 0  # No thrust after burn

    # Simplified air density calculation (assume constant for short duration)
    rho = 1.225  # Approximate air density at sea level in kg/m^3
    drag_force = 0.5 * c_w * rho * A * velocity**2  # Drag force
    drag_force = max(drag_force, 0)  # Ensure drag is not negative

    # Calculate net force
    net_force = thrust - drag_force - mass * g

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
    velocity_list[t] = velocity

# Plotting results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_list, height_list, label='Height (m)')
plt.title('Rocket Height vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_list, velocity_list, label='Velocity (m/s)', color='orange')
plt.title('Rocket Velocity vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# Final height reached
max_height_km = height / 1000
max_time_reached = current_time
print(f'Maximum height reached: {max_height_km:.2f} km at time {max_time_reached:.2f} s')