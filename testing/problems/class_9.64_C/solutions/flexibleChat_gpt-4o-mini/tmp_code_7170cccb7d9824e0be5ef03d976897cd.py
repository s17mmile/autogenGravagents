import numpy as np
import matplotlib.pyplot as plt

# Constants encapsulated in a dictionary
constants = {
    'mass_initial': 1e5,  # Total mass of the rocket (kg)
    'mass_fuel': 0.9 * 1e5,  # Mass of fuel (kg)
    'exhaust_velocity': 4000,  # Exhaust velocity (m/s)
    'burn_time': 100,  # Burn time (s)
    'radius': 0.2,  # Radius of the rocket (m)
    'drag_coefficient': 0.2,  # Drag coefficient
    'g0': 9.81,  # Gravitational acceleration at sea level (m/s^2)
    'R': 6371e3  # Radius of the Earth (m)
}

# Function to calculate air density as a function of height
def air_density(height):
    # Simplified model: density decreases exponentially with height
    return 1.225 * np.exp(-height / 8000)  # Approximation for air density (kg/m^3)

# Initial conditions
height = 0  # Initial height (m)
velocity = 0  # Initial velocity (m/s)
current_mass = constants['mass_initial']  # Initialize current mass

# Time step for simulation
# Increased dt to reduce the number of iterations
dt = 1.0  # Time step (s)

# Lists to store time, height, and velocity for plotting
results = {'time': [], 'height': [], 'velocity': []}

# Mass flow rate
mass_flow_rate = constants['mass_fuel'] / constants['burn_time']

# Maximum iterations to prevent timeout
max_iterations = 2000
max_height_threshold = 100000  # Adjusted maximum height threshold (m)

# Simulation loop
for iteration in range(max_iterations):
    time = iteration * dt
    if time < constants['burn_time']:
        # During fuel burn
        thrust = mass_flow_rate * constants['exhaust_velocity']
        weight = current_mass * constants['g0']
        net_force = thrust - weight

        # Update mass
        current_mass = constants['mass_initial'] - mass_flow_rate * time
    else:
        # After fuel burn
        thrust = 0
        weight = current_mass * (constants['g0'] * (constants['R'] / (constants['R'] + height))**2)
        net_force = thrust - weight

    # Calculate drag force
    drag_force = 0.5 * constants['drag_coefficient'] * air_density(height) * np.pi * constants['radius']**2 * velocity**2
    net_force -= drag_force

    # Update acceleration, velocity, and height
    acceleration = net_force / current_mass
    velocity += acceleration * dt
    height += velocity * dt

    # Store values for plotting
    results['time'].append(time)
    results['height'].append(height)
    results['velocity'].append(velocity)

    # Enhanced stopping condition: check if velocity is negative and height exceeds threshold
    if velocity < 0 and height > max_height_threshold:
        break

# Maximum height reached
max_height = height
print(f'The maximum height reached by the rocket is: {max_height / 1000:.2f} km')
print(f'Simulation parameters: Time step = {dt} s, Total simulation time = {len(results[