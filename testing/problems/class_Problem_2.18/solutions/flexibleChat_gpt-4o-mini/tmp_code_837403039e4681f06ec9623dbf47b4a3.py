import numpy as np
import matplotlib.pyplot as plt

# Constants
c_W = 0.5  # Drag coefficient
r = 0.05  # Radius of the softball in meters
m = 0.2  # Mass of the softball in kg
rho = 1.225  # Density of air in kg/m^3
h = 3.0  # Height of the fence in meters (adjustable)

# Cross-sectional area
A = np.pi * r**2

def simulate_trajectory(v0, angle, h):
    """
    Simulates the trajectory of a softball considering air resistance.
    Parameters:
        v0 (float): Initial speed of the ball in m/s.
        angle (float): Launch angle in radians.
        h (float): Height of the fence in meters.
    Returns:
        bool: True if the ball clears the fence, False otherwise.
        list: List of (x, y) positions of the ball during the trajectory.
    """
    g = 9.81  # Acceleration due to gravity in m/s^2
    dt = 0.01  # Time step
    t = 0  # Time
    x, y = 0, 0  # Initial position
    vx = v0 * np.cos(angle)  # Initial horizontal velocity
    vy = v0 * np.sin(angle)  # Initial vertical velocity
    trajectory = []

    while y >= 0:
        # Calculate drag force
        v = np.sqrt(vx**2 + vy**2)
        F_d = 0.5 * c_W * rho * A * v**2
        # Update velocities
        ax = -F_d * (vx / v) / m
        ay = -g - (F_d * (vy / v)) / m
        vx += ax * dt
        vy += ay * dt
        # Update position
        x += vx * dt
        y += vy * dt
        trajectory.append((x, y))
        if y >= h:
            return True, trajectory  # Cleared the fence
    return False, trajectory  # Did not clear the fence

# Find initial speed needed to clear the fence
initial_speed = 0
angle = np.pi / 4  # 45 degrees for maximum range
max_speed = 100  # Maximum limit for initial speed to avoid infinite loop
while initial_speed <= max_speed:
    cleared, trajectory = simulate_trajectory(initial_speed, angle, h)
    if cleared:
        break
    initial_speed += 0.1  # Increment speed

print(f'Initial speed needed to clear the fence: {initial_speed:.2f} m/s')

# Plot trajectory
plt.plot(*zip(*trajectory), label=f'Speed: {initial_speed:.1f} m/s')
plt.axhline(y=h, color='r', linestyle='--', label='Fence Height')
plt.xlabel('Distance (m)')
plt.ylabel('Height (m)')
plt.title('Trajectory of the Softball')
plt.legend()
plt.grid()
plt.show()