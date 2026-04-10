import numpy as np
import matplotlib.pyplot as plt

# Constants
m = 0.2  # Mass of the softball in kg
h = 1.0  # Further reduced height of the fence in meters

# Function to simulate trajectory without air resistance

def simulate_trajectory(v0, angle, h):
    """
    Simulates the trajectory of a softball ignoring air resistance.
    Parameters:
        v0 (float): Initial speed of the ball in m/s.
        angle (float): Launch angle in radians.
        h (float): Height of the fence in meters.
    Returns:
        bool: True if the ball clears the fence, False otherwise.
    """
    g = 9.81  # Acceleration due to gravity in m/s^2
    dt = 0.1  # Increased time step for efficiency
    t = 0  # Time
    x, y = 0, 0  # Initial position
    vx = v0 * np.cos(angle)  # Initial horizontal velocity
    vy = v0 * np.sin(angle)  # Initial vertical velocity

    while y >= 0:
        # Update position
        x += vx * dt
        y += vy * dt
        vy -= g * dt  # Update vertical velocity due to gravity
        if y >= h:
            return True  # Cleared the fence
        t += dt
    return False  # Did not clear the fence

# Find initial speed needed to clear the fence using binary search
low, high = 0, 50  # Speed range for binary search
angle = np.pi / 4  # 45 degrees for maximum range
initial_speed = 0

while low <= high:
    mid = (low + high) / 2
    cleared = simulate_trajectory(mid, angle, h)
    if cleared:
        initial_speed = mid  # Update initial speed
        high = mid - 0.01  # Search for a smaller speed
    else:
        low = mid + 0.01  # Increase speed

print(f'Initial speed needed to clear the fence: {initial_speed:.2f} m/s')