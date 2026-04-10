import numpy as np
import matplotlib.pyplot as plt

# Constants
c_w = 0.5  # Drag coefficient
r = 0.05  # Radius in meters
m = 0.2  # Mass in kg
rho = 1.225  # Air density in kg/m^3
v = 35.2  # Initial speed in m/s
fence_height = 3.0  # Height of the fence in meters
fence_distance = 50.0  # Distance to the fence in meters

# Cross-sectional area
A = np.pi * r**2

def drag_force(v):
    """Calculate the drag force based on velocity."""
    return 0.5 * c_w * rho * A * v**2

# Equations of motion

def equations_of_motion(launch_angle, dt, total_time):
    """Simulate the motion of the ball under the influence of gravity and drag."""
    g = 9.81  # Acceleration due to gravity
    t = np.arange(0, total_time, dt)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    vx = v * np.cos(launch_angle)
    vy = v * np.sin(launch_angle)

    for i in range(1, len(t)):
        F_d = drag_force(np.sqrt(vx**2 + vy**2))
        ax = -F_d * vx / m
        ay = -g - (F_d * vy / m)
        vx += ax * dt
        vy += ay * dt
        if vy < 0 and y[i-1] <= 0:
            break  # Stop if the ball hits the ground
        x[i] = x[i-1] + vx * dt
        y[i] = y[i-1] + vy * dt

    return x, y

# Find optimal angle
best_angle = 0
max_height = 0
angle_range = np.linspace(0, np.pi/2, 90)

for launch_angle in angle_range:
    x, y = equations_of_motion(launch_angle, 0.01, 10)
    if np.any(y >= fence_height):
        height_at_fence = y[np.where(x >= fence_distance)[0][0]]  # Height at the fence
        if height_at_fence > max_height:
            max_height = height_at_fence
            best_angle = launch_angle

# Convert best angle to degrees
best_angle_degrees = np.degrees(best_angle)
print(f'Optimal launch angle to clear the fence: {best_angle_degrees:.2f} degrees')
print(f'Maximum height achieved at this angle: {max_height:.2f} meters')