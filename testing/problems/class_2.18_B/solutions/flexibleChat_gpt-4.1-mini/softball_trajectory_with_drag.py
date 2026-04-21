# filename: softball_trajectory_with_drag.py
import numpy as np
from scipy.integrate import solve_ivp

def simulate_trajectory(theta_deg, v0, c_w, r, m, fence_x, fence_height, rho=1.225, g=9.81):
    """Simulate the trajectory of a softball with quadratic air resistance.
    Returns the height at the fence and the solution object."""
    theta = np.radians(theta_deg)
    A = np.pi * r**2

    # Initial velocity components
    v_x0 = v0 * np.cos(theta)
    v_y0 = v0 * np.sin(theta)

    def derivatives(t, y):
        x, y_pos, v_x, v_y = y
        v = np.sqrt(v_x**2 + v_y**2)
        # Drag force magnitude
        F_d = 0.5 * c_w * rho * A * v**2
        # Acceleration components
        a_x = -F_d / m * (v_x / v) if v != 0 else 0
        a_y = -g - (F_d / m) * (v_y / v) if v != 0 else -g
        return [v_x, v_y, a_x, a_y]

    # Event to stop integration when ball hits the ground (y=0)
    def hit_ground(t, y):
        return y[1]
    hit_ground.terminal = True
    hit_ground.direction = -1

    # Initial state vector
    y0 = [0, 0, v_x0, v_y0]

    # Integrate until ball hits ground or max time
    sol = solve_ivp(derivatives, [0, 10], y0, events=hit_ground, max_step=0.01)

    x_vals = sol.y[0]
    y_vals = sol.y[1]

    # Interpolate height at fence_x if within range
    if fence_x > x_vals[-1]:
        # Ball does not reach fence
        return None, None

    # Find index where x just passes fence_x
    idx = np.searchsorted(x_vals, fence_x)
    if idx == 0 or idx >= len(x_vals):
        return None, None

    # Linear interpolation for height at fence_x
    x1, x2 = x_vals[idx-1], x_vals[idx]
    y1, y2 = y_vals[idx-1], y_vals[idx]
    y_fence = y1 + (fence_x - x1) * (y2 - y1) / (x2 - x1)

    return y_fence, sol

def find_min_angle_to_clear_fence(v0, c_w, r, m, fence_x, fence_height):
    """Find the minimum elevation angle in degrees to clear the fence."""
    angles = np.linspace(1, 89, 89)  # degrees
    clearance_angles = []

    for angle in angles:
        y_fence, sol = simulate_trajectory(angle, v0, c_w, r, m, fence_x, fence_height)
        if y_fence is not None and y_fence >= fence_height:
            clearance_angles.append(angle)

    if not clearance_angles:
        return None  # No angle found to clear fence

    return min(clearance_angles)

# Constants
c_w = 0.5  # drag coefficient
r = 0.05  # radius in meters
m = 0.2  # mass in kg
v0 = 35.2  # initial speed in m/s
fence_x = 30  # horizontal distance to fence in meters (example)
fence_height = 3  # fence height in meters (example)

# Find the minimum angle to clear the fence
min_angle = find_min_angle_to_clear_fence(v0, c_w, r, m, fence_x, fence_height)

# Save results to a file
with open('softball_clearance_result.txt', 'w') as f:
    if min_angle is not None:
        f.write(f'Minimum elevation angle to clear the fence: {min_angle:.2f} degrees\n')
    else:
        f.write('No angle found to clear the fence with given parameters.\n')
