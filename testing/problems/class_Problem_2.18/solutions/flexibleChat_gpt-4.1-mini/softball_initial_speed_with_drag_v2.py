# filename: softball_initial_speed_with_drag_v2.py
import numpy as np
from scipy.integrate import solve_ivp

def projectile_motion_with_drag(t, y, params):
    x, y_pos, vx, vy = y
    m, c_w, rho, A, g = params
    v = np.sqrt(vx**2 + vy**2)
    if v == 0:
        drag_x = 0
        drag_y = 0
    else:
        drag_force = 0.5 * c_w * rho * A * v**2
        drag_x = drag_force * (vx / v)
        drag_y = drag_force * (vy / v)
    dvx_dt = -drag_x / m
    dvy_dt = -g - drag_y / m
    return [vx, vy, dvx_dt, dvy_dt]

def hit_ground(t, y):
    # Event function to stop integration when projectile hits the ground (y=0)
    return y[1]
hit_ground.terminal = True
hit_ground.direction = -1

def pass_fence(t, y, fence_distance):
    # Event function to stop integration when projectile passes the fence distance
    return y[0] - fence_distance
pass_fence.terminal = True
pass_fence.direction = 1

def clears_fence(initial_speed, angle_deg, fence_distance, fence_height, params):
    angle_rad = np.radians(angle_deg)
    vx0 = initial_speed * np.cos(angle_rad)
    vy0 = initial_speed * np.sin(angle_rad)
    y0 = [0, 0, vx0, vy0]

    sol = solve_ivp(
        projectile_motion_with_drag,
        [0, 10],
        y0,
        args=(params,),
        events=[hit_ground, lambda t, y: pass_fence(t, y, fence_distance)],
        dense_output=True,
        max_step=0.01
    )

    # Check if fence was passed
    if sol.t_events[1].size > 0:
        t_fence = sol.t_events[1][0]
        y_fence = sol.sol(t_fence)[1]  # y position at fence
        return y_fence >= fence_height
    else:
        # Fence not reached before landing
        return False

def find_min_initial_speed(angle_deg, fence_distance, fence_height, params, speed_min=1, speed_max=50, tol=0.01):
    # Binary search for minimum initial speed
    low = speed_min
    high = speed_max
    while high - low > tol:
        mid = (low + high) / 2
        if clears_fence(mid, angle_deg, fence_distance, fence_height, params):
            high = mid
        else:
            low = mid
    return (low + high) / 2

# Given parameters
c_w = 0.5  # drag coefficient
r = 0.05  # radius in meters
m = 0.2  # mass in kg
rho = 1.225  # air density kg/m^3
A = np.pi * r**2  # cross-sectional area

g = 9.81  # gravity m/s^2

params = (m, c_w, rho, A, g)

# Fence parameters (assumed)
fence_height = 3.0  # meters
fence_distance = 30.0  # meters

# Assume launch angle from previous problem or typical value
launch_angle_deg = 45  # degrees

min_speed = find_min_initial_speed(launch_angle_deg, fence_distance, fence_height, params)

print(f"Minimum initial speed to clear the fence at {fence_distance} m and {fence_height} m height with air resistance: {min_speed:.2f} m/s")
