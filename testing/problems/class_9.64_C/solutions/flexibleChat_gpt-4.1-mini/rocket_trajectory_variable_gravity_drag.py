# filename: rocket_trajectory_variable_gravity_drag.py
import numpy as np
from scipy.integrate import solve_ivp

# Constants
R_earth = 6371000  # Earth radius in meters
g0 = 9.81  # gravity at surface in m/s^2
m0 = 1e5  # initial mass in kg
fuel_fraction = 0.9
mfuel = m0 * fuel_fraction  # fuel mass
m_empty = m0 - mfuel  # dry mass
burn_time = 100  # burn time in seconds
v_e = 4000  # exhaust velocity in m/s
c_w = 0.2  # drag coefficient
radius = 0.2  # radius in meters
A = np.pi * radius**2  # cross-sectional area
rho_air = 1.225  # air density at sea level in kg/m^3 (assumed constant)

# Thrust magnitude (constant burn rate)
mass_flow_rate = mfuel / burn_time
thrust = v_e * mass_flow_rate  # thrust = exhaust velocity * mass flow rate

# Gravity as function of altitude
def gravity(h):
    return g0 * (R_earth / (R_earth + h))**2

# Drag force magnitude
def drag(v):
    return 0.5 * c_w * rho_air * A * v**2

# Equations of motion
# state vector y = [height, velocity, mass]
def rocket_ode(t, y):
    h, v, m = y
    if m > m_empty:
        # During burn
        thrust_force = thrust
        dm_dt = -mass_flow_rate
    else:
        # After burn
        thrust_force = 0
        dm_dt = 0

    g = gravity(h)
    v_abs = abs(v)
    drag_force = drag(v_abs)
    drag_force = drag_force if v > 0 else -drag_force  # drag opposes velocity

    dv_dt = (thrust_force - drag_force - m * g) / m
    dh_dt = v

    return [dh_dt, dv_dt, dm_dt]

# Initial conditions
h0 = 0
v0 = 0
m_init = m0
y0 = [h0, v0, m_init]

# Time span for integration
# We integrate long enough to reach max height (e.g. 2000 s)
t_span = (0, 2000)

# Event to stop integration when velocity becomes zero (max height)
def event_max_height(t, y):
    return y[1]  # velocity

event_max_height.terminal = True
event_max_height.direction = -1  # detect zero crossing from positive to negative velocity

# Solve ODE
sol = solve_ivp(rocket_ode, t_span, y0, events=event_max_height, max_step=1, rtol=1e-8, atol=1e-10)

max_height_m = sol.y[0, -1]
max_height_km = max_height_m / 1000

# Save results to a file
with open('rocket_max_height.txt', 'w') as f:
    f.write(f'Maximum height reached with variable gravity and drag: {max_height_km:.2f} km\n')

# Also save full trajectory data
import csv
with open('rocket_trajectory.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['time_s', 'height_m', 'velocity_m_s', 'mass_kg'])
    for i in range(len(sol.t)):
        writer.writerow([sol.t[i], sol.y[0,i], sol.y[1,i], sol.y[2,i]])
