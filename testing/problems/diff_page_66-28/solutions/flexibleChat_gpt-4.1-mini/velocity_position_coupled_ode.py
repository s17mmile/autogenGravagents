# filename: velocity_position_coupled_ode.py
import numpy as np
from scipy.integrate import solve_ivp

# Constants
m = 0.25  # mass in kg
b = 0.2   # resistance coefficient in kg/s
 g = 9.8   # acceleration due to gravity in m/s^2
h = 30    # height in meters

# Define the coupled ODE system
# y[0] = velocity v(t), y[1] = position s(t)
def ode_system(t, y):
    v = y[0]
    s = y[1]
    dvdt = g - (b/m)*v  # velocity ODE
    dsdt = v            # position ODE
    return [dvdt, dsdt]

# Initial conditions: v(0) = 0, s(0) = 0
y0 = [0, 0]

# We solve until position reaches h, so we use an event function

def hit_ground_event(t, y):
    return y[1] - h

hit_ground_event.terminal = True
hit_ground_event.direction = 1  # only trigger when position increasing through h

# Solve the ODE system with event detection
sol = solve_ivp(ode_system, [0, 20], y0, events=hit_ground_event, max_step=0.1, dense_output=True)

if sol.status == 1 and len(sol.t_events[0]) > 0:
    t_f = sol.t_events[0][0]  # time when position = h
    v_impact = sol.sol(t_f)[0]  # velocity at impact
else:
    t_f = None
    v_impact = None

# Output results
if t_f is not None:
    print(f"Time to hit the ground: {t_f:.4f} seconds")
    print(f"Velocity on impact: {v_impact:.4f} m/s")
    with open('velocity_impact_result.txt', 'w') as f:
        f.write(f"Time to hit the ground: {t_f:.4f} seconds\n")
        f.write(f"Velocity on impact: {v_impact:.4f} m/s\n")
else:
    print("Failed to find the time when the mass hits the ground.")
