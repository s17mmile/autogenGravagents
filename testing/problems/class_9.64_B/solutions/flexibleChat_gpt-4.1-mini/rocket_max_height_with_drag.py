# filename: rocket_max_height_with_drag.py
import numpy as np
from scipy.integrate import solve_ivp

# Constants
g0 = 9.81  # m/s^2, acceleration due to gravity at sea level
R_earth = 6371000  # m, Earth's radius

# Rocket parameters
total_mass = 1e5  # kg
fuel_fraction = 0.9
fuel_mass = total_mass * fuel_fraction  # kg
burn_time = 100  # s
exhaust_velocity = 4000  # m/s

# Drag parameters
c_w = 0.2  # drag coefficient
radius = 0.2  # m
area = np.pi * radius**2  # cross-sectional area, m^2
rho0 = 1.225  # kg/m^3, air density at sea level
scale_height = 8500  # m, scale height for atmosphere

# Initial conditions
initial_height = 0.0  # m
initial_velocity = 0.0  # m/s

# Mass flow rate (constant)
mass_flow_rate = fuel_mass / burn_time  # kg/s

# Thrust force (constant during burn)
thrust = exhaust_velocity * mass_flow_rate  # N

# Equations of motion
# State vector y = [height, velocity]
# dy/dt = [velocity, acceleration]

def air_density(h):
    # Exponential decrease of air density with altitude
    return rho0 * np.exp(-h / scale_height) if h >= 0 else rho0

def gravity(h):
    # Gravity decreases with altitude
    return g0 * (R_earth / (R_earth + h))**2

def drag_force(v, h):
    rho = air_density(h)
    return 0.5 * c_w * rho * area * v * abs(v)  # quadratic drag

def rocket_ode(t, y):
    h, v = y
    # Determine current mass
    if t <= burn_time:
        m = total_mass - mass_flow_rate * t
        thrust_force = thrust
    else:
        m = total_mass - fuel_mass
        thrust_force = 0.0

    g = gravity(h)
    drag = drag_force(v, h)
    gravity_force = m * g

    net_force = thrust_force - gravity_force - drag
    a = net_force / m

    return [v, a]

# Event to stop integration when velocity reaches zero after burnout (max height)
def event_max_height(t, y):
    return y[1]

event_max_height.terminal = True
event_max_height.direction = -1

# Initial state
y0 = [initial_height, initial_velocity]

# Integration time span: start at 0, max 10000 s to ensure reaching max height
t_span = (0, 10000)

# Solve ODE
sol = solve_ivp(rocket_ode, t_span, y0, events=event_max_height, max_step=1.0, rtol=1e-8, atol=1e-8)

max_height = sol.y[0][-1]  # final height at max velocity zero

# Print result
print(f"Maximum height reached with air resistance: {max_height / 1000:.2f} km")
