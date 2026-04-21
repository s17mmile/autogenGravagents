# filename: refined_rocket_max_height.py
import numpy as np
from scipy.integrate import solve_ivp

# Constants
g0 = 9.81  # m/s^2, gravity at sea level
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
rho0 = 1.225  # kg/m^3, sea level air density
scale_height = 8500  # m, atmospheric scale height

# Initial conditions
initial_height = 0.0  # m
initial_velocity = 0.0  # m/s

# Mass flow rate (constant)
mass_flow_rate = fuel_mass / burn_time  # kg/s

# Thrust profile: linearly decreasing thrust during burn (more realistic)
def thrust_at_time(t):
    if t <= burn_time:
        return exhaust_velocity * mass_flow_rate * (1 - t / burn_time)
    else:
        return 0.0

# Air density as function of altitude
def air_density(h):
    return rho0 * np.exp(-h / scale_height) if h >= 0 else rho0

# Gravity as function of altitude
def gravity(h):
    return g0 * (R_earth / (R_earth + h))**2

# Drag force (quadratic, opposes velocity)
def drag_force(v, h):
    rho = air_density(h)
    return 0.5 * c_w * rho * area * v * abs(v)

# Equations of motion
def rocket_ode(t, y):
    h, v = y
    if t <= burn_time:
        m = total_mass - mass_flow_rate * t
    else:
        m = total_mass - fuel_mass

    thrust_force = thrust_at_time(t)
    g = gravity(h)
    drag = drag_force(v, h)
    gravity_force = m * g

    net_force = thrust_force - gravity_force - drag
    a = net_force / m

    return [v, a]

# Event to stop integration when velocity crosses zero downward (max height)
def event_max_height(t, y):
    return y[1]

event_max_height.terminal = True
event_max_height.direction = -1

# Initial state
y0 = [initial_height, initial_velocity]

# Integration time span
t_span = (0, 10000)

# Solve ODE
sol = solve_ivp(rocket_ode, t_span, y0, events=event_max_height, max_step=1.0, rtol=1e-8, atol=1e-8)

max_height = sol.y[0][-1]

print(f"Maximum height reached with refined model: {max_height / 1000:.2f} km")
