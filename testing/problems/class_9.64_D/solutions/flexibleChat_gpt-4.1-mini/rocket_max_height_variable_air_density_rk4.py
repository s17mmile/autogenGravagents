# filename: rocket_max_height_variable_air_density_rk4.py
import numpy as np

# Constants
v_e = 4000.0  # exhaust velocity in m/s
m0 = 1e5      # initial total mass in kg
fuel_fraction = 0.9
m_fuel = fuel_fraction * m0
m_dry = m0 - m_fuel
burn_time = 100.0  # seconds

# Rocket parameters
radius = 0.2  # meters
area = np.pi * radius**2  # cross-sectional area
c_d = 0.2  # drag coefficient

# Earth parameters
g0 = 9.81  # m/s^2 at surface
R_earth = 6371e3  # Earth radius in meters

# Air density function (altitude h in meters)
def air_density(h):
    h_km = h / 1000.0
    log_rho = -0.05 * h_km + 0.11
    rho = 10**log_rho
    return max(rho, 0.0)

# Gravity as function of altitude
def gravity(h):
    return g0 * (R_earth / (R_earth + h))**2

# Thrust (constant during burn)
thrust = v_e * m_fuel / burn_time  # Newtons

# Time stepping parameters
dt = 0.1  # time step in seconds
max_time = 10000  # max simulation time in seconds

# State vector: [altitude (m), velocity (m/s), mass (kg)]
# Derivative function for RK4
# Returns derivatives: [dh/dt, dv/dt, dm/dt]
def derivatives(state, t):
    h, v, m = state
    # Mass flow rate
    mdot = m_fuel / burn_time if t <= burn_time else 0.0
    thrust_now = thrust if t <= burn_time else 0.0

    g = gravity(h)
    rho = air_density(h)
    drag = 0.5 * rho * v**2 * c_d * area * (-np.sign(v) if v != 0 else 0.0)  # drag opposes velocity

    dhdt = v
    dvdt = (thrust_now + drag - m * g) / m
    dmdt = -mdot

    return np.array([dhdt, dvdt, dmdt])

# Initial conditions
state = np.array([0.0, 0.0, m0])  # h=0 m, v=0 m/s, mass=m0

# Lists to store trajectory data
altitudes = []
velocities = []
times = []
masses = []

for step in range(int(max_time / dt)):
    t = step * dt
    altitudes.append(state[0])
    velocities.append(state[1])
    times.append(t)
    masses.append(state[2])

    # Stop if rocket starts falling back after burnout
    if state[0] < 0 and t > burn_time:
        break

    # RK4 integration
    k1 = derivatives(state, t)
    k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = derivatives(state + dt * k3, t + dt)

    state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    # Ensure mass does not drop below dry mass
    if state[2] < m_dry:
        state[2] = m_dry

# Convert altitude to km
altitudes_km = np.array(altitudes) / 1000.0
max_height_km = np.max(altitudes_km)
max_index = np.argmax(altitudes_km)
max_time = times[max_index]

# Save results to file
with open('rocket_trajectory_variable_air_density_rk4.txt', 'w') as f:
    f.write('time_s altitude_km velocity_m_s mass_kg\n')
    for ti, hi, vi, mi in zip(times, altitudes_km, velocities, masses):
        f.write(f'{ti:.2f} {hi:.4f} {vi:.2f} {mi:.2f}\n')

print(f'Maximum altitude reached with variable air density: {max_height_km:.2f} km at time {max_time:.2f} s')
