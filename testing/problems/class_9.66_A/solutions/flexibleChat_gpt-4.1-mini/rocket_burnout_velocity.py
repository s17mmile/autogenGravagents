# filename: rocket_burnout_velocity.py
import numpy as np
from scipy.integrate import solve_ivp

# Given parameters
I = 8.5  # total impulse in N-s (not directly used here)
m0 = 0.054  # initial total mass in kg
m_engine = 0.020  # engine mass in kg
m_propellant = 0.011  # propellant mass in kg
burn_time = 1.5  # burn time in seconds
v_e = 800  # exhaust velocity in m/s
c_w = 0.75  # drag coefficient
rho = 1.225  # air density in kg/m^3
rocket_diameter = 0.024  # rocket diameter in meters

# Cross-sectional area (m^2)
A = np.pi * (rocket_diameter / 2)**2

# Mass flow rate (negative because mass decreases) in kg/s
dm_dt = -m_propellant / burn_time

# Mass as a function of time (kg)
def mass(t):
    return m0 + dm_dt * t  # linear decrease

# Differential equation for velocity dv/dt
# v' = (thrust - drag - v * dm/dt) / m(t)
# thrust = v_e * (-dm/dt)

def dv_dt(t, v):
    m_t = mass(t)
    thrust = v_e * (-dm_dt)  # positive thrust (N)
    # Drag force always opposes motion; use sign of velocity to determine direction
    drag = 0.5 * rho * v[0]**2 * A * c_w * np.sign(v[0]) if v[0] != 0 else 0
    dvdt = (thrust - drag - v[0] * (-dm_dt)) / m_t
    return [dvdt]

# Initial velocity (m/s)
v0 = [0.0]

# Time span for integration (s)
t_span = (0, burn_time)

# Solve ODE numerically
sol = solve_ivp(dv_dt, t_span, v0, dense_output=True, max_step=0.01)

# Velocity at burnout (m/s)
v_burnout = sol.y[0, -1]

# Save result to a file
with open('rocket_burnout_speed.txt', 'w') as f:
    f.write(f'Rocket speed at burnout: {v_burnout:.2f} m/s\n')

# Save velocity vs time data for further analysis
np.savetxt('velocity_vs_time.csv', np.vstack((sol.t, sol.y[0])).T, delimiter=',', header='time,velocity', comments='')
