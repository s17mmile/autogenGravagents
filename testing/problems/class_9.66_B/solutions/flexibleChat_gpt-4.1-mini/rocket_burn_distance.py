# filename: rocket_burn_distance.py
import numpy as np
from scipy.integrate import solve_ivp

# Given constants
m0 = 0.054  # initial total mass (kg)
m_engine = 0.020  # engine mass (kg)
m_propellant = 0.011  # propellant mass (kg)
burn_time = 1.5  # burn time (s)
diameter = 0.024  # rocket diameter (m)
rho = 1.225  # air density (kg/m^3)
cw = 0.75  # drag coefficient
v_e = 800  # exhaust velocity (m/s)

# Mass flow rate (kg/s)
mdot = m_propellant / burn_time

# Cross-sectional area (m^2)
A = np.pi * (diameter / 2) ** 2

# Thrust (N) - assumed constant during burn
thrust = v_e * mdot

# Define the ODE system: state vector y = [velocity, position]
# dy/dt = [acceleration, velocity]
# acceleration = (thrust - drag) / mass - (v / m) * dm/dt
# Note: dm/dt = -mdot during burn

def rocket_ode(t, y):
    v, s = y
    m = m0 - mdot * t  # current mass
    drag = 0.5 * rho * v**2 * cw * A
    dvdt = (thrust - drag) / m - (v / m) * (-mdot)  # dm/dt = -mdot
    dsdt = v
    return [dvdt, dsdt]

# Initial conditions: velocity=0, position=0
y0 = [0, 0]

# Time span for integration
t_span = (0, burn_time)

# Solve ODE
sol = solve_ivp(rocket_ode, t_span, y0, max_step=0.01, rtol=1e-8, atol=1e-10)

# Extract final position and velocity at burnout
final_velocity = sol.y[0, -1]
final_position = sol.y[1, -1]

# Print results
print(f"Velocity at burnout: {final_velocity:.2f} m/s")
print(f"Distance traveled at burnout: {final_position:.2f} m")

# Save results to a file
with open("rocket_burn_results.txt", "w") as f:
    f.write(f"Velocity at burnout: {final_velocity:.2f} m/s\n")
    f.write(f"Distance traveled at burnout: {final_position:.2f} m\n")

# The final velocity should be close to the given 131 m/s, and the distance is the answer requested.