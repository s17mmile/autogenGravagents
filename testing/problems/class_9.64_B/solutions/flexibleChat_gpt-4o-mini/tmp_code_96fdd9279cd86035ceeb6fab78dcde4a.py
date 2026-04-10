import numpy as np
import matplotlib.pyplot as plt

# Constants
v_e = 4000  # exhaust velocity in m/s
m_total = 1e5  # total mass of the rocket in kg
fuel_mass = 0.9 * m_total  # mass of the fuel in kg
burn_time = 100  # burn time in seconds
g = 9.81  # acceleration due to gravity in m/s^2
c_w = 0.2  # drag coefficient
rho = 1.225  # air density in kg/m^3
r = 0.2  # radius of the rocket in meters
A = np.pi * r**2  # cross-sectional area in m^2

dt = 1.0  # increased time step for simulation
max_simulation_time = 60  # limit simulation duration to 60 seconds

# Initialize variables
h = 0  # initial height
v = 0  # initial velocity
m_fuel = fuel_mass  # remaining fuel mass
m_current = m_total  # current mass of the rocket
thrust = 0

# Lists to store height, velocity, and acceleration for plotting
heights = []
velocities = []
accelerations = []

# Simulation loop
for t in np.arange(0, max_simulation_time, dt):
    if m_fuel > 0:
        # Calculate mass flow rate and thrust
        m_dot = fuel_mass / burn_time
        thrust = m_dot * v_e
        m_fuel -= m_dot * dt
        m_current = m_total - (fuel_mass - m_fuel)  # update current mass
    else:
        thrust = 0  # no thrust after fuel is burnt
        m_current = m_total  # mass remains constant after fuel is burnt
        break  # exit loop if fuel is exhausted

    # Calculate drag force
    F_d = 0.5 * c_w * rho * A * v**2

    # Calculate net force
    F_net = thrust - (m_current * g + F_d)

    # Update acceleration, velocity, and height
    a = F_net / m_current
    v += a * dt
    h += v * dt

    # Store height, velocity, and acceleration
    heights.append(h)
    velocities.append(v)
    accelerations.append(a)

# Final height after fuel burns out
print(f'Maximum height reached: {h:.2f} m')

# Plotting the results
plt.figure(figsize=(15, 10))
plt.subplot(3, 1, 1)
plt.plot(np.arange(0, len(heights), 1), heights, label='Height (m)', color='blue')
plt.title('Rocket Ascent Simulation')
plt.ylabel('Height (m)')
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(np.arange(0, len(velocities), 1), velocities, label='Velocity (m/s)', color='orange')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(np.arange(0, len(accelerations), 1), accelerations, label='Acceleration (m/s^2)', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()