# filename: rocket_max_altitude_simulation_improved.py
import numpy as np

def rocket_max_altitude(v_e=4000, m0=1e5, fuel_fraction=0.9, burn_time=100, g=9.81):
    """Simulate vertical rocket flight to compute maximum altitude."""
    # Initial masses
    m_fuel = fuel_fraction * m0
    m_dry = m0 - m_fuel

    # Mass flow rate (kg/s)
    mdot = m_fuel / burn_time

    # Time discretization for burn phase
    dt = 0.1  # time step in seconds
    times = np.arange(0, burn_time + dt, dt)

    # Initialize arrays
    mass = np.zeros_like(times)
    velocity = np.zeros_like(times)
    altitude = np.zeros_like(times)

    # Initial conditions
    mass[0] = m0
    velocity[0] = 0
    altitude[0] = 0

    # Burn phase simulation
    for i in range(1, len(times)):
        mass[i] = max(mass[i-1] - mdot * dt, m_dry)  # prevent mass below dry mass
        a_thrust = v_e * mdot / mass[i]
        a_net = a_thrust - g
        velocity[i] = velocity[i-1] + a_net * dt
        # Use average velocity between steps for altitude update
        altitude[i] = altitude[i-1] + 0.5 * (velocity[i-1] + velocity[i]) * dt

    # At burnout
    v_burnout = velocity[-1]
    h_burnout = altitude[-1]

    # Coasting phase: rocket moves upward until velocity = 0
    h_coast = (v_burnout ** 2) / (2 * g)

    # Total max altitude
    h_max = h_burnout + h_coast

    return h_max

if __name__ == '__main__':
    max_altitude = rocket_max_altitude()
    print(f'Maximum altitude reached by the rocket: {max_altitude / 1000:.2f} km')
