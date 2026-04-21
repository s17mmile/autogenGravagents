# filename: calculate_min_energy_geosync_orbit.py

import math

def calculate_min_energy(mass=10000, initial_altitude=200000):
    """
    Calculate the minimum energy required to move a spacecraft from a circular orbit
    at a given initial altitude to a geosynchronous orbit (24-hour period).

    Parameters:
    mass (float): Mass of the spacecraft in kilograms.
    initial_altitude (float): Initial orbit altitude above Earth's surface in meters.

    Returns:
    tuple: (delta_energy_J, initial_orbit_radius_m, synchronous_orbit_radius_m)
        delta_energy_J: Minimum energy required in joules.
        initial_orbit_radius_m: Radius of initial orbit in meters.
        synchronous_orbit_radius_m: Radius of synchronous orbit in meters.
    """
    # Constants
    G = 6.67430e-11  # gravitational constant, m^3 kg^-1 s^-2
    M = 5.972e24     # mass of Earth, kg
    R = 6.371e6      # radius of Earth, m

    # Calculate initial orbit radius
    r1 = R + initial_altitude

    # Calculate synchronous orbit radius using Kepler's third law
    T = 24 * 3600  # orbital period for synchronous orbit, seconds
    r2 = ((G * M * T**2) / (4 * math.pi**2))**(1/3)

    # Calculate total mechanical energy for initial and synchronous orbits
    E1 = -G * M * mass / (2 * r1)
    E2 = -G * M * mass / (2 * r2)

    # Minimum energy required to move from initial to synchronous orbit
    delta_E = E2 - E1

    # delta_E > 0 means energy must be added to raise orbit

    return delta_E, r1, r2

if __name__ == '__main__':
    delta_E, r1, r2 = calculate_min_energy()
    print(f'Initial orbit radius (m): {r1:.2f}')
    print(f'Synchronous orbit radius (m): {r2:.2f}')
    print(f'Minimum energy required (MJ): {delta_E / 1e6:.2f}')
