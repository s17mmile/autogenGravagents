# filename: satellite_halfway_altitude.py
import numpy as np
from scipy.optimize import root_scalar

def kepler_equation(E, M, e):
    # Kepler's equation: E - e*sin(E) = M
    return E - e * np.sin(E) - M

def solve_kepler(M, e):
    # Solve Kepler's equation for eccentric anomaly E given mean anomaly M and eccentricity e
    # Bracket narrowed to [0, pi] since M=pi
    sol = root_scalar(kepler_equation, args=(M, e), bracket=[0, np.pi], method='brentq')
    if not sol.converged:
        raise RuntimeError('Root finding for Kepler equation did not converge')
    return sol.root

def satellite_altitude_halfway(perigee_alt_km, apogee_alt_km):
    earth_radius = 6371  # km
    r_p = earth_radius + perigee_alt_km  # Distance at perigee from Earth's center
    r_a = earth_radius + apogee_alt_km   # Distance at apogee from Earth's center

    # Semi-major axis of the elliptical orbit
    a = (r_p + r_a) / 2
    # Orbital eccentricity
    e = (r_a - r_p) / (r_a + r_p)

    # Mean anomaly halfway through orbit (half orbital period)
    M_half = np.pi

    # Solve Kepler's equation for eccentric anomaly E at M = pi
    E_half = solve_kepler(M_half, e)

    # Calculate true anomaly f from eccentric anomaly E
    cos_f = (np.cos(E_half) - e) / (1 - e * np.cos(E_half))
    sin_f = (np.sqrt(1 - e**2) * np.sin(E_half)) / (1 - e * np.cos(E_half))
    f = np.arctan2(sin_f, cos_f)

    # Radius at true anomaly f
    r = a * (1 - e**2) / (1 + e * np.cos(f))

    # Altitude above Earth's surface
    altitude = r - earth_radius
    return altitude

# Given perigee and apogee altitudes in km
perigee_altitude = 300
apogee_altitude = 3500

altitude_halfway = satellite_altitude_halfway(perigee_altitude, apogee_altitude)

# Save result to a text file
with open('satellite_halfway_altitude.txt', 'w') as f:
    f.write(f'Altitude halfway from perigee to apogee: {altitude_halfway:.2f} km\n')

# Also print the result
print(f'Altitude halfway from perigee to apogee: {altitude_halfway:.2f} km')

# Note: Requires numpy and scipy libraries to run.