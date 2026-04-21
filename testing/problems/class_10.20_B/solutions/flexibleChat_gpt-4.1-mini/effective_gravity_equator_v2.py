# filename: effective_gravity_equator_v2.py
import numpy as np

# Constants
G = 6.67430e-11  # gravitational constant, m^3 kg^-1 s^-2
M = 5.972e24     # mass of Earth, kg
r_equator = 6378e3  # equatorial radius in meters
r_polar = 6357e3    # polar radius in meters (not used directly here)

# Earth's sidereal day length in seconds (more precise)
T_sidereal = 86164  # seconds
omega = 2 * np.pi / T_sidereal  # angular velocity of Earth, rad/s

# Step 1: Gravitational acceleration at equator
g_grav = G * M / r_equator**2  # m/s^2

# Step 2: Centrifugal acceleration at equator
a_cent = omega**2 * r_equator  # m/s^2

# Step 3: Effective gravitational acceleration magnitude
g_eff_magnitude = g_grav - a_cent

# Step 4: Direction of g vector
# Coordinate system definition:
# - z-axis: along Earth's rotation axis (north pole direction)
# - x-axis: points radially outward at equator (from Earth's center to equator point)
# - y-axis: completes right-handed system
# Gravity vector points radially inward, so negative x-direction
# Centrifugal acceleration points radially outward, positive x-direction

# Effective gravity vector points inward with reduced magnitude
# Represented as vector in m/s^2

g_vector = np.array([-g_eff_magnitude, 0.0, 0.0])  # (x, y, z) components

# Output results with units
print(f"Gravitational acceleration at equator: {g_grav:.6f} m/s^2")
print(f"Centrifugal acceleration at equator: {a_cent:.6f} m/s^2")
print(f"Effective gravitational acceleration magnitude: {g_eff_magnitude:.6f} m/s^2")
print(f"Effective gravitational field vector (x, y, z) in m/s^2: {g_vector}")
