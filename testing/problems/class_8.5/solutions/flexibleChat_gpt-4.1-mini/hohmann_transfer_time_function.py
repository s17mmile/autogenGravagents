# filename: hohmann_transfer_time_function.py
import math

def hohmann_transfer_time(r1_km, r2_km, mu_km3_s2):
    """
    Calculate the time for a Hohmann transfer between two circular orbits around the Sun.

    Parameters:
    r1_km (float): Orbital radius of the starting planet in kilometers.
    r2_km (float): Orbital radius of the target planet in kilometers.
    mu_km3_s2 (float): Standard gravitational parameter of the Sun in km^3/s^2.

    Returns:
    float: Transfer time in days.
    """
    a_transfer = (r1_km + r2_km) / 2  # Semi-major axis of transfer orbit in km
    T_transfer = 2 * math.pi * math.sqrt(a_transfer**3 / mu_km3_s2)  # Orbital period in seconds
    t_transfer = T_transfer / 2  # Transfer time is half the orbital period
    t_transfer_days = t_transfer / (60 * 60 * 24)  # Convert seconds to days
    return t_transfer_days

# Constants with units
r_earth = 1.496e8  # km, Earth's average orbital radius
r_mars = 2.279e8   # km, Mars' average orbital radius
mu_sun = 1.327e11  # km^3/s^2, standard gravitational parameter of the Sun

# Calculate transfer time
transfer_time_days = hohmann_transfer_time(r_earth, r_mars, mu_sun)

print(f"Time for Hohmann transfer from Earth to Mars: {transfer_time_days:.2f} days")
