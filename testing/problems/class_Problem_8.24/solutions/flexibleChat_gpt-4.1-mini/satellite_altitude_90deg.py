# filename: satellite_altitude_90deg.py
import math

def satellite_altitude_at_90_deg(perigee_km, apogee_km, earth_radius_km=6371):
    """Calculate satellite altitude above Earth's surface at 90 degrees true anomaly from perigee.

    Args:
        perigee_km (float): Perigee altitude above Earth's surface in km.
        apogee_km (float): Apogee altitude above Earth's surface in km.
        earth_radius_km (float, optional): Earth's radius in km. Defaults to 6371 km.

    Returns:
        float: Altitude above Earth's surface at 90 degrees from perigee in km.
    """
    if perigee_km < 0 or apogee_km < 0:
        raise ValueError("Perigee and apogee altitudes must be non-negative.")
    if apogee_km < perigee_km:
        raise ValueError("Apogee altitude must be greater than or equal to perigee altitude.")

    # Convert perigee and apogee altitudes to distances from Earth's center
    r_p = earth_radius_km + perigee_km
    r_a = earth_radius_km + apogee_km

    # Semi-major axis
    a = (r_p + r_a) / 2

    # Eccentricity
    e = (r_a - r_p) / (r_a + r_p)

    # True anomaly at 90 degrees in radians
    theta = math.radians(90)

    # Radius at theta
    # Note: cos(90 degrees) = 0, so denominator simplifies to 1
    r_theta = a * (1 - e**2) / (1 + e * math.cos(theta))

    # Altitude above Earth's surface
    altitude = r_theta - earth_radius_km
    return altitude

# Given values
perigee_altitude = 300  # km
apogee_altitude = 3500  # km

altitude_90_deg = satellite_altitude_at_90_deg(perigee_altitude, apogee_altitude)

print(f"Altitude of the satellite at 90 degrees from perigee: {altitude_90_deg:.2f} km")
