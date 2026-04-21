# filename: delta_v_earth_to_venus_function.py
import math

def calculate_min_delta_v():
    """
    Calculate the minimum Delta-v required for a Hohmann transfer from Earth's orbit to Venus's orbit.
    Assumes circular, coplanar orbits and only the Sun's gravity.

    Returns:
        total_delta_v (float): Total Delta-v in km/s
    """
    # Constants (units in km, s, km^3/s^2)
    GM_sun = 1.327e11  # Gravitational parameter of the Sun (km^3/s^2)
    r_E = 1.496e8      # Earth's orbital radius (km, 1 AU)
    r_V = 1.082e8      # Venus's orbital radius (km, 0.723 AU)

    # Orbital velocities (km/s)
    v_E = math.sqrt(GM_sun / r_E)  # Earth's orbital velocity
    v_V = math.sqrt(GM_sun / r_V)  # Venus's orbital velocity

    # Semi-major axis of the Hohmann transfer ellipse (km)
    a = (r_E + r_V) / 2

    # Velocities on the transfer orbit at Earth's and Venus's orbits (km/s)
    v_trans_E = math.sqrt(GM_sun * (2 / r_E - 1 / a))  # velocity at Earth's orbit on transfer ellipse
    v_trans_V = math.sqrt(GM_sun * (2 / r_V - 1 / a))  # velocity at Venus's orbit on transfer ellipse

    # Delta-v calculations (km/s)
    delta_v1 = abs(v_trans_E - v_E)  # impulse to enter transfer orbit
    delta_v2 = abs(v_V - v_trans_V)  # impulse to circularize at Venus orbit

    total_delta_v = delta_v1 + delta_v2
    return total_delta_v

if __name__ == "__main__":
    delta_v = calculate_min_delta_v()
    print(f"Minimum Delta-v required for transfer from Earth to Venus: {delta_v:.4f} km/s")
