# filename: calculate_escape_velocity.py
import math

def calculate_escape_velocity(altitude_km):
    # Constants
    G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
    M = 5.972e24     # mass of Earth in kg
    R = 6.371e6      # radius of Earth in meters

    # Convert altitude to meters
    h = altitude_km * 1000

    # Distance from Earth's center
    r = R + h

    # Calculate escape velocity
    v_esc = math.sqrt(2 * G * M / r)
    return v_esc

# Altitude 200 km
altitude = 200
escape_velocity = calculate_escape_velocity(altitude)
print(f"Minimum escape velocity at {altitude} km altitude: {escape_velocity:.2f} m/s")
