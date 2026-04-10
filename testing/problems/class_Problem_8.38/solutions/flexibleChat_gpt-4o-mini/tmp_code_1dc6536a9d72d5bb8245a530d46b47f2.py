import math

# Constants
G = 6.674 * 10**-11  # Gravitational constant (m^3 kg^-1 s^-2)
M = 1.989 * 10**30   # Mass of the Sun (kg)

# Distances from the Sun (in meters)
r_E = 1.496 * 10**11  # Earth's distance from the Sun (m)
r_V = 1.082 * 10**11  # Venus's distance from the Sun (m)

def calculate_semi_major_axis(r_E, r_V):
    return (r_E + r_V) / 2


def calculate_velocity_transfer(r, semi_major_axis):
    return math.sqrt(G * M * (2 / r - 1 / semi_major_axis))


def calculate_delta_v():
    # Step 1: Calculate the semi-major axis of the transfer orbit
    semi_major_axis = calculate_semi_major_axis(r_E, r_V)

    # Step 2: Calculate the velocity at perihelion (Earth's orbit)
    v_perihelion = calculate_velocity_transfer(r_E, semi_major_axis)
    print(f'Velocity at perihelion (v_perihelion): {v_perihelion:.2f} m/s')

    # Step 3: Calculate the velocity at aphelion (Venus's orbit)
    v_aphelion = calculate_velocity_transfer(r_V, semi_major_axis)
    print(f'Velocity at aphelion (v_aphelion): {v_aphelion:.2f} m/s')

    # Step 4: Calculate the velocities of Earth and Venus
    v_E = math.sqrt(G * M / r_E)  # Earth's orbital velocity
    v_V = math.sqrt(G * M / r_V)  # Venus's orbital velocity
    print(f'Earth orbital velocity (v_E): {v_E:.2f} m/s')
    print(f'Venus orbital velocity (v_V): {v_V:.2f} m/s')

    # Step 5: Calculate the Delta v for the first and second burns
    Delta_v1 = v_perihelion - v_E  # First burn
    Delta_v2 = v_aphelion - v_V  # Second burn (corrected)

    # Total Delta v
    Delta_v_total = Delta_v1 + Delta_v2

    return Delta_v_total / 1000  # Convert to km/s

# Output the result
try:
    minimum_delta_v = calculate_delta_v()
    print(f'Minimum Delta v required: {minimum_delta_v:.2f} km/s')
except ZeroDivisionError:
    print('Error: Division by zero encountered in calculations.')
except Exception as e:
    print(f'An error occurred: {e}')