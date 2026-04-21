# filename: asteroid_final_speed.py

def calculate_final_speed():
    """Calculate the asteroid's final speed at Earth's surface given initial conditions."""
    # Given constants
    v_i_km_s = 12  # initial speed in km/s
    r_i_re = 10    # initial distance in Earth radii
    r_f_re = 1     # final distance in Earth radii (Earth's surface)
    g = 9.8        # acceleration due to gravity in m/s^2
    R_E = 6.371e6  # Earth's radius in meters

    # Convert initial speed to m/s
    v_i = v_i_km_s * 1000  # m/s

    # Calculate GM using g and R_E
    GM = g * R_E**2  # m^3/s^2

    # Convert distances to meters
    r_i = r_i_re * R_E
    r_f = r_f_re * R_E

    # Calculate final speed using conservation of energy
    v_f = (v_i**2 + 2 * GM * (1/r_f - 1/r_i))**0.5  # m/s

    # Convert final speed back to km/s
    v_f_km_s = v_f / 1000

    return v_f_km_s

if __name__ == '__main__':
    final_speed = calculate_final_speed()
    print(f"The asteroid's speed when it reaches Earth's surface is {final_speed:.2f} km/s.")
