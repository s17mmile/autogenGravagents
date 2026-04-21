# filename: calculate_moon_tide_height.py

def calculate_max_tidal_height():
    # Physical constants
    G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
    M_moon = 7.342e22  # mass of the Moon in kg
    R_earth = 6.371e6  # radius of the Earth in meters
    d = 3.844e8  # average distance from Earth to Moon in meters
    rho_water = 1025  # density of seawater in kg/m^3
    g = 9.81  # acceleration due to gravity on Earth in m/s^2

    # Tidal acceleration due to the Moon at Earth's surface (equilibrium tide theory)
    # a_tide = 2 * G * M_moon * R_earth / d^3
    a_tide = 2 * G * M_moon * R_earth / (d ** 3)

    # The tidal height h can be approximated by balancing tidal force and gravity:
    # h = a_tide * R_earth / g
    # This is a simplified model assuming ocean responds as a fluid layer
    h = a_tide * R_earth / g

    # Convert height to meters
    return h

if __name__ == '__main__':
    max_tide_height = calculate_max_tidal_height()
    print(f"Approximate maximum tidal height change caused by the Moon: {max_tide_height:.3f} meters")
