# filename: double_star_orbital_period.py
import math

def calculate_orbital_period(mass_solar=1.0, separation_ly=4.0):
    """
    Calculate the orbital period of two stars of equal mass orbiting each other.
    :param mass_solar: Mass of each star in solar masses (default 1.0)
    :param separation_ly: Separation between the two stars in light years (default 4.0)
    :return: Orbital period in years
    """
    # Constants
    G = 6.67430e-11  # gravitational constant in m^3 kg^-1 s^-2
    M_sun = 1.98847e30  # mass of the Sun in kg (more precise)
    light_year_m = 9.4607e15  # meters in one light year (more precise)

    # Convert inputs to SI units
    M = mass_solar * M_sun
    d = separation_ly * light_year_m

    # Each star orbits at radius r = d/2
    r = d / 2

    # Calculate orbital period T using formula:
    # T = 2 * pi * sqrt(2) * r^(3/2) / sqrt(G * M)
    numerator = 2 * math.pi * math.sqrt(2) * (r ** 1.5)
    denominator = math.sqrt(G * M)
    T_seconds = numerator / denominator

    # Convert seconds to years
    seconds_per_year = 365.25 * 24 * 3600
    T_years = T_seconds / seconds_per_year

    return T_years

if __name__ == '__main__':
    period_years = calculate_orbital_period()
    print(f"Orbital period of the double stars: {period_years:.2e} years")
