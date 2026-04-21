# filename: calculate_gravity_poles.py
import math

def calculate_gravity_poles():
    """Calculate effective gravitational acceleration at Earth's poles and compare with empirical formula."""
    # Constants
    G = 6.67430e-11  # gravitational constant, m^3 kg^-1 s^-2
    M = 5.9722e24    # mass of Earth, kg
    R_equator = 6378e3  # equatorial radius in meters
    R_pole = 6357e3      # polar radius in meters
    omega = 7.292115e-5  # angular velocity of Earth, rad/s
    g0 = 9.80665         # standard gravity at mean radius, m/s^2
    R_mean = 6371e3      # mean Earth radius in meters

    # Calculate gravitational acceleration at poles using inverse square law
    g_pole_calc = g0 * (R_mean / R_pole)**2

    # Centrifugal acceleration at poles (cos(90 deg) = 0)
    a_centrifugal_pole = omega**2 * R_pole * (math.cos(math.radians(90)))**2

    # Effective gravity at poles
    g_eff_pole = g_pole_calc - a_centrifugal_pole  # centrifugal acceleration is zero here

    # Empirical formula for gravity at latitude lambda (degrees)
    # g = 9.780356 * (1 + 0.0052885*sin^2(lambda) - 0.0000059*sin^2(2*lambda))
    lambda_pole = 90
    sin_lambda_sq = math.sin(math.radians(lambda_pole))**2
    sin_2lambda_sq = math.sin(math.radians(2*lambda_pole))**2
    g_empirical = 9.780356 * (1 + 0.0052885 * sin_lambda_sq - 0.0000059 * sin_2lambda_sq)

    # Output results
    print(f"Calculated effective gravity at poles: {g_eff_pole:.6f} m/s^2")
    print(f"Empirical formula gravity at poles: {g_empirical:.6f} m/s^2")
    print(f"Difference: {abs(g_eff_pole - g_empirical):.6f} m/s^2")

    return g_eff_pole, g_empirical

if __name__ == "__main__":
    calculate_gravity_poles()
