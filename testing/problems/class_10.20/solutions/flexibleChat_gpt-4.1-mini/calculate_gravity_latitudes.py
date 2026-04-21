# filename: calculate_gravity_latitudes.py
import math

def calculate_gravity_at_latitude(latitude_deg):
    """Calculate effective gravitational acceleration at a given latitude and compare with empirical formula."""
    # Constants
    g0 = 9.80665         # standard gravity at mean radius, m/s^2
    R_mean = 6371e3      # mean Earth radius in meters
    R_equator = 6378e3   # equatorial radius in meters
    R_pole = 6357e3      # polar radius in meters
    omega = 7.292115e-5  # angular velocity of Earth, rad/s

    # Approximate radius at latitude using cosine squared interpolation
    lat_rad = math.radians(latitude_deg)
    R_lat = R_pole + (R_equator - R_pole) * math.cos(lat_rad)**2

    # Gravitational acceleration at latitude (inverse square law)
    g_gravity = g0 * (R_mean / R_lat)**2

    # Centrifugal acceleration at latitude
    a_centrifugal = omega**2 * R_lat * math.cos(lat_rad)**2

    # Effective gravity
    g_effective = g_gravity - a_centrifugal

    # Empirical formula for gravity at latitude lambda (degrees)
    sin_lambda_sq = math.sin(lat_rad)**2
    sin_2lambda_sq = math.sin(2 * lat_rad)**2
    g_empirical = 9.780356 * (1 + 0.0052885 * sin_lambda_sq - 0.0000059 * sin_2lambda_sq)

    return g_effective, g_empirical

def main():
    latitudes = [0, 30, 45, 60, 90]  # degrees
    print(f"{'Latitude (deg)':>12} | {'Calc Gravity (m/s^2)':>20} | {'Empirical Gravity (m/s^2)':>25} | {'Difference (m/s^2)':>20}")
    print('-'*85)
    for lat in latitudes:
        g_calc, g_emp = calculate_gravity_at_latitude(lat)
        diff = abs(g_calc - g_emp)
        print(f"{lat:12.1f} | {g_calc:20.6f} | {g_emp:25.6f} | {diff:20.6f}")

if __name__ == '__main__':
    main()
