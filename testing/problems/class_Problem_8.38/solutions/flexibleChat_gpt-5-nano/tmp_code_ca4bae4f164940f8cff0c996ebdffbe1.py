import math
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.WARNING)


@dataclass
class HohmannResult:
    delta_v1: float
    delta_v2: float
    total_delta_v: float
    v1: float
    v2: float
    v_a: float
    v_p: float
    a_t: float
    r1: float
    r2: float
    mu: float


def hohmann_sun_transfer(r1: float, r2: float, mu: float) -> HohmannResult:
    """Compute the minimum impulsive delta-v for a Hohmann transfer between two circular
    heliocentric orbits around the Sun.

    Parameters:
    - r1: initial circular orbit radius (km)
    - r2: final circular orbit radius (km)
    - mu: central body gravitational parameter (km^3/s^2)

    Returns a HohmannResult containing delta_v1, delta_v2, total_delta_v, v1, v2,
    v_a, v_p, a_t, and the input radii.
    """
    if r1 <= 0 or r2 <= 0 or mu <= 0:
        raise ValueError("r1, r2 and mu must be positive scalars.")

    a_t = 0.5 * (r1 + r2)

    # Radicands for vis-viva at burn points
    rad1 = 2.0 / r1 - 1.0 / a_t
    rad2 = 2.0 / r2 - 1.0 / a_t
    if rad1 <= 0 or rad2 <= 0:
        raise ValueError("Non-physical radii for Hohmann transfer: radicands must be positive.")

    # Circular orbital speeds
    v1 = math.sqrt(mu / r1)
    v2 = math.sqrt(mu / r2)

    # Transfer ellipse speeds at burn points (vis-viva)
    v_a = math.sqrt(mu * rad1)  # at apoapsis (r = r1)
    v_p = math.sqrt(mu * rad2)  # at periapsis (r = r2)

    # Delta-v magnitudes (impulses)
    delta_v1 = abs(v1 - v_a)
    delta_v2 = abs(v_p - v2)
    total_delta_v = delta_v1 + delta_v2

    # Optional closed-form cross-checks for robustness
    delv1_cf = abs(v1 * (math.sqrt(2.0 * r2 / (r1 + r2)) - 1.0))
    delv2_cf = abs(v2 * (1.0 - math.sqrt(2.0 * r1 / (r1 + r2))))
    if abs(delta_v1 - delv1_cf) > 1e-6 or abs(delta_v2 - delv2_cf) > 1e-6:
        logging.warning("delta-v cross-check deviates beyond tolerance.")

    return HohmannResult(
        delta_v1=delta_v1,
        delta_v2=delta_v2,
        total_delta_v=total_delta_v,
        v1=v1,
        v2=v2,
        v_a=v_a,
        v_p=v_p,
        a_t=a_t,
        r1=r1,
        r2=r2,
        mu=mu,
    )


if __name__ == "__main__":
    # Preset solar system constants (km, km^3/s^2)
    mu_sun = 1.32712440018e11  # Sun's GM
    AU_km = 1.495978707e8       # 1 AU in kilometers

    # Earth to Venus example (precise Venus semi-major axis)
    r1 = 1.0 * AU_km
    r2 = 0.723332 * AU_km

    res = hohmann_sun_transfer(r1, r2, mu_sun)

    print(f"r1 = {res.r1:.6f} km, r2 = {res.r2:.6f} km")
    print(f"a_t = {res.a_t:.6f} km")
    print(f"v1 = {res.v1:.6f} km/s, v_a = {res.v_a:.6f} km/s, delta_v1 = {res.delta_v1:.6f} km/s")
    print(f"v2 = {res.v2:.6f} km/s, v_p = {res.v_p:.6f} km/s, delta_v2 = {res.delta_v2:.6f} km/s")
    print(f"Total delta_v = {res.total_delta_v:.6f} km/s")
