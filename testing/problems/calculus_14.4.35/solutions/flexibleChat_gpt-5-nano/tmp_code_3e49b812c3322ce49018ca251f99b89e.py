import math
from dataclasses import dataclass


@dataclass
class TinCylinderResult:
    R_out: float
    R_in: float
    R_mid: float
    V_approx: float
    m_approx: float
    V_shell: float
    m_shell: float


def compute_tin_cylinder(D: float, h: float, t: float, rho: float) -> TinCylinderResult:
    """Compute radii, first-order (mid-surface) volume and mass, and exact shell volume and mass
    for a closed cylindrical tin can.

    Parameters:
        D: Outer diameter (cm)
        h: Height (cm)
        t: Uniform wall thickness (cm)
        rho: Density (g/cm^3)

    Returns:
        TinCylinderResult containing R_out, R_in, R_mid, V_approx, m_approx, V_shell, m_shell
    """
    # Basic input validation
    if not all(isinstance(x, (int, float)) for x in (D, h, t, rho)):
        raise TypeError("D, h, t, and rho must be numbers.")
    if D <= 0 or h <= 0 or t <= 0 or rho <= 0:
        raise ValueError("D, h, t, and rho must be positive numbers.")
    R_out = D / 2.0
    if t >= R_out:
        raise ValueError("Thickness t must be smaller than the outer radius (t < D/2).")
    R_in = R_out - t
    R_mid = (R_out + R_in) / 2.0

    # First-order (mid-surface) volume and mass
    V_approx = 2.0 * math.pi * R_mid * h * t
    m_approx = rho * V_approx

    # Exact shell volume and mass (outer minus inner cylinder)
    V_shell = math.pi * h * (R_out**2 - R_in**2)
    m_shell = rho * V_shell

    return TinCylinderResult(
        R_out=R_out,
        R_in=R_in,
        R_mid=R_mid,
        V_approx=V_approx,
        m_approx=m_approx,
        V_shell=V_shell,
        m_shell=m_shell,
    )


def main():
    # Given values (cm and g/cm^3)
    D = 8.0    # outer diameter in cm
    h = 12.0   # height in cm
    t = 0.04   # wall thickness in cm
    rho = 7.31 # density in g/cm^3

    res = compute_tin_cylinder(D, h, t, rho)

    print(f"R_out = {res.R_out:.4f} cm; R_in = {res.R_in:.4f} cm; R_mid = {res.R_mid:.4f} cm")
    print(f"V_approx = {res.V_approx:.4f} cm^3; m_approx = {res.m_approx:.4f} g")
    print(f"V_shell = {res.V_shell:.4f} cm^3; m_shell = {res.m_shell:.4f} g")


if __name__ == "__main__":
    main()
