import math
from dataclasses import dataclass

@dataclass(frozen=True)
class TimeResult:
    t_min: float
    t_hours: float
    tau: float
    C0: float


def time_to_target(V: float, Q: float, M0_kg: float, C_target: float) -> TimeResult:
    """Compute time in minutes to reach C_target (g/gal) for a well-mixed pool.

    Parameters:
        V: pool volume in gallons
        Q: filter flow rate in gallons per minute
        M0_kg: initial mass of dye in kilograms
        C_target: target concentration in g/gal

    Returns:
        TimeResult with t_min, t_hours, tau, C0.
    """
    if V <= 0 or Q <= 0:
        raise ValueError("V and Q must be positive")
    if M0_kg < 0:
        raise ValueError("M0_kg must be non-negative")

    M0_g = M0_kg * 1000.0
    C0 = M0_g / V

    if C_target <= 0:
        return TimeResult(float('inf'), float('inf'), V / Q, C0)
    if C_target >= C0:
        return TimeResult(0.0, 0.0, V / Q, C0)

    tau = V / Q
    # Use log1p for improved precision when C_target is close to C0
    t_min = tau * math.log1p((C0 - C_target) / C_target)
    t_hours = t_min / 60.0
    return TimeResult(t_min, t_hours, tau, C0)


if __name__ == '__main__':
    V = 60_000.0
    Q = 200.0
    M0_kg = 5.0
    C_target = 0.02

    res = time_to_target(V, Q, M0_kg, C_target)
    print(f"tau = {res.tau} min")
    print(f"C0 = {res.C0} g/gal")
    print(f"t_min = {res.t_min}")
    print(f"t_hours = {res.t_hours}")