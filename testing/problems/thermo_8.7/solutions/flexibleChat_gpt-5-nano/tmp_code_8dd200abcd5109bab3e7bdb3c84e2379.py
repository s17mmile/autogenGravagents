import math


def compute_work(r: float, gamma: float, f: float = 3.0) -> tuple[float, float]:
    """
    Compute the change in surface area and the work required to expand a sphere
    from radius r to radius f*r against a surrounding surface tension gamma.

    Parameters:
        r (float): initial radius in meters, r > 0
        gamma (float): interfacial surface tension in N/m, gamma >= 0
        f (float): final radius factor relative to initial radius (default 3.0)

    Returns:
        (delta_A, W): delta_A in m^2, W in J
    """
    if r <= 0:
        raise ValueError("radius must be positive")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if f <= 0:
        raise ValueError("final radius factor f must be positive")

    # DeltaA for expansion from r to f*r: DeltaA = 4*pi*((f*r)^2 - r^2)
    delta_A = 4.0 * math.pi * ((f * r) ** 2 - r ** 2)
    W = gamma * delta_A
    return delta_A, W


def main():
    # Preset constants (no user input)
    r = 20.0e-6  # meters
    gamma_values = [0.01, 0.03, 0.10, 0.072]  # N/m, representative interfacial tensions
    f = 3.0  # final radius factor

    # Theoretical DeltaA for f = 3: DeltaA = 32*pi*r^2
    delta_A_expected = 32.0 * math.pi * (r ** 2)
    print("DeltaA (theoretical for f=3) = {0:.3e} m^2".format(delta_A_expected))

    for g in gamma_values:
        delta_A, W = compute_work(r, g, f=f)
        print("gamma = {0:.3f} N/m -> DeltaA = {1:.3e} m^2, W = {2:.3e} J ({3:.3f} nJ)".format(
            g, delta_A, W, W * 1e9))

    # Simple internal unit check to validate consistency W = gamma * DeltaA
    for g in gamma_values:
        delta_A, W = compute_work(r, g, f=f)
        assert math.isclose(W, g * delta_A, rel_tol=1e-12, abs_tol=0.0)


if __name__ == "__main__":
    main()
