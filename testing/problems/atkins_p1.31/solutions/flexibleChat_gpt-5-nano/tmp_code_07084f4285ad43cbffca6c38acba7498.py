import math


def compute_two_gas_height_and_pressure(
    P0=101325.0,
    w_N2_0=0.80,
    w_N2_target=0.90,
    T=298.15,
    g=9.81,
    R=8.314,
    M_N2=0.0280134,   # kg/mol (N2)
    M_O2=0.0319988,   # kg/mol (O2)
    verify=True
):
    """Compute the altitude where two isothermal gases (N2 and O2) reach a target
    sea-level mass fraction w_N2_target, and compute the total pressure there.

    Returns: (z, P_z, P_N2_z, P_O2_z, w_N2_z)
    where z is in meters, P_z in Pa, and w_N2_z is the mass fraction of N2 at z
    (computed for verification).
    """
    # 1) Sea-level mole ratio from mass fractions
    r0 = w_N2_0 * M_O2 / (M_N2 * (1.0 - w_N2_0))

    # 2) Height dependence constant
    c = (M_O2 - M_N2) * g / (R * T)

    # Target ratio for the desired N2 mass fraction
    R_target = w_N2_target * M_O2 / (M_N2 * (1.0 - w_N2_target))

    if R_target <= r0:
        raise ValueError("Target N2 fraction not achievable with given parameters; adjust w_N2_target.")

    # 3) Height where w_N2 = w_N2_target
    z = (1.0 / c) * math.log(R_target / r0)

    # 4) Surface mole fractions and partial pressures
    x_N2_0 = r0 / (1.0 + r0)
    x_O2_0 = 1.0 / (1.0 + r0)
    P_N2_0 = x_N2_0 * P0
    P_O2_0 = x_O2_0 * P0

    # 5) Exponential decays to height z
    P_N2_z = P_N2_0 * math.exp(- M_N2 * g * z / (R * T))
    P_O2_z = P_O2_0 * math.exp(- M_O2 * g * z / (R * T))
    P_z = P_N2_z + P_O2_z

    # Optional verification: actual w_N2 at height z
    w_N2_z = None
    if verify:
        R_z = r0 * math.exp(((M_O2 - M_N2) * g * z) / (R * T))
        w_N2_z = (M_N2 * R_z) / (M_N2 * R_z + M_O2)

    return z, P_z, P_N2_z, P_O2_z, w_N2_z


if __name__ == "__main__":
    z, Pz, Pn2, Po2, wN2z = compute_two_gas_height_and_pressure()
    print("Height z (m):", z)
    print("P_N2(z) Pa:", Pn2)
    print("P_O2(z) Pa:", Po2)
    print("Total P(z) Pa:", Pz)
    print("Total P(z) atm:", Pz / 101325.0)
    print("w_N2 at z:", wN2z)
