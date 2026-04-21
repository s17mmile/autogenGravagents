# filename: calculate_delta_g.py

def calculate_delta_g(n_mmol, V1_cm3, V2_cm3, T_K):
    """Calculate Gibbs free energy change for isothermal expansion of an ideal gas.

    Parameters:
    n_mmol (float): amount of substance in millimoles
    V1_cm3 (float): initial volume in cubic centimeters
    V2_cm3 (float): final volume in cubic centimeters
    T_K (float): temperature in Kelvin

    Returns:
    float: Gibbs free energy change in Joules
    """
    import math
    if n_mmol <= 0 or V1_cm3 <= 0 or V2_cm3 <= 0 or T_K <= 0:
        raise ValueError("All input values must be positive.")
    R = 8.314  # J/(mol K)
    n = n_mmol * 1e-3  # convert mmol to mol
    delta_g = n * R * T_K * math.log(V2_cm3 / V1_cm3)
    return delta_g


if __name__ == "__main__":
    n_mmol = 3.0
    V1_cm3 = 36
    V2_cm3 = 60
    T_K = 300
    delta_g = calculate_delta_g(n_mmol, V1_cm3, V2_cm3, T_K)
    print(f"Delta G for the expansion process: {delta_g:.4f} J")
