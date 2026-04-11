import math


def compute_lambda_D(I_M_per_L, T=298.0, epsilon_r=78.5):
    """
    Compute Debye length lambda_D in nanometers for a given ionic strength I.

    Parameters:
        I_M_per_L: ionic strength in mol/L (I)
        T: temperature in Kelvin (default 298.0 K)
        epsilon_r: relative permittivity of the solvent (default 78.5 for water at ~25 C)

    Returns:
        (lambda_D_nm, kappa, I_m3)
        - lambda_D_nm: Debye length in nanometers
        - kappa: inverse Debye length in 1/m
        - I_m3: ionic strength in mol/m^3
    """
    # Physical constants
    epsilon_0 = 8.8541878128e-12  # F/m
    k_B = 1.380649e-23            # J/K
    N_A = 6.02214076e23           # 1/mol
    e = 1.602176634e-19           # C

    # Convert I from mol/L to mol/m^3
    I_m3 = I_M_per_L * 1000.0

    # Debye screening parameter: kappa^2 = (2 N_A e^2 I) / (epsilon_r epsilon_0 k_B T)
    kappa_sq = (2.0 * N_A * e * e * I_m3) / (epsilon_r * epsilon_0 * k_B * T)
    kappa = math.sqrt(max(kappa_sq, 0.0))

    # Debye length: lambda_D = 1 / kappa
    lambda_D_m = 1.0 / kappa if kappa > 0.0 else float('inf')
    lambda_D_nm = lambda_D_m * 1.0e9
    return lambda_D_nm, kappa, I_m3


def main():
    # Scenario: potassium phosphate dissociates as K3PO4 -> 3 K+ + PO4^{3-}
    salt_concentration_M = 0.0075
    c_K = 3.0 * salt_concentration_M      # [K+] in mol/L
    c_PO4 = salt_concentration_M            # [PO4^{3-}] in mol/L

    # Ionic strength: I = 0.5 * sum(c_i z_i^2)
    I_M_per_L = 0.5 * (c_K * 1.0**2 + c_PO4 * 3.0**2)  # mol/L

    lambda_nm, kappa, I_m3 = compute_lambda_D(I_M_per_L, T=298.0, epsilon_r=78.5)

    # Cross-check using empirical relation: lambda_D (nm) ~ 0.304 / sqrt(I) with I in mol/L
    lambda_nm_check = 0.3040 / math.sqrt(I_M_per_L) if I_M_per_L > 0 else float('inf')

    print(f"I (mol/L) = {I_M_per_L:.6f}")
    print(f"I (mol/m^3) = {I_m3:.3f}")
    print(f"kappa (1/m) = {kappa:.6e}")
    print(f"lambda_D (nm) = {lambda_nm:.6f}")
    print(f"Cross-check lambda_D (nm) = {lambda_nm_check:.6f}")


if __name__ == "__main__":
    main()
