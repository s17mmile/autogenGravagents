# filename: fugacity_coefficient_argon.py
import math

R = 8.3145  # Gas constant in J/(mol K)

def fugacity_coefficient(p_atm, T_K, B_cm3_per_mol, C_cm6_per_mol2):
    """
    Calculate the fugacity coefficient of a gas obeying the virial equation of state:
    p V_m = R T (1 + B/V_m + C/V_m^2)

    Parameters:
    p_atm : float
        Pressure in atmospheres
    T_K : float
        Temperature in Kelvin
    B_cm3_per_mol : float
        Second virial coefficient B in cm^3/mol
    C_cm6_per_mol2 : float
        Third virial coefficient C in cm^6/mol^2

    Returns:
    phi : float
        Fugacity coefficient (dimensionless)
    """
    # Convert pressure from atm to Pa
    p_Pa = p_atm * 101325
    # Convert B from cm^3/mol to m^3/mol
    B_m3_per_mol = B_cm3_per_mol * 1e-6
    # Convert C from cm^6/mol^2 to m^6/mol^2
    C_m6_per_mol2 = C_cm6_per_mol2 * 1e-12

    ln_phi = (B_m3_per_mol * p_Pa) / (R * T_K) + (C_m6_per_mol2 * p_Pa**2) / (2 * (R * T_K)**2)
    phi = math.exp(ln_phi)
    return phi

# Given data
p = 1.00  # atm
T = 100.0  # K
B = -21.13  # cm^3/mol
C = 1054    # cm^6/mol^2

phi = fugacity_coefficient(p, T, B, C)
fugacity = phi * p  # fugacity in atm

print(f"Fugacity coefficient (phi) at {p} atm and {T} K: {phi:.6f}")
print(f"Fugacity (f) at {p} atm and {T} K: {fugacity:.6f} atm")
