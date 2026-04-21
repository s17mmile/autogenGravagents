# filename: adiabatic_expansion_argon.py

def calculate_final_temperature(T1, V1, V2, gamma):
    """
    Calculate the final temperature after a reversible adiabatic expansion of an ideal gas.

    Parameters:
    T1 (float): Initial temperature in Kelvin
    V1 (float): Initial volume (units must be consistent with V2, but actual units cancel out)
    V2 (float): Final volume (same units as V1)
    gamma (float): Heat capacity ratio (Cp/Cv) of the gas

    Returns:
    float: Final temperature in Kelvin
    """
    T2 = T1 * (V1 / V2) ** (gamma - 1)
    return T2

# Given data
mass_argon_g = 12.0  # grams (not needed for temperature calculation as process depends on volume and temperature ratios)
V1_dm3 = 1.0  # initial volume in dm^3
V2_dm3 = 3.0  # final volume in dm^3
T1_K = 273.15  # initial temperature in Kelvin

gamma_argon = 5.0 / 3.0  # heat capacity ratio for monatomic ideal gas

# Volumes are used as-is because the ratio V1/V2 is unitless
V1 = V1_dm3
V2 = V2_dm3

# Calculate final temperature
T2_K = calculate_final_temperature(T1_K, V1, V2, gamma_argon)

print(f"Final temperature after adiabatic expansion: {T2_K:.2f} K")
