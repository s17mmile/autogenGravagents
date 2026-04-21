# filename: calculate_delta_S_reaction.py
import numpy as np

# Constants
T1 = 298.15  # Reference temperature in K
T2 = 725.0   # Target temperature in K

# Standard entropy values at 298 K in J/(mol K)
# Values from NIST Chemistry WebBook
S_H2_298 = 130.68
S_N2_298 = 191.61
S_NH3_298 = 192.77

# Heat capacity polynomial coefficients Cp(T) = a + b*T + c*T^2 (J/mol K)
# Coefficients truncated to T^2 terms as per instruction
# Values approximated from NIST data (units consistent)
# H2
a_H2 = 29.11
b_H2 = -0.1916e-2
c_H2 = 0.4003e-5

# N2
a_N2 = 28.90
b_N2 = 0.1571e-2
c_N2 = -0.8081e-6

# NH3
a_NH3 = 35.06
b_NH3 = 1.85e-2
c_NH3 = -0.299e-5


def delta_S_from_Cp(a, b, c, T1, T2):
    """Calculate the integral of Cp(T)/T dT from T1 to T2 for Cp(T) = a + b*T + c*T^2.
    Parameters:
        a, b, c (float): Heat capacity polynomial coefficients (J/mol K)
        T1, T2 (float): Temperature limits in K (T2 > T1)
    Returns:
        float: Entropy change contribution in J/(mol K)
    """
    term1 = a * np.log(T2 / T1)
    term2 = b * (T2 - T1)
    term3 = (c / 2) * (T2**2 - T1**2)
    return term1 + term2 + term3

# Calculate entropy at T2 for each species
S_H2_725 = S_H2_298 + delta_S_from_Cp(a_H2, b_H2, c_H2, T1, T2)
S_N2_725 = S_N2_298 + delta_S_from_Cp(a_N2, b_N2, c_N2, T1, T2)
S_NH3_725 = S_NH3_298 + delta_S_from_Cp(a_NH3, b_NH3, c_NH3, T1, T2)

# Reaction: 3 H2 + N2 -> 2 NH3
# Calculate Delta S degree at 725 K
Delta_S_725 = 2 * S_NH3_725 - (3 * S_H2_725 + 1 * S_N2_725)

# Output the result
print(f"Standard entropy change (Delta S degree) for the reaction at {T2} K: {Delta_S_725:.2f} J/(mol K)")
