# filename: calculate_neon_entropy.py
import math

def molar_entropy_neon(T1=298.15, S1=146.22, T2=500.0):
    """
    Calculate the molar entropy of neon at temperature T2 given its molar entropy at temperature T1,
    assuming ideal gas behavior and constant volume.

    Parameters:
    T1 (float): Initial temperature in Kelvin (must be > 0).
    S1 (float): Molar entropy at T1 in J/(mol K).
    T2 (float): Final temperature in Kelvin (must be > 0).

    Returns:
    float: Molar entropy at temperature T2 in J/(mol K).
    """
    if T1 <= 0 or T2 <= 0:
        raise ValueError("Temperatures must be greater than zero Kelvin.")

    R = 8.314  # J/(mol K)
    C_V = 1.5 * R  # molar heat capacity at constant volume for monatomic ideal gas
    delta_S = C_V * math.log(T2 / T1)
    S2 = S1 + delta_S
    return S2

entropy_500K = molar_entropy_neon()
print(f'Molar entropy of neon at 500 K: {entropy_500K:.2f} J/(mol K)')
