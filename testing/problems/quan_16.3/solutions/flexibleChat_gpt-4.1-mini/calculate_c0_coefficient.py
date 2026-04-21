# filename: calculate_c0_coefficient.py

def calculate_c0(E_SCF, E_CI, E_Davidson):
    """
    Calculate the coefficient c0 of the reference configuration Phi_0 in the normalized CI-SD wave function.

    Parameters:
    E_SCF (float): Energy from frozen-core SCF/DZP calculation (hartrees).
    E_CI (float): Energy from CI-SD/DZP calculation (hartrees).
    E_Davidson (float): Energy after Davidson correction (hartrees).

    Returns:
    float: Coefficient c0 of Phi_0 in the normalized CI-SD wave function.

    Raises:
    ValueError: If denominator is zero or computed c0 squared is out of [0,1] range.
    """
    denominator = E_CI - E_SCF
    if denominator == 0:
        raise ValueError("Denominator (E_CI - E_SCF) is zero, cannot compute coefficient.")

    numerator = E_Davidson - E_CI
    ratio = numerator / denominator
    c0_squared = 1 - ratio

    if not (0 <= c0_squared <= 1):
        raise ValueError(f"Computed c0 squared value {c0_squared} is out of valid range [0,1].")

    c0 = c0_squared ** 0.5
    return c0

# Given energies in hartrees
E_SCF = -76.040542
E_CI = -76.243772
E_Davidson = -76.254549

c0 = calculate_c0(E_SCF, E_CI, E_Davidson)
print(f"Coefficient of Phi_0 in normalized CI-SD wave function: {c0:.6f}")
