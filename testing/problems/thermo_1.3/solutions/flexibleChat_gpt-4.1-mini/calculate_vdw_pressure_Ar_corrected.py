# filename: calculate_vdw_pressure_Ar_corrected.py

def calculate_pressure_vdw(V_m_L_per_mol, T_K, a_bar_dm6_per_mol2, b_dm3_per_mol):
    """
    Calculate pressure of Ar using van der Waals equation of state.

    Parameters:
    V_m_L_per_mol: molar volume in L/mol (1 L = 1 dm^3)
    T_K: temperature in Kelvin
    a_bar_dm6_per_mol2: van der Waals constant a in bar·dm^6/mol^2
    b_dm3_per_mol: van der Waals constant b in dm^3/mol

    Returns:
    P: pressure in bar
    dominant_force: 'attractive' or 'repulsive' indicating dominant interaction
    """
    R = 0.08314  # bar·dm^3/(mol·K)

    V_m = V_m_L_per_mol

    if V_m <= b_dm3_per_mol:
        raise ValueError("Molar volume must be greater than b to avoid division by zero.")

    # Calculate pressure using van der Waals equation
    P = (R * T_K) / (V_m - b_dm3_per_mol) - a_bar_dm6_per_mol2 / (V_m ** 2)

    # Calculate ideal gas pressure for comparison
    P_ideal = (R * T_K) / V_m

    # Calculate repulsive pressure correction (difference between ideal and volume-corrected pressure)
    repulsive_pressure_correction = P_ideal - (R * T_K) / (V_m - b_dm3_per_mol)

    # Attractive pressure correction
    attractive_pressure_correction = a_bar_dm6_per_mol2 / (V_m ** 2)

    # Determine dominant force by comparing magnitudes of pressure corrections
    if attractive_pressure_correction > abs(repulsive_pressure_correction):
        dominant_force = 'attractive'
    else:
        dominant_force = 'repulsive'

    return P, dominant_force


# Given values
V_m = 1.31  # L/mol
T = 426  # K
 a = 1.355  # bar·dm^6/mol^2
b = 0.0320  # dm^3/mol

pressure, dominant = calculate_pressure_vdw(V_m, T, a, b)

result_message = f"Calculated pressure: {pressure:.3f} bar\nDominant interaction: {dominant}"

result_message
