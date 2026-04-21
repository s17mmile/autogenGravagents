# filename: delta_Um_vdw_nitrogen.py

def calculate_delta_Um(V_i_liters, V_f_liters, a_L2_atm_per_mol2=1.39):
    """
    Calculate the change in molar internal energy (Delta U_m) for the isothermal expansion
    of nitrogen gas modeled as a van der Waals gas.

    Parameters:
    - V_i_liters: Initial volume in liters (dm^3)
    - V_f_liters: Final volume in liters (dm^3)
    - a_L2_atm_per_mol2: van der Waals constant 'a' for nitrogen in L^2 atm / mol^2 (default 1.39)

    Returns:
    - Delta U_m in Joules per mole
    """
    if V_i_liters <= 0 or V_f_liters <= 0:
        raise ValueError("Volumes must be positive numbers.")

    delta_Um_L_atm = -a_L2_atm_per_mol2 * (1 / V_f_liters - 1 / V_i_liters)
    delta_Um_J_per_mol = delta_Um_L_atm * 101.325  # Convert L atm to Joules
    return delta_Um_J_per_mol


# Given volumes in dm^3 (equivalent to liters)
V_i = 1.00  # initial volume in liters
V_f = 24.8  # final volume in liters

# Calculate Delta U_m
delta_Um = calculate_delta_Um(V_i, V_f)

print(f"Change in molar internal energy (Delta U_m) for nitrogen gas expansion:")
print(f"Delta U_m = {delta_Um:.2f} J/mol")
