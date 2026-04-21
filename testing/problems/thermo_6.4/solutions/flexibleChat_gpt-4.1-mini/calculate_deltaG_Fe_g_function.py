# filename: calculate_deltaG_Fe_g_function.py

def calculate_deltaG_Fe_g(T_new, Delta_G_298=370.7, Delta_H_298=416.3, T_298=298.15):
    """
    Calculate the standard Gibbs free energy of formation for Fe(g) at a new temperature T_new.

    Parameters:
    - T_new: Temperature at which to calculate Delta G (K)
    - Delta_G_298: Standard Gibbs free energy of formation at 298.15 K (kJ/mol)
    - Delta_H_298: Standard enthalpy of formation at 298.15 K (kJ/mol)
    - T_298: Reference temperature (K)

    Returns:
    - Delta_G_new: Standard Gibbs free energy of formation at T_new (kJ/mol)
    """
    # Calculate standard entropy of formation at 298.15 K (in kJ/(mol K))
    Delta_S = (Delta_H_298 - Delta_G_298) / T_298

    # Calculate Delta G at new temperature assuming Delta H is constant
    Delta_G_new = Delta_H_298 - T_new * Delta_S

    return Delta_G_new


# Calculate Delta G at 400 K
T_target = 400.0
Delta_G_400 = calculate_deltaG_Fe_g(T_target)

print(f"Standard Gibbs free energy of formation for Fe(g) at {T_target} K: {Delta_G_400:.2f} kJ/mol")
