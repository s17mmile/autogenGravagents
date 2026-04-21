# filename: calculate_deltaG_375K.py

def calculate_deltaG_375K():
    # Given data
    Delta_H_298 = -566.0  # kJ/mol, standard enthalpy change at 298 K
    Delta_fG_CO = -137.16  # kJ/mol, Gibbs free energy of formation of CO at 298 K
    Delta_fG_CO2 = -394.36  # kJ/mol, Gibbs free energy of formation of CO2 at 298 K

    # Reaction: 2 CO + O2 -> 2 CO2
    # Calculate Delta_rG_298 using formation Gibbs free energies
    Delta_G_298 = 2 * Delta_fG_CO2 - (2 * Delta_fG_CO + 0)  # O2 elemental form, Gf=0

    # Calculate Delta_rS_298 from Delta_H and Delta_G at 298 K
    T_298 = 298  # K
    Delta_S_298_kJ = (Delta_H_298 - Delta_G_298) / T_298  # kJ/(mol K)
    Delta_S_298 = Delta_S_298_kJ * 1000  # Convert to J/(mol K)

    # Calculate Delta_rG at 375 K using Gibbs-Helmholtz equation
    T_375 = 375  # K
    Delta_G_375 = Delta_H_298 - T_375 * Delta_S_298_kJ  # kJ/mol

    # Print results
    print(f"Delta_rG at 298 K: {Delta_G_298:.2f} kJ/mol")
    print(f"Delta_rS at 298 K: {Delta_S_298:.2f} J/(mol K)")
    print(f"Delta_rG at 375 K: {Delta_G_375:.2f} kJ/mol")

# Run the calculation
calculate_deltaG_375K()
