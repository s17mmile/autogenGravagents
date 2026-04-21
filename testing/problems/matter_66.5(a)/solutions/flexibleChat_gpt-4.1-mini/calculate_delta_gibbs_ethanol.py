# filename: calculate_delta_gibbs_ethanol.py

def calculate_delta_gibbs_energy(mass_g=35, density_g_per_cm3=0.789, molar_mass_g_per_mol=46.07, p1_atm=1, p2_atm=3000):
    """
    Calculate the change in Gibbs free energy (Delta G) for ethanol when pressure is increased isothermally.
    Assumes constant molar volume (incompressible liquid).
    Parameters:
        mass_g: mass of ethanol in grams
        density_g_per_cm3: density of ethanol in g/cm^3
        molar_mass_g_per_mol: molar mass of ethanol in g/mol
        p1_atm: initial pressure in atm
        p2_atm: final pressure in atm
    Returns:
        Delta G in Joules
    """
    # Constants
    atm_to_Pa = 1.01325e5  # 1 atm in Pascals

    # Convert mass to moles
    n_moles = mass_g / molar_mass_g_per_mol

    # Convert density to kg/m^3
    density_kg_per_m3 = density_g_per_cm3 * 1000  # 0.789 g/cm3 = 789 kg/m3

    # Calculate volume of ethanol in m^3
    volume_m3 = (mass_g / 1000) / density_kg_per_m3  # mass in kg / density

    # Calculate molar volume (volume per mole) in m^3/mol
    molar_volume_m3_per_mol = volume_m3 / n_moles

    # Convert pressures to Pascals
    p1_Pa = p1_atm * atm_to_Pa
    p2_Pa = p2_atm * atm_to_Pa

    delta_p = p2_Pa - p1_Pa

    # Calculate change in Gibbs free energy (Joules)
    delta_G = n_moles * molar_volume_m3_per_mol * delta_p

    return delta_G

# Calculate and print the change in Gibbs energy
result = calculate_delta_gibbs_energy()
print(f"Change in Gibbs free energy (Delta G) for 35 g ethanol from 1 atm to 3000 atm: {result:.2f} J")
