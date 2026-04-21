# filename: calculate_work_mg_hcl.py

def calculate_work_done_by_system(mass_mg_g=15.0, temperature_c=25.0, pressure_atm=1.0):
    """
    Calculate the work done by the system during the reaction of magnesium with dilute hydrochloric acid.

    Parameters:
    mass_mg_g (float): Mass of magnesium in grams.
    temperature_c (float): Temperature in degrees Celsius.
    pressure_atm (float): Atmospheric pressure in atm.

    Returns:
    float: Work done by the system in joules (negative value indicates work done by the system).
    """
    # Constants
    molar_mass_mg = 24.305  # g/mol
    R = 0.08206  # L atm / mol K (ideal gas constant)
    temperature_k = temperature_c + 273.15  # Convert to Kelvin

    # Step 1: Balanced chemical equation:
    # Mg (s) + 2 HCl (aq) -> MgCl2 (aq) + H2 (g)

    # Step 2: Calculate moles of Mg
    moles_mg = mass_mg_g / molar_mass_mg

    # Step 3: Moles of H2 gas produced (1 mole Mg produces 1 mole H2)
    moles_h2 = moles_mg

    # Step 4: Calculate volume of H2 gas at given T and P using ideal gas law: PV = nRT => V = nRT/P
    volume_h2_liters = (moles_h2 * R * temperature_k) / pressure_atm  # in liters

    # Step 5: Calculate work done by the system: W = -P * deltaV
    # Convert pressure to atm and volume to liters, then convert work to joules
    # 1 L atm = 101.325 J
    work_joules = -pressure_atm * volume_h2_liters * 101.325  # work done by system in joules

    return work_joules

# Example usage
work_done = calculate_work_done_by_system()
print(f"Work done by the system during the reaction: {work_done:.2f} J")
