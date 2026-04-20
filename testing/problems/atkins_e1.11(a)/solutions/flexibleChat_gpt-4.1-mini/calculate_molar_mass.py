# filename: calculate_molar_mass.py

def calculate_molar_mass(density, temperature, pressure_kpa):
    """
    Calculate the molar mass of a gaseous compound using the ideal gas law.

    Parameters:
    density (float): Density of the gas in kg/m^3
    temperature (float): Temperature in Kelvin
    pressure_kpa (float): Pressure in kilopascals

    Returns:
    float: Molar mass in grams per mole
    """
    # Validate inputs
    if density <= 0 or temperature <= 0 or pressure_kpa <= 0:
        raise ValueError("Density, temperature, and pressure must be positive values.")

    R = 8.314  # J/(mol*K), universal gas constant
    pressure_pa = pressure_kpa * 1000  # Convert kPa to Pa

    # Using ideal gas law: P = (density * R * T) / M => M = (density * R * T) / P
    molar_mass_kg_per_mol = (density * R * temperature) / pressure_pa
    molar_mass_g_per_mol = molar_mass_kg_per_mol * 1000  # Convert kg/mol to g/mol

    return molar_mass_g_per_mol

# Given values
density = 1.23  # kg/m^3
temperature = 330  # K
pressure_kpa = 20  # kPa

molar_mass = calculate_molar_mass(density, temperature, pressure_kpa)

# Save the result to a text file with error handling
try:
    with open('molar_mass_result.txt', 'w') as f:
        f.write(f'Molar mass of the compound: {molar_mass:.2f} g/mol\n')
except IOError as e:
    print(f'Error writing to file: {e}')
