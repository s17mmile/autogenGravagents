# filename: calculate_vdw_pressure.py

def calculate_vdw_pressure(mass_kg, volume_m3, temperature_K, a, b, molar_mass_kg_per_mol=0.02802, R=0.08206):
    """
    Calculate pressure of nitrogen gas using van der Waals equation.
    Parameters:
        mass_kg (float): mass of nitrogen in kilograms
        volume_m3 (float): volume in cubic meters
        temperature_K (float): temperature in Kelvin
        a (float): van der Waals constant a in dm^6 atm mol^-2
        b (float): van der Waals constant b in dm^3 mol^-1
        molar_mass_kg_per_mol (float): molar mass of nitrogen in kg/mol (default 0.02802)
        R (float): ideal gas constant in atm dm^3/(mol K) (default 0.08206)
    Returns:
        float: pressure in atm
    """
    if mass_kg <= 0:
        raise ValueError("Mass must be positive.")
    if volume_m3 <= 0:
        raise ValueError("Volume must be positive.")
    if temperature_K <= 0:
        raise ValueError("Temperature must be positive.")

    volume_dm3 = volume_m3 * 1000  # convert m^3 to dm^3
    n = mass_kg / molar_mass_kg_per_mol  # number of moles

    denominator = volume_dm3 - n * b
    if denominator <= 0:
        raise ValueError("Denominator in pressure calculation is non-positive, check inputs.")

    pressure_atm = (n * R * temperature_K) / denominator - (a * n**2) / (volume_dm3**2)
    return pressure_atm


# Given data
mass_kg = 92.4
volume_m3 = 1.000
temperature_K = 500

a = 1.39  # dm^6 atm mol^-2
b = 0.0391  # dm^3 mol^-1

pressure = calculate_vdw_pressure(mass_kg, volume_m3, temperature_K, a, b)
print(f"Approximate pressure of nitrogen at {temperature_K} K: {pressure:.2f} atm")
