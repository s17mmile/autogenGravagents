# filename: calculate_molecular_weight_function.py

def calculate_molecular_weight(osmotic_pressure_Pa, mass_concentration_kg_per_m3, temperature_K, R=8.314):
    """Calculate molecular weight in g/mol from osmotic pressure data."""
    # Basic input validation
    if osmotic_pressure_Pa <= 0:
        raise ValueError("Osmotic pressure must be positive.")
    if mass_concentration_kg_per_m3 <= 0:
        raise ValueError("Mass concentration must be positive.")
    if temperature_K <= 0:
        raise ValueError("Temperature must be positive.")

    molecular_weight_kg_per_mol = (mass_concentration_kg_per_m3 * R * temperature_K) / osmotic_pressure_Pa
    molecular_weight_g_per_mol = molecular_weight_kg_per_mol * 1000
    return molecular_weight_g_per_mol


# Given constants
osmotic_pressure_Pa = 5.30e4  # Pa
mass_concentration_kg_per_m3 = 31.2  # kg/m^3
temperature_K = 298  # K

molecular_weight = calculate_molecular_weight(osmotic_pressure_Pa, mass_concentration_kg_per_m3, temperature_K)
print(f"Molecular weight of the substance: {molecular_weight:.2f} g/mol")
