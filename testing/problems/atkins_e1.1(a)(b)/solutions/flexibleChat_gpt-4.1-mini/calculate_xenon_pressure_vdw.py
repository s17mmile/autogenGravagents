# filename: calculate_xenon_pressure_vdw.py

def calculate_vdw_pressure(mass_g, molar_mass, volume_L, temperature_C, a, b, R=0.082057):
    """Calculate pressure using the van der Waals equation."""
    temperature_K = temperature_C + 273.15
    n = mass_g / molar_mass
    denominator = volume_L - n * b
    if denominator <= 0:
        raise ValueError("Volume minus n*b must be positive to avoid division by zero or negative volume.")
    pressure = (n * R * temperature_K) / denominator - a * (n / volume_L)**2
    return pressure

# Given values
mass_xenon = 131.0  # g
molar_mass_xenon = 131.29  # g/mol
volume = 1.0  # L
temperature = 25  # Celsius

# van der Waals constants for xenon
vdw_a = 4.19  # L^2 atm / mol^2
vdw_b = 0.0510  # L / mol

pressure_atm = calculate_vdw_pressure(mass_xenon, molar_mass_xenon, volume, temperature, vdw_a, vdw_b)
print(f"Pressure exerted by xenon gas: {pressure_atm:.3f} atm")
