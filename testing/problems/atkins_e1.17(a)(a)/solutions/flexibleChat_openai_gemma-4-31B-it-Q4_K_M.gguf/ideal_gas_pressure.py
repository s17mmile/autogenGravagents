# filename: ideal_gas_pressure.py

# Given values
temperature_celsius = 27
volume_dm3 = 4.860  # dm³ = L
moles = 10.0

# Convert temperature to Kelvin
temperature_k = temperature_celsius + 273.15

# Ideal gas constant in L·atm·mol⁻¹·K⁻¹
R = 0.08206

# Calculate pressure using ideal gas law: P = nRT / V
pressure_atm = (moles * R * temperature_k) / volume_dm3

# Output the result with appropriate precision
print(f"Pressure exerted by ethane gas: {pressure_atm:.4f} atm")