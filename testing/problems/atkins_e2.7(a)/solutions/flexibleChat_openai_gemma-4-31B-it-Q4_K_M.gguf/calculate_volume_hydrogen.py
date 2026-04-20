# filename: calculate_volume_hydrogen.py

# Given values
moles_h2 = 0.617157  # mol
pressure_atm = 1.0  # atm
temperature_k = 298.15  # K
r = 0.08206  # L·atm·mol⁻¹·K⁻¹

# Calculate volume using ideal gas law: V = nRT / P
volume_l = (moles_h2 * r * temperature_k) / pressure_atm

# Output the result with 6 decimal places
print(f"Volume of hydrogen gas: {volume_l:.6f} L")