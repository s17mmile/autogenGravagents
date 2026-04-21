# filename: collisional_cross_section_ratio.py

# Given data
k_C2H2 = 0.01866  # Thermal conductivity of acetylene in J K^-1 m^-1 s^-1
k_N2 = 0.0240     # Thermal conductivity of nitrogen in J K^-1 m^-1 s^-1

# Molar masses in g/mol
M_C2H2 = 2 * 12.01 + 2 * 1.008  # Acetylene
M_N2 = 2 * 14.01                # Nitrogen

# Calculate the ratio of collisional cross sections
import math

ratio_sigma = math.sqrt(M_N2 / M_C2H2) * (k_N2 / k_C2H2)

print(f"Ratio of collisional cross section of acetylene relative to nitrogen: {ratio_sigma:.4f}")

# The output is the ratio sigma_C2H2 / sigma_N2