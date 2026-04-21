# filename: calculate_volume_change.py

# Given data
x_ethanol = 0.35  # mole fraction of ethanol
V_solution = 2.50  # volume of solution in liters

# Partial molar volumes in L/mol
Vbar_ethanol = 55.2e-3
Vbar_water = 17.8e-3

# Densities in kg/m^3 (1 kg/m^3 = 1 g/L)
rho_ethanol_g_per_L = 789  # ethanol density in g/L
rho_water_g_per_L = 997    # water density in g/L

# Molar masses in g/mol
M_ethanol = 46.07
M_water = 18.015

# Step 1: Calculate total moles in solution
# Let n_total be total moles, then moles ethanol = x_ethanol * n_total
# and moles water = (1 - x_ethanol) * n_total

# Step 2: Use partial molar volumes to find total volume in terms of n_total:
# V_solution = n_ethanol * Vbar_ethanol + n_water * Vbar_water
# V_solution = n_total * (x_ethanol * Vbar_ethanol + (1 - x_ethanol) * Vbar_water)

Vbar_mixture = x_ethanol * Vbar_ethanol + (1 - x_ethanol) * Vbar_water
n_total = V_solution / Vbar_mixture

n_ethanol = x_ethanol * n_total
n_water = (1 - x_ethanol) * n_total

# Step 3: Calculate volume of pure ethanol and water corresponding to these moles
# mass = moles * molar mass (g)
# volume = mass / density (L)

mass_ethanol = n_ethanol * M_ethanol  # in g
mass_water = n_water * M_water          # in g

V_ethanol_pure = mass_ethanol / rho_ethanol_g_per_L  # in L
V_water_pure = mass_water / rho_water_g_per_L        # in L

# Step 4: Calculate volume change
# volume change = solution volume - sum of pure component volumes

volume_change = V_solution - (V_ethanol_pure + V_water_pure)

# Output results
print(f"Total moles in solution: {n_total:.4f} mol")
print(f"Moles ethanol: {n_ethanol:.4f} mol")
print(f"Moles water: {n_water:.4f} mol")
print(f"Volume of pure ethanol: {V_ethanol_pure:.4f} L")
print(f"Volume of pure water: {V_water_pure:.4f} L")
print(f"Volume change on mixing: {volume_change:.4f} L")

# The volume change indicates contraction (negative) or expansion (positive) relative to pure components.