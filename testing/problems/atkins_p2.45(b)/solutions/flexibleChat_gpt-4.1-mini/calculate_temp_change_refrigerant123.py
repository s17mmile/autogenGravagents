# filename: calculate_temp_change_refrigerant123.py

# Estimated Joule-Thomson coefficient (mu) for refrigerant 123 at 50 C and ~1 bar
# Units: K/Pa (Kelvin per Pascal), assumed constant over pressure range
mu_estimated = 0.0005  # Typical small positive value for refrigerants

# Initial and final pressures in bar
P1_bar = 1.5
P2_bar = 0.5

# Amount of substance (moles) - included for reference, does not affect temperature change calculation
n_mol = 2.0

# Convert pressures from bar to pascal (1 bar = 1e5 Pa)
P1 = P1_bar * 1e5  # Pa
P2 = P2_bar * 1e5  # Pa

# Calculate temperature change using Delta T = mu * (P2 - P1)
# mu is an intensive property, so temperature change does not scale with amount of substance
Delta_T = mu_estimated * (P2 - P1)  # K

# Output the estimated temperature change
print(f"Estimated temperature change during adiabatic expansion of {n_mol} mol refrigerant 123 from {P1_bar} bar to {P2_bar} bar at 50 C is {Delta_T:.4f} K.")

# Return the temperature change value for further use if needed
Delta_T
