# filename: calculate_original_pressure.py

# Given data
volume_reduction_dm3 = 2.20  # Volume reduction in dm^3
final_volume_dm3 = 4.65       # Final volume in dm^3
final_pressure_bar = 5.04    # Final pressure in bar

# Calculate initial volume (before compression)
initial_volume_dm3 = final_volume_dm3 + volume_reduction_dm3

# Calculate initial pressure in bar using Boyle's Law (P1 * V1 = P2 * V2)
initial_pressure_bar = (final_pressure_bar * final_volume_dm3) / initial_volume_dm3

# Convert pressure from bar to atm (1 atm = 1.01325 bar)
initial_pressure_atm = initial_pressure_bar / 1.01325

# Output the result
print(f"Original pressure of the gas: {initial_pressure_atm:.4f} atm")
