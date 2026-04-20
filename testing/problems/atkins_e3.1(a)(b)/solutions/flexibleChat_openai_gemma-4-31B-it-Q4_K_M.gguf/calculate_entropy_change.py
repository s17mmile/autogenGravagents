# filename: calculate_entropy_change.py

# Calculate entropy change for reversible isothermal heat transfer
# Given: q_rev = 25,000 J, T = 373.15 K (100°C)

q_rev = 25000  # Reversible heat transfer in joules
T = 373.15     # Temperature in Kelvin

# Validate inputs
if T <= 0:
    raise ValueError("Temperature must be greater than 0 K.")

# Compute entropy change
delta_S = q_rev / T

# Output result with units
print(f'Change in entropy: {delta_S:.4f} J/K')