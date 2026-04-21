# filename: adiabatic_expansion_pressure.py

# Given data
P1 = 57.4  # initial pressure in kPa
V1 = 1.0   # initial volume in dm^3
V2 = 2.0   # final volume in dm^3
gamma = 1.4  # heat capacity ratio

# Calculate final pressure using adiabatic relation
P2 = P1 * (V1 / V2) ** gamma

# Output the result
print(f"Final pressure after adiabatic expansion: {P2:.2f} kPa")
