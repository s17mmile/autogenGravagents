D = 7.31e-19  # J per molecule
beta = 1.81e10  # m^-1

# Calculate beta squared
beta_squared = beta ** 2

# Calculate the force constant k
k = D * beta_squared

# Output the result
print(f'The force constant k for HCl is approximately {k:.2f} N/m')