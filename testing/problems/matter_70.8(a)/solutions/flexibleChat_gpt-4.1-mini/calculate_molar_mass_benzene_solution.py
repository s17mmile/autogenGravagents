# filename: calculate_molar_mass_benzene_solution.py

# Given data
P0 = 53.3  # vapor pressure of pure benzene in kPa
P = 51.5   # vapor pressure of solution in kPa
m_solute = 19.0  # mass of solute in grams
m_benzene = 500.0  # mass of benzene in grams
M_benzene = 78.11  # molar mass of benzene in g/mol

# Calculate mole fraction of benzene in solution
X_benzene = P / P0

# Calculate moles of benzene
n_benzene = m_benzene / M_benzene

# Calculate moles of solute using mole fraction formula
n_solute = n_benzene * (1 - X_benzene) / X_benzene

# Calculate molar mass of solute
M_solute = m_solute / n_solute

# Output the result
print(f"Molar mass of the solute: {M_solute:.2f} g/mol")
