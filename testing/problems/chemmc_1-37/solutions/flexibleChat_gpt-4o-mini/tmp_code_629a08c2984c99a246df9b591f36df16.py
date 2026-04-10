import numpy as np

# Constants
m_p = 938 * 1.6e-13  # Mass of proton in Joules (MeV to Joules)
me = 0.511 * 1.6e-13  # Mass of electron in Joules (MeV to Joules)
muon_mass = 207 * me  # Mass of muon in Joules
Z = 1  # Atomic number for proton
epsilon_0 = 8.854e-12  # Vacuum permittivity in F/m
h = 6.626e-34  # Planck's constant in Js

# Step 1: Calculate the reduced mass
mu = (m_p * muon_mass) / (m_p + muon_mass)

# Step 2: Calculate energy levels E1 and E2
E1 = - (Z**2 * mu * (1.6e-19)**4) / (8 * epsilon_0**2 * h**2)
E2 = - (Z**2 * mu * (1.6e-19)**4) / (8 * epsilon_0**2 * h**2 * 4)  # Corrected calculation for E2

# Step 3: Calculate the energy difference
Delta_E = E2 - E1

# Step 4: Calculate the frequency
frequency = Delta_E / h

# Output the frequency with improved formatting
print(f'The frequency associated with the transition from n=1 to n=2 is: {frequency:.2e} Hz')