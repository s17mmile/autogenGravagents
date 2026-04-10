import math

# Constants
tilde_omega_obs = 2886 * 100  # converting cm^-1 to m^-1
D = 440.2 * 1000  # converting kJ/mol to J/mol

# Define masses in kg (approximate values)
m_H = 1.00784 / 1000  # mass of Hydrogen in kg
m_Cl = 34.96885 / 1000  # mass of Chlorine in kg

# Calculate reduced mass (mu)
mu = (m_H * m_Cl) / (m_H + m_Cl)

# Speed of light in m/s
c = 3 * 10**8

# Calculate beta
beta = 2 * math.pi * c * (tilde_omega_obs)**0.5 * (mu / (2 * D))**0.5

# Output the result with units
print(f'Beta: {beta} m/s')