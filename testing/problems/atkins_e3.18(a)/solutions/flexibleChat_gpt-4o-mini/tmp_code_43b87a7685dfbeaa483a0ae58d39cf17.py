import math

# Constants
R = 8.314  # J/(mol·K)
T = 298.15  # K
P_i = 1  # atm
P_f = 3000  # atm
molar_mass_ethanol = 46.07  # g/mol

def calculate_delta_G(mass_ethanol_g):
    # Step 1: Calculate the number of moles of ethanol
    n = mass_ethanol_g / molar_mass_ethanol
    
    # Step 2: Calculate the change in Gibbs energy
    if P_i <= 0:
        raise ValueError('Initial pressure P_i must be greater than zero.')  
    Delta_G = n * R * T * math.log(P_f / P_i)
    return Delta_G

# Calculate and print the change in Gibbs energy for 35 grams of ethanol
mass_ethanol = 35  # grams
Delta_G = calculate_delta_G(mass_ethanol)
print(f'Change in Gibbs energy (Delta G): {Delta_G:.2f} J')