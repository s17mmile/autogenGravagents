import math

# Constants
sigma_nm2 = 0.43  # collisional cross section in nm^2
sigma_m2 = sigma_nm2 * 1e-18  # convert to m^2
k = 1.38e-23  # Boltzmann constant in J/K
T = 298  # temperature in K
molar_mass_N2_g_per_mol = 28  # molar mass of N2 in g/mol
molar_mass_N2_kg_per_mol = molar_mass_N2_g_per_mol / 1000  # convert to kg/mol
m = molar_mass_N2_kg_per_mol / 6.022e23  # mass of one N2 molecule in kg
P_atm = 1  # pressure in atm
P_Pa = P_atm * 101325  # convert pressure to Pascals

# Calculate average molecular speed v
v = math.sqrt((8 * k * T) / (math.pi * m))

# Calculate diffusion coefficient D
D = (1/3) * (sigma_m2 * v) / P_Pa

# Output the diffusion coefficient
print(f'Diffusion Coefficient D: {D} m^2/s')