# filename: collisional_frequency_co2.py
import math

# Constants
P = 101325  # Pressure in Pa (1 atm)
T = 298  # Temperature in K
k_B = 1.380649e-23  # Boltzmann constant in J/K

# Molecular properties of CO2
molecular_mass_CO2 = 44.01e-3  # kg/mol
N_A = 6.02214076e23  # Avogadro's number
m = molecular_mass_CO2 / N_A  # mass of one CO2 molecule in kg

# Approximate molecular diameter of CO2 (in meters)
# Literature values suggest about 3.3 angstroms (3.3e-10 m)
molecular_diameter = 3.3e-10

# Calculate number density n (molecules per m^3)
n = P / (k_B * T)

# Calculate collision cross-section sigma (m^2)
sigma = math.pi * (molecular_diameter ** 2)

# Calculate average relative speed v_avg (m/s)
v_avg = math.sqrt(8 * k_B * T / (math.pi * m))

# Calculate collisional frequency Z (s^-1)
Z = n * sigma * v_avg

print(f"Total collisional frequency for CO2 at 1 atm and 298 K: {Z:.2e} s^-1")
