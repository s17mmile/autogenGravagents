# filename: calculate_gibbs_energy_cl2.py
import math

# Constants
h = 6.626e-34  # Planck constant, J*s
c = 2.998e10   # Speed of light, cm/s
kB = 1.381e-23 # Boltzmann constant, J/K
R = 8.314      # Gas constant, J/(mol*K)
NA = 6.022e23  # Avogadro's number
T = 298.15     # Temperature, K
p = 1e5        # Pressure, Pa (1 bar)

# Given data
vib_wavenumber = 560.0  # cm^-1
B = 0.244               # cm^-1
sigma = 2               # Symmetry number for Cl2
molecular_mass_u = 2 * 34.97  # atomic mass units

# Convert molecular mass to kg
molecular_mass = molecular_mass_u * 1.6605e-27  # kg

# Calculate characteristic temperatures
theta_rot = (h * c * B) / kB
theta_vib = (h * c * vib_wavenumber) / kB

# Rotational partition function
q_rot = T / (sigma * theta_rot)

# Vibrational partition function
q_vib = 1.0 / (1.0 - math.exp(-theta_vib / T))

# Electronic partition function (nondegenerate ground state)
q_elec = 1.0

# Translational partition function per molecule
q_trans = ((2 * math.pi * molecular_mass * kB * T) / (h ** 2)) ** (3/2) * (kB * T) / p

# Total molecular partition function
Q = q_trans * q_rot * q_vib * q_elec

# Calculate molar Gibbs energy
G_molar = -R * T * math.log(Q) + R * T * math.log(NA)

print(f"Molar standard Gibbs energy for 35Cl35Cl at {T} K and 1 bar: {G_molar:.2f} J/mol")
