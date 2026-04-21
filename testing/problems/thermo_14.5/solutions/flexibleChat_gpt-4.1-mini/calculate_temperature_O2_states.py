# filename: calculate_temperature_O2_states.py
import scipy.constants as const

# Given values
V_cm3 = 1000  # volume in cubic centimeters
V_m3 = V_cm3 * 1e-6  # convert volume to cubic meters
N_A = const.N_A  # Avogadro's number

# Molar mass of O2 in kg/mol (using precise value)
M_O2 = 31.9988e-3

# Mass of one O2 molecule in kg
m = M_O2 / N_A

# Calculate temperature T using the formula derived from thermal de Broglie wavelength
T = (const.h**2 / (2 * const.pi * m * const.k)) * (N_A / V_m3)**(2/3)

print(f"Temperature at which there are Avogadro's number of translational states for O2 in {V_cm3} cm^3: {T:.2f} K")
