import math

# Constants
m_e = 9.10938356e-31  # Electron mass in kg

e = 1.602176634e-19    # Elementary charge in C

epsilon_0 = 8.854187817e-12  # Vacuum permittivity in F/m

hbar = 1.054571817e-34  # Reduced Planck's constant in J s

def calculate_hartree_energy():
    # Calculate e^4
    e4 = e ** 4
    # Calculate numerator: m_e * e^4
    numerator = m_e * e4
    # Calculate denominator: 16 * pi^2 * epsilon_0^2 * hbar^2
    denominator = 16 * math.pi ** 2 * (epsilon_0 ** 2) * (hbar ** 2)
    # Calculate Hartree energy in Joules
    E_h = numerator / denominator
    # Convert to eV
    E_h_eV = E_h / (1.602176634e-19)  # Conversion factor from J to eV
    return E_h, E_h_eV

# Execute the calculation
E_h, E_h_eV = calculate_hartree_energy()

# Print results with formatting
print(f"Hartree energy (E_h) in Joules: {E_h:.5e}")
print(f"Hartree energy (E_h) in eV: {E_h_eV:.5f}")