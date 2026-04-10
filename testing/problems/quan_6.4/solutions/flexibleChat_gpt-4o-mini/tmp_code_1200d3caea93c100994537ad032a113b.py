import math

# Constants
h = 6.626e-34  # Planck's constant in J*s
MHZ_TO_HZ = 1e6  # Conversion factor from MHz to Hz
PM_TO_M = 1e12  # Conversion factor from meters to picometers


def calculate_bond_length(frequency_mhz_input):
    if frequency_mhz_input <= 0:
        raise ValueError('Frequency must be a positive value.')  # Error handling for invalid frequency

    # Convert frequency to Hz
    frequency_hz = frequency_mhz_input * MHZ_TO_HZ  # frequency in Hz

    # Atomic masses in kg
    m_C = 12 * 1.66053906660e-27  # Carbon mass in kg
    m_S = 32 * 1.66053906660e-27  # Sulfur mass in kg

    # Calculate reduced mass (mu)
    mu = (m_C * m_S) / (m_C + m_S)

    # Calculate rotational constant B
    B = frequency_hz / 2  # in Hz

    # Calculate bond distance r
    r_squared = h / (8 * math.pi**2 * mu * B)

    # Calculate bond length in meters
    r = math.sqrt(r_squared)

    # Convert bond length to picometers
    bond_length_pm = r * PM_TO_M
    return bond_length_pm

# Example usage
frequency_mhz = 48991.0  # frequency in MHz
bond_length_pm = calculate_bond_length(frequency_mhz)
print(f'Bond length: {bond_length_pm} pm')