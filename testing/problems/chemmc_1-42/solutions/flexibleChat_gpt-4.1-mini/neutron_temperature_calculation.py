# filename: neutron_temperature_calculation.py

from scipy.constants import h, m_n, k

def calculate_neutron_temperature():
    """Calculate the temperature required for neutrons to have a de Broglie wavelength of 50 pm."""
    # Constants:
    # h: Planck's constant (Joule second)
    # m_n: mass of neutron (kg)
    # k: Boltzmann's constant (Joule per Kelvin)
    lambda_m = 50e-12  # de Broglie wavelength in meters (50 pm)

    # Calculate temperature T using the formula T = h^2 / (3 * m * k * lambda^2)
    T = h**2 / (3 * m_n * k * lambda_m**2)
    return T

if __name__ == '__main__':
    temperature = calculate_neutron_temperature()
    print(f'Temperature required for neutron wavelength of 50 pm: {temperature:.2f} K')
    # Simple assertion test: temperature should be positive and reasonable
    assert 0 < temperature < 1e6, "Calculated temperature is out of expected range."