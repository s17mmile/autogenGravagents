# filename: expected_height_methane.py

def expected_height_methane(T=298):
    """
    Calculate the expected height <h> for methane molecules in the atmosphere
    using the distribution function P(h) = exp(-mgh/kT).

    Parameters:
    T (float): Temperature in Kelvin (must be positive).

    Returns:
    float: Expected height in meters.
    """
    if T <= 0:
        raise ValueError("Temperature must be positive.")

    # Constants with higher precision
    k = 1.380649e-23  # Boltzmann constant in J/K
    g = 9.81          # Acceleration due to gravity in m/s^2
    molar_mass_ch4 = 16.04e-3  # kg/mol
    N_A = 6.02214076e23        # Avogadro's number in mol^-1

    # Calculate mass of one methane molecule
    m = molar_mass_ch4 / N_A

    # Calculate expected height <h>
    h_avg = (k * T) / (m * g)
    return h_avg

# Example usage
if __name__ == '__main__':
    temperature = 298  # Room temperature in Kelvin
    height = expected_height_methane(temperature)
    print(f'Expected height <h> for methane at {temperature} K: {height:.2f} meters')
