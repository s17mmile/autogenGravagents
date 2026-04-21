# filename: mole_percent_butane_v2.py

def calculate_mole_percent_butane(V_cm3, P_Pa, T_C, mass_g):
    """
    Calculate the mole percent of butane in a gas mixture of ethane and butane.

    Parameters:
    V_cm3 : float : volume of the gas in cm^3
    P_Pa : float : pressure in Pascals
    T_C : float : temperature in Celsius
    mass_g : float : mass of the gas in grams

    Returns:
    float : mole percent of butane in the mixture
    """
    # Constants
    R = 8.314  # J/(mol K)

    # Validate inputs
    if V_cm3 <= 0 or P_Pa <= 0 or mass_g <= 0:
        raise ValueError("Volume, pressure, and mass must be positive values.")

    # Convert units
    V_m3 = V_cm3 * 1e-6  # cm^3 to m^3
    T_K = T_C + 273.15   # Celsius to Kelvin

    # Molar masses in g/mol
    M_ethane = 2*12.01 + 6*1.008  # 30.07 g/mol
    M_butane = 4*12.01 + 10*1.008  # 58.12 g/mol

    # Calculate total moles using ideal gas law
    n_total = (P_Pa * V_m3) / (R * T_K)

    # Calculate mole fraction of butane (x)
    # Derived from mass = n_total * [M_ethane*(1 - x) + M_butane*x]
    x = (mass_g / n_total - M_ethane) / (M_butane - M_ethane)

    # Clamp mole fraction to [0,1] to ensure physical plausibility
    x = max(0.0, min(1.0, x))

    # Convert mole fraction to mole percent
    mole_percent_butane = x * 100

    return mole_percent_butane

# Example usage with given data
if __name__ == "__main__":
    V_cm3 = 230.0
    P_Pa = 97.5e3
    T_C = 23.1
    mass_g = 0.3554

    mole_percent = calculate_mole_percent_butane(V_cm3, P_Pa, T_C, mass_g)
    print(f'Mole percent of butane in the mixture: {mole_percent:.2f}%')
