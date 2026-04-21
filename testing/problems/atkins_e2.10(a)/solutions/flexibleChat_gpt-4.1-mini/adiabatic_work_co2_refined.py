# filename: adiabatic_work_co2_refined.py

def calculate_adiabatic_work_CO2(mass_g=2.45, T1_C=27.0, V1_cm3=500, V2_dm3=3.00, gamma=1.30):
    """
    Calculate the work done by CO2 gas during a reversible adiabatic expansion.

    Parameters:
    - mass_g: mass of CO2 in grams
    - T1_C: initial temperature in Celsius
    - V1_cm3: initial volume in cubic centimeters
    - V2_dm3: final volume in cubic decimeters
    - gamma: heat capacity ratio (Cp/Cv) for CO2

    Returns:
    - Work done in Joules
    """
    R = 8.314  # J/(mol*K), universal gas constant
    M_CO2 = 44.01  # g/mol, molar mass of CO2

    # Convert temperatures and volumes to SI units
    T1 = T1_C + 273.15  # Convert Celsius to Kelvin
    V1_dm3 = V1_cm3 / 1000  # Convert cm^3 to dm^3 (1 dm^3 = 1000 cm^3)

    # Convert volumes from dm^3 to m^3 for SI consistency
    V1_m3 = V1_dm3 * 1e-3  # m^3
    V2_m3 = V2_dm3 * 1e-3  # m^3

    # Calculate number of moles
    n = mass_g / M_CO2  # mol

    # Calculate final temperature using adiabatic relation
    T2 = T1 * (V1_m3 / V2_m3)**(gamma - 1)

    # Calculate initial and final pressures using ideal gas law
    P1 = n * R * T1 / V1_m3  # Pa
    P2 = n * R * T2 / V2_m3  # Pa

    # Calculate work done during reversible adiabatic expansion
    W = (P2 * V2_m3 - P1 * V1_m3) / (1 - gamma)  # Joules

    return W


if __name__ == "__main__":
    work_done = calculate_adiabatic_work_CO2()
    print(f"Work done by the CO2 gas during reversible adiabatic expansion: {work_done:.2f} J")
