# filename: tin_phase_equilibrium.py

def calculate_equilibrium_temperature(P1_bar, T1_C, P2_bar, rho_gray, rho_white, S_gray, S_white, molar_mass):
    """Calculate equilibrium temperature between two phases at a new pressure.

    Parameters:
    P1_bar (float): Initial pressure in bar
    T1_C (float): Initial temperature in Celsius
    P2_bar (float): New pressure in bar
    rho_gray (float): Density of gray tin in kg/m^3
    rho_white (float): Density of white tin in kg/m^3
    S_gray (float): Molar entropy of gray tin in J/(K mol)
    S_white (float): Molar entropy of white tin in J/(K mol)
    molar_mass (float): Molar mass of tin in kg/mol

    Returns:
    float: Equilibrium temperature at P2 in Celsius
    """
    # Convert initial temperature to Kelvin
    T1_K = T1_C + 273.15

    # Convert pressures from bar to Pa
    P1_Pa = P1_bar * 1e5
    P2_Pa = P2_bar * 1e5

    # Calculate molar volumes (m^3/mol)
    V_gray = molar_mass / rho_gray
    V_white = molar_mass / rho_white

    # Calculate changes in volume and entropy
    delta_V = V_white - V_gray  # m^3/mol
    delta_S = S_white - S_gray  # J/(K mol)

    if delta_S == 0:
        raise ValueError("Change in entropy (delta_S) cannot be zero.")

    # Calculate new equilibrium temperature using integrated Clapeyron equation
    T2_K = T1_K + (delta_V / delta_S) * (P2_Pa - P1_Pa)

    # Convert back to Celsius
    T2_C = T2_K - 273.15
    return T2_C

# Given data
P1 = 1  # bar
T1 = 18  # Celsius
P2 = 350  # bar
rho_gray = 5750  # kg/m^3
rho_white = 7280  # kg/m^3
S_gray = 44.14  # J/(K mol)
S_white = 51.18  # J/(K mol)
molar_mass = 0.11871  # kg/mol

# Calculate equilibrium temperature at 350 bar
T2 = calculate_equilibrium_temperature(P1, T1, P2, rho_gray, rho_white, S_gray, S_white, molar_mass)

print(f"Equilibrium temperature at {P2} bar: {T2:.2f} degrees Celsius")
