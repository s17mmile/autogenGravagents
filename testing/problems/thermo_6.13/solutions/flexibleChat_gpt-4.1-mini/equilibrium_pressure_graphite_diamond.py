# filename: equilibrium_pressure_graphite_diamond.py

def calculate_equilibrium_pressure():
    """Calculate the equilibrium pressure at which graphite and diamond are in equilibrium at 298.15 K."""
    # Given constants
    delta_G0_kJ_per_mol = 2.90  # kJ/mol
    delta_G0 = delta_G0_kJ_per_mol * 1000  # convert to J/mol
    P0 = 1e5  # standard pressure in Pa (1 bar)
    molar_mass_C = 0.01201  # kg/mol
    density_graphite = 2.25  # kg/L
    density_diamond = 3.52  # kg/L

    # Convert densities from kg/L to kg/m^3
    density_graphite_m3 = density_graphite * 1000  # kg/m^3
    density_diamond_m3 = density_diamond * 1000  # kg/m^3

    # Calculate molar volumes in m^3/mol
    V_graphite = molar_mass_C / density_graphite_m3
    V_diamond = molar_mass_C / density_diamond_m3

    # Calculate difference in molar volume
    delta_V = V_diamond - V_graphite  # m^3/mol

    # Calculate equilibrium pressure
    # At equilibrium: 0 = delta_G0 + delta_V * (P - P0)
    # => P = P0 - delta_G0 / delta_V
    P_eq = P0 - delta_G0 / delta_V  # Pa

    # Convert pressure to GPa
    P_eq_GPa = P_eq / 1e9

    return P_eq, P_eq_GPa

if __name__ == '__main__':
    P_eq, P_eq_GPa = calculate_equilibrium_pressure()
    print(f'Equilibrium pressure at 298.15 K: {P_eq:.2e} Pa ({P_eq_GPa:.2f} GPa)')
