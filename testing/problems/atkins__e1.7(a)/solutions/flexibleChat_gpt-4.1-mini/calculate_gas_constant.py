# filename: calculate_gas_constant.py

def calculate_gas_constant():
    # Given data
    V_dm3 = 20.000  # volume in dm^3
    mass_helium_g = 0.25132  # mass in grams
    T_gas_C = 500  # temperature of gas in Celsius
    h_cm = 206.402  # height of water column in cm
    T_water_C = 25  # temperature of water in Celsius
    density_water_g_cm3 = 0.99707  # density of water in g/cm^3
    molar_mass_helium_g_mol = 4.0026  # molar mass of helium in g/mol
    g = 9.81  # acceleration due to gravity in m/s^2
    P_atm_Pa = 101325  # standard atmospheric pressure in Pa

    # Unit conversions
    V_m3 = V_dm3 * 1e-3  # dm^3 to m^3
    T_gas_K = T_gas_C + 273.15  # Celsius to Kelvin
    h_m = h_cm * 1e-2  # cm to m
    density_water_kg_m3 = density_water_g_cm3 * 1000  # g/cm^3 to kg/m^3

    # Calculate pressure inside the container
    P_gas_Pa = P_atm_Pa + density_water_kg_m3 * g * h_m

    # Calculate number of moles of helium
    n_mol = mass_helium_g / molar_mass_helium_g_mol

    # Calculate gas constant R
    R = (P_gas_Pa * V_m3) / (n_mol * T_gas_K)

    return R

if __name__ == "__main__":
    R_value = calculate_gas_constant()
    print(f"Calculated gas constant R = {R_value:.4f} J/(mol K)")
