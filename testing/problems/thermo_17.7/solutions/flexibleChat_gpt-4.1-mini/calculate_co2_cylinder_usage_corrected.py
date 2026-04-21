# filename: calculate_co2_cylinder_usage_corrected.py
import math

def calculate_co2_cylinder_usage():
    # Given data
    mass_co2_kg = 22.7  # mass of CO2 in cylinder in kg
    T = 293  # temperature in K
    eta_microPoise = 146  # dynamic viscosity in microPoise
    # Convert microPoise to Poise: 1 microPoise = 1e-6 Poise
    # Convert Poise to Pa.s: 1 Poise = 0.1 Pa.s
    eta = eta_microPoise * 1e-6 * 0.1  # dynamic viscosity in Pa.s
    L = 1.00  # length of tube in meters
    d_mm = 0.75  # diameter in mm
    r = (d_mm / 2) * 1e-3  # radius in meters
    P_in_atm = 1.05  # input pressure in atm
    P_out_atm = 1.00  # output pressure in atm

    # Convert pressures from atm to Pa (1 atm = 101325 Pa)
    P_in = P_in_atm * 101325
    P_out = P_out_atm * 101325
    delta_P = P_in - P_out  # pressure difference in Pa

    # Calculate volumetric flow rate Q using Hagen-Poiseuille equation
    # Q = (pi * r^4 * delta_P) / (8 * eta * L)
    Q = (math.pi * r**4 * delta_P) / (8 * eta * L)  # in m^3/s

    # Calculate density of CO2 at output conditions using ideal gas law
    # rho = P * M / (R * T)
    # Molar mass of CO2 = 44.01 g/mol = 0.04401 kg/mol
    M = 0.04401  # kg/mol
    R = 8.314  # J/(mol K)
    rho = (P_out * M) / (R * T)  # kg/m^3

    # Calculate mass flow rate
    mass_flow_rate = Q * rho  # kg/s

    # Calculate usage time in seconds
    usage_time_seconds = mass_co2_kg / mass_flow_rate

    # Convert usage time to hours
    usage_time_hours = usage_time_seconds / 3600

    return usage_time_hours

if __name__ == '__main__':
    usage_hours = calculate_co2_cylinder_usage()
    print(f'The CO2 cylinder can be used for approximately {usage_hours:.2f} hours under the given conditions.')
