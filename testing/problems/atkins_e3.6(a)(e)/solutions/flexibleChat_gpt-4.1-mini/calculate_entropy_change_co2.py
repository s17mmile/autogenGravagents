# filename: calculate_entropy_change_co2.py

import math

# Constants
R = 8.314  # gas constant in J/(mol K)


def calculate_entropy_change():
    n = 2.0  # moles
    T1 = 298.15  # initial temperature in K
    P1_atm = 10.0  # initial pressure in atm
    P_ext_atm = 1.0  # external pressure in atm
    A_cm2 = 10.0  # cross-sectional area in cm^2
    delta_x_cm = 20.0  # piston displacement in cm
    C_V_m = 28.8  # molar heat capacity at constant volume in J/(mol K)

    # Unit conversions
    P1 = P1_atm * 101325  # convert atm to Pa
    P_ext = P_ext_atm * 101325  # convert atm to Pa
    A = A_cm2 * 1e-4  # convert cm^2 to m^2
    delta_x = delta_x_cm * 1e-2  # convert cm to m

    # Initial volume V1 (ideal gas law)
    V1 = n * R * T1 / P1  # in m^3

    # Final volume V2
    V2 = V1 + A * delta_x  # in m^3

    delta_V = V2 - V1  # volume change

    # Work done on the gas (W = -P_ext * delta_V)
    W = -P_ext * delta_V  # in J

    # Change in internal energy (Delta U = n * C_V_m * (T2 - T1))
    # Since Delta U = W (adiabatic, no heat exchange)
    T2 = T1 + W / (n * C_V_m)  # final temperature in K

    # Check physical validity of T2
    if T2 <= 0:
        raise ValueError(f"Calculated final temperature T2={T2:.2f} K is non-physical. Check input parameters.")

    # Entropy change of the gas
    # Note: Although the process is irreversible, entropy is a state function,
    # so we use the reversible path formula between initial and final states.
    Delta_S = n * C_V_m * math.log(T2 / T1) + n * R * math.log(V2 / V1)  # in J/K

    return Delta_S


if __name__ == "__main__":
    delta_S = calculate_entropy_change()
    print(f"Entropy change (Delta S) = {delta_S:.2f} J/K")
