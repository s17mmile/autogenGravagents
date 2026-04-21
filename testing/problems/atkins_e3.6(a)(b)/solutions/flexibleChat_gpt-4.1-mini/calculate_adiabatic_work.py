# filename: calculate_adiabatic_work.py

# Given data
n = 2.0  # moles of CO2
T1 = 298.15  # initial temperature in K
P1_atm = 10.0  # initial pressure in atm
A_cm2 = 10.0  # cross-sectional area in cm^2
P_ext_atm = 1.0  # external pressure in atm
delta_x_cm = 20.0  # piston displacement in cm
C_V_m = 28.8  # molar heat capacity at constant volume in J/(mol K)

# Unit conversions
A_m2 = A_cm2 * 1e-4  # cm^2 to m^2
delta_x_m = delta_x_cm * 1e-2  # cm to m
P_ext_Pa = P_ext_atm * 1.01325e5  # atm to Pa

# Calculate change in volume (m^3)
delta_V = A_m2 * delta_x_m

# Calculate work done by the gas (J)
w = -P_ext_Pa * delta_V

# Output the result
print(f"Work done by the gas during adiabatic expansion: {w:.2f} J")
