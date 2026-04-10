def calculate_delta_U(P_ext_atm, A_cm2, d_cm):
    # Convert units
    P_ext_Pa = P_ext_atm * 101325  # Convert atm to Pa
    A_m2 = A_cm2 * 1e-4  # Convert cm^2 to m^2
    d_m = d_cm * 0.01  # Convert cm to m

    # Calculate change in volume (Delta V)
    Delta_V = A_m2 * d_m  # m^3

    # Calculate work done (W)
    W = -P_ext_Pa * Delta_V  # J

    # Since q = 0 for adiabatic process, Delta U = W
    Delta_U = W
    return Delta_U

# Constants
P_ext_atm = 1.0  # External pressure in atm
A_cm2 = 10.0  # Cross-sectional area in cm^2
d_cm = 20.0  # Distance moved by the piston in cm

# Calculate Delta U
delta_U = calculate_delta_U(P_ext_atm, A_cm2, d_cm)

# Output the result
print(f'Delta U = {delta_U:.2f} J')