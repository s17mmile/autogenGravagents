def calculate_pressure_change(P1, T1, change_in_temp):
    # New temperature
    T2 = T1 + change_in_temp
    # Calculate new pressure using the proportional relationship
    P2 = P1 * (T2 / T1)
    # Calculate change in pressure
    delta_P = P2 - P1
    return delta_P

# Constants
P1 = 6.69  # Initial pressure in kPa at the triple point of water
T1 = 273.16  # Initial temperature in K
change_in_temp = 1.00  # Change in temperature in K

# Calculate the change in pressure
pressure_change = calculate_pressure_change(P1, T1, change_in_temp)

# Output the change in pressure
print(f'Change in pressure for a 1.00 K increase: {pressure_change:.2f} kPa')