# filename: dye_concentration_time.py
import math

def calculate_time_to_concentration(V_gal=60000, Q_gal_per_min=200, M0_kg=5, target_concentration_g_per_gal=0.02):
    """
    Calculate the time (in minutes) for the dye concentration in the pool to reach a target concentration.

    Parameters:
    V_gal (float): Volume of the pool in gallons.
    Q_gal_per_min (float): Flow rate of the filtering system in gallons per minute.
    M0_kg (float): Initial mass of dye in kilograms.
    target_concentration_g_per_gal (float): Target dye concentration in grams per gallon.

    Returns:
    tuple: (time in minutes (float), hours (int), minutes (int))
    """
    # Validate inputs
    if V_gal <= 0 or Q_gal_per_min <= 0 or M0_kg <= 0 or target_concentration_g_per_gal <= 0:
        raise ValueError("All input parameters must be positive numbers.")

    # Convert initial dye mass from kg to g
    M0_g = M0_kg * 1000

    # Calculate the ratio inside the logarithm
    ratio = (target_concentration_g_per_gal * V_gal) / M0_g

    if ratio <= 0 or ratio >= 1:
        raise ValueError("Invalid target concentration: resulting logarithm argument must be between 0 and 1.")

    # Calculate time T in minutes
    T_min = - (V_gal / Q_gal_per_min) * math.log(ratio)

    # Convert time to hours and minutes
    hours = int(T_min // 60)
    minutes = int(round(T_min % 60))

    return T_min, hours, minutes

# Calculate and print the result
T_min, hours, minutes = calculate_time_to_concentration()
print(f"Time to reach concentration 0.02 g/gal: {T_min:.2f} minutes ({hours} hours and {minutes} minutes)")
