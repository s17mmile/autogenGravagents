# filename: calculate_thermodynamic_force.py
import math

def calculate_thermodynamic_force(T_celsius=25, half_distance_cm=10):
    """
    Calculate the thermodynamic force on a solute given the temperature and half-distance where concentration halves.

    Parameters:
    T_celsius (float): Temperature in degrees Celsius (must be > 0).
    half_distance_cm (float): Distance in centimeters where concentration falls to half (must be > 0).

    Returns:
    float: Thermodynamic force in J/(mol*m).
    """
    if T_celsius <= 0:
        raise ValueError("Temperature must be greater than 0 Celsius.")
    if half_distance_cm <= 0:
        raise ValueError("Half-distance must be greater than 0 cm.")

    R = 8.314  # J/(mol*K), gas constant
    T = T_celsius + 273.15  # Convert Celsius to Kelvin

    half_distance_m = half_distance_cm / 100.0  # Convert cm to m

    k = math.log(2) / half_distance_m  # Decay constant in m^-1

    F = R * T * k  # Thermodynamic force in J/(mol*m)

    return F

if __name__ == '__main__':
    force = calculate_thermodynamic_force()
    print(f'Thermodynamic force on the solute: {force:.2e} J/(mol*m)')
