# filename: electron_acceleration.py

def calculate_electron_acceleration() -> float:
    """Calculate the acceleration of an electron in a uniform electric field.

    Returns:
        float: Acceleration in meters per second squared (m/s^2).
    """
    # Constants
    electron_charge = 1.602e-19  # Coulombs (absolute value used since acceleration magnitude is requested)
    electron_mass = 9.109e-31    # Kilograms
    electric_field = 2.00e4      # N/C

    # Calculate acceleration using a = qE/m
    acceleration = (electron_charge * electric_field) / electron_mass
    return acceleration

if __name__ == "__main__":
    accel = calculate_electron_acceleration()
    print(f"Acceleration of the electron: {accel:.3e} m/s^2")
