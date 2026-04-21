# filename: proton_acceleration.py

def calculate_proton_acceleration(electric_field: float) -> float:
    """Calculate the acceleration of a proton in a given electric field.

    Args:
        electric_field (float): Electric field strength in N/C.

    Returns:
        float: Acceleration in m/s^2.
    """
    # Constants
    proton_charge = 1.602e-19  # Coulombs
    proton_mass = 1.673e-27    # Kilograms

    # Calculate acceleration
    acceleration = (proton_charge * electric_field) / proton_mass
    return acceleration

if __name__ == "__main__":
    E = 2.00e4  # Electric field in N/C
    a = calculate_proton_acceleration(E)
    print(f"Acceleration of proton in electric field {E} N/C is {a:.2e} m/s^2")
