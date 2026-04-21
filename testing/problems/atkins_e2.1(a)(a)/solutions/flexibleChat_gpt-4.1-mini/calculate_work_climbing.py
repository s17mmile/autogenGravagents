# filename: calculate_work_climbing.py

def calculate_work(mass_kg, height_m, gravity=9.81):
    """Calculate the work done to climb a height against gravity.

    Parameters:
    mass_kg (float): Mass in kilograms, must be positive.
    height_m (float): Height in meters, must be positive.
    gravity (float): Acceleration due to gravity in m/s^2, default is 9.81.

    Returns:
    float: Work done in joules.

    Raises:
    ValueError: If mass_kg or height_m is not positive.
    """
    if mass_kg <= 0:
        raise ValueError("Mass must be a positive number.")
    if height_m <= 0:
        raise ValueError("Height must be a positive number.")

    work_joules = mass_kg * gravity * height_m
    return work_joules


if __name__ == "__main__":
    """Calculate and print the work done for a 65 kg person climbing 4.0 meters."""
    mass = 65.0  # kg
    height = 4.0  # meters

    work_done = calculate_work(mass, height)
    print(f"Work done to climb {height} meters by a {mass} kg person: {work_done:.2f} J")
