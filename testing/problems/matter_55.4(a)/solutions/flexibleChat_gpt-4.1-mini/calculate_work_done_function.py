# filename: calculate_work_done_function.py

def calculate_work_done(area_cm2, distance_cm, pressure_atm):
    """
    Calculate the work done by the system during piston expansion.

    Parameters:
    area_cm2 (float): Cross-sectional area in square centimeters.
    distance_cm (float): Distance piston moves in centimeters.
    pressure_atm (float): External pressure in atmospheres.

    Returns:
    float: Work done in joules.
    """
    # Input validation
    if not (isinstance(area_cm2, (int, float)) and area_cm2 > 0):
        raise ValueError("Area must be a positive number.")
    if not (isinstance(distance_cm, (int, float)) and distance_cm > 0):
        raise ValueError("Distance must be a positive number.")
    if not (isinstance(pressure_atm, (int, float)) and pressure_atm > 0):
        raise ValueError("Pressure must be a positive number.")

    # Convert units to SI
    area_m2 = area_cm2 * 1e-4  # cm^2 to m^2
    distance_m = distance_cm * 1e-2  # cm to m
    pressure_pa = pressure_atm * 1.013e5  # atm to Pa

    # Calculate volume change (m^3)
    delta_V = area_m2 * distance_m

    # Calculate work done (J)
    work_done = pressure_pa * delta_V

    return work_done


# Example usage with given data
if __name__ == "__main__":
    area = 50  # cm^2
    distance = 15  # cm
    pressure = 1.0  # atm

    work = calculate_work_done(area, distance, pressure)
    print(f"Work done by the system: {work:.2f} J")
