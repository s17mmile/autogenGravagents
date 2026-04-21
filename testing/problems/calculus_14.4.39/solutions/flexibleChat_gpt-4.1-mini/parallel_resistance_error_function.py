# filename: parallel_resistance_error_function.py

def calculate_parallel_resistance_error(R1, R2, R3, percent_error):
    """
    Calculate the total resistance of three resistors connected in parallel and estimate the maximum error.

    Parameters:
    R1, R2, R3 : float
        Resistance values in ohms (must be positive and non-zero).
    percent_error : float
        Relative error in each resistor (e.g., 0.005 for 0.5%).

    Returns:
    dict
        Dictionary containing total resistance, maximum absolute error, and relative error percentage.
    """
    # Input validation
    if any(r <= 0 for r in (R1, R2, R3)):
        raise ValueError("All resistor values must be positive and non-zero.")
    if percent_error < 0:
        raise ValueError("Percent error must be non-negative.")

    # Calculate total resistance R for parallel resistors
    inv_R = 1/R1 + 1/R2 + 1/R3
    R = 1 / inv_R

    # Calculate maximum error in f = 1/R
    partial_R1 = 1 / (R1**2)
    partial_R2 = 1 / (R2**2)
    partial_R3 = 1 / (R3**2)

    delta_R1 = percent_error * R1
    delta_R2 = percent_error * R2
    delta_R3 = percent_error * R3

    delta_f = (partial_R1 * delta_R1) + (partial_R2 * delta_R2) + (partial_R3 * delta_R3)

    relative_error_f = delta_f / inv_R
    relative_error_R = relative_error_f
    delta_R = relative_error_R * R

    return {
        'Total Resistance (Ohms)': R,
        'Maximum Absolute Error (Ohms)': delta_R,
        'Relative Error (%)': relative_error_R * 100
    }


# Example usage with given values
result = calculate_parallel_resistance_error(25.0, 40.0, 50.0, 0.005)
result
