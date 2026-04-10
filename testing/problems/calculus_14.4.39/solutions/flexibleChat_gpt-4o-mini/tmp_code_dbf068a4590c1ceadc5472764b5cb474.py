def calculate_parallel_resistance_and_error(R1, R2, R3, percentage_error):
    """
    Calculate the total resistance of three resistors in parallel and estimate the maximum error.

    Parameters:
    R1, R2, R3 : float
        Resistances in ohms.
    percentage_error : float
        Percentage error as a decimal (e.g., 0.005 for 0.5%).

    Returns:
    tuple
        Total resistance and maximum error in ohms.
    """
    # Check for valid resistance values
    if R1 <= 0 or R2 <= 0 or R3 <= 0:
        raise ValueError('Resistances must be greater than zero.')

    # Calculate total resistance R
    R_total = 1 / (1/R1 + 1/R2 + 1/R3)

    # Calculate absolute errors
    absolute_error_R1 = percentage_error * R1
    absolute_error_R2 = percentage_error * R2
    absolute_error_R3 = percentage_error * R3

    # Calculate the relative errors
    relative_error_sum = (absolute_error_R1 / R1**2) + (absolute_error_R2 / R2**2) + (absolute_error_R3 / R3**2)

    # Calculate maximum error in R
    Delta_R = R_total**2 * relative_error_sum

    return R_total, Delta_R

# Constants for resistances in ohms
R1 = 25.0
R2 = 40.0
R3 = 50.0

# Percentage error
percentage_error = 0.005  # 0.5%

# Calculate total resistance and maximum error
R_total, Delta_R = calculate_parallel_resistance_and_error(R1, R2, R3, percentage_error)

# Output the results
print(f'Total Resistance R: {R_total:.2f} Ohms')
print(f'Maximum Error in R (estimate): {Delta_R:.4f} Ohms')