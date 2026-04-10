import numpy as np

def calculate_t1(gamma):
    """
    Calculate the time at which the solution attains its maximum value for a given damping coefficient.
    
    Parameters:
    gamma (float): The damping coefficient (must be less than 2).
    
    Returns:
    float: The time t1 at which the solution attains its maximum value.
    """
    if gamma >= 2:
        raise ValueError('Gamma must be less than 2 for underdamped case.')
    omega_d = np.sqrt(max(1 - (gamma / 2) ** 2, 1e-10))
    t1 = 1 + (1 / omega_d) * np.arctan(2 * omega_d / gamma)
    return t1

# Example usage:
gamma_value = 1.0  # Set the damping coefficient
result = calculate_t1(gamma_value)
print(f'The time t1 at which the solution attains its maximum value is: {result}')