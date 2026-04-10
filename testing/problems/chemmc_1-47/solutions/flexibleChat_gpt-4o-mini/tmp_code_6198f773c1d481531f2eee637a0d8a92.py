def calculate_uncertainty_in_momentum(delta_x):
    """
    Calculate the uncertainty in momentum (Δp) based on the uncertainty in position (Δx).
    
    Parameters:
    delta_x (float): Uncertainty in position in meters.
    
    Returns:
    float: Uncertainty in momentum in kg m/s.
    """
    reduced_plancks_constant = 1.055e-34  # Reduced Planck's constant in Js
    return reduced_plancks_constant / (2 * delta_x)

# Constants
Delta_x = 10e-12   # Uncertainty in position in meters (10 pm)

# Calculate the uncertainty in momentum
Delta_p = calculate_uncertainty_in_momentum(Delta_x)

# Output the result
print(f'The calculated uncertainty in momentum (Δp) is: {Delta_p:.3e} kg m/s')