from math import floor, log10

def calculate_surface_temperature(lambda_max_nm):
    # Constants
    wien_constant = 2.898e-3  # Wien's displacement constant in m·K

    # Validate input
    if lambda_max_nm <= 0:
        raise ValueError('Wavelength must be a positive value.')

    # Convert wavelength from nm to m
    lambda_max_m = lambda_max_nm * 1e-9  # Convert nm to m

    # Calculate temperature using Wien's Displacement Law
    temperature = wien_constant / lambda_max_m  # Temperature in Kelvin

    # Round to three significant figures
    temperature_rounded = round(temperature, -int(floor(log10(temperature))) + 2)

    return temperature_rounded

# Example usage
lambda_max = 260  # Peak wavelength in nm
estimated_temperature = calculate_surface_temperature(lambda_max)
print(f'The estimated surface temperature of Sirius is approximately {estimated_temperature} K.')